#include "pathtracer/scene/path_trace_driver.h"

#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/aov_routing.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

namespace pathtracer::scene {

namespace {

// Only the buffer-starvation retry still polls: nothing signals a shared_ptr release, so there is no event for a wait to key on.
constexpr std::chrono::milliseconds kBufferRetryInterval{5};

// Incremental running mean computed in the fresh pass buffer, so the published set is only read and publish is a pointer swap.
void accumulateMean(PathTraceResult& sample, const PathTraceResult& previousMean, int n,
                     ThreadPool& threadPool) {
    const float invN = 1.0F / static_cast<float>(n);
    const std::span<const pathtracer::debug::PathTracedLane> lanes = pathtracer::debug::pathTracedLanes();
    const auto width = static_cast<std::size_t>(sample.beauty.width);
    const auto beautyChannels = static_cast<std::size_t>(sample.beauty.channels);
    threadPool.parallelFor(sample.beauty.height, [&](int y) {
        const std::size_t rowPixel = static_cast<std::size_t>(y) * width;
        // Before the lane folds overwrite it: `beauty` still holds this pass's own radiance and its source the mean of the rest.
        foldLuminanceM2(previousMean.beauty.texels.data() + (rowPixel * beautyChannels),
                        sample.beauty.texels.data() + (rowPixel * beautyChannels), beautyChannels,
                        previousMean.beautyLuminanceM2.data() + rowPixel, sample.beautyLuminanceM2.data() + rowPixel, width,
                        invN);
        for (const pathtracer::debug::PathTracedLane lane : lanes) {
            const auto channels = static_cast<std::size_t>((sample.*lane).channels);
            float* destination = (sample.*lane).texels.data() + (rowPixel * channels);
            foldRunningMean((previousMean.*lane).texels.data() + (rowPixel * channels), destination, destination,
                            width * channels, invN);
        }
    });
}

// Reduces the published mean's beauty on the thread that just wrote those texels, still in cache; one chunk per worker.
void reduceOverRange(PathTraceResult& pass, std::vector<OverRangeHistogram>& histograms,
                      std::vector<float>& peaks, ThreadPool& threadPool) {
    const pathtracer::gfx::HdrImage& beauty = pass.beauty;
    const int chunkCount =
        std::max(1, std::min(beauty.height, static_cast<int>(threadPool.threadCount())));
    const int chunkRows = (beauty.height + chunkCount - 1) / chunkCount;
    const auto channels = static_cast<std::size_t>(beauty.channels);
    const auto rowFloats = static_cast<std::size_t>(beauty.width) * channels;
    histograms.assign(static_cast<std::size_t>(chunkCount), OverRangeHistogram{});
    peaks.assign(static_cast<std::size_t>(chunkCount), 0.0F);

    threadPool.parallelFor(chunkCount, [&](int chunk) {
        OverRangeHistogram& bins = histograms[static_cast<std::size_t>(chunk)];
        const std::size_t begin = static_cast<std::size_t>(chunk * chunkRows) * rowFloats;
        const std::size_t end =
            static_cast<std::size_t>(std::min((chunk + 1) * chunkRows, beauty.height)) * rowFloats;
        const float* rgb = beauty.texels.data();
        // Peak kept in a register and stored once: `peaks` is the only array adjacent chunks could false-share.
        float peak = 0.0F;
        for (std::size_t i = begin; i < end; i += channels) {
            const float maxChannel = std::max({rgb[i], rgb[i + 1], rgb[i + 2]});
            ++bins[static_cast<std::size_t>(overRangeBin(maxChannel))];
            peak = std::max(peak, maxChannel);
        }
        peaks[static_cast<std::size_t>(chunk)] = peak;
    });

    OverRangeStats& out = pass.overRange;
    out.rawPeak = *std::max_element(peaks.begin(), peaks.end());
    // Summed chunk-major then suffix-summed in place: both walks are sequential, where folding direct would stride across chunks.
    std::fill(out.aboveBin.begin(), out.aboveBin.end(), 0U);
    for (const OverRangeHistogram& bins : histograms) {
        for (std::size_t bin = 0; bin < bins.size(); ++bin) {
            out.aboveBin[bin] += bins[bin];
        }
    }
    for (std::size_t bin = kOverRangeBinCount; bin-- > 0;) {
        out.aboveBin[bin] += out.aboveBin[bin + 1];
    }
}

double millisecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

void foldRunningMean(const float* previousMean, const float* drawn, float* mean, std::size_t count, float invN) {
    for (std::size_t i = 0; i < count; ++i) {
        mean[i] = previousMean[i] + ((drawn[i] - previousMean[i]) * invN);
    }
}

void foldLuminanceM2(const float* previousMean, const float* drawn, std::size_t channels, const float* previousM2, float* m2,
                     std::size_t pixels, float invN) {
    const glm::vec3 weights = pathtracer::debug::kRec709LuminanceWeights;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        const std::size_t i = pixel * channels;
        const float x = (drawn[i] * weights.r) + (drawn[i + 1] * weights.g) + (drawn[i + 2] * weights.b);
        const float before = (previousMean[i] * weights.r) + (previousMean[i + 1] * weights.g) + (previousMean[i + 2] * weights.b);
        // The luminance mean m_n by foldRunningMean's recurrence and invN; (x - m_{n-1})(x - m_n) is West's M2 increment.
        const float after = before + ((x - before) * invN);
        m2[pixel] = previousM2[pixel] + ((x - before) * (x - after));
    }
}

PathTraceDriver::PathTraceDriver(const EmbreeAccel& accel,
                                  const std::vector<ShadingTriangle>& shadingTriangles,
                                  const std::vector<MeshInstance>& instances,
                                  const std::vector<int>& instanceLightIndex,
                                  const EnvironmentMap& environmentMap,
                                  const std::vector<QuadLight>& quadLights,
                                  const std::vector<PathTraceSettings>& perInstanceSettings)
    : accel_(accel),
      shadingTriangles_(shadingTriangles),
      instances_(instances),
      instanceLightIndex_(instanceLightIndex),
      environmentMap_(environmentMap),
      quadLights_(quadLights),
      perInstanceSettings_(perInstanceSettings),
      thread_([this](std::stop_token stopToken) { driverLoop(std::move(stopToken)); }) {}

PathTraceDriver::~PathTraceDriver() = default;  // jthread requests stop + joins automatically

std::uint64_t PathTraceDriver::requestTrace(const Request& request) {
    std::uint64_t generation = 0;
    {
        const std::lock_guard<std::mutex> lock(requestMutex_);
        pendingRequest_.emplace(request);
        generation = generation_.fetch_add(1, std::memory_order_relaxed) + 1;
        ++wakeEpoch_;
    }
    wakeCv_.notify_all();
    return generation;
}

void PathTraceDriver::setSuspended(bool suspended) {
    {
        // Under the same lock as the epoch bump, so a driver about to idle on the old value sees the change instead of sleeping through it.
        const std::lock_guard<std::mutex> lock(requestMutex_);
        suspended_.store(suspended, std::memory_order_relaxed);
        ++wakeEpoch_;
    }
    wakeCv_.notify_all();
}

void PathTraceDriver::idleUntilWake(const std::stop_token& stopToken, std::uint64_t seen) {
    std::unique_lock<std::mutex> lock(requestMutex_);
    wakeCv_.wait(lock, stopToken, [this, seen] { return wakeEpoch_ != seen; });
}

std::shared_ptr<const PathTraceResult> PathTraceDriver::latestResult() const {
    const std::lock_guard<std::mutex> lock(resultMutex_);
    return result_;
}

// Driver-thread-only. Hands back the first pool slot nothing else holds, reallocating only on a size change.
std::shared_ptr<PathTraceResult> PathTraceDriver::acquireFreeBuffer(int width, int height) {
    for (std::shared_ptr<PathTraceResult>& slot : bufferPool_) {
        if (slot != nullptr && slot.use_count() > 1) {
            continue;
        }
        if (slot == nullptr || slot->beauty.width != width || slot->beauty.height != height) {
            slot = std::make_shared<PathTraceResult>(makePathTraceResult(width, height));
        }
        return slot;
    }
    return nullptr;
}

pathtracer::debug::PassRecord PathTraceDriver::lastPassRecord() const {
    const std::lock_guard<std::mutex> lock(statsMutex_);
    return lastPass_;
}

std::vector<pathtracer::debug::PassRecord> PathTraceDriver::takePassRecords() {
    std::vector<pathtracer::debug::PassRecord> taken;
    const std::lock_guard<std::mutex> lock(statsMutex_);
    taken.swap(untakenPasses_);
    return taken;
}

// Driver-thread-only. passStats_ is read here so the completed and cancelled call sites share one definition of a PassRecord.
void PathTraceDriver::publishPassRecord(std::uint64_t generation, int passIndex, int width,
                                         int height, double traceMs, double accumulateMs,
                                         double overRangeMs, double publishMs, double passMs,
                                         bool cancelled) {
    const pathtracer::debug::PassRecord record{generation,
                                            passIndex,
                                            width,
                                            height,
                                            traceMs,
                                            accumulateMs,
                                            overRangeMs,
                                            publishMs,
                                            passMs,
                                            passStats_.rays(),
                                            passStats_.tilesCompleted(),
                                            passStats_.tilesCancelled(),
                                            cancelled};
    const std::lock_guard<std::mutex> lock(statsMutex_);
    lastPass_ = record;
    untakenPasses_.push_back(record);
}

// Driver-thread-only. Resets the pass counters, builds this pass's LightSet and traces one pass into `pass`; returns the trace time.
double PathTraceDriver::tracePass(const Request& request, int sampleBase, std::uint64_t generation, PathTraceResult& pass) {
    passStats_.reset();
    // Built fresh each pass from this request's env state, holding references not copies, so the HUD toggle needs no invalidation path.
    const LightSet lights(request.envLightEnabled ? &environmentMap_ : nullptr, request.envRotationRadians, request.envExposure,
                          quadLights_);
    const auto traceStart = std::chrono::steady_clock::now();
    renderPathTraced(request.camera, accel_, shadingTriangles_, instances_,
                     instanceLightIndex_, lights, request.width, request.height,
                     request.showSky, request.settings, perInstanceSettings_,
                     // The generation is the scramble seed: fixed per accumulation, changing exactly when the image restarts.
                     static_cast<std::uint32_t>(generation), sampleBase, request.maxSamples,
                     generation_, generation,
                     threadPool_, passStats_, pass);
    return millisecondsSince(traceStart);
}

// Driver-thread-only. Folds a completed pass into the running mean, reduces its over-range stats, then publishes it and its record.
void PathTraceDriver::finishPass(const std::shared_ptr<PathTraceResult>& pass, std::shared_ptr<PathTraceResult>& currentMean,
                                 int passIndex, std::uint64_t generation, double traceMs,
                                 std::chrono::steady_clock::time_point passStart) {
    // passIndex == 1 leaves the pass as rendered: the running mean of one sample is that sample, and no previous mean exists.
    const auto accumulateStart = std::chrono::steady_clock::now();
    if (passIndex > 1) {
        accumulateMean(*pass, *currentMean, passIndex, threadPool_);
    } else {
        // One sample has no dispersion, so the second moment starts at exactly zero rather than at whatever the reused slot held.
        std::fill(pass->beautyLuminanceM2.begin(), pass->beautyLuminanceM2.end(), 0.0F);
    }
    const double accumulateMs = millisecondsSince(accumulateStart);

    // After the mean, before the publish: the statistics must describe the image about to go on screen.
    const auto overRangeStart = std::chrono::steady_clock::now();
    reduceOverRange(*pass, overRangeHistograms_, overRangePeaks_, threadPool_);
    const double overRangeMs = millisecondsSince(overRangeStart);
    pass->generation = generation;
    pass->samples = passIndex;
    currentMean = pass;

    const auto publishStart = std::chrono::steady_clock::now();
    {
        const std::lock_guard<std::mutex> lock(resultMutex_);
        result_ = currentMean;
    }
    const double publishMs = millisecondsSince(publishStart);

    publishPassRecord(generation, passIndex, pass->beauty.width, pass->beauty.height, traceMs,
                      accumulateMs, overRangeMs, publishMs, millisecondsSince(passStart),
                      /*cancelled=*/false);
}

// Runs until destruction on jthread's stop token, picking up the latest request whenever its generation changes.
void PathTraceDriver::driverLoop(std::stop_token stopToken) {
    // The pool slot holding the last published mean of the active generation: read as the previous mean, never written again.
    std::shared_ptr<PathTraceResult> currentMean;
    std::optional<Request> activeRequest;
    std::uint64_t activeGeneration = 0;  // 0 == no request handled yet; requestTrace's first bump makes generation_ 1
    int accumulated = 0;  // passes in currentMean; driver-thread-only, readers see it only as the published result's samples

    while (!stopToken.stop_requested()) {
        // Read before any idle decision below: whatever makes those decisions stale also bumps this, so no wake can be missed.
        std::uint64_t wakeSeen = 0;
        {
            const std::lock_guard<std::mutex> lock(requestMutex_);
            wakeSeen = wakeEpoch_;
        }
        const std::uint64_t requestedGeneration = generation_.load(std::memory_order_relaxed);
        if (requestedGeneration == 0 || suspended_.load(std::memory_order_relaxed)) {
            idleUntilWake(stopToken, wakeSeen);
            continue;
        }

        if (requestedGeneration != activeGeneration) {
            {
                const std::lock_guard<std::mutex> lock(requestMutex_);
                // Guaranteed engaged: requestedGeneration != 0 implies a requestTrace() call has completed.
                activeRequest = pendingRequest_;
            }
            activeGeneration = requestedGeneration;
            accumulated = 0;
        }

        // Engaged: requestedGeneration goes non-zero only inside requestTrace(), under one lock.
        const Request& request = *activeRequest;  // NOLINT(bugprone-unchecked-optional-access)
        // A zero-size view, or converged: nothing changes until a new request or a suspend, both bumping the epoch this wait keys on.
        if (request.width <= 0 || request.height <= 0 || (request.maxSamples > 0 && accumulated >= request.maxSamples)) {
            idleUntilWake(stopToken, wakeSeen);
            continue;
        }

        const std::shared_ptr<PathTraceResult> pass = acquireFreeBuffer(request.width, request.height);
        if (pass == nullptr) {
            std::this_thread::sleep_for(kBufferRetryInterval);
            continue;  // every buffer still referenced by the render thread -- retry rather than allocate
        }

        // Two roles (sampler.h): sampleBase continues the Sobol sequence, passIndex is that count 1-based, so this pass weighs 1/passIndex.
        const int sampleBase = accumulated;
        const int passIndex = sampleBase + 1;
        const auto passStart = std::chrono::steady_clock::now();
        const double traceMs = tracePass(request, sampleBase, activeGeneration, *pass);

        if (generation_.load(std::memory_order_relaxed) != activeGeneration) {
            // Published before the discard, not skipped: a camera drag cancels passes continuously, and their rays were still paid for.
            publishPassRecord(activeGeneration, passIndex, pass->beauty.width, pass->beauty.height,
                               traceMs, 0.0, 0.0, 0.0, millisecondsSince(passStart),
                               /*cancelled=*/true);
            continue;  // superseded mid-pass -- discard, next iteration picks up the new request
        }
        finishPass(pass, currentMean, passIndex, activeGeneration, traceMs, passStart);
        accumulated = passIndex;
    }
}

}  // namespace pathtracer::scene
