#include "pathtracer/scene/path_trace_driver.h"

#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/scene/cie.h"

#include <boost/math/distributions/chi_squared.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

namespace pathtracer::scene {

namespace {

// Only the buffer-starvation retry still polls: nothing signals a shared_ptr release, so there is no event for a wait to key on.
constexpr std::chrono::milliseconds kBufferRetryInterval{5};

// Incremental running mean computed in the fresh pass buffer, so the published set is only read and publish is a pointer swap.
void accumulateMean(PathTraceResult& sample, const PathTraceResult& previousMean, int n, std::span<const std::uint32_t> texels,
                     ThreadPool& threadPool) {
    const float invN = 1.0F / static_cast<float>(n);
    const std::span<const pathtracer::debug::PathTracedLane> lanes = pathtracer::debug::pathTracedLanes();
    const auto width = static_cast<std::uint32_t>(sample.beauty.width);
    const auto beautyChannels = static_cast<std::size_t>(sample.beauty.channels);
    const int replicates = sample.replicates;
    // Pass n-1 (0-based) is replicate (n-1) mod K's ((n-1) div K + 1)th, the mapping renderPathTraced draws its sequence by.
    const int replicate = (n - 1) % replicates;
    const int count = ((n - 1) / replicates) + 1;
    const float invCount = 1.0F / static_cast<float>(count);
    const auto stride = static_cast<std::size_t>(replicates);
    const auto fold = [&](std::uint32_t begin, std::uint32_t end) {
        const std::size_t run = end - begin;
        // Before the lane folds overwrite it: `beauty` still holds this pass's own radiance.
        foldReplicateLuminance(previousMean.replicateLuminance.data() + (begin * stride),
                               sample.beauty.texels.data() + (begin * beautyChannels), beautyChannels,
                               sample.replicateLuminance.data() + (begin * stride), run, replicates, replicate, invCount);
        for (const pathtracer::debug::PathTracedLane lane : lanes) {
            const auto channels = static_cast<std::size_t>((sample.*lane).channels);
            float* destination = (sample.*lane).texels.data() + (begin * channels);
            foldRunningMean((previousMean.*lane).texels.data() + (begin * channels), destination, destination, run * channels, invN);
        }
        std::fill(sample.texelSamples.data() + begin, sample.texelSamples.data() + end, n);
    };
    // An untraced texel keeps its estimate: the fresh buffer holds a stale pass there, so every lane, mean and count is carried over.
    const auto carry = [&](std::uint32_t begin, std::uint32_t end) {
        const float* means = previousMean.replicateLuminance.data();
        std::copy(means + (begin * stride), means + (end * stride), sample.replicateLuminance.data() + (begin * stride));
        for (const pathtracer::debug::PathTracedLane lane : lanes) {
            const auto channels = static_cast<std::size_t>((sample.*lane).channels);
            const float* previous = (previousMean.*lane).texels.data();
            std::copy(previous + (begin * channels), previous + (end * channels), (sample.*lane).texels.data() + (begin * channels));
        }
        const int* counts = previousMean.texelSamples.data();
        std::copy(counts + begin, counts + end, sample.texelSamples.data() + begin);
    };
    threadPool.parallelFor(sample.beauty.height, [&](int y) {
        const std::uint32_t rowBegin = static_cast<std::uint32_t>(y) * width;
        forEachTracedRun(texels, rowBegin, rowBegin + width, fold, carry);
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

void foldReplicateLuminance(const float* previousMeans, const float* drawn, std::size_t channels, float* means, std::size_t pixels,
                            int replicates, int replicate, float invCount) {
    const glm::vec3 weights = pathtracer::scene::cie::sceneLuminanceWeights();
    const auto stride = static_cast<std::size_t>(replicates);
    const auto slot = static_cast<std::size_t>(replicate);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        const float* previous = previousMeans + (pixel * stride);
        float* folded = means + (pixel * stride);
        // In place (headless) the other replicates are already there; into a fresh pool buffer (driver) they are carried.
        if (folded != previous) {
            std::copy(previous, previous + stride, folded);
        }
        const std::size_t i = pixel * channels;
        const float x = (drawn[i] * weights.r) + (drawn[i + 1] * weights.g) + (drawn[i + 2] * weights.b);
        folded[slot] = previous[slot] + ((x - previous[slot]) * invCount);
    }
}

ReplicateEstimate replicateEstimate(const float* means, int populated) {
    const double mean = std::accumulate(means, means + populated, 0.0) / static_cast<double>(populated);
    double squares = 0.0;
    for (int r = 0; r < populated; ++r) {
        squares += (means[r] - mean) * (means[r] - mean);
    }
    const double inverseDegrees = 1.0 / (static_cast<double>(populated) * static_cast<double>(populated - 1));
    return {mean, squares * inverseDegrees};
}

// Normal replicate means make (K-1) s^2 / sigma^2 chi-squared on K-1 dof (Cochran 1934): small at sigma below the tolerance.
double convergenceStatistic(const ReplicateEstimate& estimate, int replicates, float threshold) {
    const double tolerance = static_cast<double>(threshold) * estimate.mean;
    // Zero spread is no evidence: replicates of a few samples agree exactly when every sample missed a rare path, as at a silhouette.
    if (!(estimate.meanVariance > 0.0) || !(tolerance > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    return (static_cast<double>(replicates) - 1.0) * estimate.meanVariance / (tolerance * tolerance);
}

std::size_t stepUpRejections(std::span<const double> ascendingP, std::size_t tests, double alpha) {
    for (std::size_t k = ascendingP.size(); k > 0; --k) {
        if (ascendingP[k - 1] <= static_cast<double>(k) * alpha / static_cast<double>(tests)) {
            return k;
        }
    }
    return 0;
}

bool adaptiveCheckpoint(int n, const PathTraceSettings& settings) {
    const AdaptiveSampling& adaptive = settings.adaptive;
    return adaptive.threshold > 0.0F && n >= adaptive.minSamples && n % settings.replicates == 0 &&
           std::has_single_bit(static_cast<unsigned int>(n / settings.replicates));
}

// Texels are independent under filter importance sampling, so the step-up holds the stopped set's false share to 1 - confidence.
void retireConvergedTexels(std::vector<std::uint32_t>& active, const float* replicateLuminance, const PathTraceSettings& settings,
                           ThreadPool& threadPool) {
    const auto stride = static_cast<std::size_t>(settings.replicates);
    const double alpha = 1.0 - static_cast<double>(settings.adaptive.confidence);
    const boost::math::chi_squared_distribution<double> chiSquared(static_cast<double>(settings.replicates) - 1.0);
    // p_(k) <= k alpha / m <= alpha, so only statistics at or below the alpha quantile can be rejected; they rank first among all m.
    const double screen = boost::math::quantile(chiSquared, alpha);
    std::vector<double> statistics(active.size());
    const int chunkTexels = pathTraceChunkTexels(static_cast<int>(active.size()), threadPool.threadCount());
    const int chunks = (static_cast<int>(active.size()) + chunkTexels - 1) / chunkTexels;
    threadPool.parallelFor(chunks, [&](int chunk) {
        const auto begin = static_cast<std::size_t>(chunk) * static_cast<std::size_t>(chunkTexels);
        const std::size_t end = std::min(begin + static_cast<std::size_t>(chunkTexels), active.size());
        for (std::size_t i = begin; i < end; ++i) {
            const ReplicateEstimate estimate = replicateEstimate(replicateLuminance + (active[i] * stride), settings.replicates);
            statistics[i] = convergenceStatistic(estimate, settings.replicates, settings.adaptive.threshold);
        }
    });
    std::vector<double> candidates;
    std::copy_if(statistics.begin(), statistics.end(), std::back_inserter(candidates), [screen](double x) { return x <= screen; });
    // The CDF is monotone, so sorting statistics sorts p-values; it is evaluated only for the screened few.
    std::sort(candidates.begin(), candidates.end());
    std::vector<double> ascendingP(candidates.size());
    std::transform(candidates.begin(), candidates.end(), ascendingP.begin(), [&](double x) { return boost::math::cdf(chiSquared, x); });
    const std::size_t rejected = stepUpRejections(ascendingP, active.size(), alpha);
    if (rejected == 0) {
        return;
    }
    // Every texel at or below the k-th smallest statistic is rejected, ties included, as the procedure defines.
    const double cutoff = candidates[rejected - 1];
    std::size_t kept = 0;
    for (std::size_t i = 0; i < active.size(); ++i) {
        if (statistics[i] > cutoff) {
            active[kept++] = active[i];
        }
    }
    active.resize(kept);
}

PathTraceDriver::PathTraceDriver(const EmbreeAccel& accel,
                                  const std::vector<ShadingTriangle>& shadingTriangles,
                                  const std::vector<MeshInstance>& instances,
                                  const std::vector<int>& instanceLightIndex,
                                  const EnvironmentMap& environmentMap,
                                  const std::vector<AreaLight>& lights,
                                  const VolumeSet& volumes)
    : accel_(accel),
      shadingTriangles_(shadingTriangles),
      instances_(instances),
      instanceLightIndex_(instanceLightIndex),
      environmentMap_(environmentMap),
      lights_(lights),
      volumes_(volumes),
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
std::shared_ptr<PathTraceResult> PathTraceDriver::acquireFreeBuffer(int width, int height, int replicates) {
    for (std::shared_ptr<PathTraceResult>& slot : bufferPool_) {
        if (slot != nullptr && slot.use_count() > 1) {
            continue;
        }
        if (slot == nullptr || slot->beauty.width != width || slot->beauty.height != height || slot->replicates != replicates) {
            slot = std::make_shared<PathTraceResult>(makePathTraceResult(width, height, replicates));
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

// Driver-thread-only. passStats_ and activeTexels_ are read here so the completed and cancelled call sites share one PassRecord.
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
                                            passStats_.chunksCompleted(),
                                            passStats_.chunksCancelled(),
                                            activeTexels_.size(),
                                            cancelled};
    const std::lock_guard<std::mutex> lock(statsMutex_);
    lastPass_ = record;
    untakenPasses_.push_back(record);
}

// Driver-thread-only. Resets the pass counters, builds this pass's LightSet and traces one pass into `pass`; returns the trace time.
double PathTraceDriver::tracePass(const Request& request, int sampleBase, std::uint64_t generation, PathTraceResult& pass) {
    passStats_.reset();
    // Built fresh each pass from this request's env state, holding references not copies, so the HUD toggle needs no invalidation path.
    const LightSet lights(request.envLightEnabled ? &environmentMap_ : nullptr, request.envRotationDegrees, request.envExposure,
                          lights_);
    const auto traceStart = std::chrono::steady_clock::now();
    renderPathTraced(request.camera, accel_, shadingTriangles_, instances_,
                     instanceLightIndex_, lights, volumes_, request.width, request.height, activeTexels_,
                     request.showSky, request.settings,
                     // The generation is the scramble seed: fixed per accumulation, changing exactly when the image restarts.
                     static_cast<std::uint32_t>(generation), sampleBase, request.maxSamples * request.settings.samplesPerPixel,
                     generation_, generation,
                     threadPool_, passStats_, pass);
    return millisecondsSince(traceStart);
}

// Driver-thread-only. Folds a completed pass into the running mean, reduces its over-range stats, then publishes it and its record.
void PathTraceDriver::finishPass(const std::shared_ptr<PathTraceResult>& pass, std::shared_ptr<PathTraceResult>& currentMean,
                                 const PathTraceSettings& settings, int passIndex, std::uint64_t generation, double traceMs,
                                 std::chrono::steady_clock::time_point passStart) {
    // passIndex == 1 leaves the pass as rendered: the running mean of one sample is that sample, and no previous mean exists.
    const auto accumulateStart = std::chrono::steady_clock::now();
    if (passIndex > 1) {
        accumulateMean(*pass, *currentMean, passIndex, activeTexels_, threadPool_);
    } else {
        // Replicate 0's first pass is its mean; the rest start at exactly zero, not at whatever the reused slot held, and are not read yet.
        std::fill(pass->replicateLuminance.begin(), pass->replicateLuminance.end(), 0.0F);
        foldReplicateLuminance(pass->replicateLuminance.data(), pass->beauty.texels.data(),
                               static_cast<std::size_t>(pass->beauty.channels), pass->replicateLuminance.data(),
                               pass->beauty.texels.size() / static_cast<std::size_t>(pass->beauty.channels), pass->replicates,
                               /*replicate=*/0, /*invCount=*/1.0F);
        std::fill(pass->texelSamples.begin(), pass->texelSamples.end(), 1);
    }
    if (adaptiveCheckpoint(passIndex, settings)) {
        retireConvergedTexels(activeTexels_, pass->replicateLuminance.data(), settings, threadPool_);
    }
    pass->activeTexels = activeTexels_.size();
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
            activeTexels_ = allTexels(activeRequest->width, activeRequest->height);  // NOLINT(bugprone-unchecked-optional-access)
        }

        // Engaged: requestedGeneration goes non-zero only inside requestTrace(), under one lock.
        const Request& request = *activeRequest;  // NOLINT(bugprone-unchecked-optional-access)
        // A zero-size view, capped or every texel stopped: nothing changes until a new request or a suspend, both bumping the epoch.
        if (request.width <= 0 || request.height <= 0 || activeTexels_.empty() || (request.maxSamples > 0 && accumulated >= request.maxSamples)) {
            idleUntilWake(stopToken, wakeSeen);
            continue;
        }

        const std::shared_ptr<PathTraceResult> pass = acquireFreeBuffer(request.width, request.height, request.settings.replicates);
        if (pass == nullptr) {
            std::this_thread::sleep_for(kBufferRetryInterval);
            continue;  // every buffer still referenced by the render thread -- retry rather than allocate
        }

        // accumulated passes have drawn accumulated * spp samples, so this pass continues the sequence there and weighs 1/passIndex.
        const int passIndex = accumulated + 1;
        const auto passStart = std::chrono::steady_clock::now();
        const double traceMs = tracePass(request, accumulated * request.settings.samplesPerPixel, activeGeneration, *pass);

        if (generation_.load(std::memory_order_relaxed) != activeGeneration) {
            // Published before the discard, not skipped: a camera drag cancels passes continuously, and their rays were still paid for.
            publishPassRecord(activeGeneration, passIndex, pass->beauty.width, pass->beauty.height,
                               traceMs, 0.0, 0.0, 0.0, millisecondsSince(passStart),
                               /*cancelled=*/true);
            continue;  // superseded mid-pass -- discard, next iteration picks up the new request
        }
        finishPass(pass, currentMean, request.settings, passIndex, activeGeneration, traceMs, passStart);
        accumulated = passIndex;
    }
}

}  // namespace pathtracer::scene
