#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <vector>

#include "pathtracer/debug/render_stats.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/environment_map.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/light.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/thread_pool.h"
#include "pathtracer/scene/volume.h"

namespace pathtracer::scene {

// Drives renderPathTraced() on a background thread, accumulating short passes into a running mean while the UI stays responsive.
class PathTraceDriver {
public:
    struct Request {
        Camera camera;
        int width = 0;
        int height = 0;
        glm::vec3 envRotationDegrees{0.0F};  // rotationXyz of the environment map to world
        bool showSky = kDefaultShowSky;
        // Whether the environment is in the light set at all (NEE, MIS, miss radiance); showSky gates only the camera ray's own miss.
        bool envLightEnabled = true;
        float envExposure = 1.0F;
        PathTraceSettings settings;  // samplesPerPixel is "samples per pass", see path_tracer.h
        int maxSamples = 0;  // accumulated-pass cap; 0 = unbounded
    };

    // Every referenced scene object must be at its final address and outlive the driver: LightSet stores a pointer, not a copy.
    PathTraceDriver(const EmbreeAccel& accel, const std::vector<ShadingTriangle>& shadingTriangles,
                     const std::vector<MeshInstance>& instances,
                     const std::vector<int>& instanceLightIndex,
                     const EnvironmentMap& environmentMap, const std::vector<AreaLight>& lights, const VolumeSet& volumes);
    ~PathTraceDriver();

    PathTraceDriver(const PathTraceDriver&) = delete;
    PathTraceDriver& operator=(const PathTraceDriver&) = delete;
    PathTraceDriver(PathTraceDriver&&) = delete;
    PathTraceDriver& operator=(PathTraceDriver&&) = delete;

    // Render-thread-only. Bumps the generation and replaces the pending request; it does not queue. Returns that new generation.
    std::uint64_t requestTrace(const Request& request);

    // Render-thread-only, at most once per frame, null until the first pass. Holding two results at once can stall the driver.
    [[nodiscard]] std::shared_ptr<const PathTraceResult> latestResult() const;

    // Render-thread-only. Parks the driver at the next pass boundary. Non-destructive: generation_ is also the sampler scramble seed.
    void setSuspended(bool suspended);

    // Render-thread-only. While set only a finished accumulation is published; every pass still folds, so a switch never restarts one.
    void setPublishesFinalOnly(bool finalOnly);

    // Render-thread-only. The most recent pass's timings and ray counts, cancelled included; generation is 0 before the first pass.
    [[nodiscard]] pathtracer::debug::PassRecord lastPassRecord() const;

    // Render-thread-only, once per frame. Every record since the last call, in pass order: several passes can finish within one frame.
    [[nodiscard]] std::vector<pathtracer::debug::PassRecord> takePassRecords();

private:
    void driverLoop(std::stop_token stopToken);
    // Blocks until requestTrace or setSuspended bumps wakeEpoch_ past `seen`, or the driver stops. Replaces polling for new work.
    void idleUntilWake(const std::stop_token& stopToken, std::uint64_t seen);
    // width/height come from the rendered buffer, not the Request that asked: the numbers describing the image that actually exists.
    void publishPassRecord(std::uint64_t generation, int passIndex, int width, int height,
                            double traceMs, double accumulateMs, double overRangeMs,
                            double publishMs, double passMs, bool cancelled);
    std::shared_ptr<PathTraceResult> acquireFreeBuffer(int width, int height, int replicates);
    double tracePass(const Request& request, int sampleBase, std::uint64_t generation, PathTraceResult& pass);
    void finishPass(const std::shared_ptr<PathTraceResult>& pass, std::shared_ptr<PathTraceResult>& currentMean,
                    const Request& request, int passIndex, std::uint64_t generation, double traceMs,
                    std::chrono::steady_clock::time_point passStart);

    const EmbreeAccel& accel_;
    const std::vector<ShadingTriangle>& shadingTriangles_;
    const std::vector<MeshInstance>& instances_;
    const std::vector<int>& instanceLightIndex_;
    const EnvironmentMap& environmentMap_;
    const std::vector<AreaLight>& lights_;
    const VolumeSet& volumes_;

    std::mutex requestMutex_;
    // Camera has no default constructor, so this cannot be a plain Request. nullopt until the first requestTrace().
    std::optional<Request> pendingRequest_;
    // condition_variable_any, not condition_variable: only it takes the jthread stop token, so destruction interrupts the wait.
    std::condition_variable_any wakeCv_;
    // Bumped under requestMutex_ by anything that gives the driver work, so an idle driver waits on it instead of polling.
    std::uint64_t wakeEpoch_ = 0;

    // Bumped by requestTrace, polled lock-free by the dispatch loop and by every in-flight pass's chunk workers.
    std::atomic<std::uint64_t> generation_{0};
    // Set by setSuspended; polled by driverLoop, which idles instead of dispatching while it is true.
    std::atomic<bool> suspended_{false};
    // Set by setPublishesFinalOnly; read once per finished pass, so a switch takes effect at the next pass boundary.
    std::atomic<bool> publishesFinalOnly_{false};

    // Ray/chunk counters for the pass in flight, driver-thread-owned and reused, so a pass allocates nothing. Declared before thread_.
    pathtracer::debug::PassStats passStats_;

    // Republished every finished pass, cancelled included. Separate from resultMutex_ so a dashboard read never contends with publish.
    mutable std::mutex statsMutex_;
    pathtracer::debug::PassRecord lastPass_;
    std::vector<pathtracer::debug::PassRecord> untakenPasses_;  // emptied by takePassRecords each frame, so it holds a frame's worth

    // Republished on every completed pass, guarded by resultMutex_ against latestResult()'s render-thread read.
    mutable std::mutex resultMutex_;
    std::shared_ptr<const PathTraceResult> result_;

    // Holders of a slot at once: the driver's mean being folded from and pass being written, and the reader's one held result.
    static constexpr std::size_t kDriverHeldBuffers = 2;
    static constexpr std::size_t kReaderHeldBuffers = 1;
    // Driver-thread-only rotation of buffer sets, reused for the process life, so renderPathTraced allocates no images per pass.
    std::array<std::shared_ptr<PathTraceResult>, kDriverHeldBuffers + kReaderHeldBuffers> bufferPool_;

    // Driver-thread-only: ascending texels the next pass traces, every one at a generation's first pass.
    std::vector<std::uint32_t> activeTexels_;

    // Per-chunk private accumulators for reduceOverRange, driver-thread-owned and reused across passes, as passStats_ and bufferPool_ are.
    std::vector<OverRangeHistogram> overRangeHistograms_;
    std::vector<float> overRangePeaks_;

    // Persistent parallel dispatch for renderPathTraced and its accumulate step, declared before thread_ so workers exist before dispatch.
    ThreadPool threadPool_;

    // Declared last: constructed last, so driverLoop starts once every member exists, and destroyed first, jthread stopping and joining.
    std::jthread thread_;
};

// Finished at the pass cap (0 = unbounded) or once adaptive sampling has stopped every texel; the driver then idles.
[[nodiscard]] constexpr bool accumulationComplete(int samples, std::size_t activeTexels, int maxSamples) {
    return activeTexels == 0 || (maxSamples > 0 && samples >= maxSamples);
}

// Running mean m_n = m_{n-1} + (x_n - m_{n-1}) * invN over count floats; `mean` may alias either input. HeadlessRenderer shares it.
void foldRunningMean(const float* previousMean, const float* drawn, float* mean, std::size_t count, float invN);

// Folds drawn Beauty's scene luminance into one replicate's running mean, carrying the rest; means may alias previousMeans.
void foldReplicateLuminance(const float* previousMeans, const float* drawn, std::size_t channels, float* means, std::size_t pixels,
                            int replicates, int replicate, float invCount);

// One texel's replicated estimate of mean scene luminance and the variance of that mean.
struct ReplicateEstimate {
    double mean;
    double meanVariance;  // sum (m_r - mbar)^2 / (K(K-1)), unbiased for Var(mbar) at any replicate counts
};

// The estimate from `populated` replicate means, in double: a converged texel's deviations are rounding-scale differences of floats.
[[nodiscard]] ReplicateEstimate replicateEstimate(const float* means, int populated);

// (K-1) s^2 / (threshold mean)^2, chi-squared on K-1 dof where SD(mbar) = threshold * mean; +inf for zero spread, zero mean or NaN.
[[nodiscard]] double convergenceStatistic(const ReplicateEstimate& estimate, int replicates, float threshold);

// Benjamini & Hochberg 1995 step-up: how many of the smallest of `tests` p-values, given ascending, are rejected at rate alpha.
[[nodiscard]] std::size_t stepUpRejections(std::span<const double> ascendingP, std::size_t tests, double alpha);

// Whether pass n is a checkpoint, n = K 2^j >= minSamples, where every replicate holds a power-of-two prefix of its own sequence.
[[nodiscard]] bool adaptiveCheckpoint(int n, const PathTraceSettings& settings);

// Foveal minimum angle of resolution, degrees: the intercept Guenter et al. 2012 fit their linear MAR model w = w0 + m e with.
inline constexpr float kFovealMarDegrees = 1.0F / 48.0F;

// Per-texel factor on the adaptive threshold for an observer at the camera fixating the frame centre (Guenter et al. 2012).
class FoveatedThreshold {
public:
    // marSlope 0 is the uniform threshold; the camera must outlive this.
    FoveatedThreshold(const Camera& camera, int width, int height, float marSlope);

    // sqrt(N / N_gaze): the MAR w(e) pools N = max(1, w^2 / Omega) texels of solid angle Omega, dividing their noise by sqrt(N).
    [[nodiscard]] float scale(std::uint32_t texel) const;

private:
    // The MAR in texel sides, max(1, w / sqrt(Omega)), for a texel-centre ray and its differential; 1 where no ray exists.
    [[nodiscard]] float pooledSpan(const std::optional<Camera::RayDifferential>& ray) const;

    const Camera& camera_;
    pathtracer::gfx::PixelLattice lattice_;
    glm::vec2 frame_;
    Camera::ViewBasis basis_;
    glm::mat2 ndcPerPixel_;
    std::uint32_t width_;
    float marSlope_;
    glm::vec3 gaze_{0.0F};
    float gazeSpan_ = 1.0F;
};

// Drops the texels of ascending `active` the step-up procedure finds converged at 1 - confidence each at its foveated threshold.
void retireConvergedTexels(std::vector<std::uint32_t>& active, const float* replicateLuminance, const PathTraceSettings& settings,
                           const FoveatedThreshold& fovea, ThreadPool& threadPool);

// Splits [begin, end) into runs of consecutive texels in ascending `traced`, passed to fold(b, e), and the gaps, passed to carry(b, e).
template <typename Fold, typename Carry>
void forEachTracedRun(std::span<const std::uint32_t> traced, std::uint32_t begin, std::uint32_t end, Fold&& fold, Carry&& carry) {
    auto it = std::lower_bound(traced.begin(), traced.end(), begin);
    std::uint32_t cursor = begin;
    while (it != traced.end() && *it < end) {
        const std::uint32_t runBegin = *it;
        std::uint32_t runEnd = runBegin + 1;
        for (++it; it != traced.end() && *it == runEnd && runEnd < end; ++it) {
            ++runEnd;
        }
        if (cursor < runBegin) {
            carry(cursor, runBegin);
        }
        fold(runBegin, runEnd);
        cursor = runEnd;
    }
    if (cursor < end) {
        carry(cursor, end);
    }
}

}  // namespace pathtracer::scene
