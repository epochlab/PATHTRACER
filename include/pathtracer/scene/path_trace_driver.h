#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
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

namespace pathtracer::scene {

// Drives renderPathTraced() on a background thread, accumulating short passes into a running mean while the UI stays responsive.
class PathTraceDriver {
public:
    struct Request {
        Camera camera;
        int width = 0;
        int height = 0;
        float envRotationRadians = 0.0F;
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
                     const EnvironmentMap& environmentMap, const std::vector<QuadLight>& quadLights,
                     const std::vector<PathTraceSettings>& perInstanceSettings);
    ~PathTraceDriver();

    PathTraceDriver(const PathTraceDriver&) = delete;
    PathTraceDriver& operator=(const PathTraceDriver&) = delete;
    PathTraceDriver(PathTraceDriver&&) = delete;
    PathTraceDriver& operator=(PathTraceDriver&&) = delete;

    // Render-thread-only. Bumps the generation and replaces the pending request; it does not queue. Returns that new generation.
    std::uint64_t requestTrace(const Request& request);

    // Render-thread-only, at most once per frame. Null until the first pass. One mutex-guarded shared_ptr copy keeps the image alive.
    [[nodiscard]] std::shared_ptr<const PathTraceResult> latestResult() const;

    // Render-thread-only. Parks the driver at the next pass boundary. Non-destructive: generation_ is also the sampler scramble seed.
    void setSuspended(bool suspended);

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
    std::shared_ptr<PathTraceResult> acquireFreeBuffer(int width, int height);
    double tracePass(const Request& request, int sampleBase, std::uint64_t generation, PathTraceResult& pass);
    void finishPass(const std::shared_ptr<PathTraceResult>& pass, std::shared_ptr<PathTraceResult>& currentMean, int passIndex,
                    std::uint64_t generation, double traceMs, std::chrono::steady_clock::time_point passStart);

    const EmbreeAccel& accel_;
    const std::vector<ShadingTriangle>& shadingTriangles_;
    const std::vector<MeshInstance>& instances_;
    const std::vector<int>& instanceLightIndex_;
    const EnvironmentMap& environmentMap_;
    const std::vector<QuadLight>& quadLights_;
    const std::vector<PathTraceSettings>& perInstanceSettings_;

    std::mutex requestMutex_;
    // Camera has no default constructor, so this cannot be a plain Request. nullopt until the first requestTrace().
    std::optional<Request> pendingRequest_;
    // condition_variable_any, not condition_variable: only it takes the jthread stop token, so destruction interrupts the wait.
    std::condition_variable_any wakeCv_;
    // Bumped under requestMutex_ by anything that gives the driver work, so an idle driver waits on it instead of polling.
    std::uint64_t wakeEpoch_ = 0;

    // Bumped by requestTrace, polled lock-free by the dispatch loop and by every in-flight pass's tile workers.
    std::atomic<std::uint64_t> generation_{0};
    // Set by setSuspended; polled by driverLoop, which idles instead of dispatching while it is true.
    std::atomic<bool> suspended_{false};

    // Ray/tile counters for the pass in flight, driver-thread-owned and reused, so a pass allocates nothing. Declared before thread_.
    pathtracer::debug::PassStats passStats_;

    // Republished every finished pass, cancelled included. Separate from resultMutex_ so a dashboard read never contends with publish.
    mutable std::mutex statsMutex_;
    pathtracer::debug::PassRecord lastPass_;
    std::vector<pathtracer::debug::PassRecord> untakenPasses_;  // emptied by takePassRecords each frame, so it holds a frame's worth

    // Republished on every completed pass, guarded by resultMutex_ against latestResult()'s render-thread read.
    mutable std::mutex resultMutex_;
    std::shared_ptr<const PathTraceResult> result_;

    // Driver-thread-only rotation of buffer sets, reused for the process life, so renderPathTraced allocates no images per pass.
    std::array<std::shared_ptr<PathTraceResult>, 4> bufferPool_;

    // Per-chunk private accumulators for reduceOverRange, driver-thread-owned and reused across passes, as passStats_ and bufferPool_ are.
    std::vector<OverRangeHistogram> overRangeHistograms_;
    std::vector<float> overRangePeaks_;

    // Persistent parallel dispatch for renderPathTraced and its accumulate step, declared before thread_ so workers exist before dispatch.
    ThreadPool threadPool_;

    // Declared last: constructed last, so driverLoop starts once every member exists, and destroyed first, jthread stopping and joining.
    std::jthread thread_;
};

// Running mean m_n = m_{n-1} + (x_n - m_{n-1}) * invN over count floats; `mean` may alias either input. HeadlessRenderer shares it.
void foldRunningMean(const float* previousMean, const float* drawn, float* mean, std::size_t count, float invN);

// Welford (1962) M2 of Rec.709 luminance over `pixels` Beauty texels, West's (1979) form; call before foldRunningMean overwrites them.
void foldLuminanceM2(const float* previousMean, const float* drawn, std::size_t channels, const float* previousM2, float* m2,
                     std::size_t pixels, float invN);

}  // namespace pathtracer::scene
