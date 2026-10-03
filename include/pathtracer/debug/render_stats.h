#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace pathtracer::debug {

// Ray counts per path/tile/pass, never atomic. tracePath takes `RayCounts& __restrict`: +4.6% by value, +0.3% by plain reference (noise).
struct RayCounts {
    std::uint64_t primary = 0;  // one per sample: the camera ray at bounce 0
    std::uint64_t bounce = 0;   // BSDF-sampled continuation rays, bounce >= 1
    std::uint64_t ao = 0;       // one bounded occlusion ray per sample
    std::uint64_t shadow = 0;   // NEE occlusion rays, one per light sample

    void add(const RayCounts& other) {
        primary += other.primary;
        bounce += other.bounce;
        ao += other.ao;
        shadow += other.shadow;
    }

    [[nodiscard]] std::uint64_t total() const { return primary + bounce + ao + shadow; }
};

// Concurrent ray/tile accumulator for one renderPathTraced() call. Workers merge once per tile: a fetch_add per ray is ~15M contended RMWs.
class PassStats {
public:
    // Driver thread only, before dispatch.
    void reset() {
        primary_.store(0, std::memory_order_relaxed);
        bounce_.store(0, std::memory_order_relaxed);
        ao_.store(0, std::memory_order_relaxed);
        shadow_.store(0, std::memory_order_relaxed);
        tilesCompleted_.store(0, std::memory_order_relaxed);
        tilesCancelled_.store(0, std::memory_order_relaxed);
    }

    // Worker threads, once per completed tile.
    void addTile(const RayCounts& counts) {
        primary_.fetch_add(counts.primary, std::memory_order_relaxed);
        bounce_.fetch_add(counts.bounce, std::memory_order_relaxed);
        ao_.fetch_add(counts.ao, std::memory_order_relaxed);
        shadow_.fetch_add(counts.shadow, std::memory_order_relaxed);
        tilesCompleted_.fetch_add(1, std::memory_order_relaxed);
    }

    // Worker threads, on the stale-generation early-out: no rays, but still counted, or a cancelled pass reads as a short one.
    void addCancelledTile() { tilesCancelled_.fetch_add(1, std::memory_order_relaxed); }

    // Valid only once parallelFor has returned -- see the ordering argument above.
    [[nodiscard]] RayCounts rays() const {
        return {primary_.load(std::memory_order_relaxed), bounce_.load(std::memory_order_relaxed),
                ao_.load(std::memory_order_relaxed), shadow_.load(std::memory_order_relaxed)};
    }

    [[nodiscard]] std::uint64_t tilesCompleted() const { return tilesCompleted_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t tilesCancelled() const { return tilesCancelled_.load(std::memory_order_relaxed); }

private:
    std::atomic<std::uint64_t> primary_{0};
    std::atomic<std::uint64_t> bounce_{0};
    std::atomic<std::uint64_t> ao_{0};
    std::atomic<std::uint64_t> shadow_{0};
    std::atomic<std::uint64_t> tilesCompleted_{0};
    std::atomic<std::uint64_t> tilesCancelled_{0};
};

// Immutable snapshot of one finished or cancelled pass, copied wholesale under statsMutex_ exactly as result_ is under resultMutex_.
struct PassRecord {
    std::uint64_t generation = 0;  // which request this pass belonged to; 0 = no pass has run yet
    int passIndex = 0;             // n, the accumulated sample count this pass produced
    int width = 0;
    int height = 0;
    double traceMs = 0.0;       // renderPathTraced
    double accumulateMs = 0.0;  // accumulateMean
    double overRangeMs = 0.0;   // reduceOverRange, the HUD's over-range statistics
    double publishMs = 0.0;     // resultMutex_ hold
    double passMs = 0.0;        // the whole iteration, including buffer acquisition
    RayCounts rays;
    std::uint64_t tilesCompleted = 0;
    std::uint64_t tilesCancelled = 0;
    bool cancelled = false;  // superseded mid-flight and discarded; its rays were still traced and paid for
};

// Per-frame render-thread stage times, ms; one thread writes and reads, so unsynchronized. Zeroed each frame, so a skipped stage reads 0.
struct FrameStageTimes {
    float fenceMs = 0.0F;       // wait for the previous frame's GPU work: non-zero only when the GPU, not the display, bounds the frame
    float paceMs = 0.0F;        // DisplayLink::waitForNextVblank: slack, not engine cost
    float pollMs = 0.0F;
    float cameraMs = 0.0F;
    float gbufferMs = 0.0F;     // renderGBuffer, only on a trigger change into a G-buffer AOV
    float filterMs = 0.0F;      // evaluateFilterAov, only when a newly published pass invalidates the filter cache
    float uploadMs = 0.0F;      // the display texture upload, only when a newly published pass invalidates it
    bool uploaded = false;      // the upload ran this frame: an explicit event flag, since a timed stage can legitimately read 0
    float presentMs = 0.0F;     // presentFrame, INCLUSIVE of uploadMs and filterMs -- the blit's own cost is the difference
    float histogramMs = 0.0F;
    float overRangeMs = 0.0F;   // the O(kOverRangeBinCount) read of the driver's reduction, every frame
    float probeMs = 0.0F;       // samplePixelProbe: one HdrImage texel fetch, the filter cache already warm from presentFrame
    float hudMs = 0.0F;         // HUD draw + camera write-back + render, INCLUSIVE of hudRenderMs -- the build half is the difference
    float hudRenderMs = 0.0F;   // HudOverlay::render (ImGui::Render + RenderDrawData), unconditional so it is paid with the HUD hidden
    // swapBuffers at swap interval 0: the hand-off to the compositor, no vblank wait. Its tail is WindowServer latency, not engine time.
    float swapMs = 0.0F;
};

// RAII steady_clock scope timer into a caller-owned float. Serial contexts only, and never per-pixel or per-ray: now() is ~20ns.
class ScopedCpuTimer {
public:
    explicit ScopedCpuTimer(float& outMs) : outMs_(outMs), start_(std::chrono::steady_clock::now()) {}

    ~ScopedCpuTimer() {
        outMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

    ScopedCpuTimer(const ScopedCpuTimer&) = delete;
    ScopedCpuTimer& operator=(const ScopedCpuTimer&) = delete;
    ScopedCpuTimer(ScopedCpuTimer&&) = delete;
    ScopedCpuTimer& operator=(ScopedCpuTimer&&) = delete;

private:
    float& outMs_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace pathtracer::debug
