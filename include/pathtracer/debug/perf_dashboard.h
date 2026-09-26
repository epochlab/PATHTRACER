#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

#include "pathtracer/debug/render_stats.h"

namespace pathtracer::debug {

class FrameStats;

// Everything one dashboard redraw shows, bundled by renderFrame as HudFrameData is, so update()'s signature stops growing.
struct DashboardFrame {
    const FrameStats& frameStats;
    const FrameStageTimes& stages;
    const PassRecord& pass;
    // This frame's wall interval, over the same window as the stage times so the reconciliation row compares like with like.
    float frameMs;
    float presentGpuMs;
    int accumulatedSamples;
    int maxSamples;  // 0 = unbounded
    bool driverSuspended;
    std::size_t ramBytes;
    std::size_t gpuBytes;
    std::size_t bvhBytes;
    std::size_t systemAvailableBytes;
    std::uint64_t systemTotalBytes;
    const char* aovName;
    const char* sceneName;
    int instanceCount;
    int lightCount;
    int triangleCount;
    int imageWidth;   // profile.json's authored resolution, what the spec block prints
    int imageHeight;
    int traceWidth;   // imageWidth x renderScale, the last completed pass
    int traceHeight;
    int viewportWidth;  // the window's framebuffer, presentation only
    int viewportHeight;
    float renderScale;
    bool interactiveScale;
    double refreshHz;  // the window's display, measured by DisplayLink
    bool vsync;        // profile.json frame cap; the budget stays the refresh period either way
};

// Live terminal dashboard: a fixed-height block redrawn in place at kRefreshHz, render-thread only. Plain summaries off a TTY.
class PerfDashboard {
public:
    // 3 Hz: the draw is a blocking write(2) after swapBuffers, so it eats the next frame's headroom. Its cost is measured, not hidden.
    static constexpr int kRefreshHz = 3;
    static constexpr double kNonTtySeconds = 5.0;
    static constexpr std::size_t kBufferBytes = 8192;
    // Lines the body occupies, only to reserve the block on the first draw; later redraws move up by what the previous draw emitted.
    static constexpr int kBodyLines = 30;

    PerfDashboard();

    // Once per frame at the end of renderFrame. Rate-limited internally: between redraws this is one now(), a compare and the adds.
    void update(const DashboardFrame& frame);

private:
    void accumulate(const DashboardFrame& frame);
    void draw(const DashboardFrame& frame);
    // Split only to keep each part inside the one-screen function limit, by row group rather than column: the layout's own boundaries.
    void drawFrameHeader(const DashboardFrame& frame);
    void drawStageRows(const DashboardFrame& frame);
    void drawRayRows(const DashboardFrame& frame);
    void drawFooter(const DashboardFrame& frame);

    // Per-frame mean over the current window. A helper rather than a local so every row divides by the same n, whichever function emits it.
    [[nodiscard]] float windowMean(float sum) const;
    [[nodiscard]] float cpuTotalMs() const;
    [[nodiscard]] double burstDutyFrames(int stage) const;
    void drawNonTty(const DashboardFrame& frame);
    void append(const char* format, ...) __attribute__((format(printf, 2, 3)));
    void flush();

    // Windowed means of every per-frame stage, so the display is stable rather than flickering at the last frame's cost. Reset per redraw.
    FrameStageTimes sums_{};
    float cpuTotalSum_ = 0.0F;
    float frameMsSum_ = 0.0F;
    int windowFrames_ = 0;

    // This dashboard's own draw cost, measured around the write(2) and shown on the NEXT redraw -- the only point at which it is known.
    float lastDrawMs_ = 0.0F;
    // The same cost, held until one frame has charged for it: a redraw lands on ~1 frame in 20, so billing every frame inflates ~20x.
    float unbilledDrawMs_ = 0.0F;

    // Bursty stages run on few frames: a per-frame mean reads ~1.7ms for a 150ms stall, so these report over the frames they ran on.
    std::array<float, 3> burstLastMs_{};
    std::array<std::uint64_t, 3> burstFireCount_{};
    std::uint64_t totalFrames_ = 0;

    bool tty_ = false;
    // Lines the previous draw emitted, which is what the next cursor-up must move by. Derived, since a mismatch walks the block up-screen.
    std::size_t lastLineCount_ = 0;
    std::chrono::steady_clock::time_point lastDraw_{};
    std::array<char, kBufferBytes> buffer_{};
    std::size_t used_ = 0;
    std::size_t lines_ = 0;
};

}  // namespace pathtracer::debug
