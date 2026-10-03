#include "pathtracer/debug/perf_dashboard.h"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pathtracer/debug/frame_stats.h"

namespace pathtracer::debug {

namespace {

constexpr double kMiB = 1024.0 * 1024.0;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kMillion = 1.0e6;

// Index into PerfDashboard's burst arrays. Bursty means it runs on a small fraction of frames, on a trigger change or a republish.
enum BurstStage { kBurstGBuffer = 0, kBurstUpload = 1, kBurstFilter = 2 };

// Frames per firing, "1/88", in the column a per-frame stage puts its percentage. "never" is a real state, not a missing measurement.
void formatDuty(std::array<char, 8>& out, double framesPerFiring) {
    if (framesPerFiring > 0.0) {
        std::snprintf(out.data(), out.size(), "1/%.0f", framesPerFiring);
    } else {
        std::snprintf(out.data(), out.size(), "never");
    }
}

using Bar = std::array<char, 64>;
constexpr int kBarCells = 10;
constexpr const char* kBarBlank = "          ";

// Ten cells at 1/8-cell resolution (U+2588 to U+258F), so a stage worth 1.2% still marks where a whole-block bar would round away.
constexpr std::array<const char*, 9> kBlocks{" ", "▏", "▎", "▍", "▌",
                                              "▋", "▊", "▉", "█"};

void formatBar(Bar& out, double fraction) {
    const int eighths =
        std::clamp(static_cast<int>(std::lround(fraction * kBarCells * 8)), 0, kBarCells * 8);
    std::size_t used = 0;
    const auto put = [&](const char* text) {
        while (*text != '\0' && used + 1 < out.size()) {
            out[used++] = *text++;
        }
    };
    for (int cell = 0; cell < kBarCells; ++cell) {
        put(kBlocks[static_cast<std::size_t>(std::clamp(eighths - cell * 8, 0, 8))]);
    }
    out[used] = '\0';
}

// Display range of the residual bar, not a threshold: a healthy reconciliation sits under 1%, invisible against a 100% scale.
constexpr double kResidualFullScale = 0.10;

// Diverging bar: the residual is the one row that can fall either side of zero, which a left-aligned bar would draw identically.
void formatSignedBar(Bar& out, double fraction) {
    constexpr int kHalfCells = kBarCells / 2;
    const int eighths =
        std::clamp(static_cast<int>(std::lround(fraction * kHalfCells * 8)), -kHalfCells * 8,
                    kHalfCells * 8);
    std::size_t used = 0;
    const auto put = [&](const char* text) {
        while (*text != '\0' && used + 1 < out.size()) {
            out[used++] = *text++;
        }
    };
    const int wholeLeft = eighths < 0 ? (-eighths + 7) / 8 : 0;
    for (int cell = 0; cell < kHalfCells; ++cell) {
        put(cell >= kHalfCells - wholeLeft ? "█" : " ");
    }
    for (int cell = 0; cell < kHalfCells; ++cell) {
        put(kBlocks[static_cast<std::size_t>(std::clamp(eighths - cell * 8, 0, 8))]);
    }
    out[used] = '\0';
}

float percentOf(float value, float total) { return total > 0.0F ? (value / total) * 100.0F : 0.0F; }

double rate(std::uint64_t count, double milliseconds) {
    return milliseconds > 0.0 ? static_cast<double>(count) / (milliseconds / 1000.0) : 0.0;
}

}  // namespace

// isatty on the descriptor actually written, not std::cout: a shell redirect changes one and says nothing about the other.
PerfDashboard::PerfDashboard() : lastDraw_(std::chrono::steady_clock::now()) {
    const char* term = std::getenv("TERM");
    tty_ = isatty(STDOUT_FILENO) == 1 && (term == nullptr || std::strcmp(term, "dumb") != 0);
}

void PerfDashboard::append(const char* format, ...) {
    if (used_ >= buffer_.size()) {
        return;  // full: drop the rest of this frame's text rather than truncate mid-escape-sequence
    }
    std::va_list args;
    va_start(args, format);
    const int written = std::vsnprintf(buffer_.data() + used_, buffer_.size() - used_, format, args);
    va_end(args);
    if (written <= 0) {
        return;
    }
    // vsnprintf returns what it WOULD have written, so it must be clamped before it advances the cursor past the end.
    const std::size_t advance = std::min(static_cast<std::size_t>(written), buffer_.size() - used_ - 1);
    lines_ += static_cast<std::size_t>(
        std::count(buffer_.data() + used_, buffer_.data() + used_ + advance, '\n'));
    used_ += advance;
}

// One write(2) of the whole block: iostream can allocate through locale, and a partially flushed frame tears visibly.
void PerfDashboard::flush() {
    if (used_ > 0) {
        [[maybe_unused]] const ssize_t ignored = ::write(STDOUT_FILENO, buffer_.data(), used_);
    }
    used_ = 0;
}

void PerfDashboard::accumulate(const DashboardFrame& frame) {
    const FrameStageTimes& stages = frame.stages;
    sums_.fenceMs += stages.fenceMs;
    sums_.paceMs += stages.paceMs;
    sums_.pollMs += stages.pollMs;
    sums_.cameraMs += stages.cameraMs;
    sums_.presentMs += stages.presentMs;
    sums_.histogramMs += stages.histogramMs;
    sums_.probeMs += stages.probeMs;
    sums_.hudMs += stages.hudMs;
    sums_.hudRenderMs += stages.hudRenderMs;
    sums_.swapMs += stages.swapMs;
    sums_.uploadMs += stages.uploadMs;
    sums_.gbufferMs += stages.gbufferMs;
    sums_.filterMs += stages.filterMs;
    sums_.overRangeMs += stages.overRangeMs;
    // Every stage, bursty included: this must reconcile against the measured frame time, so it cannot exclude the costliest.
    cpuTotalSum_ += stages.fenceMs + stages.paceMs + stages.pollMs + stages.cameraMs + stages.gbufferMs + stages.presentMs +
                     stages.histogramMs + stages.overRangeMs + stages.probeMs + stages.hudMs +
                     stages.swapMs + unbilledDrawMs_;
    unbilledDrawMs_ = 0.0F;  // charged exactly once, to the frame that actually paid it
    frameMsSum_ += frame.frameMs;
    ++windowFrames_;
    ++totalFrames_;

    // A stage that did not run reads exactly 0, so a non-zero value is a firing and the value kept is real, not a diluted mean.
    const std::array<float, 3> burst{stages.gbufferMs, stages.uploadMs, stages.filterMs};
    for (std::size_t i = 0; i < burst.size(); ++i) {
        if (burst[i] > 0.0F) {
            burstLastMs_[i] = burst[i];
            ++burstFireCount_[i];
        }
    }
}

void PerfDashboard::update(const DashboardFrame& frame) {
    accumulate(frame);

    const auto now = std::chrono::steady_clock::now();
    const double sinceDraw = std::chrono::duration<double>(now - lastDraw_).count();
    const double interval = tty_ ? 1.0 / static_cast<double>(kRefreshHz) : kNonTtySeconds;
    if (sinceDraw < interval) {
        return;
    }
    lastDraw_ = now;

    {
        const ScopedCpuTimer drawTimer(lastDrawMs_);
        if (tty_) {
            draw(frame);
        } else {
            drawNonTty(frame);
        }
    }

    unbilledDrawMs_ = lastDrawMs_;

    sums_ = {};
    cpuTotalSum_ = 0.0F;
    frameMsSum_ = 0.0F;
    windowFrames_ = 0;
}

void PerfDashboard::drawNonTty(const DashboardFrame& frame) {
    const auto& pass = frame.pass;
    append("pathtracer perf: frame p50 %.2fms p95 %.2fms | pass #%d %.1fms %.2f Mray/s | rss %.2f GiB\n",
            static_cast<double>(frame.frameStats.percentileMs(0.5F)),
            static_cast<double>(frame.frameStats.percentileMs(0.95F)), pass.passIndex, pass.traceMs,
            rate(pass.rays.total(), pass.traceMs) / kMillion,
            static_cast<double>(frame.ramBytes) / kGiB);
    flush();
}

float PerfDashboard::windowMean(float sum) const {
    return sum / static_cast<float>(std::max(windowFrames_, 1));
}

float PerfDashboard::cpuTotalMs() const { return windowMean(cpuTotalSum_); }

// Frames per firing since launch. Never fired returns 0, rendered as "never" rather than divided by a floor of 1.
double PerfDashboard::burstDutyFrames(int stage) const {
    const std::uint64_t fires = burstFireCount_[static_cast<std::size_t>(stage)];
    return fires > 0 ? static_cast<double>(totalFrames_) / static_cast<double>(fires) : 0.0;
}

void PerfDashboard::draw(const DashboardFrame& frame) {
    lines_ = 0;
    if (lastLineCount_ == 0) {
        // Reserve the block on the first draw so the cursor-up always has real lines to move over, not whatever was on screen.
        for (int i = 0; i < kBodyLines; ++i) {
            append("\n");
        }
        lastLineCount_ = lines_;
        flush();
        lines_ = 0;
    }
    // Up by exactly what the previous draw emitted, erasing each line as it is rewritten. Never \x1b[2J: that destroys scrollback.
    append("\x1b[%zuA", lastLineCount_);

    drawFrameHeader(frame);
    drawStageRows(frame);
    drawRayRows(frame);
    drawFooter(frame);
    // The first draw reserves kBodyLines as an estimate; if the real block is shorter, shed lines are erased and kept blank.
    while (lines_ < lastLineCount_) {
        append("\x1b[2K\n");
    }

    lastLineCount_ = lines_;
    flush();
}

// A fixed 78-column grid, not eyeballed: left pane columns 0-42, divider at 43 on every split row, right pane 44-77.
void PerfDashboard::drawFrameHeader(const DashboardFrame& frame) {
    const double budgetMs = 1000.0 / frame.refreshHz;
    append("\x1b[2K== PATHTRACER PERF ============================================== %5.1f fps ==\n",
            static_cast<double>(frame.frameStats.fps()));
    // Percentiles lead, mean trails: a mean hides the hitch. p95 not p99, because 120 samples cannot express a 99th percentile.
    append("\x1b[2K frame    p50 %6.2f   p95 %6.2f   max %6.2f   mean %6.2f ms   (%3d frames)\n",
            static_cast<double>(frame.frameStats.percentileMs(0.5F)),
            static_cast<double>(frame.frameStats.percentileMs(0.95F)),
            static_cast<double>(frame.frameStats.maxMs()),
            static_cast<double>(windowMean(frameMsSum_)), FrameStats::kHistoryLength);
    // The vblank wait is the headroom itself, so it is excluded from the work; the fence wait is GPU time the frame did spend.
    append("\x1b[2K budget   %6.2f ms @ %6.2f Hz %-8s                  headroom %8.2f ms\n",
            budgetMs, frame.refreshHz, frame.vsync ? "vsync" : "uncapped", budgetMs - static_cast<double>(cpuTotalMs() - windowMean(sums_.paceMs)));
    append("\x1b[2K------------------------------------------------------------------------------\n");
}

// A per-frame stage's left pane: its windowed cost, share of the CPU total and bar, up to the divider. The caller appends the right pane.
void PerfDashboard::appendStageCell(const char* name, double ms, float cpu) {
    const float pct = percentOf(static_cast<float>(ms), cpu);
    Bar bar{};
    formatBar(bar, static_cast<double>(pct) / 100.0);
    append("\x1b[2K  %-15s%8.3f %5.1f %s |", name, ms, static_cast<double>(pct), bar.data());
}

// Bursty stages print their last real cost and how often they fire: a 150ms stall 1 frame in 88 averages to a harmless 1.7ms.
void PerfDashboard::appendBurstCell(const char* name, int stage, int precision) {
    std::array<char, 8> duty{};
    formatDuty(duty, burstDutyFrames(stage));
    append("\x1b[2K  %-15s%8.*f %5s %s |", name, precision, static_cast<double>(burstLastMs_[static_cast<std::size_t>(stage)]),
           duty.data(), kBarBlank);
}

void PerfDashboard::drawStageRows(const DashboardFrame& frame) {
    const float cpu = cpuTotalMs();
    const auto& pass = frame.pass;
    const double passMs = pass.traceMs + pass.accumulateMs + pass.overRangeMs + pass.publishMs;
    const auto phasePct = [&](double ms) { return passMs > 0.0 ? (ms / passMs) * 100.0 : 0.0; };
    const auto mean = [&](float sum) { return static_cast<double>(windowMean(sum)); };
    // presentMs and hudMs each measure a nested stage too, so the outer half's own cost is the difference -- arithmetic, not a probe.
    const double blitMs = std::max(mean(sums_.presentMs) - mean(sums_.uploadMs) - mean(sums_.filterMs), 0.0);
    const double hudBuildMs = std::max(mean(sums_.hudMs) - mean(sums_.hudRenderMs), 0.0);

    append("\x1b[2K RENDER THREAD     cpu ms     %%            | PATH TRACE (driver)     ms      %%\n");
    appendStageCell("poll", mean(sums_.pollMs), cpu);
    append("  pass #%-4d gen %-6llu %4dx%-4d \n", pass.passIndex, static_cast<unsigned long long>(pass.generation), pass.width,
           pass.height);
    appendStageCell("camera", mean(sums_.cameraMs), cpu);
    append("  trace           %9.2f %6.1f\n", pass.traceMs, phasePct(pass.traceMs));
    appendBurstCell("gbuffer", kBurstGBuffer, 2);
    append("  accumulate      %9.2f %6.1f\n", pass.accumulateMs, phasePct(pass.accumulateMs));
    appendBurstCell("tex upload", kBurstUpload, 3);
    append("  over-range      %9.2f %6.1f\n", pass.overRangeMs, phasePct(pass.overRangeMs));
    appendStageCell("present blit", blitMs, cpu);
    append("  publish         %9.2f %6.1f\n", pass.publishMs, phasePct(pass.publishMs));
    appendStageCell("histogram", mean(sums_.histogramMs), cpu);
    append("  --------------------------------\n");
    appendStageCell("over-range", mean(sums_.overRangeMs), cpu);
    append("  tiles     %5llu / %-5llu %-6s\n", static_cast<unsigned long long>(pass.tilesCompleted),
           static_cast<unsigned long long>(pass.tilesCompleted + pass.tilesCancelled), pass.cancelled ? "CANCEL" : "");
    appendStageCell("pixel probe", mean(sums_.probeMs), cpu);
    append("  samples   %5d / %s\n", frame.accumulatedSamples, frame.maxSamples > 0 ? "capped" : "unbounded");
    appendStageCell("hud build", hudBuildMs, cpu);
    append("  suspended %s\n", frame.driverSuspended ? "yes" : "no");
    appendBurstCell("aov filter", kBurstFilter, 2);
    append("\n");
    appendStageCell("hud render", mean(sums_.hudRenderMs), cpu);
    append("\n");
    appendStageCell("gpu wait", mean(sums_.fenceMs), cpu);
    append("\n");
    appendStageCell("vsync wait", mean(sums_.paceMs), cpu);
    append("\n");
}

void PerfDashboard::drawRayRows(const DashboardFrame& frame) {
    const float cpu = cpuTotalMs();
    const float frameMs = windowMean(frameMsSum_);
    const auto& pass = frame.pass;
    const auto& rays = pass.rays;
    const double total = static_cast<double>(rays.total());
    const auto share = [&](std::uint64_t n) { return total > 0.0 ? (static_cast<double>(n) / total) * 100.0 : 0.0; };
    const auto mray = [&](std::uint64_t n) { return rate(n, pass.traceMs) / kMillion; };
    const auto millions = [](std::uint64_t n) { return static_cast<double>(n) / kMillion; };
    Bar bar{};

    append("\x1b[2K  %-15s%8.3f %5s %s | RAYS          count  Mray/s     %%\n", "dashboard",
            static_cast<double>(lastDrawMs_), "", kBarBlank);
    formatBar(bar, static_cast<double>(percentOf(windowMean(sums_.swapMs), cpu)) / 100.0);
    append("\x1b[2K  %-15s%8.3f %5.1f %s |  %-9s%7.3f M %7.2f %5.1f\n", "swap",
            static_cast<double>(windowMean(sums_.swapMs)),
            static_cast<double>(percentOf(windowMean(sums_.swapMs), cpu)), bar.data(), "primary",
            millions(rays.primary), mray(rays.primary), share(rays.primary));
    append("\x1b[2K  ---------------------------------------- |  %-9s%7.3f M %7.2f %5.1f\n",
            "bounce", millions(rays.bounce), mray(rays.bounce), share(rays.bounce));
    append("\x1b[2K  %-15s%8.3f %5.1f %s |  %-9s%7.3f M %7.2f %5.1f\n", "cpu total",
            static_cast<double>(cpu), 100.0, kBarBlank, "ao", millions(rays.ao), mray(rays.ao), share(rays.ao));
    append("\x1b[2K  %-15s%8.3f %5s %s |  %-9s%7.3f M %7.2f %5.1f\n", "frame measured",
            static_cast<double>(frameMs), "", kBarBlank, "shadow", millions(rays.shadow),
            mray(rays.shadow), share(rays.shadow));
    // The residual is printed, not left to be subtracted: a growing gap means a stage exists that nothing is measuring.
    const double residual = static_cast<double>(percentOf(frameMs - cpu, frameMs)) / 100.0;
    formatSignedBar(bar, residual / kResidualFullScale);
    append("\x1b[2K  %-15s%8.3f %5.1f %s |  --------------------------------\n", "unaccounted",
            static_cast<double>(frameMs - cpu),
            static_cast<double>(percentOf(frameMs - cpu, frameMs)), bar.data());
    append("\x1b[2K  %-15s%8s %5s %s |  %-9s%7.3f M %7.2f %5.1f\n", "", "", "", kBarBlank, "total", millions(rays.total()),
            mray(rays.total()), 100.0);
    // Ratios rather than new counters: one primary ray is traced per sample, so both fall out of the counts already here.
    append("\x1b[2K  GPU  GL_TIME_ELAPSED                     |  rays/sample %5.2f  depth %5.2f  \n",
            rays.primary > 0 ? total / static_cast<double>(rays.primary) : 0.0,
            rays.primary > 0 ? (static_cast<double>(rays.bounce) / static_cast<double>(rays.primary)) + 1.0 : 0.0);
    append("\x1b[2K  %-15s%8.3f %5s %s |\n", "present (blit)",
            static_cast<double>(frame.presentGpuMs), "", kBarBlank);
}

void PerfDashboard::drawFooter(const DashboardFrame& frame) {
    append("\x1b[2K------------------------------------------------------------------------------\n");
    append("\x1b[2K MEMORY  rss %.2f GiB   gl %.1f MiB   bvh %.1f MiB   sys %.2f/%.2f GiB\n",
            static_cast<double>(frame.ramBytes) / kGiB, static_cast<double>(frame.gpuBytes) / kMiB,
            static_cast<double>(frame.bvhBytes) / kMiB,
            static_cast<double>(frame.systemAvailableBytes) / kGiB,
            static_cast<double>(frame.systemTotalBytes) / kGiB);
    append("\x1b[2K SCENE   %s   %d inst   %d light   %d tri   aov %s\n", frame.sceneName,
            frame.instanceCount, frame.lightCount, frame.triangleCount, frame.aovName);
    append("\x1b[2K RENDER  image %dx%d   scale %.2f%s   trace %dx%d   window %dx%d\n", frame.imageWidth,
            frame.imageHeight, static_cast<double>(frame.renderScale),
            frame.interactiveScale ? " interactive" : "", frame.traceWidth, frame.traceHeight,
            frame.viewportWidth, frame.viewportHeight);
    append("\x1b[2K==============================================================================\n");
    // Blank line inside the block, not after: it separates the table from the shell cursor and the next cursor-up still erases it.
    append("\x1b[2K\n");
}

}  // namespace pathtracer::debug
