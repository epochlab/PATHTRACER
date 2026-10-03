#include "pathtracer/debug/hud_overlay.h"

// GLEW before GLFW: see gl_debug.cpp for why.
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/frame_stats.h"
#include "pathtracer/debug/histogram.h"
#include "pathtracer/debug/system_info.h"
#include "pathtracer/scene/camera.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace pathtracer::debug {

namespace {

constexpr ImVec4 kCyan(0.0F, 0.85F, 0.85F, 1.0F);

constexpr float kHistogramHeight = 72.0F;

// 9-tap triangular smoother (radius 4) over bins in [1, 254]. Bins 0 and 255 are never read, so clipping spikes do not bleed in.
void smoothChannel(const std::array<std::uint32_t, 256>& channelBins, std::uint32_t peak,
                    std::array<float, 256>& out) {
    for (int bin = 0; bin < 256; ++bin) {
        float sum = 0.0F;
        float weightSum = 0.0F;
        for (int offset = -4; offset <= 4; ++offset) {
            const int neighbor = bin + offset;
            if (neighbor < 1 || neighbor > 254) {
                continue;
            }
            const int distance = offset < 0 ? -offset : offset;
            const float weight = 5.0F - static_cast<float>(distance);
            const float value = static_cast<float>(
                std::min(channelBins[static_cast<std::size_t>(neighbor)], peak));
            sum += weight * std::sqrt(value / static_cast<float>(peak));
            weightSum += weight;
        }
        out[static_cast<std::size_t>(bin)] = weightSum > 0.0F ? sum / weightSum : 0.0F;
    }
}

// Rescales a smoothed curve so its own maximum reaches 1.0: smoothing can only pull values down from the raw peak, never up.
void rescaleToUnitPeak(std::array<float, 256>& curve) {
    float maxValue = 0.0F;
    for (const float value : curve) {
        maxValue = std::max(maxValue, value);
    }
    if (maxValue < 1e-6F) {
        return;
    }
    for (float& value : curve) {
        value /= maxValue;
    }
}

// The same across every active channel at once, so the tallest reaches full height and the others keep their relative height.
void rescaleToUnitPeak(std::array<std::array<float, 256>, 3>& curves, const std::array<bool, 3>& active) {
    float maxValue = 0.0F;
    for (int c = 0; c < 3; ++c) {
        if (!active[static_cast<std::size_t>(c)]) {
            continue;
        }
        for (const float value : curves[static_cast<std::size_t>(c)]) {
            maxValue = std::max(maxValue, value);
        }
    }
    if (maxValue < 1e-6F) {
        return;
    }
    for (int c = 0; c < 3; ++c) {
        if (active[static_cast<std::size_t>(c)]) {
            for (float& value : curves[static_cast<std::size_t>(c)]) {
                value /= maxValue;
            }
        }
    }
}

bool isGrayscale(const std::array<std::array<std::uint32_t, 256>, 3>& bins) {
    for (int bin = 0; bin < 256; ++bin) {
        if (bins[0][static_cast<std::size_t>(bin)] != bins[1][static_cast<std::size_t>(bin)] ||
            bins[1][static_cast<std::size_t>(bin)] != bins[2][static_cast<std::size_t>(bin)]) {
            return false;
        }
    }
    return true;
}

// One continuous filled and outlined curve rather than 256 rectangles: what gives a smooth silhouette instead of a comb.
void drawHistogramCurve(ImDrawList* drawList, ImVec2 origin, float histogramWidth,
                         const std::array<float, 256>& vals, ImU32 fillColor, ImU32 lineColor) {
    const float binWidth = histogramWidth / 256.0F;
    std::array<ImVec2, 256> edge{};
    for (int bin = 0; bin < 256; ++bin) {
        edge[static_cast<std::size_t>(bin)] = ImVec2(
            origin.x + ((static_cast<float>(bin) + 0.5F) * binWidth),
            origin.y + (kHistogramHeight * (1.0F - vals[static_cast<std::size_t>(bin)])));
    }
    std::array<ImVec2, 258> poly{};
    poly[0] = ImVec2(origin.x, origin.y + kHistogramHeight);
    for (int bin = 0; bin < 256; ++bin) {
        poly[static_cast<std::size_t>(bin) + 1] = edge[static_cast<std::size_t>(bin)];
    }
    poly[257] = ImVec2(origin.x + histogramWidth, origin.y + kHistogramHeight);

    // Concave fills triangulate internally and AA seams show as diagonal lines, so AA is off here; the outline stroke restores the edge.
    const ImDrawListFlags savedFlags = drawList->Flags;
    drawList->Flags &= ~ImDrawListFlags_AntiAliasedFill;
    drawList->AddConcavePolyFilled(poly.data(), 258, fillColor);
    drawList->Flags = savedFlags;
    drawList->AddPolyline(edge.data(), 256, lineColor, 0, 1.0F);
}

// Grayscale AOVs, R==G==B, collapse to one curve on a full-range unsmoothed scale so 0/255 clipping spikes stay where they are.
void drawGrayscaleHistogram(ImDrawList* drawList, ImVec2 origin, float histogramWidth,
                             const std::array<std::array<std::uint32_t, 256>, 3>& bins) {
    std::uint64_t interiorTotal = 0;
    for (int bin = 1; bin <= 254; ++bin) {
        interiorTotal += bins[0][static_cast<std::size_t>(bin)];
    }
    // Under ~1% of pixels in the interior means near-binary content, an alpha or depth mask: full-range peak, raw sqrt, no smoothing.
    const bool nearBinary = interiorTotal < static_cast<std::uint64_t>(256 * 144 / 100);
    std::array<float, 256> heights{};
    if (nearBinary) {
        std::uint32_t peak = 1;
        for (const std::uint32_t count : bins[0]) {
            peak = std::max(peak, count);
        }
        for (int bin = 0; bin < 256; ++bin) {
            heights[static_cast<std::size_t>(bin)] =
                std::sqrt(static_cast<float>(bins[0][static_cast<std::size_t>(bin)]) /
                          static_cast<float>(peak));
        }
    } else {
        std::uint32_t peak = 1;
        for (int bin = 1; bin <= 254; ++bin) {
            peak = std::max(peak, bins[0][static_cast<std::size_t>(bin)]);
        }
        smoothChannel(bins[0], peak, heights);
        rescaleToUnitPeak(heights);
    }
    drawHistogramCurve(drawList, origin, histogramWidth, heights, IM_COL32(180, 180, 180, 130),
                        IM_COL32(220, 220, 220, 220));
}

// A structurally unused channel has every pixel at bin 0. Checked over bins 1-255 so a background-dominated channel is not mistaken for it.
std::array<bool, 3> activeChannels(const std::array<std::array<std::uint32_t, 256>, 3>& bins) {
    std::array<bool, 3> active{};
    for (int c = 0; c < 3; ++c) {
        bool empty = true;
        for (int bin = 1; bin < 256; ++bin) {
            if (bins[static_cast<std::size_t>(c)][static_cast<std::size_t>(bin)] > 0) {
                empty = false;
                break;
            }
        }
        active[static_cast<std::size_t>(c)] = !empty;
    }
    return active;
}

// One peak shared across active channels, not per channel, so a channel with more signal reads taller rather than all filling the same.
std::uint32_t sharedPeak(const std::array<std::array<std::uint32_t, 256>, 3>& bins,
                          const std::array<bool, 3>& active) {
    std::uint32_t peak = 1;
    for (int c = 0; c < 3; ++c) {
        if (!active[static_cast<std::size_t>(c)]) {
            continue;
        }
        for (int bin = 1; bin <= 254; ++bin) {
            peak = std::max(peak, bins[static_cast<std::size_t>(c)][static_cast<std::size_t>(bin)]);
        }
    }
    return peak;
}

// Each channel one smooth curve, layered back to front B/G/R, then a floor curve -- the min across active channels -- where they overlap.
void drawRgbHistogram(ImDrawList* drawList, ImVec2 origin, float histogramWidth,
                       const std::array<std::array<std::uint32_t, 256>, 3>& bins) {
    const std::array<bool, 3> active = activeChannels(bins);
    const std::uint32_t peak = sharedPeak(bins, active);

    std::array<std::array<float, 256>, 3> heights{};
    for (int c = 0; c < 3; ++c) {
        if (active[static_cast<std::size_t>(c)]) {
            smoothChannel(bins[static_cast<std::size_t>(c)], peak,
                          heights[static_cast<std::size_t>(c)]);
        }
    }
    rescaleToUnitPeak(heights, active);

    // B, G, R -- R topmost -- then the shared overlap curve last.
    if (active[2]) {
        drawHistogramCurve(drawList, origin, histogramWidth, heights[2],
                            IM_COL32(40, 80, 200, 120), IM_COL32(80, 140, 255, 220));
    }
    if (active[1]) {
        drawHistogramCurve(drawList, origin, histogramWidth, heights[1],
                            IM_COL32(40, 180, 60, 120), IM_COL32(80, 220, 100, 220));
    }
    if (active[0]) {
        drawHistogramCurve(drawList, origin, histogramWidth, heights[0],
                            IM_COL32(200, 40, 40, 120), IM_COL32(255, 100, 80, 220));
    }

    const int activeCount = (active[0] ? 1 : 0) + (active[1] ? 1 : 0) + (active[2] ? 1 : 0);
    if (activeCount < 2) {
        return;
    }
    std::array<float, 256> overlap{};
    float overlapMax = 0.0F;
    for (int bin = 0; bin < 256; ++bin) {
        float value = 1.0F;
        for (int c = 0; c < 3; ++c) {
            if (active[static_cast<std::size_t>(c)]) {
                value = std::min(value, heights[static_cast<std::size_t>(c)][static_cast<std::size_t>(bin)]);
            }
        }
        overlap[static_cast<std::size_t>(bin)] = value;
        overlapMax = std::max(overlapMax, value);
    }
    if (overlapMax > 0.02F) {
        drawHistogramCurve(drawList, origin, histogramWidth, overlap, IM_COL32(180, 180, 180, 160),
                            IM_COL32(255, 255, 255, 220));
    }
}

// Renders the current AOV's per-channel histogram, at the panel's content width so it lines up with every other section.
void drawHistogramPanel(const std::array<std::array<std::uint32_t, 256>, 3>& bins,
                         float overRangeFraction, float overRangePeakMultiple) {
    ImGui::TextColored(kCyan, "Histogram");

    const float histogramWidth = ImGui::GetContentRegionAvail().x;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 boxMax(origin.x + histogramWidth, origin.y + kHistogramHeight);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(origin, boxMax, IM_COL32(18, 18, 18, 255));

    if (isGrayscale(bins)) {
        drawGrayscaleHistogram(drawList, origin, histogramWidth, bins);
    } else {
        drawRgbHistogram(drawList, origin, histogramWidth, bins);
    }

    drawList->AddRect(origin, boxMax, IM_COL32(60, 60, 60, 180));
    ImGui::Dummy(ImVec2(histogramWidth, kHistogramHeight));
    ImGui::Text("Over-range: %.1f%%, peak %.1fx", overRangeFraction * 100.0F, overRangePeakMultiple);
}

// Centre crosshair framing overlay on the foreground draw list, independent of the panel, so it never contaminates the AOV buffers.
void drawFramingOverlays(const FramingOverlayState& state, ImVec2 displaySize) {
    if (!state.crosshair) {
        return;
    }
    ImDrawList* drawList = ImGui::GetForegroundDrawList();
    constexpr ImU32 kGuideColor = IM_COL32(255, 255, 255, 140);
    const float cx = displaySize.x * 0.5F;
    const float cy = displaySize.y * 0.5F;
    constexpr float kArmLength = 10.0F;
    drawList->AddLine(ImVec2(cx - kArmLength, cy), ImVec2(cx + kArmLength, cy), kGuideColor);
    drawList->AddLine(ImVec2(cx, cy - kArmLength), ImVec2(cx, cy + kArmLength), kGuideColor);
}

void drawGpuSection(const HudFrameData& frame) {
    ImGui::TextColored(kCyan, "GPU");
    ImGui::Text("%s", frame.gpuInfo.renderer.c_str());
    ImGui::Text("%s", frame.gpuInfo.version.c_str());
    ImGui::Text("%.2f Hz", frame.refreshHz);
    ImGui::Separator();
}

void drawFrameSection(const HudFrameData& frame) {
    ImGui::TextColored(kCyan, "Frame");
    ImGui::Text("%.0f FPS  avg %.2f ms", frame.frameStats.fps(), frame.frameStats.avgMs());
    ImGui::PlotLines("##frametime", frame.frameStats.history().data(), FrameStats::kHistoryLength,
                      frame.frameStats.cursor(), nullptr, 0.0F, 33.3F,
                      ImVec2(ImGui::GetContentRegionAvail().x, 40.0F));
    ImGui::Text("min %.2f  max %.2f ms", frame.frameStats.minMs(), frame.frameStats.maxMs());
    ImGui::Text("GPU  post %.2f ms", frame.postMs);
    ImGui::Text("Cap  %s", frame.vsync ? "vsync" : "off");
    ImGui::Text("LUT  %s", frame.lutName);
    if (frame.pathTraced.hasResult) {
        if (frame.pathTraced.maxSamples > 0) {
            ImGui::Text("path-traced  %d / %d samples  %.2f s/pass", frame.pathTraced.accumulatedSamples,
                        frame.pathTraced.maxSamples, frame.pathTraced.lastPassSeconds);
        } else {
            ImGui::Text("path-traced  %d samples  %.2f s/pass", frame.pathTraced.accumulatedSamples,
                        frame.pathTraced.lastPassSeconds);
        }
    } else {
        ImGui::TextDisabled("path-traced: no render yet");
    }
    ImGui::Separator();
}

void drawMemorySection(const HudFrameData& frame) {
    ImGui::TextColored(kCyan, "Memory");
    ImGui::Text("RAM  %.1f MB", static_cast<double>(frame.ramBytes) / (1024.0 * 1024.0));
    ImGui::Text("GPU alloc  %.1f MB", static_cast<double>(frame.gpuBytes) / (1024.0 * 1024.0));
    ImGui::Text("System  %.1f / %.1f GB free",
                static_cast<double>(frame.systemAvailableBytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(frame.systemTotalBytes) / (1024.0 * 1024.0 * 1024.0));
    ImGui::Separator();
}

void drawResolutionAndSceneSection(const HudFrameData& frame) {
    ImGui::TextColored(kCyan, "Resolution");
    ImGui::Text("%d x %d", frame.sceneStats.imageWidth, frame.sceneStats.imageHeight);
    ImGui::Separator();

    ImGui::TextColored(kCyan, "Scene");
    ImGui::Text("Objects  %d", frame.sceneStats.objectCount);
    ImGui::Text("Triangles  %lld", frame.sceneStats.trianglesTotal);
    ImGui::Text("Points  %lld", frame.sceneStats.pointsTotal);
    ImGui::Separator();
}

// Labels/order come from the shared AovId enum (pathtracer/debug/aov.h), not a locally duplicated array.
void drawAovSection(int& aov) {
    ImGui::TextColored(kCyan, "AOV");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::Combo("##aov", &aov, kAovLabels, IM_ARRAYSIZE(kAovLabels));
    ImGui::Separator();
}

// pos/rot/clip read-only; projection and filmback dropdowns; focal length, aperture, shutter, ISO, aberration sliders.
void drawCameraSection(const HudFrameData& frame, float& focalLengthMm, float& aperture,
                        float& shutterSeconds, float& iso, int& filmBackPresetIndex,
                        const std::vector<const char*>& filmBackPresetNames, int& lensProjection,
                        float& aberrationStrength) {
    ImGui::TextColored(kCyan, "Camera");
    const glm::vec3 camPos = frame.camera.position();
    ImGui::Text("pos  x %.2f  y %.2f  z %.2f", camPos.x, camPos.y, camPos.z);
    const glm::vec3 camRot = frame.camera.rotationDegrees();
    ImGui::Text("rot  x %.1f  y %.1f  z %.1f", camRot.x, camRot.y, camRot.z);
    const pathtracer::scene::Camera::FilmBack filmBack = frame.camera.filmBack();
    // heightMm > 0 is guaranteed by loadFilmBackPresets's boundary validation, so this division is well-defined.
    ImGui::Text("Filmback  %.2f x %.2f mm  (%.2f:1)", filmBack.widthMm, filmBack.heightMm,
                filmBack.widthMm / filmBack.heightMm);
    ImGui::Text("Near  %.2f  Far  %.1f", frame.camera.nearClip(), frame.camera.farClip());
    // The active projection's vertical extent, the same value the filters read, so the readout cannot disagree with the render.
    ImGui::Text("FOV  %.1f deg", glm::degrees(frame.camera.verticalAngularExtentRadians()));
    if (frame.cameraOrbiting) {
        ImGui::TextColored(kCyan, "orbiting");
    }
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::Combo("##lensProjection", &lensProjection, pathtracer::scene::kLensProjectionNames,
                 IM_ARRAYSIZE(pathtracer::scene::kLensProjectionNames));
    // The lat-long maps ndc straight to angles, so the gate and focal length frame nothing: greyed, values kept for the other lenses.
    ImGui::BeginDisabled(lensProjection == static_cast<int>(pathtracer::scene::LensProjection::Omnidirectional));
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::Combo("##filmBackPreset", &filmBackPresetIndex, filmBackPresetNames.data(),
                 static_cast<int>(filmBackPresetNames.size()));
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    // Floors at a real circular-fisheye focal length (Nikon 6mm f/2.8): 10mm was a rectilinear assumption a fisheye cannot live with.
    ImGui::SliderFloat("##focalLength", &focalLengthMm, 6.0F, 300.0F, "Focal Length  %.0f mm");
    ImGui::EndDisabled();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderFloat("##aperture", &aperture, 1.0F, 22.0F, "Aperture  f/%.1f");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderFloat("##shutterSeconds", &shutterSeconds, 1.0F / 4000.0F, 1.0F,
                        "Shutter  %.4f s", ImGuiSliderFlags_Logarithmic);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderFloat("##iso", &iso, 50.0F, 6400.0F, "ISO  %.0f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    // Beauty only -- main.cpp zeroes this elsewhere -- ranged so the full sweep stays a lens effect rather than an unreadable smear.
    ImGui::SliderFloat("##aberrationStrength", &aberrationStrength, 0.0F, 0.05F, "Aberration  %.3f");
    ImGui::Separator();
}

void drawHdriSection(bool& showSky, bool& envLightEnabled, int& envRotationDegrees,
                      float& envExposureStops) {
    ImGui::TextColored(kCyan, "HDRI");
    // Removes the environment from LightSet entirely, unlike showSky below. Off is what makes the classic Goral 1984 Cornell reachable.
    ImGui::Checkbox("Environment Light", &envLightEnabled);
    ImGui::BeginDisabled(!envLightEnabled);
    // Beauty only, main.cpp gating the sky draw on it. A no-op elsewhere, and with Environment Light off, which already removes it.
    ImGui::Checkbox("Show/Hide Background", &showSky);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderInt("##envRotation", &envRotationDegrees, 0, 359, "Y-Axis  %d deg");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    // Stops, not a multiplier. main.cpp's requestPathTrace does exp2().
    ImGui::SliderFloat("##envExposureStops", &envExposureStops, -6.0F, 6.0F, "Exposure  %+.2f EV");
    ImGui::EndDisabled();
    ImGui::Separator();
}

// Bottom-right, pinned via (1,1) pivot. Swatch + RGBA to 3dp; blank if !pixelProbe.valid.
void drawPixelProbePanel(const PixelProbeSample& pixelProbe) {
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(displaySize.x - 8.0F, displaySize.y - 8.0F), ImGuiCond_Always,
                             ImVec2(1.0F, 1.0F));
    constexpr ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;
    ImGui::Begin("##pixelProbe", nullptr, flags);

    const glm::vec4 color = pixelProbe.valid ? pixelProbe.color : glm::vec4(0.0F);
    ImGui::ColorButton("##pixelProbeSwatch", ImVec4(color.r, color.g, color.b, color.a),
                        ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoBorder,
                        ImVec2(14.0F, 14.0F));
    ImGui::SameLine();
    if (pixelProbe.valid) {
        ImGui::Text("R %.3f  G %.3f  B %.3f  A %.3f", color.r, color.g, color.b, color.a);
    } else {
        ImGui::TextDisabled("R --   G --   B --   A --");
    }

    const ImVec2 windowMin = ImGui::GetWindowPos();
    const ImVec2 windowSize = ImGui::GetWindowSize();
    ImGui::End();

    // Drawn on the foreground list after End(): a rect at the exact window bounds is clipped to near-invisibility before End().
    ImGui::GetForegroundDrawList()->AddRect(windowMin, ImVec2(windowMin.x + windowSize.x, windowMin.y + windowSize.y),
                                             IM_COL32(60, 60, 60, 180));
}

// Active R/G/B channel isolation, top-right corner -- foreground draw list, independent of the ##hud window.
void drawChannelViewCorner(int channelView) {
    if (channelView == 0) {
        return;
    }
    const char* label = channelView == 1 ? "R" : channelView == 2 ? "G" : "B";
    const ImU32 color = channelView == 1   ? IM_COL32(255, 70, 70, 255)
                        : channelView == 2 ? IM_COL32(70, 255, 70, 255)
                                           : IM_COL32(70, 70, 255, 255);
    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const ImVec2 textSize = ImGui::CalcTextSize(label);
    const ImVec2 pos(displaySize.x - textSize.x - 16.0F, 8.0F);
    ImGui::GetForegroundDrawList()->AddText(pos, color, label);
}

}  // namespace

HudOverlay::HudOverlay(GLFWwindow* nativeHandle) {
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;  // no stray imgui.ini for a fixed debug panel

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0F;
    style.WindowBorderSize = 0.0F;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.0F, 0.0F, 0.0F, 0.80F);

    if (!ImGui_ImplGlfw_InitForOpenGL(nativeHandle, true)) {
        std::cerr << "HudOverlay: ImGui_ImplGlfw_InitForOpenGL failed\n";
        std::exit(EXIT_FAILURE);
    }
    // matches this project's GL 4.1 core context
    if (!ImGui_ImplOpenGL3_Init("#version 410")) {
        std::cerr << "HudOverlay: ImGui_ImplOpenGL3_Init failed\n";
        std::exit(EXIT_FAILURE);
    }
}

HudOverlay::~HudOverlay() {
    if (owns_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
}

HudOverlay::HudOverlay(HudOverlay&& other) noexcept : owns_(other.owns_) {
    other.owns_ = false;
}

HudOverlay& HudOverlay::operator=(HudOverlay&& other) noexcept {
    if (this != &other) {
        if (owns_) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
        owns_ = other.owns_;
        other.owns_ = false;
    }
    return *this;
}

void HudOverlay::beginFrame() const {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void HudOverlay::draw(const HudFrameData& frame, int& aov, float& focalLengthMm, float& aperture,
                       float& shutterSeconds, float& iso, int& filmBackPresetIndex,
                       const std::vector<const char*>& filmBackPresetNames, int& lensProjection, bool& showSky,
                       bool& envLightEnabled, int& envRotationDegrees, float& envExposureStops,
                       float& aberrationStrength, const FramingOverlayState& framing,
                       const PixelProbeSample& pixelProbe) const {
    ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_Always);
    constexpr ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;
    ImGui::Begin("##hud", nullptr, flags);

    drawGpuSection(frame);
    drawFrameSection(frame);
    drawMemorySection(frame);
    drawResolutionAndSceneSection(frame);

    if (frame.histogram.hasData()) {
        drawHistogramPanel(frame.histogram.bins(), frame.overRangeFraction, frame.overRangePeakMultiple);
        ImGui::Separator();
    }

    drawAovSection(aov);
    drawCameraSection(frame, focalLengthMm, aperture, shutterSeconds, iso, filmBackPresetIndex,
                       filmBackPresetNames, lensProjection, aberrationStrength);
    drawHdriSection(showSky, envLightEnabled, envRotationDegrees, envExposureStops);

    ImGui::End();

    drawChannelViewCorner(frame.channelView);
    drawFramingOverlays(framing, ImGui::GetIO().DisplaySize);
    drawPixelProbePanel(pixelProbe);
}

void HudOverlay::render() const {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

bool HudOverlay::wantsCaptureMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

}  // namespace pathtracer::debug
