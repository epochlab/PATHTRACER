#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/debug/scene_stats.h"

struct GLFWwindow;

namespace pathtracer::scene {
class Camera;
}

namespace pathtracer::debug {

struct GpuInfo;
class FrameStats;
class Histogram;

// A slider's domain, shared where a test must cover what the HUD can reach.
struct SliderRange {
    float min;
    float max;
};

// The Camera section's exposure sliders; the over-range readout's window must hold the clip luminance at every corner.
inline constexpr SliderRange kApertureSlider{1.0F, 22.0F};
inline constexpr SliderRange kShutterSecondsSlider{1.0F / 4000.0F, 1.0F};
inline constexpr SliderRange kIsoSlider{50.0F, 6400.0F};

// Toggle for the centre-crosshair framing overlay, owned by main.cpp (defaults on, no runtime toggle), read here.
struct FramingOverlayState {
    bool crosshair = true;
};

// Read-only convergence status for the AOV section's readout, rebuilt each frame from the displayed result's sample count and driver state.
struct PathTracedStatus {
    bool hasResult = false;
    double lastPassSeconds = 0.0;
    int accumulatedSamples = 0;
    int samplesPerPixel = 0;  // 0 = unbounded
    bool adaptive = false;  // profile.json's adaptive.threshold is positive, so texels stop on their own error
    double samplingFraction = 1.0;  // texels the next pass traces over all texels; 0 once every one has stopped
    std::optional<double> readyMs;  // input change to the swap first showing its finished accumulation; nullopt while in flight
};

// What the bottom-right probe shows: the cursor texel's RGBA, its luminance as a spot meter, or nothing, neither sampled nor drawn.
enum class PixelProbeReadout { Colour, Photometric, None };

// Progressive shows every accumulated pass; Offline holds the last finished image on screen until the next one finishes.
enum class RenderMode { Progressive, Offline, Count };

// Index-parallel with RenderMode, so the HUD dropdown and the bench log name a mode from one table.
inline constexpr const char* kRenderModeNames[] = {"Progressive", "Offline"};
static_assert(sizeof(kRenderModeNames) / sizeof(kRenderModeNames[0]) == static_cast<int>(RenderMode::Count),
              "kRenderModeNames must stay index-parallel with RenderMode");

// Pixel under the cursor (samplePixelProbe): the display-transformed texel for Beauty, the raw texel for any other AOV.
struct PixelProbeSample {
    bool valid = false;
    glm::vec4 color{0.0F};
    std::optional<float> luminance;  // the picture texel's scene luminance in cd/m^2, before any display transform; absent off the picture
};

// Everything HudOverlay::draw needs for one frame. aov and FramingOverlayState stay separate arguments because draw writes them back.
struct HudFrameData {
    const GpuInfo& gpuInfo;
    double refreshHz;  // the window's display, measured by DisplayLink
    const FrameStats& frameStats;
    float postMs;
    std::size_t ramBytes;
    std::size_t gpuBytes;
    std::size_t systemAvailableBytes;
    std::uint64_t systemTotalBytes;
    int channelView;
    const char* lutName;
    SceneStats sceneStats;
    const pathtracer::scene::Camera& camera;
    bool cameraOrbiting;
    const Histogram& histogram;
    const PathTracedStatus& pathTraced;
    // Fraction of Beauty's texels that would clip at the display encode, and the peak as a multiple of display range.
    float overRangeFraction;
    float overRangePeakMultiple;
    bool vsync;  // profile.json frame cap, for the Cap readout
    int aovCount;  // the AOV dropdown's length: kAovCountWithoutDenoiser while profile.json disables the denoiser
    PixelProbeReadout pixelProbeReadout;  // profile.json render.pixelProbe
};

// Owns the ImGui context and GLFW/OpenGL3 backends for one window's lifetime, move-only. Composites the debug panel onto the backbuffer.
class HudOverlay {
public:
    explicit HudOverlay(GLFWwindow* nativeHandle);
    ~HudOverlay();

    HudOverlay(const HudOverlay&) = delete;
    HudOverlay& operator=(const HudOverlay&) = delete;
    HudOverlay(HudOverlay&& other) noexcept;
    HudOverlay& operator=(HudOverlay&& other) noexcept;

    // Call after window.pollEvents(), before any GL draw calls.
    void beginFrame() const;

    // Call after beginFrame(), before render(). Editable fields are passed by reference and written back in place, so nothing is returned.
    void draw(const HudFrameData& frame, int& renderMode, int& aov, float& focalLengthMm, float& aperture,
              float& shutterSeconds, float& iso, int& filmBackPresetIndex,
              const std::vector<const char*>& filmBackPresetNames, int& lensProjection, int& pixelLattice,
              float& acceptanceAngleDegrees, bool& showSky, bool& envLightEnabled, int& envRotationDegrees, float& envExposureStops,
              float& aberrationStrength, float& bloomStrength, bool& logTransform, const FramingOverlayState& framing,
              const PixelProbeSample& pixelProbe) const;

    // ImGui::Render + backend draw-data submit. Call after the post-process blit, before window.swapBuffers().
    void render() const;

    // True while ImGui wants the mouse: callers must not read an LMB click as a scene interaction while it holds.
    [[nodiscard]] bool wantsCaptureMouse() const;

private:
    bool owns_ = true;  // false on a moved-from instance; guards shutdown
};

}  // namespace pathtracer::debug
