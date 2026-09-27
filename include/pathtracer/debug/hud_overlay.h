#pragma once

#include <cstddef>
#include <cstdint>
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

// Toggle for the centre-crosshair framing overlay, owned by main.cpp (defaults on, no runtime toggle), read here.
struct FramingOverlayState {
    bool crosshair = true;
};

// Read-only convergence status for the AOV section's readout, rebuilt each frame from the displayed result's sample count and driver state.
struct PathTracedStatus {
    bool hasResult = false;
    double lastPassSeconds = 0.0;
    int accumulatedSamples = 0;
    int maxSamples = 0;  // 0 = unbounded
};

// Pixel under the cursor (samplePixelProbe): the composited post-LUT pixel for Beauty and post-filter AOVs, else the raw AOV value.
struct PixelProbeSample {
    bool valid = false;
    glm::vec4 color{0.0F};
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
    float cameraYawDegrees;
    float cameraPitchDegrees;
    bool cameraOrbiting;
    const Histogram& histogram;
    const PathTracedStatus& pathTraced;
    // Fraction of Beauty's texels that would clip at the display encode, and the peak as a multiple of display range.
    float overRangeFraction;
    float overRangePeakMultiple;
    bool vsync;  // profile.json frame cap, for the Cap readout
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
    void draw(const HudFrameData& frame, int& aov, float& focalLengthMm, float& aperture,
              float& shutterSeconds, float& iso, int& filmBackPresetIndex,
              const std::vector<const char*>& filmBackPresetNames, int& lensProjection, int& lensFit, bool& showSky,
              bool& envLightEnabled, int& envRotationDegrees, float& envExposureStops,
              float& aberrationStrength, const FramingOverlayState& framing,
              const PixelProbeSample& pixelProbe) const;

    // ImGui::Render + backend draw-data submit. Call after the post-process blit, before window.swapBuffers().
    void render() const;

    // True while ImGui wants the mouse: callers must not read an LMB click as a scene interaction while it holds.
    [[nodiscard]] bool wantsCaptureMouse() const;

private:
    bool owns_ = true;  // false on a moved-from instance; guards shutdown
};

}  // namespace pathtracer::debug
