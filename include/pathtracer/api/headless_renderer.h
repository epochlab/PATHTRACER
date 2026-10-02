#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "pathtracer/config/profile_config.h"
#include "pathtracer/config/scene_config.h"
#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/render_stats.h"
#include "pathtracer/gfx/scalar_type.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/environment_map.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/light.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/rasterizer.h"
#include "pathtracer/scene/shading_scene.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::api {

// A scene loaded once and rendered many times, no window or GL: the synchronous half of PathTraceDriver. Neither copyable nor movable.
class HeadlessRenderer {
public:
    struct Request {
        pathtracer::scene::Camera camera;
        // The view MotionVector measures motion from, either lens; nullopt is camera itself, so motion reads exactly zero.
        std::optional<pathtracer::scene::Camera> previousCamera;
        int width = 0;
        int height = 0;
        // Path-traced passes at one sample each, averaged. Ignored when every requested AOV comes from the rasterizer.
        int samples = 1;
        // Randomizes the sampler's Owen scramble, fixed across one render's passes. Callers choose the seed, never the per-pass index.
        std::uint32_t scrambleSeed = 1;
        // A G-buffer AOV requires camera.lens() to be Rectilinear: render() rejects a fisheye rather than rasterize the wrong projection.
        std::vector<pathtracer::debug::AovId> aovs;
        // nullopt keeps the scene's authored environment.lightEnabled; true/false override it, so one scene.json renders lit and unlit.
        std::optional<bool> envLightEnabled;
        // Whether a camera ray that hits nothing returns environment radiance. Gates the primary miss only, so it unlights nothing.
        std::optional<bool> showSky;
    };

    // Per-pass wall clock and ray counts for the most recent render(), so a benchmark measures the renderer rather than re-deriving it.
    struct RenderStats {
        // One entry per path-traced pass. Empty when the request needed no light transport -- not an error; do not assume a pass happened.
        std::vector<double> passMilliseconds;
        // Wall clock of the scan-conversion and Beauty-filter work, 0 when it did not run; averaging it with passes would mean nothing.
        double rasterMilliseconds = 0.0;
        double filterMilliseconds = 0.0;
        pathtracer::debug::RayCounts rays;
    };

    // nullptr on any load failure, reason in `error`. assetRoot holds config/, scenes/, geometry/ and materials/; scenePath is relative.
    [[nodiscard]] static std::unique_ptr<HeadlessRenderer> open(const std::string& assetRoot,
                                                                 const std::string& scenePath,
                                                                 std::string& error);
    ~HeadlessRenderer();

    HeadlessRenderer(const HeadlessRenderer&) = delete;
    HeadlessRenderer& operator=(const HeadlessRenderer&) = delete;
    HeadlessRenderer(HeadlessRenderer&&) = delete;
    HeadlessRenderer& operator=(HeadlessRenderer&&) = delete;

    // profile.json's authored camera, the one render_beauty and the viewer both start from.
    [[nodiscard]] const pathtracer::scene::Camera& defaultCamera() const { return defaultCamera_; }
    [[nodiscard]] int defaultWidth() const { return profile_.render.width; }
    [[nodiscard]] int defaultHeight() const { return profile_.render.height; }

    // Blocking. outputs parallels request.aovs, caller-allocated at width * height * aovChannels(aovs[i]); each producer runs at most once.
    [[nodiscard]] bool render(const Request& request, std::span<float* const> outputs,
                               std::string& error);

    // Same render with no output copy, for a C++ caller reading results in place through lastImage().
    [[nodiscard]] bool render(const Request& request, std::string& error);

    // The buffer behind one requested AOV of the most recent render(), at aovChannels, valid until the next: for an EXR write or encode.
    [[nodiscard]] const pathtracer::gfx::HdrImage& lastImage(pathtracer::debug::AovId aov) const;
    [[nodiscard]] const RenderStats& lastStats() const { return stats_; }

    // Resolved scene state a caller may need to report rather than render with: a benchmark naming its settings.
    [[nodiscard]] const pathtracer::scene::PathTraceSettings& baseSettings() const { return baseSettings_; }
    [[nodiscard]] pathtracer::gfx::ScalarType textureType() const { return profile_.render.textureType; }
    [[nodiscard]] bool defaultEnvLightEnabled() const { return defaultEnvLightEnabled_; }

private:
    HeadlessRenderer(pathtracer::config::ProfileConfig profile,
                     pathtracer::scene::LoadedModel model, std::vector<pathtracer::scene::QuadLight> quadLights,
                     std::vector<int> instanceLightIndex,
                     std::vector<pathtracer::scene::PathTraceSettings> perInstanceSettings,
                     pathtracer::scene::PathTraceSettings baseSettings, pathtracer::scene::EmbreeAccel accel,
                     pathtracer::gfx::ImageTexture environmentImage, bool envLightEnabled,
                     const pathtracer::scene::Camera& defaultCamera);

    // Sizes the reused path-traced and rasterizer buffers to this request, reallocating only on a resolution change.
    void resizeBuffers(int width, int height);

    pathtracer::config::ProfileConfig profile_;
    pathtracer::scene::LoadedModel model_;
    std::vector<int> instanceLightIndex_;
    std::vector<pathtracer::scene::PathTraceSettings> perInstanceSettings_;
    pathtracer::scene::PathTraceSettings baseSettings_;
    std::vector<pathtracer::scene::AabbBounds> instanceBounds_;
    pathtracer::scene::EmbreeAccel accel_;
    // Declaration order is load-bearing: lights_ captures &environmentMap_ and a reference to quadLights_.
    pathtracer::scene::EnvironmentMap environmentMap_;
    std::vector<pathtracer::scene::QuadLight> quadLights_;
    // Both light sets are built at load and chosen per request: LightSet stores only a pointer and a reference, so the off variant is free.
    pathtracer::scene::LightSet lights_;
    pathtracer::scene::LightSet lightsEnvOff_;
    bool defaultEnvLightEnabled_ = true;
    pathtracer::scene::Camera defaultCamera_;
    pathtracer::scene::ThreadPool threadPool_;

    // Reused across render() calls, reallocated only when the resolution changes.
    pathtracer::scene::PathTraceResult pathTraced_;
    pathtracer::scene::RasterGBuffer gbuffer_;
    // One running sum per path-traced lane this request needs, parallel to accumulatedAovs_.
    std::vector<pathtracer::debug::AovId> accumulatedAovs_;
    std::vector<pathtracer::gfx::HdrImage> accumulators_;
    // Welford second moment of the per-pass Beauty luminance, carried beside the naive sum so the published mean stays bit-identical.
    std::vector<float> beautyLuminanceM2_;
    // One evaluated filter per distinct BeautyFilter AOV, so lastImage() can return one and two AOVs sharing a filter evaluate it once.
    std::vector<pathtracer::debug::AovId> filteredAovs_;
    std::vector<pathtracer::gfx::HdrImage> filtered_;
    RenderStats stats_;
    int bufferWidth_ = 0;
    int bufferHeight_ = 0;
};

}  // namespace pathtracer::api
