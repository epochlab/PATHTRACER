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
#include "pathtracer/scene/gbuffer.h"
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
        // Path-traced passes at one sample each, averaged. Ignored when every requested AOV comes from the G-buffer.
        int samples = 1;
        // Randomizes the sampler's Owen scramble, fixed across one render's passes. Callers choose the seed, never the per-pass index.
        std::uint32_t scrambleSeed = 1;
        std::vector<pathtracer::debug::AovId> aovs;
        // nullopt keeps the scene's authored environment.lightEnabled; true/false override it, so one scene.json renders lit and unlit.
        std::optional<bool> envLightEnabled;
        // Whether a camera miss returns environment radiance; nullopt is kDefaultShowSky. Primary miss only, so it unlights nothing.
        std::optional<bool> showSky;
        // Model root rotationXyz in degrees, lights included, replacing scene.json's model.rotation; nullopt keeps it. Rebuilds the BVH.
        std::optional<glm::vec3> rootRotationDegrees;
        // One rotationXyz per scene light, under the root, replacing each lights[i].rotation; nullopt keeps them. Rebuilds the BVH.
        std::optional<std::vector<glm::vec3>> lightRotationsDegrees;
        // rotationXyz of the environment map to world, background and lighting alike; zero is the map unrotated.
        glm::vec3 envRotationDegrees{0.0F};
    };

    // Per-pass wall clock and ray counts for the most recent render(), so a benchmark measures the renderer rather than re-deriving it.
    struct RenderStats {
        // One entry per path-traced pass. Empty when the request needed no light transport -- not an error; do not assume a pass happened.
        std::vector<double> passMilliseconds;
        // Wall clock of the G-buffer and Beauty-filter work, 0 when it did not run; averaging it with passes would mean nothing.
        double gbufferMilliseconds = 0.0;
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
    [[nodiscard]] bool defaultEnvLightEnabled() const { return scene_.environment.lightEnabled; }
    // The authored scene.json: its model.rotation and lights[i].rotation are the pose a request's nullopt keeps.
    [[nodiscard]] const pathtracer::config::SceneConfig& scene() const { return scene_; }

private:
    // Everything the root and light rotations are baked into: the triangle soup, its BVH, the quads and the per-instance bounds.
    struct Geometry {
        pathtracer::scene::LoadedModel model;
        std::vector<pathtracer::scene::QuadLight> quadLights;
        std::vector<int> instanceLightIndex;  // parallel to model.instances, -1 where the instance emits nothing
        std::vector<pathtracer::scene::AabbBounds> instanceBounds;
        pathtracer::scene::EmbreeAccel accel;
    };

    HeadlessRenderer(std::string assetRoot, pathtracer::config::ProfileConfig profile, pathtracer::config::SceneConfig scene,
                     Geometry geometry, std::vector<pathtracer::scene::PathTraceSettings> perInstanceSettings,
                     pathtracer::scene::PathTraceSettings baseSettings,
                     std::shared_ptr<const pathtracer::gfx::ImageTexture> environmentTexture,
                     const pathtracer::scene::Camera& defaultCamera);

    // Loads the glTF under model's root, places lights under it and builds the BVH: open() and every pose change share it.
    [[nodiscard]] static std::optional<Geometry> buildGeometry(const std::string& assetRoot,
                                                               const pathtracer::config::ModelConfig& model,
                                                               const std::vector<pathtracer::config::QuadLightConfig>& lights,
                                                               std::string& error);
    // Rebuilds geometry_ for the request's root and light rotations unless already built for them; false leaves it untouched.
    [[nodiscard]] bool applyPose(const Request& request, std::string& error);

    // Sizes the reused path-traced and G-buffer buffers to this request, reallocating only on a resolution change.
    void resizeBuffers(int width, int height);
    // Traces request.samples one-sample passes and folds each into accumulators_, the running means PathTraceDriver publishes.
    void accumulatePathTraced(const Request& request);
    // Folds one pass into the means and the Beauty luminance M2 in place, through the driver's foldRunningMean/foldLuminanceM2.
    void accumulatePass(const Request& request, int pass, const std::vector<const std::vector<float>*>& laneSources,
                        int beautyIndex);
    // Ray-casts the G-buffer lanes, motion measured from request.previousCamera (camera itself when absent).
    void renderGBufferLanes(const Request& request);
    // Evaluates each distinct BeautyFilter AOV the request names once, over the accumulated Beauty.
    void evaluateFilters(const Request& request);

    std::string assetRoot_;
    pathtracer::config::ProfileConfig profile_;
    pathtracer::config::SceneConfig scene_;
    Geometry geometry_;
    // The rotations geometry_ is built for, compared per request so an unchanged pose costs nothing.
    glm::vec3 builtRootRotationDegrees_;
    std::vector<glm::vec3> builtLightRotationsDegrees_;
    std::vector<pathtracer::scene::PathTraceSettings> perInstanceSettings_;
    pathtracer::scene::PathTraceSettings baseSettings_;
    pathtracer::scene::EnvironmentMap environmentMap_;
    pathtracer::scene::Camera defaultCamera_;
    pathtracer::scene::ThreadPool threadPool_;

    // Reused across render() calls, reallocated only when the resolution changes.
    pathtracer::scene::PathTraceResult pathTraced_;
    pathtracer::scene::GBuffer gbuffer_;
    // One running mean per path-traced lane this request needs, parallel to accumulatedAovs_.
    std::vector<pathtracer::debug::AovId> accumulatedAovs_;
    std::vector<pathtracer::gfx::HdrImage> accumulators_;
    // Welford second moment of the per-pass Beauty luminance, bit-identical to the driver's PathTraceResult::beautyLuminanceM2.
    std::vector<float> beautyLuminanceM2_;
    // One evaluated filter per distinct BeautyFilter AOV, so lastImage() can return one and two AOVs sharing a filter evaluate it once.
    std::vector<pathtracer::debug::AovId> filteredAovs_;
    std::vector<pathtracer::gfx::HdrImage> filtered_;
    RenderStats stats_;
    int bufferWidth_ = 0;
    int bufferHeight_ = 0;
};

}  // namespace pathtracer::api
