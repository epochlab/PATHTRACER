#include "pathtracer/api/headless_renderer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/debug/render_stats.h"
#include "pathtracer/scene/material_binding.h"
#include "pathtracer/scene/path_trace_driver.h"

namespace pathtracer::api {

namespace {

using pathtracer::debug::AovId;
using pathtracer::debug::AovSource;

// The files a scene names, loaded and parsed before any geometry: what open() needs to build everything else.
struct SceneInputs {
    pathtracer::config::ProfileConfig profile;
    pathtracer::config::SceneConfig scene;
    pathtracer::config::MaterialConfig material;
    pathtracer::gfx::ImageTexture environmentImage;
};

[[nodiscard]] std::optional<SceneInputs> loadSceneInputs(const std::string& assetRoot, const std::string& scenePath,
                                                         std::string& error) {
    std::optional<pathtracer::config::ProfileConfig> profile =
        pathtracer::config::loadProfileConfig(assetRoot + "/config/profile.json");
    if (!profile) {
        error = "failed to load " + assetRoot + "/config/profile.json";
        return std::nullopt;
    }
    std::optional<pathtracer::config::SceneConfig> scene =
        pathtracer::config::loadSceneConfig(assetRoot + "/" + scenePath);
    if (!scene) {
        error = "failed to load scene " + assetRoot + "/" + scenePath;
        return std::nullopt;
    }
    std::optional<pathtracer::config::MaterialConfig> material =
        pathtracer::config::loadMaterialConfig(assetRoot + "/" + scene->materialPath);
    if (!material) {
        error = "failed to load material " + assetRoot + "/" + scene->materialPath;
        return std::nullopt;
    }
    std::optional<pathtracer::gfx::ImageTexture> environmentImage = pathtracer::gfx::loadImageTexture(
        assetRoot + "/" + scene->environment.hdriPath, profile->render.textureType, pathtracer::gfx::kRgbChannels);
    if (!environmentImage) {
        error = "failed to load environment " + assetRoot + "/" + scene->environment.hdriPath;
        return std::nullopt;
    }
    return SceneInputs{std::move(*profile), std::move(*scene), std::move(*material), std::move(*environmentImage)};
}

// profile.json names a film-back preset, assets/config/sensor.json supplies its dimensions. Resolved as initializeApp does.
[[nodiscard]] std::optional<pathtracer::scene::Camera> resolveCamera(const std::string& assetRoot,
                                                                 const pathtracer::config::ProfileConfig& profile,
                                                                 std::string& error) {
    const std::optional<std::vector<pathtracer::scene::Camera::FilmBackPreset>> presets =
        pathtracer::config::loadFilmBackPresets(assetRoot + "/config/sensor.json");
    if (!presets) {
        error = "failed to load " + assetRoot + "/config/sensor.json";
        return std::nullopt;
    }
    const auto preset = std::find_if(presets->begin(), presets->end(),
                                      [&](const pathtracer::scene::Camera::FilmBackPreset& candidate) {
                                          return candidate.name == profile.camera.defaultFilmBackPresetName;
                                      });
    if (preset == presets->end()) {
        error = "profile.json filmBackPreset \"" + profile.camera.defaultFilmBackPresetName +
                "\" not found in sensor.json";
        return std::nullopt;
    }
    const pathtracer::config::CameraConfig& config = profile.camera;
    pathtracer::scene::Camera camera(config.position, config.rotation, preset->filmBack,
                                     config.focalLengthMm, config.nearClip, config.farClip, config.aperture,
                                     config.shutterSeconds, config.iso, config.lens);
    if (!camera.validate(error)) {
        error = "profile.json camera: " + error;
        return std::nullopt;
    }
    return camera;
}

[[nodiscard]] bool requestsSource(const std::vector<AovId>& aovs, AovSource source) {
    return std::any_of(aovs.begin(), aovs.end(), [source](AovId aov) { return pathtracer::debug::aovSource(aov) == source; });
}

[[nodiscard]] std::vector<glm::vec3> rotationsOf(const std::vector<pathtracer::config::QuadLightConfig>& lights) {
    std::vector<glm::vec3> rotations;
    rotations.reserve(lights.size());
    for (const pathtracer::config::QuadLightConfig& light : lights) {
        rotations.push_back(light.rotation);
    }
    return rotations;
}

[[nodiscard]] bool isFinite(const glm::vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Rejects a request no producer can serve, before any buffer is touched; error names the first fault.
[[nodiscard]] bool validRequest(const HeadlessRenderer::Request& request, std::size_t lightCount, std::string& error) {
    if (request.width <= 0 || request.height <= 0) {
        error = "resolution must be positive";
        return false;
    }
    if (request.samples <= 0) {
        error = "samples must be at least 1";
        return false;
    }
    if (request.aovs.empty()) {
        error = "no AOVs requested";
        return false;
    }
    if (!request.camera.validate(error)) {
        return false;
    }
    if (request.previousCamera.has_value() && !request.previousCamera->validate(error)) {
        error = "previousCamera: " + error;
        return false;
    }
    if (request.rootRotationDegrees.has_value() && !isFinite(*request.rootRotationDegrees)) {
        error = "root rotation is not finite";
        return false;
    }
    if (request.lightRotationsDegrees.has_value()) {
        const std::vector<glm::vec3>& rotations = *request.lightRotationsDegrees;
        if (rotations.size() != lightCount) {
            error = "light rotations number " + std::to_string(rotations.size()) + ", the scene has " + std::to_string(lightCount) +
                    " lights";
            return false;
        }
        if (!std::all_of(rotations.begin(), rotations.end(), isFinite)) {
            error = "a light rotation is not finite";
            return false;
        }
    }
    if (!isFinite(request.envRotationDegrees)) {
        error = "environment rotation is not finite";
        return false;
    }
    return true;
}

// The path-traced lanes a request needs, each once. Beauty joins whenever a filter is asked for, every filter reading accumulated Beauty.
[[nodiscard]] std::vector<AovId> accumulatedAovsFor(const std::vector<AovId>& aovs, bool wantsFilter) {
    std::vector<AovId> accumulated;
    for (const AovId aov : aovs) {
        if (pathtracer::debug::aovSource(aov) == AovSource::PathTraced &&
            std::find(accumulated.begin(), accumulated.end(), aov) == accumulated.end()) {
            accumulated.push_back(aov);
        }
    }
    if (wantsFilter && std::find(accumulated.begin(), accumulated.end(), AovId::Beauty) == accumulated.end()) {
        accumulated.push_back(AovId::Beauty);
    }
    return accumulated;
}

}  // namespace

std::optional<HeadlessRenderer::Geometry> HeadlessRenderer::buildGeometry(
    const std::string& assetRoot, const pathtracer::config::ModelConfig& model,
    const std::vector<pathtracer::config::QuadLightConfig>& lights, std::string& error) {
    const glm::mat4 rootTransform = pathtracer::scene::rootTransformOf(model);
    std::optional<pathtracer::scene::LoadedModel> loaded = pathtracer::scene::loadGltf(assetRoot + "/" + model.gltfPath, rootTransform);
    if (!loaded) {
        error = "failed to load glTF " + assetRoot + "/" + model.gltfPath;
        return std::nullopt;
    }
    std::vector<int> instanceLightIndex(loaded->instances.size(), -1);
    std::vector<pathtracer::scene::QuadLight> quadLights = pathtracer::scene::buildQuadLights(lights, rootTransform);
    pathtracer::scene::appendQuadLights(*loaded, quadLights, instanceLightIndex);
    // After appendQuadLights, so the light panels' own instances are bounded too.
    std::vector<pathtracer::scene::AabbBounds> instanceBounds =
        pathtracer::scene::computeInstanceBounds(loaded->shadingTriangles, static_cast<int>(loaded->instances.size()));
    std::optional<pathtracer::scene::EmbreeAccel> accel = pathtracer::scene::EmbreeAccel::build(std::move(loaded->worldTriangles));
    if (!accel) {
        error = "Embree scene build failed";
        return std::nullopt;
    }
    return Geometry{std::move(*loaded), std::move(quadLights), std::move(instanceLightIndex), std::move(instanceBounds),
                    std::move(*accel)};
}

std::unique_ptr<HeadlessRenderer> HeadlessRenderer::open(const std::string& assetRoot,
                                                          const std::string& scenePath,
                                                          std::string& error) {
    std::optional<SceneInputs> inputs = loadSceneInputs(assetRoot, scenePath, error);
    if (!inputs) {
        return nullptr;
    }
    const pathtracer::config::SceneConfig& scene = inputs->scene;
    std::optional<Geometry> geometry = buildGeometry(assetRoot, scene.model, scene.lights, error);
    if (!geometry) {
        return nullptr;
    }
    if (!pathtracer::scene::bindSceneTextures(geometry->model.instances, scene.textures, assetRoot,
                                              inputs->profile.render.textureType)) {
        error = "failed to bind scene textures";
        return nullptr;
    }

    std::optional<pathtracer::scene::Camera> camera = resolveCamera(assetRoot, inputs->profile, error);
    if (!camera) {
        return nullptr;
    }

    // One sample per pass: convergence comes from accumulating passes, as the driver and render_beauty both drive the integrator.
    const pathtracer::scene::PathTraceSettings baseSettings =
        pathtracer::scene::baseSettingsOf(inputs->profile, inputs->material, /*samplesPerPixel=*/1);
    std::optional<std::vector<pathtracer::scene::PathTraceSettings>> perInstanceSettings =
        pathtracer::scene::resolvePerInstanceSettings(baseSettings, geometry->model.instances,
                                                   scene.materialOverrides, assetRoot);
    if (!perInstanceSettings) {
        error = "failed to resolve per-instance material overrides";
        return nullptr;
    }

    return std::unique_ptr<HeadlessRenderer>(new HeadlessRenderer(
        assetRoot, std::move(inputs->profile), std::move(inputs->scene), std::move(*geometry), std::move(*perInstanceSettings),
        baseSettings, std::move(inputs->environmentImage), *camera));
}

HeadlessRenderer::HeadlessRenderer(std::string assetRoot, pathtracer::config::ProfileConfig profile,
                                   pathtracer::config::SceneConfig scene, Geometry geometry,
                                   std::vector<pathtracer::scene::PathTraceSettings> perInstanceSettings,
                                   pathtracer::scene::PathTraceSettings baseSettings,
                                   pathtracer::gfx::ImageTexture environmentImage,
                                   const pathtracer::scene::Camera& defaultCamera)
    : assetRoot_(std::move(assetRoot)),
      profile_(std::move(profile)),
      scene_(std::move(scene)),
      geometry_(std::move(geometry)),
      builtRootRotationDegrees_(scene_.model.rotation),
      builtLightRotationsDegrees_(rotationsOf(scene_.lights)),
      perInstanceSettings_(std::move(perInstanceSettings)),
      baseSettings_(baseSettings),
      environmentMap_(std::move(environmentImage)),
      defaultCamera_(defaultCamera) {}

HeadlessRenderer::~HeadlessRenderer() = default;

bool HeadlessRenderer::applyPose(const Request& request, std::string& error) {
    const glm::vec3 rootRotation = request.rootRotationDegrees.value_or(scene_.model.rotation);
    std::vector<glm::vec3> lightRotations = request.lightRotationsDegrees.value_or(rotationsOf(scene_.lights));
    if (rootRotation == builtRootRotationDegrees_ && lightRotations == builtLightRotationsDegrees_) {
        return true;
    }
    pathtracer::config::ModelConfig model = scene_.model;
    model.rotation = rootRotation;
    std::vector<pathtracer::config::QuadLightConfig> lights = scene_.lights;
    for (std::size_t i = 0; i < lights.size(); ++i) {
        lights[i].rotation = lightRotations[i];
    }
    std::optional<Geometry> geometry = buildGeometry(assetRoot_, model, lights, error);
    if (!geometry) {
        return false;
    }
    // Same glTF and light count, so instances align one to one: the bound materials carry over rather than reloading every texture.
    for (std::size_t i = 0; i < geometry->model.instances.size(); ++i) {
        geometry->model.instances[i].material = std::move(geometry_.model.instances[i].material);
    }
    geometry_ = std::move(*geometry);
    builtRootRotationDegrees_ = rootRotation;
    builtLightRotationsDegrees_ = std::move(lightRotations);
    return true;
}

void HeadlessRenderer::resizeBuffers(int width, int height) {
    if (bufferWidth_ == width && bufferHeight_ == height) {
        return;
    }
    pathTraced_ = pathtracer::scene::makePathTraceResult(width, height);
    // renderGBuffer reallocates on a size change and clears per row otherwise; resetting the stamp says this buffer holds nothing.
    gbuffer_ = pathtracer::scene::GBuffer{};
    accumulators_.clear();
    bufferWidth_ = width;
    bufferHeight_ = height;
}

const pathtracer::gfx::HdrImage& HeadlessRenderer::lastImage(AovId aov) const {
    switch (pathtracer::debug::aovSource(aov)) {
        // at(), not []: an AOV absent from the parallel name vector indexes exactly one past the end, which render() must not do silently.
        case AovSource::PathTraced: {
            const auto it = std::find(accumulatedAovs_.begin(), accumulatedAovs_.end(), aov);
            return accumulators_.at(static_cast<std::size_t>(it - accumulatedAovs_.begin()));
        }
        case AovSource::GBuffer:
            return gbuffer_.*pathtracer::debug::gbufferLane(aov);
        case AovSource::BeautyFilter: {
            const auto it = std::find(filteredAovs_.begin(), filteredAovs_.end(), aov);
            return filtered_.at(static_cast<std::size_t>(it - filteredAovs_.begin()));
        }
    }
    // Not dead: a scoped enum holds any value of its underlying type, so falling off a covered switch is still undefined behaviour.
    return gbuffer_.depth;
}

bool HeadlessRenderer::render(const Request& request, std::span<float* const> outputs,
                               std::string& error) {
    if (outputs.size() != request.aovs.size()) {
        error = "one output buffer is required per requested AOV";
        return false;
    }
    if (!render(request, error)) {
        return false;
    }
    // Every lane is stored at its aovChannels count, the caller's layout, so the boundary costs one straight copy.
    for (std::size_t i = 0; i < request.aovs.size(); ++i) {
        const std::vector<float>& texels = lastImage(request.aovs[i]).texels;
        std::copy(texels.begin(), texels.end(), outputs[i]);
    }
    return true;
}

bool HeadlessRenderer::render(const Request& request, std::string& error) {
    if (!validRequest(request, scene_.lights.size(), error) || !applyPose(request, error)) {
        return false;
    }
    const bool wantsFilter = requestsSource(request.aovs, AovSource::BeautyFilter);
    resizeBuffers(request.width, request.height);
    accumulatedAovs_ = accumulatedAovsFor(request.aovs, wantsFilter);

    stats_.passMilliseconds.clear();
    stats_.gbufferMilliseconds = 0.0;
    stats_.filterMilliseconds = 0.0;
    stats_.rays = pathtracer::debug::RayCounts{};
    if (!accumulatedAovs_.empty()) {
        accumulatePathTraced(request);
    }
    if (requestsSource(request.aovs, AovSource::GBuffer)) {
        renderGBufferLanes(request);
    }
    filteredAovs_.clear();
    filtered_.clear();
    if (wantsFilter) {
        evaluateFilters(request);
    }
    return true;
}

void HeadlessRenderer::accumulatePathTraced(const Request& request) {
    accumulators_.clear();
    for (const AovId aov : accumulatedAovs_) {
        accumulators_.push_back(
            pathtracer::gfx::makeImage(request.width, request.height, pathtracer::debug::aovChannels(aov)));
    }
    // A synchronous caller uses no cooperative cancellation, so generation stays where requestedGeneration asks and never goes stale.
    const std::atomic<std::uint64_t> generation{1};
    pathtracer::debug::PassStats stats;
    // Built per request, as the driver builds one per pass: LightSet holds a pointer and a reference, so it costs one matrix.
    const bool envLightEnabled = request.envLightEnabled.value_or(scene_.environment.lightEnabled);
    const pathtracer::scene::LightSet lights(envLightEnabled ? &environmentMap_ : nullptr, request.envRotationDegrees,
                                             /*envExposure=*/1.0F, geometry_.quadLights);
    // Loop-invariant like `lights` above it: the background is a property of the request, not of the pass index.
    const bool showSky = request.showSky.value_or(pathtracer::scene::kDefaultShowSky);
    stats_.passMilliseconds.reserve(static_cast<std::size_t>(request.samples));
    // Resolved once: the lane set is fixed for this request, and pathTracedLane is a switch the row loop would otherwise re-run.
    std::vector<const std::vector<float>*> laneSources;
    laneSources.reserve(accumulatedAovs_.size());
    for (const AovId aov : accumulatedAovs_) {
        laneSources.push_back(&(pathTraced_.*pathtracer::debug::pathTracedLane(aov)).texels);
    }
    // Only Beauty carries a second moment, and only where it is accumulated at all; -1 leaves the lane zero and SNR black.
    const auto beautyLane = std::find(accumulatedAovs_.begin(), accumulatedAovs_.end(), AovId::Beauty);
    const auto beautyIndex = beautyLane == accumulatedAovs_.end()
                                 ? -1
                                 : static_cast<int>(beautyLane - accumulatedAovs_.begin());
    beautyLuminanceM2_.assign(static_cast<std::size_t>(request.width) * static_cast<std::size_t>(request.height), 0.0F);
    for (int pass = 0; pass < request.samples; ++pass) {
        // Only the trace is timed: the accumulation below it is O(pixels) and identical across revisions.
        const auto passStart = std::chrono::steady_clock::now();
        // scrambleSeed fixed, sampleBase advancing: the pair that keeps accumulated samples stratified rather than N independent draws.
        pathtracer::scene::renderPathTraced(request.camera, geometry_.accel, geometry_.model.shadingTriangles,
                                         geometry_.model.instances, geometry_.instanceLightIndex, lights, request.width, request.height,
                                         showSky, baseSettings_, perInstanceSettings_,
                                         request.scrambleSeed, /*sampleBase=*/pass,
                                         /*sampleCount=*/request.samples, generation,
                                         /*requestedGeneration=*/1U, threadPool_, stats, pathTraced_);
        stats_.passMilliseconds.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - passStart).count());
        accumulatePass(request, pass, laneSources, beautyIndex);
    }
    stats_.rays = stats.rays();
}

void HeadlessRenderer::accumulatePass(const Request& request, int pass,
                                      const std::vector<const std::vector<float>*>& laneSources, int beautyIndex) {
    const auto width = static_cast<std::size_t>(request.width);
    // The driver's own arithmetic, folded in place: same kernels, same invN, same per-element order, so the two means are bit-identical.
    const float invN = 1.0F / static_cast<float>(pass + 1);
    threadPool_.parallelFor(request.height, [&](int y) {
        const std::size_t rowPixel = static_cast<std::size_t>(y) * width;
        // The first pass is kept as drawn, as the driver keeps it: folding into a zero mean would turn -0 into +0 and M2 into NaN at inf.
        if (pass == 0) {
            for (std::size_t lane = 0; lane < laneSources.size(); ++lane) {
                const auto channels = static_cast<std::size_t>(accumulators_[lane].channels);
                const float* drawn = laneSources[lane]->data() + (rowPixel * channels);
                std::copy(drawn, drawn + (width * channels), accumulators_[lane].texels.data() + (rowPixel * channels));
            }
            return;
        }
        if (beautyIndex >= 0) {
            const auto channels = static_cast<std::size_t>(pathTraced_.beauty.channels);
            const auto beauty = static_cast<std::size_t>(beautyIndex);
            const float* mean = accumulators_[beauty].texels.data() + (rowPixel * channels);
            const float* drawn = laneSources[beauty]->data() + (rowPixel * channels);
            float* m2 = beautyLuminanceM2_.data() + rowPixel;
            pathtracer::scene::foldLuminanceM2(mean, drawn, channels, m2, m2, width, invN);
        }
        for (std::size_t lane = 0; lane < laneSources.size(); ++lane) {
            const auto channels = static_cast<std::size_t>(accumulators_[lane].channels);
            float* mean = accumulators_[lane].texels.data() + (rowPixel * channels);
            pathtracer::scene::foldRunningMean(mean, laneSources[lane]->data() + (rowPixel * channels), mean,
                                               width * channels, invN);
        }
    });
}

void HeadlessRenderer::renderGBufferLanes(const Request& request) {
    const auto gbufferStart = std::chrono::steady_clock::now();
    pathtracer::scene::renderGBuffer(request.camera, request.previousCamera.value_or(request.camera), geometry_.accel,
                                     geometry_.model.shadingTriangles, geometry_.model.instances, perInstanceSettings_,
                                     geometry_.instanceBounds,
                                     request.width, request.height, threadPool_, gbuffer_);
    stats_.gbufferMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gbufferStart).count();
}

// Filters read accumulated Beauty, so they run after the loop, each distinct one evaluated once however many AOVs ask for it.
void HeadlessRenderer::evaluateFilters(const Request& request) {
    const auto filterStart = std::chrono::steady_clock::now();
    const pathtracer::gfx::HdrImage& beauty = lastImage(AovId::Beauty);
    const pathtracer::debug::FilterInput filterInput{beauty, beautyLuminanceM2_.data(), request.samples,
                                                     pathtracer::scene::wrapsHorizontally(request.camera.lens().projection)};
    for (const AovId aov : request.aovs) {
        if (pathtracer::debug::aovSource(aov) != AovSource::BeautyFilter ||
            std::find(filteredAovs_.begin(), filteredAovs_.end(), aov) != filteredAovs_.end()) {
            continue;
        }
        filteredAovs_.push_back(aov);
        filtered_.push_back(pathtracer::debug::evaluateFilterAov(aov, filterInput, threadPool_));
    }
    stats_.filterMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - filterStart).count();
}

}  // namespace pathtracer::api
