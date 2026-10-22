#include "pathtracer/api/headless_renderer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/debug/render_stats.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/material_binding.h"
#include "pathtracer/scene/path_trace_driver.h"

namespace pathtracer::api {

namespace {

using pathtracer::debug::AovId;
using pathtracer::debug::AovSource;
using pathtracer::debug::PathTracedLane;

// The files a scene names, loaded and parsed before any geometry: what open() needs to build everything else.
struct SceneInputs {
    pathtracer::config::ProfileConfig profile;
    pathtracer::config::SceneConfig scene;
    std::shared_ptr<const pathtracer::gfx::ImageTexture> environmentTexture;
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
    pathtracer::config::applySceneCamera(*scene, *profile);
    std::shared_ptr<const pathtracer::gfx::ImageTexture> environmentTexture =
        pathtracer::gfx::openTexture(assetRoot + "/" + scene->environment.hdriPath, pathtracer::gfx::kRgbChannels,
                                     pathtracer::gfx::ImageRole::Colour, pathtracer::gfx::TextureWrap::LatLong, scene->environment.colorSpace);
    if (!environmentTexture) {
        error = "failed to load environment " + assetRoot + "/" + scene->environment.hdriPath;
        return std::nullopt;
    }
    return SceneInputs{std::move(*profile), std::move(*scene), std::move(environmentTexture)};
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
                                     config.shutterSeconds, config.iso, config.lens, config.pixelLattice, config.acceptanceAngleDegrees);
    if (!camera.validate(error)) {
        error = "profile.json camera: " + error;
        return std::nullopt;
    }
    return camera;
}

[[nodiscard]] bool requestsSource(const std::vector<AovId>& aovs, AovSource source) {
    return std::any_of(aovs.begin(), aovs.end(), [source](AovId aov) { return pathtracer::debug::aovSource(aov) == source; });
}

[[nodiscard]] std::vector<glm::vec3> rotationsOf(const std::vector<pathtracer::config::LightConfig>& lights) {
    std::vector<glm::vec3> rotations;
    rotations.reserve(lights.size());
    for (const pathtracer::config::LightConfig& light : lights) {
        rotations.push_back(light.rotation);
    }
    return rotations;
}

[[nodiscard]] bool isFinite(const glm::vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Each light's colour for this request: its kelvin override resolved on the Planckian locus, else the authored colour.
[[nodiscard]] std::optional<std::vector<glm::vec3>> lightColours(const HeadlessRenderer::Request& request,
                                                                 const std::vector<pathtracer::config::LightConfig>& lights,
                                                                 std::string& error) {
    std::vector<glm::vec3> colours;
    colours.reserve(lights.size());
    for (std::size_t i = 0; i < lights.size(); ++i) {
        if (!request.lightTemperaturesKelvin.has_value()) {
            colours.push_back(lights[i].color);
            continue;
        }
        const float kelvin = (*request.lightTemperaturesKelvin)[i];
        const std::optional<glm::dvec3> rgb = pathtracer::scene::cie::blackbodyToScene(kelvin);
        if (!rgb) {
            error = "light temperature " + std::to_string(kelvin) + " K is not positive or lies outside the scene gamut";
            return std::nullopt;
        }
        colours.emplace_back(*rgb);
    }
    return colours;
}

// Bloom is an energy fraction, so [0, 1]; the fringe keeps R's magnification 1 - a positive, so [0, 1). NaN fails every comparison.
[[nodiscard]] bool validDisplayOptics(const pathtracer::gfx::DisplayOptics& optics, std::string& error) {
    if (!(optics.bloomStrength >= 0.0F && optics.bloomStrength <= 1.0F)) {
        error = "bloom strength " + std::to_string(optics.bloomStrength) + " lies outside [0, 1]";
        return false;
    }
    if (!(optics.aberration >= 0.0F && optics.aberration < 1.0F)) {
        error = "aberration " + std::to_string(optics.aberration) + " lies outside [0, 1)";
        return false;
    }
    return true;
}

// Rejects a request no producer can serve, before any buffer is touched; error names the first fault.
[[nodiscard]] bool validRequest(const HeadlessRenderer::Request& request, std::size_t lightCount, bool denoiserEnabled, std::string& error) {
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
    if (!denoiserEnabled && std::find(request.aovs.begin(), request.aovs.end(), AovId::Denoised) != request.aovs.end()) {
        error = "denoised was requested, but profile.json disables the denoiser";
        return false;
    }
    if (!request.camera.validate(error)) {
        return false;
    }
    const pathtracer::gfx::PixelLattice lattice = request.camera.pixelLattice();
    const auto unsupported = std::find_if(request.aovs.begin(), request.aovs.end(),
                                          [lattice](AovId aov) { return !pathtracer::debug::aovSupportsLattice(aov, lattice); });
    if (unsupported != request.aovs.end()) {
        error = std::string(pathtracer::debug::kAovNames[static_cast<int>(*unsupported)]) + " runs a network trained on square images, so the " +
                pathtracer::gfx::kPixelLatticeNames[static_cast<int>(lattice)] + " lattice cannot render it";
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
    if (request.lightTemperaturesKelvin.has_value() && request.lightTemperaturesKelvin->size() != lightCount) {
        error = "light temperatures number " + std::to_string(request.lightTemperaturesKelvin->size()) + ", the scene has " +
                std::to_string(lightCount) + " lights";
        return false;
    }
    if (!isFinite(request.envRotationDegrees)) {
        error = "environment rotation is not finite";
        return false;
    }
    return validDisplayOptics(request.displayOptics, error);
}

// The path-traced lanes a request needs, each once. Beauty joins whenever a filter is asked for, every filter reading accumulated Beauty.
[[nodiscard]] std::vector<PathTracedLane> accumulatedLanesFor(const std::vector<AovId>& aovs, bool wantsFilter) {
    std::vector<PathTracedLane> accumulated;
    const auto add = [&accumulated](PathTracedLane lane) {
        if (std::find(accumulated.begin(), accumulated.end(), lane) == accumulated.end()) {
            accumulated.push_back(lane);
        }
    };
    for (const AovId aov : aovs) {
        if (const PathTracedLane lane = pathtracer::debug::pathTracedLane(aov)) {
            add(lane);
        }
        // Denoised reads OIDN's auxiliary features, which must come from the same samples as the Beauty it denoises.
        if (aov == AovId::Denoised) {
            for (const PathTracedLane lane : pathtracer::debug::kDenoiseFeatureLanes) {
                add(lane);
            }
        }
        // Shadow is the ratio of two lane means, both accumulated over the request's samples.
        if (aov == AovId::Shadow) {
            for (const PathTracedLane lane : pathtracer::debug::kShadowLanes) {
                add(lane);
            }
        }
    }
    if (wantsFilter) {
        add(&pathtracer::scene::PathTraceResult::beauty);
    }
    return accumulated;
}

}  // namespace

std::optional<HeadlessRenderer::Geometry> HeadlessRenderer::buildGeometry(
    const std::string& assetRoot, const pathtracer::config::ModelConfig& model,
    const std::vector<pathtracer::config::LightConfig>& lights, std::string& error) {
    const glm::mat4 rootTransform = pathtracer::scene::rootTransformOf(model);
    // No glTF is an empty model: the lights below are then the only triangles.
    std::optional<pathtracer::scene::LoadedModel> loaded =
        model.gltfPath ? pathtracer::scene::loadGltf(assetRoot + "/" + *model.gltfPath, rootTransform, model.vertexColor)
                       : pathtracer::scene::LoadedModel{};
    if (!loaded) {
        error = "failed to load glTF " + assetRoot + "/" + *model.gltfPath;
        return std::nullopt;
    }
    std::vector<int> instanceLightIndex(loaded->instances.size(), -1);
    std::optional<std::vector<pathtracer::scene::AreaLight>> areaLights = pathtracer::scene::buildLights(lights, rootTransform, assetRoot);
    if (!areaLights) {
        error = "failed to build the scene's lights";
        return std::nullopt;
    }
    pathtracer::scene::appendLightGeometry(*loaded, *areaLights, instanceLightIndex);
    // After appendLightGeometry, so the lights' own instances are bounded too.
    std::vector<pathtracer::scene::AabbBounds> instanceBounds =
        pathtracer::scene::computeInstanceBounds(loaded->shadingTriangles, static_cast<int>(loaded->instances.size()), loaded->shapes);
    std::optional<pathtracer::scene::EmbreeAccel> accel =
        pathtracer::scene::EmbreeAccel::build(std::move(loaded->worldTriangles), std::move(loaded->shapes));
    if (!accel) {
        error = "Embree scene build failed";
        return std::nullopt;
    }
    return Geometry{std::move(*loaded), std::move(*areaLights), std::move(instanceLightIndex), std::move(instanceBounds),
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
    // Materials before textures, which bind over the constants; each call logs the reason it fails. Without a model there are none.
    if (scene.materialPath &&
        (!pathtracer::scene::applySceneMaterials(geometry->model.instances, *scene.materialPath, scene.materialOverrides, assetRoot) ||
         !pathtracer::scene::bindSceneTextures(geometry->model.instances, scene.textures, assetRoot))) {
        error = "failed to bind scene materials";
        return nullptr;
    }
    // World-space media, independent of the pose a request may change, so loaded once here rather than in buildGeometry.
    std::optional<pathtracer::scene::VolumeSet> volumes = pathtracer::scene::VolumeSet::load(scene.volumes, assetRoot, error);
    if (!volumes) {
        return nullptr;
    }

    std::optional<pathtracer::scene::Camera> camera = resolveCamera(assetRoot, inputs->profile, error);
    if (!camera) {
        return nullptr;
    }

    pathtracer::scene::EnvironmentMap environmentMap(std::move(inputs->environmentTexture), scene.environment.illuminanceLux);
    if (scene.environment.illuminanceLux && !(environmentMap.upwardIlluminance() > 0.0F)) {
        error = "environment.lux needs a map with light above its horizon";
        return nullptr;
    }

    const pathtracer::scene::PathTraceSettings baseSettings = pathtracer::scene::baseSettingsOf(inputs->profile);

    return std::unique_ptr<HeadlessRenderer>(new HeadlessRenderer(
        assetRoot, std::move(inputs->profile), std::move(inputs->scene), std::move(*geometry), std::move(*volumes),
        baseSettings, std::move(environmentMap), *camera));
}

HeadlessRenderer::HeadlessRenderer(std::string assetRoot, pathtracer::config::ProfileConfig profile,
                                   pathtracer::config::SceneConfig scene, Geometry geometry, pathtracer::scene::VolumeSet volumes,
                                   pathtracer::scene::PathTraceSettings baseSettings,
                                   pathtracer::scene::EnvironmentMap environmentMap, const pathtracer::scene::Camera& defaultCamera)
    : assetRoot_(std::move(assetRoot)),
      profile_(std::move(profile)),
      scene_(std::move(scene)),
      geometry_(std::move(geometry)),
      volumes_(std::move(volumes)),
      builtRootRotationDegrees_(scene_.model.rotation),
      builtLightRotationsDegrees_(rotationsOf(scene_.lights)),
      baseSettings_(baseSettings),
      environmentMap_(std::move(environmentMap)),
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
    std::vector<pathtracer::config::LightConfig> lights = scene_.lights;
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
    pathTraced_ = pathtracer::scene::makePathTraceResult(width, height, baseSettings_.replicates);
    // renderGBuffer reallocates on a size change and clears per row otherwise; resetting the stamp says this buffer holds nothing.
    gbuffer_ = pathtracer::scene::GBuffer{};
    accumulators_.clear();
    bufferWidth_ = width;
    bufferHeight_ = height;
}

const pathtracer::gfx::HdrImage& HeadlessRenderer::accumulatedImage(PathTracedLane lane) const {
    // at(), not []: a lane absent from the parallel lane vector indexes exactly one past the end, which render() must not do silently.
    const auto it = std::find(accumulatedLanes_.begin(), accumulatedLanes_.end(), lane);
    return accumulators_.at(static_cast<std::size_t>(it - accumulatedLanes_.begin()));
}

const pathtracer::gfx::HdrImage& HeadlessRenderer::lastImage(AovId aov) const {
    switch (pathtracer::debug::aovSource(aov)) {
        case AovSource::PathTraced:
            return accumulatedImage(pathtracer::debug::pathTracedLane(aov));
        case AovSource::GBuffer:
            return gbuffer_.*pathtracer::debug::gbufferLane(aov);
        // at(), not []: an AOV absent from the parallel name vector indexes exactly one past the end, which render() must not do silently.
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
    if (!validRequest(request, scene_.lights.size(), profile_.denoiser.enabled, error)) {
        return false;
    }
    const std::optional<std::vector<glm::vec3>> colours = lightColours(request, scene_.lights, error);
    if (!colours || !applyPose(request, error)) {
        return false;
    }
    // Radiance is all a light's colour reaches and selection is uniform, so a new colour rewrites it here and never the BVH.
    for (std::size_t i = 0; i < colours->size(); ++i) {
        const float luminance = pathtracer::scene::emittedLuminance(scene_.lights[i], geometry_.lights[i]);
        std::visit([&](auto& light) { light.radiance = (*colours)[i] * luminance; }, geometry_.lights[i]);
    }
    const bool wantsFilter = requestsSource(request.aovs, AovSource::BeautyFilter);
    resizeBuffers(request.width, request.height);
    accumulatedLanes_ = accumulatedLanesFor(request.aovs, wantsFilter);

    stats_.passMilliseconds.clear();
    stats_.gbufferMilliseconds = 0.0;
    stats_.filterMilliseconds = 0.0;
    stats_.opticsMilliseconds = 0.0;
    stats_.rays = pathtracer::debug::RayCounts{};
    if (!accumulatedLanes_.empty()) {
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
    applyDisplayOptics(request);
    return true;
}

void HeadlessRenderer::accumulatePathTraced(const Request& request) {
    accumulators_.clear();
    for (const PathTracedLane lane : accumulatedLanes_) {
        accumulators_.push_back(pathtracer::gfx::makeImage(request.width, request.height, (pathTraced_.*lane).channels));
    }
    // A synchronous caller uses no cooperative cancellation, so generation stays where requestedGeneration asks and never goes stale.
    const std::atomic<std::uint64_t> generation{1};
    pathtracer::debug::PassStats stats;
    // Built per request, as the driver builds one per pass: LightSet holds a pointer and a reference, so it costs one matrix.
    const bool envLightEnabled = request.envLightEnabled.value_or(scene_.environment.lightEnabled);
    const pathtracer::scene::LightSet lights(envLightEnabled ? &environmentMap_ : nullptr, request.envRotationDegrees,
                                             /*envExposure=*/1.0F, geometry_.lights);
    // Loop-invariant like `lights` above it: the background is a property of the request, not of the pass index.
    const bool showSky = request.showSky.value_or(pathtracer::scene::kDefaultShowSky);
    stats_.passMilliseconds.reserve(static_cast<std::size_t>(request.samples));
    // Resolved once: the lane set is fixed for this request, and pathTracedLane is a switch the row loop would otherwise re-run.
    std::vector<const std::vector<float>*> laneSources;
    laneSources.reserve(accumulatedLanes_.size());
    for (const PathTracedLane lane : accumulatedLanes_) {
        laneSources.push_back(&(pathTraced_.*lane).texels);
    }
    // renderPathTraced writes images only, so pathTraced_ holds the replicate means and counts across passes, folded in place from zero.
    std::fill(pathTraced_.replicateLuminance.begin(), pathTraced_.replicateLuminance.end(), 0.0F);
    std::fill(pathTraced_.texelSamples.begin(), pathTraced_.texelSamples.end(), 0);
    activeTexels_ = pathtracer::scene::allTexels(request.width, request.height);
    // Ends at request.samples, or earlier once the stopping rule has retired every texel.
    for (int pass = 0; pass < request.samples && !activeTexels_.empty(); ++pass) {
        // Only the trace is timed: the accumulation below it is O(pixels) and identical across revisions.
        const auto passStart = std::chrono::steady_clock::now();
        // scrambleSeed fixed, sampleBase advancing: the pair that keeps accumulated samples stratified rather than N independent draws.
        pathtracer::scene::renderPathTraced(request.camera, geometry_.accel, geometry_.model.shadingTriangles,
                                         geometry_.model.instances, geometry_.instanceLightIndex, lights, volumes_, request.width,
                                         request.height, activeTexels_,
                                         showSky, baseSettings_,
                                         request.scrambleSeed, /*sampleBase=*/pass,
                                         /*sampleCount=*/request.samples, generation,
                                         /*requestedGeneration=*/1U, threadPool_, stats, pathTraced_);
        stats_.passMilliseconds.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - passStart).count());
        accumulatePass(request, pass, laneSources);
        if (pathtracer::scene::adaptiveCheckpoint(pass + 1, baseSettings_)) {
            const pathtracer::scene::FoveatedThreshold fovea(request.camera, request.width, request.height,
                                                             baseSettings_.adaptive.marSlope);
            pathtracer::scene::retireConvergedTexels(activeTexels_, pathTraced_.replicateLuminance.data(), baseSettings_, fovea,
                                                     threadPool_);
        }
    }
    pathTraced_.activeTexels = activeTexels_.size();
    stats_.rays = stats.rays();
}

void HeadlessRenderer::accumulatePass(const Request& request, int pass, const std::vector<const std::vector<float>*>& laneSources) {
    const auto width = static_cast<std::uint32_t>(request.width);
    // The driver's own arithmetic, folded in place: same kernels, same invN, same per-element order, so the two means are bit-identical.
    const float invN = 1.0F / static_cast<float>(pass + 1);
    const int replicates = baseSettings_.replicates;
    // The driver's mapping: pass p is replicate p mod K's (p div K + 1)th, folded in place from zeroed slots.
    const int replicate = pass % replicates;
    const int count = (pass / replicates) + 1;
    const float invCount = 1.0F / static_cast<float>(count);
    const auto stride = static_cast<std::size_t>(replicates);
    const auto beautyChannels = static_cast<std::size_t>(pathTraced_.beauty.channels);
    const auto fold = [&](std::uint32_t begin, std::uint32_t end) {
        const std::size_t run = end - begin;
        // From the pass's own Beauty, which renderPathTraced always writes, so the means exist whichever lanes are accumulated.
        float* means = pathTraced_.replicateLuminance.data() + (begin * stride);
        pathtracer::scene::foldReplicateLuminance(means, pathTraced_.beauty.texels.data() + (begin * beautyChannels), beautyChannels, means,
                                                  run, replicates, replicate, invCount);
        std::fill(pathTraced_.texelSamples.data() + begin, pathTraced_.texelSamples.data() + end, pass + 1);
        for (std::size_t lane = 0; lane < laneSources.size(); ++lane) {
            const auto channels = static_cast<std::size_t>(accumulators_[lane].channels);
            const float* drawn = laneSources[lane]->data() + (begin * channels);
            float* mean = accumulators_[lane].texels.data() + (begin * channels);
            // The first pass is kept as drawn, as the driver keeps it: folding into a zero mean would turn -0 into +0.
            if (pass == 0) {
                std::copy(drawn, drawn + (run * channels), mean);
            } else {
                pathtracer::scene::foldRunningMean(mean, drawn, mean, run * channels, invN);
            }
        }
    };
    // In place, an untraced texel's mean is already where it stays.
    const auto carry = [](std::uint32_t, std::uint32_t) {};
    threadPool_.parallelFor(request.height, [&](int y) {
        const std::uint32_t rowBegin = static_cast<std::uint32_t>(y) * width;
        pathtracer::scene::forEachTracedRun(activeTexels_, rowBegin, rowBegin + width, fold, carry);
    });
}

void HeadlessRenderer::renderGBufferLanes(const Request& request) {
    const auto gbufferStart = std::chrono::steady_clock::now();
    pathtracer::scene::renderGBuffer(request.camera, request.previousCamera.value_or(request.camera), geometry_.accel,
                                     geometry_.model.shadingTriangles, geometry_.model.instances, baseSettings_.lookaheadDistance,
                                     geometry_.instanceBounds,
                                     request.width, request.height, threadPool_, gbuffer_);
    stats_.gbufferMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gbufferStart).count();
}

// Filters read accumulated Beauty, so they run after the loop, each distinct one evaluated once however many AOVs ask for it.
void HeadlessRenderer::evaluateFilters(const Request& request) {
    const auto filterStart = std::chrono::steady_clock::now();
    const pathtracer::gfx::HdrImage& beauty = lastImage(AovId::Beauty);
    const bool denoises = std::find(request.aovs.begin(), request.aovs.end(), AovId::Denoised) != request.aovs.end();
    const bool shadows = std::find(request.aovs.begin(), request.aovs.end(), AovId::Shadow) != request.aovs.end();
    if (denoises && denoiser_ == nullptr) {
        denoiser_ = std::make_unique<pathtracer::debug::Denoiser>(profile_.denoiser.options);
    }
    const pathtracer::debug::FilterInput filterInput{beauty,
                                                     pathTraced_.replicateLuminance.data(),
                                                     pathTraced_.replicates,
                                                     request.samples,
                                                     pathTraced_.texelSamples.data(),
                                                     pathtracer::scene::wrapsHorizontally(request.camera.lens().projection),
                                                     request.camera.pixelLattice(),
                                                     denoises ? &accumulatedImage(pathtracer::debug::kDenoiseFeatureLanes[0]) : nullptr,
                                                     denoises ? &accumulatedImage(pathtracer::debug::kDenoiseFeatureLanes[1]) : nullptr,
                                                     denoiser_.get(),
                                                     shadows ? &accumulatedImage(pathtracer::debug::kShadowLanes[0]) : nullptr,
                                                     shadows ? &accumulatedImage(pathtracer::debug::kShadowLanes[1]) : nullptr,
                                                     &request.camera};
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

// Picture AOVs are path-traced (Beauty) or Beauty-derived (Denoised), never G-buffer, so each lives in an accumulator or a filter result.
void HeadlessRenderer::applyDisplayOptics(const Request& request) {
    if (request.displayOptics.aberration == 0.0F && request.displayOptics.bloomStrength == 0.0F) {
        return;
    }
    const auto opticsStart = std::chrono::steady_clock::now();
    const pathtracer::gfx::PixelLattice lattice = request.camera.pixelLattice();
    const bool wraps = pathtracer::scene::wrapsHorizontally(request.camera.lens().projection);
    std::vector<AovId> applied;
    for (const AovId aov : request.aovs) {
        if (!pathtracer::debug::aovIsPicture(aov) || std::find(applied.begin(), applied.end(), aov) != applied.end()) {
            continue;
        }
        applied.push_back(aov);
        const PathTracedLane lane = pathtracer::debug::pathTracedLane(aov);
        pathtracer::gfx::HdrImage& image =
            pathtracer::debug::aovSource(aov) == AovSource::PathTraced
                ? accumulators_.at(static_cast<std::size_t>(std::find(accumulatedLanes_.begin(), accumulatedLanes_.end(), lane) -
                                                            accumulatedLanes_.begin()))
                : filtered_.at(static_cast<std::size_t>(std::find(filteredAovs_.begin(), filteredAovs_.end(), aov) -
                                                        filteredAovs_.begin()));
        pathtracer::gfx::applyDisplayOptics(image, lattice, wraps, request.displayOptics, threadPool_);
    }
    stats_.opticsMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - opticsStart).count();
}

}  // namespace pathtracer::api
