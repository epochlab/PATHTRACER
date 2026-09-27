#include "pathtracer/api/headless_renderer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <optional>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/debug/render_stats.h"
#include "pathtracer/scene/material_binding.h"

namespace pathtracer::api {

namespace {

using pathtracer::debug::AovId;
using pathtracer::debug::AovSource;
using pathtracer::gfx::HdrImage;

// Scene-level placement, order X,Y,Z -- identical to main.cpp's and render_beauty's, or headless places the scene differently.
[[nodiscard]] glm::mat4 rootTransformOf(const pathtracer::config::SceneConfig& scene) {
    return glm::translate(glm::mat4(1.0F), scene.model.position) *
           glm::rotate(glm::mat4(1.0F), glm::radians(scene.model.rotation.z), glm::vec3(0.0F, 0.0F, 1.0F)) *
           glm::rotate(glm::mat4(1.0F), glm::radians(scene.model.rotation.y), glm::vec3(0.0F, 1.0F, 0.0F)) *
           glm::rotate(glm::mat4(1.0F), glm::radians(scene.model.rotation.x), glm::vec3(1.0F, 0.0F, 0.0F));
}

// samplesPerPixel is 1 here; convergence comes from accumulating passes, as the driver and render_beauty both drive the integrator.
[[nodiscard]] pathtracer::scene::PathTraceSettings baseSettingsOf(const pathtracer::config::ProfileConfig& profile,
                                                               const pathtracer::config::MaterialConfig& material) {
    return pathtracer::scene::PathTraceSettings{
        .samplesPerPixel = 1,
        .maxBounces = profile.pathTracer.maxBounces,
        .russianRouletteStartBounce = profile.pathTracer.russianRouletteStartBounce,
        .aoMaxDistance = profile.pathTracer.aoMaxDistance,
        .lookaheadDistance = profile.pathTracer.lookaheadDistance,
        .bumpStrength = material.bumpStrength,
        .roughnessMin = material.roughnessMin,
        .roughnessMax = material.roughnessMax,
        .diffuseColour = material.diffuseColour,
        .ior = material.ior,
        .abbe = material.abbe,
        .transmissionFactor = material.transmissionFactor,
        .metallicFactor = material.metallicFactor,
        .roughnessFactor = material.roughnessFactor,
        .diffuseRoughness = material.diffuseRoughness,
        .transmissionColor = material.transmissionColor,
        .transmissionDepth = material.transmissionDepth,
        .edgeTint = material.edgeTint,
    };
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
    const pathtracer::config::CameraConfig& camera = profile.camera;
    return pathtracer::scene::Camera(camera.position, camera.yawDegrees, camera.pitchDegrees, preset->filmBack,
                                  camera.focalLengthMm, camera.nearClip, camera.farClip, camera.aperture,
                                  camera.shutterSeconds, camera.iso, camera.lens);
}

// Gathers `channels` of each texel out of HdrImage's fixed RGBA into a packed destination: the one copy the boundary costs.
void packChannels(const HdrImage& source, int channels, float* destination,
                  pathtracer::scene::ThreadPool& threadPool) {
    const int width = source.width;
    threadPool.parallelFor(source.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            for (int c = 0; c < channels; ++c) {
                destination[(pixel * static_cast<std::size_t>(channels)) + static_cast<std::size_t>(c)] =
                    source.rgba[(pixel * 4) + static_cast<std::size_t>(c)];
            }
        }
    });
}

}  // namespace

std::unique_ptr<HeadlessRenderer> HeadlessRenderer::open(const std::string& assetRoot,
                                                          const std::string& scenePath,
                                                          std::string& error) {
    const std::optional<pathtracer::config::ProfileConfig> profile =
        pathtracer::config::loadProfileConfig(assetRoot + "/config/profile.json");
    if (!profile) {
        error = "failed to load " + assetRoot + "/config/profile.json";
        return nullptr;
    }
    const std::optional<pathtracer::config::SceneConfig> scene =
        pathtracer::config::loadSceneConfig(assetRoot + "/" + scenePath);
    if (!scene) {
        error = "failed to load scene " + assetRoot + "/" + scenePath;
        return nullptr;
    }
    const std::optional<pathtracer::config::MaterialConfig> material =
        pathtracer::config::loadMaterialConfig(assetRoot + "/" + scene->materialPath);
    if (!material) {
        error = "failed to load material " + assetRoot + "/" + scene->materialPath;
        return nullptr;
    }
    std::optional<pathtracer::gfx::ImageTexture> environmentImage = pathtracer::gfx::loadImageTexture(
        assetRoot + "/" + scene->environment.hdriPath, profile->render.textureType);
    if (!environmentImage) {
        error = "failed to load environment " + assetRoot + "/" + scene->environment.hdriPath;
        return nullptr;
    }

    const glm::mat4 rootTransform = rootTransformOf(*scene);
    std::optional<pathtracer::scene::LoadedModel> model = pathtracer::scene::loadGltf(
        assetRoot + "/" + scene->model.gltfPath, profile->render.textureType, rootTransform,
        scene->model.texturePath.empty() ? "" : assetRoot + "/" + scene->model.texturePath);
    if (!model) {
        error = "failed to load glTF " + assetRoot + "/" + scene->model.gltfPath;
        return nullptr;
    }

    std::vector<int> instanceLightIndex(model->instances.size(), -1);
    std::vector<pathtracer::scene::QuadLight> quadLights =
        pathtracer::scene::buildQuadLights(scene->lights, rootTransform);
    pathtracer::scene::appendQuadLights(*model, quadLights, instanceLightIndex);

    std::optional<pathtracer::scene::Camera> camera = resolveCamera(assetRoot, *profile, error);
    if (!camera) {
        return nullptr;
    }

    const pathtracer::scene::PathTraceSettings baseSettings = baseSettingsOf(*profile, *material);
    std::optional<std::vector<pathtracer::scene::PathTraceSettings>> perInstanceSettings =
        pathtracer::scene::resolvePerInstanceSettings(baseSettings, model->instances,
                                                   scene->materialOverrides, assetRoot);
    if (!perInstanceSettings) {
        error = "failed to resolve per-instance material overrides";
        return nullptr;
    }

    std::optional<pathtracer::scene::EmbreeAccel> accel =
        pathtracer::scene::EmbreeAccel::build(std::move(model->worldTriangles));
    if (!accel) {
        error = "Embree scene build failed";
        return nullptr;
    }

    return std::unique_ptr<HeadlessRenderer>(new HeadlessRenderer(
        *profile, std::move(*model), std::move(quadLights), std::move(instanceLightIndex),
        std::move(*perInstanceSettings), baseSettings, std::move(*accel), std::move(*environmentImage),
        scene->environment.lightEnabled, *camera));
}

HeadlessRenderer::HeadlessRenderer(pathtracer::config::ProfileConfig profile,
                                    pathtracer::scene::LoadedModel model,
                                    std::vector<pathtracer::scene::QuadLight> quadLights,
                                    std::vector<int> instanceLightIndex,
                                    std::vector<pathtracer::scene::PathTraceSettings> perInstanceSettings,
                                    pathtracer::scene::PathTraceSettings baseSettings,
                                    pathtracer::scene::EmbreeAccel accel,
                                    pathtracer::gfx::ImageTexture environmentImage, bool envLightEnabled,
                                    const pathtracer::scene::Camera& defaultCamera)
    : profile_(std::move(profile)),
      model_(std::move(model)),
      instanceLightIndex_(std::move(instanceLightIndex)),
      perInstanceSettings_(std::move(perInstanceSettings)),
      baseSettings_(baseSettings),
      instanceBounds_(pathtracer::scene::computeInstanceBounds(model_.shadingTriangles, static_cast<int>(model_.instances.size()))),
      accel_(std::move(accel)),
      environmentMap_(std::move(environmentImage)),
      quadLights_(std::move(quadLights)),
      lights_(&environmentMap_, /*envRotationRadians=*/0.0F, /*envExposure=*/1.0F, quadLights_),
      lightsEnvOff_(nullptr, /*envRotationRadians=*/0.0F, /*envExposure=*/1.0F, quadLights_),
      defaultEnvLightEnabled_(envLightEnabled),
      defaultCamera_(defaultCamera) {}

HeadlessRenderer::~HeadlessRenderer() = default;

void HeadlessRenderer::resizeBuffers(int width, int height) {
    if (bufferWidth_ == width && bufferHeight_ == height) {
        return;
    }
    pathTraced_ = pathtracer::scene::makePathTraceResult(width, height);
    // renderRasterGBuffer reallocates on a size change and clears per row otherwise; resetting the stamp says this buffer holds nothing.
    gbuffer_ = pathtracer::scene::RasterGBuffer{};
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
    for (std::size_t i = 0; i < request.aovs.size(); ++i) {
        packChannels(lastImage(request.aovs[i]), pathtracer::debug::aovChannels(request.aovs[i]), outputs[i],
                     threadPool_);
    }
    return true;
}

bool HeadlessRenderer::render(const Request& request, std::string& error) {
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
    const bool wantsFilter = std::any_of(request.aovs.begin(), request.aovs.end(), [](AovId aov) {
        return pathtracer::debug::aovSource(aov) == AovSource::BeautyFilter;
    });
    const bool wantsGBuffer = std::any_of(request.aovs.begin(), request.aovs.end(), [](AovId aov) {
        return pathtracer::debug::aovSource(aov) == AovSource::GBuffer;
    });
    // Scan conversion is a perspective divide, which a fisheye has no equivalent of: rejected rather than silently approximated.
    if (wantsGBuffer && request.camera.lens().projection != pathtracer::scene::LensProjection::Rectilinear) {
        error = "G-buffer AOVs require a rectilinear lens: a fisheye has no rasterizer projection";
        return false;
    }
    resizeBuffers(request.width, request.height);

    // The path-traced lanes this request needs. Beauty joins whenever a filter is asked for, every filter reading accumulated Beauty.
    accumulatedAovs_.clear();
    for (const AovId aov : request.aovs) {
        if (pathtracer::debug::aovSource(aov) == AovSource::PathTraced &&
            std::find(accumulatedAovs_.begin(), accumulatedAovs_.end(), aov) == accumulatedAovs_.end()) {
            accumulatedAovs_.push_back(aov);
        }
    }
    if (wantsFilter && std::find(accumulatedAovs_.begin(), accumulatedAovs_.end(), AovId::Beauty) ==
                            accumulatedAovs_.end()) {
        accumulatedAovs_.push_back(AovId::Beauty);
    }

    stats_.passMilliseconds.clear();
    stats_.rasterMilliseconds = 0.0;
    stats_.filterMilliseconds = 0.0;
    stats_.rays = pathtracer::debug::RayCounts{};
    if (!accumulatedAovs_.empty()) {
        accumulators_.assign(accumulatedAovs_.size(),
                              pathtracer::gfx::HdrImage{request.width, request.height,
                                                     std::vector<float>(static_cast<std::size_t>(request.width) *
                                                                            static_cast<std::size_t>(request.height) * 4,
                                                                        0.0F)});
        // A synchronous caller uses no cooperative cancellation, so generation stays where requestedGeneration asks and never goes stale.
        const std::atomic<std::uint64_t> generation{1};
        pathtracer::debug::PassStats stats;
        const pathtracer::scene::LightSet& lights =
            request.envLightEnabled.value_or(defaultEnvLightEnabled_) ? lights_ : lightsEnvOff_;
        // Loop-invariant like `lights` above it: the background is a property of the request, not of the pass index.
        const bool showSky = request.showSky.value_or(true);
        stats_.passMilliseconds.reserve(static_cast<std::size_t>(request.samples));
        // Resolved once: the lane set is fixed for this request, and pathTracedLane is a switch the row loop would otherwise re-run.
        std::vector<const std::vector<float>*> laneSources;
        laneSources.reserve(accumulatedAovs_.size());
        for (const AovId aov : accumulatedAovs_) {
            laneSources.push_back(&(pathTraced_.*pathtracer::debug::pathTracedLane(aov)).rgba);
        }
        const auto rowFloats = static_cast<std::size_t>(request.width) * 4;
        // Only Beauty carries a second moment, and only where it is accumulated at all; -1 leaves the lane zero and SNR black.
        const auto beautyLane = std::find(accumulatedAovs_.begin(), accumulatedAovs_.end(), AovId::Beauty);
        const auto beautyIndex = beautyLane == accumulatedAovs_.end()
                                     ? -1
                                     : static_cast<int>(beautyLane - accumulatedAovs_.begin());
        beautyLuminanceM2_.assign(static_cast<std::size_t>(request.width) * static_cast<std::size_t>(request.height), 0.0F);
        const glm::vec3 weights = pathtracer::debug::kRec709LuminanceWeights;
        for (int pass = 0; pass < request.samples; ++pass) {
            // Only the trace is timed: the accumulation below it is O(pixels) and identical across revisions.
            const auto passStart = std::chrono::steady_clock::now();
            // scrambleSeed fixed, sampleBase advancing: the pair that keeps accumulated samples stratified rather than N independent draws.
            pathtracer::scene::renderPathTraced(request.camera, accel_, model_.shadingTriangles, model_.instances,
                                             instanceLightIndex_, lights, request.width, request.height,
                                             showSky, baseSettings_, perInstanceSettings_,
                                             request.scrambleSeed, /*sampleBase=*/pass,
                                             /*sampleCount=*/request.samples, generation,
                                             /*requestedGeneration=*/1U, threadPool_, stats, pathTraced_);
            stats_.passMilliseconds.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - passStart).count());
            // By row, as the interactive driver's accumulateMean is: each element's chain stays in order, so the sum is bit-identical.
            threadPool_.parallelFor(request.height, [&](int y) {
                const std::size_t begin = static_cast<std::size_t>(y) * rowFloats;
                if (beautyIndex >= 0) {
                    const float* drawnLane = laneSources[static_cast<std::size_t>(beautyIndex)]->data();
                    // Still the passes before this one, so dividing recovers exactly the mean the driver's own lane would hold.
                    const float* runningSum = accumulators_[static_cast<std::size_t>(beautyIndex)].rgba.data();
                    const float inversePrevious = pass == 0 ? 0.0F : 1.0F / static_cast<float>(pass);
                    const float inverseCount = 1.0F / static_cast<float>(pass + 1);
                    for (std::size_t i = begin; i < begin + rowFloats; i += 4) {
                        const float drawn = (drawnLane[i] * weights.r) + (drawnLane[i + 1] * weights.g) +
                                            (drawnLane[i + 2] * weights.b);
                        // At the first pass the mean is the draw, so Welford's product is exactly zero with no branch of its own.
                        const float before = pass == 0 ? drawn
                                                       : ((runningSum[i] * weights.r) + (runningSum[i + 1] * weights.g) +
                                                          (runningSum[i + 2] * weights.b)) * inversePrevious;
                        const float after = before + ((drawn - before) * inverseCount);
                        beautyLuminanceM2_[i / 4] += (drawn - before) * (drawn - after);
                    }
                }
                for (std::size_t lane = 0; lane < laneSources.size(); ++lane) {
                    const float* source = laneSources[lane]->data();
                    float* sum = accumulators_[lane].rgba.data();
                    for (std::size_t i = begin; i < begin + rowFloats; ++i) {
                        sum[i] += source[i];
                    }
                }
            });
        }
        stats_.rays = stats.rays();
        const auto passes = static_cast<float>(request.samples);
        threadPool_.parallelFor(request.height, [&](int y) {
            const std::size_t begin = static_cast<std::size_t>(y) * rowFloats;
            for (pathtracer::gfx::HdrImage& accumulator : accumulators_) {
                float* rgba = accumulator.rgba.data();
                for (std::size_t i = begin; i < begin + rowFloats; ++i) {
                    rgba[i] /= passes;
                }
            }
        });
    }

    if (wantsGBuffer) {
        const auto rasterStart = std::chrono::steady_clock::now();
        pathtracer::scene::renderRasterGBuffer(request.camera, model_.shadingTriangles, model_.instances,
                                            perInstanceSettings_, instanceBounds_, request.width,
                                            request.height, threadPool_, gbuffer_);
        stats_.rasterMilliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rasterStart).count();
    }

    // Filters read accumulated Beauty, so they run after the loop, each distinct one evaluated once however many AOVs ask for it.
    filteredAovs_.clear();
    filtered_.clear();
    if (wantsFilter) {
        const auto filterStart = std::chrono::steady_clock::now();
        const pathtracer::gfx::HdrImage& beauty = lastImage(AovId::Beauty);
        const pathtracer::debug::FilterInput filterInput{beauty, request.camera.pixelsPerRadian(beauty.height),
                                                          beautyLuminanceM2_.data(), request.samples};
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

    return true;
}

}  // namespace pathtracer::api
