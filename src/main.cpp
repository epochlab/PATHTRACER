#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// GLEW before GLFW: see gl_debug.cpp for why.
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <OpenColorIO/OpenColorIO.h>

#include "pathtracer/config/profile_config.h"
#include "pathtracer/config/scene_config.h"
#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/bench_log.h"
#include "pathtracer/debug/colormap.h"
#include "pathtracer/debug/frame_stats.h"
#include "pathtracer/debug/gpu_timer.h"
#include "pathtracer/debug/histogram.h"
#include "pathtracer/debug/perf_dashboard.h"
#include "pathtracer/debug/render_stats.h"
#include "pathtracer/debug/spec_report.h"
#include "pathtracer/debug/hud_overlay.h"
#include "pathtracer/debug/memory_tracker.h"
#include "pathtracer/debug/scene_stats.h"
#include "pathtracer/debug/system_info.h"
#include "pathtracer/gfx/gl_debug.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/ocio_display_transform.h"
#include "pathtracer/gfx/post_process_pass.h"
#include "pathtracer/gfx/shader_program.h"
#include "pathtracer/gfx/texture.h"
#include "pathtracer/gfx/viewport.h"
#include "pathtracer/platform/display_link.h"
#include "pathtracer/platform/window.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/debug_camera_controller.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/environment_map.h"
#include "pathtracer/scene/false_color.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/light.h"
#include "pathtracer/scene/material_binding.h"
#include "pathtracer/scene/path_trace_driver.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/rasterizer.h"
#include "pathtracer/scene/thread_pool.h"

namespace OCIO = OCIO_NAMESPACE;

namespace {

void glfwErrorCallback(int error, const char* description) {
    std::cerr << "GLFW error " << error << ": " << description << '\n';
}

const char* lutName(pathtracer::gfx::OcioDisplayTransform::Lut lut) {
    using Lut = pathtracer::gfx::OcioDisplayTransform::Lut;
    return lut == Lut::SRGB ? "sRGB" : lut == Lut::Rec709 ? "Rec709" : "Raw";
}

// Camera geometry, every renderRasterGBuffer input that can change: factored so the two producers compare the same fields.
struct ViewInputState {
    glm::vec3 cameraPosition{0.0F};
    float cameraYawDegrees = 0.0F;
    float cameraPitchDegrees = 0.0F;
    float focalLengthMm = 0.0F;
    // The only FilmBack component feeding the render; widthMm is display-only, so tracking it would retrace for no visible effect.
    float filmBackHeightMm = 0.0F;

    bool operator==(const ViewInputState&) const = default;
};

// Every input renderPathTraced depends on bar the resolution, compared frame to frame.
struct PathTraceInputState {
    ViewInputState view;
    int envRotationDegrees = 0;
    bool showSky = false;
    bool envLightEnabled = true;
    float envExposureStops = 0.0F;

    bool operator==(const PathTraceInputState&) const = default;
};

// The inputs plus the scale they render at. Split, because folding renderScale in would make the settle promotion read as fresh input.
struct PathTraceTriggerState {
    PathTraceInputState input;
    float renderScale = 0.0F;  // sentinel, never a real value: profile_config.h bounds it to (0,1]

    bool operator==(const PathTraceTriggerState&) const = default;
};

// The rasterizer's own last-rendered state: separate because an environment change must retrace without re-rasterizing a G-buffer.
struct RasterTriggerState {
    ViewInputState view;
    float renderScale = 0.0F;  // same sentinel, same bound

    bool operator==(const RasterTriggerState&) const = default;
};

// Seconds of no input before promoting back to full renderScale: longer than the gaps between drag events, short enough to feel immediate.
constexpr double kInteractiveSettleSeconds = 0.25;

// max(1) so a non-empty framebuffer never scales to a zero-pixel target; a genuinely empty one stays 0 for the caller's own guard.
int scaledExtent(int framebufferExtent, float scale) {
    if (framebufferExtent <= 0) {
        return 0;
    }
    return std::max(1, static_cast<int>(std::lround(static_cast<float>(framebufferExtent) * scale)));
}

// Everything the loop touches each frame plus state outliving the run. -bench appends raw columns at exit; stage 0 is the warm-up.
struct BenchCapture {
    std::string logPath;
    std::vector<std::string> argv;
    std::vector<int> aovs;  // the -bench-aovs schedule; empty = single-stage, and `stage` then never leaves 0
    std::size_t stage = 0;  // index into aovs of the AOV currently selected
    std::chrono::steady_clock::time_point stageStart;
    std::vector<float> stageWallMs;  // one entry per measured stage, so stages 1..aovs.size()-1
    std::uint64_t generation = 0;  // requestTrace's generation for the accumulation being captured
    std::vector<pathtracer::debug::PassRecord> passes;
    std::vector<pathtracer::debug::FrameStageTimes> frames;
    std::vector<float> frameMs;
    std::vector<float> presentGpuMs;
    // Per frame, not in config: a measurement of the display link's period, so it differs run to run, where config demands equality.
    std::vector<float> refreshHz;
    std::vector<float> uploadMs;  // one entry per display-texture upload, not per frame
    bool complete = false;

    void restart(std::uint64_t requestGeneration) {
        generation = requestGeneration;
        if (stage > 0) {
            return;  // past the warm-up every restart is part of the measured schedule and its columns are kept
        }
        passes.clear();
        frames.clear();
        frameMs.clear();
        presentGpuMs.clear();
        refreshHz.clear();
        uploadMs.clear();
    }
};

// The one evaluated BeautyFilter AOV, held across frames. One entry, not a map: exactly one AOV is on screen at a time.
struct FilterCache {
    pathtracer::debug::AovId aov = pathtracer::debug::AovId::Count;
    std::shared_ptr<const void> owner;  // identity and lifetime of the published pass this was filtered from
    std::uint64_t generation = 0;
    int samples = 0;
    // Bumped per re-evaluation, so the display texture's cache key changes even though `image`'s address never does.
    std::uint64_t revision = 0;
    pathtracer::gfx::HdrImage image;
};

struct AppResources {
    pathtracer::gfx::OcioDisplayTransform ocioTransform;
    pathtracer::scene::EmbreeAccel sceneAccel;     // path tracer scene intersection
    // stumpModel.shadingTriangles indexes sceneAccel's triangles 1:1, no separate field needed.
    pathtracer::scene::EnvironmentMap environmentMap;
    pathtracer::scene::LoadedModel stumpModel;
    // Parallel to stumpModel.instances: -1 for ordinary geometry, else the index into quadLights this instance's triangles emit as.
    std::vector<int> instanceLightIndex;
    std::vector<pathtracer::scene::QuadLight> quadLights;
    int totalTriangles;
    int totalPoints;

    pathtracer::gfx::PostProcessPass postProcess;
    pathtracer::debug::HudOverlay hud;
    pathtracer::debug::FrameStats frameStats;
    pathtracer::debug::GpuTimer postTimer;
    pathtracer::debug::Histogram histogram;
    // Zeroed at the top of every frame -- see FrameStageTimes.
    pathtracer::debug::FrameStageTimes stages;
    pathtracer::debug::PerfDashboard dashboard;
    // Companion to histogram: that bins the post-transform, post-clamp framebuffer and cannot tell 1.01 from 100.0, both saturating 255.
    float overRangeFraction = 0.0F;
    float overRangePeakMultiple = 0.0F;
    pathtracer::scene::DebugCameraController debugCamera;
    pathtracer::debug::GpuInfo gpuInfo;
    FilterCache filterCache;

    // HUD-editable UI/run state.
    int aov;
    // Film-back preset catalogue, loaded once at startup; filmBackPresetNames is index-parallel .c_str() pointers built for ImGui::Combo.
    std::vector<pathtracer::scene::Camera::FilmBackPreset> filmBackPresets;
    std::vector<const char*> filmBackPresetNames;
    int filmBackPresetIndex;
    int channelView;
    pathtracer::gfx::OcioDisplayTransform::Lut userLut;
    pathtracer::debug::FramingOverlayState framingState;
    bool showSky;
    // Whether the environment is in LightSet at all, unlike showSky. Off is what makes the classic Goral 1984 Cornell reachable.
    bool envLightEnabled;
    int envRotationDegrees;
    float envExposureStops;  // stops, not a multiplier; requestPathTrace does exp2()
    bool invert;   // 1.0 - colour, applied to the final display-referred image -- the 'I' debug toggle
    bool statsEnabled;  // -stats: the live terminal dashboard. The instrumentation behind it always runs; this only gates the drawing.
    // 'H' toggle; gates HudOverlay::draw only -- beginFrame/render stay unconditional so ImGui's frame pairing holds.
    bool showHud;
    bool vsync;  // profile.json: pace each frame to the vblank, or run uncapped
    // Chromatic aberration strength (0 = off), radial UV offset passed to OcioDisplayTransform::setAberration -- HUD slider only.
    float aberrationStrength;
    // The rasterizer runs only on a trigger change, so its cost is a last-actual plus duty cycle rather than a per-frame average.
    float lastRasterMs;

    // Async path-traced view, selected by `aov`; requestTrace() is called only from requestPathTraceIfTriggerChanged.
    pathtracer::scene::PathTraceSettings pathTraceSettings;
    // Per-instance material fields, parallel to stumpModel.instances, overridden by name. Renderer-only fields stay scene-wide.
    std::vector<pathtracer::scene::PathTraceSettings> perInstanceSettings;
    // World-space AABB per instance, parallel to stumpModel.instances. Static geometry, so computed once and read for the Wireframe AOV.
    std::vector<pathtracer::scene::AabbBounds> instanceBounds;
    int maxSamples;  // accumulated-pass cap for PathTraceDriver; 0 = unbounded
    pathtracer::gfx::ScalarType displayFormat;  // pathTraceDisplayTexture's component type (profile.json displayBitDepth)
    pathtracer::gfx::ScalarType textureType;    // scene textures' storage (profile.json textureBitDepth), recorded in -bench configs
    std::unique_ptr<pathtracer::scene::PathTraceDriver> pathTraceDriver;
    std::optional<pathtracer::gfx::Texture> pathTraceDisplayTexture;
    // Which image the texture holds, not the AovId that selected it, so AOVs sharing a buffer share one upload.
    const pathtracer::gfx::HdrImage* pathTraceDisplayedImage;  // nullptr = nothing uploaded yet
    // Max raw Depth in the last rebuilt pathTraceDisplayTexture; only meaningful when aov==Depth.
    float pathTraceDisplayedDepthMax;
    // Which RasterGBuffer generation the texture holds; 0 when it was built from a PathTraceResult instead.
    std::uint64_t pathTraceDisplayedGeneration;
    // Strong ref, not just an identity pointer, to whichever published object pathTraceDisplayTexture currently reflects.
    std::shared_ptr<const void> pathTraceDisplayedOwner;
    PathTraceTriggerState lastPathTraceTrigger;  // sentinel-initialized, see its own doc comment
    RasterTriggerState lastRasterTrigger;        // the same, for the rasterizer's independent refresh
    // The authored image in pixels (profile.json render.width/height), fixed for the session and independent of the window.
    int imageWidth;
    int imageHeight;
    // Fraction of imageWidth x imageHeight traced: renderScale settled, interactiveRenderScale while input changes.
    float renderScale;
    float interactiveRenderScale;
    std::chrono::steady_clock::time_point lastInputChange;

    // Synchronous CPU rasterizer for the 14 primary-hit AOVs, their only producer. unique_ptr: ThreadPool owns threads and cannot move.
    std::unique_ptr<pathtracer::scene::ThreadPool> rasterThreadPool;
    // Allocated once and rendered into in place, never republished: its `generation`, not its address, tells one render from the next.
    std::shared_ptr<pathtracer::scene::RasterGBuffer> rasterGBuffer;
    // Scene file's basename, for the dashboard's one-line SCENE row -- the full path is in the spec block, and the row has no space for it.
    std::string sceneName;

    // Orbit-pick and RAM-sampling state carried frame to frame.
    bool orbitPickRequested;
    double lastCursorX;
    double lastCursorY;
    std::size_t ramBytes;
    std::size_t bvhBytes;  // Embree's own device accounting, fixed after startup -- see embreeAllocatedBytes
    std::size_t systemAvailableBytes;
    std::uint64_t systemTotalBytes;
    std::chrono::steady_clock::time_point lastRamSample;
    std::chrono::steady_clock::time_point lastFrameTime;
    std::optional<BenchCapture> bench;  // engaged by -bench
    GLsync frameFence;  // the last presented frame's GPU completion, waited on before the next frame begins
    double refreshHz;   // 1 / DisplayLink::refreshPeriodSeconds, refreshed each frame so it follows the window across displays
};

// All one-time startup work: camera, model, OCIO and environment loading, and the Embree build.
std::optional<AppResources> initializeApp(const pathtracer::config::SceneConfig& sceneConfig,
                                           const pathtracer::config::ProfileConfig& profileConfig,
                                           const pathtracer::platform::Window& window,
                                           const std::string& scenePath, bool statsEnabled,
                                           double refreshHz) {
    const pathtracer::debug::GpuInfo gpuInfo = pathtracer::debug::queryGpuInfo();

    // Loaded separately from profile.json, then resolved by name against defaultFilmBackPresetName, as loadMaterialConfig already does.
    std::optional<std::vector<pathtracer::scene::Camera::FilmBackPreset>> filmBackPresets =
        pathtracer::config::loadFilmBackPresets(ASSET_ROOT_DIR "/config/sensor.json");
    if (!filmBackPresets) {
        std::cerr << "main: film back preset load failed, aborting startup\n";
        return std::nullopt;
    }
    const auto filmBackPresetIt =
        std::find_if(filmBackPresets->begin(), filmBackPresets->end(),
                     [&](const pathtracer::scene::Camera::FilmBackPreset& preset) {
                         return preset.name == profileConfig.camera.defaultFilmBackPresetName;
                     });
    if (filmBackPresetIt == filmBackPresets->end()) {
        std::cerr << "main: profile.json filmBackPreset \"" << profileConfig.camera.defaultFilmBackPresetName
                   << "\" not found in sensor.json\n";
        return std::nullopt;
    }
    const int filmBackPresetIndex =
        static_cast<int>(std::distance(filmBackPresets->begin(), filmBackPresetIt));
    // Computed before filmBackPresets is moved: the move keeps element addresses valid, but a moved-from vector would give none.
    std::vector<const char*> filmBackPresetNames;
    filmBackPresetNames.reserve(filmBackPresets->size());
    for (const pathtracer::scene::Camera::FilmBackPreset& preset : *filmBackPresets) {
        filmBackPresetNames.push_back(preset.name.c_str());
    }

    // ev100() is logged, not consumed: relativeExposureEv() derives the display-stage multiplier, not this log line.
    pathtracer::scene::DebugCameraController debugCamera(
        profileConfig.camera.position, profileConfig.camera.yawDegrees,
        profileConfig.camera.pitchDegrees, filmBackPresetIt->filmBack,
        profileConfig.camera.focalLengthMm, profileConfig.camera.nearClip,
        profileConfig.camera.farClip, profileConfig.camera.aperture,
        profileConfig.camera.shutterSeconds, profileConfig.camera.iso,
        profileConfig.controls.flySpeedMetersPerSecond,
        profileConfig.controls.orbitSensitivityDegPerPixel);
    // Scene-level placement (scene.json model.position/model.rotation), order X,Y,Z.
    const glm::mat4 sceneTransform =
        glm::translate(glm::mat4(1.0F), sceneConfig.model.position) *
        glm::rotate(glm::mat4(1.0F), glm::radians(sceneConfig.model.rotation.z), glm::vec3(0.0F, 0.0F, 1.0F)) *
        glm::rotate(glm::mat4(1.0F), glm::radians(sceneConfig.model.rotation.y), glm::vec3(0.0F, 1.0F, 0.0F)) *
        glm::rotate(glm::mat4(1.0F), glm::radians(sceneConfig.model.rotation.x), glm::vec3(1.0F, 0.0F, 0.0F));

    const auto loadStart = std::chrono::steady_clock::now();
    std::optional<pathtracer::scene::LoadedModel> stumpModel = pathtracer::scene::loadGltf(
        std::string(ASSET_ROOT_DIR) + "/" + sceneConfig.model.gltfPath, profileConfig.render.textureType, sceneTransform,
        std::string(ASSET_ROOT_DIR) + "/" + sceneConfig.model.texturePath);
    const double loadMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStart)
            .count();
    int totalTriangles = 0;
    std::vector<int> instanceLightIndex;
    std::vector<pathtracer::scene::QuadLight> quadLights;
    if (stumpModel) {
        instanceLightIndex.assign(stumpModel->instances.size(), -1);
        quadLights = pathtracer::scene::buildQuadLights(sceneConfig.lights, sceneTransform);
        pathtracer::scene::appendQuadLights(*stumpModel, quadLights, instanceLightIndex);

        totalTriangles = static_cast<int>(stumpModel->worldTriangles.size());
    }
    // "Points": total vertex-index count, i.e. 3 per triangle -- derived rather than tracked separately.
    const int totalPoints = totalTriangles * 3;

    std::optional<pathtracer::gfx::OcioDisplayTransform> ocioTransform =
        pathtracer::gfx::OcioDisplayTransform::create();
    // Decoded once here, not through a texture-upload helper: the path tracer samples this CPU ImageTexture directly, with no GPU upload.
    std::optional<pathtracer::gfx::ImageTexture> environmentImage = pathtracer::gfx::loadImageTexture(
        std::string(ASSET_ROOT_DIR) + "/" + sceneConfig.environment.hdriPath, profileConfig.render.textureType);
    std::optional<pathtracer::config::MaterialConfig> materialConfig = pathtracer::config::loadMaterialConfig(
        std::string(ASSET_ROOT_DIR) + "/" + sceneConfig.materialPath);

    if (!ocioTransform || !stumpModel || !environmentImage || !materialConfig) {
        std::cerr << "main: OCIO setup, model load, environment map load, or material "
                     "load failed, aborting startup\n";
        return std::nullopt;
    }

    pathtracer::gfx::PostProcessPass postProcess;
    pathtracer::debug::HudOverlay hud(window.nativeHandle());
    pathtracer::debug::FrameStats frameStats;
    pathtracer::debug::GpuTimer postTimer;
    pathtracer::debug::Histogram histogram;

    pathtracer::scene::EnvironmentMap environmentMap(std::move(*environmentImage));

    const auto accelBuildStart = std::chrono::steady_clock::now();
    std::optional<pathtracer::scene::EmbreeAccel> sceneAccel =
        pathtracer::scene::EmbreeAccel::build(std::move(stumpModel->worldTriangles));
    if (!sceneAccel) {
        std::cerr << "main: Embree scene build failed, aborting startup\n";
        return std::nullopt;
    }
    const double accelBuildMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                    accelBuildStart)
            .count();

    const pathtracer::scene::PathTraceSettings basePathTraceSettings{
        .samplesPerPixel = profileConfig.pathTracer.samplesPerPixel,
        .maxBounces = profileConfig.pathTracer.maxBounces,
        .russianRouletteStartBounce = profileConfig.pathTracer.russianRouletteStartBounce,
        .aoMaxDistance = profileConfig.pathTracer.aoMaxDistance,
        .lookaheadDistance = profileConfig.pathTracer.lookaheadDistance,
        .bumpStrength = materialConfig->bumpStrength,
        .roughnessMin = materialConfig->roughnessMin,
        .roughnessMax = materialConfig->roughnessMax,
        .diffuseColour = materialConfig->diffuseColour,
        .ior = materialConfig->ior,
        .abbe = materialConfig->abbe,
        .transmissionFactor = materialConfig->transmissionFactor,
        .metallicFactor = materialConfig->metallicFactor,
        .roughnessFactor = materialConfig->roughnessFactor,
        .diffuseRoughness = materialConfig->diffuseRoughness,
        .transmissionColor = materialConfig->transmissionColor,
        .transmissionDepth = materialConfig->transmissionDepth,
        .edgeTint = materialConfig->edgeTint,
    };

    std::optional<std::vector<pathtracer::scene::PathTraceSettings>> perInstanceSettings =
        pathtracer::scene::resolvePerInstanceSettings(basePathTraceSettings, stumpModel->instances,
                                                   sceneConfig.materialOverrides, ASSET_ROOT_DIR);
    if (!perInstanceSettings) {
        std::cerr << "main: material override resolution failed, aborting startup\n";
        return std::nullopt;
    }

    // After appendQuadLights, so the light panels' own instances are bounded too.
    std::vector<pathtracer::scene::AabbBounds> instanceBounds = pathtracer::scene::computeInstanceBounds(
        stumpModel->shadingTriangles, static_cast<int>(stumpModel->instances.size()));

    // Printed at the end of startup, not where each value becomes known: every number is real by now, and one block survives being piped.
    const pathtracer::debug::EngineSpec spec{
        scenePath.c_str(),
        sceneConfig.environment.hdriPath.c_str(),
        pathtracer::debug::kAovNames[profileConfig.render.defaultAov],
        profileConfig.render.width,
        profileConfig.render.height,
        profileConfig.render.renderScale,
        profileConfig.render.interactiveRenderScale,
        basePathTraceSettings.samplesPerPixel,
        basePathTraceSettings.maxBounces,
        basePathTraceSettings.russianRouletteStartBounce,
        profileConfig.pathTracer.maxSamples,
        basePathTraceSettings.aoMaxDistance,
        // Both pools take ThreadPool's default; PathTraceDriver owns its own, unconstructed until AppResources is at its final address.
        pathtracer::scene::ThreadPool::defaultThreadCount(),
        pathtracer::scene::ThreadPool::defaultThreadCount(),
        pathtracer::scene::kPathTraceTileSize,
        static_cast<int>(stumpModel->instances.size()),
        static_cast<int>(quadLights.size()),
        totalTriangles,
        loadMs,
        accelBuildMs,
        pathtracer::scene::embreeAllocatedBytes(),
        pathtracer::gfx::khrDebugAvailable(),
        pathtracer::debug::gpuTimerQueryAvailable(),
        refreshHz,
    };
    pathtracer::debug::printSpec(spec, gpuInfo);

    return AppResources{
        .ocioTransform = std::move(*ocioTransform),
        .sceneAccel = std::move(*sceneAccel),
        .environmentMap = std::move(environmentMap),
        .stumpModel = std::move(*stumpModel),
        .instanceLightIndex = std::move(instanceLightIndex),
        .quadLights = std::move(quadLights),
        .totalTriangles = totalTriangles,
        .totalPoints = totalPoints,
        .postProcess = std::move(postProcess),
        .hud = std::move(hud),
        .frameStats = std::move(frameStats),
        .postTimer = std::move(postTimer),
        .histogram = std::move(histogram),
        .stages = {},
        .dashboard = {},
        .debugCamera = std::move(debugCamera),
        .gpuInfo = gpuInfo,
        .filterCache = {},
        // aov selects which lane the snapshot supplies; userLut stays separate because non-Beauty AOVs force Raw and must not overwrite it.
        .aov = profileConfig.render.defaultAov,
        .filmBackPresets = std::move(*filmBackPresets),
        .filmBackPresetNames = std::move(filmBackPresetNames),
        .filmBackPresetIndex = filmBackPresetIndex,
        .channelView = 0,
        .userLut = profileConfig.render.defaultLut,
        .framingState = pathtracer::debug::FramingOverlayState{},
        // "Show/Hide Background" HDRI-section checkbox -- off by default; only takes visible effect for the Beauty AOV, see presentFrame.
        .showSky = false,
        // The scene's own authored default, not a separate runtime one -- the HUD checkbox edits this in place.
        .envLightEnabled = sceneConfig.environment.lightEnabled,
        // HDR environment Y rotation in degrees, affecting background and lighting alike, rotated at query time rather than re-baked.
        .envRotationDegrees = 0,
        // HDRI Exposure slider, stops. requestPathTrace does exp2() -> path_tracer.cpp miss-ray sampleDirection.
        .envExposureStops = 0.0F,
        .invert = false,
        .statsEnabled = statsEnabled,
        .showHud = true,
        .vsync = profileConfig.render.vsync,
        .aberrationStrength = 0.0F,
        .lastRasterMs = 0.0F,
        .pathTraceSettings = basePathTraceSettings,
        .perInstanceSettings = std::move(*perInstanceSettings),
        .instanceBounds = std::move(instanceBounds),
        .maxSamples = profileConfig.pathTracer.maxSamples,
        .displayFormat = profileConfig.render.displayFormat,
        .textureType = profileConfig.render.textureType,
        // Constructed in main() right after initializeApp(): its reference members must bind to objects at their final address.
        .pathTraceDriver = nullptr,
        .pathTraceDisplayTexture = std::nullopt,
        .pathTraceDisplayedImage = nullptr,
        .pathTraceDisplayedDepthMax = 0.0F,
        .pathTraceDisplayedGeneration = 0,
        .pathTraceDisplayedOwner = nullptr,
        .lastPathTraceTrigger = PathTraceTriggerState{},
        .lastRasterTrigger = RasterTriggerState{},
        .imageWidth = profileConfig.render.width,
        .imageHeight = profileConfig.render.height,
        .renderScale = profileConfig.render.renderScale,
        .interactiveRenderScale = profileConfig.render.interactiveRenderScale,
        .lastInputChange = std::chrono::steady_clock::time_point{},
        .rasterThreadPool = std::make_unique<pathtracer::scene::ThreadPool>(),
        .rasterGBuffer = std::make_shared<pathtracer::scene::RasterGBuffer>(),
        .sceneName = std::filesystem::path(scenePath).filename().string(),
        .orbitPickRequested = false,
        .lastCursorX = 0.0,
        .lastCursorY = 0.0,
        // task_info() is a real syscall and the HUD is read by eyes, so sampling 4x/sec rather than per frame drops jitter for free.
        .ramBytes = pathtracer::debug::residentSetBytes(),
        // Embree has finished building by now, so its device counter is final.
        .bvhBytes = pathtracer::scene::embreeAllocatedBytes(),
        .systemAvailableBytes = pathtracer::debug::availableSystemBytes(),
        // Fixed for the machine, unlike the other two -- queried once here rather than resampled alongside them.
        .systemTotalBytes = pathtracer::debug::totalSystemBytes(),
        .lastRamSample = std::chrono::steady_clock::now(),
        .lastFrameTime = std::chrono::steady_clock::now(),
        .bench = std::nullopt,
        .frameFence = nullptr,
        .refreshHz = refreshHz,
    };
}

// Debug-only: 'L' cycles the viewer LUT (sRGB -> Rec709 -> Raw); 'R'/'G'/'B' isolate a channel, pressing the active one again clearing it.
void wireCallbacks(pathtracer::platform::Window& window, AppResources& app) {
    window.setKeyCallback([&app, &window](int key, int action) {
        if (action != GLFW_PRESS) {
            return;
        }
        using Lut = pathtracer::gfx::OcioDisplayTransform::Lut;
        if (key == GLFW_KEY_L) {
            app.userLut = app.userLut == Lut::SRGB     ? Lut::Rec709
                          : app.userLut == Lut::Rec709 ? Lut::Raw
                                                        : Lut::SRGB;
        } else if (key == GLFW_KEY_R) {
            app.channelView = app.channelView == 1 ? 0 : 1;
        } else if (key == GLFW_KEY_G) {
            app.channelView = app.channelView == 2 ? 0 : 2;
        } else if (key == GLFW_KEY_B) {
            app.channelView = app.channelView == 3 ? 0 : 3;
        } else if (key == GLFW_KEY_0) {
            app.debugCamera.resetToDefault();
        } else if (key == GLFW_KEY_I) {
            app.invert = !app.invert;
        } else if (key == GLFW_KEY_H) {
            app.showHud = !app.showHud;
        } else if (key == GLFW_KEY_ESCAPE) {
            window.setShouldClose(true);
        }
    });

    // LMB begins and ends an orbit, gated on the HUD not wanting the click; the release always ends one wherever the cursor is.
    window.setMouseButtonCallback([&app, &window](int button, int action) {
        if (button != GLFW_MOUSE_BUTTON_LEFT) {
            return;
        }
        if (action == GLFW_PRESS) {
            if (!app.hud.wantsCaptureMouse()) {
                app.orbitPickRequested = true;
            }
        } else if (action == GLFW_RELEASE && app.debugCamera.isOrbiting()) {
            app.debugCamera.endOrbit();
            window.setCursorLocked(false);
        }
    });
}

pathtracer::scene::Camera updateCamera(const pathtracer::platform::Window& window, AppResources& app,
                                    float dtSeconds) {
    if (app.debugCamera.isOrbiting()) {
        const auto [cursorX, cursorY] = window.cursorPosition();
        app.debugCamera.applyOrbitDelta(static_cast<float>(cursorX - app.lastCursorX),
                                         static_cast<float>(cursorY - app.lastCursorY));
        app.lastCursorX = cursorX;
        app.lastCursorY = cursorY;
    } else {
        app.debugCamera.applyFlyInput(window, dtSeconds);
    }
    return app.debugCamera.snapshot();
}

// Direct texel read, no bilinear -- gbuffer AOVs are per-pixel snapshots.
glm::vec3 sampleTexel(const pathtracer::gfx::HdrImage& image, int x, int y) {
    const std::size_t idx = ((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) +
                              static_cast<std::size_t>(x)) *
                             4;
    return {image.rgba[idx + 0], image.rgba[idx + 1], image.rgba[idx + 2]};
}

// Cached CPU OCIO processors for the Beauty probe, built once: getProcessor parses the builtin config, as the GPU shaders do at startup.
const OCIO::ConstCPUProcessorRcPtr& ocioCpuProcessor(pathtracer::gfx::OcioDisplayTransform::Lut lut) {
    static const OCIO::ConstConfigRcPtr config =
        OCIO::Config::CreateFromBuiltinConfig(pathtracer::gfx::kOcioConfigName);
    static const OCIO::ConstCPUProcessorRcPtr srgbProcessor =
        config
            ->getProcessor(pathtracer::gfx::kOcioSceneColorSpace, pathtracer::gfx::kOcioSrgbDisplay,
                            pathtracer::gfx::kOcioView, OCIO::TRANSFORM_DIR_FORWARD)
            ->getDefaultCPUProcessor();
    static const OCIO::ConstCPUProcessorRcPtr rec709Processor =
        config
            ->getProcessor(pathtracer::gfx::kOcioSceneColorSpace, pathtracer::gfx::kOcioRec709Display,
                            pathtracer::gfx::kOcioView, OCIO::TRANSFORM_DIR_FORWARD)
            ->getDefaultCPUProcessor();
    return lut == pathtracer::gfx::OcioDisplayTransform::Lut::Rec709 ? rec709Processor : srgbProcessor;
}

// The Beauty probe's display transform on one texel: channel isolation, exposure, the OCIO curve, then invert -- the shaders' own order.
glm::vec3 applyBeautyDisplayTransform(glm::vec3 hdrColor, const AppResources& app) {
    if (app.channelView == 1) {
        hdrColor = glm::vec3(hdrColor.r);
    } else if (app.channelView == 2) {
        hdrColor = glm::vec3(hdrColor.g);
    } else if (app.channelView == 3) {
        hdrColor = glm::vec3(hdrColor.b);
    }
    glm::vec3 displayColor = hdrColor * std::pow(2.0F, app.debugCamera.relativeExposureEv());
    if (app.userLut != pathtracer::gfx::OcioDisplayTransform::Lut::Raw) {
        std::array<float, 3> pixel{displayColor.r, displayColor.g, displayColor.b};
        ocioCpuProcessor(app.userLut)->applyRGB(pixel.data());
        displayColor = {pixel[0], pixel[1], pixel[2]};
    }
    if (app.invert) {
        displayColor = glm::vec3(1.0F) - displayColor;
    }
    return displayColor;
}

// Orbit pivot from a single Embree ray down the view centre. Reading the G-buffer centre texel instead forced a full rasterization.
void resolveOrbitPick(pathtracer::platform::Window& window, AppResources& app,
                       const pathtracer::scene::Camera& camera) {
    if (!app.orbitPickRequested) {
        return;
    }
    app.orbitPickRequested = false;

    // primaryRay at ndc (0,0) reduces to the camera's forward axis, both terms vanishing, so it needs no aspect and no projection.
    const pathtracer::scene::Ray ray{camera.position(), camera.forward(), camera.nearClip(),
                                  camera.farClip()};
    const std::optional<pathtracer::scene::Hit> hit = app.sceneAccel.intersect(ray);
    const glm::vec3 pivot =
        hit ? ray.origin + (hit->t * ray.dir) : camera.position() + (3.0F * camera.forward());

    app.debugCamera.beginOrbit(pivot);
    window.setCursorLocked(true);
    const auto [cursorX, cursorY] = window.cursorPosition();
    app.lastCursorX = cursorX;
    app.lastCursorY = cursorY;
}

// Bundles the HdrImage an AOV displays with a type-erased strong ref to its owner, which is also an ABA-safe cache key.
struct DisplayedAovSource {
    const pathtracer::gfx::HdrImage* image = nullptr;
    std::shared_ptr<const void> owner;
    // A counter that changes whenever the image's contents do: the rasterizer's, the filter cache's, or 0 for a published lane.
    std::uint64_t generation = 0;
};

// Re-runs the selected image-space filter only when the pass it reads, or the AOV itself, changed. Filters cost tens of milliseconds.
const pathtracer::gfx::HdrImage* ensureFilterImage(
    AppResources& app, const std::shared_ptr<const pathtracer::scene::PathTraceResult>& snapshot,
    pathtracer::debug::AovId aov, const pathtracer::scene::Camera& camera) {
    if (!snapshot) {
        return nullptr;
    }
    FilterCache& cache = app.filterCache;
    if (cache.revision != 0 && cache.aov == aov && cache.owner == snapshot &&
        cache.generation == snapshot->generation && cache.samples == snapshot->samples) {
        return &cache.image;
    }
    const pathtracer::debug::ScopedCpuTimer filterTimer(app.stages.filterMs);
    cache.image = pathtracer::debug::evaluateFilterAov(
        aov, pathtracer::debug::FilterInput{snapshot->beauty, camera.verticalFovRadians()}, *app.rasterThreadPool);
    cache.aov = aov;
    cache.owner = snapshot;
    cache.generation = snapshot->generation;
    cache.samples = snapshot->samples;
    ++cache.revision;
    return &cache.image;
}

// Null image if the AOV's producer has not published, so callers show black. The one place an AovId becomes an HdrImage.
DisplayedAovSource resolveAovImage(AppResources& app,
                                   const std::shared_ptr<const pathtracer::scene::PathTraceResult>& snapshot,
                                   pathtracer::debug::AovId aov, const pathtracer::scene::Camera& camera) {
    if (const pathtracer::debug::GBufferLane lane = pathtracer::debug::gbufferLane(aov)) {
        // The buffer is allocated for the process's life now, so a null check no longer distinguishes "no render yet" -- generation 0 does.
        return app.rasterGBuffer->generation == 0
                   ? DisplayedAovSource{}
                   : DisplayedAovSource{&(*app.rasterGBuffer.*lane), app.rasterGBuffer, app.rasterGBuffer->generation};
    }
    if (const pathtracer::debug::PathTracedLane lane = pathtracer::debug::pathTracedLane(aov)) {
        return snapshot ? DisplayedAovSource{&(*snapshot.*lane), snapshot} : DisplayedAovSource{};
    }
    const pathtracer::gfx::HdrImage* filtered = ensureFilterImage(app, snapshot, aov, camera);
    return filtered != nullptr ? DisplayedAovSource{filtered, snapshot, app.filterCache.revision}
                               : DisplayedAovSource{};
}

// Cursor offset within imageRect, in framebuffer pixels with GL's bottom-left origin. Nullopt off-window or over a letterbox bar.
std::optional<std::pair<int, int>> cursorInImageRect(const pathtracer::platform::Window& window,
                                                      pathtracer::gfx::ViewportRect imageRect) {
    const auto [windowWidth, windowHeight] = window.windowSize();
    const auto [fbWidth, fbHeight] = window.framebufferSize();
    if (windowWidth <= 0 || windowHeight <= 0 || fbWidth <= 0 || fbHeight <= 0) {
        return std::nullopt;
    }
    const auto [cursorX, cursorY] = window.cursorPosition();
    if (cursorX < 0.0 || cursorY < 0.0 || cursorX >= windowWidth || cursorY >= windowHeight) {
        return std::nullopt;
    }
    // Points to pixels by the framebuffer ratio, which is the display's content scale, then flipped into GL's bottom-up rows.
    const int x = static_cast<int>(cursorX / windowWidth * fbWidth) - imageRect.x;
    const int y = (fbHeight - 1 - static_cast<int>(cursorY / windowHeight * fbHeight)) - imageRect.y;
    if (x < 0 || y < 0 || x >= imageRect.width || y >= imageRect.height) {
        return std::nullopt;
    }
    return std::pair{x, y};
}

// Bottom-right HUD probe: every AOV now reads its own raw HdrImage texel, so nothing here stalls on the GPU.
pathtracer::debug::PixelProbeSample samplePixelProbe(
    const pathtracer::platform::Window& window,
    const std::shared_ptr<const pathtracer::scene::PathTraceResult>& pathTraceSnapshot, AppResources& app,
    const pathtracer::scene::Camera& camera, pathtracer::debug::AovId aovId,
    pathtracer::gfx::ViewportRect imageRect, float& probeMs) {
    // Timed here so updateHud stays one screen. A filter AOV hits the cache presentFrame filled, so this stays a single texel fetch.
    const pathtracer::debug::ScopedCpuTimer probeTimer(probeMs);
    const std::optional<std::pair<int, int>> cursor = cursorInImageRect(window, imageRect);
    if (!cursor.has_value()) {
        return {};
    }
    const auto [rectX, rectY] = *cursor;
    const DisplayedAovSource source = resolveAovImage(app, pathTraceSnapshot, aovId, camera);
    if (source.image == nullptr) {
        return {};
    }
    // Normalised by the rect, not the window: the image occupies only the rect, and its rows run top-down.
    const double u = rectX / static_cast<double>(imageRect.width);
    const double v = (imageRect.height - 1 - rectY) / static_cast<double>(imageRect.height);
    const int imgX = std::min(source.image->width - 1, static_cast<int>(u * source.image->width));
    const int imgY = std::min(source.image->height - 1, static_cast<int>(v * source.image->height));
    const glm::vec3 texel = sampleTexel(*source.image, imgX, imgY);
    const glm::vec3 color =
        aovId == pathtracer::debug::AovId::Beauty ? applyBeautyDisplayTransform(texel, app) : texel;
    return {true, glm::vec4(color, 1.0F)};
}

// Re-uploads only when the owning published object changed: 33MB a frame for texels the GPU holds is work without a reason.
void ensurePathTraceDisplayTexture(AppResources& app, const std::shared_ptr<const void>& owner,
                                    const pathtracer::gfx::HdrImage& image, std::uint64_t generation) {
    if (app.pathTraceDisplayTexture.has_value() && app.pathTraceDisplayedImage == &image &&
        app.pathTraceDisplayedOwner == owner && app.pathTraceDisplayedGeneration == generation) {
        return;
    }
    // Started after the cache-key check, never before: on a hit this does nothing and must report 0, not the last real upload's cost.
    const pathtracer::debug::ScopedCpuTimer uploadTimer(app.stages.uploadMs);
    app.stages.uploaded = true;
    if (app.aov == static_cast<int>(pathtracer::debug::AovId::Depth)) {
        float maxDepth = 0.0F;
        for (int i = 0; i < image.width * image.height; ++i) {
            maxDepth = std::max(maxDepth, image.rgba[static_cast<std::size_t>(i) * 4]);
        }
        app.pathTraceDisplayedDepthMax = maxDepth;
    }
    // BounceCount is a scalar mapped through Turbo on the CPU before upload; it runs once per rebuilt pass, so it costs nothing per frame.
    if (app.aov == static_cast<int>(pathtracer::debug::AovId::BounceCount)) {
        const float maxBounceCount = static_cast<float>(app.pathTraceSettings.maxBounces) + 1.0F;
        pathtracer::gfx::HdrImage mapped;
        mapped.width = image.width;
        mapped.height = image.height;
        mapped.rgba.resize(image.rgba.size());
        for (int i = 0; i < image.width * image.height; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i) * 4;
            const float t = image.rgba[idx] / maxBounceCount;
            const glm::vec3 mappedColor = pathtracer::debug::turbo(t);
            mapped.rgba[idx + 0] = mappedColor.r;
            mapped.rgba[idx + 1] = mappedColor.g;
            mapped.rgba[idx + 2] = mappedColor.b;
            mapped.rgba[idx + 3] = image.rgba[idx + 3];
        }
        if (app.pathTraceDisplayTexture.has_value()) {
            app.pathTraceDisplayTexture->upload(mapped.width, mapped.height, mapped.rgba.data());
        } else {
            app.pathTraceDisplayTexture = pathtracer::gfx::Texture::createFromFloatPixels(
                mapped.width, mapped.height, mapped.rgba.data(), app.displayFormat);
        }
        app.pathTraceDisplayedImage = &image;
        app.pathTraceDisplayedOwner = owner;
        app.pathTraceDisplayedGeneration = generation;
        return;
    }
    // Uploaded straight from the HdrImage: the vertex shader resolves row order, and upload reallocates only on a resolution change.
    if (app.pathTraceDisplayTexture.has_value()) {
        app.pathTraceDisplayTexture->upload(image.width, image.height, image.rgba.data());
    } else {
        app.pathTraceDisplayTexture =
            pathtracer::gfx::Texture::createFromFloatPixels(image.width, image.height, image.rgba.data(),
                                                         app.displayFormat);
    }
    app.pathTraceDisplayedImage = &image;
    app.pathTraceDisplayedOwner = owner;
    app.pathTraceDisplayedGeneration = generation;
}

// Clears the whole viewport, so the letterbox bars are black and an AOV with no buffer leaves no stale contents on screen.
void clearToBlack(int viewportWidth, int viewportHeight) {
    GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_CALL(glViewport(0, 0, viewportWidth, viewportHeight));
    GL_CALL(glClearColor(0.0F, 0.0F, 0.0F, 1.0F));
    GL_CALL(glClear(GL_COLOR_BUFFER_BIT));
}

// Blits the selected AOV through the shared OCIO path. Beauty uses the user's LUT; everything else forces Raw, not being radiance.
void presentFrame(AppResources& app,
                   const std::shared_ptr<const pathtracer::scene::PathTraceResult>& pathTraceSnapshot,
                   const pathtracer::scene::Camera& camera, int viewportWidth, int viewportHeight,
                   pathtracer::gfx::ViewportRect imageRect) {
    const auto aovId = static_cast<pathtracer::debug::AovId>(app.aov);
    // Unconditional: the draw covers only imageRect, so the bars need clearing whether or not there is an image to draw into it.
    clearToBlack(viewportWidth, viewportHeight);

    const DisplayedAovSource source = resolveAovImage(app, pathTraceSnapshot, aovId, camera);
    if (source.image == nullptr) {
        return;
    }
    ensurePathTraceDisplayTexture(app, source.owner, *source.image, source.generation);
    const bool isBeauty = aovId == pathtracer::debug::AovId::Beauty;
    app.ocioTransform.setActiveLut(isBeauty ? app.userLut
                                             : pathtracer::gfx::OcioDisplayTransform::Lut::Raw);
    // Radiance-carrying AOVs take the photographic exposure; Depth auto-ranges to its own max, farClip being a tMax, not a scene extent.
    float exposureEv = 0.0F;
    if (pathtracer::debug::aovCarriesRadiance(aovId)) {
        exposureEv = app.debugCamera.relativeExposureEv();
    } else if (aovId == pathtracer::debug::AovId::Depth) {
        exposureEv = -std::log2(std::max(app.pathTraceDisplayedDepthMax, 1e-4F));
    }
    app.ocioTransform.setExposureEv(exposureEv);
    app.ocioTransform.setChannelView(app.channelView);
    app.ocioTransform.setInvert(app.invert);
    // Beauty only -- an artistic lens effect over the rendered image, not meaningful on a raw data AOV like Normal/Depth/Albedo.
    app.ocioTransform.setAberration(isBeauty ? app.aberrationStrength : 0.0F);
    app.ocioTransform.bind();
    app.postProcess.draw(app.pathTraceDisplayTexture->id(), app.ocioTransform.activeShader(), imageRect);
}

// Non-blocking: hands a fresh request to PathTraceDriver, which restarts accumulation at this pose and size on its own thread.
std::uint64_t requestPathTrace(AppResources& app, const pathtracer::scene::Camera& camera, int traceWidth,
                               int traceHeight) {
    return app.pathTraceDriver->requestTrace(pathtracer::scene::PathTraceDriver::Request{
        camera, traceWidth, traceHeight, glm::radians(static_cast<float>(app.envRotationDegrees)),
        app.showSky, app.envLightEnabled, std::exp2(app.envExposureStops), app.pathTraceSettings,
        app.maxSamples});
}

// Once per frame, re-tracing on any input that changes the image. The rasterizer is synchronous, 21 ms at 1024x576, so it stays gated.
void requestPathTraceIfTriggerChanged(AppResources& app, const pathtracer::scene::Camera& camera,
                                       std::chrono::steady_clock::time_point now) {
    const ViewInputState view{camera.position(),
                               app.debugCamera.yawDegrees(),
                               app.debugCamera.pitchDegrees(),
                               app.debugCamera.focalLengthMm(),
                               app.debugCamera.filmBack().heightMm};
    const PathTraceInputState input{view, app.envRotationDegrees, app.showSky, app.envLightEnabled,
                                     app.envExposureStops};

    // Interaction is a change in anything but resolution, compared against the inputs alone so the scale promotion cannot re-arm the timer.
    if (input != app.lastPathTraceTrigger.input) {
        app.lastInputChange = now;
    }
    const bool settled =
        std::chrono::duration<double>(now - app.lastInputChange).count() >= kInteractiveSettleSeconds;
    const float renderScale = settled ? app.renderScale : app.interactiveRenderScale;
    const int traceWidth = scaledExtent(app.imageWidth, renderScale);
    const int traceHeight = scaledExtent(app.imageHeight, renderScale);
    const bool needsLightTransport = aovNeedsLightTransport(static_cast<pathtracer::debug::AovId>(app.aov));

    // Park the driver when the selected AOV is not its own: otherwise it accumulates an off-screen image on every core.
    app.pathTraceDriver->setSuspended(!needsLightTransport);

    const PathTraceTriggerState pathTrace{input, renderScale};
    if (pathTrace != app.lastPathTraceTrigger) {
        const std::uint64_t generation = requestPathTrace(app, camera, traceWidth, traceHeight);
        if (app.bench) {
            app.bench->restart(generation);
        }
        app.lastPathTraceTrigger = pathTrace;
    }

    const RasterTriggerState raster{view, renderScale};
    if (needsLightTransport || raster == app.lastRasterTrigger || traceWidth <= 0 || traceHeight <= 0) {
        return;
    }
    {
        const pathtracer::debug::ScopedCpuTimer rasterTimer(app.stages.rasterMs);
        pathtracer::scene::renderRasterGBuffer(camera, app.stumpModel.shadingTriangles,
                                            app.stumpModel.instances, app.perInstanceSettings,
                                            app.instanceBounds,
                                            traceWidth, traceHeight, *app.rasterThreadPool,
                                            *app.rasterGBuffer);
    }
    app.lastRasterTrigger = raster;
}

// Fraction of texels clipping at the display encode plus that peak, from pre-transform beauty rather than the framebuffer Histogram reads.
void updateOverRangeStats(AppResources& app,
                           const std::shared_ptr<const pathtracer::scene::PathTraceResult>& pathTraceSnapshot) {
    const pathtracer::debug::ScopedCpuTimer overRangeTimer(app.stages.overRangeMs);
    if (!pathTraceSnapshot) {
        app.overRangeFraction = 0.0F;
        app.overRangePeakMultiple = 0.0F;
        return;
    }
    const pathtracer::scene::OverRangeStats& stats = pathTraceSnapshot->overRange;
    const float exposure = std::pow(2.0F, app.debugCamera.relativeExposureEv());
    app.overRangePeakMultiple = exposure * stats.rawPeak;
    // Texels binned strictly above the one holding 1/exposure: exact above that edge, the threshold quantised to 1/1024 of a stop.
    const auto bin = static_cast<std::size_t>(pathtracer::scene::overRangeBin(1.0F / exposure));
    const std::uint32_t texelCount = stats.aboveBin.front();
    app.overRangeFraction =
        texelCount > 0 ? static_cast<float>(stats.aboveBin[bin + 1]) / static_cast<float>(texelCount)
                        : 0.0F;
}

void updateHud(AppResources& app, const pathtracer::platform::Window& window,
               const pathtracer::scene::Camera& camera,
               const std::shared_ptr<const pathtracer::scene::PathTraceResult>& pathTraceSnapshot,
               pathtracer::gfx::ViewportRect imageRect) {
    const pathtracer::debug::PathTracedStatus pathTracedStatus{
        pathTraceSnapshot != nullptr,
        // PassRecord is the single source of truth for pass timing now; the HUD wants seconds, the record carries milliseconds.
        app.pathTraceDriver != nullptr ? app.pathTraceDriver->lastPassRecord().passMs / 1000.0 : 0.0,
        pathTraceSnapshot != nullptr ? pathTraceSnapshot->samples : 0, app.maxSamples};
    const pathtracer::debug::SceneStats sceneStats{
        static_cast<int>(app.stumpModel.instances.size()),
        app.totalTriangles,
        app.totalPoints,
        app.imageWidth,
        app.imageHeight,
    };
    const pathtracer::debug::HudFrameData hudFrameData{
        app.gpuInfo,
        app.refreshHz,
        app.frameStats,
        app.postTimer.millisecondsElapsed(),
        app.ramBytes,
        pathtracer::debug::gpuAllocatedBytes(),
        app.systemAvailableBytes,
        app.systemTotalBytes,
        app.channelView,
        lutName(app.ocioTransform.activeLut()),
        sceneStats,
        camera,
        app.debugCamera.yawDegrees(),
        app.debugCamera.pitchDegrees(),
        app.debugCamera.isOrbiting(),
        app.histogram,
        pathTracedStatus,
        app.overRangeFraction,
        app.overRangePeakMultiple,
        app.vsync,
    };
    // Round-tripped through locals so the sliders can bind plain float&s; DebugCameraController stays the authoritative owner.
    float focalLengthMm = app.debugCamera.focalLengthMm();
    float aperture = app.debugCamera.aperture();
    float shutterSeconds = app.debugCamera.shutterSeconds();
    float iso = app.debugCamera.iso();
    int filmBackPresetIndex = app.filmBackPresetIndex;
    // Only the HUD reads it, so with the HUD hidden the fetch is skipped outright rather than computed and thrown away.
    const pathtracer::debug::PixelProbeSample pixelProbe =
        app.showHud ? samplePixelProbe(window, pathTraceSnapshot, app, camera,
                                        static_cast<pathtracer::debug::AovId>(app.aov), imageRect, app.stages.probeMs)
                     : pathtracer::debug::PixelProbeSample{};
    const pathtracer::debug::ScopedCpuTimer hudTimer(app.stages.hudMs);
    if (app.showHud) {
        app.hud.draw(hudFrameData, app.aov, focalLengthMm, aperture, shutterSeconds, iso,
                     filmBackPresetIndex, app.filmBackPresetNames, app.showSky, app.envLightEnabled,
                     app.envRotationDegrees, app.envExposureStops, app.aberrationStrength,
                     app.framingState, pixelProbe);
    }
    app.debugCamera.setFocalLengthMm(focalLengthMm);
    app.debugCamera.setAperture(aperture);
    app.debugCamera.setShutterSeconds(shutterSeconds);
    app.debugCamera.setIso(iso);
    app.filmBackPresetIndex = filmBackPresetIndex;
    app.debugCamera.setFilmBack(
        app.filmBackPresets[static_cast<std::size_t>(filmBackPresetIndex)].filmBack);
    {
        const pathtracer::debug::ScopedCpuTimer hudRenderTimer(app.stages.hudRenderMs);
        app.hud.render();
    }
}

// The per-frame read-backs after the composited image lands and before the HUD draws: only the histogram reads the framebuffer.
void sampleDisplayedFrame(AppResources& app,
                           const std::shared_ptr<const pathtracer::scene::PathTraceResult>& pathTraceSnapshot,
                           pathtracer::gfx::ViewportRect imageRect) {
    {
        const pathtracer::debug::ScopedCpuTimer histogramTimer(app.stages.histogramMs);
        app.histogram.update(imageRect);
    }
    updateOverRangeStats(app, pathTraceSnapshot);

    // task_info() is a real syscall; nothing per-frame reads these, so 4Hz keeps one source of frame-time jitter out of the loop.
    const auto now = std::chrono::steady_clock::now();
    if (now - app.lastRamSample >= std::chrono::milliseconds(250)) {
        app.ramBytes = pathtracer::debug::residentSetBytes();
        app.systemAvailableBytes = pathtracer::debug::availableSystemBytes();
        app.lastRamSample = now;
    }
}

// Assembles one DashboardFrame and hands it over; the dashboard decides whether to redraw. Split so renderFrame stays readable.
void updateDashboard(AppResources& app,
                     const std::shared_ptr<const pathtracer::scene::PathTraceResult>& pathTraceSnapshot, float frameMs,
                     int viewportWidth, int viewportHeight) {
    const pathtracer::debug::PassRecord pass =
        app.pathTraceDriver != nullptr ? app.pathTraceDriver->lastPassRecord()
                                        : pathtracer::debug::PassRecord{};
    const bool interactive = app.lastPathTraceTrigger.renderScale == app.interactiveRenderScale;
    const pathtracer::debug::DashboardFrame frame{
        app.frameStats,
        app.stages,
        pass,
        frameMs,
        app.postTimer.millisecondsElapsed(),
        pathTraceSnapshot != nullptr ? pathTraceSnapshot->samples : 0,
        app.maxSamples,
        !aovNeedsLightTransport(static_cast<pathtracer::debug::AovId>(app.aov)),
        app.ramBytes,
        pathtracer::debug::gpuAllocatedBytes(),
        app.bvhBytes,
        app.systemAvailableBytes,
        app.systemTotalBytes,
        pathtracer::debug::kAovNames[app.aov],
        app.sceneName.c_str(),
        static_cast<int>(app.stumpModel.instances.size()),
        static_cast<int>(app.quadLights.size()),
        static_cast<int>(app.totalTriangles),
        app.imageWidth,
        app.imageHeight,
        pass.width,
        pass.height,
        viewportWidth,
        viewportHeight,
        app.lastPathTraceTrigger.renderScale,
        interactive,
        app.refreshHz,
        app.vsync,
    };
    app.dashboard.update(frame);
}

// True once the selected AOV's producer has nothing left to do: a rasterizer AOV is finished as soon as a G-buffer exists.
bool stageComplete(const AppResources& app, const BenchCapture& bench) {
    if (!aovNeedsLightTransport(static_cast<pathtracer::debug::AovId>(app.aov))) {
        return app.rasterGBuffer->generation != 0;
    }
    return !bench.passes.empty() && bench.passes.back().generation == bench.generation &&
           bench.passes.back().passIndex == app.maxSamples;
}

// Appends this frame's stage times and any newly finished pass. `pass` was read before this frame's snapshot, the driver publishing first.
void captureBenchFrame(pathtracer::platform::Window& window, AppResources& app, BenchCapture& bench,
                       const pathtracer::debug::PassRecord& pass,
                       const std::shared_ptr<const pathtracer::scene::PathTraceResult>& snapshot, float frameMs,
                       std::chrono::steady_clock::time_point now) {
    const bool ours = pass.generation == bench.generation && !pass.cancelled;
    // Keyed on the generation too: consecutive generations both number passes from 1, so an index-only test would drop a restart's first.
    const bool unseen = bench.passes.empty() || bench.passes.back().generation != pass.generation ||
                        bench.passes.back().passIndex != pass.passIndex;
    if (ours && unseen) {
        bench.passes.push_back(pass);
    }
    bench.frames.push_back(app.stages);
    bench.frameMs.push_back(frameMs);
    bench.presentGpuMs.push_back(app.postTimer.millisecondsElapsed());
    bench.refreshHz.push_back(static_cast<float>(app.refreshHz));
    // Only uploads the displayed AOV's producer issued: a superseded request can still publish, and a raster upload is never the driver's.
    const bool ourUpload = !aovNeedsLightTransport(static_cast<pathtracer::debug::AovId>(app.aov)) ||
                           (snapshot != nullptr && snapshot->generation == bench.generation);
    if (app.stages.uploaded && ourUpload) {
        bench.uploadMs.push_back(app.stages.uploadMs);
    }

    if (!stageComplete(app, bench)) {
        return;
    }
    if (bench.stage > 0) {
        bench.stageWallMs.push_back(std::chrono::duration<float, std::milli>(now - bench.stageStart).count());
    }
    // An empty schedule leaves stage at 0 and ends here, which is the single-stage capture unchanged.
    if (bench.stage + 1 >= bench.aovs.size()) {
        bench.complete = true;
        window.setShouldClose(true);
        return;
    }
    bench.stageStart = now;
    ++bench.stage;
    app.aov = bench.aovs[bench.stage];
}

// Everything the captured workload's cost depends on; two engine records are comparable iff these are equal.
nlohmann::json benchConfig(const AppResources& app, const BenchCapture& bench) {
    const pathtracer::scene::Camera camera = app.debugCamera.snapshot();
    std::vector<std::string> schedule;
    schedule.reserve(bench.aovs.size());
    for (const int aov : bench.aovs) {
        schedule.emplace_back(pathtracer::debug::kAovNames[aov]);
    }
    nlohmann::json config = {{"scene", app.sceneName},
            {"width", bench.passes.back().width},
            {"height", bench.passes.back().height},
            {"max_samples", app.maxSamples},
            {"spp_per_pass", app.pathTraceSettings.samplesPerPixel},
            {"max_bounces", app.pathTraceSettings.maxBounces},
            {"rr_start_bounce", app.pathTraceSettings.russianRouletteStartBounce},
            {"ao_max_distance", app.pathTraceSettings.aoMaxDistance},
            {"aov", pathtracer::debug::kAovNames[app.aov]},
            {"camera", {{"position", {camera.position().x, camera.position().y, camera.position().z}},
                        {"yaw", app.debugCamera.yawDegrees()},
                        {"pitch", app.debugCamera.pitchDegrees()},
                        {"focal_mm", app.debugCamera.focalLengthMm()},
                        {"film_height_mm", app.debugCamera.filmBack().heightMm}}},
            {"env", {{"rotation_deg", app.envRotationDegrees}, {"exposure_stops", app.envExposureStops},
                     {"light", app.envLightEnabled}, {"show_sky", app.showSky}}},
            {"hud", app.showHud},
            {"vsync", app.vsync},
            {"display_type", pathtracer::gfx::scalarTypeName(app.displayFormat)},
            {"texture_type", pathtracer::gfx::scalarTypeName(app.textureType)}};
    // Only with -bench-aovs, so a single-stage record stays comparable with every one logged before this existed.
    if (!schedule.empty()) {
        config["aov_schedule"] = schedule;
    }
    return config;
}

// Raw columns, one entry per event: per captured frame for render stages, per upload for upload_ms, per pass for driver phases.
nlohmann::json benchSamples(const BenchCapture& bench) {
    const auto frameColumn = [&](float pathtracer::debug::FrameStageTimes::*stage) {
        std::vector<float> column;
        column.reserve(bench.frames.size());
        for (const pathtracer::debug::FrameStageTimes& frame : bench.frames) {
            column.push_back(frame.*stage);
        }
        return column;
    };
    const auto passColumn = [&](double pathtracer::debug::PassRecord::*phase) {
        std::vector<double> column;
        column.reserve(bench.passes.size());
        for (const pathtracer::debug::PassRecord& pass : bench.passes) {
            column.push_back(pass.*phase);
        }
        return column;
    };
    using Stages = pathtracer::debug::FrameStageTimes;
    using Pass = pathtracer::debug::PassRecord;
    nlohmann::json samples = {{"frame_ms", bench.frameMs},
            {"fence_ms", frameColumn(&Stages::fenceMs)},
            {"pace_ms", frameColumn(&Stages::paceMs)},
            {"poll_ms", frameColumn(&Stages::pollMs)},
            {"camera_ms", frameColumn(&Stages::cameraMs)},
            {"raster_ms", frameColumn(&Stages::rasterMs)},
            {"filter_ms", frameColumn(&Stages::filterMs)},
            {"upload_ms", bench.uploadMs},
            {"present_ms", frameColumn(&Stages::presentMs)},
            {"present_gpu_ms", bench.presentGpuMs},
            {"refresh_hz", bench.refreshHz},
            {"histogram_ms", frameColumn(&Stages::histogramMs)},
            {"over_range_ms", frameColumn(&Stages::overRangeMs)},
            {"probe_ms", frameColumn(&Stages::probeMs)},
            {"hud_ms", frameColumn(&Stages::hudMs)},
            {"hud_render_ms", frameColumn(&Stages::hudRenderMs)},
            {"swap_ms", frameColumn(&Stages::swapMs)},
            {"pass_trace_ms", passColumn(&Pass::traceMs)},
            {"pass_accumulate_ms", passColumn(&Pass::accumulateMs)},
            {"pass_over_range_ms", passColumn(&Pass::overRangeMs)},
            {"pass_publish_ms", passColumn(&Pass::publishMs)},
            {"pass_ms", passColumn(&Pass::passMs)}};
    // Absent without -bench-aovs, so a single-stage record keeps exactly the columns it has always had.
    if (!bench.stageWallMs.empty()) {
        samples["stage_wall_ms"] = bench.stageWallMs;
    }
    return samples;
}

// Writes the captured accumulation as one record. Refuses an incomplete capture rather than log a workload differing from its config.
bool finishBench(const AppResources& app, const BenchCapture& bench) {
    if (!bench.complete) {
        std::cerr << "pathtracer: -bench closed before the schedule finished; nothing logged\n";
        return false;
    }
    // Contiguous from 1 within each generation, not across the column: each restart numbers its own. A gap means a record went unread.
    std::uint64_t generation = 0;
    int expected = 0;
    for (const pathtracer::debug::PassRecord& pass : bench.passes) {
        if (pass.generation != generation) {
            generation = pass.generation;
            expected = 0;
        }
        if (pass.passIndex != ++expected) {
            std::cerr << "pathtracer: -bench missed a pass record of generation " << generation << " (two passes finished within one frame); nothing logged\n";
            return false;
        }
    }
    pathtracer::debug::RayCounts rays;
    for (const pathtracer::debug::PassRecord& pass : bench.passes) {
        rays.add(pass.rays);
    }
    const pathtracer::debug::BenchRecord record{
        .tool = "pathtracer",
        .argv = bench.argv,
        .config = benchConfig(app, bench),
        .samples = benchSamples(bench),
        .work = {{"rays", {{"primary", rays.primary}, {"bounce", rays.bounce}, {"ao", rays.ao}, {"shadow", rays.shadow}}},
                 {"crc32", pathtracer::debug::floatCrc32(app.pathTraceDriver->latestResult()->beauty.rgba)}},
    };
    return pathtracer::debug::appendBenchRecord(bench.logPath, record);
}

// Bounds GPU work to one frame in flight: at swap interval 0 nothing else stops the command queue outgrowing the GPU.
void waitForPreviousFrame(AppResources& app) {
    if (app.frameFence == nullptr) {
        return;
    }
    GL_CALL(glClientWaitSync(app.frameFence, GL_SYNC_FLUSH_COMMANDS_BIT, std::numeric_limits<GLuint64>::max()));
    GL_CALL(glDeleteSync(app.frameFence));
    app.frameFence = nullptr;
}

// One frame: vblank -> previous GPU completion -> poll -> camera -> retrace if changed -> orbit-pick -> blit -> swap.
void renderFrame(pathtracer::platform::Window& window, pathtracer::platform::DisplayLink& displayLink, AppResources& app) {
    // Every stage zeroed first: one that does not run must read 0, or the dashboard reports the last time it did as if it still were.
    app.stages = {};
    {
        // Before the poll, so input is sampled right after the vblank. Timed either way, so an uncapped frame logs 0 rather than absent.
        const pathtracer::debug::ScopedCpuTimer paceTimer(app.stages.paceMs);
        if (app.vsync) {
            displayLink.waitForNextVblank();
        }
    }
    {
        // After the vblank wait, which the GPU drains the previous frame during, so this is non-zero only for a GPU-bound frame.
        const pathtracer::debug::ScopedCpuTimer fenceTimer(app.stages.fenceMs);
        waitForPreviousFrame(app);
    }
    app.refreshHz = 1.0 / displayLink.refreshPeriodSeconds();
    {
        const pathtracer::debug::ScopedCpuTimer pollTimer(app.stages.pollMs);
        window.pollEvents();
        app.hud.beginFrame();
    }
    app.frameStats.tick();

    const auto frameNow = std::chrono::steady_clock::now();
    const float dtSeconds = std::chrono::duration<float>(frameNow - app.lastFrameTime).count();
    app.lastFrameTime = frameNow;

    const pathtracer::scene::Camera camera = [&] {
        const pathtracer::debug::ScopedCpuTimer cameraTimer(app.stages.cameraMs);
        return updateCamera(window, app, dtSeconds);
    }();
    const auto [viewportWidth, viewportHeight] = window.framebufferSize();
    // The window only frames the image; what is traced is the authored resolution, so a resize never restarts an accumulation.
    const pathtracer::gfx::ViewportRect imageRect =
        pathtracer::gfx::fitAspect(app.imageWidth, app.imageHeight, viewportWidth, viewportHeight);

    requestPathTraceIfTriggerChanged(app, camera, frameNow);

    // Read before the snapshot so a final record guarantees the snapshot holds the final image -- see captureBenchFrame.
    const pathtracer::debug::PassRecord benchPass =
        app.bench ? app.pathTraceDriver->lastPassRecord() : pathtracer::debug::PassRecord{};
    // Held for the rest of the frame so the images stay valid if the driver publishes mid-frame. Null until the first pass completes.
    const std::shared_ptr<const pathtracer::scene::PathTraceResult> pathTraceSnapshot =
        app.pathTraceDriver != nullptr ? app.pathTraceDriver->latestResult() : nullptr;

    resolveOrbitPick(window, app, camera);

    app.postTimer.begin();
    {
        // Inclusive of the display-texture upload inside it; the blit's own cost is the difference, which the dashboard subtracts.
        const pathtracer::debug::ScopedCpuTimer presentTimer(app.stages.presentMs);
        presentFrame(app, pathTraceSnapshot, camera, viewportWidth, viewportHeight, imageRect);
    }
    app.postTimer.end();

    // Captured after the composited image lands in the default framebuffer, before the HUD draws on top of it.
    sampleDisplayedFrame(app, pathTraceSnapshot, imageRect);

    updateHud(app, window, camera, pathTraceSnapshot, imageRect);

    {
        const pathtracer::debug::ScopedCpuTimer swapTimer(app.stages.swapMs);
        window.swapBuffers();
    }
    GL_CALL(app.frameFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0));

    if (app.statsEnabled) {
        // After swapBuffers, so the dashboard's write(2) lands in the frame's slack. It times its own draw, the stages being zeroed first.
        updateDashboard(app, pathTraceSnapshot, dtSeconds * 1000.0F, viewportWidth, viewportHeight);
    }
    if (app.bench) {
        captureBenchFrame(window, app, *app.bench, benchPass, pathTraceSnapshot, dtSeconds * 1000.0F, frameNow);
    }
}

struct Options {
    std::string scenePath = ASSET_ROOT_DIR "/scenes/cornell.json";
    // Off by default: the live dashboard redraws in place, wrong for anything scripted. Always compiled in, so -stats measures what ships.
    bool stats = false;
    // -bench PATH: run one accumulation to profile.json's maxSamples, append it to this benchmark log (bench_log.h), exit.
    std::string benchLogPath;
    // -bench-aovs A,B,C: AovId sequence the run walks, one stage per entry. Empty = today's single-stage capture.
    std::vector<int> benchAovs;
};

// Resolves a comma-separated AOV list against kAovNames, so -bench-aovs and the HUD name the same AOVs. nullopt on an unknown name.
std::optional<std::vector<int>> parseAovList(const char* list) {
    std::vector<int> aovs;
    const std::string text(list);
    for (std::size_t begin = 0; begin <= text.size();) {
        const std::size_t comma = text.find(',', begin);
        const std::string name = text.substr(begin, comma - begin);
        const auto* match = std::find_if(std::begin(pathtracer::debug::kAovNames), std::end(pathtracer::debug::kAovNames),
                                          [&name](const char* candidate) { return name == candidate; });
        if (match == std::end(pathtracer::debug::kAovNames)) {
            std::cerr << "pathtracer: -bench-aovs has no AOV named '" << name << "'; valid names are";
            for (const char* candidate : pathtracer::debug::kAovNames) {
                std::cerr << " '" << candidate << "'";
            }
            std::cerr << '\n';
            return std::nullopt;
        }
        aovs.push_back(static_cast<int>(match - std::begin(pathtracer::debug::kAovNames)));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return aovs;
}

// nullopt on an unrecognized flag or a missing value: argv is a system boundary, so a bad value is surfaced, not defaulted around.
std::optional<Options> parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-scene") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "pathtracer: -scene expects a value\n";
                return std::nullopt;
            }
            options.scenePath = argv[++i];
        } else if (std::strcmp(argv[i], "-stats") == 0) {
            options.stats = true;
        } else if (std::strcmp(argv[i], "-bench") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "pathtracer: -bench expects a value\n";
                return std::nullopt;
            }
            options.benchLogPath = argv[++i];
        } else if (std::strcmp(argv[i], "-bench-aovs") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "pathtracer: -bench-aovs expects a value\n";
                return std::nullopt;
            }
            std::optional<std::vector<int>> aovs = parseAovList(argv[++i]);
            if (!aovs) {
                return std::nullopt;
            }
            options.benchAovs = std::move(*aovs);
        } else {
            std::cerr << "pathtracer: unknown flag " << argv[i]
                       << "\n  usage: pathtracer [-scene path/to/scene.json] [-stats] [-bench log.jsonl] [-bench-aovs Beauty,Normal,...]\n";
            return std::nullopt;
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    const std::optional<Options> options = parseOptions(argc, argv);
    if (!options) {
        return EXIT_FAILURE;
    }

    glfwSetErrorCallback(&glfwErrorCallback);

    if (glfwInit() != GLFW_TRUE) {
        std::cerr << "main: glfwInit failed\n";
        return EXIT_FAILURE;
    }

    int exitCode = EXIT_SUCCESS;
    {
        // profile.json's window size must be known before Window is constructed, so config loads first, before any GL object exists.
        std::optional<pathtracer::config::SceneConfig> sceneConfig =
            pathtracer::config::loadSceneConfig(options->scenePath);
        std::optional<pathtracer::config::ProfileConfig> profileConfig =
            pathtracer::config::loadProfileConfig(ASSET_ROOT_DIR "/config/profile.json");

        if (!sceneConfig || !profileConfig) {
            std::cerr << "main: scene/profile config load failed, aborting startup\n";
            exitCode = EXIT_FAILURE;
        } else if (options->benchLogPath.empty() && !options->benchAovs.empty()) {
            std::cerr << "main: -bench-aovs is the schedule -bench walks; it does nothing on its own\n";
            exitCode = EXIT_FAILURE;
        } else if (!options->benchLogPath.empty() &&
                   (profileConfig->pathTracer.maxSamples <= 0 ||
                    !aovNeedsLightTransport(static_cast<pathtracer::debug::AovId>(
                        options->benchAovs.empty() ? profileConfig->render.defaultAov
                                                   : options->benchAovs.front())))) {
            // An unbounded accumulation never ends and a rasterizer AOV parks the driver, so neither is a benchmark workload.
            std::cerr << "main: -bench needs profile.json maxSamples > 0 and a path-traced first AOV\n";
            exitCode = EXIT_FAILURE;
        } else {
            // Window construction creates the GL 4.1 core context and makes it current; a fatal failure inside exits the process.
            pathtracer::platform::Window window(profileConfig->render.width, profileConfig->render.height,
                                             "PATHTRACER");

            glewExperimental = GL_TRUE;
            const GLenum glewStatus = glewInit();
            // GLEW's init is known to leave a spurious error even on success; drain it here so it's never misattributed to a later GL_CALL.
            while (glGetError() != GL_NO_ERROR) {
            }

            if (glewStatus != GLEW_OK) {
                std::cerr << "main: glewInit failed: "
                          << reinterpret_cast<const char*>(glewGetErrorString(glewStatus)) << '\n';
                exitCode = EXIT_FAILURE;
            } else {
                // Off: NSGL's interval lets two swaps through per refresh, and paces an occluded window at a fixed 60 Hz usleep.
                glfwSwapInterval(0);
                pathtracer::platform::DisplayLink displayLink(window);

                std::optional<AppResources> app =
                    initializeApp(*sceneConfig, *profileConfig, window, options->scenePath,
                                   options->stats, 1.0 / displayLink.refreshPeriodSeconds());
                if (!app) {
                    exitCode = EXIT_FAILURE;
                } else {
                    // Constructed here, not in the initializer list: this local is where the scene objects reach their final address.
                    app->pathTraceDriver = std::make_unique<pathtracer::scene::PathTraceDriver>(
                        app->sceneAccel, app->stumpModel.shadingTriangles, app->stumpModel.instances,
                        app->instanceLightIndex, app->environmentMap, app->quadLights,
                        app->perInstanceSettings);

                    wireCallbacks(window, *app);
                    if (!options->benchLogPath.empty()) {
                        app->bench.emplace();
                        app->bench->logPath = options->benchLogPath;
                        app->bench->argv.assign(argv, argv + argc);
                        app->bench->aovs = options->benchAovs;
                        if (!app->bench->aovs.empty()) {
                            app->aov = app->bench->aovs.front();
                        }
                    }

                    while (!window.shouldClose()) {
                        renderFrame(window, displayLink, *app);
                    }
                    GL_CALL(glDeleteSync(app->frameFence));
                    if (app->bench && !finishBench(*app, *app->bench)) {
                        exitCode = EXIT_FAILURE;
                    }
                }
            }
        }
    }  // Window destroyed here, while GLFW is still initialized.

    glfwTerminate();
    return exitCode;
}
