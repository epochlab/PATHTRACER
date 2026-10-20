#pragma once

#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/debug/denoiser.h"
#include "pathtracer/debug/hud_overlay.h"
#include "pathtracer/gfx/ocio_display_transform.h"
#include "pathtracer/gfx/scalar_type.h"
#include "pathtracer/scene/camera.h"

namespace pathtracer::config {

struct CameraConfig {
    glm::vec3 position;
    glm::vec3 rotation;  // degrees; Rz*Ry*Rx as scene.json's rotation keys, so X applies first

    // Resolved against assets/config/sensor.json by name at startup; loadProfileConfig cannot validate it, not loading that file.
    std::string defaultFilmBackPresetName;
    float focalLengthMm;
    float nearClip;
    float farClip;
    float aperture;
    float shutterSeconds;
    float iso;
    // Projection selector plus the polynomial; Camera::validate checks both whichever is active, as the HUD dropdown switches either way.
    pathtracer::scene::Lens lens;
    pathtracer::gfx::PixelLattice pixelLattice;
    float acceptanceAngleDegrees;  // a compound-eye facet's half-maximum width; 0 is a camera's film filter
};

struct ControlsConfig {
    float flySpeedMetersPerSecond;
    float orbitSensitivityDegPerPixel;
};

struct RenderConfig {
    // The authored image, in pixels. Independent of the window, which fits it at launch and zooms it by powers of two.
    int width;
    int height;
    // The window's framebuffer, in pixels (2x its points on Retina); the image is drawn inside it with a 1 px bounds ring.
    int displayWidth;
    int displayHeight;
    // Fraction of width x height traced, magnified to the viewport by the display draw. Both in (0,1].
    float renderScale;
    float interactiveRenderScale;  // the same, while the camera is moving
    // Index into pathtracer::debug::AovId / kAovNames (aov.h) (0 = Beauty).
    int defaultAov;
    pathtracer::gfx::OcioDisplayTransform::Lut defaultLut;
    // true paces each frame to the display's vblank (DisplayLink); false runs uncapped, bounded only by the one-frame-in-flight fence.
    bool vsync;
    // profile.json displayBitDepth, the path-traced display texture's GL storage: 16 -> Float16, 32 -> Float32. 8-bit UNORM would clamp.
    pathtracer::gfx::ScalarType displayFormat;
    // The glare PSF's scattered fraction in [0,1], 0 = off: the HUD Bloom slider's starting value, applied to picture AOVs only.
    float bloomStrength;
    pathtracer::debug::PixelProbeReadout pixelProbe;  // profile.json "colour", "photometric" or "none"
};

struct PathTracerConfig {
    int samplesPerPixel;   // path tracer startup default
    int maxBounces;        // path tracer startup default; secondary/indirect bounces beyond the primary hit, 0 = direct lighting only
    int russianRouletteStartBounce;  // 0-based bounce index RR kicks in from
    int maxSamples;  // accumulated-pass cap for PathTraceDriver; 0 = unbounded
    float aoMaxDistance;  // ray-traced AO occlusion range, scene units; occluders beyond it don't darken
    float lookaheadDistance;  // horizon of the Lookahead AOV's ramp, scene units; geometry at or beyond it reads 0
    int replicates;  // independent scrambles interleaved by pass, whose spread is the SNR's standard error; divides maxSamples
    // pathTracer.adaptive: texels stop once their relative luminance error is within threshold, at this false-discovery confidence.
    float adaptiveThreshold;
    float adaptiveConfidence;
    int adaptiveMinSamples;  // the first checkpoint, replicates times a power of two
};

// profile.json's denoiser block: OIDN for the Denoised AOV, and the feature chain its albedo and normal come from.
struct DenoiserConfig {
    bool enabled;  // false skips the integrator's feature chain and withholds the Denoised AOV
    pathtracer::debug::DenoiserOptions options;
};

// Session-wide defaults: the controller's initial and reset pose, lens, exposure and tuning constants -- what is not specific to one scene.
struct ProfileConfig {
    CameraConfig camera;
    ControlsConfig controls;
    RenderConfig render;
    PathTracerConfig pathTracer;
    DenoiserConfig denoiser;
};

// The sample, bounce and size counts and their cross-constraints, logged against `source`; main re-checks after its overrides.
[[nodiscard]] bool validCounts(const RenderConfig& render, const PathTracerConfig& pathTracer, const std::string& source);

// Reads and parses path; nullopt and a stderr log if missing, unreadable or unparseable. User input: failure is surfaced, not asserted.
[[nodiscard]] std::optional<ProfileConfig> loadProfileConfig(const std::string& path);

// Reads the film-back preset catalogue, a JSON array of {name, widthMm, heightMm}; each must pass Camera::validFilmBack.
[[nodiscard]] std::optional<std::vector<pathtracer::scene::Camera::FilmBackPreset>> loadFilmBackPresets(
    const std::string& path);

}  // namespace pathtracer::config
