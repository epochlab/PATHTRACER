#pragma once

#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

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
};

struct ControlsConfig {
    float flySpeedMetersPerSecond;
    float orbitSensitivityDegPerPixel;
};

struct RenderConfig {
    // The authored image, in pixels. Independent of the window: the display viewport letterboxes and magnifies it.
    int width;
    int height;
    // Fraction of width x height traced, magnified to the viewport by the display draw. Both in (0,1].
    float renderScale;
    float interactiveRenderScale;  // the same, while the camera is moving
    // Index into pathtracer::debug::AovId / kAovNames (aov.h) (0 = Beauty).
    int defaultAov;
    pathtracer::gfx::OcioDisplayTransform::Lut defaultLut;
    // true paces each frame to the display's vblank (DisplayLink); false runs uncapped, bounded only by the one-frame-in-flight fence.
    bool vsync;
    // profile.json bit depths: 16 -> Float16, 32 -> Float32. 8-bit UNORM is not offered: it clamps scene-referred data before exposure.
    pathtracer::gfx::ScalarType displayFormat;  // displayBitDepth: the path-traced display texture's GL storage
    pathtracer::gfx::ScalarType textureType;    // textureBitDepth: environment HDRI and every material texture's CPU storage
};

struct PathTracerConfig {
    int samplesPerPixel;   // path tracer startup default
    int maxBounces;        // path tracer startup default; secondary/indirect bounces beyond the primary hit, 0 = direct lighting only
    int russianRouletteStartBounce;  // 0-based bounce index RR kicks in from
    int maxSamples;  // accumulated-pass cap for PathTraceDriver; 0 = unbounded
    float aoMaxDistance;  // ray-traced AO occlusion range, scene units; occluders beyond it don't darken
    float lookaheadDistance;  // horizon of the Lookahead AOV's ramp, scene units; geometry at or beyond it reads 0
};

// Session-wide defaults: the controller's initial and reset pose, lens, exposure and tuning constants -- what is not specific to one scene.
struct ProfileConfig {
    CameraConfig camera;
    ControlsConfig controls;
    RenderConfig render;
    PathTracerConfig pathTracer;
};

// Reads and parses path; nullopt and a stderr log if missing, unreadable or unparseable. User input: failure is surfaced, not asserted.
[[nodiscard]] std::optional<ProfileConfig> loadProfileConfig(const std::string& path);

// Reads the film-back preset catalogue, a JSON array of {name, widthMm, heightMm}; each must pass Camera::validFilmBack.
[[nodiscard]] std::optional<std::vector<pathtracer::scene::Camera::FilmBackPreset>> loadFilmBackPresets(
    const std::string& path);

}  // namespace pathtracer::config
