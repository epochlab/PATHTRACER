#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/scene/material.h"

namespace pathtracer::config {

// What geometry to load and where to place it. gltfPath is relative to ASSET_ROOT_DIR, as every asset path is.
struct ModelConfig {
    std::string gltfPath;
    glm::vec3 position;         // model root, composed on top of the glTF's own node transforms
    glm::vec3 rotation;  // degrees; the root rotation is Rz*Ry*Rx, so X applies first
};

// What lights the scene, IBL half.
struct EnvironmentConfig {
    std::string hdriPath;  // environment map, relative to ASSET_ROOT_DIR
    // Whether the environment is a light (NEE, MIS, contributing to every miss) rather than just the background. Optional, default true.
    bool lightEnabled = true;
    std::optional<std::string> colorSpace;  // the map's OCIO colour space, overriding its file's tag. Optional
};

// One texture binding: the image relative to ASSET_ROOT_DIR and, for a colour slot only, an OCIO space overriding its file's tag.
struct TextureConfig {
    std::string path;
    std::optional<std::string> colorSpace;
};

// Rectangular area light: a size.x by size.y quad centred on its local origin in the XY plane, color * intensity leaving along -Z.
struct QuadLightConfig {
    glm::vec3 position;  // quad centre, before the model root transform
    glm::vec3 rotation;  // degrees; Rz*Ry*Rx as ModelConfig::rotation, so X applies first
    glm::vec2 size;  // width along local X, height along local Y; loadSceneConfig rejects non-positive extents
    glm::vec3 color;  // loadSceneConfig rejects a negative color or intensity
    float intensity;
    bool twoSided = false;
};

// Tunable shading constants turning a material's textures into BSDF input, externalized so they need no recompile.
struct MaterialConfig {
    float bumpStrength;   // world-space height per unit bump texture value (metres): h = bumpStrength*H; see buildShadingFrame
    float roughnessMin;   // floor applied to the roughness texture sample, avoids a near-zero-roughness GGX singularity
    float roughnessMax;   // ceiling applied to the roughness texture sample
    glm::vec3 diffuseColour;      // multiplies Material::baseColor
    // Dielectric IOR, non-metal lobes only. Optional, default 1.5: nullified by (1-metallic), so a pure conductor need not declare it.
    float ior = 1.5F;
    // Abbe number V_d = (n_d-1)/(n_F-n_C), giving ior a wavelength dependence. Optional, default 0.0 = no dispersion (Arnold, OpenPBR).
    float abbe = 0.0F;
    // KHR_materials_transmission-style factor, 0 = opaque. Optional, default 0.0, a true no-op (transmitProb=0), so opaque files omit it.
    float transmissionFactor = 0.0F;
    // Optional, default 0.0: a true no-op, so a non-metal material (e.g. clay.json, glass.json) need not declare it.
    float metallicFactor = 0.0F;
    float roughnessFactor;         // multiplies the roughness texture sample, before roughnessMin/Max clamp
    // EON rough-diffuse r in [0,1]; 0 = Lambertian. Optional, default 0.0: dead at metallicFactor=1 or transmissionFactor=1.
    float diffuseRoughness = 0.0F;
    // The transmission lobe's only tint; baseColor tints reflection alone. Optional, default [1,1,1], the identity in both regimes.
    glm::vec3 transmissionColor = glm::vec3(1.0F);
    // Distance at which transmittance reaches transmissionColor by Beer's law -- the medium's density. Optional, default 0.0 = no medium.
    float transmissionDepth = 0.0F;
    // Gulbrandsen 2014 edgetint for the conductor lobe. Optional, default [1,1,1]: the no-dip edge Schlick always produced.
    glm::vec3 edgeTint = glm::vec3(1.0F);
    // Optional "shadingModel" key, default "standard"; "constant" admits only diffuseColour and leaves every BSDF field at identity.
    pathtracer::scene::ShadingModel shadingModel = pathtracer::scene::ShadingModel::Standard;
};

// The asset to load and how to shade and light it: what is specific to this scene, as against ProfileConfig's session-wide defaults.
struct SceneConfig {
    ModelConfig model;
    EnvironmentConfig environment;
    // Relative to ASSET_ROOT_DIR, a material JSON (materials/clay.json), matching ModelConfig::gltfPath.
    std::string materialPath;
    // glTF node name -> material JSON path, overriding materialPath for that instance. Optional; absent means every instance uses it.
    std::map<std::string, std::string> materialOverrides;
    // glTF node name -> {Material slot name -> texture}, a path string or {path, colorSpace}; the only texture source. Optional.
    std::map<std::string, std::map<std::string, TextureConfig>> textures;
    // Rectangular area lights, in the glTF's own vertex space (ModelConfig position/rotation applies too). Optional; absent means none.
    std::vector<QuadLightConfig> lights;
};

// Reads and parses path; nullopt and a stderr log if missing, unreadable or unparseable. User input: failure is surfaced, not asserted.
[[nodiscard]] std::optional<SceneConfig> loadSceneConfig(const std::string& path);

// Reads and parses a standalone material file (e.g. assets/materials/clay.json). Same failure contract as loadSceneConfig.
[[nodiscard]] std::optional<MaterialConfig> loadMaterialConfig(const std::string& path);

}  // namespace pathtracer::config
