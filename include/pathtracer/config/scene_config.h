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

// A height map supplying a normal input by bump: the image and the world-space height, in metres, of one unit of its value.
struct BumpConfig {
    std::string path;
    float heightMetres;
};

// One input's texture relative to ASSET_ROOT_DIR; colorSpace only on a colour input, channel only on a scalar one, bump only on a normal.
struct TextureConfig {
    std::optional<std::string> path;  // required except on a normal input, which may be a bump alone
    std::optional<std::string> colorSpace;  // OCIO space overriding the file's tag
    int channel = 0;  // R, G or B as 0, 1, 2: the channel a scalar input reads, so a packed glTF metallic-roughness map binds
    std::optional<BumpConfig> bump;
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

// A material file: OpenPBR Surface v1.1.1 constants under their specification names, every omitted input at its default.
using MaterialConfig = pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>;

// The asset to load and how to shade and light it: what is specific to this scene, as against ProfileConfig's session-wide defaults.
struct SceneConfig {
    ModelConfig model;
    EnvironmentConfig environment;
    // Relative to ASSET_ROOT_DIR, a material JSON (materials/clay.json), matching ModelConfig::gltfPath.
    std::string materialPath;
    // glTF node name -> material JSON path, overriding materialPath for that instance. Optional; absent means every instance uses it.
    std::map<std::string, std::string> materialOverrides;
    // glTF node name -> {OpenPBR input name -> texture}, a path string or a TextureConfig object; the only texture source. Optional.
    std::map<std::string, std::map<std::string, TextureConfig>> textures;
    // Rectangular area lights, in the glTF's own vertex space (ModelConfig position/rotation applies too). Optional; absent means none.
    std::vector<QuadLightConfig> lights;
};

// Reads and parses path; nullopt and a stderr log if missing, unreadable or unparseable. User input: failure is surfaced, not asserted.
[[nodiscard]] std::optional<SceneConfig> loadSceneConfig(const std::string& path);

// Reads and parses a standalone material file (e.g. assets/materials/clay.json). Same failure contract as loadSceneConfig.
[[nodiscard]] std::optional<MaterialConfig> loadMaterialConfig(const std::string& path);

}  // namespace pathtracer::config
