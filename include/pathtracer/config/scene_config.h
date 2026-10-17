#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/config/profile_config.h"
#include "pathtracer/scene/material.h"

namespace pathtracer::config {

// What geometry to load and where to place it: the root pose lights and requests turn even when no glTF is named.
struct ModelConfig {
    std::optional<std::string> gltfPath;  // relative to ASSET_ROOT_DIR, as every asset path is; nullopt when the scene has no model
    glm::vec3 position;         // model root, composed on top of the glTF's own node transforms
    glm::vec3 rotation;  // degrees; the root rotation is Rz*Ry*Rx, so X applies first
    bool vertexColor = true;  // glTF COLOR_0 multiplies base_color; false reads every vertex as white. Optional, default true
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

// A size.x by size.y rectangle in the local XY plane, width along X and height along Y; loadSceneConfig rejects non-positive extents.
struct QuadShape {
    glm::vec2 size;
};

// A disk of `radius` in the local XY plane; loadSceneConfig rejects a non-positive one.
struct DiskShape {
    float radius;
};

// A sphere of `radius`, closed so emitting outward only: it takes no twoSided key. loadSceneConfig rejects a non-positive radius.
struct SphereShape {
    float radius;
};

// A glTF's triangles, materials ignored, pivoting on their bounding-box centre and uniformly scaled about it before rotation.
struct MeshShape {
    std::string gltfPath;  // relative to ASSET_ROOT_DIR
    float scale = 1.0F;    // optional, default 1; loadSceneConfig rejects a non-positive one
};

// An area light centred on its local origin, color * intensity leaving along local -Z (quad, disk), outward (sphere) or each face's front.
struct LightConfig {
    glm::vec3 position;  // the shape's centre, before the model root transform
    glm::vec3 rotation;  // degrees; Rz*Ry*Rx as ModelConfig::rotation, so X applies first
    std::variant<QuadShape, DiskShape, SphereShape, MeshShape> shape;
    glm::vec3 color;  // authored, or resolved from a kelvin temperature at unit luminance; loadSceneConfig rejects a negative one
    float intensity;
    bool twoSided = false;  // a sphere's stays false
};

// A volume material file, MaterialX anisotropic_vdf's inputs: coefficients per unit of the grid's own space at density 1.
struct VolumeMaterialConfig {
    glm::vec3 absorption;
    glm::vec3 scattering;
    float anisotropy;  // Henyey-Greenstein g in (-1, 1), positive forward
    float density = 1.0F;  // multiplies the grid's values: thickness without changing colour or albedo. Optional, default 1
};

// A scene's own camera pose and far clip over profile.json's, where the shared default cannot frame it; lens and film stay the profile's.
struct SceneCameraConfig {
    glm::vec3 position;
    glm::vec3 rotation;  // degrees; Rz*Ry*Rx as every rotation key, so X applies first
    float farClip;
};

// A VDB density grid as a heterogeneous medium in world space, not under the model root: sigma(x) = density(x) * coefficient.
struct VolumeConfig {
    std::string vdbPath;  // relative to ASSET_ROOT_DIR
    std::string grid;  // the FloatGrid's name in the file
    glm::vec3 position;
    glm::vec3 rotation;  // degrees; Rz*Ry*Rx as ModelConfig::rotation, so X applies first
    float scale = 1.0F;  // uniform, applied first; coefficients are in grid units, so scaling keeps the optical depth
    std::string materialPath;  // a VolumeMaterialConfig file, relative to ASSET_ROOT_DIR
};

// A material file: OpenPBR Surface v1.1.1 constants under their specification names, every omitted input at its default.
struct MaterialConfig : pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant> {
    pathtracer::scene::RandomCheckerInputs randomCheckers;  // inputs a {"randomChecker": {...}} object drives; their constants stay default
};

// The asset to load and how to shade and light it: what is specific to this scene, as against ProfileConfig's session-wide defaults.
struct SceneConfig {
    ModelConfig model;  // optional in the file: absent is no glTF at the identity pose
    EnvironmentConfig environment;
    // Relative to ASSET_ROOT_DIR, a material JSON (materials/clay.json), matching ModelConfig::gltfPath; required exactly with a model.
    std::optional<std::string> materialPath;
    // glTF node name -> material JSON path, overriding materialPath for that instance. Optional; absent means every instance uses it.
    std::map<std::string, std::string> materialOverrides;
    // glTF node name -> {OpenPBR input name -> texture}, a path string or a TextureConfig object; the only texture source. Optional.
    std::map<std::string, std::map<std::string, TextureConfig>> textures;
    // Area lights, in the glTF's own vertex space (ModelConfig position/rotation applies too). Optional; absent means none.
    std::vector<LightConfig> lights;
    // VDB media in world space. Optional; absent means none.
    std::vector<VolumeConfig> volumes;
    // Optional; absent keeps profile.json's camera.
    std::optional<SceneCameraConfig> camera;
};

// Reads and parses path; nullopt and a stderr log if missing, unreadable or unparseable. User input: failure is surfaced, not asserted.
[[nodiscard]] std::optional<SceneConfig> loadSceneConfig(const std::string& path);

// Puts scene.camera, where the scene names one, over profile.camera: the one place a scene overrides the session's camera.
void applySceneCamera(const SceneConfig& scene, ProfileConfig& profile);

// Reads and parses a standalone material file (e.g. assets/materials/clay.json). Same failure contract as loadSceneConfig.
[[nodiscard]] std::optional<MaterialConfig> loadMaterialConfig(const std::string& path);

// Reads and validates a volume material file (e.g. assets/materials/volume.json). Same failure contract as loadSceneConfig.
[[nodiscard]] std::optional<VolumeMaterialConfig> loadVolumeMaterialConfig(const std::string& path);

}  // namespace pathtracer::config
