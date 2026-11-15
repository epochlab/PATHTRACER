#pragma once

#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/config/profile_config.h"
#include "pathtracer/config/scene_config.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/path_tracer.h"

namespace pathtracer::scene {

// Each instance's constants: materialPath's file, or its node's materialOverrides file; false on a bad key or file.
[[nodiscard]] bool applySceneMaterials(std::span<MeshInstance> instances, const std::string& materialPath,
                                       const std::map<std::string, std::string>& materialOverrides, const std::string& assetRoot);

// Binds SceneConfig::textures over each named node's inputs; false on a bad key, input or binding, or a non-finite texel.
[[nodiscard]] bool bindSceneTextures(std::span<MeshInstance> instances,
                                     const std::map<std::string, std::map<std::string, pathtracer::config::TextureConfig>>& textures,
                                     const std::string& assetRoot);

// Places each light by sceneTransform * its own placement, mesh glTFs loaded under assetRoot; nullopt logs why. Rigid sceneTransform.
[[nodiscard]] std::optional<std::vector<AreaLight>> buildLights(const std::vector<pathtracer::config::LightConfig>& lights,
                                                                const glm::mat4& sceneTransform, const std::string& assetRoot);

// The luminance in cd/m^2 `light` emits once placed as `placed`: its authored luminance, or its lumens over pi A sides.
[[nodiscard]] float emittedLuminance(const pathtracer::config::LightConfig& light, const AreaLight& placed);

// scene.json's model placement: translate * rotationXyz, the order every caller places the scene in.
[[nodiscard]] glm::mat4 rootTransformOf(const pathtracer::config::ModelConfig& model);

// Everything the root and light rotations are baked into: the triangle soup, its BVH, the lights and the per-instance bounds.
struct SceneGeometry {
    LoadedModel model;  // the glTF's instances, then one per light, appended last
    std::vector<AreaLight> lights;
    std::vector<int> instanceLightIndex;  // parallel to model.instances, -1 where the instance emits nothing
    std::vector<AabbBounds> instanceBounds;
    EmbreeAccel accel;
    double accelBuildMs;  // the Embree build alone, for the viewer's spec block
};

// Loads the glTF under model's root (none: lights only), places lights under it and builds the BVH; nullopt with error on failure.
[[nodiscard]] std::optional<SceneGeometry> buildSceneGeometry(const std::string& assetRoot, const pathtracer::config::ModelConfig& model,
                                                              const std::vector<pathtracer::config::LightConfig>& lights, std::string& error);

// The glTF's own instances, ahead of the lights' emitting ones: what scene materials and textures bind to; an emitter has no BSDF.
[[nodiscard]] inline std::span<MeshInstance> surfaceInstances(SceneGeometry& geometry) {
    return std::span(geometry.model.instances).first(geometry.model.instances.size() - geometry.lights.size());
}

// The profile's integrator limits at one sample per pass, so passes count samples for the cap, replicates and checkpoints alike.
[[nodiscard]] PathTraceSettings baseSettingsOf(const pathtracer::config::ProfileConfig& profile);

}  // namespace pathtracer::scene
