#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/config/profile_config.h"
#include "pathtracer/config/scene_config.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/path_tracer.h"

namespace pathtracer::scene {

// Each instance's constants: materialPath's file, or its node's materialOverrides file; false on a bad key or file.
[[nodiscard]] bool applySceneMaterials(std::vector<MeshInstance>& instances, const std::string& materialPath,
                                       const std::map<std::string, std::string>& materialOverrides, const std::string& assetRoot);

// Binds SceneConfig::textures over each named node's inputs; false on a bad key, input or binding, or a non-finite texel.
[[nodiscard]] bool bindSceneTextures(std::vector<MeshInstance>& instances,
                                     const std::map<std::string, std::map<std::string, pathtracer::config::TextureConfig>>& textures,
                                     const std::string& assetRoot);

// Places each authored quad by sceneTransform * its own placement, as QuadLight corner + edges. sceneTransform must be rigid.
[[nodiscard]] std::vector<QuadLight> buildQuadLights(
    const std::vector<pathtracer::config::QuadLightConfig>& lights, const glm::mat4& sceneTransform);

// scene.json's model placement: translate * rotationXyz, the order every caller places the scene in.
[[nodiscard]] glm::mat4 rootTransformOf(const pathtracer::config::ModelConfig& model);

// The profile's integrator limits; samplesPerPixel is the caller's, interactive or headless.
[[nodiscard]] PathTraceSettings baseSettingsOf(const pathtracer::config::ProfileConfig& profile, int samplesPerPixel);

}  // namespace pathtracer::scene
