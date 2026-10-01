#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/config/scene_config.h"
#include "pathtracer/gfx/scalar_type.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/path_tracer.h"

namespace pathtracer::scene {

// Binds the scene's EXRs (SceneConfig::textures) into each named node's Material slots; false on a bad key, slot or file.
[[nodiscard]] bool bindSceneTextures(std::vector<MeshInstance>& instances,
                                     const std::map<std::string, std::map<std::string, std::string>>& textures,
                                     const std::string& assetRoot, pathtracer::gfx::ScalarType textureType);

// One PathTraceSettings per instance from the scene's material overrides, parallel to `instances`. nullopt on a bad file or unknown key.
[[nodiscard]] std::optional<std::vector<PathTraceSettings>> resolvePerInstanceSettings(
    const PathTraceSettings& base, const std::vector<MeshInstance>& instances,
    const std::map<std::string, std::string>& materialOverrides, const std::string& assetRoot);

// Transforms authored quad lights into world space: origin as a point, edge0/edge1 as displacements. sceneTransform must be rigid.
[[nodiscard]] std::vector<QuadLight> buildQuadLights(
    const std::vector<pathtracer::config::QuadLightConfig>& lights, const glm::mat4& sceneTransform);

}  // namespace pathtracer::scene
