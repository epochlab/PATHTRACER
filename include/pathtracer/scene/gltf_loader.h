#pragma once

#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/scene/light.h"
#include "pathtracer/scene/material.h"
#include "pathtracer/scene/ray_types.h"
#include "pathtracer/scene/shading_scene.h"

namespace pathtracer::scene {

// One glTF primitive: its material and its baked world-space transform. Geometry lives only in LoadedModel's triangle soup.
struct MeshInstance {
    Material material;
    glm::mat4 transform;
    std::string name;  // owning glTF node's name, empty if the node has none; keys SceneConfig::materialOverrides/textures
};

struct LoadedModel {
    std::vector<MeshInstance> instances;
    // Every triangle across every instance, pre-transformed to world space at the one point where both are in hand.
    std::vector<Triangle> worldTriangles;
    // Per-vertex normal/uv/tangent, parallel-indexed with worldTriangles -- Hit::triangleIndex resolves directly into this.
    std::vector<ShadingTriangle> shadingTriangles;
};

// Parses path's geometry via cgltf, ignoring its materials, into neutral Material{}s; vertexColor false reads COLOR_0 as absent (white).
std::optional<LoadedModel> loadGltf(const std::string& path, const glm::mat4& rootTransform = glm::mat4(1.0F),
                                    bool vertexColor = true);

// Appends each light's emitting geometry to `model`, plus one entry per light to `instanceLightIndex`, pre-sized to instances, all -1.
void appendQuadLights(LoadedModel& model, const std::vector<QuadLight>& lights,
                       std::vector<int>& instanceLightIndex);

}  // namespace pathtracer::scene
