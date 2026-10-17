#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/scene/ray_types.h"

namespace pathtracer::scene {

struct AabbBounds {
    glm::vec3 min;
    glm::vec3 max;
};

// World-space, baked from the glTF loader's per-vertex data -- the path tracer's only source of per-vertex shading data.
struct ShadingVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec4 tangent;              // xyz = tangent, w = bitangent handedness (glTF convention)
    glm::vec3 colour = glm::vec3(1.0F);  // COLOR_0, multiplies baseColor; white = no vertex tint
};

// Indexed identically to the world-space Triangle list (ray_types.h) -- Hit::triangleIndex resolves directly into this.
struct ShadingTriangle {
    ShadingVertex v0;
    ShadingVertex v1;
    ShadingVertex v2;
    int instanceIndex;  // indexes LoadedModel::instances, owner of this triangle's Material
};

// Barycentric interpolation (u/v = Hit's Moller-Trumbore convention, w=1-u-v on v0); normal/tangent renormalized after blending.
[[nodiscard]] ShadingVertex interpolateShading(const ShadingTriangle& tri, float u, float v);

// Ray origin past the hit's rounding-error bound along unit geoNormal (PBR 4e 6.8.6), then Chiang's terminator lift; tMin 0 suffices.
[[nodiscard]] glm::vec3 spawnOrigin(const ShadingTriangle& tri, float u, float v, const glm::vec3& geoNormal, bool normalSide);

// True for the inverted box computeInstanceBounds leaves on an instance with no triangles: a min/max identity, not a sentinel.
[[nodiscard]] bool isEmpty(const AabbBounds& box);

// World-space AABB per instance, indexed by instanceIndex, over its triangles and analytic shapes: one linear pass over each.
[[nodiscard]] std::vector<AabbBounds> computeInstanceBounds(const std::vector<ShadingTriangle>& triangles, int instanceCount,
                                                             const std::vector<AnalyticShape>& shapes = {});

// The shading record at `point` on a shape: sphere uv lat-long in its frame, disc uv its frame's planar [0,1]^2 as a quad's; colour white.
[[nodiscard]] ShadingVertex shadingAt(const AnalyticShape& shape, const glm::vec3& point);

}  // namespace pathtracer::scene
