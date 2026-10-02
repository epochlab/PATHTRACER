#pragma once

#include <glm/glm.hpp>

#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/material.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/shading_scene.h"

namespace pathtracer::scene {

// Primary-hit G-buffer sampling shared by the path tracer (tracePath bounce 0) and the rasterizer, so both resolve materials alike.

// near: p within thicknessPx of segment [a,b], clamped to its extent, not the infinite line. t: the closest point's [0,1] parameter.
struct LineProximity {
    bool near;
    float t;
};
[[nodiscard]] LineProximity nearLineSegmentPx(glm::vec2 p, glm::vec2 a, glm::vec2 b, float thicknessPx);

// glTF core order: baseColorFactor * baseColor * COLOR_0 (commutative). vertexColour is white with no COLOR_0 attribute.
[[nodiscard]] glm::vec3 resolveBaseColor(const Material& material, glm::vec2 uv, const glm::vec3& vertexColour,
                                          const PathTraceSettings& settings);

// heroChannel: the RGB channel a dispersive path committed to, setting the wavelength ior resolves at; nullopt keeps the d-line ior.
[[nodiscard]] BsdfParams resolveBsdfParams(const Material& material, glm::vec2 uv,
                                            const glm::vec3& vertexColour,
                                            const PathTraceSettings& settings,
                                            std::optional<int> heroChannel);

// Gram-Schmidt re-orthogonalized tangent frame, normal- and bump-mapped.
[[nodiscard]] ShadingFrame buildShadingFrame(const ShadingVertex& shading, const Material& material,
                                              const PathTraceSettings& settings);

[[nodiscard]] glm::vec3 geometricNormalOf(const ShadingTriangle& tri);

}  // namespace pathtracer::scene
