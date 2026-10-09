#pragma once

#include <glm/glm.hpp>

#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/material.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/shading_scene.h"

namespace pathtracer::scene {

// Primary-hit G-buffer sampling shared by the path tracer (tracePath bounce 0) and renderGBuffer, so both resolve materials alike.

// near: p within thicknessPx of segment [a,b], clamped to its extent, not the infinite line. t: the closest point's [0,1] parameter.
struct LineProximity {
    bool near;
    float t;
};
[[nodiscard]] LineProximity nearLineSegmentPx(glm::vec2 p, glm::vec2 a, glm::vec2 b, float thicknessPx);

// A primary hit's st footprint (Igehy 1999): dirFootprint's direction offsets moved onto the hit plane, then into uv.
[[nodiscard]] pathtracer::gfx::TextureFootprint primaryHitFootprint(const ShadingTriangle& triangle, const Ray& ray, float t,
                                                                    const glm::mat2x3& dirFootprint);

// The surface inputs at a shading point, filtered ones prefiltered over footprint; COLOR_0 tints base_color. Others keep defaults.
[[nodiscard]] OpenPbrInputs<Constant> resolveInputs(const Material& material, glm::vec2 uv,
                                                   const pathtracer::gfx::TextureFootprint& footprint, const glm::vec3& vertexColour);

// A positive dispersion scale on a transmissive base: the path must commit to one wavelength there.
[[nodiscard]] bool isDispersive(const OpenPbrInputs<Constant>& inputs);

// emission_luminance * emission_color at a hit: scene-linear radiance, one unit a luminance of 1 cd/m^2; both inputs filtered.
[[nodiscard]] glm::vec3 emittedRadiance(const Material& material, glm::vec2 uv, const pathtracer::gfx::TextureFootprint& footprint);

// Gram-Schmidt re-orthogonalized tangent frame, normal- and bump-mapped. triangle: the hit's corners, whose edges carry dP/duv.
[[nodiscard]] ShadingFrame buildShadingFrame(const ShadingTriangle& triangle, const ShadingVertex& shading, const Material& material);

[[nodiscard]] glm::vec3 geometricNormalOf(const ShadingTriangle& tri);

}  // namespace pathtracer::scene
