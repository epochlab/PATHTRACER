#pragma once

#include <glm/glm.hpp>

#include "pathtracer/scene/bsdf.h"

// OpenPBR's diffuse slab: EON (Portsmouth, Kutz, Hill 2025, JCGT 14(1)), ported from the paper's GLSL, never hand-derived.
namespace pathtracer::scene {

// The slab at wo for single-scattering albedo rho and base_diffuse_roughness r, its sampling state precomputed; wo.z >= 0.
[[nodiscard]] DiffuseSlab makeDiffuseSlab(const glm::vec3& rho, float r, const glm::vec3& wo);

// Cosine-weighted EON value at wi and the density of the slab's one-sample MIS sampler there; zero below the surface.
struct DiffuseEval {
    glm::vec3 value;
    float pdf;
};
[[nodiscard]] DiffuseEval evaluateDiffuse(const DiffuseSlab& slab, const glm::vec3& wo, const glm::vec3& wi);

// One-sample MIS between the clipped LTC and the uniform hemisphere (paper Sec. 4). Direction only; evaluateDiffuse owns the density.
[[nodiscard]] glm::vec3 sampleDiffuse(const DiffuseSlab& slab, glm::vec2 u);

}  // namespace pathtracer::scene
