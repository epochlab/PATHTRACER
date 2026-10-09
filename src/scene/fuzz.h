#pragma once

#include <glm/glm.hpp>

#include "pathtracer/scene/bsdf.h"

// OpenPBR's fuzz: the LTC fit to a volumetric SGGX sheen (Zeltner, Burley, Chiang 2022), ported from the authors' reference code.
namespace pathtracer::scene {

// The slab at wo, in the fuzz frame with wo.z >= 0, for fuzz_color and fuzz_roughness.
[[nodiscard]] FuzzSlab makeFuzzSlab(const glm::vec3& color, float roughness, const glm::vec3& wo);

// Cosine-weighted value color * E_fuzz * D_ltc(wi) and the LTC's own density there: the lobe is sampled exactly.
struct FuzzEval {
    glm::vec3 value;
    float pdf;
};
[[nodiscard]] FuzzEval evaluateFuzz(const FuzzSlab& slab, const glm::vec3& wi);

// A direction drawn from the LTC, in the fuzz frame.
[[nodiscard]] glm::vec3 sampleFuzz(const FuzzSlab& slab, glm::vec2 u);

}  // namespace pathtracer::scene
