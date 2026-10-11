#pragma once

#include <optional>

#include <glm/glm.hpp>

#include "pathtracer/scene/bsdf.h"

// GGX primitives and OpenPBR's two microfacet slabs, conductor and dielectric interface, each in its own frame with wo.z >= 0.
namespace pathtracer::scene {

// Perceptual roughness to GGX alpha, OpenPBR's alpha = r^2: every consumer reads this one mapping.
[[nodiscard]] float alphaForRoughness(float roughness);

// GGX's denominator pi*d^2 falls to pi*alpha^4 at its peak: below FLT_MIN it underflows and D overflows, so the lobe is a delta.
[[nodiscard]] bool isSmooth(float alpha);

// Heitz 2018 VNDF sampling; wo.z >= 0.
[[nodiscard]] glm::vec3 sampleGGXVNDF(const glm::vec3& wo, float alpha, glm::vec2 u);

// The metal's F82-tint Fresnel (OpenPBR; Hoffman 2023) at cosTheta in [0,1], k from the slab.
[[nodiscard]] glm::vec3 fresnelF82(float cosTheta, const glm::vec3& f0, const glm::vec3& k);

// The conductor slab at wo: f0 = base_weight * base_color, specular_color the F82 tint, specular_weight the Fresnel scale.
[[nodiscard]] ConductorSlab makeConductorSlab(float roughness, const glm::vec3& f0, const glm::vec3& tint, float scale, float muO);

// The dielectric interface at wo between etaI (wo's side) and etaT; refractWeight and the tints as DielectricSlab documents.
[[nodiscard]] DielectricSlab makeDielectricSlab(float roughness, float etaI, float etaT, const glm::vec3& tint, float refractWeight,
                                                const glm::vec3& transmitTint, float muO);

// The interface's reflection albedo at wo, single plus multiple scattering, untinted: E_spec of OpenPBR's albedo scaling.
[[nodiscard]] inline float reflectAlbedo(const DielectricSlab& slab) { return slab.reflectSingle + slab.multiReflect; }

[[nodiscard]] inline bool isIndexMatched(const DielectricSlab& slab) { return slab.etaI == slab.etaT; }

// A slab's cosine-weighted value at wi and the density of each of its techniques there, unweighted by selection mass.
struct ConductorEval {
    glm::vec3 value;
    float pdfSingle;
    float pdfMulti;
};

struct DielectricEval {
    glm::vec3 reflect;
    glm::vec3 transmit;
    float pdfSingle;
    float pdfMultiReflect;
    float pdfMultiTransmit;
};

[[nodiscard]] ConductorEval evaluateConductor(const ConductorSlab& slab, const glm::vec3& wo, const glm::vec3& wi);
[[nodiscard]] DielectricEval evaluateDielectric(const DielectricSlab& slab, const glm::vec3& wo, const glm::vec3& wi);

// The rough conductor's VNDF reflection and its Kulla-Conty lobe's direction; nullopt below the surface.
[[nodiscard]] std::optional<glm::vec3> sampleConductorSingle(const ConductorSlab& slab, const glm::vec3& wo, glm::vec2 u);
[[nodiscard]] glm::vec3 sampleConductorMulti(const ConductorSlab& slab, glm::vec2 u);

// The rough interface's VNDF facet, reflected or refracted by uSplit against the facet's Fresnel split; nullopt on a wrong-side draw.
struct InterfaceSample {
    glm::vec3 wi;
    bool refracted;
};
[[nodiscard]] std::optional<InterfaceSample> sampleDielectricSingle(const DielectricSlab& slab, const glm::vec3& wo, glm::vec2 u,
                                                                    float uSplit);

// Escape-table multiple scattering's direction about +z; the caller mirrors z for the refracted lobe.
[[nodiscard]] glm::vec3 sampleEscapeShape(const MsTransmitRow& shape, glm::vec2 u);

}  // namespace pathtracer::scene
