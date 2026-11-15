#pragma once

#include <cmath>
#include <optional>

#include <glm/glm.hpp>

#include "pathtracer/scene/bsdf.h"
#include "shading_math.h"

// GGX primitives and OpenPBR's two microfacet slabs, conductor and dielectric interface, each in its own frame with wo.z >= 0.
namespace pathtracer::scene {

// OpenPBR's (alpha_t, alpha_b) = r^2 sqrt(2/(1 + (1-a)^2)) (1, 1-a), RMS r^2: every consumer reads this one mapping.
[[nodiscard]] glm::vec2 alphaForRoughness(float roughness, float anisotropy);

// GGX's denominator pi*d^2 falls to pi*alpha^4 at its peak: below FLT_MIN it underflows and D overflows, so the lobe is a delta.
[[nodiscard]] bool isSmooth(float alpha);

// GGX D, cancellation-free (Filament 4.4.2) and anisotropic (Burley 2012): ax ay/(pi d^2), d = ax ay hz^2 + (ay/ax) hx^2 + (ax/ay) hy^2.
[[nodiscard]] inline float distributionGGX(const glm::vec3& h, const glm::vec2& alpha) {
    const float d = (alpha.x * alpha.y * h.z * h.z) + ((alpha.y / alpha.x) * h.x * h.x) + ((alpha.x / alpha.y) * h.y * h.y);
    return (alpha.x * alpha.y) / (kPi * d * d);
}

// |(ax wx, ay wy, wz)|, the Smith Lambda radical scaled by cosine (Heitz 2014): 1 + Lambda(w) = (|wz| + radical)/(2|wz|), finite at wz = 0.
[[nodiscard]] inline float smithRadical(const glm::vec3& w, const glm::vec2& alpha) {
    return std::sqrt((alpha.x * alpha.x * w.x * w.x) + (alpha.y * alpha.y * w.y * w.y) + (w.z * w.z));
}

// Visible normals seen from w anywhere off the downward pole, a spherical cap in the stretched configuration (Dupuy & Benyoub 2023).
[[nodiscard]] glm::vec3 sampleVisibleNormal(const glm::vec3& w, const glm::vec2& alpha, glm::vec2 u);

// The conductor slab at wo: f0 = base_weight * base_color, specular_color the F82 tint, specular_weight the Fresnel scale.
[[nodiscard]] ConductorSlab makeConductorSlab(float roughness, float anisotropy, const glm::vec3& f0, const glm::vec3& tint, float scale,
                                              const FilmLayer& film, const glm::vec3& wo);

// A slab's Fresnel at a facet cosine: the conductor's F82, the interface's dielectric, each mixed with the film's by its weight.
[[nodiscard]] glm::vec3 conductorFresnel(const ConductorSlab& slab, float cosTheta);
// The interface's from wo's side, or from below it: the Fresnel ratio inverted and the film met from its other face.
[[nodiscard]] glm::vec3 interfaceFresnel(const DielectricSlab& slab, float cosTheta, bool below = false);

// One dielectric interface: geometry ratio etaI/etaT bends refraction, the Fresnel ratio sets R and T (OpenPBR's coat moves it alone).
struct InterfaceInputs {
    float roughness;
    float anisotropy;
    float etaI;
    float etaT;
    float fresnelEtaI;
    float fresnelEtaT;
    glm::vec3 tint;
    float refractWeight;
    glm::vec3 transmitTint;
    FilmLayer film;
    float baseIor;   // n_b, the film's substrate from outside or its incident medium from inside
    bool fromBase;
};

// The dielectric interface at wo; refractWeight and the tints as DielectricSlab documents.
[[nodiscard]] DielectricSlab makeDielectricSlab(const InterfaceInputs& interface, const glm::vec3& wo);

// An interface's untinted reflection albedo at w, single plus multiple scattering, from the side etaI faces.
[[nodiscard]] float reflectionAlbedo(float roughness, float anisotropy, float etaI, float etaT, const glm::vec3& w);

// The interface's reflection albedo at wo, single plus multiple scattering, untinted: E_spec of OpenPBR's albedo scaling.
[[nodiscard]] inline glm::vec3 reflectAlbedo(const DielectricSlab& slab) { return slab.reflectSingle + slab.multiReflect; }

// Fresnel-matched and unfilmed: the interface reflects nothing at any angle.
[[nodiscard]] inline bool isIndexMatched(const DielectricSlab& slab) {
    return slab.fresnelEtaI == slab.fresnelEtaT && slab.film.weight == 0.0F;
}

// A thin wall's albedos at cosine mu, every internal bounce summed per polarisation: exact for a smooth wall (Born & Wolf 1.6).
struct SheetLadder {
    glm::vec3 reflect;   // R', untinted
    glm::vec3 transmit;  // T', transmission_color along the refracted path included
};
[[nodiscard]] SheetLadder sheetLadder(const SheetInterfaces& sheet, float mu);

// The thin wall's translucent share at wo, weight (1 - M) T, its lobe the white conductor at the base roughness.
[[nodiscard]] SheetSlab makeSheetSlab(const SheetInterfaces& interfaces, float weight, float roughness, float anisotropy, const glm::vec3& wo);

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
    float lobePdfSingle;  // pdfSingle given its reflect or refract branch: the VNDF lobe's own density, wi and h one-to-one
};

[[nodiscard]] ConductorEval evaluateConductor(const ConductorSlab& slab, const glm::vec3& wo, const glm::vec3& wi);
[[nodiscard]] DielectricEval evaluateDielectric(const DielectricSlab& slab, const glm::vec3& wo, const glm::vec3& wi);

// The rough conductor's VNDF reflection and its Kulla-Conty lobe's direction; nullopt below the surface.
[[nodiscard]] std::optional<glm::vec3> sampleConductorSingle(const ConductorSlab& slab, const glm::vec3& wo, glm::vec2 u);
[[nodiscard]] glm::vec3 sampleConductorMulti(const ConductorSlab& slab, glm::vec2 u);

// The odds the interface reflects about a facet of this Fresnel (Walter 2007 5.3), by its tinted and refracting energies; 1 when opaque.
[[nodiscard]] float facetReflectProbability(const DielectricSlab& slab, const glm::vec3& fresnel);

// The rough interface's VNDF facet, reflected or refracted by uSplit against the facet's Fresnel split; nullopt on a wrong-side draw.
struct InterfaceSample {
    glm::vec3 wi;
    bool refracted;
};
[[nodiscard]] std::optional<InterfaceSample> sampleDielectricSingle(const DielectricSlab& slab, const glm::vec3& wo, glm::vec2 u,
                                                                    float uSplit);

}  // namespace pathtracer::scene
