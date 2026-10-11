#pragma once

#include <array>
#include <optional>

#include <glm/glm.hpp>

#include "pathtracer/scene/sampler.h"

namespace pathtracer::scene {

// OpenPBR's base substrate at a hit, resolved from its inputs: metal and dielectric bases mixed by metalness (Surface v1.1.1).
struct BsdfParams {
    float metalness;            // base_metalness: the metal and dielectric bases mix linearly, so metal is weighted once
    float transmissionWeight;   // transmission_weight: the dielectric base's translucent share, the rest glossy-diffuse
    float roughness;            // specular_roughness, every microfacet lobe's; alpha = roughness^2, floored to avoid a delta lobe
    glm::vec3 metalF0;          // base_weight * base_color, the metal's normal-incidence reflectance
    // specular_color: the metal's F82 tint, and the dielectric's reflection tint for light arriving from above.
    glm::vec3 specularColor;
    float specularWeight;       // specular_weight, scaling the metal's Fresnel; for the dielectric it is already in ior
    float ior;                  // the dielectric's index ratio over the ambient medium, specular_weight-modulated and dispersed
    // EON rough-diffuse r in [0,1] (Portsmouth, Kutz, Hill 2025, JCGT 14(1)); 0 = Lambertian. base_diffuse_roughness.
    float diffuseRoughness;
    glm::vec3 diffuseRho;       // EON single-scattering albedo whose observed albedo is base_weight * base_color
    // transmission_color on the surface when transmission_depth is 0; white when the interior medium carries it instead.
    glm::vec3 transmissionTint;
};

// Orthonormal shading basis, columns (tangent, bitangent, normal): frame * local is world, world * frame its transpose, local.
using ShadingFrame = glm::mat3;

// Which lobe sampleBsdf drew from; path_tracer.cpp buckets the transport AOVs by it. Transmission is delta only below the smooth threshold.
enum class LobeType { Diffuse, SpecularReflection, Transmission };

struct BsdfSample {
    glm::vec3 wiLocal;            // sampled direction, local shading frame
    glm::vec3 throughputWeight;   // f(wi)*|cosThetaI| / pdf(wi)
    LobeType type;
    // The mixture density wiLocal was drawn from, equal to pdfBsdf's. Zero for the smooth-glass delta branch, which MIS keys on.
    float pdf;
};

// The BSDF's continuous lobes at one wi, split by transport type in one pass.
struct BsdfEval {
    glm::vec3 diffuse;
    glm::vec3 specular;
    glm::vec3 transmission;
    float pdf;
    [[nodiscard]] glm::vec3 total() const { return diffuse + specular + transmission; }
};

// Bilinear (roughness, eta) weights over four tabulated escape rows, a mu lookup away from any directional value at that vertex.
struct EscapeRow {
    std::array<int, 4> base;
    std::array<float, 4> weight;
};

// An escape row over the escape-deficit shape, with the reciprocal of its blended total.
struct MsTransmitRow {
    EscapeRow row;
    float scale;
};

// Per-strategy selection mass plus the Fresnel, albedo and escape state a vertex's lobes share. Built by makeBsdfClosure.
struct LobeProbabilities {
    float specular;
    float diffuse;
    float msReflect;    // opaque multiple-scattering reflection, drawn from kMsReflectDensity over the near hemisphere
    float msReflectTransmissive;  // a transmissive interface's reflected multiple scattering, drawn from reflectShape
    float transmit;     // delta refraction mass, 0 for a rough interface, whose refraction is the specular strategy's VNDF branch
    float msTransmit;   // multiple-scattering transmission, drawn from kMsTransmitDensity over the far hemisphere
    float etaI;
    float etaT;
    float dielectricWeight;      // 1 - metalness
    glm::vec3 dielectricTint;    // specular_color for reflection from above, white from below
    float metalWeight;           // metalness * specular_weight, scaling the metal's F82 Fresnel
    glm::vec3 metalF0;
    glm::vec3 metalK;            // F82's correction weight per channel, fit so F(1/7) = specular_color * Schlick(1/7)
    float albedoWo;              // E(mu_o, roughness), Fresnel-free single scattering
    float msReflectScaleWo;      // (1 - E(mu_o)) / (pi * (1 - Eavg)), the wo half of the Kulla-Conty lobe
    // Opaque multiple scattering per unit (1 - E(mu_o))(1 - E(mu_i)): the metal and glossy-diffuse Kulla-Conty tints, weighted.
    glm::vec3 msReflectTint;
    // The dielectric interface's escape row at the forward eta, its glossy-diffuse Kulla-Conty tint, and the diffuse coupling at wo.
    EscapeRow dielectricRow;
    float glossyFms;
    glm::vec3 diffuseCouplingWo;  // glossy-diffuse weight * (1 - E_spec(mu_o)) / (1 - mean E_spec), Kelemen's reciprocal form
    // EON's CLTC/uniform mixing weight at wo: a function of wo.z and diffuseRoughness alone, so its pow() is not a per-call cost.
    float eonUniformMix;
    // EON's clipped-LTC fit at wo: coefficients (a,b,c,d), the transposed LTC basis and its normalisation, none depending on wi.
    glm::vec4 eonLtcM;
    glm::mat3 eonLtcBasisT;
    float eonLtcS;
    // Multiple-scattering state for a transmissive interface: a facet reflects or refracts, so the escape is Fresnel-weighted.
    float escapeWo;         // R_ss(mu_o) + T_ss(mu_o), the Fresnel-weighted escaping fraction
    // Escape-deficit shape at the reciprocal eta (etaT/etaI); scale 0 where no transmitted multiple scattering exists.
    MsTransmitRow transmitShape;
    MsTransmitRow reflectShape;   // the same at the forward eta (etaI/etaT), for the reflected share whose wi stays in wo's medium
    float transmitShare;    // of the multiple-scattered energy, the fraction leaving refracted
    float etaSq;            // (etaI/etaT)^2, the radiance compression the transmit lobe must carry
    // (1 - metalness) * the translucent share (1 on the exiting side): how much transmission happens.
    float transmitWeight;
    float transmitPhysicalValue;  // the delta branch's (1-F(mu_o)) * transmitWeight energy
    // Refraction's value per unit (1-F) in the VNDF strategy's reflect/refract split; 0 where that strategy only reflects.
    float facetTransmit;
};

// One shading vertex's BSDF: everything derived from (params, woLocal), built once and then both sampled and evaluated at that vertex.
struct BsdfClosure {
    BsdfParams params;
    glm::vec3 wo;  // woLocal mirrored into the +z hemisphere, which every lobe below assumes
    float sign;    // the mirror that produced wo; wiLocal crosses it on the way in and the sampled wi on the way out
    float alpha;
    LobeProbabilities lobes;
};

// A vertex whose every strategy is massless has a zero BSDF: it can only emit, so neither NEE nor a continuation carries light from it.
[[nodiscard]] inline bool scatters(const BsdfClosure& closure) {
    const LobeProbabilities& lobes = closure.lobes;
    return lobes.specular + lobes.diffuse + lobes.msReflect + lobes.msReflectTransmissive + lobes.transmit + lobes.msTransmit > 0.0F;
}

// Builds the closure. Consumes no sampler dimensions, so where it is called relative to a draw does not move the sample stream.
[[nodiscard]] BsdfClosure makeBsdfClosure(const BsdfParams& params, const glm::vec3& woLocal);

// The closure forms, for a caller that both samples a continuation and evaluates toward a light at one vertex, as the integrator does.
[[nodiscard]] BsdfEval evaluateBsdfSplit(const BsdfClosure& closure, const glm::vec3& wiLocal);
[[nodiscard]] std::optional<BsdfSample> sampleBsdf(const BsdfClosure& closure, Sampler& sampler);

// Macro-surface Fresnel at cosTheta = dot(n, wo), entering: metal F82 and tinted dielectric mixed by metalness. For the Fresnel AOV.
[[nodiscard]] glm::vec3 fresnelAtViewAngle(const BsdfParams& params, float cosTheta);

// VNDF half-vector (Heitz 2018) through fresnelAtViewAngle at dot(wo, wh), per Walter 2007. One sample of E[F]: THE CALLER MUST AVERAGE.
[[nodiscard]] glm::vec3 fresnelAtMicrofacet(const BsdfParams& params, const glm::vec3& woLocal, glm::vec2 u);

// Cosine-weighted hemisphere direction about +z, pdf = cos(theta)/pi. The AO lane relies on the pdf cancelling the cosine (Miller 1994).
[[nodiscard]] glm::vec3 sampleCosineHemisphere(glm::vec2 u);

// Cosine-weighted average Fresnel, 2*int_0^1 F(mu)*mu dmu, the Kulla-Conty tint input. The metal's is closed-form.
[[nodiscard]] glm::vec3 metalFresnelAvg(const glm::vec3& f0, const glm::vec3& tint);
[[nodiscard]] float dielectricFresnelAvg(float ior);

// F82-split directional albedo E = F0*a + b - k*c, as (a, b, c), and its cosine-weighted mean.
[[nodiscard]] glm::vec3 directionalAlbedoSplit(float mu, float roughness);
[[nodiscard]] glm::vec3 averageAlbedoSplit(float roughness);

// The grid those two index. mu is uniform in sqrt(mu), so never assume k/(res-1).
[[nodiscard]] glm::ivec2 albedoGridRes();
[[nodiscard]] float albedoGridRoughness(float index);
[[nodiscard]] float albedoGridMu(float index);

// EON Appendix A: the rho whose normal-incidence directional albedo equals albedo under uniform light. Identity at r=0 and at albedo=1.
[[nodiscard]] glm::vec3 eonAlbedoInversion(const glm::vec3& albedo, float r);

// Representative wavelength per RGB channel (Adobe's OpenPBR reference); three discrete bands, so dispersion shows RGB banding.
inline constexpr glm::vec3 kRgbWavelengthsNm(620.0F, 540.0F, 450.0F);

// Cauchy n(lambda) from (n_d, V_d) per OpenPBR's dispersion; V_d = transmission_dispersion_abbe_number / transmission_dispersion_scale.
[[nodiscard]] float cauchyIor(float iorD, float abbe, float lambdaNm);

// Value and pdf of the continuous lobes at wiLocal, split by transport type; one call, so GGX, Fresnel and albedo are computed once.
[[nodiscard]] BsdfEval evaluateBsdfSplit(const BsdfParams& params, const glm::vec3& woLocal,
                                          const glm::vec3& wiLocal);

// evaluateBsdfSplit's total() and pdf as named entry points for the validators; anything needing both calls evaluateBsdfSplit once instead.
[[nodiscard]] float pdfBsdf(const BsdfParams& params, const glm::vec3& woLocal,
                            const glm::vec3& wiLocal);
[[nodiscard]] glm::vec3 evaluateBsdf(const BsdfParams& params, const glm::vec3& woLocal,
                                      const glm::vec3& wiLocal);

// Samples one lobe by Fresnel/energy probability, returning its throughput or nullopt.
[[nodiscard]] std::optional<BsdfSample> sampleBsdf(const BsdfParams& params,
                                                    const glm::vec3& woLocal, Sampler& sampler);

}  // namespace pathtracer::scene
