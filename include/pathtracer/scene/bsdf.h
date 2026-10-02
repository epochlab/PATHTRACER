#pragma once

#include <array>
#include <optional>

#include <glm/glm.hpp>

#include "pathtracer/scene/sampler.h"

namespace pathtracer::scene {

// Resolved shading parameters at a hit point (textures already sampled by the caller).
struct BsdfParams {
    // OpenPBR base_color, the normal-incidence reflection colour under uniform light; resolveBsdfParams derives f0 and diffuseRho from it.
    glm::vec3 baseColor;
    float metallic;
    float roughness;  // perceptual; alpha = roughness^2, floored to avoid a delta lobe
    // Specular reflectance at normal incidence; also Gulbrandsen reflectivity r, clamped to [1e-4, 0.9999] at use.
    glm::vec3 f0;
    // Gulbrandsen 2014 edgetint g: 1 = white edge (Schlick's value), 0 = max dip. Inverts with f0 to a complex IOR. Inert at metallic=0.
    glm::vec3 edgeTint;
    float ior;  // dielectric IOR, non-metal lobes only
    float transmissionFactor;  // KHR_materials_transmission, 0 = opaque
    // EON rough-diffuse r in [0,1] (Portsmouth, Kutz, Hill 2025, JCGT 14(1)); 0 = Lambertian. Not `roughness`, which drives specular.
    float diffuseRoughness;
    // EON single-scattering albedo rho, the diffuse lobe's input, not the authored colour.
    glm::vec3 diffuseRho;
    // The transmission lobe's only tint (OpenPBR/Arnold): carried by Beer-Lambert at transmissionDepth > 0, applied per crossing at 0.
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

// Bilinear (roughness, eta) weights over four tabulated rows of the escape-deficit shape, with the reciprocal of their blended total.
struct MsTransmitRow {
    std::array<int, 4> base;
    std::array<float, 4> weight;
    float scale;
};

// Per-strategy selection mass plus the Fresnel, albedo and escape state a vertex's lobes share. Built by makeBsdfClosure.
struct LobeProbabilities {
    float specular;
    float diffuse;
    float msReflect;    // multiple-scattering reflection, drawn from kMsReflectDensity over the near hemisphere
    float msReflectTransmissive;  // a transmissive interface's reflected multiple scattering, drawn from reflectShape
    float transmit;     // delta refraction mass, 0 for a rough interface, whose refraction is the specular strategy's VNDF branch
    float msTransmit;   // multiple-scattering transmission, drawn from kMsTransmitDensity over the far hemisphere
    float etaI;
    float etaT;
    float diffuseKd;              // evaluateDiffuseLobe's wo-side energy factor, 0 on the exiting side
    float transmitPhysicalValue;  // transmission's true (1-F)*t energy fraction -- see below
    // Energy-compensation state, hoisted so the wo-side table lookups happen once per evaluation, not per lobe call.
    float albedoWo;         // E(mu_o, roughness), Fresnel-free
    float albedoAvg;        // Eavg(roughness)
    float coatF0;           // dielectric f0 implied by ior, for the diffuse coupling
    // The coat's own cosine-mean Fresnel, separate from the metallic-blended fresnelAvg below, which is wrong for the coat.
    float coatFresnelAvg;
    glm::vec3 fresnelAvg;
    // Kulla-Conty tint per channel, a pure function of fresnelAvg and albedoAvg, so the reflection lobe reads it per evaluation.
    glm::vec3 multiScatterFms;
    // EON's CLTC/uniform mixing weight at wo: a function of wo.z and diffuseRoughness alone, so its pow() is not a per-call cost.
    float eonUniformMix;
    // EON's clipped-LTC fit at wo: coefficients (a,b,c,d), the transposed LTC basis and its normalisation, none depending on wi.
    glm::vec4 eonLtcM;
    glm::mat3 eonLtcBasisT;
    float eonLtcS;
    // Complex IOR inverted from (f0, edgeTint) once per evaluation. Index-matched (1, 0) at metallic==0, where no consumer reads them.
    glm::vec3 conductorN;
    glm::vec3 conductorK;
    // Multiple-scattering state for a transmissive interface: a facet reflects or refracts, so the escape is Fresnel-weighted.
    float escapeWo;         // R_ss(mu_o) + T_ss(mu_o), the Fresnel-weighted escaping fraction
    // Escape-deficit shape at the reciprocal eta (etaT/etaI); scale 0 where no transmitted multiple scattering exists.
    MsTransmitRow transmitShape;
    MsTransmitRow reflectShape;   // the same at the forward eta (etaI/etaT), for the reflected share whose wi stays in wo's medium
    float transmitShare;    // of the multiple-scattered energy, the fraction leaving refracted
    float etaSq;            // (etaI/etaT)^2, the radiance compression the transmit lobe must carry
    // effectiveTransmission*(1-metallic): how much transmission happens. Scales single-scatter and multiple-scattering transmit alike.
    float transmitWeight;
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

// Builds the closure. Consumes no sampler dimensions, so where it is called relative to a draw does not move the sample stream.
[[nodiscard]] BsdfClosure makeBsdfClosure(const BsdfParams& params, const glm::vec3& woLocal);

// The closure forms, for a caller that both samples a continuation and evaluates toward a light at one vertex, as the integrator does.
[[nodiscard]] BsdfEval evaluateBsdfSplit(const BsdfClosure& closure, const glm::vec3& wiLocal);
[[nodiscard]] std::optional<BsdfSample> sampleBsdf(const BsdfClosure& closure, Sampler& sampler);

// Macro-surface Fresnel at cosTheta = dot(n, wo), dielectric and conductor mixed by metallic. Exists so the Fresnel AOV shows the curve.
[[nodiscard]] glm::vec3 fresnelAtViewAngle(const BsdfParams& params, float cosTheta);

// VNDF half-vector (Heitz 2018) through fresnelAtViewAngle at dot(wo, wh), per Walter 2007. One sample of E[F]: THE CALLER MUST AVERAGE.
[[nodiscard]] glm::vec3 fresnelAtMicrofacet(const BsdfParams& params, const glm::vec3& woLocal, glm::vec2 u);

// Cosine-weighted hemisphere direction about +z, pdf = cos(theta)/pi. The AO lane relies on the pdf cancelling the cosine (Miller 1994).
[[nodiscard]] glm::vec3 sampleCosineHemisphere(glm::vec2 u);

// Cosine-weighted average Fresnel, 2*int_0^1 F(mu)*mu dmu, the Kulla-Conty tint.
[[nodiscard]] glm::vec3 conductorFresnelAvg(const glm::vec3& n, const glm::vec3& k);
[[nodiscard]] float dielectricFresnelAvg(float ior);

// Schlick-split directional albedo E(mu, roughness) = a+b and its mean Eavg.
[[nodiscard]] glm::vec2 directionalAlbedoSplit(float mu, float roughness);
[[nodiscard]] glm::vec2 averageAlbedoSplit(float roughness);

// The grid those two index. mu is uniform in sqrt(mu), so never assume k/(res-1).
[[nodiscard]] glm::ivec2 albedoGridRes();
[[nodiscard]] float albedoGridRoughness(float index);
[[nodiscard]] float albedoGridMu(float index);

// EON Appendix A: the rho whose normal-incidence directional albedo equals albedo under uniform light. Identity at r=0 and at albedo=1.
[[nodiscard]] glm::vec3 eonAlbedoInversion(const glm::vec3& albedo, float r);

// Representative wavelength per RGB channel (Adobe's OpenPBR reference); three discrete bands, so dispersion shows RGB banding.
inline constexpr glm::vec3 kRgbWavelengthsNm(620.0F, 540.0F, 450.0F);

// Cauchy n(lambda) from an authored (ior at d line, Abbe V_d), per KHR_materials_dispersion.
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
