#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <glm/glm.hpp>

#include "pathtracer/scene/openpbr.h"
#include "pathtracer/scene/sampler.h"

namespace pathtracer::scene {

// Orthonormal shading basis, columns (tangent, bitangent, normal): frame * local is world, world * frame its transpose, local.
using ShadingFrame = glm::mat3;

// Which lobe sampleBsdf drew from; path_tracer.cpp buckets the transport AOVs by it.
enum class LobeType { Diffuse, SpecularReflection, Transmission };

struct BsdfSample {
    glm::vec3 wiLocal;            // sampled direction, local shading frame
    glm::vec3 throughputWeight;   // f(wi)*|cosThetaI| / pdf(wi)
    glm::vec3 transmitWeight;     // its transmission lobes' share: what a direction crossing the geometric surface carries
    LobeType type;
    // The mixture density wiLocal was drawn from, equal to pdfBsdf's; zero for a delta branch.
    float pdf;
    bool delta;  // a smooth reflection or refraction: no density for NEE to share, so MIS gives its continuation full weight
    bool passThrough = false;  // a smooth thin wall's undeviated transmission along -wo: a null vertex, the ray continues unchanged
};

// The BSDF's continuous lobes at one wi, cosine-weighted, f*|cos| about each lobe's own normal (Mitsuba 3's convention), by type.
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

// The four (roughness, anisotropy) rows of the anisotropic tables about one slab, each row's node alphas setting its azimuth axis.
struct AnisoRows {
    std::array<int, 4> row;
    std::array<float, 4> weight;
    std::array<glm::vec2, 4> alpha;
};

// An escape row over the escape-deficit shape, with the reciprocal of its blended total; scale 0 where the shape holds no energy.
struct EscapeShape {
    EscapeRow row;
    float scale;
};

// OpenPBR's thin film over a base slab's Fresnel: weight 0 where absent or zero-thick; ambient-side index 1, or the coat's n_c with C.
struct FilmLayer {
    float weight = 0.0F;   // thin_film_weight, mixing the filmed and bare Fresnel
    float thicknessNm;
    float ior;             // thin_film_ior
    float coatWeight;      // the medium on the film's ambient side is a statistical mix of the ambient and the coat
    float coatIor;
};

// OpenPBR's metal slab at one wo: GGX with the F82-tint Fresnel (Hoffman 2023) and its Kulla-Conty lobe (Kulla & Conty 2017).
struct ConductorSlab {
    float roughness;
    float anisotropy;      // specular_roughness_anisotropy, selecting the 4D tables over the isotropic ones
    AnisoRows rows;        // where anisotropy > 0, the tables' rows about (roughness, anisotropy), for the deficit at wi
    glm::vec2 alpha;       // (alpha_t, alpha_b) along the frame's tangent and bitangent
    glm::vec3 f0;          // base_weight * base_color
    glm::vec3 k;           // F82's correction weight, fit so F(1/7) = specular_color * Schlick(1/7)
    float scale;           // specular_weight
    glm::vec3 msTint;      // Kulla-Conty Fms per channel, the share of the single-scattering deficit surviving repeated bounces
    float msScaleWo;       // (1 - E(mu_o)) / (pi * (1 - Eavg)), the wo half of the Kulla-Conty lobe
    float singleEnergy;    // single-scattering albedo at wo, channel mean, the selection mass before weighting
    float multiEnergy;
    glm::vec3 albedo;      // E(mu_o) per channel, single plus multiple scattering
    FilmLayer film;
    glm::vec3 filmEta;     // the conductor's complex index under the film, Gulbrandsen 2014 from base_color and specular_color
    glm::vec3 filmKappa;
};

// A GGX dielectric interface at one wo (OpenPBR's specular slab): reflection, refraction and their escape-table multiple scattering.
struct DielectricSlab {
    glm::vec2 alpha;       // (alpha_t, alpha_b) along the frame's tangent and bitangent
    float etaI;            // the media either side, wo's first: refraction bends by etaI/etaT
    float etaT;
    float fresnelEtaI;     // the ratio Fresnel and the escape tables read; OpenPBR's coat moves it toward n_b/n_c, not the bend
    float fresnelEtaT;
    bool deltaRefraction;  // smooth, or index-matched media: refraction is undeviated or Snell's, never a lobe
    glm::vec3 tint;          // specular_color on reflection from above, white from inside
    float tintMean;
    // Of the light the interface transmits, the share leaving refracted: transmission_weight from above, 1 from inside, the rest diffuse.
    float refractWeight;
    glm::vec3 transmitTint;  // transmission_color on the surface when transmission_depth is 0; white when the medium carries it
    EscapeRow row;           // forward eta etaI/etaT
    EscapeShape reflectShape;   // escape-deficit shape at the forward eta, sampling the multiple scattering leaving on wo's side
    EscapeShape transmitShape;  // the same at the reciprocal eta, for multiple scattering leaving refracted
    glm::vec3 reflectSingle;   // R_ss(mu_o), the film's through the kernel rule; F(mu_o) when smooth
    glm::vec3 transmitSingle;  // T_ss(mu_o), a selection mass: 1 - F(mu_o) when smooth; undeviated, plus transmitted multiple scattering
    float multiReflect;      // the escape deficit at mu_o times the random walks' reflected share of 2+ bounce energy
    float multiReflectScaleWo;  // multiReflect / (pi * (1 - Eavg)), the wo half of the reflected multiple-scattering lobe
    float multiTransmit;     // the same deficit's transmitted share
    float etaSq;             // (etaI/etaT)^2, the radiance compression refraction carries
    FilmLayer film;
    float baseIor;           // the base's own index n_b, the film's substrate from outside and its incident medium from inside
    bool fromBase;           // wo inside the base: the film is met from below
};

// OpenPBR's diffuse slab at one wo: EON (Portsmouth, Kutz, Hill 2025) with its clipped-LTC/uniform sampling state.
struct DiffuseSlab {
    glm::vec3 rho;       // base_weight * base_color: in albedo scaling OpenPBR states rho = C, no inversion to an observed albedo
    float roughness;     // base_diffuse_roughness
    float uniformMix;    // the uniform hemisphere's share of the one-sample MIS, a function of mu_o and roughness alone
    glm::vec4 ltcM;      // clipped-LTC coefficients (a,b,c,d)
    glm::mat3 ltcBasisT; // the transposed LTC basis aligned to wo's azimuth
    float ltcS;          // the clipped LTC's normalisation
};

// A thin wall's two parallel faces (OpenPBR's thin-walled translucent base): what its ladder of internal reflections depends on.
struct SheetInterfaces {
    glm::vec3 tint;    // specular_color, on reflection from the wall
    glm::vec3 color;   // transmission_color, one crossing at normal incidence; transmission_depth has no meaning without an interior
    float fresnelIor;  // the facing face's Fresnel ratio, the coat's shift included
    float ior;         // the far face's ratio and the bend, specular_ior modulated by specular_weight
    FilmLayer film;    // on the facing face
};

// The translucent share of a thin wall at one wo: its ladder's albedos over a white GGX lobe, mirrored through the wall to transmit.
struct SheetSlab {
    SheetInterfaces interfaces;
    float weight;          // (1 - base_metalness) * transmission_weight
    ConductorSlab lobe;    // f0 = tint = 1: single plus Kulla-Conty scattering integrate to 1 at every mu_o
    glm::vec3 reflect;     // weight * specular_color * R'(mu_o)
    glm::vec3 transmit;    // weight * T'(mu_o)
    float reflectShare;    // mean(reflect) / mean(reflect + transmit): the side a draw from the lobe takes
};

// OpenPBR's fuzz slab at one wo: the volumetric SGGX sheen's LTC fit (Zeltner, Burley, Chiang 2022), in the fuzz frame.
struct FuzzSlab {
    glm::vec3 color;     // fuzz_color, tinting the fuzz's own reflection alone
    float aInv;          // the LTC's inverse-matrix coefficients at (fuzz_roughness, mu_o)
    float bInv;
    float albedo;        // E_fuzz(mu_o), the sheen's directional albedo
    glm::mat3 basisT;    // to the LTC frame, wo's azimuth on +x
};

// One slab's sampling strategy; the closure's selection masses are indexed by it, so the order is the sample stream's contract.
enum class Technique : std::uint8_t {
    Fuzz,                    // the sheen LTC, sampled exactly
    CoatSingle,              // the coat's VNDF reflection in its own frame, or its mirror when smooth
    CoatMulti,               // the coat's reflected multiple scattering
    MetalSingle,             // VNDF reflection, or the mirror when smooth
    MetalMulti,              // the Kulla-Conty lobe, drawn from kMsReflectDensity
    DielectricSingle,        // VNDF reflection or refraction split by the facet's Fresnel (Walter 2007), or the mirror when smooth
    DielectricRefract,       // smooth or index-matched refraction, a delta
    DielectricMultiReflect,  // escape-table multiple scattering leaving on wo's side
    DielectricMultiTransmit, // escape-table multiple scattering leaving refracted
    Diffuse,                 // EON
    SheetSingle,             // a thin wall's VNDF lobe, reflected or mirrored through it by side; the mirror or -wo when smooth
    SheetMulti,              // its Kulla-Conty lobe, reflected or mirrored through the wall
    ThinSubsurface,          // a thin wall's subsurface, an albedo-1 EON reflected or mirrored through it by its anisotropy
    Count,
};
inline constexpr std::size_t kTechniqueCount = static_cast<std::size_t>(Technique::Count);

// One vertex's OpenPBR surface, layer(coat, mix(dielectric, metal, M)); a slab is read only where built, so unbuilt ones stay unwritten.
struct BsdfClosure {
    glm::vec3 wo;  // woLocal mirrored into the +z hemisphere, which every slab assumes
    float sign;    // the mirror that produced wo; wiLocal crosses it on the way in and the sampled wi on the way out
    bool exiting;  // wo inside a transmissive base: only its interface faces the ray, and only transmission crosses coat and fuzz
    bool thinWalled;  // geometry_thin_walled: no interior, so the wall is the same from either side and never exits
    glm::vec2 baseAlpha;  // the base slabs' one (alpha_t, alpha_b), specular_roughness after the coat's roughening
    // OpenPBR's fuzz over the coated base: fuzz_weight, its slab in its own frame, and 1 - F E_fuzz(wo), what passes beneath it.
    float fuzzWeight = 0.0F;
    glm::mat3 toFuzz;
    FuzzSlab fuzz;             // built where fuzzWeight > 0 and wo is outside
    float fuzzRoughness;
    float belowFuzz;
    // OpenPBR's coat: coat_weight of the surface, a dielectric slab in the frame toCoat takes the mirrored base frame to, x its tangent.
    float coatWeight = 0.0F;
    glm::mat3 toCoat;
    DielectricSlab coat;       // built where coatWeight > 0 and wo is outside
    glm::vec3 coatColor;       // coat_color, the coat's squared normal-incidence transmittance
    float coatIor;
    float coatRoughness;
    float coatAnisotropy;
    // The base under the coat: reflection weighs baseBare + baseUnder * T(mu_i) under Delta, refraction transmitUnder under 1/(1 - E_b K).
    glm::vec3 baseBare;
    glm::vec3 baseUnder;
    glm::vec3 transmitUnder;
    float metalWeight;        // base_metalness
    float dielectricWeight;   // the dielectric interface's share: 1 - base_metalness, times 1 - transmission_weight when thin-walled
    // OpenPBR's albedo scaling of the diffuse under the interface: (1 - M)(1 - T)(1 - S)(1 - E_spec(mu_o)), zero from inside.
    glm::vec3 diffuseWeight = glm::vec3(0.0F);
    ConductorSlab metal;        // built where metalWeight > 0
    DielectricSlab dielectric;  // built where dielectricWeight > 0
    DiffuseSlab diffuse;        // built where its selection mass is positive
    SheetSlab sheet;            // built where the thin wall's translucent share is positive
    // The thin wall's subsurface under the interface, (1 - M)(1 - T) S (1 - E_spec(mu_o)) subsurface_color, reflected (1 - g)/2.
    glm::vec3 subsurfaceWeight;
    float subsurfaceReflectShare;
    DiffuseSlab subsurface;     // rho = 1, built where its selection mass is positive
    bool thinTransmission = false;  // a thin wall's sheet or subsurface transmits: light behind it reaches wo
    std::array<float, kTechniqueCount> mass{};  // selection probabilities: each technique's energy at wo over the total, or all zero
};

// Resolved inputs, wo, a hero RGB band dispersing specular_ior and the coat's normal and tangent in the base frame build it. No draws.
[[nodiscard]] BsdfClosure makeBsdfClosure(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal,
                                          std::optional<int> heroChannel = std::nullopt,
                                          const glm::vec3& coatNormalLocal = glm::vec3(0.0F, 0.0F, 1.0F),
                                          const glm::vec3& coatTangentLocal = glm::vec3(1.0F, 0.0F, 0.0F));

// A vertex whose every strategy is massless has a zero BSDF: it can only emit, so neither NEE nor a continuation carries light from it.
[[nodiscard]] inline bool scatters(const BsdfClosure& closure) {
    for (const float mass : closure.mass) {
        if (mass > 0.0F) {
            return true;
        }
    }
    return false;
}

// Light behind the vertex reaches wo only through a refracting interface or a thin wall, so NEE samples the far side only here.
[[nodiscard]] inline bool transmits(const BsdfClosure& closure) {
    return (closure.dielectricWeight > 0.0F && closure.dielectric.refractWeight > 0.0F) || closure.thinTransmission;
}

// A smooth thin wall's undeviated transmission toward -wo, through fuzz, coat and sheet; zero for any other closure.
[[nodiscard]] glm::vec3 passThrough(const BsdfClosure& closure);

// Schussler et al. 2017's default model (eq. 23) in the perturbed frame: facet +z and a mirror wall over the interpolated normal.
struct NormalMappedBsdf {
    bool perturbed = false;  // no normal or bump input: facet alone is the BSDF and every call forwards to it unchanged
    BsdfClosure facet;       // at wo, the walk meeting the perturbed facet first; built where lambdaP > 0
    BsdfClosure viaWall;     // at wo mirrored by the wall, the walk meeting the wall first; built where lambdaT > 0
    glm::vec3 geometric;     // omega_g
    glm::vec3 wall;          // omega_t: perpendicular to omega_g, in the plane of omega_g and +z, against the facet's tilt
    float cosP;              // <omega_p, omega_g>
    float sinP;
    float side;              // sign <wo, omega_g>: seen from below, the microsurface is the point reflection of the one above
    float lambdaP = 0.0F;    // the facets' shares of wo's projected area (eq. 9)
    float lambdaT = 0.0F;
};

// The closure at wo with geometricLocal, the interpolated normal in the perturbed frame, where a normal or bump input bends it.
[[nodiscard]] NormalMappedBsdf makeNormalMappedBsdf(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal,
                                                    std::optional<glm::vec3> geometricLocal, std::optional<int> heroChannel,
                                                    const glm::vec3& coatNormalLocal, const glm::vec3& coatTangentLocal);

// The continuous lobes at wiLocal, split by transport type; one call, so GGX, Fresnel and table reads are computed once.
[[nodiscard]] BsdfEval evaluateBsdfSplit(const BsdfClosure& closure, const glm::vec3& wiLocal);

// Selects a technique by its mass and draws from it; the throughput divides by the whole mixture's density (one-sample MIS).
[[nodiscard]] std::optional<BsdfSample> sampleBsdf(const BsdfClosure& closure, Sampler& sampler);

// The microsurface's forms: eq. 23 evaluated exactly, its walk sampled with the walk's exact density, components by omega_g's side.
[[nodiscard]] bool scatters(const NormalMappedBsdf& bsdf);
[[nodiscard]] bool transmits(const NormalMappedBsdf& bsdf);
[[nodiscard]] glm::vec3 passThrough(const NormalMappedBsdf& bsdf);
[[nodiscard]] BsdfEval evaluateBsdfSplit(const NormalMappedBsdf& bsdf, const glm::vec3& wiLocal);
[[nodiscard]] std::optional<BsdfSample> sampleBsdf(const NormalMappedBsdf& bsdf, Sampler& sampler);
[[nodiscard]] glm::vec3 fresnelAtMicrofacet(const NormalMappedBsdf& bsdf, glm::vec2 u);

// The slabs' reflected Fresnel at cosTheta = dot(h, wo), each at its layer and mix weight: coat, metal F82, tinted dielectric.
[[nodiscard]] glm::vec3 fresnelAtViewAngle(const BsdfClosure& closure, float cosTheta);

// fresnelAtViewAngle at one VNDF half-vector (Heitz 2018), per Walter 2007. One sample of E[F]: THE CALLER MUST AVERAGE.
[[nodiscard]] glm::vec3 fresnelAtMicrofacet(const BsdfClosure& closure, glm::vec2 u);

// The closure forms for validators with one direction to answer for: one closure, used once.
[[nodiscard]] BsdfEval evaluateBsdfSplit(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, const glm::vec3& wiLocal);
[[nodiscard]] float pdfBsdf(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, const glm::vec3& wiLocal);
[[nodiscard]] glm::vec3 evaluateBsdf(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, const glm::vec3& wiLocal);
[[nodiscard]] std::optional<BsdfSample> sampleBsdf(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, Sampler& sampler);

// The dielectric's index ratio: heroChannel disperses specular_ior, specular_weight modulates it (OpenPBR's epsilon, eta').
[[nodiscard]] float modulatedIor(const OpenPbrInputs<Constant>& inputs, std::optional<int> heroChannel);

// E_fuzz(mu) at fuzz_roughness, the sheen's directional albedo: what the fuzz reflects and so withholds from the layers beneath.
[[nodiscard]] float fuzzAlbedo(float roughness, float mu);

// The coat's one-pass transmittance at coat cosine mu: coat_color^(1/(2 mu_t)), mu_t refracted into the coat (OpenPBR).
[[nodiscard]] glm::vec3 coatTransmittance(const glm::vec3& coatColor, float coatIor, float mu);

// Cosine-weighted hemisphere direction about +z, pdf = cos(theta)/pi. The AO lane relies on the pdf cancelling the cosine (Miller 1994).
[[nodiscard]] glm::vec3 sampleCosineHemisphere(glm::vec2 u);

// The metal's cosine-weighted average Fresnel 2*int_0^1 F82(mu)*mu dmu of the clamped F82, the Kulla-Conty tint input.
[[nodiscard]] glm::vec3 metalFresnelAvg(const glm::vec3& f0, const glm::vec3& tint);

// The dielectric's 2*int_0^1 F(mu) mu dmu entering a medium of relative index eta, in closed form; 0 at eta = 1. The coat's E_F.
[[nodiscard]] float fresnelAverage(float eta);

// F82-split directional albedo E = F0*a + b - k*c and the deficit 1 - E, as (a, b, c, 1 - E), and the deficit's cosine-weighted mean.
[[nodiscard]] glm::vec4 directionalAlbedoSplit(float mu, float roughness);
[[nodiscard]] float averageAlbedoDeficit(float roughness);

// The grid those two index. mu is uniform in sqrt(mu), so never assume k/(res-1).
[[nodiscard]] glm::ivec2 albedoGridRes();
[[nodiscard]] float albedoGridRoughness(float index);
[[nodiscard]] float albedoGridMu(float index);

// Representative wavelength per RGB channel (Adobe's OpenPBR reference); three discrete bands, so dispersion shows RGB banding.
inline constexpr glm::vec3 kRgbWavelengthsNm(620.0F, 540.0F, 450.0F);

// Cauchy n(lambda) from (n_d, V_d) per OpenPBR's dispersion; V_d = transmission_dispersion_abbe_number / transmission_dispersion_scale.
[[nodiscard]] float cauchyIor(float iorD, float abbe, float lambdaNm);

}  // namespace pathtracer::scene
