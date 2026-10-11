#include "pathtracer/scene/bsdf.h"

#include <algorithm>
#include <cmath>

#include "eon.h"
#include "fuzz.h"
#include "microfacet.h"
#include "pathtracer/scene/fresnel_dielectric.h"
#include "shading_math.h"

namespace pathtracer::scene {

namespace {

// Fraunhofer d, F and C lines as OpenPBR states them (656.3, 587.6, 486.1 nm), where V_d = (n_d - 1)/(n_F - n_C) is defined.
constexpr float kLambdaDNm = 587.6F;
constexpr float kLambdaFNm = 486.1F;
constexpr float kLambdaCNm = 656.3F;

float& massOf(BsdfClosure& closure, Technique technique) { return closure.mass[static_cast<std::size_t>(technique)]; }

float massOf(const BsdfClosure& closure, Technique technique) { return closure.mass[static_cast<std::size_t>(technique)]; }

// Rows (t, b, n) of an orthonormal basis about unit n, branchless (Duff et al. 2017); the identity at n = +z.
glm::mat3 toFrameAbout(const glm::vec3& n) {
    const float s = std::copysign(1.0F, n.z);
    const float a = -1.0F / (s + n.z);
    const float b = n.x * n.y * a;
    return glm::transpose(glm::mat3(glm::vec3(1.0F + (s * n.x * n.x * a), s * b, -s * n.x), glm::vec3(b, s + (n.y * n.y * a), -n.y), n));
}

// Rows (t, b, n) about unit n with t projected into its plane: an anisotropic lobe's alpha_t runs along t. Identity at (+z, +x).
glm::mat3 toFrameAlong(const glm::vec3& n, const glm::vec3& t) {
    const glm::vec3 tangent = glm::normalize(t - (glm::dot(t, n) * n));
    return glm::transpose(glm::mat3(tangent, glm::cross(n, tangent), n));
}

// OpenPBR's specular_weight: F0 = xi*F_s, capped at the largest float below 1 so the modulated ratio (1+eps)/(1-eps) stays finite.
float modulatedRatio(float eta, float specularWeight) {
    const float reflectance = ((eta - 1.0F) / (eta + 1.0F)) * ((eta - 1.0F) / (eta + 1.0F));
    const float epsilon = std::copysign(std::sqrt(std::min(specularWeight * reflectance, std::nextafter(1.0F, 0.0F))), eta - 1.0F);
    // At xi = 1 the modulation is the identity; taking the index as authored keeps it free of the ratio's rounding.
    return specularWeight == 1.0F ? eta : (1.0F + epsilon) / (1.0F - epsilon);
}

// specular_ior at the path's hero wavelength: dispersion enters here alone, so every index at the vertex is spectrally consistent.
float dispersedIor(const OpenPbrInputs<Constant>& inputs, std::optional<int> heroChannel) {
    // A zero transmission_dispersion_scale is V_d infinite, no dispersion, whatever the Abbe number.
    if (!heroChannel.has_value() || inputs.transmissionDispersionScale == 0.0F) {
        return inputs.specularIor;
    }
    // V_d's least value keeping the longest hero band's index on its side of the surround (the spec allows V_d = 0, unbounded dispersion).
    const float longest = std::max({kRgbWavelengthsNm.x, kRgbWavelengthsNm.y, kRgbWavelengthsNm.z});
    const float leastAbbe = ((1.0F / (kLambdaDNm * kLambdaDNm)) - (1.0F / (longest * longest))) /
                            ((1.0F / (kLambdaFNm * kLambdaFNm)) - (1.0F / (kLambdaCNm * kLambdaCNm)));
    const float abbe = std::max(inputs.transmissionDispersionAbbeNumber / inputs.transmissionDispersionScale, leastAbbe);
    // A medium rarer than its surround disperses as its reciprocal, the denser side, so short wavelengths still bend the most.
    const float n = inputs.specularIor;
    const float dense = cauchyIor(std::max(n, 1.0F / n), abbe, kRgbWavelengthsNm[*heroChannel]);
    return n >= 1.0F ? dense : 1.0F / dense;
}

// What passes beneath the fuzz toward wi, untinted (OpenPBR): 1 - F E_fuzz(wo) from outside, from inside only leaving transmission.
float underFuzz(const BsdfClosure& closure, const glm::vec3& wi) {
    if (closure.fuzzWeight == 0.0F) {
        return 1.0F;
    }
    if (!closure.exiting) {
        return closure.belowFuzz;
    }
    return wi.z >= 0.0F ? 1.0F
                        : 1.0F - (closure.fuzzWeight * fuzzAlbedo(closure.fuzzRoughness, std::max((closure.toFuzz * wi).z, 0.0F)));
}

// The base's weight under the coat toward wi (OpenPBR's lerp(1, T_coat (1 - E_coat), C), darkened): side decides which rays cross it.
glm::vec3 underCoat(const BsdfClosure& closure, const glm::vec3& wi) {
    if (closure.coatWeight == 0.0F) {
        return glm::vec3(1.0F);
    }
    // A coat normal tilted past wi leaves wi at the grazing limit of the coat, where the transmittance is its own limit.
    glm::vec3 coatWi = closure.toCoat * wi;
    coatWi.z = std::max(coatWi.z, 0.0F);
    const float mu = coatWi.z;
    if (closure.exiting) {
        // From inside only transmission crosses the coat, once, leaving toward wi; the base's internal reflection never reaches it.
        if (wi.z >= 0.0F) {
            return glm::vec3(1.0F);
        }
        const float coatAlbedo = reflectionAlbedo(closure.coatRoughness, closure.coatAnisotropy, 1.0F, closure.coatIor, coatWi);
        return glm::vec3(1.0F - closure.coatWeight) +
               ((closure.coatWeight * (1.0F - coatAlbedo)) * coatTransmittance(closure.coatColor, closure.coatIor, mu));
    }
    return wi.z < 0.0F ? closure.transmitUnder : closure.baseBare + (closure.baseUnder * coatTransmittance(closure.coatColor, closure.coatIor, mu));
}

// OpenPBR's thin film over the base's Fresnel; a zero thickness is no film at all, its exact limit.
FilmLayer filmOf(const OpenPbrInputs<Constant>& inputs) {
    FilmLayer film;
    if (inputs.thinFilmWeight > 0.0F && inputs.thinFilmThickness > 0.0F) {
        // thin_film_thickness is in micrometres.
        film = {inputs.thinFilmWeight, inputs.thinFilmThickness * 1000.0F, inputs.thinFilmIor, inputs.coatWeight, inputs.coatIor};
    }
    return film;
}

// The base interface from wo's side: coat shift and film from outside, no tint inside; a thin wall's refracts nothing (the sheet does).
InterfaceInputs baseInterface(const OpenPbrInputs<Constant>& inputs, float roughness, float ior, float fresnelIor, bool exiting) {
    // OpenPBR's two exclusive regimes: at transmission_depth > 0 the interior medium carries the colour; at 0 it tints the surface.
    const glm::vec3 transmitTint = inputs.transmissionDepth > 0.0F ? glm::vec3(1.0F) : inputs.transmissionColor;
    return InterfaceInputs{.roughness = roughness,
                           .anisotropy = inputs.specularRoughnessAnisotropy,
                           .etaI = exiting ? ior : 1.0F,
                           .etaT = exiting ? 1.0F : ior,
                           .fresnelEtaI = exiting ? ior : 1.0F,
                           .fresnelEtaT = exiting ? 1.0F : fresnelIor,
                           .tint = exiting ? glm::vec3(1.0F) : inputs.specularColor,
                           .refractWeight = exiting ? 1.0F : inputs.geometryThinWalled ? 0.0F : inputs.transmissionWeight,
                           .transmitTint = transmitTint,
                           .film = filmOf(inputs),
                           .baseIor = ior,
                           .fromBase = exiting};
}

// The metal: base_metalness of the mix, its Fresnel tinted by specular_color and scaled by specular_weight.
void addMetal(BsdfClosure& closure, const OpenPbrInputs<Constant>& inputs, const glm::vec3& baseAlbedo, float roughness) {
    closure.metal = makeConductorSlab(roughness, inputs.specularRoughnessAnisotropy, baseAlbedo, inputs.specularColor, inputs.specularWeight,
                                      filmOf(inputs), closure.wo);
    massOf(closure, Technique::MetalSingle) = closure.metalWeight * closure.metal.singleEnergy;
    massOf(closure, Technique::MetalMulti) = closure.metalWeight * closure.metal.multiEnergy;
}

// The dielectric's interface over mix(glossy-diffuse, translucent, T), or thin-walled over mix(glossy-diffuse, subsurface, S) alone.
void addDielectric(BsdfClosure& closure, const OpenPbrInputs<Constant>& inputs, const glm::vec3& baseAlbedo, float roughness, float ior,
                   float fresnelIor) {
    // Only the interface faces a ray from inside: the translucent share is whole there, with no diffuse and no specular_color tint.
    const bool exiting = closure.exiting;
    const DielectricSlab& slab = closure.dielectric = makeDielectricSlab(baseInterface(inputs, roughness, ior, fresnelIor, exiting), closure.wo);
    const float weight = closure.dielectricWeight;
    const float refracted = slab.refractWeight * channelMean(slab.transmitSingle);
    massOf(closure, Technique::DielectricSingle) =
        weight * (channelMean(slab.tint * slab.reflectSingle) + (slab.deltaRefraction ? 0.0F : refracted));
    massOf(closure, Technique::DielectricRefract) = slab.deltaRefraction ? weight * refracted : 0.0F;
    massOf(closure, Technique::DielectricMultiReflect) = weight * slab.tintMean * slab.multiReflect;
    massOf(closure, Technique::DielectricMultiTransmit) = weight * slab.refractWeight * slab.multiTransmit;
    const float opaque = closure.thinWalled ? 1.0F : 1.0F - inputs.transmissionWeight;
    if (exiting || opaque == 0.0F) {
        return;
    }
    // Albedo scaling, f = f_spec + (1 - E_spec(wo)) f_diffuse: energy-exact for any substrate, non-reciprocal by the spec's definition.
    const glm::vec3 underInterface = (weight * opaque) * (1.0F - (slab.tint * reflectAlbedo(slab)));
    closure.diffuseWeight = (1.0F - inputs.subsurfaceWeight) * underInterface;
    massOf(closure, Technique::Diffuse) = channelMean(closure.diffuseWeight * baseAlbedo);
    if (massOf(closure, Technique::Diffuse) > 0.0F) {
        closure.diffuse = makeDiffuseSlab(baseAlbedo, inputs.baseDiffuseRoughness, closure.wo);
    }
    // OpenPBR's thin-walled subsurface: albedo-1 lobes f+ and f- in proportions (1 -/+ g)/2, tinted by subsurface_color (spec 1266).
    closure.subsurfaceWeight = (inputs.subsurfaceWeight * underInterface) * inputs.subsurfaceColor;
    massOf(closure, Technique::ThinSubsurface) = channelMean(closure.subsurfaceWeight);
    if (massOf(closure, Technique::ThinSubsurface) > 0.0F) {
        closure.subsurfaceReflectShare = 0.5F * (1.0F - inputs.subsurfaceScatterAnisotropy);
        closure.subsurface = makeDiffuseSlab(glm::vec3(1.0F), inputs.baseDiffuseRoughness, closure.wo);
        closure.thinTransmission = closure.thinTransmission || closure.subsurfaceReflectShare < 1.0F;
    }
}

// The thin wall's two faces from wo's side: the facing one at the coat-aware Fresnel ratio and under the film.
SheetInterfaces sheetInterfaces(const OpenPbrInputs<Constant>& inputs, float ior, float fresnelIor) {
    return {inputs.specularColor, inputs.transmissionColor, fresnelIor, ior, filmOf(inputs)};
}

// OpenPBR's thin-walled translucent base: the sheet's reflection and its transmission mirrored through the wall (KC17).
void addSheet(BsdfClosure& closure, const OpenPbrInputs<Constant>& inputs, float roughness, float ior, float fresnelIor) {
    const SheetSlab& sheet = closure.sheet = makeSheetSlab(sheetInterfaces(inputs, ior, fresnelIor),
                                                           (1.0F - inputs.baseMetalness) * inputs.transmissionWeight, roughness,
                                                           inputs.specularRoughnessAnisotropy, closure.wo);
    const float energy = channelMean(sheet.reflect) + channelMean(sheet.transmit);
    massOf(closure, Technique::SheetSingle) = energy * sheet.lobe.singleEnergy;
    massOf(closure, Technique::SheetMulti) = energy * sheet.lobe.multiEnergy;
    closure.thinTransmission = closure.thinTransmission || channelMean(sheet.transmit) > 0.0F;
}

// The whole base's albedo at normal incidence, E_b of OpenPBR's coat darkening: metal, and the glossy-diffuse or translucent dielectric.
glm::vec3 baseAlbedoAtNormal(const OpenPbrInputs<Constant>& inputs, const glm::vec3& baseAlbedo, float roughness, float ior,
                             float fresnelIor) {
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    const glm::vec3 metal = makeConductorSlab(roughness, inputs.specularRoughnessAnisotropy, baseAlbedo, inputs.specularColor,
                                              inputs.specularWeight, filmOf(inputs), normal)
                                .albedo;
    const glm::vec3 specular =
        inputs.specularColor * reflectAlbedo(makeDielectricSlab(baseInterface(inputs, roughness, ior, fresnelIor, false), normal));
    if (!inputs.geometryThinWalled) {
        const glm::vec3 dielectric = specular + ((1.0F - inputs.transmissionWeight) * (1.0F - specular) * baseAlbedo);
        return (inputs.baseMetalness * metal) + ((1.0F - inputs.baseMetalness) * dielectric);
    }
    // Thin-walled: the sheet's reflection, and the subsurface's reflected share beside the diffuse under the interface.
    const glm::vec3 subsurface = 0.5F * (1.0F - inputs.subsurfaceScatterAnisotropy) * inputs.subsurfaceColor;
    const glm::vec3 underInterface = (1.0F - specular) * glm::mix(baseAlbedo, subsurface, inputs.subsurfaceWeight);
    const glm::vec3 sheet = inputs.specularColor * sheetLadder(sheetInterfaces(inputs, ior, fresnelIor), 1.0F).reflect;
    const glm::vec3 dielectric = glm::mix(specular + underInterface, sheet, inputs.transmissionWeight);
    return (inputs.baseMetalness * metal) + ((1.0F - inputs.baseMetalness) * dielectric);
}

// OpenPBR's fuzz: its slab at wo in its own frame, F * fuzz_color * E_fuzz its mass, and 1 - F E_fuzz(wo) left for every layer beneath.
void addFuzz(BsdfClosure& closure, const OpenPbrInputs<Constant>& inputs) {
    const glm::vec3 wo = closure.toFuzz * closure.wo;
    // A view below a tilted fuzz plane meets no fiber it could reflect from: the fuzz is undrawn and passes everything beneath.
    closure.belowFuzz = 1.0F;
    if (!(wo.z > 0.0F)) {
        return;
    }
    closure.fuzz = makeFuzzSlab(inputs.fuzzColor, closure.fuzzRoughness, wo);
    closure.belowFuzz = 1.0F - (closure.fuzzWeight * closure.fuzz.albedo);
    for (std::size_t t = static_cast<std::size_t>(Technique::CoatSingle); t < kTechniqueCount; ++t) {
        closure.mass[t] *= closure.belowFuzz;
    }
    massOf(closure, Technique::Fuzz) = closure.fuzzWeight * channelMean(inputs.fuzzColor) * closure.fuzz.albedo;
}

// OpenPBR's coat: its slab at wo in its own frame, then the base's absorption, (1 - E_coat) and darkening weights under it.
void addCoat(BsdfClosure& closure, const OpenPbrInputs<Constant>& inputs, const glm::vec3& baseAlbedo, float roughness, float ior,
             float fresnelIor, float specularRatio) {
    const float weight = closure.coatWeight;
    glm::vec3 coatWo = closure.toCoat * closure.wo;
    coatWo.z = std::max(coatWo.z, 0.0F);
    const float muO = coatWo.z;
    const DielectricSlab& coat = closure.coat = makeDielectricSlab(InterfaceInputs{.roughness = closure.coatRoughness,
                                                                                    .anisotropy = closure.coatAnisotropy,
                                                                                    .etaI = 1.0F,
                                                                                    .etaT = closure.coatIor,
                                                                                    .fresnelEtaI = 1.0F,
                                                                                    .fresnelEtaT = closure.coatIor,
                                                                                    .tint = glm::vec3(1.0F),
                                                                                    .refractWeight = 0.0F,
                                                                                    .transmitTint = glm::vec3(1.0F),
                                                                                    .film = {},
                                                                                    .baseIor = closure.coatIor,
                                                                                    .fromBase = false},
                                                                   coatWo);
    // A view below a tilted coat plane meets no coat facet it could reflect from: the lobes stay undrawn, the absorption at its limit.
    if ((closure.toCoat * closure.wo).z > 0.0F) {
        massOf(closure, Technique::CoatSingle) = weight * channelMean(coat.reflectSingle);
        massOf(closure, Technique::CoatMulti) = weight * coat.multiReflect;
    }
    // Darkening (OpenPBR): Delta = (1-K)/(1-E_b K), K the internal reflectance between the smooth (F) and Lambertian (K_r) base limits.
    glm::vec3 darkening(1.0F);
    glm::vec3 transmitGain(1.0F);
    if (inputs.coatDarkening > 0.0F) {
        const float nc2 = closure.coatIor * closure.coatIor;
        const float smoothK = fresnelDielectric(muO, 1.0F, closure.coatIor);
        const float roughK = 1.0F - ((1.0F - fresnelAverage(closure.coatIor)) / nc2);
        // r_d = lerp(1, r, xi_s F_s), F_s at the coat-aware eta_s before modulation, xi_s F_s clamped to [0, 1] (spec 997-1000).
        const float specularF0 = ((specularRatio - 1.0F) / (specularRatio + 1.0F)) * ((specularRatio - 1.0F) / (specularRatio + 1.0F));
        const float dielectricRoughness = lerp1(1.0F, inputs.specularRoughness, std::min(inputs.specularWeight * specularF0, 1.0F));
        const float baseRoughness = lerp1(dielectricRoughness, inputs.specularRoughness, inputs.baseMetalness);
        const float k = lerp1(smoothK, roughK, baseRoughness);
        const glm::vec3 below = 1.0F - (baseAlbedoAtNormal(inputs, baseAlbedo, roughness, ior, fresnelIor) * k);
        // E_b K <= 1 with equality only at E_b = K = 1, where nothing is absorbed or transmitted: both factors' limit there is exactly 1.
        glm::vec3 delta(1.0F);
        glm::vec3 series(1.0F);
        for (int c = 0; c < 3; ++c) {
            if (below[c] > 0.0F) {
                delta[c] = (1.0F - k) / below[c];
                series[c] = 1.0F / below[c];
            }
        }
        darkening = glm::vec3(1.0F) + ((weight * inputs.coatDarkening) * (delta - 1.0F));
        // Light the base transmits never re-crosses the coat: the same series sums to T/(1 - E_b K), with no (1 - K) exit (Elias 2001).
        transmitGain = glm::vec3(1.0F) + ((weight * inputs.coatDarkening) * (series - 1.0F));
    }
    const glm::vec3 throughCoat = (weight * (1.0F - reflectAlbedo(coat))) * coatTransmittance(closure.coatColor, closure.coatIor, muO);
    closure.baseBare = (1.0F - weight) * darkening;
    closure.baseUnder = throughCoat * darkening;
    closure.transmitUnder = ((1.0F - weight) + throughCoat) * transmitGain;
}

// Every slab's continuous value and the mixture density at wi, wo's hemisphere being +z; the base's values carry its weight under the coat.
BsdfEval evaluateSlabs(const BsdfClosure& closure, const glm::vec3& wi) {
    BsdfEval eval{};
    const glm::vec3& wo = closure.wo;
    const float beneath = underFuzz(closure, wi);
    const glm::vec3 under = beneath * underCoat(closure, wi);
    if (closure.metalWeight > 0.0F && wi.z > 0.0F) {
        const ConductorEval metal = evaluateConductor(closure.metal, wo, wi);
        eval.specular += (closure.metalWeight * under) * metal.value;
        eval.pdf += (massOf(closure, Technique::MetalSingle) * metal.pdfSingle) + (massOf(closure, Technique::MetalMulti) * metal.pdfMulti);
    }
    if (closure.dielectricWeight > 0.0F) {
        const DielectricEval dielectric = evaluateDielectric(closure.dielectric, wo, wi);
        eval.specular += (closure.dielectricWeight * under) * dielectric.reflect;
        eval.transmission += (closure.dielectricWeight * under) * dielectric.transmit;
        eval.pdf += (massOf(closure, Technique::DielectricSingle) * dielectric.pdfSingle) +
                    (massOf(closure, Technique::DielectricMultiReflect) * dielectric.pdfMultiReflect) +
                    (massOf(closure, Technique::DielectricMultiTransmit) * dielectric.pdfMultiTransmit);
    }
    if (massOf(closure, Technique::Diffuse) > 0.0F) {
        const DiffuseEval diffuse = evaluateDiffuse(closure.diffuse, wo, wi);
        eval.diffuse = (closure.diffuseWeight * under) * diffuse.value;
        eval.pdf += massOf(closure, Technique::Diffuse) * diffuse.pdf;
    }
    // A thin wall's lobes transmit as their reflection mirrored through the wall: one lobe, evaluated at wi's mirror below.
    const bool reflected = wi.z > 0.0F;
    const glm::vec3 above(wi.x, wi.y, std::abs(wi.z));
    if (massOf(closure, Technique::SheetSingle) + massOf(closure, Technique::SheetMulti) > 0.0F) {
        const SheetSlab& sheet = closure.sheet;
        const ConductorEval lobe = evaluateConductor(sheet.lobe, wo, above);
        (reflected ? eval.specular : eval.transmission) += (under * (reflected ? sheet.reflect : sheet.transmit)) * lobe.value;
        eval.pdf += (reflected ? sheet.reflectShare : 1.0F - sheet.reflectShare) *
                    ((massOf(closure, Technique::SheetSingle) * lobe.pdfSingle) + (massOf(closure, Technique::SheetMulti) * lobe.pdfMulti));
    }
    if (massOf(closure, Technique::ThinSubsurface) > 0.0F) {
        const float share = reflected ? closure.subsurfaceReflectShare : 1.0F - closure.subsurfaceReflectShare;
        const DiffuseEval subsurface = evaluateDiffuse(closure.subsurface, wo, above);
        (reflected ? eval.diffuse : eval.transmission) += (share * closure.subsurfaceWeight * under) * subsurface.value;
        eval.pdf += massOf(closure, Technique::ThinSubsurface) * share * subsurface.pdf;
    }
    // The coat reflects in its own frame, cosine-weighted about its own normal; its slab exists only where it can be drawn.
    if (massOf(closure, Technique::CoatSingle) + massOf(closure, Technique::CoatMulti) > 0.0F) {
        const DielectricEval coat = evaluateDielectric(closure.coat, closure.toCoat * wo, closure.toCoat * wi);
        eval.specular += (closure.coatWeight * beneath) * coat.reflect;
        eval.pdf += (massOf(closure, Technique::CoatSingle) * coat.pdfSingle) + (massOf(closure, Technique::CoatMulti) * coat.pdfMultiReflect);
    }
    // The fuzz reflects in its own frame, its color tinting its own lobe alone; a sheen is reflection, so it reports as specular.
    if (massOf(closure, Technique::Fuzz) > 0.0F) {
        const FuzzEval fuzz = evaluateFuzz(closure.fuzz, closure.toFuzz * wi);
        eval.specular += closure.fuzzWeight * fuzz.value;
        eval.pdf += massOf(closure, Technique::Fuzz) * fuzz.pdf;
    }
    return eval;
}

// One-sample MIS: whichever technique drew wi, the throughput divides by the whole mixture's density at it.
std::optional<BsdfSample> weighSample(const BsdfClosure& closure, const glm::vec3& wi, LobeType type) {
    const BsdfEval eval = evaluateSlabs(closure, wi);
    if (!(eval.pdf > 0.0F)) {
        return std::nullopt;
    }
    return BsdfSample{glm::vec3(wi.x, wi.y, wi.z * closure.sign), eval.total() / eval.pdf, eval.transmission / eval.pdf, type, eval.pdf, false};
}

// A delta technique's sample: its value over its own mass, pdf 0, so NEE has no density to double-count against.
BsdfSample deltaSample(const BsdfClosure& closure, const glm::vec3& wi, const glm::vec3& value, Technique technique, LobeType type) {
    const glm::vec3 weight = value / massOf(closure, technique);
    return BsdfSample{glm::vec3(wi.x, wi.y, wi.z * closure.sign), weight, type == LobeType::Transmission ? weight : glm::vec3(0.0F), type, 0.0F, true};
}

// wi mirrored through the wall when uSplit falls past the reflected share: a thin wall's transmission is its reflection's mirror.
glm::vec3 throughWall(const glm::vec3& wi, float uSplit, float reflectShare) {
    return uSplit < reflectShare ? wi : glm::vec3(wi.x, wi.y, -wi.z);
}

// The smooth sheet's two deltas, the mirror or -wo by side, each its value over the technique's mass times its side's probability.
BsdfSample sampleSmoothSheet(const BsdfClosure& closure, float uSplit) {
    const SheetSlab& sheet = closure.sheet;
    const glm::vec3& wo = closure.wo;
    const float mass = massOf(closure, Technique::SheetSingle);
    if (uSplit < sheet.reflectShare) {
        const glm::vec3 mirror(-wo.x, -wo.y, wo.z);
        const glm::vec3 weight = (sheet.reflect * underFuzz(closure, mirror) * underCoat(closure, mirror)) / (mass * sheet.reflectShare);
        return BsdfSample{glm::vec3(mirror.x, mirror.y, mirror.z * closure.sign), weight, glm::vec3(0.0F), LobeType::SpecularReflection, 0.0F,
                          true};
    }
    const glm::vec3 weight = passThrough(closure) / (mass * (1.0F - sheet.reflectShare));
    return BsdfSample{glm::vec3(-wo.x, -wo.y, -wo.z * closure.sign), weight, weight, LobeType::Transmission, 0.0F, true, true};
}

// Smooth refraction by Snell; its mass is the (1-F) energy, exactly 0 past the critical angle, so this draw never meets TIR.
BsdfSample sampleDeltaRefraction(const BsdfClosure& closure) {
    const DielectricSlab& slab = closure.dielectric;
    const glm::vec3& wo = closure.wo;
    const float eta = slab.etaI / slab.etaT;
    const glm::vec3 wt(-eta * wo.x, -eta * wo.y, -std::sqrt(cos2Transmitted(wo.z, eta)));
    // Non-symmetric radiance compression for camera-originated transport (Veach 1997 sec. 5.2): eta^2 = (etaI/etaT)^2.
    const glm::vec3 value = (slab.transmitTint * underFuzz(closure, wt) * underCoat(closure, wt)) *
                            (closure.dielectricWeight * slab.refractWeight * slab.transmitSingle * slab.etaSq);
    return deltaSample(closure, wt, value, Technique::DielectricRefract, LobeType::Transmission);
}

std::optional<BsdfSample> sampleTechnique(const BsdfClosure& closure, Technique technique, float uSplit, Sampler& sampler) {
    const glm::vec3& wo = closure.wo;
    const glm::vec3 mirror(-wo.x, -wo.y, wo.z);
    switch (technique) {
        case Technique::Fuzz:
            return weighSample(closure, glm::transpose(closure.toFuzz) * sampleFuzz(closure.fuzz, sampler.next2D()), LobeType::SpecularReflection);
        case Technique::CoatSingle: {
            const DielectricSlab& slab = closure.coat;
            const glm::vec3 woCoat = closure.toCoat * wo;
            const glm::mat3 fromCoat = glm::transpose(closure.toCoat);
            if (isSmooth(slab.alpha.x)) {
                const glm::vec3 wi = fromCoat * glm::vec3(-woCoat.x, -woCoat.y, woCoat.z);
                const glm::vec3 value = (closure.coatWeight * underFuzz(closure, wi)) * slab.reflectSingle;
                return deltaSample(closure, wi, value, technique, LobeType::SpecularReflection);
            }
            const std::optional<InterfaceSample> sample = sampleDielectricSingle(slab, woCoat, sampler.next2D(), uSplit);
            return sample ? weighSample(closure, fromCoat * sample->wi, LobeType::SpecularReflection) : std::nullopt;
        }
        case Technique::CoatMulti:
            return weighSample(closure, glm::transpose(closure.toCoat) * sampleEscapeShape(closure.coat.reflectShape, sampler.next2D()),
                               LobeType::SpecularReflection);
        case Technique::MetalSingle: {
            const ConductorSlab& slab = closure.metal;
            if (isSmooth(slab.alpha.x)) {
                const glm::vec3 value =
                    (closure.metalWeight * underFuzz(closure, mirror) * underCoat(closure, mirror)) * conductorFresnel(slab, wo.z);
                return deltaSample(closure, mirror, value, technique, LobeType::SpecularReflection);
            }
            const std::optional<glm::vec3> wi = sampleConductorSingle(slab, wo, sampler.next2D());
            return wi ? weighSample(closure, *wi, LobeType::SpecularReflection) : std::nullopt;
        }
        case Technique::MetalMulti:
            return weighSample(closure, sampleConductorMulti(closure.metal, sampler.next2D()), LobeType::SpecularReflection);
        case Technique::DielectricSingle: {
            const DielectricSlab& slab = closure.dielectric;
            if (isSmooth(slab.alpha.x)) {
                const glm::vec3 value =
                    (slab.tint * underFuzz(closure, mirror) * underCoat(closure, mirror)) * (closure.dielectricWeight * slab.reflectSingle);
                return deltaSample(closure, mirror, value, technique, LobeType::SpecularReflection);
            }
            const std::optional<InterfaceSample> sample = sampleDielectricSingle(slab, wo, sampler.next2D(), uSplit);
            if (!sample) {
                return std::nullopt;
            }
            return weighSample(closure, sample->wi, sample->refracted ? LobeType::Transmission : LobeType::SpecularReflection);
        }
        case Technique::DielectricRefract:
            return sampleDeltaRefraction(closure);
        // The multiple-scattering lobes report SpecularReflection: repeated GGX bounces are specular however broad their exitant lobe.
        case Technique::DielectricMultiReflect:
            return weighSample(closure, sampleEscapeShape(closure.dielectric.reflectShape, sampler.next2D()), LobeType::SpecularReflection);
        case Technique::DielectricMultiTransmit: {
            const glm::vec3 wi = sampleEscapeShape(closure.dielectric.transmitShape, sampler.next2D());
            return weighSample(closure, glm::vec3(wi.x, wi.y, -wi.z), LobeType::Transmission);
        }
        case Technique::Diffuse:
            return weighSample(closure, sampleDiffuse(closure.diffuse, sampler.next2D()), LobeType::Diffuse);
        case Technique::SheetSingle: {
            const SheetSlab& sheet = closure.sheet;
            if (isSmooth(sheet.lobe.alpha.x)) {
                return sampleSmoothSheet(closure, uSplit);
            }
            const std::optional<glm::vec3> wi = sampleConductorSingle(sheet.lobe, wo, sampler.next2D());
            if (!wi) {
                return std::nullopt;
            }
            const glm::vec3 side = throughWall(*wi, uSplit, sheet.reflectShare);
            return weighSample(closure, side, side.z > 0.0F ? LobeType::SpecularReflection : LobeType::Transmission);
        }
        case Technique::SheetMulti: {
            const glm::vec3 side = throughWall(sampleConductorMulti(closure.sheet.lobe, sampler.next2D()), uSplit, closure.sheet.reflectShare);
            return weighSample(closure, side, side.z > 0.0F ? LobeType::SpecularReflection : LobeType::Transmission);
        }
        case Technique::ThinSubsurface: {
            const glm::vec3 side = throughWall(sampleDiffuse(closure.subsurface, sampler.next2D()), uSplit, closure.subsurfaceReflectShare);
            return weighSample(closure, side, side.z > 0.0F ? LobeType::Diffuse : LobeType::Transmission);
        }
        case Technique::Count:
            break;
    }
    return std::nullopt;
}

}  // namespace

// Malley's method, a uniform disk point lifted to the hemisphere (PBR 4th ed. 13.6.3). External for the AO lane, where the pdf cancels.
glm::vec3 sampleCosineHemisphere(glm::vec2 u) {
    const float r = std::sqrt(u.x);
    const float phi = 2.0F * kPi * u.y;
    return {r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0F, 1.0F - u.x))};
}

// Cauchy n(lambda) = A + B/lambda^2, (A,B) from (n_d, V_d); V_d infinite, dispersion scale 0, gives B = 0 and the index itself.
float cauchyIor(float iorD, float abbe, float lambdaNm) {
    const float b = (iorD - 1.0F) / (abbe * ((1.0F / (kLambdaFNm * kLambdaFNm)) - (1.0F / (kLambdaCNm * kLambdaCNm))));
    const float a = iorD - (b / (kLambdaDNm * kLambdaDNm));
    return a + (b / (lambdaNm * lambdaNm));
}

float modulatedIor(const OpenPbrInputs<Constant>& inputs, std::optional<int> heroChannel) {
    return modulatedRatio(dispersedIor(inputs, heroChannel), inputs.specularWeight);
}

glm::vec3 coatTransmittance(const glm::vec3& coatColor, float coatIor, float mu) {
    // Past the coat's critical cosine (coat_ior < 1) no ray refracts in: mu_t = 0 and the exponent's limit is exact.
    const float muT = std::sqrt(std::max(1.0F - ((1.0F - (mu * mu)) / (coatIor * coatIor)), 0.0F));
    return glm::pow(coatColor, glm::vec3(0.5F / muT));
}

BsdfClosure makeBsdfClosure(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, std::optional<int> heroChannel,
                            const glm::vec3& coatNormalLocal, const glm::vec3& coatTangentLocal) {
    BsdfClosure closure;
    closure.sign = woLocal.z >= 0.0F ? 1.0F : -1.0F;
    closure.wo = glm::vec3(woLocal.x, woLocal.y, woLocal.z * closure.sign);
    closure.thinWalled = inputs.geometryThinWalled;
    closure.exiting = !closure.thinWalled && closure.sign < 0.0F && inputs.transmissionWeight > 0.0F;
    closure.metalWeight = inputs.baseMetalness;
    closure.dielectricWeight = (1.0F - inputs.baseMetalness) * (closure.thinWalled ? 1.0F - inputs.transmissionWeight : 1.0F);
    const glm::vec3 baseAlbedo = inputs.baseWeight * inputs.baseColor;
    // A thin wall transmits undeviated, so the hero band disperses nothing there.
    const std::optional<int> hero = closure.thinWalled ? std::nullopt : heroChannel;
    const float ior = modulatedIor(inputs, hero);
    float roughness = inputs.specularRoughness;
    float fresnelIor = ior;
    // eta_s, the base's coat-aware index ratio before specular_weight modulates it: darkening's F_s reads it.
    float specularRatio = dispersedIor(inputs, hero);
    closure.coatWeight = inputs.coatWeight;
    // Coat and fuzz lie on the outside, past the base from within; a thin wall has no within, so they face wo on either side.
    const float layerSign = closure.thinWalled ? 1.0F : closure.sign;
    if (closure.coatWeight > 0.0F) {
        closure.coatColor = inputs.coatColor;
        closure.coatIor = inputs.coatIor;
        closure.coatRoughness = inputs.coatRoughness;
        closure.coatAnisotropy = inputs.coatRoughnessAnisotropy;
        // The coat frame crosses wo's mirror, every coat cosine the unmirrored one; a thin wall's coat faces wo from either side.
        const auto mirrored = [&](const glm::vec3& v) { return glm::vec3(v.x, v.y, v.z * layerSign); };
        closure.toCoat = toFrameAlong(mirrored(coatNormalLocal), mirrored(coatTangentLocal));
        // OpenPBR's coat roughens the base, r' = lerp(r, min(1, r^4 + 2 r_c^4)^(1/4), C): the coat's blur crossed twice, in alpha^2.
        const float r2 = roughness * roughness;
        const float rc2 = inputs.coatRoughness * inputs.coatRoughness;
        roughness = lerp1(roughness, std::sqrt(std::sqrt(std::min(1.0F, (r2 * r2) + (2.0F * rc2 * rc2)))), closure.coatWeight);
        // Under the coat the base meets n_c, its Fresnel ratio n_b/n_c, inverted where n_c > n_b so no TIR appears; the bend is unchanged.
        const float base = specularRatio;
        const float coated = inputs.coatIor > base ? inputs.coatIor / base : base / inputs.coatIor;
        specularRatio = lerp1(base, coated, closure.coatWeight);
        fresnelIor = modulatedRatio(specularRatio, inputs.specularWeight);
    }
    closure.baseAlpha = alphaForRoughness(roughness, inputs.specularRoughnessAnisotropy);
    if (closure.metalWeight > 0.0F) {
        addMetal(closure, inputs, baseAlbedo, roughness);
    }
    if (closure.dielectricWeight > 0.0F) {
        addDielectric(closure, inputs, baseAlbedo, roughness, ior, fresnelIor);
    }
    if (closure.thinWalled && inputs.baseMetalness < 1.0F && inputs.transmissionWeight > 0.0F) {
        addSheet(closure, inputs, roughness, ior, fresnelIor);
    }
    closure.fuzzWeight = inputs.fuzzWeight;
    if (closure.fuzzWeight > 0.0F) {
        // OpenPBR's fuzz normal, lerp(base, coat, C), normalised, crossing the same mirror as the coat.
        const glm::vec3 normal = glm::normalize(glm::vec3(0.0F, 0.0F, 1.0F) + (closure.coatWeight * (coatNormalLocal - glm::vec3(0.0F, 0.0F, 1.0F))));
        closure.toFuzz = toFrameAbout(glm::vec3(normal.x, normal.y, normal.z * layerSign));
        closure.fuzzRoughness = inputs.fuzzRoughness;
    }
    if (closure.coatWeight > 0.0F && !closure.exiting) {
        addCoat(closure, inputs, baseAlbedo, roughness, ior, fresnelIor, specularRatio);
        // The base's selection masses at its weight toward wo: selection only, so one scalar serves reflection and refraction alike.
        const float baseScale = channelMean(underCoat(closure, glm::vec3(-closure.wo.x, -closure.wo.y, closure.wo.z)));
        for (std::size_t t = static_cast<std::size_t>(Technique::MetalSingle); t < kTechniqueCount; ++t) {
            closure.mass[t] *= baseScale;
        }
    }
    if (closure.fuzzWeight > 0.0F && !closure.exiting) {
        addFuzz(closure, inputs);
    }
    float total = 0.0F;
    for (const float mass : closure.mass) {
        total += mass;
    }
    // Each technique's mass is its energy at wo over the total; a massless technique is never drawn, and a black vertex draws nothing.
    const float inverseTotal = total > 0.0F ? 1.0F / total : 0.0F;
    for (float& mass : closure.mass) {
        mass *= inverseTotal;
    }
    return closure;
}

BsdfEval evaluateBsdfSplit(const BsdfClosure& closure, const glm::vec3& wiLocal) {
    return evaluateSlabs(closure, glm::vec3(wiLocal.x, wiLocal.y, wiLocal.z * closure.sign));
}

glm::vec3 passThrough(const BsdfClosure& closure) {
    if (!(massOf(closure, Technique::SheetSingle) > 0.0F) || !isSmooth(closure.sheet.lobe.alpha.x)) {
        return glm::vec3(0.0F);
    }
    const glm::vec3 through = -closure.wo;
    return closure.sheet.transmit * underFuzz(closure, through) * underCoat(closure, through);
}

std::optional<BsdfSample> sampleBsdf(const BsdfClosure& closure, Sampler& sampler) {
    const float u = sampler.next1D();
    // The technique whose cumulative mass first exceeds u; the masses sum to 1 only in float, so a u past the sum takes the last drawable.
    float below = 0.0F;
    float chosenBelow = 0.0F;
    std::optional<std::size_t> chosen;
    for (std::size_t t = 0; t < kTechniqueCount; ++t) {
        if (!(closure.mass[t] > 0.0F)) {
            continue;
        }
        chosen = t;
        chosenBelow = below;
        if (u < below + closure.mass[t]) {
            break;
        }
        below += closure.mass[t];
    }
    if (!chosen) {
        return std::nullopt;
    }
    // u's position within the chosen mass is uniform on [0,1): the interface's facet split reuses it rather than drawing a dimension.
    const float uSplit = std::min((u - chosenBelow) / closure.mass[*chosen], 1.0F);
    return sampleTechnique(closure, static_cast<Technique>(*chosen), uSplit, sampler);
}

namespace {

// The base slabs' reflected Fresnel at cosTheta, each at its mix weight: metal F82 and tinted dielectric.
glm::vec3 baseFresnel(const BsdfClosure& closure, float cosTheta) {
    glm::vec3 fresnel(0.0F);
    if (closure.metalWeight > 0.0F) {
        const ConductorSlab& slab = closure.metal;
        fresnel += closure.metalWeight * conductorFresnel(slab, cosTheta);
    }
    if (closure.dielectricWeight > 0.0F) {
        const DielectricSlab& slab = closure.dielectric;
        fresnel += slab.tint * closure.dielectricWeight * interfaceFresnel(slab, cosTheta);
    }
    if (massOf(closure, Technique::SheetSingle) + massOf(closure, Technique::SheetMulti) > 0.0F) {
        const SheetSlab& sheet = closure.sheet;
        fresnel += sheet.interfaces.tint * sheet.weight * sheetLadder(sheet.interfaces, cosTheta).reflect;
    }
    return fresnel;
}

// A slab's visible facet cosine dot(wo, h) for one VNDF draw; the macro normal's own when smooth, where the distribution is a delta.
float facetCosine(const glm::vec3& wo, const glm::vec2& alpha, glm::vec2 u) {
    // Clamped for rounding at a grazing visible facet: fresnelDielectric reads a negative cosine as the other side of the interface.
    return isSmooth(alpha.x) ? wo.z : std::max(glm::dot(wo, sampleGGXVNDF(wo, alpha, u)), 0.0F);
}

}  // namespace

glm::vec3 fresnelAtViewAngle(const BsdfClosure& closure, float cosTheta) {
    const glm::vec3 mirror(-closure.wo.x, -closure.wo.y, closure.wo.z);
    glm::vec3 fresnel = underCoat(closure, mirror) * baseFresnel(closure, cosTheta);
    if (massOf(closure, Technique::CoatSingle) > 0.0F) {
        fresnel += glm::vec3(closure.coatWeight * fresnelDielectric(cosTheta, 1.0F, closure.coatIor));
    }
    return underFuzz(closure, mirror) * fresnel;
}

glm::vec3 fresnelAtMicrofacet(const BsdfClosure& closure, glm::vec2 u) {
    const glm::vec3 mirror(-closure.wo.x, -closure.wo.y, closure.wo.z);
    glm::vec3 fresnel = underCoat(closure, mirror) * baseFresnel(closure, facetCosine(closure.wo, closure.baseAlpha, u));
    if (massOf(closure, Technique::CoatSingle) > 0.0F) {
        const float cosine = facetCosine(closure.toCoat * closure.wo, closure.coat.alpha, u);
        fresnel += glm::vec3(closure.coatWeight * fresnelDielectric(cosine, 1.0F, closure.coatIor));
    }
    return underFuzz(closure, mirror) * fresnel;
}

BsdfEval evaluateBsdfSplit(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, const glm::vec3& wiLocal) {
    return evaluateBsdfSplit(makeBsdfClosure(inputs, woLocal), wiLocal);
}

float pdfBsdf(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, const glm::vec3& wiLocal) {
    return evaluateBsdfSplit(inputs, woLocal, wiLocal).pdf;
}

glm::vec3 evaluateBsdf(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, const glm::vec3& wiLocal) {
    return evaluateBsdfSplit(inputs, woLocal, wiLocal).total();
}

std::optional<BsdfSample> sampleBsdf(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, Sampler& sampler) {
    return sampleBsdf(makeBsdfClosure(inputs, woLocal), sampler);
}

namespace {

enum class Facet { Perturbed, Wall };

// The medium w lies in, by omega_g's side: +1 above, -1 below; the microsurface it meets is the one on that side.
float mediumOf(const NormalMappedBsdf& bsdf, const glm::vec3& w) { return std::copysign(1.0F, glm::dot(w, bsdf.geometric)); }

// w mirrored by the wall, a plane mirror, so the mapping is its own inverse and preserves solid angle.
glm::vec3 offWall(const NormalMappedBsdf& bsdf, const glm::vec3& w) { return w - ((2.0F * glm::dot(w, bsdf.wall)) * bsdf.wall); }

// The unmasked share of a facet's projection toward w (eq. 13), the facets' areas a_p, a_t (eq. 8) on w's sigma side.
float masking(const NormalMappedBsdf& bsdf, const glm::vec3& w, Facet facet, float sigma) {
    const float perturbed = std::max(sigma * w.z, 0.0F) / bsdf.cosP;
    const float wall = std::max(sigma * glm::dot(w, bsdf.wall), 0.0F) * bsdf.sinP / bsdf.cosP;
    if (!((facet == Facet::Perturbed ? perturbed : wall) > 0.0F)) {
        return 0.0F;
    }
    return std::clamp(sigma * glm::dot(w, bsdf.geometric) / (perturbed + wall), 0.0F, 1.0F);
}

// The walk's chance to leave the facet along w rather than meet the wall and leave along its mirror; 1 where neither carries.
float escapeShare(float maskFacet, float maskMirrorWall) {
    const float total = maskFacet + ((1.0F - maskFacet) * maskMirrorWall);
    return total > 0.0F ? maskFacet / total : 1.0F;
}

// One facet closure's lobes at w toward the medium the microsurface leaves into, weighted: reflection on wo's side, else transmission.
void accumulate(BsdfEval& eval, const BsdfEval& lobes, float weight, bool reflected) {
    if (reflected) {
        eval.diffuse += weight * lobes.diffuse;
        eval.specular += weight * lobes.specular;
    } else {
        eval.transmission += weight * lobes.transmission;
    }
}

}  // namespace

NormalMappedBsdf makeNormalMappedBsdf(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, std::optional<glm::vec3> geometricLocal,
                                      std::optional<int> heroChannel, const glm::vec3& coatNormalLocal, const glm::vec3& coatTangentLocal) {
    NormalMappedBsdf bsdf;
    const auto closureAt = [&](const glm::vec3& wo) { return makeBsdfClosure(inputs, wo, heroChannel, coatNormalLocal, coatTangentLocal); };
    if (!geometricLocal) {
        bsdf.facet = closureAt(woLocal);
        return bsdf;
    }
    bsdf.perturbed = true;
    bsdf.geometric = *geometricLocal;
    bsdf.cosP = bsdf.geometric.z;
    const glm::vec3 tilt = glm::vec3(0.0F, 0.0F, 1.0F) - (bsdf.cosP * bsdf.geometric);
    bsdf.sinP = glm::length(tilt);
    // An untilted facet has a wall of no area, any direction across omega_g serving; past 90 degrees no microsurface exists.
    bsdf.wall = bsdf.sinP > 0.0F ? -tilt / bsdf.sinP : glm::vec3(1.0F, 0.0F, 0.0F);
    if (!(bsdf.cosP > 0.0F)) {
        return bsdf;
    }
    bsdf.side = mediumOf(bsdf, woLocal);
    const float perturbed = std::max(bsdf.side * woLocal.z, 0.0F);
    const float wall = std::max(bsdf.side * glm::dot(woLocal, bsdf.wall), 0.0F) * bsdf.sinP;
    if (!(perturbed + wall > 0.0F)) {
        return bsdf;
    }
    bsdf.lambdaP = perturbed / (perturbed + wall);
    bsdf.lambdaT = wall / (perturbed + wall);
    if (bsdf.lambdaP > 0.0F) {
        bsdf.facet = closureAt(woLocal);
    }
    if (bsdf.lambdaT > 0.0F) {
        bsdf.viaWall = closureAt(offWall(bsdf, woLocal));
    }
    return bsdf;
}

bool scatters(const NormalMappedBsdf& bsdf) {
    return bsdf.perturbed ? (bsdf.lambdaP > 0.0F && scatters(bsdf.facet)) || (bsdf.lambdaT > 0.0F && scatters(bsdf.viaWall))
                          : scatters(bsdf.facet);
}

bool transmits(const NormalMappedBsdf& bsdf) {
    return bsdf.perturbed ? (bsdf.lambdaP > 0.0F && transmits(bsdf.facet)) || (bsdf.lambdaT > 0.0F && transmits(bsdf.viaWall))
                          : transmits(bsdf.facet);
}

// Along -wo exactly only the facet-first walk's direct escape: the wall turns every other undeviated ray aside.
glm::vec3 passThrough(const NormalMappedBsdf& bsdf) {
    if (!bsdf.perturbed) {
        return passThrough(bsdf.facet);
    }
    if (!(bsdf.lambdaP > 0.0F)) {
        return glm::vec3(0.0F);
    }
    const glm::vec3 through = -bsdf.facet.wo * glm::vec3(1.0F, 1.0F, bsdf.facet.sign);
    return (bsdf.lambdaP * masking(bsdf, through, Facet::Perturbed, -bsdf.side)) * passThrough(bsdf.facet);
}

// Eq. 23 by component, i -> p -> o, i -> p -> t -> o and i -> t -> p -> o; the density is the walk's own, branch by branch.
BsdfEval evaluateBsdfSplit(const NormalMappedBsdf& bsdf, const glm::vec3& wiLocal) {
    if (!bsdf.perturbed) {
        return evaluateBsdfSplit(bsdf.facet, wiLocal);
    }
    BsdfEval eval{};
    const float sigma = mediumOf(bsdf, wiLocal);
    const bool reflected = sigma == bsdf.side;
    const float maskFacet = masking(bsdf, wiLocal, Facet::Perturbed, sigma);
    if (bsdf.lambdaP > 0.0F) {
        const glm::vec3 mirrored = offWall(bsdf, wiLocal);
        const BsdfEval direct = evaluateBsdfSplit(bsdf.facet, wiLocal);
        accumulate(eval, direct, bsdf.lambdaP * maskFacet, reflected);
        eval.pdf += bsdf.lambdaP * direct.pdf * escapeShare(maskFacet, masking(bsdf, mirrored, Facet::Wall, sigma));
        const float maskWall = masking(bsdf, wiLocal, Facet::Wall, sigma);
        if (maskWall > 0.0F) {
            const BsdfEval bounced = evaluateBsdfSplit(bsdf.facet, mirrored);
            const float maskMirrored = masking(bsdf, mirrored, Facet::Perturbed, sigma);
            accumulate(eval, bounced, bsdf.lambdaP * (1.0F - maskMirrored) * maskWall, reflected);
            eval.pdf += bsdf.lambdaP * bounced.pdf * (1.0F - escapeShare(maskMirrored, maskWall));
        }
    }
    if (bsdf.lambdaT > 0.0F) {
        const BsdfEval viaWall = evaluateBsdfSplit(bsdf.viaWall, wiLocal);
        accumulate(eval, viaWall, bsdf.lambdaT * maskFacet, reflected);
        eval.pdf += bsdf.lambdaT * viaWall.pdf;
    }
    return eval;
}

// Algorithm 2 to second order: the first facet by lambda, a facet draw, then escape or the wall by the masking odds at that draw.
std::optional<BsdfSample> sampleBsdf(const NormalMappedBsdf& bsdf, Sampler& sampler) {
    if (!bsdf.perturbed) {
        return sampleBsdf(bsdf.facet, sampler);
    }
    if (!(bsdf.lambdaP + bsdf.lambdaT > 0.0F)) {
        return std::nullopt;
    }
    const bool facetFirst = bsdf.lambdaT == 0.0F || (bsdf.lambdaP > 0.0F && sampler.next1D() < bsdf.lambdaP);
    const std::optional<BsdfSample> drawn = sampleBsdf(facetFirst ? bsdf.facet : bsdf.viaWall, sampler);
    if (!drawn) {
        return std::nullopt;
    }
    glm::vec3 wi = drawn->wiLocal;
    const float sigma = mediumOf(bsdf, wi);
    const float maskFacet = masking(bsdf, wi, Facet::Perturbed, sigma);
    // A delta's weight on the direction the walk leaves along: the branch's odds cancel against eq. 23's lambda.
    float walkWeight = maskFacet;
    bool escaped = true;
    if (facetFirst) {
        const glm::vec3 mirrored = offWall(bsdf, wi);
        const float maskWall = masking(bsdf, mirrored, Facet::Wall, sigma);
        const float escape = escapeShare(maskFacet, maskWall);
        escaped = escape == 1.0F || (escape > 0.0F && sampler.next1D() < escape);
        walkWeight = escaped ? maskFacet / escape : (1.0F - maskFacet) * maskWall / (1.0F - escape);
        wi = escaped ? wi : mirrored;
    }
    const bool reflected = sigma == bsdf.side;
    const LobeType type = !reflected ? LobeType::Transmission : drawn->type == LobeType::Transmission ? LobeType::SpecularReflection : drawn->type;
    if (drawn->delta) {
        const glm::vec3 weight = walkWeight * (reflected ? drawn->throughputWeight - drawn->transmitWeight : drawn->transmitWeight);
        return BsdfSample{wi, weight, reflected ? glm::vec3(0.0F) : weight, type, 0.0F, true, drawn->passThrough && facetFirst && escaped};
    }
    // A continuous draw divides by the whole walk's density over both branches and both exits, one-sample MIS as weighSample does.
    const BsdfEval eval = evaluateBsdfSplit(bsdf, wi);
    if (!(eval.pdf > 0.0F)) {
        return std::nullopt;
    }
    return BsdfSample{wi, eval.total() / eval.pdf, eval.transmission / eval.pdf, type, eval.pdf, false};
}

glm::vec3 fresnelAtMicrofacet(const NormalMappedBsdf& bsdf, glm::vec2 u) {
    if (!bsdf.perturbed) {
        return fresnelAtMicrofacet(bsdf.facet, u);
    }
    glm::vec3 fresnel(0.0F);
    if (bsdf.lambdaP > 0.0F) {
        fresnel += bsdf.lambdaP * fresnelAtMicrofacet(bsdf.facet, u);
    }
    if (bsdf.lambdaT > 0.0F) {
        fresnel += bsdf.lambdaT * fresnelAtMicrofacet(bsdf.viaWall, u);
    }
    return fresnel;
}

}  // namespace pathtracer::scene
