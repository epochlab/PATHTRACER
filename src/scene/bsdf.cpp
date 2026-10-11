#include "pathtracer/scene/bsdf.h"

#include <algorithm>
#include <cmath>

#include "eon.h"
#include "microfacet.h"
#include "pathtracer/scene/fresnel_dielectric.h"

namespace pathtracer::scene {

namespace {

constexpr float kPi = 3.14159265F;

float channelMean(const glm::vec3& v) { return (v.x + v.y + v.z) / 3.0F; }

float lerp1(float a, float b, float t) { return a + ((b - a) * t); }

float& massOf(BsdfClosure& closure, Technique technique) { return closure.mass[static_cast<std::size_t>(technique)]; }

float massOf(const BsdfClosure& closure, Technique technique) { return closure.mass[static_cast<std::size_t>(technique)]; }

// Rows (t, b, n) of an orthonormal basis about unit n, branchless (Duff et al. 2017); the identity at n = +z.
glm::mat3 toFrameAbout(const glm::vec3& n) {
    const float s = std::copysign(1.0F, n.z);
    const float a = -1.0F / (s + n.z);
    const float b = n.x * n.y * a;
    return glm::transpose(glm::mat3(glm::vec3(1.0F + (s * n.x * n.x * a), s * b, -s * n.x), glm::vec3(b, s + (n.y * n.y * a), -n.y), n));
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
    const float abbe = inputs.transmissionDispersionAbbeNumber / inputs.transmissionDispersionScale;
    return heroChannel.has_value() ? cauchyIor(inputs.specularIor, abbe, kRgbWavelengthsNm[*heroChannel]) : inputs.specularIor;
}

// The base's weight under the coat toward wi (OpenPBR's lerp(1, T_coat (1 - E_coat), C), darkened): side decides which rays cross it.
glm::vec3 underCoat(const BsdfClosure& closure, const glm::vec3& wi) {
    if (closure.coatWeight == 0.0F) {
        return glm::vec3(1.0F);
    }
    // A coat normal tilted past wi leaves wi at the grazing limit of the coat, where the transmittance is its own limit.
    const float mu = std::max((closure.toCoat * wi).z, 0.0F);
    if (closure.exiting) {
        // From inside only transmission crosses the coat, once, leaving toward wi; the base's internal reflection never reaches it.
        if (wi.z >= 0.0F) {
            return glm::vec3(1.0F);
        }
        const float coatAlbedo = reflectionAlbedo(closure.coatRoughness, 1.0F, closure.coatIor, mu);
        return glm::vec3(1.0F - closure.coatWeight) +
               ((closure.coatWeight * (1.0F - coatAlbedo)) * coatTransmittance(closure.coatColor, closure.coatIor, mu));
    }
    return wi.z < 0.0F ? closure.transmitUnder : closure.baseBare + (closure.baseUnder * coatTransmittance(closure.coatColor, closure.coatIor, mu));
}

// The metal: base_metalness of the mix, its Fresnel tinted by specular_color and scaled by specular_weight.
void addMetal(BsdfClosure& closure, const OpenPbrInputs<Constant>& inputs, const glm::vec3& baseAlbedo, float roughness) {
    closure.metal = makeConductorSlab(roughness, baseAlbedo, inputs.specularColor, inputs.specularWeight, closure.wo.z);
    massOf(closure, Technique::MetalSingle) = closure.metalWeight * closure.metal.singleEnergy;
    massOf(closure, Technique::MetalMulti) = closure.metalWeight * closure.metal.multiEnergy;
}

// The dielectric: mix(glossy-diffuse, translucent, T) sharing one interface, the diffuse under it by OpenPBR's albedo scaling.
void addDielectric(BsdfClosure& closure, const OpenPbrInputs<Constant>& inputs, const glm::vec3& baseAlbedo, float roughness, float ior,
                   float fresnelIor) {
    // OpenPBR's two exclusive regimes: at transmission_depth > 0 the interior medium carries the colour; at 0 it tints the surface.
    const glm::vec3 transmitTint = inputs.transmissionDepth > 0.0F ? glm::vec3(1.0F) : inputs.transmissionColor;
    // Only the interface faces a ray from inside: the translucent share is whole there, with no diffuse and no specular_color tint.
    const bool exiting = closure.exiting;
    const DielectricSlab& slab = closure.dielectric = makeDielectricSlab(
        InterfaceInputs{.roughness = roughness,
                        .etaI = exiting ? ior : 1.0F,
                        .etaT = exiting ? 1.0F : ior,
                        .fresnelEtaI = exiting ? ior : 1.0F,
                        .fresnelEtaT = exiting ? 1.0F : fresnelIor,
                        .tint = exiting ? glm::vec3(1.0F) : inputs.specularColor,
                        .refractWeight = exiting ? 1.0F : inputs.transmissionWeight,
                        .transmitTint = transmitTint},
        closure.wo.z);
    const float weight = closure.dielectricWeight;
    const float refracted = slab.refractWeight * slab.transmitSingle;
    massOf(closure, Technique::DielectricSingle) = weight * ((slab.tintMean * slab.reflectSingle) + (slab.deltaRefraction ? 0.0F : refracted));
    massOf(closure, Technique::DielectricRefract) = slab.deltaRefraction ? weight * refracted : 0.0F;
    massOf(closure, Technique::DielectricMultiReflect) = weight * slab.tintMean * slab.multiReflect;
    massOf(closure, Technique::DielectricMultiTransmit) = weight * slab.refractWeight * slab.multiTransmit;
    if (exiting || inputs.transmissionWeight == 1.0F) {
        return;
    }
    // Albedo scaling, f = f_spec + (1 - E_spec(wo)) f_diffuse: energy-exact for any substrate, non-reciprocal by the spec's definition.
    closure.diffuseWeight = (weight * (1.0F - inputs.transmissionWeight)) * (1.0F - (slab.tint * reflectAlbedo(slab)));
    const glm::vec3 rho = eonAlbedoInversion(baseAlbedo, inputs.baseDiffuseRoughness);
    massOf(closure, Technique::Diffuse) = channelMean(closure.diffuseWeight * rho);
    if (massOf(closure, Technique::Diffuse) > 0.0F) {
        closure.diffuse = makeDiffuseSlab(rho, inputs.baseDiffuseRoughness, closure.wo);
    }
}

// The whole base's albedo at normal incidence, E_b of OpenPBR's coat darkening: metal, and the glossy-diffuse or translucent dielectric.
glm::vec3 baseAlbedoAtNormal(const OpenPbrInputs<Constant>& inputs, const glm::vec3& baseAlbedo, float roughness, float fresnelIor) {
    const glm::vec3 metal = conductorAlbedo(roughness, baseAlbedo, inputs.specularColor, inputs.specularWeight, 1.0F);
    const glm::vec3 specular = inputs.specularColor * reflectionAlbedo(roughness, 1.0F, fresnelIor, 1.0F);
    const glm::vec3 dielectric = specular + ((1.0F - inputs.transmissionWeight) * (1.0F - specular) * baseAlbedo);
    return (inputs.baseMetalness * metal) + ((1.0F - inputs.baseMetalness) * dielectric);
}

// OpenPBR's coat: its slab at wo in its own frame, then the base's absorption, (1 - E_coat) and darkening weights under it.
void addCoat(BsdfClosure& closure, const OpenPbrInputs<Constant>& inputs, const glm::vec3& baseAlbedo, float roughness, float fresnelIor) {
    const float weight = closure.coatWeight;
    const float muO = std::max((closure.toCoat * closure.wo).z, 0.0F);
    const DielectricSlab& coat = closure.coat = makeDielectricSlab(InterfaceInputs{.roughness = closure.coatRoughness,
                                                                                    .etaI = 1.0F,
                                                                                    .etaT = closure.coatIor,
                                                                                    .fresnelEtaI = 1.0F,
                                                                                    .fresnelEtaT = closure.coatIor,
                                                                                    .tint = glm::vec3(1.0F),
                                                                                    .refractWeight = 0.0F,
                                                                                    .transmitTint = glm::vec3(1.0F)},
                                                                   muO);
    // A view below a tilted coat plane meets no coat facet it could reflect from: the lobes stay undrawn, the absorption at its limit.
    if ((closure.toCoat * closure.wo).z > 0.0F) {
        massOf(closure, Technique::CoatSingle) = weight * coat.reflectSingle;
        massOf(closure, Technique::CoatMulti) = weight * coat.multiReflect;
    }
    // Darkening (OpenPBR): Delta = (1-K)/(1-E_b K), K the internal reflectance between the smooth (F) and Lambertian (K_r) base limits.
    glm::vec3 darkening(1.0F);
    if (inputs.coatDarkening > 0.0F) {
        const float nc2 = closure.coatIor * closure.coatIor;
        const float smoothK = fresnelDielectric(muO, 1.0F, closure.coatIor);
        const float roughK = 1.0F - ((1.0F - fresnelAverage(closure.coatIor)) / nc2);
        const float specularF0 = ((inputs.specularIor - 1.0F) / (inputs.specularIor + 1.0F)) * ((inputs.specularIor - 1.0F) / (inputs.specularIor + 1.0F));
        const float dielectricRoughness = lerp1(1.0F, inputs.specularRoughness, std::min(inputs.specularWeight * specularF0, 1.0F));
        const float baseRoughness = lerp1(dielectricRoughness, inputs.specularRoughness, inputs.baseMetalness);
        const float k = lerp1(smoothK, roughK, baseRoughness);
        const glm::vec3 delta = (1.0F - k) / (1.0F - (baseAlbedoAtNormal(inputs, baseAlbedo, roughness, fresnelIor) * k));
        darkening = glm::vec3(1.0F) + ((weight * inputs.coatDarkening) * (delta - 1.0F));
    }
    const glm::vec3 throughCoat = (weight * (1.0F - reflectAlbedo(coat))) * coatTransmittance(closure.coatColor, closure.coatIor, muO);
    closure.baseBare = (1.0F - weight) * darkening;
    closure.baseUnder = throughCoat * darkening;
    closure.transmitUnder = ((1.0F - weight) + throughCoat) * darkening;
}

// Every slab's continuous value and the mixture density at wi, wo's hemisphere being +z; the base's values carry its weight under the coat.
BsdfEval evaluateSlabs(const BsdfClosure& closure, const glm::vec3& wi) {
    BsdfEval eval{};
    const glm::vec3& wo = closure.wo;
    const glm::vec3 under = underCoat(closure, wi);
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
    // The coat reflects in its own frame, cosine-weighted about its own normal; its slab exists only where it can be drawn.
    if (massOf(closure, Technique::CoatSingle) + massOf(closure, Technique::CoatMulti) > 0.0F) {
        const DielectricEval coat = evaluateDielectric(closure.coat, closure.toCoat * wo, closure.toCoat * wi);
        eval.specular += closure.coatWeight * coat.reflect;
        eval.pdf += (massOf(closure, Technique::CoatSingle) * coat.pdfSingle) + (massOf(closure, Technique::CoatMulti) * coat.pdfMultiReflect);
    }
    return eval;
}

// One-sample MIS: whichever technique drew wi, the throughput divides by the whole mixture's density at it.
std::optional<BsdfSample> weighSample(const BsdfClosure& closure, const glm::vec3& wi, LobeType type) {
    const BsdfEval eval = evaluateSlabs(closure, wi);
    if (!(eval.pdf > 0.0F)) {
        return std::nullopt;
    }
    return BsdfSample{glm::vec3(wi.x, wi.y, wi.z * closure.sign), eval.total() / eval.pdf, type, eval.pdf, false};
}

// A delta technique's sample: its value over its own mass, pdf 0, so NEE has no density to double-count against.
BsdfSample deltaSample(const BsdfClosure& closure, const glm::vec3& wi, const glm::vec3& value, Technique technique, LobeType type) {
    return BsdfSample{glm::vec3(wi.x, wi.y, wi.z * closure.sign), value / massOf(closure, technique), type, 0.0F, true};
}

// Smooth refraction by Snell; its mass is the (1-F) energy, exactly 0 past the critical angle, so this draw never meets TIR.
BsdfSample sampleDeltaRefraction(const BsdfClosure& closure) {
    const DielectricSlab& slab = closure.dielectric;
    const glm::vec3& wo = closure.wo;
    const float eta = slab.etaI / slab.etaT;
    const glm::vec3 wt(-eta * wo.x, -eta * wo.y, -std::sqrt(cos2Transmitted(wo.z, eta)));
    // Non-symmetric radiance compression for camera-originated transport (Veach 1997 sec. 5.2): eta^2 = (etaI/etaT)^2.
    const glm::vec3 value =
        (slab.transmitTint * underCoat(closure, wt)) * (closure.dielectricWeight * slab.refractWeight * slab.transmitSingle * slab.etaSq);
    return deltaSample(closure, wt, value, Technique::DielectricRefract, LobeType::Transmission);
}

std::optional<BsdfSample> sampleTechnique(const BsdfClosure& closure, Technique technique, float uSplit, Sampler& sampler) {
    const glm::vec3& wo = closure.wo;
    const glm::vec3 mirror(-wo.x, -wo.y, wo.z);
    switch (technique) {
        case Technique::CoatSingle: {
            const DielectricSlab& slab = closure.coat;
            const glm::vec3 woCoat = closure.toCoat * wo;
            const glm::mat3 fromCoat = glm::transpose(closure.toCoat);
            if (isSmooth(slab.alpha)) {
                const glm::vec3 value(closure.coatWeight * slab.reflectSingle);
                return deltaSample(closure, fromCoat * glm::vec3(-woCoat.x, -woCoat.y, woCoat.z), value, technique, LobeType::SpecularReflection);
            }
            const std::optional<InterfaceSample> sample = sampleDielectricSingle(slab, woCoat, sampler.next2D(), uSplit);
            return sample ? weighSample(closure, fromCoat * sample->wi, LobeType::SpecularReflection) : std::nullopt;
        }
        case Technique::CoatMulti:
            return weighSample(closure, glm::transpose(closure.toCoat) * sampleEscapeShape(closure.coat.reflectShape, sampler.next2D()),
                               LobeType::SpecularReflection);
        case Technique::MetalSingle: {
            const ConductorSlab& slab = closure.metal;
            if (isSmooth(slab.alpha)) {
                const glm::vec3 value = (closure.metalWeight * slab.scale * underCoat(closure, mirror)) * fresnelF82(wo.z, slab.f0, slab.k);
                return deltaSample(closure, mirror, value, technique, LobeType::SpecularReflection);
            }
            const std::optional<glm::vec3> wi = sampleConductorSingle(slab, wo, sampler.next2D());
            return wi ? weighSample(closure, *wi, LobeType::SpecularReflection) : std::nullopt;
        }
        case Technique::MetalMulti:
            return weighSample(closure, sampleConductorMulti(closure.metal, sampler.next2D()), LobeType::SpecularReflection);
        case Technique::DielectricSingle: {
            const DielectricSlab& slab = closure.dielectric;
            if (isSmooth(slab.alpha)) {
                const glm::vec3 value = (slab.tint * underCoat(closure, mirror)) * (closure.dielectricWeight * slab.reflectSingle);
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

// Fraunhofer d, F and C lines, where V_d = (n_d-1)/(n_F-n_C) is defined: physical constants of the definition, not tuning.
constexpr float kLambdaDNm = 587.56F;
constexpr float kLambdaFNm = 486.13F;
constexpr float kLambdaCNm = 656.27F;

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
                            const glm::vec3& coatNormalLocal) {
    BsdfClosure closure;
    closure.sign = woLocal.z >= 0.0F ? 1.0F : -1.0F;
    closure.wo = glm::vec3(woLocal.x, woLocal.y, woLocal.z * closure.sign);
    closure.exiting = closure.sign < 0.0F && inputs.transmissionWeight > 0.0F;
    closure.metalWeight = inputs.baseMetalness;
    closure.dielectricWeight = 1.0F - inputs.baseMetalness;
    const glm::vec3 baseAlbedo = inputs.baseWeight * inputs.baseColor;
    const float ior = modulatedIor(inputs, heroChannel);
    float roughness = inputs.specularRoughness;
    float fresnelIor = ior;
    closure.coatWeight = inputs.coatWeight;
    if (closure.coatWeight > 0.0F) {
        closure.coatColor = inputs.coatColor;
        closure.coatIor = inputs.coatIor;
        closure.coatRoughness = inputs.coatRoughness;
        // The coat normal crosses the same mirror as wo, so every coat cosine is the unmirrored one.
        closure.toCoat = toFrameAbout(glm::vec3(coatNormalLocal.x, coatNormalLocal.y, coatNormalLocal.z * closure.sign));
        // OpenPBR's coat roughens the base, r' = lerp(r, min(1, r^4 + 2 r_c^4)^(1/4), C): the coat's blur crossed twice, in alpha^2.
        const float r2 = roughness * roughness;
        const float rc2 = inputs.coatRoughness * inputs.coatRoughness;
        roughness = lerp1(roughness, std::sqrt(std::sqrt(std::min(1.0F, (r2 * r2) + (2.0F * rc2 * rc2)))), closure.coatWeight);
        // Under the coat the base meets n_c, its Fresnel ratio n_b/n_c, inverted where n_c > n_b so no TIR appears; the bend is unchanged.
        const float base = dispersedIor(inputs, heroChannel);
        const float coated = inputs.coatIor > base ? inputs.coatIor / base : base / inputs.coatIor;
        fresnelIor = modulatedRatio(lerp1(base, coated, closure.coatWeight), inputs.specularWeight);
    }
    if (closure.metalWeight > 0.0F) {
        addMetal(closure, inputs, baseAlbedo, roughness);
    }
    if (closure.dielectricWeight > 0.0F) {
        addDielectric(closure, inputs, baseAlbedo, roughness, ior, fresnelIor);
    }
    if (closure.coatWeight > 0.0F && !closure.exiting) {
        addCoat(closure, inputs, baseAlbedo, roughness, fresnelIor);
        // The base's selection masses at its weight toward wo: selection only, so one scalar serves reflection and refraction alike.
        const float baseScale = channelMean(underCoat(closure, glm::vec3(-closure.wo.x, -closure.wo.y, closure.wo.z)));
        for (std::size_t t = static_cast<std::size_t>(Technique::MetalSingle); t < kTechniqueCount; ++t) {
            closure.mass[t] *= baseScale;
        }
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
        fresnel += closure.metalWeight * slab.scale * fresnelF82(cosTheta, slab.f0, slab.k);
    }
    if (closure.dielectricWeight > 0.0F) {
        const DielectricSlab& slab = closure.dielectric;
        fresnel += slab.tint * (closure.dielectricWeight * fresnelDielectric(cosTheta, slab.fresnelEtaI, slab.fresnelEtaT));
    }
    return fresnel;
}

// A slab's visible facet cosine dot(wo, h) for one VNDF draw; the macro normal's own when smooth, where the distribution is a delta.
float facetCosine(const glm::vec3& wo, float alpha, glm::vec2 u) {
    // Clamped for rounding at a grazing visible facet: fresnelDielectric reads a negative cosine as the other side of the interface.
    return isSmooth(alpha) ? wo.z : std::max(glm::dot(wo, sampleGGXVNDF(wo, alpha, u)), 0.0F);
}

}  // namespace

glm::vec3 fresnelAtViewAngle(const BsdfClosure& closure, float cosTheta) {
    const glm::vec3 mirror(-closure.wo.x, -closure.wo.y, closure.wo.z);
    glm::vec3 fresnel = underCoat(closure, mirror) * baseFresnel(closure, cosTheta);
    if (massOf(closure, Technique::CoatSingle) > 0.0F) {
        fresnel += glm::vec3(closure.coatWeight * fresnelDielectric(cosTheta, 1.0F, closure.coatIor));
    }
    return fresnel;
}

glm::vec3 fresnelAtMicrofacet(const BsdfClosure& closure, glm::vec2 u) {
    const glm::vec3 mirror(-closure.wo.x, -closure.wo.y, closure.wo.z);
    // One specular_roughness serves both base slabs, so either built slab's alpha is the base's.
    const float alpha = closure.metalWeight > 0.0F ? closure.metal.alpha : closure.dielectric.alpha;
    glm::vec3 fresnel = underCoat(closure, mirror) * baseFresnel(closure, facetCosine(closure.wo, alpha, u));
    if (massOf(closure, Technique::CoatSingle) > 0.0F) {
        const float cosine = facetCosine(closure.toCoat * closure.wo, closure.coat.alpha, u);
        fresnel += glm::vec3(closure.coatWeight * fresnelDielectric(cosine, 1.0F, closure.coatIor));
    }
    return fresnel;
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

}  // namespace pathtracer::scene
