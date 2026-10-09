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

float& massOf(BsdfClosure& closure, Technique technique) { return closure.mass[static_cast<std::size_t>(technique)]; }

float massOf(const BsdfClosure& closure, Technique technique) { return closure.mass[static_cast<std::size_t>(technique)]; }

// The metal: base_metalness of the mix, its Fresnel tinted by specular_color and scaled by specular_weight.
void addMetal(BsdfClosure& closure, const OpenPbrInputs<Constant>& inputs, const glm::vec3& baseAlbedo) {
    closure.metal = makeConductorSlab(inputs.specularRoughness, baseAlbedo, inputs.specularColor, inputs.specularWeight, closure.wo.z);
    massOf(closure, Technique::MetalSingle) = closure.metalWeight * closure.metal.singleEnergy;
    massOf(closure, Technique::MetalMulti) = closure.metalWeight * closure.metal.multiEnergy;
}

// The dielectric: mix(glossy-diffuse, translucent, T) sharing one interface, the diffuse under it by OpenPBR's albedo scaling.
void addDielectric(BsdfClosure& closure, const OpenPbrInputs<Constant>& inputs, const glm::vec3& baseAlbedo, float ior) {
    // Only the interface faces a ray from inside: the translucent share is whole there, with no diffuse and no specular_color tint.
    const bool exiting = closure.sign < 0.0F && inputs.transmissionWeight > 0.0F;
    // OpenPBR's two exclusive regimes: at transmission_depth > 0 the interior medium carries the colour; at 0 it tints the surface.
    const glm::vec3 transmitTint = inputs.transmissionDepth > 0.0F ? glm::vec3(1.0F) : inputs.transmissionColor;
    const DielectricSlab& slab = closure.dielectric =
        makeDielectricSlab(inputs.specularRoughness, exiting ? ior : 1.0F, exiting ? 1.0F : ior, exiting ? glm::vec3(1.0F) : inputs.specularColor,
                           exiting ? 1.0F : inputs.transmissionWeight, transmitTint, closure.wo.z);
    const float weight = closure.dielectricWeight;
    const bool delta = isIndexMatched(slab) || isSmooth(slab.alpha);
    const float refracted = slab.refractWeight * slab.transmitSingle;
    massOf(closure, Technique::DielectricSingle) =
        weight * ((slab.tintMean * slab.reflectSingle) + (delta ? 0.0F : refracted));
    massOf(closure, Technique::DielectricRefract) = delta ? weight * refracted : 0.0F;
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

// Every slab's continuous value and the mixture density at wi, wo's hemisphere being +z.
BsdfEval evaluateSlabs(const BsdfClosure& closure, const glm::vec3& wi) {
    BsdfEval eval{};
    const glm::vec3& wo = closure.wo;
    if (closure.metalWeight > 0.0F && wi.z > 0.0F) {
        const ConductorEval metal = evaluateConductor(closure.metal, wo, wi);
        eval.specular += closure.metalWeight * metal.value;
        eval.pdf += (massOf(closure, Technique::MetalSingle) * metal.pdfSingle) + (massOf(closure, Technique::MetalMulti) * metal.pdfMulti);
    }
    if (closure.dielectricWeight > 0.0F) {
        const DielectricEval dielectric = evaluateDielectric(closure.dielectric, wo, wi);
        eval.specular += closure.dielectricWeight * dielectric.reflect;
        eval.transmission += closure.dielectricWeight * dielectric.transmit;
        eval.pdf += (massOf(closure, Technique::DielectricSingle) * dielectric.pdfSingle) +
                    (massOf(closure, Technique::DielectricMultiReflect) * dielectric.pdfMultiReflect) +
                    (massOf(closure, Technique::DielectricMultiTransmit) * dielectric.pdfMultiTransmit);
    }
    if (massOf(closure, Technique::Diffuse) > 0.0F) {
        const DiffuseEval diffuse = evaluateDiffuse(closure.diffuse, wo, wi);
        eval.diffuse = closure.diffuseWeight * diffuse.value;
        eval.pdf += massOf(closure, Technique::Diffuse) * diffuse.pdf;
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
    const glm::vec3 value = slab.transmitTint * (closure.dielectricWeight * slab.refractWeight * slab.transmitSingle * slab.etaSq);
    return deltaSample(closure, wt, value, Technique::DielectricRefract, LobeType::Transmission);
}

std::optional<BsdfSample> sampleTechnique(const BsdfClosure& closure, Technique technique, float uSplit, Sampler& sampler) {
    const glm::vec3& wo = closure.wo;
    const glm::vec3 mirror(-wo.x, -wo.y, wo.z);
    switch (technique) {
        case Technique::MetalSingle: {
            const ConductorSlab& slab = closure.metal;
            if (isSmooth(slab.alpha)) {
                const glm::vec3 value = closure.metalWeight * slab.scale * fresnelF82(wo.z, slab.f0, slab.k);
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
                const glm::vec3 value = slab.tint * (closure.dielectricWeight * slab.reflectSingle);
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
    // Dispersion enters here alone: every downstream ior consumer reads this one scalar, so the vertex stays spectrally consistent.
    const float abbe = inputs.transmissionDispersionAbbeNumber / inputs.transmissionDispersionScale;
    const float dispersed = heroChannel.has_value() ? cauchyIor(inputs.specularIor, abbe, kRgbWavelengthsNm[*heroChannel]) : inputs.specularIor;
    // OpenPBR's specular_weight: F0 = xi*F_s, capped at the largest float below 1 so the modulated ratio (1+eps)/(1-eps) stays finite.
    const float reflectance = ((dispersed - 1.0F) / (dispersed + 1.0F)) * ((dispersed - 1.0F) / (dispersed + 1.0F));
    const float epsilon =
        std::copysign(std::sqrt(std::min(inputs.specularWeight * reflectance, std::nextafter(1.0F, 0.0F))), dispersed - 1.0F);
    // At xi = 1 the modulation is the identity; taking the index as authored keeps it free of the ratio's rounding.
    return inputs.specularWeight == 1.0F ? dispersed : (1.0F + epsilon) / (1.0F - epsilon);
}

BsdfClosure makeBsdfClosure(const OpenPbrInputs<Constant>& inputs, const glm::vec3& woLocal, std::optional<int> heroChannel) {
    BsdfClosure closure{};
    closure.sign = woLocal.z >= 0.0F ? 1.0F : -1.0F;
    closure.wo = glm::vec3(woLocal.x, woLocal.y, woLocal.z * closure.sign);
    closure.metalWeight = inputs.baseMetalness;
    closure.dielectricWeight = 1.0F - inputs.baseMetalness;
    const glm::vec3 baseAlbedo = inputs.baseWeight * inputs.baseColor;
    if (closure.metalWeight > 0.0F) {
        addMetal(closure, inputs, baseAlbedo);
    }
    if (closure.dielectricWeight > 0.0F) {
        addDielectric(closure, inputs, baseAlbedo, modulatedIor(inputs, heroChannel));
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

glm::vec3 fresnelAtViewAngle(const BsdfClosure& closure, float cosTheta) {
    glm::vec3 fresnel(0.0F);
    if (closure.metalWeight > 0.0F) {
        const ConductorSlab& slab = closure.metal;
        fresnel += closure.metalWeight * slab.scale * fresnelF82(cosTheta, slab.f0, slab.k);
    }
    if (closure.dielectricWeight > 0.0F) {
        const DielectricSlab& slab = closure.dielectric;
        fresnel += slab.tint * (closure.dielectricWeight * fresnelDielectric(cosTheta, slab.etaI, slab.etaT));
    }
    return fresnel;
}

glm::vec3 fresnelAtMicrofacet(const BsdfClosure& closure, glm::vec2 u) {
    // One specular_roughness serves both slabs, so either built slab's alpha is the base's.
    const float alpha = closure.metalWeight > 0.0F ? closure.metal.alpha : closure.dielectric.alpha;
    // A smooth surface's only facet is the macro normal, where the visible-normal distribution collapses to a delta.
    if (isSmooth(alpha)) {
        return fresnelAtViewAngle(closure, closure.wo.z);
    }
    // Clamped for rounding at a grazing visible facet: fresnelDielectric reads a negative cosine as the other side of the interface.
    return fresnelAtViewAngle(closure, std::max(glm::dot(closure.wo, sampleGGXVNDF(closure.wo, alpha, u)), 0.0F));
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
