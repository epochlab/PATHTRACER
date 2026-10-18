#include "medium.h"

#include <cmath>
#include <limits>

#include <glm/gtx/component_wise.hpp>

namespace pathtracer::scene {

namespace {

// van de Hulst's semi-infinite albedo C = (1 - s)(1 - a s)/(1 + b s), s = sqrt((1 - alpha)/(1 - alpha g)) (OpenPBR 1.1.1).
constexpr float kVanDeHulstA = 0.139F;
constexpr float kVanDeHulstB = 1.17F;

// The subsurface medium: mu_t = 1/(radius * scale) per open channel, albedo by van de Hulst; HG at g = 1 is a forward delta, so clear.
HomogeneousMedium subsurfaceMedium(const OpenPbrInputs<Constant>& inputs) {
    const float g = inputs.subsurfaceScatterAnisotropy;
    if (g == 1.0F) {
        return {};
    }
    const glm::vec3 open = subsurfaceGate(inputs);
    // A gated channel takes the glossy-diffuse lobe and never enters, so its coefficients stay 0.
    glm::vec3 extinction(0.0F);
    for (int c = 0; c < 3; ++c) {
        if (open[c] > 0.0F) {
            extinction[c] = 1.0F / (inputs.subsurfaceRadius * inputs.subsurfaceRadiusScale[c]);
        }
    }
    const glm::vec3 albedo = singleScatterAlbedo(inputs.subsurfaceColor, g);
    return {(1.0F - albedo) * extinction, albedo * extinction, g};
}

// The transmission medium: mu_t = -ln(T)/depth, mu_s = S/depth, mu_a = mu_t - mu_s shifted by grey where negative (OpenPBR).
HomogeneousMedium transmissionMedium(const OpenPbrInputs<Constant>& inputs) {
    const float depth = inputs.transmissionDepth;
    const glm::vec3 extinction = -glm::log(inputs.transmissionColor) / depth;
    // HG at g = 1 changes no direction, so its scattering is null collisions: the medium is its absorption alone.
    const glm::vec3 scattering = inputs.transmissionScatterAnisotropy == 1.0F ? glm::vec3(0.0F) : inputs.transmissionScatter / depth;
    glm::vec3 absorption = extinction - (inputs.transmissionScatter / depth);
    absorption -= glm::min(glm::compMin(absorption), 0.0F);
    return {absorption, scattering, inputs.transmissionScatterAnisotropy};
}

}  // namespace

glm::vec3 transmittance(const HomogeneousMedium& medium, float distance) {
    const glm::vec3 extinction = medium.absorption + medium.scattering;
    glm::vec3 result(1.0F);
    for (int c = 0; c < 3; ++c) {
        if (extinction[c] > 0.0F && distance > 0.0F) {
            result[c] = std::exp(-extinction[c] * distance);
        }
    }
    return result;
}

glm::vec3 subsurfaceGate(const OpenPbrInputs<Constant>& inputs) {
    // An MFP whose reciprocal overflows float is no more representable as a medium than zero is: both take the limit.
    const glm::vec3 extinction = 1.0F / (inputs.subsurfaceRadius * inputs.subsurfaceRadiusScale);
    return {std::isfinite(extinction.x) ? 1.0F : 0.0F, std::isfinite(extinction.y) ? 1.0F : 0.0F, std::isfinite(extinction.z) ? 1.0F : 0.0F};
}

glm::vec3 singleScatterAlbedo(const glm::vec3& multipleScatter, float anisotropy) {
    // a s^2 - (1 + a + b C) s + (1 - C) = 0; its root in [0, 1] as 2c/(B + sqrt(B^2 - 4ac)), free of cancellation as C -> 1.
    const glm::vec3 b = 1.0F + kVanDeHulstA + (kVanDeHulstB * multipleScatter);
    const glm::vec3 c = 1.0F - multipleScatter;
    const glm::vec3 s = (2.0F * c) / (b + glm::sqrt((b * b) - (4.0F * kVanDeHulstA * c)));
    return (1.0F - (s * s)) / (1.0F - (anisotropy * s * s));
}

HomogeneousMedium interiorMedium(const OpenPbrInputs<Constant>& inputs) {
    // Adobe's blend (openpbr_prepare_volume) by per-channel share of the light entering; OpenPBR leaves the mix undefined.
    const glm::vec3 transmission(inputs.transmissionWeight);
    const glm::vec3 subsurface = ((1.0F - inputs.transmissionWeight) * inputs.subsurfaceWeight) * subsurfaceGate(inputs);
    const glm::vec3 total = transmission + subsurface;
    HomogeneousMedium blend;
    float meanCosine = 0.0F;
    const auto add = [&](const HomogeneousMedium& medium, const glm::vec3& share) {
        blend.absorption += share * medium.absorption;
        blend.scattering += share * medium.scattering;
        meanCosine += glm::compAdd(share * medium.scattering) * medium.anisotropy;
    };
    // At depth 0 transmission is a surface tint: its medium is clear but keeps its share of total, as in the oracle.
    if (inputs.transmissionWeight > 0.0F && inputs.transmissionDepth > 0.0F) {
        add(transmissionMedium(inputs), transmission / total);
    }
    if (subsurface != glm::vec3(0.0F)) {
        add(subsurfaceMedium(inputs), glm::vec3(subsurface.x > 0.0F ? subsurface.x / total.x : 0.0F,
                                                subsurface.y > 0.0F ? subsurface.y / total.y : 0.0F,
                                                subsurface.z > 0.0F ? subsurface.z / total.z : 0.0F));
    }
    // A phase mixture's mean cosine is its scattering-weighted mean of g: the blend's HG keeps that first moment exactly.
    const float scattering = glm::compAdd(blend.scattering);
    blend.anisotropy = scattering > 0.0F ? meanCosine / scattering : 0.0F;
    return blend;
}

glm::vec3 channelOdds(const HomogeneousMedium& medium, const glm::vec3& throughput) {
    const glm::vec3 extinction = medium.absorption + medium.scattering;
    glm::vec3 drive(0.0F);
    for (int c = 0; c < 3; ++c) {
        if (extinction[c] > 0.0F) {
            drive[c] = throughput[c] * medium.scattering[c] / extinction[c];
        }
    }
    const float total = glm::compAdd(drive);
    return total > 0.0F ? drive / total : glm::vec3(1.0F / 3.0F);
}

float sampleCollision(const HomogeneousMedium& medium, const glm::vec3& odds, glm::vec2 u) {
    // The channel whose cumulative odds first exceed u.x; past the float sum, the last with positive odds.
    int channel = 0;
    float below = 0.0F;
    for (int c = 0; c < 3; ++c) {
        if (!(odds[c] > 0.0F)) {
            continue;
        }
        channel = c;
        if (u.x < below + odds[c]) {
            break;
        }
        below += odds[c];
    }
    const float extinction = medium.absorption[channel] + medium.scattering[channel];
    return extinction > 0.0F ? -std::log1p(-u.y) / extinction : std::numeric_limits<float>::infinity();
}

glm::vec3 collisionWeight(const HomogeneousMedium& medium, const glm::vec3& odds, float distance) {
    const glm::vec3 extinction = medium.absorption + medium.scattering;
    const glm::vec3 surviving = transmittance(medium, distance);
    // Zero terms are skipped, not formed: 0 * inf reads NaN where T sigma_t's limit is 0 (an unchosen or opaque channel).
    float density = 0.0F;
    for (int c = 0; c < 3; ++c) {
        if (odds[c] > 0.0F && surviving[c] > 0.0F) {
            density += odds[c] * surviving[c] * extinction[c];
        }
    }
    return surviving * medium.scattering / density;
}

glm::vec3 passWeight(const HomogeneousMedium& medium, const glm::vec3& odds, float distance) {
    const glm::vec3 surviving = transmittance(medium, distance);
    return surviving / glm::dot(odds, surviving);
}

}  // namespace pathtracer::scene
