#include "medium.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pathtracer::scene {

namespace {

// van de Hulst's semi-infinite albedo C = (1 - s)(1 - a s)/(1 + b s), s = sqrt((1 - alpha)/(1 - alpha g)) (OpenPBR 1.1.1).
constexpr float kVanDeHulstA = 0.139F;
constexpr float kVanDeHulstB = 1.17F;

// The subsurface medium: mu_t = 1/(radius * scale) per open lane, albedo by van de Hulst; HG at g = 1 is a forward delta, so clear.
HomogeneousMedium subsurfaceMedium(const OpenPbrInputs<Spectral>& inputs) {
    const float g = inputs.subsurfaceScatterAnisotropy;
    if (g == 1.0F) {
        return {};
    }
    const SampledSpectrum open = subsurfaceGate(inputs);
    // A gated lane takes the glossy-diffuse lobe and never enters, so its coefficients stay 0.
    SampledSpectrum extinction(0.0F);
    for (int c = 0; c < kSpectrumSamples; ++c) {
        if (open[c] > 0.0F) {
            extinction[c] = 1.0F / (inputs.subsurfaceRadius * inputs.subsurfaceRadiusScale[c]);
        }
    }
    const SampledSpectrum albedo = singleScatterAlbedo(inputs.subsurfaceColor, g);
    return {(1.0F - albedo) * extinction, albedo * extinction, g};
}

// The transmission medium: mu_t = -ln(T)/depth, mu_s = S/depth, mu_a = mu_t - mu_s shifted by grey where negative (OpenPBR).
HomogeneousMedium transmissionMedium(const OpenPbrInputs<Spectral>& inputs) {
    const float depth = inputs.transmissionDepth;
    const SampledSpectrum extinction = -log(inputs.transmissionColor) / depth;
    // HG at g = 1 changes no direction, so its scattering is null collisions: the medium is its absorption alone.
    const SampledSpectrum scattering = inputs.transmissionScatterAnisotropy == 1.0F ? SampledSpectrum(0.0F) : inputs.transmissionScatter / depth;
    SampledSpectrum absorption = extinction - (inputs.transmissionScatter / depth);
    absorption -= std::min(minChannel(absorption), 0.0F);
    return {absorption, scattering, inputs.transmissionScatterAnisotropy};
}

}  // namespace

SampledSpectrum transmittance(const HomogeneousMedium& medium, float distance) {
    const SampledSpectrum extinction = medium.absorption + medium.scattering;
    SampledSpectrum result(1.0F);
    for (int c = 0; c < kSpectrumSamples; ++c) {
        if (extinction[c] > 0.0F && distance > 0.0F) {
            result[c] = std::exp(-extinction[c] * distance);
        }
    }
    return result;
}

SampledSpectrum subsurfaceGate(const OpenPbrInputs<Spectral>& inputs) {
    // An MFP whose reciprocal overflows float is no more representable as a medium than zero is: both take the limit.
    const SampledSpectrum extinction = 1.0F / (inputs.subsurfaceRadius * inputs.subsurfaceRadiusScale);
    SampledSpectrum open(0.0F);
    for (int c = 0; c < kSpectrumSamples; ++c) {
        open[c] = std::isfinite(extinction[c]) ? 1.0F : 0.0F;
    }
    return open;
}

SampledSpectrum singleScatterAlbedo(const SampledSpectrum& multipleScatter, float anisotropy) {
    // a s^2 - (1 + a + b C) s + (1 - C) = 0; its root in [0, 1] as 2c/(B + sqrt(B^2 - 4ac)), free of cancellation as C -> 1.
    const SampledSpectrum b = 1.0F + kVanDeHulstA + (kVanDeHulstB * multipleScatter);
    const SampledSpectrum c = 1.0F - multipleScatter;
    const SampledSpectrum s = (2.0F * c) / (b + sqrt((b * b) - (4.0F * kVanDeHulstA * c)));
    return (1.0F - (s * s)) / (1.0F - (anisotropy * s * s));
}

HomogeneousMedium interiorMedium(const OpenPbrInputs<Spectral>& inputs) {
    // Adobe's blend (openpbr_prepare_volume) by per-lane share of the light entering; OpenPBR leaves the mix undefined.
    const SampledSpectrum transmission(inputs.transmissionWeight);
    const SampledSpectrum subsurface = ((1.0F - inputs.transmissionWeight) * inputs.subsurfaceWeight) * subsurfaceGate(inputs);
    const SampledSpectrum total = transmission + subsurface;
    HomogeneousMedium blend;
    float meanCosine = 0.0F;
    const auto add = [&](const HomogeneousMedium& medium, const SampledSpectrum& share) {
        blend.absorption += share * medium.absorption;
        blend.scattering += share * medium.scattering;
        meanCosine += channelSum(share * medium.scattering) * medium.anisotropy;
    };
    // At depth 0 transmission is a surface tint: its medium is clear but keeps its share of total, as in the oracle.
    if (inputs.transmissionWeight > 0.0F && inputs.transmissionDepth > 0.0F) {
        add(transmissionMedium(inputs), transmission / total);
    }
    if (!isBlack(subsurface)) {
        SampledSpectrum share(0.0F);
        for (int c = 0; c < kSpectrumSamples; ++c) {
            share[c] = subsurface[c] > 0.0F ? subsurface[c] / total[c] : 0.0F;
        }
        add(subsurfaceMedium(inputs), share);
    }
    // A phase mixture's mean cosine is its scattering-weighted mean of g: the blend's HG keeps that first moment exactly.
    const float scattering = channelSum(blend.scattering);
    blend.anisotropy = scattering > 0.0F ? meanCosine / scattering : 0.0F;
    return blend;
}

SampledSpectrum channelOdds(const HomogeneousMedium& medium, const SampledSpectrum& throughput) {
    const SampledSpectrum extinction = medium.absorption + medium.scattering;
    SampledSpectrum drive(0.0F);
    for (int c = 0; c < kSpectrumSamples; ++c) {
        if (extinction[c] > 0.0F) {
            drive[c] = throughput[c] * medium.scattering[c] / extinction[c];
        }
    }
    const float total = channelSum(drive);
    return total > 0.0F ? drive / total : SampledSpectrum(1.0F / kSpectrumSamples);
}

int sampleLane(const SampledSpectrum& odds, float u) {
    int lane = 0;
    float below = 0.0F;
    for (int c = 0; c < kSpectrumSamples; ++c) {
        if (!(odds[c] > 0.0F)) {
            continue;
        }
        lane = c;
        if (u < below + odds[c]) {
            break;
        }
        below += odds[c];
    }
    return lane;
}

float sampleCollision(const HomogeneousMedium& medium, const SampledSpectrum& odds, glm::vec2 u) {
    const int lane = sampleLane(odds, u.x);
    const float extinction = medium.absorption[lane] + medium.scattering[lane];
    return extinction > 0.0F ? -std::log1p(-u.y) / extinction : std::numeric_limits<float>::infinity();
}

SampledSpectrum collisionWeight(const SampledSpectrum& odds, const SampledSpectrum& surviving, const SampledSpectrum& scattering,
                                const SampledSpectrum& extinction) {
    // Zero terms are skipped, not formed: 0 * inf reads NaN where T sigma_t's limit is 0 (an unchosen or opaque lane).
    float density = 0.0F;
    for (int c = 0; c < kSpectrumSamples; ++c) {
        if (odds[c] > 0.0F && surviving[c] > 0.0F) {
            density += odds[c] * surviving[c] * extinction[c];
        }
    }
    return surviving * scattering / density;
}

SampledSpectrum passWeight(const SampledSpectrum& odds, const SampledSpectrum& surviving) {
    return surviving / channelSum(odds * surviving);
}

}  // namespace pathtracer::scene
