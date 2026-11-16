#pragma once

#include <glm/glm.hpp>

#include "pathtracer/scene/openpbr.h"
#include "pathtracer/scene/spectrum.h"

namespace pathtracer::scene {

// A homogeneous medium per world unit: absorption and scattering coefficients and its Henyey-Greenstein asymmetry.
struct HomogeneousMedium {
    SampledSpectrum absorption{0.0F};
    SampledSpectrum scattering{0.0F};
    float anisotropy = 0.0F;
};

// A bulk surface bounding an interior a path can enter: transmission or subsurface refracts into it.
[[nodiscard]] inline bool hasInterior(const OpenPbrInputs<Spectral>& inputs) {
    return !inputs.geometryThinWalled && (inputs.transmissionWeight > 0.0F || inputs.subsurfaceWeight > 0.0F);
}

[[nodiscard]] inline bool scatters(const HomogeneousMedium& medium) { return !isBlack(medium.scattering); }

// exp(-mu_t d) per channel, exactly 1 at zero extinction or distance and 0 at infinite extinction, where the product is undefined.
[[nodiscard]] SampledSpectrum transmittance(const HomogeneousMedium& medium, float distance);

// 1 where subsurface_radius * subsurface_radius_scale is positive; at zero MFP OpenPBR's subsurface is its glossy-diffuse limit.
[[nodiscard]] SampledSpectrum subsurfaceGate(const OpenPbrInputs<Spectral>& inputs);

// The single-scattering albedo whose semi-infinite medium reflects multipleScatter: van de Hulst's relation inverted (OpenPBR).
[[nodiscard]] SampledSpectrum singleScatterAlbedo(const SampledSpectrum& multipleScatter, float anisotropy);

// A bulk material's interior (OpenPBR 1.1.1): transmission and subsurface media blended by their shares of the dielectric.
[[nodiscard]] HomogeneousMedium interiorMedium(const OpenPbrInputs<Spectral>& inputs);

// Each channel's odds of driving the distance draw, throughput times albedo normalised; uniform where that product is zero.
[[nodiscard]] SampledSpectrum channelOdds(const HomogeneousMedium& medium, const SampledSpectrum& throughput);

// The lane whose cumulative odds first exceed u; past the float sum, the last with positive odds.
[[nodiscard]] int sampleLane(const SampledSpectrum& odds, float u);

// A collision distance from the odds' mixture of per-channel exponentials (Wilkie et al. 2014); infinite in a clear channel.
[[nodiscard]] float sampleCollision(const HomogeneousMedium& medium, const SampledSpectrum& odds, glm::vec2 u);

// A scattering where T survives weighs T sigma_s / sum_c p_c T_c sigma_t,c: the event's one-sample estimate over the mixture.
[[nodiscard]] SampledSpectrum collisionWeight(const SampledSpectrum& odds, const SampledSpectrum& surviving,
                                              const SampledSpectrum& scattering, const SampledSpectrum& extinction);

// The flight passing with T surviving weighs T / sum_c p_c T_c, the mixture's odds of drawing beyond it.
[[nodiscard]] SampledSpectrum passWeight(const SampledSpectrum& odds, const SampledSpectrum& surviving);

}  // namespace pathtracer::scene
