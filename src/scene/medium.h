#pragma once

#include <glm/glm.hpp>

#include "pathtracer/scene/openpbr.h"
#include "pathtracer/scene/rgb.h"

namespace pathtracer::scene {

// A homogeneous medium per world unit: absorption and scattering coefficients and its Henyey-Greenstein asymmetry.
struct HomogeneousMedium {
    Rgb absorption{0.0F};
    Rgb scattering{0.0F};
    float anisotropy = 0.0F;
};

// A bulk surface bounding an interior a path can enter: transmission or subsurface refracts into it.
[[nodiscard]] inline bool hasInterior(const OpenPbrInputs<Constant>& inputs) {
    return !inputs.geometryThinWalled && (inputs.transmissionWeight > 0.0F || inputs.subsurfaceWeight > 0.0F);
}

[[nodiscard]] inline bool scatters(const HomogeneousMedium& medium) { return !isBlack(medium.scattering); }

// exp(-mu_t d) per channel, exactly 1 at zero extinction or distance and 0 at infinite extinction, where the product is undefined.
[[nodiscard]] Rgb transmittance(const HomogeneousMedium& medium, float distance);

// 1 where subsurface_radius * subsurface_radius_scale is positive; at zero MFP OpenPBR's subsurface is its glossy-diffuse limit.
[[nodiscard]] Rgb subsurfaceGate(const OpenPbrInputs<Constant>& inputs);

// The single-scattering albedo whose semi-infinite medium reflects multipleScatter: van de Hulst's relation inverted (OpenPBR).
[[nodiscard]] Rgb singleScatterAlbedo(const Rgb& multipleScatter, float anisotropy);

// A bulk material's interior (OpenPBR 1.1.1): transmission and subsurface media blended by their shares of the dielectric.
[[nodiscard]] HomogeneousMedium interiorMedium(const OpenPbrInputs<Constant>& inputs);

// Each channel's odds of driving the distance draw, throughput times albedo normalised; uniform where that product is zero.
[[nodiscard]] Rgb channelOdds(const HomogeneousMedium& medium, const Rgb& throughput);

// A collision distance from the odds' mixture of per-channel exponentials (Wilkie et al. 2014); infinite in a clear channel.
[[nodiscard]] float sampleCollision(const HomogeneousMedium& medium, const Rgb& odds, glm::vec2 u);

// A scattering at distance d weighs T sigma_s / sum_c p_c T_c sigma_t,c: the event's one-sample estimate over the mixture.
[[nodiscard]] Rgb collisionWeight(const HomogeneousMedium& medium, const Rgb& odds, float distance);

// The flight passing distance d weighs T / sum_c p_c T_c, the mixture's odds of drawing beyond it.
[[nodiscard]] Rgb passWeight(const HomogeneousMedium& medium, const Rgb& odds, float distance);

}  // namespace pathtracer::scene
