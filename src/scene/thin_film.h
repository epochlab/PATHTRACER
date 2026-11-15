#pragma once

#include <glm/glm.hpp>

#include "pathtracer/scene/rgb.h"

// OpenPBR's thin film: Belcour & Barla 2017's Airy reflectance, its spectrum integrated exactly against CIE 1931 x D65 in the scene space.
namespace pathtracer::scene {

// A film of thin_film_thickness (nm) and thin_film_ior between an outer medium and a substrate.
struct ThinFilm {
    float thicknessNm;
    float ior;
};

// Reflectance through the film at outer cosine cosTheta onto a substrate of complex index (eta, kappa) per channel, in linear scene space.
[[nodiscard]] Rgb filmReflectance(const ThinFilm& film, float cosTheta, float outerIor, const Rgb& substrateIor,
                                  const Rgb& substrateKappa);

// The same per polarisation, each clamped to [0, 1]: a thin sheet's ladder sums each mode apart before averaging.
struct PolarisedReflectance {
    Rgb parallel;
    Rgb perpendicular;
};
[[nodiscard]] PolarisedReflectance filmReflectancePolarised(const ThinFilm& film, float cosTheta, float outerIor,
                                                            const Rgb& substrateIor, const Rgb& substrateKappa);

// Gulbrandsen 2014's complex index (n, k) of a conductor from its normal-incidence reflectance r and edge tint g, per channel.
struct ComplexIor {
    Rgb eta;
    Rgb kappa;
};
[[nodiscard]] ComplexIor conductorIor(const Rgb& reflectance, const Rgb& edgeTint);

}  // namespace pathtracer::scene
