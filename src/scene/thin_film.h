#pragma once

#include <glm/glm.hpp>

#include "pathtracer/scene/spectrum.h"

// OpenPBR's thin film: Belcour & Barla 2017's Airy reflectance, coherent and exact at each of the path's wavelengths.
namespace pathtracer::scene {

// A film of thin_film_thickness (nm) and thin_film_ior between an outer medium and a substrate, seen at the lanes' wavelengths.
struct ThinFilm {
    float thicknessNm;
    float ior;
    glm::vec4 wavelengthNm;
};

// Reflectance through the film at outer cosine cosTheta onto a substrate of complex index (eta, kappa), per lane.
[[nodiscard]] SampledSpectrum filmReflectance(const ThinFilm& film, float cosTheta, float outerIor, const SampledSpectrum& substrateIor,
                                  const SampledSpectrum& substrateKappa);

// The same per polarisation, each clamped to [0, 1]: a thin sheet's ladder sums each mode apart before averaging.
struct PolarisedReflectance {
    SampledSpectrum parallel;
    SampledSpectrum perpendicular;
};
[[nodiscard]] PolarisedReflectance filmReflectancePolarised(const ThinFilm& film, float cosTheta, float outerIor,
                                                            const SampledSpectrum& substrateIor, const SampledSpectrum& substrateKappa);

// Gulbrandsen 2014's complex index (n, k) of a conductor from its normal-incidence reflectance r and edge tint g, per lane.
struct ComplexIor {
    SampledSpectrum eta;
    SampledSpectrum kappa;
};
[[nodiscard]] ComplexIor conductorIor(const SampledSpectrum& reflectance, const SampledSpectrum& edgeTint);

}  // namespace pathtracer::scene
