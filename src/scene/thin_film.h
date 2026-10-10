#pragma once

#include <glm/glm.hpp>

// OpenPBR's thin film: Belcour & Barla 2017's Airy reflectance, its spectrum integrated exactly against CIE 1931 x D65 in Rec.709.
namespace pathtracer::scene {

// A film of thin_film_thickness (nm) and thin_film_ior between an outer medium and a substrate.
struct ThinFilm {
    float thicknessNm;
    float ior;
};

// Reflectance through the film at outer cosine cosTheta onto a substrate of complex index (eta, kappa) per channel, in linear Rec.709.
[[nodiscard]] glm::vec3 filmReflectance(const ThinFilm& film, float cosTheta, float outerIor, const glm::vec3& substrateIor,
                                        const glm::vec3& substrateKappa);

// The same per polarisation, each clamped to [0, 1]: a thin sheet's ladder sums each mode apart before averaging.
struct PolarisedReflectance {
    glm::vec3 parallel;
    glm::vec3 perpendicular;
};
[[nodiscard]] PolarisedReflectance filmReflectancePolarised(const ThinFilm& film, float cosTheta, float outerIor,
                                                            const glm::vec3& substrateIor, const glm::vec3& substrateKappa);

// Gulbrandsen 2014's complex index (n, k) of a conductor from its normal-incidence reflectance r and edge tint g, per channel.
struct ComplexIor {
    glm::vec3 eta;
    glm::vec3 kappa;
};
[[nodiscard]] ComplexIor conductorIor(const glm::vec3& reflectance, const glm::vec3& edgeTint);

}  // namespace pathtracer::scene
