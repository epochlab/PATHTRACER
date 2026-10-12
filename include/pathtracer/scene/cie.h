#pragma once

#include <array>
#include <optional>

#include <glm/glm.hpp>

// CIE 1931 colorimetry: spectral reflectance to linear Rec.709 (OCIO's scene space), per CIE 015:2018.
namespace pathtracer::scene::cie {

// The observer's full tabulated range at its native 1 nm interval, CIE 015:2018's recommended summation interval.
inline constexpr int kLambdaMinNm = 360;
inline constexpr int kLambdaMaxNm = 830;
inline constexpr int kSampleCount = kLambdaMaxNm - kLambdaMinNm + 1;

// A spectral quantity sampled at kLambdaMinNm + i nm.
using Spectrum = std::array<double, kSampleCount>;

[[nodiscard]] constexpr double wavelengthNm(int index) { return kLambdaMinNm + index; }

// One tabulated row: the observer's (xBar, yBar, zBar) and D65's relative power at wavelengthNm(index).
struct TableRow {
    glm::dvec3 colourMatching;
    double d65;
};
[[nodiscard]] TableRow tableRow(int index);

// XYZ to linear Rec.709 from the BT.709 primaries and tabulated D65 white (SMPTE RP 177), so the perfect reflector maps to (1, 1, 1).
[[nodiscard]] const glm::dmat3& xyzToRec709();

// Linear Rec.709 of a reflectance under D65 (Rec.709's white), via CIE 015:2018 tristimulus integration.
[[nodiscard]] glm::dvec3 reflectanceToRec709(const Spectrum& reflectance);

// Linear Rec.709 of a Planckian radiator at unit luminance (Y = 1); nullopt unless kelvin is finite, positive and inside the gamut.
[[nodiscard]] std::optional<glm::dvec3> blackbodyToRec709(double kelvin);

}  // namespace pathtracer::scene::cie
