#pragma once

#include <array>
#include <optional>

#include <glm/glm.hpp>

// CIE 1931 colorimetry: spectral reflectance to linear ACEScg (AP1, OCIO's scene space), per CIE 015:2018.
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

// CIE daylight's tabulated interval; its components are interpolated linearly onto the observer's 1 nm (CIE 015:2018).
inline constexpr int kDaylightIntervalNm = 5;
inline constexpr int kDaylightKnotCount = ((kLambdaMaxNm - kLambdaMinNm) / kDaylightIntervalNm) + 1;

// CIE daylight's (S0, S1, S2) at kLambdaMinNm + knot * kDaylightIntervalNm.
[[nodiscard]] glm::dvec3 daylightComponents(int knot);

// XYZ to linear AP1: Bradford-adapts the tabulated D65 white onto the ACES white (TB-2014-004), so the perfect reflector maps to (1, 1, 1).
[[nodiscard]] const glm::dmat3& xyzToScene();

// The luminance (Y of the D65-lit stimulus) of a scene RGB triple as a dot product: the Y row of xyzToScene's inverse, summing to 1.
[[nodiscard]] const glm::vec3& sceneLuminanceWeights();

// OpenEXR-order xy (R, G, B, white) of the scene space's own primaries and white, for tagging written files.
[[nodiscard]] std::array<float, 8> sceneChromaticities();

// Scene RGB of a reflectance under D65, via CIE 015:2018 tristimulus integration.
[[nodiscard]] glm::dvec3 reflectanceToScene(const Spectrum& reflectance);

// The spectrum a kelvin names: a Planckian radiator, or CIE daylight on its own locus just above the Planckian one (CIE 015:2018).
enum class Illuminant { Blackbody, Daylight };

// CIE daylight's defined range; the locus polynomial is fitted only within it (CIE 015:2018).
inline constexpr double kDaylightMinKelvin = 4000.0;
inline constexpr double kDaylightMaxKelvin = 25000.0;

// Scene RGB of the illuminant at kelvin, unit luminance (Y = 1); nullopt outside its domain or the scene gamut.
[[nodiscard]] std::optional<glm::dvec3> kelvinToScene(Illuminant illuminant, double kelvin);

}  // namespace pathtracer::scene::cie
