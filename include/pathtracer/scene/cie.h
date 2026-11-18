#pragma once

#include <array>
#include <optional>
#include <span>
#include <string_view>

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

// 1 at every sample: a clear filter's transmittance, which multiplies a spectrum exactly.
inline constexpr Spectrum kUnitSpectrum = [] {
    Spectrum unit{};
    unit.fill(1.0);
    return unit;
}();

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

// XYZ to linear AP1 with Bradford adaptation of `white` (XYZ, Y = 1) onto the ACES white (TB-2014-004): that white maps to (1, 1, 1).
[[nodiscard]] glm::dmat3 xyzToSceneFrom(const glm::dvec3& white);

// xyzToSceneFrom the tabulated D65 white: the scene's own matrix, so the perfect reflector under D65 maps to (1, 1, 1).
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

// The yBar-weighted sum of power over the samples: its luminance up to K_m and the 1 nm interval.
[[nodiscard]] double luminanceSum(const Spectrum& power);

// (nm, value) samples at strictly increasing nm, two or more: linear between them, zero outside (the CIE datasets' convention).
[[nodiscard]] Spectrum resample(std::span<const glm::dvec2> samples);

// Sprague's quintic reads six samples about each interval, so fewer cannot seed it.
inline constexpr std::size_t kSpragueStencil = 6;

// Sprague (1880) quintic through kSpragueStencil or more uniform samples (CIE 167:2005); nearest value beyond their ends (CIE 015:2018).
[[nodiscard]] Spectrum resampleSprague(double firstNm, double intervalNm, std::span<const double> samples);

// A CIE 015:2018 tabulated lamp by CIE name: FL1-FL12, FL3.1-FL3.15, HP1-HP5, LED-B1-B5, LED-BH1, LED-RGB1, LED-V1, LED-V2.
[[nodiscard]] std::optional<Spectrum> lampSpectrum(std::string_view name);

// The illuminant's relative spectral power at kelvin; nullopt outside its domain or where Planck's law underflows to no luminance.
[[nodiscard]] std::optional<Spectrum> illuminantSpectrum(Illuminant illuminant, double kelvin);

}  // namespace pathtracer::scene::cie
