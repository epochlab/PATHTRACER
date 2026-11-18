#include "pathtracer/scene/cie.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace pathtracer::scene::cie {

namespace {

struct Sample {
    double xBar;
    double yBar;
    double zBar;
    double d65;
};

constexpr std::array<Sample, kSampleCount> kTable = {{
#include "cie_1931.inc"
}};

struct DaylightComponents {
    double s0;
    double s1;
    double s2;
};

constexpr std::array<DaylightComponents, kDaylightKnotCount> kDaylight = {{
#include "cie_daylight.inc"
}};

// CIE 015:2018 lamp tables, rows {nm, one value per lamp}: fluorescent (Table 10) and LED (Table 12) at 1 nm, high pressure at 5 nm.
constexpr std::array<std::array<double, 28>, 401> kFluorescent = {{
#include "cie_fl.inc"
}};
constexpr std::array<std::array<double, 10>, 401> kLed = {{
#include "cie_led.inc"
}};
constexpr std::array<std::array<double, 6>, 81> kHighPressure = {{
#include "cie_hp.inc"
}};

// Each table's lamp names: its CIE metadata's column titles after lambda, in column order.
constexpr std::array<std::string_view, 27> kFluorescentNames = {
    "FL1", "FL2", "FL3", "FL4", "FL5", "FL6", "FL7", "FL8", "FL9", "FL10", "FL11", "FL12", "FL3.1",
    "FL3.2", "FL3.3", "FL3.4", "FL3.5", "FL3.6", "FL3.7", "FL3.8", "FL3.9", "FL3.10", "FL3.11", "FL3.12", "FL3.13", "FL3.14", "FL3.15"};
constexpr std::array<std::string_view, 9> kLedNames = {
    "LED-B1", "LED-B2", "LED-B3", "LED-B4", "LED-B5", "LED-BH1", "LED-RGB1", "LED-V1", "LED-V2"};
constexpr std::array<std::string_view, 5> kHighPressureNames = {"HP1", "HP2", "HP3", "HP4", "HP5"};

// ACEScg's AP1 primary and white chromaticities (x, y), SMPTE ST 2065-1 and ACES S-2014-004.
constexpr glm::dvec2 kScenePrimaryRed(0.713, 0.293);
constexpr glm::dvec2 kScenePrimaryGreen(0.165, 0.830);
constexpr glm::dvec2 kScenePrimaryBlue(0.128, 0.044);
constexpr glm::dvec2 kSceneWhite(0.32168, 0.33767);

// Bradford's sharpened cone response (Lam 1985), the von Kries space ACES specifies for white-point adaptation.
const glm::dmat3 kBradford(0.8951, -0.7502, 0.0389, 0.2664, 1.7135, -0.0685, -0.1614, 0.0367, 1.0296);

// XYZ of chromaticity xy at Y = 1.
glm::dvec3 xyzAtUnitLuminance(const glm::dvec2& xy) { return {xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y}; }

// D65-weighted yBar sum, the normaliser giving the perfect reflector Y = 1 (CIE 015:2018 eq 7.3's k); the 1 nm interval cancels.
double whiteLuminanceSum() {
    double sum = 0.0;
    for (const Sample& s : kTable) {
        sum += s.d65 * s.yBar;
    }
    return sum;
}

// Tristimulus values of a reflectance under D65, normalised so the perfect reflector has Y = 1.
glm::dvec3 reflectanceToXyz(const Spectrum& reflectance) {
    static const double kNormaliser = 1.0 / whiteLuminanceSum();
    glm::dvec3 xyz(0.0);
    for (int i = 0; i < kSampleCount; ++i) {
        const Sample& s = kTable[i];
        xyz += (reflectance[i] * s.d65) * glm::dvec3(s.xBar, s.yBar, s.zBar);
    }
    return xyz * kNormaliser;
}

// The von Kries scaling of Bradford cone responses taking white `from` to white `to`.
glm::dmat3 bradfordAdaptation(const glm::dvec3& from, const glm::dvec3& to) {
    const glm::dvec3 gain = (kBradford * to) / (kBradford * from);
    return glm::inverse(kBradford) * glm::dmat3(gain.x, 0.0, 0.0, 0.0, gain.y, 0.0, 0.0, 0.0, gain.z) * kBradford;
}

}  // namespace

// SMPTE RP 177 eq 1-4: scale each primary's unit-luminance column so the three sum to the scene white, then invert; white adapts first.
glm::dmat3 xyzToSceneFrom(const glm::dvec3& white) {
    const glm::dmat3 primaries(xyzAtUnitLuminance(kScenePrimaryRed), xyzAtUnitLuminance(kScenePrimaryGreen),
                               xyzAtUnitLuminance(kScenePrimaryBlue));
    const glm::dvec3 sceneWhite = xyzAtUnitLuminance(kSceneWhite);
    const glm::dvec3 scale = glm::inverse(primaries) * sceneWhite;
    const glm::dmat3 rgbToXyz = primaries * glm::dmat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
    return glm::inverse(rgbToXyz) * bradfordAdaptation(white, sceneWhite);
}

TableRow tableRow(int index) {
    const Sample& s = kTable[index];
    return {{s.xBar, s.yBar, s.zBar}, s.d65};
}

glm::dvec3 daylightComponents(int knot) {
    const DaylightComponents& c = kDaylight[knot];
    return {c.s0, c.s1, c.s2};
}

const glm::dmat3& xyzToScene() {
    static const glm::dmat3 kMatrix = [] {
        Spectrum perfectReflector;
        perfectReflector.fill(1.0);
        return xyzToSceneFrom(reflectanceToXyz(perfectReflector));
    }();
    return kMatrix;
}

const glm::vec3& sceneLuminanceWeights() {
    static const glm::vec3 kWeights = [] {
        const glm::dmat3 sceneToXyz = glm::inverse(xyzToScene());
        return glm::vec3(sceneToXyz[0][1], sceneToXyz[1][1], sceneToXyz[2][1]);
    }();
    return kWeights;
}

std::array<float, 8> sceneChromaticities() {
    const std::array<glm::dvec2, 4> xy = {kScenePrimaryRed, kScenePrimaryGreen, kScenePrimaryBlue, kSceneWhite};
    return {static_cast<float>(xy[0].x), static_cast<float>(xy[0].y), static_cast<float>(xy[1].x), static_cast<float>(xy[1].y),
            static_cast<float>(xy[2].x), static_cast<float>(xy[2].y), static_cast<float>(xy[3].x), static_cast<float>(xy[3].y)};
}

glm::dvec3 reflectanceToScene(const Spectrum& reflectance) { return xyzToScene() * reflectanceToXyz(reflectance); }

namespace {

// Planck's law at each tabulated wavelength; c1 and the 1 nm interval cancel in a light's unit-luminance normalisation.
Spectrum blackbodySpectrum(double kelvin) {
    // Planck's second radiation constant, CIE 015:2018's ITS-90 value.
    constexpr double kC2 = 1.4388e-2;
    Spectrum power;
    for (int i = 0; i < kSampleCount; ++i) {
        const double lambda = wavelengthNm(i) * 1e-9;
        power[i] = 1.0 / (std::pow(lambda, 5) * std::expm1(kC2 / (lambda * kelvin)));
    }
    return power;
}

// CIE 015:2018 daylight: S0 + M1 S1 + M2 S2 at the locus xy; M1, M2 unrounded so the colour is continuous in kelvin.
Spectrum daylightSpectrum(double kelvin) {
    const double t = 1e3 / kelvin;
    const double x = kelvin <= 7000.0 ? 0.244063 + (t * (0.09911 + (t * (2.9678 - (t * 4.6070)))))
                                      : 0.237040 + (t * (0.24748 + (t * (1.9018 - (t * 2.0064)))));
    const double y = (-3.000 * x * x) + (2.870 * x) - 0.275;
    const double m = 0.0241 + (0.2562 * x) - (0.7341 * y);
    const double m1 = (-1.3515 - (1.7703 * x) + (5.9114 * y)) / m;
    const double m2 = (0.0300 - (31.4424 * x) + (30.0717 * y)) / m;
    Spectrum power;
    for (int i = 0; i < kSampleCount; ++i) {
        const int knot = std::min(i / kDaylightIntervalNm, kDaylightKnotCount - 2);
        const double f = static_cast<double>(i - (knot * kDaylightIntervalNm)) / kDaylightIntervalNm;
        const DaylightComponents& a = kDaylight[knot];
        const DaylightComponents& b = kDaylight[knot + 1];
        power[i] = ((1.0 - f) * (a.s0 + (m1 * a.s1) + (m2 * a.s2))) + (f * (b.s0 + (m1 * b.s1) + (m2 * b.s2)));
    }
    return power;
}

// The named lamp's column of a CIE table resampled onto the observer's samples; nullopt when the table has no such lamp.
template <std::size_t Rows, std::size_t Fields>
std::optional<Spectrum> lampColumn(const std::array<std::array<double, Fields>, Rows>& table,
                                   const std::array<std::string_view, Fields - 1>& names, std::string_view name) {
    const auto it = std::ranges::find(names, name);
    if (it == names.end()) {
        return std::nullopt;
    }
    const auto field = static_cast<std::size_t>(it - names.begin()) + 1;
    std::array<glm::dvec2, Rows> samples;
    for (std::size_t row = 0; row < Rows; ++row) {
        samples[row] = {table[row][0], table[row][field]};
    }
    return resample(samples);
}

}  // namespace

double luminanceSum(const Spectrum& power) {
    double sum = 0.0;
    for (int i = 0; i < kSampleCount; ++i) {
        sum += power[static_cast<std::size_t>(i)] * kTable[static_cast<std::size_t>(i)].yBar;
    }
    return sum;
}

Spectrum resample(std::span<const glm::dvec2> samples) {
    Spectrum out{};
    std::size_t k = 0;
    for (int i = 0; i < kSampleCount; ++i) {
        const double nm = wavelengthNm(i);
        if (nm < samples.front().x || nm > samples.back().x) {
            continue;
        }
        // Wavelengths ascend, so the bracketing segment only moves forward: one pass over both grids.
        while (samples[k + 1].x < nm) {
            ++k;
        }
        const glm::dvec2& a = samples[k];
        const glm::dvec2& b = samples[k + 1];
        out[static_cast<std::size_t>(i)] = glm::mix(a.y, b.y, (nm - a.x) / (b.x - a.x));
    }
    return out;
}

std::optional<Spectrum> lampSpectrum(std::string_view name) {
    if (std::optional<Spectrum> lamp = lampColumn(kFluorescent, kFluorescentNames, name)) {
        return lamp;
    }
    if (std::optional<Spectrum> lamp = lampColumn(kLed, kLedNames, name)) {
        return lamp;
    }
    return lampColumn(kHighPressure, kHighPressureNames, name);
}

std::optional<Spectrum> illuminantSpectrum(Illuminant illuminant, double kelvin) {
    if (!(kelvin > 0.0) || !std::isfinite(kelvin)) {
        return std::nullopt;
    }
    if (illuminant == Illuminant::Daylight && (kelvin < kDaylightMinKelvin || kelvin > kDaylightMaxKelvin)) {
        return std::nullopt;
    }
    const Spectrum power = illuminant == Illuminant::Blackbody ? blackbodySpectrum(kelvin) : daylightSpectrum(kelvin);
    const double luminance = luminanceSum(power);
    // A blackbody cold enough that Planck's law underflows at every sample has no luminance to normalise a light by.
    if (!(luminance > 0.0) || !std::isfinite(luminance)) {
        return std::nullopt;
    }
    return power;
}

}  // namespace pathtracer::scene::cie
