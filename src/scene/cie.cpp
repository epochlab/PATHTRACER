#include "pathtracer/scene/cie.h"

#include <cmath>

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

// SMPTE RP 177 eq 1-4: scale each primary's unit-luminance column so the three sum to the scene white, then invert; D65 adapts first.
glm::dmat3 deriveXyzToScene() {
    const glm::dmat3 primaries(xyzAtUnitLuminance(kScenePrimaryRed), xyzAtUnitLuminance(kScenePrimaryGreen),
                               xyzAtUnitLuminance(kScenePrimaryBlue));
    const glm::dvec3 sceneWhite = xyzAtUnitLuminance(kSceneWhite);
    const glm::dvec3 scale = glm::inverse(primaries) * sceneWhite;
    const glm::dmat3 rgbToXyz = primaries * glm::dmat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
    Spectrum perfectReflector;
    perfectReflector.fill(1.0);
    return glm::inverse(rgbToXyz) * bradfordAdaptation(reflectanceToXyz(perfectReflector), sceneWhite);
}

}  // namespace

TableRow tableRow(int index) {
    const Sample& s = kTable[index];
    return {{s.xBar, s.yBar, s.zBar}, s.d65};
}

const glm::dmat3& xyzToScene() {
    static const glm::dmat3 kMatrix = deriveXyzToScene();
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

std::optional<glm::dvec3> blackbodyToScene(double kelvin) {
    if (!(kelvin > 0.0) || !std::isfinite(kelvin)) {
        return std::nullopt;
    }
    // Planck's second radiation constant, CIE 015:2018's ITS-90 value; c1 and the 1 nm interval cancel in the Y = 1 normalisation.
    constexpr double kC2 = 1.4388e-2;
    glm::dvec3 xyz(0.0);
    for (int i = 0; i < kSampleCount; ++i) {
        const double lambda = wavelengthNm(i) * 1e-9;
        const Sample& s = kTable[i];
        xyz += glm::dvec3(s.xBar, s.yBar, s.zBar) / (std::pow(lambda, 5) * std::expm1(kC2 / (lambda * kelvin)));
    }
    const glm::dvec3 rgb = xyzToScene() * (xyz / xyz.y);
    // Cool of the gamut the locus leaves the primaries (a channel < 0) and an underflowed spectrum is NaN: a clamp shifts hue, so refuse.
    if (!glm::all(glm::greaterThanEqual(rgb, glm::dvec3(0.0)))) {
        return std::nullopt;
    }
    return rgb;
}

}  // namespace pathtracer::scene::cie
