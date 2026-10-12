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

// ITU-R BT.709-6 item 1.4 primary chromaticities (x, y).
constexpr glm::dvec2 kRec709Red(0.640, 0.330);
constexpr glm::dvec2 kRec709Green(0.300, 0.600);
constexpr glm::dvec2 kRec709Blue(0.150, 0.060);

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

// SMPTE RP 177 eq 1-4: scale each primary's unit-luminance column so the three sum to the white point, then invert.
glm::dmat3 deriveXyzToRec709() {
    const glm::dmat3 primaries(xyzAtUnitLuminance(kRec709Red), xyzAtUnitLuminance(kRec709Green),
                               xyzAtUnitLuminance(kRec709Blue));
    Spectrum perfectReflector;
    perfectReflector.fill(1.0);
    const glm::dvec3 white = reflectanceToXyz(perfectReflector);
    const glm::dvec3 scale = glm::inverse(primaries) * white;
    return glm::inverse(primaries * glm::dmat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z));
}

}  // namespace

TableRow tableRow(int index) {
    const Sample& s = kTable[index];
    return {{s.xBar, s.yBar, s.zBar}, s.d65};
}

const glm::dmat3& xyzToRec709() {
    static const glm::dmat3 kMatrix = deriveXyzToRec709();
    return kMatrix;
}

glm::dvec3 reflectanceToRec709(const Spectrum& reflectance) { return xyzToRec709() * reflectanceToXyz(reflectance); }

std::optional<glm::dvec3> blackbodyToRec709(double kelvin) {
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
    const glm::dvec3 rgb = xyzToRec709() * (xyz / xyz.y);
    // Below 1900 K the locus leaves Rec.709 (blue < 0): a clamp would shift hue, and an underflowed spectrum is NaN, so both are refused.
    if (!glm::all(glm::greaterThanEqual(rgb, glm::dvec3(0.0)))) {
        return std::nullopt;
    }
    return rgb;
}

}  // namespace pathtracer::scene::cie
