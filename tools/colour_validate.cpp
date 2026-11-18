// Gate for CIE 1931 (scene/cie.cpp), measured spectral data (config/spectral_data.cpp, scene/measured.cpp) and how they shade.

#include <algorithm>
#include <cfloat>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <OpenColorIO/OpenColorIO.h>

#include "check.h"
#include "conductor_reference.h"
#include "pathtracer/config/scene_config.h"
#include "pathtracer/config/spectral_data.h"
#include "pathtracer/gfx/ocio_display_transform.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/measured.h"
#include "pathtracer/scene/spectrum.h"

namespace {

namespace cie = pathtracer::scene::cie;
using pathtracer::config::ChartPatch;

const std::string kSpectra = std::string(ASSET_ROOT_DIR) + "/spectra/";
const std::string kColorCheckerTable = kSpectra + "colorchecker_babelcolor_average.csv";

// The Classic chart's patch count, 6 x 4.
constexpr std::size_t kPatchCount = 24;

// Wavelengths on the 1 nm grid, where a tabulated spectrum is its double sample rounded once to float.
pathtracer::scene::SampledWavelengths gridWavelengths(const glm::vec4& nm) {
    pathtracer::scene::SampledWavelengths lambda = pathtracer::scene::sampleWavelengths(0.5F);
    lambda.nm = nm;
    return lambda;
}

// Rows "lambda_um,n,k" as written, linear in nm at one wavelength: the oracle reads the file apart from loadNkCsv.
std::complex<double> nkAt(const std::string& file, double nm) {
    std::ifstream in(file);
    std::string line;
    glm::dvec3 below(0.0);
    while (std::getline(in, line)) {
        glm::dvec3 row(0.0);
        if (line.empty() || line.front() == '#' || std::sscanf(line.c_str(), "%lf,%lf,%lf", &row.x, &row.y, &row.z) != 3) {
            continue;
        }
        row.x *= 1000.0;
        if (row.x >= nm) {
            const double t = (nm - below.x) / (row.x - below.x);
            return {std::lerp(below.y, row.y, t), std::lerp(below.z, row.z, t)};
        }
        below = row;
    }
    return {0.0, 0.0};
}

// Two 3x3 double inversions on O(1) entries round at ~1e-15; three orders of headroom still resolves any real defect.
constexpr double kDoubleRoundoff = 1e-12;

// The renderer is float: one rounding of the authored value, one through its Fresnel evaluation (measured worst 0.63 FLT_EPSILON).
constexpr double kFloatResolution = 2.0 * FLT_EPSILON;

glm::dvec2 chromaticity(const glm::dvec3& xyz) { return glm::dvec2(xyz) / (xyz.x + xyz.y + xyz.z); }

double maxAbs(const glm::dvec3& v) { return std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)}); }

// An off-by-one row or a truncated transcription moves both normalisation points, each fixed by definition rather than by measurement.
PT_CHECK(cie_tables_sit_on_their_normalisation_points, Fast, Exact) {
    ctx.plan(2);
    int peak = 0;
    for (int i = 1; i < cie::kSampleCount; ++i) {
        peak = cie::tableRow(i).colourMatching.y > cie::tableRow(peak).colourMatching.y ? i : peak;
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "yBar peaks at %g nm with %.9g; V(lambda) is 1 at 555 nm (CIE 018:2019)",
                  cie::wavelengthNm(peak), cie::tableRow(peak).colourMatching.y);
    PT_EXPECT(ctx, cie::wavelengthNm(peak) == 555.0 && cie::tableRow(peak).colourMatching.y == 1.0, detail);
    const double d65At560 = cie::tableRow(560 - cie::kLambdaMinNm).d65;
    std::snprintf(detail, sizeof(detail), "D65 at 560 nm is %.9g; ISO/CIE 11664-2 normalises it to 100", d65At560);
    PT_EXPECT(ctx, d65At560 == 100.0, detail);
}

// The scene matrix's independent oracle is the pinned OCIO config: ACEScg to its D65 XYZ interchange, the matrix the display path inverts.
glm::dmat3 ocioSceneToXyz() {
    const OCIO_NAMESPACE::ConstConfigRcPtr config = OCIO_NAMESPACE::Config::CreateFromBuiltinConfig(pathtracer::gfx::kOcioConfigName);
    const OCIO_NAMESPACE::ConstCPUProcessorRcPtr toXyz =
        config->getProcessor(pathtracer::gfx::kOcioSceneColorSpace, OCIO_NAMESPACE::ROLE_INTERCHANGE_DISPLAY)->getDefaultCPUProcessor();
    glm::dmat3 matrix(0.0);
    for (int c = 0; c < 3; ++c) {
        std::array<float, 3> xyz{};
        xyz[static_cast<std::size_t>(c)] = 1.0F;
        toXyz->applyRGB(xyz.data());
        matrix[c] = glm::dvec3(xyz[0], xyz[1], xyz[2]);
    }
    return matrix;
}

// The tabulated D65 sits ~1e-4 in xy from OCIO's nominal one and OCIO's matrix is float: 5e-4 bounds both, below any real defect.
constexpr double kOcioAgreement = 5e-4;

// The matrix is pinned by AP1's primaries, the ACES white and Bradford alone; the pinned config applies the same to the display.
PT_CHECK(scene_matrix_matches_the_ocio_config_and_maps_white, Fast, Exact) {
    ctx.plan(6);
    const glm::dmat3 rgbToXyz = glm::inverse(cie::xyzToScene());
    const glm::dmat3 ocio = ocioSceneToXyz();
    double worst = 0.0;
    for (int c = 0; c < 3; ++c) {
        worst = std::max(worst, maxAbs(rgbToXyz[c] - ocio[c]));
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "scene-to-XYZ matrix is %.3e from OCIO's %s to the D65 interchange", worst, pathtracer::gfx::kOcioSceneColorSpace);
    PT_EXPECT(ctx, worst <= kOcioAgreement, detail);
    cie::Spectrum flat;
    flat.fill(1.0);
    const double whiteError = maxAbs(cie::reflectanceToScene(flat) - glm::dvec3(1.0));
    std::snprintf(detail, sizeof(detail), "perfect reflector maps %.3e away from (1, 1, 1)", whiteError);
    PT_EXPECT(ctx, whiteError <= kDoubleRoundoff, detail);
    constexpr double kMidGrey = 0.18;
    flat.fill(kMidGrey);
    const double greyError = maxAbs(cie::reflectanceToScene(flat) - glm::dvec3(kMidGrey));
    std::snprintf(detail, sizeof(detail), "flat %.2f reflectance maps %.3e away from neutral", kMidGrey, greyError);
    PT_EXPECT(ctx, greyError <= kDoubleRoundoff, detail);
    const glm::vec3& weights = cie::sceneLuminanceWeights();
    std::snprintf(detail, sizeof(detail), "luminance weights sum to %.9g, not 1", static_cast<double>(weights.x + weights.y + weights.z));
    PT_EXPECT(ctx, std::abs(static_cast<double>(weights.x + weights.y + weights.z) - 1.0) <= kFloatResolution, detail);
    const double weightError = maxAbs(glm::dvec3(weights) - glm::dvec3(ocio[0][1], ocio[1][1], ocio[2][1]));
    std::snprintf(detail, sizeof(detail), "luminance weights are %.3e from OCIO's Y row", weightError);
    PT_EXPECT(ctx, weightError <= kOcioAgreement, detail);
    const std::array<float, 8> tagged = cie::sceneChromaticities();
    const std::array<float, 8> published = {0.713F, 0.293F, 0.165F, 0.830F, 0.128F, 0.044F, 0.32168F, 0.33767F};
    PT_EXPECT(ctx, tagged == published, "tagged chromaticities are not AP1's primaries and the ACES white");
}

// XYZ of a relative spectral power on the CIE 1 nm grid, CIE 015:2018's summation.
glm::dvec3 spectrumXyz(const cie::Spectrum& power) {
    glm::dvec3 xyz(0.0);
    for (int i = 0; i < cie::kSampleCount; ++i) {
        xyz += power[static_cast<std::size_t>(i)] * cie::tableRow(i).colourMatching;
    }
    return xyz;
}

// CIE A is Planck at c2/T = 1.435e-2/2848 (CIE 015:2018); the summation interval moves its 5th decimal, so 4 are asserted.
PT_CHECK(blackbody_reproduces_illuminant_a, Fast, Exact) {
    ctx.plan(1);
    constexpr double kKelvin = 2848.0 * 1.4388e-2 / 1.435e-2;
    constexpr glm::dvec2 kPublished(0.44757, 0.40745);
    const std::optional<cie::Spectrum> power = cie::illuminantSpectrum(cie::Illuminant::Blackbody, kKelvin);
    const glm::dvec2 xy = power ? chromaticity(spectrumXyz(*power)) : glm::dvec2(0.0);
    const double error = std::max(std::abs(xy.x - kPublished.x), std::abs(xy.y - kPublished.y));
    char detail[160];
    std::snprintf(detail, sizeof(detail), "Planck at CIE A's exponent gives xy (%.6f, %.6f), %.3e from (0.44757, 0.40745)", xy.x, xy.y,
                  error);
    PT_EXPECT(ctx, error <= 5e-5, detail);
}

// Any temperature with visible luminance resolves, out of AP1 or not; outside the domain, or underflowed at every sample, is refused.
PT_CHECK(blackbody_resolves_and_refuses_outside_its_domain, Fast, Exact) {
    constexpr std::array<double, 8> kResolved{500.0, 1000.0, 2000.0, 2856.0, 4000.0, 6504.0, 10000.0, 1e6};
    constexpr std::array<double, 5> kRefused{0.0, -2856.0, 10.0, std::numeric_limits<double>::infinity(),
                                             std::numeric_limits<double>::quiet_NaN()};
    ctx.plan(static_cast<int>(kResolved.size() + kRefused.size()));
    char detail[160];
    for (const double kelvin : kResolved) {
        const std::optional<cie::Spectrum> power = cie::illuminantSpectrum(cie::Illuminant::Blackbody, kelvin);
        const double luminanceError = power ? std::abs(pathtracer::scene::EmissionSpectrum(*power).luminance() - 1.0) : 1.0;
        std::snprintf(detail, sizeof(detail), "%g K: %s, unit-luminance light off 1 by %.3e", kelvin, power ? "resolved" : "refused",
                      luminanceError);
        PT_EXPECT(ctx, power && luminanceError <= kFloatResolution, detail);
    }
    for (const double kelvin : kRefused) {
        std::snprintf(detail, sizeof(detail), "%g K resolved to a spectrum; it is outside the domain or has no visible luminance", kelvin);
        PT_EXPECT(ctx, !cie::illuminantSpectrum(cie::Illuminant::Blackbody, kelvin), detail);
    }
}

// D65 is daylight at 6500 K on the 1.4380e-2 c2 scale; CIE tabulated it with M1, M2 rounded to 3 decimals, which bounds the gap.
PT_CHECK(daylight_reproduces_d65, Fast, Exact) {
    ctx.plan(1);
    constexpr double kKelvin = 6500.0 * 1.4388e-2 / 1.4380e-2;
    constexpr double kRoundingHalfStep = 5e-4;
    glm::dvec3 d65(0.0);
    glm::dvec3 s1(0.0);
    glm::dvec3 s2(0.0);
    for (int i = 0; i < cie::kSampleCount; ++i) {
        const cie::TableRow row = cie::tableRow(i);
        const int knot = std::min(i / cie::kDaylightIntervalNm, cie::kDaylightKnotCount - 2);
        const double f = static_cast<double>(i - (knot * cie::kDaylightIntervalNm)) / cie::kDaylightIntervalNm;
        const glm::dvec3 components = ((1.0 - f) * cie::daylightComponents(knot)) + (f * cie::daylightComponents(knot + 1));
        d65 += row.d65 * row.colourMatching;
        s1 += components.y * row.colourMatching;
        s2 += components.z * row.colourMatching;
    }
    // The tabulated spectrum moved by each rounding corner: chromaticity is near-linear at this scale, so a corner is the extreme.
    const glm::dvec2 tabulated = chromaticity(d65);
    double bound = 0.0;
    for (const double a : {-kRoundingHalfStep, kRoundingHalfStep}) {
        for (const double b : {-kRoundingHalfStep, kRoundingHalfStep}) {
            const glm::dvec2 corner = chromaticity(d65 + (a * s1) + (b * s2));
            bound = std::max(bound, std::max(std::abs(corner.x - tabulated.x), std::abs(corner.y - tabulated.y)));
        }
    }
    const std::optional<cie::Spectrum> power = cie::illuminantSpectrum(cie::Illuminant::Daylight, kKelvin);
    const glm::dvec2 xy = power ? chromaticity(spectrumXyz(*power)) : glm::dvec2(0.0);
    const double error = std::max(std::abs(xy.x - tabulated.x), std::abs(xy.y - tabulated.y));
    char detail[192];
    std::snprintf(detail, sizeof(detail), "daylight at %.2f K gives xy (%.6f, %.6f), %.3e from D65's (%.6f, %.6f); rounding bounds %.3e",
                  kKelvin, xy.x, xy.y, error, tabulated.x, tabulated.y, bound);
    PT_EXPECT(ctx, error <= bound, detail);
}

// Daylight resolves across CIE's defined 4000-25000 K, both locus branches; outside it is refused.
PT_CHECK(daylight_resolves_and_refuses_outside_its_range, Fast, Exact) {
    constexpr std::array<double, 6> kInRange{cie::kDaylightMinKelvin, 5003.0, 6504.0, 7000.0, 7001.0, cie::kDaylightMaxKelvin};
    const std::array<double, 6> kRefused{std::nextafter(cie::kDaylightMinKelvin, 0.0), std::nextafter(cie::kDaylightMaxKelvin, 1e9),
                                             0.0, -6504.0, std::numeric_limits<double>::infinity(),
                                             std::numeric_limits<double>::quiet_NaN()};
    ctx.plan(static_cast<int>(kInRange.size() + kRefused.size()));
    char detail[160];
    for (const double kelvin : kInRange) {
        const std::optional<cie::Spectrum> power = cie::illuminantSpectrum(cie::Illuminant::Daylight, kelvin);
        const double luminanceError = power ? std::abs(pathtracer::scene::EmissionSpectrum(*power).luminance() - 1.0) : 1.0;
        std::snprintf(detail, sizeof(detail), "%g K daylight: %s, unit-luminance light off 1 by %.3e", kelvin, power ? "resolved" : "refused",
                      luminanceError);
        PT_EXPECT(ctx, power && luminanceError <= kFloatResolution, detail);
    }
    for (const double kelvin : kRefused) {
        std::snprintf(detail, sizeof(detail), "%.17g K resolved as daylight; CIE defines it over 4000-25000 K only", kelvin);
        PT_EXPECT(ctx, !cie::illuminantSpectrum(cie::Illuminant::Daylight, kelvin), detail);
    }
}

// XYZ of power's linear interpolant times the observer's, integrated exactly per 1 nm bin: what the sensor's estimator converges to.
glm::dvec3 interpolantXyz(const cie::Spectrum& power) {
    glm::dvec3 xyz(0.0);
    for (int i = 0; i + 1 < cie::kSampleCount; ++i) {
        const double a = power[static_cast<std::size_t>(i)];
        const double b = power[static_cast<std::size_t>(i) + 1];
        const glm::dvec3 ca = cie::tableRow(i).colourMatching;
        const glm::dvec3 cb = cie::tableRow(i + 1).colourMatching;
        xyz += (((a * ca) + (b * cb)) / 3.0) + (((a * cb) + (b * ca)) / 6.0);
    }
    return xyz;
}

// The sensor over 2^16 stratified heroes reads a unit-luminance light (CIE summation) at xyzToScene of its interpolant's XYZ.
void expectSensorReads(tools::check::Context& ctx, const char* name, const cie::Spectrum& power) {
    constexpr int kStrata = 1 << 16;
    const glm::dvec3 expected = cie::xyzToScene() * (interpolantXyz(power) / spectrumXyz(power).y);
    const pathtracer::scene::EmissionSpectrum light(power);
    glm::dvec3 projected(0.0);
    for (int k = 0; k < kStrata; ++k) {
        const pathtracer::scene::SampledWavelengths lambda = pathtracer::scene::sampleWavelengths((static_cast<float>(k) + 0.5F) / kStrata);
        projected += glm::dvec3(pathtracer::scene::toScene(light(lambda), lambda).vec3());
    }
    projected /= kStrata;
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%s reads (%.5f, %.5f, %.5f), its tristimulus colour is (%.5f, %.5f, %.5f)", name, projected.x,
                  projected.y, projected.z, expected.x, expected.y, expected.z);
    PT_EXPECT(ctx, maxAbs(projected - expected) <= 1e-4, detail);
}

PT_CHECK(sensor_reads_kelvin_lights_as_their_tristimulus_colour, Fast, Exact) {
    const std::array<std::pair<cie::Illuminant, double>, 3> kLights{
        {{cie::Illuminant::Blackbody, 2856.0}, {cie::Illuminant::Blackbody, 4000.0}, {cie::Illuminant::Daylight, 6504.0}}};
    ctx.plan(static_cast<int>(kLights.size()));
    for (const auto& [illuminant, kelvin] : kLights) {
        char name[32];
        std::snprintf(name, sizeof(name), "%g K", kelvin);
        expectSensorReads(ctx, name, *cie::illuminantSpectrum(illuminant, kelvin));
    }
}

// CIE 015:2018's published xy of its tabulated lamps (Tables 10, 11, 12), as colour-science 0.4.6's CCS_ILLUMINANTS carries them.
struct LampChromaticity {
    const char* name;
    double x;
    double y;
    int intervalNm;  // the vendored table's own interval: 1 nm for FL and LED, 5 nm for HP
};
constexpr std::array<LampChromaticity, 41> kLamps{{
    {"FL1", 0.3131, 0.3371, 1}, {"FL2", 0.3721, 0.3751, 1}, {"FL3", 0.4091, 0.3941, 1}, {"FL4", 0.4402, 0.4031, 1},
    {"FL5", 0.3138, 0.3452, 1}, {"FL6", 0.3779, 0.3882, 1}, {"FL7", 0.3129, 0.3292, 1}, {"FL8", 0.3458, 0.3586, 1},
    {"FL9", 0.3741, 0.3727, 1}, {"FL10", 0.3458, 0.3588, 1}, {"FL11", 0.3805, 0.3769, 1}, {"FL12", 0.4370, 0.4042, 1},
    {"FL3.1", 0.4407, 0.4033, 1}, {"FL3.2", 0.3808, 0.3734, 1}, {"FL3.3", 0.3153, 0.3439, 1}, {"FL3.4", 0.4429, 0.4043, 1},
    {"FL3.5", 0.3749, 0.3672, 1}, {"FL3.6", 0.3488, 0.3600, 1}, {"FL3.7", 0.4384, 0.4045, 1}, {"FL3.8", 0.3820, 0.3832, 1},
    {"FL3.9", 0.3499, 0.3591, 1}, {"FL3.10", 0.3455, 0.3560, 1}, {"FL3.11", 0.3245, 0.3434, 1}, {"FL3.12", 0.4377, 0.4037, 1},
    {"FL3.13", 0.3830, 0.3724, 1}, {"FL3.14", 0.3447, 0.3609, 1}, {"FL3.15", 0.3127, 0.3288, 1}, {"HP1", 0.5330, 0.4150, 5},
    {"HP2", 0.4778, 0.4158, 5}, {"HP3", 0.4302, 0.4075, 5}, {"HP4", 0.3812, 0.3797, 5}, {"HP5", 0.3776, 0.3713, 5},
    {"LED-B1", 0.4560, 0.4078, 1}, {"LED-B2", 0.4357, 0.4012, 1}, {"LED-B3", 0.3756, 0.3723, 1}, {"LED-B4", 0.3422, 0.3502, 1},
    {"LED-B5", 0.3118, 0.3236, 1}, {"LED-BH1", 0.4474, 0.4066, 1}, {"LED-RGB1", 0.4557, 0.4211, 1}, {"LED-V1", 0.4548, 0.4044, 1},
    {"LED-V2", 0.3781, 0.3775, 1}
}};

// Summed over 380-780 nm at its table's own interval, as CIE computed them, every lamp lands on its published xy to the last digit.
PT_CHECK(lamps_reproduce_cie_015_chromaticities, Fast, Exact) {
    constexpr double kPublishedDigit = 1e-4;
    ctx.plan(static_cast<int>(kLamps.size()) + 1);
    PT_EXPECT(ctx, !cie::lampSpectrum("FL13") && !cie::lampSpectrum("fl1") && !cie::lampSpectrum("D65"),
              "an unknown or miscased lamp name resolved to a table column");
    for (const LampChromaticity& lamp : kLamps) {
        const std::optional<cie::Spectrum> power = cie::lampSpectrum(lamp.name);
        glm::dvec3 xyz(0.0);
        for (int nm = 380; power && nm <= 780; nm += lamp.intervalNm) {
            const int i = nm - cie::kLambdaMinNm;
            xyz += (*power)[static_cast<std::size_t>(i)] * cie::tableRow(i).colourMatching;
        }
        const glm::dvec2 xy = chromaticity(xyz);
        char detail[160];
        std::snprintf(detail, sizeof(detail), "%s sums to xy (%.5f, %.5f), CIE 015:2018 publishes (%.4f, %.4f)", lamp.name, xy.x, xy.y,
                      lamp.x, lamp.y);
        PT_EXPECT(ctx, power && std::abs(xy.x - lamp.x) <= kPublishedDigit && std::abs(xy.y - lamp.y) <= kPublishedDigit, detail);
    }
}

// The renderer reads every lamp at its tristimulus colour, the 1 nm line spectra (FL3.x, LED-RGB1) included.
PT_CHECK(sensor_reads_lamps_as_their_tristimulus_colour, Fast, Exact) {
    ctx.plan(static_cast<int>(kLamps.size()));
    for (const LampChromaticity& lamp : kLamps) {
        expectSensorReads(ctx, lamp.name, *cie::lampSpectrum(lamp.name));
    }
}

// Linear between samples, exact on them, zero outside: the CIE datasets' stated interpolation and extrapolation.
PT_CHECK(resample_is_linear_inside_and_zero_outside, Fast, Exact) {
    const std::array<glm::dvec2, 3> samples{{{400.5, 1.0}, {402.5, 3.0}, {403.0, 5.0}}};
    const cie::Spectrum out = cie::resample(samples);
    const auto at = [&](int nm) { return out[static_cast<std::size_t>(nm - cie::kLambdaMinNm)]; };
    ctx.plan(2);
    PT_EXPECT(ctx, at(400) == 0.0 && at(404) == 0.0 && at(cie::kLambdaMinNm) == 0.0 && at(cie::kLambdaMaxNm) == 0.0,
              "a wavelength outside the samples is not zero");
    PT_EXPECT(ctx, at(401) == 1.5 && at(402) == 2.5 && at(403) == 5.0, "a wavelength inside is not the linear interpolant or the knot");
}

// CIELAB of XYZ against a white, CIE 015:2018.
glm::dvec3 lab(const glm::dvec3& xyz, const glm::dvec3& white) {
    constexpr double kDelta = 6.0 / 29.0;
    const auto f = [](double t) { return t > kDelta * kDelta * kDelta ? std::cbrt(t) : (t / (3.0 * kDelta * kDelta)) + (4.0 / 29.0); };
    const glm::dvec3 t = xyz / white;
    return {(116.0 * f(t.y)) - 16.0, 500.0 * (f(t.x) - f(t.y)), 200.0 * (f(t.y) - f(t.z))};
}

// CIEDE2000 (CIE 142:2001), in Sharma, Wu and Dalal's (2005) tested formulation.
double deltaE2000(const glm::dvec3& lab1, const glm::dvec3& lab2) {
    constexpr double kPi = 3.14159265358979323846;
    const auto degrees = [&](double r) { return r * 180.0 / kPi; };
    const auto radians = [&](double d) { return d * kPi / 180.0; };
    const double cBar = 0.5 * (std::hypot(lab1.y, lab1.z) + std::hypot(lab2.y, lab2.z));
    const double g = 0.5 * (1.0 - std::sqrt(std::pow(cBar, 7.0) / (std::pow(cBar, 7.0) + std::pow(25.0, 7.0))));
    const double a1 = (1.0 + g) * lab1.y;
    const double a2 = (1.0 + g) * lab2.y;
    const double c1 = std::hypot(a1, lab1.z);
    const double c2 = std::hypot(a2, lab2.z);
    const auto hue = [&](double a, double b) { return a == 0.0 && b == 0.0 ? 0.0 : std::fmod(degrees(std::atan2(b, a)) + 360.0, 360.0); };
    const double h1 = hue(a1, lab1.z);
    const double h2 = hue(a2, lab2.z);
    double dh = 0.0;
    if (c1 * c2 != 0.0) {
        dh = h2 - h1;
        dh = dh > 180.0 ? dh - 360.0 : (dh < -180.0 ? dh + 360.0 : dh);
    }
    const double dL = lab2.x - lab1.x;
    const double dC = c2 - c1;
    const double dH = 2.0 * std::sqrt(c1 * c2) * std::sin(radians(dh / 2.0));
    const double lBar = 0.5 * (lab1.x + lab2.x);
    const double cBarPrime = 0.5 * (c1 + c2);
    double hBar = h1 + h2;
    if (c1 * c2 != 0.0) {
        hBar = std::abs(h1 - h2) <= 180.0 ? 0.5 * (h1 + h2) : (h1 + h2 < 360.0 ? 0.5 * (h1 + h2 + 360.0) : 0.5 * (h1 + h2 - 360.0));
    }
    const double t = 1.0 - (0.17 * std::cos(radians(hBar - 30.0))) + (0.24 * std::cos(radians(2.0 * hBar))) +
                     (0.32 * std::cos(radians((3.0 * hBar) + 6.0))) - (0.20 * std::cos(radians((4.0 * hBar) - 63.0)));
    const double dTheta = 30.0 * std::exp(-std::pow((hBar - 275.0) / 25.0, 2.0));
    const double rc = 2.0 * std::sqrt(std::pow(cBarPrime, 7.0) / (std::pow(cBarPrime, 7.0) + std::pow(25.0, 7.0)));
    const double sl = 1.0 + (0.015 * (lBar - 50.0) * (lBar - 50.0) / std::sqrt(20.0 + ((lBar - 50.0) * (lBar - 50.0))));
    const double sc = 1.0 + (0.045 * cBarPrime);
    const double sh = 1.0 + (0.015 * cBarPrime * t);
    const double rt = -std::sin(radians(2.0 * dTheta)) * rc;
    return std::sqrt(std::pow(dL / sl, 2.0) + std::pow(dC / sc, 2.0) + std::pow(dH / sh, 2.0) + (rt * (dC / sc) * (dH / sh)));
}

// Sharma et al. 2005's published pairs 1, 7, 17 and 25: the formula's hue rotation, near-neutral and large-difference branches.
PT_CHECK(ciede2000_matches_sharmas_test_data, Fast, Exact) {
    struct Pair {
        glm::dvec3 a;
        glm::dvec3 b;
        double expected;
    };
    const std::array<Pair, 4> kPairs{{{{50.0, 2.6772, -79.7751}, {50.0, 0.0, -82.7485}, 2.0425},
                                      {{50.0, 0.0, 0.0}, {50.0, -1.0, 2.0}, 2.3669},
                                      {{50.0, 2.5, 0.0}, {73.0, 25.0, -18.0}, 27.1492},
                                      {{60.2574, -34.0099, 36.2677}, {60.4626, -34.1751, 39.4387}, 1.2644}}};
    ctx.plan(static_cast<int>(kPairs.size()));
    for (const Pair& pair : kPairs) {
        const double got = deltaE2000(pair.a, pair.b);
        PT_EXPECT(ctx, std::abs(got - pair.expected) <= 5e-5, "dE00 " + std::to_string(got) + " against " + std::to_string(pair.expected));
    }
}

// Scene RGB of the table's fit through the renderer's own sigmoid on the CIE 1 nm grid, under D65 as the fit was made.
glm::dvec3 sigmoidToScene(const pathtracer::scene::RgbSigmoid& sigmoid) {
    cie::Spectrum reflectance{};
    for (int i = 0; i < cie::kSampleCount; ++i) {
        reflectance[static_cast<std::size_t>(i)] = sigmoid(static_cast<float>(cie::wavelengthNm(i)));
    }
    return cie::reflectanceToScene(reflectance);
}

// Measured ColorChecker reflectances' scene RGB under D65 round-trip through the table within a tenth of a JND.
PT_CHECK(rgb_spectrum_table_round_trips_measured_reflectances, Fast, Exact) {
    const std::optional<std::vector<ChartPatch>> patches = pathtracer::config::loadChartCsv(kColorCheckerTable, "colour_validate");
    ctx.plan(1);
    if (!patches) {
        PT_EXPECT(ctx, false, "ColorChecker table unavailable");
        return;
    }
    glm::dvec3 white(0.0);
    double whiteY = 0.0;
    for (int i = 0; i < cie::kSampleCount; ++i) {
        white += cie::tableRow(i).d65 * cie::tableRow(i).colourMatching;
        whiteY += cie::tableRow(i).d65 * cie::tableRow(i).colourMatching.y;
    }
    white /= whiteY;
    const glm::dmat3 toXyz = glm::inverse(cie::xyzToScene());
    double worst = 0.0;
    std::string worstName;
    for (const ChartPatch& patch : *patches) {
        const glm::dvec3 scene = cie::reflectanceToScene(patch.reflectance);
        const glm::dvec3 rendered = sigmoidToScene(pathtracer::scene::rgbSigmoid(pathtracer::scene::Rgb(glm::vec3(scene))));
        const double error = deltaE2000(lab(toXyz * scene, white), lab(toXyz * rendered, white));
        if (error > worst) {
            worst = error;
            worstName = patch.name;
        }
    }
    std::printf("    worst ColorChecker round trip dE00 %.5f (%s)\n", worst, worstName.c_str());
    PT_EXPECT(ctx, worst <= 0.1, "table round trip dE00 " + std::to_string(worst) + " at " + worstName);
}

// Equal RGB is a flat reflectance with no lookup; an RGB (1, 1, 1) emitter is D65 at unit luminance, every lane.
PT_CHECK(grey_upsamples_flat_and_white_emits_d65, Fast, Exact) {
    using pathtracer::scene::Rgb;
    using pathtracer::scene::SampledSpectrum;
    ctx.plan(3);
    const pathtracer::scene::SampledWavelengths lambda = pathtracer::scene::sampleWavelengths(0.3F);
    PT_EXPECT(ctx, pathtracer::scene::reflectanceSpectrum(Rgb(0.37F), lambda) == SampledSpectrum(0.37F), "grey 0.37 was not flat");
    PT_EXPECT(ctx, pathtracer::scene::reflectanceSpectrum(Rgb(1.0F), lambda) == SampledSpectrum(1.0F), "white was not flat 1");
    PT_EXPECT(ctx, pathtracer::scene::illuminantSpectrum(Rgb(2.0F), lambda) == 2.0F * lambda.white,
              "a grey emitter was not the scene white's D65");
}

// Every lane's density is positive and finite, so no wavelength the sensor sees is unreachable; lanes along the CDF are in order.
PT_CHECK(wavelength_lanes_stratify_the_sensitivity_cdf, Fast, Exact) {
    ctx.plan(2);
    constexpr int kStrata = 1 << 16;
    int unbounded = 0;
    for (int k = 0; k < kStrata; ++k) {
        const pathtracer::scene::SampledWavelengths lambda = pathtracer::scene::sampleWavelengths((static_cast<float>(k) + 0.5F) / kStrata);
        for (int j = 0; j < pathtracer::scene::kSpectrumSamples; ++j) {
            unbounded += lambda.pdf[j] > 0.0F && std::isfinite(lambda.pdf[j]) ? 0 : 1;
        }
    }
    PT_EXPECT(ctx, unbounded == 0, std::to_string(unbounded) + " lanes drew a zero or non-finite density");
    // Lane j sits at CDF position u + j/n mod 1, one per quarter; the inverse CDF is monotone, so that order is the wavelength order.
    constexpr int kSpectrumSamples = pathtracer::scene::kSpectrumSamples;
    int misplaced = 0;
    for (int k = 0; k < 4096; ++k) {
        const float u = (static_cast<float>(k) + 0.5F) / 4096.0F;
        const pathtracer::scene::SampledWavelengths lambda = pathtracer::scene::sampleWavelengths(u);
        for (int j = 0; j < kSpectrumSamples; ++j) {
            const float uj = std::fmod(u + (static_cast<float>(j) / kSpectrumSamples), 1.0F);
            for (int i = 0; i < kSpectrumSamples; ++i) {
                const float ui = std::fmod(u + (static_cast<float>(i) / kSpectrumSamples), 1.0F);
                misplaced += ui < uj && !(lambda.nm[i] <= lambda.nm[j]) ? 1 : 0;
            }
        }
    }
    PT_EXPECT(ctx, misplaced == 0, std::to_string(misplaced) + " lanes out of CDF order");
}

// The sensor's estimator: stratified over u, the scene white at 1 cd/m^2 projects to (1, 1, 1) and a flat reflectance under it too.
PT_CHECK(sensor_projects_the_scene_white_to_one, Fast, Exact) {
    ctx.plan(2);
    constexpr int kStrata = 1 << 16;
    glm::dvec3 radiance(0.0);
    glm::dvec3 reflectance(0.0);
    for (int k = 0; k < kStrata; ++k) {
        const pathtracer::scene::SampledWavelengths lambda = pathtracer::scene::sampleWavelengths((static_cast<float>(k) + 0.5F) / kStrata);
        radiance += glm::dvec3(pathtracer::scene::toScene(lambda.white, lambda).vec3());
        reflectance += glm::dvec3(pathtracer::scene::reflectanceToScene(pathtracer::scene::SampledSpectrum(1.0F), lambda).vec3());
    }
    radiance /= kStrata;
    reflectance /= kStrata;
    char detail[160];
    std::snprintf(detail, sizeof(detail), "white radiance reads (%.6f, %.6f, %.6f)", radiance.x, radiance.y, radiance.z);
    PT_EXPECT(ctx, maxAbs(radiance - glm::dvec3(1.0)) <= 1e-4, detail);
    std::snprintf(detail, sizeof(detail), "flat reflectance reads (%.6f, %.6f, %.6f)", reflectance.x, reflectance.y, reflectance.z);
    PT_EXPECT(ctx, maxAbs(reflectance - glm::dvec3(1.0)) <= 1e-4, detail);
}

// Grid wavelengths spanning the visible: lane j's table entry is that wavelength's double sample, rounded once.
constexpr glm::vec4 kGridNm(420.0F, 510.0F, 580.0F, 690.0F);

// A conductor's shading at mu = 1 and mu-bar = 1/7 meets exact Fresnel of the file's n,k (F82's two points), uncapped lanes to float.
void expectMeasuredConductor(tools::check::Context& ctx, const std::string& table, const pathtracer::scene::BsdfClosure& closure) {
    const pathtracer::scene::SampledSpectrum normal = pathtracer::scene::fresnelAtViewAngle(closure, 1.0F);
    const pathtracer::scene::SampledSpectrum grazing = pathtracer::scene::fresnelAtViewAngle(closure, static_cast<float>(tools::reference::kMuBar));
    char detail[200];
    for (int lane = 0; lane < pathtracer::scene::kSpectrumSamples; ++lane) {
        const std::complex<double> eta = nkAt(table, kGridNm[lane]);
        const double exactNormal = tools::reference::referenceConductorFresnelAt(eta, 1.0);
        const double exactGrazing = tools::reference::referenceConductorFresnelAt(eta, tools::reference::kMuBar);
        const double schlick = exactNormal + ((1.0 - exactNormal) * std::pow(1.0 - tools::reference::kMuBar, 5.0));
        // A tint past OpenPBR's 1 caps there: F82 then meets Schlick at mu-bar, its closest in range.
        const double target = std::min(exactGrazing, schlick);
        std::snprintf(detail, sizeof(detail), "%s %.0f nm: R(1) %.9g vs exact %.9g", table.c_str(), kGridNm[lane], normal[lane], exactNormal);
        PT_EXPECT(ctx, std::abs(normal[lane] - exactNormal) <= kFloatResolution, detail);
        std::snprintf(detail, sizeof(detail), "%s %.0f nm: R(1/7) %.9g vs exact %.9g", table.c_str(), kGridNm[lane], grazing[lane], target);
        PT_EXPECT(ctx, std::abs(grazing[lane] - target) <= kFloatResolution, detail);
    }
}

// A metal material shades from its nk file: chrome (chromium) and gold, each at the file's exact Fresnel.
PT_CHECK(measured_metal_materials_meet_exact_fresnel, Fast, Exact) {
    const std::array<std::pair<const char*, const char*>, 2> metals{{{"chrome.json", "chromium_johnson_christy_1974.csv"},
                                                                     {"gold.json", "gold_johnson_christy_1972.csv"}}};
    ctx.plan(static_cast<int>(metals.size()) * (1 + (2 * pathtracer::scene::kSpectrumSamples)));
    for (const auto& [materialFile, table] : metals) {
        const auto material = pathtracer::config::loadMaterialConfig(std::string(ASSET_ROOT_DIR) + "/materials/" + materialFile, ASSET_ROOT_DIR);
        PT_EXPECT(ctx, material.has_value() && material->measured.conductorEta != nullptr, std::string(materialFile) + " binds no nk");
        if (!material || material->measured.conductorEta == nullptr) {
            continue;
        }
        const pathtracer::scene::SampledWavelengths lambda = gridWavelengths(kGridNm);
        expectMeasuredConductor(ctx, kSpectra + table,
                                pathtracer::scene::makeBsdfClosure(pathtracer::scene::toSpectral(*material, lambda), glm::vec3(0.0F, 0.0F, 1.0F), lambda.nm));
    }
}

// Every shipped conductor table, bound bare (metalness 1, specular_weight 1, smooth), shades at its exact Fresnel.
PT_CHECK(measured_conductor_tables_meet_exact_fresnel, Fast, Exact) {
    const std::array<const char*, 5> tables{"chromium_johnson_christy_1974.csv", "gold_johnson_christy_1972.csv", "silver_johnson_christy_1972.csv",
                                            "copper_johnson_christy_1972.csv", "aluminium_rakic_1995.csv"};
    ctx.plan(static_cast<int>(tables.size()) * (1 + (2 * pathtracer::scene::kSpectrumSamples)));
    for (const char* table : tables) {
        const std::optional<pathtracer::config::ComplexIndexSpectrum> index = pathtracer::config::loadNkCsv(kSpectra + table, "colour_validate");
        PT_EXPECT(ctx, index.has_value(), std::string(table) + " rejected");
        if (!index) {
            continue;
        }
        pathtracer::scene::MeasuredMaterial measured;
        const pathtracer::scene::ConductorColours colours = pathtracer::scene::conductorColours(index->n, index->k);
        measured.baseColor = pathtracer::scene::measuredColour(colours.f0);
        measured.specularColor = pathtracer::scene::measuredColour(colours.tint);
        pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant> inputs;
        inputs.baseMetalness = 1.0F;
        inputs.specularRoughness = 0.0F;
        inputs.measured = pathtracer::scene::uniformView(measured);
        const pathtracer::scene::SampledWavelengths lambda = gridWavelengths(kGridNm);
        expectMeasuredConductor(ctx, kSpectra + table,
                                pathtracer::scene::makeBsdfClosure(pathtracer::scene::toSpectral(inputs, lambda), glm::vec3(0.0F, 0.0F, 1.0F), lambda.nm));
    }
}

// SCHOTT states N-BK7's n_d 1.5168 and V_d 64.17 (4 and 2 decimals) beside its Sellmeier terms: glass.json's must reproduce them.
PT_CHECK(glass_sellmeier_reproduces_the_catalogue, Fast, Exact) {
    ctx.plan(5);
    const auto glass = pathtracer::config::loadMaterialConfig(std::string(ASSET_ROOT_DIR) + "/materials/glass.json", ASSET_ROOT_DIR);
    PT_EXPECT(ctx, glass.has_value() && glass->measured.specularIor != nullptr && glass->measured.transmissionColor != nullptr,
              "glass.json binds no sellmeier index or transmittance");
    if (!glass || glass->measured.specularIor == nullptr || glass->measured.transmissionColor == nullptr) {
        return;
    }
    // He d 587.5618, H F 486.1327, H C 656.2725 nm: the lines the catalogue's n_d and V_d are stated at.
    const auto index = [&](float nm) { return static_cast<double>(pathtracer::scene::sampleTabulated(*glass->measured.specularIor, nm)); };
    const double nd = index(587.5618F);
    const double vd = (nd - 1.0) / (index(486.1327F) - index(656.2725F));
    char detail[160];
    std::snprintf(detail, sizeof(detail), "n_d %.6f vs the catalogue's 1.5168", nd);
    PT_EXPECT(ctx, std::abs(nd - 1.5168) <= 0.5e-4, detail);
    std::snprintf(detail, sizeof(detail), "V_d %.4f vs the catalogue's 64.17", vd);
    PT_EXPECT(ctx, std::abs(vd - 64.17) <= 0.5e-2, detail);
    // The 25 mm transmittance at 400 nm and 546 nm, rows the catalogue states: the interior absorbs as measured, at transmission_depth.
    const pathtracer::scene::SampledSpectrum transmittance =
        pathtracer::scene::toSpectral(*glass, gridWavelengths(glm::vec4(400.0F, 546.0F, 400.0F, 546.0F))).transmissionColor;
    std::snprintf(detail, sizeof(detail), "25 mm transmittance %.4f at 400 nm, %.4f at 546 nm vs the catalogue's 0.992, 0.996 at depth %.4f m",
                  transmittance[0], transmittance[1], glass->transmissionDepth);
    PT_EXPECT(ctx, std::abs(transmittance[0] - 0.992F) <= FLT_EPSILON && std::abs(transmittance[1] - 0.996F) <= FLT_EPSILON, detail);
    PT_EXPECT(ctx, glass->transmissionDepth == 0.025F, detail);
}

// Each row is a table some real transcription or sourcing mistake produces; the shipped table proves the rejections reject only those.
PT_CHECK(nk_table_rejects_invalid_input, Fast, Exact) {
    const auto load = [](const char* name, const std::string& text) {
        const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
        std::ofstream(path) << text;
        return pathtracer::config::loadNkCsv(path.string(), "colour_validate").has_value();
    };
    ctx.plan(7);
    PT_EXPECT(ctx, pathtracer::config::loadNkCsv(kSpectra + "chromium_johnson_christy_1974.csv", "colour_validate").has_value(),
              "shipped chromium table rejected");
    PT_EXPECT(ctx, load("engine_colour_crlf.csv", "0.30,1,1\r\n0.90,1,1\r\n"), "CRLF table rejected");
    PT_EXPECT(ctx, !load("engine_colour_gap.csv", "0.30,1,1\n0.80,1,1\n"), "table ending at 800 nm accepted");
    PT_EXPECT(ctx, !load("engine_colour_order.csv", "0.30,1,1\n0.50,1,1\n0.50,1,1\n0.90,1,1\n"), "repeated wavelength accepted");
    PT_EXPECT(ctx, !load("engine_colour_k.csv", "0.30,1,1\n0.90,1,-0.1\n"), "negative k accepted");
    PT_EXPECT(ctx, !load("engine_colour_n.csv", "0.30,0,1\n0.90,1,1\n"), "zero n accepted");
    PT_EXPECT(ctx, !load("engine_colour_row.csv", "0.30,1,1\n0.90 1 1\n"), "malformed row accepted");
}

// Interior property: two intervals from either end, where no boundary point enters the stencil, Sprague reproduces any quartic exactly.
PT_CHECK(sprague_reproduces_quartics, Fast, Exact) {
    ctx.plan(1);
    constexpr double kFirstNm = 380.0;
    constexpr double kIntervalNm = 10.0;
    constexpr int kSamples = 36;
    constexpr double kLastNm = kFirstNm + (kIntervalNm * (kSamples - 1));
    // Unit-scale in the centred variable, so the roundoff bound below applies without rescaling.
    const auto quartic = [](double nm) {
        const double u = (nm - 555.0) / 175.0;
        return 0.5 + (u * (0.3 + (u * (-0.2 + (u * (0.1 + (0.05 * u)))))));
    };
    std::vector<double> samples;
    for (int i = 0; i < kSamples; ++i) {
        samples.push_back(quartic(kFirstNm + (kIntervalNm * i)));
    }
    const cie::Spectrum resampled = cie::resampleSprague(kFirstNm, kIntervalNm, samples);
    double worst = 0.0;
    for (int i = 0; i < cie::kSampleCount; ++i) {
        const double nm = cie::wavelengthNm(i);
        if (nm >= kFirstNm + (2.0 * kIntervalNm) && nm <= kLastNm - (2.0 * kIntervalNm)) {
            worst = std::max(worst, std::abs(resampled[i] - quartic(nm)));
        }
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "Sprague misses a quartic by %.3e in the interior", worst);
    PT_EXPECT(ctx, worst <= kDoubleRoundoff, detail);
}

// Oracle: colorchecker_oracle.py, colour-science 0.4.6 Sprague align, Integration XYZ, AP1 NPM, Bradford, fed cie_1931.inc's CIE CSVs.
PT_CHECK(colorchecker_matches_colour_science, Fast, Exact) {
    // Printed to 12 decimals, so each oracle entry carries 5e-13 of quantisation on top of either side's double roundoff.
    const std::array<glm::dvec3, kPatchCount> oracle{{
        {0.136914534534, 0.089621054728, 0.062821513397},  // dark skin
        {0.447685891163, 0.314892111832, 0.232738721435},  // light skin
        {0.150315350402, 0.192552069902, 0.316023818371},  // blue sky
        {0.117128602679, 0.145656759749, 0.063940836275},  // foliage
        {0.231888328817, 0.221051083218, 0.402732382940},  // blue flower
        {0.270877866637, 0.489712999511, 0.411217375232},  // bluish green
        {0.508353947984, 0.233003741687, 0.060189274841},  // orange
        {0.094512650529, 0.107193938687, 0.354421390741},  // purplish blue
        {0.367815136402, 0.120546853997, 0.125473112058},  // moderate red
        {0.085383280674, 0.049339045378, 0.128381458468},  // purple
        {0.392127302608, 0.490907757834, 0.104919438439},  // yellow green
        {0.599729647293, 0.379467904022, 0.073590580570},  // orange yellow
        {0.044855437251, 0.050608819773, 0.259555700200},  // blue
        {0.145546038501, 0.282085268377, 0.090536626812},  // green
        {0.275966615599, 0.059895628428, 0.047296664921},  // red
        {0.721015418109, 0.587446458030, 0.087534586491},  // yellow
        {0.353141172959, 0.120863571220, 0.285892583621},  // magenta
        {0.085143551111, 0.231282248558, 0.359852293240},  // cyan
        {0.913858512399, 0.914864480358, 0.876450190967},  // white 9.5 (.05 D)
        {0.585134712457, 0.590343663849, 0.584730254791},  // neutral 8 (.23 D)
        {0.357341104017, 0.360505290980, 0.359128976931},  // neutral 6.5 (.44 D)
        {0.189389532725, 0.192003766174, 0.191744501772},  // neutral 5 (.70 D)
        {0.088262270534, 0.089861312977, 0.090683810813},  // neutral 3.5 (1.05 D)
        {0.032046858247, 0.031949207125, 0.032511305585},  // black 2 (1.5 D)

    }};
    const auto patches = pathtracer::config::loadChartCsv(kColorCheckerTable, "colour_validate");
    ctx.plan(1 + static_cast<int>(kPatchCount));
    PT_EXPECT(ctx, patches.has_value() && patches->size() == kPatchCount, "shipped ColorChecker table rejected");
    if (!patches || patches->size() != kPatchCount) {
        return;
    }
    for (std::size_t i = 0; i < kPatchCount; ++i) {
        const ChartPatch& patch = (*patches)[i];
        const double error = maxAbs(cie::reflectanceToScene(patch.reflectance) - oracle[i]);
        char detail[200];
        std::snprintf(detail, sizeof(detail), "%s: ACEScg off colour-science by %.3e", patch.name.c_str(), error);
        PT_EXPECT(ctx, error <= kDoubleRoundoff, detail);
    }
}

// Each row is a transcription mistake the loader must refuse rather than resample into a plausible-looking wrong chart.
PT_CHECK(colorchecker_table_rejects_invalid_input, Fast, Exact) {
    const auto load = [](const char* name, const std::string& text) {
        const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
        std::ofstream(path) << text;
        return pathtracer::config::loadChartCsv(path.string(), "colour_validate").has_value();
    };
    const std::string header = "patch,400,410,420,430,440,450\n";
    std::string rows;
    for (std::size_t i = 0; i < kPatchCount; ++i) {
        rows += "p" + std::to_string(i) + ",0.1,0.2,0.3,0.4,0.5,0.6\n";
    }
    ctx.plan(6);
    PT_EXPECT(ctx, load("engine_colour_cc_valid.csv", header + rows), "well-formed table rejected");
    PT_EXPECT(ctx, !load("engine_colour_cc_uneven.csv", "patch,400,410,420,430,440,460\n" + rows), "non-uniform grid accepted");
    PT_EXPECT(ctx, !load("engine_colour_cc_short.csv", "patch,400,410,420,430,440\n" + rows), "five-sample header accepted");
    PT_EXPECT(ctx, !load("engine_colour_cc_empty.csv", header), "table of no patches accepted");
    PT_EXPECT(ctx, !load("engine_colour_cc_range.csv", header + rows.substr(0, rows.rfind("0.6")) + "1.2\n"),
                  "reflectance above 1 accepted");
    PT_EXPECT(ctx, !load("engine_colour_cc_ragged.csv", header + rows.substr(0, rows.rfind(",0.6")) + "\n"),
                  "row shorter than the header accepted");
}

}  // namespace

PT_CHECK_MAIN("colour")
