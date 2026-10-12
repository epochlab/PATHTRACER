// Gate for CIE 1931 (scene/cie.cpp), the cone space on it (scene/cone_space.cpp), the metal fit, and the Fresnel shading chrome.

#include <cfloat>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>

#include <OpenColorIO/OpenColorIO.h>

#include "check.h"
#include "conductor_reference.h"
#include "pathtracer/config/scene_config.h"
#include "pathtracer/gfx/ocio_display_transform.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/cone_space.h"
#include "colorchecker.h"
#include "metal_fit.h"

namespace {

namespace cie = pathtracer::scene::cie;
namespace cone = pathtracer::scene::cone;
using tools::colorchecker::Patch;
using tools::metal_fit::Interpolation;

const std::string kChromiumTable = std::string(TOOLS_DATA_DIR) + "/chromium_johnson_christy_1974.csv";
const std::string kGoldTable = std::string(TOOLS_DATA_DIR) + "/gold_johnson_christy_1972.csv";
const std::string kColorCheckerTable = std::string(TOOLS_DATA_DIR) + "/colorchecker_babelcolor_average.csv";

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

// CIE A is Planck at c2/T = 1.435e-2/2848 (CIE 015:2018); the summation interval moves its 5th decimal, so 4 are asserted.
PT_CHECK(blackbody_reproduces_illuminant_a, Fast, Exact) {
    ctx.plan(1);
    constexpr double kKelvin = 2848.0 * 1.4388e-2 / 1.435e-2;
    constexpr glm::dvec2 kPublished(0.44757, 0.40745);
    const std::optional<glm::dvec3> rgb = cie::blackbodyToScene(kKelvin);
    const glm::dvec2 xy = rgb ? chromaticity(glm::inverse(cie::xyzToScene()) * *rgb) : glm::dvec2(0.0);
    const double error = std::max(std::abs(xy.x - kPublished.x), std::abs(xy.y - kPublished.y));
    char detail[160];
    std::snprintf(detail, sizeof(detail), "Planck at CIE A's exponent gives xy (%.6f, %.6f), %.3e from (0.44757, 0.40745)", xy.x, xy.y,
                  error);
    PT_EXPECT(ctx, error <= 5e-5, detail);
}

// Unit luminance makes intensity mean the same for a temperature as for color [1, 1, 1]; outside the gamut or domain is refused.
PT_CHECK(blackbody_has_unit_luminance_and_refuses_outside_the_scene_gamut, Fast, Exact) {
    constexpr std::array<double, 7> kInGamut{1000.0, 2000.0, 2856.0, 4000.0, 6504.0, 10000.0, 1e6};
    constexpr std::array<double, 5> kRefused{0.0, -2856.0, 500.0, std::numeric_limits<double>::infinity(),
                                             std::numeric_limits<double>::quiet_NaN()};
    ctx.plan(static_cast<int>(kInGamut.size() + kRefused.size()));
    const glm::dmat3 rgbToXyz = glm::inverse(cie::xyzToScene());
    char detail[160];
    for (const double kelvin : kInGamut) {
        const std::optional<glm::dvec3> rgb = cie::blackbodyToScene(kelvin);
        const double luminanceError = rgb ? std::abs((rgbToXyz * *rgb).y - 1.0) : 1.0;
        std::snprintf(detail, sizeof(detail), "%g K: %s, luminance off 1 by %.3e", kelvin, rgb ? "resolved" : "refused", luminanceError);
        PT_EXPECT(ctx, rgb && luminanceError <= kDoubleRoundoff, detail);
    }
    for (const double kelvin : kRefused) {
        std::snprintf(detail, sizeof(detail), "%g K resolved to a colour; it is outside the domain or the scene gamut", kelvin);
        PT_EXPECT(ctx, !cie::blackbodyToScene(kelvin), detail);
    }
}

// A tint above 1 caps, leaving F82 short of the data at mu-bar by (tint - 1) Schlick(F0): 1e-3 bounds gold's, measured 6.6e-4.
constexpr double kCappedTintMisfit = 1e-3;

// A material must be metal_fit's exact output; shading it must meet CIE-projected measurement at mu = 1 and, uncapped, mu-bar = 1/7.
void expectMeasuredMetal(tools::check::Context& ctx, const std::string& table, const char* materialFile) {
    constexpr int kExpectations = 15;
    ctx.plan(kExpectations);
    const auto nk = tools::metal_fit::loadNkTable(table);
    const auto material =
        pathtracer::config::loadMaterialConfig((std::filesystem::path(ASSET_ROOT_DIR) / "materials" / materialFile).string());
    const auto fit = nk ? tools::metal_fit::fitF82(*nk, Interpolation::Wavelength) : std::nullopt;
    if (!fit || !material) {
        for (int i = 0; i < kExpectations; ++i) {
            PT_EXPECT(ctx, false, "table, fit or material unavailable");
        }
        return;
    }
    const pathtracer::scene::BsdfClosure closure = pathtracer::scene::makeBsdfClosure(*material, glm::vec3(0.0F, 0.0F, 1.0F));
    const glm::vec3 normal = pathtracer::scene::fresnelAtViewAngle(closure, 1.0F);
    const glm::vec3 grazing = pathtracer::scene::fresnelAtViewAngle(closure, static_cast<float>(tools::reference::kMuBar));
    char detail[200];
    for (int c = 0; c < 3; ++c) {
        std::snprintf(detail, sizeof(detail), "%s base_color[%d] %.9g != metal_fit %.9g; re-run metal_fit", materialFile, c,
                      material->baseColor[c], static_cast<float>(fit->baseColor[c]));
        PT_EXPECT(ctx, material->baseColor[c] == static_cast<float>(fit->baseColor[c]), detail);
        std::snprintf(detail, sizeof(detail), "%s specular_color[%d] %.9g != metal_fit %.9g; re-run metal_fit", materialFile, c,
                      material->specularColor[c], static_cast<float>(fit->specularColor[c]));
        PT_EXPECT(ctx, material->specularColor[c] == static_cast<float>(fit->specularColor[c]), detail);
        const double normalError = std::abs(normal[c] - fit->baseColor[c]);
        std::snprintf(detail, sizeof(detail), "channel %d renderer R(mu=1) off the CIE target by %.3e", c, normalError);
        PT_EXPECT(ctx, normalError <= kFloatResolution, detail);
        const double model = tools::reference::referenceF82(fit->baseColor[c], fit->specularColor[c], tools::reference::kMuBar);
        const double grazingError = std::abs(grazing[c] - model);
        std::snprintf(detail, sizeof(detail), "channel %d renderer R(mu=1/7) off the F82 model by %.3e", c, grazingError);
        PT_EXPECT(ctx, grazingError <= kFloatResolution, detail);
        const double misfit = std::abs(model - fit->atMuBar[c]);
        std::snprintf(detail, sizeof(detail), "channel %d F82 misses the CIE target at mu-bar by %.3e (capped: %d)", c, misfit,
                      static_cast<int>(fit->specularCapped[c]));
        PT_EXPECT(ctx, fit->specularCapped[c] ? misfit <= kCappedTintMisfit : misfit <= kFloatResolution, detail);
    }
}

PT_CHECK(chrome_matches_measured_chromium, Fast, Exact) { expectMeasuredMetal(ctx, kChromiumTable, "chrome.json"); }

// Gold's F0 red is 1.038 in Rec.709 and no base_color carries it; in ACEScg all three F0 fit, and only the green tint caps.
PT_CHECK(gold_matches_measured_gold_in_the_scene_gamut, Fast, Exact) {
    expectMeasuredMetal(ctx, kGoldTable, "gold.json");
}

std::optional<std::vector<tools::metal_fit::NkSample>> loadText(const char* name, const std::string& text) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << text;
    return tools::metal_fit::loadNkTable(path.string());
}

// Each row is a table some real transcription or sourcing mistake produces; the shipped row proves the rejections reject only those.
PT_CHECK(nk_table_and_fit_reject_invalid_input, Fast, Exact) {
    ctx.plan(9);
    PT_EXPECT(ctx, tools::metal_fit::loadNkTable(kChromiumTable).has_value(), "shipped chromium table rejected");
    PT_EXPECT(ctx, loadText("engine_colour_crlf.csv", "0.30,1,1\r\n0.90,1,1\r\n").has_value(), "CRLF table rejected");
    PT_EXPECT(ctx, !loadText("engine_colour_gap.csv", "0.30,1,1\n0.80,1,1\n"), "table ending at 800 nm accepted");
    PT_EXPECT(ctx, !loadText("engine_colour_order.csv", "0.30,1,1\n0.50,1,1\n0.50,1,1\n0.90,1,1\n"),
                  "repeated wavelength accepted");
    PT_EXPECT(ctx, !loadText("engine_colour_k.csv", "0.30,1,1\n0.90,1,-0.1\n"), "negative k accepted");
    PT_EXPECT(ctx, !loadText("engine_colour_n.csv", "0.30,0,1\n0.90,1,1\n"), "zero n accepted");
    PT_EXPECT(ctx, !loadText("engine_colour_row.csv", "0.30,1,1\n0.90 1 1\n"), "malformed row accepted");
    // n = 0.01, k = 100 reflects 1 - 4e-6 at normal incidence: F82 has no reflectivity domain to leave, so a near-perfect mirror fits.
    const auto mirror = loadText("engine_colour_mirror.csv", "0.30,0.01,100\n0.90,0.01,100\n");
    PT_EXPECT(ctx, mirror && tools::metal_fit::fitF82(*mirror, Interpolation::Wavelength), "a near-perfect mirror was rejected");
    // Black but for a mirror band at 510-520 nm, bluer-green than AP1's green primary: its red projects below 0, which no base_color holds.
    const auto band = loadText("engine_colour_band.csv", "0.30,1,0\n0.50,1,0\n0.51,0.01,100\n0.52,0.01,100\n0.53,1,0\n0.90,1,0\n");
    PT_EXPECT(ctx, band && !tools::metal_fit::fitF82(*band, Interpolation::Wavelength), "an out-of-gamut base_color was accepted");
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
    const cie::Spectrum resampled = tools::colorchecker::resampleSprague(kFirstNm, kIntervalNm, samples);
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
    const std::array<glm::dvec3, tools::colorchecker::kPatchCount> oracle{{
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
    const auto patches = tools::colorchecker::loadColorChecker(kColorCheckerTable);
    ctx.plan(1 + tools::colorchecker::kPatchCount);
    PT_EXPECT(ctx, patches.has_value(), "shipped ColorChecker table rejected");
    if (!patches) {
        return;
    }
    for (int i = 0; i < tools::colorchecker::kPatchCount; ++i) {
        const Patch& patch = (*patches)[static_cast<std::size_t>(i)];
        const double error = maxAbs(patch.scene - oracle[static_cast<std::size_t>(i)]);
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
        return tools::colorchecker::loadColorChecker(path.string()).has_value();
    };
    const std::string header = "patch,400,410,420,430,440,450\n";
    std::string rows;
    for (int i = 0; i < tools::colorchecker::kPatchCount; ++i) {
        rows += "p" + std::to_string(i) + ",0.1,0.2,0.3,0.4,0.5,0.6\n";
    }
    ctx.plan(6);
    PT_EXPECT(ctx, load("engine_colour_cc_valid.csv", header + rows), "well-formed table rejected");
    PT_EXPECT(ctx, !load("engine_colour_cc_uneven.csv", "patch,400,410,420,430,440,460\n" + rows), "non-uniform grid accepted");
    PT_EXPECT(ctx, !load("engine_colour_cc_short.csv", "patch,400,410,420,430,440\n" + rows), "five-sample header accepted");
    PT_EXPECT(ctx, !load("engine_colour_cc_count.csv", header + rows + "extra,0.1,0.2,0.3,0.4,0.5,0.6\n"),
                  "25-patch table accepted");
    PT_EXPECT(ctx, !load("engine_colour_cc_range.csv", header + rows.substr(0, rows.rfind("0.6")) + "1.2\n"),
                  "reflectance above 1 accepted");
    PT_EXPECT(ctx, !load("engine_colour_cc_ragged.csv", header + rows.substr(0, rows.rfind(",0.6")) + "\n"),
                  "row shorter than the header accepted");
}

}  // namespace

// Hunt-Pointer-Estevez is the cone basis defined AS a 3x3 on CIE 1931 XYZ: S is Z alone, and equal energy excites the three equally.
PT_CHECK(cone_fundamentals_are_the_equal_energy_normalised_transform, Fast, Exact) {
    ctx.plan(2);
    const glm::dmat3& matrix = cone::xyzToLms();
    const glm::dvec3 shortRow(matrix[0][2], matrix[1][2], matrix[2][2]);
    PT_EXPECT(ctx, shortRow == glm::dvec3(0.0, 0.0, 1.0),
                  "the S row is not Z alone, so the transform is not stated on CIE 1931 XYZ");
    // Equal energy is XYZ (1,1,1) since the CMFs share a normalisation; the residual is the published matrix's own five decimals.
    const glm::dvec3 equalEnergy = matrix * glm::dvec3(1.0);
    const double spread = maxAbs(equalEnergy - glm::dvec3(equalEnergy.y));
    PT_EXPECT(ctx, spread <= 1e-4,
                  "equal energy excites the cones unequally, by " + std::to_string(spread));
}

// The per-texel basis folds four matrices and a white point into three rows, so it must reproduce the chromaticity it was folded from.
PT_CHECK(opponent_basis_matches_the_cone_chromaticity_definition, Fast, Exact) {
    const std::array<glm::dvec3, 5> stimuli = {glm::dvec3(1.0, 0.0, 0.0), glm::dvec3(0.0, 1.0, 0.0),
                                               glm::dvec3(0.0, 0.0, 1.0), glm::dvec3(0.31, 0.72, 0.09),
                                               glm::dvec3(0.5, 0.25, 0.875)};
    ctx.plan(static_cast<int>(stimuli.size()));
    const cone::OpponentBasis& basis = cone::opponentBasis();
    const glm::dmat3 rgbToXyz = glm::inverse(cie::xyzToScene());
    const glm::dvec2 white = cone::whiteConeChromaticity();
    // The s axis is rescaled so that one unit is one S excitation per unit luminance, which is this sum and nothing else.
    const double axisScale = static_cast<double>(basis.denominator.r) + static_cast<double>(basis.denominator.g) +
                             static_cast<double>(basis.denominator.b);

    char detail[192];
    for (std::size_t index = 0; index < stimuli.size(); ++index) {
        const glm::dvec3& rgb = stimuli[index];
        const glm::dvec2 chromaticity = cone::coneChromaticity(cone::xyzToLms() * (rgbToXyz * rgb));
        const glm::dvec2 expected(chromaticity.x - white.x, axisScale * (chromaticity.y - white.y));
        const glm::dvec2 difference(rgb.r - rgb.g, rgb.b - rgb.g);
        const double denominator = (basis.denominator.r * rgb.r) + (basis.denominator.g * rgb.g) +
                                   (basis.denominator.b * rgb.b);
        const glm::dvec2 folded(((basis.redGreenNumerator.x * difference.x) + (basis.redGreenNumerator.y * difference.y)) /
                                    denominator,
                                ((basis.blueYellowNumerator.x * difference.x) +
                                 (basis.blueYellowNumerator.y * difference.y)) / denominator);
        const double error = std::max(std::abs(folded.x - expected.x), std::abs(folded.y - expected.y));
        std::snprintf(detail, sizeof(detail), "stimulus %zu folds to (%.9g, %.9g) against (%.9g, %.9g)", index, folded.x,
                      folded.y, expected.x, expected.y);
        PT_EXPECT(ctx, error <= kFloatResolution * std::max(1.0, std::abs(expected.y)), detail);
    }
}

PT_CHECK_MAIN("colour")
