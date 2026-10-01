// Gate for CIE 1931 (scene/cie.cpp), the cone space on it (scene/cone_space.cpp), the metal fit, and the Fresnel shading chrome.

#include <cfloat>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include "check.h"
#include "conductor_reference.h"
#include "pathtracer/config/scene_config.h"
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

// RP 177's matrix is pinned by these alone: the three primary chromaticities fix each column's direction, the white fixes their scales.
PT_CHECK(rec709_matrix_reproduces_primaries_and_white, Fast, Exact) {
    ctx.plan(5);
    const glm::dmat3 rgbToXyz = glm::inverse(cie::xyzToRec709());
    const std::array<glm::dvec2, 3> primaries = {glm::dvec2(0.640, 0.330), glm::dvec2(0.300, 0.600),
                                                 glm::dvec2(0.150, 0.060)};
    char detail[160];
    for (int c = 0; c < 3; ++c) {
        const double error = glm::length(chromaticity(rgbToXyz[c]) - primaries[c]);
        std::snprintf(detail, sizeof(detail), "primary %d chromaticity off BT.709 by %.3e", c, error);
        PT_EXPECT(ctx, error <= kDoubleRoundoff, detail);
    }
    cie::Spectrum flat;
    flat.fill(1.0);
    const double whiteError = maxAbs(cie::reflectanceToRec709(flat) - glm::dvec3(1.0));
    std::snprintf(detail, sizeof(detail), "perfect reflector maps %.3e away from (1, 1, 1)", whiteError);
    PT_EXPECT(ctx, whiteError <= kDoubleRoundoff, detail);
    constexpr double kMidGrey = 0.18;
    flat.fill(kMidGrey);
    const double greyError = maxAbs(cie::reflectanceToRec709(flat) - glm::dvec3(kMidGrey));
    std::snprintf(detail, sizeof(detail), "flat %.2f reflectance maps %.3e away from neutral", kMidGrey, greyError);
    PT_EXPECT(ctx, greyError <= kDoubleRoundoff, detail);
}

double rendererAverage(const pathtracer::scene::BsdfParams& params, int channel) {
    return tools::reference::cosineAverageFresnel(
        [&](double mu) { return static_cast<double>(pathtracer::scene::fresnelAtViewAngle(params, static_cast<float>(mu))[channel]); });
}

// chrome.json must be metal_fit's exact output; shading it must match CIE-projected measured chromium at normal incidence and on average.
PT_CHECK(chrome_matches_measured_chromium, Fast, Exact) {
    ctx.plan(12);
    const auto table = tools::metal_fit::loadNkTable(kChromiumTable);
    const auto material =
        pathtracer::config::loadMaterialConfig((std::filesystem::path(ASSET_ROOT_DIR) / "materials" / "chrome.json").string());
    const auto fit = table ? tools::metal_fit::fitGulbrandsen(*table, Interpolation::Wavelength) : std::nullopt;
    if (!fit || !material) {
        for (int i = 0; i < 12; ++i) {
            PT_EXPECT(ctx, false, "chromium table, fit or chrome.json unavailable");
        }
        return;
    }
    const pathtracer::scene::BsdfParams params{.baseColor = material->diffuseColour,
                                           .metallic = material->metallicFactor,
                                           .roughness = material->roughnessFactor,
                                           .f0 = material->diffuseColour,
                                           .edgeTint = material->edgeTint,
                                           .ior = material->ior,
                                           .transmissionFactor = 0.0F,
                                           .diffuseRoughness = 0.0F,
                                           .diffuseRho = material->diffuseColour,
                                           .transmissionTint = glm::vec3(1.0F)};
    const glm::vec3 normal = pathtracer::scene::fresnelAtViewAngle(params, 1.0F);
    char detail[200];
    for (int c = 0; c < 3; ++c) {
        std::snprintf(detail, sizeof(detail), "chrome.json diffuseColour[%d] %.9g != metal_fit %.9g; re-run metal_fit", c,
                      material->diffuseColour[c], static_cast<float>(fit->reflectivity[c]));
        PT_EXPECT(ctx, material->diffuseColour[c] == static_cast<float>(fit->reflectivity[c]), detail);
        std::snprintf(detail, sizeof(detail), "chrome.json edgeTint[%d] %.9g != metal_fit %.9g; re-run metal_fit", c,
                      material->edgeTint[c], static_cast<float>(fit->edgeTint[c]));
        PT_EXPECT(ctx, material->edgeTint[c] == static_cast<float>(fit->edgeTint[c]), detail);
        const double normalError = std::abs(normal[c] - fit->reflectivity[c]);
        std::snprintf(detail, sizeof(detail), "channel %d renderer R(mu=1) off the CIE target by %.3e", c, normalError);
        PT_EXPECT(ctx, normalError <= kFloatResolution, detail);
        const double averageError = std::abs(rendererAverage(params, c) - fit->averageTarget[c]);
        std::snprintf(detail, sizeof(detail), "channel %d renderer average off the CIE target by %.3e", c, averageError);
        PT_EXPECT(ctx, averageError <= kFloatResolution, detail);
    }
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
    // n = 0.01, k = 100 reflects 1 - 4e-6 at normal incidence, above the 0.9999 bsdf.cpp would clamp it to.
    const auto mirror = loadText("engine_colour_mirror.csv", "0.30,0.01,100\n0.90,0.01,100\n");
    PT_EXPECT(ctx, mirror && !tools::metal_fit::fitGulbrandsen(*mirror, Interpolation::Wavelength),
                  "reflectivity above the Gulbrandsen domain accepted");
    // A dielectric stepping from n = 1.2 to n = 10 at 555 nm projects to an average below every edgeTint at its reflectivity.
    const auto step = loadText("engine_colour_step.csv", "0.30,1.2,0\n0.55,1.2,0\n0.56,10,0\n0.90,10,0\n");
    PT_EXPECT(ctx, step && !tools::metal_fit::fitGulbrandsen(*step, Interpolation::Wavelength),
                  "average outside the edgeTint range accepted");
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

// Oracle: colour-science 0.4.6 Sprague align, Integration XYZ, BT.709 NPM, fed cie_1931.inc's CIE CSVs (its own D65 is 5 nm-interpolated).
PT_CHECK(colorchecker_matches_colour_science, Fast, Exact) {
    // Printed to 12 decimals, so each oracle entry carries 5e-13 of quantisation on top of either side's double roundoff.
    const std::array<glm::dvec3, tools::colorchecker::kPatchCount> oracle{{
        {0.172480673006, 0.083742975237, 0.057585578315},  // dark skin
        {0.548125514103, 0.298460038633, 0.216980613957},  // light skin
        {0.110266878137, 0.196752190552, 0.335929333768},  // blue sky
        {0.103821901447, 0.150234506643, 0.052123162248},  // foliage
        {0.224404585523, 0.217723858121, 0.430269691533},  // blue flower
        {0.123162717692, 0.519046931914, 0.404461871984},  // bluish green
        {0.716822676018, 0.198957573449, 0.027136959786},  // orange
        {0.064995722249, 0.106239257106, 0.392552066080},  // purplish blue
        {0.541695427015, 0.088284772070, 0.120289679529},  // moderate red
        {0.104210323250, 0.043810412110, 0.139609612723},  // purple
        {0.354628911947, 0.507844885402, 0.048234441858},  // yellow green
        {0.780444229676, 0.354000699765, 0.021502170189},  // orange yellow
        {0.023408077059, 0.049155285872, 0.291663042582},  // blue
        {0.065248390752, 0.301890964771, 0.064507829370},  // green
        {0.429315816563, 0.031882203252, 0.040180924337},  // red
        {0.856774201400, 0.575317964882, 0.007840168280},  // yellow
        {0.503128686293, 0.088866221202, 0.305565062881},  // magenta
        {-0.028565975691, 0.248963298201, 0.383032645845},  // cyan
        {0.916346914439, 0.915400535569, 0.870596899190},  // white 9.5 (.05 D)
        {0.581930354497, 0.591081386351, 0.583996466577},  // neutral 8 (.23 D)
        {0.355225344379, 0.360931981377, 0.358994374019},  // neutral 6.5 (.44 D)
        {0.187568435802, 0.192347040051, 0.191767604263},  // neutral 5 (.70 D)
        {0.087066702355, 0.090060938708, 0.090848048863},  // neutral 3.5 (1.05 D)
        {0.032068901815, 0.031930560325, 0.032594962823},  // black 2 (1.5 D)
    }};
    const auto patches = tools::colorchecker::loadColorChecker(kColorCheckerTable);
    ctx.plan(1 + tools::colorchecker::kPatchCount);
    PT_EXPECT(ctx, patches.has_value(), "shipped ColorChecker table rejected");
    if (!patches) {
        return;
    }
    for (int i = 0; i < tools::colorchecker::kPatchCount; ++i) {
        const Patch& patch = (*patches)[static_cast<std::size_t>(i)];
        const double error = maxAbs(patch.rec709 - oracle[static_cast<std::size_t>(i)]);
        char detail[200];
        std::snprintf(detail, sizeof(detail), "%s: linear Rec.709 off colour-science by %.3e", patch.name.c_str(), error);
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
    const glm::dmat3 rgbToXyz = glm::inverse(cie::xyzToRec709());
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
