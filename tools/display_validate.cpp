// Correctness gate for the CPU-evaluable display and statistics path: the OCIO display transform and the frame-time ring buffer.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "check.h"
#include "pathtracer/debug/frame_stats.h"
#include "pathtracer/debug/power_spectrum.h"
#include "pathtracer/gfx/ocio_cpu_transform.h"
#include "pathtracer/gfx/viewport.h"
#include "pathtracer/scene/sampler.h"

namespace {

// Applies the transform to one RGB triple and returns red; it is per-channel identical for a neutral input, so one channel suffices.
float transformScalar(float value) {
    std::vector<float> rgb{value, value, value};
    pathtracer::gfx::applyOcioDisplayTransform(rgb, 1, 1);
    return rgb[0];
}

// The anchors the pin exists to guarantee: scene-referred 0 must display as 0 and 1 as 1, which a filmic view would roll off.
PT_CHECK(ocio_view_is_colorimetric_at_its_anchors, Fast, Exact) {
    ctx.plan(2);
    // sRGB's curve is steep near zero but passes exactly through the origin; this tolerance is round-trip noise, not a tone allowance.
    constexpr float kAnchorTolerance = 1e-5F;
    const float atZero = transformScalar(0.0F);
    const float atOne = transformScalar(1.0F);
    char zeroDetail[160];
    std::snprintf(zeroDetail, sizeof(zeroDetail), "scene-referred 0.0 displays as %.9g, must be 0", 
                  static_cast<double>(atZero));
    PT_EXPECT(ctx, std::fabs(atZero) <= kAnchorTolerance, zeroDetail);
    char oneDetail[192];
    std::snprintf(oneDetail, sizeof(oneDetail),
                  "scene-referred 1.0 displays as %.9g, must be 1 -- a filmic view would roll this off",
                  static_cast<double>(atOne));
    PT_EXPECT(ctx, std::fabs(atOne - 1.0F) <= kAnchorTolerance, oneDetail);
}

// A display transform must be order-preserving, or a brighter scene value could display darker. Swept well above display white.
PT_CHECK(ocio_transform_is_monotone, Slow, Exact) {
    constexpr int kSteps = 256;
    ctx.plan(1);
    float previous = transformScalar(0.0F);
    int inversions = 0;
    float worstAt = 0.0F;
    for (int i = 1; i <= kSteps; ++i) {
        // Geometric-ish sweep to 16.0: linear steps would spend nearly every sample above display white and none in the toe.
        const float x = 16.0F * std::pow(static_cast<float>(i) / static_cast<float>(kSteps), 3.0F);
        const float y = transformScalar(x);
        if (!(y >= previous)) {
            if (inversions == 0) {
                worstAt = x;
            }
            ++inversions;
        }
        previous = y;
    }
    char detail[176];
    std::snprintf(detail, sizeof(detail), "%d inversions over %d steps to 16.0 (first at input %.6g)", inversions,
                  kSteps, static_cast<double>(worstAt));
    PT_EXPECT(ctx, inversions == 0, detail);
}

// The view is the scene gamut's matrix to the display primaries, then a per-channel curve: crosstalk is the matrix's, the curve's neutral.
PT_CHECK(ocio_transform_is_the_gamut_matrix_then_the_curve, Fast, Exact) {
    ctx.plan(3);
    // The scene space's own basis in the display primaries, read from the config (linear Rec.709 into the scene space, inverted).
    glm::mat3 rec709ToScene;
    for (int c = 0; c < 3; ++c) {
        std::array<float, 3> unit{};
        unit[static_cast<std::size_t>(c)] = 1.0F;
        pathtracer::gfx::convertToSceneSpace(unit, pathtracer::gfx::kOcioLinearRec709);
        rec709ToScene[c] = glm::vec3(unit[0], unit[1], unit[2]);
    }
    const glm::vec3 scene(0.3F, 0.5F, 0.2F);
    const glm::vec3 display = glm::inverse(rec709ToScene) * scene;
    std::vector<float> rgb{scene.r, scene.g, scene.b};
    pathtracer::gfx::applyOcioDisplayTransform(rgb, 1, 1);
    constexpr float kTolerance = 1e-5F;
    char detail[192];
    for (int c = 0; c < 3; ++c) {
        const float expected = transformScalar(display[c]);
        std::snprintf(detail, sizeof(detail), "channel %d is %.6g, the curve of the matrix output is %.6g", c, static_cast<double>(rgb[c]),
                      static_cast<double>(expected));
        PT_EXPECT(ctx, std::fabs(rgb[static_cast<std::size_t>(c)] - expected) <= kTolerance, detail);
    }
}

// Exercises the packed-image path at a real stride: a wrong stride or channel order transforms right values in wrong places.
PT_CHECK(ocio_transform_handles_a_real_image, Slow, Exact) {
    constexpr int kWidth = 13;  // deliberately not a multiple of any SIMD width
    constexpr int kHeight = 7;
    ctx.plan(1);
    std::vector<float> image(static_cast<std::size_t>(kWidth) * kHeight * 3);
    for (std::size_t i = 0; i < image.size(); ++i) {
        image[i] = static_cast<float>(i % 17) / 16.0F;
    }
    const std::vector<float> original = image;
    pathtracer::gfx::applyOcioDisplayTransform(image, kWidth, kHeight);

    std::size_t mismatched = 0;
    for (std::size_t texel = 0; texel < image.size(); texel += 3) {
        std::vector<float> expected(original.begin() + static_cast<std::ptrdiff_t>(texel), original.begin() + static_cast<std::ptrdiff_t>(texel) + 3);
        pathtracer::gfx::applyOcioDisplayTransform(expected, 1, 1);
        for (std::size_t lane = 0; lane < 3; ++lane) {
            mismatched += std::fabs(image[texel + lane] - expected[lane]) > 1e-6F ? 1 : 0;
        }
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu of %zu texel channels disagree with a one-texel transform of the same triple",
                  mismatched, image.size());
    PT_EXPECT(ctx, mismatched == 0, detail);
}

// The probe's Rec.709 LUT is BT.1886's inverse EOTF at Lw = 1, Lb = 0 (ITU-R BT.1886 Annex 1): V = L^(1/2.4), not sRGB's piecewise curve.
PT_CHECK(rec709_lut_is_the_bt1886_inverse_eotf, Fast, Exact) {
    constexpr int kSteps = 64;
    // OCIO's float pow round-off, the anchor check's bound; the sRGB curve in its place fails it.
    constexpr float kTolerance = 1e-5F;
    ctx.plan(1);
    float worst = 0.0F;
    for (int i = 0; i <= kSteps; ++i) {
        const float v = static_cast<float>(i) / kSteps;
        std::vector<float> rgb{v, v, v};
        pathtracer::gfx::applyOcioDisplayTransform(rgb, 1, 1, pathtracer::gfx::OcioDisplayTransform::Lut::Rec709);
        worst = std::max(worst, std::fabs(rgb[0] - std::pow(v, 1.0F / 2.4F)));
    }
    PT_EXPECT(ctx, worst <= kTolerance, "max |Rec.709 - L^(1/2.4)| = " + std::to_string(worst));
}

// The row-wise encode against a whole-image reference: transform the mapped frame in one OCIO call, then encode it untransformed.
PT_CHECK(encode_matches_whole_image_transform, Fast, Exact) {
    constexpr int kWidth = 13;  // deliberately not a multiple of any SIMD width
    constexpr int kHeight = 7;
    constexpr std::size_t kFloats = static_cast<std::size_t>(kWidth) * kHeight * 3;
    ctx.plan(3);
    const glm::vec3 gain(1.7F, 0.9F, 1.2F);
    const glm::vec3 offset(0.01F, -0.02F, 0.0F);
    // Negative, in-gamut and far over-range values, so the curve's linear toe, its power segment and the clamp are all exercised.
    std::mt19937_64 rng(ctx.seed());
    std::uniform_real_distribution<float> radiance(-0.5F, 8.0F);
    std::vector<float> rgb(kFloats);
    std::vector<float> mapped(kFloats);
    for (std::size_t i = 0; i < kFloats; ++i) {
        rgb[i] = radiance(rng);
        mapped[i] = (rgb[i] * gain[static_cast<glm::length_t>(i % 3)]) + offset[static_cast<glm::length_t>(i % 3)];
    }
    std::vector<unsigned char> rowWise(kFloats);
    pathtracer::gfx::encodeForDisplay(rgb, kWidth, kHeight, gain, true, offset, rowWise);
    std::vector<float> transformed = mapped;
    pathtracer::gfx::applyOcioDisplayTransform(transformed, kWidth, kHeight);
    std::vector<unsigned char> reference(kFloats);
    pathtracer::gfx::encodeForDisplay(transformed, kWidth, kHeight, glm::vec3(1.0F), false, glm::vec3(0.0F), reference);
    std::size_t differing = 0;
    for (std::size_t i = 0; i < kFloats; ++i) {
        differing += rowWise[i] != reference[i] ? 1 : 0;
    }
    PT_EXPECT(ctx, differing == 0, std::to_string(differing) + " of " + std::to_string(kFloats) + " bytes differ from the reference");
    // Independent of the encode's addressing: TPDF dither keeps |d| < 1/255, so each code lies within 1.5 of 255 clamp(v).
    std::size_t outOfBound = 0;
    for (std::size_t i = 0; i < kFloats; ++i) {
        const float expected = 255.0F * std::clamp(transformed[i], 0.0F, 1.0F);
        outOfBound += std::fabs(static_cast<float>(reference[i]) - expected) < 1.5F ? 0 : 1;
    }
    PT_EXPECT(ctx, outOfBound == 0, std::to_string(outOfBound) + " codes lie beyond the dither's reach of their value");
    std::vector<float> raw = mapped;
    pathtracer::gfx::applyOcioDisplayTransform(raw, kWidth, kHeight, pathtracer::gfx::OcioDisplayTransform::Lut::Raw);
    PT_EXPECT(ctx, raw == mapped, "the Raw LUT must leave every value untouched");
}

// --- Dither: one 128x128 tile of the mask, so every quantile of the TPDF appears exactly once in each test field.

constexpr int kTile = pathtracer::gfx::kDitherMaskSize;
constexpr int kTileTexels = pathtracer::gfx::kDitherMaskTexels;

// Codes of a flat field at display-referred v over one tile, untransformed at unit gain so only the dither and quantize act.
std::vector<unsigned char> encodeFlatTile(float v) {
    const std::vector<float> rgb(static_cast<std::size_t>(kTileTexels) * 3, v);
    std::vector<unsigned char> codes(rgb.size());
    pathtracer::gfx::encodeForDisplay(rgb, kTile, kTile, glm::vec3(1.0F), false, glm::vec3(0.0F), codes);
    return codes;
}

// Flat-field levels whose code +-1.5 stays inside [0, 255], so the clamp never acts and the dither alone decides each code.
std::vector<float> unclampedLevels(std::uint64_t seed) {
    constexpr int kLevels = 16;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> level(1.5F / 255.0F, 253.5F / 255.0F);
    std::vector<float> levels(kLevels);
    std::generate(levels.begin(), levels.end(), [&] { return level(rng); });
    return levels;
}

// TPDF on (-1, 1) codes: mean 0, variance 1/6. Midpoint quantiles of a function monotone on each half err by at most 2/N.
PT_CHECK(dither_mask_is_tpdf, Fast, Exact) {
    const std::array<float, kTileTexels>& mask = pathtracer::gfx::displayDitherMask();
    double sum = 0.0;
    double squares = 0.0;
    double peak = 0.0;
    for (const float d : mask) {
        const double code = 255.0 * d;
        sum += code;
        squares += code * code;
        peak = std::max(peak, std::fabs(code));
    }
    const double mean = sum / kTileTexels;
    const double variance = (squares / kTileTexels) - (mean * mean);
    constexpr double kQuadratureBound = 2.0 / kTileTexels;
    char detail[160];
    std::snprintf(detail, sizeof(detail), "mean %.3g, variance %.9f vs 1/6, peak %.9f codes", mean, variance, peak);
    ctx.plan(3);
    PT_EXPECT(ctx, std::fabs(mean) <= kQuadratureBound, detail);
    PT_EXPECT(ctx, std::fabs(variance - (1.0 / 6.0)) <= kQuadratureBound, detail);
    PT_EXPECT(ctx, peak < 1.0, detail);
}

// TPDF meets Schuchman's condition (1964), so E[Q(x + d)] = x; over the tile's midpoint quantiles, a <= 2-code step errs <= 2/N.
PT_CHECK(encode_of_flat_field_is_unbiased, Fast, Exact) {
    constexpr double kQuadratureBound = 2.0 / kTileTexels;
    double worst = 0.0;
    float worstLevel = 0.0F;
    for (const float v : unclampedLevels(ctx.seed())) {
        const std::vector<unsigned char> codes = encodeFlatTile(v);
        const double mean = std::accumulate(codes.begin(), codes.end(), 0.0) / static_cast<double>(codes.size());
        const double bias = std::fabs(mean - (255.0 * static_cast<double>(v)));
        if (bias > worst) {
            worst = bias;
            worstLevel = v;
        }
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "worst mean-code bias %.3g at v = %.6f, bound %.3g", worst,
                  static_cast<double>(worstLevel), kQuadratureBound);
    ctx.plan(1);
    PT_EXPECT(ctx, worst <= kQuadratureBound, detail);
}

// A flat field's quantization error must be blue, every threshold of a void-and-cluster mask being so; sampler_validate's gate.
PT_CHECK(encode_error_is_blue_noise, Fast, Statistical) {
    constexpr int kLowBand = 3;  // everything below an eighth of Nyquist, as sampler_validate's mask_is_blue_noise
    const std::array<double, pathtracer::debug::kSpectrumBands> null = pathtracer::debug::whiteNoiseBandShare(kTile, kTile);
    const double expected = std::accumulate(null.begin() + kLowBand, null.end(), 0.0);
    double worst = 0.0;
    for (const float v : unclampedLevels(ctx.seed())) {
        const std::vector<unsigned char> codes = encodeFlatTile(v);
        std::vector<double> error(static_cast<std::size_t>(kTileTexels));
        for (std::size_t i = 0; i < error.size(); ++i) {
            error[i] = static_cast<double>(codes[i * 3]) - (255.0 * static_cast<double>(v));
        }
        const std::array<double, pathtracer::debug::kSpectrumBands> bands =
            pathtracer::debug::octaveBandPower(error, kTile, kTile);
        const double measured = std::accumulate(bands.begin() + kLowBand, bands.end(), 0.0) /
                                std::accumulate(bands.begin(), bands.end(), 0.0);
        worst = std::max(worst, measured);
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "worst low band %.5f%% of power vs %.3f%% white-noise null", 100.0 * worst,
                  100.0 * expected);
    ctx.plan(1);
    PT_EXPECT(ctx, worst < 0.5 * expected, detail);
}

// Every sampler dither channel is a toroidal shift of its mask, so all 128^2 shifts cover every channel. sampler_validate's |r| gate.
PT_CHECK(dither_mask_independent_of_sampler, Fast, Statistical) {
    const std::array<float, kTileTexels>& mask = pathtracer::gfx::displayDitherMask();
    std::vector<double> display(mask.begin(), mask.end());
    std::vector<double> sampler(static_cast<std::size_t>(kTileTexels));
    for (int y = 0; y < kTile; ++y) {
        for (int x = 0; x < kTile; ++x) {
            sampler[(static_cast<std::size_t>(y) * kTile) + static_cast<std::size_t>(x)] =
                pathtracer::scene::blueNoiseDither(x, y, 0) - 0.5;  // mean-centred: the mask is uniform on [0,1)
        }
    }
    const double displayNorm = std::inner_product(display.begin(), display.end(), display.begin(), 0.0);
    const double samplerNorm = std::inner_product(sampler.begin(), sampler.end(), sampler.begin(), 0.0);
    const double norm = std::sqrt(displayNorm * samplerNorm);
    double worst = 0.0;
    for (int dy = 0; dy < kTile; ++dy) {
        for (int dx = 0; dx < kTile; ++dx) {
            double dot = 0.0;
            for (int y = 0; y < kTile; ++y) {
                const double* a = &display[static_cast<std::size_t>(y) * kTile];
                const double* b = &sampler[static_cast<std::size_t>((y + dy) % kTile) * kTile];
                // The row's shift split at the wrap, so the inner loops carry no modulo.
                for (int x = 0; x < kTile - dx; ++x) {
                    dot += a[x] * b[x + dx];
                }
                for (int x = kTile - dx; x < kTile; ++x) {
                    dot += a[x] * b[x + dx - kTile];
                }
            }
            worst = std::max(worst, std::fabs(dot) / norm);
        }
    }
    char detail[96];
    std::snprintf(detail, sizeof(detail), "worst |r| = %.4f over all %d shifts", worst, kTileTexels);
    ctx.plan(1);
    PT_EXPECT(ctx, worst < 0.1, detail);
}

// --- FrameStats: testable exactly only because tick() takes an injectable clock, steady_clock having made every assertion a race.

pathtracer::debug::FrameStats tickedWith(const std::vector<float>& millisecondGaps) {
    pathtracer::debug::FrameStats stats;
    auto now = std::chrono::steady_clock::time_point{};
    stats.tick(now);  // the first tick establishes the baseline and records nothing
    for (const float gap : millisecondGaps) {
        now += std::chrono::microseconds(static_cast<long long>(gap * 1000.0F));
        stats.tick(now);
    }
    return stats;
}

// The first tick has no predecessor, so it must record no interval: counting it enters a garbage frame time at startup.
PT_CHECK(frame_stats_ignores_the_first_tick, Fast, Exact) {
    ctx.plan(2);
    pathtracer::debug::FrameStats stats;
    stats.tick(std::chrono::steady_clock::time_point{});
    PT_EXPECT(ctx, stats.avgMs() == 0.0F, "a single tick must record no interval");
    PT_EXPECT(ctx, stats.fps() == 0.0F, "fps must be zero before any interval has been measured");
}

PT_CHECK(frame_stats_reports_exact_intervals, Fast, Exact) {
    ctx.plan(3);
    const pathtracer::debug::FrameStats stats = tickedWith({10.0F, 20.0F, 30.0F});
    char avgDetail[160];
    std::snprintf(avgDetail, sizeof(avgDetail), "mean of 10/20/30 ms read as %.6g", static_cast<double>(stats.avgMs()));
    PT_EXPECT(ctx, std::fabs(stats.avgMs() - 20.0F) <= 1e-3F, avgDetail);
    PT_EXPECT(ctx, std::fabs(stats.minMs() - 10.0F) <= 1e-3F, "minimum interval must be 10 ms");
    PT_EXPECT(ctx, std::fabs(stats.maxMs() - 30.0F) <= 1e-3F, "maximum interval must be 30 ms");
}

// The ring's wrap is where an off-by-one would mix a stale frame time into the window; identical intervals make a survivor visible.
PT_CHECK(frame_stats_ring_buffer_wraps_cleanly, Fast, Exact) {
    ctx.plan(2);
    std::vector<float> gaps;
    gaps.reserve(static_cast<std::size_t>(pathtracer::debug::FrameStats::kHistoryLength) * 2);
    for (int i = 0; i < pathtracer::debug::FrameStats::kHistoryLength; ++i) {
        gaps.push_back(99.0F);  // the stale value that must be fully overwritten
    }
    for (int i = 0; i < pathtracer::debug::FrameStats::kHistoryLength; ++i) {
        gaps.push_back(5.0F);
    }
    const pathtracer::debug::FrameStats stats = tickedWith(gaps);
    char detail[176];
    std::snprintf(detail, sizeof(detail), "after a full wrap the window reads min %.6g max %.6g, both must be 5 ms",
                  static_cast<double>(stats.minMs()), static_cast<double>(stats.maxMs()));
    PT_EXPECT(ctx, std::fabs(stats.maxMs() - 5.0F) <= 1e-3F, detail);
    PT_EXPECT(ctx, std::fabs(stats.minMs() - 5.0F) <= 1e-3F, detail);
}

// p50 and p95 against a known distribution. 120 samples cannot express a p99, so the assertion stops where the data does.
PT_CHECK(frame_stats_percentiles_are_correct, Fast, Exact) {
    ctx.plan(2);
    std::vector<float> gaps;
    gaps.reserve(100);
    for (int i = 1; i <= 100; ++i) {
        gaps.push_back(static_cast<float>(i));  // 1..100 ms, so the k-th percentile is k
    }
    const pathtracer::debug::FrameStats stats = tickedWith(gaps);
    char p50[160];
    std::snprintf(p50, sizeof(p50), "p50 of 1..100 ms read as %.6g", static_cast<double>(stats.percentileMs(0.5F)));
    PT_EXPECT(ctx, std::fabs(stats.percentileMs(0.5F) - 51.0F) <= 1.5F, p50);
    char p95[160];
    std::snprintf(p95, sizeof(p95), "p95 of 1..100 ms read as %.6g", static_cast<double>(stats.percentileMs(0.95F)));
    PT_EXPECT(ctx, std::fabs(stats.percentileMs(0.95F) - 96.0F) <= 1.5F, p95);
}

// The letterbox contract: the fitted rect stays inside the viewport, is centred in it, and never distorts the authored aspect.
PT_CHECK(viewport_fit_preserves_aspect_and_centres, Fast, Exact) {
    struct Case {
        const char* name;
        double imageWidth;
        double imageHeight;
        int viewportWidth;
        int viewportHeight;
        pathtracer::gfx::ViewportRect expected;
    };
    // Hand-checkable ratios: exact halves and thirds, so the expected rect is arithmetic rather than a recorded output.
    const std::vector<Case> cases = {
        {"matching aspect fills the viewport", 1024, 576, 1024, 576, {0, 0, 1024, 576}},
        {"matching aspect magnified 2x", 1024, 576, 2048, 1152, {0, 0, 2048, 1152}},
        {"wider viewport bars on x", 1024, 576, 2048, 576, {512, 0, 1024, 576}},
        {"taller viewport bars on y", 1024, 576, 1024, 1152, {0, 288, 1024, 576}},
        {"square viewport, 16:9 image", 1600, 900, 1000, 1000, {0, 218, 1000, 563}},
        {"minified below the authored size", 1024, 576, 512, 288, {0, 0, 512, 288}},
        {"non-positive viewport yields nothing", 1024, 576, 0, 576, {0, 0, 0, 0}},
        {"non-positive image yields nothing", 0, 576, 1024, 576, {0, 0, 0, 0}},
        // 256 columns by 128 hexagonal rows span 256 x 110.85 pitches: x8 fills the width, and 886.8 rows round to 887.
        {"hexagonal film's fractional height", 256, 128.0 * std::sqrt(3.0) / 2.0, 2048, 1024, {0, 68, 2048, 887}},
    };
    ctx.plan(static_cast<int>(cases.size()) * 3);
    for (const Case& testCase : cases) {
        const pathtracer::gfx::ViewportRect rect = pathtracer::gfx::fitAspect(
            testCase.imageWidth, testCase.imageHeight, testCase.viewportWidth, testCase.viewportHeight);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "%s: got {%d,%d,%d,%d}, expected {%d,%d,%d,%d}", testCase.name, rect.x,
                      rect.y, rect.width, rect.height, testCase.expected.x, testCase.expected.y,
                      testCase.expected.width, testCase.expected.height);
        PT_EXPECT(ctx, rect.x == testCase.expected.x && rect.y == testCase.expected.y &&
                           rect.width == testCase.expected.width && rect.height == testCase.expected.height,
                  detail);
        char bounds[224];
        std::snprintf(bounds, sizeof(bounds), "%s: rect {%d,%d,%d,%d} escapes a %dx%d viewport", testCase.name, rect.x,
                      rect.y, rect.width, rect.height, testCase.viewportWidth, testCase.viewportHeight);
        PT_EXPECT(ctx, rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= std::max(0, testCase.viewportWidth) &&
                           rect.y + rect.height <= std::max(0, testCase.viewportHeight),
                  bounds);
        // Half a pixel of rounding on each extent, propagated through w/h: d(w/h) <= (0.5 + 0.5*aspect)/h. Anything above that distorts.
        const bool degenerate = rect.width == 0 || rect.height == 0;
        const double imageAspect = degenerate ? 0.0
                                              : testCase.imageWidth / testCase.imageHeight;
        const double rectAspect = degenerate ? 0.0 : static_cast<double>(rect.width) / rect.height;
        const double allowed = degenerate ? 0.0 : (0.5 + (0.5 * imageAspect)) / rect.height;
        char aspect[224];
        std::snprintf(aspect, sizeof(aspect), "%s: aspect %.6f against the image's %.6f", testCase.name, rectAspect,
                      imageAspect);
        PT_EXPECT(ctx, std::fabs(rectAspect - imageAspect) <= allowed, aspect);
    }
}

}  // namespace

PT_CHECK_MAIN("display")
