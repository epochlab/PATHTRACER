// Correctness gate for the CPU-evaluable display and statistics path: the OCIO display transform and the frame-time ring buffer.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "pathtracer/debug/frame_stats.h"
#include "pathtracer/gfx/ocio_cpu_transform.h"
#include "pathtracer/gfx/viewport.h"

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

// The transform must act per channel and identically on each: a matrix where a curve was intended leaves grey neutral but tints the rest.
PT_CHECK(ocio_transform_is_channel_independent, Fast, Exact) {
    ctx.plan(3);
    std::vector<float> rgb{0.25F, 0.5F, 0.75F};
    pathtracer::gfx::applyOcioDisplayTransform(rgb, 1, 1);
    const float r = transformScalar(0.25F);
    const float g = transformScalar(0.5F);
    const float b = transformScalar(0.75F);
    constexpr float kTolerance = 1e-6F;
    char detail[192];
    std::snprintf(detail, sizeof(detail), "packed (%.6g, %.6g, %.6g) vs per-channel (%.6g, %.6g, %.6g)",
                  static_cast<double>(rgb[0]), static_cast<double>(rgb[1]), static_cast<double>(rgb[2]),
                  static_cast<double>(r), static_cast<double>(g), static_cast<double>(b));
    PT_EXPECT(ctx, std::fabs(rgb[0] - r) <= kTolerance, detail);
    PT_EXPECT(ctx, std::fabs(rgb[1] - g) <= kTolerance, detail);
    PT_EXPECT(ctx, std::fabs(rgb[2] - b) <= kTolerance, detail);
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
    for (std::size_t i = 0; i < image.size(); ++i) {
        const float expected = transformScalar(original[i]);
        mismatched += std::fabs(image[i] - expected) > 1e-6F ? 1 : 0;
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu of %zu texel channels disagree with a per-scalar transform of the same value",
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
        int imageWidth;
        int imageHeight;
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
                                              : static_cast<double>(testCase.imageWidth) / testCase.imageHeight;
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
