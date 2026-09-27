// Correctness gate for the headless path: the AOV tables, the four CPU Beauty filters, and headless_renderer's dispatch.

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <numbers>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "check.h"
#include "pathtracer/api/headless_renderer.h"
#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/colormap.h"
#include "pathtracer/debug/scale_space.h"
#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using pathtracer::debug::AovId;
using pathtracer::debug::AovSource;
using pathtracer::gfx::HdrImage;

constexpr int kAovCount = static_cast<int>(AovId::Count);

[[nodiscard]] HdrImage makeImage(int width, int height, float value) {
    return HdrImage{width, height,
                    std::vector<float>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4, value)};
}

void setTexel(HdrImage& image, int x, int y, float r, float g, float b) {
    const std::size_t texel =
        ((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) + static_cast<std::size_t>(x)) * 4;
    image.rgba[texel] = r;
    image.rgba[texel + 1] = g;
    image.rgba[texel + 2] = b;
    image.rgba[texel + 3] = 1.0F;
}

[[nodiscard]] float texelR(const HdrImage& image, int x, int y) {
    return image.rgba[((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) +
                       static_cast<std::size_t>(x)) * 4];
}

}  // namespace

// Every AovId must be classified, sized and owned by the producer its classification names; an unclassified one routes to the rasterizer.
PT_CHECK(aov_tables_are_total_and_consistent, Fast, Exact) {
    ctx.plan(kAovCount * 3);
    for (int i = 0; i < kAovCount; ++i) {
        const auto aov = static_cast<AovId>(i);
        const int channels = pathtracer::debug::aovChannels(aov);
        PT_EXPECT(ctx, channels >= 1 && channels <= 3,
                      std::string(pathtracer::debug::kAovNames[i]) + " channels=" + std::to_string(channels));

        const pathtracer::debug::PathTracedLane traced = pathtracer::debug::pathTracedLane(aov);
        const pathtracer::debug::GBufferLane raster = pathtracer::debug::gbufferLane(aov);
        const AovSource source = pathtracer::debug::aovSource(aov);
        // Exactly one lane accessor may answer, and only the one the classification points at.
        const bool ownedCorrectly = (source == AovSource::PathTraced && traced != nullptr && raster == nullptr) ||
                                     (source == AovSource::GBuffer && raster != nullptr && traced == nullptr) ||
                                     (source == AovSource::BeautyFilter && traced == nullptr && raster == nullptr);
        PT_EXPECT(ctx, ownedCorrectly, std::string(pathtracer::debug::kAovNames[i]) + " lane/source disagree");

        // aovNeedsLightTransport is derived from aovSource; this pins the derivation itself.
        PT_EXPECT(ctx, pathtracer::debug::aovNeedsLightTransport(aov) == (source != AovSource::GBuffer),
                      std::string(pathtracer::debug::kAovNames[i]) + " transport flag disagrees with source");
    }
}

// The display names are the vocabulary every consumer spells an AOV in, so each must resolve, case- and separator-insensitively.
PT_CHECK(aov_names_round_trip, Fast, Exact) {
    ctx.plan(kAovCount + 5);
    for (int i = 0; i < kAovCount; ++i) {
        PT_EXPECT(ctx, pathtracer::debug::aovIdFromName(pathtracer::debug::kAovNames[i]) == static_cast<AovId>(i),
                      std::string("name does not resolve: ") + pathtracer::debug::kAovNames[i]);
    }
    PT_EXPECT(ctx, pathtracer::debug::aovIdFromName("bounce-count") == AovId::BounceCount, "hyphen form");
    PT_EXPECT(ctx, pathtracer::debug::aovIdFromName("BOUNCE_COUNT") == AovId::BounceCount, "upper snake form");
    PT_EXPECT(ctx, pathtracer::debug::aovIdFromName("indirectspecular") == AovId::IndirectSpecular, "run-together form");
    PT_EXPECT(ctx, pathtracer::debug::aovIdFromName("colour-opponent") == AovId::ColourOpponent, "two-word hyphen form");
    PT_EXPECT(ctx, pathtracer::debug::aovIdFromName("not an aov") == AovId::Count, "unknown name must not resolve");
}

// Morlet's admissibility term subtracts the envelope's own response to a constant, so the bank's parameters must leave zero mean.
PT_CHECK(morlet_bank_is_admissible_and_covers_every_orientation, Fast, Exact) {
    const std::array<float, 4> variances{1.0F, 4.0F, pathtracer::debug::innerScaleVariance(), 64.0F};
    ctx.plan(static_cast<int>(variances.size()) + 3);
    // Petkov 1995 eq. 4 read back: sigma*omega is fixed by the bandwidth alone, at one octave 2 sqrt(ln2/2) * 3.
    const double expected = 2.0 * std::sqrt(std::numbers::ln2 / 2.0) * 3.0;
    // Swept over the envelope, because the product is what the bandwidth fixes: a carrier not falling as 1/sigma would still pass at one.
    for (const float variance : variances) {
        const double sigmaOmega = pathtracer::debug::morletCarrier(variance) * std::sqrt(static_cast<double>(variance));
        PT_EXPECT(ctx, std::fabs(sigmaOmega - expected) <= 1e-6 * expected,
                      "at t=" + std::to_string(variance) + " sigma*omega is " + std::to_string(sigmaOmega) + ", not " +
                          std::to_string(expected));
    }

    const float variance = pathtracer::debug::innerScaleVariance();
    const double carrier = pathtracer::debug::morletCarrier(variance);
    // Below the Nyquist of the grid it runs on, or the carrier itself aliases and the bank measures a frequency that is not there.
    PT_EXPECT(ctx, carrier < std::numbers::pi,
                  "carrier " + std::to_string(carrier) + " is at or past the grid Nyquist");
    // Half response at a chord of sqrt(2 ln2)/sigma from the carrier, so this is how much of a half turn one orientation covers.
    const double angularWidth = 4.0 * std::asin(std::sqrt(2.0 * std::numbers::ln2) / (2.0 * expected));
    const double needed = std::numbers::pi / angularWidth;
    const int orientations = pathtracer::debug::morletOrientations();
    PT_EXPECT(ctx, static_cast<double>(orientations) >= needed,
                  std::to_string(orientations) + " orientations leave a gap, " + std::to_string(needed) + " being needed");
    // And no more than that: one fewer must fail to cover, or the count is not the one the bandwidth forces but a larger choice.
    PT_EXPECT(ctx, static_cast<double>(orientations - 1) < needed,
                  std::to_string(orientations) + " orientations exceed the " + std::to_string(needed) + " the bandwidth needs");
}

// No gradient and no AC content, so both filters must read exactly zero, including at the border where clamping repeats the constant.
PT_CHECK(filters_are_zero_on_a_constant_field, Fast, Exact) {
    ctx.plan(2);
    pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const HdrImage flat = makeImage(24, 16, 0.375F);

    // Read the response channel per texel, not the raw rgba span: alpha is 1 by the broadcast convention and would dominate a reduction.
    const HdrImage sobel = pathtracer::debug::sobelAov(flat, pool);
    float worstSobel = 0.0F;
    for (int y = 0; y < sobel.height; ++y) {
        for (int x = 0; x < sobel.width; ++x) {
            worstSobel = std::max(worstSobel, texelR(sobel, x, y));
        }
    }
    PT_EXPECT(ctx, worstSobel == 0.0F, "peak Sobel response " + std::to_string(worstSobel));

    const HdrImage gabor = pathtracer::debug::gaborAov(flat, pool);
    float worstGabor = 0.0F;
    for (int y = 0; y < gabor.height; ++y) {
        for (int x = 0; x < gabor.width; ++x) {
            worstGabor = std::max(worstGabor, texelR(gabor, x, y));
        }
    }
    // Two epsilons for the separability gap the filter suite bounds, plus one each for scaling that term by the field and differencing.
    const double bound = static_cast<double>(0.375F) * 4.0 * static_cast<double>(FLT_EPSILON);
    PT_EXPECT(ctx, static_cast<double>(worstGabor) <= bound,
                  "peak Morlet response " + std::to_string(worstGabor) + " over the cancellation bound " + std::to_string(bound));
}

// A bipolar AOV claims both signs are meaningful, so each must actually produce both on a pattern with structure at several scales.
PT_CHECK(bipolar_aovs_produce_both_signs, Fast, Exact) {
    pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    // Large enough for the octave ladder to reach two levels: below that DoG has no band to difference and correctly reads zero.
    constexpr int kWidth = 256;
    constexpr int kHeight = 192;
    HdrImage pattern = makeImage(kWidth, kHeight, 0.0F);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            // Per-channel phases, so the chromatic axes swing either side of white as well as the achromatic ones.
            const auto fx = static_cast<float>(x);
            const auto fy = static_cast<float>(y);
            setTexel(pattern, x, y, 0.5F + (0.4F * std::sin(0.5F * fx)), 0.5F + (0.4F * std::sin(0.11F * fy)),
                     0.5F + (0.4F * std::cos(0.23F * (fx + fy))));
        }
    }

    std::vector<AovId> bipolar;
    for (int i = 0; i < kAovCount; ++i) {
        const auto aov = static_cast<AovId>(i);
        if (pathtracer::debug::aovIsBipolar(aov) && pathtracer::debug::aovSource(aov) == AovSource::BeautyFilter) {
            bipolar.push_back(aov);
        }
    }
    ctx.plan(static_cast<int>(bipolar.size()) + 1);
    PT_EXPECT(ctx, !bipolar.empty(), "no bipolar filter AOV exists, so this check would assert nothing");

    for (const AovId aov : bipolar) {
        const HdrImage out = pathtracer::debug::evaluateFilterAov(
            aov, pathtracer::debug::FilterInput{pattern, static_cast<float>(pattern.height) / glm::radians(30.0F), nullptr, 0}, pool);
        const int channels = pathtracer::debug::aovChannels(aov);
        float lowest = 0.0F;
        float highest = 0.0F;
        for (int i = 0; i < out.width * out.height; ++i) {
            for (int c = 0; c < channels; ++c) {
                const float value = out.rgba[(static_cast<std::size_t>(i) * 4) + static_cast<std::size_t>(c)];
                lowest = std::min(lowest, value);
                highest = std::max(highest, value);
            }
        }
        PT_EXPECT(ctx, lowest < 0.0F && highest > 0.0F,
                      std::string(pathtracer::debug::kAovNames[static_cast<int>(aov)]) + " spans [" +
                          std::to_string(lowest) + ", " + std::to_string(highest) + "], not both signs");
    }
}

// The auto-range must equal the peak on a field with no outlier and fall below it on one, or a lone texel crushes the whole preview.
PT_CHECK(bipolar_range_caps_a_lone_outlier, Fast, Exact) {
    ctx.plan(4);
    constexpr int kPixels = 4096;
    std::vector<float> rgba(static_cast<std::size_t>(kPixels) * 4, 0.0F);
    // A zero-centred square wave: every sample has the same magnitude, so the peak is exactly the RMS and no tail exists to cap.
    for (int i = 0; i < kPixels; ++i) {
        rgba[static_cast<std::size_t>(i) * 4] = (i % 2) == 0 ? 1.0F : -1.0F;
    }
    // The gain is the offset divided by the range, so a unit range must reproduce the offset exactly rather than within a rounding.
    const float flat = pathtracer::debug::bipolarDisplay(rgba, 1).gain[0];
    // sqrt(2 ln n) > 1 for any n > 1, so the extreme-value cap cannot bite here and the true peak must survive it.
    PT_EXPECT(ctx, flat == pathtracer::debug::kBipolarDisplayOffset,
                  "a constant-magnitude field gained " + std::to_string(flat) + ", not the unit-range offset");

    rgba[0] = 1024.0F;
    const float outlier = pathtracer::debug::bipolarDisplay(rgba, 1).gain[0];
    PT_EXPECT(ctx, outlier > pathtracer::debug::kBipolarDisplayOffset / 1024.0F,
                  "a lone outlier still set the range to its own peak, gaining " + std::to_string(outlier));
    // One sample of 1024 among 4096 unit samples lifts the RMS to sqrt(1 + 1024^2/4096) = 16.03, and sqrt(2 ln 4096) = 4.08 scales it.
    const double rms = std::sqrt(1.0 + ((1024.0 * 1024.0) - 1.0) / 4096.0);
    const double expected =
        static_cast<double>(pathtracer::debug::kBipolarDisplayOffset) / (rms * std::sqrt(2.0 * std::log(4096.0)));
    PT_EXPECT(ctx, std::fabs(static_cast<double>(outlier) - expected) <= expected * 1e-6,
                  "gain " + std::to_string(outlier) + " against the extreme-value prediction " + std::to_string(expected));

    const std::vector<float> zeros(static_cast<std::size_t>(kPixels) * 4, 0.0F);
    PT_EXPECT(ctx, pathtracer::debug::bipolarDisplay(zeros, 1).gain[0] == 1.0F,
                  "a field with no range did not fall back to unit gain");
}

// The preview's affine map must carry -range to 0 and +range to 1 at every amplitude, or a signed AOV clips or collapses to mid-grey.
PT_CHECK(bipolar_display_maps_the_range_to_the_unit_interval, Fast, Exact) {
    // Powers of two, so the gain and its product with the amplitude are both exact and the comparison needs no tolerance to hide in.
    const std::vector<float> amplitudes{1.0F, 0.5F, 1.0F / 1024.0F, 4096.0F, 0.0F};
    ctx.plan(static_cast<int>(amplitudes.size()) * 2);
    constexpr int kPixels = 4096;
    for (const float amplitude : amplitudes) {
        std::vector<float> rgba(static_cast<std::size_t>(kPixels) * 4, 0.0F);
        // Constant magnitude again, so the field's range is exactly the amplitude and this gates the map rather than the range scan.
        for (int i = 0; i < kPixels; ++i) {
            rgba[static_cast<std::size_t>(i) * 4] = (i % 2) == 0 ? amplitude : -amplitude;
        }
        const pathtracer::debug::BipolarDisplay display = pathtracer::debug::bipolarDisplay(rgba, 1);
        const float low = (-amplitude * display.gain[0]) + display.offset[0];
        const float high = (amplitude * display.gain[0]) + display.offset[0];
        // A zero amplitude is the one degenerate case: there is nothing to map, and both ends must land on mid-grey rather than diverge.
        const float expectedLow = amplitude > 0.0F ? 0.0F : pathtracer::debug::kBipolarDisplayOffset;
        const float expectedHigh = amplitude > 0.0F ? 1.0F : pathtracer::debug::kBipolarDisplayOffset;
        PT_EXPECT(ctx, low == expectedLow, "amplitude " + std::to_string(amplitude) + " maps its floor to " + std::to_string(low));
        PT_EXPECT(ctx, high == expectedHigh, "amplitude " + std::to_string(amplitude) + " maps its ceiling to " + std::to_string(high));
    }
}

// Two lanes of one AOV can be incomparable quantities: Colour Opponent's axes differ by 77x over the gamut, so each ranges alone.
PT_CHECK(bipolar_lanes_range_independently, Fast, Exact) {
    ctx.plan(3);
    constexpr int kPixels = 4096;
    constexpr float kNarrow = 1.0F / 128.0F;
    constexpr float kWide = 1024.0F;
    std::vector<float> rgba(static_cast<std::size_t>(kPixels) * 4, 0.0F);
    for (int i = 0; i < kPixels; ++i) {
        const float sign = (i % 2) == 0 ? 1.0F : -1.0F;
        rgba[static_cast<std::size_t>(i) * 4] = sign * kNarrow;
        rgba[(static_cast<std::size_t>(i) * 4) + 1] = sign * kWide;
    }
    const pathtracer::debug::BipolarDisplay display = pathtracer::debug::bipolarDisplay(rgba, 2);
    const float narrow = (kNarrow * display.gain[0]) + display.offset[0];
    const float wide = (kWide * display.gain[1]) + display.offset[1];
    PT_EXPECT(ctx, narrow == 1.0F, "the narrow lane reaches only " + std::to_string(narrow) + " of the display, not 1");
    PT_EXPECT(ctx, wide == 1.0F, "the wide lane reaches " + std::to_string(wide) + ", not 1");
    // A pooled range would set both gains from the wider lane, so this ratio is exactly what separates the per-lane map from that one.
    PT_EXPECT(ctx, display.gain[0] == display.gain[1] * (kWide / kNarrow),
                  "the lanes' gains differ by " + std::to_string(display.gain[0] / display.gain[1]) + ", not the " +
                      std::to_string(kWide / kNarrow) + " their amplitudes do");
}

// A lane the AOV never defined carries no measurement, so the preview must leave it black rather than assert a zero response at mid-grey.
PT_CHECK(bipolar_absent_lanes_render_black, Fast, Exact) {
    ctx.plan(4);
    constexpr int kPixels = 256;
    std::vector<float> rgba(static_cast<std::size_t>(kPixels) * 4, 0.0F);
    std::vector<float> broadcast(static_cast<std::size_t>(kPixels) * 4, 0.0F);
    for (int i = 0; i < kPixels; ++i) {
        const float sign = (i % 2) == 0 ? 1.0F : -1.0F;
        rgba[static_cast<std::size_t>(i) * 4] = sign;
        rgba[(static_cast<std::size_t>(i) * 4) + 1] = sign * 0.5F;
        for (int c = 0; c < 3; ++c) {
            broadcast[(static_cast<std::size_t>(i) * 4) + static_cast<std::size_t>(c)] = sign;
        }
    }
    const pathtracer::debug::BipolarDisplay two = pathtracer::debug::bipolarDisplay(rgba, 2);
    PT_EXPECT(ctx, two.gain[2] == 0.0F && two.offset[2] == 0.0F,
                  "a two-channel AOV's third lane gained " + std::to_string(two.gain[2]) + " at offset " +
                      std::to_string(two.offset[2]) + ", so an undefined lane would show");
    PT_EXPECT(ctx, two.offset[0] == pathtracer::debug::kBipolarDisplayOffset &&
                       two.offset[1] == pathtracer::debug::kBipolarDisplayOffset,
                  "a defined lane lost the mid-grey offset");
    // writeScalar broadcasts a scalar AOV across all three lanes, so declaring one channel must still light all three or its grey tints.
    const pathtracer::debug::BipolarDisplay one = pathtracer::debug::bipolarDisplay(broadcast, 1);
    PT_EXPECT(ctx, one.gain == glm::vec3(one.gain[0]), "a scalar AOV's three lanes gained differently, so its grey would tint");
    PT_EXPECT(ctx, one.offset == glm::vec3(pathtracer::debug::kBipolarDisplayOffset),
                  "a scalar AOV lost the mid-grey offset on a lane");
}

// Builds one interleaved RGBA frame from a scalar broadcast across its three lanes, as writeScalar does for every scalar AOV.
[[nodiscard]] std::vector<float> broadcastScalarFrame(const std::vector<float>& values) {
    std::vector<float> rgba(values.size() * 4, 0.0F);
    for (std::size_t i = 0; i < values.size(); ++i) {
        for (std::size_t lane = 0; lane < 3; ++lane) {
            rgba[(i * 4) + lane] = values[i];
        }
        rgba[(i * 4) + 3] = 1.0F;
    }
    return rgba;
}

// Reads a pre-mapped display lane, or a sentinel neither pre-map can produce, so a vanished pre-map fails an assertion not the process.
[[nodiscard]] float displayLane(const pathtracer::debug::AovDisplay& display, std::size_t index) {
    return index < display.rgba.size() ? display.rgba[index] : -1.0F;
}

// The log window's ceiling is the ratio sqrt(n), the SNR of a texel at unit per-sample coefficient of variation, so it must read white.
PT_CHECK(snr_display_maps_the_root_of_the_pass_count_to_white, Fast, Exact) {
    // Powers of four, so sqrt(n) is an exact power of two and log(sqrt(n)) * 2/log(n) rounds to exactly 1 rather than near it.
    const std::vector<int> passCounts{4, 64, 1024, 4096};
    ctx.plan(static_cast<int>(passCounts.size()) * 2);
    for (const int samples : passCounts) {
        const float ceiling = std::sqrt(static_cast<float>(samples));
        const std::vector<float> rgba = broadcastScalarFrame({ceiling, ceiling * 4.0F});
        const pathtracer::debug::AovDisplay display =
            pathtracer::debug::aovDisplay(AovId::SNR, rgba, {samples, 0});
        PT_EXPECT(ctx, displayLane(display, 0) == 1.0F,
                      "the ratio sqrt(" + std::to_string(samples) + ") displayed " + std::to_string(displayLane(display, 0)) +
                          ", not white");
        // Beyond the reference there is no more window, so the clamp must hold rather than let a well-converged texel run past 1.
        PT_EXPECT(ctx, displayLane(display, 4) == 1.0F,
                      "a ratio past the reference displayed " + std::to_string(displayLane(display, 4)) + ", unclamped");
    }
}

// Decibels are a log scale, so squaring the ratio must double the displayed value; any linear or power map fails this identically.
PT_CHECK(snr_display_is_logarithmic_in_the_ratio, Fast, Exact) {
    const std::vector<float> ratios{1.5F, 2.0F, 3.0F, 4.0F};
    ctx.plan(static_cast<int>(ratios.size()));
    constexpr int kSamples = 1024;
    for (const float ratio : ratios) {
        const std::vector<float> rgba = broadcastScalarFrame({ratio, ratio * ratio});
        const pathtracer::debug::AovDisplay display =
            pathtracer::debug::aovDisplay(AovId::SNR, rgba, {kSamples, 0});
        // Toleranced at float's unit roundoff, not exact: the identity is the property, and its bit-exactness here is a rounding accident.
        PT_EXPECT(ctx, std::fabs(displayLane(display, 4) - (2.0F * displayLane(display, 0))) <= 0x1p-23F * displayLane(display, 4),
                      "ratio " + std::to_string(ratio) + " displayed " + std::to_string(displayLane(display, 0)) +
                          " but its square displayed " + std::to_string(displayLane(display, 4)) + ", not twice that");
    }
}

// Below the unit ratio a texel's value is under its own uncertainty and nothing is resolved, which is also where snrAov's zero lands.
PT_CHECK(snr_display_floors_the_unit_ratio_and_below, Fast, Exact) {
    const std::vector<float> ratios{1.0F, 0.5F, 0.0F};
    ctx.plan(static_cast<int>(ratios.size()) + 1);
    const std::vector<float> rgba = broadcastScalarFrame(ratios);
    const pathtracer::debug::AovDisplay display = pathtracer::debug::aovDisplay(AovId::SNR, rgba, {1024, 0});
    for (std::size_t i = 0; i < ratios.size(); ++i) {
        PT_EXPECT(ctx, displayLane(display, i * 4) == 0.0F,
                      "the ratio " + std::to_string(ratios[i]) + " displayed " + std::to_string(displayLane(display, i * 4)) +
                          " rather than the floor");
    }
    // A scalar AOV is broadcast, so a lane that drifted from lane 0 would tint the grey the whole point of this display is to keep.
    PT_EXPECT(ctx, displayLane(display, 1) == displayLane(display, 0) && displayLane(display, 2) == displayLane(display, 0),
                  "the three display lanes disagree, so the preview would tint");
}

// A variance needs two passes, so below that snrAov is uniformly zero and log(n) offers no positive ceiling to divide by.
PT_CHECK(snr_display_is_absent_below_two_passes, Fast, Exact) {
    ctx.plan(3);
    const std::vector<float> rgba = broadcastScalarFrame({4.0F, 16.0F});
    for (const int samples : {0, 1}) {
        PT_EXPECT(ctx, pathtracer::debug::aovDisplay(AovId::SNR, rgba, {samples, 0}).rgba.empty(),
                      "SNR pre-mapped at " + std::to_string(samples) + " passes, where its own value is undefined");
    }
    PT_EXPECT(ctx, !pathtracer::debug::aovDisplay(AovId::SNR, rgba, {2, 0}).rgba.empty(),
                  "SNR did not pre-map at two passes, the fewest a variance is defined for");
}

// Bounce Count's domain is maxBounces + 1 terminations, a real bound, so the shared path must reproduce the colormap exactly.
PT_CHECK(bounce_count_display_matches_the_turbo_colormap, Fast, Exact) {
    constexpr int kMaxBounces = 8;
    const std::vector<float> counts{0.0F, 1.0F, 2.5F, 4.0F, 8.0F, 9.0F};
    ctx.plan(static_cast<int>(counts.size()));
    const std::vector<float> rgba = broadcastScalarFrame(counts);
    const pathtracer::debug::AovDisplay display =
        pathtracer::debug::aovDisplay(AovId::BounceCount, rgba, {0, kMaxBounces});
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const glm::vec3 expected = pathtracer::debug::turbo(counts[i] / (static_cast<float>(kMaxBounces) + 1.0F));
        const glm::vec3 got{displayLane(display, i * 4), displayLane(display, (i * 4) + 1), displayLane(display, (i * 4) + 2)};
        PT_EXPECT(ctx, got == expected,
                      "bounce depth " + std::to_string(counts[i]) + " displayed (" + std::to_string(got.r) + ", " +
                          std::to_string(got.g) + ", " + std::to_string(got.b) + "), not the colormap's colour");
    }
}

// Raw metres quantize to white and farClip is a ray bound rather than a depth span, so Depth's gain comes from the depth present.
PT_CHECK(depth_display_ranges_to_its_own_maximum, Fast, Exact) {
    // Powers of two, so 2^-log2(max) is exact and the product with the maximum must be exactly 1 rather than within a rounding.
    const std::vector<float> maxima{0.25F, 1.0F, 64.0F};
    ctx.plan(static_cast<int>(maxima.size()));
    for (const float maximum : maxima) {
        const std::vector<float> rgba = broadcastScalarFrame({maximum * 0.5F, maximum});
        const pathtracer::debug::AovDisplay display = pathtracer::debug::aovDisplay(AovId::Depth, rgba, {0, 0});
        const float white = (maximum * display.affine.gain[0]) + display.affine.offset[0];
        PT_EXPECT(ctx, white == 1.0F,
                      "the farthest depth " + std::to_string(maximum) + " displayed " + std::to_string(white) + ", not white");
    }
}

// An auto-ranged AOV has already absorbed the scene's scale, so applying the photographic exposure on top would range it twice.
PT_CHECK(an_auto_ranged_aov_never_also_takes_the_exposure, Fast, Exact) {
    ctx.plan(kAovCount);
    // Spread over two decades and positive, so every arm of aovDisplay that ranges at all leaves a map this can tell from identity.
    const std::vector<float> rgba = broadcastScalarFrame({0.5F, 4.0F, 16.0F});
    for (int i = 0; i < kAovCount; ++i) {
        const auto aov = static_cast<AovId>(i);
        const pathtracer::debug::AovDisplay display = pathtracer::debug::aovDisplay(aov, rgba, {1024, 8});
        // Whatever aovDisplay decided from the values -- a pre-map or any map but unity gain at zero offset -- is a range already taken.
        const bool ranged = !display.rgba.empty() || display.affine.gain != glm::vec3(1.0F) ||
                            display.affine.offset != glm::vec3(0.0F);
        PT_EXPECT(ctx, !(ranged && pathtracer::debug::aovTakesDisplayExposure(aov)),
                      std::string(pathtracer::debug::kAovNames[i]) + " auto-ranges and takes the exposure, so it would range twice");
    }
}

// A nonlinear pre-map costs a full frame copy, so it must apply only where the affine map genuinely cannot express the display.
PT_CHECK(display_premap_applies_to_exactly_two_aovs, Fast, Exact) {
    ctx.plan(kAovCount);
    const std::vector<float> rgba = broadcastScalarFrame({0.5F, 4.0F, 16.0F});
    for (int i = 0; i < kAovCount; ++i) {
        const auto aov = static_cast<AovId>(i);
        const bool premapped = aov == AovId::SNR || aov == AovId::BounceCount;
        PT_EXPECT(ctx, pathtracer::debug::aovDisplay(aov, rgba, {1024, 8}).rgba.empty() != premapped,
                      std::string(pathtracer::debug::kAovNames[i]) +
                          (premapped ? " lost its pre-map" : " gained a pre-map it does not need"));
    }
}

// A unit step edge has a closed-form Sobel magnitude: both adjacent columns read exactly 4 and everything further exactly 0.
PT_CHECK(sobel_step_edge_matches_closed_form, Fast, Exact) {
    ctx.plan(4);
    pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 16;
    constexpr int kHeight = 12;
    constexpr int kEdge = 8;

    HdrImage step = makeImage(kWidth, kHeight, 0.0F);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            // White luminance 1.0 requires all three channels at 1, since the Rec.709 weights sum to 1.
            const float value = x >= kEdge ? 1.0F : 0.0F;
            setTexel(step, x, y, value, value, value);
        }
    }

    const HdrImage sobel = pathtracer::debug::sobelAov(step, pool);
    const int midRow = kHeight / 2;
    PT_EXPECT(ctx, texelR(sobel, kEdge - 1, midRow) == 4.0F,
                  "left of edge " + std::to_string(texelR(sobel, kEdge - 1, midRow)));
    PT_EXPECT(ctx, texelR(sobel, kEdge, midRow) == 4.0F,
                  "right of edge " + std::to_string(texelR(sobel, kEdge, midRow)));
    PT_EXPECT(ctx, texelR(sobel, kEdge - 3, midRow) == 0.0F,
                  "far left " + std::to_string(texelR(sobel, kEdge - 3, midRow)));
    PT_EXPECT(ctx, texelR(sobel, kEdge + 3, midRow) == 0.0F,
                  "far right " + std::to_string(texelR(sobel, kEdge + 3, midRow)));
}

// Luminance IS the Rec.709 weighted sum (ITU-R BT.709-6), so each primary returns its own weight exactly: a definition, not a measurement.
PT_CHECK(luminance_returns_rec709_weights, Fast, Exact) {
    ctx.plan(4);
    pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    HdrImage primaries = makeImage(4, 1, 0.0F);
    setTexel(primaries, 0, 0, 1.0F, 0.0F, 0.0F);
    setTexel(primaries, 1, 0, 0.0F, 1.0F, 0.0F);
    setTexel(primaries, 2, 0, 0.0F, 0.0F, 1.0F);
    setTexel(primaries, 3, 0, 1.0F, 1.0F, 1.0F);

    const HdrImage luminance = pathtracer::debug::luminanceAov(primaries, pool);
    PT_EXPECT(ctx, texelR(luminance, 0, 0) == pathtracer::debug::kRec709LuminanceWeights.r, "red weight");
    PT_EXPECT(ctx, texelR(luminance, 1, 0) == pathtracer::debug::kRec709LuminanceWeights.g, "green weight");
    PT_EXPECT(ctx, texelR(luminance, 2, 0) == pathtracer::debug::kRec709LuminanceWeights.b, "blue weight");
    // The three weights are defined to sum to 1, so white maps to 1 and the AOV is a true relative luminance.
    PT_EXPECT(ctx, std::fabs(texelR(luminance, 3, 0) - 1.0F) < 1e-6F,
                  "white luminance " + std::to_string(texelR(luminance, 3, 0)));
}

// HSV is a bijection on the RGB cube away from the achromatic axis, so inverting must return the original triple, degenerate rows included.
PT_CHECK(hsv_inverts_to_rgb, Fast, Exact) {
    pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const std::vector<glm::vec3> samples = {
        {0.8F, 0.2F, 0.4F}, {0.1F, 0.9F, 0.3F}, {0.25F, 0.25F, 0.95F}, {1.0F, 0.5F, 0.0F},
        {0.5F, 0.5F, 0.5F},  // achromatic: hue and saturation are 0 by convention
        {0.0F, 0.0F, 0.0F},  // black: saturation undefined, 0 by convention
        {4.0F, 1.5F, 0.25F}, // scene-referred highlight, well above display white
    };
    ctx.plan(static_cast<int>(samples.size()));

    HdrImage image = makeImage(static_cast<int>(samples.size()), 1, 0.0F);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        setTexel(image, static_cast<int>(i), 0, samples[i].r, samples[i].g, samples[i].b);
    }
    const HdrImage hsv = pathtracer::debug::hsvAov(image, pool);

    for (std::size_t i = 0; i < samples.size(); ++i) {
        const std::size_t texel = i * 4;
        const float h = hsv.rgba[texel];
        const float s = hsv.rgba[texel + 1];
        const float v = hsv.rgba[texel + 2];
        // Standard HSV-to-RGB inverse (Smith 1978), stated here rather than imported, so the check does not depend on the code it checks.
        const float sector = h * 6.0F;
        const auto index = static_cast<int>(std::floor(sector)) % 6;
        const float f = sector - std::floor(sector);
        const float p = v * (1.0F - s);
        const float q = v * (1.0F - (s * f));
        const float t = v * (1.0F - (s * (1.0F - f)));
        glm::vec3 back{};
        switch (index) {
            case 0:  back = {v, t, p}; break;
            case 1:  back = {q, v, p}; break;
            case 2:  back = {p, v, t}; break;
            case 3:  back = {p, q, v}; break;
            case 4:  back = {t, p, v}; break;
            default: back = {v, p, q}; break;
        }
        const float worst = std::max({std::fabs(back.r - samples[i].r), std::fabs(back.g - samples[i].g),
                                       std::fabs(back.b - samples[i].b)});
        // Forward-error bound of the divide-and-remultiply round trip, which scales with the value being represented.
        const float bound = 1e-6F * std::max(1.0F, samples[i].r + samples[i].g + samples[i].b);
        PT_EXPECT(ctx, worst <= bound, "round-trip error " + std::to_string(worst));
    }
}

// Every AOV must render at its declared channel count with no NaN or infinity: a new AOV cannot pass without being produced.
PT_CHECK(every_aov_renders_finite, Slow, Exact) {
    ctx.plan(kAovCount);
    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    if (!renderer) {
        PT_EXPECT(ctx, false, "scene load failed: " + error);
        return;
    }

    constexpr int kWidth = 24;
    constexpr int kHeight = 18;
    for (int i = 0; i < kAovCount; ++i) {
        const auto aov = static_cast<AovId>(i);
        const pathtracer::api::HeadlessRenderer::Request request{
            .camera = renderer->defaultCamera(),
            .width = kWidth,
            .height = kHeight,
            .samples = 2,
            .scrambleSeed = 1,
            .aovs = {aov},
        };

        std::vector<float> buffer(static_cast<std::size_t>(kWidth) * kHeight *
                                  static_cast<std::size_t>(pathtracer::debug::aovChannels(aov)));
        float* pointer = buffer.data();
        if (!renderer->render(request, std::span<float* const>(&pointer, 1), error)) {
            PT_EXPECT(ctx, false, std::string(pathtracer::debug::kAovNames[i]) + ": " + error);
            continue;
        }
        const bool finite = std::all_of(buffer.begin(), buffer.end(),
                                         [](float value) { return std::isfinite(value); });
        PT_EXPECT(ctx, finite, std::string(pathtracer::debug::kAovNames[i]) + " contains a non-finite value");
    }
}

// Sharing one accumulation, one rasterizer pass and one Beauty across a multi-AOV request must be bit-identical to requesting them singly.
PT_CHECK(multi_aov_request_matches_single_aov_requests, Slow, Exact) {
    const std::vector<AovId> combined = {AovId::Beauty, AovId::Depth, AovId::Normal, AovId::Sobel, AovId::AO};
    ctx.plan(static_cast<int>(combined.size()));

    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    if (!renderer) {
        for (std::size_t i = 0; i < combined.size(); ++i) {
            PT_EXPECT(ctx, false, "scene load failed: " + error);
        }
        return;
    }

    constexpr int kWidth = 32;
    constexpr int kHeight = 24;
    const auto makeRequest = [&](const std::vector<AovId>& aovs) {
        return pathtracer::api::HeadlessRenderer::Request{
            .camera = renderer->defaultCamera(),
            .width = kWidth,
            .height = kHeight,
            .samples = 4,
            .scrambleSeed = 7,
            .aovs = aovs,
        };
    };

    std::vector<std::vector<float>> together(combined.size());
    std::vector<float*> pointers(combined.size());
    for (std::size_t i = 0; i < combined.size(); ++i) {
        together[i].resize(static_cast<std::size_t>(kWidth) * kHeight *
                           static_cast<std::size_t>(pathtracer::debug::aovChannels(combined[i])));
        pointers[i] = together[i].data();
    }
    if (!renderer->render(makeRequest(combined), std::span<float* const>(pointers.data(), pointers.size()), error)) {
        for (std::size_t i = 0; i < combined.size(); ++i) {
            PT_EXPECT(ctx, false, "combined render failed: " + error);
        }
        return;
    }

    for (std::size_t i = 0; i < combined.size(); ++i) {
        std::vector<float> alone(together[i].size());
        float* pointer = alone.data();
        if (!renderer->render(makeRequest({combined[i]}), std::span<float* const>(&pointer, 1), error)) {
            PT_EXPECT(ctx, false, "single render failed: " + error);
            continue;
        }
        PT_EXPECT(ctx, alone == together[i],
                      std::string(pathtracer::debug::kAovNames[static_cast<int>(combined[i])]) +
                          " differs between a combined and a single-AOV request");
    }
}

// showSky gates the primary ray's own miss, so it must blacken the background and leave every texel the camera hits untouched.
PT_CHECK(show_sky_changes_only_the_background, Slow, Exact) {
    ctx.plan(3);
    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    if (!renderer) {
        PT_EXPECT(ctx, false, "scene load failed: " + error);
        return;
    }

    constexpr int kWidth = 64;
    constexpr int kHeight = 36;
    const auto render = [&](std::optional<bool> showSky, std::vector<float>& out) {
        const pathtracer::api::HeadlessRenderer::Request request{
            .camera = renderer->defaultCamera(),
            .width = kWidth,
            .height = kHeight,
            .samples = 2,
            .scrambleSeed = 1,
            .aovs = {AovId::Beauty},
            .showSky = showSky,
        };
        out.assign(static_cast<std::size_t>(kWidth) * kHeight * 3, 0.0F);
        float* pointer = out.data();
        return renderer->render(request, std::span<float* const>(&pointer, 1), error);
    };

    std::vector<float> sky;
    std::vector<float> without;
    std::vector<float> defaulted;
    if (!render(true, sky) || !render(false, without) || !render(std::nullopt, defaulted)) {
        PT_EXPECT(ctx, false, "render failed: " + error);
        return;
    }

    // nullopt must keep the sky on, or every existing headless caller silently changes output.
    PT_EXPECT(ctx, defaulted == sky, "the default must match showSky=true");

    // The top-left corner sits outside the box, where a primary ray misses; the centre is box interior the camera hits.
    const auto texel = [](int x, int y) { return ((static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)) * 3; };
    bool backgroundLit = false;
    bool backgroundBlack = true;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            for (int c = 0; c < 3; ++c) {
                backgroundLit = backgroundLit || sky[texel(x, y) + static_cast<std::size_t>(c)] > 0.0F;
                backgroundBlack = backgroundBlack && without[texel(x, y) + static_cast<std::size_t>(c)] == 0.0F;
            }
        }
    }
    PT_EXPECT(ctx, backgroundLit && backgroundBlack, "showSky must light the background and showSky=false must blacken it");

    bool interiorIdentical = true;
    for (int y = kHeight / 3; y < (2 * kHeight) / 3; ++y) {
        for (int x = kWidth / 3; x < (2 * kWidth) / 3; ++x) {
            for (int c = 0; c < 3; ++c) {
                const std::size_t i = texel(x, y) + static_cast<std::size_t>(c);
                interiorIdentical = interiorIdentical && sky[i] == without[i];
            }
        }
    }
    PT_EXPECT(ctx, interiorIdentical, "showSky must not change a texel the camera hits");
}

// A new filter AOV reaching the dispatch's default arm is exactly the class of bug this catches: identical output for two distinct AOVs.
PT_CHECK(aov_filter_dispatch_is_total, Fast, Exact) {
    pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    // Large enough for two pyramid octaves, so the scale-space AOVs are not legitimately empty, and chromatic so none of the six agree.
    constexpr int kWidth = 96;
    constexpr int kHeight = 64;
    HdrImage source = makeImage(kWidth, kHeight, 0.0F);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const auto fx = static_cast<float>(x);
            const auto fy = static_cast<float>(y);
            setTexel(source, x, y, 0.2F + (0.03F * fx), 0.5F + (0.01F * fy), 0.1F + (0.02F * ((x * y) % 5)));
        }
    }

    std::vector<AovId> filters;
    for (int i = 0; i < kAovCount; ++i) {
        if (pathtracer::debug::aovSource(static_cast<AovId>(i)) == AovSource::BeautyFilter) {
            filters.push_back(static_cast<AovId>(i));
        }
    }
    const int pairs = static_cast<int>(filters.size() * (filters.size() - 1) / 2);
    ctx.plan(static_cast<int>(filters.size()) + pairs);

    std::vector<HdrImage> outputs;
    outputs.reserve(filters.size());
    for (const AovId aov : filters) {
        outputs.push_back(pathtracer::debug::evaluateFilterAov(
            aov, pathtracer::debug::FilterInput{source, static_cast<float>(kHeight) / 0.5F}, pool));
        const HdrImage& out = outputs.back();
        PT_EXPECT(ctx, out.width == kWidth && out.height == kHeight &&
                          out.rgba.size() == source.rgba.size(),
                      std::string(pathtracer::debug::kAovNames[static_cast<int>(aov)]) + " returned a wrong-sized image");
    }
    for (std::size_t a = 0; a < outputs.size(); ++a) {
        for (std::size_t b = a + 1; b < outputs.size(); ++b) {
            PT_EXPECT(ctx, outputs[a].rgba != outputs[b].rgba,
                          std::string(pathtracer::debug::kAovNames[static_cast<int>(filters[a])]) + " and " +
                              pathtracer::debug::kAovNames[static_cast<int>(filters[b])] + " returned identical images");
        }
    }
}

// The one projection/producer pairing that has no answer: scan conversion inverts a perspective divide the fisheye does not have.
PT_CHECK(gbuffer_aov_with_a_fisheye_lens_is_rejected, Fast, Exact) {
    ctx.plan(4);
    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    if (!renderer) {
        for (int i = 0; i < 4; ++i) {
            PT_EXPECT(ctx, false, "scene load failed: " + error);
        }
        return;
    }

    constexpr int kWidth = 16;
    constexpr int kHeight = 12;
    const pathtracer::scene::Camera& spherical = renderer->defaultCamera();
    // Equidistant, so the lens is admissible on every axis but the one under test: the rejection can only be the projection.
    const pathtracer::scene::Lens fisheyeLens{pathtracer::scene::LensProjection::FisheyePolynomial, {}, 180.0F};
    const pathtracer::scene::Camera fisheye{spherical.position(),
                                            spherical.yawDegrees(),
                                            spherical.pitchDegrees(),
                                            spherical.filmBack(),
                                            spherical.focalLengthMm(),
                                            spherical.nearClip(),
                                            spherical.farClip(),
                                            spherical.aperture(),
                                            spherical.shutterSeconds(),
                                            spherical.iso(),
                                            fisheyeLens};

    const auto request = [&](const pathtracer::scene::Camera& camera, AovId aov) {
        return pathtracer::api::HeadlessRenderer::Request{
            .camera = camera, .width = kWidth, .height = kHeight, .samples = 1, .scrambleSeed = 3, .aovs = {aov}};
    };
    const auto attempt = [&](const pathtracer::scene::Camera& camera, AovId aov) {
        std::vector<float> buffer(static_cast<std::size_t>(kWidth) * kHeight *
                                  static_cast<std::size_t>(pathtracer::debug::aovChannels(aov)));
        float* pointer = buffer.data();
        error.clear();
        return renderer->render(request(camera, aov), std::span<float* const>(&pointer, 1), error);
    };

    PT_EXPECT(ctx, !attempt(fisheye, AovId::Depth), "a G-buffer AOV was rendered through a fisheye lens");
    PT_EXPECT(ctx, error.find("fisheye") != std::string::npos,
              "the rejection did not name the fisheye lens: " + error);
    PT_EXPECT(ctx, attempt(fisheye, AovId::Beauty), "a path-traced AOV failed under a fisheye lens: " + error);
    PT_EXPECT(ctx, attempt(spherical, AovId::Depth),
              "a G-buffer AOV failed under the default spherical lens: " + error);
}

PT_CHECK_MAIN("api")
