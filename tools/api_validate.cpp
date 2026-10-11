// Correctness gate for the headless path: the AOV tables, the four CPU Beauty filters, and headless_renderer's dispatch.

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "pathtracer/api/headless_renderer.h"
#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/colormap.h"
#include "pathtracer/debug/scale_space.h"
#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/rotation.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using pathtracer::debug::AovId;
using pathtracer::debug::AovSource;
using pathtracer::gfx::HdrImage;

constexpr int kAovCount = static_cast<int>(AovId::Count);

// An RGB image filled with `value`, the layout Beauty has and every filter reads.
[[nodiscard]] HdrImage makeRgbImage(int width, int height, float value) {
    return HdrImage{width, height, pathtracer::gfx::kRgbChannels,
                    std::vector<float>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                                           pathtracer::gfx::kRgbChannels,
                                       value)};
}

void setTexel(HdrImage& image, int x, int y, float r, float g, float b) {
    writeTexel(image, x, y, glm::vec3(r, g, b));
}

[[nodiscard]] float texelR(const HdrImage& image, int x, int y) {
    return image.texels[((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) +
                         static_cast<std::size_t>(x)) * static_cast<std::size_t>(image.channels)];
}

// One row of `channels`-channel texels, the shape every display check below feeds aovDisplay or bipolarDisplay.
[[nodiscard]] HdrImage frame(int channels, std::vector<float> texels) {
    const auto width = static_cast<int>(texels.size() / static_cast<std::size_t>(channels));
    return HdrImage{width, 1, channels, std::move(texels)};
}

}  // namespace

// Every AovId must be classified, sized and owned by the producer its classification names; an unclassified one routes to the G-buffer.
PT_CHECK(aov_tables_are_total_and_consistent, Fast, Exact) {
    ctx.plan(kAovCount * 3);
    for (int i = 0; i < kAovCount; ++i) {
        const auto aov = static_cast<AovId>(i);
        const int channels = pathtracer::debug::aovChannels(aov);
        PT_EXPECT(ctx, channels >= 1 && channels <= 3,
                      std::string(pathtracer::debug::kAovNames[i]) + " channels=" + std::to_string(channels));

        const pathtracer::debug::PathTracedLane traced = pathtracer::debug::pathTracedLane(aov);
        const pathtracer::debug::GBufferLane gbuffer = pathtracer::debug::gbufferLane(aov);
        const AovSource source = pathtracer::debug::aovSource(aov);
        // Exactly one lane accessor may answer, and only the one the classification points at.
        const bool ownedCorrectly = (source == AovSource::PathTraced && traced != nullptr && gbuffer == nullptr) ||
                                     (source == AovSource::GBuffer && gbuffer != nullptr && traced == nullptr) ||
                                     (source == AovSource::BeautyFilter && traced == nullptr && gbuffer == nullptr);
        PT_EXPECT(ctx, ownedCorrectly, std::string(pathtracer::debug::kAovNames[i]) + " lane/source disagree");

        // aovNeedsLightTransport is derived from aovSource; this pins the derivation itself.
        PT_EXPECT(ctx, pathtracer::debug::aovNeedsLightTransport(aov) == (source != AovSource::GBuffer),
                      std::string(pathtracer::debug::kAovNames[i]) + " transport flag disagrees with source");
    }
}

// The camelCase names are the vocabulary every consumer spells an AOV in, so each must resolve, and only exactly.
PT_CHECK(aov_names_round_trip, Fast, Exact) {
    ctx.plan(kAovCount + 6);
    for (int i = 0; i < kAovCount; ++i) {
        PT_EXPECT(ctx, pathtracer::debug::aovIdFromName(pathtracer::debug::kAovNames[i]) == static_cast<AovId>(i),
                      std::string("name does not resolve: ") + pathtracer::debug::kAovNames[i]);
    }
    PT_EXPECT(ctx, pathtracer::debug::aovIdFromName("bounceCount") == AovId::BounceCount, "camelCase form");
    PT_EXPECT(ctx, pathtracer::debug::aovIdFromName("bounce-count") == AovId::Count, "hyphen form must not resolve");
    PT_EXPECT(ctx, pathtracer::debug::aovIdFromName("BounceCount") == AovId::Count, "PascalCase form must not resolve");
    PT_EXPECT(ctx, pathtracer::debug::aovIdFromName("Bounce Count") == AovId::Count, "spaced form must not resolve");
    PT_EXPECT(ctx, pathtracer::debug::aovIdFromName("HSV") == AovId::Count, "upper-case acronym label must not resolve");
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
    const HdrImage flat = makeRgbImage(24, 16, 0.375F);

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
    // Wider than DoG's coarse support: below that DoG has no band to difference and correctly reads zero.
    constexpr int kWidth = 256;
    constexpr int kHeight = 192;
    HdrImage pattern = makeRgbImage(kWidth, kHeight, 0.0F);
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
            aov, pathtracer::debug::FilterInput{pattern}, pool);
        const int channels = pathtracer::debug::aovChannels(aov);
        float lowest = 0.0F;
        float highest = 0.0F;
        for (int i = 0; i < out.width * out.height; ++i) {
            for (int c = 0; c < channels; ++c) {
                const float value = out.texels[(static_cast<std::size_t>(i) * static_cast<std::size_t>(channels)) +
                                               static_cast<std::size_t>(c)];
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
    std::vector<float> wave(kPixels, 0.0F);
    // A zero-centred square wave: every sample has the same magnitude, so the peak is exactly the RMS and no tail exists to cap.
    for (int i = 0; i < kPixels; ++i) {
        wave[static_cast<std::size_t>(i)] = (i % 2) == 0 ? 1.0F : -1.0F;
    }
    // The gain is the offset divided by the range, so a unit range must reproduce the offset exactly rather than within a rounding.
    const float flat = pathtracer::debug::bipolarDisplay(frame(1, wave)).gain[0];
    // sqrt(2 ln n) > 1 for any n > 1, so the extreme-value cap cannot bite here and the true peak must survive it.
    PT_EXPECT(ctx, flat == pathtracer::debug::kBipolarDisplayOffset,
                  "a constant-magnitude field gained " + std::to_string(flat) + ", not the unit-range offset");

    wave[0] = 1024.0F;
    const float outlier = pathtracer::debug::bipolarDisplay(frame(1, wave)).gain[0];
    PT_EXPECT(ctx, outlier > pathtracer::debug::kBipolarDisplayOffset / 1024.0F,
                  "a lone outlier still set the range to its own peak, gaining " + std::to_string(outlier));
    // One sample of 1024 among 4096 unit samples lifts the RMS to sqrt(1 + 1024^2/4096) = 16.03, and sqrt(2 ln 4096) = 4.08 scales it.
    const double rms = std::sqrt(1.0 + ((1024.0 * 1024.0) - 1.0) / 4096.0);
    const double expected =
        static_cast<double>(pathtracer::debug::kBipolarDisplayOffset) / (rms * std::sqrt(2.0 * std::log(4096.0)));
    PT_EXPECT(ctx, std::fabs(static_cast<double>(outlier) - expected) <= expected * 1e-6,
                  "gain " + std::to_string(outlier) + " against the extreme-value prediction " + std::to_string(expected));

    PT_EXPECT(ctx, pathtracer::debug::bipolarDisplay(frame(1, std::vector<float>(kPixels, 0.0F))).gain[0] == 1.0F,
                  "a field with no range did not fall back to unit gain");
}

// The preview's affine map must carry -range to 0 and +range to 1 at every amplitude, or a signed AOV clips or collapses to mid-grey.
PT_CHECK(bipolar_display_maps_the_range_to_the_unit_interval, Fast, Exact) {
    // Powers of two, so the gain and its product with the amplitude are both exact and the comparison needs no tolerance to hide in.
    const std::vector<float> amplitudes{1.0F, 0.5F, 1.0F / 1024.0F, 4096.0F, 0.0F};
    ctx.plan(static_cast<int>(amplitudes.size()) * 2);
    constexpr int kPixels = 4096;
    for (const float amplitude : amplitudes) {
        std::vector<float> wave(kPixels, 0.0F);
        // Constant magnitude again, so the field's range is exactly the amplitude and this gates the map rather than the range scan.
        for (int i = 0; i < kPixels; ++i) {
            wave[static_cast<std::size_t>(i)] = (i % 2) == 0 ? amplitude : -amplitude;
        }
        const pathtracer::debug::BipolarDisplay display = pathtracer::debug::bipolarDisplay(frame(1, wave));
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
    std::vector<float> pairs(static_cast<std::size_t>(kPixels) * 2, 0.0F);
    for (int i = 0; i < kPixels; ++i) {
        const float sign = (i % 2) == 0 ? 1.0F : -1.0F;
        pairs[static_cast<std::size_t>(i) * 2] = sign * kNarrow;
        pairs[(static_cast<std::size_t>(i) * 2) + 1] = sign * kWide;
    }
    const pathtracer::debug::BipolarDisplay display = pathtracer::debug::bipolarDisplay(frame(2, pairs));
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
    std::vector<float> pairs(static_cast<std::size_t>(kPixels) * 2, 0.0F);
    std::vector<float> scalars(kPixels, 0.0F);
    for (int i = 0; i < kPixels; ++i) {
        const float sign = (i % 2) == 0 ? 1.0F : -1.0F;
        pairs[static_cast<std::size_t>(i) * 2] = sign;
        pairs[(static_cast<std::size_t>(i) * 2) + 1] = sign * 0.5F;
        scalars[static_cast<std::size_t>(i)] = sign;
    }
    const pathtracer::debug::BipolarDisplay two = pathtracer::debug::bipolarDisplay(frame(2, pairs));
    PT_EXPECT(ctx, two.gain[2] == 0.0F && two.offset[2] == 0.0F,
                  "a two-channel AOV's third lane gained " + std::to_string(two.gain[2]) + " at offset " +
                      std::to_string(two.offset[2]) + ", so an undefined lane would show");
    PT_EXPECT(ctx, two.offset[0] == pathtracer::debug::kBipolarDisplayOffset &&
                       two.offset[1] == pathtracer::debug::kBipolarDisplayOffset,
                  "a defined lane lost the mid-grey offset");
    // The display swizzle broadcasts a scalar AOV across all three lanes, so its one channel must light all three or its grey tints.
    const pathtracer::debug::BipolarDisplay one = pathtracer::debug::bipolarDisplay(frame(1, scalars));
    PT_EXPECT(ctx, one.gain == glm::vec3(one.gain[0]), "a scalar AOV's three lanes gained differently, so its grey would tint");
    PT_EXPECT(ctx, one.offset == glm::vec3(pathtracer::debug::kBipolarDisplayOffset),
                  "a scalar AOV lost the mid-grey offset on a lane");
}

// Reads a pre-mapped display float, or a sentinel neither pre-map can produce, so a vanished pre-map fails an assertion not the process.
[[nodiscard]] float displayLane(const pathtracer::debug::AovDisplay& display, std::size_t index) {
    return index < display.mapped.texels.size() ? display.mapped.texels[index] : -1.0F;
}

// The log window's ceiling is the ratio sqrt(n), the SNR of a texel at unit per-sample coefficient of variation, so it must read white.
PT_CHECK(snr_display_maps_the_root_of_the_pass_count_to_white, Fast, Exact) {
    // Powers of four, so sqrt(n) is an exact power of two and log(sqrt(n)) * 2/log(n) rounds to exactly 1 rather than near it.
    const std::vector<int> passCounts{4, 64, 1024, 4096};
    ctx.plan(static_cast<int>(passCounts.size()) * 2);
    for (const int samples : passCounts) {
        const float ceiling = std::sqrt(static_cast<float>(samples));
        const pathtracer::debug::AovDisplay display =
            pathtracer::debug::aovDisplay(AovId::SNR, frame(1, {ceiling, ceiling * 4.0F}), {samples, 0});
        PT_EXPECT(ctx, displayLane(display, 0) == 1.0F,
                      "the ratio sqrt(" + std::to_string(samples) + ") displayed " + std::to_string(displayLane(display, 0)) +
                          ", not white");
        // Beyond the reference there is no more window, so the clamp must hold rather than let a well-converged texel run past 1.
        PT_EXPECT(ctx, displayLane(display, 1) == 1.0F,
                      "a ratio past the reference displayed " + std::to_string(displayLane(display, 1)) + ", unclamped");
    }
}

// Decibels are a log scale, so squaring the ratio must double the displayed value; any linear or power map fails this identically.
PT_CHECK(snr_display_is_logarithmic_in_the_ratio, Fast, Exact) {
    const std::vector<float> ratios{1.5F, 2.0F, 3.0F, 4.0F};
    ctx.plan(static_cast<int>(ratios.size()));
    constexpr int kSamples = 1024;
    for (const float ratio : ratios) {
        const pathtracer::debug::AovDisplay display =
            pathtracer::debug::aovDisplay(AovId::SNR, frame(1, {ratio, ratio * ratio}), {kSamples, 0});
        // Toleranced at float's unit roundoff, not exact: the identity is the property, and its bit-exactness here is a rounding accident.
        PT_EXPECT(ctx, std::fabs(displayLane(display, 1) - (2.0F * displayLane(display, 0))) <= 0x1p-23F * displayLane(display, 1),
                      "ratio " + std::to_string(ratio) + " displayed " + std::to_string(displayLane(display, 0)) +
                          " but its square displayed " + std::to_string(displayLane(display, 1)) + ", not twice that");
    }
}

// Below the unit ratio a texel's value is under its own uncertainty and nothing is resolved, which is also where snrAov's zero lands.
PT_CHECK(snr_display_floors_the_unit_ratio_and_below, Fast, Exact) {
    const std::vector<float> ratios{1.0F, 0.5F, 0.0F};
    ctx.plan(static_cast<int>(ratios.size()) + 1);
    const pathtracer::debug::AovDisplay display = pathtracer::debug::aovDisplay(AovId::SNR, frame(1, ratios), {1024, 0});
    for (std::size_t i = 0; i < ratios.size(); ++i) {
        PT_EXPECT(ctx, displayLane(display, i) == 0.0F,
                      "the ratio " + std::to_string(ratios[i]) + " displayed " + std::to_string(displayLane(display, i)) +
                          " rather than the floor");
    }
    // One grey channel, which the display swizzle broadcasts: a second stored lane is one that could drift and tint the grey.
    PT_EXPECT(ctx, display.mapped.channels == pathtracer::gfx::kScalarChannels,
                  "the SNR pre-map stored " + std::to_string(display.mapped.channels) + " channels, not one grey");
}

// A variance needs two passes, so below that snrAov is uniformly zero and log(n) offers no positive ceiling to divide by.
PT_CHECK(snr_display_is_absent_below_two_passes, Fast, Exact) {
    ctx.plan(3);
    const HdrImage snr = frame(1, {4.0F, 16.0F});
    for (const int samples : {0, 1}) {
        PT_EXPECT(ctx, pathtracer::debug::aovDisplay(AovId::SNR, snr, {samples, 0}).mapped.texels.empty(),
                      "SNR pre-mapped at " + std::to_string(samples) + " passes, where its own value is undefined");
    }
    PT_EXPECT(ctx, !pathtracer::debug::aovDisplay(AovId::SNR, snr, {2, 0}).mapped.texels.empty(),
                  "SNR did not pre-map at two passes, the fewest a variance is defined for");
}

// Bounce Count's domain is maxBounces + 1 terminations, a real bound, so the shared path must reproduce the colormap exactly.
PT_CHECK(bounce_count_display_matches_the_turbo_colormap, Fast, Exact) {
    constexpr int kMaxBounces = 8;
    const std::vector<float> counts{0.0F, 1.0F, 2.5F, 4.0F, 8.0F, 9.0F};
    ctx.plan(static_cast<int>(counts.size()));
    const pathtracer::debug::AovDisplay display =
        pathtracer::debug::aovDisplay(AovId::BounceCount, frame(1, counts), {0, kMaxBounces});
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const glm::vec3 expected = pathtracer::debug::turbo(counts[i] / (static_cast<float>(kMaxBounces) + 1.0F));
        const glm::vec3 got{displayLane(display, i * 3), displayLane(display, (i * 3) + 1), displayLane(display, (i * 3) + 2)};
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
        const pathtracer::debug::AovDisplay display =
            pathtracer::debug::aovDisplay(AovId::Depth, frame(1, {maximum * 0.5F, maximum}), {0, 0});
        const float white = (maximum * display.affine.gain[0]) + display.affine.offset[0];
        PT_EXPECT(ctx, white == 1.0F,
                      "the farthest depth " + std::to_string(maximum) + " displayed " + std::to_string(white) + ", not white");
    }
}

// Three texels at the AOV's channel count, each spread over two decades and positive, so any ranging arm departs from identity.
[[nodiscard]] HdrImage aovFrame(AovId aov) {
    const int channels = pathtracer::debug::aovChannels(aov);
    std::vector<float> texels;
    for (const float value : {0.5F, 4.0F, 16.0F}) {
        texels.insert(texels.end(), static_cast<std::size_t>(channels), value);
    }
    return frame(channels, texels);
}

// An auto-ranged AOV has already absorbed the scene's scale, so applying the photographic exposure on top would range it twice.
PT_CHECK(an_auto_ranged_aov_never_also_takes_the_exposure, Fast, Exact) {
    ctx.plan(kAovCount);
    for (int i = 0; i < kAovCount; ++i) {
        const auto aov = static_cast<AovId>(i);
        const pathtracer::debug::AovDisplay display = pathtracer::debug::aovDisplay(aov, aovFrame(aov), {1024, 8});
        // Whatever aovDisplay decided from the values -- a pre-map or any map but unity gain at zero offset -- is a range already taken.
        const bool ranged = !display.mapped.texels.empty() || display.affine.gain != glm::vec3(1.0F) ||
                            display.affine.offset != glm::vec3(0.0F);
        PT_EXPECT(ctx, !(ranged && pathtracer::debug::aovTakesDisplayExposure(aov)),
                      std::string(pathtracer::debug::kAovNames[i]) + " auto-ranges and takes the exposure, so it would range twice");
    }
}

// A nonlinear pre-map costs a full frame copy, so it must apply only where the affine map genuinely cannot express the display.
PT_CHECK(display_premap_applies_to_exactly_two_aovs, Fast, Exact) {
    ctx.plan(kAovCount);
    for (int i = 0; i < kAovCount; ++i) {
        const auto aov = static_cast<AovId>(i);
        const bool premapped = aov == AovId::SNR || aov == AovId::BounceCount;
        PT_EXPECT(ctx, pathtracer::debug::aovDisplay(aov, aovFrame(aov), {1024, 8}).mapped.texels.empty() != premapped,
                      std::string(pathtracer::debug::kAovNames[i]) +
                          (premapped ? " lost its pre-map" : " gained a pre-map it does not need"));
    }
}

// dx and dy share the pixel as their unit, so one zero-centred pooled range keeps a motion's direction on screen.
PT_CHECK(motion_vector_display_pools_its_two_lanes, Fast, Exact) {
    ctx.plan(2);
    // Lane ranges far apart, so per-lane ranging would give the two gains different values.
    const HdrImage motion = frame(2, {0.5F, 16.0F, -4.0F, 0.125F, 2.0F, -8.0F});
    const pathtracer::debug::AovDisplay display = pathtracer::debug::aovDisplay(AovId::MotionVector, motion, {1024, 8});
    PT_EXPECT(ctx, display.affine.gain.x == display.affine.gain.y && display.affine.gain.x > 0.0F,
                  "dx and dy do not share one positive gain");
    PT_EXPECT(ctx, display.affine.offset == glm::vec3(0.5F, 0.5F, 0.0F), "zero motion does not land on mid-grey in R and G");
}

// A unit step edge has a closed-form Sobel magnitude: both adjacent columns read exactly 4 and everything further exactly 0.
PT_CHECK(sobel_step_edge_matches_closed_form, Fast, Exact) {
    ctx.plan(4);
    pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 16;
    constexpr int kHeight = 12;
    constexpr int kEdge = 8;

    HdrImage step = makeRgbImage(kWidth, kHeight, 0.0F);
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
    HdrImage primaries = makeRgbImage(4, 1, 0.0F);
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

    HdrImage image = makeRgbImage(static_cast<int>(samples.size()), 1, 0.0F);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        setTexel(image, static_cast<int>(i), 0, samples[i].r, samples[i].g, samples[i].b);
    }
    const HdrImage hsv = pathtracer::debug::hsvAov(image, pool);

    for (std::size_t i = 0; i < samples.size(); ++i) {
        const glm::vec3 hsvTexel = hsv.rgb(i);
        const float h = hsvTexel.x;
        const float s = hsvTexel.y;
        const float v = hsvTexel.z;
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

// The same camera through either lens, so a check can hold everything but the projection fixed.
pathtracer::scene::Camera withLens(const pathtracer::scene::Camera& camera, pathtracer::scene::Lens lens) {
    return pathtracer::scene::Camera{camera.position(), camera.rotationDegrees(), camera.filmBack(),
                                     camera.focalLengthMm(), camera.nearClip(), camera.farClip(), camera.aperture(),
                                     camera.shutterSeconds(), camera.iso(), lens};
}

// Every AOV must render at its declared channel count with no NaN or infinity: a new AOV cannot pass without being produced.
PT_CHECK(every_aov_renders_finite, Slow, Exact) {
    ctx.plan(2 * 3 * kAovCount);
    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    if (!renderer) {
        PT_EXPECT(ctx, false, "scene load failed: " + error);
        return;
    }

    constexpr int kWidth = 24;
    constexpr int kHeight = 18;
    // Every AOV through every projection: no producer is lens-specific, so a fisheye and the lat-long must serve each one too.
    const std::array<pathtracer::scene::Camera, 3> cameras{
        renderer->defaultCamera(),
        withLens(renderer->defaultCamera(), {pathtracer::scene::LensProjection::FisheyePolynomial, {}, 180.0F}),
        withLens(renderer->defaultCamera(), {pathtracer::scene::LensProjection::Omnidirectional})};
    for (int i = 0; i < static_cast<int>(cameras.size()) * kAovCount; ++i) {
        const auto aov = static_cast<AovId>(i % kAovCount);
        const pathtracer::api::HeadlessRenderer::Request request{
            .camera = cameras[static_cast<std::size_t>(i / kAovCount)],
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
            PT_EXPECT(ctx, false, std::string(pathtracer::debug::kAovNames[i % kAovCount]) + ": " + error);
            continue;
        }
        const bool finite = std::all_of(buffer.begin(), buffer.end(),
                                         [](float value) { return std::isfinite(value); });
        PT_EXPECT(ctx, finite, std::string(pathtracer::debug::kAovNames[i % kAovCount]) + " contains a non-finite value");
        // Its producer's lane is stored at exactly the channels it declares: the memory the declared count promises, and no padding lane.
        const HdrImage& lane = renderer->lastImage(aov);
        PT_EXPECT(ctx, lane.channels == pathtracer::debug::aovChannels(aov) && lane.texels.size() == buffer.size(),
                  std::string(pathtracer::debug::kAovNames[i % kAovCount]) + " is stored at " + std::to_string(lane.channels) +
                      " channels, not its declared " + std::to_string(pathtracer::debug::aovChannels(aov)));
    }
}

// Sharing one accumulation, one G-buffer pass and one Beauty across a multi-AOV request must be bit-identical to requesting them singly.
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

    // nullopt is the shared default, which hides the sky as the viewer does.
    PT_EXPECT(ctx, defaulted == without, "the default must match showSky=false");

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
    // Wider than DoG's coarse support, so its band is not legitimately empty, and chromatic so no two of the seven filters agree.
    constexpr int kWidth = 96;
    constexpr int kHeight = 64;
    HdrImage source = makeRgbImage(kWidth, kHeight, 0.0F);
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
            aov, pathtracer::debug::FilterInput{source}, pool));
        const HdrImage& out = outputs.back();
        const int channels = pathtracer::debug::aovChannels(aov);
        PT_EXPECT(ctx, out.width == kWidth && out.height == kHeight && out.channels == channels &&
                          out.texels.size() == static_cast<std::size_t>(kWidth) * kHeight * static_cast<std::size_t>(channels),
                      std::string(pathtracer::debug::kAovNames[static_cast<int>(aov)]) + " returned a wrong-shaped image");
    }
    for (std::size_t a = 0; a < outputs.size(); ++a) {
        for (std::size_t b = a + 1; b < outputs.size(); ++b) {
            PT_EXPECT(ctx, outputs[a].texels != outputs[b].texels,
                          std::string(pathtracer::debug::kAovNames[static_cast<int>(filters[a])]) + " and " +
                              pathtracer::debug::kAovNames[static_cast<int>(filters[b])] + " returned identical images");
        }
    }
}

// On the optical axis every lens casts the same ray, so an odd-sized frame's centre texel reads one ray distance through each.
PT_CHECK(gbuffer_depth_agrees_across_lenses_on_the_axis, Fast, Exact) {
    ctx.plan(5);
    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    if (!renderer) {
        for (int i = 0; i < 5; ++i) {
            PT_EXPECT(ctx, false, "scene load failed: " + error);
        }
        return;
    }
    constexpr int kWidth = 17;
    constexpr int kHeight = 13;
    constexpr std::size_t kCentre = ((kHeight / 2) * kWidth) + (kWidth / 2);
    const auto centreDepth = [&](const pathtracer::scene::Camera& camera) {
        const pathtracer::api::HeadlessRenderer::Request request{
            .camera = camera, .width = kWidth, .height = kHeight, .aovs = {AovId::Depth}};
        error.clear();
        return renderer->render(request, error) ? renderer->lastImage(AovId::Depth).texels[kCentre] : -1.0F;
    };
    const float rectilinear = centreDepth(renderer->defaultCamera());
    const float fisheye =
        centreDepth(withLens(renderer->defaultCamera(), {pathtracer::scene::LensProjection::FisheyePolynomial, {}, 180.0F}));
    PT_EXPECT(ctx, rectilinear > 0.0F, "the rectilinear centre ray hit nothing: " + error);
    PT_EXPECT(ctx, fisheye > 0.0F, "the fisheye centre ray hit nothing: " + error);
    // The pinhole re-normalises an already unit forward: at most one rounding of the direction, carried through one intersection.
    PT_EXPECT(ctx, std::fabs(rectilinear - fisheye) <= 4.0F * std::numeric_limits<float>::epsilon() * rectilinear,
              "the axis depth differs between lenses: " + std::to_string(rectilinear) + " vs " + std::to_string(fisheye));
    // cos(float(pi/2)) leaves the lat-long's axis ray 4.4e-8 rad off forward, a second-order 1e-15 change in the distance.
    const float omnidirectional = centreDepth(withLens(renderer->defaultCamera(), {pathtracer::scene::LensProjection::Omnidirectional}));
    PT_EXPECT(ctx, omnidirectional > 0.0F, "the omnidirectional centre ray hit nothing: " + error);
    PT_EXPECT(ctx, std::fabs(rectilinear - omnidirectional) <= 4.0F * std::numeric_limits<float>::epsilon() * rectilinear,
              "the axis depth differs between lenses: " + std::to_string(rectilinear) + " vs " + std::to_string(omnidirectional));
}

// MotionVector measures from Request::previousCamera: absent or equal, motion is exactly zero; a turn left moves the whole frame right.
PT_CHECK(motion_vector_follows_the_previous_camera, Fast, Exact) {
    ctx.plan(4);
    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    if (!renderer) {
        for (int i = 0; i < 4; ++i) {
            PT_EXPECT(ctx, false, "scene load failed: " + error);
        }
        return;
    }
    const pathtracer::scene::Camera& camera = renderer->defaultCamera();
    const auto posed = [&camera](float yawOffset, pathtracer::scene::Lens lens) {
        return pathtracer::scene::Camera{camera.position(), camera.rotationDegrees() + glm::vec3(0.0F, yawOffset, 0.0F), camera.filmBack(),
                                         camera.focalLengthMm(), camera.nearClip(), camera.farClip(), camera.aperture(),
                                         camera.shutterSeconds(), camera.iso(), lens};
    };
    const auto motion = [&](std::optional<pathtracer::scene::Camera> previous) {
        const pathtracer::api::HeadlessRenderer::Request request{
            .camera = camera, .previousCamera = previous, .width = 24, .height = 18, .aovs = {AovId::MotionVector}};
        error.clear();
        return renderer->render(request, error) ? renderer->lastImage(AovId::MotionVector).texels : std::vector<float>{};
    };
    const auto allZero = [](const std::vector<float>& texels) {
        return !texels.empty() && std::all_of(texels.begin(), texels.end(), [](float value) { return value == 0.0F; });
    };
    PT_EXPECT(ctx, allZero(motion(std::nullopt)), "no previous camera did not read exactly zero motion: " + error);
    PT_EXPECT(ctx, allZero(motion(camera)), "the camera as its own previous view did not read exactly zero motion: " + error);
    // A positive turn about +Y is to the left (right-handed, -Z forward), so the previous view at y - 0.5 sees the scene shifted right.
    const std::vector<float> turned = motion(posed(-0.5F, camera.lens()));
    bool rightward = !turned.empty();
    for (std::size_t i = 0; i < turned.size(); i += 2) {
        rightward = rightward && turned[i] > 0.0F;
    }
    PT_EXPECT(ctx, rightward, "a turn to the left did not move every pixel to the right: " + error);
    // A lens toggle between views: the previous fisheye projects through its own forward model, the current one through the pinhole's.
    const pathtracer::scene::Lens fisheyeLens{pathtracer::scene::LensProjection::FisheyePolynomial, {}, 180.0F};
    const std::vector<float> toggled = motion(posed(0.0F, fisheyeLens));
    PT_EXPECT(ctx, !toggled.empty() && std::all_of(toggled.begin(), toggled.end(), [](float value) { return std::isfinite(value); }),
              "a fisheye previous view failed or gave a non-finite motion: " + error);
}

// render() is the boundary for C++ and C ABI callers alike: an invalid camera or previous camera fails it, the error naming which.
PT_CHECK(render_rejects_an_invalid_camera, Fast, Exact) {
    ctx.plan(3);
    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    if (!renderer) {
        for (int i = 0; i < 3; ++i) {
            PT_EXPECT(ctx, false, "scene load failed: " + error);
        }
        return;
    }
    const pathtracer::scene::Camera& camera = renderer->defaultCamera();
    const pathtracer::scene::Camera zeroFieldOfView = withLens(camera, {pathtracer::scene::LensProjection::Rectilinear, {}, 0.0F});
    const auto render = [&](const pathtracer::scene::Camera& current, std::optional<pathtracer::scene::Camera> previous) {
        const pathtracer::api::HeadlessRenderer::Request request{
            .camera = current, .previousCamera = previous, .width = 8, .height = 8, .aovs = {AovId::Depth}};
        error.clear();
        return renderer->render(request, error);
    };
    PT_EXPECT(ctx, render(camera, camera), "the default camera was rejected: " + error);
    PT_EXPECT(ctx, !render(zeroFieldOfView, std::nullopt) && error.find("field of view") != std::string::npos,
              "a zero field of view was not rejected by name: " + error);
    PT_EXPECT(ctx, !render(camera, zeroFieldOfView) && error.starts_with("previousCamera: "),
              "an invalid previous camera was not rejected under its own name: " + error);
}

// Cornell's scene.json, parsed, for a check to re-author in another pose.
[[nodiscard]] nlohmann::json cornellScene() {
    std::ifstream file(std::filesystem::path(ASSET_ROOT_DIR) / "scenes" / "cornell.json");
    return nlohmann::json::parse(file);
}

// An asset root whose scenes/posed.json is `scene`, every other directory linked to the real one, so the scene file is all that differs.
[[nodiscard]] std::filesystem::path posedAssetRoot(const nlohmann::json& scene) {
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "engine_api_posed_assets";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "scenes");
    for (const char* directory : {"config", "geometry", "materials", "textures"}) {
        std::filesystem::create_directory_symlink(std::filesystem::path(ASSET_ROOT_DIR) / directory, root / directory);
    }
    std::ofstream(root / "scenes" / "posed.json") << scene.dump();
    return root;
}

// A request's pose renders exactly what scene.json authored in that pose renders, and returning to the authored pose restores it.
PT_CHECK(pose_override_matches_the_scene_authored_in_that_pose, Slow, Exact) {
    ctx.plan(5);
    const glm::vec3 rootRotation(10.0F, 35.0F, -5.0F);
    const glm::vec3 lightRotation(-70.0F, 20.0F, 0.0F);
    nlohmann::json authored = cornellScene();
    authored["model"]["rotation"] = {rootRotation.x, rootRotation.y, rootRotation.z};
    authored["lights"][0]["rotation"] = {lightRotation.x, lightRotation.y, lightRotation.z};
    const std::filesystem::path posedRoot = posedAssetRoot(authored);
    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    const auto posed = pathtracer::api::HeadlessRenderer::open(posedRoot.string(), "scenes/posed.json", error);
    if (!renderer || !posed) {
        for (int i = 0; i < 5; ++i) {
            PT_EXPECT(ctx, false, "scene load failed: " + error);
        }
        std::filesystem::remove_all(posedRoot);
        return;
    }

    const std::vector<AovId> aovs = {AovId::Beauty, AovId::Depth, AovId::Normal};
    using Rotations = std::optional<std::vector<glm::vec3>>;
    const auto render = [&](pathtracer::api::HeadlessRenderer& target, std::optional<glm::vec3> root, Rotations lights) {
        const pathtracer::api::HeadlessRenderer::Request request{.camera = renderer->defaultCamera(),
                                                                 .width = 32,
                                                                 .height = 24,
                                                                 .samples = 2,
                                                                 .scrambleSeed = 5,
                                                                 .aovs = aovs,
                                                                 .rootRotationDegrees = root,
                                                                 .lightRotationsDegrees = std::move(lights)};
        std::vector<std::vector<float>> lanes;
        if (target.render(request, error)) {
            for (const AovId aov : aovs) {
                lanes.push_back(target.lastImage(aov).texels);
            }
        }
        return lanes;
    };
    std::vector<glm::vec3> authoredLights;
    for (const pathtracer::config::QuadLightConfig& light : renderer->scene().lights) {
        authoredLights.push_back(light.rotation);
    }
    const auto authoredPose = render(*renderer, std::nullopt, std::nullopt);
    const auto explicitAuthored = render(*renderer, renderer->scene().model.rotation, authoredLights);
    const auto overridden = render(*renderer, rootRotation, std::vector{lightRotation});
    const auto reference = render(*posed, std::nullopt, std::nullopt);
    const auto restored = render(*renderer, std::nullopt, std::nullopt);
    std::filesystem::remove_all(posedRoot);

    PT_EXPECT(ctx, !authoredPose.empty() && !overridden.empty() && !reference.empty(), "a render failed: " + error);
    PT_EXPECT(ctx, explicitAuthored == authoredPose, "the authored pose given explicitly differs from no pose at all");
    PT_EXPECT(ctx, overridden != authoredPose, "the pose changed nothing, so the comparison below is vacuous");
    PT_EXPECT(ctx, overridden == reference, "the overridden pose differs from scene.json authored in that pose");
    // A rebuild, then a rebuild back: the BVH and the bound materials must come back exactly, not merely close.
    PT_EXPECT(ctx, restored == authoredPose, "returning to the authored pose did not reproduce it bit for bit");
}

// render() refuses a pose it cannot build, naming the field, and the refusal leaves the renderer serving the authored pose.
PT_CHECK(render_rejects_an_invalid_pose, Fast, Exact) {
    ctx.plan(5);
    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    if (!renderer) {
        for (int i = 0; i < 5; ++i) {
            PT_EXPECT(ctx, false, "scene load failed: " + error);
        }
        return;
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const auto render = [&](std::optional<glm::vec3> root, std::optional<std::vector<glm::vec3>> lights, glm::vec3 env) {
        const pathtracer::api::HeadlessRenderer::Request request{.camera = renderer->defaultCamera(),
                                                                 .width = 8,
                                                                 .height = 8,
                                                                 .aovs = {AovId::Depth},
                                                                 .rootRotationDegrees = root,
                                                                 .lightRotationsDegrees = std::move(lights),
                                                                 .envRotationDegrees = env};
        error.clear();
        return renderer->render(request, error);
    };
    PT_EXPECT(ctx, !render(glm::vec3(0.0F, nan, 0.0F), std::nullopt, glm::vec3(0.0F)) && error == "root rotation is not finite",
              "a NaN root rotation was not rejected by name: " + error);
    PT_EXPECT(ctx, !render(std::nullopt, std::vector<glm::vec3>(2, glm::vec3(0.0F)), glm::vec3(0.0F)) &&
                       error.starts_with("light rotations number 2"),
              "two rotations for cornell's one light were not rejected by count: " + error);
    PT_EXPECT(ctx, !render(std::nullopt, std::vector{glm::vec3(inf, 0.0F, 0.0F)}, glm::vec3(0.0F)) &&
                       error == "a light rotation is not finite",
              "an infinite light rotation was not rejected by name: " + error);
    PT_EXPECT(ctx, !render(std::nullopt, std::nullopt, glm::vec3(0.0F, 0.0F, nan)) && error == "environment rotation is not finite",
              "a NaN environment rotation was not rejected by name: " + error);
    PT_EXPECT(ctx, render(std::nullopt, std::nullopt, glm::vec3(0.0F)), "the authored pose failed after the refusals: " + error);
}

// Turning model, lights, environment and camera by one rotation leaves the image: the four share one convention and one sense.
PT_CHECK(turning_the_world_with_the_camera_leaves_the_image, Slow, Exact) {
    ctx.plan(4);
    std::string error;
    const auto renderer = pathtracer::api::HeadlessRenderer::open(ASSET_ROOT_DIR, "scenes/cornell.json", error);
    if (!renderer) {
        for (int i = 0; i < 4; ++i) {
            PT_EXPECT(ctx, false, "scene load failed: " + error);
        }
        return;
    }
    // Cornell authors its root unrotated, so the turned root is the turn itself; the light, authored under the root, follows it.
    const glm::vec3 turn(20.0F, 35.0F, -15.0F);
    const glm::mat3 turnMatrix = pathtracer::scene::rotationXyz(turn);
    const pathtracer::scene::Camera& camera = renderer->defaultCamera();
    // The lat-long sees every direction, so the comparison covers the whole environment and every surface around the eye.
    const pathtracer::scene::Camera unturned =
        withLens(camera, {pathtracer::scene::LensProjection::Omnidirectional});
    // The camera turns on the left of its own rotation, as the world does, and goes back to Euler through the controller's inverse.
    const glm::vec3 turnedRotation = pathtracer::scene::eulerXyzDegrees(turnMatrix * pathtracer::scene::rotationXyz(camera.rotationDegrees()));
    const pathtracer::scene::Camera turned{turnMatrix * camera.position(), turnedRotation, camera.filmBack(),
                                           camera.focalLengthMm(), camera.nearClip(), camera.farClip(), camera.aperture(),
                                           camera.shutterSeconds(), camera.iso(), unturned.lens()};
    // 4:3, not the map's 2:1: OIIO probes int(2a - 1) times along a footprint of st aspect a, so a = 2 would step on a few-ulp turn.
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    const auto render = [&](const pathtracer::scene::Camera& view, std::optional<glm::vec3> root, glm::vec3 env) {
        const pathtracer::api::HeadlessRenderer::Request request{.camera = view,
                                                                 .width = kWidth,
                                                                 .height = kHeight,
                                                                 .samples = 1,
                                                                 .scrambleSeed = 3,
                                                                 .aovs = {AovId::Beauty, AovId::Depth},
                                                                 .showSky = true,
                                                                 .rootRotationDegrees = root,
                                                                 .envRotationDegrees = env};
        std::pair<std::vector<float>, std::vector<float>> lanes;
        if (renderer->render(request, error)) {
            lanes = {renderer->lastImage(AovId::Beauty).texels, renderer->lastImage(AovId::Depth).texels};
        }
        return lanes;
    };
    const auto [beauty, depth] = render(unturned, std::nullopt, glm::vec3(0.0F));
    const auto [turnedBeauty, turnedDepth] = render(turned, turn, turn);
    const auto [worldOnlyBeauty, worldOnlyDepth] = render(unturned, turn, turn);
    PT_EXPECT(ctx, !depth.empty() && !turnedDepth.empty() && !worldOnlyDepth.empty(), "a render failed: " + error);
    PT_EXPECT(ctx, worldOnlyBeauty != beauty, "turning the world alone changed nothing, so the comparison below is vacuous");

    // A convention slip moves the image by whole pixels. Rounding alone leaves depth a few ulp apart, and turns each ray by a few ulp.
    constexpr float kDepthTolerance = 16.0F * std::numeric_limits<float>::epsilon();
    // A few-ulp turn is ~1e-4 of a 2k lat-long texel; bilinear radiance moves that fraction of a texel step, under 1e-3 at the sun's edge.
    constexpr float kSkyTolerance = 1e-3F;
    float worstDepth = 0.0F;
    float worstSky = 0.0F;
    bool coverageAgrees = true;
    for (std::size_t pixel = 0; pixel < depth.size(); ++pixel) {
        coverageAgrees = coverageAgrees && ((depth[pixel] == 0.0F) == (turnedDepth[pixel] == 0.0F));
        if (depth[pixel] > 0.0F) {
            worstDepth = std::max(worstDepth, std::fabs(turnedDepth[pixel] - depth[pixel]) / depth[pixel]);
            continue;
        }
        // A primary miss reads the environment alone, so the background pixels compare the environment's turn with the camera's.
        for (std::size_t c = 0; c < 3; ++c) {
            const float sky = beauty[(3 * pixel) + c];
            worstSky = std::max(worstSky, std::fabs(turnedBeauty[(3 * pixel) + c] - sky) / std::max(sky, 1e-6F));
        }
    }
    PT_EXPECT(ctx, coverageAgrees && worstDepth <= kDepthTolerance,
              "turned depth differs: coverage " + std::string(coverageAgrees ? "agrees" : "differs") + ", worst relative " +
                  std::to_string(worstDepth));
    PT_EXPECT(ctx, worstSky <= kSkyTolerance, "turned sky radiance differs: worst relative " + std::to_string(worstSky));
}

PT_CHECK_MAIN("api")
