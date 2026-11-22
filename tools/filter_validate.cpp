// Correctness gate for the Beauty filters and adaptive sampling: SNR, the replicate stopping rule, foveation, OIDN.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "check.h"
#include "stats.h"
#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/denoiser.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/path_trace_driver.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/rgb.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using pathtracer::gfx::HdrImage;
using pathtracer::scene::ThreadPool;

constexpr float kFloatEpsilon = 0x1p-23F;  // distance from 1 to the next float, the unit in the last place, twice the unit roundoff

[[nodiscard]] float texelAt(const HdrImage& image, int x, int y, int channel) {
    return image.texels[(((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) +
                          static_cast<std::size_t>(x)) * static_cast<std::size_t>(image.channels)) +
                        static_cast<std::size_t>(channel)];
}

// A grey texel's replicateStride means, one channel's replicates in all three: where the colour rule reduces to the scalar one.
[[nodiscard]] std::vector<float> greyTexel(std::span<const float> means) {
    std::vector<float> texel;
    texel.reserve(means.size() * pathtracer::scene::kReplicateChannels);
    for (int c = 0; c < pathtracer::scene::kReplicateChannels; ++c) {
        texel.insert(texel.end(), means.begin(), means.end());
    }
    return texel;
}

// An exposure and range that leave the luminance window inert on unit means: clip 2^20 above, floor 2^40 below, damping ~2^-40.
constexpr float kInertGain = 0x1p-20F;
constexpr float kInertDynamicRange = 60.0F;

// The stopping statistic with the luminance window inert, where it reduces to the plain relative rule.
[[nodiscard]] double inertStatistic(const pathtracer::scene::ReplicateEstimate& estimate, int replicates, float threshold) {
    return pathtracer::scene::convergenceStatistic(
        pathtracer::scene::dampedRelativeVariance(estimate, kInertGain, std::exp2(-static_cast<double>(kInertDynamicRange))), replicates,
        threshold);
}

// A width x height frame holding `means`, replicateStride floats a texel, every texel `samples` passes deep.
[[nodiscard]] pathtracer::scene::PathTraceResult frameOf(int width, int height, int replicates, std::span<const float> means, int samples,
                                                         pathtracer::gfx::PixelLattice lattice, bool wrapsHorizontally) {
    pathtracer::scene::PathTraceResult result = pathtracer::scene::makePathTraceResult(width, height, replicates);
    std::copy(means.begin(), means.end(), result.replicateColour.begin());
    std::fill(result.texelSamples.begin(), result.texelSamples.end(), samples);
    result.pixelLattice = lattice;
    result.wrapsHorizontally = wrapsHorizontally;
    return result;
}

}  // namespace

// Replicate means mu +/- delta over an even K sum K delta^2 of squared deviations, so the ratio mu sqrt(K-1) / delta is exact.
PT_CHECK(snr_matches_the_closed_form_of_alternating_replicates, Fast, Exact) {
    const std::array<int, 5> counts{2, 4, 8, 16, 64};
    ctx.plan(static_cast<int>(counts.size()));
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;
    // Powers of two, so every replicate mean and their exact average are representable and the reference has a closed form.
    constexpr float kMean = 0.25F;
    constexpr float kDelta = 0.03125F;
    for (const int replicates : counts) {
        // Even K alternates identically in each channel block, so every texel is grey.
        std::vector<float> means(kPixels * pathtracer::scene::replicateStride(replicates));
        for (std::size_t i = 0; i < means.size(); ++i) {
            means[i] = (i % 2 == 0) ? kMean + kDelta : kMean - kDelta;
        }
        const HdrImage snr = pathtracer::debug::snrAov(kWidth, kHeight, means.data(), replicates, replicates, pool);
        const double expected = (static_cast<double>(kMean) * std::sqrt(static_cast<double>(replicates) - 1.0)) / kDelta;
        double worst = 0.0;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                worst = std::max(worst, std::fabs(static_cast<double>(texelAt(snr, x, y, 0)) - expected) / expected);
            }
        }
        // Computed in double, so the one float rounding of the stored ratio dominates.
        PT_EXPECT(ctx, worst <= static_cast<double>(kFloatEpsilon),
                  "K=" + std::to_string(replicates) + " reports a relative " + std::to_string(worst) + " off " +
                      std::to_string(expected));
    }
}

// Mid-round only min(n, K) replicates are drawn: the rest hold no estimate and must not be read, so NaN there must not reach the ratio.
PT_CHECK(snr_reads_only_the_populated_replicates, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 16;
    constexpr int kHeight = 8;
    constexpr int kReplicates = 8;
    constexpr int kSamples = 5;
    constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;
    constexpr float kMean = 0.5F;
    constexpr float kDelta = 0.0625F;
    // Five drawn: mu+d, mu-d, mu+d, mu-d, mu average to mu with squared deviations 4 d^2, so SE^2 = 4 d^2 / (5 * 4).
    const std::array<float, kReplicates> pixel{kMean + kDelta, kMean - kDelta, kMean + kDelta, kMean - kDelta, kMean,
                                               std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
                                               std::numeric_limits<float>::quiet_NaN()};
    const std::vector<float> texel = greyTexel(pixel);
    std::vector<float> means;
    means.reserve(kPixels * texel.size());
    for (std::size_t i = 0; i < kPixels; ++i) {
        means.insert(means.end(), texel.begin(), texel.end());
    }
    const HdrImage snr = pathtracer::debug::snrAov(kWidth, kHeight, means.data(), kReplicates, kSamples, pool);
    const double expected = (static_cast<double>(kMean) * std::sqrt(static_cast<double>(kSamples))) / kDelta;
    const bool exact = std::all_of(snr.texels.begin(), snr.texels.end(), [&](float value) {
        return std::fabs(static_cast<double>(value) - expected) <= static_cast<double>(kFloatEpsilon) * expected;
    });
    PT_EXPECT(ctx, exact, "an incomplete round read an undrawn replicate or mis-weighted the drawn ones");
}

// Undefined is zero (no estimate, one replicate, no signal); a positive signal every replicate agreed on exactly is known, so +inf.
PT_CHECK(snr_is_zero_where_undefined_and_infinite_where_exact, Fast, Exact) {
    ctx.plan(6);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 32;
    constexpr int kHeight = 24;
    constexpr int kReplicates = 4;
    constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;
    constexpr std::size_t kStride = pathtracer::scene::replicateStride(kReplicates);
    std::vector<float> spread(kPixels * kStride);
    for (std::size_t i = 0; i < spread.size(); ++i) {
        spread[i] = (i % 2 == 0) ? 0.75F : 0.5F;
    }
    const std::vector<float> dark(kPixels * kStride, 0.0F);
    const std::vector<float> agreed(kPixels * kStride, 0.75F);
    const auto all = [](const HdrImage& image, float expected) {
        return std::all_of(image.texels.begin(), image.texels.end(), [expected](float value) { return value == expected; });
    };
    constexpr float kInfinity = std::numeric_limits<float>::infinity();
    PT_EXPECT(ctx, all(pathtracer::debug::snrAov(kWidth, kHeight, nullptr, kReplicates, 64, pool), 0.0F),
              "absent replicate means did not read zero");
    PT_EXPECT(ctx, all(pathtracer::debug::snrAov(kWidth, kHeight, spread.data(), kReplicates, 0, pool), 0.0F),
              "zero passes did not read zero");
    PT_EXPECT(ctx, all(pathtracer::debug::snrAov(kWidth, kHeight, spread.data(), kReplicates, 1, pool), 0.0F),
              "a single populated replicate did not read zero");
    PT_EXPECT(ctx, all(pathtracer::debug::snrAov(kWidth, kHeight, spread.data(), 1, 64, pool), 0.0F),
              "one replicate, a single sequence with no independent spread, did not read zero");
    PT_EXPECT(ctx, all(pathtracer::debug::snrAov(kWidth, kHeight, dark.data(), kReplicates, 64, pool), 0.0F),
              "no signal did not read zero");
    PT_EXPECT(ctx, all(pathtracer::debug::snrAov(kWidth, kHeight, agreed.data(), kReplicates, 64, pool), kInfinity),
              "a positive signal every replicate agreed on did not read +inf");
}

// --- Adaptive sampling's stopping rule, on the replicate estimate the SNR above is built from.

// Alternating mu +/- delta over even K: mean mu and sum of squares K delta^2 are exact, so the variance is one rounding off delta^2/(K-1).
PT_CHECK(replicate_estimate_matches_the_closed_form, Fast, Exact) {
    constexpr std::array<int, 4> kCounts{2, 4, 8, 64};
    constexpr float kMean = 1.0F;
    constexpr float kDelta = 0.015625F;
    ctx.plan(static_cast<int>(kCounts.size()));
    for (const int replicates : kCounts) {
        std::vector<float> means(static_cast<std::size_t>(replicates));
        for (int r = 0; r < replicates; ++r) {
            means[static_cast<std::size_t>(r)] = r % 2 == 0 ? kMean + kDelta : kMean - kDelta;
        }
        const pathtracer::scene::ReplicateEstimate estimate =
            pathtracer::scene::replicateEstimate(greyTexel(means).data(), replicates, replicates);
        const double expected = static_cast<double>(kDelta) * kDelta / (replicates - 1.0);
        // Two roundings, 1/(K(K-1)) and its product, each within half an ulp of double.
        const double bound = 2.0 * std::numeric_limits<double>::epsilon() * expected;
        char detail[160];
        std::snprintf(detail, sizeof(detail), "K=%d: mean %.17g, variance %.17g against %.17g", replicates, estimate.level(),
                      estimate.variance[0], expected);
        PT_EXPECT(ctx, estimate.level() == kMean && std::fabs(estimate.variance[0] - expected) <= bound, detail);
    }
}

// Benjamini & Hochberg 1995's own worked example: 15 p-values at a false discovery rate of 0.05 reject exactly the four smallest.
PT_CHECK(step_up_matches_benjamini_hochbergs_example, Fast, Exact) {
    constexpr std::array<double, 15> kP{0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
                                        0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000};
    ctx.plan(3);
    PT_EXPECT(ctx, pathtracer::scene::stepUpRejections(kP, kP.size(), 0.05) == 4, "the paper's example did not reject its four smallest");
    // Only the p-values at or below alpha can be rejected, so passing those nine against all fifteen tests must agree.
    PT_EXPECT(ctx, pathtracer::scene::stepUpRejections(std::span(kP).first(9), kP.size(), 0.05) == 4,
              "the screened p-values ranked against the whole family rejected a different set");
    PT_EXPECT(ctx, pathtracer::scene::stepUpRejections({}, 15, 0.05) == 0, "an empty screen rejected something");
}

// Replicates that agree exactly carry no variance to test, whatever their mean, and NaN none; a zero mean is judged at the floor.
PT_CHECK(convergence_statistic_needs_spread, Fast, Exact) {
    ctx.plan(5);
    constexpr int kReplicates = 4;
    const auto statistic = [&](std::array<float, kReplicates> means, float threshold) {
        const std::vector<float> texel = greyTexel(means);
        return inertStatistic(pathtracer::scene::replicateEstimate(texel.data(), kReplicates, kReplicates), kReplicates, threshold);
    };
    constexpr float kHuge = std::numeric_limits<float>::max();
    constexpr double kNoEvidence = std::numeric_limits<double>::infinity();
    PT_EXPECT(ctx, statistic({0.0F, 0.0F, 0.0F, 0.0F}, kHuge) == kNoEvidence, "replicates agreeing on zero gave evidence of convergence");
    PT_EXPECT(ctx, statistic({0.5F, 0.5F, 0.5F, 0.5F}, kHuge) == kNoEvidence, "replicates agreeing on a positive value gave evidence");
    PT_EXPECT(ctx, statistic({0.5F, 0.5F, 0.5F, std::nextafter(0.5F, 1.0F)}, kHuge) < 1e-30,
              "the least spread under any threshold was not converged");
    PT_EXPECT(ctx, std::isfinite(statistic({0.25F, -0.25F, 0.25F, -0.25F}, kHuge)), "a zero mean with spread was not judged at the floor");
    PT_EXPECT(ctx, statistic({0.5F, std::numeric_limits<float>::quiet_NaN(), 0.5F, 0.5F}, kHuge) == kNoEvidence,
              "a NaN replicate gave evidence");
}

// Channel blocks R, G, B: the level is their mean, the variance the worst channel's, and one noisy channel is evidence for the texel.
PT_CHECK(replicate_estimate_takes_the_worst_channel, Fast, Exact) {
    ctx.plan(3);
    constexpr int kReplicates = 4;
    constexpr float kSmall = 0.015625F;
    constexpr float kLarge = 0.0625F;
    // R spreads by kSmall about 1, G by kLarge about 2, B agrees on 3: exact powers of two throughout.
    const std::array<float, pathtracer::scene::replicateStride(kReplicates)> texel{
        1.0F + kSmall, 1.0F - kSmall, 1.0F + kSmall, 1.0F - kSmall, 2.0F + kLarge, 2.0F - kLarge, 2.0F + kLarge, 2.0F - kLarge,
        3.0F,          3.0F,          3.0F,          3.0F};
    const pathtracer::scene::ReplicateEstimate estimate = pathtracer::scene::replicateEstimate(texel.data(), kReplicates, kReplicates);
    const double expected = static_cast<double>(kLarge) * kLarge / (kReplicates - 1.0);
    const double bound = 2.0 * std::numeric_limits<double>::epsilon() * expected;
    const double worst = *std::max_element(estimate.variance.begin(), estimate.variance.end());
    PT_EXPECT(ctx, estimate.level() == 2.0, "the level was not the mean of the channel means: " + std::to_string(estimate.level()));
    PT_EXPECT(ctx, std::fabs(worst - expected) <= bound, "the worst channel's variance was wrong: " + std::to_string(worst));
    PT_EXPECT(ctx, std::isfinite(inertStatistic(estimate, kReplicates, 1.0F)),
              "two agreeing channels beside a noisy one withheld the texel's evidence");
}

// Chroma noise with luminance held inside the tolerance: the luminance rule would stop the texel, the colour rule must not.
PT_CHECK(chroma_noise_keeps_a_texel_the_luminance_rule_stops, Fast, Exact) {
    ctx.plan(2);
    constexpr int kReplicates = 4;
    constexpr float kThreshold = 0.01F;
    const glm::vec3 weights = pathtracer::scene::cie::sceneLuminanceWeights();
    // R and B swing by +/-0.1 against each other in the ratio that cancels their luminance; G's 1e-4 swing leaves luminance spread.
    const float swing = 0.1F;
    const float blue = -swing * weights.x / weights.z;
    const float green = 1e-4F;
    std::array<float, pathtracer::scene::replicateStride(kReplicates)> colour{};
    std::array<float, kReplicates> luminance{};
    for (int r = 0; r < kReplicates; ++r) {
        const float sign = r % 2 == 0 ? 1.0F : -1.0F;
        const pathtracer::scene::Rgb rgb(1.0F + (sign * swing), 1.0F + (sign * green), 1.0F + (sign * blue));
        colour[static_cast<std::size_t>(r)] = rgb.r;
        colour[static_cast<std::size_t>(kReplicates + r)] = rgb.g;
        colour[static_cast<std::size_t>((2 * kReplicates) + r)] = rgb.b;
        luminance[static_cast<std::size_t>(r)] = pathtracer::scene::luminance(rgb);
    }
    // A lone texel's step-up rejects at p <= alpha, p the statistic's lower chi-squared tail.
    const auto pValue = [](double x) { return 1.0 - tools::stats::chiSquareUpperTail(x, kReplicates - 1); };
    constexpr double kAlpha = 0.05;
    const double byLuminance =
        inertStatistic(pathtracer::scene::replicateEstimate(greyTexel(luminance).data(), kReplicates, kReplicates), kReplicates, kThreshold);
    const double byColour =
        inertStatistic(pathtracer::scene::replicateEstimate(colour.data(), kReplicates, kReplicates), kReplicates, kThreshold);
    PT_EXPECT(ctx, pValue(byLuminance) <= kAlpha, "the luminance rule did not stop the chroma-noisy texel: " + std::to_string(byLuminance));
    PT_EXPECT(ctx, pValue(byColour) > kAlpha, "the colour rule stopped a texel with 10% chroma noise: " + std::to_string(byColour));
}

// The IUT's size: alpha when one channel sits on the boundary and the others far inside; below alpha with all three on it.
PT_CHECK(colour_rule_size_is_alpha_at_its_least_favourable_point, Fast, Statistical) {
    constexpr int kReplicates = 8;
    constexpr double kAlpha = 0.05;
    constexpr float kThreshold = 1e-3F;
    constexpr long long kTrials = 200000;
    ctx.plan(2);
    std::mt19937_64 rng(ctx.seed());
    const double boundary = static_cast<double>(kThreshold) * std::sqrt(static_cast<double>(kReplicates));
    std::normal_distribution<double> onBoundary(1.0, boundary);
    std::normal_distribution<double> inside(1.0, 0.01 * boundary);
    std::array<float, pathtracer::scene::replicateStride(kReplicates)> texel{};
    const auto rejectionRate = [&](int boundaryChannels) {
        long long rejections = 0;
        for (long long trial = 0; trial < kTrials; ++trial) {
            for (std::size_t i = 0; i < texel.size(); ++i) {
                texel[i] = static_cast<float>(static_cast<int>(i / kReplicates) < boundaryChannels ? onBoundary(rng) : inside(rng));
            }
            const double x = inertStatistic(pathtracer::scene::replicateEstimate(texel.data(), kReplicates, kReplicates), kReplicates, kThreshold);
            rejections += 1.0 - tools::stats::chiSquareUpperTail(x, kReplicates - 1) <= kAlpha ? 1 : 0;
        }
        return tools::stats::clopperPearsonBand(rejections, kTrials, ctx.alpha());
    };
    const tools::stats::Band one = rejectionRate(1);
    const tools::stats::Band all = rejectionRate(pathtracer::scene::kReplicateChannels);
    char detail[160];
    std::snprintf(detail, sizeof(detail), "one boundary channel: band [%.5f, %.5f] against %.2f", one.lo, one.hi, kAlpha);
    PT_EXPECT(ctx, one.lo <= kAlpha && kAlpha <= one.hi, detail);
    std::snprintf(detail, sizeof(detail), "three boundary channels: band [%.5f, %.5f], must sit below %.2f", all.lo, all.hi, kAlpha);
    PT_EXPECT(ctx, all.hi < kAlpha, detail);
}

// Checkpoints are exactly n = K 2^j >= minSamples, and none exist with the threshold at zero.
PT_CHECK(adaptive_checkpoints_are_power_of_two_rounds, Fast, Exact) {
    constexpr std::array<int, 3> kCounts{2, 4, 8};
    constexpr int kPasses = 4096;
    ctx.plan(static_cast<int>(kCounts.size()) * 3);
    for (const int replicates : kCounts) {
        for (const int minRounds : {1, 4}) {
            pathtracer::scene::PathTraceSettings settings{};
            settings.replicates = replicates;
            settings.adaptive = {0.01F, 0.95F, replicates * minRounds};
            int wrong = 0;
            for (int n = 1; n <= kPasses; ++n) {
                const bool expected = n >= settings.adaptive.minSamples && n % replicates == 0 &&
                                      std::has_single_bit(static_cast<unsigned>(n / replicates));
                wrong += pathtracer::scene::adaptiveCheckpoint(n, settings) != expected ? 1 : 0;
            }
            char detail[128];
            std::snprintf(detail, sizeof(detail), "K=%d, minSamples %d: %d of %d passes misclassified", replicates,
                          settings.adaptive.minSamples, wrong, kPasses);
            PT_EXPECT(ctx, wrong == 0, detail);
        }
        pathtracer::scene::PathTraceSettings off{};
        off.replicates = replicates;
        off.adaptive = {0.0F, 0.95F, replicates};
        bool any = false;
        for (int n = 1; n <= kPasses; ++n) {
            any = any || pathtracer::scene::adaptiveCheckpoint(n, off);
        }
        PT_EXPECT(ctx, !any, "a zero threshold produced a checkpoint");
    }
}

// The test's size: at a true standard error of exactly threshold * mean, the statistic is chi-squared, so P(p <= alpha) = alpha.
PT_CHECK(convergence_statistic_is_chi_squared_at_the_boundary, Fast, Statistical) {
    constexpr std::array<int, 4> kCounts{2, 4, 8, 16};
    constexpr double kAlpha = 0.05;
    // Small, so the sampled mean's own error moves the tolerance by O(threshold^2), far inside the band below.
    constexpr float kThreshold = 1e-3F;
    constexpr long long kTrials = 200000;
    ctx.plan(static_cast<int>(kCounts.size()));
    std::mt19937_64 rng(ctx.seed());
    for (const int replicates : kCounts) {
        // Each replicate mean's spread is sqrt(K) times their average's, which is set to sit exactly on the threshold.
        std::normal_distribution<double> replicate(1.0, static_cast<double>(kThreshold) * std::sqrt(static_cast<double>(replicates)));
        std::vector<float> means(static_cast<std::size_t>(replicates));
        long long rejections = 0;
        for (long long trial = 0; trial < kTrials; ++trial) {
            for (float& mean : means) {
                mean = static_cast<float>(replicate(rng));
            }
            const pathtracer::scene::ReplicateEstimate estimate =
                pathtracer::scene::replicateEstimate(greyTexel(means).data(), replicates, replicates);
            // The p-value through stats.h's own incomplete gamma, independent of the Boost.Math CDF the renderer ranks by.
            const double x = inertStatistic(estimate, replicates, kThreshold);
            rejections += 1.0 - tools::stats::chiSquareUpperTail(x, replicates - 1) <= kAlpha ? 1 : 0;
        }
        const tools::stats::Band band = tools::stats::clopperPearsonBand(rejections, kTrials, ctx.alpha());
        char detail[160];
        std::snprintf(detail, sizeof(detail), "K=%d: p <= %.2f in %lld of %lld, band [%.5f, %.5f]", replicates, kAlpha, rejections, kTrials,
                      band.lo, band.hi);
        PT_EXPECT(ctx, band.lo <= kAlpha && kAlpha <= band.hi, detail);
    }
}

// Half the texels converged far inside the tolerance, half exactly on it, where each null's p-value is uniform.
struct FdrField {
    static constexpr int kReplicates = 4;
    static constexpr int kTexels = 2000;
    static constexpr int kTrials = 400;
    static constexpr float kThreshold = 1e-3F;
    static constexpr float kConfidence = 0.95F;
    static constexpr double kNullShare = 0.5;

    [[nodiscard]] static bool isNull(std::size_t texel) { return static_cast<double>(texel) < kNullShare * kTexels; }

    // One trial's replicate means, nulls first.
    [[nodiscard]] static std::vector<float> draw(std::mt19937_64& rng) {
        const double boundarySpread = static_cast<double>(kThreshold) * std::sqrt(static_cast<double>(kReplicates));
        std::normal_distribution<double> unconverged(1.0, boundarySpread);
        std::normal_distribution<double> converged(1.0, 0.1 * boundarySpread);
        constexpr std::size_t kStride = pathtracer::scene::replicateStride(kReplicates);
        std::vector<float> means(static_cast<std::size_t>(kTexels) * kStride);
        std::array<float, kReplicates> draws{};
        for (std::size_t texel = 0; texel < static_cast<std::size_t>(kTexels); ++texel) {
            for (float& value : draws) {
                value = static_cast<float>(isNull(texel) ? unconverged(rng) : converged(rng));
            }
            const std::vector<float> grey = greyTexel(draws);
            std::copy(grey.begin(), grey.end(), means.begin() + static_cast<std::ptrdiff_t>(texel * kStride));
        }
        return means;
    }

    // The stopped set's false share, given which texels stopped.
    [[nodiscard]] static double falseShare(const std::vector<bool>& stopped) {
        double nulls = 0.0;
        double total = 0.0;
        for (std::size_t texel = 0; texel < stopped.size(); ++texel) {
            nulls += stopped[texel] && isNull(texel) ? 1.0 : 0.0;
            total += stopped[texel] ? 1.0 : 0.0;
        }
        return total > 0.0 ? nulls / total : 0.0;
    }
};

// The step-up's guarantee is sharp: with nulls on the boundary, the stopped set's false share averages exactly pi0 (1 - confidence).
PT_CHECK(step_up_holds_the_false_discovery_rate, Fast, Statistical) {
    using F = FdrField;
    ctx.plan(1);
    std::mt19937_64 rng(ctx.seed());
    const double alpha = 1.0 - static_cast<double>(F::kConfidence);
    constexpr std::size_t kStride = pathtracer::scene::replicateStride(F::kReplicates);
    tools::stats::Welford share;
    for (int trial = 0; trial < F::kTrials; ++trial) {
        const std::vector<float> means = F::draw(rng);
        std::vector<double> p(F::kTexels);
        for (std::size_t texel = 0; texel < p.size(); ++texel) {
            const double x = inertStatistic(pathtracer::scene::replicateEstimate(means.data() + (texel * kStride), F::kReplicates, F::kReplicates),
                                            F::kReplicates, F::kThreshold);
            p[texel] = 1.0 - tools::stats::chiSquareUpperTail(x, F::kReplicates - 1);
        }
        std::vector<double> ascending = p;
        std::sort(ascending.begin(), ascending.end());
        const std::size_t rejected = pathtracer::scene::stepUpRejections(ascending, ascending.size(), alpha);
        std::vector<bool> stopped(p.size(), false);
        for (std::size_t texel = 0; texel < p.size() && rejected > 0; ++texel) {
            stopped[texel] = p[texel] <= ascending[rejected - 1];
        }
        share.add(F::falseShare(stopped));
    }
    const double expected = F::kNullShare * alpha;
    const double half = tools::stats::studentTTwoSided(ctx.alpha(), F::kTrials - 1) * share.standardError();
    char detail[192];
    std::snprintf(detail, sizeof(detail), "mean false share %.5f +/- %.5f against pi0 * alpha = %.5f", share.mean(), half, expected);
    PT_EXPECT(ctx, std::fabs(share.mean() - expected) <= half, detail);
}

// The footprint withdraws stops after the step-up, which bounds no ratio; checked one-sided against pi0 (1 - confidence).
PT_CHECK(retirement_holds_the_false_discovery_rate, Fast, Statistical) {
    using F = FdrField;
    ctx.plan(1);
    pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    pathtracer::scene::PathTraceSettings settings{};
    settings.replicates = F::kReplicates;
    settings.adaptive = {F::kThreshold, F::kConfidence, F::kReplicates, 0.0F, kInertDynamicRange};
    // marSlope 0: the uniform threshold the guarantee is stated at, so the camera is never read.
    const pathtracer::scene::Camera camera{glm::vec3(0.0F), glm::vec3(0.0F), {36.0F, 24.0F}, 35.0F, 0.01F, 100.0F, 2.8F, 0.008F, 400.0F};
    const pathtracer::scene::FoveatedThreshold fovea(camera, F::kTexels, 1, settings.adaptive.marSlope);
    std::mt19937_64 rng(ctx.seed());
    tools::stats::Welford share;
    for (int trial = 0; trial < F::kTrials; ++trial) {
        const std::vector<float> means = F::draw(rng);
        const pathtracer::scene::PathTraceResult frame =
            frameOf(F::kTexels, 1, F::kReplicates, means, F::kReplicates, pathtracer::gfx::PixelLattice::Square, false);
        std::vector<std::uint32_t> active = pathtracer::scene::allTexels(F::kTexels, 1);
        pathtracer::scene::retireConvergedTexels(active, frame, settings, kInertGain, fovea, pool);
        std::vector<bool> stopped(F::kTexels, true);
        for (const std::uint32_t texel : active) {
            stopped[texel] = false;
        }
        share.add(F::falseShare(stopped));
    }
    const double bound = F::kNullShare * (1.0 - static_cast<double>(F::kConfidence));
    const double half = tools::stats::studentTTwoSided(ctx.alpha(), F::kTrials - 1) * share.standardError();
    char detail[192];
    std::snprintf(detail, sizeof(detail), "mean false share %.5f - %.5f above pi0 * alpha = %.5f", share.mean(), half, bound);
    PT_EXPECT(ctx, share.mean() - half <= bound, detail);
}

// The window: absolute error at the floor below black, relative mid-range, each channel damped past the clip; gain only sets units.
PT_CHECK(luminance_window_matches_the_closed_form, Fast, Exact) {
    using pathtracer::scene::ReplicateEstimate;
    using pathtracer::scene::dampedRelativeVariance;
    ctx.plan(4);
    constexpr double kGain = 0.25;
    constexpr double kFloor = 0x1p-12;
    constexpr double kVariance = 0x1p-30;
    const auto ulps = [](double measured, double expected) { return std::fabs(measured - expected) / (std::fabs(expected) * 0x1p-52); };

    // Black: the floor alone, g^2 v / f^2, every channel.
    const std::array<double, 3> black = dampedRelativeVariance(ReplicateEstimate{{0.0, 0.0, 0.0}, {kVariance, kVariance, kVariance}}, kGain, kFloor);
    PT_EXPECT(ctx, black[0] == kGain * kGain * kVariance / (kFloor * kFloor) && black[1] == black[0] && black[2] == black[0],
              "a black texel was not judged at the floor: " + std::to_string(black[0]));

    // Mid-range, 2^-6 exposed: within (f / l)^2 + mu^2 of the plain relative variance v / l^2.
    constexpr double kLevel = 0x1p-6 / kGain;
    const std::array<double, 3> grey = dampedRelativeVariance(ReplicateEstimate{{kLevel, kLevel, kLevel}, {kVariance, kVariance, kVariance}}, kGain, kFloor);
    const double exposed = kGain * kLevel;
    const double departure = std::fabs((grey[0] * kLevel * kLevel / kVariance) - 1.0);
    PT_EXPECT(ctx, departure <= 1.01 * ((kFloor * kFloor / (exposed * exposed)) + (exposed * exposed)),
              "mid-grey departed from the relative rule by " + std::to_string(departure));

    // Red 4x past the clip beside green and blue below it: only red's relative variance falls, by (1 + mu_g^2) / 17.
    const ReplicateEstimate clipped{{4.0 / kGain, 0.5 / kGain, 0.5 / kGain}, {kVariance, kVariance, kVariance}};
    const std::array<double, 3> damped = dampedRelativeVariance(clipped, kGain, kFloor);
    PT_EXPECT(ctx, ulps(damped[0] / damped[1], 1.25 / 17.0) <= 4.0 && damped[1] == damped[2],
              "the clipped channel's damping was " + std::to_string(damped[0] / damped[1]));

    // Scaling radiance by 2^k and gain by 2^-k is exact in binary floating point, so the window must be unchanged bit for bit.
    const ReplicateEstimate scaled{{16.0 / kGain, 2.0 / kGain, 2.0 / kGain}, {16.0 * kVariance, 16.0 * kVariance, 16.0 * kVariance}};
    PT_EXPECT(ctx, dampedRelativeVariance(scaled, kGain / 4.0, kFloor) == damped, "the window depended on units, not exposed radiance");
}

// Support spans against direct enumeration of centre distances, and their runs against the wrapped or clamped frame they cover.
PT_CHECK(support_spans_match_brute_force, Fast, Exact) {
    using pathtracer::gfx::PixelLattice;
    constexpr int kWidth = 41;
    constexpr int kHeight = 23;
    struct Shape {
        PixelLattice lattice;
        double radius;
        bool box;
    };
    std::vector<Shape> shapes;
    for (const PixelLattice lattice : {PixelLattice::Square, PixelLattice::Hexagonal}) {
        for (const int radius : pathtracer::scene::kPoolRadii) {
            shapes.push_back({lattice, static_cast<double>(radius), false});
        }
        shapes.push_back({lattice, static_cast<double>(pathtracer::scene::kFilterRadius), lattice == PixelLattice::Square});
    }
    ctx.plan(static_cast<int>(shapes.size()) * 2);
    for (const Shape& shape : shapes) {
        const auto pitch = static_cast<double>(pathtracer::gfx::rowPitch(shape.lattice));
        const auto column = [&](int y) { return static_cast<double>(pathtracer::gfx::pixelCentre(shape.lattice, 0, y).x); };
        const auto inside = [&](int x, int y, int px, int py) {
            const double dx = (px - x) + column(py) - column(y);
            const double dy = (py - y) * pitch;
            return shape.box ? std::max(std::fabs(dx), std::fabs(dy)) <= shape.radius : (dx * dx) + (dy * dy) <= shape.radius * shape.radius;
        };
        const std::array<std::vector<pathtracer::scene::RowSpan>, 2> spans = pathtracer::scene::supportSpans(shape.lattice, shape.radius, shape.box);
        const int reach = static_cast<int>(shape.radius) + 2;
        int spanErrors = 0;
        int runErrors = 0;
        for (const bool wraps : {false, true}) {
            for (int y = 0; y < kHeight; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    std::vector<int> covered(static_cast<std::size_t>(kWidth) * kHeight, 0);
                    for (const pathtracer::scene::RowSpan& span : spans[static_cast<std::size_t>(y & 1)]) {
                        pathtracer::scene::forEachSpanRun(span, x, y, kWidth, kHeight, wraps, [&](int row, int begin, int end) {
                            for (int column = begin; column < end; ++column) {
                                ++covered[(static_cast<std::size_t>(row) * kWidth) + static_cast<std::size_t>(column)];
                            }
                        });
                    }
                    std::vector<int> expected(covered.size(), 0);
                    for (int py = y - reach; py <= y + reach; ++py) {
                        for (int px = x - reach - kWidth; px <= x + reach + kWidth; ++px) {
                            if (py < 0 || py >= kHeight || !inside(x, y, px, py) || (!wraps && (px < 0 || px >= kWidth))) {
                                continue;
                            }
                            // A wrapped disc wider than the frame still counts each texel once.
                            expected[(static_cast<std::size_t>(py) * kWidth) + static_cast<std::size_t>(pathtracer::gfx::wrapIndex(px, kWidth))] = 1;
                        }
                    }
                    runErrors += covered == expected ? 0 : 1;
                }
            }
        }
        for (int parity = 0; parity < 2; ++parity) {
            for (int dy = -reach; dy <= reach; ++dy) {
                int lo = reach + 1;
                int hi = -reach - 1;
                for (int dx = -reach; dx <= reach; ++dx) {
                    if (inside(0, parity, dx, parity + dy)) {
                        lo = std::min(lo, dx);
                        hi = std::max(hi, dx);
                    }
                }
                const auto& list = spans[static_cast<std::size_t>(parity)];
                const auto found = std::find_if(list.begin(), list.end(), [dy](const pathtracer::scene::RowSpan& s) { return s.dy == dy; });
                const bool match = lo > hi ? found == list.end() : found != list.end() && found->lo == lo && found->hi == hi;
                spanErrors += match ? 0 : 1;
            }
        }
        const std::string name = std::string(shape.lattice == PixelLattice::Square ? "square" : "hexagonal") + " radius " +
                                 std::to_string(shape.radius) + (shape.box ? " box" : " disc");
        PT_EXPECT(ctx, spanErrors == 0, name + ": " + std::to_string(spanErrors) + " rows disagree with enumeration");
        PT_EXPECT(ctx, runErrors == 0, name + ": " + std::to_string(runErrors) + " centres covered the frame wrongly");
    }
}

// A lone active texel whose own spread is tight stops among quiet neighbours and is kept among noisy ones it would pool with.
PT_CHECK(pooling_vetoes_a_lucky_texel, Fast, Exact) {
    constexpr int kSize = 33;
    constexpr int kReplicates = 4;
    constexpr int kSamples = 64;
    constexpr float kThreshold = 0.01F;
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    pathtracer::scene::PathTraceSettings settings{};
    settings.replicates = kReplicates;
    settings.adaptive = {kThreshold, 0.95F, kReplicates, 0.0F, kInertDynamicRange};
    const pathtracer::scene::Camera camera{glm::vec3(0.0F), glm::vec3(0.0F), {36.0F, 24.0F}, 35.0F, 0.01F, 100.0F, 2.8F, 0.008F, 400.0F};
    const pathtracer::scene::FoveatedThreshold fovea(camera, kSize, kSize, 0.0F);
    const auto centre = static_cast<std::uint32_t>(((kSize / 2) * kSize) + (kSize / 2));
    // Means 1 -+ a give SD(mbar) = a / sqrt(3): the centre at 1/100 of its tolerance, noisy neighbours at 10x theirs.
    const auto retired = [&](float neighbourSpread) {
        std::vector<float> means(static_cast<std::size_t>(kSize) * kSize * pathtracer::scene::replicateStride(kReplicates));
        for (std::size_t i = 0; i < means.size(); ++i) {
            const bool isCentre = i / pathtracer::scene::replicateStride(kReplicates) == centre;
            const float a = std::sqrt(3.0F) * kThreshold * (isCentre ? 0.01F : neighbourSpread);
            means[i] = (i % 2 == 0) ? 1.0F - a : 1.0F + a;
        }
        const pathtracer::scene::PathTraceResult frame =
            frameOf(kSize, kSize, kReplicates, means, kSamples, pathtracer::gfx::PixelLattice::Square, false);
        // The rest stopped at earlier checkpoints: they pool but are not live, so no footprint can hold the centre.
        std::vector<std::uint32_t> active{centre};
        pathtracer::scene::retireConvergedTexels(active, frame, settings, kInertGain, fovea, pool);
        return active.empty();
    };
    PT_EXPECT(ctx, retired(0.01F), "a converged texel among converged neighbours did not stop");
    PT_EXPECT(ctx, !retired(10.0F), "a texel with a lucky spread stopped among neighbours at 10x the tolerance");
}

// One texel at its tolerance amid converged ones: the step-up stops the rest, and the footprint keeps exactly its support's neighbours.
PT_CHECK(footprint_keeps_live_neighbours_live, Fast, Exact) {
    using pathtracer::gfx::PixelLattice;
    constexpr int kWidth = 16;
    constexpr int kHeight = 9;
    constexpr int kReplicates = 4;
    constexpr float kThreshold = 0.01F;
    struct Case {
        PixelLattice lattice;
        bool wraps;
        int x;
        int y;
        std::size_t neighbours;
    };
    // Square's separable filter reaches the 8 around it, hexagonal's radial one the 6 at distance 1; x = 0 wraps across the seam.
    constexpr std::array<Case, 5> kCases{{{PixelLattice::Square, false, 7, 4, 8},
                                          {PixelLattice::Square, true, 0, 4, 8},
                                          {PixelLattice::Hexagonal, false, 7, 4, 6},
                                          {PixelLattice::Hexagonal, false, 7, 5, 6},
                                          {PixelLattice::Hexagonal, true, 0, 3, 6}}};
    ctx.plan(static_cast<int>(kCases.size()));
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    pathtracer::scene::PathTraceSettings settings{};
    settings.replicates = kReplicates;
    settings.adaptive = {kThreshold, 0.95F, kReplicates, 0.0F, kInertDynamicRange};
    const pathtracer::scene::Camera camera{glm::vec3(0.0F), glm::vec3(0.0F), {36.0F, 24.0F}, 35.0F, 0.01F, 100.0F, 2.8F, 0.008F, 400.0F};
    const pathtracer::scene::FoveatedThreshold fovea(camera, kWidth, kHeight, 0.0F);
    for (const Case& testCase : kCases) {
        const auto noisy = static_cast<std::uint32_t>((testCase.y * kWidth) + testCase.x);
        std::vector<float> means(static_cast<std::size_t>(kWidth) * kHeight * pathtracer::scene::replicateStride(kReplicates));
        for (std::size_t i = 0; i < means.size(); ++i) {
            // On its tolerance the noisy texel's statistic is K-1, far above the screen; its neighbours pool it well inside theirs.
            const float a = std::sqrt(3.0F) * kThreshold * (i / pathtracer::scene::replicateStride(kReplicates) == noisy ? 1.0F : 1e-3F);
            means[i] = (i % 2 == 0) ? 1.0F - a : 1.0F + a;
        }
        const pathtracer::scene::PathTraceResult frame =
            frameOf(kWidth, kHeight, kReplicates, means, kReplicates, testCase.lattice, testCase.wraps);
        std::vector<std::uint32_t> active = pathtracer::scene::allTexels(kWidth, kHeight);
        pathtracer::scene::retireConvergedTexels(active, frame, settings, kInertGain, fovea, pool);
        const glm::vec2 centre = pathtracer::gfx::pixelCentre(testCase.lattice, testCase.x, testCase.y);
        bool adjacent = true;
        for (const std::uint32_t texel : active) {
            const int x = static_cast<int>(texel) % kWidth;
            const int y = static_cast<int>(texel) / kWidth;
            glm::vec2 offset = pathtracer::gfx::pixelCentre(testCase.lattice, x, y) - centre;
            offset.x -= testCase.wraps ? static_cast<float>(kWidth) * std::round(offset.x / static_cast<float>(kWidth)) : 0.0F;
            adjacent = adjacent && std::max(std::fabs(offset.x), std::fabs(offset.y)) < 1.5F && glm::length(offset) < 1.5F;
        }
        const bool kept = std::binary_search(active.begin(), active.end(), noisy);
        PT_EXPECT(ctx, kept && adjacent && active.size() == testCase.neighbours + 1,
                  "kept " + std::to_string(active.size()) + " texels, expected the noisy one and " + std::to_string(testCase.neighbours));
    }
}

// Pinhole closed form: e = atan(r), Omega = s_x s_y cos^3 e, scale = max(1, w(e)/sqrt(Omega)) / max(1, w0/sqrt(Omega_gaze)).
PT_CHECK(foveated_threshold_matches_the_pinhole_closed_form, Fast, Exact) {
    using pathtracer::scene::Camera;
    using pathtracer::scene::FoveatedThreshold;
    constexpr float kMarSlope = 0.022F;
    // Texels coarser than w0, where the gaze pools nothing, and finer, where it pools several; strided so the fine frame stays fast.
    constexpr std::array<std::array<int, 3>, 2> kFrames{{{33, 17, 1}, {4096, 2731, 97}}};
    ctx.plan(static_cast<int>(kFrames.size()) + 1);
    const Camera pinhole{glm::vec3(0.0F), glm::vec3(0.0F), {36.0F, 24.0F}, 35.0F, 0.01F, 100.0F, 2.8F, 0.008F, 400.0F};
    const double w0 = static_cast<double>(pathtracer::scene::kFovealMarDegrees) * std::numbers::pi / 180.0;
    for (const auto& [width, height, stride] : kFrames) {
        const FoveatedThreshold fovea(pinhole, width, height, kMarSlope);
        const double halfHeight = std::tan(0.5 * static_cast<double>(pinhole.verticalFovRadians()));
        const double halfWidth = halfHeight * width / height;
        const double cell = (2.0 * halfWidth / width) * (2.0 * halfHeight / height);
        const double gazeSpan = std::max(1.0, w0 / std::sqrt(cell));
        double worst = 0.0;
        for (int texel = 0; texel < width * height; texel += stride) {
            const double u = (((texel % width) + 0.5) / width * 2.0 - 1.0) * halfWidth;
            const double v = (1.0 - ((texel / width) + 0.5) / height * 2.0) * halfHeight;
            const double eccentricity = std::atan(std::hypot(u, v));
            const double solidAngle = cell * std::pow(std::cos(eccentricity), 3.0);
            const double mar = w0 + (static_cast<double>(kMarSlope) * eccentricity);
            const double expected = std::max(1.0, mar / std::sqrt(solidAngle)) / gazeSpan;
            const auto measured = static_cast<double>(fovea.scale(static_cast<std::uint32_t>(texel)));
            worst = std::max(worst, std::abs(measured - expected) / expected);
        }
        // A handful of float trig, cross products and a square root deep: well inside 16 ulp of the double reference.
        PT_EXPECT(ctx, worst <= 16.0 * static_cast<double>(kFloatEpsilon),
                  std::to_string(width) + "x" + std::to_string(height) + " scale relative error " + std::to_string(worst));
    }
    const FoveatedThreshold off(pinhole, 33, 17, 0.0F);
    bool uniform = true;
    for (std::uint32_t texel = 0; texel < 33U * 17U; ++texel) {
        uniform = uniform && off.scale(texel) == 1.0F;
    }
    PT_EXPECT(ctx, uniform, "marSlope 0 scaled a texel's threshold");
}

// Equal replicate spread everywhere, twice the gaze's tolerance: only texels loosened past it stop, outermost first.
PT_CHECK(retirement_loosens_with_eccentricity, Fast, Exact) {
    constexpr int kWidth = 64;
    constexpr int kHeight = 43;
    constexpr int kReplicates = 4;
    constexpr float kThreshold = 0.01F;
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    pathtracer::scene::PathTraceSettings settings{};
    settings.replicates = kReplicates;
    // Steep, so a 64-texel frame's 0.6-degree texels pool well inside its 27-degree half-width.
    settings.adaptive = {kThreshold, 0.95F, kReplicates, 0.5F, kInertDynamicRange};
    const pathtracer::scene::Camera camera{glm::vec3(0.0F), glm::vec3(0.0F), {36.0F, 24.0F}, 35.0F, 0.01F, 100.0F, 2.8F, 0.008F, 400.0F};
    const pathtracer::scene::FoveatedThreshold fovea(camera, kWidth, kHeight, settings.adaptive.marSlope);
    // Means 1 -+ a give SD(mbar) = a / sqrt(3); a = 2 sqrt(3) threshold puts it at twice the unscaled tolerance.
    const float a = 2.0F * std::sqrt(3.0F) * kThreshold;
    // Even K alternates identically in each channel block, so every texel is grey.
    std::vector<float> means(static_cast<std::size_t>(kWidth) * kHeight * pathtracer::scene::replicateStride(kReplicates));
    for (std::size_t i = 0; i < means.size(); ++i) {
        means[i] = (i % 2 == 0) ? 1.0F - a : 1.0F + a;
    }
    const pathtracer::scene::PathTraceResult frame =
        frameOf(kWidth, kHeight, kReplicates, means, kReplicates, pathtracer::gfx::PixelLattice::Square, false);
    std::vector<std::uint32_t> active = pathtracer::scene::allTexels(kWidth, kHeight);
    pathtracer::scene::retireConvergedTexels(active, frame, settings, kInertGain, fovea, pool);
    std::vector<bool> kept(static_cast<std::size_t>(kWidth) * kHeight, false);
    for (const std::uint32_t texel : active) {
        kept[texel] = true;
    }
    float minStopped = std::numeric_limits<float>::infinity();
    for (std::uint32_t texel = 0; texel < kept.size(); ++texel) {
        minStopped = kept[texel] ? minStopped : std::min(minStopped, fovea.scale(texel));
    }
    // The test kept every texel below the stopped scales; the footprint may add only their 8 neighbours, never a lone outer texel.
    const auto core = [&](int x, int y) {
        const auto texel = static_cast<std::uint32_t>((y * kWidth) + x);
        return x >= 0 && x < kWidth && y >= 0 && y < kHeight && kept[texel] && fovea.scale(texel) < minStopped;
    };
    int strays = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            bool beside = false;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    beside = beside || core(x + dx, y + dy);
                }
            }
            strays += kept[static_cast<std::size_t>((y * kWidth) + x)] && !beside ? 1 : 0;
        }
    }
    const auto centre = static_cast<std::uint32_t>(((kHeight / 2) * kWidth) + (kWidth / 2));
    PT_EXPECT(ctx, kept[centre], "the gaze texel stopped at twice its tolerance");
    PT_EXPECT(ctx, !kept.front() && !kept.back(), "a corner texel kept sampling under foveation");
    PT_EXPECT(ctx, strays == 0, std::to_string(strays) + " kept texels lie neither below the stopped scales nor beside one that does");
}

// A smooth radiance ramp, Denoised's ground truth: any noise added to it is error a denoiser must remove, not detail it must keep.
[[nodiscard]] HdrImage rampImage(int width, int height) {
    HdrImage image = pathtracer::gfx::makeImage(width, height, pathtracer::gfx::kRgbChannels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(width);
            const float v = static_cast<float>(y) / static_cast<float>(height);
            float* texel = image.texels.data() + ((static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x)) * 3;
            texel[0] = 0.2F + (0.6F * u);
            texel[1] = 0.3F + (0.4F * v);
            texel[2] = 0.5F;
        }
    }
    return image;
}

[[nodiscard]] HdrImage constantImage(int width, int height, const glm::vec3& value) {
    HdrImage image = pathtracer::gfx::makeImage(width, height, pathtracer::gfx::kRgbChannels);
    for (std::size_t texel = 0; texel < image.texels.size(); texel += 3) {
        image.texels[texel] = value.x;
        image.texels[texel + 1] = value.y;
        image.texels[texel + 2] = value.z;
    }
    return image;
}

[[nodiscard]] double rmse(const HdrImage& a, const HdrImage& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.texels.size(); ++i) {
        const double d = static_cast<double>(a.texels[i]) - static_cast<double>(b.texels[i]);
        sum += d * d;
    }
    return std::sqrt(sum / static_cast<double>(a.texels.size()));
}

// Every profile setting removes noise from a clean ramp, keeps its shape, and only reads its inputs: published lanes must never be written.
PT_CHECK(denoised_reduces_error_and_leaves_its_inputs_unchanged, Fast, Exact) {
    using pathtracer::debug::DenoiseQuality;
    constexpr std::array<DenoiseQuality, 3> kQualities{DenoiseQuality::Fast, DenoiseQuality::Balanced, DenoiseQuality::High};
    ctx.plan(static_cast<int>(kQualities.size()) * 2 * 4);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    const HdrImage clean = rampImage(kWidth, kHeight);
    HdrImage noisy = clean;
    // Multiplicative, as Monte Carlo noise scales with the signal; fixed seed, so the check is deterministic.
    std::mt19937 rng(1234U);
    std::normal_distribution<float> gauss(0.0F, 0.3F);
    for (float& value : noisy.texels) {
        value = std::max(0.0F, value * (1.0F + gauss(rng)));
    }
    const HdrImage albedo = constantImage(kWidth, kHeight, glm::vec3(0.5F));
    const HdrImage normal = constantImage(kWidth, kHeight, glm::vec3(0.0F, 0.0F, 1.0F));
    const HdrImage noisyCopy = noisy;
    const HdrImage albedoCopy = albedo;
    const HdrImage normalCopy = normal;
    std::vector<pathtracer::debug::DenoiserOptions> settings;
    for (const DenoiseQuality quality : kQualities) {
        for (const bool prefilter : {true, false}) {
            settings.push_back({quality, prefilter});
        }
    }
    for (const pathtracer::debug::DenoiserOptions& options : settings) {
        const std::string setting = std::string(pathtracer::debug::kDenoiseQualityNames[static_cast<int>(options.quality)]) +
                                    (options.prefilterAuxiliary ? ", prefiltered" : ", unprefiltered");
        pathtracer::debug::Denoiser denoiser(options);
        pathtracer::debug::FilterInput input{noisy};
        input.denoiseAlbedo = &albedo;
        input.denoiseNormal = &normal;
        input.denoiser = &denoiser;
        const HdrImage denoised = pathtracer::debug::evaluateFilterAov(pathtracer::debug::AovId::Denoised, input, pool);
        PT_EXPECT(ctx, denoised.width == kWidth && denoised.height == kHeight && denoised.channels == pathtracer::gfx::kRgbChannels &&
                           denoised.texels.size() == noisy.texels.size(),
                  setting + ": denoised is not an RGB image of the beauty's size");
        PT_EXPECT(ctx, std::all_of(denoised.texels.begin(), denoised.texels.end(), [](float value) { return std::isfinite(value); }),
                  setting + ": denoised holds a non-finite value");
        PT_EXPECT(ctx, rmse(denoised, clean) < rmse(noisy, clean), setting + ": denoising did not bring the noisy ramp closer to the clean one");
        PT_EXPECT(ctx, noisy.texels == noisyCopy.texels && albedo.texels == albedoCopy.texels && normal.texels == normalCopy.texels,
                  setting + ": denoising wrote to one of its inputs");
    }
}

// Every misuse is reported, never absorbed: a shape mismatch and a missing feature are std::invalid_argument, OIDN's own errors runtime.
PT_CHECK(denoised_reports_misuse_and_oidn_errors, Fast, Exact) {
    ctx.plan(5);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const HdrImage beauty = rampImage(16, 8);
    const HdrImage albedo = constantImage(16, 8, glm::vec3(0.5F));
    const HdrImage normal = constantImage(16, 8, glm::vec3(0.0F, 0.0F, 1.0F));
    pathtracer::debug::Denoiser denoiser({pathtracer::debug::DenoiseQuality::High, true});
    const auto throwsInvalid = [](const auto& call) {
        try {
            static_cast<void>(call());
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    PT_EXPECT(ctx, throwsInvalid([&] { return denoiser.denoise(beauty, rampImage(8, 8), normal); }), "an albedo of another size was accepted");
    PT_EXPECT(ctx, throwsInvalid([&] { return denoiser.denoise(beauty, albedo, pathtracer::gfx::makeImage(16, 8, 1)); }),
              "a one-channel normal was accepted");
    PT_EXPECT(ctx, throwsInvalid([&] { return pathtracer::debug::evaluateFilterAov(pathtracer::debug::AovId::Denoised, {beauty}, pool); }),
              "Denoised ran without its features and a Denoiser");
    // OIDN rejects an empty image as unbound, which only it can detect: its message must reach the caller, prefixed so its origin shows.
    std::string message;
    try {
        static_cast<void>(denoiser.denoise(rampImage(0, 0), rampImage(0, 0), rampImage(0, 0)));
    } catch (const std::runtime_error& error) {
        message = error.what();
    }
    PT_EXPECT(ctx, message.rfind("OIDN: ", 0) == 0 && message.size() > 6, "OIDN's error did not surface with its own message: " + message);
    // The failure is the call's alone: OIDN clears its error on read, so the same denoiser serves the next valid call.
    PT_EXPECT(ctx, denoiser.denoise(beauty, albedo, normal).texels.size() == beauty.texels.size(), "a denoiser stayed broken after an OIDN error");
}

PT_CHECK_MAIN("filter")
