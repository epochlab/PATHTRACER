// Correctness gate for the Beauty filters and adaptive sampling: SNR, the replicate stopping rule, foveation, OIDN.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
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
        std::vector<float> means(kPixels * static_cast<std::size_t>(replicates));
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
    std::vector<float> means;
    means.reserve(kPixels * kReplicates);
    for (std::size_t i = 0; i < kPixels; ++i) {
        means.insert(means.end(), pixel.begin(), pixel.end());
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
    std::vector<float> spread(kPixels * kReplicates);
    for (std::size_t i = 0; i < spread.size(); ++i) {
        spread[i] = (i % 2 == 0) ? 0.75F : 0.5F;
    }
    const std::vector<float> dark(kPixels * kReplicates, 0.0F);
    const std::vector<float> agreed(kPixels * kReplicates, 0.75F);
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
        const pathtracer::scene::ReplicateEstimate estimate = pathtracer::scene::replicateEstimate(means.data(), replicates);
        const double expected = static_cast<double>(kDelta) * kDelta / (replicates - 1.0);
        // Two roundings, 1/(K(K-1)) and its product, each within half an ulp of double.
        const double bound = 2.0 * std::numeric_limits<double>::epsilon() * expected;
        char detail[160];
        std::snprintf(detail, sizeof(detail), "K=%d: mean %.17g, variance %.17g against %.17g", replicates, estimate.mean,
                      estimate.meanVariance, expected);
        PT_EXPECT(ctx, estimate.mean == kMean && std::fabs(estimate.meanVariance - expected) <= bound, detail);
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

// Replicates that agree exactly carry no variance to test, whatever their mean; zero mean with spread and NaN meet no relative tolerance.
PT_CHECK(convergence_statistic_needs_spread_and_signal, Fast, Exact) {
    ctx.plan(5);
    constexpr int kReplicates = 4;
    const auto statistic = [&](std::array<float, kReplicates> means, float threshold) {
        return pathtracer::scene::convergenceStatistic(pathtracer::scene::replicateEstimate(means.data(), kReplicates), kReplicates,
                                                       threshold);
    };
    constexpr float kHuge = std::numeric_limits<float>::max();
    constexpr double kNoEvidence = std::numeric_limits<double>::infinity();
    PT_EXPECT(ctx, statistic({0.0F, 0.0F, 0.0F, 0.0F}, kHuge) == kNoEvidence, "replicates agreeing on zero gave evidence of convergence");
    PT_EXPECT(ctx, statistic({0.5F, 0.5F, 0.5F, 0.5F}, kHuge) == kNoEvidence, "replicates agreeing on a positive value gave evidence");
    PT_EXPECT(ctx, statistic({0.5F, 0.5F, 0.5F, std::nextafter(0.5F, 1.0F)}, kHuge) < 1e-30,
              "the least spread under any threshold was not converged");
    PT_EXPECT(ctx, statistic({0.25F, -0.25F, 0.25F, -0.25F}, kHuge) == kNoEvidence, "a zero mean with spread gave evidence of convergence");
    PT_EXPECT(ctx, statistic({0.5F, std::numeric_limits<float>::quiet_NaN(), 0.5F, 0.5F}, kHuge) == kNoEvidence,
              "a NaN replicate gave evidence");
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
            const pathtracer::scene::ReplicateEstimate estimate = pathtracer::scene::replicateEstimate(means.data(), replicates);
            // The p-value through stats.h's own incomplete gamma, independent of the Boost.Math CDF the renderer ranks by.
            const double x = pathtracer::scene::convergenceStatistic(estimate, replicates, kThreshold);
            rejections += 1.0 - tools::stats::chiSquareUpperTail(x, replicates - 1) <= kAlpha ? 1 : 0;
        }
        const tools::stats::Band band = tools::stats::clopperPearsonBand(rejections, kTrials, ctx.alpha());
        char detail[160];
        std::snprintf(detail, sizeof(detail), "K=%d: p <= %.2f in %lld of %lld, band [%.5f, %.5f]", replicates, kAlpha, rejections, kTrials,
                      band.lo, band.hi);
        PT_EXPECT(ctx, band.lo <= kAlpha && kAlpha <= band.hi, detail);
    }
}

// The retirement's guarantee: with unconverged texels on the boundary, the stopped set's false share averages pi0 * (1 - confidence).
PT_CHECK(retirement_holds_the_false_discovery_rate, Fast, Statistical) {
    constexpr int kReplicates = 4;
    constexpr int kTexels = 2000;
    constexpr int kTrials = 400;
    constexpr float kThreshold = 1e-3F;
    constexpr float kConfidence = 0.95F;
    // Half the texels converged far inside the tolerance, half exactly on it, where each null's p-value is uniform.
    constexpr double kNullShare = 0.5;
    ctx.plan(1);
    pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    pathtracer::scene::PathTraceSettings settings{};
    settings.replicates = kReplicates;
    settings.adaptive = {kThreshold, kConfidence, kReplicates};
    // marSlope 0: the uniform threshold the guarantee is stated at, so the camera is never read.
    const pathtracer::scene::Camera camera{glm::vec3(0.0F), glm::vec3(0.0F), {36.0F, 24.0F}, 35.0F, 0.01F, 100.0F, 2.8F, 0.008F, 400.0F};
    const pathtracer::scene::FoveatedThreshold fovea(camera, kTexels, 1, settings.adaptive.marSlope);
    std::mt19937_64 rng(ctx.seed());
    const double boundarySpread = static_cast<double>(kThreshold) * std::sqrt(static_cast<double>(kReplicates));
    std::normal_distribution<double> unconverged(1.0, boundarySpread);
    std::normal_distribution<double> converged(1.0, 0.1 * boundarySpread);
    tools::stats::Welford falseShare;
    std::vector<float> means(static_cast<std::size_t>(kTexels) * kReplicates);
    for (int trial = 0; trial < kTrials; ++trial) {
        for (std::size_t i = 0; i < means.size(); ++i) {
            const bool isNull = static_cast<double>(i / kReplicates) < kNullShare * kTexels;
            means[i] = static_cast<float>(isNull ? unconverged(rng) : converged(rng));
        }
        std::vector<std::uint32_t> active(kTexels);
        std::iota(active.begin(), active.end(), 0U);
        pathtracer::scene::retireConvergedTexels(active, means.data(), settings, fovea, pool);
        // Survivors are ascending, so the stopped nulls are the null indices missing from them.
        const auto nullsKept = static_cast<double>(std::count_if(active.begin(), active.end(), [](std::uint32_t texel) {
            return static_cast<double>(texel) < kNullShare * kTexels;
        }));
        const double stopped = kTexels - static_cast<double>(active.size());
        falseShare.add(stopped > 0.0 ? ((kNullShare * kTexels) - nullsKept) / stopped : 0.0);
    }
    const double expected = kNullShare * (1.0 - static_cast<double>(kConfidence));
    const double half = tools::stats::studentTTwoSided(ctx.alpha(), kTrials - 1) * falseShare.standardError();
    char detail[192];
    std::snprintf(detail, sizeof(detail), "mean false share %.5f +/- %.5f against pi0 * alpha = %.5f", falseShare.mean(), half, expected);
    PT_EXPECT(ctx, std::fabs(falseShare.mean() - expected) <= half, detail);
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
    settings.adaptive = {kThreshold, 0.95F, kReplicates, 0.5F};
    const pathtracer::scene::Camera camera{glm::vec3(0.0F), glm::vec3(0.0F), {36.0F, 24.0F}, 35.0F, 0.01F, 100.0F, 2.8F, 0.008F, 400.0F};
    const pathtracer::scene::FoveatedThreshold fovea(camera, kWidth, kHeight, settings.adaptive.marSlope);
    // Means 1 -+ a give SD(mbar) = a / sqrt(3); a = 2 sqrt(3) threshold puts it at twice the unscaled tolerance.
    const float a = 2.0F * std::sqrt(3.0F) * kThreshold;
    std::vector<float> means(static_cast<std::size_t>(kWidth) * kHeight * kReplicates);
    for (std::size_t i = 0; i < means.size(); ++i) {
        means[i] = (i % 2 == 0) ? 1.0F - a : 1.0F + a;
    }
    std::vector<std::uint32_t> active = pathtracer::scene::allTexels(kWidth, kHeight);
    pathtracer::scene::retireConvergedTexels(active, means.data(), settings, fovea, pool);
    std::vector<bool> kept(static_cast<std::size_t>(kWidth) * kHeight, false);
    for (const std::uint32_t texel : active) {
        kept[texel] = true;
    }
    float maxKept = 0.0F;
    float minStopped = std::numeric_limits<float>::infinity();
    for (std::uint32_t texel = 0; texel < kept.size(); ++texel) {
        if (kept[texel]) {
            maxKept = std::max(maxKept, fovea.scale(texel));
        } else {
            minStopped = std::min(minStopped, fovea.scale(texel));
        }
    }
    const auto centre = static_cast<std::uint32_t>(((kHeight / 2) * kWidth) + (kWidth / 2));
    PT_EXPECT(ctx, kept[centre], "the gaze texel stopped at twice its tolerance");
    PT_EXPECT(ctx, !kept.front() && !kept.back(), "a corner texel kept sampling under foveation");
    PT_EXPECT(ctx, maxKept < minStopped, "kept scale " + std::to_string(maxKept) + " not below stopped " + std::to_string(minStopped));
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
