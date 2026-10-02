// Correctness gate for the Optic Flow estimator: exact zeros and priors, gain invariance, and errors explained by the reported sigma.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/optic_flow.h"
#include "pathtracer/debug/scale_space.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using pathtracer::gfx::HdrImage;
using pathtracer::scene::ThreadPool;

// Two octaves fit 192 x 128, so every check exercises the coarse-to-fine hand-off as well as the finest octave.
constexpr int kWidth = 192;
constexpr int kHeight = 128;

// 5 orientations, 2 unknowns: Q/2 ~ F(2, 3) (Hotelling 1931), so P(|e| <= 3 sigma_max) >= 1 - (1 + 9/3)^(-3/2) = 0.875 at worst.
constexpr double kCoverageFloor = 0.875;

struct PlaneWave {
    double kx;
    double ky;
    double phase;
};

// A field of plane waves inside the bank's passband, so any sub-pixel translation of it is exact: no resampling enters the truth.
[[nodiscard]] std::vector<PlaneWave> waveField(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    // Up to half the finest carrier, the band both octaves at this size respond to.
    const double reach = 0.5 * static_cast<double>(pathtracer::debug::morletCarrier(pathtracer::debug::innerScaleVariance()));
    std::vector<PlaneWave> waves(256);
    for (PlaneWave& wave : waves) {
        const double radius = reach * std::sqrt(unit(rng));
        const double angle = 2.0 * std::numbers::pi * unit(rng);
        wave = {radius * std::cos(angle), radius * std::sin(angle), 2.0 * std::numbers::pi * unit(rng)};
    }
    return waves;
}

// One channel, the field displaced by (dx, dy) and offset to stay positive, as a radiance would be.
[[nodiscard]] HdrImage translated(const std::vector<PlaneWave>& waves, double dx, double dy) {
    HdrImage image = pathtracer::gfx::makeImage(kWidth, kHeight, pathtracer::gfx::kScalarChannels);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            double value = 0.0;
            for (const PlaneWave& wave : waves) {
                value += std::cos((wave.kx * (static_cast<double>(x) - dx)) + (wave.ky * (static_cast<double>(y) - dy)) + wave.phase);
            }
            image.texels[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)] =
                static_cast<float>(1.0 + (value / static_cast<double>(waves.size())));
        }
    }
    return image;
}

// Flow from `before` to `after`, both deterministic: no variance, so only rounding and border bounds weigh the phases.
[[nodiscard]] HdrImage flowBetween(const HdrImage& after, const HdrImage* before, ThreadPool& pool) {
    return pathtracer::debug::opticFlowAov({&after, nullptr}, {before, nullptr}, pool);
}

[[nodiscard]] const float* flowAt(const HdrImage& flow, std::size_t pixel) {
    return flow.texels.data() + (pixel * static_cast<std::size_t>(flow.channels));
}

// Fraction of texels whose error from (dx, dy) lies within three of their own reported sigma.
[[nodiscard]] double coverage(const HdrImage& flow, double dx, double dy) {
    std::size_t covered = 0;
    const auto pixels = static_cast<std::size_t>(flow.width) * static_cast<std::size_t>(flow.height);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        const float* texel = flowAt(flow, pixel);
        const double error = std::hypot(static_cast<double>(texel[0]) - dx, static_cast<double>(texel[1]) - dy);
        covered += error <= 3.0 * static_cast<double>(texel[2]) ? 1U : 0U;
    }
    return static_cast<double>(covered) / static_cast<double>(pixels);
}

// Fraction of texels whose estimate is closer to (dx, dy) than the prior's zero: the data must have improved on the prior's guess.
[[nodiscard]] double beatsPrior(const HdrImage& flow, double dx, double dy) {
    std::size_t better = 0;
    const auto pixels = static_cast<std::size_t>(flow.width) * static_cast<std::size_t>(flow.height);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        const float* texel = flowAt(flow, pixel);
        better += std::hypot(static_cast<double>(texel[0]) - dx, static_cast<double>(texel[1]) - dy) < std::hypot(dx, dy) ? 1U : 0U;
    }
    return static_cast<double>(better) / static_cast<double>(pixels);
}

// True when every texel reads (0, 0, spread) exactly.
[[nodiscard]] bool readsPrior(const HdrImage& flow, float spread) {
    const auto pixels = static_cast<std::size_t>(flow.width) * static_cast<std::size_t>(flow.height);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        const float* texel = flowAt(flow, pixel);
        if (texel[0] != 0.0F || texel[1] != 0.0F || texel[2] != spread) {
            return false;
        }
    }
    return true;
}

}  // namespace

// Identical frames make every phase step exactly zero, so the posterior mean is exactly zero and its sigma finite and inside the prior.
PT_CHECK(identical_frames_give_exactly_zero_flow, Fast, Exact) {
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const HdrImage frame = translated(waveField(ctx.seed()), 0.0, 0.0);
    const HdrImage flow = flowBetween(frame, &frame, pool);
    const float prior = pathtracer::debug::opticFlowPriorStd(kWidth, kHeight);
    bool zero = true;
    bool bounded = true;
    for (std::size_t pixel = 0; pixel < frame.texels.size(); ++pixel) {
        const float* texel = flowAt(flow, pixel);
        zero = zero && texel[0] == 0.0F && texel[1] == 0.0F;
        bounded = bounded && std::isfinite(texel[2]) && texel[2] > 0.0F && texel[2] <= prior;
    }
    PT_EXPECT(ctx, zero, "identical frames produced a nonzero displacement");
    PT_EXPECT(ctx, bounded, "a sigma is non-finite, non-positive, or wider than the prior");
}

// With nothing to pair against the posterior is the prior, bit for bit: no previous, another size, another channel count, too small.
PT_CHECK(an_unpaired_view_reads_the_prior, Fast, Exact) {
    ctx.plan(5);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const std::vector<PlaneWave> waves = waveField(ctx.seed());
    const HdrImage frame = translated(waves, 0.0, 0.0);
    const float prior = pathtracer::debug::opticFlowPriorStd(kWidth, kHeight);
    PT_EXPECT(ctx, readsPrior(flowBetween(frame, nullptr, pool), prior), "no previous frame");
    const HdrImage narrower = pathtracer::gfx::makeImage(kWidth - 1, kHeight, pathtracer::gfx::kScalarChannels);
    PT_EXPECT(ctx, readsPrior(flowBetween(frame, &narrower, pool), prior), "a previous frame of another width");
    const HdrImage colour = pathtracer::gfx::makeImage(kWidth, kHeight, pathtracer::gfx::kRgbChannels);
    PT_EXPECT(ctx, readsPrior(flowBetween(frame, &colour, pool), prior), "a previous frame of another depth");
    // One pixel short of the finest envelope's support: no octave fits, so the finest one's range stands in and stays finite.
    const float inner = pathtracer::debug::innerScaleVariance();
    const int support = (2 * (static_cast<int>(pathtracer::debug::discreteGaussianKernel(inner).size()) - 1)) + 1;
    const HdrImage tiny = pathtracer::gfx::makeImage(support - 1, support - 1, pathtracer::gfx::kScalarChannels);
    const auto finest = static_cast<float>(std::numbers::pi / static_cast<double>(pathtracer::debug::morletCarrier(inner)));
    PT_EXPECT(ctx, pathtracer::debug::opticFlowPriorStd(support - 1, support - 1) == finest, "a frame below every octave");
    PT_EXPECT(ctx, readsPrior(flowBetween(tiny, &tiny, pool), finest), "a pair below every octave");
}

// The prior's spread is the coarsest fitting octave's half carrier wavelength, the widest displacement a phase difference can tell.
PT_CHECK(prior_is_the_coarsest_unambiguous_range, Fast, Exact) {
    const std::array<std::pair<int, int>, 4> sizes{{{64, 64}, {256, 128}, {kWidth, kHeight}, {2048, 1152}}};
    ctx.plan(static_cast<int>(sizes.size()));
    for (const auto& [width, height] : sizes) {
        const int coarsest = pathtracer::debug::octaveLevelCount(width, height) - 1;
        const float variance = std::ldexp(pathtracer::debug::innerScaleVariance(), 2 * coarsest);
        const auto expected = static_cast<float>(std::numbers::pi / static_cast<double>(pathtracer::debug::morletCarrier(variance)));
        PT_EXPECT(ctx, pathtracer::debug::opticFlowPriorStd(width, height) == expected,
                      std::to_string(width) + "x" + std::to_string(height) + " prior is not pi / omega at octave " +
                          std::to_string(coarsest));
    }
}

// Sub-pixel and integer shifts, both signs: the estimate must beat the prior, and its error be explained by a sigma inside the prior.
PT_CHECK(translations_are_explained_by_their_sigma, Fast, Statistical) {
    const std::array<std::pair<double, double>, 4> shifts{{{0.3, -0.2}, {1.5, 0.7}, {-2.0, 1.0}, {3.0, 0.0}}};
    ctx.plan(3 * static_cast<int>(shifts.size()));
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const std::vector<PlaneWave> waves = waveField(ctx.seed());
    const HdrImage before = translated(waves, 0.0, 0.0);
    const float prior = pathtracer::debug::opticFlowPriorStd(kWidth, kHeight);
    for (const auto& [dx, dy] : shifts) {
        const HdrImage flow = flowBetween(translated(waves, dx, dy), &before, pool);
        const std::string label = "shift (" + std::to_string(dx) + ", " + std::to_string(dy) + ")";
        const double improved = beatsPrior(flow, dx, dy);
        PT_EXPECT(ctx, improved >= kCoverageFloor, label + " beats the prior's zero at " + std::to_string(improved));
        const double covered = coverage(flow, dx, dy);
        PT_EXPECT(ctx, covered >= kCoverageFloor, label + " covered by 3 sigma at " + std::to_string(covered));
        bool bounded = true;
        for (std::size_t pixel = 0; pixel < before.texels.size(); ++pixel) {
            bounded = bounded && flowAt(flow, pixel)[2] <= prior;
        }
        PT_EXPECT(ctx, bounded, label + " reported a sigma wider than the prior");
    }
}

// At an integer shift the converged warp is the shift, so responses match to rounding: every texel the mirror cannot reach, not most.
PT_CHECK(an_integer_translation_converges_at_every_interior_texel, Fast, Exact) {
    const std::array<std::pair<int, int>, 2> shifts{{{3, 0}, {-2, 1}}};
    ctx.plan(static_cast<int>(shifts.size()));
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const std::vector<PlaneWave> waves = waveField(ctx.seed());
    const HdrImage before = translated(waves, 0.0, 0.0);
    // The coarsest envelope's base-grid radius: nearer the edge than it plus the shift, some response reads the mirror instead.
    const int coarsest = pathtracer::debug::octaveLevelCount(kWidth, kHeight) - 1;
    const float coarsestVariance = std::ldexp(pathtracer::debug::innerScaleVariance(), 2 * coarsest);
    const int reach = static_cast<int>(pathtracer::debug::discreteGaussianKernel(coarsestVariance).size()) - 1;
    for (const auto& [dx, dy] : shifts) {
        const HdrImage flow = flowBetween(translated(waves, dx, dy), &before, pool);
        const int margin = reach + std::max(std::abs(dx), std::abs(dy));
        std::size_t outside = 0;
        for (int y = margin; y < kHeight - margin; ++y) {
            for (int x = margin; x < kWidth - margin; ++x) {
                const float* texel = flowAt(flow, (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x));
                const double error = std::hypot(static_cast<double>(texel[0]) - dx, static_cast<double>(texel[1]) - dy);
                outside += error > 3.0 * static_cast<double>(texel[2]) ? 1U : 0U;
            }
        }
        PT_EXPECT(ctx, outside == 0, "shift (" + std::to_string(dx) + ", " + std::to_string(dy) + ") left " + std::to_string(outside) +
                                         " interior texels outside 3 sigma");
    }
}

// Within the coarsest envelope's reach of an edge the responses read the mirror's stand-in for unseen content: sigma must still cover it.
PT_CHECK(the_border_band_is_explained_by_its_sigma, Fast, Statistical) {
    const std::array<std::pair<double, double>, 2> shifts{{{1.5, 0.7}, {-2.0, 1.0}}};
    ctx.plan(static_cast<int>(shifts.size()));
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const std::vector<PlaneWave> waves = waveField(ctx.seed());
    const HdrImage before = translated(waves, 0.0, 0.0);
    const int coarsest = pathtracer::debug::octaveLevelCount(kWidth, kHeight) - 1;
    const float coarsestVariance = std::ldexp(pathtracer::debug::innerScaleVariance(), 2 * coarsest);
    const int reach = static_cast<int>(pathtracer::debug::discreteGaussianKernel(coarsestVariance).size()) - 1;
    for (const auto& [dx, dy] : shifts) {
        const HdrImage flow = flowBetween(translated(waves, dx, dy), &before, pool);
        std::size_t band = 0;
        std::size_t covered = 0;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                if (std::min({x, y, kWidth - 1 - x, kHeight - 1 - y}) >= reach) {
                    continue;
                }
                const float* texel = flowAt(flow, (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x));
                ++band;
                covered += std::hypot(static_cast<double>(texel[0]) - dx, static_cast<double>(texel[1]) - dy) <=
                                   3.0 * static_cast<double>(texel[2])
                               ? 1U
                               : 0U;
            }
        }
        const double fraction = static_cast<double>(covered) / static_cast<double>(band);
        PT_EXPECT(ctx, fraction >= kCoverageFloor, "border band of shift (" + std::to_string(dx) + ", " + std::to_string(dy) +
                                                       ") covered by 3 sigma at " + std::to_string(fraction));
    }
}

// A textured half and a flat half under independent noise of known variance: where only noise is seen, sigma must stay honest.
PT_CHECK(known_noise_keeps_a_flat_region_uncertain, Fast, Statistical) {
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const std::vector<PlaneWave> waves = waveField(ctx.seed());
    constexpr double kDx = 1.5;
    constexpr double kDy = 0.5;
    // Noise a quarter of the texture's amplitude, so the textured half is well above it and the flat half is noise alone.
    const float deviation = 0.25F / std::sqrt(static_cast<float>(waves.size()));
    std::mt19937_64 rng(ctx.subSeed("noise"));
    std::normal_distribution<float> normal(0.0F, deviation);
    const auto observed = [&](double dx, double dy) {
        HdrImage image = translated(waves, dx, dy);
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                float& texel = image.texels[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)];
                texel = (x < kWidth / 2 ? texel : 1.0F) + normal(rng);
            }
        }
        return image;
    };
    const HdrImage before = observed(0.0, 0.0);
    const HdrImage after = observed(kDx, kDy);
    HdrImage variance = pathtracer::gfx::makeImage(kWidth, kHeight, pathtracer::gfx::kScalarChannels);
    std::fill(variance.texels.begin(), variance.texels.end(), deviation * deviation);
    const HdrImage flow = pathtracer::debug::opticFlowAov({&after, &variance}, {&before, &variance}, pool);
    std::size_t flatCovered = 0;
    std::size_t texturedBetter = 0;
    const std::size_t half = static_cast<std::size_t>(kWidth / 2) * kHeight;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const float* texel = flowAt(flow, (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x));
            const double error = std::hypot(static_cast<double>(texel[0]) - kDx, static_cast<double>(texel[1]) - kDy);
            if (x < kWidth / 2) {
                texturedBetter += error < std::hypot(kDx, kDy) ? 1U : 0U;
            } else {
                flatCovered += error <= 3.0 * static_cast<double>(texel[2]) ? 1U : 0U;
            }
        }
    }
    const double covered = static_cast<double>(flatCovered) / static_cast<double>(half);
    const double improved = static_cast<double>(texturedBetter) / static_cast<double>(half);
    PT_EXPECT(ctx, covered >= kCoverageFloor, "the noise-only half is covered by 3 sigma at " + std::to_string(covered));
    PT_EXPECT(ctx, improved >= kCoverageFloor, "the textured half beats the prior at " + std::to_string(improved));
}

// A grating along x has no phase structure along y: the normal flow is measured and the tangent must not be claimed beyond its sigma.
PT_CHECK(a_grating_measures_only_its_normal_flow, Fast, Statistical) {
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const double carrier = pathtracer::debug::morletCarrier(pathtracer::debug::innerScaleVariance());
    const auto grating = [carrier](double shift) {
        HdrImage image = pathtracer::gfx::makeImage(kWidth, kHeight, pathtracer::gfx::kScalarChannels);
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                image.texels[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)] =
                    static_cast<float>(1.0 + (0.5 * std::cos(carrier * (static_cast<double>(x) - shift))));
            }
        }
        return image;
    };
    constexpr double kShift = 0.7;
    const HdrImage still = grating(0.0);
    const HdrImage flow = flowBetween(grating(kShift), &still, pool);
    std::size_t normal = 0;
    std::size_t tangent = 0;
    const auto pixels = static_cast<std::size_t>(kWidth) * kHeight;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        const float* texel = flowAt(flow, pixel);
        normal += std::fabs(static_cast<double>(texel[0]) - kShift) <= 3.0 * static_cast<double>(texel[2]) ? 1U : 0U;
        tangent += std::fabs(texel[1]) <= 3.0F * texel[2] ? 1U : 0U;
    }
    const double improved = beatsPrior(flow, kShift, 0.0);
    PT_EXPECT(ctx, improved >= kCoverageFloor, "normal flow beats the prior's zero at " + std::to_string(improved));
    PT_EXPECT(ctx, static_cast<double>(normal) >= kCoverageFloor * static_cast<double>(pixels),
                  "normal flow covered at " + std::to_string(static_cast<double>(normal) / static_cast<double>(pixels)));
    PT_EXPECT(ctx, static_cast<double>(tangent) >= kCoverageFloor * static_cast<double>(pixels),
                  "tangential flow covered at " + std::to_string(static_cast<double>(tangent) / static_cast<double>(pixels)));
}

// Every step is homogeneous in the field and a power of two scales floats exactly, so a joint 2^k gain changes no output bit.
PT_CHECK(a_joint_power_of_two_gain_is_bit_identical, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const std::vector<PlaneWave> waves = waveField(ctx.seed());
    HdrImage before = translated(waves, 0.0, 0.0);
    HdrImage after = translated(waves, 1.25, -0.5);
    const HdrImage reference = flowBetween(after, &before, pool);
    for (HdrImage* image : {&before, &after}) {
        std::transform(image->texels.begin(), image->texels.end(), image->texels.begin(), [](float value) { return 4.0F * value; });
    }
    const HdrImage scaled = flowBetween(after, &before, pool);
    PT_EXPECT(ctx, scaled.texels == reference.texels, "a joint gain of 4 changed the flow");
}

// The five orientations are closed under a horizontal mirror, so mirroring both frames must negate dx and keep dy, within both sigmas.
PT_CHECK(a_horizontal_mirror_negates_horizontal_flow, Fast, Statistical) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const std::vector<PlaneWave> waves = waveField(ctx.seed());
    const HdrImage before = translated(waves, 0.0, 0.0);
    const HdrImage after = translated(waves, 1.25, -0.5);
    const auto mirrored = [](const HdrImage& image) {
        HdrImage out = image;
        for (int y = 0; y < kHeight; ++y) {
            std::reverse(out.texels.begin() + (static_cast<std::ptrdiff_t>(y) * kWidth),
                         out.texels.begin() + (static_cast<std::ptrdiff_t>(y + 1) * kWidth));
        }
        return out;
    };
    const HdrImage flow = flowBetween(after, &before, pool);
    const HdrImage mirroredBefore = mirrored(before);
    const HdrImage mirror = flowBetween(mirrored(after), &mirroredBefore, pool);
    std::size_t agreed = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const float* direct = flowAt(flow, (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x));
            const float* reflected = flowAt(mirror, (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(kWidth - 1 - x));
            const double gap = std::hypot(static_cast<double>(direct[0] + reflected[0]), static_cast<double>(direct[1] - reflected[1]));
            agreed += gap <= 3.0 * std::hypot(static_cast<double>(direct[2]), static_cast<double>(reflected[2])) ? 1U : 0U;
        }
    }
    const double fraction = static_cast<double>(agreed) / static_cast<double>(static_cast<std::size_t>(kWidth) * kHeight);
    PT_EXPECT(ctx, fraction >= kCoverageFloor, "mirrored flow agrees within 3 sigma at " + std::to_string(fraction));
}

PT_CHECK_MAIN("flow")
