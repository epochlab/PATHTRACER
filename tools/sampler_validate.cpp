// Correctness gate for sampler.cpp, asserting EXACT combinatorial net properties rather than Monte Carlo tolerances.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "check.h"
#include "stats.h"
#include "pathtracer/debug/power_spectrum.h"
#include "pathtracer/scene/sampler.h"

using pathtracer::scene::Sampler;

namespace {

constexpr std::uint32_t kSeed = 0x9E3779B9U;
// Every check here measures a 2^kM-sample prefix, so the sampler is told that length -- the same value main() uses for kM.
constexpr int kSampleCount = 128;
constexpr int kPixelX = 37;
constexpr int kPixelY = 41;
// Must match sampler.cpp's kMaskSize, the one place it is restated. 2^7 = 128 points, matching profile.json's maxSamples cap.
constexpr int kM = 7;
// A 12-bounce path consumes up to 6 sets per bounce plus 1 at the camera, so set 64+ is reached, where an unpadded sampler degraded.
constexpr int kSetCount = 72;
// Depths spanning the padded range, including the last two sets, where a shuffle running out of distinct scrambles shows first.
constexpr std::array<int, 7> kNetDepths = {0, 1, 2, 7, 31, 64, 71};
constexpr int kMaskSize = 128;
constexpr int kMaskPixels = kMaskSize * kMaskSize;

// Recovers the underlying sequence by subtracting the dither shift: both are multiples of 2^-24, so the difference is exact.
float unshift(float value, int pixelX, int pixelY, int ditherChannel) {
    const float dither = pathtracer::scene::blueNoiseDither(pixelX, pixelY, ditherChannel);
    return value >= dither ? value - dither : (value - dither) + 1.0F;
}

// Draws dimension set `set` exactly as the renderer would -- fresh Sampler per sample, index advancing, seed fixed -- shift removed.
float draw1DAt(int pixelX, int pixelY, int sampleIndex, int set) {
    Sampler sampler(pixelX, pixelY, sampleIndex, kSampleCount, kSeed);
    float value = 0.0F;
    for (int i = 0; i <= set; ++i) {
        value = sampler.next1D();
    }
    return unshift(value, pixelX, pixelY, 2 * set);
}

float draw1D(int sampleIndex, int set) { return draw1DAt(kPixelX, kPixelY, sampleIndex, set); }

// Draws the 2D value of dimension set `set`, with the preceding sets consumed as 1D draws.
glm::vec2 draw2D(int sampleIndex, int set) {
    Sampler sampler(kPixelX, kPixelY, sampleIndex, kSampleCount, kSeed);
    for (int i = 0; i < set; ++i) {
        static_cast<void>(sampler.next1D());
    }
    const glm::vec2 value = sampler.next2D();
    return {unshift(value.x, kPixelX, kPixelY, 2 * set), unshift(value.y, kPixelX, kPixelY, (2 * set) + 1)};
}

std::size_t binOf(float value, int binCount) {
    const auto bin = static_cast<std::size_t>(value * static_cast<float>(binCount));
    return bin >= static_cast<std::size_t>(binCount) ? static_cast<std::size_t>(binCount) - 1 : bin;
}

// (0,m,1)-net: the first 2^m samples put exactly one point in each of the 2^m intervals, which the scramble and shuffle preserve.
PT_CHECK(one_dimensional_net, Fast, Exact) {
    const int m = kM;
    const int setCount = kSetCount;
    const int n = 1 << m;
    int worstSet = -1;
    for (int set = 0; set < setCount && worstSet < 0; ++set) {
        std::vector<int> bins(static_cast<std::size_t>(n), 0);
        for (int i = 0; i < n; ++i) {
            ++bins[binOf(draw1D(i, set), n)];
        }
        for (const int count : bins) {
            if (count != 1) {
                worstSet = set;
                break;
            }
        }
    }
    char detail[128];
    if (worstSet < 0) {
        std::snprintf(detail, sizeof(detail), "%d sets, 2^%d samples each", setCount, m);
    } else {
        std::snprintf(detail, sizeof(detail), "set %d is not a (0,m,1)-net", worstSet);
    }
    ctx.plan(1);
    PT_EXPECT(ctx, worstSet < 0, detail);
}

// Net quality t: the smallest t for which the first 2^m samples form a (t,m,2)-net. t = 0 is perfect; m means no stratification.
int measureNetQuality(int m, int set) {
    const int n = 1 << m;
    std::vector<glm::vec2> points(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        points[static_cast<std::size_t>(i)] = draw2D(i, set);
    }

    for (int t = 0; t <= m; ++t) {
        bool holds = true;
        for (int a = 0; a <= m - t && holds; ++a) {
            const int columns = 1 << a;
            const int rows = 1 << (m - t - a);
            std::vector<int> cells(static_cast<std::size_t>(columns) * static_cast<std::size_t>(rows), 0);
            for (const glm::vec2& p : points) {
                ++cells[(binOf(p.y, rows) * static_cast<std::size_t>(columns)) + binOf(p.x, columns)];
            }
            for (const int count : cells) {
                if (count != (1 << t)) {
                    holds = false;
                    break;
                }
            }
        }
        if (holds) {
            return t;
        }
    }
    return m;
}

// What padding buys: EVERY 2D set is a perfect (0,m,2)-net, where Sobol's (62,63) projection is only (4,m,2), sixteen points per cell.
PT_CHECK(every_set_is_perfect_net, Fast, Exact) {
    const int m = kM;
    const std::array<int, 7>& sets = kNetDepths;
    int worstSet = -1;
    int worstT = 0;
    std::string measured;
    for (const int set : sets) {
        const int t = measureNetQuality(m, set);
        measured += "s" + std::to_string(set) + ":t=" + std::to_string(t) + " ";
        if (t != 0 && worstSet < 0) {
            worstSet = set;
            worstT = t;
        }
    }
    char detail[256];
    if (worstSet < 0) {
        std::snprintf(detail, sizeof(detail), "%s", measured.c_str());
    } else {
        std::snprintf(detail, sizeof(detail), "set %d is only a (%d,m,2)-net", worstSet, worstT);
    }
    ctx.plan(1);
    PT_EXPECT(ctx, worstSet < 0, detail);
}

// The regression test for the defect this sampler replaces: the old white noise left ~N/e (36.8%) of bins empty, a correct sequence none.
PT_CHECK(pass_direction_occupancy, Fast, Exact) {
    const int m = kM;
    const int setCount = kSetCount;
    const int n = 1 << m;
    int worstSet = -1;
    int worstEmpty = 0;
    for (int set = 0; set < setCount; ++set) {
        std::vector<int> bins(static_cast<std::size_t>(n), 0);
        for (int pass = 0; pass < n; ++pass) {
            ++bins[binOf(draw1D(pass, set), n)];
        }
        int empty = 0;
        for (const int count : bins) {
            empty += count == 0 ? 1 : 0;
        }
        if (empty > worstEmpty) {
            worstEmpty = empty;
            worstSet = set;
        }
    }
    char detail[128];
    if (worstSet < 0) {
        std::snprintf(detail, sizeof(detail), "0 empty bins across %d sets", setCount);
    } else {
        std::snprintf(detail, sizeof(detail), "set %d leaves %d/%d bins empty (%.1f%%)", worstSet, worstEmpty, n,
                      100.0 * worstEmpty / n);
    }
    ctx.plan(1);
    PT_EXPECT(ctx, worstSet < 0, detail);
}

// Sobol's index-0 point is all zeros before scrambling, so the dither shift is what must spread the first pass across pixels.
PT_CHECK(index_zero_is_scrambled, Fast, Statistical) {
    constexpr int kPixels = 4096;
    constexpr int kBins = 16;
    std::vector<int> bins(kBins, 0);
    for (int i = 0; i < kPixels; ++i) {
        Sampler sampler(i % 64, i / 64, 0, kSampleCount, kSeed);
        ++bins[binOf(sampler.next1D(), kBins)];
    }
    // Chi-square against uniform over 16 bins: a correct scramble sits near its 15 dof, the all-zeros case at kPixels * 15.
    double chiSquare = 0.0;
    const double expected = static_cast<double>(kPixels) / kBins;
    for (const int count : bins) {
        const double delta = count - expected;
        chiSquare += delta * delta / expected;
    }
    // One-sided upper tail: a chi2 well BELOW its dof is EXPECTED here, a blue-noise mask being more even than independent draws.
    constexpr int kDof = kBins - 1;
    ctx.plan(1);
    const double p = tools::stats::chiSquareUpperTail(chiSquare, kDof);
    char detail[192];
    std::snprintf(detail, sizeof(detail), "chi2 = %.1f over %d dof, p = %.3g vs alpha %.3g (degenerate would be %d)",
                  chiSquare, kDof, p, ctx.alpha(), kPixels * 15);
    PT_EXPECT(ctx, p >= ctx.alpha(), detail);
}

// Padding's other requirement: consecutive sets must be uncorrelated, or a path's successive decisions would move in lockstep.
PT_CHECK(sets_are_decorrelated, Fast, Exact) {
    constexpr int kSamples = 128;
    int identical = 0;
    for (int i = 0; i < kSamples; ++i) {
        identical += draw1D(i, 0) == draw1D(i, 1) ? 1 : 0;
    }
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%d/%d collisions between set 0 and set 1", identical, kSamples);
    ctx.plan(1);
    PT_EXPECT(ctx, identical == 0, detail);
}

// Neighbouring pixels must not draw the same values; the mask being a permutation is what guarantees adjacent pixels differ.
PT_CHECK(pixels_are_decorrelated, Fast, Exact) {
    constexpr int kSamples = 128;
    int identical = 0;
    for (int i = 0; i < kSamples; ++i) {
        Sampler a(kPixelX, kPixelY, i, kSampleCount, kSeed);
        Sampler b(kPixelX + 1, kPixelY, i, kSampleCount, kSeed);
        identical += a.next1D() == b.next1D() ? 1 : 0;
    }
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%d/%d collisions between adjacent pixels", identical, kSamples);
    ctx.plan(1);
    PT_EXPECT(ctx, identical == 0, detail);
}

// The shift must be a property of the pixel ALONE: one drifting per sample index is plain Monte Carlo with stratification thrown away.
PT_CHECK(shift_is_rigid, Fast, Exact) {
    const int setCount = kSetCount;
    constexpr int kSamples = 64;
    int mismatches = 0;
    for (int sample = 0; sample < kSamples; ++sample) {
        for (int set = 0; set < setCount; ++set) {
            mismatches += draw1DAt(kPixelX, kPixelY, sample, set) != draw1DAt(kPixelX + 1, kPixelY + 3, sample, set)
                              ? 1
                              : 0;
        }
    }
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%d/%d draws disagree after unshifting", mismatches, kSamples * setCount);
    ctx.plan(1);
    PT_EXPECT(ctx, mismatches == 0, detail);
}

// Distinct dither channels must be uncorrelated: one shared shift put the error at 199x white noise. Gate |r| < 0.1, about 13 SD.
PT_CHECK(channels_are_decorrelated, Fast, Statistical) {
    const int channelCount = (2 * kSetCount) + 2;
    std::vector<std::vector<double>> fields(static_cast<std::size_t>(channelCount));
    for (int c = 0; c < channelCount; ++c) {
        std::vector<double>& field = fields[static_cast<std::size_t>(c)];
        field.resize(kMaskPixels);
        for (int y = 0; y < kMaskSize; ++y) {
            for (int x = 0; x < kMaskSize; ++x) {
                field[(static_cast<std::size_t>(y) * kMaskSize) + static_cast<std::size_t>(x)] =
                    pathtracer::scene::blueNoiseDither(x, y, c) - 0.5;  // mean-centred: the mask is uniform on [0,1)
            }
        }
    }
    double worst = 0.0;
    int worstA = -1;
    int worstB = -1;
    for (int a = 0; a < channelCount; ++a) {
        for (int b = a + 1; b < channelCount; ++b) {
            double dot = 0.0;
            double normA = 0.0;
            double normB = 0.0;
            for (std::size_t i = 0; i < static_cast<std::size_t>(kMaskPixels); ++i) {
                const double va = fields[static_cast<std::size_t>(a)][i];
                const double vb = fields[static_cast<std::size_t>(b)][i];
                dot += va * vb;
                normA += va * va;
                normB += vb * vb;
            }
            const double r = std::abs(dot / std::sqrt(normA * normB));
            if (r > worst) {
                worst = r;
                worstA = a;
                worstB = b;
            }
        }
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "worst |r| = %.4f between channels %d and %d, over %d channels", worst,
                  worstA, worstB, channelCount);
    ctx.plan(1);
    PT_EXPECT(ctx, worst < 0.1, detail);
}

// The mask must be a permutation of [0, kMaskPixels): every shift used exactly once, recovered exactly rather than rounded.
PT_CHECK(mask_is_permutation, Fast, Exact) {
    std::vector<int> seen(kMaskPixels, 0);
    int bad = 0;
    for (int y = 0; y < kMaskSize; ++y) {
        for (int x = 0; x < kMaskSize; ++x) {
            const float value = pathtracer::scene::blueNoiseDither(x, y, 0);
            const int rank = static_cast<int>((value * static_cast<float>(kMaskPixels)) - 0.5F);
            if (rank < 0 || rank >= kMaskPixels || seen[static_cast<std::size_t>(rank)] != 0) {
                ++bad;
                continue;
            }
            seen[static_cast<std::size_t>(rank)] = 1;
        }
    }
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%d/%d ranks out of range or repeated", bad, kMaskPixels);
    ctx.plan(1);
    PT_EXPECT(ctx, bad == 0, detail);
}

// The mask must be blue noise, measured against the analytic null: a sampled control reproduced the mask's own generator sequence.
PT_CHECK(mask_is_blue_noise, Fast, Statistical) {
    std::vector<double> mask(kMaskPixels);
    for (int y = 0; y < kMaskSize; ++y) {
        for (int x = 0; x < kMaskSize; ++x) {
            mask[(static_cast<std::size_t>(y) * kMaskSize) + static_cast<std::size_t>(x)] =
                pathtracer::scene::blueNoiseDither(x, y, 0);
        }
    }

    // Bands 3 and up are everything below an eighth of Nyquist: the low-frequency error a blue-noise mask exists to suppress.
    constexpr int kLowBand = 3;
    const std::array<double, pathtracer::debug::kSpectrumBands> bands =
        pathtracer::debug::octaveBandPower(mask, kMaskSize, kMaskSize);
    const std::array<double, pathtracer::debug::kSpectrumBands> null =
        pathtracer::debug::whiteNoiseBandShare(kMaskSize, kMaskSize);
    const double measured = std::accumulate(bands.begin() + kLowBand, bands.end(), 0.0) /
                             std::accumulate(bands.begin(), bands.end(), 0.0);
    const double expected = std::accumulate(null.begin() + kLowBand, null.end(), 0.0);

    char detail[160];
    std::snprintf(detail, sizeof(detail), "low band %.5f%% of power vs %.3f%% white-noise null (%.0fx suppressed)",
                  100.0 * measured, 100.0 * expected, expected / measured);
    ctx.plan(1);
    PT_EXPECT(ctx, measured < 0.5 * expected, detail);
}

}  // namespace

PT_CHECK_MAIN("sampler")
