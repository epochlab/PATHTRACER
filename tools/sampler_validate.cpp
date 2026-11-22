// Correctness gate for sampler.cpp, asserting EXACT combinatorial net properties rather than Monte Carlo tolerances.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <numeric>
#include <random>
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
// Every single-pixel check measures a 2^kM-sample prefix, so the sampler is told that length.
constexpr int kM = 7;
constexpr int kSampleCount = 1 << kM;
constexpr int kPixelX = 37;
constexpr int kPixelY = 41;
// A 12-bounce path consumes up to 6 sets per bounce plus 1 at the camera, so set 64+ is reached, where an unpadded sampler degraded.
constexpr int kSetCount = 72;
// Depths spanning the padded range, including the last two sets, where a shuffle running out of distinct scrambles shows first.
constexpr std::array<int, 7> kNetDepths = {0, 1, 2, 7, 31, 64, 71};
// The shipped profile's 64 spp over 4 replicates: the per-replicate count the renderer's blue noise is aimed at.
constexpr int kReplicateSamples = 16;

// Dimension set `set` as the renderer draws it: a fresh Sampler per sample, index advancing, seed fixed, earlier sets consumed as 1D.
float draw1DAt(int pixelX, int pixelY, int sampleIndex, int sampleCount, std::uint32_t seed, int set) {
    Sampler sampler(pixelX, pixelY, sampleIndex, sampleCount, seed);
    for (int i = 0; i < set; ++i) {
        static_cast<void>(sampler.next1D());
    }
    return sampler.next1D();
}

glm::vec2 draw2DAt(int pixelX, int pixelY, int sampleIndex, int sampleCount, std::uint32_t seed, int set) {
    Sampler sampler(pixelX, pixelY, sampleIndex, sampleCount, seed);
    for (int i = 0; i < set; ++i) {
        static_cast<void>(sampler.next1D());
    }
    return sampler.next2D();
}

float draw1D(int sampleIndex, int set) { return draw1DAt(kPixelX, kPixelY, sampleIndex, kSampleCount, kSeed, set); }

std::size_t binOf(float value, int binCount) {
    const auto bin = static_cast<std::size_t>(value * static_cast<float>(binCount));
    return bin >= static_cast<std::size_t>(binCount) ? static_cast<std::size_t>(binCount) - 1 : bin;
}

// Net quality t: the smallest t for which 2^m points form a (t,m,2)-net. t = 0 is perfect; m means no stratification.
int netQuality(const std::vector<glm::vec2>& points, int m) {
    for (int t = 0; t <= m; ++t) {
        bool holds = true;
        for (int a = 0; a <= m - t && holds; ++a) {
            const int columns = 1 << a;
            const int rows = 1 << (m - t - a);
            std::vector<int> cells(static_cast<std::size_t>(columns) * static_cast<std::size_t>(rows), 0);
            for (const glm::vec2& p : points) {
                ++cells[(binOf(p.y, rows) * static_cast<std::size_t>(columns)) + binOf(p.x, columns)];
            }
            holds = std::all_of(cells.begin(), cells.end(), [t](int count) { return count == (1 << t); });
        }
        if (holds) {
            return t;
        }
    }
    return m;
}

// A pixel's first n samples of one set, n a power of two.
std::vector<glm::vec2> pixelPoints(int pixelX, int pixelY, int n, int sampleCount, std::uint32_t seed, int set) {
    std::vector<glm::vec2> points(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        points[static_cast<std::size_t>(i)] = draw2DAt(pixelX, pixelY, i, sampleCount, seed, set);
    }
    return points;
}

// (0,m,1)-net: the first 2^m samples put exactly one point in each of the 2^m intervals, which the scramble and shuffle preserve.
PT_CHECK(one_dimensional_net, Fast, Exact) {
    const int n = kSampleCount;
    int worstSet = -1;
    for (int set = 0; set < kSetCount && worstSet < 0; ++set) {
        std::vector<int> bins(static_cast<std::size_t>(n), 0);
        for (int i = 0; i < n; ++i) {
            ++bins[binOf(draw1D(i, set), n)];
        }
        worstSet = std::all_of(bins.begin(), bins.end(), [](int count) { return count == 1; }) ? -1 : set;
    }
    char detail[128];
    if (worstSet < 0) {
        std::snprintf(detail, sizeof(detail), "%d sets, 2^%d samples each", kSetCount, kM);
    } else {
        std::snprintf(detail, sizeof(detail), "set %d is not a (0,m,1)-net", worstSet);
    }
    ctx.plan(1);
    PT_EXPECT(ctx, worstSet < 0, detail);
}

// What padding buys: EVERY 2D set is a perfect (0,m,2)-net, where Sobol's (62,63) projection is only (4,m,2), sixteen points per cell.
PT_CHECK(every_set_is_perfect_net, Fast, Exact) {
    int worstSet = -1;
    int worstT = 0;
    std::string measured;
    for (const int set : kNetDepths) {
        const int t = netQuality(pixelPoints(kPixelX, kPixelY, kSampleCount, kSampleCount, kSeed, set), kM);
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
    const int n = kSampleCount;
    int worstSet = -1;
    int worstEmpty = 0;
    for (int set = 0; set < kSetCount; ++set) {
        std::vector<int> bins(static_cast<std::size_t>(n), 0);
        for (int pass = 0; pass < n; ++pass) {
            ++bins[binOf(draw1D(pass, set), n)];
        }
        const auto empty = static_cast<int>(std::count(bins.begin(), bins.end(), 0));
        if (empty > worstEmpty) {
            worstEmpty = empty;
            worstSet = set;
        }
    }
    char detail[128];
    if (worstSet < 0) {
        std::snprintf(detail, sizeof(detail), "0 empty bins across %d sets", kSetCount);
    } else {
        std::snprintf(detail, sizeof(detail), "set %d leaves %d/%d bins empty (%.1f%%)", worstSet, worstEmpty, n,
                      100.0 * worstEmpty / n);
    }
    ctx.plan(1);
    PT_EXPECT(ctx, worstSet < 0, detail);
}

// Unbounded, each pixel is its own tile: index 0, Sobol's all-zeros point, must still land independently uniform per pixel.
PT_CHECK(index_zero_is_scrambled, Fast, Statistical) {
    constexpr int kPixels = 4096;
    constexpr int kBins = 16;
    std::vector<int> bins(kBins, 0);
    const auto seed = static_cast<std::uint32_t>(ctx.seed());
    for (int i = 0; i < kPixels; ++i) {
        Sampler sampler(i % 64, i / 64, 0, /*sampleCount=*/0, seed);
        ++bins[binOf(sampler.next1D(), kBins)];
    }
    // Chi-square against uniform over 16 bins: independent pixels sit near its 15 dof, the all-zeros case at kPixels * 15.
    double chiSquare = 0.0;
    const double expected = static_cast<double>(kPixels) / kBins;
    for (const int count : bins) {
        const double delta = count - expected;
        chiSquare += delta * delta / expected;
    }
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
    int identical = 0;
    for (int i = 0; i < kSampleCount; ++i) {
        identical += draw1D(i, 0) == draw1D(i, 1) ? 1 : 0;
    }
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%d/%d collisions between set 0 and set 1", identical, kSampleCount);
    ctx.plan(1);
    PT_EXPECT(ctx, identical == 0, detail);
}

// The closed form against Joe & Kuo's recurrence v_b = v_(b-1) ^ (v_(b-1) >> 1), every 16-bit index and random 32-bit ones.
PT_CHECK(sobol_dimension_one_matches_direction_numbers, Fast, Exact) {
    std::array<std::uint32_t, 32> directions{};
    directions[0] = 1U << 31U;
    for (std::size_t b = 1; b < directions.size(); ++b) {
        directions[b] = directions[b - 1] ^ (directions[b - 1] >> 1U);
    }
    const auto reference = [&directions](std::uint32_t index) {
        std::uint32_t x = 0;
        for (std::size_t b = 0; b < directions.size(); ++b) {
            x ^= ((index >> b) & 1U) != 0U ? directions[b] : 0U;
        }
        return x;
    };
    const auto reverse = [](std::uint32_t x) {
        std::uint32_t r = 0;
        for (int b = 0; b < 32; ++b) {
            r = (r << 1U) | ((x >> static_cast<unsigned>(b)) & 1U);
        }
        return r;
    };
    std::vector<std::uint32_t> indices(1U << 16U);
    std::iota(indices.begin(), indices.end(), 0U);
    std::mt19937 rng(kSeed);
    for (int i = 0; i < (1 << 16); ++i) {
        indices.push_back(static_cast<std::uint32_t>(rng()));
    }
    const auto mismatches = std::count_if(indices.begin(), indices.end(), [&](std::uint32_t index) {
        return reverse(pathtracer::scene::reversedSobolDimension1(index)) != reference(index);
    });
    ctx.plan(1);
    PT_EXPECT(ctx, mismatches == 0, std::to_string(mismatches) + " of " + std::to_string(indices.size()) + " indices differ");
}

// Every power-of-two prefix of every pixel is a (0,m,2)-net, so adaptive checkpoints at K 2^j keep full stratification.
PT_CHECK(pixel_prefixes_are_nets, Fast, Exact) {
    constexpr int kRegion = 16;
    int failures = 0;
    std::string first;
    for (int y = kPixelY; y < kPixelY + kRegion; ++y) {
        for (int x = kPixelX; x < kPixelX + kRegion; ++x) {
            for (const int set : kNetDepths) {
                const std::vector<glm::vec2> points = pixelPoints(x, y, kSampleCount, kSampleCount, kSeed, set);
                for (int m = 0; m <= kM; ++m) {
                    const std::vector<glm::vec2> prefix(points.begin(), points.begin() + (1 << m));
                    if (netQuality(prefix, m) != 0) {
                        first = first.empty() ? "(" + std::to_string(x) + "," + std::to_string(y) + ") set " + std::to_string(set) +
                                                    " prefix 2^" + std::to_string(m)
                                              : first;
                        ++failures;
                    }
                }
            }
        }
    }
    ctx.plan(1);
    PT_EXPECT(ctx, failures == 0, failures == 0 ? "every prefix of 256 pixels at 7 depths" : std::to_string(failures) + " fail, first " + first);
}

// Ahmed & Wonka's property: an aligned 2^k x 2^k pixel square holds one aligned block of the sequence, a (0, 2k+b, 2)-net.
PT_CHECK(aligned_pixel_blocks_are_nets, Fast, Exact) {
    constexpr int kOrigin = 32;  // a multiple of every block side tested
    constexpr int kRegion = 16;
    const int sampleBits = std::countr_zero(static_cast<unsigned>(kReplicateSamples));
    int failures = 0;
    int blocks = 0;
    for (int k = 1; k <= 3; ++k) {
        const int side = 1 << k;
        for (const int set : kNetDepths) {
            for (int by = kOrigin; by < kOrigin + kRegion; by += side) {
                for (int bx = kOrigin; bx < kOrigin + kRegion; bx += side) {
                    std::vector<glm::vec2> points;
                    for (int y = by; y < by + side; ++y) {
                        for (int x = bx; x < bx + side; ++x) {
                            const std::vector<glm::vec2> pixel = pixelPoints(x, y, kReplicateSamples, kReplicateSamples, kSeed, set);
                            points.insert(points.end(), pixel.begin(), pixel.end());
                        }
                    }
                    failures += netQuality(points, (2 * k) + sampleBits) != 0 ? 1 : 0;
                    ++blocks;
                }
            }
        }
    }
    ctx.plan(1);
    PT_EXPECT(ctx, failures == 0, std::to_string(failures) + " of " + std::to_string(blocks) + " aligned squares are not perfect nets");
}

// Which pixel pair of a 2x2 square shares a half-block: |, - or x. The base-4 scramble must pick each a third of the time.
PT_CHECK(morton_pairing_is_isotropic, Fast, Statistical) {
    constexpr int kSquares = 64;  // per axis, so 4096 squares
    const auto seed = static_cast<std::uint32_t>(ctx.seed());
    const int sampleBits = std::countr_zero(static_cast<unsigned>(kReplicateSamples));
    // Pixel offsets of the square's two pairs under each pairing: vertical, horizontal, diagonal.
    constexpr std::array<std::array<glm::ivec2, 4>, 3> kPairings = {{
        {{{0, 0}, {0, 1}, {1, 0}, {1, 1}}},
        {{{0, 0}, {1, 0}, {0, 1}, {1, 1}}},
        {{{0, 0}, {1, 1}, {1, 0}, {0, 1}}},
    }};
    std::array<int, 3> counts{};
    int ambiguous = 0;
    for (int sy = 0; sy < kSquares; ++sy) {
        for (int sx = 0; sx < kSquares; ++sx) {
            int found = -1;
            for (std::size_t pairing = 0; pairing < kPairings.size(); ++pairing) {
                std::vector<glm::vec2> pair;
                for (int member = 0; member < 2; ++member) {
                    const glm::ivec2 at = (2 * glm::ivec2(sx, sy)) + kPairings[pairing][static_cast<std::size_t>(member)];
                    const std::vector<glm::vec2> pixel = pixelPoints(at.x, at.y, kReplicateSamples, kReplicateSamples, seed, 0);
                    pair.insert(pair.end(), pixel.begin(), pixel.end());
                }
                if (netQuality(pair, sampleBits + 1) == 0) {
                    found = found < 0 ? static_cast<int>(pairing) : 3;
                }
            }
            if (found >= 0 && found < 3) {
                ++counts[static_cast<std::size_t>(found)];
            } else {
                ++ambiguous;
            }
        }
    }
    const double expected = (kSquares * kSquares - ambiguous) / 3.0;
    double chiSquare = 0.0;
    for (const int count : counts) {
        chiSquare += (count - expected) * (count - expected) / expected;
    }
    ctx.plan(2);
    PT_EXPECT(ctx, ambiguous == 0, std::to_string(ambiguous) + " squares have no unique sibling pairing");
    const double p = tools::stats::chiSquareUpperTail(chiSquare, 2);
    char detail[192];
    std::snprintf(detail, sizeof(detail), "| %d  - %d  x %d, chi2 = %.2f over 2 dof, p = %.3g vs alpha %.3g", counts[0], counts[1], counts[2],
                  chiSquare, p, ctx.alpha());
    std::printf("  %s\n", detail);
    PT_EXPECT(ctx, p >= ctx.alpha(), detail);
}

// 2^20 samples leave 6-bit tiles: pixels astride a seam, or a tile apart, draw distinct sequences and each stays a perfect net.
PT_CHECK(tiles_fit_the_index, Fast, Exact) {
    constexpr int kLargeCount = 1 << 20;
    constexpr int kTileSide = 64;  // (32 - 20) / 2 bits
    const std::array<glm::ivec2, 4> pixels = {{{kTileSide - 1, 5}, {kTileSide, 5}, {(2 * kTileSide) - 1, 5}, {kTileSide - 1, kTileSide + 5}}};
    int collisions = 0;
    int notNets = 0;
    std::vector<std::vector<glm::vec2>> sets;
    for (const glm::ivec2& pixel : pixels) {
        sets.push_back(pixelPoints(pixel.x, pixel.y, kSampleCount, kLargeCount, kSeed, 0));
        notNets += netQuality(sets.back(), kM) != 0 ? 1 : 0;
    }
    for (std::size_t a = 0; a < sets.size(); ++a) {
        for (std::size_t b = a + 1; b < sets.size(); ++b) {
            for (int i = 0; i < kSampleCount; ++i) {
                collisions += sets[a][static_cast<std::size_t>(i)] == sets[b][static_cast<std::size_t>(i)] ? 1 : 0;
            }
        }
    }
    ctx.plan(2);
    PT_EXPECT(ctx, notNets == 0, std::to_string(notNets) + " pixels near tile seams are not perfect nets");
    PT_EXPECT(ctx, collisions == 0, std::to_string(collisions) + " equal draws between pixels in different tiles");
}

// Screen-space error of random oriented Heavisides (Heitz et al. 2019) must be improbably low-band for white noise at 1 and 16 spp.
PT_CHECK(error_is_blue_noise, Fast, Statistical) {
    constexpr int kSize = 128;
    constexpr int kPixels = kSize * kSize;
    constexpr int kIntegrands = 4;
    constexpr std::array<int, 4> kDepths = {0, 7, 31, 64};
    constexpr std::array<int, 2> kCounts = {1, kReplicateSamples};
    // Bands 3 and up are everything below an eighth of Nyquist: the low-frequency error blue noise exists to suppress.
    constexpr int kLowBand = 3;
    const auto seed = static_cast<std::uint32_t>(ctx.seed());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const std::array<double, pathtracer::debug::kSpectrumBands> null = pathtracer::debug::whiteNoiseBandShare(kSize, kSize);
    const double expected = std::accumulate(null.begin() + kLowBand, null.end(), 0.0);
    // The white null's share is its frequency count over the field's kPixels - 1 nonzero ones.
    const int lowFrequencies = static_cast<int>(std::lround(expected * (kPixels - 1)));
    double worst = 0.0;
    std::vector<double> error(static_cast<std::size_t>(kPixels));
    for (const int count : kCounts) {
        for (const int set : kDepths) {
            for (int integrand = 0; integrand < kIntegrands; ++integrand) {
                const double angle = 2.0 * std::numbers::pi * uniform(rng);
                const glm::dvec2 normal(std::cos(angle), std::sin(angle));
                const glm::dvec2 centre(uniform(rng), uniform(rng));
                for (int y = 0; y < kSize; ++y) {
                    for (int x = 0; x < kSize; ++x) {
                        double sum = 0.0;
                        for (int i = 0; i < count; ++i) {
                            sum += glm::dot(glm::dvec2(draw2DAt(x, y, i, count, seed, set)) - centre, normal) > 0.0 ? 1.0 : 0.0;
                        }
                        // The integral is a constant over pixels, which the DC term absorbs, so the estimate stands in for its error.
                        error[(static_cast<std::size_t>(y) * kSize) + static_cast<std::size_t>(x)] = sum / count;
                    }
                }
                const std::array<double, pathtracer::debug::kSpectrumBands> bands = pathtracer::debug::octaveBandPower(error, kSize, kSize);
                const double total = std::accumulate(bands.begin(), bands.end(), 0.0);
                // An edge missing every pixel's points leaves no error at all, nothing to test; its share would be 0/0.
                if (total > 0.0) {
                    worst = std::max(worst, std::accumulate(bands.begin() + kLowBand, bands.end(), 0.0) / total);
                }
            }
        }
    }
    // Every field must reject white noise, an intersection of rejections, so each holds the whole alpha uncorrected (Berger 1982).
    ctx.plan(1);
    const double p = tools::stats::whiteNoiseShareLowerTail(worst, lowFrequencies, kPixels);
    char detail[192];
    std::snprintf(detail, sizeof(detail), "worst low band %.4f%% of power vs %.3f%% white-noise null (%.1fx suppressed), P(white this low) %.3g",
                  100.0 * worst, 100.0 * expected, expected / worst, p);
    std::printf("  %s\n", detail);
    PT_EXPECT(ctx, p <= ctx.alpha(), detail);
}

// A stream shares its pixel's block under its own scramble: still a perfect net at every depth, and no draw equal to its parent's.
PT_CHECK(streams_are_distinct_nets, Fast, Exact) {
    constexpr std::uint32_t kStreamOffset = 0x9E3779B9U;
    int notNets = 0;
    int collisions = 0;
    for (const int set : kNetDepths) {
        std::vector<glm::vec2> points(static_cast<std::size_t>(kSampleCount));
        for (int i = 0; i < kSampleCount; ++i) {
            const Sampler parent(kPixelX, kPixelY, i, kSampleCount, kSeed);
            Sampler stream = parent.stream(kStreamOffset);
            for (int k = 0; k < set; ++k) {
                static_cast<void>(stream.next1D());
            }
            points[static_cast<std::size_t>(i)] = stream.next2D();
            collisions += points[static_cast<std::size_t>(i)] == draw2DAt(kPixelX, kPixelY, i, kSampleCount, kSeed, set) ? 1 : 0;
        }
        notNets += netQuality(points, kM) != 0 ? 1 : 0;
    }
    ctx.plan(2);
    PT_EXPECT(ctx, notNets == 0, std::to_string(notNets) + " stream sets are not perfect nets");
    PT_EXPECT(ctx, collisions == 0, std::to_string(collisions) + " stream draws equal their parent's");
}

// Replicate 0 is the plain sequence and replicate seeds differ by >= 2^16, so seeds nearer than that never alias two replicates' streams.
PT_CHECK(replicate_seeds_are_identity_then_far_apart, Fast, Exact) {
    constexpr int kReplicates = 64;
    constexpr std::uint32_t kNearSeeds = 1U << 16U;
    ctx.plan(2);
    PT_EXPECT(ctx, pathtracer::scene::replicateSeed(kSeed, 0) == kSeed, "replicate 0 changed the scramble seed");
    std::uint32_t closest = 0xFFFFFFFFU;
    for (int a = 0; a < kReplicates; ++a) {
        for (int b = a + 1; b < kReplicates; ++b) {
            closest = std::min(closest, pathtracer::scene::replicateSeed(0U, a) ^ pathtracer::scene::replicateSeed(0U, b));
        }
    }
    char detail[128];
    std::snprintf(detail, sizeof(detail), "two of %d replicate seeds differ by XOR 0x%08X, under 2^16", kReplicates, closest);
    PT_EXPECT(ctx, closest >= kNearSeeds, detail);
}

// invertCdf's search is upper_bound's bin, bit for bit: zero-mass bins, ties, every knot and its neighbours, 0, below 1 and past 1.
PT_CHECK(invert_cdf_matches_upper_bound, Fast, Exact) {
    constexpr int kMaxBins = 80;  // past 64, so every power-of-two search length and its neighbours occur
    std::mt19937 rng(kSeed);
    std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
    int mismatches = 0;
    int queries = 0;
    for (int count = 1; count <= kMaxBins; ++count) {
        std::vector<float> cdf(static_cast<std::size_t>(count) + 1, 0.0F);
        for (int b = 1; b <= count; ++b) {
            const float mass = uniform(rng) < 0.3F ? 0.0F : uniform(rng);
            cdf[static_cast<std::size_t>(b)] = cdf[static_cast<std::size_t>(b) - 1] + mass;
        }
        const float total = cdf.back();
        for (float& knot : cdf) {
            knot = total > 0.0F ? knot / total : 0.0F;
        }
        cdf.back() = 1.0F;
        std::vector<float> us = {0.0F, std::nextafter(1.0F, 0.0F), 1.0F, 1.5F};
        for (const float knot : cdf) {
            us.insert(us.end(), {knot, std::nextafter(knot, 0.0F), std::nextafter(knot, 2.0F)});
        }
        for (int i = 0; i < 64; ++i) {
            us.push_back(uniform(rng));
        }
        for (const float u : us) {
            const int index = std::clamp(static_cast<int>(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin()) - 1, 0, count - 1);
            const float lo = cdf[static_cast<std::size_t>(index)];
            const float hi = cdf[static_cast<std::size_t>(index) + 1];
            const float fraction = hi > lo ? std::clamp((u - lo) / (hi - lo), 0.0F, 1.0F) : 0.5F;
            const pathtracer::scene::CdfSample got = pathtracer::scene::invertCdf(cdf.data(), count, u);
            mismatches += got.index != index || got.fraction != fraction ? 1 : 0;
            ++queries;
        }
    }
    ctx.plan(1);
    PT_EXPECT(ctx, mismatches == 0,
              std::to_string(mismatches) + " of " + std::to_string(queries) + " inversions differ from upper_bound's bin");
}

}  // namespace

PT_CHECK_MAIN("sampler")
