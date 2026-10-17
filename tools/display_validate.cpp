// Correctness gate for the CPU-evaluable display and statistics path: the OCIO display transform, lens optics and frame-time ring buffer.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "check.h"
#include "pathtracer/debug/frame_stats.h"
#include "pathtracer/debug/histogram_curve.h"
#include "pathtracer/debug/power_spectrum.h"
#include "pathtracer/gfx/display_optics.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/pixel_lattice.h"
#include "pathtracer/gfx/ocio_cpu_transform.h"
#include "pathtracer/gfx/viewport.h"
#include "pathtracer/scene/sampler.h"
#include "pathtracer/scene/thread_pool.h"

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

// --- Display optics: the CPU path against the viewer's shaders, transcribed here in double precision with GL's own sampling rules.

using pathtracer::gfx::BloomTap;
using pathtracer::gfx::DisplayOptics;
using pathtracer::gfx::HdrImage;
using pathtracer::gfx::PixelLattice;

HdrImage randomImage(int width, int height, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> radiance(0.0F, 4.0F);
    HdrImage image = pathtracer::gfx::makeImage(width, height, 3);
    std::generate(image.texels.begin(), image.texels.end(), [&] { return radiance(rng); });
    return image;
}

struct ShaderImage {
    int width;
    int height;
    std::vector<glm::dvec3> rgb;
};

// GL_LINEAR at (u, v): x = u * w - 1/2, each of the 2x2 indices addressed after the floor; S repeats or clamps, T clamps.
glm::dvec3 glLinear(const ShaderImage& image, double u, double v, bool wrapsS) {
    const auto address = [](int index, int size, bool wraps) {
        return wraps ? (((index % size) + size) % size) : std::clamp(index, 0, size - 1);
    };
    const double x = (u * image.width) - 0.5;
    const double y = (v * image.height) - 0.5;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const double fx = x - x0;
    const double fy = y - y0;
    const auto at = [&](int i, int j) {
        return image.rgb[(static_cast<std::size_t>(address(j, image.height, false)) * static_cast<std::size_t>(image.width)) +
                         static_cast<std::size_t>(address(i, image.width, wrapsS))];
    };
    return ((1.0 - fx) * (1.0 - fy) * at(x0, y0)) + (fx * (1.0 - fy) * at(x0 + 1, y0)) + ((1.0 - fx) * fy * at(x0, y0 + 1)) +
           (fx * fy * at(x0 + 1, y0 + 1));
}

// One bloom shader pass: target texel (i, j) sums its taps at uv = (i + 1/2, j + 1/2) / target + offset / source.
ShaderImage shaderPass(const ShaderImage& source, int width, int height, std::span<const BloomTap> taps, bool wrapsS) {
    ShaderImage target{width, height, std::vector<glm::dvec3>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height))};
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            glm::dvec3 sum(0.0);
            for (const BloomTap& tap : taps) {
                const double u = ((i + 0.5) / width) + (static_cast<double>(tap.dx) / source.width);
                const double v = ((j + 0.5) / height) + (static_cast<double>(tap.dy) / source.height);
                sum += static_cast<double>(tap.weight) * glLinear(source, u, v, wrapsS);
            }
            target.rgb[(static_cast<std::size_t>(j) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(i)] = sum;
        }
    }
    return target;
}

// BloomPass::build then the display shader's mix(I, bloom, k): separate down and up chains, each up adding its down octave, the last 1/L.
ShaderImage shaderBloom(const HdrImage& image, double k, bool wraps) {
    ShaderImage input{image.width, image.height, {}};
    for (std::size_t i = 0; i < image.texels.size(); i += 3) {
        input.rgb.emplace_back(image.texels[i], image.texels[i + 1], image.texels[i + 2]);
    }
    const int octaves = pathtracer::gfx::bloomOctaves(image.width, image.height);
    std::vector<ShaderImage> down;
    const ShaderImage* source = &input;
    for (int octave = 1; octave <= octaves; ++octave) {
        down.push_back(shaderPass(*source, std::max(1, image.width >> octave), std::max(1, image.height >> octave),
                                  pathtracer::gfx::kBloomDownTaps, wraps));
        source = &down.back();
    }
    ShaderImage up = down.back();
    for (int octave = octaves - 1; octave >= 0; --octave) {
        ShaderImage next = shaderPass(up, std::max(1, image.width >> octave), std::max(1, image.height >> octave),
                                      pathtracer::gfx::kBloomUpTaps, wraps);
        for (std::size_t i = 0; i < next.rgb.size(); ++i) {
            next.rgb[i] = octave > 0 ? next.rgb[i] + down[static_cast<std::size_t>(octave - 1)].rgb[i] : next.rgb[i] / double(octaves);
        }
        up = std::move(next);
    }
    for (std::size_t i = 0; i < up.rgb.size(); ++i) {
        up.rgb[i] = ((1.0 - k) * input.rgb[i]) + (k * up.rgb[i]);
    }
    return up;
}

// Bloom against the shader transcription, odd and degenerate sizes, both seam modes; float error is bounded by the pass depth.
PT_CHECK(bloom_matches_the_shader_passes, Fast, Exact) {
    struct Case {
        int width;
        int height;
        bool wraps;
    };
    constexpr std::array<Case, 4> kCases{{{37, 23, false}, {64, 15, true}, {3, 1, false}, {1, 1, true}}};
    constexpr float kStrength = 0.6F;
    ctx.plan(static_cast<int>(kCases.size()));
    pathtracer::scene::ThreadPool pool;
    for (const Case& testCase : kCases) {
        HdrImage image = randomImage(testCase.width, testCase.height, ctx.seed());
        const ShaderImage reference = shaderBloom(image, kStrength, testCase.wraps);
        pathtracer::gfx::applyDisplayOptics(image, PixelLattice::Square, testCase.wraps, DisplayOptics{.bloomStrength = kStrength}, pool);
        double worst = 0.0;
        for (std::size_t i = 0; i < reference.rgb.size(); ++i) {
            for (glm::length_t lane = 0; lane < 3; ++lane) {
                worst = std::max(worst, std::fabs(image.texels[(i * 3) + static_cast<std::size_t>(lane)] - reference.rgb[i][lane]));
            }
        }
        // 2L + 1 dependent passes of at most 3 roundings per tap plus the sum, on intermediates no larger than L times the input's 4.
        const int octaves = pathtracer::gfx::bloomOctaves(testCase.width, testCase.height);
        const double bound = (2.0 * octaves + 1.0) * ((3.0 * pathtracer::gfx::kBloomDownTaps.size()) + 1.0) *
                             std::numeric_limits<float>::epsilon() * octaves * 4.0;
        char detail[160];
        std::snprintf(detail, sizeof(detail), "%dx%d %s: worst |CPU - shader| %.3g against the float bound %.3g", testCase.width,
                      testCase.height, testCase.wraps ? "wrapped" : "clamped", worst, bound);
        PT_EXPECT(ctx, worst <= bound, detail);
    }
}

// A lat-long rolled by 2^L columns rolls every octave by whole texels, so its bloom must roll with it, bit for bit, across the seam.
PT_CHECK(bloom_wraps_a_lat_long_seam, Fast, Exact) {
    constexpr int kWidth = 64;
    constexpr int kHeight = 16;
    const int roll = 1 << pathtracer::gfx::bloomOctaves(kWidth, kHeight);
    ctx.plan(1);
    pathtracer::scene::ThreadPool pool;
    HdrImage image = randomImage(kWidth, kHeight, ctx.seed());
    HdrImage rolled = image;
    for (int y = 0; y < kHeight; ++y) {
        const auto row = image.texels.begin() + (static_cast<std::ptrdiff_t>(y) * kWidth * 3);
        std::rotate_copy(row, row + (static_cast<std::ptrdiff_t>(kWidth - roll) * 3), row + (static_cast<std::ptrdiff_t>(kWidth) * 3),
                         rolled.texels.begin() + (static_cast<std::ptrdiff_t>(y) * kWidth * 3));
    }
    const DisplayOptics optics{.bloomStrength = 1.0F};
    pathtracer::gfx::applyDisplayOptics(image, PixelLattice::Square, true, optics, pool);
    pathtracer::gfx::applyDisplayOptics(rolled, PixelLattice::Square, true, optics, pool);
    std::size_t differing = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            for (int lane = 0; lane < 3; ++lane) {
                const std::size_t row = static_cast<std::size_t>(y) * kWidth;
                const std::size_t at = ((row + static_cast<std::size_t>(x)) * 3) + static_cast<std::size_t>(lane);
                const std::size_t shifted = ((row + static_cast<std::size_t>((x + roll) % kWidth)) * 3) + static_cast<std::size_t>(lane);
                differing += image.texels[at] != rolled.texels[shifted] ? 1 : 0;
            }
        }
    }
    PT_EXPECT(ctx, differing == 0, std::to_string(differing) + " floats differ after rolling by " + std::to_string(roll) + " columns");
}

// The six axial unit steps of the hexagonal lattice: the smoothing step S's ring and the edges linear prolongation P halves.
constexpr std::array<glm::ivec2, 6> kHexagonalRing{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, -1}, {-1, 1}}};

// floor(a / 2) for either sign, as the shaders write it.
int halfFloor(int a) {
    return (a - (((a % 2) + 2) % 2)) / 2;
}

// Offset cell (x, y) of axial (q, r), and back: the lattice basis in which the aperture-4 coarse lattice is axial / 2.
glm::ivec2 offsetOfAxial(glm::ivec2 axial) {
    return {axial.x + halfFloor(axial.y), axial.y};
}

glm::ivec2 axialOfOffset(int x, int y) {
    return {x - halfFloor(y), y};
}

// A level's value at any offset cell, folded by the lattice's mirrors or wrapped, as every hexagonal pass fetches it.
glm::dvec3 foldedAt(const ShaderImage& level, glm::ivec2 cell, bool wraps) {
    const glm::ivec2 folded = pathtracer::gfx::hexagonalReflection(cell.x, cell.y, level.width, level.height, wraps);
    return level.rgb[(static_cast<std::size_t>(folded.y) * static_cast<std::size_t>(level.width)) + static_cast<std::size_t>(folded.x)];
}

// S: the centre at 1/4 and its six neighbours at 1/8, folded at this level's own frame.
ShaderImage hexagonalSmooth(const ShaderImage& level, bool wraps) {
    ShaderImage out{level.width, level.height, std::vector<glm::dvec3>(level.rgb.size())};
    for (int y = 0; y < level.height; ++y) {
        for (int x = 0; x < level.width; ++x) {
            const glm::ivec2 axial = axialOfOffset(x, y);
            glm::dvec3 sum = 0.25 * foldedAt(level, {x, y}, wraps);
            for (const glm::ivec2 step : kHexagonalRing) {
                sum += 0.125 * foldedAt(level, offsetOfAxial(axial + step), wraps);
            }
            out.rgb[(static_cast<std::size_t>(y) * static_cast<std::size_t>(level.width)) + static_cast<std::size_t>(x)] = sum;
        }
    }
    return out;
}

// P at any fine cell: a coarse vertex copies its value, an edge midpoint averages its two ends; coarse cells fold at the coarse frame.
glm::dvec3 hexagonalProlong(const ShaderImage& coarse, glm::ivec2 fineAxial, bool wraps) {
    const auto vertex = [&](glm::ivec2 axial) { return foldedAt(coarse, offsetOfAxial(axial / 2), wraps); };
    const glm::ivec2 parity((fineAxial.x % 2 + 2) % 2, (fineAxial.y % 2 + 2) % 2);
    if (parity == glm::ivec2(0)) {
        return vertex(fineAxial);
    }
    // The one unit step with this parity runs along the edge: (1,0), (0,1) or (1,-1).
    const glm::ivec2 along = parity == glm::ivec2(1, 1) ? glm::ivec2(1, -1) : parity;
    return 0.5 * (vertex(fineAxial - along) + vertex(fineAxial + along));
}

// Loop subdivision factorised, S after P: P is evaluated on S's whole stencil, so only the coarse frame folds.
ShaderImage hexagonalUpReference(const ShaderImage& coarse, int width, int height, bool wraps) {
    ShaderImage out{width, height, std::vector<glm::dvec3>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height))};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const glm::ivec2 axial = axialOfOffset(x, y);
            glm::dvec3 sum = 0.25 * hexagonalProlong(coarse, axial, wraps);
            for (const glm::ivec2 step : kHexagonalRing) {
                sum += 0.125 * hexagonalProlong(coarse, axial + step, wraps);
            }
            out.rgb[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x)] = sum;
        }
    }
    return out;
}

// Its adjoint over 4, factorised: S twice on the fine frame, the fold a lattice symmetry each commutes with, sampled at coarse vertices.
ShaderImage hexagonalDownReference(const ShaderImage& fine, int width, int height, bool wraps) {
    const ShaderImage smoothed = hexagonalSmooth(hexagonalSmooth(fine, wraps), wraps);
    ShaderImage out{width, height, std::vector<glm::dvec3>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height))};
    for (int m = 0; m < height; ++m) {
        for (int j = 0; j < width; ++j) {
            out.rgb[(static_cast<std::size_t>(m) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(j)] =
                foldedAt(smoothed, {(2 * j) + (m & 1), 2 * m}, wraps);
        }
    }
    return out;
}

// The hexagonal pyramid with the display mix, from the factorised operators: the oracle the shipped 19-entry mask must equal.
ShaderImage hexagonalBloomReference(const HdrImage& image, double k, bool wraps) {
    ShaderImage input{image.width, image.height, {}};
    for (std::size_t i = 0; i < image.texels.size(); i += 3) {
        input.rgb.emplace_back(image.texels[i], image.texels[i + 1], image.texels[i + 2]);
    }
    const int octaves = pathtracer::gfx::bloomOctaves(image.width, image.height);
    const auto size = [&](int octave) { return pathtracer::gfx::bloomLevelSize(PixelLattice::Hexagonal, image.width, image.height, octave, wraps); };
    std::vector<ShaderImage> down;
    const ShaderImage* source = &input;
    for (int octave = 1; octave <= octaves; ++octave) {
        down.push_back(hexagonalDownReference(*source, size(octave).x, size(octave).y, wraps));
        source = &down.back();
    }
    ShaderImage up = down.back();
    for (int octave = octaves - 1; octave >= 0; --octave) {
        ShaderImage next = hexagonalUpReference(up, size(octave).x, size(octave).y, wraps);
        for (std::size_t i = 0; i < next.rgb.size(); ++i) {
            next.rgb[i] = octave > 0 ? next.rgb[i] + down[static_cast<std::size_t>(octave - 1)].rgb[i] : next.rgb[i] / double(octaves);
        }
        up = std::move(next);
    }
    for (std::size_t i = 0; i < up.rgb.size(); ++i) {
        up.rgb[i] = ((1.0 - k) * input.rgb[i]) + (k * up.rgb[i]);
    }
    return up;
}

// Each hexagonal level is the least whose far mirror lines lie on or past its finer level's, so every octave holds the whole frame.
PT_CHECK(hexagonal_bloom_levels_cover_the_frame, Fast, Exact) {
    ctx.plan(1);
    int failures = 0;
    for (int width = 1; width <= 70; ++width) {
        for (int height = 1; height <= 70; ++height) {
            for (const bool wraps : {false, true}) {
                glm::ivec2 finer(width, height);
                for (int octave = 1; octave <= pathtracer::gfx::bloomOctaves(width, height); ++octave) {
                    const glm::ivec2 level = pathtracer::gfx::bloomLevelSize(PixelLattice::Hexagonal, width, height, octave, wraps);
                    // Coarse row m is finer row 2m, so the far mirror row is 2(h' - 1); one row fewer would fall short of h - 1.
                    const bool rows = 2 * (level.y - 1) >= finer.y - 1 && 2 * (level.y - 2) < finer.y - 1;
                    // Half-pitch column index: coarse 2w' - 1 is finer 4w' - 2; a wrapped width halves its period exactly.
                    const bool columns =
                        wraps ? level.x == (finer.x + 1) / 2 : (4 * level.x) - 2 >= (2 * finer.x) - 1 && (4 * level.x) - 6 < (2 * finer.x) - 1;
                    failures += rows && columns ? 0 : 1;
                    finer = level;
                }
            }
        }
    }
    PT_EXPECT(ctx, failures == 0, std::to_string(failures) + " levels fall short of, or needlessly exceed, their finer level's frame");
}

// The shipped mask is Loop's: six-fold symmetric, unit weight per coset, and over 4 exactly S * S, the 7-tap smoothing composed twice.
PT_CHECK(hexagonal_bloom_mask_is_loops, Fast, Exact) {
    ctx.plan(3);
    const auto weightAt = [](glm::ivec2 offset) {
        for (const pathtracer::gfx::HexBloomTap& tap : pathtracer::gfx::kHexBloomMask) {
            if (tap.dq == offset.x && tap.dr == offset.y) {
                return static_cast<double>(tap.weight);
            }
        }
        return 0.0;
    };
    bool symmetric = true;
    for (const pathtracer::gfx::HexBloomTap& tap : pathtracer::gfx::kHexBloomMask) {
        // A sixth of a turn in axial coordinates: (q, r) -> (-r, q + r).
        symmetric = symmetric && weightAt({-tap.dr, tap.dq + tap.dr}) == static_cast<double>(tap.weight);
    }
    PT_EXPECT(ctx, symmetric, "the mask is not invariant under a sixth of a turn");
    bool cosets = true;
    for (int coset = 0; coset < 4; ++coset) {
        const std::vector<pathtracer::gfx::HexBloomTap> taps = pathtracer::gfx::hexBloomCoset(coset & 1, coset >> 1);
        const double sum = std::accumulate(taps.begin(), taps.end(), 0.0, [](double s, const auto& tap) { return s + tap.weight; });
        cosets = cosets && sum == 1.0 && taps.size() == (coset == 0 ? 7U : 4U);
    }
    PT_EXPECT(ctx, cosets, "a coset does not carry unit weight over 7 (vertex) or 4 (edge) entries");
    bool composed = true;
    for (int r = -3; r <= 3; ++r) {
        for (int q = -3; q <= 3; ++q) {
            double sum = 0.0;
            for (int s = -1; s <= 1; ++s) {
                for (int p = -1; p <= 1; ++p) {
                    const auto smooth = [](glm::ivec2 d) {
                        const bool ring = std::find(kHexagonalRing.begin(), kHexagonalRing.end(), d) != kHexagonalRing.end();
                        return d == glm::ivec2(0) ? 0.25 : (ring ? 0.125 : 0.0);
                    };
                    sum += smooth({p, s}) * smooth(glm::ivec2(q, r) - glm::ivec2(p, s));
                }
            }
            composed = composed && sum == weightAt({q, r}) / 4.0;
        }
    }
    PT_EXPECT(ctx, composed, "mask / 4 is not the 7-tap smoothing composed with itself");
}

// The shipped single-mask pyramid against its factorisation in double, odd, even and degenerate sizes, both seam modes.
PT_CHECK(hexagonal_bloom_is_loop_subdivision_and_its_adjoint, Fast, Exact) {
    struct Case {
        int width;
        int height;
        bool wraps;
    };
    constexpr std::array<Case, 5> kCases{{{37, 23, false}, {38, 24, false}, {64, 16, true}, {3, 2, false}, {1, 1, true}}};
    constexpr float kStrength = 0.6F;
    ctx.plan(static_cast<int>(kCases.size()));
    pathtracer::scene::ThreadPool pool;
    for (const Case& testCase : kCases) {
        HdrImage image = randomImage(testCase.width, testCase.height, ctx.seed());
        const ShaderImage reference = hexagonalBloomReference(image, kStrength, testCase.wraps);
        pathtracer::gfx::applyDisplayOptics(image, PixelLattice::Hexagonal, testCase.wraps, DisplayOptics{.bloomStrength = kStrength}, pool);
        double worst = 0.0;
        for (std::size_t i = 0; i < reference.rgb.size(); ++i) {
            for (glm::length_t lane = 0; lane < 3; ++lane) {
                worst = std::max(worst, std::fabs(image.texels[(i * 3) + static_cast<std::size_t>(lane)] - reference.rgb[i][lane]));
            }
        }
        // 2L + 1 dependent passes of at most one rounding per tap plus the sum, on intermediates no larger than L times the input's 4.
        const int octaves = pathtracer::gfx::bloomOctaves(testCase.width, testCase.height);
        const double bound = (2.0 * octaves + 1.0) * ((2.0 * pathtracer::gfx::kHexBloomMask.size()) + 1.0) *
                             std::numeric_limits<float>::epsilon() * octaves * 4.0;
        char detail[160];
        std::snprintf(detail, sizeof(detail), "%dx%d %s: worst |CPU - factorised| %.3g against the float bound %.3g", testCase.width,
                      testCase.height, testCase.wraps ? "wrapped" : "mirrored", worst, bound);
        PT_EXPECT(ctx, worst <= bound, detail);
    }
}

// Every hexagonal pass carries unit weight per output, so a constant frame blooms to itself, border cells included.
PT_CHECK(hexagonal_bloom_preserves_a_constant_field, Fast, Exact) {
    ctx.plan(2);
    pathtracer::scene::ThreadPool pool;
    constexpr int kWidth = 37;
    constexpr int kHeight = 23;
    constexpr float kLevel = 0.75F;
    for (const bool wraps : {false, true}) {
        HdrImage image = pathtracer::gfx::makeImage(kWidth, kHeight, 3);
        std::fill(image.texels.begin(), image.texels.end(), kLevel);
        pathtracer::gfx::applyDisplayOptics(image, PixelLattice::Hexagonal, wraps, DisplayOptics{.bloomStrength = 1.0F}, pool);
        double worst = 0.0;
        for (const float value : image.texels) {
            worst = std::max(worst, std::fabs(static_cast<double>(value) - kLevel));
        }
        // Dyadic weights sum exactly, so only the per-pass accumulations and the final 1/L round: 2L + 1 passes of 20 additions.
        const int octaves = pathtracer::gfx::bloomOctaves(kWidth, kHeight);
        const double bound = (2.0 * octaves + 1.0) * 20.0 * std::numeric_limits<float>::epsilon() * octaves * kLevel;
        PT_EXPECT(ctx, worst <= bound, std::string(wraps ? "wrapped" : "mirrored") + ": a constant moved by " + std::to_string(worst));
    }
}

// A lat-long rolled by 2^L columns rolls every hexagonal octave by whole cells, so its bloom must roll with it, bit for bit.
PT_CHECK(hexagonal_bloom_wraps_a_lat_long_seam, Fast, Exact) {
    constexpr int kWidth = 64;
    constexpr int kHeight = 16;
    const int roll = 1 << pathtracer::gfx::bloomOctaves(kWidth, kHeight);
    ctx.plan(1);
    pathtracer::scene::ThreadPool pool;
    HdrImage image = randomImage(kWidth, kHeight, ctx.seed());
    HdrImage rolled = image;
    for (int y = 0; y < kHeight; ++y) {
        const auto row = image.texels.begin() + (static_cast<std::ptrdiff_t>(y) * kWidth * 3);
        std::rotate_copy(row, row + (static_cast<std::ptrdiff_t>(kWidth - roll) * 3), row + (static_cast<std::ptrdiff_t>(kWidth) * 3),
                         rolled.texels.begin() + (static_cast<std::ptrdiff_t>(y) * kWidth * 3));
    }
    const DisplayOptics optics{.bloomStrength = 1.0F};
    pathtracer::gfx::applyDisplayOptics(image, PixelLattice::Hexagonal, true, optics, pool);
    pathtracer::gfx::applyDisplayOptics(rolled, PixelLattice::Hexagonal, true, optics, pool);
    std::size_t differing = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            for (int lane = 0; lane < 3; ++lane) {
                const std::size_t row = static_cast<std::size_t>(y) * kWidth;
                const std::size_t at = ((row + static_cast<std::size_t>(x)) * 3) + static_cast<std::size_t>(lane);
                const std::size_t shifted = ((row + static_cast<std::size_t>((x + roll) % kWidth)) * 3) + static_cast<std::size_t>(lane);
                differing += image.texels[at] != rolled.texels[shifted] ? 1 : 0;
            }
        }
    }
    PT_EXPECT(ctx, differing == 0, std::to_string(differing) + " floats differ after rolling by " + std::to_string(roll) + " columns");
}

// The display shader's fetch at 1:1 as a cell: GL_NEAREST magnification on the square grid, latticeTexel on the hexagonal one.
glm::ivec2 shaderCell(PixelLattice lattice, glm::vec2 uv, int width, int height, bool wraps) {
    if (lattice == PixelLattice::Square) {
        const int column = static_cast<int>(std::floor(uv.x * static_cast<float>(width)));
        const int row = std::clamp(static_cast<int>(std::floor(uv.y * static_cast<float>(height))), 0, height - 1);
        return {wraps ? (((column % width) + width) % width) : std::clamp(column, 0, width - 1), row};
    }
    const float rowPitch = pathtracer::gfx::rowPitch(lattice);
    const glm::vec2 film = uv * glm::vec2(static_cast<float>(width), static_cast<float>(height) * rowPitch);
    const int below = std::clamp(static_cast<int>(std::floor((film.y / rowPitch) - 0.5F)), 0, height - 1);
    glm::ivec2 nearest(0, below);
    float nearestDistance = -1.0F;
    for (int row = below; row <= std::min(below + 1, height - 1); ++row) {
        const float offset = 0.25F + (0.5F * static_cast<float>(row & 1));
        int column = static_cast<int>(std::floor(film.x - offset + 0.5F));
        column = wraps ? column : std::clamp(column, 0, width - 1);
        const glm::vec2 delta = glm::vec2(static_cast<float>(column) + offset, (static_cast<float>(row) + 0.5F) * rowPitch) - film;
        const float distance = glm::dot(delta, delta);
        if (nearestDistance < 0.0F || distance < nearestDistance) {
            nearestDistance = distance;
            nearest = glm::ivec2(column, row);
        }
    }
    nearest.x = (nearest.x + width) % width;
    return nearest;
}

// The fringe reads R at uv - (uv - 1/2) a and B at uv + (uv - 1/2) a from the shader's cell at each pixel centre; G stays put.
PT_CHECK(aberration_reads_the_shader_cell_of_the_scaled_film, Fast, Exact) {
    struct Case {
        PixelLattice lattice;
        bool wraps;
    };
    constexpr std::array<Case, 4> kCases{{{PixelLattice::Square, false},
                                          {PixelLattice::Square, true},
                                          {PixelLattice::Hexagonal, false},
                                          {PixelLattice::Hexagonal, true}}};
    constexpr int kWidth = 29;
    constexpr int kHeight = 17;
    // Not a short decimal: 0.3 lands whole columns of 29 exactly on a cell edge, where GL's own rounding is unspecified.
    constexpr float kAberration = 0.2718282F;
    ctx.plan(static_cast<int>(kCases.size()));
    pathtracer::scene::ThreadPool pool;
    for (const Case& testCase : kCases) {
        const HdrImage input = randomImage(kWidth, kHeight, ctx.seed());
        HdrImage image = input;
        pathtracer::gfx::applyDisplayOptics(image, testCase.lattice, testCase.wraps, DisplayOptics{.aberration = kAberration}, pool);
        const glm::vec2 frame = pathtracer::gfx::frameExtent(testCase.lattice, kWidth, kHeight);
        const auto lane = [&](glm::ivec2 cell, int channel) {
            return input.texels[(((static_cast<std::size_t>(cell.y) * kWidth) + static_cast<std::size_t>(cell.x)) * 3) +
                                static_cast<std::size_t>(channel)];
        };
        std::size_t differing = 0;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const glm::vec2 uv = pathtracer::gfx::pixelCentre(testCase.lattice, x, y) / frame;
                const glm::vec2 shift = (uv - 0.5F) * kAberration;
                const glm::vec3 expected(lane(shaderCell(testCase.lattice, uv - shift, kWidth, kHeight, testCase.wraps), 0),
                                         lane({x, y}, 1),
                                         lane(shaderCell(testCase.lattice, uv + shift, kWidth, kHeight, testCase.wraps), 2));
                const std::size_t at = ((static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)) * 3;
                differing += glm::vec3(image.texels[at], image.texels[at + 1], image.texels[at + 2]) != expected ? 1 : 0;
            }
        }
        PT_EXPECT(ctx, differing == 0,
                  std::string(pathtracer::gfx::kPixelLatticeNames[static_cast<int>(testCase.lattice)]) +
                      (testCase.wraps ? " wrapped: " : " clamped: ") + std::to_string(differing) + " pixels read another cell");
    }
}

// Off is the identity, not a pass that happens to reproduce its input: every float of the image is left as it was.
PT_CHECK(display_optics_off_is_identity, Fast, Exact) {
    ctx.plan(1);
    pathtracer::scene::ThreadPool pool;
    const HdrImage input = randomImage(13, 7, ctx.seed());
    HdrImage image = input;
    pathtracer::gfx::applyDisplayOptics(image, PixelLattice::Square, false, DisplayOptics{}, pool);
    PT_EXPECT(ctx, image.texels == input.texels, "zero bloom and zero aberration changed the image");
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


// --- HUD histogram: the binning and curve shapes the panel draws, on bins whose answers follow from the definitions in closed form.

using pathtracer::debug::Histogram;
using pathtracer::debug::HistogramActive;
using pathtracer::debug::HistogramBins;
using pathtracer::debug::HistogramCurve;

constexpr std::uint32_t kCapturePixels = Histogram::kWidth * Histogram::kHeight;
constexpr double kUnitRoundoff = 0x1p-24;

// Every channel holding the same counts, the panel's grayscale case.
HistogramBins greyBins(const std::array<std::uint32_t, Histogram::kBins>& counts) {
    return HistogramBins{counts, counts, counts};
}

// The smoother's definition in double: triangle weights 5-|d| over neighbours inside [1, 254], renormalised by the weight they carry.
std::array<double, Histogram::kBins> triangleSmooth(const std::array<std::uint32_t, Histogram::kBins>& counts, double peak) {
    std::array<double, Histogram::kBins> out{};
    for (int bin = 0; bin < Histogram::kBins; ++bin) {
        double sum = 0.0;
        double weights = 0.0;
        for (int n = std::max(1, bin - 4); n <= std::min(Histogram::kBins - 2, bin + 4); ++n) {
            const double weight = 5.0 - std::abs(n - bin);
            sum += weight * std::sqrt(std::min(static_cast<double>(counts[static_cast<std::size_t>(n)]), peak) / peak);
            weights += weight;
        }
        out[static_cast<std::size_t>(bin)] = sum / weights;
    }
    return out;
}

// Each pixel lands in exactly one bin per channel, and a re-bin replaces rather than accumulates.
PT_CHECK(histogram_bins_count_every_pixel_once, Fast, Exact) {
    ctx.plan(4);
    std::vector<unsigned char> rgb(static_cast<std::size_t>(kCapturePixels) * 3);
    for (std::uint32_t i = 0; i < kCapturePixels; ++i) {
        rgb[(i * 3) + 0] = static_cast<unsigned char>(i % Histogram::kBins);
        rgb[(i * 3) + 1] = static_cast<unsigned char>(i / Histogram::kHeight);
        rgb[(i * 3) + 2] = i < kCapturePixels / 4 ? 0 : 255;
    }
    HistogramBins bins{};
    pathtracer::debug::binPixels(rgb, bins);
    pathtracer::debug::binPixels(rgb, bins);
    // R cycles every bin and G steps once per kHeight pixels, so both are flat at kHeight; B splits a quarter / three quarters.
    const auto flat = [](const std::array<std::uint32_t, Histogram::kBins>& channel) {
        return std::all_of(channel.begin(), channel.end(), [](std::uint32_t count) { return count == Histogram::kHeight; });
    };
    PT_EXPECT(ctx, flat(bins[0]), "R is not flat at kHeight per bin");
    PT_EXPECT(ctx, flat(bins[1]), "G is not flat at kHeight per bin");
    PT_EXPECT(ctx, bins[2][0] == kCapturePixels / 4 && bins[2][255] == kCapturePixels - (kCapturePixels / 4),
              "B splits " + std::to_string(bins[2][0]) + " / " + std::to_string(bins[2][255]));
    PT_EXPECT(ctx, std::accumulate(bins[2].begin(), bins[2].end(), 0U) == kCapturePixels, "B does not sum to the capture");
}

// A spike's curve is the triangle itself, renormalised where it meets the clipped extremes, then at unit peak.
PT_CHECK(histogram_smoothing_is_the_triangle, Fast, Exact) {
    const std::array<int, 3> spikes{1, 128, Histogram::kBins - 2};
    ctx.plan(static_cast<int>(spikes.size()));
    for (const int spike : spikes) {
        std::array<std::uint32_t, Histogram::kBins> counts{};
        counts[static_cast<std::size_t>(spike)] = kCapturePixels;
        const HistogramCurve heights = pathtracer::debug::grayscaleHeights(greyBins(counts));
        const std::array<double, Histogram::kBins> expected = triangleSmooth(counts, kCapturePixels);
        const double peak = *std::max_element(expected.begin(), expected.end());
        double worst = 0.0;
        for (std::size_t bin = 0; bin < heights.size(); ++bin) {
            worst = std::max(worst, std::abs(static_cast<double>(heights[bin]) - (expected[bin] / peak)));
        }
        // Integer weights sum exactly; the smoother's divide and the rescale's divide round once each on values at most 1: gamma_2.
        const double bound = (2.0 * kUnitRoundoff) / (1.0 - (2.0 * kUnitRoundoff));
        PT_EXPECT(ctx, worst <= bound, "spike at " + std::to_string(spike) + " deviates by " + std::to_string(worst));
    }
}

// The tallest active curve reaches exactly 1, and one shared peak keeps a quarter-count channel at exactly half height.
PT_CHECK(histogram_curves_peak_at_one, Fast, Exact) {
    ctx.plan(3);
    HistogramBins bins{};
    for (std::size_t bin = 1; bin + 1 < Histogram::kBins; ++bin) {
        bins[0][bin] = 4U * static_cast<std::uint32_t>(1 + ((bin * 37) % 61));
        bins[1][bin] = bins[0][bin] / 4U;
        bins[2][bin] = static_cast<std::uint32_t>(1 + ((bin * 11) % 23));
    }
    const HistogramActive active = pathtracer::debug::activeChannels(bins);
    const std::array<HistogramCurve, 3> heights = pathtracer::debug::rgbHeights(bins, active);
    float tallest = 0.0F;
    for (const HistogramCurve& curve : heights) {
        tallest = std::max(tallest, *std::max_element(curve.begin(), curve.end()));
    }
    PT_EXPECT(ctx, tallest == 1.0F, "tallest curve reaches " + std::to_string(tallest));
    // A quarter of the counts is half the sqrt, and every later step is linear or a power-of-two scale, so the ratio is exact.
    bool half = true;
    for (std::size_t bin = 0; bin < Histogram::kBins; ++bin) {
        half = half && heights[1][bin] == 0.5F * heights[0][bin];
    }
    PT_EXPECT(ctx, half, "the quarter-count channel is not at exactly half height");
    const HistogramCurve grey = pathtracer::debug::grayscaleHeights(greyBins(bins[2]));
    PT_EXPECT(ctx, *std::max_element(grey.begin(), grey.end()) == 1.0F, "the grey curve does not reach 1");
}

// Clipped 0/255 mass never shapes a smoothed curve; under 1% interior pixels the curve is the raw sqrt that keeps the clip spikes.
PT_CHECK(histogram_ignores_clipping_bins, Fast, Exact) {
    ctx.plan(4);
    std::array<std::uint32_t, Histogram::kBins> counts{};
    for (std::size_t bin = 1; bin + 1 < Histogram::kBins; ++bin) {
        counts[bin] = static_cast<std::uint32_t>(1 + ((bin * 29) % 17));
    }
    const HistogramCurve interior = pathtracer::debug::grayscaleHeights(greyBins(counts));
    std::array<std::uint32_t, Histogram::kBins> clipped = counts;
    clipped[0] = kCapturePixels / 2;
    clipped[Histogram::kBins - 1] = kCapturePixels / 3;
    PT_EXPECT(ctx, pathtracer::debug::grayscaleHeights(greyBins(clipped)) == interior, "clip-bin mass changed the smoothed curve");

    // The near-binary switch sits at 1% of the capture: one pixel under it the curve is raw, at it the curve is smoothed.
    const std::uint32_t threshold = kCapturePixels / 100;
    std::array<std::uint32_t, Histogram::kBins> mask{};
    mask[128] = threshold - 1;
    mask[Histogram::kBins - 1] = threshold / 2;
    mask[0] = kCapturePixels - mask[128] - mask[Histogram::kBins - 1];
    const HistogramCurve raw = pathtracer::debug::grayscaleHeights(greyBins(mask));
    const double spike = std::sqrt(static_cast<double>(mask[Histogram::kBins - 1]) / mask[0]);
    // One divide and one root, each rounding once on a value at most 1.
    const double bound = (2.0 * kUnitRoundoff) / (1.0 - (2.0 * kUnitRoundoff));
    PT_EXPECT(ctx, raw[0] == 1.0F && std::abs(static_cast<double>(raw[Histogram::kBins - 1]) - spike) <= bound,
              "near-binary clip spikes read " + std::to_string(raw[0]) + ", " + std::to_string(raw[Histogram::kBins - 1]));
    PT_EXPECT(ctx, raw[127] == 0.0F, "the raw curve smoothed into an empty bin");
    mask[128] += 1;
    PT_EXPECT(ctx, pathtracer::debug::grayscaleHeights(greyBins(mask))[127] > 0.0F, "at 1% interior the curve is not smoothed");
}

// Grey means all three channels agree everywhere; a channel all at bin 0 is unused; the overlap is the floor of the active ones only.
PT_CHECK(histogram_classifies_channels, Fast, Exact) {
    ctx.plan(4);
    std::array<std::uint32_t, Histogram::kBins> counts{};
    counts[200] = kCapturePixels;
    HistogramBins bins = greyBins(counts);
    PT_EXPECT(ctx, pathtracer::debug::isGrayscale(bins), "equal channels are not grey");
    bins[2][200] -= 1;
    bins[2][201] += 1;
    PT_EXPECT(ctx, !pathtracer::debug::isGrayscale(bins), "one moved pixel still reads grey");

    bins[1] = {};
    bins[1][0] = kCapturePixels;
    bins[2] = {};
    bins[2][Histogram::kBins - 1] = kCapturePixels;
    const HistogramActive active = pathtracer::debug::activeChannels(bins);
    PT_EXPECT(ctx, active[0] && !active[1] && active[2], "only the all-zero channel may be inactive");

    std::array<HistogramCurve, 3> heights{};
    for (std::size_t bin = 0; bin < Histogram::kBins; ++bin) {
        heights[0][bin] = static_cast<float>(bin) / Histogram::kBins;
        heights[1][bin] = 0.0F;
        heights[2][bin] = 1.0F - heights[0][bin];
    }
    const HistogramCurve overlap = pathtracer::debug::overlapHeights(heights, active);
    bool floor = true;
    for (std::size_t bin = 0; bin < Histogram::kBins; ++bin) {
        floor = floor && overlap[bin] == std::min(heights[0][bin], heights[2][bin]);
    }
    PT_EXPECT(ctx, floor, "overlap is not the minimum of the active channels alone");
}

}  // namespace

PT_CHECK_MAIN("display")
