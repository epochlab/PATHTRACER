// Correctness gate for the shared scale space and the AOVs built on it: kernel identities, the octave cascade, then DoG and LoG.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <glm/gtc/constants.hpp>

#include "check.h"
#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/scale_space.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using pathtracer::debug::ScaleSpaceLevel;
using pathtracer::gfx::HdrImage;
using pathtracer::scene::ThreadPool;

constexpr float kFloatEpsilon = 0x1p-23F;  // distance from 1 to the next float, the unit in the last place, twice the unit roundoff

// The exact transfer of T(.;t): sum_n T(n;t) exp(-i n w) = exp(-t (1 - cos w)), from the generating function of the modified Bessels.
[[nodiscard]] double analyticTransfer(double t, double omega) {
    return std::exp(-t * (1.0 - std::cos(omega)));
}

// Transfer measured from the shipped half-kernel, so a truncation or normalisation regression shows up as a transfer error.
[[nodiscard]] double measuredTransfer(const std::vector<float>& kernel, double omega) {
    double sum = kernel.front();
    for (std::size_t n = 1; n < kernel.size(); ++n) {
        sum += 2.0 * static_cast<double>(kernel[n]) * std::cos(static_cast<double>(n) * omega);
    }
    return sum;
}

// A grey field, so the Rec.709 reduction returns the plane value scaled by the weight sum rather than mixing three different fields.
[[nodiscard]] HdrImage greyImage(int width, int height, const std::vector<float>& plane) {
    HdrImage image{width, height,
                   std::vector<float>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4, 0.0F)};
    for (std::size_t pixel = 0; pixel < plane.size(); ++pixel) {
        const std::size_t texel = pixel * 4;
        image.rgba[texel] = plane[pixel];
        image.rgba[texel + 1] = plane[pixel];
        image.rgba[texel + 2] = plane[pixel];
        image.rgba[texel + 3] = 1.0F;
    }
    return image;
}

// Rec.709 weights sum to one only in decimal, so the luminance of a grey field is this factor, not exactly the field value.
[[nodiscard]] float greyLuminanceGain() {
    return (1.0F * pathtracer::debug::kRec709LuminanceWeights.r) +
           (1.0F * pathtracer::debug::kRec709LuminanceWeights.g) +
           (1.0F * pathtracer::debug::kRec709LuminanceWeights.b);
}

// Exactly representable affine field: small integer values, so every tap pair in the centre-relative convolution cancels bit-exactly.
[[nodiscard]] std::vector<float> affinePlane(int width, int height) {
    std::vector<float> plane(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            plane[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x)] =
                static_cast<float>(8 + (2 * x) + (3 * y));
        }
    }
    return plane;
}

// Radius in base pixels inside which a DoG or LoG sample sees no mirrored tap: each cascade step's radius on the grid it ran on.
[[nodiscard]] int scaleSpaceSupportRadius(int width, int height) {
    const float decimationAt = pathtracer::debug::decimationVariance();
    int support = 0;
    int reach = 0;
    int decimation = 1;
    int levelWidth = width;
    int levelHeight = height;
    float ownVariance = 0.0F;
    for (float baseVariance = pathtracer::debug::innerScaleVariance();; baseVariance *= 4.0F) {
        const float target = baseVariance / static_cast<float>(decimation * decimation);
        const float step = target - ownVariance;
        const auto radius = static_cast<int>(pathtracer::debug::discreteGaussianKernel(step).size()) - 1;
        if ((2 * radius) + 1 > std::min(levelWidth, levelHeight)) {
            return reach;
        }
        support += radius * decimation;
        // The Laplacian stencil and the bilinear expand each reach one level pixel, which is `decimation` base pixels wide.
        reach = support + (2 * decimation);
        ownVariance = target;
        if (ownVariance >= decimationAt) {
            ownVariance *= 0.25F;
            decimation *= 2;
            levelWidth = (levelWidth + 1) / 2;
            levelHeight = (levelHeight + 1) / 2;
        }
    }
}

[[nodiscard]] float texelAt(const HdrImage& image, int x, int y, int channel) {
    return image.rgba[(((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) +
                        static_cast<std::size_t>(x)) * 4) + static_cast<std::size_t>(channel)];
}

}  // namespace

// The Bessel generating function gives sum_n I_n(t) = exp(t), so the emitted taps must carry unit mass to float storage error alone.
PT_CHECK(discrete_gaussian_sums_to_one, Fast, Exact) {
    const std::vector<double> variances{0.25, 1.0, 4.0, 8.317766, 16.635532, 24.953298, 100.0, 1000.0};
    ctx.plan(static_cast<int>(variances.size()));
    for (const double t : variances) {
        const std::vector<float> kernel = pathtracer::debug::discreteGaussianKernel(static_cast<float>(t));
        const double error = std::abs(measuredTransfer(kernel, 0.0) - 1.0);
        // Every tap is stored with relative error at most half an ulp and the weights sum to one, so the absolute error is under one ulp.
        PT_EXPECT(ctx, error <= static_cast<double>(kFloatEpsilon),
                      "t=" + std::to_string(t) + " sum deviates by " + std::to_string(error));
    }
}

// The transfer is the strongest statement available: it implies unit mass, variance t, and the band-limiting the cascade decimates on.
PT_CHECK(discrete_gaussian_transfer_is_exact, Fast, Exact) {
    const std::vector<double> variances{1.0, 8.317766, 24.953298, 100.0};
    const std::vector<double> fractions{0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0};
    ctx.plan(static_cast<int>(variances.size() * fractions.size()));
    for (const double t : variances) {
        const std::vector<float> kernel = pathtracer::debug::discreteGaussianKernel(static_cast<float>(t));
        for (const double fraction : fractions) {
            const double omega = fraction * std::numbers::pi;
            const double error = std::abs(measuredTransfer(kernel, omega) - analyticTransfer(t, omega));
            // Discarded mass is under 2^-24 by construction and |cos| <= 1, so that plus one ulp of storage bounds the whole deviation.
            const double bound = pathtracer::debug::kFloat32Roundoff + static_cast<double>(kFloatEpsilon);
            PT_EXPECT(ctx, error <= bound,
                          "t=" + std::to_string(t) + " w/pi=" + std::to_string(fraction) + " transfer error " +
                              std::to_string(error));
        }
    }
}

// Variance is exactly t for the untruncated kernel, so what is left is the discarded second moment, which the radius and 2^-24 bound.
PT_CHECK(discrete_gaussian_variance_is_t, Fast, Exact) {
    const std::vector<double> variances{1.0, 8.317766, 24.953298, 100.0, 1000.0};
    ctx.plan(static_cast<int>(variances.size()));
    for (const double t : variances) {
        const std::vector<float> kernel = pathtracer::debug::discreteGaussianKernel(static_cast<float>(t));
        const auto radius = static_cast<double>(kernel.size() - 1);
        double variance = 0.0;
        for (std::size_t n = 1; n < kernel.size(); ++n) {
            variance += 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(kernel[n]);
        }
        // Taps decay super-exponentially past the radius, so the discarded mass sits within an offset or two of it: bound it there.
        const double bound = (pathtracer::debug::kFloat32Roundoff * (radius + 1.0) * (radius + 1.0) / t) +
                             static_cast<double>(kFloatEpsilon);
        const double error = std::abs((variance / t) - 1.0);
        PT_EXPECT(ctx, error <= bound,
                      "t=" + std::to_string(t) + " variance/t-1 = " + std::to_string(error) + " bound " +
                          std::to_string(bound));
    }
}

// T(t1) * T(t2) = T(t1+t2) on the grid; the octave cascade adds variance in steps and is only correct because this composition holds.
PT_CHECK(discrete_gaussian_semigroup_is_exact, Fast, Exact) {
    ctx.plan(1);
    const double first = 8.317766;
    const double second = 24.953298;
    const std::vector<float> a = pathtracer::debug::discreteGaussianKernel(static_cast<float>(first));
    const std::vector<float> b = pathtracer::debug::discreteGaussianKernel(static_cast<float>(second));
    const std::vector<float> composed = pathtracer::debug::discreteGaussianKernel(static_cast<float>(first + second));

    const auto radiusA = static_cast<int>(a.size()) - 1;
    const auto radiusB = static_cast<int>(b.size()) - 1;
    const auto tap = [](const std::vector<float>& kernel, int n) {
        const int index = std::abs(n);
        return index < static_cast<int>(kernel.size()) ? static_cast<double>(kernel[static_cast<std::size_t>(index)]) : 0.0;
    };
    double worst = 0.0;
    for (int n = 0; n <= radiusA + radiusB; ++n) {
        double sum = 0.0;
        for (int k = -radiusA; k <= radiusA; ++k) {
            sum += tap(a, k) * tap(b, n - k);
        }
        worst = std::max(worst, std::abs(sum - tap(composed, n)));
    }
    // Truncating both factors discards at most 2^-24 of each, and the convolution adds float storage error on the order of one ulp.
    const double bound = (2.0 * pathtracer::debug::kFloat32Roundoff) + static_cast<double>(kFloatEpsilon);
    PT_EXPECT(ctx, worst <= bound, "worst tap deviation " + std::to_string(worst) + " bound " + std::to_string(bound));
}

// A symmetric kernel of unit mass annihilates the first moment, so in the centre-relative form every tap pair of an affine field cancels.
PT_CHECK(diffusion_reproduces_affine_fields, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 96;
    constexpr int kHeight = 64;
    const std::vector<float> reference = affinePlane(kWidth, kHeight);
    std::vector<float> plane = reference;
    const float t = pathtracer::debug::innerScaleVariance();
    const int radius = static_cast<int>(pathtracer::debug::discreteGaussianKernel(t).size()) - 1;
    pathtracer::debug::diffuse(plane, kWidth, kHeight, t, pool);

    bool identical = true;
    // Interior only: the mirror turns a ramp into a tent, so the boundary response is a different field, not a rounding difference.
    for (int y = radius; y < kHeight - radius; ++y) {
        for (int x = radius; x < kWidth - radius; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x);
            identical = identical && plane[pixel] == reference[pixel];
        }
    }
    PT_EXPECT(ctx, identical, "diffusion must leave an affine field bit-identical in the interior");
}

// The 5-point stencil's weights sum to zero and its offsets are symmetric, so it kills both the constant and the linear term exactly.
PT_CHECK(laplacian5_annihilates_affine_fields, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    const std::vector<float> plane = affinePlane(kWidth, kHeight);
    std::vector<float> response(plane.size());
    pathtracer::debug::laplacian5(plane, kWidth, kHeight, response, pool);
    float worst = 0.0F;
    // Interior only: the mirror folds the ramp back on itself at the border, which is a genuinely different field, not rounding.
    for (int y = 1; y < kHeight - 1; ++y) {
        for (int x = 1; x < kWidth - 1; ++x) {
            worst = std::max(worst, std::fabs(response[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)]));
        }
    }
    PT_EXPECT(ctx, worst == 0.0F, "peak interior Laplacian of an affine field " + std::to_string(worst));
}

// The whole cascade in one assertion: every level must equal a direct full-variance diffusion of the original at its own samples.
PT_CHECK(pyramid_matches_direct_diffusion, Fast, Exact) {
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 96;
    constexpr int kHeight = 64;
    std::vector<float> source(static_cast<std::size_t>(kWidth) * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            // Broadband and asymmetric, so a level that silently aliased on decimation cannot match the direct reference by luck.
            source[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)] =
                (0.5F * std::sin(0.31F * static_cast<float>(x))) + (0.25F * std::cos(0.73F * static_cast<float>(y))) +
                ((x + (2 * y)) % 7 == 0 ? 1.0F : 0.0F);
        }
    }
    const std::vector<ScaleSpaceLevel> pyramid =
        pathtracer::debug::buildOctavePyramid(source, kWidth, kHeight, pool);
    ctx.plan(static_cast<int>(pyramid.size()) + 1);
    PT_EXPECT(ctx, pyramid.size() >= 2 && pyramid[0].decimation == 1 && pyramid[1].decimation == 1,
                  "the two finest octaves must share the base grid, which is what lets DoG difference them directly");

    for (const ScaleSpaceLevel& level : pyramid) {
        std::vector<float> direct = source;
        pathtracer::debug::diffuse(direct, kWidth, kHeight, level.baseVariance, pool);
        const int radius = static_cast<int>(pathtracer::debug::discreteGaussianKernel(level.baseVariance).size()) - 1;
        float worst = 0.0F;
        // Co-located samples only, so no interpolation enters, and the interior only, the boundary being a different operator per route.
        for (int y = 0; y < level.height; ++y) {
            for (int x = 0; x < level.width; ++x) {
                const int baseX = x * level.decimation;
                const int baseY = y * level.decimation;
                if (baseX < radius || baseX >= kWidth - radius || baseY < radius || baseY >= kHeight - radius) {
                    continue;
                }
                const float cascaded = level.plane[(static_cast<std::size_t>(y) * static_cast<std::size_t>(level.width)) +
                                                   static_cast<std::size_t>(x)];
                const float reference = direct[(static_cast<std::size_t>(baseY) * kWidth) + static_cast<std::size_t>(baseX)];
                worst = std::max(worst, std::fabs(cascaded - reference));
            }
        }
        // Each decimation discards under 2^-24 of the level's energy and each pass rounds, so the budget grows with the level index.
        const double bound = static_cast<double>(kFloatEpsilon) * 8.0 * (1.0 + std::log2(static_cast<double>(level.decimation)));
        PT_EXPECT(ctx, static_cast<double>(worst) <= bound,
                      "level t=" + std::to_string(level.baseVariance) + " deviates by " + std::to_string(worst) +
                          " bound " + std::to_string(bound));
    }
}

// A constant field survives the mirror exactly, so both scale-space AOVs must read zero at every pixel, border included.
PT_CHECK(scale_space_aovs_are_zero_on_a_constant_field, Fast, Exact) {
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 128;
    constexpr int kHeight = 96;
    const HdrImage flat = greyImage(kWidth, kHeight, std::vector<float>(static_cast<std::size_t>(kWidth) * kHeight, 0.375F));

    const HdrImage dog = pathtracer::debug::dogAov(flat, pool);
    float worstDog = 0.0F;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            worstDog = std::max(worstDog, std::fabs(texelAt(dog, x, y, 0)));
        }
    }
    PT_EXPECT(ctx, worstDog == 0.0F, "peak DoG response " + std::to_string(worstDog));

    const HdrImage log = pathtracer::debug::logAov(flat, pool);
    float worstLog = 0.0F;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            worstLog = std::max(worstLog, std::fabs(texelAt(log, x, y, 0)));
        }
    }
    PT_EXPECT(ctx, worstLog == 0.0F, "peak LoG magnitude " + std::to_string(worstLog));
}

// Every cascade step reproduces an affine field and the stencil annihilates it, so neither AOV may fire on a smooth gradient.
PT_CHECK(scale_space_aovs_are_flat_on_an_affine_field, Fast, Exact) {
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    // Sized so the ladder stops at the two base-grid octaves, the deepest pyramid whose cascade support still fits inside the frame.
    constexpr int kWidth = 256;
    constexpr int kHeight = 112;
    const HdrImage image = greyImage(kWidth, kHeight, affinePlane(kWidth, kHeight));
    const int radius = scaleSpaceSupportRadius(kWidth, kHeight);
    PT_EXPECT(ctx, (2 * radius) < kHeight,
                  "support radius " + std::to_string(radius) + " leaves no interior in " + std::to_string(kHeight) + " rows");

    const HdrImage dog = pathtracer::debug::dogAov(image, pool);
    const HdrImage log = pathtracer::debug::logAov(image, pool);
    float worstDog = 0.0F;
    float worstLog = 0.0F;
    for (int y = radius; y < kHeight - radius; ++y) {
        for (int x = radius; x < kWidth - radius; ++x) {
            worstDog = std::max(worstDog, std::fabs(texelAt(dog, x, y, 0)));
            worstLog = std::max(worstLog, std::fabs(texelAt(log, x, y, 0)));
        }
    }
    // The Rec.709 reduction rounds, so the plane is affine only to an ulp; carried through the widest step that is the bound, not zero.
    const float peak = greyLuminanceGain() * static_cast<float>(8 + (2 * (kWidth - 1)) + (3 * (kHeight - 1)));
    const auto taps = static_cast<double>((2 * (static_cast<int>(pathtracer::debug::discreteGaussianKernel(
                                                    3.0F * pathtracer::debug::innerScaleVariance()).size()) - 1)) + 1);
    const double quantisation = taps * static_cast<double>(peak) * static_cast<double>(kFloatEpsilon);
    PT_EXPECT(ctx, static_cast<double>(worstDog) <= quantisation,
                  "interior DoG " + std::to_string(worstDog) + " exceeds the plane quantisation " + std::to_string(quantisation));
    // The gamma=1 normalisation multiplies the stencil output by the level's own variance, so the same quantisation scales with it.
    const double normalised = quantisation * static_cast<double>(4.0F * pathtracer::debug::innerScaleVariance());
    PT_EXPECT(ctx, static_cast<double>(worstLog) <= normalised,
                  "interior LoG " + std::to_string(worstLog) + " exceeds the normalised quantisation " + std::to_string(normalised));
}

// On a cosine the two octaves are eigenfunctions, so DoG must equal the analytic difference of transfers -- amplitude and sign both.
PT_CHECK(dog_matches_the_analytic_transfer, Fast, Exact) {
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 128;
    constexpr int kHeight = 96;
    const std::vector<double> fractions{0.05, 0.1, 0.2, 0.35};
    ctx.plan(static_cast<int>(fractions.size()));
    const double fine = static_cast<double>(pathtracer::debug::innerScaleVariance());
    const double coarse = 4.0 * fine;

    for (const double cyclesPerPixel : fractions) {
        const double omega = 2.0 * std::numbers::pi * cyclesPerPixel;
        std::vector<float> plane(static_cast<std::size_t>(kWidth) * kHeight);
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                plane[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)] =
                    static_cast<float>(std::cos(omega * static_cast<double>(x)));
            }
        }
        const HdrImage dog = pathtracer::debug::dogAov(greyImage(kWidth, kHeight, plane), pool);
        // The y transfer at w=0 is unity, so only the x direction attenuates: the expected response is one transfer difference.
        const double gain = static_cast<double>(greyLuminanceGain()) *
                            (analyticTransfer(fine, omega) - analyticTransfer(coarse, omega));
        const int radius = static_cast<int>(pathtracer::debug::discreteGaussianKernel(static_cast<float>(coarse)).size()) - 1;
        double worst = 0.0;
        for (int y = radius; y < kHeight - radius; ++y) {
            for (int x = radius; x < kWidth - radius; ++x) {
                const double expected = gain * std::cos(omega * static_cast<double>(x));
                worst = std::max(worst, std::abs(static_cast<double>(texelAt(dog, x, y, 0)) - expected));
            }
        }
        PT_EXPECT(ctx, worst <= 1e-6,
                      "f=" + std::to_string(cyclesPerPixel) + " cyc/px deviates by " + std::to_string(worst));
    }
}

// Lindeberg 1998: the gamma=1 response t*laplacian5 is stationary at t = t0, and on a discrete blob its value is closed-form exact.
PT_CHECK(log_response_peaks_at_the_blob_scale, Fast, Exact) {
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 192;
    constexpr int kHeight = 192;
    constexpr double kBlobVariance = 16.0;
    const auto pixels = static_cast<std::size_t>(kWidth) * kHeight;
    const std::size_t centre = (static_cast<std::size_t>(kHeight / 2) * kWidth) + static_cast<std::size_t>(kWidth / 2);
    const int blobRadius =
        static_cast<int>(pathtracer::debug::discreteGaussianKernel(static_cast<float>(kBlobVariance)).size()) - 1;

    std::vector<float> blob(pixels, 0.0F);
    blob[centre] = 1.0F;
    pathtracer::debug::diffuse(blob, kWidth, kHeight, static_cast<float>(kBlobVariance), pool);

    // A ladder of the validator's own choosing, four rungs per octave, far finer than the shipped one, so the argmax localises in t.
    std::vector<double> ladder;
    for (int step = -6; step <= 6; ++step) {
        ladder.push_back(kBlobVariance * std::pow(2.0, static_cast<double>(step) / 4.0));
    }
    ctx.plan(static_cast<int>(ladder.size()) + 2);

    double bestResponse = 0.0;
    double bestVariance = 0.0;
    for (const double t : ladder) {
        std::vector<float> level = blob;
        pathtracer::debug::diffuse(level, kWidth, kHeight, static_cast<float>(t), pool);
        std::vector<float> response(pixels);
        pathtracer::debug::laplacian5(level, kWidth, kHeight, response, pool);
        const double measured = t * static_cast<double>(response[centre]);

        // Diffusing an impulse to s gives the separable T(.;s), and dL/dt = laplacian5(L)/2 then makes the centre response exact.
        const double total = kBlobVariance + t;
        const std::vector<float> kernel = pathtracer::debug::discreteGaussianKernel(static_cast<float>(total));
        const auto centreTap = static_cast<double>(kernel[0]);
        const auto firstTap = static_cast<double>(kernel[1]);
        const double exact = 4.0 * t * centreTap * (firstTap - centreTap);

        // The plane carries both diffusions' accumulated rounding and the stencil differences it, so the bound scales by centre/response.
        const int radius =
            blobRadius + static_cast<int>(pathtracer::debug::discreteGaussianKernel(static_cast<float>(t)).size()) - 1;
        const double bound = 2.0 * static_cast<double>((2 * radius) + 1) * static_cast<double>(kFloatEpsilon) *
                             std::abs(t * static_cast<double>(level[centre]));
        PT_EXPECT(ctx, std::abs(measured - exact) <= bound,
                      "t=" + std::to_string(t) + " response " + std::to_string(measured) + " exact " +
                          std::to_string(exact) + " bound " + std::to_string(bound));
        if (std::abs(measured) > std::abs(bestResponse)) {
            bestResponse = measured;
            bestVariance = t;
        }
    }
    PT_EXPECT(ctx, bestVariance == kBlobVariance,
                  "scale selection peaked at t=" + std::to_string(bestVariance) + ", blob is t=" +
                      std::to_string(kBlobVariance));

    // Marr & Hildreth's continuous peak is 1/(4 pi t0); the discrete kernel's own central value lifts it by 1/(2s) at s = 2 t0.
    const double continuousPeak = 1.0 / (4.0 * std::numbers::pi * kBlobVariance);
    const double ratio = std::abs(bestResponse) / continuousPeak;
    const double leading = 1.0 / (2.0 * 2.0 * kBlobVariance);
    PT_EXPECT(ctx, std::abs(ratio - 1.0 - leading) <= 1.0 / (4.0 * kBlobVariance * kBlobVariance),
                  "peak/continuous-1 = " + std::to_string(ratio - 1.0) + ", derived leading term " + std::to_string(leading));
}

// One family, one sign convention: both AOVs must read positive on a bright blob, which is why logAov negates the raw Laplacian.
PT_CHECK(log_and_dog_agree_on_polarity, Fast, Exact) {
    ctx.plan(4);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 96;
    constexpr int kHeight = 96;
    const auto pixels = static_cast<std::size_t>(kWidth) * kHeight;
    const int centreX = kWidth / 2;
    const int centreY = kHeight / 2;
    const std::size_t centre = (static_cast<std::size_t>(centreY) * kWidth) + static_cast<std::size_t>(centreX);

    std::vector<float> bright(pixels, 0.25F);
    bright[centre] += 1.0F;
    pathtracer::debug::diffuse(bright, kWidth, kHeight, pathtracer::debug::innerScaleVariance(), pool);
    const HdrImage brightField = greyImage(kWidth, kHeight, bright);
    const float brightLog = texelAt(pathtracer::debug::logAov(brightField, pool), centreX, centreY, 0);
    const float brightDog = texelAt(pathtracer::debug::dogAov(brightField, pool), centreX, centreY, 0);
    PT_EXPECT(ctx, brightLog > 0.0F, "bright-on-dark LoG read " + std::to_string(brightLog));
    PT_EXPECT(ctx, brightDog > 0.0F, "bright-on-dark DoG read " + std::to_string(brightDog));

    std::vector<float> dark(pixels, 1.25F);
    dark[centre] -= 1.0F;
    pathtracer::debug::diffuse(dark, kWidth, kHeight, pathtracer::debug::innerScaleVariance(), pool);
    const HdrImage darkField = greyImage(kWidth, kHeight, dark);
    const float darkLog = texelAt(pathtracer::debug::logAov(darkField, pool), centreX, centreY, 0);
    const float darkDog = texelAt(pathtracer::debug::dogAov(darkField, pool), centreX, centreY, 0);
    PT_EXPECT(ctx, darkLog < 0.0F, "dark-on-bright LoG read " + std::to_string(darkLog));
    PT_EXPECT(ctx, darkDog < 0.0F, "dark-on-bright DoG read " + std::to_string(darkDog));
}

// Every stage from the luminance dot to the normalised Laplacian is linear, so a power-of-two gain must pass through exactly.
PT_CHECK(log_is_homogeneous_of_degree_one, Fast, Exact) {
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 128;
    constexpr int kHeight = 96;
    constexpr float kGain = 64.0F;
    std::vector<float> plane(static_cast<std::size_t>(kWidth) * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            // A pattern with structure at several octaves, so the argmax lands on different levels across the frame.
            plane[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)] =
                0.5F + (0.25F * std::sin(0.4F * static_cast<float>(x))) + (0.25F * std::cos(0.05F * static_cast<float>(y)));
        }
    }
    std::vector<float> scaled = plane;
    for (float& value : scaled) {
        value *= kGain;
    }

    const HdrImage base = pathtracer::debug::logAov(greyImage(kWidth, kHeight, plane), pool);
    const HdrImage gained = pathtracer::debug::logAov(greyImage(kWidth, kHeight, scaled), pool);
    float worst = 0.0F;
    float peak = 0.0F;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            peak = std::max(peak, std::fabs(texelAt(base, x, y, 0)));
            worst = std::max(worst, std::fabs(texelAt(gained, x, y, 0) - (kGain * texelAt(base, x, y, 0))));
        }
    }
    PT_EXPECT(ctx, peak > 0.0F, "the pattern produced no response at all, so the homogeneity check is vacuous");
    PT_EXPECT(ctx, worst == 0.0F, "worst gain mismatch " + std::to_string(worst) + " against peak " + std::to_string(peak));
}

// The two AOVs are not redundant: DoG sees one fixed band next to pixel Nyquist, LoG the whole ladder, so a coarse blob separates them.
PT_CHECK(log_responds_where_dog_cannot, Fast, Exact) {
    ctx.plan(4);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 256;
    constexpr int kHeight = 256;
    // Four octaves above the inner scale, so the blob sits far outside the band DoG differences but well inside the ladder.
    const float blobVariance = 16.0F * pathtracer::debug::innerScaleVariance();
    const auto pixels = static_cast<std::size_t>(kWidth) * kHeight;
    const int centreX = kWidth / 2;
    const int centreY = kHeight / 2;

    std::vector<float> blob(pixels, 0.0F);
    blob[(static_cast<std::size_t>(centreY) * kWidth) + static_cast<std::size_t>(centreX)] = 1.0F;
    pathtracer::debug::diffuse(blob, kWidth, kHeight, blobVariance, pool);
    const HdrImage field = greyImage(kWidth, kHeight, blob);

    const float log = texelAt(pathtracer::debug::logAov(field, pool), centreX, centreY, 0);
    const float dog = texelAt(pathtracer::debug::dogAov(field, pool), centreX, centreY, 0);
    PT_EXPECT(ctx, log > 0.0F, "LoG read " + std::to_string(log) + " on a bright blob");

    // Lindeberg 1998: the gamma=1 response to a blob of scale t0 peaks at t = t0 with magnitude 1/(4 pi t0), here in luminance units.
    const auto gain = static_cast<double>(greyLuminanceGain());
    const double peak = gain / (4.0 * std::numbers::pi * static_cast<double>(blobVariance));
    // The ladder quantises scale to one octave, so the winning rung is at worst t0/2 or 2 t0, where -t/(pi (t0+t)^2) is 8/9 of the peak.
    const double laddered = peak * 8.0 / 9.0;
    PT_EXPECT(ctx, static_cast<double>(log) > laddered,
                  "LoG " + std::to_string(log) + " fell below the octave-laddered Lindeberg peak " + std::to_string(laddered));

    // DoG's own oracle, no threshold: an impulse diffused to variance s has separable centre value T(0;s)^2, and DoG differences two.
    const std::vector<ScaleSpaceLevel> pyramid =
        pathtracer::debug::buildOctavePyramid(std::vector<float>(pixels, 0.0F), kWidth, kHeight, pool);
    const auto centreValue = [&](float levelVariance) {
        const auto tap = static_cast<double>(
            pathtracer::debug::discreteGaussianKernel(static_cast<float>(blobVariance) + levelVariance)[0]);
        return gain * tap * tap;
    };
    const double fineCentre = centreValue(pyramid[0].baseVariance);
    const double dogExact = fineCentre - centreValue(pyramid[1].baseVariance);
    const auto taps = static_cast<double>((2 * static_cast<int>(pathtracer::debug::discreteGaussianKernel(
                                               blobVariance + pyramid[1].baseVariance).size())) - 1);
    const double bound = 2.0 * taps * static_cast<double>(kFloatEpsilon) * fineCentre;
    PT_EXPECT(ctx, std::abs(static_cast<double>(dog) - dogExact) <= bound,
                  "DoG " + std::to_string(dog) + " against its oracle " + std::to_string(dogExact) + " bound " +
                      std::to_string(bound));
    // The separation is between two closed forms, not against a chosen threshold: the fixed band simply does not reach this scale.
    PT_EXPECT(ctx, dogExact < laddered,
                  "DoG's analytic response " + std::to_string(dogExact) + " is not below LoG's " + std::to_string(laddered));
}

// Justifies the retinex fast path: with every texel valid the normalising mask is a constant field, which the cascade must return intact.
PT_CHECK(expansion_reproduces_a_constant_level_exactly, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 320;
    constexpr int kHeight = 208;
    const std::vector<float> ones(static_cast<std::size_t>(kWidth) * kHeight, 1.0F);
    const std::vector<ScaleSpaceLevel> pyramid = pathtracer::debug::buildOctavePyramid(ones, kWidth, kHeight, pool);
    const std::vector<float> expanded = pathtracer::debug::expandToBase(pyramid.back(), kWidth, kHeight, pool);
    const auto extremes = std::minmax_element(expanded.begin(), expanded.end());
    PT_EXPECT(ctx, *extremes.first == 1.0F && *extremes.second == 1.0F,
                  "coarsest level expands to [" + std::to_string(*extremes.first) + ", " +
                      std::to_string(*extremes.second) + "], not exactly one");
}

// A cone chromaticity is a ratio of two linear forms, and writing the numerators on (R-G, B-G) makes the achromatic axis exactly zero.
PT_CHECK(colour_opponent_is_zero_on_every_achromatic_texel, Fast, Exact) {
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    // Intensities spanning eleven decades and deliberately not powers of two, so no exactness can come from the exponent alone.
    std::vector<float> grey(static_cast<std::size_t>(kWidth) * kHeight);
    for (std::size_t pixel = 0; pixel < grey.size(); ++pixel) {
        grey[pixel] = 1e-5F * std::pow(1.37F, static_cast<float>(pixel % 97));
    }
    const HdrImage opponent = pathtracer::debug::colourOpponentAov(greyImage(kWidth, kHeight, grey), pool);
    float worstRedGreen = 0.0F;
    float worstBlueYellow = 0.0F;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            worstRedGreen = std::max(worstRedGreen, std::fabs(texelAt(opponent, x, y, 0)));
            worstBlueYellow = std::max(worstBlueYellow, std::fabs(texelAt(opponent, x, y, 1)));
        }
    }
    PT_EXPECT(ctx, worstRedGreen == 0.0F, "peak achromatic R-G " + std::to_string(worstRedGreen));
    PT_EXPECT(ctx, worstBlueYellow == 0.0F, "peak achromatic B-Y " + std::to_string(worstBlueYellow));
}

// MacLeod & Boynton coordinates are ratios, so a gain cancels; at a power of two every product is exact and the cancellation is bitwise.
PT_CHECK(colour_opponent_is_invariant_to_exposure, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 48;
    constexpr int kHeight = 32;
    HdrImage image{kWidth, kHeight, std::vector<float>(static_cast<std::size_t>(kWidth) * kHeight * 4, 0.0F)};
    HdrImage scaled = image;
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(kWidth) * kHeight; ++pixel) {
        for (int channel = 0; channel < 3; ++channel) {
            const float value = 0.125F + (0.0625F * static_cast<float>((pixel * 7 + channel * 5) % 13));
            image.rgba[(pixel * 4) + static_cast<std::size_t>(channel)] = value;
            scaled.rgba[(pixel * 4) + static_cast<std::size_t>(channel)] = value * 256.0F;
        }
    }
    const HdrImage base = pathtracer::debug::colourOpponentAov(image, pool);
    const HdrImage gained = pathtracer::debug::colourOpponentAov(scaled, pool);
    PT_EXPECT(ctx, base.rgba == gained.rgba, "a 256x gain changed the opponent response");
}

// The two axes are the L-versus-M and S-versus-(L+M) cardinal directions, so each Rec.709 primary must sit on its own side of white.
PT_CHECK(colour_opponent_separates_the_rec709_primaries, Fast, Exact) {
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const auto axis = [&pool](const glm::vec3& rgb, int channel) {
        HdrImage image{1, 1, {rgb.r, rgb.g, rgb.b, 1.0F}};
        return texelAt(pathtracer::debug::colourOpponentAov(image, pool), 0, 0, channel);
    };
    const float red = axis({1.0F, 0.0F, 0.0F}, 0);
    const float green = axis({0.0F, 1.0F, 0.0F}, 0);
    const float blue = axis({0.0F, 0.0F, 1.0F}, 1);
    PT_EXPECT(ctx, red > 0.0F, "Rec.709 red reads R-G " + std::to_string(red));
    PT_EXPECT(ctx, green < 0.0F, "Rec.709 green reads R-G " + std::to_string(green));
    PT_EXPECT(ctx, blue > 0.0F, "Rec.709 blue reads B-Y " + std::to_string(blue));
}

// A field equal to its own surround has unit reflectance by definition, and the cascade returns a constant intact, so this is exact.
PT_CHECK(retinex_is_exactly_one_on_a_uniform_field, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 160;
    constexpr int kHeight = 112;
    const HdrImage flat = greyImage(kWidth, kHeight, std::vector<float>(static_cast<std::size_t>(kWidth) * kHeight, 0.375F));
    const HdrImage retinex = pathtracer::debug::retinexAov(flat, pool);
    float worst = 0.0F;
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(kWidth) * kHeight; ++pixel) {
        for (int channel = 0; channel < 3; ++channel) {
            worst = std::max(worst, std::fabs(retinex.rgba[(pixel * 4) + static_cast<std::size_t>(channel)] - 1.0F));
        }
    }
    PT_EXPECT(ctx, worst == 0.0F, "uniform-field reflectance departs from one by " + std::to_string(worst));
}

// Normalised convolution (Knutsson & Westin 1993) divides the blurred field by the blurred mask, so a dark hole leaves the surround alone.
PT_CHECK(retinex_normalises_the_surround_by_the_validity_mask, Fast, Exact) {
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 160;
    constexpr int kHeight = 112;
    constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;
    // Powers of two three stops apart, so the relative log is an exact small integer and this reference needs no rounding budget.
    std::vector<float> plane(kPixels);
    std::vector<float> logRadiance(kPixels, 0.0F);
    std::vector<float> mask(kPixels, 1.0F);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const auto pixel = (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x);
            const bool hole = y >= 40 && y < 72 && x >= 48 && x < 112;
            plane[pixel] = hole ? 0.0F : (x < kWidth / 2 ? 0.25F : 2.0F);
            mask[pixel] = hole ? 0.0F : 1.0F;
            logRadiance[pixel] = hole ? 0.0F : (x < kWidth / 2 ? 0.0F : 3.0F);
        }
    }

    // The same two cascades the filter runs, spelled out against the public facility: the value over the mask, both at the coarsest scale.
    const std::vector<float> value =
        pathtracer::debug::expandToBase(pathtracer::debug::buildOctavePyramid(logRadiance, kWidth, kHeight, pool).back(),
                                        kWidth, kHeight, pool);
    const std::vector<float> weight = pathtracer::debug::expandToBase(
        pathtracer::debug::buildOctavePyramid(mask, kWidth, kHeight, pool).back(), kWidth, kHeight, pool);

    const HdrImage retinex = pathtracer::debug::retinexAov(greyImage(kWidth, kHeight, plane), pool);
    float worst = 0.0F;
    float unmaskedGap = 0.0F;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const auto pixel = (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x);
            const float expected =
                mask[pixel] > 0.0F ? std::exp2(logRadiance[pixel] - (value[pixel] / weight[pixel])) : 0.0F;
            worst = std::max(worst, std::fabs(texelAt(retinex, x, y, 0) - expected));
            const float unmasked = mask[pixel] > 0.0F ? std::exp2(logRadiance[pixel] - value[pixel]) : 0.0F;
            unmaskedGap = std::max(unmaskedGap, std::fabs(unmasked - expected));
        }
    }
    PT_EXPECT(ctx, worst == 0.0F, "retinex departs from normalised convolution by " + std::to_string(worst));
    // Without the division the hole enters the surround as a zero, so the two must not agree: otherwise the check above proves nothing.
    PT_EXPECT(ctx, unmaskedGap > 0.0F, "dropping the mask changed nothing, so the normalisation is untested");
}

// Retinex divides out its own surround, so a frame-wide gain cancels; at a power of two the relative log is untouched, so bitwise.
PT_CHECK(retinex_is_invariant_to_a_uniform_gain, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 128;
    constexpr int kHeight = 96;
    const std::vector<float> plane = affinePlane(kWidth, kHeight);
    std::vector<float> gained(plane.size());
    std::transform(plane.begin(), plane.end(), gained.begin(), [](float value) { return value * 512.0F; });
    const HdrImage base = pathtracer::debug::retinexAov(greyImage(kWidth, kHeight, plane), pool);
    const HdrImage bright = pathtracer::debug::retinexAov(greyImage(kWidth, kHeight, gained), pool);
    PT_EXPECT(ctx, base.rgba == bright.rgba, "a 512x gain changed the retinex reflectance");
}

// No dynamic range is no histogram to equalise, and the operator is then exactly the identity rather than approximately one.
PT_CHECK(clahe_passes_a_uniform_field_through_unchanged, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 128;
    constexpr int kHeight = 96;
    const HdrImage flat = greyImage(kWidth, kHeight, std::vector<float>(static_cast<std::size_t>(kWidth) * kHeight, 0.375F));
    const HdrImage clahe = pathtracer::debug::claheAov(flat, glm::radians(30.0F), pool);
    PT_EXPECT(ctx, clahe.rgba == flat.rgba, "a uniform field was not passed through unchanged");
}

// Every bin index, bin width and transfer value is a difference of log radiances, so a power-of-two gain moves only the output exponent.
PT_CHECK(clahe_is_homogeneous_of_degree_one, Fast, Exact) {
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 192;
    constexpr int kHeight = 128;
    const HdrImage image = greyImage(kWidth, kHeight, affinePlane(kWidth, kHeight));
    std::vector<float> gainedPlane = affinePlane(kWidth, kHeight);
    std::transform(gainedPlane.begin(), gainedPlane.end(), gainedPlane.begin(), [](float v) { return v * 64.0F; });
    const HdrImage base = pathtracer::debug::claheAov(image, glm::radians(30.0F), pool);
    const HdrImage gained = pathtracer::debug::claheAov(greyImage(kWidth, kHeight, gainedPlane), glm::radians(30.0F), pool);
    float worst = 0.0F;
    // Colour only: alpha is a coverage flag, not a radiance, so it is the one channel a gain must leave alone rather than scale.
    for (std::size_t pixel = 0; pixel < base.rgba.size() / 4; ++pixel) {
        for (int channel = 0; channel < 3; ++channel) {
            const std::size_t index = (pixel * 4) + static_cast<std::size_t>(channel);
            worst = std::max(worst, std::fabs((base.rgba[index] * 64.0F) - gained.rgba[index]));
        }
    }
    PT_EXPECT(ctx, worst == 0.0F, "a 64x gain did not scale the CLAHE response exactly, worst " + std::to_string(worst));
    // Homogeneity is trivial for the identity, so the same configuration must be shown to move the frame at all.
    PT_EXPECT(ctx, base.rgba != image.rgba, "CLAHE left this frame untouched, so the homogeneity above proves nothing");
}

// A cumulative histogram is non-decreasing and spans its whole range, so with one tile the operator is a monotone onto map in luminance.
PT_CHECK(clahe_transfer_is_monotone_and_range_preserving, Fast, Exact) {
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    // One degree of visual angle per tile, so a one-degree field is a single tile and the bilinear blend cannot mask a transfer defect.
    constexpr int kWidth = 96;
    constexpr int kHeight = 96;
    const float oneTileFov = glm::radians(1.0F);
    std::vector<float> plane(static_cast<std::size_t>(kWidth) * kHeight);
    for (std::size_t pixel = 0; pixel < plane.size(); ++pixel) {
        // Radiances spread over eight stops in a scrambled order, so monotonicity cannot come from the scan order.
        plane[pixel] = std::exp2(-4.0F + (8.0F * static_cast<float>((pixel * 2654435761U) % 4096U) / 4095.0F));
    }
    const HdrImage image = greyImage(kWidth, kHeight, plane);
    const HdrImage clahe = pathtracer::debug::claheAov(image, oneTileFov, pool);

    std::vector<std::pair<float, float>> pairs(plane.size());
    for (std::size_t pixel = 0; pixel < plane.size(); ++pixel) {
        pairs[pixel] = {image.rgba[pixel * 4], clahe.rgba[pixel * 4]};
    }
    std::sort(pairs.begin(), pairs.end());
    bool monotone = true;
    for (std::size_t index = 1; index < pairs.size(); ++index) {
        monotone = monotone && pairs[index].second >= pairs[index - 1].second;
    }
    PT_EXPECT(ctx, monotone, "the single-tile transfer is not non-decreasing in input radiance");
    // The cumulative runs from zero to one over the occupied range, so its endpoints land back on the extremes they came from.
    PT_EXPECT(ctx, pairs.front().second == pairs.front().first,
                  "the darkest texel moved from " + std::to_string(pairs.front().first) + " to " +
                      std::to_string(pairs.front().second));
    const double topError = std::fabs(static_cast<double>(pairs.back().second - pairs.back().first));
    // The cumulative reaches one only up to the rounding of its own bin sum, which the exponential then carries at ln2 per unit.
    const double bound = static_cast<double>(pairs.back().first) * std::numbers::ln2 *
                         static_cast<double>(kFloatEpsilon) * 64.0;
    PT_EXPECT(ctx, topError <= bound,
                  "the brightest texel moved by " + std::to_string(topError) + ", bound " + std::to_string(bound));
}

// Zuiderveld's four tile weights are written as nested lerps, so they sum to exactly one and equal inputs cannot map to unequal outputs.
PT_CHECK(clahe_blend_is_a_partition_of_unity, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    // Tiles of exactly eight pixels, carrying a period-eight pattern, so every tile holds the same histogram and the same transfer.
    constexpr int kWidth = 256;
    constexpr int kHeight = 256;
    constexpr int kPeriod = 8;
    const float fov = glm::radians(static_cast<float>(kHeight) / static_cast<float>(kPeriod));
    std::vector<float> plane(static_cast<std::size_t>(kWidth) * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            plane[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)] =
                std::exp2(static_cast<float>(((y % kPeriod) * kPeriod) + (x % kPeriod)) * 0.125F);
        }
    }
    const HdrImage clahe = pathtracer::debug::claheAov(greyImage(kWidth, kHeight, plane), fov, pool);
    std::vector<float> firstSeen(static_cast<std::size_t>(kPeriod) * kPeriod, -1.0F);
    float worst = 0.0F;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const auto key = static_cast<std::size_t>(((y % kPeriod) * kPeriod) + (x % kPeriod));
            const float value = texelAt(clahe, x, y, 0);
            worst = firstSeen[key] < 0.0F ? worst : std::max(worst, std::fabs(value - firstSeen[key]));
            firstSeen[key] = value;
        }
    }
    PT_EXPECT(ctx, worst == 0.0F, "identical tiles mapped equal radiances differently, by " + std::to_string(worst));
}

// A degree-wide tile cannot be finer than the sampling, so below one pixel per degree the grid caps and each tile holds one sample.
PT_CHECK(clahe_is_the_identity_below_one_tile_per_pixel, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    // 36 rows across 90 degrees: every pixel spans more than a degree, so there is no sub-pixel adaptation left to model.
    constexpr int kWidth = 48;
    constexpr int kHeight = 36;
    std::vector<float> plane(static_cast<std::size_t>(kWidth) * kHeight);
    for (std::size_t pixel = 0; pixel < plane.size(); ++pixel) {
        plane[pixel] = std::exp2(-3.0F + (6.0F * static_cast<float>((pixel * 2654435761U) % 997U) / 996.0F));
    }
    const HdrImage image = greyImage(kWidth, kHeight, plane);
    const HdrImage clahe = pathtracer::debug::claheAov(image, glm::radians(90.0F), pool);
    const auto extremes = std::minmax_element(plane.begin(), plane.end());
    const double range = std::log2(static_cast<double>(*extremes.second) / static_cast<double>(*extremes.first));
    double worst = 0.0;
    for (std::size_t pixel = 0; pixel < plane.size(); ++pixel) {
        worst = std::max(worst, std::fabs(static_cast<double>(clahe.rgba[pixel * 4] - image.rgba[pixel * 4]) /
                                          static_cast<double>(image.rgba[pixel * 4])));
    }
    // One bin makes the cumulative the linear transfer, so only the log round trip is left: two roundings of the exponent, at ln2 a stop.
    const double bound = 2.0 * std::numbers::ln2 * range * static_cast<double>(kFloatEpsilon);
    PT_EXPECT(ctx, worst <= bound, "a sub-pixel tile pitch moved the frame by a relative " + std::to_string(worst) +
                                       ", bound " + std::to_string(bound));
}

// Only luminance is remapped, so every texel keeps its chromaticity: the output is its input times one positive scalar, to a rounding.
PT_CHECK(clahe_preserves_chromaticity, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 160;
    constexpr int kHeight = 128;
    HdrImage image{kWidth, kHeight, std::vector<float>(static_cast<std::size_t>(kWidth) * kHeight * 4, 0.0F)};
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(kWidth) * kHeight; ++pixel) {
        const float base = std::exp2(-3.0F + (6.0F * static_cast<float>(pixel % 251U) / 250.0F));
        image.rgba[pixel * 4] = base;
        image.rgba[(pixel * 4) + 1] = base * 0.375F;
        image.rgba[(pixel * 4) + 2] = base * 2.25F;
        image.rgba[(pixel * 4) + 3] = 1.0F;
    }
    const HdrImage clahe = pathtracer::debug::claheAov(image, glm::radians(30.0F), pool);
    double worst = 0.0;
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(kWidth) * kHeight; ++pixel) {
        const double red = clahe.rgba[pixel * 4] / static_cast<double>(image.rgba[pixel * 4]);
        const double blue = clahe.rgba[(pixel * 4) + 2] / static_cast<double>(image.rgba[(pixel * 4) + 2]);
        worst = std::max(worst, std::fabs(red - blue) / red);
    }
    // One rounding of the scale into each of the two products, so the two recovered ratios may differ by two ulps and no more.
    PT_EXPECT(ctx, worst <= 2.0 * static_cast<double>(kFloatEpsilon),
                  "per-channel scale factors differ by a relative " + std::to_string(worst));
}

// x_i = mu +/- delta over an even count gives M2 = n delta^2 exactly, so the reported ratio has a closed form and no tolerance.
PT_CHECK(snr_matches_the_closed_form_of_an_alternating_sequence, Fast, Exact) {
    // 65536 is past the 46341 at which the degrees-of-freedom product leaves int, so this sweep covers that widening too.
    const std::array<int, 5> counts{2, 8, 64, 1024, 65536};
    ctx.plan(static_cast<int>(counts.size()));
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;
    // Powers of two, so mu, delta and every product below are exact and the comparison is against an exactly representable value.
    constexpr float kValue = 0.25F;
    constexpr float kDelta = 0.03125F;
    const float mean = greyLuminanceGain() * kValue;
    const HdrImage beauty = greyImage(kWidth, kHeight, std::vector<float>(kPixels, kValue));

    for (const int samples : counts) {
        const std::vector<float> secondMoment(kPixels, static_cast<float>(samples) * kDelta * kDelta);
        const HdrImage snr = pathtracer::debug::snrAov(beauty, secondMoment.data(), samples, pool);
        // SEM = delta / sqrt(n-1), so the ratio is mu sqrt(n-1) / delta.
        const double expected = (static_cast<double>(mean) * std::sqrt(static_cast<double>(samples) - 1.0)) / kDelta;
        double worst = 0.0;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                worst = std::max(worst, std::fabs(static_cast<double>(texelAt(snr, x, y, 0)) - expected) / expected);
            }
        }
        // One rounding each in the luminance dot, the product under the root, the root itself and the divide.
        PT_EXPECT(ctx, worst <= 4.0 * static_cast<double>(kFloatEpsilon),
                      "n=" + std::to_string(samples) + " reports a relative " + std::to_string(worst) + " off " +
                          std::to_string(expected));
    }
}

// Independent draws of variance sigma^2 have E[M2] = (n-1) sigma^2, so the standard error of the mean must fall as sigma/sqrt(n).
PT_CHECK(snr_falls_with_the_root_of_the_pass_count, Fast, Exact) {
    const std::array<int, 5> counts{4, 16, 64, 256, 1024};
    ctx.plan(static_cast<int>(counts.size()));
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 32;
    constexpr int kHeight = 32;
    constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;
    constexpr float kValue = 0.5F;
    constexpr float kSigma = 0.125F;
    const float mean = greyLuminanceGain() * kValue;
    const HdrImage beauty = greyImage(kWidth, kHeight, std::vector<float>(kPixels, kValue));

    for (const int samples : counts) {
        const std::vector<float> secondMoment(kPixels, static_cast<float>(samples - 1) * kSigma * kSigma);
        const HdrImage snr = pathtracer::debug::snrAov(beauty, secondMoment.data(), samples, pool);
        const double expected = (static_cast<double>(mean) * std::sqrt(static_cast<double>(samples))) / kSigma;
        const double worst = std::fabs(static_cast<double>(texelAt(snr, 0, 0, 0)) - expected) / expected;
        PT_EXPECT(ctx, worst <= 4.0 * static_cast<double>(kFloatEpsilon),
                      "n=" + std::to_string(samples) + " reports a relative " + std::to_string(worst) + " off " +
                          std::to_string(expected));
    }
}

// A variance needs two draws, and a texel every pass agreed on has none: both report zero rather than an infinity or a stale number.
PT_CHECK(snr_is_zero_where_the_ratio_is_undefined, Fast, Exact) {
    ctx.plan(4);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 32;
    constexpr int kHeight = 24;
    constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;
    const HdrImage beauty = greyImage(kWidth, kHeight, std::vector<float>(kPixels, 0.75F));
    const std::vector<float> secondMoment(kPixels, 0.125F);
    const std::vector<float> converged(kPixels, 0.0F);
    // Colour only: alpha is 1 by the broadcast convention, so admitting it here would let an all-ones response pass as zero.
    const auto allZero = [](const HdrImage& image) {
        for (std::size_t pixel = 0; pixel < image.rgba.size() / 4; ++pixel) {
            for (int channel = 0; channel < 3; ++channel) {
                if (image.rgba[(pixel * 4) + static_cast<std::size_t>(channel)] != 0.0F) {
                    return false;
                }
            }
        }
        return true;
    };
    PT_EXPECT(ctx, allZero(pathtracer::debug::snrAov(beauty, nullptr, 64, pool)),
                  "an absent second moment did not read zero");
    PT_EXPECT(ctx, allZero(pathtracer::debug::snrAov(beauty, secondMoment.data(), 0, pool)),
                  "zero passes did not read zero");
    PT_EXPECT(ctx, allZero(pathtracer::debug::snrAov(beauty, secondMoment.data(), 1, pool)),
                  "a single pass did not read zero");
    PT_EXPECT(ctx, allZero(pathtracer::debug::snrAov(beauty, converged.data(), 64, pool)),
                  "a texel with no dispersion did not read zero");
}

// The Morlet bank's admissibility term is a 2-D blur written as two 1-D ones, which is only sound if the mirror separates too.
PT_CHECK(the_blurred_plane_wave_separates_into_its_two_axes, Fast, Exact) {
    const std::array<std::pair<int, int>, 5> shapes{{{24, 16}, {31, 9}, {1, 40}, {40, 1}, {97, 61}}};
    ctx.plan(static_cast<int>(shapes.size()));
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const float variance = pathtracer::debug::innerScaleVariance();
    const double carrier = pathtracer::debug::morletCarrier(variance);
    const int orientations = pathtracer::debug::morletOrientations();

    for (const auto& [width, height] : shapes) {
        double worst = 0.0;
        for (int orientation = 0; orientation < orientations; ++orientation) {
            const double theta = (std::numbers::pi * static_cast<double>(orientation)) / static_cast<double>(orientations);
            const double stepX = carrier * std::cos(theta);
            const double stepY = carrier * std::sin(theta);
            // The whole plane wave through the 2-D blur, which is what the bank would have to compute if it did not separate.
            std::vector<float> real(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
            std::vector<float> imaginary(real.size());
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const double phase = (stepX * static_cast<double>(x)) + (stepY * static_cast<double>(y));
                    real[(static_cast<std::size_t>(y) * width) + static_cast<std::size_t>(x)] = static_cast<float>(std::cos(phase));
                    imaginary[(static_cast<std::size_t>(y) * width) + static_cast<std::size_t>(x)] = static_cast<float>(-std::sin(phase));
                }
            }
            pathtracer::debug::diffuse(real, width, height, variance, pool);
            pathtracer::debug::diffuse(imaginary, width, height, variance, pool);

            std::vector<float> cosX(static_cast<std::size_t>(width));
            std::vector<float> sinX(static_cast<std::size_t>(width));
            std::vector<float> cosY(static_cast<std::size_t>(height));
            std::vector<float> sinY(static_cast<std::size_t>(height));
            for (int x = 0; x < width; ++x) {
                cosX[static_cast<std::size_t>(x)] = static_cast<float>(std::cos(stepX * static_cast<double>(x)));
                sinX[static_cast<std::size_t>(x)] = static_cast<float>(std::sin(stepX * static_cast<double>(x)));
            }
            for (int y = 0; y < height; ++y) {
                cosY[static_cast<std::size_t>(y)] = static_cast<float>(std::cos(stepY * static_cast<double>(y)));
                sinY[static_cast<std::size_t>(y)] = static_cast<float>(std::sin(stepY * static_cast<double>(y)));
            }
            // A width x 1 and a 1 x height plane exercise one separable pass each, the other folding to a no-op on a unit extent.
            pathtracer::debug::diffuse(cosX, width, 1, variance, pool);
            pathtracer::debug::diffuse(sinX, width, 1, variance, pool);
            pathtracer::debug::diffuse(cosY, 1, height, variance, pool);
            pathtracer::debug::diffuse(sinY, 1, height, variance, pool);

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const auto pixel = (static_cast<std::size_t>(y) * width) + static_cast<std::size_t>(x);
                    const float productReal = (cosX[static_cast<std::size_t>(x)] * cosY[static_cast<std::size_t>(y)]) -
                                               (sinX[static_cast<std::size_t>(x)] * sinY[static_cast<std::size_t>(y)]);
                    const float productImaginary = -((sinX[static_cast<std::size_t>(x)] * cosY[static_cast<std::size_t>(y)]) +
                                                     (cosX[static_cast<std::size_t>(x)] * sinY[static_cast<std::size_t>(y)]));
                    worst = std::max({worst, std::fabs(static_cast<double>(productReal - real[pixel])),
                                      std::fabs(static_cast<double>(productImaginary - imaginary[pixel]))});
                }
            }
        }
        // Both routes convolve unit-magnitude values with a unit-mass kernel, so the gap is the two extra roundings of the product.
        PT_EXPECT(ctx, worst <= 2.0 * static_cast<double>(kFloatEpsilon),
                      std::to_string(width) + "x" + std::to_string(height) + " separates to within " + std::to_string(worst));
    }
}

PT_CHECK_MAIN("filter")
