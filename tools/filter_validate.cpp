// Correctness gate for the shared scale space and the AOVs built on it: kernel identities, the octave cascade, then DoG and LoG.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
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

    const HdrImage log = pathtracer::debug::logAov(flat, glm::radians(30.0F), pool);
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
    const HdrImage log = pathtracer::debug::logAov(image, glm::radians(30.0F), pool);
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

// A bright blob has a negative Laplacian at its centre, so the reported polarity must read +1 there and -1 for the inverted field.
PT_CHECK(log_polarity_separates_bright_and_dark_blobs, Fast, Exact) {
    ctx.plan(2);
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
    const HdrImage brightLog = pathtracer::debug::logAov(greyImage(kWidth, kHeight, bright), glm::radians(30.0F), pool);
    PT_EXPECT(ctx, texelAt(brightLog, centreX, centreY, 2) == 1.0F,
                  "bright-on-dark polarity read " + std::to_string(texelAt(brightLog, centreX, centreY, 2)));

    std::vector<float> dark(pixels, 1.25F);
    dark[centre] -= 1.0F;
    pathtracer::debug::diffuse(dark, kWidth, kHeight, pathtracer::debug::innerScaleVariance(), pool);
    const HdrImage darkLog = pathtracer::debug::logAov(greyImage(kWidth, kHeight, dark), glm::radians(30.0F), pool);
    PT_EXPECT(ctx, texelAt(darkLog, centreX, centreY, 2) == -1.0F,
                  "dark-on-bright polarity read " + std::to_string(texelAt(darkLog, centreX, centreY, 2)));
}

PT_CHECK_MAIN("filter")
