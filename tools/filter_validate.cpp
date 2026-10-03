// Correctness gate for the shared scale space and the Beauty filters: kernel identities, DoG, Colour Opponent, SNR, the Morlet bank.

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
#include "pathtracer/debug/scale_space.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

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
    HdrImage image = pathtracer::gfx::makeImage(width, height, pathtracer::gfx::kRgbChannels);
    for (std::size_t pixel = 0; pixel < plane.size(); ++pixel) {
        std::fill_n(image.texels.begin() + static_cast<std::ptrdiff_t>(pixel * pathtracer::gfx::kRgbChannels),
                    pathtracer::gfx::kRgbChannels, plane[pixel]);
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

// Radius in base pixels inside which a DoG sample sees no mirrored tap: the fine diffusion's radius plus the coarse step's.
[[nodiscard]] int dogSupportRadius() {
    const float fine = pathtracer::debug::innerScaleVariance();
    const auto radius = [](float t) { return static_cast<int>(pathtracer::debug::discreteGaussianKernel(t).size()) - 1; };
    return radius(fine) + radius((4.0F * fine) - fine);
}

[[nodiscard]] float texelAt(const HdrImage& image, int x, int y, int channel) {
    return image.texels[(((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) +
                          static_cast<std::size_t>(x)) * static_cast<std::size_t>(image.channels)) +
                        static_cast<std::size_t>(channel)];
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

// T(t1) * T(t2) = T(t1+t2) on the grid; DoG reaches its coarse octave by a second step and is only correct because this composition holds.
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

// A constant field survives the mirror exactly, so DoG must read zero at every pixel, border included.
PT_CHECK(dog_is_zero_on_a_constant_field, Fast, Exact) {
    ctx.plan(1);
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
}

// Both diffusions reproduce an affine field, so DoG may not fire on a smooth gradient anywhere its support avoids the mirror.
PT_CHECK(dog_is_flat_on_an_affine_field, Fast, Exact) {
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 256;
    constexpr int kHeight = 112;
    const HdrImage image = greyImage(kWidth, kHeight, affinePlane(kWidth, kHeight));
    const int radius = dogSupportRadius();
    PT_EXPECT(ctx, (2 * radius) < kHeight,
                  "support radius " + std::to_string(radius) + " leaves no interior in " + std::to_string(kHeight) + " rows");

    const HdrImage dog = pathtracer::debug::dogAov(image, pool);
    float worstDog = 0.0F;
    for (int y = radius; y < kHeight - radius; ++y) {
        for (int x = radius; x < kWidth - radius; ++x) {
            worstDog = std::max(worstDog, std::fabs(texelAt(dog, x, y, 0)));
        }
    }
    // The Rec.709 reduction rounds, so the plane is affine only to an ulp; carried through the widest step that is the bound, not zero.
    const float peak = greyLuminanceGain() * static_cast<float>(8 + (2 * (kWidth - 1)) + (3 * (kHeight - 1)));
    const auto taps = static_cast<double>((2 * (static_cast<int>(pathtracer::debug::discreteGaussianKernel(
                                                    3.0F * pathtracer::debug::innerScaleVariance()).size()) - 1)) + 1);
    const double quantisation = taps * static_cast<double>(peak) * static_cast<double>(kFloatEpsilon);
    PT_EXPECT(ctx, static_cast<double>(worstDog) <= quantisation,
                  "interior DoG " + std::to_string(worstDog) + " exceeds the plane quantisation " + std::to_string(quantisation));
}

// The band exists exactly when the coarse step's kernel fits the frame: one row short is empty, the fitting frame is not.
PT_CHECK(dog_is_empty_exactly_below_its_coarse_support, Fast, Exact) {
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const float fine = pathtracer::debug::innerScaleVariance();
    const int support = (2 * (static_cast<int>(pathtracer::debug::discreteGaussianKernel((4.0F * fine) - fine).size()) - 1)) + 1;
    constexpr int kWidth = 128;
    const auto peakResponse = [&pool](int height) {
        std::vector<float> plane(static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(height), 0.0F);
        // A unit impulse at the centre, which every non-empty DoG answers with a nonzero centre tap.
        plane[(static_cast<std::size_t>(height / 2) * kWidth) + (kWidth / 2)] = 1.0F;
        const HdrImage dog = pathtracer::debug::dogAov(greyImage(kWidth, height, plane), pool);
        float peak = 0.0F;
        for (const float response : dog.texels) {
            peak = std::max(peak, std::fabs(response));
        }
        return peak;
    };
    PT_EXPECT(ctx, peakResponse(support - 1) == 0.0F, "a frame one row short of the coarse support must yield an empty band");
    PT_EXPECT(ctx, peakResponse(support) > 0.0F, "a frame exactly the coarse support must yield a band");
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

// Fine minus coarse is the centre-surround sign: a bright blob reads positive and a dark one negative, the convention the preview shows.
PT_CHECK(dog_polarity_follows_the_blob, Fast, Exact) {
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
    const float brightDog = texelAt(pathtracer::debug::dogAov(greyImage(kWidth, kHeight, bright), pool), centreX, centreY, 0);
    PT_EXPECT(ctx, brightDog > 0.0F, "bright-on-dark DoG read " + std::to_string(brightDog));

    std::vector<float> dark(pixels, 1.25F);
    dark[centre] -= 1.0F;
    pathtracer::debug::diffuse(dark, kWidth, kHeight, pathtracer::debug::innerScaleVariance(), pool);
    const float darkDog = texelAt(pathtracer::debug::dogAov(greyImage(kWidth, kHeight, dark), pool), centreX, centreY, 0);
    PT_EXPECT(ctx, darkDog < 0.0F, "dark-on-bright DoG read " + std::to_string(darkDog));
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
    HdrImage image = pathtracer::gfx::makeImage(kWidth, kHeight, pathtracer::gfx::kRgbChannels);
    HdrImage scaled = image;
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(kWidth) * kHeight; ++pixel) {
        for (int channel = 0; channel < pathtracer::gfx::kRgbChannels; ++channel) {
            const float value = 0.125F + (0.0625F * static_cast<float>((pixel * 7 + channel * 5) % 13));
            image.texels[(pixel * pathtracer::gfx::kRgbChannels) + static_cast<std::size_t>(channel)] = value;
            scaled.texels[(pixel * pathtracer::gfx::kRgbChannels) + static_cast<std::size_t>(channel)] = value * 256.0F;
        }
    }
    const HdrImage base = pathtracer::debug::colourOpponentAov(image, pool);
    const HdrImage gained = pathtracer::debug::colourOpponentAov(scaled, pool);
    PT_EXPECT(ctx, base.texels == gained.texels, "a 256x gain changed the opponent response");
}

// The two axes are the L-versus-M and S-versus-(L+M) cardinal directions, so each Rec.709 primary must sit on its own side of white.
PT_CHECK(colour_opponent_separates_the_rec709_primaries, Fast, Exact) {
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const auto axis = [&pool](const glm::vec3& rgb, int channel) {
        const HdrImage image{1, 1, pathtracer::gfx::kRgbChannels, {rgb.r, rgb.g, rgb.b}};
        return texelAt(pathtracer::debug::colourOpponentAov(image, pool), 0, 0, channel);
    };
    const float red = axis({1.0F, 0.0F, 0.0F}, 0);
    const float green = axis({0.0F, 1.0F, 0.0F}, 0);
    const float blue = axis({0.0F, 0.0F, 1.0F}, 1);
    PT_EXPECT(ctx, red > 0.0F, "Rec.709 red reads R-G " + std::to_string(red));
    PT_EXPECT(ctx, green < 0.0F, "Rec.709 green reads R-G " + std::to_string(green));
    PT_EXPECT(ctx, blue > 0.0F, "Rec.709 blue reads B-Y " + std::to_string(blue));
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
    const auto allZero = [](const HdrImage& image) {
        return std::all_of(image.texels.begin(), image.texels.end(), [](float value) { return value == 0.0F; });
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

// On a cycle the lattice harmonics are the heat kernel's eigenfunctions: periodic diffusion scales cos(2 pi m x / W) by its transfer.
PT_CHECK(periodic_diffusion_keeps_lattice_harmonics_as_eigenfunctions, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 64;
    constexpr int kHarmonic = 3;
    const float t = pathtracer::debug::innerScaleVariance();
    const std::vector<float> kernel = pathtracer::debug::discreteGaussianKernel(t);
    const double omega = 2.0 * std::numbers::pi * kHarmonic / kWidth;
    std::vector<float> row(kWidth);
    for (int x = 0; x < kWidth; ++x) {
        row[static_cast<std::size_t>(x)] = static_cast<float>(std::cos(omega * x));
    }
    pathtracer::debug::diffuse(row, kWidth, 1, t, pool, /*wrapsHorizontally=*/true);
    // Against the shipped kernel's own transfer, so truncation cancels and what is left is the 2R+1 tap accumulation's rounding.
    const double transfer = measuredTransfer(kernel, omega);
    const double bound = static_cast<double>(2 * kernel.size() - 1) * static_cast<double>(kFloatEpsilon);
    double worst = 0.0;
    for (int x = 0; x < kWidth; ++x) {
        worst = std::max(worst, std::abs(static_cast<double>(row[static_cast<std::size_t>(x)]) - (transfer * std::cos(omega * x))));
    }
    PT_EXPECT(ctx, worst <= bound, "worst harmonic deviation " + std::to_string(worst) + " bound " + std::to_string(bound));
}

// A wrapping image has no seam to the filters: rolling its columns rolls Sobel and DoG bit for bit, and Gabor to its own rounding.
PT_CHECK(wrapped_filters_commute_with_a_cyclic_column_shift, Fast, Exact) {
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    constexpr int kShift = 23;
    std::mt19937 rng(static_cast<std::uint32_t>(ctx.subSeed("cyclic-shift")));
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    std::vector<float> plane(static_cast<std::size_t>(kWidth) * kHeight);
    for (float& value : plane) {
        value = unit(rng);
    }
    std::vector<float> rolledPlane(plane.size());
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            rolledPlane[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>((x + kShift) % kWidth)] =
                plane[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)];
        }
    }
    const HdrImage image = greyImage(kWidth, kHeight, plane);
    const HdrImage rolled = greyImage(kWidth, kHeight, rolledPlane);
    // Worst |f(roll(I)) - roll(f(I))| over the frame.
    const auto worstShiftError = [&](const HdrImage& original, const HdrImage& shifted) {
        float worst = 0.0F;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                worst = std::max(worst, std::fabs(texelAt(shifted, (x + kShift) % kWidth, y, 0) - texelAt(original, x, y, 0)));
            }
        }
        return worst;
    };
    const float sobel = worstShiftError(pathtracer::debug::sobelAov(image, pool, true), pathtracer::debug::sobelAov(rolled, pool, true));
    PT_EXPECT(ctx, sobel == 0.0F, "Sobel moved by " + std::to_string(sobel) + " under a cyclic shift");
    const float dog = worstShiftError(pathtracer::debug::dogAov(image, pool, true), pathtracer::debug::dogAov(rolled, pool, true));
    PT_EXPECT(ctx, dog == 0.0F, "DoG moved by " + std::to_string(dog) + " under a cyclic shift");
    // The roll moves the carrier's phase origin, so demodulation, both blur passes and the baseband each round anew: 2R+1 ulps apiece.
    const auto radius = static_cast<double>(pathtracer::debug::discreteGaussianKernel(pathtracer::debug::innerScaleVariance()).size() - 1);
    const double bound = 4.0 * ((2.0 * radius) + 1.0) * static_cast<double>(kFloatEpsilon * greyLuminanceGain());
    const float gabor = worstShiftError(pathtracer::debug::gaborAov(image, pool, true), pathtracer::debug::gaborAov(rolled, pool, true));
    PT_EXPECT(ctx, gabor <= bound, "Gabor moved by " + std::to_string(gabor) + " under a cyclic shift, bound " + std::to_string(bound));
}

PT_CHECK_MAIN("filter")
