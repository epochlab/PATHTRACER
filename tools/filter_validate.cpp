// Correctness gate for the shared scale space and the Beauty filters: kernel identities, DoG, Colour Opponent, SNR, the Morlet bank, OIDN.

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <cstddef>
#include <numbers>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "check.h"
#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/denoiser.h"
#include "pathtracer/debug/scale_space.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using pathtracer::gfx::HdrImage;
using pathtracer::gfx::PixelLattice;
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
    return (1.0F * pathtracer::scene::cie::sceneLuminanceWeights().r) +
           (1.0F * pathtracer::scene::cie::sceneLuminanceWeights().g) +
           (1.0F * pathtracer::scene::cie::sceneLuminanceWeights().b);
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

// The hexagonal lattice's three axes, 0, 60 and 120 degrees, each one column pitch long: the directions diffuse's passes run along.
constexpr std::array<glm::dvec2, 3> kHexagonalAxes{{{1.0, 0.0}, {0.5, std::numbers::sqrt3 / 2.0}, {-0.5, std::numbers::sqrt3 / 2.0}}};

// Hexagonal transfer at wavevector k: the product of the three axis passes' T(.;2t/3) transfers, each at its axis phase k.u.
[[nodiscard]] double hexagonalTransfer(const std::vector<float>& axisKernel, glm::dvec2 k) {
    double transfer = 1.0;
    for (const glm::dvec2& axis : kHexagonalAxes) {
        transfer *= measuredTransfer(axisKernel, glm::dot(k, axis));
    }
    return transfer;
}

// Film centre of hexagonal cell (x, y) in double, from pixel_lattice.h's formula, so a phase is not rounded through float first.
[[nodiscard]] glm::dvec2 hexagonalCentre(int x, int y) {
    return {static_cast<double>(x) + 0.25 + (0.5 * static_cast<double>(y & 1)), (static_cast<double>(y) + 0.5) * (std::numbers::sqrt3 / 2.0)};
}

// Axial coordinates (q, r) of offset cell (x, y): the lattice basis (1,0), (1/2, sqrt3/2), in which every lattice translation is integer.
[[nodiscard]] glm::ivec2 axialOf(int x, int y) {
    return {x - ((y - (y & 1)) / 2), y};
}

// Offset cell of axial (q, r), the inverse of axialOf.
[[nodiscard]] glm::ivec2 offsetOf(glm::ivec2 axial) {
    return {axial.x + ((axial.y - (axial.y & 1)) / 2), axial.y};
}

[[nodiscard]] std::vector<float> uniformPlane(int width, int height, std::mt19937& rng) {
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    std::vector<float> plane(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (float& value : plane) {
        value = unit(rng);
    }
    return plane;
}

// Per-axis radius of hexagonal diffusion at variance t: each pass carries T(.;2t/3).
[[nodiscard]] int hexagonalAxisRadius(float t) {
    return static_cast<int>(pathtracer::debug::discreteGaussianKernel(static_cast<float>((2.0 * static_cast<double>(t)) / 3.0)).size()) - 1;
}

// Rounding of three float passes of 2r+1 taps each over a field bounded by one: gamma_n per pass, summed.
[[nodiscard]] double hexagonalRoundingBound(float t) {
    return 3.0 * static_cast<double>((2 * hexagonalAxisRadius(t)) + 1) * static_cast<double>(kFloatEpsilon);
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

// The hexagonal inner scale is the same 2^-24 criterion on the lattice's own Nyquist boundary, whose least-attenuated point is M, not K.
PT_CHECK(hexagonal_inner_scale_meets_the_roundoff_on_the_zone_boundary, Fast, Exact) {
    ctx.plan(3);
    const float t = pathtracer::debug::innerScaleVariance(PixelLattice::Hexagonal);
    const double expected = 9.0 * std::numbers::ln2;
    PT_EXPECT(ctx, std::abs(static_cast<double>(t) - expected) <= expected * static_cast<double>(kFloatEpsilon),
              "hexagonal inner scale " + std::to_string(t) + " is not 9 ln2");
    // The zone edge from M (0, 2pi/sqrt3) to K (2pi/3, 2pi/sqrt3); by the lattice's symmetry it stands for the whole boundary.
    const auto analytic = [t](double along) {
        const glm::dvec2 k(along * 2.0 * std::numbers::pi / 3.0, 2.0 * std::numbers::pi / std::numbers::sqrt3);
        double exponent = 0.0;
        for (const glm::dvec2& axis : kHexagonalAxes) {
            exponent += std::cos(glm::dot(k, axis)) - 1.0;
        }
        return std::exp((2.0 * static_cast<double>(t) / 3.0) * exponent);
    };
    double edgePeak = 0.0;
    for (int step = 0; step <= 64; ++step) {
        edgePeak = std::max(edgePeak, analytic(static_cast<double>(step) / 64.0));
    }
    // t is rounded to float, which moves exp(-8t/3) by (8/3) t of its relative ulp.
    const double transferUlps = (8.0 / 3.0) * static_cast<double>(t) * static_cast<double>(kFloatEpsilon);
    PT_EXPECT(ctx, edgePeak == analytic(0.0) && std::abs((edgePeak / pathtracer::debug::kFloat32Roundoff) - 1.0) <= transferUlps,
              "the zone boundary peaks at " + std::to_string(edgePeak / pathtracer::debug::kFloat32Roundoff) + " x 2^-24, not at M");
    // The shipped kernel's transfer at M: each axis carries truncation 2^-24 plus an ulp, as discrete_gaussian_transfer_is_exact bounds.
    const std::vector<float> axis = pathtracer::debug::discreteGaussianKernel(static_cast<float>((2.0 * static_cast<double>(t)) / 3.0));
    const double measured = hexagonalTransfer(axis, glm::dvec2(0.0, 2.0 * std::numbers::pi / std::numbers::sqrt3));
    const double bound = pathtracer::debug::kFloat32Roundoff + (3.0 * (pathtracer::debug::kFloat32Roundoff + static_cast<double>(kFloatEpsilon)));
    PT_EXPECT(ctx, std::abs(measured) <= bound, "measured transfer at M " + std::to_string(measured) + " exceeds " + std::to_string(bound));
}

// Plane waves are the lattice's eigenfunctions: inside the reach, hexagonal diffusion scales each by the three axis transfers' product.
PT_CHECK(hexagonal_diffusion_transfer_is_the_three_axis_product, Fast, Exact) {
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 112;
    constexpr int kHeight = 96;
    const std::vector<double> anglesDegrees{0.0, 15.0, 30.0, 90.0};
    const std::vector<double> fractionsOfM{0.2, 0.5, 0.9};
    ctx.plan(static_cast<int>(anglesDegrees.size() * fractionsOfM.size()));
    const float t = pathtracer::debug::innerScaleVariance(PixelLattice::Hexagonal);
    const std::vector<float> axis = pathtracer::debug::discreteGaussianKernel(static_cast<float>((2.0 * static_cast<double>(t)) / 3.0));
    const glm::ivec2 reach = pathtracer::debug::diffusionReach(t, PixelLattice::Hexagonal);
    // Against the shipped kernel's own transfer, so truncation cancels; what is left is rounding of the stored wave and the passes.
    const double bound = hexagonalRoundingBound(t) + static_cast<double>(kFloatEpsilon);
    for (const double degrees : anglesDegrees) {
        for (const double fraction : fractionsOfM) {
            const double magnitude = fraction * 2.0 * std::numbers::pi / std::numbers::sqrt3;
            const glm::dvec2 k = magnitude * glm::dvec2(std::cos(degrees * std::numbers::pi / 180.0), std::sin(degrees * std::numbers::pi / 180.0));
            std::vector<float> plane(static_cast<std::size_t>(kWidth) * kHeight);
            for (int y = 0; y < kHeight; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    plane[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)] =
                        static_cast<float>(std::cos(glm::dot(k, hexagonalCentre(x, y)) + 0.3));
                }
            }
            pathtracer::debug::diffuse(plane, kWidth, kHeight, t, pool, false, PixelLattice::Hexagonal);
            const double transfer = hexagonalTransfer(axis, k);
            double worst = 0.0;
            for (int y = reach.y; y < kHeight - reach.y; ++y) {
                for (int x = reach.x; x < kWidth - reach.x; ++x) {
                    const double expected = transfer * std::cos(glm::dot(k, hexagonalCentre(x, y)) + 0.3);
                    worst = std::max(worst, std::abs(static_cast<double>(plane[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)]) - expected));
                }
            }
            PT_EXPECT(ctx, worst <= bound,
                      std::to_string(degrees) + " deg, " + std::to_string(fraction) + " M: deviates by " + std::to_string(worst) + " bound " +
                          std::to_string(bound));
        }
    }
}

// Every axis line of an axial-affine field is affine, so each centre-relative pass cancels exactly and the interior is untouched.
PT_CHECK(hexagonal_diffusion_reproduces_affine_fields, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 112;
    constexpr int kHeight = 96;
    std::vector<float> reference(static_cast<std::size_t>(kWidth) * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const glm::ivec2 axial = axialOf(x, y);
            reference[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)] = static_cast<float>(8 + (2 * axial.x) + (3 * axial.y));
        }
    }
    std::vector<float> plane = reference;
    const float t = pathtracer::debug::innerScaleVariance(PixelLattice::Hexagonal);
    const glm::ivec2 reach = pathtracer::debug::diffusionReach(t, PixelLattice::Hexagonal);
    pathtracer::debug::diffuse(plane, kWidth, kHeight, t, pool, false, PixelLattice::Hexagonal);
    bool identical = reach.y < kHeight - reach.y && reach.x < kWidth - reach.x;
    for (int y = reach.y; y < kHeight - reach.y; ++y) {
        for (int x = reach.x; x < kWidth - reach.x; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x);
            identical = identical && plane[pixel] == reference[pixel];
        }
    }
    PT_EXPECT(ctx, identical, "hexagonal diffusion must leave an affine field bit-identical in the interior");
}

// The lattice fold extends the frame by its own symmetry, which diffusion keeps, so t then 3t equals 4t right up to the border.
PT_CHECK(hexagonal_diffusion_semigroup_holds_to_the_border, Fast, Exact) {
    ctx.plan(2);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 41;
    constexpr int kHeight = 30;
    std::mt19937 rng(static_cast<std::uint32_t>(ctx.subSeed("hexagonal-semigroup")));
    const std::vector<float> source = uniformPlane(kWidth, kHeight, rng);
    const float t = pathtracer::debug::innerScaleVariance(PixelLattice::Hexagonal);
    // Rounding of nine passes, plus each axis' composed truncation (2 x 2^-24 + an ulp, as discrete_gaussian_semigroup_is_exact bounds).
    const double bound = hexagonalRoundingBound(t) + hexagonalRoundingBound(3.0F * t) + hexagonalRoundingBound(4.0F * t) +
                         (3.0 * ((2.0 * pathtracer::debug::kFloat32Roundoff) + static_cast<double>(kFloatEpsilon)));
    for (const bool wraps : {false, true}) {
        std::vector<float> stepped = source;
        pathtracer::debug::diffuse(stepped, kWidth, kHeight, t, pool, wraps, PixelLattice::Hexagonal);
        pathtracer::debug::diffuse(stepped, kWidth, kHeight, 3.0F * t, pool, wraps, PixelLattice::Hexagonal);
        std::vector<float> direct = source;
        pathtracer::debug::diffuse(direct, kWidth, kHeight, 4.0F * t, pool, wraps, PixelLattice::Hexagonal);
        double worst = 0.0;
        for (std::size_t pixel = 0; pixel < source.size(); ++pixel) {
            worst = std::max(worst, std::abs(static_cast<double>(stepped[pixel]) - static_cast<double>(direct[pixel])));
        }
        PT_EXPECT(ctx, worst <= bound,
                  std::string(wraps ? "wrapped" : "mirrored") + ": t then 3t differs from 4t by " + std::to_string(worst) + " bound " +
                      std::to_string(bound));
    }
}

// Frame symmetries that are lattice symmetries commute with diffusion: 180 degree turns and cyclic rolls bitwise, a row flip to rounding.
PT_CHECK(hexagonal_diffusion_commutes_with_the_frame_symmetries, Fast, Exact) {
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const float t = pathtracer::debug::innerScaleVariance(PixelLattice::Hexagonal);
    std::mt19937 rng(static_cast<std::uint32_t>(ctx.subSeed("hexagonal-symmetry")));
    const auto at = [](const std::vector<float>& plane, int width, int x, int y) {
        return plane[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x)];
    };
    // Returns the worst |f(S(I)) - S(f(I))| for the index map S over a width x height frame.
    const auto worstError = [&](int width, int height, bool wraps, const auto& map) {
        const std::vector<float> source = uniformPlane(width, height, rng);
        std::vector<float> mapped(source.size());
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const glm::ivec2 to = map(x, y);
                mapped[(static_cast<std::size_t>(to.y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(to.x)] = at(source, width, x, y);
            }
        }
        std::vector<float> diffused = source;
        pathtracer::debug::diffuse(diffused, width, height, t, pool, wraps, PixelLattice::Hexagonal);
        pathtracer::debug::diffuse(mapped, width, height, t, pool, wraps, PixelLattice::Hexagonal);
        double worst = 0.0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const glm::ivec2 to = map(x, y);
                worst = std::max(worst, std::abs(static_cast<double>(at(mapped, width, to.x, to.y)) - static_cast<double>(at(diffused, width, x, y))));
            }
        }
        return worst;
    };
    // An even row count makes the index turn (w-1-x, h-1-y) the frame's half turn, which maps every axis onto itself.
    const double turn = worstError(37, 24, false, [](int x, int y) { return glm::ivec2(36 - x, 23 - y); });
    PT_EXPECT(ctx, turn == 0.0, "a half turn moved hexagonal diffusion by " + std::to_string(turn));
    const double roll = worstError(40, 24, true, [](int x, int y) { return glm::ivec2((x + 13) % 40, y); });
    PT_EXPECT(ctx, roll == 0.0, "a cyclic roll moved wrapped hexagonal diffusion by " + std::to_string(roll));
    // An odd row count makes y -> h-1-y a lattice mirror; it swaps the 60 and 120 degree passes, so their order rounds differently.
    const double flip = worstError(37, 25, false, [](int x, int y) { return glm::ivec2(x, 24 - y); });
    const double bound = 2.0 * hexagonalRoundingBound(t);
    PT_EXPECT(ctx, flip <= bound, "a row flip moved hexagonal diffusion by " + std::to_string(flip) + " bound " + std::to_string(bound));
}

// Far from any border an impulse spreads isotropically: six-fold symmetric, unit mass and covariance t I in film units.
PT_CHECK(hexagonal_impulse_response_is_six_fold_symmetric, Fast, Exact) {
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    const float t = 4.0F * pathtracer::debug::innerScaleVariance(PixelLattice::Hexagonal);
    const glm::ivec2 reach = pathtracer::debug::diffusionReach(t, PixelLattice::Hexagonal);
    const int width = (4 * reach.x) + 1;
    const int height = (4 * reach.y) + 2;
    const glm::ivec2 centre(width / 2, height / 2);
    std::vector<float> plane(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0.0F);
    plane[(static_cast<std::size_t>(centre.y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(centre.x)] = 1.0F;
    pathtracer::debug::diffuse(plane, width, height, t, pool, false, PixelLattice::Hexagonal);
    const glm::ivec2 origin = axialOf(centre.x, centre.y);
    const auto response = [&](glm::ivec2 offset) {
        const glm::ivec2 cell = offsetOf(origin + offset);
        return static_cast<double>(plane[(static_cast<std::size_t>(cell.y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(cell.x)]);
    };
    const int radius = hexagonalAxisRadius(t);
    double asymmetry = 0.0;
    double mass = 0.0;
    glm::dmat2 covariance(0.0);
    for (int r = -2 * radius; r <= 2 * radius; ++r) {
        for (int q = -2 * radius; q <= 2 * radius; ++q) {
            const glm::ivec2 offset(q, r);
            // A sixth of a turn in axial coordinates: (q, r) -> (-r, q + r).
            asymmetry = std::max(asymmetry, std::abs(response(glm::ivec2(-r, q + r)) - response(offset)));
            const double weight = response(offset);
            const glm::dvec2 film = (static_cast<double>(q) * kHexagonalAxes[0]) + (static_cast<double>(r) * kHexagonalAxes[1]);
            mass += weight;
            covariance += weight * glm::outerProduct(film, film);
        }
    }
    const double rounding = 2.0 * hexagonalRoundingBound(t);
    PT_EXPECT(ctx, asymmetry <= rounding, "a sixth turn moves the impulse response by " + std::to_string(asymmetry));
    PT_EXPECT(ctx, std::abs(mass - 1.0) <= 3.0 * static_cast<double>(kFloatEpsilon), "impulse mass is " + std::to_string(mass));
    // Each axis' discarded second moment, as discrete_gaussian_variance_is_t bounds it, scaled to the pass' variance 2t/3.
    const double axisVariance = (2.0 * static_cast<double>(t)) / 3.0;
    const double bound = 3.0 * ((pathtracer::debug::kFloat32Roundoff * (radius + 1.0) * (radius + 1.0) / axisVariance) + static_cast<double>(kFloatEpsilon));
    const double error = std::max({std::abs((covariance[0][0] / static_cast<double>(t)) - 1.0), std::abs((covariance[1][1] / static_cast<double>(t)) - 1.0),
                                   std::abs(covariance[0][1] / static_cast<double>(t))});
    PT_EXPECT(ctx, error <= bound, "covariance deviates from t I by " + std::to_string(error) + " bound " + std::to_string(bound));
}

// The boundary oracle without the padding code: a frame's response is the free-space kernel summed over the impulse's reflection orbit.
PT_CHECK(hexagonal_diffusion_matches_the_method_of_images, Fast, Exact) {
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 9;
    constexpr int kHeight = 7;
    const std::vector<glm::ivec2> sources{{0, 0}, {1, 1}, {8, 6}, {4, 3}};
    ctx.plan(2 * static_cast<int>(sources.size()));
    const float t = pathtracer::debug::innerScaleVariance(PixelLattice::Hexagonal);
    const glm::ivec2 reach = pathtracer::debug::diffusionReach(t, PixelLattice::Hexagonal);
    // Free-space kernel on a canvas whose every border is twice the reach from the impulse, so no image of it reaches the centre's support.
    const int canvasWidth = (4 * reach.x) + 1;
    const int canvasHeight = (4 * reach.y) + 2;
    const glm::ivec2 canvasCentre(canvasWidth / 2, canvasHeight / 2);
    std::vector<float> kernel(static_cast<std::size_t>(canvasWidth) * static_cast<std::size_t>(canvasHeight), 0.0F);
    kernel[(static_cast<std::size_t>(canvasCentre.y) * static_cast<std::size_t>(canvasWidth)) + static_cast<std::size_t>(canvasCentre.x)] = 1.0F;
    pathtracer::debug::diffuse(kernel, canvasWidth, canvasHeight, t, pool, false, PixelLattice::Hexagonal);
    const glm::ivec2 origin = axialOf(canvasCentre.x, canvasCentre.y);
    const auto free = [&](glm::ivec2 axialOffset) {
        const glm::ivec2 cell = offsetOf(origin + axialOffset);
        const bool inside = cell.x >= 0 && cell.x < canvasWidth && cell.y >= 0 && cell.y < canvasHeight;
        return inside ? static_cast<double>(kernel[(static_cast<std::size_t>(cell.y) * static_cast<std::size_t>(canvasWidth)) + static_cast<std::size_t>(cell.x)]) : 0.0;
    };
    const double bound = 2.0 * hexagonalRoundingBound(t);
    for (const bool wraps : {false, true}) {
        for (const glm::ivec2 source : sources) {
            // The orbit, in half-pitch column j = 2x + parity and row: rows mirror with period 2h-2, j mirrors with 4w-2 or wraps with 2w.
            const int rowPeriod = (2 * kHeight) - 2;
            const int columnPeriod = wraps ? 2 * kWidth : (4 * kWidth) - 2;
            const int column = (2 * source.x) + (source.y & 1);
            std::set<std::pair<int, int>> orbit;
            for (int k = -2 * ((reach.y / rowPeriod) + 2); k <= 2 * ((reach.y / rowPeriod) + 2); ++k) {
                for (int m = -2 * ((reach.x / columnPeriod) + 2); m <= 2 * ((reach.x / columnPeriod) + 2); ++m) {
                    for (const int row : {source.y + (k * rowPeriod), -source.y + (k * rowPeriod)}) {
                        for (const int j : {column + (m * columnPeriod), (wraps ? column : -column) + (m * columnPeriod)}) {
                            orbit.emplace(j, row);
                        }
                    }
                }
            }
            std::vector<float> plane(static_cast<std::size_t>(kWidth) * kHeight, 0.0F);
            plane[(static_cast<std::size_t>(source.y) * kWidth) + static_cast<std::size_t>(source.x)] = 1.0F;
            pathtracer::debug::diffuse(plane, kWidth, kHeight, t, pool, wraps, PixelLattice::Hexagonal);
            double worst = 0.0;
            for (int y = 0; y < kHeight; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    double expected = 0.0;
                    for (const auto& [j, row] : orbit) {
                        expected += free(axialOf(x, y) - axialOf((j - (row & 1)) / 2, row));
                    }
                    worst = std::max(worst, std::abs(static_cast<double>(plane[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)]) - expected));
                }
            }
            PT_EXPECT(ctx, worst <= bound,
                      std::string(wraps ? "wrapped" : "mirrored") + " impulse at " + std::to_string(source.x) + "," + std::to_string(source.y) +
                          " misses its images by " + std::to_string(worst) + " bound " + std::to_string(bound));
        }
    }
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

// A grating is two plane waves, eigenfunctions of every pass, so each bank channel and so the envelope have a closed form per texel.
PT_CHECK(gabor_matches_the_analytic_morlet_response, Fast, Exact) {
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kSize = 96;
    constexpr double kMean = 0.5;
    constexpr double kAmplitude = 0.25;
    const float variance = pathtracer::debug::innerScaleVariance();
    const std::vector<float> kernel = pathtracer::debug::discreteGaussianKernel(variance);
    const int radius = static_cast<int>(kernel.size()) - 1;
    const double carrier = static_cast<double>(pathtracer::debug::morletCarrier(variance));
    const int orientations = pathtracer::debug::morletOrientations();
    const double gain = static_cast<double>(greyLuminanceGain());

    // diffuse's centre-relative pass applies 1 + sum 2 w_n (cos nw - 1), the exact transfer of its float taps.
    const auto passTransfer = [&](double omega) {
        double sum = 1.0;
        for (int n = 1; n <= radius; ++n) {
            sum += 2.0 * static_cast<double>(kernel[static_cast<std::size_t>(n)]) * (std::cos(n * omega) - 1.0);
        }
        return sum;
    };
    const auto transfer = [&](const glm::dvec2& k) { return passTransfer(k.x) * passTransfer(k.y); };
    const auto bankCarrier = [&](int index) {
        const double theta = (std::numbers::pi * index) / orientations;
        return carrier * glm::dvec2(std::cos(theta), std::sin(theta));
    };

    struct Grating {
        double omega;
        double angle;
        bool halfResponse;  // a point the bank's design claims at half response or better
    };
    std::vector<Grating> gratings;
    for (int i = 0; i < orientations; ++i) {
        const double theta = (std::numbers::pi * i) / orientations;
        gratings.push_back({carrier, theta, false});
        gratings.push_back({carrier, theta + (std::numbers::pi / (2.0 * orientations)), true});
    }
    // The Gaussian envelope halves at sqrt(2 ln2)/sigma from the carrier, the radial edges of kMorletOctaves' band.
    const double radialHalfWidth = std::sqrt(2.0 * std::numbers::ln2 / static_cast<double>(variance));
    gratings.push_back({carrier - radialHalfWidth, 0.0, true});
    gratings.push_back({carrier + radialHalfWidth, 0.0, true});
    const auto halfResponses = std::count_if(gratings.begin(), gratings.end(), [](const Grating& g) { return g.halfResponse; });
    ctx.plan(static_cast<int>(gratings.size() + static_cast<std::size_t>(halfResponses)));

    // Higham 3.1: the longest rounding chain is the admissibility product, a 1-D table blur times the 2-D lowpass, 3R+16, +4 to the root.
    const double chain = (3.0 * radius) + 20.0;
    const double gammaChain = (chain * pathtracer::debug::kFloat32Roundoff) / (1.0 - (chain * pathtracer::debug::kFloat32Roundoff));
    // Each part is two terms of absolute mass 9M (3 per centre-relative pass), so 18 sqrt2 + 1 per orientation, doubled for the envelope.
    const double bound = 2.0 * ((18.0 * std::numbers::sqrt2) + 1.0) * gammaChain * gain * (kMean + kAmplitude);

    for (const Grating& grating : gratings) {
        const glm::dvec2 k = grating.omega * glm::dvec2(std::cos(grating.angle), std::sin(grating.angle));
        std::vector<float> plane(static_cast<std::size_t>(kSize) * kSize);
        for (int y = 0; y < kSize; ++y) {
            for (int x = 0; x < kSize; ++x) {
                plane[(static_cast<std::size_t>(y) * kSize) + static_cast<std::size_t>(x)] =
                    static_cast<float>(kMean + (kAmplitude * std::cos(glm::dot(k, glm::dvec2(x, y)))));
            }
        }
        const HdrImage gabor = pathtracer::debug::gaborAov(greyImage(kSize, kSize, plane), pool);
        // Per orientation the sidebands k-q and -(k+q) pass H(k-+q) less the admissibility term H(q)H(k); the mean cancels exactly.
        std::vector<std::array<double, 2>> sidebands;
        for (int i = 0; i < orientations; ++i) {
            const glm::dvec2 q = bankCarrier(i);
            const double admissibility = transfer(q) * transfer(k);
            sidebands.push_back({transfer(k - q) - admissibility, transfer(k + q) - admissibility});
        }
        double worst = 0.0;
        double weakest = std::numeric_limits<double>::infinity();
        // Interior only: R from every edge no tap reaches the mirror, where the plane waves stop being eigenfunctions.
        for (int y = radius; y < kSize - radius; ++y) {
            for (int x = radius; x < kSize - radius; ++x) {
                const glm::dvec2 r(x, y);
                double peak = 0.0;
                for (int i = 0; i < orientations; ++i) {
                    const glm::dvec2 q = bankCarrier(i);
                    const std::array<double, 2>& h = sidebands[static_cast<std::size_t>(i)];
                    peak = std::max(peak, std::abs((h[0] * std::polar(1.0, glm::dot(k - q, r))) +
                                                   (h[1] * std::polar(1.0, -glm::dot(k + q, r)))));
                }
                // 2 max|c_i| with c_i = (A/2)(...): the envelope is A times the peak sideband sum.
                const double expected = gain * kAmplitude * peak;
                const double measured = static_cast<double>(texelAt(gabor, x, y, 0));
                worst = std::max(worst, std::abs(measured - expected));
                weakest = std::min(weakest, measured);
            }
        }
        const std::string label = "w=" + std::to_string(grating.omega) + " angle=" + std::to_string(grating.angle);
        PT_EXPECT(ctx, worst <= bound, label + " deviates by " + std::to_string(worst) + ", bound " + std::to_string(bound));
        if (grating.halfResponse) {
            PT_EXPECT(ctx, weakest >= 0.5 * gain * kAmplitude,
                      label + " reads " + std::to_string(weakest / (gain * kAmplitude)) + " of its amplitude, under half");
        }
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
    // The roll moves the carrier's phase, so demodulation, both blurs and the baseband round anew: 2R+1 ulps apiece, x2 envelope.
    const auto radius = static_cast<double>(pathtracer::debug::discreteGaussianKernel(pathtracer::debug::innerScaleVariance()).size() - 1);
    const double bound = 2.0 * 4.0 * ((2.0 * radius) + 1.0) * static_cast<double>(kFloatEpsilon * greyLuminanceGain());
    const float gabor = worstShiftError(pathtracer::debug::gaborAov(image, pool, true), pathtracer::debug::gaborAov(rolled, pool, true));
    PT_EXPECT(ctx, gabor <= bound, "Gabor moved by " + std::to_string(gabor) + " under a cyclic shift, bound " + std::to_string(bound));
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
