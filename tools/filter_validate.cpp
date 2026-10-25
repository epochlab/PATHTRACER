// Correctness gate for the shared scale space and the Beauty filters: kernel identities, Colour Opponent, SNR, the Morlet bank, OIDN.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <limits>
#include <cstddef>
#include <numbers>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "check.h"
#include "stats.h"
#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/denoiser.h"
#include "pathtracer/debug/scale_space.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/path_trace_driver.h"
#include "pathtracer/scene/path_tracer.h"
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

// T(t1) * T(t2) = T(t1+t2) on the grid; Contrast Sensitivity steps octave to octave and is only correct because this composition holds.
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

// A grating is two plane waves, eigenfunctions of every pass, so each bank channel and so the envelope have a closed form per texel.
PT_CHECK(gabor_matches_the_analytic_morlet_response, Fast, Exact) {
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kSize = 96;
    constexpr double kMean = 0.5;
    constexpr double kAmplitude = 0.25;
    const double gain = static_cast<double>(greyLuminanceGain());
    const int orientations = pathtracer::debug::morletOrientations();
    // Per lattice: two gratings per orientation and the two radial edges, all but the on-carrier ones also checked for half response.
    ctx.plan(2 * (((2 * orientations) + 2) + (orientations + 2)));
    for (const PixelLattice lattice : {PixelLattice::Square, PixelLattice::Hexagonal}) {
        const bool hexagonal = lattice == PixelLattice::Hexagonal;
        const float variance = pathtracer::debug::innerScaleVariance(lattice);
        // One pass' kernel: T(.;t) along each square axis, T(.;2t/3) along each of the three hexagonal ones.
        const auto axisVariance = static_cast<float>(hexagonal ? (2.0 * static_cast<double>(variance)) / 3.0 : static_cast<double>(variance));
        const std::vector<float> kernel = pathtracer::debug::discreteGaussianKernel(axisVariance);
        const int radius = static_cast<int>(kernel.size()) - 1;
        const glm::ivec2 reach = pathtracer::debug::diffusionReach(variance, lattice);
        const double carrier = static_cast<double>(pathtracer::debug::morletCarrier(variance));

        // diffuse's centre-relative pass applies 1 + sum 2 w_n (cos nw - 1), the exact transfer of its float taps.
        const auto passTransfer = [&](double omega) {
            double sum = 1.0;
            for (int n = 1; n <= radius; ++n) {
                sum += 2.0 * static_cast<double>(kernel[static_cast<std::size_t>(n)]) * (std::cos(n * omega) - 1.0);
            }
            return sum;
        };
        const auto transfer = [&](const glm::dvec2& k) {
            if (!hexagonal) {
                return passTransfer(k.x) * passTransfer(k.y);
            }
            return passTransfer(glm::dot(k, kHexagonalAxes[0])) * passTransfer(glm::dot(k, kHexagonalAxes[1])) *
                   passTransfer(glm::dot(k, kHexagonalAxes[2]));
        };
        const auto position = [hexagonal](int x, int y) { return hexagonal ? hexagonalCentre(x, y) : glm::dvec2(x, y); };
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

        // Higham 3.1: the longest chain is the admissibility product, the wave's blur times the lowpass (3R square, 6R hexagonal), +20.
        const double chain = ((hexagonal ? 6.0 : 3.0) * radius) + 20.0;
        const double gammaChain = (chain * pathtracer::debug::kFloat32Roundoff) / (1.0 - (chain * pathtracer::debug::kFloat32Roundoff));
        // Each part is two terms of absolute mass 3 per centre-relative pass, so 2 3^p sqrt2 + 1 per orientation, doubled for the envelope.
        const double mass = hexagonal ? 27.0 : 9.0;
        const double bound = 2.0 * ((2.0 * mass * std::numbers::sqrt2) + 1.0) * gammaChain * gain * (kMean + kAmplitude);
        const std::string name = pathtracer::gfx::kPixelLatticeNames[static_cast<int>(lattice)];

        for (const Grating& grating : gratings) {
            const glm::dvec2 k = grating.omega * glm::dvec2(std::cos(grating.angle), std::sin(grating.angle));
            std::vector<float> plane(static_cast<std::size_t>(kSize) * kSize);
            for (int y = 0; y < kSize; ++y) {
                for (int x = 0; x < kSize; ++x) {
                    plane[(static_cast<std::size_t>(y) * kSize) + static_cast<std::size_t>(x)] =
                        static_cast<float>(kMean + (kAmplitude * std::cos(glm::dot(k, position(x, y)))));
                }
            }
            const HdrImage gabor = pathtracer::debug::gaborAov(greyImage(kSize, kSize, plane), pool, false, lattice);
            // Per orientation the sidebands k-q and -(k+q) pass H(k-+q) less the admissibility term H(q)H(k); the mean cancels exactly.
            std::vector<std::array<double, 2>> sidebands;
            for (int i = 0; i < orientations; ++i) {
                const glm::dvec2 q = bankCarrier(i);
                const double admissibility = transfer(q) * transfer(k);
                sidebands.push_back({transfer(k - q) - admissibility, transfer(k + q) - admissibility});
            }
            double worst = 0.0;
            double weakest = std::numeric_limits<double>::infinity();
            // Interior only: past the reach from every edge no tap reaches the fold, where the plane waves stop being eigenfunctions.
            for (int y = reach.y; y < kSize - reach.y; ++y) {
                for (int x = reach.x; x < kSize - reach.x; ++x) {
                    const glm::dvec2 r = position(x, y);
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
            const std::string label = name + " w=" + std::to_string(grating.omega) + " angle=" + std::to_string(grating.angle);
            PT_EXPECT(ctx, worst <= bound, label + " deviates by " + std::to_string(worst) + ", bound " + std::to_string(bound));
            if (grating.halfResponse) {
                PT_EXPECT(ctx, weakest >= 0.5 * gain * kAmplitude,
                          label + " reads " + std::to_string(weakest / (gain * kAmplitude)) + " of its amplitude, under half");
            }
        }
    }
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

// Row-axis plane waves are diffusion's eigenfunctions, so its symbol must reproduce the shipped kernels' transfer on both lattices.
PT_CHECK(row_axis_symbol_is_the_diffusion_transfer, Fast, Exact) {
    const std::vector<double> fractions{0.0, 0.125, 0.25, 0.5, 0.75, 0.9, 1.0};
    ctx.plan(static_cast<int>(3 * fractions.size()));
    const double t = 2.0;
    const std::vector<float> squareKernel = pathtracer::debug::discreteGaussianKernel(static_cast<float>(t));
    const std::vector<float> axisKernel = pathtracer::debug::discreteGaussianKernel(static_cast<float>((2.0 * t) / 3.0));
    for (const double fraction : fractions) {
        const double square = fraction * pathtracer::debug::rowAxisBoundary(PixelLattice::Square);
        const double hexagonal = fraction * pathtracer::debug::rowAxisBoundary(PixelLattice::Hexagonal);
        // Each pass carries truncation 2^-24 plus an ulp of storage, as discrete_gaussian_transfer_is_exact bounds.
        const double bound = pathtracer::debug::kFloat32Roundoff + static_cast<double>(kFloatEpsilon);
        const double squareError =
            std::abs(measuredTransfer(squareKernel, square) - std::exp(-t * pathtracer::debug::rowAxisSymbol(square, PixelLattice::Square)));
        const double hexagonalError = std::abs(hexagonalTransfer(axisKernel, glm::dvec2(hexagonal, 0.0)) -
                                               std::exp(-t * pathtracer::debug::rowAxisSymbol(hexagonal, PixelLattice::Hexagonal)));
        PT_EXPECT(ctx, squareError <= bound, "square w/pi=" + std::to_string(fraction) + " transfer error " + std::to_string(squareError));
        PT_EXPECT(ctx, hexagonalError <= 3.0 * bound,
                  "hexagonal w/K=" + std::to_string(fraction) + " transfer error " + std::to_string(hexagonalError));
        // The inverse is acos of the symbol: its conditioning sin(w) vanishes at the ends, so it is checked forward, in symbol space.
        double worst = 0.0;
        for (const PixelLattice lattice : {PixelLattice::Square, PixelLattice::Hexagonal}) {
            const double omega = fraction * pathtracer::debug::rowAxisBoundary(lattice);
            const double a = pathtracer::debug::rowAxisSymbol(omega, lattice);
            worst = std::max(worst, std::abs(pathtracer::debug::rowAxisSymbol(pathtracer::debug::rowAxisFrequency(a, lattice), lattice) - a));
        }
        // A handful of double operations each way, on symbols at most 3.
        PT_EXPECT(ctx, worst <= 64.0 * std::numeric_limits<double>::epsilon(),
                  "w/boundary=" + std::to_string(fraction) + " inverse misses by " + std::to_string(worst));
    }
}

// colour-science 0.4.6 (results/contrast_sensitivity/barten_reference.py): L cd/m^2, X0 deg, u cycles/deg, pupil mm, sensitivity.
constexpr std::array<std::array<double, 5>, 48> kBartenReference{{
    {0.01, 10.0, 0.5, 7.570770130708059, 50.643133515859006},
    {0.01, 10.0, 2.0, 7.570770130708059, 36.91862906512951},
    {0.01, 10.0, 8.0, 7.570770130708059, 10.24484700811024},
    {0.01, 10.0, 32.0, 7.570770130708059, 0.10236209216367767},
    {0.01, 40.0, 0.5, 6.992110310803547, 70.7875915666346},
    {0.01, 40.0, 2.0, 6.992110310803547, 41.863383960856645},
    {0.01, 40.0, 8.0, 6.992110310803547, 10.263178230655962},
    {0.01, 40.0, 32.0, 6.992110310803547, 0.134211584257269},
    {0.01, 60.0, 0.5, 6.733346411225377, 71.03060450918636},
    {0.01, 60.0, 2.0, 6.733346411225377, 41.43129444759229},
    {0.01, 60.0, 8.0, 6.733346411225377, 10.168704153255602},
    {0.01, 60.0, 32.0, 6.733346411225377, 0.149793896320777},
    {1.0, 10.0, 0.5, 6.342687385490272, 206.06414763016508},
    {1.0, 10.0, 2.0, 6.342687385490272, 301.28509678655035},
    {1.0, 10.0, 8.0, 6.342687385490272, 96.64118095620103},
    {1.0, 10.0, 32.0, 6.342687385490272, 1.7239479761531373},
    {1.0, 40.0, 0.5, 5.0, 287.891971108829},
    {1.0, 40.0, 2.0, 5.0, 312.74996120965403},
    {1.0, 40.0, 8.0, 5.0, 87.59967327046563},
    {1.0, 40.0, 32.0, 5.0, 2.5693071526506164},
    {1.0, 60.0, 0.5, 4.5801546165686595, 289.126241386437},
    {1.0, 60.0, 2.0, 4.5801546165686595, 298.37703371488925},
    {1.0, 60.0, 8.0, 4.5801546165686595, 82.9717607179329},
    {1.0, 60.0, 32.0, 4.5801546165686595, 2.7841461227414994},
    {100.0, 10.0, 0.5, 4.075951886659953, 227.870280923516},
    {100.0, 10.0, 2.0, 4.075951886659953, 621.633397822374},
    {100.0, 10.0, 8.0, 4.075951886659953, 466.1220609717308},
    {100.0, 10.0, 32.0, 4.075951886659953, 20.125311819962086},
    {100.0, 40.0, 0.5, 3.007889689196453, 328.1845818655644},
    {100.0, 40.0, 2.0, 3.007889689196453, 713.4431083919145},
    {100.0, 40.0, 8.0, 3.007889689196453, 429.0388591331311},
    {100.0, 40.0, 32.0, 3.007889689196453, 23.072921202226855},
    {100.0, 60.0, 0.5, 2.7931307084858403, 334.78453885875086},
    {100.0, 60.0, 2.0, 2.7931307084858403, 713.4270204895842},
    {100.0, 60.0, 8.0, 2.7931307084858403, 415.6138081601761},
    {100.0, 60.0, 32.0, 2.7931307084858403, 23.174647608300816},
    {10000.0, 10.0, 0.5, 2.5790152167839246, 228.36061742208275},
    {10000.0, 10.0, 2.0, 2.5790152167839246, 641.9854056746868},
    {10000.0, 10.0, 8.0, 2.5790152167839246, 599.967509049232},
    {10000.0, 10.0, 32.0, 2.5790152167839246, 38.59436085429881},
    {10000.0, 40.0, 0.5, 2.234994336780586, 329.3463489354062},
    {10000.0, 40.0, 2.0, 2.234994336780586, 751.3932544929431},
    {10000.0, 40.0, 8.0, 2.234994336780586, 610.5363378048489},
    {10000.0, 40.0, 32.0, 2.234994336780586, 40.888108470058384},
    {10000.0, 60.0, 0.5, 2.179016589937908, 336.13741334923964},
    {10000.0, 60.0, 2.0, 2.179016589937908, 756.5840343659006},
    {10000.0, 60.0, 8.0, 2.179016589937908, 610.976463419274},
    {10000.0, 60.0, 32.0, 2.179016589937908, 41.230218491732586},
}};

// An independent implementation of Barten 1999's pupil, retinal illuminance and CSF, so a transcription slip in any one shows here.
PT_CHECK(barten_csf_matches_colour_science, Fast, Exact) {
    ctx.plan(2);
    double worstPupil = 0.0;
    double worstSensitivity = 0.0;
    for (const std::array<double, 5>& row : kBartenReference) {
        const double pupil = pathtracer::debug::bartenPupilDiameter(row[0], row[1]);
        const double sensitivity = pathtracer::debug::bartenSensitivity(pathtracer::debug::bartenObserver(row[0], row[1]), row[2], row[0]);
        worstPupil = std::max(worstPupil, std::abs(pupil - row[3]) / row[3]);
        worstSensitivity = std::max(worstSensitivity, std::abs(sensitivity - row[4]) / row[4]);
    }
    // Both sides evaluate ~40 double operations in a different order; exp's argument reaches 2 pi^2 sigma^2 u^2 < 2, doubling its ulps.
    const double bound = 2.0 * 40.0 * std::numeric_limits<double>::epsilon();
    PT_EXPECT(ctx, worstPupil <= bound, "pupil diameter relative error " + std::to_string(worstPupil));
    PT_EXPECT(ctx, worstSensitivity <= bound, "sensitivity relative error " + std::to_string(worstSensitivity));
}

// Closed forms: the lat-long's pitch^2 is (2 pi / W)(pi / H) sin(colatitude); the pinhole's on-axis pitch is 2 tan(vfov/2) / H.
PT_CHECK(texel_angular_pitch_matches_the_lens, Fast, Exact) {
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    using pathtracer::scene::Camera;
    using pathtracer::scene::Lens;
    using pathtracer::scene::LensProjection;
    const Camera::FilmBack fullFrame{36.0F, 24.0F};
    constexpr int kWidth = 33;
    constexpr int kHeight = 17;
    const Camera latLong{glm::vec3(0.0F), glm::vec3(0.0F), fullFrame, 35.0F, 0.01F, 100.0F, 2.8F, 0.008F, 400.0F,
                         Lens{LensProjection::Omnidirectional}};
    const std::vector<float> latLongPitch = pathtracer::debug::texelAngularPitch(latLong, kWidth, kHeight, pool);
    double worstLatLong = 0.0;
    for (int y = 0; y < kHeight; ++y) {
        const double colatitude = std::numbers::pi * (static_cast<double>(y) + 0.5) / kHeight;
        const double expected = std::sqrt((2.0 * std::numbers::pi / kWidth) * (std::numbers::pi / kHeight) * std::sin(colatitude));
        for (int x = 0; x < kWidth; ++x) {
            const auto measured = static_cast<double>(latLongPitch[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)]);
            worstLatLong = std::max(worstLatLong, std::abs(measured - expected) / expected);
        }
    }
    // The differential's closed form is a few float trig calls and products deep, as camera_validate's 16-ulp budget allows.
    PT_EXPECT(ctx, worstLatLong <= 16.0 * static_cast<double>(kFloatEpsilon), "lat-long pitch relative error " + std::to_string(worstLatLong));

    const Camera pinhole{glm::vec3(0.0F), glm::vec3(0.0F), fullFrame, 35.0F, 0.01F, 100.0F, 2.8F, 0.008F, 400.0F};
    const std::vector<float> pinholePitch = pathtracer::debug::texelAngularPitch(pinhole, kWidth, kHeight, pool);
    const double axial = static_cast<double>(pinholePitch[((kHeight / 2) * kWidth) + (kWidth / 2)]);
    const double expectedAxial = 2.0 * std::tan(0.5 * static_cast<double>(pinhole.verticalFovRadians())) / kHeight;
    PT_EXPECT(ctx, std::abs(axial - expectedAxial) <= 16.0 * static_cast<double>(kFloatEpsilon) * expectedAxial,
              "pinhole on-axis pitch " + std::to_string(axial) + " expected " + std::to_string(expectedAxial));

    // An 8 mm lens' 180-degree circle has a 12.6 mm radius, inside the gate's 21.6 mm corners, so they see nothing.
    const Camera fisheye{glm::vec3(0.0F), glm::vec3(0.0F), fullFrame, 8.0F, 0.01F, 100.0F, 2.8F, 0.008F, 400.0F,
                         Lens{LensProjection::FisheyePolynomial, {}, 180.0F}};
    const std::vector<float> fisheyePitch = pathtracer::debug::texelAngularPitch(fisheye, kWidth, kHeight, pool);
    PT_EXPECT(ctx, fisheyePitch.front() == 0.0F && fisheyePitch[((kHeight / 2) * kWidth) + (kWidth / 2)] > 0.0F,
              "fisheye corner pitch " + std::to_string(fisheyePitch.front()) + ", centre " +
                  std::to_string(fisheyePitch[((kHeight / 2) * kWidth) + (kWidth / 2)]));
}

// Diffusion leaves a constant exactly, so every band and its contrast vanish: nothing in a uniform field is visible.
PT_CHECK(contrast_sensitivity_reads_zero_on_a_uniform_field, Fast, Exact) {
    ctx.plan(4);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    const HdrImage grey = greyImage(kWidth, kHeight, std::vector<float>(static_cast<std::size_t>(kWidth) * kHeight, 37.0F));
    const std::vector<float> pitch(static_cast<std::size_t>(kWidth) * kHeight, 0.001F);
    for (const PixelLattice lattice : {PixelLattice::Square, PixelLattice::Hexagonal}) {
        for (const bool wraps : {false, true}) {
            const HdrImage out = pathtracer::debug::contrastSensitivityAov(grey, pitch, pool, wraps, lattice);
            const float peak = *std::max_element(out.texels.begin(), out.texels.end());
            PT_EXPECT(ctx, peak == 0.0F, std::string(pathtracer::gfx::kPixelLatticeNames[static_cast<int>(lattice)]) +
                                             (wraps ? " wrapped" : "") + " uniform field reads " + std::to_string(peak));
        }
    }
}

namespace {

// A wrapped row diffused in double: its DFT times the square lattice's exact transfer exp(-t (1 - cos w_j)).
[[nodiscard]] std::vector<double> diffusedRow(const std::vector<double>& row, double t) {
    const std::size_t width = row.size();
    std::vector<double> out(width, 0.0);
    for (std::size_t j = 0; j < width; ++j) {
        const double omega = 2.0 * std::numbers::pi * static_cast<double>(j) / static_cast<double>(width);
        double cosine = 0.0;
        double sine = 0.0;
        for (std::size_t m = 0; m < width; ++m) {
            cosine += row[m] * std::cos(omega * static_cast<double>(m));
            sine += row[m] * std::sin(omega * static_cast<double>(m));
        }
        const double transfer = analyticTransfer(t, omega) / static_cast<double>(width);
        for (std::size_t n = 0; n < width; ++n) {
            out[n] += transfer * ((cosine * std::cos(omega * static_cast<double>(n))) + (sine * std::sin(omega * static_cast<double>(n))));
        }
    }
    return out;
}

// Rounding bound of one square diffusion of a field bounded by one: per pass, truncation 2^-24, a storage ulp and gamma_(2r+1).
[[nodiscard]] double squareDiffusionBound(double t) {
    const auto taps = static_cast<double>((2 * pathtracer::debug::discreteGaussianKernel(static_cast<float>(t)).size()) - 1);
    return 2.0 * (pathtracer::debug::kFloat32Roundoff + static_cast<double>(kFloatEpsilon) + (taps * static_cast<double>(kFloatEpsilon)));
}

// The texel-wise closed form of contrastSensitivityAov on a wrapped, row-constant square field, and its rounding bound.
struct VisibilityReference {
    std::vector<double> visibility;
    std::vector<double> bound;
};

[[nodiscard]] VisibilityReference visibilityReference(const std::vector<double>& luminance, int height, float pitch) {
    const std::size_t width = luminance.size();
    const double peakLuminance = *std::max_element(luminance.begin(), luminance.end());
    double meanLuminance = 0.0;
    for (const double value : luminance) {
        meanLuminance += value;
    }
    meanLuminance /= static_cast<double>(width);
    const auto cell = static_cast<double>(pitch) * static_cast<double>(pitch);
    const double field = std::sqrt(cell * static_cast<double>(width) * height) * (180.0 / std::numbers::pi);
    const pathtracer::debug::BartenObserver observer = pathtracer::debug::bartenObserver(meanLuminance, field);
    const double pitchDegrees = static_cast<double>(pitch) * (180.0 / std::numbers::pi);
    const auto eps = static_cast<double>(kFloatEpsilon);
    VisibilityReference reference{std::vector<double>(width, 0.0), std::vector<double>(width, 0.0)};
    std::vector<double> fine = luminance;
    double fineError = 0.0;
    double fineVariance = 0.0;
    double coarseVariance = std::log(4.0) / 3.0;
    while ((2 * pathtracer::debug::diffusionReach(static_cast<float>(coarseVariance), PixelLattice::Square).y) + 1 <= height) {
        const auto step = static_cast<double>(static_cast<float>(coarseVariance - fineVariance));
        const std::vector<double> coarse = diffusedRow(fine, step);
        const double coarseError = fineError + (peakLuminance * squareDiffusionBound(step));
        const double symbol = fineVariance > 0.0 ? std::log(coarseVariance / fineVariance) / (coarseVariance - fineVariance) : 2.0;
        const double omega = std::acos(1.0 - symbol);
        const double gain = std::exp(-fineVariance * symbol) - std::exp(-coarseVariance * symbol);
        const double sensitivityFrequency = omega / (2.0 * std::numbers::pi * pitchDegrees);
        std::vector<double> power(width);
        double peakContrast = 0.0;
        double minCoarse = coarse.front();
        for (std::size_t x = 0; x < width; ++x) {
            const double contrast = (fine[x] - coarse[x]) / coarse[x];
            power[x] = contrast * contrast;
            peakContrast = std::max(peakContrast, std::abs(contrast));
            minCoarse = std::min(minCoarse, coarse[x]);
        }
        const std::vector<double> pooled = diffusedRow(power, static_cast<double>(static_cast<float>(coarseVariance)));
        // Contrast error from both lowpasses' error and the float subtract and divide; squared, then pooled with its own rounding.
        const double contrastError = ((fineError + coarseError + (peakContrast * coarseError)) / minCoarse) + (2.0 * eps * peakContrast);
        const double powerError = (2.0 * peakContrast * contrastError) + (contrastError * contrastError) +
                                  (peakContrast * peakContrast * (eps + squareDiffusionBound(static_cast<double>(static_cast<float>(coarseVariance)))));
        for (std::size_t x = 0; x < width; ++x) {
            const double sensitivity = pathtracer::debug::bartenSensitivity(observer, sensitivityFrequency, luminance[x]);
            const double envelope = std::sqrt(2.0 * pooled[x]);
            const double visibility = (envelope / gain) * sensitivity;
            // |sqrt a - sqrt b| <= min(sqrt|a-b|, |a-b| / sqrt a); the sensitivity reads the exact float luminance both sides share.
            const double envelopeError = std::min(std::sqrt(2.0 * powerError), (2.0 * powerError) / std::max(envelope, 0x1p-1074));
            const double error = ((envelopeError / gain) * sensitivity) + (visibility * eps);
            if (visibility > reference.visibility[x]) {
                reference.visibility[x] = visibility;
            }
            reference.bound[x] = std::max(reference.bound[x], error);
        }
        fine = coarse;
        fineError = coarseError;
        fineVariance = coarseVariance;
        coarseVariance *= 4.0;
    }
    return reference;
}

// The float luminance row contrastSensitivityAov reads from a row-constant grating L0 (1 + m cos w x), w = 2 pi n / W.
[[nodiscard]] std::vector<double> gratingLuminance(int width, int height, double modulation, int cycles, ThreadPool& pool, HdrImage& image) {
    std::vector<float> plane(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double phase = 2.0 * std::numbers::pi * cycles * x / width;
            plane[(static_cast<std::size_t>(y) * width) + static_cast<std::size_t>(x)] = static_cast<float>(40.0 * (1.0 + (modulation * std::cos(phase))));
        }
    }
    image = greyImage(width, height, plane);
    const HdrImage luminance = pathtracer::debug::luminanceAov(image, pool);
    return {luminance.texels.begin(), luminance.texels.begin() + width};
}

}  // namespace

// A wrapped grating has a closed form at every texel: exact transfers for the cascade, the contrast power's pooling and Barten's CSF.
PT_CHECK(contrast_sensitivity_matches_the_closed_form_on_a_grating, Fast, Exact) {
    ctx.plan(3);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 64;
    constexpr int kHeight = 64;
    constexpr int kCycles = 8;
    // 3 arcmin a texel puts the grating at 2.5 cycles/deg, near the CSF's peak, and the frame's Nyquist at 10.
    const auto pitchRadians = static_cast<float>(0.05 * std::numbers::pi / 180.0);
    const std::vector<float> pitch(static_cast<std::size_t>(kWidth) * kHeight, pitchRadians);
    struct Comparison {
        double worst = 0.0;  // largest deviation as a fraction of its texel's bound
        double measuredPeak = 0.0;
        double expectedPeak = 0.0;
        double peakBound = 0.0;
    };
    const auto compare = [&](double modulation) {
        HdrImage image;
        const std::vector<double> row = gratingLuminance(kWidth, kHeight, modulation, kCycles, pool, image);
        const VisibilityReference reference = visibilityReference(row, kHeight, pitchRadians);
        const HdrImage out = pathtracer::debug::contrastSensitivityAov(image, pitch, pool, true);
        Comparison result;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const auto column = static_cast<std::size_t>(x);
                const auto measured = static_cast<double>(texelAt(out, x, y, 0));
                result.worst = std::max(result.worst, std::abs(measured - reference.visibility[column]) / reference.bound[column]);
                result.measuredPeak = std::max(result.measuredPeak, measured);
                if (reference.visibility[column] > result.expectedPeak) {
                    result.expectedPeak = reference.visibility[column];
                    result.peakBound = reference.bound[column];
                }
            }
        }
        return result;
    };
    const Comparison grating = compare(0.25);
    PT_EXPECT(ctx, grating.worst <= 1.0, "m=0.25 grating deviates by " + std::to_string(grating.worst) + " of its bound");
    // Visibility is m times a factor varying at first order in m, so one rescale m = m0 / V(m0) lands within 2 m0 of the threshold.
    constexpr double kProbe = 0.01;
    HdrImage probeImage;
    const VisibilityReference probe =
        visibilityReference(gratingLuminance(kWidth, kHeight, kProbe, kCycles, pool, probeImage), kHeight, pitchRadians);
    const double modulation = kProbe / *std::max_element(probe.visibility.begin(), probe.visibility.end());
    const Comparison threshold = compare(modulation);
    PT_EXPECT(ctx, threshold.worst <= 1.0, "threshold grating deviates by " + std::to_string(threshold.worst) + " of its bound");
    // The lowpass and the adapting luminance each move by at most a factor 1 +/- m0, which the rescale leaves; then the peak's rounding.
    PT_EXPECT(ctx, std::abs(threshold.measuredPeak - 1.0) <= (2.0 * kProbe) + threshold.peakBound,
              "threshold grating (m=" + std::to_string(modulation) + ") peaks at " + std::to_string(threshold.measuredPeak));
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

// A constant has no gradient and no band: hexagonal Sobel reads exact zeros, and Gabor's admissibility term cancels it at the border too.
PT_CHECK(hexagonal_filters_are_zero_on_a_constant_field, Fast, Exact) {
    ctx.plan(4);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 37;
    constexpr int kHeight = 24;
    constexpr float kLevel = 0.375F;
    const HdrImage flat = greyImage(kWidth, kHeight, std::vector<float>(static_cast<std::size_t>(kWidth) * kHeight, kLevel));
    const float variance = pathtracer::debug::innerScaleVariance(PixelLattice::Hexagonal);
    // The field's blur and the wave's blur each round by the three-pass bound; scaling and differencing add two ulps; root 2, envelope 2.
    const double bound = 2.0 * std::numbers::sqrt2 * ((2.0 * hexagonalRoundingBound(variance)) + (2.0 * static_cast<double>(kFloatEpsilon))) *
                         static_cast<double>(kLevel * greyLuminanceGain());
    for (const bool wraps : {false, true}) {
        const std::string name = wraps ? "wrapped" : "mirrored";
        const HdrImage sobel = pathtracer::debug::sobelAov(flat, pool, wraps, PixelLattice::Hexagonal);
        const float sobelPeak = *std::max_element(sobel.texels.begin(), sobel.texels.end());
        PT_EXPECT(ctx, sobelPeak == 0.0F, name + ": peak hexagonal Sobel " + std::to_string(sobelPeak));
        const HdrImage gabor = pathtracer::debug::gaborAov(flat, pool, wraps, PixelLattice::Hexagonal);
        const float gaborPeak = *std::max_element(gabor.texels.begin(), gabor.texels.end());
        PT_EXPECT(ctx, static_cast<double>(gaborPeak) <= bound,
                  name + ": peak hexagonal Morlet response " + std::to_string(gaborPeak) + " over " + std::to_string(bound));
    }
}

// On any affine field hexagonal Sobel reads 8 |grad f|, Sobel's own gain: the square operator's 8s on an x ramp, and so on both lattices.
PT_CHECK(hexagonal_sobel_reads_eight_times_the_affine_slope, Fast, Exact) {
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 40;
    constexpr int kHeight = 30;
    const double rowPitch = std::numbers::sqrt3 / 2.0;
    // Axial slopes (b, c) of f = 8 + b q + c r; (1, 1/2) is the pure film-x ramp of slope 1, the square operator's 8 reference.
    const std::vector<glm::dvec2> slopes{{1.0, 0.5}, {2.0, 3.0}, {0.0, 1.0}, {-1.5, 0.25}};
    ctx.plan(static_cast<int>(slopes.size()) + 1);
    for (const glm::dvec2 slope : slopes) {
        std::vector<float> plane(static_cast<std::size_t>(kWidth) * kHeight);
        float peak = 0.0F;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const glm::ivec2 axial = axialOf(x, y);
                const auto value = static_cast<float>(8.0 + (slope.x * axial.x) + (slope.y * axial.y));
                plane[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)] = value;
                peak = std::max(peak, std::fabs(value));
            }
        }
        const HdrImage sobel = pathtracer::debug::sobelAov(greyImage(kWidth, kHeight, plane), pool, false, PixelLattice::Hexagonal);
        // grad q = (1, -1/(2 rowPitch)) and grad r = (0, 1/rowPitch) in film units, from x = q + r/2 + 1/4 and y = (r + 1/2) rowPitch.
        const double expected = 8.0 * static_cast<double>(greyLuminanceGain()) *
                                std::hypot(slope.x, (slope.y - (0.5 * slope.x)) / rowPitch);
        // Each tap's luminance rounds by peak eps; the stencil's absolute weights sum below 8, times the 8/3 gain.
        const double bound = 8.0 * (8.0 / 3.0) * static_cast<double>(peak * kFloatEpsilon);
        double worst = 0.0;
        for (int y = 1; y < kHeight - 1; ++y) {
            for (int x = 1; x < kWidth - 1; ++x) {
                worst = std::max(worst, std::abs(static_cast<double>(texelAt(sobel, x, y, 0)) - expected));
            }
        }
        PT_EXPECT(ctx, worst <= bound,
                  "slope (" + std::to_string(slope.x) + ", " + std::to_string(slope.y) + ") reads off 8|grad f| by " + std::to_string(worst));
    }
    std::vector<float> ramp(static_cast<std::size_t>(kWidth) * kHeight);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            ramp[(static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)] = static_cast<float>(x);
        }
    }
    const float square = texelAt(pathtracer::debug::sobelAov(greyImage(kWidth, kHeight, ramp), pool), kWidth / 2, kHeight / 2, 0);
    PT_EXPECT(ctx, std::abs(static_cast<double>(square) - (8.0 * static_cast<double>(greyLuminanceGain()))) <= 8.0 * kWidth * kFloatEpsilon,
              "square Sobel reads " + std::to_string(square) + " on a unit x ramp, not 8");
}

// A half turn maps each neighbour onto its opposite, negating both gradient components exactly, so the magnitude is bit-identical.
PT_CHECK(hexagonal_sobel_commutes_with_a_half_turn, Fast, Exact) {
    ctx.plan(1);
    ThreadPool pool(static_cast<unsigned int>(ctx.threads()));
    constexpr int kWidth = 37;
    constexpr int kHeight = 24;
    std::mt19937 rng(static_cast<std::uint32_t>(ctx.subSeed("hexagonal-sobel-turn")));
    const std::vector<float> plane = uniformPlane(kWidth, kHeight, rng);
    std::vector<float> turned(plane.size());
    for (std::size_t pixel = 0; pixel < plane.size(); ++pixel) {
        turned[plane.size() - 1 - pixel] = plane[pixel];
    }
    const HdrImage sobel = pathtracer::debug::sobelAov(greyImage(kWidth, kHeight, plane), pool, false, PixelLattice::Hexagonal);
    const HdrImage turnedSobel = pathtracer::debug::sobelAov(greyImage(kWidth, kHeight, turned), pool, false, PixelLattice::Hexagonal);
    bool identical = true;
    for (std::size_t pixel = 0; pixel < plane.size(); ++pixel) {
        identical = identical && turnedSobel.texels[plane.size() - 1 - pixel] == sobel.texels[pixel];
    }
    PT_EXPECT(ctx, identical, "a half turn of an even-row hexagonal frame changed its Sobel");
}

// A wrapping image has no seam to the filters: rolling its columns rolls Sobel bit for bit, and Gabor to its own rounding.
PT_CHECK(wrapped_filters_commute_with_a_cyclic_column_shift, Fast, Exact) {
    ctx.plan(4);
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
    for (const PixelLattice lattice : {PixelLattice::Square, PixelLattice::Hexagonal}) {
        const std::string name = pathtracer::gfx::kPixelLatticeNames[static_cast<int>(lattice)];
        const float sobel = worstShiftError(pathtracer::debug::sobelAov(image, pool, true, lattice),
                                            pathtracer::debug::sobelAov(rolled, pool, true, lattice));
        PT_EXPECT(ctx, sobel == 0.0F, name + ": Sobel moved by " + std::to_string(sobel) + " under a cyclic shift");
    }
    // The roll moves the carrier's phase, so demodulation, both blurs and the baseband round anew: 2R+1 ulps apiece, x2 envelope.
    const auto radius = static_cast<double>(pathtracer::debug::discreteGaussianKernel(pathtracer::debug::innerScaleVariance()).size() - 1);
    const double bound = 2.0 * 4.0 * ((2.0 * radius) + 1.0) * static_cast<double>(kFloatEpsilon * greyLuminanceGain());
    const float gabor = worstShiftError(pathtracer::debug::gaborAov(image, pool, true), pathtracer::debug::gaborAov(rolled, pool, true));
    PT_EXPECT(ctx, gabor <= bound, "Gabor moved by " + std::to_string(gabor) + " under a cyclic shift, bound " + std::to_string(bound));
    // Hexagonally the wave is diffused too, so its blur rounds anew alongside the field's: two three-pass bounds apiece, x2 envelope.
    const float hexagonalVariance = pathtracer::debug::innerScaleVariance(PixelLattice::Hexagonal);
    const double hexagonalBound = 2.0 * 4.0 * hexagonalRoundingBound(hexagonalVariance) * static_cast<double>(greyLuminanceGain());
    const float hexagonalGabor = worstShiftError(pathtracer::debug::gaborAov(image, pool, true, PixelLattice::Hexagonal),
                                                 pathtracer::debug::gaborAov(rolled, pool, true, PixelLattice::Hexagonal));
    PT_EXPECT(ctx, hexagonalGabor <= hexagonalBound,
              "hexagonal Gabor moved by " + std::to_string(hexagonalGabor) + " under a cyclic shift, bound " + std::to_string(hexagonalBound));
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
