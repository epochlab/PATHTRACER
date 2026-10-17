#include "pathtracer/debug/scale_space.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace pathtracer::debug {

namespace {

using pathtracer::gfx::mirrorIndex;
using pathtracer::gfx::PixelLattice;
using pathtracer::gfx::wrapIndex;
using pathtracer::scene::ThreadPool;

// T(n;t) = exp(-t) I_n(t) via the ascending series for I_n, whose terms are all positive, so the sum carries no cancellation at all.
[[nodiscard]] double discreteGaussianTap(int n, double t) {
    const double halfT = 0.5 * t;
    // Magnitude kept in the exponent and the series summed relative to its own first term, so no intermediate can overflow or vanish.
    double logScale = (static_cast<double>(n) * std::log(halfT)) - std::lgamma(static_cast<double>(n) + 1.0) - t;
    double term = 1.0;
    double sum = 1.0;
    // The term ratio is at most (halfT/k)^2, under 1/2 past sqrt(2)*halfT, so the discarded tail is below the last term kept.
    for (int k = 1; term > sum * 0x1p-53; ++k) {
        term *= (halfT * halfT) / (static_cast<double>(k) * static_cast<double>(n + k));
        sum += term;
        // Rescaled by a power of two, so exactly, well before the growing positive sum could reach the top of double's range.
        if (sum > 0x1p512) {
            sum *= 0x1p-512;
            term *= 0x1p-512;
            logScale += 512.0 * std::numbers::ln2;
        }
    }
    return std::exp(logScale + std::log(sum));
}

// Folded indices for one line, built once per pass: the row loop would otherwise pay a modulo per boundary tap per row.
[[nodiscard]] std::vector<int> foldTable(int extent, int radius, bool periodic) {
    std::vector<int> table(static_cast<std::size_t>(extent + (2 * radius)));
    for (int i = 0; i < extent + (2 * radius); ++i) {
        table[static_cast<std::size_t>(i)] = periodic ? wrapIndex(i - radius, extent) : mirrorIndex(i - radius, extent);
    }
    return table;
}

// Columns carried in registers across the tap loop. Without it each tap restreams the accumulator and the centre, 2.5x the traffic.
constexpr int kConvolutionBlock = 32;

// One separable pass in the centre-relative form: an exactly representable affine field's tap pairs cancel, so its interior is kept.
void diffuseRows(const float* __restrict src, float* __restrict dst, const std::vector<float>& kernel, int width,
                 int height, bool wrapsHorizontally, ThreadPool& threadPool) {
    const int radius = static_cast<int>(kernel.size()) - 1;
    const std::vector<int> fold = foldTable(width, radius, wrapsHorizontally);
    // Columns nearer than radius to either edge need the fold; the interior span between them indexes directly and vectorises.
    const int interiorBegin = std::min(radius, width);
    const int interiorEnd = std::max(width - radius, interiorBegin);
    threadPool.parallelFor(height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        const float* __restrict in = src + row;
        float* __restrict out = dst + row;
        for (int x0 = 0; x0 < width; x0 += kConvolutionBlock) {
            const int span = std::min(kConvolutionBlock, width - x0);
            std::array<float, kConvolutionBlock> value{};
            std::array<float, kConvolutionBlock> accumulator{};
            for (int j = 0; j < span; ++j) {
                value[static_cast<std::size_t>(j)] = in[x0 + j];
                accumulator[static_cast<std::size_t>(j)] = value[static_cast<std::size_t>(j)];
            }
            const bool interior = x0 >= interiorBegin && (x0 + span) <= interiorEnd;
            for (int n = 1; n <= radius; ++n) {
                const float weight = kernel[static_cast<std::size_t>(n)];
                if (interior) {
                    for (int j = 0; j < span; ++j) {
                        accumulator[static_cast<std::size_t>(j)] +=
                            weight * ((in[(x0 + j) - n] + in[(x0 + j) + n]) - (2.0F * value[static_cast<std::size_t>(j)]));
                    }
                    continue;
                }
                for (int j = 0; j < span; ++j) {
                    const int left = fold[static_cast<std::size_t>(((x0 + j) - n) + radius)];
                    const int right = fold[static_cast<std::size_t>(((x0 + j) + n) + radius)];
                    accumulator[static_cast<std::size_t>(j)] +=
                        weight * ((in[left] + in[right]) - (2.0F * value[static_cast<std::size_t>(j)]));
                }
            }
            for (int j = 0; j < span; ++j) {
                out[x0 + j] = accumulator[static_cast<std::size_t>(j)];
            }
        }
    });
}

// The column pass, same form and blocking, but the fold is per row rather than per sample, so every inner loop is a stride-1 run.
void diffuseColumns(const float* __restrict src, float* __restrict dst, const std::vector<float>& kernel, int width,
                    int height, ThreadPool& threadPool) {
    const int radius = static_cast<int>(kernel.size()) - 1;
    const std::vector<int> fold = foldTable(height, radius, false);
    threadPool.parallelFor(height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        const float* __restrict centre = src + row;
        float* __restrict out = dst + row;
        for (int x0 = 0; x0 < width; x0 += kConvolutionBlock) {
            const int span = std::min(kConvolutionBlock, width - x0);
            std::array<float, kConvolutionBlock> value{};
            std::array<float, kConvolutionBlock> accumulator{};
            for (int j = 0; j < span; ++j) {
                value[static_cast<std::size_t>(j)] = centre[x0 + j];
                accumulator[static_cast<std::size_t>(j)] = value[static_cast<std::size_t>(j)];
            }
            for (int n = 1; n <= radius; ++n) {
                const float weight = kernel[static_cast<std::size_t>(n)];
                const float* __restrict above =
                    src + (static_cast<std::size_t>(fold[static_cast<std::size_t>((y - n) + radius)]) *
                           static_cast<std::size_t>(width));
                const float* __restrict below =
                    src + (static_cast<std::size_t>(fold[static_cast<std::size_t>((y + n) + radius)]) *
                           static_cast<std::size_t>(width));
                for (int j = 0; j < span; ++j) {
                    accumulator[static_cast<std::size_t>(j)] +=
                        weight * ((above[x0 + j] + below[x0 + j]) - (2.0F * value[static_cast<std::size_t>(j)]));
                }
            }
            for (int j = 0; j < span; ++j) {
                out[x0 + j] = accumulator[static_cast<std::size_t>(j)];
            }
        }
    });
}

// Half-pitch column reach of n rows along a 60 or 120 degree axis, ceil(n/2): odd rows sit half a pitch right of even ones.
[[nodiscard]] int diagonalReach(int rows) {
    return (rows + 1) >> 1;
}

// Two diagonal passes reach 2r rows and 2 ceil(r/2) columns on top of the row pass' r, for per-axis radius r.
[[nodiscard]] glm::ivec2 hexagonalReach(int radius) {
    return {radius + (2 * diagonalReach(radius)), 2 * radius};
}

// One hexagonal 60 or 120 degree pass over rows [firstRow, lastRow) of a padded plane, columns inside the axis' reach of the edge.
void diffuseDiagonal(const float* __restrict src, float* __restrict dst, const std::vector<float>& kernel, int width,
                     int firstRow, int lastRow, bool sixtyDegrees, ThreadPool& threadPool) {
    const int radius = static_cast<int>(kernel.size()) - 1;
    const int margin = diagonalReach(radius);
    threadPool.parallelFor(lastRow - firstRow, [&](int index) {
        const int y = firstRow + index;
        const int parity = y & 1;
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        const float* __restrict centre = src + row;
        float* __restrict out = dst + row;
        for (int x0 = margin; x0 < width - margin; x0 += kConvolutionBlock) {
            const int span = std::min(kConvolutionBlock, width - margin - x0);
            std::array<float, kConvolutionBlock> value{};
            std::array<float, kConvolutionBlock> accumulator{};
            for (int j = 0; j < span; ++j) {
                value[static_cast<std::size_t>(j)] = centre[x0 + j];
                accumulator[static_cast<std::size_t>(j)] = value[static_cast<std::size_t>(j)];
            }
            for (int n = 1; n <= radius; ++n) {
                const float weight = kernel[static_cast<std::size_t>(n)];
                // n rows along the axis move n/2 pitches: right by (n+p)>>1 or left by (n+1-p)>>1 columns from a row of parity p.
                const int right = (n + parity) >> 1;
                const int left = -((n + 1 - parity) >> 1);
                const int belowShift = sixtyDegrees ? right : left;
                const int aboveShift = sixtyDegrees ? left : right;
                const float* __restrict below =
                    src + (static_cast<std::ptrdiff_t>(y + n) * width) + belowShift;
                const float* __restrict above =
                    src + (static_cast<std::ptrdiff_t>(y - n) * width) + aboveShift;
                for (int j = 0; j < span; ++j) {
                    accumulator[static_cast<std::size_t>(j)] +=
                        weight * ((above[x0 + j] + below[x0 + j]) - (2.0F * value[static_cast<std::size_t>(j)]));
                }
            }
            for (int j = 0; j < span; ++j) {
                out[x0 + j] = accumulator[static_cast<std::size_t>(j)];
            }
        }
    });
}

// Six-neighbour generator (2/3) sum D_a over commuting 0/60/120 degree axes: three T(.;2t/3) passes, covariance t I (Lindeberg 1994).
void diffuseHexagonal(std::span<float> plane, int width, int height, float t, bool wrapsHorizontally, ThreadPool& threadPool) {
    const std::vector<float> kernel = discreteGaussianKernel(static_cast<float>((2.0 * static_cast<double>(t)) / 3.0));
    if (kernel.size() == 1) {
        return;
    }
    // The frame mirrors swap the 60 and 120 degree axes, so no per-line fold is exact: pad by the cascade's reach through the lattice fold.
    const int radius = static_cast<int>(kernel.size()) - 1;
    const glm::ivec2 pad = hexagonalReach(radius);
    const int paddedWidth = width + (2 * pad.x);
    const int paddedHeight = height + (2 * pad.y);
    std::vector<float> padded(static_cast<std::size_t>(paddedWidth) * static_cast<std::size_t>(paddedHeight));
    threadPool.parallelFor(paddedHeight, [&](int y) {
        float* out = padded.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(paddedWidth));
        // The row pad is even, so padded row y has the parity of frame row y - pad.y and the lattice geometry carries over unchanged.
        for (int x = 0; x < paddedWidth; ++x) {
            const glm::ivec2 cell =
                pathtracer::gfx::hexagonalReflection(x - pad.x, y - pad.y, width, height, wrapsHorizontally);
            out[x] = plane[(static_cast<std::size_t>(cell.y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(cell.x)];
        }
    });
    std::vector<float> scratch(padded.size());
    // Each pass only needs rows and columns the later passes read; the 0 degree pass's own edge fold lands in margins never cropped.
    diffuseRows(padded.data(), scratch.data(), kernel, paddedWidth, paddedHeight, false, threadPool);
    diffuseDiagonal(scratch.data(), padded.data(), kernel, paddedWidth, radius, paddedHeight - radius, true, threadPool);
    diffuseDiagonal(padded.data(), scratch.data(), kernel, paddedWidth, pad.y, pad.y + height, false, threadPool);
    threadPool.parallelFor(height, [&](int y) {
        const float* in = scratch.data() + (static_cast<std::size_t>(y + pad.y) * static_cast<std::size_t>(paddedWidth)) + pad.x;
        std::copy_n(in, width, plane.begin() + (static_cast<std::ptrdiff_t>(y) * width));
    });
}

}  // namespace

float innerScaleVariance(PixelLattice lattice) {
    if (lattice == PixelLattice::Hexagonal) {
        // Three-axis transfer exp(-(2t/3) sum(1-cos k.u)) peaks on the zone boundary at the M point, sum 4: exp(-8t/3) <= 2^-24, t = 9 ln2.
        return static_cast<float>(-0.375 * std::log(kFloat32Roundoff));
    }
    // exp(-t(1-cos w)) at w = pi gives exp(-2t), so the criterion exp(-2t) <= 2^-24 gives t = 12 ln2.
    return static_cast<float>(-0.5 * std::log(kFloat32Roundoff));
}

glm::ivec2 diffusionReach(float t, PixelLattice lattice) {
    if (lattice == PixelLattice::Hexagonal) {
        return hexagonalReach(static_cast<int>(discreteGaussianKernel(static_cast<float>((2.0 * static_cast<double>(t)) / 3.0)).size()) - 1);
    }
    const int radius = static_cast<int>(discreteGaussianKernel(t).size()) - 1;
    return {radius, radius};
}

std::vector<float> discreteGaussianKernel(float t) {
    if (!(t > 0.0F)) {
        return {1.0F};
    }
    const auto variance = static_cast<double>(t);
    std::vector<double> taps{discreteGaussianTap(0, variance)};
    double mass = taps.front();
    // Emitted outward until the discarded tail mass cannot perturb a float result: the radius is derived from 2^-24, never authored.
    while (1.0 - mass > kFloat32Roundoff) {
        const double value = discreteGaussianTap(static_cast<int>(taps.size()), variance);
        taps.push_back(value);
        mass += 2.0 * value;
    }
    // Normalised over the emitted taps, not against the infinite sum: a truncated kernel with gain 1-2^-24 would compound per level.
    const double scale = 1.0 / mass;
    std::vector<float> kernel(taps.size());
    for (std::size_t n = 0; n < taps.size(); ++n) {
        kernel[n] = static_cast<float>(taps[n] * scale);
    }
    return kernel;
}

void diffuse(std::span<float> plane, int width, int height, float t, ThreadPool& threadPool, bool wrapsHorizontally,
             PixelLattice lattice) {
    if (!(t > 0.0F) || width <= 0 || height <= 0) {
        return;
    }
    if (lattice == PixelLattice::Hexagonal) {
        diffuseHexagonal(plane, width, height, t, wrapsHorizontally, threadPool);
        return;
    }
    const std::vector<float> kernel = discreteGaussianKernel(t);
    if (kernel.size() == 1) {
        return;
    }
    std::vector<float> scratch(plane.size());
    diffuseRows(plane.data(), scratch.data(), kernel, width, height, wrapsHorizontally, threadPool);
    diffuseColumns(scratch.data(), plane.data(), kernel, width, height, threadPool);
}

}  // namespace pathtracer::debug
