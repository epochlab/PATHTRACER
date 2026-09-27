#include "pathtracer/debug/scale_space.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace pathtracer::debug {

namespace {

using pathtracer::scene::ThreadPool;

// Reflection about the edge samples without repeating them, so every tap lands on real data. Valid for any offset and any extent.
[[nodiscard]] int mirror(int i, int extent) {
    if (extent <= 1) {
        return 0;
    }
    const int period = (2 * extent) - 2;
    int folded = i % period;
    if (folded < 0) {
        folded += period;
    }
    return folded < extent ? folded : period - folded;
}

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

// Folded column indices for one row, built once per pass: the row loop would otherwise pay a modulo per boundary tap per row.
[[nodiscard]] std::vector<int> foldTable(int extent, int radius) {
    std::vector<int> table(static_cast<std::size_t>(extent + (2 * radius)));
    for (int i = 0; i < extent + (2 * radius); ++i) {
        table[static_cast<std::size_t>(i)] = mirror(i - radius, extent);
    }
    return table;
}

// Columns carried in registers across the tap loop. Without it each tap restreams the accumulator and the centre, 2.5x the traffic.
constexpr int kConvolutionBlock = 32;

// One separable pass in the centre-relative form: an exactly representable affine field's tap pairs cancel, so its interior is kept.
void diffuseRows(const float* __restrict src, float* __restrict dst, const std::vector<float>& kernel, int width,
                 int height, ThreadPool& threadPool) {
    const int radius = static_cast<int>(kernel.size()) - 1;
    const std::vector<int> fold = foldTable(width, radius);
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
    const std::vector<int> fold = foldTable(height, radius);
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

}  // namespace

float decimationVariance() {
    // exp(-t(1-cos w)) at w = pi/2 gives exp(-t), so the criterion is exp(-t) <= 2^-24, i.e. t = 24 ln2.
    return static_cast<float>(-std::log(kFloat32Roundoff));
}

float innerScaleVariance() {
    // At the full-grid Nyquist w = pi the exponent doubles, (1-cos pi) = 2 (1-cos pi/2), so the same bound halves the variance.
    return 0.5F * decimationVariance();
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

void diffuse(std::span<float> plane, int width, int height, float t, ThreadPool& threadPool) {
    if (!(t > 0.0F) || width <= 0 || height <= 0) {
        return;
    }
    const std::vector<float> kernel = discreteGaussianKernel(t);
    if (kernel.size() == 1) {
        return;
    }
    std::vector<float> scratch(plane.size());
    diffuseRows(plane.data(), scratch.data(), kernel, width, height, threadPool);
    diffuseColumns(scratch.data(), plane.data(), kernel, width, height, threadPool);
}

std::vector<ScaleSpaceLevel> buildOctavePyramid(std::span<const float> plane, int width, int height,
                                                ThreadPool& threadPool, std::size_t maxLevels) {
    const float decimationAt = decimationVariance();

    std::vector<ScaleSpaceLevel> levels;
    std::vector<float> current(plane.begin(), plane.end());
    int currentWidth = width;
    int currentHeight = height;
    int decimation = 1;
    float ownVariance = 0.0F;

    for (float baseVariance = innerScaleVariance();; baseVariance *= 4.0F) {
        const float targetOwn = baseVariance / static_cast<float>(decimation * decimation);
        const float step = targetOwn - ownVariance;
        const auto radius = static_cast<int>(discreteGaussianKernel(step).size()) - 1;
        // Ends where the step's support outgrows its own plane: such a level reports the mirror, not the image, at every sample.
        if ((2 * radius) + 1 > std::min(currentWidth, currentHeight)) {
            break;
        }
        diffuse(current, currentWidth, currentHeight, step, threadPool);
        ownVariance = targetOwn;
        levels.push_back(ScaleSpaceLevel{current, currentWidth, currentHeight, decimation, baseVariance});
        if (levels.size() == maxLevels) {
            break;
        }

        // Halving the grid discards nothing once the level is band-limited to the halved Nyquist, which decimationVariance defines.
        if (ownVariance < decimationAt) {
            continue;
        }
        const int nextWidth = (currentWidth + 1) / 2;
        const int nextHeight = (currentHeight + 1) / 2;
        std::vector<float> next(static_cast<std::size_t>(nextWidth) * static_cast<std::size_t>(nextHeight));
        // Plain subsampling, not an average: an average would add variance the cascade's own bookkeeping does not account for.
        threadPool.parallelFor(nextHeight, [&](int y) {
            const std::size_t source = static_cast<std::size_t>(2 * y) * static_cast<std::size_t>(currentWidth);
            const std::size_t destination = static_cast<std::size_t>(y) * static_cast<std::size_t>(nextWidth);
            for (int x = 0; x < nextWidth; ++x) {
                next[destination + static_cast<std::size_t>(x)] = current[source + static_cast<std::size_t>(2 * x)];
            }
        });
        current = std::move(next);
        currentWidth = nextWidth;
        currentHeight = nextHeight;
        decimation *= 2;
        ownVariance *= 0.25F;
    }
    return levels;
}

std::vector<float> expandToBase(const ScaleSpaceLevel& level, int baseWidth, int baseHeight,
                                ThreadPool& threadPool) {
    std::vector<float> out(static_cast<std::size_t>(baseWidth) * static_cast<std::size_t>(baseHeight), 0.0F);
    // Unit weight onto zero is the resample itself, exact but for a -0 sample, which becomes +0.
    addExpanded(level, 1.0F, out, baseWidth, baseHeight, threadPool);
    return out;
}

void addExpanded(const ScaleSpaceLevel& level, float weight, std::span<float> target, int targetWidth, int targetHeight,
                 ThreadPool& threadPool, int targetDecimation) {
    // A ratio of powers of two, so exact: target sample i sits on level coordinate i*targetDecimation/decimation.
    const float inverseDecimation = static_cast<float>(targetDecimation) / static_cast<float>(level.decimation);
    threadPool.parallelFor(targetHeight, [&](int y) {
        // Level sample j sits on target sample j*decimation/targetDecimation: a plain scale with no half-pixel offset.
        const float v = static_cast<float>(y) * inverseDecimation;
        const int y0 = std::min(static_cast<int>(v), level.height - 1);
        // Past the last level sample the level's own mirror applies, as its diffusion did, rather than a zeroth-order hold.
        const int y1 = mirror(y0 + 1, level.height);
        const float fractionY = v - static_cast<float>(y0);
        const float* upper = level.plane.data() + (static_cast<std::size_t>(y0) * static_cast<std::size_t>(level.width));
        const float* lower = level.plane.data() + (static_cast<std::size_t>(y1) * static_cast<std::size_t>(level.width));
        float* row = target.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(targetWidth));
        for (int x = 0; x < targetWidth; ++x) {
            const float u = static_cast<float>(x) * inverseDecimation;
            const int x0 = std::min(static_cast<int>(u), level.width - 1);
            // The fold's modulo only past the last sample: interior columns take the next sample directly, off the per-texel path.
            const int x1 = x0 + 1 < level.width ? x0 + 1 : mirror(x0 + 1, level.width);
            const float fractionX = u - static_cast<float>(x0);
            const float top = upper[x0] + ((upper[x1] - upper[x0]) * fractionX);
            const float bottom = lower[x0] + ((lower[x1] - lower[x0]) * fractionX);
            row[x] += weight * (top + ((bottom - top) * fractionY));
        }
    });
}

std::vector<float> octaveMean(const std::vector<ScaleSpaceLevel>& pyramid, ThreadPool& threadPool) {
    const float weight = 1.0F / static_cast<float>(pyramid.size());
    const auto scaled = [&](const ScaleSpaceLevel& level) {
        std::vector<float> plane(level.plane.size());
        std::transform(level.plane.begin(), level.plane.end(), plane.begin(), [weight](float value) { return weight * value; });
        return plane;
    };
    // The running sum lives on each level's own grid and moves one octave finer per step: about 4/3 of one base pass in all.
    ScaleSpaceLevel sum{scaled(pyramid.back()), pyramid.back().width, pyramid.back().height, pyramid.back().decimation, 0.0F};
    for (std::size_t k = pyramid.size() - 1; k-- > 0;) {
        const ScaleSpaceLevel& level = pyramid[k];
        std::vector<float> finer = scaled(level);
        addExpanded(sum, 1.0F, finer, level.width, level.height, threadPool, level.decimation);
        sum = ScaleSpaceLevel{std::move(finer), level.width, level.height, level.decimation, 0.0F};
    }
    // The finest level is on the base grid by construction, so the collapsed sum is the base plane.
    return std::move(sum.plane);
}

void laplacian5(std::span<const float> plane, int width, int height, std::span<float> out,
                ThreadPool& threadPool) {
    threadPool.parallelFor(height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        const float* centre = plane.data() + row;
        const float* above = plane.data() + (static_cast<std::size_t>(mirror(y - 1, height)) * static_cast<std::size_t>(width));
        const float* below = plane.data() + (static_cast<std::size_t>(mirror(y + 1, height)) * static_cast<std::size_t>(width));
        float* destination = out.data() + row;
        // Differences taken against the centre before summing, the same cancellation-free form the diffusion passes use.
        for (int x = 0; x < width; ++x) {
            const float value = centre[x];
            // The fold's modulo is paid only at the two edge columns; everywhere else the neighbours index directly.
            const float left = centre[x > 0 ? x - 1 : mirror(x - 1, width)];
            const float right = centre[x + 1 < width ? x + 1 : mirror(x + 1, width)];
            destination[x] = ((left - value) + (right - value)) + ((above[x] - value) + (below[x] - value));
        }
    });
}

}  // namespace pathtracer::debug
