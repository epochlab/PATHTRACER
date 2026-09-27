#include "pathtracer/debug/aov_filters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

#include <glm/gtc/constants.hpp>

#include "pathtracer/debug/scale_space.h"
#include "pathtracer/scene/cone_space.h"

namespace pathtracer::debug {

namespace {

using pathtracer::gfx::HdrImage;
using pathtracer::scene::ThreadPool;

// Single-channel Rec.709 luminance, shared by every filter that reads intensity alone. Materialised once: Sobel reads 8 neighbours.
[[nodiscard]] std::vector<float> luminancePlane(const HdrImage& beauty, ThreadPool& threadPool) {
    std::vector<float> plane(static_cast<std::size_t>(beauty.width) * static_cast<std::size_t>(beauty.height));
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const std::size_t texel = (row + static_cast<std::size_t>(x)) * 4;
            plane[row + static_cast<std::size_t>(x)] =
                (beauty.rgba[texel] * kRec709LuminanceWeights.r) +
                (beauty.rgba[texel + 1] * kRec709LuminanceWeights.g) +
                (beauty.rgba[texel + 2] * kRec709LuminanceWeights.b);
        }
    });
    return plane;
}

// Neighbour fetch with edge clamping, matching the display texture's GL_CLAMP_TO_EDGE.
[[nodiscard]] float tap(const std::vector<float>& plane, int width, int height, int x, int y) {
    const int cx = std::clamp(x, 0, width - 1);
    const int cy = std::clamp(y, 0, height - 1);
    return plane[(static_cast<std::size_t>(cy) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(cx)];
}

[[nodiscard]] HdrImage makeBroadcastImage(int width, int height) {
    return HdrImage{width, height,
                    std::vector<float>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4, 0.0F)};
}

// Writes one scalar to RGB with alpha 1, the broadcast convention every AOV uses, so it goes straight through HdrImage to the GPU.
void writeScalar(HdrImage& out, std::size_t pixel, float value) {
    const std::size_t texel = pixel * 4;
    out.rgba[texel] = value;
    out.rgba[texel + 1] = value;
    out.rgba[texel + 2] = value;
    out.rgba[texel + 3] = 1.0F;
}

// A plane of log2(value / reference) and the reference it is measured from; zero at a texel carrying no light, where the log is undefined.
struct RelativeLog {
    std::vector<float> plane;
    float reference = 0.0F;  // the smallest positive value present, so every entry of `plane` is non-negative
};

// Exponent differences are exact integers and mantissas are untouched by a power-of-two gain, so this cancels one rather than rounding it.
[[nodiscard]] RelativeLog relativeLog2(std::span<const float> values, int width, int height, ThreadPool& threadPool) {
    RelativeLog out{std::vector<float>(values.size(), 0.0F), 0.0F};
    for (const float value : values) {
        out.reference = value > 0.0F && (out.reference == 0.0F || value < out.reference) ? value : out.reference;
    }
    if (out.reference == 0.0F) {
        return out;
    }
    int referenceExponent = 0;
    const float referenceLog = std::log2(std::frexp(out.reference, &referenceExponent));
    // Threaded over rows: one frexp and one log per texel is the whole cost, and it is the same arithmetic wherever it runs.
    threadPool.parallelFor(height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        for (int x = 0; x < width; ++x) {
            const std::size_t index = row + static_cast<std::size_t>(x);
            if (values[index] <= 0.0F) {
                continue;
            }
            int exponent = 0;
            const float mantissa = std::frexp(values[index], &exponent);
            out.plane[index] = static_cast<float>(exponent - referenceExponent) + (std::log2(mantissa) - referenceLog);
        }
    });
    return out;
}

// IEC 61966-2-1 reference medium of kOcioSrgbDisplay, the encode every CPU capture uses: white 80 cd/m^2 over black 0.2 cd/m^2.
constexpr double kSrgbReferenceWhite = 80.0;
constexpr double kSrgbReferenceBlack = 0.2;

// Ward Larson 1997 section 5 ceiling at its fixed point T = sum min(f, cT); empty iff K occupied bins fit (K*c <= 1, g concave).
[[nodiscard]] std::vector<double> truncatedHistogram(const std::vector<double>& counts, double ceilingPerCount) {
    std::vector<double> sorted;
    for (const double count : counts) {
        if (count > 0.0) {
            sorted.push_back(count);
        }
    }
    if (static_cast<double>(sorted.size()) * ceilingPerCount <= 1.0) {
        return {};
    }
    std::sort(sorted.begin(), sorted.end(), std::greater<>{});
    std::vector<double> suffix(sorted.size() + 1, 0.0);
    for (std::size_t j = sorted.size(); j-- > 0;) {
        suffix[j] = suffix[j + 1] + sorted[j];
    }
    // First breakpoint T = f_j/c, downward, with g >= 0 brackets the root; integer counts make tied breakpoints evaluate identically.
    const auto g = [&](std::size_t j) {
        return (static_cast<double>(j) * sorted[j]) + suffix[j] - (sorted[j] / ceilingPerCount);
    };
    // Terminates by the last breakpoint: all K clipped there gives g = f(K - 1/c) > 0, since K*c > 1.
    std::size_t clipped = 0;
    while (g(clipped) < 0.0) {
        ++clipped;
    }
    const double retained = suffix[clipped] / (1.0 - (static_cast<double>(clipped) * ceilingPerCount));
    std::vector<double> truncated(counts.size());
    for (std::size_t bin = 0; bin < counts.size(); ++bin) {
        truncated[bin] = std::min(counts[bin], ceilingPerCount * retained);
    }
    return truncated;
}

// log2(value / reference) as an exact exponent difference plus a mantissa term, so a power-of-two gain on both cancels bitwise.
class LogReference {
public:
    explicit LogReference(double reference) : mantissaLog_(std::log2(std::frexp(reference, &exponent_))) {}

    [[nodiscard]] double operator()(double value) const {
        int exponent = 0;
        const double mantissa = std::frexp(value, &exponent);
        return static_cast<double>(exponent - exponent_) + (std::log2(mantissa) - mantissaLog_);
    }

private:
    int exponent_ = 0;
    double mantissaLog_;
};

// Ward Larson 1997's foveal image: mean luminance over 1-degree blocks, an exact partition of the frame, at most one block per pixel.
[[nodiscard]] std::vector<double> fovealMeans(const std::vector<float>& luminance, int width, int height, float pixelsPerRadian,
                                              ThreadPool& threadPool) {
    const float pixelsPerDegree = glm::radians(pixelsPerRadian);
    const int blocksX = std::clamp(static_cast<int>(std::lround(static_cast<float>(width) / pixelsPerDegree)), 1, width);
    const int blocksY = std::clamp(static_cast<int>(std::lround(static_cast<float>(height) / pixelsPerDegree)), 1, height);
    std::vector<double> means(static_cast<std::size_t>(blocksX) * static_cast<std::size_t>(blocksY), 0.0);
    // Row y is in block row y*blocksY/height, so each task owns whole block rows and the double sums need no atomics.
    threadPool.parallelFor(blocksY, [&](int by) {
        const auto rowBegin = static_cast<int>(((static_cast<std::int64_t>(by) * height) + blocksY - 1) / blocksY);
        const auto rowEnd = static_cast<int>(((static_cast<std::int64_t>(by + 1) * height) + blocksY - 1) / blocksY);
        std::vector<int> counts(static_cast<std::size_t>(blocksX), 0);
        double* sums = means.data() + (static_cast<std::size_t>(by) * static_cast<std::size_t>(blocksX));
        for (int y = rowBegin; y < rowEnd; ++y) {
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            for (int x = 0; x < width; ++x) {
                const auto bx = static_cast<std::size_t>((static_cast<std::int64_t>(x) * blocksX) / width);
                sums[bx] += static_cast<double>(luminance[row + static_cast<std::size_t>(x)]);
                ++counts[bx];
            }
        }
        for (std::size_t bx = 0; bx < counts.size(); ++bx) {
            sums[bx] /= static_cast<double>(counts[bx]);
        }
    });
    return means;
}

// The adjusted cumulative over [0, range] of relative log2 luminance, sampled at bin boundaries, top knot exactly 1.
struct WardTransfer {
    std::vector<float> knots;
    double range = 0.0;
};

// Histogram of the foveal log samples under the linear ceiling onto `displayStops`; nullopt where the occupied range already fits.
[[nodiscard]] std::optional<WardTransfer> wardTransfer(std::vector<double>& samples, double displayStops) {
    const double range = *std::max_element(samples.begin(), samples.end());
    // Necessary for any adjustment, the occupied bins spanning at most the range; truncatedHistogram decides exactly.
    if (!(range > displayStops)) {
        return std::nullopt;
    }
    // Freedman & Diaconis 1981, the L2-optimal width for the foveal sample, with the interquartile range as the robust scale.
    const auto sampleCount = static_cast<double>(samples.size());
    const auto quantile = [&samples](double fraction) {
        const auto index = static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1));
        std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(index), samples.end());
        return samples[index];
    };
    double binWidth = 2.0 * (quantile(0.75) - quantile(0.25)) / std::cbrt(sampleCount);
    if (!(binWidth > 0.0)) {
        // Scott 1979 where the quartiles coincide: (24 sqrt(pi))^(1/3) sigma n^(-1/3), the same optimum under a Gaussian reference.
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / sampleCount;
        double variance = 0.0;
        for (const double sample : samples) {
            variance += (sample - mean) * (sample - mean);
        }
        binWidth = std::cbrt(24.0 * std::sqrt(std::numbers::pi)) * std::sqrt(variance / sampleCount) / std::cbrt(sampleCount);
    }
    // A histogram cannot occupy more bins than it has samples, so the asymptotic rules are capped where the expected count reaches one.
    const int bins = static_cast<int>(std::clamp(std::ceil(range / binWidth), 1.0, sampleCount));
    const double binStops = range / static_cast<double>(bins);
    std::vector<double> counts(static_cast<std::size_t>(bins), 0.0);
    for (const double sample : samples) {
        counts[static_cast<std::size_t>(std::min(bins - 1, static_cast<int>(sample / binStops)))] += 1.0;
    }
    // Display contrast may never exceed the world's: a bin's share of the display stops is at most its own share of world stops.
    const std::vector<double> truncated = truncatedHistogram(counts, binStops / displayStops);
    if (truncated.empty()) {
        return std::nullopt;
    }
    // Normalised by the cumulative's own total, so the top knot is exactly one and the brightest foveal level maps onto itself.
    WardTransfer transfer{std::vector<float>(static_cast<std::size_t>(bins) + 1, 0.0F), range};
    const double total = std::accumulate(truncated.begin(), truncated.end(), 0.0);
    double cumulative = 0.0;
    for (std::size_t bin = 0; bin < truncated.size(); ++bin) {
        cumulative += truncated[bin];
        transfer.knots[bin + 1] = static_cast<float>(cumulative / total);
    }
    transfer.knots.back() = 1.0F;
    return transfer;
}

// Land 1986's inverse-square surround, equal weight per octave over every level the frame supports: integral of G_t dt/t = 1/(pi r^2).
[[nodiscard]] std::vector<float> landSurround(const std::vector<float>& plane, int width, int height, ThreadPool& threadPool) {
    const std::vector<ScaleSpaceLevel> pyramid = buildOctavePyramid(plane, width, height, threadPool);
    if (pyramid.empty()) {
        return plane;
    }
    // Octave rungs are uniform in log t, so dt/t is the same at every rung and the Riemann sum over them is a plain mean.
    return octaveMean(pyramid, threadPool);
}

// Gamma-normalised Laplacian of one rung on the base grid: own-grid stencil times own-grid variance, the two decimations cancelling.
[[nodiscard]] std::vector<float> normalisedLaplacian(const ScaleSpaceLevel& level, int baseWidth, int baseHeight,
                                                     ThreadPool& threadPool) {
    std::vector<float> response(level.plane.size());
    laplacian5(level.plane, level.width, level.height, response, threadPool);
    const auto decimation = static_cast<float>(level.decimation);
    std::vector<float> expanded(static_cast<std::size_t>(baseWidth) * static_cast<std::size_t>(baseHeight), 0.0F);
    // Negated once here: a bright blob has a negative Laplacian, and DoG's fine-minus-coarse reads positive on one.
    addExpanded(ScaleSpaceLevel{std::move(response), level.width, level.height, level.decimation, level.baseVariance},
                -level.baseVariance / (decimation * decimation), expanded, baseWidth, baseHeight, threadPool);
    return expanded;
}

// Vertex of the parabola through three rungs equally spaced in log t (Lowe 2004 section 4), in the extremum's own sign.
[[nodiscard]] float scalePeak(float previous, float current, float next) {
    const float sign = std::copysign(1.0F, current);
    const float below = sign * previous;
    const float centre = sign * current;
    const float above = sign * next;
    const float curvature = below - (2.0F * centre) + above;
    // At a final argmax both neighbours are at most the centre, so curvature is negative unless all three are equal.
    return curvature < 0.0F ? sign * (centre - (((below - above) * (below - above)) / (8.0F * curvature))) : current;
}

}  // namespace

HdrImage luminanceAov(const HdrImage& beauty, ThreadPool& threadPool) {
    const std::vector<float> plane = luminancePlane(beauty, threadPool);
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            writeScalar(out, row + static_cast<std::size_t>(x), plane[row + static_cast<std::size_t>(x)]);
        }
    });
    return out;
}

HdrImage sobelAov(const HdrImage& beauty, ThreadPool& threadPool) {
    const std::vector<float> plane = luminancePlane(beauty, threadPool);
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    const int width = beauty.width;
    const int height = beauty.height;
    threadPool.parallelFor(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            const float tl = tap(plane, width, height, x - 1, y - 1);
            const float t = tap(plane, width, height, x, y - 1);
            const float tr = tap(plane, width, height, x + 1, y - 1);
            const float l = tap(plane, width, height, x - 1, y);
            const float r = tap(plane, width, height, x + 1, y);
            const float bl = tap(plane, width, height, x - 1, y + 1);
            const float b = tap(plane, width, height, x, y + 1);
            const float br = tap(plane, width, height, x + 1, y + 1);
            const float gx = (-tl - (2.0F * l) - bl) + tr + (2.0F * r) + br;
            const float gy = (-bl - (2.0F * b) - br) + tl + (2.0F * t) + tr;
            writeScalar(out, (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x),
                        std::sqrt((gx * gx) + (gy * gy)));
        }
    });
    return out;
}

float morletCarrier(float variance) {
    // Petkov 1995 eq. 4: the half-response bandwidth in octaves fixes sigma/lambda alone, so the carrier follows from the envelope.
    const double octaves = static_cast<double>(kMorletOctaves);
    const double ratio = (std::exp2(octaves) + 1.0) / (std::exp2(octaves) - 1.0);
    return static_cast<float>((2.0 * std::sqrt(std::numbers::ln2 / 2.0) * ratio) / std::sqrt(static_cast<double>(variance)));
}

int morletOrientations() {
    // sigma*omega depends only on the bandwidth, so the angular half-response width does too, and with it the cover of a half turn.
    const double octaves = static_cast<double>(kMorletOctaves);
    const double sigmaOmega = 2.0 * std::sqrt(std::numbers::ln2 / 2.0) * ((std::exp2(octaves) + 1.0) / (std::exp2(octaves) - 1.0));
    // Half response at a chord of sqrt(2 ln2)/sigma from the carrier, so adjacent orientations may be no further apart than this.
    const double angularWidth = 4.0 * std::asin(std::sqrt(2.0 * std::numbers::ln2) / (2.0 * sigmaOmega));
    return static_cast<int>(std::ceil(std::numbers::pi / angularWidth));
}

HdrImage gaborAov(const HdrImage& beauty, ThreadPool& threadPool) {
    const int width = beauty.width;
    const int height = beauty.height;
    const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::vector<float> plane = luminancePlane(beauty, threadPool);
    HdrImage out = makeBroadcastImage(width, height);

    // The finest scale the grid resolves, so the bank sits on the same rung the octave ladder starts from.
    const float variance = innerScaleVariance();
    const float carrier = morletCarrier(variance);
    const int orientations = morletOrientations();

    std::vector<float> lowpass = plane;
    diffuse(lowpass, width, height, variance, threadPool);

    std::vector<float> peak(pixels, 0.0F);
    std::vector<float> real(pixels);
    std::vector<float> imaginary(pixels);
    std::vector<float> phaseCos(static_cast<std::size_t>(width));
    std::vector<float> phaseSin(static_cast<std::size_t>(width));
    for (int orientation = 0; orientation < orientations; ++orientation) {
        // A half turn covers the bank: the magnitude at theta and theta+pi is the same, the two being complex conjugates.
        const double theta = (std::numbers::pi * static_cast<double>(orientation)) / static_cast<double>(orientations);
        const double stepX = static_cast<double>(carrier) * std::cos(theta);
        const double stepY = static_cast<double>(carrier) * std::sin(theta);
        // Split by the angle-addition formula, so the row loop carries no transcendental at all.
        for (int x = 0; x < width; ++x) {
            phaseCos[static_cast<std::size_t>(x)] = static_cast<float>(std::cos(stepX * static_cast<double>(x)));
            phaseSin[static_cast<std::size_t>(x)] = static_cast<float>(std::sin(stepX * static_cast<double>(x)));
        }
        // Morlet's admissibility term as the blurred plane wave itself, under the same mirror: zero mean at the border, not just inside.
        std::vector<float> meanCosX(phaseCos);
        std::vector<float> meanSinX(phaseSin);
        diffuse(meanCosX, width, 1, variance, threadPool);
        diffuse(meanSinX, width, 1, variance, threadPool);
        // Separable, so its own blur is the product of a width-long and a height-long one and costs nothing against the 2-D passes.
        std::vector<float> meanCosY(static_cast<std::size_t>(height));
        std::vector<float> meanSinY(static_cast<std::size_t>(height));
        for (int y = 0; y < height; ++y) {
            meanCosY[static_cast<std::size_t>(y)] = static_cast<float>(std::cos(stepY * static_cast<double>(y)));
            meanSinY[static_cast<std::size_t>(y)] = static_cast<float>(std::sin(stepY * static_cast<double>(y)));
        }
        diffuse(meanCosY, 1, height, variance, threadPool);
        diffuse(meanSinY, 1, height, variance, threadPool);
        threadPool.parallelFor(height, [&](int y) {
            const auto rowCos = static_cast<float>(std::cos(stepY * static_cast<double>(y)));
            const auto rowSin = static_cast<float>(std::sin(stepY * static_cast<double>(y)));
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = row + static_cast<std::size_t>(x);
                const float cosPhase = (phaseCos[static_cast<std::size_t>(x)] * rowCos) - (phaseSin[static_cast<std::size_t>(x)] * rowSin);
                const float sinPhase = (phaseSin[static_cast<std::size_t>(x)] * rowCos) + (phaseCos[static_cast<std::size_t>(x)] * rowSin);
                // Demodulate, blur, remodulate: the plane wave factors out of the convolution, so every orientation stays separable.
                real[pixel] = plane[pixel] * cosPhase;
                imaginary[pixel] = -plane[pixel] * sinPhase;
            }
        });
        diffuse(real, width, height, variance, threadPool);
        diffuse(imaginary, width, height, variance, threadPool);
        // Remodulating is a rotation by the carrier's phase, which a magnitude discards, so only the admissibility term is left to apply.
        threadPool.parallelFor(height, [&](int y) {
            const float rowCos = meanCosY[static_cast<std::size_t>(y)];
            const float rowSin = meanSinY[static_cast<std::size_t>(y)];
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = row + static_cast<std::size_t>(x);
                // The blurred plane wave under the same mirror, which is what a constant field would have produced right here.
                const float meanReal = (meanCosX[static_cast<std::size_t>(x)] * rowCos) -
                                        (meanSinX[static_cast<std::size_t>(x)] * rowSin);
                const float meanImaginary = (meanSinX[static_cast<std::size_t>(x)] * rowCos) +
                                             (meanCosX[static_cast<std::size_t>(x)] * rowSin);
                const float centredReal = real[pixel] - (meanReal * lowpass[pixel]);
                const float centredImaginary = imaginary[pixel] + (meanImaginary * lowpass[pixel]);
                // Squared, so the root is paid once per texel at the end rather than once per texel per orientation; max commutes with it.
                peak[pixel] = std::max(peak[pixel], (centredReal * centredReal) + (centredImaginary * centredImaginary));
            }
        });
    }

    threadPool.parallelFor(height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        for (int x = 0; x < width; ++x) {
            writeScalar(out, row + static_cast<std::size_t>(x), std::sqrt(peak[row + static_cast<std::size_t>(x)]));
        }
    });
    return out;
}

HdrImage hsvAov(const HdrImage& beauty, ThreadPool& threadPool) {
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const std::size_t texel = (row + static_cast<std::size_t>(x)) * 4;
            const float r = beauty.rgba[texel];
            const float g = beauty.rgba[texel + 1];
            const float b = beauty.rgba[texel + 2];
            const float value = std::max({r, g, b});
            const float chroma = value - std::min({r, g, b});
            // Exact branches, not the shader's 1e-10 guard: hue is undefined on the achromatic axis, saturation on black (Smith 1978).
            const float saturation = value > 0.0F ? chroma / value : 0.0F;
            float hue = 0.0F;
            if (chroma > 0.0F) {
                if (value == r) {
                    hue = (g - b) / chroma;
                } else if (value == g) {
                    hue = 2.0F + ((b - r) / chroma);
                } else {
                    hue = 4.0F + ((r - g) / chroma);
                }
                // Sextant index to a [0,1) turn, wrapping the negative sixth the red sector produces.
                hue /= 6.0F;
                if (hue < 0.0F) {
                    hue += 1.0F;
                }
            }
            out.rgba[texel] = hue;
            out.rgba[texel + 1] = saturation;
            out.rgba[texel + 2] = value;
            out.rgba[texel + 3] = 1.0F;
        }
    });
    return out;
}

HdrImage dogAov(const HdrImage& beauty, ThreadPool& threadPool) {
    const std::vector<float> plane = luminancePlane(beauty, threadPool);
    // Only the two finest rungs are read, so the cascade stops there rather than building and copying the coarser octaves.
    const std::vector<ScaleSpaceLevel> pyramid = buildOctavePyramid(plane, beauty.width, beauty.height, threadPool, 2);
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    // A frame narrower than the second rung's kernel support holds fewer than two octaves, so the band is empty there.
    if (pyramid.size() < 2) {
        return out;
    }
    // Both finest levels sit on the base grid: the inner scale is half the decimation variance, so the cascade cannot have halved yet.
    const std::vector<float>& fine = pyramid[0].plane;
    const std::vector<float>& coarse = pyramid[1].plane;
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            // Signed, not a magnitude: polarity separates a bright blob from a dark one, and Sobel already reports gradient magnitude.
            writeScalar(out, pixel, fine[pixel] - coarse[pixel]);
        }
    });
    return out;
}

HdrImage logAov(const HdrImage& beauty, ThreadPool& threadPool) {
    const std::vector<float> plane = luminancePlane(beauty, threadPool);
    const std::vector<ScaleSpaceLevel> pyramid = buildOctavePyramid(plane, beauty.width, beauty.height, threadPool);
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    const auto pixels = static_cast<std::size_t>(beauty.width) * static_cast<std::size_t>(beauty.height);

    // The largest-magnitude rung so far per texel, and its value refined across scale: a sliding window of three rungs suffices.
    std::vector<float> rungPeak(pixels, 0.0F);
    std::vector<float> extremum(pixels, 0.0F);
    std::vector<float> previous;
    std::vector<float> current =
        pyramid.empty() ? std::vector<float>{} : normalisedLaplacian(pyramid.front(), beauty.width, beauty.height, threadPool);
    for (std::size_t rung = 0; rung < pyramid.size(); ++rung) {
        std::vector<float> next = rung + 1 < pyramid.size()
                                      ? normalisedLaplacian(pyramid[rung + 1], beauty.width, beauty.height, threadPool)
                                      : std::vector<float>{};
        const bool interior = !previous.empty() && !next.empty();
        threadPool.parallelFor(beauty.height, [&](int y) {
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
            for (int x = 0; x < beauty.width; ++x) {
                const std::size_t pixel = row + static_cast<std::size_t>(x);
                if (!(std::fabs(current[pixel]) > std::fabs(rungPeak[pixel]))) {
                    continue;
                }
                rungPeak[pixel] = current[pixel];
                extremum[pixel] = interior ? scalePeak(previous[pixel], current[pixel], next[pixel]) : current[pixel];
            }
        });
        previous = std::move(current);
        current = std::move(next);
    }

    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            // Signed: the zero crossings are the edges (Marr & Hildreth 1980), which a magnitude would erase.
            writeScalar(out, pixel, extremum[pixel]);
        }
    });
    return out;
}

HdrImage colourOpponentAov(const HdrImage& beauty, ThreadPool& threadPool) {
    const pathtracer::scene::cone::OpponentBasis& basis = pathtracer::scene::cone::opponentBasis();
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const std::size_t texel = (row + static_cast<std::size_t>(x)) * 4;
            const float red = beauty.rgba[texel];
            const float green = beauty.rgba[texel + 1];
            const float blue = beauty.rgba[texel + 2];
            const glm::vec2 difference(red - green, blue - green);
            const float sum = (basis.denominator.r * red) + (basis.denominator.g * green) + (basis.denominator.b * blue);
            // Zero where no light arrives: the cone chromaticity of black is undefined, not achromatic, and must not read as white.
            const float scale = sum > 0.0F ? 1.0F / sum : 0.0F;
            out.rgba[texel] = glm::dot(basis.redGreenNumerator, difference) * scale;
            out.rgba[texel + 1] = glm::dot(basis.blueYellowNumerator, difference) * scale;
            out.rgba[texel + 2] = 0.0F;
            out.rgba[texel + 3] = 1.0F;
        }
    });
    return out;
}

HdrImage retinexAov(const HdrImage& beauty, ThreadPool& threadPool) {
    const int width = beauty.width;
    const int height = beauty.height;
    const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    HdrImage out = makeBroadcastImage(width, height);
    std::vector<float> radiance(pixels);
    std::vector<float> mask(pixels);
    // Reused while consecutive channels share a validity pattern, which they do wherever a texel is either lit or wholly dark.
    std::vector<float> cachedMask;
    std::vector<float> cachedMaskSurround;

    for (int channel = 0; channel < 3; ++channel) {
        threadPool.parallelFor(height, [&](int y) {
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = row + static_cast<std::size_t>(x);
                radiance[pixel] = beauty.rgba[(pixel * 4) + static_cast<std::size_t>(channel)];
                // Log radiance is undefined at zero, and a channel can be zero where its neighbours are not, so validity is per channel.
                mask[pixel] = radiance[pixel] > 0.0F ? 1.0F : 0.0F;
            }
        });
        // The reference cancels between the log and its own surround, so retinex reads the same reflectance whatever the frame's gain.
        const std::vector<float> logRadiance = relativeLog2(radiance, width, height, threadPool).plane;
        const bool everywhereValid = std::find(mask.begin(), mask.end(), 0.0F) == mask.end();
        const std::vector<float> surround = landSurround(logRadiance, width, height, threadPool);
        if (!everywhereValid && mask != cachedMask) {
            cachedMaskSurround = landSurround(mask, width, height, threadPool);
            cachedMask = mask;
        }
        threadPool.parallelFor(height, [&](int y) {
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = row + static_cast<std::size_t>(x);
                // Normalised convolution (Knutsson & Westin 1993): on a fully valid channel the weight is exactly 1, so this is plain blur.
                const float weight = everywhereValid ? 1.0F : cachedMaskSurround[pixel];
                const float reflectance =
                    mask[pixel] > 0.0F && weight > 0.0F ? std::exp2(logRadiance[pixel] - (surround[pixel] / weight)) : 0.0F;
                out.rgba[(pixel * 4) + static_cast<std::size_t>(channel)] = reflectance;
                out.rgba[(pixel * 4) + 3] = 1.0F;
            }
        });
    }
    return out;
}

HdrImage claheAov(const HdrImage& beauty, float pixelsPerRadian, ThreadPool& threadPool) {
    const std::vector<float> luminance = luminancePlane(beauty, threadPool);
    // Unlit texels pass through untouched, so the copy is both the identity fallback and the carrier of the preserved chromaticity.
    HdrImage out = beauty;
    const std::vector<double> foveal = fovealMeans(luminance, beauty.width, beauty.height, pixelsPerRadian, threadPool);

    // The darkest lit foveal sample is the log reference, so every sample is non-negative and the top one is the world range.
    double reference = 0.0;
    for (const double mean : foveal) {
        reference = mean > 0.0 && (reference == 0.0 || mean < reference) ? mean : reference;
    }
    if (reference == 0.0) {
        return out;
    }
    const LogReference relativeLog(reference);
    std::vector<double> samples;
    samples.reserve(foveal.size());
    for (const double mean : foveal) {
        if (mean > 0.0) {
            samples.push_back(relativeLog(mean));
        }
    }
    const double displayStops = std::log2(kSrgbReferenceWhite / kSrgbReferenceBlack);
    const std::optional<WardTransfer> transfer = wardTransfer(samples, displayStops);
    // No transfer: the occupied world range fits the display, Ward Larson's linear case, which at unit gain is the identity.
    if (!transfer.has_value()) {
        return out;
    }

    const int bins = static_cast<int>(transfer->knots.size()) - 1;
    const auto brightest = static_cast<float>(transfer->range);
    const auto stops = static_cast<float>(displayStops);
    const auto inverseBinStops = static_cast<float>(static_cast<double>(bins) / transfer->range);
    const auto referenceFloat = static_cast<float>(reference);
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            out.rgba[(pixel * 4) + 3] = 1.0F;
            if (luminance[pixel] <= 0.0F) {
                continue;
            }
            // Past the foveal extremes the cumulative is flat, 0 or 1: the display floor and white Ward Larson clamps to.
            const float position = std::clamp(static_cast<float>(relativeLog(luminance[pixel])) * inverseBinStops, 0.0F,
                                              static_cast<float>(bins));
            const int bin = std::min(static_cast<int>(position), bins - 1);
            const float* knot = transfer->knots.data() + bin;
            const float cdf = knot[0] + ((knot[1] - knot[0]) * (position - static_cast<float>(bin)));
            // log L_d = log L_dmax - D (1 - P), anchored at the brightest foveal level so the operator stays degree one in radiance.
            const float scale = (referenceFloat * std::exp2(brightest - (stops * (1.0F - cdf)))) / luminance[pixel];
            out.rgba[pixel * 4] *= scale;
            out.rgba[(pixel * 4) + 1] *= scale;
            out.rgba[(pixel * 4) + 2] *= scale;
        }
    });
    return out;
}

HdrImage snrAov(const HdrImage& beauty, const float* beautyLuminanceM2, int samples, ThreadPool& threadPool) {
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    // A single sample carries no dispersion, so the ratio is undefined rather than infinite, and a black frame says so.
    if (beautyLuminanceM2 == nullptr || samples < 2) {
        return out;
    }
    // Standard error of the MEAN, not of one sample: `beauty` is a mean, and the question is how well its value is known.
    const float passes = static_cast<float>(samples);
    // Widened before the product, not after: n(n-1) leaves int at 46341 passes, which an uncapped accumulation reaches.
    const float inverseDegrees = 1.0F / (passes * (passes - 1.0F));
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            const std::size_t texel = pixel * 4;
            const float luminance = (beauty.rgba[texel] * kRec709LuminanceWeights.r) +
                                    (beauty.rgba[texel + 1] * kRec709LuminanceWeights.g) +
                                    (beauty.rgba[texel + 2] * kRec709LuminanceWeights.b);
            const float standardError = std::sqrt(beautyLuminanceM2[pixel] * inverseDegrees);
            // Zero dispersion is a converged texel, not an infinite ratio; a background pixel every pass agrees on is the usual case.
            writeScalar(out, pixel, standardError > 0.0F ? luminance / standardError : 0.0F);
        }
    });
    return out;
}

HdrImage evaluateFilterAov(AovId aov, const FilterInput& input, ThreadPool& threadPool) {
    switch (aov) {
        case AovId::HSV:       return hsvAov(input.beauty, threadPool);
        case AovId::Luminance: return luminanceAov(input.beauty, threadPool);
        case AovId::Sobel:     return sobelAov(input.beauty, threadPool);
        case AovId::Gabor:     return gaborAov(input.beauty, threadPool);
        case AovId::DoG:       return dogAov(input.beauty, threadPool);
        case AovId::LoG:       return logAov(input.beauty, threadPool);
        case AovId::ColourOpponent:  return colourOpponentAov(input.beauty, threadPool);
        case AovId::Retinex:   return retinexAov(input.beauty, threadPool);
        case AovId::CLAHE:     return claheAov(input.beauty, input.pixelsPerRadian, threadPool);
        case AovId::SNR:       return snrAov(input.beauty, input.beautyLuminanceM2, input.samples, threadPool);

        // The lanes their own producers write. No default arm: -Werror then makes an unrouted new filter a compile error.
        case AovId::Beauty:
        case AovId::Wireframe:
        case AovId::Alpha:
        case AovId::Depth:
        case AovId::Lookahead:
        case AovId::WorldPos:
        case AovId::UV:
        case AovId::Normal:
        case AovId::GeomNormal:
        case AovId::Albedo:
        case AovId::Metallic:
        case AovId::Roughness:
        case AovId::Tangent:
        case AovId::ObjectID:
        case AovId::AO:
        case AovId::Fresnel:
        case AovId::IOR:
        case AovId::BounceCount:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::Shadow:
        case AovId::Count:
            break;
    }
    return {};
}

}  // namespace pathtracer::debug
