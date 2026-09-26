#include "pathtracer/debug/aov_filters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <numeric>
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

// One tile's transfer function, sampled at bin boundaries: the clipped cumulative histogram Zuiderveld 1994 equalises with.
void clippedTransfer(const float* counts, int bins, int samples, float* knots) {
    int occupied = 0;
    for (int bin = 0; bin < bins; ++bin) {
        occupied += counts[bin] > 0.0F ? 1 : 0;
    }
    if (occupied == 0) {
        // No samples is no evidence, and the uniform density a contrast-preserving map would give is the linear transfer itself.
        for (int bin = 0; bin <= bins; ++bin) {
            knots[bin] = static_cast<float>(bin) / static_cast<float>(bins);
        }
        return;
    }
    // Ward Larson 1997's contrast ceiling read adaptively: uniform over the support this tile occupies, the only non-vacuous reference.
    const float ceiling = 1.0F / static_cast<float>(occupied);
    const float inverseSamples = 1.0F / static_cast<float>(samples);
    std::vector<float> descending(static_cast<std::size_t>(bins));
    for (int bin = 0; bin < bins; ++bin) {
        descending[static_cast<std::size_t>(bin)] = counts[bin] * inverseSamples;
    }
    std::sort(descending.begin(), descending.end(), std::greater<>{});

    // Clipping every bin totals bins/occupied >= 1, so the sweep below always has this level as its last admissible one.
    float redistribution = ceiling;
    double clipped = 0.0;
    for (int k = 0; k < bins; ++k) {
        // The fixed point of Ward Larson's truncate-and-redistribute: p = min(phat + delta, ceiling) with delta set by the total being 1.
        const auto candidate =
            static_cast<float>((clipped - (static_cast<double>(k) * ceiling)) / static_cast<double>(bins - k));
        const bool aboveLower = k == 0 || descending[static_cast<std::size_t>(k) - 1] + candidate >= ceiling;
        if (candidate >= 0.0F && aboveLower && descending[static_cast<std::size_t>(k)] + candidate <= ceiling) {
            redistribution = candidate;
            break;
        }
        clipped += static_cast<double>(descending[static_cast<std::size_t>(k)]);
    }
    knots[0] = 0.0F;
    for (int bin = 0; bin < bins; ++bin) {
        knots[bin + 1] = knots[bin] + std::min((counts[bin] * inverseSamples) + redistribution, ceiling);
    }
}

// The coarsest scale the frame supports, as a base-grid plane: the pyramid's endpoint, so no surround extent has to be chosen.
[[nodiscard]] std::vector<float> coarsestSurround(const std::vector<float>& plane, int width, int height,
                                                  ThreadPool& threadPool) {
    const std::vector<ScaleSpaceLevel> pyramid = buildOctavePyramid(plane, width, height, threadPool);
    if (pyramid.empty()) {
        return plane;
    }
    return expandToBase(pyramid.back(), width, height, threadPool);
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
    const std::vector<ScaleSpaceLevel> pyramid = buildOctavePyramid(plane, beauty.width, beauty.height, threadPool);
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    // Below roughly six pixels the inner scale already fills the frame, so there are not two octaves to difference and the band is empty.
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
    std::vector<float> extremum(pixels, 0.0F);

    for (const ScaleSpaceLevel& level : pyramid) {
        std::vector<float> response(level.plane.size());
        laplacian5(level.plane, level.width, level.height, response, threadPool);
        // Own-grid Laplacian times own-grid variance is exactly the base-grid gamma-normalised response, the two decimations cancelling.
        const auto decimation = static_cast<float>(level.decimation);
        const float ownVariance = level.baseVariance / (decimation * decimation);
        const std::vector<float> expanded =
            expandToBase(ScaleSpaceLevel{std::move(response), level.width, level.height, level.decimation,
                                         level.baseVariance},
                         beauty.width, beauty.height, threadPool);
        threadPool.parallelFor(beauty.height, [&](int y) {
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
            for (int x = 0; x < beauty.width; ++x) {
                const std::size_t pixel = row + static_cast<std::size_t>(x);
                // Sign negated once here: a bright blob has a negative Laplacian, and DoG's fine-minus-coarse reads positive on one.
                const float normalised = -ownVariance * expanded[pixel];
                if (std::fabs(normalised) > std::fabs(extremum[pixel])) {
                    extremum[pixel] = normalised;
                }
            }
        });
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
        const std::vector<float> surround = coarsestSurround(logRadiance, width, height, threadPool);
        if (!everywhereValid && mask != cachedMask) {
            cachedMaskSurround = coarsestSurround(mask, width, height, threadPool);
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

HdrImage claheAov(const HdrImage& beauty, float verticalFovRadians, ThreadPool& threadPool) {
    const int width = beauty.width;
    const int height = beauty.height;
    const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::vector<float> luminance = luminancePlane(beauty, threadPool);
    // Unlit texels pass through untouched, so the copy is both the identity fallback and the carrier of the preserved chromaticity.
    HdrImage out = beauty;

    // Equalisation's uniform-output criterion is only meaningful on a perceptually uniform axis; luminance response is log (Weber-Fechner).
    const RelativeLog logLuminance = relativeLog2(luminance, width, height, threadPool);
    if (logLuminance.reference == 0.0F) {
        return out;
    }
    std::vector<float> samples;
    samples.reserve(pixels);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        if (luminance[pixel] > 0.0F) {
            samples.push_back(logLuminance.plane[pixel]);
        }
    }
    const float range = *std::max_element(samples.begin(), samples.end());
    // A frame with no dynamic range has no histogram to equalise, and the operator is exactly the identity there.
    if (!(range > 0.0F)) {
        return out;
    }

    // One degree of visual angle, the extent retinal light adaptation pools over (Ward Larson et al. 1997), so the grid is FOV-anchored.
    const float pixelsPerDegree = static_cast<float>(height) / glm::degrees(verticalFovRadians);
    // Capped at one tile per pixel: below that the adaptation extent is finer than the sampling, and a tile row could hold no rows at all.
    const int tilesX = std::clamp(static_cast<int>(std::lround(static_cast<float>(width) / pixelsPerDegree)), 1, width);
    const int tilesY = std::clamp(static_cast<int>(std::lround(static_cast<float>(height) / pixelsPerDegree)), 1, height);
    const int tileCount = tilesX * tilesY;
    const float pitchX = static_cast<float>(width) / static_cast<float>(tilesX);
    const float pitchY = static_cast<float>(height) / static_cast<float>(tilesY);
    const double samplesPerTile = static_cast<double>(samples.size()) / static_cast<double>(tileCount);

    // Freedman & Diaconis 1981, the L2-optimal width for the sample one tile holds, with the interquartile range as the robust scale.
    const auto quantile = [&samples](double fraction) {
        const auto index = static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1));
        std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(index), samples.end());
        return static_cast<double>(samples[index]);
    };
    double binWidth = 2.0 * (quantile(0.75) - quantile(0.25)) / std::cbrt(samplesPerTile);
    if (!(binWidth > 0.0)) {
        // Scott 1979 where the quartiles coincide: the same asymptotic optimum under a Gaussian reference, on the only scale left.
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
        double variance = 0.0;
        for (const float sample : samples) {
            variance += (static_cast<double>(sample) - mean) * (static_cast<double>(sample) - mean);
        }
        binWidth = 3.49 * std::sqrt(variance / static_cast<double>(samples.size())) / std::cbrt(samplesPerTile);
    }
    // A histogram cannot occupy more bins than it has samples, so the asymptotic rules are capped where the expected count reaches one.
    const int bins = std::clamp(static_cast<int>(std::ceil(static_cast<double>(range) / binWidth)), 1,
                                std::max(1, static_cast<int>(samplesPerTile)));
    const float binStops = range / static_cast<float>(bins);

    // Shared by the scatter and the blend, so a texel's histogram and its transfer functions cannot disagree about which tile it is in.
    const auto tileIndexX = [&](int x) {
        return std::min(tilesX - 1, static_cast<int>((static_cast<float>(x) + 0.5F) / pitchX));
    };
    // Descending, so the last write to each entry is that tile row's first image row; entry tilesY stays the sentinel end.
    std::vector<int> bandBegin(static_cast<std::size_t>(tilesY) + 1, height);
    for (int y = height - 1; y >= 0; --y) {
        bandBegin[static_cast<std::size_t>(std::min(tilesY - 1, static_cast<int>((static_cast<float>(y) + 0.5F) / pitchY)))] = y;
    }

    std::vector<float> counts(static_cast<std::size_t>(tileCount) * static_cast<std::size_t>(bins), 0.0F);
    std::vector<int> tileSamples(static_cast<std::size_t>(tileCount), 0);
    // Parallel over tile rows, not pixels: each task owns one row of tiles outright, so the scatter needs no atomics.
    threadPool.parallelFor(tilesY, [&](int ty) {
        for (int y = bandBegin[static_cast<std::size_t>(ty)]; y < bandBegin[static_cast<std::size_t>(ty) + 1]; ++y) {
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = row + static_cast<std::size_t>(x);
                if (luminance[pixel] <= 0.0F) {
                    continue;
                }
                const int tile = (ty * tilesX) + tileIndexX(x);
                const int bin = std::min(bins - 1, static_cast<int>(logLuminance.plane[pixel] / binStops));
                counts[(static_cast<std::size_t>(tile) * static_cast<std::size_t>(bins)) + static_cast<std::size_t>(bin)] += 1.0F;
                ++tileSamples[static_cast<std::size_t>(tile)];
            }
        }
    });

    // One cumulative transfer function per tile, sampled at bin boundaries, so evaluating it is a lerp between two adjacent knots.
    std::vector<float> transfer(static_cast<std::size_t>(tileCount) * static_cast<std::size_t>(bins + 1));
    threadPool.parallelFor(tileCount, [&](int tile) {
        clippedTransfer(counts.data() + (static_cast<std::size_t>(tile) * static_cast<std::size_t>(bins)), bins,
                        tileSamples[static_cast<std::size_t>(tile)],
                        transfer.data() + (static_cast<std::size_t>(tile) * static_cast<std::size_t>(bins + 1)));
    });

    threadPool.parallelFor(height, [&](int y) {
        // Zuiderveld 1994's bilinear blend of the four surrounding tiles, written as nested lerps so the four weights sum to exactly one.
        const float gridY = ((static_cast<float>(y) + 0.5F) / pitchY) - 0.5F;
        const auto floorY = static_cast<int>(std::floor(gridY));
        const float fractionY = gridY - static_cast<float>(floorY);
        const int rowA = std::clamp(floorY, 0, tilesY - 1) * tilesX;
        const int rowB = std::clamp(floorY + 1, 0, tilesY - 1) * tilesX;
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            out.rgba[(pixel * 4) + 3] = 1.0F;
            if (luminance[pixel] <= 0.0F) {
                continue;
            }
            const float position = logLuminance.plane[pixel] / binStops;
            const int bin = std::clamp(static_cast<int>(position), 0, bins - 1);
            const float fractionBin = position - static_cast<float>(bin);
            const float gridX = ((static_cast<float>(x) + 0.5F) / pitchX) - 0.5F;
            const auto floorX = static_cast<int>(std::floor(gridX));
            const float fractionX = gridX - static_cast<float>(floorX);
            const int columnA = std::clamp(floorX, 0, tilesX - 1);
            const int columnB = std::clamp(floorX + 1, 0, tilesX - 1);
            const auto evaluate = [&](int tile) {
                const float* knots = transfer.data() + (static_cast<std::size_t>(tile) * static_cast<std::size_t>(bins + 1));
                return knots[bin] + ((knots[bin + 1] - knots[bin]) * fractionBin);
            };
            const float upper = evaluate(rowA + columnA);
            const float lower = evaluate(rowB + columnA);
            const float upperRight = evaluate(rowA + columnB);
            const float lowerRight = evaluate(rowB + columnB);
            const float top = upper + ((upperRight - upper) * fractionX);
            const float bottom = lower + ((lowerRight - lower) * fractionX);
            const float equalised = top + ((bottom - top) * fractionY);
            // Chromaticity preserved: only the luminance is remapped, and the texel keeps its ratio to the new value exactly.
            const float scale = (logLuminance.reference * std::exp2(equalised * range)) / luminance[pixel];
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
        case AovId::CLAHE:     return claheAov(input.beauty, input.verticalFovRadians, threadPool);
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
