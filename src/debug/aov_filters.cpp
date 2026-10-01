#include "pathtracer/debug/aov_filters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

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

    // The finest scale the grid resolves, the same fine scale DoG starts from.
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
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    const float fineVariance = innerScaleVariance();
    // One octave up, sigma doubled: variance 4t, reached from the fine level by the semigroup's step 3t.
    const float coarseStep = (4.0F * fineVariance) - fineVariance;
    // A frame narrower than the coarse step's kernel support would report the mirror, not the image, so the band is empty there.
    if ((2 * (static_cast<int>(discreteGaussianKernel(coarseStep).size()) - 1)) + 1 > std::min(beauty.width, beauty.height)) {
        return out;
    }
    std::vector<float> fine = luminancePlane(beauty, threadPool);
    diffuse(fine, beauty.width, beauty.height, fineVariance, threadPool);
    std::vector<float> coarse = fine;
    diffuse(coarse, beauty.width, beauty.height, coarseStep, threadPool);
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
        case AovId::ColourOpponent:  return colourOpponentAov(input.beauty, threadPool);
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
