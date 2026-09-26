#include "pathtracer/debug/aov_filters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <glm/gtc/constants.hpp>

#include "pathtracer/debug/scale_space.h"

namespace pathtracer::debug {

namespace {

using pathtracer::gfx::HdrImage;
using pathtracer::scene::ThreadPool;

// Single-channel Rec.709 luminance, shared by Sobel and Gabor. Materialised once: Sobel reads 8 neighbours per pixel and Gabor 25.
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

std::array<float, kGaborKernelSize> buildGaborKernel() {
    constexpr float kSigma = 1.4F;
    constexpr float kLambda = 4.0F;
    constexpr float kGamma = 0.5F;
    constexpr std::array<float, kGaborOrientations> kOrientationsDeg = {0.0F, 45.0F, 90.0F, 135.0F};

    std::array<float, kGaborKernelSize> kernel{};
    for (int o = 0; o < kGaborOrientations; ++o) {
        const float theta = glm::radians(kOrientationsDeg[static_cast<std::size_t>(o)]);
        int tapIndex = 0;
        for (int dy = -kGaborRadius; dy <= kGaborRadius; ++dy) {
            for (int dx = -kGaborRadius; dx <= kGaborRadius; ++dx) {
                const auto x = static_cast<float>(dx);
                const auto y = static_cast<float>(dy);
                const float xp = (x * std::cos(theta)) + (y * std::sin(theta));
                const float yp = (-x * std::sin(theta)) + (y * std::cos(theta));
                const float envelope = std::exp(
                    -((xp * xp) + (kGamma * kGamma * yp * yp)) / (2.0F * kSigma * kSigma));
                // Odd/quadrature carrier (sin, not cos) -- edge-sensitive, not bar/ridge-sensitive.
                const float carrier = std::sin(2.0F * glm::pi<float>() * xp / kLambda);
                kernel[(static_cast<std::size_t>(o) * kGaborTaps) + static_cast<std::size_t>(tapIndex)] =
                    envelope * carrier;
                ++tapIndex;
            }
        }
    }
    return kernel;
}

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

HdrImage gaborAov(const HdrImage& beauty, ThreadPool& threadPool) {
    // Built once per process, not per call: the bank depends only on its compile-time parameters, as at shader setup.
    static const std::array<float, kGaborKernelSize> kernel = buildGaborKernel();

    const std::vector<float> plane = luminancePlane(beauty, threadPool);
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    const int width = beauty.width;
    const int height = beauty.height;
    threadPool.parallelFor(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            std::array<float, kGaborOrientations> response{};
            int tapIndex = 0;
            // Each neighbourhood texel is fetched once and reused across all four orientations, as the shader does.
            for (int dy = -kGaborRadius; dy <= kGaborRadius; ++dy) {
                for (int dx = -kGaborRadius; dx <= kGaborRadius; ++dx) {
                    const float lum = tap(plane, width, height, x + dx, y + dy);
                    for (int o = 0; o < kGaborOrientations; ++o) {
                        response[static_cast<std::size_t>(o)] +=
                            kernel[(static_cast<std::size_t>(o) * kGaborTaps) + static_cast<std::size_t>(tapIndex)] * lum;
                    }
                    ++tapIndex;
                }
            }
            float magnitude = 0.0F;
            for (const float value : response) {
                magnitude = std::max(magnitude, std::fabs(value));
            }
            writeScalar(out, (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x),
                        magnitude);
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

HdrImage logAov(const HdrImage& beauty, float verticalFovRadians, ThreadPool& threadPool) {
    const std::vector<float> plane = luminancePlane(beauty, threadPool);
    const std::vector<ScaleSpaceLevel> pyramid = buildOctavePyramid(plane, beauty.width, beauty.height, threadPool);
    HdrImage out = makeBroadcastImage(beauty.width, beauty.height);
    const auto pixels = static_cast<std::size_t>(beauty.width) * static_cast<std::size_t>(beauty.height);
    std::vector<float> extremum(pixels, 0.0F);
    std::vector<float> extremumFrequency(pixels, 0.0F);
    // Derived from the traced height, not the authored one, so the frequency axis holds unchanged at any interactive render scale.
    const float pixelsPerDegree = static_cast<float>(beauty.height) / glm::degrees(verticalFovRadians);

    for (const ScaleSpaceLevel& level : pyramid) {
        std::vector<float> response(level.plane.size());
        laplacian5(level.plane, level.width, level.height, response, threadPool);
        // Own-grid Laplacian times own-grid variance is exactly the base-grid gamma-normalised response, the two decimations cancelling.
        const auto decimation = static_cast<float>(level.decimation);
        const float ownVariance = level.baseVariance / (decimation * decimation);
        // The band's peak radial frequency: |-w^2 exp(-w^2 t/2)| is stationary at w = sqrt(2/t), carried to the camera's angular scale.
        const float frequency =
            (std::sqrt(2.0F / level.baseVariance) / (2.0F * glm::pi<float>())) * pixelsPerDegree;
        const std::vector<float> expanded =
            expandToBase(ScaleSpaceLevel{std::move(response), level.width, level.height, level.decimation,
                                         level.baseVariance},
                         beauty.width, beauty.height, threadPool);
        threadPool.parallelFor(beauty.height, [&](int y) {
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
            for (int x = 0; x < beauty.width; ++x) {
                const std::size_t pixel = row + static_cast<std::size_t>(x);
                const float normalised = ownVariance * expanded[pixel];
                if (std::fabs(normalised) > std::fabs(extremum[pixel])) {
                    extremum[pixel] = normalised;
                    extremumFrequency[pixel] = frequency;
                }
            }
        });
    }

    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            const std::size_t texel = pixel * 4;
            out.rgba[texel] = std::fabs(extremum[pixel]);
            out.rgba[texel + 1] = extremumFrequency[pixel];
            // A bright blob has a negative Laplacian at its centre, so the reported polarity negates the response's sign.
            out.rgba[texel + 2] = extremum[pixel] > 0.0F ? -1.0F : (extremum[pixel] < 0.0F ? 1.0F : 0.0F);
            out.rgba[texel + 3] = 1.0F;
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
        case AovId::LoG:       return logAov(input.beauty, input.verticalFovRadians, threadPool);

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
