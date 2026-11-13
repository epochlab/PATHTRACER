#include "pathtracer/debug/aov_filters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/path_trace_driver.h"

namespace pathtracer::debug {

namespace {

using pathtracer::gfx::HdrImage;
using pathtracer::scene::ThreadPool;

// The filter's output image, zeroed at the channel count aovChannels declares for it.
[[nodiscard]] HdrImage makeAovImage(AovId aov, int width, int height) {
    return pathtracer::gfx::makeImage(width, height, aovChannels(aov));
}

}  // namespace

HdrImage luminanceAov(const HdrImage& beauty, ThreadPool& threadPool) {
    const glm::vec3 weights = pathtracer::scene::cie::sceneLuminanceWeights();
    HdrImage out = makeAovImage(AovId::Luminance, beauty.width, beauty.height);
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const std::size_t texel = (row + static_cast<std::size_t>(x)) * static_cast<std::size_t>(beauty.channels);
            out.texels[row + static_cast<std::size_t>(x)] = (beauty.texels[texel] * weights.r) + (beauty.texels[texel + 1] * weights.g) +
                                                            (beauty.texels[texel + 2] * weights.b);
        }
    });
    return out;
}

HdrImage hsvAov(const HdrImage& beauty, ThreadPool& threadPool) {
    HdrImage out = makeAovImage(AovId::HSV, beauty.width, beauty.height);
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const glm::vec3 rgb = beauty.rgb(row + static_cast<std::size_t>(x));
            const float r = rgb.r;
            const float g = rgb.g;
            const float b = rgb.b;
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
            writeTexel(out, x, y, glm::vec3(hue, saturation, value));
        }
    });
    return out;
}

HdrImage snrAov(int width, int height, const float* replicateLuminance, int replicates, int samples, ThreadPool& threadPool) {
    HdrImage out = makeAovImage(AovId::SNR, width, height);
    // Replicates not yet drawn hold no estimate; fewer than two populated carry no spread, so the ratio is undefined and reads zero.
    const int populated = std::min(samples, replicates);
    if (replicateLuminance == nullptr || populated < 2) {
        return out;
    }
    const auto stride = static_cast<std::size_t>(replicates);
    threadPool.parallelFor(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
            // The adaptive stopping rule's own estimate, so the AOV shows exactly the error the sampler retires texels by.
            const pathtracer::scene::ReplicateEstimate estimate =
                pathtracer::scene::replicateEstimate(replicateLuminance + (pixel * stride), populated);
            const double standardError = std::sqrt(estimate.meanVariance);
            // No signal reads zero; a positive signal every replicate agreed on exactly is known exactly, so its SNR is +inf, not zero.
            const double ratio = estimate.mean > 0.0 ? estimate.mean / standardError : 0.0;
            writeTexel(out, x, y, static_cast<float>(ratio));
        }
    });
    return out;
}

HdrImage sampleCountAov(int width, int height, const int* texelSamples, ThreadPool& threadPool) {
    HdrImage out = makeAovImage(AovId::SampleCount, width, height);
    if (texelSamples == nullptr) {
        return out;
    }
    threadPool.parallelFor(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
            writeTexel(out, x, y, static_cast<float>(texelSamples[pixel]));
        }
    });
    return out;
}

HdrImage shadowAov(const HdrImage& lost, const HdrImage& unoccluded, ThreadPool& threadPool) {
    HdrImage out = makeAovImage(AovId::Shadow, lost.width, lost.height);
    threadPool.parallelFor(lost.height, [&](int y) {
        for (int x = 0; x < lost.width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(lost.width)) + static_cast<std::size_t>(x);
            // No direct light to lose means no shadow, so zero rather than an undefined 0/0.
            const float total = unoccluded.texels[pixel];
            writeTexel(out, x, y, total > 0.0F ? lost.texels[pixel] / total : 0.0F);
        }
    });
    return out;
}

HdrImage evaluateFilterAov(AovId aov, const FilterInput& input, ThreadPool& threadPool) {
    switch (aov) {
        case AovId::HSV:       return hsvAov(input.beauty, threadPool);
        case AovId::Luminance: return luminanceAov(input.beauty, threadPool);
        case AovId::SNR:
            return snrAov(input.beauty.width, input.beauty.height, input.replicateLuminance, input.replicates, input.samples, threadPool);
        case AovId::SampleCount:
            return sampleCountAov(input.beauty.width, input.beauty.height, input.texelSamples, threadPool);
        case AovId::Denoised:
            if (input.denoiseAlbedo == nullptr || input.denoiseNormal == nullptr || input.denoiser == nullptr) {
                throw std::invalid_argument("denoised: needs the denoiseAlbedo and denoiseNormal lanes and a Denoiser");
            }
            return input.denoiser->denoise(input.beauty, *input.denoiseAlbedo, *input.denoiseNormal);
        case AovId::Shadow:
            if (input.shadowLost == nullptr || input.shadowUnoccluded == nullptr) {
                throw std::invalid_argument("shadow: needs the shadowLost and shadowUnoccluded lanes");
            }
            return shadowAov(*input.shadowLost, *input.shadowUnoccluded, threadPool);

        // The lanes their own producers write. No default arm: -Werror then makes an unrouted new filter a compile error.
        case AovId::Beauty:
        case AovId::Wireframe:
        case AovId::Alpha:
        case AovId::Depth:
        case AovId::WorldPos:
        case AovId::UV:
        case AovId::MotionVector:
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
        case AovId::DirectVolume:
        case AovId::IndirectVolume:
        case AovId::DirectSubsurface:
        case AovId::IndirectSubsurface:
        case AovId::Count:
            break;
    }
    return {};
}

}  // namespace pathtracer::debug
