#include "pathtracer/gfx/ocio_cpu_transform.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

#include <OpenColorIO/OpenColorIO.h>
#include <glm/glm.hpp>

#include "pathtracer/gfx/ocio_display_transform.h"

namespace OCIO = OCIO_NAMESPACE;

namespace pathtracer::gfx {

namespace {

// Triangular-PDF dither, byte for byte the shader's ditherOffset() (ocio_display_transform.cpp), so output matches the viewer.
glm::vec3 ditherOffset(float u, float v) {
    const auto rand = [](float x, float y) {
        const float s = std::sin((x * 12.9898F) + (y * 78.233F)) * 43758.5453F;
        return s - std::floor(s);
    };
    const float d = (rand(u, v) - rand(u + 0.618F, v + 0.618F)) / 255.0F;
    return {d, d, d};
}

}  // namespace

void applyOcioDisplayTransform(std::vector<float>& rgb, int width, int height) {
    const OCIO::ConstConfigRcPtr config = OCIO::Config::CreateFromBuiltinConfig(kOcioConfigName);
    const OCIO::ConstProcessorRcPtr processor =
        config->getProcessor(kOcioSceneColorSpace, kOcioSrgbDisplay, kOcioView, OCIO::TRANSFORM_DIR_FORWARD);
    const OCIO::PackedImageDesc desc(rgb.data(), width, height, OCIO::CHANNEL_ORDERING_RGB);
    processor->getDefaultCPUProcessor()->apply(desc);
}

std::vector<unsigned char> encodeForDisplay(std::span<const float> rgb, int width, int height,
                                             const glm::vec3& gain, bool applyDisplayTransform,
                                             const glm::vec3& displayOffset) {
    // The one copy the encode needs: OCIO applies in place and the caller's buffer is const.
    std::vector<float> exposed(rgb.size());
    // Packed RGB triples, walked a texel at a time so the lane is the inner index rather than a division on every element.
    for (std::size_t texel = 0; texel < rgb.size(); texel += 3) {
        for (glm::length_t lane = 0; lane < 3; ++lane) {
            const std::size_t i = texel + static_cast<std::size_t>(lane);
            exposed[i] = (rgb[i] * gain[lane]) + displayOffset[lane];
        }
    }

    if (applyDisplayTransform) {
        applyOcioDisplayTransform(exposed, width, height);
    }

    std::vector<unsigned char> out(exposed.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t i = ((static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) +
                                    static_cast<std::size_t>(x)) * 3;
            const glm::vec3 dither = ditherOffset((static_cast<float>(x) + 0.5F) / static_cast<float>(width),
                                                   (static_cast<float>(y) + 0.5F) / static_cast<float>(height));
            for (int c = 0; c < 3; ++c) {
                const float value = std::clamp(exposed[i + static_cast<std::size_t>(c)] + dither[c], 0.0F, 1.0F);
                out[i + static_cast<std::size_t>(c)] = static_cast<unsigned char>(std::lround(value * 255.0F));
            }
        }
    }
    return out;
}

}  // namespace pathtracer::gfx
