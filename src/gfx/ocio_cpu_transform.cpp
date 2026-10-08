#include "pathtracer/gfx/ocio_cpu_transform.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <OpenColorIO/OpenColorIO.h>
#include <glm/glm.hpp>

#include "pathtracer/gfx/ocio_display_transform.h"

namespace OCIO = OCIO_NAMESPACE;

namespace pathtracer::gfx {

namespace {

// RGBA float, OCIO's native packed layout, which its CPU processor transforms in place without a conversion pass.
constexpr std::size_t kScratchLanes = 4;

// Triangular-PDF dither, byte for byte the shader's ditherOffset() (ocio_display_transform.cpp), so output matches the viewer.
glm::vec3 ditherOffset(float u, float v) {
    const auto rand = [](float x, float y) {
        const float s = std::sin((x * 12.9898F) + (y * 78.233F)) * 43758.5453F;
        return s - std::floor(s);
    };
    const float d = (rand(u, v) - rand(u + 0.618F, v + 0.618F)) / 255.0F;
    return {d, d, d};
}

const OCIO::ConstConfigRcPtr& pinnedConfig() {
    static const OCIO::ConstConfigRcPtr config = OCIO::Config::CreateFromBuiltinConfig(kOcioConfigName);
    return config;
}

OCIO::ConstCPUProcessorRcPtr buildCpuProcessor(const char* display) {
    return pinnedConfig()->getProcessor(kOcioSceneColorSpace, display, kOcioView, OCIO::TRANSFORM_DIR_FORWARD)->getDefaultCPUProcessor();
}

// Each built on its first use (static init is thread-safe) and shared: parsing the config and compiling the ops costs milliseconds.
const OCIO::ConstCPUProcessorRcPtr& cpuProcessor(OcioDisplayTransform::Lut lut) {
    if (lut == OcioDisplayTransform::Lut::Rec709) {
        static const OCIO::ConstCPUProcessorRcPtr rec709 = buildCpuProcessor(kOcioRec709Display);
        return rec709;
    }
    static const OCIO::ConstCPUProcessorRcPtr srgb = buildCpuProcessor(kOcioSrgbDisplay);
    return srgb;
}

// Row y's affine map and transform into the RGBA scratch row, the stage both encodeForDisplay overloads share.
void encodeRow(std::span<const float> rgb, int width, int y, const glm::vec3& gain, bool applyDisplayTransform,
               const glm::vec3& displayOffset, std::vector<float>& row) {
    const auto rowTexels = static_cast<std::size_t>(width);
    const std::size_t rowStart = static_cast<std::size_t>(y) * rowTexels * 3;
    for (std::size_t x = 0; x < rowTexels; ++x) {
        for (glm::length_t lane = 0; lane < 3; ++lane) {
            row[(x * kScratchLanes) + static_cast<std::size_t>(lane)] =
                (rgb[rowStart + (x * 3) + static_cast<std::size_t>(lane)] * gain[lane]) + displayOffset[lane];
        }
    }
    if (applyDisplayTransform) {
        OCIO::PackedImageDesc desc(row.data(), width, 1, OCIO::CHANNEL_ORDERING_RGBA);
        cpuProcessor(OcioDisplayTransform::Lut::SRGB)->apply(desc);
    }
}

}  // namespace

void applyOcioDisplayTransform(std::span<float> rgb, int width, int height, OcioDisplayTransform::Lut lut) {
    if (lut == OcioDisplayTransform::Lut::Raw) {
        return;
    }
    OCIO::PackedImageDesc desc(rgb.data(), width, height, OCIO::CHANNEL_ORDERING_RGB);
    cpuProcessor(lut)->apply(desc);
}

void encodeForDisplay(std::span<const float> rgb, int width, int height, const glm::vec3& gain, bool applyDisplayTransform,
                      const glm::vec3& displayOffset, std::span<unsigned char> out) {
    const auto rowTexels = static_cast<std::size_t>(width);
    // One row of scratch rather than a full-frame copy: OCIO applies in place, the input is const, and a row stays in cache.
    std::vector<float> row(rowTexels * kScratchLanes, 1.0F);
    for (int y = 0; y < height; ++y) {
        const std::size_t rowStart = static_cast<std::size_t>(y) * rowTexels * 3;
        encodeRow(rgb, width, y, gain, applyDisplayTransform, displayOffset, row);
        for (std::size_t x = 0; x < rowTexels; ++x) {
            const glm::vec3 dither = ditherOffset((static_cast<float>(x) + 0.5F) / static_cast<float>(width),
                                                  (static_cast<float>(y) + 0.5F) / static_cast<float>(height));
            for (glm::length_t lane = 0; lane < 3; ++lane) {
                const float value = std::clamp(row[(x * kScratchLanes) + static_cast<std::size_t>(lane)] + dither[lane], 0.0F, 1.0F);
                out[rowStart + (x * 3) + static_cast<std::size_t>(lane)] = static_cast<unsigned char>(std::lround(value * 255.0F));
            }
        }
    }
}

void encodeForDisplay(std::span<const float> rgb, int width, int height, const glm::vec3& gain, bool applyDisplayTransform,
                      const glm::vec3& displayOffset, std::span<float> out) {
    const auto rowTexels = static_cast<std::size_t>(width);
    std::vector<float> row(rowTexels * kScratchLanes, 1.0F);
    for (int y = 0; y < height; ++y) {
        const std::size_t rowStart = static_cast<std::size_t>(y) * rowTexels * 3;
        encodeRow(rgb, width, y, gain, applyDisplayTransform, displayOffset, row);
        for (std::size_t x = 0; x < rowTexels; ++x) {
            for (std::size_t lane = 0; lane < 3; ++lane) {
                out[rowStart + (x * 3) + lane] = row[(x * kScratchLanes) + lane];
            }
        }
    }
}

std::array<float, 8> chromaticitiesOf(const char* colorSpace) {
    // Unit RGB vectors are the primaries and (1, 1, 1) the white: a transfer function fixes 0 and 1, so encoded spaces hold too.
    const OCIO::ConstCPUProcessorRcPtr toXyz =
        pinnedConfig()->getProcessor(colorSpace, OCIO::ROLE_INTERCHANGE_DISPLAY)->getDefaultCPUProcessor();
    constexpr std::array<std::array<float, 3>, 4> kCorners{{{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, {1.0F, 1.0F, 1.0F}}};
    std::array<float, 8> xy{};
    for (std::size_t corner = 0; corner < kCorners.size(); ++corner) {
        std::array<float, 3> xyz = kCorners[corner];
        toXyz->applyRGB(xyz.data());
        const float sum = xyz[0] + xyz[1] + xyz[2];
        xy[2 * corner] = xyz[0] / sum;
        xy[(2 * corner) + 1] = xyz[1] / sum;
    }
    return xy;
}

}  // namespace pathtracer::gfx
