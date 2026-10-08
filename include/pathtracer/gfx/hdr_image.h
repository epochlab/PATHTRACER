#pragma once

#include <cassert>
#include <cstddef>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/gfx/scalar_type.h"

namespace pathtracer::gfx {

// Linear float image of 1-4 interleaved channels, row-major, row 0 top: every AOV lane at its aovChannels count, and image I/O.
struct HdrImage {
    int width = 0;
    int height = 0;
    int channels = 0;  // 0 only for an empty image, as width and height are
    std::vector<float> texels;

    // RGB of `pixel` as Texture's swizzle displays it: one channel broadcasts, a second leaves blue 0.
    [[nodiscard]] glm::vec3 rgb(std::size_t pixel) const;
};

// Zeroed width x height image of `channels` interleaved floats.
[[nodiscard]] HdrImage makeImage(int width, int height, int channels);

// Stores value's L components at (x, y): L is the stride, so it must equal the channels the image was allocated at.
template <glm::length_t L>
void writeTexel(HdrImage& image, int x, int y, const glm::vec<L, float>& value) {
    assert(image.channels == L);
    const std::size_t pixel =
        (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) + static_cast<std::size_t>(x);
    float* texel = image.texels.data() + (pixel * L);
    for (glm::length_t c = 0; c < L; ++c) {
        texel[c] = value[c];
    }
}

inline void writeTexel(HdrImage& image, int x, int y, float value) {
    writeTexel(image, x, y, glm::vec1(value));
}

// ImageTexture channel counts: what its reader consumes, never alpha. R for a scalar map (roughness, bump), RGB otherwise.
inline constexpr int kScalarChannels = 1;
inline constexpr int kRgbChannels = 3;

// A scene input image (environment HDRI, material texture) at a chosen ScalarType: `channels` interleaved, row-major, row 0 top.
struct ImageTexture {
    int width = 0;
    int height = 0;
    int channels = 0;  // kScalarChannels or kRgbChannels; 0 only for an empty image, as width and height are
    std::variant<std::vector<float>, std::vector<Half>> texels;

    // Texel (x, y), widened to float; x in [0, width), y in [0, height). Channels it lacks read 0, as GL_RED swizzles.
    [[nodiscard]] glm::vec3 texel(int x, int y) const;
};

// The first `channels` of R,G,B decoded straight into type's storage; nullopt if one is absent. Float16 rejects over-range as Inf.
[[nodiscard]] std::optional<ImageTexture> loadImageTexture(const std::string& path, ScalarType type, int channels);

// Colour: RGB in a colour space, converted to and from the working space. Data: normals, depths, heights, read and written raw.
enum class ImageRole { Colour, Data };

// A linear EXR's leading R, RG or RGB as float in the working space, Colour converted by OCIO; nullopt, logged, unless all finite.
[[nodiscard]] std::optional<HdrImage> loadImage(const std::string& path, ImageRole role);

// Full-float EXR of the image's first `channels` of R,G,B, lossless through loadImage; Colour tags working space and chromaticities.
[[nodiscard]] bool writeExr(const std::string& path, const HdrImage& image, ImageRole role);

// Display-referred RGB or RGBA (straight alpha), clamped to [0, 1], as a 16-bit PNG tagged with the display colour space it encodes.
[[nodiscard]] bool writeDisplayPng(const std::string& path, const HdrImage& image);

// How the v axis resolves outside [0,1). u always repeats; ClampV is for an equirect map, whose top and bottom rows are poles.
enum class WrapMode { Repeat, ClampV };

// Bilinear sample at uv (GL_REPEAT equivalent by default), filtered in float after widening each texel.
[[nodiscard]] glm::vec3 sampleBilinear(const ImageTexture& image, glm::vec2 uv,
                                        WrapMode wrap = WrapMode::Repeat);

}  // namespace pathtracer::gfx
