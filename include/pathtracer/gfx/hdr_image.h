#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

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

// Texture channel counts: what a lookup consumes, never alpha. R for a scalar map (roughness, bump), RGB otherwise.
inline constexpr int kScalarChannels = 1;
inline constexpr int kRgbChannels = 3;

// Colour: RGB in a colour space, converted to and from the working space. Data: normals, depths, heights, read and written raw.
enum class ImageRole { Colour, Data };

// A linear EXR's leading R, RG or RGB as finite float in the working space; colorSpace overrides a mis-tagged file's own tag.
[[nodiscard]] std::optional<HdrImage> loadImage(const std::string& path, ImageRole role,
                                                const std::optional<std::string>& colorSpace = std::nullopt);

// Full-float EXR of the image's first `channels` of R,G,B, lossless through loadImage; Colour tags working space and chromaticities.
[[nodiscard]] bool writeExr(const std::string& path, const HdrImage& image, ImageRole role);

// Display-referred RGB or RGBA (straight alpha), clamped to [0, 1], as a 16-bit PNG tagged with the display colour space it encodes.
[[nodiscard]] bool writeDisplayPng(const std::string& path, const HdrImage& image);

// How st resolves outside [0, 1): Repeat is GL_REPEAT on both; LatLong repeats s and clamps t, whose first and last rows are poles.
enum class TextureWrap { Repeat, LatLong };

// d(st)/d(pixel x) and d(st)/d(pixel y): the lookup's filter footprint. Zero is a point lookup at the finest MIP level.
struct TextureFootprint {
    glm::vec2 dx{0.0F};
    glm::vec2 dy{0.0F};
};

// A filtered scalar lookup with its derivatives per unit s and t, the rates a bump map's surface gradient is built from.
struct TextureGradient {
    float value;
    glm::vec2 dst;
};

// A scene image served by OIIO's TextureSystem: cached at its stored format, MIP-mapped and filtered. Defined where OIIO is.
struct ImageTexture;

// A linear EXR's `channels` from R, G, B starting channelOffset in; nullptr, logged, on failure. Colour converts to the working space once.
[[nodiscard]] std::shared_ptr<const ImageTexture> openTexture(const std::string& path, int channels, ImageRole role, TextureWrap wrap,
                                                             const std::optional<std::string>& colorSpace = std::nullopt,
                                                             int channelOffset = 0);

// A generated working-space image as a texture: what a procedural source or a fixture has in place of a file.
[[nodiscard]] std::shared_ptr<const ImageTexture> makeTexture(const HdrImage& image, TextureWrap wrap);

// The texture's finest level as float, every texel exactly as lookups filter it; nullopt, logged, if its file cannot be read.
[[nodiscard]] std::optional<HdrImage> readTexels(const ImageTexture& texture);

// Bilinear within a MIP level, anisotropic across levels by footprint: interpolating, so a point lookup at a texel centre is the texel.
[[nodiscard]] glm::vec3 sampleTexture(const ImageTexture& texture, glm::vec2 st, const TextureFootprint& footprint = {});

// Channel 0 by smart-bicubic B-spline with its analytic d/ds, d/dt: C1, so a bump's slope does not facet at texel boundaries.
[[nodiscard]] TextureGradient sampleTextureGradient(const ImageTexture& texture, glm::vec2 st, const TextureFootprint& footprint = {});

}  // namespace pathtracer::gfx
