#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/gfx/scalar_type.h"

namespace pathtracer::gfx {

// CPU-side decode of a linear scanline EXR, shared by Texture's GPU upload and any CPU consumer wanting the same pixels.
struct HdrImage {
    int width = 0;
    int height = 0;
    std::vector<float> rgba;  // row-major, 4 floats/texel, linear light
};

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

// Linear EXR read as RGBA float; nullopt on I/O failure or absent R/G/B, absent alpha reads 1.0, all texels finite.
[[nodiscard]] std::optional<HdrImage> loadExr(const std::string& path);

// Writes a linear scanline EXR, loadExr's inverse with the same full-float channels, so a round trip is lossless.
[[nodiscard]] bool writeExr(const std::string& path, const HdrImage& image);

// How the v axis resolves outside [0,1). u always repeats; ClampV is for an equirect map, whose top and bottom rows are poles.
enum class WrapMode { Repeat, ClampV };

// Bilinear sample at uv (GL_REPEAT equivalent by default), filtered in float after widening each texel.
[[nodiscard]] glm::vec3 sampleBilinear(const ImageTexture& image, glm::vec2 uv,
                                        WrapMode wrap = WrapMode::Repeat);

}  // namespace pathtracer::gfx
