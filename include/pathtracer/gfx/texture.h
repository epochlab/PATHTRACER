#pragma once

#include <cstddef>

#include "pathtracer/gfx/scalar_type.h"

namespace pathtracer::gfx {

// Owns one GL_TEXTURE_2D object, move-only.
class Texture {
public:
    ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    // Uploads width*height texels of `channels` (1-3) floats, row-major and unpadded; binary16 storage overflows above 65504.
    static Texture createFromFloatPixels(int width, int height, int channels, const float* texels, ScalarType format);

    // Replaces the texels through the pixel buffer, reallocating only on a size or channel change. Sampled as HdrImage::rgb expands them.
    void upload(int width, int height, int channels, const float* texels);

    void bind(unsigned int unit) const;

    // GL_REPEAT in s for an image whose left and right edges are one meridian, else GL_CLAMP_TO_EDGE; t always clamps.
    void setWrapsHorizontally(bool wrapsHorizontally);

    // Raw GL texture id, for callers needing it directly (PostProcessPass::draw) rather than through bind().
    [[nodiscard]] unsigned int id() const { return id_; }

private:
    Texture(unsigned int id, unsigned int pixelBuffer, ScalarType format);

    unsigned int id_ = 0;
    // GL_PIXEL_UNPACK_BUFFER the texels are packed straight into, so the transfer is the driver's DMA, not a client-memory copy.
    unsigned int pixelBuffer_ = 0;
    ScalarType format_ = ScalarType::Float16;  // component type every (re)allocation in upload uses
    int width_ = 0;   // current storage shape, so upload can tell a reallocation from an in-place update
    int height_ = 0;
    int channels_ = 0;
    std::size_t byteSize_ = 0;  // the texture's storage, reported to pathtracer::debug's GPU memory tracker
    std::size_t pixelBufferBytes_ = 0;  // the pixel buffer's storage, reported likewise
    bool wrapsHorizontally_ = false;  // GL_TEXTURE_WRAP_S is GL_REPEAT, so setWrapsHorizontally touches GL only on a change
};

}  // namespace pathtracer::gfx
