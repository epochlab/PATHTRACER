#pragma once

#include <cstddef>
#include <vector>

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

    // Replaces the texels, reallocating only on a size or channel change. Sampled as HdrImage::rgb expands them, alpha 1.
    void upload(int width, int height, int channels, const float* texels);

    void bind(unsigned int unit) const;

    // Raw GL texture id, for callers needing it directly (PostProcessPass::draw) rather than through bind().
    [[nodiscard]] unsigned int id() const { return id_; }

private:
    Texture(unsigned int id, ScalarType format);

    unsigned int id_ = 0;
    ScalarType format_ = ScalarType::Float16;  // component type every (re)allocation in upload uses
    int width_ = 0;   // current storage shape, so upload can tell a reallocation from an in-place update
    int height_ = 0;
    int channels_ = 0;
    // Upload staging in the texture's own component type and lane count, reused across uploads; only format_'s one is ever filled.
    std::vector<Half> halfStaging_;
    std::vector<float> floatStaging_;
    std::size_t byteSize_ = 0;  // reported to pathtracer::debug's GPU memory tracker
};

}  // namespace pathtracer::gfx
