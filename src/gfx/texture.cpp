#include "pathtracer/gfx/texture.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <utility>

#include <GL/glew.h>

#include "pathtracer/debug/memory_tracker.h"
#include "pathtracer/gfx/gl_debug.h"

namespace pathtracer::gfx {

namespace {

// Lanes stored for `channels` (1-3): Metal, Vulkan and D3D have no sampled 3-lane half or float format, so RGB is stored as RGBA.
constexpr std::array<int, 3> kStoredLanes{1, 2, 4};

int storedLanes(int channels) {
    return kStoredLanes[static_cast<std::size_t>(channels - 1)];
}

// Sized float internal format of storedLanes(channels) lanes of each component type.
GLint glInternalFormat(int channels, ScalarType format) {
    constexpr std::array<GLint, 3> kHalf{GL_R16F, GL_RG16F, GL_RGBA16F};
    constexpr std::array<GLint, 3> kFloat{GL_R32F, GL_RG32F, GL_RGBA32F};
    const auto index = static_cast<std::size_t>(channels - 1);
    switch (format) {
        case ScalarType::Float16:
            return kHalf[index];
        case ScalarType::Float32:
            return kFloat[index];
    }
    return 0;
}

GLenum glPixelFormat(int channels) {
    constexpr std::array<GLenum, 3> kFormats{GL_RED, GL_RG, GL_RGBA};
    return kFormats[static_cast<std::size_t>(channels - 1)];
}

GLenum glComponentType(ScalarType format) {
    return format == ScalarType::Float16 ? GL_HALF_FLOAT : GL_FLOAT;
}

// Converts `channels`-wide float texels to `Lanes`-wide T, padding with 1; compile-time widths so the loop vectorises.
template <typename T, int Channels, int Lanes>
void packTexels(const float* texels, std::size_t pixels, T* out) {
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        for (int c = 0; c < Channels; ++c) {
            out[(pixel * Lanes) + c] = static_cast<T>(texels[(pixel * Channels) + c]);
        }
        for (int c = Channels; c < Lanes; ++c) {
            out[(pixel * Lanes) + c] = T(1);
        }
    }
}

// Writes the texels to `out` in the texture's own type and lane count, so the driver transfers them rather than converting.
template <typename T>
void stage(const float* texels, std::size_t pixels, int channels, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        if (storedLanes(channels) == channels) {
            std::memcpy(out, texels, pixels * static_cast<std::size_t>(channels) * sizeof(float));
            return;
        }
    }
    switch (channels) {
        case 1:
            packTexels<T, 1, 1>(texels, pixels, out);
            break;
        case 2:
            packTexels<T, 2, 2>(texels, pixels, out);
            break;
        default:
            packTexels<T, 3, 4>(texels, pixels, out);
            break;
    }
}

// HdrImage::rgb's expansion done by the sampler: R broadcasts for a scalar, an absent blue reads 0, alpha reads 1.
std::array<GLint, 4> glSwizzle(int channels) {
    constexpr std::array<std::array<GLint, 4>, 3> kSwizzles{{{GL_RED, GL_RED, GL_RED, GL_ONE},
                                                              {GL_RED, GL_GREEN, GL_ZERO, GL_ONE},
                                                              {GL_RED, GL_GREEN, GL_BLUE, GL_ONE}}};
    return kSwizzles[static_cast<std::size_t>(channels - 1)];
}

}  // namespace

Texture::Texture(unsigned int id, unsigned int pixelBuffer, ScalarType format) : id_(id), pixelBuffer_(pixelBuffer), format_(format) {}

Texture::~Texture() {
    if (id_ != 0) {
        pathtracer::debug::trackGpuFree(byteSize_ + pixelBufferBytes_);
        glDeleteTextures(1, &id_);
        glDeleteBuffers(1, &pixelBuffer_);
    }
}

Texture::Texture(Texture&& other) noexcept
    : id_(std::exchange(other.id_, 0)),
      pixelBuffer_(std::exchange(other.pixelBuffer_, 0)),
      format_(other.format_),
      width_(std::exchange(other.width_, 0)),
      height_(std::exchange(other.height_, 0)),
      channels_(std::exchange(other.channels_, 0)),
      byteSize_(std::exchange(other.byteSize_, 0)),
      pixelBufferBytes_(std::exchange(other.pixelBufferBytes_, 0)),
      wrapsHorizontally_(std::exchange(other.wrapsHorizontally_, false)) {}

Texture& Texture::operator=(Texture&& other) noexcept {
    if (this != &other) {
        if (id_ != 0) {
            pathtracer::debug::trackGpuFree(byteSize_ + pixelBufferBytes_);
            glDeleteTextures(1, &id_);
            glDeleteBuffers(1, &pixelBuffer_);
        }
        id_ = std::exchange(other.id_, 0);
        pixelBuffer_ = std::exchange(other.pixelBuffer_, 0);
        format_ = other.format_;
        width_ = std::exchange(other.width_, 0);
        height_ = std::exchange(other.height_, 0);
        channels_ = std::exchange(other.channels_, 0);
        byteSize_ = std::exchange(other.byteSize_, 0);
        pixelBufferBytes_ = std::exchange(other.pixelBufferBytes_, 0);
        wrapsHorizontally_ = std::exchange(other.wrapsHorizontally_, false);
    }
    return *this;
}

Texture Texture::createFromFloatPixels(int width, int height, int channels, const float* texels, ScalarType format) {
    unsigned int id = 0;
    unsigned int pixelBuffer = 0;
    GL_CALL(glGenTextures(1, &id));
    GL_CALL(glGenBuffers(1, &pixelBuffer));
    GL_CALL(glBindTexture(GL_TEXTURE_2D, id));

    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    // GL_LINEAR rather than GL_LINEAR_MIPMAP_LINEAR, since there is no longer a mip chain to select from.
    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    // Magnification shows traced pixels as blocks rather than interpolating values never traced, so the probe matches what is on screen.
    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

    Texture texture(id, pixelBuffer, format);
    texture.upload(width, height, channels, texels);
    GL_CALL(glBindTexture(GL_TEXTURE_2D, 0));
    return texture;
}

// Not wrapped in GL_CALL on the in-place path: it runs every frame the image changes and glGetError is a driver sync point.
void Texture::upload(int width, int height, int channels, const float* texels) {
    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::size_t bytes = pixels * static_cast<std::size_t>(storedLanes(channels)) * scalarBytes(format_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer_);
    if (bytes != pixelBufferBytes_) {
        GL_CALL(glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, GL_STREAM_DRAW));
        pathtracer::debug::trackGpuFree(pixelBufferBytes_);
        pixelBufferBytes_ = bytes;
        pathtracer::debug::trackGpuAlloc(pixelBufferBytes_);
    }
    // Invalidation orphans the store a previous transfer may still read (Hrabcak & Masserann 2012), so the map never waits on it.
    void* mapped = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, static_cast<GLsizeiptr>(bytes),
                                    GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if (mapped == nullptr) {
        std::cerr << "Texture::upload: mapping the " << bytes << "-byte pixel buffer failed; the texture keeps its previous texels\n";
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        return;
    }
    if (format_ == ScalarType::Float16) {
        stage(texels, pixels, channels, static_cast<Half*>(mapped));
    } else {
        stage(texels, pixels, channels, static_cast<float*>(mapped));
    }
    // GL_FALSE: the store was corrupted while mapped (e.g. a display change), so its contents are undefined and must not be uploaded.
    if (glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER) == GL_FALSE) {
        std::cerr << "Texture::upload: the pixel buffer was lost while mapped; the texture keeps its previous texels\n";
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        return;
    }
    // Rows are tightly packed, and an R16F row of odd width is not a multiple of the default 4-byte alignment.
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (width == width_ && height == height_ && channels == channels_) {
        glBindTexture(GL_TEXTURE_2D, id_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, glPixelFormat(channels), glComponentType(format_), nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        return;
    }
    GL_CALL(glBindTexture(GL_TEXTURE_2D, id_));
    GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, glInternalFormat(channels, format_), width, height, 0, glPixelFormat(channels),
                         glComponentType(format_), nullptr));
    GL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    GL_CALL(glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, glSwizzle(channels).data()));
    width_ = width;
    height_ = height;
    channels_ = channels;
    pathtracer::debug::trackGpuFree(byteSize_);
    // No mip chain, so no ~1/3 addition: the lanes actually stored, RGB counted at the four it is padded to.
    byteSize_ = pixels * static_cast<std::size_t>(storedLanes(channels)) * scalarBytes(format_);
    pathtracer::debug::trackGpuAlloc(byteSize_);
}

// Not wrapped in GL_CALL: runs every frame.
void Texture::bind(unsigned int unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, id_);
}

void Texture::setWrapsHorizontally(bool wrapsHorizontally) {
    // Called on every new pass, so the GL calls and their glGetError syncs run only on the rare change of lens.
    if (wrapsHorizontally == wrapsHorizontally_) {
        return;
    }
    wrapsHorizontally_ = wrapsHorizontally;
    GL_CALL(glBindTexture(GL_TEXTURE_2D, id_));
    GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapsHorizontally ? GL_REPEAT : GL_CLAMP_TO_EDGE));
    GL_CALL(glBindTexture(GL_TEXTURE_2D, 0));
}

}  // namespace pathtracer::gfx
