#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "pathtracer/gfx/scalar_type.h"
#include "pathtracer/gfx/shader_program.h"

namespace pathtracer::gfx {

// Glare PSF as an equal-weight Gaussian pyramid (Jimenez 2014): sigma doubles per octave, so P ~ 1/r^2 (Stiles-Holladay).
class BloomPass {
public:
    // Compiles the down and up programs; nullopt on a GLSL compile or link failure, per ShaderProgram's contract.
    [[nodiscard]] static std::optional<BloomPass> create();
    ~BloomPass();

    BloomPass(const BloomPass&) = delete;
    BloomPass& operator=(const BloomPass&) = delete;
    BloomPass(BloomPass&& other) noexcept;
    BloomPass& operator=(BloomPass&& other) noexcept;

    // Convolves sourceTexture (width x height) with the PSF into the full-resolution result, reallocating only on a shape change.
    void build(unsigned int sourceTexture, int width, int height, ScalarType format, bool wrapsHorizontally);

    // Binds the last build's full-resolution result; always a complete texture, so valid to bind with bloom off.
    void bind(unsigned int unit) const;

private:
    BloomPass(ShaderProgram downShader, ShaderProgram upShader);

    struct Level {
        unsigned int texture;
        unsigned int fbo;  // texture attached at allocation; 0 for the source, which is only ever read
        int width;
        int height;
    };

    void allocate(int width, int height, ScalarType format);
    void releaseLevels();
    void destroy();
    void setWrapsHorizontally(bool wrapsHorizontally);
    void draw(const Level& source, const Level& target, int sourceTexelLoc, int targetTexelLoc) const;

    ShaderProgram downShader_;
    ShaderProgram upShader_;
    int downSourceTexelLoc_;
    int downTargetTexelLoc_;
    int upSourceTexelLoc_;
    int upTargetTexelLoc_;
    int upScaleLoc_;
    int upHasBaseLoc_;
    unsigned int vao_ = 0;
    std::vector<Level> down_;  // octaves 1..L, each half the last; down_[0] is half resolution
    std::vector<Level> up_;    // octaves 0..L-1, each the down octave plus the tent-upsampled coarser one; up_[0] is full resolution
    ScalarType format_ = ScalarType::Float16;
    std::size_t byteSize_ = 0;  // every level of both chains, reported to pathtracer::debug's GPU memory tracker
    bool wrapsHorizontally_ = false;  // GL_TEXTURE_WRAP_S of every level, so build touches GL only on a change
};

}  // namespace pathtracer::gfx
