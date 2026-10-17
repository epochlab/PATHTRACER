#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "pathtracer/gfx/pixel_lattice.h"
#include "pathtracer/gfx/scalar_type.h"
#include "pathtracer/gfx/shader_program.h"

namespace pathtracer::gfx {

// Glare PSF as an equal-weight pyramid, sigma doubling per octave so P ~ 1/r^2 (Stiles-Holladay): Jimenez 2014 square, Loop hexagonal.
class BloomPass {
public:
    // Compiles both lattices' down and up programs; nullopt on a GLSL compile or link failure, per ShaderProgram's contract.
    [[nodiscard]] static std::optional<BloomPass> create();
    ~BloomPass();

    BloomPass(const BloomPass&) = delete;
    BloomPass& operator=(const BloomPass&) = delete;
    BloomPass(BloomPass&& other) noexcept;
    BloomPass& operator=(BloomPass&& other) noexcept;

    // Convolves sourceTexture (width x height) with the lattice's PSF into the full-resolution result, reallocating on a shape change.
    void build(unsigned int sourceTexture, int width, int height, ScalarType format, bool wrapsHorizontally, PixelLattice lattice);

    // Binds the last build's full-resolution result; always a complete texture, so valid to bind with bloom off.
    void bind(unsigned int unit) const;

private:
    // One lattice's down and up programs and their uniforms; a location the program lacks is -1, which glUniform ignores.
    struct Pyramid {
        ShaderProgram down;
        ShaderProgram up;
        int downSourceTexelLoc;
        int downTargetTexelLoc;
        int downWrapsLoc;
        int upSourceTexelLoc;
        int upTargetTexelLoc;
        int upWrapsLoc;
        int upScaleLoc;
        int upHasBaseLoc;
    };

    BloomPass(Pyramid square, Pyramid hexagonal);

    struct Level {
        unsigned int texture;
        unsigned int fbo;  // texture attached at allocation; 0 for the source, which is only ever read
        int width;
        int height;
    };

    void allocate(int width, int height, ScalarType format, PixelLattice lattice);
    void releaseLevels();
    void destroy();
    void setWrapsHorizontally(bool wrapsHorizontally);
    void draw(const Level& source, const Level& target, int sourceTexelLoc, int targetTexelLoc) const;

    Pyramid square_;
    Pyramid hexagonal_;
    unsigned int vao_ = 0;
    std::vector<Level> down_;  // octaves 1..L, each half the last; down_[0] is half resolution
    std::vector<Level> up_;    // octaves 0..L-1, each the down octave plus the upsampled coarser one; up_[0] is full resolution
    ScalarType format_ = ScalarType::Float16;
    PixelLattice lattice_ = PixelLattice::Square;  // the levels' geometry, so build reallocates when the lattice changes
    std::size_t byteSize_ = 0;  // every level of both chains, reported to pathtracer::debug's GPU memory tracker
    bool wrapsHorizontally_ = false;  // GL_TEXTURE_WRAP_S of every level, so build touches GL only on a change
};

}  // namespace pathtracer::gfx
