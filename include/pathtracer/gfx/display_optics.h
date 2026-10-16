#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <numeric>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/pixel_lattice.h"
#include "pathtracer/scene/thread_pool.h"

// The viewer's lens optics on the CPU, from the same kernels as BloomPass's shaders: one definition, so a comparison cannot drift.
namespace pathtracer::gfx {

// One GL_LINEAR fetch of a bloom pass: (dx, dy) source texels from the target texel's centre, and its weight.
struct BloomTap {
    int dx;
    int dy;
    float weight;
};

// Jimenez 2014's 13 taps: five bilinear 2x2 boxes, the centre box weighted 1/2 and the four corner boxes 1/8, so weights sum to 1.
inline constexpr std::array<BloomTap, 13> kBloomDownTaps{{{-1, 1, 0.125F},
                                                          {1, 1, 0.125F},
                                                          {-1, -1, 0.125F},
                                                          {1, -1, 0.125F},
                                                          {0, 0, 0.125F},
                                                          {0, 2, 0.0625F},
                                                          {-2, 0, 0.0625F},
                                                          {2, 0, 0.0625F},
                                                          {0, -2, 0.0625F},
                                                          {-2, 2, 0.03125F},
                                                          {2, 2, 0.03125F},
                                                          {-2, -2, 0.03125F},
                                                          {2, -2, 0.03125F}}};

// 3x3 tent (1 2 1)/4 squared at one coarse texel; each up pass adds the down octave of its own size, so every octave enters once.
inline constexpr std::array<BloomTap, 9> kBloomUpTaps{{{0, 0, 0.25F},
                                                       {0, 1, 0.125F},
                                                       {-1, 0, 0.125F},
                                                       {1, 0, 0.125F},
                                                       {0, -1, 0.125F},
                                                       {-1, 1, 0.0625F},
                                                       {1, 1, 0.0625F},
                                                       {-1, -1, 0.0625F},
                                                       {1, -1, 0.0625F}}};

static_assert(std::accumulate(kBloomDownTaps.begin(), kBloomDownTaps.end(), 0.0F,
                              [](float sum, const BloomTap& tap) { return sum + tap.weight; }) == 1.0F,
              "the down kernel must be normalised, or bloom adds or removes energy");
static_assert(std::accumulate(kBloomUpTaps.begin(), kBloomUpTaps.end(), 0.0F,
                              [](float sum, const BloomTap& tap) { return sum + tap.weight; }) == 1.0F,
              "the up kernel must be normalised, or bloom adds or removes energy");

// L = floor(log2(min(w, h))) octaves, the last with a 1-texel short side: the PSF's 1/r^2 reaches the frame, then stops.
[[nodiscard]] constexpr int bloomOctaves(int width, int height) {
    return std::max(1, std::bit_width(static_cast<unsigned>(std::min(width, height))) - 1);
}

// 0 = off for each. aberration in [0, 1): R and B sampled at film scaled 1 -/+ aberration about the frame centre. bloomStrength in [0, 1].
struct DisplayOptics {
    float aberration = 0.0F;
    float bloomStrength = 0.0F;
};

// In place on 3 channels, the viewer's 1:1 frame: (1 - k) A(I) + k (I * PSF), A the nearest-cell fringe. Bloom needs a square lattice.
void applyDisplayOptics(HdrImage& image, PixelLattice lattice, bool wrapsHorizontally, const DisplayOptics& optics,
                        pathtracer::scene::ThreadPool& threadPool);

}  // namespace pathtracer::gfx
