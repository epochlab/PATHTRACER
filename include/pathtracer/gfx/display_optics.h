#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <numeric>
#include <vector>

#include <glm/glm.hpp>

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

// One entry of the hexagonal pyramid's mask: axial offset (dq, dr) in fine cells from a coarse vertex, and its weight.
struct HexBloomTap {
    int dq;
    int dr;
    float weight;
};

// Loop's regular mask (Loop 1987), the three-direction box spline's: vertex 10/16, then rings at 1, sqrt3 and 2 pitches 6, 2, 1 /16.
inline constexpr std::array<HexBloomTap, 19> kHexBloomMask{{{0, 0, 0.625F},   {1, 0, 0.375F},   {-1, 0, 0.375F},  {0, 1, 0.375F},
                                                            {0, -1, 0.375F},  {1, -1, 0.375F},  {-1, 1, 0.375F},  {1, 1, 0.125F},
                                                            {-1, -1, 0.125F}, {2, -1, 0.125F},  {-2, 1, 0.125F},  {1, -2, 0.125F},
                                                            {-1, 2, 0.125F},  {2, 0, 0.0625F},  {-2, 0, 0.0625F}, {0, 2, 0.0625F},
                                                            {0, -2, 0.0625F}, {2, -2, 0.0625F}, {-2, 2, 0.0625F}}};

// Mask weight landing on fine cells of axial parity (q, r): the coset a coarse vertex (0, 0) or an edge midpoint feeds.
[[nodiscard]] constexpr float hexBloomCosetWeight(int q, int r) {
    float sum = 0.0F;
    for (const HexBloomTap& tap : kHexBloomMask) {
        sum += (tap.dq - q) % 2 == 0 && (tap.dr - r) % 2 == 0 ? tap.weight : 0.0F;
    }
    return sum;
}

static_assert(hexBloomCosetWeight(0, 0) == 1.0F && hexBloomCosetWeight(1, 0) == 1.0F && hexBloomCosetWeight(0, 1) == 1.0F &&
                  hexBloomCosetWeight(1, 1) == 1.0F,
              "each fine coset must take unit weight, or the up pass adds or removes energy");
static_assert(std::accumulate(kHexBloomMask.begin(), kHexBloomMask.end(), 0.0F,
                              [](float sum, const HexBloomTap& tap) { return sum + tap.weight; }) == 4.0F,
              "the aperture-4 mask must sum to 4, so its adjoint over 4 is a normalised down kernel");

// The mask entries reaching a fine cell of axial parity (q & 1, r & 1), in mask order: 7 at a coarse vertex, 4 at an edge midpoint.
[[nodiscard]] std::vector<HexBloomTap> hexBloomCoset(int qParity, int rParity);

// Octave `octave`'s level: square halves and floors. Hexagonal coarse (j, m) is fine (2j + (m & 1), 2m), anchored at the top-left.
[[nodiscard]] constexpr glm::ivec2 bloomLevelSize(PixelLattice lattice, int width, int height, int octave, bool wrapsHorizontally) {
    if (lattice != PixelLattice::Hexagonal) {
        return {std::max(1, width >> octave), std::max(1, height >> octave)};
    }
    for (int level = 0; level < octave; ++level) {
        // floor(n/2) + 1 is the least level whose far mirror lines lie on or past its finer level's, so every octave covers the frame.
        height = (height / 2) + 1;
        // A wrapped width is one period, which a level must halve exactly to stay periodic.
        width = wrapsHorizontally ? (width + 1) / 2 : (width / 2) + 1;
    }
    return {width, height};
}

// 0 = off for each. aberration in [0, 1): R and B sampled at film scaled 1 -/+ aberration about the frame centre. bloomStrength in [0, 1].
struct DisplayOptics {
    float aberration = 0.0F;
    float bloomStrength = 0.0F;
};

// In place on 3 channels, the viewer's 1:1 frame: (1 - k) A(I) + k (I * PSF), A the nearest-cell fringe, the PSF the lattice's pyramid.
void applyDisplayOptics(HdrImage& image, PixelLattice lattice, bool wrapsHorizontally, const DisplayOptics& optics,
                        pathtracer::scene::ThreadPool& threadPool);

}  // namespace pathtracer::gfx
