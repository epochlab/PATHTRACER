#pragma once

#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/gfx/pixel_lattice.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// Lindeberg's discrete scale space over one float plane: kernel T(n;t) = exp(-t) I_n(t), the shared Gaussian facility.

// Unit roundoff of binary32, the one threshold this file is built on: a contribution below it cannot perturb a float result.
inline constexpr double kFloat32Roundoff = 0x1p-24;

// Smallest variance whose transfer is at or below kFloat32Roundoff on the lattice's Nyquist boundary: 12 ln2 square, 9 ln2 hexagonal.
[[nodiscard]] float innerScaleVariance(pathtracer::gfx::PixelLattice lattice = pathtracer::gfx::PixelLattice::Square);

// Half-kernel of T(.;t): index 0 the centre tap, index n the pair T(+/-n;t), radius kernel.size()-1, truncated at tail mass 2^-24.
[[nodiscard]] std::vector<float> discreteGaussianKernel(float t);

// Half-support of diffuse's 2-D kernel at variance t, in storage columns and rows: how far a texel's response reaches.
[[nodiscard]] glm::ivec2 diffusionReach(float t, pathtracer::gfx::PixelLattice lattice);

// Adds variance t in place: square mirrors about the edge sample, hexagonal folds by the lattice's mirrors; x may wrap. No-op at t <= 0.
void diffuse(std::span<float> plane, int width, int height, float t, pathtracer::scene::ThreadPool& threadPool,
             bool wrapsHorizontally = false, pathtracer::gfx::PixelLattice lattice = pathtracer::gfx::PixelLattice::Square);

}  // namespace pathtracer::debug
