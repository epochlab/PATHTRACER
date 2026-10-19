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

// diffuse's generator symbol along the row axis: variance t scales a row-axis plane wave of omega radians per pitch by exp(-t a).
[[nodiscard]] double rowAxisSymbol(double omega, pathtracer::gfx::PixelLattice lattice);

// The row axis' zone boundary, where its symbol peaks: pi square, the K point's 4 pi/3 hexagonal.
[[nodiscard]] double rowAxisBoundary(pathtracer::gfx::PixelLattice lattice);

// rowAxisSymbol's inverse on [0, rowAxisBoundary]: the row-axis frequency whose symbol is a, in closed form on both lattices.
[[nodiscard]] double rowAxisFrequency(double a, pathtracer::gfx::PixelLattice lattice);

// Half-kernel of T(.;t): index 0 the centre tap, index n the pair T(+/-n;t), radius kernel.size()-1, truncated at tail mass 2^-24.
[[nodiscard]] std::vector<float> discreteGaussianKernel(float t);

// Half-support of diffuse's 2-D kernel at variance t, in storage columns and rows: how far a texel's response reaches.
[[nodiscard]] glm::ivec2 diffusionReach(float t, pathtracer::gfx::PixelLattice lattice);

// Adds variance t in place: square mirrors about the edge sample, hexagonal folds by the lattice's mirrors; x may wrap. No-op at t <= 0.
void diffuse(std::span<float> plane, int width, int height, float t, pathtracer::scene::ThreadPool& threadPool,
             bool wrapsHorizontally = false, pathtracer::gfx::PixelLattice lattice = pathtracer::gfx::PixelLattice::Square);

}  // namespace pathtracer::debug
