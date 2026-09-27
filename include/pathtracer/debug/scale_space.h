#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// Lindeberg's discrete scale space over one float plane: kernel T(n;t) = exp(-t) I_n(t), the shared Gaussian facility. See PIPELINE.md.

// Unit roundoff of binary32, the one threshold this file is built on: a contribution below it cannot perturb a float result.
inline constexpr double kFloat32Roundoff = 0x1p-24;

// Smallest variance whose transfer exp(-t(1-cos w)) is at or below kFloat32Roundoff at the grid Nyquist w=pi, so t = 12 ln2.
[[nodiscard]] float innerScaleVariance();

// The same criterion at the post-decimation Nyquist w=pi/2: t = 24 ln2, above which halving the grid discards nothing.
[[nodiscard]] float decimationVariance();

// Half-kernel of T(.;t): index 0 the centre tap, index n the pair T(+/-n;t), radius kernel.size()-1, truncated at tail mass 2^-24.
[[nodiscard]] std::vector<float> discreteGaussianKernel(float t);

// One octave of the pyramid: the plane, its own grid, and the base-grid variance it carries. Own variance is baseVariance/decimation^2.
struct ScaleSpaceLevel {
    std::vector<float> plane;
    int width = 0;
    int height = 0;
    int decimation = 1;         // base pixels per level pixel, a power of two
    float baseVariance = 0.0F;  // t in base-grid pixels squared
};

// Adds variance t in place by two separable passes, mirroring about the edge sample so every tap lands on real data. No-op at t <= 0.
void diffuse(std::span<float> plane, int width, int height, float t, pathtracer::scene::ThreadPool& threadPool);

// Octave cascade (Burt & Adelson 1983, Lindeberg kernels): level k carries innerScaleVariance()*4^k while it fits, maxLevels at most.
[[nodiscard]] std::vector<ScaleSpaceLevel> buildOctavePyramid(std::span<const float> plane, int width, int height,
                                                              pathtracer::scene::ThreadPool& threadPool,
                                                              std::size_t maxLevels = std::numeric_limits<std::size_t>::max());

// Bilinear resample of a level onto the base grid, exact at co-located samples: a coarse level holds nothing between its own samples.
[[nodiscard]] std::vector<float> expandToBase(const ScaleSpaceLevel& level, int baseWidth, int baseHeight,
                                              pathtracer::scene::ThreadPool& threadPool);

// target += weight * the level resampled onto a finer grid of `targetDecimation` base pixels per sample, the base grid by default.
void addExpanded(const ScaleSpaceLevel& level, float weight, std::span<float> target, int targetWidth, int targetHeight,
                 pathtracer::scene::ThreadPool& threadPool, int targetDecimation = 1);

// Equal-weight mean of a non-empty pyramid on the base grid, collapsed coarse to fine: bilinear restricts exactly to a nested lattice.
[[nodiscard]] std::vector<float> octaveMean(const std::vector<ScaleSpaceLevel>& pyramid, pathtracer::scene::ThreadPool& threadPool);

// The 5-point Laplacian under the same mirror, so it is zero-flux there; weights sum to zero, so constants map to exactly zero.
void laplacian5(std::span<const float> plane, int width, int height, std::span<float> out,
                pathtracer::scene::ThreadPool& threadPool);

}  // namespace pathtracer::debug
