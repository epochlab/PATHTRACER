#pragma once

#include <span>
#include <vector>

#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// Lindeberg's discrete scale space over one float plane: kernel T(n;t) = exp(-t) I_n(t), the shared Gaussian facility.

// Unit roundoff of binary32, the one threshold this file is built on: a contribution below it cannot perturb a float result.
inline constexpr double kFloat32Roundoff = 0x1p-24;

// Smallest variance whose transfer exp(-t(1-cos w)) is at or below kFloat32Roundoff at the grid Nyquist w=pi, so t = 12 ln2.
[[nodiscard]] float innerScaleVariance();

// Half-kernel of T(.;t): index 0 the centre tap, index n the pair T(+/-n;t), radius kernel.size()-1, truncated at tail mass 2^-24.
[[nodiscard]] std::vector<float> discreteGaussianKernel(float t);

// One octave of the cascade: the plane, its own grid, and the base-grid variance it carries. Own variance is baseVariance/decimation^2.
struct ScaleSpaceLevel {
    std::vector<float> plane;
    int width = 0;
    int height = 0;
    int decimation = 1;         // base pixels per level pixel, a power of two
    float baseVariance = 0.0F;  // t in base-grid pixels squared
};

// Adds variance t in place by two separable passes, mirroring about the edge sample so every tap lands on real data. No-op at t <= 0.
void diffuse(std::span<float> plane, int width, int height, float t, pathtracer::scene::ThreadPool& threadPool);

// Octave level k (Burt & Adelson 1983, Lindeberg kernels): variance innerScaleVariance()*4^k, the grid halved past decimationVariance.
[[nodiscard]] ScaleSpaceLevel octaveLevel(std::span<const float> plane, int width, int height, int level,
                                          pathtracer::scene::ThreadPool& threadPool);

// Levels whose own kernel support fits their own grid: past it a level reports the mirror, not the image, at every sample.
[[nodiscard]] int octaveLevelCount(int width, int height);

// target += weight * the level resampled onto a finer grid of `targetDecimation` base pixels per sample, the base grid by default.
void addExpanded(const ScaleSpaceLevel& level, float weight, std::span<float> target, int targetWidth, int targetHeight,
                 pathtracer::scene::ThreadPool& threadPool, int targetDecimation = 1);

}  // namespace pathtracer::debug
