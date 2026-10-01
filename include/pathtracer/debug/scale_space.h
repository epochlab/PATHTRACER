#pragma once

#include <span>
#include <vector>

#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// Lindeberg's discrete scale space over one float plane: kernel T(n;t) = exp(-t) I_n(t), the shared Gaussian facility. See PIPELINE.md.

// Unit roundoff of binary32, the one threshold this file is built on: a contribution below it cannot perturb a float result.
inline constexpr double kFloat32Roundoff = 0x1p-24;

// Smallest variance whose transfer exp(-t(1-cos w)) is at or below kFloat32Roundoff at the grid Nyquist w=pi, so t = 12 ln2.
[[nodiscard]] float innerScaleVariance();

// Half-kernel of T(.;t): index 0 the centre tap, index n the pair T(+/-n;t), radius kernel.size()-1, truncated at tail mass 2^-24.
[[nodiscard]] std::vector<float> discreteGaussianKernel(float t);

// Adds variance t in place by two separable passes, mirroring about the edge sample so every tap lands on real data. No-op at t <= 0.
void diffuse(std::span<float> plane, int width, int height, float t, pathtracer::scene::ThreadPool& threadPool);

}  // namespace pathtracer::debug
