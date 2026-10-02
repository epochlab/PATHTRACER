#pragma once

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// Image flow: Morlet phase constraints (Fleet & Jepson 1990), pooled per texel (Simoncelli & Heeger 1998), Bayesian coarse to fine.


// Prior std in pixels: half the carrier wavelength at the coarsest octave that fits, the largest displacement a phase difference can tell.
[[nodiscard]] float opticFlowPriorStd(int width, int height);

// One view of the flow source: its mean image and, where the producer knows it, the per-texel variance of that mean (null: deterministic).
struct OpticFlowView {
    const pathtracer::gfx::HdrImage* mean = nullptr;
    const pathtracer::gfx::HdrImage* variance = nullptr;
};

// (dx, dy, sigma) on current's grid, current(x) ~ previous(x - d), sigma the worst-axis posterior std; no same-shape previous: the prior.
[[nodiscard]] pathtracer::gfx::HdrImage opticFlowAov(const OpticFlowView& current, const OpticFlowView& previous,
                                                     pathtracer::scene::ThreadPool& threadPool);

}  // namespace pathtracer::debug
