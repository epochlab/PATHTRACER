#pragma once

#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/denoiser.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// CPU implementations of the derived AOVs, each a per-texel function of accumulated lanes.

// Scene luminance, one channel.
[[nodiscard]] pathtracer::gfx::HdrImage luminanceAov(const pathtracer::gfx::HdrImage& beauty,
                                                  pathtracer::scene::ThreadPool& threadPool);

// HSV as three channels: hue and saturation normalised to [0,1], value left scene-referred so it is not clamped at white.
[[nodiscard]] pathtracer::gfx::HdrImage hsvAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool);

// Scene luminance SNR from replicated RQMC: the replicates' mean over its standard error, from their spread (L'Ecuyer & Lemieux 2002).
[[nodiscard]] pathtracer::gfx::HdrImage snrAov(int width, int height, const float* replicateLuminance, int replicates, int samples,
                                            pathtracer::scene::ThreadPool& threadPool);

// Each texel's pass count as a float, the samples its estimate averages; null counts, before any pass, read zero.
[[nodiscard]] pathtracer::gfx::HdrImage sampleCountAov(int width, int height, const int* texelSamples, pathtracer::scene::ThreadPool& threadPool);

// 1 - direct luminance / unoccluded direct luminance at the primary vertex, as a ratio of means; 0 where no direct light arrives.
[[nodiscard]] pathtracer::gfx::HdrImage shadowAov(const pathtracer::gfx::HdrImage& lost, const pathtracer::gfx::HdrImage& unoccluded,
                                               pathtracer::scene::ThreadPool& threadPool);

// Everything a BeautyFilter AOV reads, as explicit fields rather than a PathTraceResult, so a validator can build one with no renderer.
struct FilterInput {
    const pathtracer::gfx::HdrImage& beauty;  // published mean radiance, linear scene space, scene-referred and unbounded
    // Per-replicate running luminance means, `replicates` per texel, and the passes averaged in: min(samples, replicates) are populated.
    const float* replicateLuminance = nullptr;
    int replicates = 1;
    int samples = 0;
    const int* texelSamples = nullptr;  // passes per texel, `samples` wherever adaptive sampling never stopped it
    // Denoised's auxiliary features from Beauty's own samples, and the OIDN filter state reused across calls; Denoised needs all three.
    const pathtracer::gfx::HdrImage* denoiseAlbedo = nullptr;
    const pathtracer::gfx::HdrImage* denoiseNormal = nullptr;
    Denoiser* denoiser = nullptr;
    // Shadow's two accumulated means, the occluded share's numerator and its denominator; Shadow needs both.
    const pathtracer::gfx::HdrImage* shadowLost = nullptr;
    const pathtracer::gfx::HdrImage* shadowUnoccluded = nullptr;
};

// The one dispatch for every BeautyFilter AOV; a non-filter id yields an empty image. No default arm, so a new filter must be routed.
[[nodiscard]] pathtracer::gfx::HdrImage evaluateFilterAov(AovId aov, const FilterInput& input,
                                                       pathtracer::scene::ThreadPool& threadPool);

}  // namespace pathtracer::debug
