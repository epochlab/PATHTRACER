#pragma once

#include <cstddef>

#include <glm/glm.hpp>

#include "pathtracer/debug/aov.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// CPU implementations of the seven Beauty-reading AOVs. Sobel clamps at the edge; DoG and Gabor, built on scale_space.h, mirror.

// Rec.709 luminance weights (ITU-R BT.709-6), the one triple every luminance reduction in the engine dots against.
inline constexpr glm::vec3 kRec709LuminanceWeights{0.2126F, 0.7152F, 0.0722F};

// Radial half-response bandwidth of the Morlet bank, in octaves: DoG's fine-to-coarse spacing, so both operators resolve one band.
inline constexpr float kMorletOctaves = 1.0F;

// Carrier frequency in radians per pixel at envelope variance t: sigma*omega depends on the bandwidth alone (Petkov 1995 eq. 4).
[[nodiscard]] float morletCarrier(float variance);

// Orientations needed to cover every direction at half response or better, from the angular bandwidth the carrier already fixes.
[[nodiscard]] int morletOrientations();

// Rec.709 luminance, one channel. Single centre tap, no neighbourhood.
[[nodiscard]] pathtracer::gfx::HdrImage luminanceAov(const pathtracer::gfx::HdrImage& beauty,
                                                  pathtracer::scene::ThreadPool& threadPool);

// Gradient magnitude of Luminance under the fixed 3x3 Sobel operator (Sobel & Feldman 1968), one channel.
[[nodiscard]] pathtracer::gfx::HdrImage sobelAov(const pathtracer::gfx::HdrImage& beauty,
                                              pathtracer::scene::ThreadPool& threadPool);

// Peak quadrature magnitude of the 2-D Morlet wavelet bank over Luminance (Morlet 1982; Antoine & Murenzi 1996), one channel.
[[nodiscard]] pathtracer::gfx::HdrImage gaborAov(const pathtracer::gfx::HdrImage& beauty,
                                              pathtracer::scene::ThreadPool& threadPool);

// HSV as three channels: hue and saturation normalised to [0,1], value left scene-referred so it is not clamped at white.
[[nodiscard]] pathtracer::gfx::HdrImage hsvAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool);

// Signed first band of the Laplacian pyramid (Burt & Adelson 1983), finest octave minus the next, one channel. Zero if it cannot fit.
[[nodiscard]] pathtracer::gfx::HdrImage dogAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool);

// Cone-opponent displacement from Rec.709 white (cone_space.h): (l - l_white, s - s_white), two channels, invariant to a positive gain.
[[nodiscard]] pathtracer::gfx::HdrImage colourOpponentAov(const pathtracer::gfx::HdrImage& beauty,
                                                          pathtracer::scene::ThreadPool& threadPool);

// Signal-to-noise ratio of each texel's published radiance: its Rec.709 luminance over the standard error of that mean.
[[nodiscard]] pathtracer::gfx::HdrImage snrAov(const pathtracer::gfx::HdrImage& beauty,
                                            const float* beautyLuminanceM2, int samples,
                                            pathtracer::scene::ThreadPool& threadPool);

// Everything a BeautyFilter AOV reads, as explicit fields rather than a PathTraceResult, so a validator can build one with no renderer.
struct FilterInput {
    const pathtracer::gfx::HdrImage& beauty;  // published mean radiance, linear Rec.709, scene-referred and unbounded
    // Welford second moment of the per-pass luminance and the passes averaged in; a variance is undefined below two of them.
    const float* beautyLuminanceM2 = nullptr;
    int samples = 0;
};

// The one dispatch for every BeautyFilter AOV; a non-filter id yields an empty image. No default arm, so a new filter must be routed.
[[nodiscard]] pathtracer::gfx::HdrImage evaluateFilterAov(AovId aov, const FilterInput& input,
                                                       pathtracer::scene::ThreadPool& threadPool);

}  // namespace pathtracer::debug
