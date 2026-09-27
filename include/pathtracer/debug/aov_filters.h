#pragma once

#include <cstddef>

#include <glm/glm.hpp>

#include "pathtracer/debug/aov.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// CPU implementations of the ten Beauty-reading AOVs. Sobel clamps at the edge; everything built on scale_space.h mirrors, Gabor included.

// Rec.709 luminance weights (ITU-R BT.709-6), the one triple every luminance reduction in the engine dots against.
inline constexpr glm::vec3 kRec709LuminanceWeights{0.2126F, 0.7152F, 0.0722F};

// Radial half-response bandwidth of the Morlet bank, in octaves: the shared ladder's own spacing, so the bank tiles its frequency plane.
inline constexpr float kMorletOctaves = 1.0F;

// Carrier frequency in radians per pixel at envelope variance t: sigma*omega depends on the bandwidth alone (Petkov 1995 eq. 4).
[[nodiscard]] float morletCarrier(float variance);

// Orientations needed to cover every direction at half response or better, from the angular bandwidth the carrier already fixes.
[[nodiscard]] int morletOrientations();

// Rec.709 luminance, broadcast to RGB. Single centre tap, no neighbourhood.
[[nodiscard]] pathtracer::gfx::HdrImage luminanceAov(const pathtracer::gfx::HdrImage& beauty,
                                                  pathtracer::scene::ThreadPool& threadPool);

// Gradient magnitude of Luminance under the fixed 3x3 Sobel operator (Sobel & Feldman 1968), broadcast to RGB.
[[nodiscard]] pathtracer::gfx::HdrImage sobelAov(const pathtracer::gfx::HdrImage& beauty,
                                              pathtracer::scene::ThreadPool& threadPool);

// Peak quadrature magnitude of the 2-D Morlet wavelet bank over Luminance (Morlet 1982; Antoine & Murenzi 1996), broadcast to RGB.
[[nodiscard]] pathtracer::gfx::HdrImage gaborAov(const pathtracer::gfx::HdrImage& beauty,
                                              pathtracer::scene::ThreadPool& threadPool);

// Hue, saturation and value in RGB; hue and saturation normalised to [0,1], value left scene-referred so it is not clamped at white.
[[nodiscard]] pathtracer::gfx::HdrImage hsvAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool);

// Signed first band of the Laplacian pyramid (Burt & Adelson 1983), finest octave minus the next, broadcast to RGB. Zero if absent.
[[nodiscard]] pathtracer::gfx::HdrImage dogAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool);

// Signed scale-normalised Laplacian extremum over the octave ladder (Lindeberg 1998, gamma=1), positive on a bright blob like DoG.
[[nodiscard]] pathtracer::gfx::HdrImage logAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool);

// Cone-opponent displacement from Rec.709 white (cone_space.h): (l - l_white, s - s_white), exactly invariant to a positive gain.
[[nodiscard]] pathtracer::gfx::HdrImage colourOpponentAov(const pathtracer::gfx::HdrImage& beauty,
                                                          pathtracer::scene::ThreadPool& threadPool);

// Per-channel retinex in log radiance (Stockham 1972) under Land 1986's inverse-square surround: a dimensionless reflectance estimate.
[[nodiscard]] pathtracer::gfx::HdrImage retinexAov(const pathtracer::gfx::HdrImage& beauty,
                                                pathtracer::scene::ThreadPool& threadPool);

// Ward Larson, Rushmeier & Piatko 1997 histogram adjustment, linear ceiling, onto the sRGB reference display's range; chromaticity kept.
[[nodiscard]] pathtracer::gfx::HdrImage claheAov(const pathtracer::gfx::HdrImage& beauty, float pixelsPerRadian,
                                              pathtracer::scene::ThreadPool& threadPool);

// Signal-to-noise ratio of each texel's published radiance: its Rec.709 luminance over the standard error of that mean.
[[nodiscard]] pathtracer::gfx::HdrImage snrAov(const pathtracer::gfx::HdrImage& beauty,
                                            const float* beautyLuminanceM2, int samples,
                                            pathtracer::scene::ThreadPool& threadPool);

// Everything a BeautyFilter AOV reads, as explicit fields rather than a PathTraceResult, so a validator can build one with no renderer.
struct FilterInput {
    const pathtracer::gfx::HdrImage& beauty;  // published mean radiance, linear Rec.709, scene-referred and unbounded
    float pixelsPerRadian;                    // Camera::pixelsPerRadian at beauty's height: the one anchor from pixels to visual angle
    // Welford second moment of the per-pass luminance and the passes averaged in; a variance is undefined below two of them.
    const float* beautyLuminanceM2 = nullptr;
    int samples = 0;
};

// The one dispatch for every BeautyFilter AOV; a non-filter id yields an empty image. No default arm, so a new filter must be routed.
[[nodiscard]] pathtracer::gfx::HdrImage evaluateFilterAov(AovId aov, const FilterInput& input,
                                                       pathtracer::scene::ThreadPool& threadPool);

}  // namespace pathtracer::debug
