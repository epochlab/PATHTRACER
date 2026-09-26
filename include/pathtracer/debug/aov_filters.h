#pragma once

#include <array>
#include <cstddef>

#include <glm/glm.hpp>

#include "pathtracer/debug/aov.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// CPU implementations of the nine Beauty-reading AOVs. Sobel and Gabor clamp at the edge; the scale-space AOVs mirror, see scale_space.h.

// Rec.709 luminance weights (ITU-R BT.709-6), the one triple every luminance reduction in the engine dots against.
inline constexpr glm::vec3 kRec709LuminanceWeights{0.2126F, 0.7152F, 0.0722F};

inline constexpr int kGaborOrientations = 4;
inline constexpr int kGaborRadius = 2;  // 5x5 support
inline constexpr int kGaborTaps = (2 * kGaborRadius + 1) * (2 * kGaborRadius + 1);
// Extent of the flattened orientation-major bank. size_t because it is only ever an array bound.
inline constexpr std::size_t kGaborKernelSize = std::size_t{kGaborOrientations} * std::size_t{kGaborTaps};

// 4 orientations x 25 taps of an odd-carrier Gabor filter (Gabor 1946; Daugman 1985), orientation-major: index o * 25 + tap.
[[nodiscard]] std::array<float, kGaborKernelSize> buildGaborKernel();

// Rec.709 luminance, broadcast to RGB. Single centre tap, no neighbourhood.
[[nodiscard]] pathtracer::gfx::HdrImage luminanceAov(const pathtracer::gfx::HdrImage& beauty,
                                                  pathtracer::scene::ThreadPool& threadPool);

// Gradient magnitude of Luminance under the fixed 3x3 Sobel operator (Sobel & Feldman 1968), broadcast to RGB.
[[nodiscard]] pathtracer::gfx::HdrImage sobelAov(const pathtracer::gfx::HdrImage& beauty,
                                              pathtracer::scene::ThreadPool& threadPool);

// Maximum absolute response of the 4-orientation Gabor bank over Luminance, broadcast to RGB.
[[nodiscard]] pathtracer::gfx::HdrImage gaborAov(const pathtracer::gfx::HdrImage& beauty,
                                              pathtracer::scene::ThreadPool& threadPool);

// Hue, saturation and value in RGB; hue and saturation normalised to [0,1], value left scene-referred so it is not clamped at white.
[[nodiscard]] pathtracer::gfx::HdrImage hsvAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool);

// Signed difference of the two finest pyramid octaves (Marr & Hildreth 1980), broadcast to RGB. Zero if the frame carries no such band.
[[nodiscard]] pathtracer::gfx::HdrImage dogAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool);

// Scale-normalised Laplacian extremum over the octave ladder (Lindeberg 1998, gamma=1): magnitude, its cycles/degree, and its polarity.
[[nodiscard]] pathtracer::gfx::HdrImage logAov(const pathtracer::gfx::HdrImage& beauty, float verticalFovRadians,
                                            pathtracer::scene::ThreadPool& threadPool);

// Cone-opponent displacement from Rec.709 white (cone_space.h): (l - l_white, s - s_white), exactly invariant to a positive gain.
[[nodiscard]] pathtracer::gfx::HdrImage opponentAov(const pathtracer::gfx::HdrImage& beauty,
                                                 pathtracer::scene::ThreadPool& threadPool);

// Per-channel Gaussian-surround retinex (Land 1986; Stockham 1972) at the pyramid's coarsest scale: a dimensionless reflectance estimate.
[[nodiscard]] pathtracer::gfx::HdrImage retinexAov(const pathtracer::gfx::HdrImage& beauty,
                                                pathtracer::scene::ThreadPool& threadPool);

// Contrast-limited adaptive histogram equalisation (Zuiderveld 1994) on log2 luminance under Ward Larson 1997's linear contrast ceiling.
[[nodiscard]] pathtracer::gfx::HdrImage claheAov(const pathtracer::gfx::HdrImage& beauty, float verticalFovRadians,
                                              pathtracer::scene::ThreadPool& threadPool);

// Everything a BeautyFilter AOV reads, as explicit fields rather than a PathTraceResult, so a validator can build one with no renderer.
struct FilterInput {
    const pathtracer::gfx::HdrImage& beauty;  // published mean radiance, linear Rec.709, scene-referred and unbounded
    float verticalFovRadians;                 // the rendering camera's, the sole anchor turning pixels into cycles per degree
};

// The one dispatch for every BeautyFilter AOV; a non-filter id yields an empty image. No default arm, so a new filter must be routed.
[[nodiscard]] pathtracer::gfx::HdrImage evaluateFilterAov(AovId aov, const FilterInput& input,
                                                       pathtracer::scene::ThreadPool& threadPool);

}  // namespace pathtracer::debug
