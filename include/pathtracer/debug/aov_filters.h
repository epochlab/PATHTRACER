#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/denoiser.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// CPU implementations of the nine derived AOVs. Sobel clamps at the edge, DoG and Gabor mirror; all three wrap a periodic x.

// Radial half-response bandwidth of the Morlet bank, in octaves: DoG's fine-to-coarse spacing, so both operators resolve one band.
inline constexpr float kMorletOctaves = 1.0F;

// Carrier frequency in radians per pixel at envelope variance t: sigma*omega depends on the bandwidth alone (Petkov 1995 eq. 4).
[[nodiscard]] float morletCarrier(float variance);

// Orientations needed to cover every direction at half response or better, from the angular bandwidth the carrier already fixes.
[[nodiscard]] int morletOrientations();

// One bank orientation's plane wave exp(i(kx x + ky y)), held as row and column factors so no texel pays a transcendental.
struct MorletPlaneWave {
    double stepX = 0.0;  // kx, radians per pixel
    double stepY = 0.0;  // ky, radians per pixel
    std::vector<float> cosX, sinX;  // cos(kx x), sin(kx x), one per column
    std::vector<float> cosY, sinY;  // cos(ky y), sin(ky y), one per row
};

// Orientation `index` of `count` spread over a half turn at `carrier` radians per pixel, tabulated on a width x height grid.
[[nodiscard]] MorletPlaneWave morletPlaneWave(float carrier, int index, int count, int width, int height);

// real + i imaginary = plane * exp(-i(kx x + ky y)): the carrier factors out, so a Gaussian blur after it is the Morlet envelope.
void demodulate(std::span<const float> plane, const MorletPlaneWave& wave, int width, int height, std::span<float> real,
                std::span<float> imaginary, pathtracer::scene::ThreadPool& threadPool);

// Admissible baseband at (x, y): the blurred demodulated pair minus what a constant field of the local lowpass would have produced.
[[nodiscard]] inline glm::vec2 morletBaseband(const MorletPlaneWave& blurredWave, int x, int y, float real, float imaginary,
                                              float lowpass) {
    const float rowCos = blurredWave.cosY[static_cast<std::size_t>(y)];
    const float rowSin = blurredWave.sinY[static_cast<std::size_t>(y)];
    // The blurred plane wave under the same mirror, which is what a constant field would have produced right here.
    const float meanReal = (blurredWave.cosX[static_cast<std::size_t>(x)] * rowCos) -
                           (blurredWave.sinX[static_cast<std::size_t>(x)] * rowSin);
    const float meanImaginary = (blurredWave.sinX[static_cast<std::size_t>(x)] * rowCos) +
                                (blurredWave.cosX[static_cast<std::size_t>(x)] * rowSin);
    const float centredReal = real - (meanReal * lowpass);
    const float centredImaginary = imaginary + (meanImaginary * lowpass);
    return {centredReal, centredImaginary};
}

// Scene luminance, one channel. Single centre tap, no neighbourhood.
[[nodiscard]] pathtracer::gfx::HdrImage luminanceAov(const pathtracer::gfx::HdrImage& beauty,
                                                  pathtracer::scene::ThreadPool& threadPool);

// Gradient magnitude of Luminance under the fixed 3x3 Sobel operator (Sobel & Feldman 1968), one channel.
[[nodiscard]] pathtracer::gfx::HdrImage sobelAov(const pathtracer::gfx::HdrImage& beauty,
                                              pathtracer::scene::ThreadPool& threadPool, bool wrapsHorizontally = false);

// Peak envelope of the 2-D Morlet bank over Luminance (Morlet 1982; Antoine & Murenzi 1996): a tuned grating's amplitude, one channel.
[[nodiscard]] pathtracer::gfx::HdrImage gaborAov(const pathtracer::gfx::HdrImage& beauty,
                                              pathtracer::scene::ThreadPool& threadPool, bool wrapsHorizontally = false);

// HSV as three channels: hue and saturation normalised to [0,1], value left scene-referred so it is not clamped at white.
[[nodiscard]] pathtracer::gfx::HdrImage hsvAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool);

// Signed first band of the Laplacian pyramid (Burt & Adelson 1983), finest octave minus the next, one channel. Zero if it cannot fit.
[[nodiscard]] pathtracer::gfx::HdrImage dogAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool, bool wrapsHorizontally = false);

// Cone-opponent displacement from scene white (cone_space.h): (l - l_white, s - s_white), two channels, invariant to a positive gain.
[[nodiscard]] pathtracer::gfx::HdrImage colourOpponentAov(const pathtracer::gfx::HdrImage& beauty,
                                                          pathtracer::scene::ThreadPool& threadPool);

// Scene luminance SNR from replicated RQMC: the replicates' mean over its standard error, from their spread (L'Ecuyer & Lemieux 2002).
[[nodiscard]] pathtracer::gfx::HdrImage snrAov(int width, int height, const float* replicateLuminance, int replicates, int samples,
                                            pathtracer::scene::ThreadPool& threadPool);

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
    bool wrapsHorizontally = false;  // the lens joins the image's left and right edges, so neighbourhood filters wrap in x
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
