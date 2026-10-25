#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/denoiser.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::debug {

// CPU implementations of the derived AOVs. Sobel clamps at the edge, Gabor mirrors, hexagonal folds by its mirrors; x may wrap.

// Radial half-response bandwidth of the Morlet bank, in octaves: one band an octave wide about the carrier.
inline constexpr float kMorletOctaves = 1.0F;

// Carrier frequency in radians per pitch at envelope variance t: sigma*omega depends on the bandwidth alone (Petkov 1995 eq. 4).
[[nodiscard]] float morletCarrier(float variance);

// Orientations needed to cover every direction at half response or better, from the angular bandwidth the carrier already fixes.
[[nodiscard]] int morletOrientations();

// One bank orientation's plane wave exp(i(kx x + ky y)), held as row and column factors so no texel pays a transcendental.
struct MorletPlaneWave {
    double stepX = 0.0;  // kx, radians per pitch
    double stepY = 0.0;  // ky, radians per pitch
    std::vector<float> cosX, sinX;  // cos(kx x), sin(kx x), one per column
    std::vector<float> cosY, sinY;  // cos, sin of row y's phase: ky y, plus kx/2 on odd hexagonal rows and the sqrt(3)/2 row pitch
};

// Orientation `index` of `count` spread over a half turn at `carrier` radians per pitch, tabulated at the lattice's texel centres.
[[nodiscard]] MorletPlaneWave morletPlaneWave(float carrier, int index, int count, int width, int height,
                                              pathtracer::gfx::PixelLattice lattice = pathtracer::gfx::PixelLattice::Square);

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

// Gradient magnitude of Luminance under 3x3 Sobel (Sobel & Feldman 1968), or its six-neighbour gain-matched peer on hexagonal; one channel.
[[nodiscard]] pathtracer::gfx::HdrImage sobelAov(const pathtracer::gfx::HdrImage& beauty,
                                              pathtracer::scene::ThreadPool& threadPool, bool wrapsHorizontally = false,
                                              pathtracer::gfx::PixelLattice lattice = pathtracer::gfx::PixelLattice::Square);

// Peak envelope of the 2-D Morlet bank over Luminance (Morlet 1982; Antoine & Murenzi 1996): a tuned grating's amplitude, one channel.
[[nodiscard]] pathtracer::gfx::HdrImage gaborAov(const pathtracer::gfx::HdrImage& beauty,
                                              pathtracer::scene::ThreadPool& threadPool, bool wrapsHorizontally = false,
                                              pathtracer::gfx::PixelLattice lattice = pathtracer::gfx::PixelLattice::Square);

// HSV as three channels: hue and saturation normalised to [0,1], value left scene-referred so it is not clamped at white.
[[nodiscard]] pathtracer::gfx::HdrImage hsvAov(const pathtracer::gfx::HdrImage& beauty,
                                            pathtracer::scene::ThreadPool& threadPool);

// Cone-opponent displacement from scene white (cone_space.h): (l - l_white, s - s_white), two channels, invariant to a positive gain.
[[nodiscard]] pathtracer::gfx::HdrImage colourOpponentAov(const pathtracer::gfx::HdrImage& beauty,
                                                          pathtracer::scene::ThreadPool& threadPool);

// Scene luminance SNR from replicated RQMC: the replicates' mean over its standard error, from their spread (L'Ecuyer & Lemieux 2002).
[[nodiscard]] pathtracer::gfx::HdrImage snrAov(int width, int height, const float* replicateLuminance, int replicates, int samples,
                                            pathtracer::scene::ThreadPool& threadPool);

// Each texel's pass count as a float, the samples its estimate averages; null counts, before any pass, read zero.
[[nodiscard]] pathtracer::gfx::HdrImage sampleCountAov(int width, int height, const int* texelSamples, pathtracer::scene::ThreadPool& threadPool);

// Barten's (1999) standard observer adapted to one field: the pupil and field-size terms its CSF shares across every texel and band.
struct BartenObserver {
    double sigmaSquared = 0.0;  // squared optical line-spread sigma, deg^2: sigma0^2 + (Cab d)^2
    double trolandsPerLuminance = 0.0;  // retinal illuminance per cd/m^2: pupil area pi d^2/4 with the Stiles-Crawford correction
    double fieldTerm = 0.0;  // 1/X0^2 + 1/Xmax^2, deg^-2: the spatial integration area's frequency-independent part
};

// Barten 1999 pupil diameter in mm for field luminance L in cd/m^2 subtending X0 x X0 degrees: 5 - 3 tanh(0.4 log10(L X0^2 / 40^2)).
[[nodiscard]] double bartenPupilDiameter(double luminance, double fieldDegrees);

// The observer for a field of mean luminance L > 0 over X0 x X0 degrees, its pupil from bartenPupilDiameter.
[[nodiscard]] BartenObserver bartenObserver(double luminance, double fieldDegrees);

// Barten 1999 contrast sensitivity, 1 / threshold modulation, at u > 0 cycles/deg and local luminance L cd/m^2; 0 at L = 0.
[[nodiscard]] double bartenSensitivity(const BartenObserver& observer, double cyclesPerDegree, double luminance);

// Barten's ceiling over every u, L and field: M_opt <= 1, saturated inhibition, no photon noise, X0 -> inf: sqrt(T Xmax^2 / 2 Phi0) / k.
[[nodiscard]] double bartenSensitivityBound();

// Isotropic angle per lattice pitch at each texel centre, radians: sqrt |d(dir)/dx x d(dir)/dy|; 0 where the lens casts no ray.
[[nodiscard]] std::vector<float> texelAngularPitch(const pathtracer::scene::Camera& camera, int width, int height,
                                                   pathtracer::scene::ThreadPool& threadPool);

// Max over octave bands of Peli's (1990) local band contrast times Barten's sensitivity, one channel: 1 is threshold, 0 invisible.
[[nodiscard]] pathtracer::gfx::HdrImage contrastSensitivityAov(const pathtracer::gfx::HdrImage& beauty, std::span<const float> angularPitch,
                                                               pathtracer::scene::ThreadPool& threadPool, bool wrapsHorizontally = false,
                                                               pathtracer::gfx::PixelLattice lattice = pathtracer::gfx::PixelLattice::Square);

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
    bool wrapsHorizontally = false;  // the lens joins the image's left and right edges, so neighbourhood filters wrap in x
    pathtracer::gfx::PixelLattice lattice = pathtracer::gfx::PixelLattice::Square;  // where the texels sit, so a stencil finds neighbours
    // Denoised's auxiliary features from Beauty's own samples, and the OIDN filter state reused across calls; Denoised needs all three.
    const pathtracer::gfx::HdrImage* denoiseAlbedo = nullptr;
    const pathtracer::gfx::HdrImage* denoiseNormal = nullptr;
    Denoiser* denoiser = nullptr;
    // Shadow's two accumulated means, the occluded share's numerator and its denominator; Shadow needs both.
    const pathtracer::gfx::HdrImage* shadowLost = nullptr;
    const pathtracer::gfx::HdrImage* shadowUnoccluded = nullptr;
    const pathtracer::scene::Camera* camera = nullptr;  // the camera Beauty was traced through, for Contrast Sensitivity's angular pitch
};

// The one dispatch for every BeautyFilter AOV; a non-filter id yields an empty image. No default arm, so a new filter must be routed.
[[nodiscard]] pathtracer::gfx::HdrImage evaluateFilterAov(AovId aov, const FilterInput& input,
                                                       pathtracer::scene::ThreadPool& threadPool);

}  // namespace pathtracer::debug
