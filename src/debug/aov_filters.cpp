#include "pathtracer/debug/aov_filters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <vector>

#include "pathtracer/debug/scale_space.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/cone_space.h"
#include "pathtracer/scene/path_trace_driver.h"

namespace pathtracer::debug {

namespace {

using pathtracer::gfx::HdrImage;
using pathtracer::gfx::PixelLattice;
using pathtracer::scene::ThreadPool;

// Sobel's response to the ramp f = s x: its (1, 2, 1) column weights times the central difference 2s.
constexpr float kSobelAffineGain = 2.0F * (1.0F + 2.0F + 1.0F);
// sum_k f_k u_k.x over the six unit neighbours on f = s x: the row pair's 2s and four diagonals at (1/2)^2 each, so 3s.
constexpr float kHexagonalAffineGain = 2.0F + (4.0F * 0.25F);

// Scene luminance, materialised once (Sobel reads 8 neighbours); `pad` wrapped columns each side extend a periodic image past its seam.
[[nodiscard]] std::vector<float> luminancePlane(const HdrImage& beauty, ThreadPool& threadPool, int pad = 0) {
    const glm::vec3 weights = pathtracer::scene::cie::sceneLuminanceWeights();
    const int width = beauty.width + (2 * pad);
    std::vector<float> plane(static_cast<std::size_t>(width) * static_cast<std::size_t>(beauty.height));
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        const std::size_t sourceRow = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < width; ++x) {
            const auto column = static_cast<std::size_t>((((x - pad) % beauty.width) + beauty.width) % beauty.width);
            const std::size_t texel = (sourceRow + column) * static_cast<std::size_t>(beauty.channels);
            plane[row + static_cast<std::size_t>(x)] = (beauty.texels[texel] * weights.r) + (beauty.texels[texel + 1] * weights.g) +
                                                       (beauty.texels[texel + 2] * weights.b);
        }
    });
    return plane;
}

// Neighbour fetch clamped at the edge, as the display texture's GL_CLAMP_TO_EDGE, or wrapped in x where the image is periodic.
[[nodiscard]] float tap(const std::vector<float>& plane, int width, int height, int x, int y, bool wrapsHorizontally) {
    const int cx = wrapsHorizontally ? (x + width) % width : std::clamp(x, 0, width - 1);
    const int cy = std::clamp(y, 0, height - 1);
    return plane[(static_cast<std::size_t>(cy) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(cx)];
}

// Least-squares gradient over the six unit neighbours (Staunton 1989), at Sobel's affine gain; the frame's edge folds by its mirrors.
void hexagonalSobel(const std::vector<float>& plane, HdrImage& out, bool wrapsHorizontally, ThreadPool& threadPool) {
    const int width = out.width;
    const int height = out.height;
    const float gain = kSobelAffineGain / kHexagonalAffineGain;
    const float rowPitch = pathtracer::gfx::rowPitch(PixelLattice::Hexagonal);
    threadPool.parallelFor(height, [&](int y) {
        // A row's up and down neighbours sit at columns x-1+p and x+p, p its parity, half a pitch either side of x.
        const int parity = y & 1;
        for (int x = 0; x < width; ++x) {
            const bool interior = y > 0 && y < height - 1 && x > 0 && x < width - 1;
            const auto at = [&](int column, int row) {
                const glm::ivec2 cell =
                    interior ? glm::ivec2(column, row) : pathtracer::gfx::hexagonalReflection(column, row, width, height, wrapsHorizontally);
                return plane[(static_cast<std::size_t>(cell.y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(cell.x)];
            };
            const float l = at(x - 1, y);
            const float r = at(x + 1, y);
            const float ul = at(x - 1 + parity, y - 1);
            const float ur = at(x + parity, y - 1);
            const float dl = at(x - 1 + parity, y + 1);
            const float dr = at(x + parity, y + 1);
            // Differences first, so a constant field cancels exactly; the diagonals carry u.x = 1/2 and u.y = rowPitch.
            const float gx = (r - l) + (0.5F * ((ur - ul) + (dr - dl)));
            const float gy = rowPitch * ((dl + dr) - (ul + ur));
            writeTexel(out, x, y, gain * std::sqrt((gx * gx) + (gy * gy)));
        }
    });
}

// Barten 1999 standard observer as ITU-R BT.2246 and Cowan et al. 2004 tabulate it, in the model's degrees, seconds and trolands.
constexpr double kBartenSignalToNoise = 3.0;  // k, the detection signal-to-noise ratio
constexpr double kBartenIntegrationSeconds = 0.1;  // T, the eye's integration time
constexpr double kBartenQuantumEfficiency = 0.03;  // eta
constexpr double kBartenPhotonsPerTroland = 1.2274e6;  // p, photons / (s deg^2 Td)
constexpr double kBartenNeuralNoise = 3e-8;  // Phi0, neural noise spectral density, s deg^2
constexpr double kBartenInhibitionCyclesPerDegree = 7.0;  // u0, lateral inhibition's spatial frequency
constexpr double kBartenSigma0Degrees = 0.5 / 60.0;  // optical line spread with no pupil aberration, 0.5 arcmin
constexpr double kBartenAberrationDegreesPerMm = 0.08 / 60.0;  // Cab, line-spread growth per mm of pupil, 0.08 arcmin/mm
constexpr double kBartenMaxFieldDegrees = 12.0;  // Xmax, the largest angle the eye integrates over
constexpr double kBartenMaxCycles = 15.0;  // Nmax, the most cycles the eye integrates over

// The filter's output image, zeroed at the channel count aovChannels declares for it.
[[nodiscard]] HdrImage makeAovImage(AovId aov, int width, int height) {
    return pathtracer::gfx::makeImage(width, height, aovChannels(aov));
}

}  // namespace

HdrImage luminanceAov(const HdrImage& beauty, ThreadPool& threadPool) {
    return {beauty.width, beauty.height, pathtracer::gfx::kScalarChannels, luminancePlane(beauty, threadPool)};
}

HdrImage sobelAov(const HdrImage& beauty, ThreadPool& threadPool, bool wrapsHorizontally, PixelLattice lattice) {
    const std::vector<float> plane = luminancePlane(beauty, threadPool);
    HdrImage out = makeAovImage(AovId::Sobel, beauty.width, beauty.height);
    if (lattice == PixelLattice::Hexagonal) {
        hexagonalSobel(plane, out, wrapsHorizontally, threadPool);
        return out;
    }
    const int width = beauty.width;
    const int height = beauty.height;
    threadPool.parallelFor(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            const float tl = tap(plane, width, height, x - 1, y - 1, wrapsHorizontally);
            const float t = tap(plane, width, height, x, y - 1, wrapsHorizontally);
            const float tr = tap(plane, width, height, x + 1, y - 1, wrapsHorizontally);
            const float l = tap(plane, width, height, x - 1, y, wrapsHorizontally);
            const float r = tap(plane, width, height, x + 1, y, wrapsHorizontally);
            const float bl = tap(plane, width, height, x - 1, y + 1, wrapsHorizontally);
            const float b = tap(plane, width, height, x, y + 1, wrapsHorizontally);
            const float br = tap(plane, width, height, x + 1, y + 1, wrapsHorizontally);
            // Differences first, so a constant field cancels exactly rather than to the rounding of 4 * value.
            const float gx = (tr - tl) + (2.0F * (r - l)) + (br - bl);
            const float gy = (tl - bl) + (2.0F * (t - b)) + (tr - br);
            writeTexel(out, x, y, std::sqrt((gx * gx) + (gy * gy)));
        }
    });
    return out;
}

float morletCarrier(float variance) {
    // Petkov 1995 eq. 4: the half-response bandwidth in octaves fixes sigma/lambda alone, so the carrier follows from the envelope.
    const double octaves = static_cast<double>(kMorletOctaves);
    const double ratio = (std::exp2(octaves) + 1.0) / (std::exp2(octaves) - 1.0);
    return static_cast<float>((2.0 * std::sqrt(std::numbers::ln2 / 2.0) * ratio) / std::sqrt(static_cast<double>(variance)));
}

int morletOrientations() {
    // sigma*omega depends only on the bandwidth, so the angular half-response width does too, and with it the cover of a half turn.
    const double octaves = static_cast<double>(kMorletOctaves);
    const double sigmaOmega = 2.0 * std::sqrt(std::numbers::ln2 / 2.0) * ((std::exp2(octaves) + 1.0) / (std::exp2(octaves) - 1.0));
    // Half response at a chord of sqrt(2 ln2)/sigma from the carrier, so adjacent orientations may be no further apart than this.
    const double angularWidth = 4.0 * std::asin(std::sqrt(2.0 * std::numbers::ln2) / (2.0 * sigmaOmega));
    return static_cast<int>(std::ceil(std::numbers::pi / angularWidth));
}

MorletPlaneWave morletPlaneWave(float carrier, int index, int count, int width, int height, PixelLattice lattice) {
    MorletPlaneWave wave;
    const double theta = (std::numbers::pi * static_cast<double>(index)) / static_cast<double>(count);
    wave.stepX = static_cast<double>(carrier) * std::cos(theta);
    wave.stepY = static_cast<double>(carrier) * std::sin(theta);
    wave.cosX.resize(static_cast<std::size_t>(width));
    wave.sinX.resize(static_cast<std::size_t>(width));
    wave.cosY.resize(static_cast<std::size_t>(height));
    wave.sinY.resize(static_cast<std::size_t>(height));
    // Split by the angle-addition formula, so the row loop carries no transcendental at all.
    for (int x = 0; x < width; ++x) {
        wave.cosX[static_cast<std::size_t>(x)] = static_cast<float>(std::cos(wave.stepX * static_cast<double>(x)));
        wave.sinX[static_cast<std::size_t>(x)] = static_cast<float>(std::sin(wave.stepX * static_cast<double>(x)));
    }
    // A hexagonal row's half-pitch parity shift and sqrt(3)/2 pitch go in its own factor, so the phase still splits by column and row.
    const bool hexagonal = lattice == PixelLattice::Hexagonal;
    const auto rowPitch = static_cast<double>(pathtracer::gfx::rowPitch(lattice));
    for (int y = 0; y < height; ++y) {
        const double shift = hexagonal ? 0.5 * static_cast<double>(y & 1) : 0.0;
        const double phase = (wave.stepX * shift) + (wave.stepY * static_cast<double>(y) * rowPitch);
        wave.cosY[static_cast<std::size_t>(y)] = static_cast<float>(std::cos(phase));
        wave.sinY[static_cast<std::size_t>(y)] = static_cast<float>(std::sin(phase));
    }
    return wave;
}

void demodulate(std::span<const float> plane, const MorletPlaneWave& wave, int width, int height, std::span<float> real,
                std::span<float> imaginary, ThreadPool& threadPool) {
    threadPool.parallelFor(height, [&](int y) {
        const float rowCos = wave.cosY[static_cast<std::size_t>(y)];
        const float rowSin = wave.sinY[static_cast<std::size_t>(y)];
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            const float cosPhase = (wave.cosX[static_cast<std::size_t>(x)] * rowCos) - (wave.sinX[static_cast<std::size_t>(x)] * rowSin);
            const float sinPhase = (wave.sinX[static_cast<std::size_t>(x)] * rowCos) + (wave.cosX[static_cast<std::size_t>(x)] * rowSin);
            // Demodulate, blur, remodulate: the plane wave factors out of the convolution, so every orientation stays separable.
            real[pixel] = plane[pixel] * cosPhase;
            imaginary[pixel] = -plane[pixel] * sinPhase;
        }
    });
}

HdrImage gaborAov(const HdrImage& beauty, ThreadPool& threadPool, bool wrapsHorizontally, PixelLattice lattice) {
    // The finest scale the grid resolves.
    const float variance = innerScaleVariance(lattice);
    // Off-harmonic carriers make demodulation Bloch-periodic, J(x+W) = J(x)exp(-ikW): wrapped padding to the blur's reach keeps taps real.
    const int pad = wrapsHorizontally ? diffusionReach(variance, lattice).x : 0;
    const int width = beauty.width + (2 * pad);
    const int height = beauty.height;
    const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::vector<float> plane = luminancePlane(beauty, threadPool, pad);
    HdrImage out = makeAovImage(AovId::Gabor, beauty.width, height);

    const float carrier = morletCarrier(variance);
    const int orientations = morletOrientations();
    const bool hexagonal = lattice == PixelLattice::Hexagonal;

    std::vector<float> lowpass = plane;
    diffuse(lowpass, width, height, variance, threadPool, false, lattice);

    std::vector<float> peak(pixels, 0.0F);
    std::vector<float> real(pixels);
    std::vector<float> imaginary(pixels);
    // Hexagonal only: the blurred plane wave, which no longer separates into a row and a column factor once the passes run off-axis.
    const std::vector<float> ones(hexagonal ? pixels : 0, 1.0F);
    std::vector<float> waveReal(hexagonal ? pixels : 0);
    std::vector<float> waveImaginary(hexagonal ? pixels : 0);
    for (int orientation = 0; orientation < orientations; ++orientation) {
        // A half turn covers the bank: the magnitude at theta and theta+pi is the same, the two being complex conjugates.
        const MorletPlaneWave wave = morletPlaneWave(carrier, orientation, orientations, width, height, lattice);
        // Morlet's admissibility term as the blurred plane wave itself, under the same boundary: zero mean at the border, not just inside.
        MorletPlaneWave mean = hexagonal ? MorletPlaneWave{} : wave;
        if (hexagonal) {
            demodulate(ones, wave, width, height, waveReal, waveImaginary, threadPool);
            diffuse(waveReal, width, height, variance, threadPool, false, lattice);
            diffuse(waveImaginary, width, height, variance, threadPool, false, lattice);
        } else {
            // Separable, so its own blur is the product of a width-long and a height-long one and costs nothing against the 2-D passes.
            diffuse(mean.cosX, width, 1, variance, threadPool);
            diffuse(mean.sinX, width, 1, variance, threadPool);
            diffuse(mean.cosY, 1, height, variance, threadPool);
            diffuse(mean.sinY, 1, height, variance, threadPool);
        }
        demodulate(plane, wave, width, height, real, imaginary, threadPool);
        diffuse(real, width, height, variance, threadPool, false, lattice);
        diffuse(imaginary, width, height, variance, threadPool, false, lattice);
        // Remodulating is a rotation by the carrier's phase, which a magnitude discards, so only the admissibility term is left to apply.
        threadPool.parallelFor(height, [&](int y) {
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = row + static_cast<std::size_t>(x);
                const glm::vec2 centred =
                    hexagonal ? glm::vec2(real[pixel] - (waveReal[pixel] * lowpass[pixel]), imaginary[pixel] - (waveImaginary[pixel] * lowpass[pixel]))
                              : morletBaseband(mean, x, y, real[pixel], imaginary[pixel], lowpass[pixel]);
                // Squared, so the root is paid once per texel at the end rather than once per texel per orientation; max commutes with it.
                peak[pixel] = std::max(peak[pixel], (centred.x * centred.x) + (centred.y * centred.y));
            }
        });
    }

    threadPool.parallelFor(height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        for (int x = 0; x < beauty.width; ++x) {
            // One sideband is half a real grating; twice it is the analytic signal's envelope (Gabor 1946), the grating's amplitude.
            writeTexel(out, x, y, 2.0F * std::sqrt(peak[row + static_cast<std::size_t>(x + pad)]));
        }
    });
    return out;
}

HdrImage hsvAov(const HdrImage& beauty, ThreadPool& threadPool) {
    HdrImage out = makeAovImage(AovId::HSV, beauty.width, beauty.height);
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const glm::vec3 rgb = beauty.rgb(row + static_cast<std::size_t>(x));
            const float r = rgb.r;
            const float g = rgb.g;
            const float b = rgb.b;
            const float value = std::max({r, g, b});
            const float chroma = value - std::min({r, g, b});
            // Exact branches, not the shader's 1e-10 guard: hue is undefined on the achromatic axis, saturation on black (Smith 1978).
            const float saturation = value > 0.0F ? chroma / value : 0.0F;
            float hue = 0.0F;
            if (chroma > 0.0F) {
                if (value == r) {
                    hue = (g - b) / chroma;
                } else if (value == g) {
                    hue = 2.0F + ((b - r) / chroma);
                } else {
                    hue = 4.0F + ((r - g) / chroma);
                }
                // Sextant index to a [0,1) turn, wrapping the negative sixth the red sector produces.
                hue /= 6.0F;
                if (hue < 0.0F) {
                    hue += 1.0F;
                }
            }
            writeTexel(out, x, y, glm::vec3(hue, saturation, value));
        }
    });
    return out;
}

HdrImage colourOpponentAov(const HdrImage& beauty, ThreadPool& threadPool) {
    const pathtracer::scene::cone::OpponentBasis& basis = pathtracer::scene::cone::opponentBasis();
    HdrImage out = makeAovImage(AovId::ColourOpponent, beauty.width, beauty.height);
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const glm::vec3 rgb = beauty.rgb(row + static_cast<std::size_t>(x));
            const float red = rgb.r;
            const float green = rgb.g;
            const float blue = rgb.b;
            const glm::vec2 difference(red - green, blue - green);
            const float sum = (basis.denominator.r * red) + (basis.denominator.g * green) + (basis.denominator.b * blue);
            // Zero where no light arrives: the cone chromaticity of black is undefined, not achromatic, and must not read as white.
            const float scale = sum > 0.0F ? 1.0F / sum : 0.0F;
            writeTexel(out, x, y,
                       glm::vec2(glm::dot(basis.redGreenNumerator, difference), glm::dot(basis.blueYellowNumerator, difference)) *
                           scale);
        }
    });
    return out;
}

double bartenPupilDiameter(double luminance, double fieldDegrees) {
    // Barten 1999: the field's luminous size L X0^2 against his 40-degree reference field sets the light-adapted pupil.
    return 5.0 - (3.0 * std::tanh(0.4 * std::log10((luminance * fieldDegrees * fieldDegrees) / (40.0 * 40.0))));
}

BartenObserver bartenObserver(double luminance, double fieldDegrees) {
    const double pupil = bartenPupilDiameter(luminance, fieldDegrees);
    const double spread = kBartenAberrationDegreesPerMm * pupil;
    // Stiles-Crawford: off-axis rays through a wide pupil are less effective, 1 - (d/9.7)^2 + (d/12.4)^4 (Barten 1999).
    const double stilesCrawford = 1.0 - std::pow(pupil / 9.7, 2.0) + std::pow(pupil / 12.4, 4.0);
    return {(kBartenSigma0Degrees * kBartenSigma0Degrees) + (spread * spread),
            (std::numbers::pi * pupil * pupil / 4.0) * stilesCrawford,
            (1.0 / (fieldDegrees * fieldDegrees)) + (1.0 / (kBartenMaxFieldDegrees * kBartenMaxFieldDegrees))};
}

double bartenSensitivity(const BartenObserver& observer, double cyclesPerDegree, double luminance) {
    const double u = cyclesPerDegree;
    const double opticalMtf = std::exp(-2.0 * std::numbers::pi * std::numbers::pi * observer.sigmaSquared * u * u);
    const double integrationArea = observer.fieldTerm + ((u * u) / (kBartenMaxCycles * kBartenMaxCycles));
    // M_lat^2 = 1 - exp(-(u/u0)^2), by expm1 so a low frequency keeps its digits.
    const double inhibition = -std::expm1(-(u * u) / (kBartenInhibitionCyclesPerDegree * kBartenInhibitionCyclesPerDegree));
    const double photons = kBartenQuantumEfficiency * kBartenPhotonsPerTroland * observer.trolandsPerLuminance * luminance;
    // Barten's 1/(eta p E) + Phi0/M_lat^2 noise sum multiplied through by eta p E M_lat^2, so E = 0 gives 0 with no division by zero.
    return (opticalMtf / kBartenSignalToNoise) *
           std::sqrt((photons * inhibition) /
                     ((2.0 / kBartenIntegrationSeconds) * integrationArea * (inhibition + (photons * kBartenNeuralNoise))));
}

double bartenSensitivityBound() {
    return std::sqrt((kBartenIntegrationSeconds * kBartenMaxFieldDegrees * kBartenMaxFieldDegrees) / (2.0 * kBartenNeuralNoise)) /
           kBartenSignalToNoise;
}

std::vector<float> texelAngularPitch(const pathtracer::scene::Camera& camera, int width, int height, ThreadPool& threadPool) {
    const PixelLattice lattice = camera.pixelLattice();
    const glm::vec2 frame = pathtracer::gfx::frameExtent(lattice, width, height);
    const pathtracer::scene::Camera::ViewBasis basis = camera.viewBasis(frame.x / frame.y);
    // d(ndc) per column pitch along each film axis: two NDC units span the film, and +Y flips because row 0 is the top.
    const glm::mat2 ndcPerPitch(2.0F / frame.x, 0.0F, 0.0F, -2.0F / frame.y);
    std::vector<float> pitch(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    threadPool.parallelFor(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            const glm::vec2 centre = pathtracer::gfx::pixelCentre(lattice, x, y);
            const glm::vec2 ndc(((centre.x / frame.x) * 2.0F) - 1.0F, 1.0F - ((centre.y / frame.y) * 2.0F));
            const std::optional<pathtracer::scene::Camera::RayDifferential> ray = camera.primaryRayDifferential(basis, ndc.x, ndc.y);
            const glm::mat2x3 dirPerPitch = ray ? ray->dirPerNdc * ndcPerPitch : glm::mat2x3(0.0F);
            pitch[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x)] =
                std::sqrt(glm::length(glm::cross(dirPerPitch[0], dirPerPitch[1])));
        }
    });
    return pitch;
}

namespace {

// The observer's field is the frame: its solid angle and mean luminance, each texel weighted by its cell's pitch^2 rowPitch sr.
[[nodiscard]] std::optional<BartenObserver> frameObserver(const std::vector<float>& luminance, std::span<const float> angularPitch,
                                                          PixelLattice lattice) {
    double solidAngle = 0.0;
    double weightedLuminance = 0.0;
    for (std::size_t pixel = 0; pixel < luminance.size(); ++pixel) {
        const double cell = static_cast<double>(angularPitch[pixel]) * static_cast<double>(angularPitch[pixel]);
        solidAngle += cell;
        weightedLuminance += cell * static_cast<double>(luminance[pixel]);
    }
    // A black field adapts no pupil and shows no detail.
    if (!(weightedLuminance > 0.0)) {
        return std::nullopt;
    }
    // The square of equal solid angle, exact for the pupil, which reads only the area X0 Y0; rowPitch cancels from the mean.
    const double fieldDegrees = std::sqrt(solidAngle * static_cast<double>(pathtracer::gfx::rowPitch(lattice))) * (180.0 / std::numbers::pi);
    return bartenObserver(weightedLuminance / solidAngle, fieldDegrees);
}

// One octave G(t_f) - G(t_c): the row-axis frequency its discrete transfer peaks at, and that peak gain.
struct OctaveBand {
    double cyclesPerPitch = 0.0;
    double gain = 0.0;
};

[[nodiscard]] OctaveBand octaveBand(double fineVariance, double coarseVariance, PixelLattice lattice) {
    const double boundary = rowAxisBoundary(lattice);
    const double peakSymbol = rowAxisSymbol(boundary, lattice);
    // H = exp(-t_f a) - exp(-t_c a) peaks at a = ln(t_c/t_f)/(t_c - t_f); the highpass (t_f = 0) and any later peak clamp to the boundary.
    const double symbol =
        fineVariance > 0.0 ? std::min(std::log(coarseVariance / fineVariance) / (coarseVariance - fineVariance), peakSymbol) : peakSymbol;
    const double omega = symbol < peakSymbol ? rowAxisFrequency(symbol, lattice) : boundary;
    // The peak gain, so a grating at the band's frequency reads its own amplitude.
    return {omega / (2.0 * std::numbers::pi), std::exp(-fineVariance * symbol) - std::exp(-coarseVariance * symbol)};
}

// Raises each texel's visibility to this band's RMS contrast `energy` times Barten's sensitivity at the texel's own luminance.
void foldBandVisibility(const std::vector<float>& luminance, const std::vector<float>& energy, std::span<const float> angularPitch,
                        const OctaveBand& band, const BartenObserver& observer, int width, int height, std::vector<float>& visibility,
                        ThreadPool& threadPool) {
    threadPool.parallelFor(height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            // Photoreceptors adapt to the light reaching them, not to a lowpass tail; at threshold the two agree to first order in m.
            const auto adaptation = static_cast<double>(luminance[pixel]);
            const double pitchDegrees = static_cast<double>(angularPitch[pixel]) * (180.0 / std::numbers::pi);
            // No ray, or no light to adapt to: nothing there is visible.
            if (!(pitchDegrees > 0.0) || !(adaptation > 0.0)) {
                continue;
            }
            // Centre-relative accumulation can round a vanishing mean power a few ulp below zero.
            const double contrast = std::sqrt(2.0 * std::max(static_cast<double>(energy[pixel]), 0.0)) / band.gain;
            const double sensitivity = bartenSensitivity(observer, band.cyclesPerPitch / pitchDegrees, adaptation);
            visibility[pixel] = std::max(visibility[pixel], static_cast<float>(contrast * sensitivity));
        }
    });
}

// Contrast Sensitivity at the angular pitch of the camera Beauty was traced through.
[[nodiscard]] HdrImage tracedContrastSensitivity(const FilterInput& input, ThreadPool& threadPool) {
    if (input.camera == nullptr) {
        throw std::invalid_argument("contrastSensitivity: needs the camera Beauty was traced through");
    }
    return contrastSensitivityAov(input.beauty, texelAngularPitch(*input.camera, input.beauty.width, input.beauty.height, threadPool),
                                  threadPool, input.wrapsHorizontally, input.lattice);
}

}  // namespace

HdrImage contrastSensitivityAov(const HdrImage& beauty, std::span<const float> angularPitch, ThreadPool& threadPool,
                                bool wrapsHorizontally, PixelLattice lattice) {
    const int width = beauty.width;
    const int height = beauty.height;
    HdrImage out = makeAovImage(AovId::ContrastSensitivity, width, height);
    const std::vector<float> luminance = luminancePlane(beauty, threadPool);
    const std::optional<BartenObserver> observer = frameObserver(luminance, angularPitch, lattice);
    if (!observer) {
        return out;
    }
    // Octaves t_k = t0 4^k under a highpass band from t = 0, t0 placing band 0's peak at half the row axis' zone boundary.
    double fineVariance = 0.0;
    double coarseVariance = std::log(4.0) / (3.0 * rowAxisSymbol(0.5 * rowAxisBoundary(lattice), lattice));
    std::vector<float> fine = luminance;
    std::vector<float> coarse(fine.size());
    std::vector<float> energy(fine.size());
    std::vector<float> visibility(fine.size(), 0.0F);
    // A band whose window outreaches the frame would measure the mirror, not the image; a periodic width has no mirror.
    for (glm::ivec2 reach = diffusionReach(static_cast<float>(coarseVariance), lattice);
         (2 * reach.y) + 1 <= height && (wrapsHorizontally || (2 * reach.x) + 1 <= width);
         reach = diffusionReach(static_cast<float>(coarseVariance), lattice)) {
        coarse = fine;
        diffuse(coarse, width, height, static_cast<float>(coarseVariance - fineVariance), threadPool, wrapsHorizontally, lattice);
        threadPool.parallelFor(height, [&](int y) {
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixel = row + static_cast<std::size_t>(x);
                // Peli's pointwise contrast b / l, bounded where both fade together; undefined with no light to adapt to, so zero.
                const float contrast = coarse[pixel] > 0.0F ? (fine[pixel] - coarse[pixel]) / coarse[pixel] : 0.0F;
                energy[pixel] = contrast * contrast;
            }
        });
        // Its power over the band's window: sqrt(2 mean c^2) of a grating is its amplitude contrast, with no zero-crossing ripple.
        diffuse(energy, width, height, static_cast<float>(coarseVariance), threadPool, wrapsHorizontally, lattice);
        foldBandVisibility(luminance, energy, angularPitch, octaveBand(fineVariance, coarseVariance, lattice), *observer, width, height,
                           visibility, threadPool);
        fine.swap(coarse);
        fineVariance = coarseVariance;
        coarseVariance *= 4.0;
    }
    threadPool.parallelFor(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            writeTexel(out, x, y, visibility[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x)]);
        }
    });
    return out;
}

HdrImage snrAov(int width, int height, const float* replicateLuminance, int replicates, int samples, ThreadPool& threadPool) {
    HdrImage out = makeAovImage(AovId::SNR, width, height);
    // Replicates not yet drawn hold no estimate; fewer than two populated carry no spread, so the ratio is undefined and reads zero.
    const int populated = std::min(samples, replicates);
    if (replicateLuminance == nullptr || populated < 2) {
        return out;
    }
    const auto stride = static_cast<std::size_t>(replicates);
    threadPool.parallelFor(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
            // The adaptive stopping rule's own estimate, so the AOV shows exactly the error the sampler retires texels by.
            const pathtracer::scene::ReplicateEstimate estimate =
                pathtracer::scene::replicateEstimate(replicateLuminance + (pixel * stride), populated);
            const double standardError = std::sqrt(estimate.meanVariance);
            // No signal reads zero; a positive signal every replicate agreed on exactly is known exactly, so its SNR is +inf, not zero.
            const double ratio = estimate.mean > 0.0 ? estimate.mean / standardError : 0.0;
            writeTexel(out, x, y, static_cast<float>(ratio));
        }
    });
    return out;
}

HdrImage sampleCountAov(int width, int height, const int* texelSamples, ThreadPool& threadPool) {
    HdrImage out = makeAovImage(AovId::SampleCount, width, height);
    if (texelSamples == nullptr) {
        return out;
    }
    threadPool.parallelFor(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
            writeTexel(out, x, y, static_cast<float>(texelSamples[pixel]));
        }
    });
    return out;
}

HdrImage shadowAov(const HdrImage& lost, const HdrImage& unoccluded, ThreadPool& threadPool) {
    HdrImage out = makeAovImage(AovId::Shadow, lost.width, lost.height);
    threadPool.parallelFor(lost.height, [&](int y) {
        for (int x = 0; x < lost.width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(lost.width)) + static_cast<std::size_t>(x);
            // No direct light to lose means no shadow, so zero rather than an undefined 0/0.
            const float total = unoccluded.texels[pixel];
            writeTexel(out, x, y, total > 0.0F ? lost.texels[pixel] / total : 0.0F);
        }
    });
    return out;
}

HdrImage evaluateFilterAov(AovId aov, const FilterInput& input, ThreadPool& threadPool) {
    switch (aov) {
        case AovId::HSV:       return hsvAov(input.beauty, threadPool);
        case AovId::Luminance: return luminanceAov(input.beauty, threadPool);
        case AovId::Sobel:     return sobelAov(input.beauty, threadPool, input.wrapsHorizontally, input.lattice);
        case AovId::Gabor:     return gaborAov(input.beauty, threadPool, input.wrapsHorizontally, input.lattice);
        case AovId::ColourOpponent:  return colourOpponentAov(input.beauty, threadPool);
        case AovId::ContrastSensitivity: return tracedContrastSensitivity(input, threadPool);
        case AovId::SNR:
            return snrAov(input.beauty.width, input.beauty.height, input.replicateLuminance, input.replicates, input.samples, threadPool);
        case AovId::SampleCount:
            return sampleCountAov(input.beauty.width, input.beauty.height, input.texelSamples, threadPool);
        case AovId::Denoised:
            if (input.denoiseAlbedo == nullptr || input.denoiseNormal == nullptr || input.denoiser == nullptr) {
                throw std::invalid_argument("denoised: needs the denoiseAlbedo and denoiseNormal lanes and a Denoiser");
            }
            return input.denoiser->denoise(input.beauty, *input.denoiseAlbedo, *input.denoiseNormal);
        case AovId::Shadow:
            if (input.shadowLost == nullptr || input.shadowUnoccluded == nullptr) {
                throw std::invalid_argument("shadow: needs the shadowLost and shadowUnoccluded lanes");
            }
            return shadowAov(*input.shadowLost, *input.shadowUnoccluded, threadPool);

        // The lanes their own producers write. No default arm: -Werror then makes an unrouted new filter a compile error.
        case AovId::Beauty:
        case AovId::Wireframe:
        case AovId::Alpha:
        case AovId::Depth:
        case AovId::Lookahead:
        case AovId::WorldPos:
        case AovId::UV:
        case AovId::MotionVector:
        case AovId::Normal:
        case AovId::GeomNormal:
        case AovId::Albedo:
        case AovId::Metallic:
        case AovId::Roughness:
        case AovId::Tangent:
        case AovId::ObjectID:
        case AovId::AO:
        case AovId::Fresnel:
        case AovId::IOR:
        case AovId::BounceCount:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::DirectVolume:
        case AovId::IndirectVolume:
        case AovId::DirectSubsurface:
        case AovId::IndirectSubsurface:
        case AovId::Count:
            break;
    }
    return {};
}

}  // namespace pathtracer::debug
