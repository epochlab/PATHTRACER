#include "pathtracer/debug/aov_filters.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "pathtracer/debug/scale_space.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/cone_space.h"

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
    // The finest scale the grid resolves, the same fine scale DoG starts from.
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

HdrImage dogAov(const HdrImage& beauty, ThreadPool& threadPool, bool wrapsHorizontally, PixelLattice lattice) {
    HdrImage out = makeAovImage(AovId::DoG, beauty.width, beauty.height);
    const float fineVariance = innerScaleVariance(lattice);
    // One octave up, sigma doubled: variance 4t, reached from the fine level by the semigroup's step 3t.
    const float coarseStep = (4.0F * fineVariance) - fineVariance;
    // A frame narrower than the coarse step's kernel support would report the mirror, not the image; a periodic width has no mirror.
    const glm::ivec2 reach = diffusionReach(coarseStep, lattice);
    if ((2 * reach.y) + 1 > beauty.height || (!wrapsHorizontally && (2 * reach.x) + 1 > beauty.width)) {
        return out;
    }
    std::vector<float> fine = luminancePlane(beauty, threadPool);
    diffuse(fine, beauty.width, beauty.height, fineVariance, threadPool, wrapsHorizontally, lattice);
    std::vector<float> coarse = fine;
    diffuse(coarse, beauty.width, beauty.height, coarseStep, threadPool, wrapsHorizontally, lattice);
    threadPool.parallelFor(beauty.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(beauty.width);
        for (int x = 0; x < beauty.width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            // Signed, not a magnitude: polarity separates a bright blob from a dark one, and Sobel already reports gradient magnitude.
            writeTexel(out, x, y, fine[pixel] - coarse[pixel]);
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

HdrImage snrAov(int width, int height, const float* replicateLuminance, int replicates, int samples, ThreadPool& threadPool) {
    HdrImage out = makeAovImage(AovId::SNR, width, height);
    // Replicates not yet drawn hold no estimate; fewer than two populated carry no spread, so the ratio is undefined and reads zero.
    const int populated = std::min(samples, replicates);
    if (replicateLuminance == nullptr || populated < 2) {
        return out;
    }
    // Independent replicate means m_r of one expectation: sum (m_r - mbar)^2 / (K(K-1)) is unbiased for Var(mbar) at any counts.
    const double inverseDegrees = 1.0 / (static_cast<double>(populated) * static_cast<double>(populated - 1));
    const auto stride = static_cast<std::size_t>(replicates);
    threadPool.parallelFor(height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
            const float* means = replicateLuminance + (pixel * stride);
            // Double: the deviations of a converged texel's replicates are rounding-scale differences of float means.
            const double mean = std::accumulate(means, means + populated, 0.0) / static_cast<double>(populated);
            double squares = 0.0;
            for (int r = 0; r < populated; ++r) {
                squares += (means[r] - mean) * (means[r] - mean);
            }
            const double standardError = std::sqrt(squares * inverseDegrees);
            // No signal reads zero; a positive signal every replicate agreed on exactly is known exactly, so its SNR is +inf, not zero.
            const double ratio = mean > 0.0 ? mean / standardError : 0.0;
            writeTexel(out, x, y, static_cast<float>(ratio));
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
        case AovId::DoG:       return dogAov(input.beauty, threadPool, input.wrapsHorizontally, input.lattice);
        case AovId::ColourOpponent:  return colourOpponentAov(input.beauty, threadPool);
        case AovId::SNR:
            return snrAov(input.beauty.width, input.beauty.height, input.replicateLuminance, input.replicates, input.samples, threadPool);
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
