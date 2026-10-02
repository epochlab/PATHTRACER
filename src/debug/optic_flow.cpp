#include "pathtracer/debug/optic_flow.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/debug/scale_space.h"

namespace pathtracer::debug {

namespace {

using pathtracer::gfx::HdrImage;
using pathtracer::scene::ThreadPool;

// Envelope variance of octave k in base pixels squared, exact: a power-of-two multiple of the inner scale.
[[nodiscard]] float levelVariance(int level) {
    return std::ldexp(innerScaleVariance(), 2 * level);
}

// pi/omega_k, half the carrier wavelength at octave k: the largest displacement whose phase difference stays inside (-pi, pi].
[[nodiscard]] double unambiguousRange(int level) {
    return std::numbers::pi / static_cast<double>(morletCarrier(levelVariance(level)));
}

// One channel of an interleaved image as its own plane, the unit the scale space and the bank operate on.
[[nodiscard]] std::vector<float> channelPlane(const HdrImage& image, int channel, ThreadPool& threadPool) {
    std::vector<float> plane(static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height));
    threadPool.parallelFor(image.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width);
        for (int x = 0; x < image.width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            plane[pixel] = image.texels[(pixel * static_cast<std::size_t>(image.channels)) + static_cast<std::size_t>(channel)];
        }
    });
    return plane;
}

// A view's per-channel variance planes, or none for a deterministic source, whose noise is only rounding.
[[nodiscard]] std::vector<std::vector<float>> channelVariances(const OpticFlowView& view, ThreadPool& threadPool) {
    std::vector<std::vector<float>> planes;
    if (view.variance != nullptr) {
        for (int channel = 0; channel < view.variance->channels; ++channel) {
            planes.push_back(channelPlane(*view.variance, channel, threadPool));
        }
    }
    return planes;
}

// Every channel of an image as planes: each is a separate observation of the same motion.
[[nodiscard]] std::vector<std::vector<float>> channelPlanes(const HdrImage& image, ThreadPool& threadPool) {
    std::vector<std::vector<float>> planes;
    for (int channel = 0; channel < image.channels; ++channel) {
        planes.push_back(channelPlane(image, channel, threadPool));
    }
    return planes;
}

// The bank at one octave: each orientation's base-grid plane wave, to demodulate with, and its cascaded tables on the level grid.
struct LevelBank {
    std::vector<MorletPlaneWave> waves;
    std::vector<MorletPlaneWave> blurred;
};

[[nodiscard]] LevelBank levelBank(int level, int width, int height, ThreadPool& threadPool) {
    const float carrier = morletCarrier(levelVariance(level));
    const int orientations = morletOrientations();
    LevelBank bank;
    for (int orientation = 0; orientation < orientations; ++orientation) {
        MorletPlaneWave wave = morletPlaneWave(carrier, orientation, orientations, width, height);
        MorletPlaneWave blurred;
        // Separable, so the plane wave cascades as one row and one column; diffusion is the identity along a unit-length axis.
        blurred.cosX = octaveLevel(wave.cosX, width, 1, level, threadPool).plane;
        blurred.sinX = octaveLevel(wave.sinX, width, 1, level, threadPool).plane;
        blurred.cosY = octaveLevel(wave.cosY, 1, height, level, threadPool).plane;
        blurred.sinY = octaveLevel(wave.sinY, 1, height, level, threadPool).plane;
        bank.waves.push_back(std::move(wave));
        bank.blurred.push_back(std::move(blurred));
    }
    return bank;
}

// A complex plane on one level's grid.
struct Baseband {
    std::vector<float> real;
    std::vector<float> imaginary;
};

// Admissible Morlet baseband of one plane at one orientation and octave: demodulated on the base grid, then the pair is cascaded.
[[nodiscard]] Baseband baseband(std::span<const float> plane, int width, int height, const ScaleSpaceLevel& lowpass,
                                const MorletPlaneWave& wave, const MorletPlaneWave& blurred, int level, ThreadPool& threadPool) {
    std::vector<float> real(plane.size());
    std::vector<float> imaginary(plane.size());
    demodulate(plane, wave, width, height, real, imaginary, threadPool);
    Baseband out{octaveLevel(real, width, height, level, threadPool).plane,
                 octaveLevel(imaginary, width, height, level, threadPool).plane};
    threadPool.parallelFor(lowpass.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(lowpass.width);
        for (int x = 0; x < lowpass.width; ++x) {
            const std::size_t pixel = row + static_cast<std::size_t>(x);
            const glm::vec2 centred = morletBaseband(blurred, x, y, out.real[pixel], out.imaginary[pixel], lowpass.plane[pixel]);
            out.real[pixel] = centred.x;
            out.imaginary[pixel] = centred.y;
        }
    });
    return out;
}

// Phase-constraint normal equations of one source channel per level texel, in double: A = sum w g g^T, b = -sum w g dphi.
struct ChannelSums {
    explicit ChannelSums(std::size_t texels)
        : axx(texels), axy(texels), ayy(texels), bx(texels), by(texels), phaseEnergy(texels), constraints(texels) {}

    void clear(std::size_t pixel) {
        axx[pixel] = axy[pixel] = ayy[pixel] = bx[pixel] = by[pixel] = phaseEnergy[pixel] = 0.0;
        constraints[pixel] = 0;
    }

    std::vector<double> axx, axy, ayy, bx, by;
    std::vector<double> phaseEnergy;  // sum w dphi^2, the residual before any fit
    std::vector<int> constraints;  // orientations with w > 0
};

// a * conj(b) with each product its own statement: no contraction can fuse them, so a * conj(a) has an exactly zero imaginary part.
[[nodiscard]] glm::vec2 timesConjugate(glm::vec2 a, glm::vec2 b) {
    const float realProduct = a.x * b.x;
    const float imaginaryProduct = a.y * b.y;
    const float crossAB = a.y * b.x;
    const float crossBA = a.x * b.y;
    return {realProduct + imaginaryProduct, crossAB - crossBA};
}

// Texel (x, y) of a baseband on a grid `width` wide.
[[nodiscard]] glm::vec2 sampleAt(const Baseband& band, int width, int x, int y) {
    const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
    return {band.real[pixel], band.imaginary[pixel]};
}

// Integer displacement of each level texel, in level samples: the previous frame's response is read at x - r.
struct LevelWarp {
    std::vector<int> x;
    std::vector<int> y;
};

// C(p + forward) * conj(C(p - backward)): one frame's neighbour phasor about p, whose argument is the phase advance across the pair.
[[nodiscard]] glm::vec2 neighbourPhasor(const Baseband& band, int width, glm::ivec2 p, glm::ivec2 backward, glm::ivec2 forward) {
    const glm::ivec2 ahead = p + forward;
    const glm::ivec2 behind = p - backward;
    return timesConjugate(sampleAt(band, width, ahead.x, ahead.y), sampleAt(band, width, behind.x, behind.y));
}

// Adds one orientation's constraint g . delta = -dphi at every level texel, current(x) against previous(x - r).
void accumulateOrientation(const Baseband& previous, const Baseband& current, const std::vector<float>& previousNoise,
                           const std::vector<float>& currentNoise, const MorletPlaneWave& wave, const ScaleSpaceLevel& grid,
                           const LevelWarp& warp, const std::vector<std::uint8_t>& stale, ChannelSums& sums, ThreadPool& threadPool) {
    const auto pitch = static_cast<double>(grid.decimation);
    const auto inside = [&grid](glm::ivec2 p) { return p.x >= 0 && p.y >= 0 && p.x < grid.width && p.y < grid.height; };
    threadPool.parallelFor(grid.height, [&](int y) {
        for (int x = 0; x < grid.width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(grid.width)) + static_cast<std::size_t>(x);
            if (stale[pixel] == 0) {
                continue;
            }
            const glm::ivec2 here(x, y);
            const glm::ivec2 there = here - glm::ivec2(warp.x[pixel], warp.y[pixel]);
            // One step either side where the current frame has the neighbour: the slope is read from it alone, the grid flow lives on.
            const glm::ivec2 backward(inside(here - glm::ivec2(1, 0)) ? 1 : 0, inside(here - glm::ivec2(0, 1)) ? 1 : 0);
            const glm::ivec2 forward(inside(here + glm::ivec2(1, 0)) ? 1 : 0, inside(here + glm::ivec2(0, 1)) ? 1 : 0);
            // Displaced off the grid the previous frame holds no observation here, and none is made up.
            if (!inside(there)) {
                continue;
            }
            const glm::vec2 before = sampleAt(previous, grid.width, there.x, there.y);
            const glm::vec2 after = sampleAt(current, grid.width, x, y);
            const std::size_t source =
                (static_cast<std::size_t>(there.y) * static_cast<std::size_t>(grid.width)) + static_cast<std::size_t>(there.x);
            const auto noiseBefore = static_cast<double>(previousNoise[source]);
            const auto noiseAfter = static_cast<double>(currentNoise[pixel]);
            // Rice: |C|^2 - v is the unbiased signal power; at or below zero the phase is noise's, uniform, and carries nothing.
            const double signalBefore = static_cast<double>(timesConjugate(before, before).x) - noiseBefore;
            const double signalAfter = static_cast<double>(timesConjugate(after, after).x) - noiseAfter;
            if (!(signalBefore > 0.0) || !(signalAfter > 0.0)) {
                continue;
            }
            // Phase variance v / 2P per frame at the current frame's power, which a true match shares: warp-independent information.
            const double weight = 1.0 / ((noiseBefore + noiseAfter) / (2.0 * signalAfter));
            const glm::vec2 rotation = timesConjugate(after, before);
            // The baseband omits the carrier, so previous(x - r) lacks the rotation omega u . r that the full response carries.
            const double carrierShift = ((wave.stepX * static_cast<double>(warp.x[pixel])) +
                                         (wave.stepY * static_cast<double>(warp.y[pixel]))) * pitch;
            const double phaseStep =
                std::remainder(static_cast<double>(std::atan2(rotation.y, rotation.x)) + carrierShift, 2.0 * std::numbers::pi);
            const glm::ivec2 alongX(1, 0);
            const glm::ivec2 alongY(0, 1);
            const glm::vec2 slopeX = neighbourPhasor(current, grid.width, here, alongX * backward.x, alongX * forward.x);
            const glm::vec2 slopeY = neighbourPhasor(current, grid.width, here, alongY * backward.y, alongY * forward.y);
            // The carrier is analytic, so only the baseband's own slope is measured: phase(R) = omega u . x + phase(C).
            const double gx = wave.stepX + (static_cast<double>(std::atan2(slopeX.y, slopeX.x)) /
                                            (static_cast<double>(backward.x + forward.x) * pitch));
            const double gy = wave.stepY + (static_cast<double>(std::atan2(slopeY.y, slopeY.x)) /
                                            (static_cast<double>(backward.y + forward.y) * pitch));
            sums.axx[pixel] += weight * gx * gx;
            sums.axy[pixel] += weight * gx * gy;
            sums.ayy[pixel] += weight * gy * gy;
            sums.bx[pixel] -= weight * gx * phaseStep;
            sums.by[pixel] -= weight * gy * phaseStep;
            sums.phaseEnergy[pixel] += weight * phaseStep * phaseStep;
            ++sums.constraints[pixel];
        }
    });
}

// Eigen-decomposition of a symmetric 2x2 A: the eigenvalues and the largest one's unit vector; the other is its perpendicular.
struct SymmetricEigen {
    double largest = 0.0;
    double smallest = 0.0;
    double ux = 1.0;
    double uy = 0.0;
};

[[nodiscard]] SymmetricEigen symmetricEigen(double axx, double axy, double ayy) {
    SymmetricEigen eigen;
    const double half = 0.5 * (axx + ayy);
    const double root = std::hypot(0.5 * (axx - ayy), axy);
    eigen.largest = half + root;
    eigen.smallest = half - root;
    // (A - smallest I) has every column along the largest eigenvector, so the longer column gives it without cancellation.
    const double firstX = axx - eigen.smallest;
    const double secondY = ayy - eigen.smallest;
    const bool first = std::hypot(firstX, axy) >= std::hypot(axy, secondY);
    const double vx = first ? firstX : axy;
    const double vy = first ? axy : secondY;
    const double length = std::hypot(vx, vy);
    if (length > 0.0) {
        eigen.ux = vx / length;
        eigen.uy = vy / length;
    }
    return eigen;
}

// Square-root information at one texel (Bierman 1977): R upper triangular with R^T R the information, R d = z.
struct RootInformation {
    double r11 = 0.0;
    double r12 = 0.0;
    double r22 = 0.0;
    double z1 = 0.0;
    double z2 = 0.0;

    // Folds the observation row h . d = y in by two Givens rotations: orthogonal, so no information is formed by subtraction.
    void fold(double hx, double hy, double y) {
        const double first = std::hypot(r11, hx);
        const double c1 = r11 / first;
        const double s1 = hx / first;
        r11 = first;
        const double rotatedR12 = (c1 * r12) + (s1 * hy);
        const double remainingHy = (c1 * hy) - (s1 * r12);
        const double rotatedZ1 = (c1 * z1) + (s1 * y);
        const double remainingY = (c1 * y) - (s1 * z1);
        r12 = rotatedR12;
        z1 = rotatedZ1;
        const double second = std::hypot(r22, remainingHy);
        const double c2 = r22 / second;
        const double s2 = remainingHy / second;
        r22 = second;
        z2 = (c2 * z2) + (s2 * remainingY);
    }
};

// Gaussian belief over displacement in base pixels on one level grid: mean, and P = U U^T with U = [[a, b], [0, c]] upper triangular.
enum PosteriorField : std::size_t { kMeanX, kMeanY, kRootA, kRootB, kRootC, kPosteriorFields };
using Posterior = std::array<ScaleSpaceLevel, kPosteriorFields>;

// The prior's square-root information R = U^-1 and z = R m: upper triangular, so it seeds the rotation with no factorisation.
[[nodiscard]] RootInformation priorRoot(const Posterior& belief, std::size_t pixel) {
    const auto a = static_cast<double>(belief[kRootA].plane[pixel]);
    const auto b = static_cast<double>(belief[kRootB].plane[pixel]);
    const auto c = static_cast<double>(belief[kRootC].plane[pixel]);
    const auto mx = static_cast<double>(belief[kMeanX].plane[pixel]);
    const auto my = static_cast<double>(belief[kMeanY].plane[pixel]);
    RootInformation root;
    root.r11 = 1.0 / a;
    root.r12 = -b / (a * c);
    root.r22 = 1.0 / c;
    root.z1 = (root.r11 * mx) + (root.r12 * my);
    root.z2 = root.r22 * my;
    return root;
}

// The belief's mean in level samples, rounded: the integer displacement a measurement linearises about.
[[nodiscard]] LevelWarp levelWarp(const Posterior& belief) {
    const ScaleSpaceLevel& grid = belief[kMeanX];
    // A power of two, so dividing by it is exact and the rounding is the only approximation.
    const float inversePitch = 1.0F / static_cast<float>(grid.decimation);
    LevelWarp warp{std::vector<int>(grid.plane.size()), std::vector<int>(grid.plane.size())};
    for (std::size_t pixel = 0; pixel < grid.plane.size(); ++pixel) {
        warp.x[pixel] = static_cast<int>(std::lround(belief[kMeanX].plane[pixel] * inversePitch));
        warp.y[pixel] = static_cast<int>(std::lround(belief[kMeanY].plane[pixel] * inversePitch));
    }
    return warp;
}

// Envelope energy sum T(n)^2 on the base grid: per level column and row, the part landing inside the frame, and the whole axis's.
struct EnvelopeEnergy {
    std::vector<double> insideX;
    std::vector<double> insideY;
    double axis = 0.0;
};

// Each level sample sits on base pixel i * decimation; the kernel's taps beyond 0 or extent - 1 read the mirror, not the scene.
[[nodiscard]] std::vector<double> insideEnergy(const std::vector<float>& kernel, int samples, int decimation, int extent) {
    const auto radius = static_cast<int>(kernel.size()) - 1;
    std::vector<double> energy(static_cast<std::size_t>(samples), 0.0);
    for (int i = 0; i < samples; ++i) {
        for (int n = -radius; n <= radius; ++n) {
            const int base = (i * decimation) + n;
            if (base >= 0 && base < extent) {
                const auto tap = static_cast<double>(kernel[static_cast<std::size_t>(std::abs(n))]);
                energy[static_cast<std::size_t>(i)] += tap * tap;
            }
        }
    }
    return energy;
}

[[nodiscard]] EnvelopeEnergy envelopeEnergy(const ScaleSpaceLevel& grid, int width, int height) {
    // The cascade is the base-grid discrete Gaussian of the level's variance to within the decimated band, so its energy is that kernel's.
    const std::vector<float> kernel = discreteGaussianKernel(grid.baseVariance);
    EnvelopeEnergy energy{insideEnergy(kernel, grid.width, grid.decimation, width),
                          insideEnergy(kernel, grid.height, grid.decimation, height), 0.0};
    for (std::size_t n = 0; n < kernel.size(); ++n) {
        energy.axis += (n == 0 ? 1.0 : 2.0) * static_cast<double>(kernel[n]) * static_cast<double>(kernel[n]);
    }
    return energy;
}

// One channel's least-squares fit at a texel: its numerical rank, the retained eigen-directions, and the chi-square it leaves.
struct ChannelFit {
    SymmetricEigen eigen;
    int rank = 0;
    int freedom = 0;
    double residual = 0.0;
};

[[nodiscard]] ChannelFit channelFit(const ChannelSums& sums, std::size_t pixel) {
    ChannelFit fit;
    fit.eigen = symmetricEigen(sums.axx[pixel], sums.axy[pixel], sums.ayy[pixel]);
    // Numerical rank: an eigenvalue under 2^-24 of the largest carries no information a float constraint could have encoded.
    fit.rank = fit.eigen.largest > 0.0 ? (fit.eigen.smallest > kFloat32Roundoff * fit.eigen.largest ? 2 : 1) : 0;
    fit.freedom = std::max(sums.constraints[pixel] - fit.rank, 0);
    const std::array<double, 2> values{fit.eigen.largest, fit.eigen.smallest};
    const std::array<glm::dvec2, 2> vectors{glm::dvec2(fit.eigen.ux, fit.eigen.uy), glm::dvec2(-fit.eigen.uy, fit.eigen.ux)};
    const glm::dvec2 b(sums.bx[pixel], sums.by[pixel]);
    double explained = 0.0;
    for (int i = 0; i < fit.rank; ++i) {
        explained += glm::dot(vectors[i], b) * glm::dot(vectors[i], b) / values[i];
    }
    fit.residual = std::max(sums.phaseEnergy[pixel] - explained, 0.0);
    return fit;
}

// Every texel's fit, computed once: the pooled dispersion and the fold both read it.
[[nodiscard]] std::vector<ChannelFit> channelFits(const ChannelSums& sums, const ScaleSpaceLevel& grid, ThreadPool& threadPool) {
    std::vector<ChannelFit> fits(grid.plane.size());
    threadPool.parallelFor(grid.height, [&](int y) {
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(grid.width);
        for (std::size_t pixel = row; pixel < row + static_cast<std::size_t>(grid.width); ++pixel) {
            fits[pixel] = channelFit(sums, pixel);
        }
    });
    return fits;
}

// Overdispersion pooled over the octave's envelope (McCullagh & Nelder 1989): a texel's own few constraints cannot measure their misfit.
[[nodiscard]] std::vector<float> pooledDispersion(const std::vector<ChannelFit>& fits, const ScaleSpaceLevel& grid,
                                                  ThreadPool& threadPool) {
    std::vector<float> chiSquare(fits.size());
    std::vector<float> freedom(fits.size());
    std::transform(fits.begin(), fits.end(), chiSquare.begin(), [](const ChannelFit& fit) { return static_cast<float>(fit.residual); });
    std::transform(fits.begin(), fits.end(), freedom.begin(), [](const ChannelFit& fit) { return static_cast<float>(fit.freedom); });
    const float ownVariance = grid.baseVariance / static_cast<float>(grid.decimation * grid.decimation);
    diffuse(chiSquare, grid.width, grid.height, ownVariance, threadPool);
    diffuse(freedom, grid.width, grid.height, ownVariance, threadPool);
    std::vector<float> dispersion(fits.size());
    // Weights are absolute, so chi-square over its freedom is 1 when the model fits; above it (occlusion, noise phase) only widens.
    std::transform(chiSquare.begin(), chiSquare.end(), freedom.begin(), dispersion.begin(),
                   [](float chi, float dof) { return dof > 0.0F ? std::max(chi / dof, 1.0F) : 1.0F; });
    return dispersion;
}

// Folds one channel's normal equations A (d - r) = b, compressed to its retained eigen-rows and widened by the pooled dispersion.
void foldChannel(const ChannelSums& sums, const ScaleSpaceLevel& grid, const LevelWarp& warp, std::vector<RootInformation>& roots,
                 ThreadPool& threadPool) {
    const std::vector<ChannelFit> fits = channelFits(sums, grid, threadPool);
    const std::vector<float> dispersion = pooledDispersion(fits, grid, threadPool);
    threadPool.parallelFor(grid.height, [&](int y) {
        for (int x = 0; x < grid.width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(grid.width)) + static_cast<std::size_t>(x);
            const ChannelFit& fit = fits[pixel];
            const std::array<double, 2> values{fit.eigen.largest, fit.eigen.smallest};
            const std::array<glm::dvec2, 2> vectors{glm::dvec2(fit.eigen.ux, fit.eigen.uy), glm::dvec2(-fit.eigen.uy, fit.eigen.ux)};
            const glm::dvec2 b(sums.bx[pixel], sums.by[pixel]);
            const double noise = std::sqrt(static_cast<double>(dispersion[pixel]));
            // The displacement the previous response was read at, in base pixels.
            const glm::dvec2 shift = glm::dvec2(warp.x[pixel], warp.y[pixel]) * static_cast<double>(grid.decimation);
            for (int i = 0; i < fit.rank; ++i) {
                const double scale = std::sqrt(values[i]);
                const glm::dvec2 row = vectors[i] * (scale / noise);
                roots[pixel].fold(row.x, row.y, ((glm::dot(vectors[i], b) / scale) + (scale * glm::dot(vectors[i], shift))) / noise);
            }
        }
    });
}

// Zero mean at the coarsest octave's unambiguous range: the slow-speed prior (Weiss et al. 2002) with no authored width.
[[nodiscard]] Posterior flatPrior(const ScaleSpaceLevel& grid, int level) {
    const auto range = static_cast<float>(unambiguousRange(level));
    const std::vector<float> zero(grid.plane.size(), 0.0F);
    const std::vector<float> spread(grid.plane.size(), range);
    const auto field = [&grid](const std::vector<float>& plane) {
        return ScaleSpaceLevel{plane, grid.width, grid.height, grid.decimation, grid.baseVariance};
    };
    return {field(zero), field(zero), field(spread), field(zero), field(spread)};
}

// The belief resampled onto a finer grid; the covariance factor interpolates, so U U^T stays positive definite by construction.
[[nodiscard]] Posterior prolong(const Posterior& coarse, const ScaleSpaceLevel& grid, ThreadPool& threadPool) {
    Posterior fine;
    for (std::size_t field = 0; field < kPosteriorFields; ++field) {
        fine[field] = ScaleSpaceLevel{std::vector<float>(grid.plane.size(), 0.0F), grid.width, grid.height, grid.decimation,
                                      grid.baseVariance};
        addExpanded(coarse[field], 1.0F, fine[field].plane, grid.width, grid.height, threadPool, grid.decimation);
    }
    return fine;
}

// Back-substitutes R d = z into a copy of the prior's grid, with U = R^-1 as the posterior's covariance factor.
[[nodiscard]] Posterior solvePosterior(const std::vector<RootInformation>& roots, const Posterior& prior, ThreadPool& threadPool) {
    Posterior posterior = prior;
    const int width = prior[kMeanX].width;
    threadPool.parallelFor(prior[kMeanX].height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
            const RootInformation& root = roots[pixel];
            const double dy = root.z2 / root.r22;
            posterior[kMeanX].plane[pixel] = static_cast<float>((root.z1 - (root.r12 * dy)) / root.r11);
            posterior[kMeanY].plane[pixel] = static_cast<float>(dy);
            posterior[kRootA].plane[pixel] = static_cast<float>(1.0 / root.r11);
            posterior[kRootB].plane[pixel] = static_cast<float>(-root.r12 / (root.r11 * root.r22));
            posterior[kRootC].plane[pixel] = static_cast<float>(1.0 / root.r22);
        }
    });
    return posterior;
}

// One frame at one octave: per source channel, every orientation's admissible baseband, and the octave's grid. Filtered once per frame.
struct FrameBands {
    ScaleSpaceLevel grid;
    std::vector<std::vector<Baseband>> bands;
    // Per channel, each response's error variance: the source's own noise through the envelope, plus its rounding and border bounds.
    std::vector<std::vector<float>> noise;
};

[[nodiscard]] FrameBands frameBands(int level, const LevelBank& bank, const std::vector<std::vector<float>>& planes,
                                    const std::vector<std::vector<float>>& variances, int width, int height, ThreadPool& threadPool) {
    FrameBands frame;
    for (std::size_t channel = 0; channel < planes.size(); ++channel) {
        const std::vector<float>& plane = planes[channel];
        ScaleSpaceLevel lowpass = octaveLevel(plane, width, height, level, threadPool);
        const EnvelopeEnergy energy = envelopeEnergy(lowpass, width, height);
        std::vector<float> absolute(plane.size());
        std::vector<float> square(plane.size());
        std::transform(plane.begin(), plane.end(), absolute.begin(), [](float value) { return std::fabs(value); });
        std::transform(plane.begin(), plane.end(), square.begin(), [](float value) { return value * value; });
        const std::vector<float> magnitude = octaveLevel(absolute, width, height, level, threadPool).plane;
        const std::vector<float> secondMoment = octaveLevel(square, width, height, level, threadPool).plane;
        // No variance means a deterministic source, uncertain only by its rounding and by what the frame does not show.
        const std::vector<float> spread = variances.empty() ? std::vector<float>(lowpass.plane.size(), 0.0F)
                                                            : octaveLevel(variances[channel], width, height, level, threadPool).plane;
        std::vector<float>& noise = frame.noise.emplace_back(lowpass.plane.size());
        const double whole = energy.axis * energy.axis;
        threadPool.parallelFor(lowpass.height, [&](int y) {
            for (int x = 0; x < lowpass.width; ++x) {
                const std::size_t pixel =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(lowpass.width)) + static_cast<std::size_t>(x);
                const double inside = energy.insideX[static_cast<std::size_t>(x)] * energy.insideY[static_cast<std::size_t>(y)];
                const auto mean = static_cast<double>(lowpass.plane[pixel]);
                const double localVariance = std::max(static_cast<double>(secondMoment[pixel]) - (mean * mean), 0.0);
                const double rounding = kFloat32Roundoff * static_cast<double>(magnitude[pixel]);
                // Texel noise through the envelope, sum h^2 V; the mirror's unseen content as a like draw, 2 V_I sum_out h^2; rounding.
                noise[pixel] = static_cast<float>((whole * static_cast<double>(spread[pixel])) + (2.0 * (whole - inside) * localVariance) +
                                                  (rounding * rounding));
            }
        });
        std::vector<Baseband>& bands = frame.bands.emplace_back();
        for (std::size_t orientation = 0; orientation < bank.waves.size(); ++orientation) {
            bands.push_back(baseband(plane, width, height, lowpass, bank.waves[orientation], bank.blurred[orientation], level,
                                     threadPool));
        }
        if (frame.grid.plane.empty()) {
            frame.grid = std::move(lowpass);
        }
    }
    return frame;
}

// Both frames' bands at one octave and the bank that made them.
struct OctavePair {
    LevelBank bank;
    FrameBands previous;
    FrameBands current;
};

// One measurement about `warp`: constraints depend on a texel's own warp alone, so `sums` carry over and only `stale` texels are re-read.
[[nodiscard]] Posterior measureLevel(const OctavePair& octave, const Posterior& prior, const LevelWarp& warp,
                                     const std::vector<std::uint8_t>& stale, std::vector<ChannelSums>& sums, ThreadPool& threadPool) {
    const ScaleSpaceLevel& grid = prior[kMeanX];
    std::vector<RootInformation> roots(grid.plane.size());
    threadPool.parallelFor(grid.height, [&](int y) {
        for (int x = 0; x < grid.width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(grid.width)) + static_cast<std::size_t>(x);
            roots[pixel] = priorRoot(prior, pixel);
            for (ChannelSums& channel : sums) {
                if (stale[pixel] != 0) {
                    channel.clear(pixel);
                }
            }
        }
    });
    for (std::size_t channel = 0; channel < octave.current.bands.size(); ++channel) {
        for (std::size_t orientation = 0; orientation < octave.bank.waves.size(); ++orientation) {
            accumulateOrientation(octave.previous.bands[channel][orientation], octave.current.bands[channel][orientation],
                                  octave.previous.noise[channel], octave.current.noise[channel], octave.bank.waves[orientation], grid,
                                  warp, stale, sums[channel], threadPool);
        }
        foldChannel(sums[channel], grid, warp, roots, threadPool);
    }
    return solvePosterior(roots, prior, threadPool);
}

// Iterated update: phase constancy is linear in d only near the warp, so re-read at each new estimate, against the same prior and data.
[[nodiscard]] Posterior refineLevel(const OctavePair& octave, const Posterior& prior, ThreadPool& threadPool) {
    const std::size_t texels = prior[kMeanX].plane.size();
    std::vector<ChannelSums> sums(octave.current.bands.size(), ChannelSums(texels));
    std::vector<std::uint8_t> stale(texels, 1U);
    LevelWarp warp = levelWarp(prior);
    LevelWarp earlier = warp;
    std::size_t lastMoved = texels + 1;
    for (;;) {
        Posterior estimate = measureLevel(octave, prior, warp, stale, sums, threadPool);
        LevelWarp next = levelWarp(estimate);
        std::size_t moved = 0;
        for (std::size_t pixel = 0; pixel < texels; ++pixel) {
            // Back to the warp of two reads ago: the mean sits on a rounding boundary, where either read is within half a sample of it.
            if (next.x[pixel] == earlier.x[pixel] && next.y[pixel] == earlier.y[pixel]) {
                next.x[pixel] = warp.x[pixel];
                next.y[pixel] = warp.y[pixel];
            }
            stale[pixel] = next.x[pixel] != warp.x[pixel] || next.y[pixel] != warp.y[pixel] ? 1U : 0U;
            moved += stale[pixel];
        }
        // Done at a fixed point of the warp, or once the moved count stops shrinking: strictly decreasing, so no iteration cap is needed.
        if (moved == 0 || moved >= lastMoved) {
            return estimate;
        }
        lastMoved = moved;
        earlier = std::move(warp);
        warp = std::move(next);
    }
}

// Writes (0, 0, prior std) everywhere: the answer with no observation to condition on.
void writePrior(HdrImage& out) {
    const float spread = opticFlowPriorStd(out.width, out.height);
    for (std::size_t pixel = 0; pixel < out.texels.size(); pixel += static_cast<std::size_t>(out.channels)) {
        out.texels[pixel + 2] = spread;
    }
}

// Base-grid (dx, dy, sqrt of the largest eigenvalue of U U^T) from a level-0 belief, which lives on the base grid.
void writePosterior(const Posterior& belief, HdrImage& out, ThreadPool& threadPool) {
    threadPool.parallelFor(out.height, [&](int y) {
        for (int x = 0; x < out.width; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(out.width)) + static_cast<std::size_t>(x);
            const auto a = static_cast<double>(belief[kRootA].plane[pixel]);
            const auto b = static_cast<double>(belief[kRootB].plane[pixel]);
            const auto c = static_cast<double>(belief[kRootC].plane[pixel]);
            const double covXX = (a * a) + (b * b);
            const double covXY = b * c;
            const double covYY = c * c;
            const double largest = (0.5 * (covXX + covYY)) + std::hypot(0.5 * (covXX - covYY), covXY);
            writeTexel(out, x, y, glm::vec3(belief[kMeanX].plane[pixel], belief[kMeanY].plane[pixel],
                                            static_cast<float>(std::sqrt(largest))));
        }
    });
}

}  // namespace

float opticFlowPriorStd(int width, int height) {
    // A frame too small for even the finest octave keeps the finest one's range, so the prior stays finite and never reads as zero.
    return static_cast<float>(unambiguousRange(std::max(octaveLevelCount(width, height) - 1, 0)));
}

HdrImage opticFlowAov(const OpticFlowView& current, const OpticFlowView& previous, ThreadPool& threadPool) {
    const HdrImage& image = *current.mean;
    HdrImage out = pathtracer::gfx::makeImage(image.width, image.height, aovChannels(AovId::OpticFlow));
    const int levels = octaveLevelCount(image.width, image.height);
    const bool paired = previous.mean != nullptr && previous.mean->width == image.width && previous.mean->height == image.height &&
                        previous.mean->channels == image.channels;
    if (!paired || levels == 0) {
        writePrior(out);
        return out;
    }
    const std::vector<std::vector<float>> after = channelPlanes(image, threadPool);
    const std::vector<std::vector<float>> before = channelPlanes(*previous.mean, threadPool);
    const std::vector<std::vector<float>> afterVariance = channelVariances(current, threadPool);
    const std::vector<std::vector<float>> beforeVariance = channelVariances(previous, threadPool);
    Posterior belief;
    // Coarse to fine: each octave's posterior is the next finer one's prior, so large motion is caught before fine detail aliases it.
    for (int level = levels - 1; level >= 0; --level) {
        OctavePair octave{levelBank(level, image.width, image.height, threadPool), {}, {}};
        octave.previous = frameBands(level, octave.bank, before, beforeVariance, image.width, image.height, threadPool);
        octave.current = frameBands(level, octave.bank, after, afterVariance, image.width, image.height, threadPool);
        const ScaleSpaceLevel& grid = octave.current.grid;
        const Posterior prior = level == levels - 1 ? flatPrior(grid, level) : prolong(belief, grid, threadPool);
        belief = refineLevel(octave, prior, threadPool);
    }
    writePosterior(belief, out, threadPool);
    return out;
}

}  // namespace pathtracer::debug
