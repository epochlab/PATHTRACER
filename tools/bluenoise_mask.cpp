// Offline generator for the blue-noise mask .inc files by Ulichney's void-and-cluster method (Proc. SPIE 1913, 1993), transcribed.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/pixel_lattice.h"

namespace {

// 128^2, Georgiev & Fajardo's size (SIGGRAPH 2016 Talks). Must match sampler.cpp kMaskSize and ocio_cpu_transform.h kDitherMaskSize.
constexpr int kSize = 128;
// Even, so the hexagonal torus's row parity is consistent across its wrap.
static_assert(kSize % 2 == 0);
constexpr int kPixels = kSize * kSize;

// Ulichney Sec. 2: a Gaussian at sigma 1.5, untruncated, the wrap-around form being what makes the array tileable.
constexpr double kSigma = 1.5;

// Ulichney Sec. 3: the initial pattern starts from white noise whose 1s are ~10%, governing how much rearranging is needed, not quality.
constexpr int kInitialOnes = kPixels / 10;

// Energy field of a set of marked cells. One class, not two: phases I/II mark the 1s and phase III the 0s, the same search either way.
class EnergyField {
public:
    explicit EnergyField(pathtracer::gfx::PixelLattice lattice) {
        // By wrapped offset and source row parity, which flips offset rows' half-pitch: Ulichney Sec. 2's wrap-around form, on the film.
        const double rowPitch = pathtracer::gfx::rowPitch(lattice);
        for (int parity = 0; parity < 2; ++parity) {
            const double sourceX = pathtracer::gfx::pixelCentre(lattice, 0, parity).x;
            for (int dy = 0; dy < kSize; ++dy) {
                const double wy = std::min(dy, kSize - dy) * rowPitch;
                for (int dx = 0; dx < kSize; ++dx) {
                    const double wx = std::abs(std::remainder(pathtracer::gfx::pixelCentre(lattice, dx, parity + dy).x - sourceX, kSize));
                    kernel_[static_cast<std::size_t>(parity)][(static_cast<std::size_t>(dy) * kSize) + static_cast<std::size_t>(dx)] =
                        std::exp(-((wx * wx) + (wy * wy)) / (2.0 * kSigma * kSigma));
                }
            }
        }
    }

    // Rebuilt from the pattern, not rewound: recomputing costs one pass where rewinding would carry each phase's rounding into the next.
    void reset(const std::array<std::uint8_t, kPixels>& marked) {
        marked_ = marked;
        markedCount_ = 0;
        energy_.fill(0.0);
        for (int i = 0; i < kPixels; ++i) {
            if (marked_[static_cast<std::size_t>(i)] != 0) {
                splat(i, 1.0);
                ++markedCount_;
            }
        }
    }

    void insert(int i) {
        marked_[static_cast<std::size_t>(i)] = 1;
        splat(i, 1.0);
        ++markedCount_;
    }

    void erase(int i) {
        marked_[static_cast<std::size_t>(i)] = 0;
        splat(i, -1.0);
        --markedCount_;
    }

    [[nodiscard]] int tightestCluster() const { return extremum(1, /*wantMax=*/true); }
    [[nodiscard]] int largestVoid() const { return extremum(0, /*wantMax=*/false); }
    [[nodiscard]] int markedCount() const { return markedCount_; }
    [[nodiscard]] const std::array<std::uint8_t, kPixels>& marked() const { return marked_; }

private:
    // Adds `sign` times the kernel centred on i. The x offset is advanced and wrapped, the modulo being the generator's dominant cost.
    void splat(int i, double sign) {
        const int iy = i / kSize;
        const int ix = i % kSize;
        for (int jy = 0; jy < kSize; ++jy) {
            const int dy = (jy - iy + kSize) % kSize;
            const double* kernelRow = &kernel_[static_cast<std::size_t>(iy & 1)][static_cast<std::size_t>(dy) * kSize];
            double* energyRow = &energy_[static_cast<std::size_t>(jy) * kSize];
            int dx = (kSize - ix) % kSize;
            for (int jx = 0; jx < kSize; ++jx) {
                energyRow[jx] += sign * kernelRow[dx];
                dx = dx + 1 == kSize ? 0 : dx + 1;
            }
        }
    }

    // Ties resolve to the lowest index, which is what makes the generator reproducible at exactly equal energy.
    [[nodiscard]] int extremum(std::uint8_t state, bool wantMax) const {
        int best = -1;
        double bestEnergy = 0.0;
        for (int i = 0; i < kPixels; ++i) {
            if (marked_[static_cast<std::size_t>(i)] != state) {
                continue;
            }
            const double e = energy_[static_cast<std::size_t>(i)];
            if (best < 0 || (wantMax ? e > bestEnergy : e < bestEnergy)) {
                best = i;
                bestEnergy = e;
            }
        }
        return best;
    }

    std::array<std::array<double, kPixels>, 2> kernel_{};
    std::array<double, kPixels> energy_{};
    std::array<std::uint8_t, kPixels> marked_{};
    int markedCount_ = 0;
};

// Ulichney Sec. 3 / Fig. 2. The shuffle is written out because only mt19937's raw output is specified exactly by the standard.
std::array<std::uint8_t, kPixels> initialBinaryPattern(std::uint32_t seed, EnergyField& field) {
    std::array<int, kPixels> order{};
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    for (int i = kPixels - 1; i > 0; --i) {
        std::swap(order[static_cast<std::size_t>(i)], order[rng() % static_cast<std::uint32_t>(i + 1)]);
    }

    std::array<std::uint8_t, kPixels> pattern{};
    for (int i = 0; i < kInitialOnes; ++i) {
        pattern[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])] = 1;
    }

    field.reset(pattern);
    for (;;) {
        const int cluster = field.tightestCluster();
        field.erase(cluster);
        const int emptiest = field.largestVoid();
        if (emptiest == cluster) {
            field.insert(cluster);  // removing this 1 is what opened the largest void -- converged
            return field.marked();
        }
        field.insert(emptiest);
    }
}

// Ulichney Sec. 4's three phases, which between them assign every cell a distinct rank in [0, kPixels).
std::array<std::uint16_t, kPixels> rankPattern(const std::array<std::uint8_t, kPixels>& initial, EnergyField& field) {
    std::array<std::uint16_t, kPixels> ranks{};

    field.reset(initial);
    while (field.markedCount() > 0) {
        const int i = field.tightestCluster();
        field.erase(i);
        ranks[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(field.markedCount());
    }

    field.reset(initial);
    while (field.markedCount() <= kPixels / 2) {
        const int i = field.largestVoid();
        ranks[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(field.markedCount());
        field.insert(i);
    }

    std::array<std::uint8_t, kPixels> complement{};
    for (int i = 0; i < kPixels; ++i) {
        complement[static_cast<std::size_t>(i)] = field.marked()[static_cast<std::size_t>(i)] ^ 1U;
    }
    int ones = kPixels - std::accumulate(complement.begin(), complement.end(), 0);
    field.reset(complement);
    while (field.markedCount() > 0) {
        const int i = field.tightestCluster();
        field.erase(i);
        ranks[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(ones);
        ++ones;
    }
    return ranks;
}

// The one property the rest of the system assumes: the mask is a bijection onto [0, kPixels), checked before the .inc is written.
bool isPermutation(const std::array<std::uint16_t, kPixels>& ranks) {
    std::array<std::uint8_t, kPixels> seen{};
    for (const std::uint16_t rank : ranks) {
        if (rank >= kPixels || seen[rank] != 0) {
            return false;
        }
        seen[rank] = 1;
    }
    return true;
}

bool writeInc(const std::string& path, const std::array<std::uint16_t, kPixels>& ranks, std::uint32_t seed,
              pathtracer::gfx::PixelLattice lattice) {
    const bool hexagonal = lattice == pathtracer::gfx::PixelLattice::Hexagonal;
    std::ofstream out(path);
    if (!out) {
        std::cerr << "bluenoise_mask: cannot write " << path << "\n";
        return false;
    }
    out << "// Generated by tools/bluenoise_mask.cpp -- do not edit. Regenerate with:\n"
        << "//   ./build/bluenoise_mask --seed " << seed << (hexagonal ? " --lattice hexagonal" : "") << " --out " << path << "\n"
        << "// Void-and-cluster dither array (Ulichney 1993), " << kSize << "x" << kSize << (hexagonal ? " offset rows" : "")
        << ", sigma " << kSigma << ", initial pattern seed " << seed << ".\n"
        << "// Row-major ranks, a permutation of [0, " << kPixels << ").\n";
    for (int i = 0; i < kPixels; ++i) {
        out << (i % 16 == 0 ? "\n" : " ") << ranks[static_cast<std::size_t>(i)] << ",";
    }
    out << "\n";
    return out.good();
}

// Greyscale dump for looking at. The radial spectrum in sampler_validate is the gate; this catches streaks and tile seams it hides.
bool writePreview(const std::string& path, const std::array<std::uint16_t, kPixels>& ranks, pathtracer::gfx::PixelLattice lattice) {
    // Three replicated channels, not one: a lone R plane reads as red in a viewer, and this file exists only to be looked at.
    pathtracer::gfx::HdrImage image = pathtracer::gfx::makeImage(kSize, kSize, pathtracer::gfx::kRgbChannels);
    for (int i = 0; i < kPixels; ++i) {
        const auto value = static_cast<float>((ranks[static_cast<std::size_t>(i)] + 0.5) / kPixels);
        writeTexel(image, i % kSize, i / kSize, glm::vec3(value));
    }
    return pathtracer::gfx::writeExr(path, image, pathtracer::gfx::ImageRole::Data, lattice);
}

}  // namespace

int main(int argc, char** argv) {
    std::string outPath = "src/scene/blue_noise_mask.inc";
    std::string previewPath;
    std::uint32_t seed = 1;
    pathtracer::gfx::PixelLattice lattice = pathtracer::gfx::PixelLattice::Square;
    for (int i = 1; i < argc; ++i) {
        const bool hasValue = i + 1 < argc;
        if (std::strcmp(argv[i], "--out") == 0 && hasValue) {
            outPath = argv[++i];
        } else if (std::strcmp(argv[i], "--preview") == 0 && hasValue) {
            previewPath = argv[++i];
        } else if (std::strcmp(argv[i], "--seed") == 0 && hasValue) {
            seed = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--lattice") == 0 && hasValue && std::strcmp(argv[i + 1], "hexagonal") == 0) {
            lattice = pathtracer::gfx::PixelLattice::Hexagonal;
            ++i;
        } else {
            std::cerr << "bluenoise_mask: unknown or incomplete argument '" << argv[i]
                      << "'\nusage: bluenoise_mask [--out path.inc] [--preview path.exr] [--seed N] [--lattice hexagonal]\n";
            return EXIT_FAILURE;
        }
    }

    EnergyField field(lattice);
    const std::array<std::uint8_t, kPixels> initial = initialBinaryPattern(seed, field);
    const std::array<std::uint16_t, kPixels> ranks = rankPattern(initial, field);
    if (!isPermutation(ranks)) {
        std::cerr << "bluenoise_mask: ranks are not a permutation of [0, " << kPixels << ") -- generator is wrong\n";
        return EXIT_FAILURE;
    }
    if (!writeInc(outPath, ranks, seed, lattice)) {
        return EXIT_FAILURE;
    }
    std::cout << "bluenoise_mask: wrote " << outPath << " (" << kSize << "x" << kSize << ", sigma " << kSigma
              << ", seed " << seed << ")\n";
    if (!previewPath.empty() && !writePreview(previewPath, ranks, lattice)) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
