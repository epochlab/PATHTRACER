#include "pathtracer/gfx/display_optics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

namespace pathtracer::gfx {

namespace {

constexpr int kLanes = 3;

// GL_LINEAR along one axis: texels i0 and i1 blended by f, each index addressed after the floor, as GL addresses them.
struct AxisFetch {
    std::size_t i0;
    std::size_t i1;
    float f;
};

// Each target texel's centre moved `offset` source texels, in the shader's float arithmetic; the axis repeats or clamps to edge.
std::vector<AxisFetch> axisFetches(int targetSize, int sourceSize, int offset, bool wraps) {
    const float targetTexel = 1.0F / static_cast<float>(targetSize);
    const float sourceTexel = 1.0F / static_cast<float>(sourceSize);
    const auto address = [sourceSize, wraps](int index) {
        return static_cast<std::size_t>(wraps ? (((index % sourceSize) + sourceSize) % sourceSize)
                                              : std::clamp(index, 0, sourceSize - 1));
    };
    std::vector<AxisFetch> fetches(static_cast<std::size_t>(targetSize));
    for (int i = 0; i < targetSize; ++i) {
        const float u = ((static_cast<float>(i) + 0.5F) * targetTexel) + (static_cast<float>(offset) * sourceTexel);
        const float x = (u * static_cast<float>(sourceSize)) - 0.5F;
        const float floorX = std::floor(x);
        const int index = static_cast<int>(floorX);
        fetches[static_cast<std::size_t>(i)] = {address(index), address(index + 1), x - floorX};
    }
    return fetches;
}

// Per-tap column and row fetches of one pass, hoisted: the taps form a tensor grid, so each depends on one axis alone.
template <std::size_t N>
struct PassFetches {
    std::array<std::vector<AxisFetch>, N> columns;
    std::array<std::vector<AxisFetch>, N> rows;
};

// T always clamps to edge; S repeats on a lat-long, whose left and right edges are one meridian (BloomPass::setWrapsHorizontally).
template <std::size_t N>
PassFetches<N> passFetches(const std::array<BloomTap, N>& taps, const HdrImage& source, int targetWidth, int targetHeight,
                           bool wrapsHorizontally) {
    PassFetches<N> fetches;
    for (std::size_t t = 0; t < N; ++t) {
        fetches.columns[t] = axisFetches(targetWidth, source.width, taps[t].dx, wrapsHorizontally);
        fetches.rows[t] = axisFetches(targetHeight, source.height, taps[t].dy, false);
    }
    return fetches;
}

glm::vec3 texel(const HdrImage& image, std::size_t column, std::size_t row) {
    const float* rgb = image.texels.data() + (((row * static_cast<std::size_t>(image.width)) + column) * kLanes);
    return {rgb[0], rgb[1], rgb[2]};
}

void store(HdrImage& image, std::size_t column, std::size_t row, const glm::vec3& color) {
    float* rgb = image.texels.data() + (((row * static_cast<std::size_t>(image.width)) + column) * kLanes);
    rgb[0] = color.r;
    rgb[1] = color.g;
    rgb[2] = color.b;
}

// One shader invocation's tap sum at target texel (x, y): each tap a bilinear blend of its 2x2 source texels.
template <std::size_t N>
glm::vec3 tapSum(const HdrImage& source, const std::array<BloomTap, N>& taps, const PassFetches<N>& fetches, std::size_t x,
                 std::size_t y) {
    glm::vec3 sum(0.0F);
    for (std::size_t t = 0; t < N; ++t) {
        const AxisFetch& column = fetches.columns[t][x];
        const AxisFetch& row = fetches.rows[t][y];
        const glm::vec3 near = glm::mix(texel(source, column.i0, row.i0), texel(source, column.i1, row.i0), column.f);
        const glm::vec3 far = glm::mix(texel(source, column.i0, row.i1), texel(source, column.i1, row.i1), column.f);
        sum += taps[t].weight * glm::mix(near, far, row.f);
    }
    return sum;
}

// One pyramid pass over target; accumulating adds target's own texel, the up pass's down octave of the same size, read where written.
template <std::size_t N>
void filterPass(const HdrImage& source, HdrImage& target, const std::array<BloomTap, N>& taps, bool accumulate, bool wrapsHorizontally,
                pathtracer::scene::ThreadPool& threadPool) {
    const PassFetches<N> fetches = passFetches(taps, source, target.width, target.height, wrapsHorizontally);
    threadPool.parallelFor(target.height, [&](int y) {
        const auto row = static_cast<std::size_t>(y);
        for (std::size_t x = 0; x < static_cast<std::size_t>(target.width); ++x) {
            const glm::vec3 sum = tapSum(source, taps, fetches, x, row);
            store(target, x, row, accumulate ? sum + texel(target, x, row) : sum);
        }
    });
}

// A hexagonal level's texel at (x, y), folded into the level by the lattice's mirrors or wrapped, as the hexagonal shaders fetch.
glm::vec3 hexagonalTexel(const HdrImage& level, int x, int y, bool wrapsHorizontally) {
    const glm::ivec2 cell = hexagonalReflection(x, y, level.width, level.height, wrapsHorizontally);
    return texel(level, static_cast<std::size_t>(cell.x), static_cast<std::size_t>(cell.y));
}

// The normalised adjoint of Loop subdivision at coarse (j, m), fine (2j + (m & 1), 2m): every mask entry over 4, in mask order.
glm::vec3 hexagonalDown(const HdrImage& fine, int j, int m, bool wrapsHorizontally) {
    const int x = (2 * j) + (m & 1);
    const int y = 2 * m;
    glm::vec3 sum(0.0F);
    for (const HexBloomTap& tap : kHexBloomMask) {
        // Axial (dq, dr) from an even row is dq + floor(dr / 2) columns: the arithmetic shift floors a negative dr.
        sum += (0.25F * tap.weight) * hexagonalTexel(fine, x + tap.dq + (tap.dr >> 1), y + tap.dr, wrapsHorizontally);
    }
    return sum;
}

// Loop subdivision at fine (x, y): the coarse vertices whose mask reaches its coset, each at axial (q - dq, r - dr) / 2.
glm::vec3 hexagonalUp(const HdrImage& coarse, const std::array<std::vector<HexBloomTap>, 4>& cosets, int x, int y,
                      bool wrapsHorizontally) {
    const int q = x - (y >> 1);
    glm::vec3 sum(0.0F);
    for (const HexBloomTap& tap : cosets[static_cast<std::size_t>((q & 1) | ((y & 1) << 1))]) {
        const int row = (y - tap.dr) >> 1;
        sum += tap.weight * hexagonalTexel(coarse, ((q - tap.dq) >> 1) + (row >> 1), row, wrapsHorizontally);
    }
    return sum;
}

// One hexagonal pyramid pass over target; accumulating adds target's own texel, the up pass's down octave of the same size.
void hexagonalPass(const HdrImage& source, HdrImage& target, bool up, const std::array<std::vector<HexBloomTap>, 4>& cosets,
                   bool wrapsHorizontally, pathtracer::scene::ThreadPool& threadPool) {
    threadPool.parallelFor(target.height, [&](int y) {
        const auto row = static_cast<std::size_t>(y);
        for (int x = 0; x < target.width; ++x) {
            const auto column = static_cast<std::size_t>(x);
            const glm::vec3 sum = up ? hexagonalUp(source, cosets, x, y, wrapsHorizontally) : hexagonalDown(source, x, y, wrapsHorizontally);
            store(target, column, row, up ? sum + texel(target, column, row) : sum);
        }
    });
}

}  // namespace

std::vector<HexBloomTap> hexBloomCoset(int qParity, int rParity) {
    std::vector<HexBloomTap> coset;
    for (const HexBloomTap& tap : kHexBloomMask) {
        if ((tap.dq - qParity) % 2 == 0 && (tap.dr - rParity) % 2 == 0) {
            coset.push_back(tap);
        }
    }
    return coset;
}

void applyDisplayOptics(HdrImage& image, PixelLattice lattice, bool wrapsHorizontally, const DisplayOptics& optics,
                        pathtracer::scene::ThreadPool& threadPool) {
    if (optics.aberration == 0.0F && optics.bloomStrength == 0.0F) {
        return;
    }
    const bool blooms = optics.bloomStrength > 0.0F;
    const bool hexagonal = lattice == PixelLattice::Hexagonal;
    const int octaves = bloomOctaves(image.width, image.height);
    std::array<std::vector<HexBloomTap>, 4> cosets;
    if (blooms && hexagonal) {
        cosets = {hexBloomCoset(0, 0), hexBloomCoset(1, 0), hexBloomCoset(0, 1), hexBloomCoset(1, 1)};
    }
    // levels[k] is octave k + 1: the down chain, then each up octave summed into the down octave of its size, so octave 1 ends as up_1.
    std::vector<HdrImage> levels;
    if (blooms) {
        levels.reserve(static_cast<std::size_t>(octaves));
        const HdrImage* source = &image;
        for (int octave = 1; octave <= octaves; ++octave) {
            const glm::ivec2 size = bloomLevelSize(lattice, image.width, image.height, octave, wrapsHorizontally);
            levels.push_back(makeImage(size.x, size.y, kLanes));
            if (hexagonal) {
                hexagonalPass(*source, levels.back(), false, cosets, wrapsHorizontally, threadPool);
            } else {
                filterPass(*source, levels.back(), kBloomDownTaps, false, wrapsHorizontally, threadPool);
            }
            source = &levels.back();
        }
        for (std::size_t k = levels.size() - 1; k > 0; --k) {
            if (hexagonal) {
                hexagonalPass(levels[k], levels[k - 1], true, cosets, wrapsHorizontally, threadPool);
            } else {
                filterPass(levels[k], levels[k - 1], kBloomUpTaps, true, wrapsHorizontally, threadPool);
            }
        }
    }
    // The display shader's composite fused with the last up pass, whose 1/L weights every octave equally: mix(A(I), up_0, k).
    using UpFetches = PassFetches<kBloomUpTaps.size()>;
    const UpFetches finest =
        blooms && !hexagonal ? passFetches(kBloomUpTaps, levels.front(), image.width, image.height, wrapsHorizontally) : UpFetches{};
    const float octaveWeight = 1.0F / static_cast<float>(octaves);
    const glm::vec2 frame = frameExtent(lattice, image.width, image.height);
    HdrImage out = makeImage(image.width, image.height, kLanes);
    threadPool.parallelFor(image.height, [&](int y) {
        const auto row = static_cast<std::size_t>(y);
        for (int x = 0; x < image.width; ++x) {
            const auto column = static_cast<std::size_t>(x);
            glm::vec3 color = texel(image, column, row);
            // The shader's R at uv - (uv - 1/2) a and B at uv + (uv - 1/2) a, nearest-cell as its GL_NEAREST magnification reads at 1:1.
            if (optics.aberration > 0.0F) {
                const glm::vec2 uv = pixelCentre(lattice, x, y) / frame;
                const glm::vec2 shift = (uv - 0.5F) * optics.aberration;
                const glm::ivec2 red = nearestPixel(lattice, (uv - shift) * frame, image.width, image.height, wrapsHorizontally);
                const glm::ivec2 blue = nearestPixel(lattice, (uv + shift) * frame, image.width, image.height, wrapsHorizontally);
                color.r = texel(image, static_cast<std::size_t>(red.x), static_cast<std::size_t>(red.y)).r;
                color.b = texel(image, static_cast<std::size_t>(blue.x), static_cast<std::size_t>(blue.y)).b;
            }
            if (blooms) {
                const glm::vec3 up = hexagonal ? hexagonalUp(levels.front(), cosets, x, y, wrapsHorizontally)
                                               : tapSum(levels.front(), kBloomUpTaps, finest, column, row);
                color = glm::mix(color, up * octaveWeight, optics.bloomStrength);
            }
            store(out, column, row, color);
        }
    });
    image = std::move(out);
}

}  // namespace pathtracer::gfx
