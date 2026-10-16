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

}  // namespace

void applyDisplayOptics(HdrImage& image, PixelLattice lattice, bool wrapsHorizontally, const DisplayOptics& optics,
                        pathtracer::scene::ThreadPool& threadPool) {
    if (optics.aberration == 0.0F && optics.bloomStrength == 0.0F) {
        return;
    }
    const bool blooms = optics.bloomStrength > 0.0F;
    const int octaves = bloomOctaves(image.width, image.height);
    // levels[k] is octave k + 1: the down chain, then each up octave summed into the down octave of its size, so octave 1 ends as up_1.
    std::vector<HdrImage> levels;
    if (blooms) {
        levels.reserve(static_cast<std::size_t>(octaves));
        const HdrImage* source = &image;
        for (int octave = 1; octave <= octaves; ++octave) {
            levels.push_back(makeImage(std::max(1, image.width >> octave), std::max(1, image.height >> octave), kLanes));
            filterPass(*source, levels.back(), kBloomDownTaps, false, wrapsHorizontally, threadPool);
            source = &levels.back();
        }
        for (std::size_t k = levels.size() - 1; k > 0; --k) {
            filterPass(levels[k], levels[k - 1], kBloomUpTaps, true, wrapsHorizontally, threadPool);
        }
    }
    // The display shader's composite fused with the last up pass, whose 1/L weights every octave equally: mix(A(I), up_0, k).
    using UpFetches = PassFetches<kBloomUpTaps.size()>;
    const UpFetches finest = blooms ? passFetches(kBloomUpTaps, levels.front(), image.width, image.height, wrapsHorizontally) : UpFetches{};
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
                color = glm::mix(color, tapSum(levels.front(), kBloomUpTaps, finest, column, row) * octaveWeight, optics.bloomStrength);
            }
            store(out, column, row, color);
        }
    });
    image = std::move(out);
}

}  // namespace pathtracer::gfx
