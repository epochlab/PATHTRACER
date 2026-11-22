#include "pathtracer/scene/sampler.h"

#include <algorithm>
#include <array>
#include <bit>

namespace pathtracer::scene {

namespace {

// SplitMix64 finalizer (Vigna): an avalanched 64-bit key from a small integer and a seed, its halves two independent 32-bit seeds.
std::uint64_t hashSeed(int extra, std::uint32_t runSeed) {
    std::uint64_t h = static_cast<std::uint64_t>(static_cast<std::uint32_t>(extra)) * 0x165667B19E3779F9ULL;
    h ^= static_cast<std::uint64_t>(runSeed);
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27;
    h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return h;
}

// 2^32 points: the index is 32-bit, so a tile's pixels x samples must fit it.
constexpr int kIndexBits = 32;

std::uint32_t reverseBits(std::uint32_t x) {
#if defined(__has_builtin)
#if __has_builtin(__builtin_bitreverse32)
    return __builtin_bitreverse32(x);  // one RBIT instruction on arm64
#endif
#endif
    x = (x >> 16) | (x << 16);
    x = ((x & 0xFF00FF00U) >> 8) | ((x & 0x00FF00FFU) << 8);
    x = ((x & 0xF0F0F0F0U) >> 4) | ((x & 0x0F0F0F0FU) << 4);
    x = ((x & 0xCCCCCCCCU) >> 2) | ((x & 0x33333333U) << 2);
    return ((x & 0xAAAAAAAAU) >> 1) | ((x & 0x55555555U) << 1);
}

// Base-2 Owen scrambling (Owen 1995; Burley 2020) via Vegdahl's Laine-Karras hash as shipped by Cycles: transcribed, never derived.
std::uint32_t reversedBitOwen(std::uint32_t n, std::uint32_t seed) {
    n ^= n * 0x3D20ADEAU;
    n += seed;
    n *= (seed >> 16) | 1U;
    n ^= n * 0x05526C56U;
    n ^= n * 0x53A22864U;
    return n;
}

std::uint32_t nestedUniformScramble(std::uint32_t x, std::uint32_t seed) {
    return reverseBits(reversedBitOwen(reverseBits(x), seed));
}

// Interleaves a coordinate below 2^16 into the even bits: one axis of a Morton code.
std::uint32_t spreadBits(std::uint32_t x) {
    x = (x | (x << 8)) & 0x00FF00FFU;
    x = (x | (x << 4)) & 0x0F0F0F0FU;
    x = (x | (x << 2)) & 0x33333333U;
    return (x | (x << 1)) & 0x55555555U;
}

// All 24 orders of a base-4 digit, lexicographic, so a uniform pick among them is a uniform random child permutation.
constexpr std::array<std::array<std::uint32_t, 4>, 24> kDigitOrders = [] {
    std::array<std::array<std::uint32_t, 4>, 24> orders{};
    std::array<std::uint32_t, 4> order = {0, 1, 2, 3};
    for (std::array<std::uint32_t, 4>& entry : orders) {
        entry = order;
        std::next_permutation(order.begin(), order.end());
    }
    return orders;
}();

// Base-4 Owen scrambling (Owen 1995) as pbrt-v4's ZSobolSampler: each digit takes one of all 4! orders, hashed from the digits above.
std::uint32_t scrambleBase4(std::uint32_t code, int digits, std::uint32_t seed) {
    std::uint32_t scrambled = 0;
    for (int d = digits - 1; d >= 0; --d) {
        const int shift = 2 * d;
        // The digits above behind a sentinel 1, so each tree node has a distinct key whatever its depth.
        const std::uint32_t node = (1U << (2 * (digits - 1 - d))) | static_cast<std::uint32_t>(std::uint64_t{code} >> (shift + 2));
        // Lemire's multiply-shift onto [0, 24): bias below 24 / 2^32.
        const std::uint64_t pick = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(hashSeed(static_cast<int>(node), seed))) * 24U) >> 32U;
        scrambled |= kDigitOrders[pick][(code >> shift) & 3U] << shift;
    }
    return scrambled;
}

// Owen-scrambled coordinate from its bit-reversed Sobol point, the domain the Laine-Karras hash works in, so no reversal is spent.
float scrambledUnitFloat(std::uint32_t reversedPoint, std::uint32_t seed) {
    return static_cast<float>(reverseBits(reversedBitOwen(reversedPoint, seed)) >> 8) * 0x1.0p-24F;
}

}  // namespace

std::uint32_t reversedSobolDimension1(std::uint32_t index) {
    // Superset-sum over the 5-bit digit positions, one position bit per step, so each digit gathers every b containing it.
    index ^= (index >> 1) & 0x55555555U;
    index ^= (index >> 2) & 0x33333333U;
    index ^= (index >> 4) & 0x0F0F0F0FU;
    index ^= (index >> 8) & 0x00FF00FFU;
    return index ^ ((index >> 16) & 0x0000FFFFU);
}

std::uint32_t replicateSeed(std::uint32_t scrambleSeed, int replicate) {
    // SplitMix64 of r alone is 0 at r = 0 and avalanched otherwise; XOR is the kAoSeedOffset idiom for an independent stream.
    return scrambleSeed ^ static_cast<std::uint32_t>(hashSeed(replicate, 0U));
}

Sampler::Sampler(int pixelX, int pixelY, int sampleIndex, int sampleCount, std::uint32_t scrambleSeed) {
    const int sampleBits = sampleCount > 0 ? std::bit_width(static_cast<std::uint32_t>(sampleCount) - 1U) : kIndexBits;
    // The largest square tile whose pixels' blocks fit the index; unbounded leaves 0, one independent sequence per pixel.
    const int tileBits = (kIndexBits - sampleBits) / 2;
    const auto x = static_cast<std::uint32_t>(pixelX);
    const auto y = static_cast<std::uint32_t>(pixelY);
    const std::uint64_t tileHash =
        hashSeed(static_cast<int>(x >> tileBits), static_cast<std::uint32_t>(hashSeed(static_cast<int>(y >> tileBits), scrambleSeed)));
    seed_ = static_cast<std::uint32_t>(tileHash);
    const std::uint64_t tileMask = (std::uint64_t{1} << (2 * tileBits)) - 1U;
    const std::uint32_t morton = (spreadBits(x & static_cast<std::uint32_t>((1U << tileBits) - 1U)) << 1) |
                                 spreadBits(y & static_cast<std::uint32_t>((1U << tileBits) - 1U));
    // Ahmed & Wonka 2020: a base-4-scrambled Morton rank picks the pixel's block, so aligned pixel squares hold aligned blocks.
    const std::uint32_t rank = scrambleBase4(morton, tileBits, static_cast<std::uint32_t>(tileHash >> 32));
    index_ = static_cast<std::uint32_t>((std::uint64_t{rank} << sampleBits) | static_cast<std::uint32_t>(sampleIndex));
    indexMask_ = static_cast<std::uint32_t>((tileMask << sampleBits) | ((std::uint64_t{1} << sampleBits) - 1U));
}

Sampler Sampler::stream(std::uint32_t streamOffset) const {
    Sampler other = *this;
    // XOR into the avalanched base, the kAoSeedOffset idiom: every set's hash then differs from this stream's.
    other.seed_ ^= streamOffset;
    other.dimensionSet_ = 0;
    return other;
}

float Sampler::next1D() {
    // Two seeds per set from the halves of one avalanched hash: SplitMix64 already decorrelates them, as kAoSeedOffset also rests on.
    const std::uint64_t setHash = hashSeed(2 * dimensionSet_, seed_);
    ++dimensionSet_;
    // Shuffling the index per set (Burley 2020 Sec. 5.2) decorrelates sets; nesting maps every aligned block to an aligned block.
    const std::uint32_t shuffled = nestedUniformScramble(index_, static_cast<std::uint32_t>(setHash)) & indexMask_;
    return scrambledUnitFloat(shuffled, static_cast<std::uint32_t>(setHash >> 32U));
}

glm::vec2 Sampler::next2D() {
    // Both coordinates share one shuffled index, being one 2D point; their Owen scrambles differ, per Owen's independence rule (1995).
    const std::uint64_t setHash = hashSeed(2 * dimensionSet_, seed_);
    const std::uint64_t scrambleHash = hashSeed((2 * dimensionSet_) + 1, seed_);
    ++dimensionSet_;
    const std::uint32_t shuffled = nestedUniformScramble(index_, static_cast<std::uint32_t>(setHash)) & indexMask_;
    return {scrambledUnitFloat(shuffled, static_cast<std::uint32_t>(setHash >> 32U)),
            scrambledUnitFloat(reversedSobolDimension1(shuffled), static_cast<std::uint32_t>(scrambleHash))};
}

CdfSample invertCdf(const float* cdf, int count, float u) {
    // Last knot <= u, upper_bound's bin, by branch-free halving (Khuong & Morin 2017): a select per step, no mispredicted branch.
    const float* base = cdf;
    for (int n = count + 1; n > 1; n -= n / 2) {
        base = base[n / 2] <= u ? base + (n / 2) : base;
    }
    const int index = std::min(static_cast<int>(base - cdf), count - 1);
    const float lo = cdf[index];
    const float hi = cdf[index + 1];
    const float fraction = hi > lo ? std::clamp((u - lo) / (hi - lo), 0.0F, 1.0F) : 0.5F;
    return {index, fraction};
}

}  // namespace pathtracer::scene
