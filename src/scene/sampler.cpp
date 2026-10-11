#include "pathtracer/scene/sampler.h"

#include <algorithm>
#include <array>
#include <bit>

namespace pathtracer::scene {

namespace {

// SplitMix64 finalizer (Vigna). Deliberately not a function of the pixel: the toroidal shift, not the scramble, separates pixels.
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

// Only two Sobol dimensions exist, which is the point of padding: every draw reuses this stratified pair under a fresh randomization.
constexpr int kSobolDimensions = 2;
constexpr int kSobolBits = 32;  // direction vectors are scaled by 2^32, so a 32-bit index spans the sequence's period

constexpr int kMaxDegree = 1;  // largest polynomial degree across the generated rows, reported by generate_sobol_table
struct SobolPolynomial {
    std::uint32_t degree;
    std::uint32_t coefficients;
    std::array<std::uint32_t, kMaxDegree> initial;
};

// Sobol dimension 2 only; dimension 1 needs no seeds and is generated directly below.
constexpr std::array<SobolPolynomial, kSobolDimensions - 1> kSobolPolynomials = {{
#include "sobol_direction_seeds.inc"
}};

// Direction vectors V from Joe & Kuo (ACM Alg. 659), derived at static init.
using SobolDirections = std::array<std::array<std::uint32_t, kSobolBits>, kSobolDimensions>;

SobolDirections buildSobolDirections() {
    SobolDirections directions{};

    // Sobol dimension 1: every initial direction number is 1, giving the van der Corput sequence in base 2.
    for (int b = 0; b < kSobolBits; ++b) {
        directions[0][static_cast<std::size_t>(b)] = 1U << (kSobolBits - 1 - b);
    }

    for (std::size_t d = 1; d < kSobolDimensions; ++d) {
        const SobolPolynomial& poly = kSobolPolynomials[d - 1];
        const auto degree = static_cast<int>(poly.degree);
        std::array<std::uint32_t, kSobolBits>& v = directions[d];

        for (int b = 0; b < degree; ++b) {
            v[static_cast<std::size_t>(b)] = poly.initial[static_cast<std::size_t>(b)] << (kSobolBits - 1 - b);
        }
        for (int b = degree; b < kSobolBits; ++b) {
            const std::uint32_t previous = v[static_cast<std::size_t>(b - degree)];
            std::uint32_t next = previous ^ (previous >> degree);
            for (int k = 1; k < degree; ++k) {
                const std::uint32_t coefficient = (poly.coefficients >> (degree - 1 - k)) & 1U;
                next ^= coefficient * v[static_cast<std::size_t>(b - k)];
            }
            v[static_cast<std::size_t>(b)] = next;
        }
    }
    return directions;
}

const SobolDirections kSobolDirections = buildSobolDirections();

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

// Sobol point for one dimension (Sobol 1967), random-access form. Dimension 1 is a bit reversal by identity: 62 -> 48 ns/draw.
std::uint32_t sobolPoint(std::uint32_t index, int dimension) {
    if (dimension == 0) {
        return reverseBits(index);
    }
    std::uint32_t x = 0;
    for (std::uint32_t bits = index; bits != 0U; bits &= bits - 1U) {
        x ^= kSobolDirections[static_cast<std::size_t>(dimension)][static_cast<std::size_t>(std::countr_zero(bits))];
    }
    return x;
}

// Blue-noise dithering (Georgiev & Fajardo 2016): one point set, toroidally shifted per pixel. Apparent cleanliness, not convergence.
constexpr int kMaskSize = 128;  // the size Georgiev & Fajardo used for their rendered images
constexpr int kMaskPixels = kMaskSize * kMaskSize;
constexpr std::array<std::uint16_t, kMaskPixels> kBlueNoiseRanks = {{
#include "blue_noise_mask.inc"
}};

// R2 (Roberts 2018): frac(c * (1/phi2, 1/phi2^2)). A hash lands offsets too close, measured |r| = 0.14 between channels 8 and 23.
constexpr std::uint32_t kR2AlphaX = 0xC13FA9A9U;
constexpr std::uint32_t kR2AlphaY = 0x91E10DA6U;
// Keeps the top log2(kMaskSize) bits of the fixed-point fraction, so the offset follows kMaskSize rather than a literal.
constexpr int kR2Shift = 32 - std::countr_zero(static_cast<unsigned>(kMaskSize));

// The shift in the sampler's 24-bit output space, exactly representable, which lets sampler_validate invert it at zero tolerance.
std::uint32_t ditherFixed(int pixelX, int pixelY, int ditherChannel) {
    const auto channel = static_cast<std::uint32_t>(ditherChannel);
    const auto x = (static_cast<std::uint32_t>(pixelX) + ((channel * kR2AlphaX) >> kR2Shift)) & (kMaskSize - 1U);
    const auto y = (static_cast<std::uint32_t>(pixelY) + ((channel * kR2AlphaY) >> kR2Shift)) & (kMaskSize - 1U);
    return (static_cast<std::uint32_t>(kBlueNoiseRanks[(y * kMaskSize) + x]) << 10U) | 512U;
}

// One dimension set's output: 24 significant bits of the scrambled point, shifted by this pixel's dither for that set.
float toShiftedUnitFloat(std::uint32_t x, std::uint32_t dither) {
    return static_cast<float>(((x >> 8) + dither) & 0xFFFFFFU) * 0x1.0p-24F;
}

}  // namespace

std::uint32_t replicateSeed(std::uint32_t scrambleSeed, int replicate) {
    // SplitMix64 of r alone is 0 at r = 0 and avalanched otherwise; XOR is the kAoSeedOffset idiom for an independent stream.
    return scrambleSeed ^ static_cast<std::uint32_t>(hashSeed(replicate, 0U));
}

float blueNoiseDither(int pixelX, int pixelY, int ditherChannel) {
    return static_cast<float>(ditherFixed(pixelX, pixelY, ditherChannel)) * 0x1.0p-24F;
}

// The index mask bounds the sequence to the length drawn: 62 ns/draw unmasked, 14 masked, at 128 samples. sampleCount <= 0 = unbounded.
Sampler::Sampler(int pixelX, int pixelY, int sampleIndex, int sampleCount, std::uint32_t scrambleSeed)
    : pixelX_(pixelX),
      pixelY_(pixelY),
      sampleIndex_(static_cast<std::uint32_t>(sampleIndex)),
      indexMask_(sampleCount > 0 ? std::bit_ceil(static_cast<std::uint32_t>(std::max(sampleCount, sampleIndex + 1))) - 1U
                                 : 0xFFFFFFFFU),
      scrambleSeed_(scrambleSeed) {}

float Sampler::next1D() {
    // Two seeds per set from the halves of one avalanched hash: SplitMix64 already decorrelates them, as kAoSeedOffset also rests on.
    const int set = dimensionSet_;
    const std::uint64_t setHash = hashSeed(2 * set, scrambleSeed_);
    ++dimensionSet_;
    // Shuffling the index per set (Burley 2020 Sec. 5.2) decorrelates sets; Owen-scrambling it keeps a power-of-two prefix stratified.
    const auto indexSeed = static_cast<std::uint32_t>(setHash);
    const auto scramble = static_cast<std::uint32_t>(setHash >> 32U);
    const std::uint32_t shuffled = nestedUniformScramble(sampleIndex_, indexSeed) & indexMask_;
    return toShiftedUnitFloat(nestedUniformScramble(sobolPoint(shuffled, 0), scramble),
                               ditherFixed(pixelX_, pixelY_, 2 * set));
}

glm::vec2 Sampler::next2D() {
    // Both coordinates share one shuffled index, being one 2D point; their Owen scrambles differ, per Owen's independence rule (1995).
    const int set = dimensionSet_;
    const std::uint64_t setHash = hashSeed(2 * set, scrambleSeed_);
    const std::uint64_t scrambleHash = hashSeed((2 * set) + 1, scrambleSeed_);
    ++dimensionSet_;
    const auto indexSeed = static_cast<std::uint32_t>(setHash);
    const auto scrambleX = static_cast<std::uint32_t>(setHash >> 32U);
    const auto scrambleY = static_cast<std::uint32_t>(scrambleHash);
    const std::uint32_t shuffled = nestedUniformScramble(sampleIndex_, indexSeed) & indexMask_;
    return {toShiftedUnitFloat(nestedUniformScramble(sobolPoint(shuffled, 0), scrambleX),
                               ditherFixed(pixelX_, pixelY_, 2 * set)),
            toShiftedUnitFloat(nestedUniformScramble(sobolPoint(shuffled, 1), scrambleY),
                               ditherFixed(pixelX_, pixelY_, (2 * set) + 1))};
}

CdfSample invertCdf(const float* cdf, int count, float u) {
    // upper_bound finds the first entry > u; the bin before it is u's. cdf[0]==0 is never > u, so searching it is harmless.
    const float* it = std::upper_bound(cdf, cdf + count + 1, u);
    const int index = std::clamp(static_cast<int>(it - cdf) - 1, 0, count - 1);
    const float lo = cdf[index];
    const float hi = cdf[index + 1];
    const float fraction = hi > lo ? std::clamp((u - lo) / (hi - lo), 0.0F, 1.0F) : 0.5F;
    return {index, fraction};
}

}  // namespace pathtracer::scene
