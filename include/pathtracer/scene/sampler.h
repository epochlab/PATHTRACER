#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace pathtracer::scene {

// Owen-scrambled shuffled Sobol (Burley 2020), pixels taking blocks in scrambled Morton order (Ahmed & Wonka 2020): blue-noise error.
class Sampler {
public:
    // sampleIndex < sampleCount advances per sample; sampleCount <= 0 is unbounded; scrambleSeed is fixed across an accumulation.
    Sampler(int pixelX, int pixelY, int sampleIndex, int sampleCount, std::uint32_t scrambleSeed);
    // Each call consumes one padded dimension set. next2D's pair is jointly stratified; two next1D calls are not.
    [[nodiscard]] float next1D();
    [[nodiscard]] glm::vec2 next2D();
    // The same pixel block from set 0 under an independent scramble: a sample's streams share one Morton rank, computed once.
    [[nodiscard]] Sampler stream(std::uint32_t streamOffset) const;

private:
    std::uint32_t index_;  // the pixel's block of its tile's sequence, offset by sampleIndex
    std::uint32_t indexMask_;  // the tile's sequence length less one, keeping a shuffled index inside the tile
    std::uint32_t seed_;  // the tile's set-seed base
    int dimensionSet_ = 0;
};

// Replicate r's scramble seed (Owen 1997 replicated RQMC): independent scrambles per r, and the identity at r = 0 so K = 1 is one sequence.
[[nodiscard]] std::uint32_t replicateSeed(std::uint32_t scrambleSeed, int replicate);

// Sobol dimension 1 (Joe & Kuo's degree-1 row) bit-reversed: digit j is the XOR of index bits b with C(b, j) odd, Lucas's b superset j.
[[nodiscard]] std::uint32_t reversedSobolDimension1(std::uint32_t index);

// A piecewise-constant CDF inverted at u: the bin and u's fractional position within that bin's mass.
struct CdfSample {
    int index;
    float fraction;  // [0,1] position within the selected bin
};

// cdf is count + 1 non-decreasing knots from 0 to 1; a zero-mass bin returns its midpoint.
[[nodiscard]] CdfSample invertCdf(const float* cdf, int count, float u);

}  // namespace pathtracer::scene
