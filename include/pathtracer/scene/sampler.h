#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace pathtracer::scene {

// Owen-scrambled shuffled Sobol (Burley 2020), blue-noise dithered (Georgiev 2016).
class Sampler {
public:
    // sampleIndex advances per sample; scrambleSeed is fixed across an accumulation. Swapping the two degrades this to white noise.
    Sampler(int pixelX, int pixelY, int sampleIndex, int sampleCount, std::uint32_t scrambleSeed);
    // Each call consumes one padded dimension set. next2D's pair is jointly stratified; two next1D calls are not.
    [[nodiscard]] float next1D();
    [[nodiscard]] glm::vec2 next2D();

private:
    int pixelX_;
    int pixelY_;
    std::uint32_t sampleIndex_;
    std::uint32_t indexMask_;
    std::uint32_t scrambleSeed_;
    int dimensionSet_ = 0;
};

// Replicate r's scramble seed (Owen 1997 replicated RQMC): independent scrambles per r, and the identity at r = 0 so K = 1 is one sequence.
[[nodiscard]] std::uint32_t replicateSeed(std::uint32_t scrambleSeed, int replicate);

// The shift Sampler adds at (pixelX, pixelY) on one dither channel; sampler_validate undoes it exactly, both being multiples of 2^-24.
[[nodiscard]] float blueNoiseDither(int pixelX, int pixelY, int ditherChannel);

}  // namespace pathtracer::scene
