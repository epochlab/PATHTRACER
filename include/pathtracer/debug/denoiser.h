#pragma once

#include <memory>

#include "pathtracer/gfx/hdr_image.h"

namespace pathtracer::debug {

// OIDN's filter quality: High for final frames; Balanced and Fast give up detail for speed at interactive rates.
enum class DenoiseQuality { Fast, Balanced, High };
// profile.json's names for them, index-parallel with DenoiseQuality: OIDN's own, lower-cased as the lens projection's are.
inline constexpr const char* kDenoiseQualityNames[] = {"fast", "balanced", "high"};

// profile.json's denoiser settings OIDN itself takes.
struct DenoiserOptions {
    DenoiseQuality quality;
    bool prefilterAuxiliary;  // OIDN's final-frame path: true prefilters albedo and normal; false is its one-filter interactive path
    int numThreads;           // the CPU device's worker threads, 0 for OIDN's default of one per logical core
};

// Intel Open Image Denoise's RT filter on a CPU device: HDR Beauty guided by albedo and normal, prefiltered or taken as they are.
class Denoiser {
public:
    // Commits the CPU device with the options' thread count, or throws std::runtime_error carrying OIDN's own message.
    explicit Denoiser(const DenoiserOptions& options);
    ~Denoiser();

    Denoiser(const Denoiser&) = delete;
    Denoiser& operator=(const Denoiser&) = delete;
    Denoiser(Denoiser&&) = delete;
    Denoiser& operator=(Denoiser&&) = delete;

    // Denoised beauty from three equal-sized RGB images, which are only read; std::invalid_argument on a shape mismatch.
    [[nodiscard]] pathtracer::gfx::HdrImage denoise(const pathtracer::gfx::HdrImage& beauty, const pathtracer::gfx::HdrImage& albedo,
                                                    const pathtracer::gfx::HdrImage& normal);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pathtracer::debug
