#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/debug/render_stats.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/light.h"
#include "pathtracer/scene/material.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::scene {

// Square destination tiles, one owner each; splatting crosses pixel bounds. Halo re-traces 4.2% here vs 6.3% at 64, and 128 ties with 96.
inline constexpr int kPathTraceTileSize = 96;
// Floor for the resolution-derived size: the halo re-trace is (4t+4)/t^2, 4.3% at 96 but 12.9% at 32, so balance stops paying below it.
inline constexpr int kMinPathTraceTileSize = 32;
// Tiles per worker the split aims for: two leaves no worker idle and bounds the straggler tail at half a worker's share, without more halo.
inline constexpr int kTilesPerThread = 2;

// Tile edge renderPathTraced splits `width` x `height` into for `threadCount` workers: kPathTraceTileSize unless that leaves the pool idle.
[[nodiscard]] int pathTraceTileSize(int width, int height, unsigned int threadCount);

// samplesPerPixel is per renderPathTraced() call -- under PathTraceDriver, samples per accumulated pass (typically 1), not the total.
struct PathTraceSettings {
    int samplesPerPixel;
    int maxBounces;  // secondary/indirect bounces beyond the always-traced primary hit; 0 = direct lighting only
    int russianRouletteStartBounce;
    // AO ray bound and obscurance falloff scale (Zhukov 1998; Iones 2003): a hit at t gives 1-(1-t/aoMaxDistance)^2 visibility.
    float aoMaxDistance = 1.0F;
    // Horizon of the Lookahead AOV ramp (gbuffer.h), scene units: ray distance maps linearly to 1 at the camera and 0 here.
    float lookaheadDistance = 10.0F;
    float rrMinProb = 0.05F;  // floor: stops a near-zero-throughput path being killed with near-certainty
    // Exactly 1.0: a path at full throughput must never be terminated; a lower cap costs more in variance than the deep paths it saves.
    float rrMaxProb = 1.0F;
    // From the scene's material file (SceneConfig::materialPath) -- see resolveRoughness/buildShadingFrame.
    float bumpStrength;
    float roughnessMin;
    float roughnessMax;
    // Global material override from the scene's material file (SceneConfig::materialPath) -- see resolveBsdfParams.
    glm::vec3 diffuseColour;
    float ior;
    float abbe = 0.0F;  // Abbe number pairing with ior for dispersion; 0 = none, see bsdf.h's cauchyIor
    float transmissionFactor;
    float metallicFactor;
    float roughnessFactor;
    float diffuseRoughness;  // EON rough-diffuse parameter r in [0,1]; 0 = Lambertian, see bsdf.h's BsdfParams
    // The transmission lobe's only tint (transmissionFactor>0), realized in volume or on surface by transmissionDepth. See bsdf.h.
    glm::vec3 transmissionColor = glm::vec3(1.0F);
    float transmissionDepth = 0.0F;
    // Gulbrandsen 2014 edgetint for the conductor lobe; 1 = white edge, see bsdf.h's BsdfParams.
    glm::vec3 edgeTint = glm::vec3(1.0F);
    // Constant short-circuits tracePath to emit resolveBaseColor and terminate; see material.h's ShadingModel.
    ShadingModel shadingModel = ShadingModel::Standard;
};

// Stops either side of unity the readout is exact over: the next binade above the +/-13.95 EV the exposure controls reach.
inline constexpr int kOverRangeEvRadius = 16;
inline constexpr float kOverRangeMin = 1.0F / static_cast<float>(1U << kOverRangeEvRadius);

// Mantissa bits above the shift, 1024 bins/stop. Error 2.6/0.65/0.18/0.047 pp at 4/6/8/10 bits, cornell 2048x1152.
inline constexpr int kOverRangeSubBinBits = 10;
inline constexpr int kOverRangeBinShift = 23 - kOverRangeSubBinBits;  // float32 carries 23 mantissa bits
inline constexpr int kOverRangeBinCount = 2 * kOverRangeEvRadius * (1 << kOverRangeSubBinBits);
inline constexpr int kOverRangeBinOrigin =
    static_cast<int>(std::bit_cast<std::uint32_t>(kOverRangeMin) >> kOverRangeBinShift);

// Positive floats are monotone under integer bit compare, so a shift bins them; the clamp folds -0 to bin 0 and NaN/inf to the top bin.
[[nodiscard]] inline int overRangeBin(float value) {
    if (std::signbit(value)) {
        return 0;
    }
    const auto shifted = static_cast<int>(std::bit_cast<std::uint32_t>(value) >> kOverRangeBinShift);
    return std::clamp(shifted - kOverRangeBinOrigin, 0, kOverRangeBinCount - 1);
}

using OverRangeHistogram = std::array<std::uint32_t, kOverRangeBinCount>;

// Exposure-free reduction of `beauty` for the HUD's over-range readout, on the driver thread, so a slider drag stays O(1) render-side.
struct OverRangeStats {
    float rawPeak = 0.0F;  // max over texels of max(R,G,B) before exposure; exact, only the fraction is quantised
    // Complementary CDF, not the histogram: aboveBin[b] counts texels with bin >= b, so aboveBin[0] is the texel count, the last 0.
    std::array<std::uint32_t, kOverRangeBinCount + 1> aboveBin{};
};

// One renderPathTraced() call's output, republished in full each pass. Per-sample quantities, each lane at its aovChannels count.
struct PathTraceResult {
    pathtracer::gfx::HdrImage beauty;
    pathtracer::gfx::HdrImage bounceHeatmap;   // mean bounce depth at termination, across samples
    // Cosine-weighted obscurance (Zhukov 1998; Iones 2003), one bounded ray/sample. 1.0 = unoccluded, the OPPOSITE polarity to `shadow`.
    pathtracer::gfx::HdrImage ao;
    // Bounce-0 shadow: 1.0 on a surface unless the one NEE shadow ray was unoccluded, 0.0 on background; stays 1.0 with no NEE sample.
    pathtracer::gfx::HdrImage shadow;

    // These five plus the background sum to beauty per channel, bucketed by the bounce-0 lobe; transmission sticks to refraction.
    pathtracer::gfx::HdrImage directDiffuse;
    pathtracer::gfx::HdrImage indirectDiffuse;
    pathtracer::gfx::HdrImage directSpecular;
    pathtracer::gfx::HdrImage indirectSpecular;
    pathtracer::gfx::HdrImage refraction;

    // E[F(wo.wh)] at the primary hit, wh ~ D_vis(wo), one VNDF draw per sample (bsdf.h). 0 where bounce 0 has no BSDF vertex.
    pathtracer::gfx::HdrImage fresnel;

    // Welford second moment of each texel's per-pass Rec.709 luminance, one float and not an image: an SNR needs no chromaticity.
    std::vector<float> beautyLuminanceM2;

    // Reduced from `beauty` immediately before publish, so it describes exactly the pixels published with it, unlike PassRecord.
    OverRangeStats overRange;
    std::uint64_t generation = 0;  // the request whose accumulation this is, stamped by PathTraceDriver at publish; 0 from renderPathTraced
    bool wrapsHorizontally = false;  // the lens it was traced through joins its left and right edges, so neighbour reads wrap in x
    int samples = 0;  // passes averaged in, stamped with generation so image and count publish as one snapshot
};

// All 10 images at their aovChannels counts and the second-moment lane, zeroed at width x height: renderPathTraced's `out`.
[[nodiscard]] PathTraceResult makePathTraceResult(int width, int height);

// Whether a primary ray that misses returns environment radiance when nothing overrides it; viewer, driver and headless share it.
inline constexpr bool kDefaultShowSky = false;

// Blocking multithreaded path trace: BSDF bounces, NEE with MIS, RR; a worker drops its tile when generation != requestedGeneration.
void renderPathTraced(const Camera& camera, const EmbreeAccel& accel,
                       const std::vector<ShadingTriangle>& shadingTriangles,
                       const std::vector<MeshInstance>& instances,
                       const std::vector<int>& instanceLightIndex, const LightSet& lights,
                       int width, int height, bool showSky, const PathTraceSettings& settings,
                       const std::vector<PathTraceSettings>& perInstanceSettings,
                       std::uint32_t scrambleSeed, int sampleBase, int sampleCount,
                       const std::atomic<std::uint64_t>& generation,
                       std::uint64_t requestedGeneration, ThreadPool& threadPool,
                       pathtracer::debug::PassStats& stats, PathTraceResult& out);

}  // namespace pathtracer::scene
