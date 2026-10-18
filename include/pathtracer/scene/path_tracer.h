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
#include "pathtracer/scene/volume.h"

namespace pathtracer::scene {

// Square tiles, one owner each; pixels are independent under filter importance sampling, so the size sets only scheduling granularity.
inline constexpr int kPathTraceTileSize = 96;
// Tiles per worker the split aims for: two leaves no worker idle and bounds the straggler tail at half a worker's share.
inline constexpr int kTilesPerThread = 2;

// Blackman-Harris (Harris 1978) at Arnold's default 1.5 px radius, importance-sampled (Ernst et al. 2006): pdf proportional to f, weight 1.
inline constexpr float kFilterRadius = 1.5F;

// A direction a facet's acceptance lobe draws, and its density per steradian.
struct AcceptanceSample {
    glm::vec3 dir;
    float pdf;
};

// von Mises-Fisher about unit axis at kappa (Fisher 1953), weight 1 as Millward et al. 2022; Shirley-Chiu's map, Jakob 2012's inverse.
[[nodiscard]] AcceptanceSample sampleAcceptance(const glm::vec3& axis, float kappa, glm::vec2 u);

// The lobe's equivalent solid angle, 1 / its peak density 2 pi (1 - e^-2k) / k: pi rho^2 / (4 ln2) at small rho (Snyder 1977).
[[nodiscard]] float acceptanceSolidAngle(float kappa);

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
    // Independent scrambles K, interleaved by pass (replicated RQMC, Owen 1997): their spread is the SNR's error; 1 is one sequence.
    int replicates = 1;
    // OIDN's albedo and normal lanes; false leaves the chain unopened, so the lanes read 0 and the paths pay nothing for them.
    bool denoiseFeatures = false;
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
    // Cosine-weighted obscurance (Zhukov 1998; Iones 2003), one bounded ray/sample. 1.0 = unoccluded, the opposite polarity to Shadow.
    pathtracer::gfx::HdrImage ao;
    // Shadow's two means, no AOV of their own: bounce-0 direct luminance the occluders remove, and the same with visibility 1.
    pathtracer::gfx::HdrImage shadowLost;
    pathtracer::gfx::HdrImage shadowUnoccluded;

    // These nine plus the background sum to beauty per channel, bucketed by the bounce-0 lobe; transmission sticks to refraction.
    pathtracer::gfx::HdrImage directDiffuse;
    pathtracer::gfx::HdrImage indirectDiffuse;
    pathtracer::gfx::HdrImage directSpecular;
    pathtracer::gfx::HdrImage indirectSpecular;
    pathtracer::gfx::HdrImage refraction;
    // Paths whose bounce-0 vertex is a scattering in a volume, rather than a surface lobe.
    pathtracer::gfx::HdrImage directVolume;
    pathtracer::gfx::HdrImage indirectVolume;
    // Paths whose bounce-0 transmission enters a scattering interior; direct is light reaching that walk's boundary from a light.
    pathtracer::gfx::HdrImage directSubsurface;
    pathtracer::gfx::HdrImage indirectSubsurface;

    // E[F(wo.wh)] at the primary hit, wh ~ D_vis(wo), one VNDF draw per sample (bsdf.h). 0 where bounce 0 has no BSDF vertex.
    pathtracer::gfx::HdrImage fresnel;

    // Filtered primary opacity: opaque on a hit, a shown sky or a real collision; a primary escape through media is 1 - its mean weight.
    pathtracer::gfx::HdrImage alpha;

    // OIDN's auxiliary features from Beauty's own samples, past deltas and sub-pixel lobes: directional albedo and world shading normal.
    pathtracer::gfx::HdrImage denoiseAlbedo;
    pathtracer::gfx::HdrImage denoiseNormal;

    // Running mean of each replicate's per-pass scene luminance, `replicates` floats per texel, pixel-major: the SNR's error estimate.
    std::vector<float> replicateLuminance;
    int replicates = 1;

    // Reduced from `beauty` immediately before publish, so it describes exactly the pixels published with it, unlike PassRecord.
    OverRangeStats overRange;
    std::uint64_t generation = 0;  // the request whose accumulation this is, stamped by PathTraceDriver at publish; 0 from renderPathTraced
    bool wrapsHorizontally = false;  // the lens it was traced through joins its left and right edges, so neighbour reads wrap in x
    pathtracer::gfx::PixelLattice pixelLattice = pathtracer::gfx::PixelLattice::Square;  // where its texels sit on the film
    int samples = 0;  // passes averaged in, stamped with generation so image and count publish as one snapshot
};

// The 12 AOV lanes at their aovChannels counts, the 2 RGB denoise features, the 2 shadow means and the replicate means, zeroed.
[[nodiscard]] PathTraceResult makePathTraceResult(int width, int height, int replicates);

// Whether a primary ray that misses returns environment radiance when nothing overrides it; viewer, driver and headless share it.
inline constexpr bool kDefaultShowSky = false;

// Blocking multithreaded path trace: BSDF bounces, NEE with MIS, RR; a worker drops its tile when generation != requestedGeneration.
void renderPathTraced(const Camera& camera, const EmbreeAccel& accel,
                       const std::vector<ShadingTriangle>& shadingTriangles,
                       const std::vector<MeshInstance>& instances,
                       const std::vector<int>& instanceLightIndex, const LightSet& lights, const VolumeSet& volumes,
                       int width, int height, bool showSky, const PathTraceSettings& settings,
                       std::uint32_t scrambleSeed, int sampleBase, int sampleCount,
                       const std::atomic<std::uint64_t>& generation,
                       std::uint64_t requestedGeneration, ThreadPool& threadPool,
                       pathtracer::debug::PassStats& stats, PathTraceResult& out);

}  // namespace pathtracer::scene
