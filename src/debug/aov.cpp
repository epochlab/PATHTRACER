#include "pathtracer/debug/aov.h"

#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/debug/colormap.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <string_view>
#include <vector>

namespace pathtracer::debug {

AovSource aovSource(AovId aov) {
    switch (aov) {
        // Accumulated by renderPathTraced into PathTraceResult's 10 lanes.
        case AovId::Beauty:
        case AovId::BounceCount:
        case AovId::AO:
        case AovId::Shadow:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::Fresnel:
            return AovSource::PathTraced;

        // Image-space filters over a finished Beauty, so they need light transport but are not lanes of it.
        case AovId::HSV:
        case AovId::Luminance:
        case AovId::Sobel:
        case AovId::Gabor:
        case AovId::DoG:
        case AovId::ColourOpponent:
        case AovId::SNR:
            return AovSource::BeautyFilter;

        // The 15 primary-hit lanes renderRasterGBuffer scan-converts. No default: -Werror makes an unclassified AovId a compile error.
        case AovId::Wireframe:
        case AovId::Alpha:
        case AovId::Depth:
        case AovId::Lookahead:
        case AovId::WorldPos:
        case AovId::UV:
        case AovId::MotionVector:
        case AovId::Normal:
        case AovId::GeomNormal:
        case AovId::Albedo:
        case AovId::Metallic:
        case AovId::Roughness:
        case AovId::Tangent:
        case AovId::ObjectID:
        case AovId::IOR:
        case AovId::Count:
            return AovSource::GBuffer;
    }
    return AovSource::GBuffer;
}

int aovChannels(AovId aov) {
    switch (aov) {
        // Scalars, broadcast to RGB for display but carrying one value: a depth, an occlusion fraction, a filter response.
        case AovId::Depth:
        case AovId::Lookahead:
        case AovId::Alpha:
        case AovId::Metallic:
        case AovId::Roughness:
        case AovId::IOR:
        case AovId::AO:
        case AovId::Shadow:
        case AovId::BounceCount:
        case AovId::SNR:
        case AovId::Luminance:
        case AovId::Sobel:
        case AovId::Gabor:
        case AovId::DoG:
            return 1;

        // Two-component lanes: UV and image-plane motion have no third axis; Colour Opponent spans the two cardinal chromatic axes only.
        case AovId::UV:
        case AovId::MotionVector:
        case AovId::ColourOpponent:
            return 2;

        // Radiance triples, world-space vectors and the two false-coloured lanes, all three-channel.
        case AovId::Beauty:
        case AovId::HSV:
        case AovId::WorldPos:
        case AovId::Normal:
        case AovId::GeomNormal:
        case AovId::Albedo:
        case AovId::Tangent:
        case AovId::ObjectID:
        case AovId::Wireframe:
        case AovId::Fresnel:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::Count:
            return 3;
    }
    return 3;
}

bool aovCarriesRadiance(AovId aov) {
    switch (aov) {
        // Radiance lanes and the filters that are positively homogeneous of degree one in it, so a gain commutes with the filter.
        case AovId::Beauty:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::Luminance:
        case AovId::Sobel:
        case AovId::Gabor:
        case AovId::DoG:
            return true;

        // Ratios, reflectances, counts, lengths, directions and frequencies: scaling any of them by an exposure means nothing.
        case AovId::HSV:
        case AovId::ColourOpponent:
        case AovId::MotionVector:
        case AovId::Wireframe:
        case AovId::Alpha:
        case AovId::Depth:
        case AovId::Lookahead:
        case AovId::WorldPos:
        case AovId::UV:
        case AovId::Normal:
        case AovId::GeomNormal:
        case AovId::Albedo:
        case AovId::Metallic:
        case AovId::Roughness:
        case AovId::Tangent:
        case AovId::ObjectID:
        case AovId::AO:
        case AovId::Fresnel:
        case AovId::IOR:
        case AovId::BounceCount:
        case AovId::SNR:
        case AovId::Shadow:
        case AovId::Count:
            return false;
    }
    return false;
}

bool aovIsBipolar(AovId aov) {
    switch (aov) {
        // Responses of a zero-mean operator: zero is the operator's own centre and the two signs are equally meaningful.
        case AovId::DoG:
        case AovId::ColourOpponent:
            return true;

        // Signed but not bipolar: a normal spans a sphere; motion's two signed lanes share one unit, so it takes its own display.
        case AovId::MotionVector:
        case AovId::Beauty:
        case AovId::Wireframe:
        case AovId::Alpha:
        case AovId::Depth:
        case AovId::Lookahead:
        case AovId::HSV:
        case AovId::Luminance:
        case AovId::Sobel:
        case AovId::Gabor:
        case AovId::WorldPos:
        case AovId::UV:
        case AovId::Normal:
        case AovId::GeomNormal:
        case AovId::Albedo:
        case AovId::Metallic:
        case AovId::Roughness:
        case AovId::Tangent:
        case AovId::ObjectID:
        case AovId::AO:
        case AovId::Fresnel:
        case AovId::IOR:
        case AovId::BounceCount:
        case AovId::SNR:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::Shadow:
        case AovId::Count:
            return false;
    }
    return false;
}

namespace {

// Symmetric display range over the pooled samples of `lanes`, so lanes sharing one unit share one scale.
[[nodiscard]] float pooledRange(const pathtracer::gfx::HdrImage& image, std::span<const int> lanes) {
    const auto stride = static_cast<std::size_t>(image.channels);
    double sumOfSquares = 0.0;
    float peak = 0.0F;
    std::size_t count = 0;
    for (const int lane : lanes) {
        for (std::size_t sample = static_cast<std::size_t>(lane); sample < image.texels.size(); sample += stride) {
            const float value = image.texels[sample];
            sumOfSquares += static_cast<double>(value) * static_cast<double>(value);
            peak = std::max(peak, std::fabs(value));
            ++count;
        }
    }
    if (count == 0) {
        return 0.0F;
    }
    // Zero is the operator's own centre, so the scale is the RMS about zero rather than a sample deviation about an estimated mean.
    const double rms = std::sqrt(sumOfSquares / static_cast<double>(count));
    // Cramer 1946: the max of n iid normals concentrates at sigma*sqrt(2 ln n), Donoho-Johnstone's universal threshold at this RMS.
    const double expectedMaximum = rms * std::sqrt(2.0 * std::log(static_cast<double>(count)));
    // The smaller of the two: a Gaussian field keeps its true peak, and a heavy-tailed one stops a lone outlier crushing the preview.
    return std::min(peak, static_cast<float>(expectedMaximum));
}

// A lane with no range has no scale to fit, and every finite gain then reads mid-grey, so unity is the choice that assumes least.
[[nodiscard]] float bipolarGain(float range) {
    return range > 0.0F ? kBipolarDisplayOffset / range : 1.0F;
}

}  // namespace

BipolarDisplay bipolarDisplay(const pathtracer::gfx::HdrImage& image) {
    // Ranged per lane, because each lane is its own operator in its own unit: sharing one range would crush the narrower axis to nothing.
    BipolarDisplay display{glm::vec3(0.0F), glm::vec3(0.0F)};
    // Texture's swizzle broadcasts a scalar to all three display lanes, so channel 0's range is theirs, not absences left at zero gain.
    const bool scalar = image.channels == pathtracer::gfx::kScalarChannels;
    const int lanes = scalar ? glm::vec3::length() : image.channels;
    const std::array<int, 1> firstLane{0};
    const float scalarRange = scalar ? pooledRange(image, firstLane) : 0.0F;
    for (int lane = 0; lane < lanes; ++lane) {
        const std::array<int, 1> ownLane{lane};
        display.gain[lane] = bipolarGain(scalar ? scalarRange : pooledRange(image, ownLane));
        display.offset[lane] = kBipolarDisplayOffset;
    }
    return display;
}

namespace {

// SNR's preview is decibels, the domain-standard unit for a ratio of like-dimensioned amplitudes, so the factor is 20 (EMVA 1288).
void mapSnrForDisplay(const pathtracer::gfx::HdrImage& snr, int samples, pathtracer::gfx::HdrImage& out) {
    // From 0 dB, the ratio 1 where a texel equals its own uncertainty, to 10*log10(n), the ratio sqrt(n); the shared factor cancels.
    const double inverseCeiling = 2.0 / std::log(static_cast<double>(samples));
    out = pathtracer::gfx::makeImage(snr.width, snr.height, pathtracer::gfx::kScalarChannels);
    for (std::size_t pixel = 0; pixel < snr.texels.size(); ++pixel) {
        // snrAov stores the ratio as its one channel; at or below the floor nothing is resolved, which absorbs its zero too.
        const float ratio = snr.texels[pixel];
        // sqrt(n) is the SNR at unit per-sample coefficient of variation, since SNR = sqrt(n)*mu/sigma, so that reference is white.
        out.texels[pixel] = static_cast<float>(
            ratio > 1.0F ? std::min(std::log(static_cast<double>(ratio)) * inverseCeiling, 1.0) : 0.0);
    }
}

// Bounce Count's domain is a real bound -- maxBounces + 1 possible terminations -- so the colormap's argument is exact and frame-stable.
void mapBounceCountForDisplay(const pathtracer::gfx::HdrImage& bounces, int maxBounces, pathtracer::gfx::HdrImage& out) {
    const float ceiling = static_cast<float>(maxBounces) + 1.0F;
    out = pathtracer::gfx::makeImage(bounces.width, bounces.height, pathtracer::gfx::kRgbChannels);
    for (std::size_t pixel = 0; pixel < bounces.texels.size(); ++pixel) {
        const glm::vec3 mapped = turbo(bounces.texels[pixel] / ceiling);
        float* texel = out.texels.data() + (pixel * pathtracer::gfx::kRgbChannels);
        texel[0] = mapped.r;
        texel[1] = mapped.g;
        texel[2] = mapped.b;
    }
}

}  // namespace

AovDisplay aovDisplay(AovId aov, const pathtracer::gfx::HdrImage& image, const AovDisplayContext& context) {
    AovDisplay display{{}, {glm::vec3(1.0F), glm::vec3(0.0F)}};
    if (aovIsBipolar(aov)) {
        display.affine = bipolarDisplay(image);
        return display;
    }
    switch (aov) {
        // Raw metres quantize to white and farClip is a ray bound, not a depth span, so Depth ranges to the depth actually present.
        case AovId::Depth: {
            float maxDepth = 0.0F;
            for (const float depth : image.texels) {
                maxDepth = std::max(maxDepth, depth);
            }
            display.affine.gain = glm::vec3(std::pow(2.0F, -std::log2(std::max(maxDepth, 1e-4F))));
            return display;
        }
        // A variance is undefined below two passes, where snrAov is uniformly zero and the log window has no positive ceiling.
        case AovId::SNR:
            if (context.samples >= 2) {
                mapSnrForDisplay(image, context.samples, display.mapped);
            }
            return display;
        case AovId::BounceCount:
            mapBounceCountForDisplay(image, context.maxBounces, display.mapped);
            return display;
        // dx and dy share the pixel, so one pooled range keeps the direction; the swizzle's zero third lane stays black.
        case AovId::MotionVector: {
            const std::array<int, 2> displacementLanes{0, 1};
            const float gain = bipolarGain(pooledRange(image, displacementLanes));
            display.affine.gain = glm::vec3(gain, gain, 0.0F);
            display.affine.offset = glm::vec3(kBipolarDisplayOffset, kBipolarDisplayOffset, 0.0F);
            return display;
        }
        // Every other AOV's values reach the display unchanged, under the unity gain and zero offset this was initialised with.
        default:
            return display;
    }
}

PathTracedLane pathTracedLane(AovId aov) {
    using Result = pathtracer::scene::PathTraceResult;
    switch (aov) {
        case AovId::Beauty:           return &Result::beauty;
        case AovId::BounceCount:      return &Result::bounceHeatmap;
        case AovId::AO:               return &Result::ao;
        case AovId::Shadow:           return &Result::shadow;
        case AovId::DirectDiffuse:    return &Result::directDiffuse;
        case AovId::IndirectDiffuse:  return &Result::indirectDiffuse;
        case AovId::DirectSpecular:   return &Result::directSpecular;
        case AovId::IndirectSpecular: return &Result::indirectSpecular;
        case AovId::Refraction:       return &Result::refraction;
        case AovId::Fresnel:          return &Result::fresnel;
        default:                      return nullptr;
    }
}

GBufferLane gbufferLane(AovId aov) {
    using GBuffer = pathtracer::scene::RasterGBuffer;
    switch (aov) {
        case AovId::IOR:        return &GBuffer::iorAov;
        case AovId::Depth:      return &GBuffer::depth;
        case AovId::Lookahead:  return &GBuffer::lookahead;
        case AovId::WorldPos:   return &GBuffer::worldPos;
        case AovId::UV:         return &GBuffer::uv;
        case AovId::MotionVector: return &GBuffer::motionVector;
        case AovId::Normal:     return &GBuffer::normal;
        case AovId::GeomNormal: return &GBuffer::geomNormal;
        case AovId::Albedo:     return &GBuffer::albedo;
        case AovId::Metallic:   return &GBuffer::metallic;
        case AovId::Roughness:  return &GBuffer::roughness;
        case AovId::Tangent:    return &GBuffer::tangent;
        case AovId::ObjectID:   return &GBuffer::objectId;
        case AovId::Alpha:      return &GBuffer::alpha;
        case AovId::Wireframe:  return &GBuffer::wireframe;
        default:                return nullptr;
    }
}

namespace {

// The non-null images of laneOf over every AovId, in AovId order.
template <typename Lane>
std::vector<Lane> collectLanes(Lane (*laneOf)(AovId)) {
    std::vector<Lane> lanes;
    for (int i = 0; i < static_cast<int>(AovId::Count); ++i) {
        if (const Lane lane = laneOf(static_cast<AovId>(i))) {
            lanes.push_back(lane);
        }
    }
    return lanes;
}

}  // namespace

std::span<const PathTracedLane> pathTracedLanes() {
    static const std::vector<PathTracedLane> lanes = collectLanes(&pathTracedLane);
    return lanes;
}

std::span<const GBufferLane> gbufferLanes() {
    static const std::vector<GBufferLane> lanes = collectLanes(&gbufferLane);
    return lanes;
}

AovId aovIdFromName(std::string_view name) {
    for (int i = 0; i < static_cast<int>(AovId::Count); ++i) {
        if (name == kAovNames[i]) {
            return static_cast<AovId>(i);
        }
    }
    return AovId::Count;
}

}  // namespace pathtracer::debug
