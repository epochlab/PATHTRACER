#include "pathtracer/debug/aov.h"

#include "pathtracer/debug/aov_filters.h"
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
        // Accumulated by renderPathTraced into PathTraceResult's 14 AOV lanes.
        case AovId::Beauty:
        case AovId::Alpha:
        case AovId::BounceCount:
        case AovId::AO:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::DirectVolume:
        case AovId::IndirectVolume:
        case AovId::DirectSubsurface:
        case AovId::IndirectSubsurface:
        case AovId::Fresnel:
            return AovSource::PathTraced;

        // Derived from finished accumulated lanes, Beauty or its companions, so they need light transport but are not lanes of it.
        case AovId::HSV:
        case AovId::Luminance:
        case AovId::Sobel:
        case AovId::Gabor:
        case AovId::ColourOpponent:
        case AovId::ContrastSensitivity:
        case AovId::SNR:
        case AovId::SampleCount:
        case AovId::Shadow:
        case AovId::Denoised:
            return AovSource::BeautyFilter;

        // The 14 primary-hit lanes renderGBuffer ray-casts. No default: -Werror makes an unclassified AovId a compile error.
        case AovId::Wireframe:
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
        case AovId::SampleCount:
        case AovId::Luminance:
        case AovId::Sobel:
        case AovId::Gabor:
        case AovId::ContrastSensitivity:
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
        case AovId::DirectVolume:
        case AovId::IndirectVolume:
        case AovId::DirectSubsurface:
        case AovId::IndirectSubsurface:
        case AovId::Denoised:
        case AovId::Count:
            return 3;
    }
    return 3;
}

bool aovCarriesRadiance(AovId aov) {
    switch (aov) {
        // Radiance lanes and the filters that are positively homogeneous of degree one in it, so a gain commutes with the filter.
        case AovId::Beauty:
        case AovId::Denoised:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::DirectVolume:
        case AovId::IndirectVolume:
        case AovId::DirectSubsurface:
        case AovId::IndirectSubsurface:
        case AovId::Luminance:
        case AovId::Sobel:
        case AovId::Gabor:
            return true;

        // Ratios, reflectances, counts, lengths, directions and frequencies: scaling any of them by an exposure means nothing.
        case AovId::HSV:
        case AovId::ColourOpponent:
        case AovId::ContrastSensitivity:
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
        case AovId::SampleCount:
        case AovId::Shadow:
        case AovId::Count:
            return false;
    }
    return false;
}

bool aovIsPicture(AovId aov) {
    return aov == AovId::Beauty || aov == AovId::Denoised;
}

bool aovTakesLogTransform(AovId aov) {
    return aovIsPicture(aov) || aov == AovId::Luminance;
}

bool aovIsColour(AovId aov) {
    switch (aov) {
        // Working-space RGB: radiance lanes and per-channel reflectances, whose values change with the primaries they are written in.
        case AovId::Beauty:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::DirectVolume:
        case AovId::IndirectVolume:
        case AovId::DirectSubsurface:
        case AovId::IndirectSubsurface:
        case AovId::Albedo:
        case AovId::Fresnel:
        case AovId::Denoised:
            return true;

        // Scalars, geometry, derived colour coordinates and false colours: no change of primaries applies to them.
        case AovId::Luminance:
        case AovId::Sobel:
        case AovId::Gabor:
        case AovId::HSV:
        case AovId::ColourOpponent:
        case AovId::ContrastSensitivity:
        case AovId::MotionVector:
        case AovId::Wireframe:
        case AovId::Alpha:
        case AovId::Depth:
        case AovId::Lookahead:
        case AovId::WorldPos:
        case AovId::UV:
        case AovId::Normal:
        case AovId::GeomNormal:
        case AovId::Metallic:
        case AovId::Roughness:
        case AovId::Tangent:
        case AovId::ObjectID:
        case AovId::AO:
        case AovId::IOR:
        case AovId::BounceCount:
        case AovId::SNR:
        case AovId::SampleCount:
        case AovId::Shadow:
        case AovId::Count:
            return false;
    }
    return false;
}

bool aovIsBipolar(AovId aov) {
    switch (aov) {
        // Responses of a zero-mean operator: zero is the operator's own centre and the two signs are equally meaningful.
        case AovId::ColourOpponent:
            return true;

        // Signed but not bipolar: a normal spans a sphere; motion's two signed lanes share one unit, so it takes its own display.
        case AovId::ContrastSensitivity:
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
        case AovId::SampleCount:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::DirectVolume:
        case AovId::IndirectVolume:
        case AovId::DirectSubsurface:
        case AovId::IndirectSubsurface:
        case AovId::Shadow:
        case AovId::Denoised:
        case AovId::Count:
            return false;
    }
    return false;
}

namespace {

// The peak, capped at Donoho-Johnstone's sigma*sqrt(2 ln n), where n iid normals' max concentrates (Cramer 1946): outliers can't crush it.
[[nodiscard]] float cappedRange(float peak, double rms, std::size_t count) {
    return std::min(peak, static_cast<float>(rms * std::sqrt(2.0 * std::log(static_cast<double>(count)))));
}

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
    return cappedRange(peak, std::sqrt(sumOfSquares / static_cast<double>(count)), count);
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

// log2 radiance about its mean, the receptor's adaptation level, as a bipolar signal: pooled over lanes, so their log ratios survive.
[[nodiscard]] BipolarDisplay logWindow(const pathtracer::gfx::HdrImage& image) {
    std::size_t count = 0;
    double mean = 0.0;
    double squaredDeviations = 0.0;  // Welford's M2: no cancellation against a mean many stops from zero
    double lowest = 0.0;
    double highest = 0.0;
    for (const float value : image.texels) {
        // Only a positive texel has a logarithm; the rest clamp below the window at the display's log stage.
        if (!(value > 0.0F)) {
            continue;
        }
        const double level = std::log2(static_cast<double>(value));
        lowest = count == 0 ? level : std::min(lowest, level);
        highest = count == 0 ? level : std::max(highest, level);
        ++count;
        const double delta = level - mean;
        mean += delta / static_cast<double>(count);
        squaredDeviations += delta * (level - mean);
    }
    const float peak = static_cast<float>(std::max(highest - mean, mean - lowest));
    const float range = count > 0 ? cappedRange(peak, std::sqrt(squaredDeviations / static_cast<double>(count)), count) : 0.0F;
    // Every lane shares one gain and offset, so a lane's log ratio to another, its chromaticity, reaches the display unscaled.
    const float gain = bipolarGain(range);
    return {glm::vec3(gain), glm::vec3(kBipolarDisplayOffset - (static_cast<float>(mean) * gain))};
}

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

// A count over its real bound, maxBounces + 1 terminations or the passes run, so the colormap's argument is exact and frame-stable.
void mapCountForDisplay(const pathtracer::gfx::HdrImage& counts, float ceiling, pathtracer::gfx::HdrImage& out) {
    out = pathtracer::gfx::makeImage(counts.width, counts.height, pathtracer::gfx::kRgbChannels);
    for (std::size_t pixel = 0; pixel < counts.texels.size(); ++pixel) {
        const glm::vec3 mapped = turbo(counts.texels[pixel] / ceiling);
        float* texel = out.texels.data() + (pixel * pathtracer::gfx::kRgbChannels);
        texel[0] = mapped.r;
        texel[1] = mapped.g;
        texel[2] = mapped.b;
    }
}

// Visibility on a log10 turbo ramp over [1/S, S], S Barten's ceiling: threshold at the centre, a full-contrast grating at the top.
void mapVisibilityForDisplay(const pathtracer::gfx::HdrImage& visibility, pathtracer::gfx::HdrImage& out) {
    const double inverseSpan = 1.0 / (2.0 * std::log10(bartenSensitivityBound()));
    out = pathtracer::gfx::makeImage(visibility.width, visibility.height, pathtracer::gfx::kRgbChannels);
    for (std::size_t pixel = 0; pixel < visibility.texels.size(); ++pixel) {
        // Nothing visible, V = 0, has no logarithm and takes the ramp's floor, as any V below 1/S does.
        const float value = visibility.texels[pixel];
        const glm::vec3 mapped = turbo(value > 0.0F ? static_cast<float>(0.5 + (std::log10(static_cast<double>(value)) * inverseSpan)) : 0.0F);
        float* texel = out.texels.data() + (pixel * pathtracer::gfx::kRgbChannels);
        texel[0] = mapped.r;
        texel[1] = mapped.g;
        texel[2] = mapped.b;
    }
}

}  // namespace

AovDisplay aovDisplay(AovId aov, const pathtracer::gfx::HdrImage& image, const AovDisplayContext& context) {
    AovDisplay display{{}, {glm::vec3(1.0F), glm::vec3(0.0F)}};
    if (context.logTransform) {
        display.affine = logWindow(image);
        return display;
    }
    if (aovIsBipolar(aov)) {
        display.affine = bipolarDisplay(image);
        return display;
    }
    switch (aov) {
        // Raw metres quantize to white and farClip is not a depth span, so the depth present is white; an all-miss frame keeps unity.
        case AovId::Depth:
            display.affine.gain = glm::vec3(context.depthMax > 0.0F ? 1.0F / context.depthMax : 1.0F);
            return display;
        // A variance is undefined below two passes, where snrAov is uniformly zero and the log window has no positive ceiling.
        case AovId::SNR:
            if (context.samples >= 2) {
                mapSnrForDisplay(image, context.samples, display.mapped);
            }
            return display;
        case AovId::BounceCount:
            mapCountForDisplay(image, static_cast<float>(context.maxBounces) + 1.0F, display.mapped);
            return display;
        // Full red is a texel traced every pass; no pass yet leaves nothing to scale by.
        case AovId::SampleCount:
            if (context.samples >= 1) {
                mapCountForDisplay(image, static_cast<float>(context.samples), display.mapped);
            }
            return display;
        case AovId::ContrastSensitivity:
            mapVisibilityForDisplay(image, display.mapped);
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
        case AovId::DirectDiffuse:    return &Result::directDiffuse;
        case AovId::IndirectDiffuse:  return &Result::indirectDiffuse;
        case AovId::DirectSpecular:   return &Result::directSpecular;
        case AovId::IndirectSpecular: return &Result::indirectSpecular;
        case AovId::Refraction:       return &Result::refraction;
        case AovId::DirectVolume:     return &Result::directVolume;
        case AovId::IndirectVolume:   return &Result::indirectVolume;
        case AovId::DirectSubsurface: return &Result::directSubsurface;
        case AovId::IndirectSubsurface: return &Result::indirectSubsurface;
        case AovId::Fresnel:          return &Result::fresnel;
        case AovId::Alpha:            return &Result::alpha;
        default:                      return nullptr;
    }
}

GBufferLane gbufferLane(AovId aov) {
    using GBuffer = pathtracer::scene::GBuffer;
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
    static const std::vector<PathTracedLane> lanes = [] {
        std::vector<PathTracedLane> all = collectLanes(&pathTracedLane);
        all.insert(all.end(), kDenoiseFeatureLanes.begin(), kDenoiseFeatureLanes.end());
        all.insert(all.end(), kShadowLanes.begin(), kShadowLanes.end());
        return all;
    }();
    return lanes;
}

std::span<const GBufferLane> gbufferLanes() {
    static const std::vector<GBufferLane> lanes = collectLanes(&gbufferLane);
    return lanes;
}

bool aovSupportsLattice(AovId aov, pathtracer::gfx::PixelLattice lattice) {
    if (lattice == pathtracer::gfx::PixelLattice::Square) {
        return true;
    }
    switch (aov) {
        // OIDN's network is trained on square images, so it has no meaning over offset rows.
        case AovId::Denoised:
            return false;

        // One texel each, a primary ray at the pixel's own centre, or a stencil with a hexagonal version.
        case AovId::Beauty:
        case AovId::Sobel:
        case AovId::Gabor:
        case AovId::Wireframe:
        case AovId::Alpha:
        case AovId::Depth:
        case AovId::Lookahead:
        case AovId::HSV:
        case AovId::Luminance:
        case AovId::WorldPos:
        case AovId::UV:
        case AovId::MotionVector:
        case AovId::ColourOpponent:
        case AovId::ContrastSensitivity:
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
        case AovId::SampleCount:
        case AovId::DirectDiffuse:
        case AovId::IndirectDiffuse:
        case AovId::DirectSpecular:
        case AovId::IndirectSpecular:
        case AovId::Refraction:
        case AovId::DirectVolume:
        case AovId::IndirectVolume:
        case AovId::DirectSubsurface:
        case AovId::IndirectSubsurface:
        case AovId::Shadow:
        case AovId::Count:
            break;
    }
    return true;
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
