#include "pathtracer/debug/aov.h"

#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/debug/colormap.h"

#include <cctype>
#include <algorithm>
#include <cmath>
#include <string>
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

        // The 14 primary-hit lanes renderRasterGBuffer scan-converts. No default: -Werror makes an unclassified AovId a compile error.
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

        // Two-component lanes: UV's third channel is structurally zero, and Colour Opponent spans the two cardinal chromatic axes only.
        case AovId::UV:
        case AovId::ColourOpponent:
            return 2;

        // Radiance triples, world-space vectors and the two false-coloured lanes, all needing three channels, not a broadcast scalar.
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

        // Direction and position lanes are signed but not bipolar: a normal's components span a sphere, not a response about zero.
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

BipolarDisplay bipolarDisplay(std::span<const float> rgba, int channels) {
    const auto laneRange = [rgba](int lane) {
        double sumOfSquares = 0.0;
        float peak = 0.0F;
        std::size_t count = 0;
        for (std::size_t sample = static_cast<std::size_t>(lane); sample < rgba.size(); sample += 4) {
            const float value = rgba[sample];
            sumOfSquares += static_cast<double>(value) * static_cast<double>(value);
            peak = std::max(peak, std::fabs(value));
            ++count;
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
    };

    // Ranged per lane, because each lane is its own operator in its own unit: sharing one range would crush the narrower axis to nothing.
    BipolarDisplay display{glm::vec3(0.0F), glm::vec3(0.0F)};
    // writeScalar broadcasts a scalar AOV to all three lanes: replicas, so lane 0's range is theirs, not absences left at zero gain.
    const int lanes = channels == 1 ? 3 : std::min(channels, 3);
    const float scalarRange = channels == 1 ? laneRange(0) : 0.0F;
    for (int lane = 0; lane < lanes; ++lane) {
        const float range = channels == 1 ? scalarRange : laneRange(lane);
        // A lane with no range has no scale to fit, and every finite gain then reads mid-grey, so unity is the choice that assumes least.
        display.gain[lane] = range > 0.0F ? kBipolarDisplayOffset / range : 1.0F;
        display.offset[lane] = kBipolarDisplayOffset;
    }
    return display;
}

namespace {

// SNR's preview is decibels, the domain-standard unit for a ratio of like-dimensioned amplitudes, so the factor is 20 (EMVA 1288).
void mapSnrForDisplay(std::span<const float> rgba, int samples, std::vector<float>& out) {
    // From 0 dB, the ratio 1 where a texel equals its own uncertainty, to 10*log10(n), the ratio sqrt(n); the shared factor cancels.
    const double inverseCeiling = 2.0 / std::log(static_cast<double>(samples));
    out.resize(rgba.size());
    for (std::size_t texel = 0; texel + 3 < rgba.size(); texel += 4) {
        // snrAov broadcasts its scalar, so lane 0 is the ratio; at or below the floor nothing is resolved, which absorbs its zero too.
        const float ratio = rgba[texel];
        // sqrt(n) is the SNR at unit per-sample coefficient of variation, since SNR = sqrt(n)*mu/sigma, so that reference is white.
        const auto grey = static_cast<float>(
            ratio > 1.0F ? std::min(std::log(static_cast<double>(ratio)) * inverseCeiling, 1.0) : 0.0);
        out[texel] = grey;
        out[texel + 1] = grey;
        out[texel + 2] = grey;
        out[texel + 3] = rgba[texel + 3];
    }
}

// Bounce Count's domain is a real bound -- maxBounces + 1 possible terminations -- so the colormap's argument is exact and frame-stable.
void mapBounceCountForDisplay(std::span<const float> rgba, int maxBounces, std::vector<float>& out) {
    const float ceiling = static_cast<float>(maxBounces) + 1.0F;
    out.resize(rgba.size());
    for (std::size_t texel = 0; texel + 3 < rgba.size(); texel += 4) {
        const glm::vec3 mapped = turbo(rgba[texel] / ceiling);
        out[texel] = mapped.r;
        out[texel + 1] = mapped.g;
        out[texel + 2] = mapped.b;
        out[texel + 3] = rgba[texel + 3];
    }
}

}  // namespace

AovDisplay aovDisplay(AovId aov, std::span<const float> rgba, const AovDisplayContext& context) {
    AovDisplay display{{}, {glm::vec3(1.0F), glm::vec3(0.0F)}};
    if (aovIsBipolar(aov)) {
        display.affine = bipolarDisplay(rgba, aovChannels(aov));
        return display;
    }
    switch (aov) {
        // Raw metres quantize to white and farClip is a ray bound, not a depth span, so Depth ranges to the depth actually present.
        case AovId::Depth: {
            float maxDepth = 0.0F;
            for (std::size_t texel = 0; texel < rgba.size(); texel += 4) {
                maxDepth = std::max(maxDepth, rgba[texel]);
            }
            display.affine.gain = glm::vec3(std::pow(2.0F, -std::log2(std::max(maxDepth, 1e-4F))));
            return display;
        }
        // A variance is undefined below two passes, where snrAov is uniformly zero and the log window has no positive ceiling.
        case AovId::SNR:
            if (context.samples >= 2) {
                mapSnrForDisplay(rgba, context.samples, display.rgba);
            }
            return display;
        case AovId::BounceCount:
            mapBounceCountForDisplay(rgba, context.maxBounces, display.rgba);
            return display;
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

// Drops separators and folds case, so a CLI flag, a C string from a foreign runtime and a HUD label all reduce to the same key.
[[nodiscard]] std::string normalizeAovName(std::string_view name) {
    std::string out;
    out.reserve(name.size());
    for (const char c : name) {
        if (c == ' ' || c == '-' || c == '_') {
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

}  // namespace

AovId aovIdFromName(std::string_view name) {
    const std::string wanted = normalizeAovName(name);
    for (int i = 0; i < static_cast<int>(AovId::Count); ++i) {
        if (normalizeAovName(kAovNames[i]) == wanted) {
            return static_cast<AovId>(i);
        }
    }
    return AovId::Count;
}

}  // namespace pathtracer::debug
