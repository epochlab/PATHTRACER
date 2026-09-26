#include "pathtracer/debug/aov.h"

#include "pathtracer/debug/aov_routing.h"

#include <cctype>
#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>

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
        case AovId::LoG:
        case AovId::Opponent:
        case AovId::Retinex:
        case AovId::CLAHE:
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
        case AovId::LoG:
            return 1;

        // Two-component lanes: UV's third channel is structurally zero, and Opponent spans the two cardinal chromatic axes only.
        case AovId::UV:
        case AovId::Opponent:
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
        case AovId::Retinex:
        case AovId::CLAHE:
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
        case AovId::LoG:
        case AovId::CLAHE:
            return true;

        // Ratios, reflectances, counts, lengths, directions and frequencies: scaling any of them by an exposure means nothing.
        case AovId::HSV:
        case AovId::Opponent:
        case AovId::Retinex:
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
        case AovId::LoG:
        case AovId::Opponent:
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
        case AovId::Retinex:
        case AovId::CLAHE:
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

float bipolarDisplayRange(std::span<const float> rgba, int channels) {
    double sumOfSquares = 0.0;
    float peak = 0.0F;
    std::size_t count = 0;
    for (std::size_t texel = 0; texel + 3 < rgba.size(); texel += 4) {
        for (int c = 0; c < channels; ++c) {
            const float value = rgba[texel + static_cast<std::size_t>(c)];
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
    // Cramer 1946: the maximum of n standard normals concentrates at sigma*sqrt(2 ln n), so this is where a well-behaved field ends.
    const double expectedMaximum = rms * std::sqrt(2.0 * std::log(static_cast<double>(count)));
    // The smaller of the two: a Gaussian field keeps its true peak, and a heavy-tailed one stops a lone outlier crushing the preview.
    return std::min(peak, static_cast<float>(expectedMaximum));
}

float bipolarDisplayExposureEv(float range) {
    // A field that is exactly zero everywhere has no range, and any finite gain then gives mid-grey, so unity is the natural choice.
    return range > 0.0F ? -std::log2(range / kBipolarDisplayOffset) : 0.0F;
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
