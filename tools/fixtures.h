#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/environment_map.h"
#include "pathtracer/scene/material.h"
#include "pathtracer/scene/sampler.h"

// Scenes, materials and analytic references shared by the validators. Oracles re-deriving shipped math stay independent transcriptions.
namespace tools::fixtures {

constexpr float kPi = 3.14159265F;

// Uniform hemisphere direction about +z, pdf = 1/(2*pi). Draws cosTheta then phi: the order is contract, callers depending on rng order.
inline glm::vec3 sampleUniformHemisphere(std::mt19937& rng) {
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    const float cosTheta = unit(rng);
    const float sinTheta = std::sqrt(std::max(0.0F, 1.0F - (cosTheta * cosTheta)));
    const float phi = 2.0F * kPi * unit(rng);
    return {sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta};
}

// Independent ground truth: Lo(wo) = int evaluateBsdf(wo,wi)*wi.z dwi at L0 = 1. Uniform MC, so it under-samples a sharp GGX peak.
inline float referenceLo(const pathtracer::scene::BsdfParams& params, const glm::vec3& wo, int sampleCount,
                          std::mt19937& rng) {
    constexpr float kUniformPdf = 1.0F / (2.0F * kPi);
    glm::vec3 accum(0.0F);
    for (int i = 0; i < sampleCount; ++i) {
        const glm::vec3 wi = sampleUniformHemisphere(rng);
        accum += pathtracer::scene::evaluateBsdf(params, wo, wi) * wi.z / kUniformPdf;
    }
    return std::max({accum.x, accum.y, accum.z}) / static_cast<float>(sampleCount);
}

// A white infinite slab under uniform L0 = 1 by BSDF sampling alone: the BSDF's own round-trip energy closure, free of integrator.
struct SlabWalk {
    double mean;
    double verticesPerPath;
    long long truncated;
};

inline SlabWalk slabWalkLo(const pathtracer::scene::BsdfParams& params, int paths, std::uint32_t seed) {
    constexpr int kMaxVertices = 256;
    double sum = 0.0;
    long long vertices = 0;
    long long truncated = 0;
    for (int i = 0; i < paths; ++i) {
        pathtracer::scene::Sampler sampler(0, 0, i, paths, seed);
        glm::vec3 direction(0.0F, 0.0F, -1.0F);
        glm::vec3 throughput(1.0F);
        bool top = true;
        int vertex = 0;
        for (; vertex < kMaxVertices; ++vertex) {
            const float zSign = top ? 1.0F : -1.0F;
            const std::optional<pathtracer::scene::BsdfSample> sample =
                pathtracer::scene::sampleBsdf(params, glm::vec3(-direction.x, -direction.y, -direction.z * zSign), sampler);
            if (!sample.has_value()) {
                break;
            }
            ++vertices;
            throughput *= sample->throughputWeight;
            direction = glm::vec3(sample->wiLocal.x, sample->wiLocal.y, sample->wiLocal.z * zSign);
            if (top ? direction.z > 0.0F : direction.z < 0.0F) {
                sum += std::max({throughput.x, throughput.y, throughput.z});
                break;
            }
            top = !top;
        }
        truncated += vertex == kMaxVertices ? 1 : 0;
    }
    const auto count = static_cast<double>(paths);
    return {sum / count, static_cast<double>(vertices) / count, truncated};
}

// Interleaved float test data as a texture, served by the TextureSystem as a scene file is.
inline std::shared_ptr<const pathtracer::gfx::ImageTexture> makeTexture(
    int width, int height, int channels, std::vector<float> texels,
    pathtracer::gfx::TextureWrap wrap = pathtracer::gfx::TextureWrap::Repeat) {
    return pathtracer::gfx::makeTexture({width, height, channels, std::move(texels)}, wrap);
}

// An equirect environment of interleaved RGB test radiance, through the same texture path a scene's HDRI takes.
inline pathtracer::scene::EnvironmentMap makeEnvironment(int width, int height, std::vector<float> rgb) {
    return pathtracer::scene::EnvironmentMap(
        makeTexture(width, height, pathtracer::gfx::kRgbChannels, std::move(rgb), pathtracer::gfx::TextureWrap::LatLong));
}

// Uniform-radiance (L0 = 1) equirect environment: a real image, so EnvironmentMap's CDFs run their normal path, not the all-black fallback.
inline pathtracer::scene::EnvironmentMap makeUniformEnvironment() {
    constexpr int kWidth = 64;
    constexpr int kHeight = 32;
    return makeEnvironment(kWidth, kHeight,
                           std::vector<float>(static_cast<std::size_t>(kWidth) * kHeight * pathtracer::gfx::kRgbChannels, 1.0F));
}

// The neutral default (white baseColor, flat normal, constant bump) with constant roughness and specular f0.
inline pathtracer::scene::Material makeMaterial(float roughness, glm::vec3 f0) {
    return pathtracer::scene::Material{.roughness = roughness, .specular = f0};
}

}  // namespace tools::fixtures
