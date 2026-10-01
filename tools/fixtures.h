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

// Interleaved float test data of `channels` per texel stored at type, rounded to nearest even as loadImageTexture's read does.
inline pathtracer::gfx::ImageTexture makeImageTexture(int width, int height, int channels, const std::vector<float>& texels,
                                                  pathtracer::gfx::ScalarType type) {
    if (type == pathtracer::gfx::ScalarType::Float32) {
        return {width, height, channels, texels};
    }
    return {width, height, channels, std::vector<pathtracer::gfx::Half>(texels.begin(), texels.end())};
}

// Uniform-radiance (L0 = 1) equirect environment: a real image, so EnvironmentMap's CDFs run their normal path, not the all-black fallback.
inline pathtracer::scene::EnvironmentMap makeUniformEnvironment() {
    constexpr int kWidth = 64;
    constexpr int kHeight = 32;
    return pathtracer::scene::EnvironmentMap(makeImageTexture(
        kWidth, kHeight, pathtracer::gfx::kRgbChannels,
        std::vector<float>(static_cast<std::size_t>(kWidth) * kHeight * pathtracer::gfx::kRgbChannels, 1.0F),
        pathtracer::gfx::ScalarType::Float32));
}

inline pathtracer::gfx::ImageTexture makeConstantTexture(glm::vec3 rgb) {
    return makeImageTexture(1, 1, pathtracer::gfx::kRgbChannels, {rgb.x, rgb.y, rgb.z}, pathtracer::gfx::ScalarType::Float32);
}

// The neutral default (white baseColor, flat normal, constant bump) with roughness in .r and f0 in the specular slot.
inline pathtracer::scene::Material makeMaterial(float roughness, glm::vec3 f0) {
    pathtracer::scene::Material material = pathtracer::scene::makeDefaultMaterial();
    material.roughnessTexture =
        makeImageTexture(1, 1, pathtracer::gfx::kScalarChannels, {roughness}, pathtracer::gfx::ScalarType::Float32);
    material.specularTexture = makeConstantTexture(f0);
    return material;
}

}  // namespace tools::fixtures
