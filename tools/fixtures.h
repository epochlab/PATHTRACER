#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <random>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/environment_map.h"
#include "pathtracer/scene/material.h"
#include "pathtracer/scene/sampler.h"
#include "pathtracer/scene/thread_pool.h"
#include "parallel.h"
#include "stats.h"

// Scenes, materials and analytic references shared by the validators. Oracles re-deriving shipped math stay independent transcriptions.
namespace tools::fixtures {

constexpr float kPi = std::numbers::pi_v<float>;

// estimate(seed + r) for r < kReplicates on up to `threads` workers: independent scrambles whose means are iid (Owen 1997b).
template <typename Estimate>
auto replicated(int threads, std::uint64_t seed, Estimate estimate) {
    std::array<decltype(estimate(seed)), tools::stats::kReplicates> results{};
    tools::parallelFor(tools::stats::kReplicates, threads,
                       [&](int r) { results[static_cast<std::size_t>(r)] = estimate(seed + static_cast<std::uint64_t>(r)); });
    return results;
}

// Uniform hemisphere direction about +z, pdf = 1/(2*pi). Draws cosTheta then phi: the order is contract, callers depending on rng order.
inline glm::vec3 sampleUniformHemisphere(std::mt19937& rng) {
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    const float cosTheta = unit(rng);
    const float sinTheta = std::sqrt(std::max(0.0F, 1.0F - (cosTheta * cosTheta)));
    const float phi = 2.0F * kPi * unit(rng);
    return {sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta};
}

// One pool for the binary, sized from the runner's --threads so concurrent ctest jobs do not oversubscribe the machine.
inline pathtracer::scene::ThreadPool& sharedPool(int threads) {
    static pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(threads));
    return pool;
}

// One closure at woLocal, used once: the validators' single-direction forms of the closure API; a draw walks on its own sampler.
inline pathtracer::scene::BsdfEval evaluateBsdfSplit(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs,
                                                     const glm::vec3& woLocal, const glm::vec3& wiLocal, pathtracer::scene::Sampler& walk) {
    return pathtracer::scene::evaluateBsdfSplit(pathtracer::scene::makeBsdfClosure(inputs, woLocal), wiLocal, walk);
}
inline pathtracer::scene::BsdfEval evaluateClosedForm(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs,
                                                      const glm::vec3& woLocal, const glm::vec3& wiLocal) {
    return pathtracer::scene::evaluateClosedForm(pathtracer::scene::makeBsdfClosure(inputs, woLocal), wiLocal);
}
inline float pdfBsdf(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs, const glm::vec3& woLocal,
                     const glm::vec3& wiLocal) {
    return pathtracer::scene::pdfBsdf(pathtracer::scene::makeBsdfClosure(inputs, woLocal), wiLocal);
}
inline pathtracer::scene::Rgb evaluateBsdf(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs,
                                           const glm::vec3& woLocal, const glm::vec3& wiLocal, pathtracer::scene::Sampler& walk) {
    return evaluateBsdfSplit(inputs, woLocal, wiLocal, walk).total();
}
inline std::optional<pathtracer::scene::BsdfSample> sampleBsdf(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs,
                                                               const glm::vec3& woLocal, pathtracer::scene::Sampler& sampler) {
    return pathtracer::scene::sampleBsdf(pathtracer::scene::makeBsdfClosure(inputs, woLocal), sampler, sampler);
}

// Independent ground truth: Lo(wo) = int evaluateBsdf(wo,wi) dwi at L0 = 1, cosine-weighted. Uniform MC: under-samples a sharp GGX peak.
inline double referenceLo(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs, const glm::vec3& wo,
                          int sampleCount, std::mt19937& rng) {
    // Summed in double: a float sum's ulp at 1e4 draws rivals the standard error the callers' bands are built from.
    glm::dvec3 accum(0.0);
    pathtracer::scene::Sampler walk(0, 0, 0, 1, rng());
    for (int i = 0; i < sampleCount; ++i) {
        accum += glm::dvec3((evaluateBsdf(inputs, wo, sampleUniformHemisphere(rng), walk)).vec3());
    }
    return 2.0 * std::numbers::pi * std::max({accum.x, accum.y, accum.z}) / static_cast<double>(sampleCount);
}

// A white infinite slab under uniform L0 = 1 by BSDF sampling alone, to maxVertices scatterings: the BSDF's own round-trip energy closure.
struct SlabWalk {
    double mean;
    double verticesPerPath;
    long long truncated;
};

inline SlabWalk slabWalkLo(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs, int paths, std::uint32_t seed,
                           int maxVertices) {
    double sum = 0.0;
    long long vertices = 0;
    long long truncated = 0;
    for (int i = 0; i < paths; ++i) {
        pathtracer::scene::Sampler sampler(0, 0, i, paths, seed);
        glm::vec3 direction(0.0F, 0.0F, -1.0F);
        pathtracer::scene::Rgb throughput(1.0F);
        bool top = true;
        int vertex = 0;
        for (; vertex < maxVertices; ++vertex) {
            const float zSign = top ? 1.0F : -1.0F;
            const std::optional<pathtracer::scene::BsdfSample> sample =
                sampleBsdf(inputs, glm::vec3(-direction.x, -direction.y, -direction.z * zSign), sampler);
            if (!sample.has_value()) {
                break;
            }
            ++vertices;
            throughput *= sample->throughputWeight;
            direction = glm::vec3(sample->wiLocal.x, sample->wiLocal.y, sample->wiLocal.z * zSign);
            if (top ? direction.z > 0.0F : direction.z < 0.0F) {
                sum += pathtracer::scene::maxChannel(throughput);
                break;
            }
            top = !top;
        }
        truncated += vertex == maxVertices ? 1 : 0;
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
inline pathtracer::scene::EnvironmentMap makeEnvironment(int width, int height, std::vector<float> rgb,
                                                         std::optional<float> illuminanceLux = std::nullopt) {
    return pathtracer::scene::EnvironmentMap(
        makeTexture(width, height, pathtracer::gfx::kRgbChannels, std::move(rgb), pathtracer::gfx::TextureWrap::LatLong), illuminanceLux);
}

// Uniform-radiance (L0 = 1) equirect environment: a real image, so EnvironmentMap's CDFs run their normal path, not the all-black fallback.
inline pathtracer::scene::EnvironmentMap makeUniformEnvironment() {
    constexpr int kWidth = 64;
    constexpr int kHeight = 32;
    return makeEnvironment(kWidth, kHeight,
                           std::vector<float>(static_cast<std::size_t>(kWidth) * kHeight * pathtracer::gfx::kRgbChannels, 1.0F));
}

// A dielectric over a white diffuse base at specular_roughness, every other input at its OpenPBR default.
inline pathtracer::scene::Material makeMaterial(float specularRoughness) {
    pathtracer::scene::Material material;
    material.baseColor = pathtracer::scene::Rgb(1.0F);
    material.specularRoughness = specularRoughness;
    return material;
}

// OpenPBR defaults with a fractional metalness, varying only the two inputs the G-buffer AOVs under test read.
inline pathtracer::scene::Material makeMaterial(pathtracer::scene::Rgb baseColor, float roughness) {
    pathtracer::scene::Material material;
    material.baseColor = baseColor;
    material.baseMetalness = 0.2F;
    material.specularRoughness = roughness;
    return material;
}

// A unit tangent perpendicular to normal, +w handedness: any basis serves a surface whose inputs ignore anisotropy.
inline glm::vec4 tangentFor(const glm::vec3& normal) {
    const glm::vec3 up = std::fabs(normal.y) < 0.99F ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::vec3(1.0F, 0.0F, 0.0F);
    return glm::vec4(glm::normalize(glm::cross(up, normal)), 1.0F);
}

}  // namespace tools::fixtures
