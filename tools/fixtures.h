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

// The lanes every BSDF check shades at: the hero at u = 1/2, one fixed set of wavelengths for films, dispersion and upsampling.
inline const pathtracer::scene::SampledWavelengths& pinnedWavelengths() {
    static const pathtracer::scene::SampledWavelengths kLambda = pathtracer::scene::sampleWavelengths(0.5F);
    return kLambda;
}

// A colour's channels as lanes 0-2, lane 3 repeating green: per-lane BSDF math meets the per-channel oracles exactly, unupsampled.
inline pathtracer::scene::SampledSpectrum lanesOf(const pathtracer::scene::Rgb& c) {
    return pathtracer::scene::SampledSpectrum(glm::vec4(c.r, c.g, c.b, c.g));
}

// lanesOf's inverse: the three channel lanes as a vec3, for the oracles written per channel.
inline glm::vec3 channels(const pathtracer::scene::SampledSpectrum& s) { return glm::vec3(s.lanes); }

// Authored inputs with every colour mapped by lanesOf: the BSDF checks test the closure's math, colour_validate the upsampling.
inline pathtracer::scene::OpenPbrInputs<pathtracer::scene::Spectral> laneInputs(
    const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs) {
    pathtracer::scene::OpenPbrInputs<pathtracer::scene::Spectral> lanes = pathtracer::scene::toSpectral(inputs, pinnedWavelengths());
    lanes.baseColor = lanesOf(inputs.baseColor);
    lanes.specularColor = lanesOf(inputs.specularColor);
    lanes.transmissionColor = lanesOf(inputs.transmissionColor);
    lanes.transmissionScatter = lanesOf(inputs.transmissionScatter);
    lanes.subsurfaceColor = lanesOf(inputs.subsurfaceColor);
    lanes.subsurfaceRadiusScale = lanesOf(inputs.subsurfaceRadiusScale);
    lanes.coatColor = lanesOf(inputs.coatColor);
    lanes.fuzzColor = lanesOf(inputs.fuzzColor);
    return lanes;
}

// The closure of authored inputs, colours as lanes, at the pinned wavelengths a thin film interferes at.
inline pathtracer::scene::BsdfClosure closureOf(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs,
                                                const glm::vec3& woLocal, std::optional<int> heroLane = std::nullopt,
                                                const glm::vec3& coatNormalLocal = glm::vec3(0.0F, 0.0F, 1.0F),
                                                const glm::vec3& coatTangentLocal = glm::vec3(1.0F, 0.0F, 0.0F), float ambientIor = 1.0F) {
    return pathtracer::scene::makeBsdfClosure(laneInputs(inputs), woLocal, pinnedWavelengths().nm, heroLane, coatNormalLocal, coatTangentLocal,
                                              ambientIor);
}

// One closure at woLocal, used once: the validators' single-direction forms of the closure API; a draw walks on its own sampler.
inline pathtracer::scene::BsdfEval evaluateBsdfSplit(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs,
                                                     const glm::vec3& woLocal, const glm::vec3& wiLocal, pathtracer::scene::Sampler& walk) {
    return pathtracer::scene::evaluateBsdfSplit(closureOf(inputs, woLocal), wiLocal, walk);
}
inline pathtracer::scene::BsdfEval evaluateClosedForm(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs,
                                                      const glm::vec3& woLocal, const glm::vec3& wiLocal) {
    return pathtracer::scene::evaluateClosedForm(closureOf(inputs, woLocal), wiLocal);
}
inline float pdfBsdf(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs, const glm::vec3& woLocal,
                     const glm::vec3& wiLocal) {
    return pathtracer::scene::pdfBsdf(closureOf(inputs, woLocal), wiLocal);
}
inline pathtracer::scene::SampledSpectrum evaluateBsdf(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs,
                                                       const glm::vec3& woLocal, const glm::vec3& wiLocal, pathtracer::scene::Sampler& walk) {
    return evaluateBsdfSplit(inputs, woLocal, wiLocal, walk).total();
}
inline std::optional<pathtracer::scene::BsdfSample> sampleBsdf(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs,
                                                               const glm::vec3& woLocal, pathtracer::scene::Sampler& sampler) {
    return pathtracer::scene::sampleBsdf(closureOf(inputs, woLocal), sampler, sampler);
}

// The sensor's exact expectation of a spectrum's per-sample scene RGB, stratified over 2^16 heroes: what a render converges to.
template <typename SpectrumAt>
glm::dvec3 expectedSceneRgb(SpectrumAt spectrumAt, bool reflectance) {
    constexpr int kStrata = 1 << 16;
    glm::dvec3 sum(0.0);
    for (int k = 0; k < kStrata; ++k) {
        const pathtracer::scene::SampledWavelengths lambda = pathtracer::scene::sampleWavelengths((static_cast<float>(k) + 0.5F) / kStrata);
        const pathtracer::scene::SampledSpectrum value = spectrumAt(lambda);
        sum += glm::dvec3((reflectance ? pathtracer::scene::reflectanceToScene(value, lambda) : pathtracer::scene::toScene(value, lambda)).vec3());
    }
    return sum / static_cast<double>(kStrata);
}

// A grey light of `luminance` cd/m^2: the scene white (D65), which the sensor reads as luminance * (1, 1, 1) in expectation.
inline pathtracer::scene::EmissionSpectrum greyEmission(float luminance) {
    pathtracer::scene::EmissionSpectrum spectrum = pathtracer::scene::rgbEmission(pathtracer::scene::Rgb(1.0F));
    spectrum *= luminance;
    return spectrum;
}

// An emitter of `value` W/(sr m^2 nm) at every wavelength (Illuminant E): a light-geometry check reads one known radiance in each lane.
inline pathtracer::scene::EmissionSpectrum flatEmission(float value) {
    pathtracer::scene::cie::Spectrum flat;
    flat.fill(1.0);
    double yBarSum = 0.0;
    for (int i = 0; i < pathtracer::scene::cie::kSampleCount; ++i) {
        yBarSum += pathtracer::scene::cie::tableRow(i).colourMatching.y;
    }
    // Unit luminance puts 1 / (K_m sum yBar) at each sample; this scale lifts it to value.
    pathtracer::scene::EmissionSpectrum spectrum(flat);
    spectrum *= static_cast<float>(static_cast<double>(value) * static_cast<double>(pathtracer::scene::kLuminousEfficacy) * yBarSum);
    return spectrum;
}

// Independent ground truth: Lo(wo) = int evaluateBsdf(wo,wi) dwi at L0 = 1, cosine-weighted. Uniform MC: under-samples a sharp GGX peak.
inline double referenceLo(const pathtracer::scene::OpenPbrInputs<pathtracer::scene::Constant>& inputs, const glm::vec3& wo,
                          int sampleCount, std::mt19937& rng) {
    // Summed in double: a float sum's ulp at 1e4 draws rivals the standard error the callers' bands are built from.
    glm::dvec4 accum(0.0);
    pathtracer::scene::Sampler walk(0, 0, 0, 1, rng());
    for (int i = 0; i < sampleCount; ++i) {
        accum += glm::dvec4((evaluateBsdf(inputs, wo, sampleUniformHemisphere(rng), walk)).lanes);
    }
    return 2.0 * std::numbers::pi * std::max({accum.x, accum.y, accum.z, accum.w}) / static_cast<double>(sampleCount);
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
        pathtracer::scene::SampledSpectrum throughput(1.0F);
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
