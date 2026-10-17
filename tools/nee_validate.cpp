// NEE+MIS (Veach & Guibas 1995 power heuristic) against EnvironmentMap and LightSet.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <OpenImageIO/imageio.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/epsilon.hpp>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/scalar_type.h"
#include "check.h"
#include "fixtures.h"
#include "stats.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/environment_map.h"
#include "pathtracer/scene/lat_long.h"
#include "pathtracer/scene/light.h"
#include "pathtracer/scene/rotation.h"
#include "pathtracer/scene/sampler.h"

namespace {

using pathtracer::scene::AreaLight;
using pathtracer::scene::Constant;
using pathtracer::scene::DiskLight;
using pathtracer::scene::EnvironmentMap;
using pathtracer::scene::LightSample;
using pathtracer::scene::LightSet;
using pathtracer::scene::LobeType;
using pathtracer::scene::MeshLight;
using pathtracer::scene::OpenPbrInputs;
using pathtracer::scene::QuadLight;
using pathtracer::scene::Sampler;
using pathtracer::scene::SphereLight;
using pathtracer::scene::Triangle;

using tools::fixtures::kPi;
using tools::fixtures::makeUniformEnvironment;
using tools::fixtures::referenceLo;
using tools::fixtures::sampleUniformHemisphere;

// A white base, worst case for albedo, at the given metalness and specular_roughness; every other input at its OpenPBR default.
OpenPbrInputs<Constant> makeParams(float roughness, float metalness) {
    OpenPbrInputs<Constant> inputs;
    inputs.baseColor = glm::vec3(1.0F);
    inputs.baseMetalness = metalness;
    inputs.specularRoughness = roughness;
    return inputs;
}

// Dim background with one bright patch: a uniform map gives linear CDFs, where a mis-scaled Jacobian or an off-by-one bin would pass.
EnvironmentMap makeStructuredEnvironment() {
    constexpr int kWidth = 64;
    constexpr int kHeight = 32;
    constexpr int kChannels = pathtracer::gfx::kRgbChannels;
    std::vector<float> rgb(static_cast<std::size_t>(kWidth) * kHeight * kChannels, 0.05F);
    for (int y = 8; y < 12; ++y) {
        for (int x = 20; x < 26; ++x) {
            const std::size_t idx = ((static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)) * kChannels;
            rgb[idx + 0] = 400.0F;
            rgb[idx + 1] = 380.0F;
            rgb[idx + 2] = 300.0F;
        }
    }
    return tools::fixtures::makeEnvironment(kWidth, kHeight, rgb);
}

// sample returns a density, pdf() recovers it from a direction; MIS uses both, so disagreement corrupts every MIS weight in the renderer.
PT_CHECK(environment_pdf_consistency, Fast, Exact) {
    constexpr int kSampleCount = 20000;
    constexpr float kTolerance = 1e-3F;
    // Sampling rotates by R and querying by R^T, so a transpose slip cancels at identity; all three axes, so no axis can hide it.
    const glm::mat3 rotation = pathtracer::scene::rotationXyz(glm::vec3(25.0F, 40.0F, -60.0F));
    ctx.plan(1);
    const EnvironmentMap env = makeStructuredEnvironment();
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    int worstIndex = -1;
    float worstRelative = 0.0F;
    for (int i = 0; i < kSampleCount; ++i) {
        const EnvironmentMap::EnvSample sample =
            env.importanceSampleDirection(glm::vec2(unit(rng), unit(rng)), rotation);
        const float queried = env.pdf(sample.direction, rotation);
        const float relative = std::fabs(queried - sample.pdf) / std::max(sample.pdf, 1e-6F);
        if (relative > worstRelative) {
            worstRelative = relative;
            worstIndex = i;
        }
    }
    // Two code paths over one direction: exactness with a float tolerance, not statistics -- they agree or every MIS weight is wrong.
    char detail[224];
    std::snprintf(detail, sizeof(detail),
                  "worst relative mismatch %.3e at sample %d, between importanceSampleDirection's own pdf and pdf()",
                  static_cast<double>(worstRelative), worstIndex);
    PT_EXPECT(ctx, worstRelative <= kTolerance, detail);
}

// CDFs must come from the values the map returns: a half EXR stores 1 + 2^-11 as 1.0, and the cache keeps the file's half.
PT_CHECK(environment_pdf_tracks_stored_luminance, Fast, Exact) {
    constexpr int kWidth = 64;
    constexpr int kHeight = 32;
    constexpr int kPatchX = 20;
    constexpr int kPatchY = 16;
    constexpr int kBackgroundX = 40;
    const float midpoint = 1.0F + pathtracer::gfx::kHalfUnitRoundoff;
    constexpr int kChannels = pathtracer::gfx::kRgbChannels;
    std::vector<float> rgb(static_cast<std::size_t>(kWidth) * kHeight * kChannels, 1.0F);
    const std::size_t patch = ((static_cast<std::size_t>(kPatchY) * kWidth) + kPatchX) * kChannels;
    rgb[patch + 0] = midpoint;
    rgb[patch + 1] = midpoint;
    rgb[patch + 2] = midpoint;

    // Direction through a texel's centre: the shared chart on the map's frame (right, up, forward) = (-X, +Y, +Z).
    const auto centre = [](int x, int y) {
        const glm::vec3 local = pathtracer::scene::latLongDirection(
            glm::vec2((static_cast<float>(x) + 0.5F) / kWidth, (static_cast<float>(y) + 0.5F) / kHeight));
        return glm::vec3(-local.x, local.y, local.z);
    };
    ctx.plan(2);
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "engine_nee_half_environment.exr";
    OIIO::ImageSpec spec(kWidth, kHeight, kChannels, OIIO::TypeHalf);
    const std::unique_ptr<OIIO::ImageOutput> output = OIIO::ImageOutput::create("openexr");
    const bool written = output && output->open(path.string(), spec) && output->write_image(OIIO::TypeFloat, rgb.data()) && output->close();
    const std::shared_ptr<const pathtracer::gfx::ImageTexture> stored =
        written ? pathtracer::gfx::openTexture(path.string(), kChannels, pathtracer::gfx::ImageRole::Colour, pathtracer::gfx::TextureWrap::LatLong)
                : nullptr;
    const std::shared_ptr<const pathtracer::gfx::ImageTexture> exact =
        tools::fixtures::makeTexture(kWidth, kHeight, kChannels, rgb, pathtracer::gfx::TextureWrap::LatLong);
    for (const auto& [name, texture, expected] : {std::tuple{"half EXR", stored, 1.0F}, std::tuple{"float", exact, midpoint}}) {
        if (!texture) {
            PT_EXPECT(ctx, false, std::string(name) + ": the probe texture did not open");
            continue;
        }
        const EnvironmentMap env(texture);
        // Same row, so sin(theta) cancels and the solid-angle pdf ratio is the stored luminance ratio, exactly.
        const float pdfRatio = env.pdf(centre(kPatchX, kPatchY)) / env.pdf(centre(kBackgroundX, kPatchY));
        char detail[192];
        std::snprintf(detail, sizeof(detail), "%s: pdf ratio %.9g, stored luminance ratio %.9g", name, static_cast<double>(pdfRatio),
                      static_cast<double>(expected));
        PT_EXPECT(ctx, std::fabs(pdfRatio - expected) < std::fabs(pdfRatio - (expected == 1.0F ? midpoint : 1.0F)), detail);
    }
    std::filesystem::remove(path);
}

// MIS-combined NEE + BSDF estimator mirroring tracePath: power heuristic, and with no geometry both strategies reach the environment.
float misCombinedLo(const OpenPbrInputs<Constant>& params, const glm::vec3& wo, const EnvironmentMap& env,
                     int sampleCount, std::uint32_t seed) {
    glm::vec3 accum(0.0F);
    for (int i = 0; i < sampleCount; ++i) {
        Sampler sampler(0, 0, i, sampleCount, seed);

        // NEE.
        const EnvironmentMap::EnvSample lightSample =
            env.importanceSampleDirection(sampler.next2D());
        if (lightSample.direction.z > 0.0F) {
            const glm::vec3 bsdfValue = pathtracer::scene::evaluateBsdf(params, wo, lightSample.direction);
            const float bsdfPdf = pathtracer::scene::pdfBsdf(params, wo, lightSample.direction);
            if (bsdfPdf > 0.0F && (bsdfValue.x > 0.0F || bsdfValue.y > 0.0F || bsdfValue.z > 0.0F)) {
                const float lightPdf2 = lightSample.pdf * lightSample.pdf;
                const float bsdfPdf2 = bsdfPdf * bsdfPdf;
                const float misWeight = lightPdf2 / (lightPdf2 + bsdfPdf2);
                accum += bsdfValue * misWeight / lightSample.pdf;
            }
        }

        // BSDF-sampled.
        const std::optional<pathtracer::scene::BsdfSample> sample = pathtracer::scene::sampleBsdf(params, wo, sampler);
        if (sample.has_value() && sample->type != LobeType::Transmission) {
            const float bsdfPdf = pathtracer::scene::pdfBsdf(params, wo, sample->wiLocal);
            const float lightPdf = env.pdf(sample->wiLocal);
            const float bsdfPdf2 = bsdfPdf * bsdfPdf;
            const float lightPdf2 = lightPdf * lightPdf;
            const float misWeight = bsdfPdf2 / (bsdfPdf2 + lightPdf2);
            accum += sample->throughputWeight * misWeight;  // L0=1 folded in via the uniform environment already
        }
    }
    return std::max({accum.x, accum.y, accum.z}) / static_cast<float>(sampleCount);
}

// An equirect map's v axis is polar, not periodic: the top and bottom rows are opposite poles, so wrapping v blends them together.
PT_CHECK(environment_poles_do_not_blend_opposite_rows, Fast, Exact) {
    constexpr int kWidth = 16;
    constexpr int kHeight = 8;
    const glm::vec3 north(1.0F, 0.0F, 0.0F);
    const glm::vec3 south(0.0F, 0.0F, 1.0F);
    constexpr int kChannels = pathtracer::gfx::kRgbChannels;
    std::vector<float> rgb(static_cast<std::size_t>(kWidth) * kHeight * kChannels, 0.0F);
    for (int y = 0; y < kHeight; ++y) {
        // Row 0 is theta = 0 (the zenith) and the last row theta = pi; the band between them is mid grey, so a blend is unmistakable.
        const glm::vec3 row = y == 0 ? north : (y == kHeight - 1 ? south : glm::vec3(0.5F));
        for (int x = 0; x < kWidth; ++x) {
            const std::size_t idx = ((static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)) * kChannels;
            rgb[idx + 0] = row.x;
            rgb[idx + 1] = row.y;
            rgb[idx + 2] = row.z;
        }
    }
    const EnvironmentMap env = tools::fixtures::makeEnvironment(kWidth, kHeight, rgb);

    const glm::vec3 zenith = env.sampleDirection(glm::vec3(0.0F, 1.0F, 0.0F));
    const glm::vec3 nadir = env.sampleDirection(glm::vec3(0.0F, -1.0F, 0.0F));
    ctx.plan(2);
    char detail[224];
    std::snprintf(detail, sizeof(detail), "zenith reads (%.4f, %.4f, %.4f), expected the top row (1, 0, 0)",
                  static_cast<double>(zenith.x), static_cast<double>(zenith.y), static_cast<double>(zenith.z));
    PT_EXPECT(ctx, glm::all(glm::epsilonEqual(zenith, north, 1e-6F)), detail);
    std::snprintf(detail, sizeof(detail), "nadir reads (%.4f, %.4f, %.4f), expected the bottom row (0, 0, 1)",
                  static_cast<double>(nadir.x), static_cast<double>(nadir.y), static_cast<double>(nadir.z));
    PT_EXPECT(ctx, glm::all(glm::epsilonEqual(nadir, south, 1e-6F)), detail);
}

// A camera facing the map's centre sees u grow to its right, as a panorama viewed from inside does: the map is not mirrored.
PT_CHECK(environment_lat_long_is_unmirrored, Fast, Exact) {
    constexpr int kWidth = 16;
    constexpr int kHeight = 8;
    constexpr int kChannels = pathtracer::gfx::kRgbChannels;
    const glm::vec3 left(1.0F, 0.0F, 0.0F);
    const glm::vec3 right(0.0F, 0.0F, 1.0F);
    std::vector<float> rgb(static_cast<std::size_t>(kWidth) * kHeight * kChannels);
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const glm::vec3 texel = x < kWidth / 2 ? left : right;
            const std::size_t idx = ((static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)) * kChannels;
            rgb[idx + 0] = texel.x;
            rgb[idx + 1] = texel.y;
            rgb[idx + 2] = texel.z;
        }
    }
    const EnvironmentMap env = tools::fixtures::makeEnvironment(kWidth, kHeight, rgb);
    // Yaw 180 faces +Z, the map's centre column; a quarter frame either side stays well inside one half of the map.
    const pathtracer::scene::Camera camera(glm::vec3(0.0F), glm::vec3(0.0F, 180.0F, 0.0F), {36.0F, 24.0F}, 35.0F, 0.1F, 100.0F, 2.8F,
                                           0.01F, 100.0F);
    const pathtracer::scene::Camera::ViewBasis basis = camera.viewBasis(1.5F);
    const glm::vec3 seenRight = env.sampleDirection(camera.primaryRay(basis, 0.5F, 0.0F)->dir);
    const glm::vec3 seenLeft = env.sampleDirection(camera.primaryRay(basis, -0.5F, 0.0F)->dir);
    ctx.plan(2);
    char detail[224];
    std::snprintf(detail, sizeof(detail), "image right reads (%.4f, %.4f, %.4f), expected the map's right half (0, 0, 1)",
                  static_cast<double>(seenRight.x), static_cast<double>(seenRight.y), static_cast<double>(seenRight.z));
    PT_EXPECT(ctx, glm::all(glm::epsilonEqual(seenRight, right, 1e-6F)), detail);
    std::snprintf(detail, sizeof(detail), "image left reads (%.4f, %.4f, %.4f), expected the map's left half (1, 0, 0)",
                  static_cast<double>(seenLeft.x), static_cast<double>(seenLeft.y), static_cast<double>(seenLeft.z));
    PT_EXPECT(ctx, glm::all(glm::epsilonEqual(seenLeft, left, 1e-6F)), detail);
}

// The lens's chart is the map's: an omnidirectional camera posed in the map's frame, at its resolution, reads each texel at its centre.
PT_CHECK(omnidirectional_camera_reads_the_environment_texel_for_texel, Fast, Exact) {
    constexpr int kWidth = 32;
    constexpr int kHeight = 16;
    constexpr int kChannels = pathtracer::gfx::kRgbChannels;
    // Every texel distinct, so a mirrored, shifted or transposed chart cannot land on an equal value by coincidence.
    std::vector<float> rgb(static_cast<std::size_t>(kWidth) * kHeight * kChannels);
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = static_cast<float>(i + 1) / static_cast<float>(rgb.size());
    }
    const EnvironmentMap env = tools::fixtures::makeEnvironment(kWidth, kHeight, rgb);
    // Yaw 180 faces +Z with -X on the right and +Y up: the map's (right, up, forward), so pixel (x, y) and texel (x, y) share (u, v).
    const pathtracer::scene::Camera camera(glm::vec3(0.0F), glm::vec3(0.0F, 180.0F, 0.0F), {36.0F, 24.0F}, 35.0F, 0.1F, 100.0F, 2.8F,
                                           0.01F, 100.0F, pathtracer::scene::Lens{pathtracer::scene::LensProjection::Omnidirectional});
    const pathtracer::scene::Camera::ViewBasis basis = camera.viewBasis(2.0F);
    // The (u, v) round trip carries 16 ulps of its pi-scaled angles; longitude's is amplified by 1/sin(colatitude) toward a pole.
    constexpr float kRoundTripUlps = 16.0F;
    int mismatches = 0;
    for (int y = 0; y < kHeight; ++y) {
        const float sinColatitude = std::sin(glm::pi<float>() * (static_cast<float>(y) + 0.5F) / kHeight);
        for (int x = 0; x < kWidth; ++x) {
            const float ndcX = (((static_cast<float>(x) + 0.5F) / kWidth) * 2.0F) - 1.0F;
            const float ndcY = 1.0F - (((static_cast<float>(y) + 0.5F) / kHeight) * 2.0F);
            const glm::vec3 seen = env.sampleDirection(camera.primaryRay(basis, ndcX, ndcY)->dir);
            const std::size_t index = ((static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)) * kChannels;
            const glm::vec3 texel(rgb[index], rgb[index + 1], rgb[index + 2]);
            // Off-centre by `offset` texels, bilinear blends in at most that fraction of a neighbour, whose difference is bounded here.
            const float offset = kRoundTripUlps * std::numeric_limits<float>::epsilon() * static_cast<float>(kWidth) / sinColatitude;
            const float neighbourSpan = static_cast<float>(kWidth) / static_cast<float>(rgb.size()) * kChannels;
            mismatches += glm::all(glm::lessThanEqual(glm::abs(seen - texel), glm::vec3(2.0F * offset * neighbourSpan))) ? 0 : 1;
        }
    }
    ctx.plan(1);
    PT_EXPECT(ctx, mismatches == 0, std::to_string(mismatches) + " pixels read a texel other than their own");
}

// MIS weights sum to 1, so both strategies must evaluate ONE Le. Structured, not uniform: on a uniform map every lookup agrees.
PT_CHECK(environment_nee_and_miss_share_one_radiance, Fast, Exact) {
    const EnvironmentMap env = makeStructuredEnvironment();
    const std::vector<AreaLight> noLights;
    const LightSet lights(&env, /*envRotationDegrees=*/glm::vec3(0.0F), /*envExposure=*/1.0F, noLights);

    constexpr int kSamples = 4096;
    int drawn = 0;
    int mismatches = 0;
    float worstRelative = 0.0F;
    for (int i = 0; i < kSamples; ++i) {
        Sampler sampler(i % 64, i / 64, i, kSamples, 0x9E3779B9U);
        const std::optional<LightSample> sample = lights.sample(glm::vec3(0.0F), sampler);
        if (!sample.has_value()) {
            continue;
        }
        ++drawn;
        const glm::vec3 miss = lights.environmentRadiance(sample->direction);
        const glm::vec3 difference = glm::abs(miss - sample->radiance);
        const float scale = std::max({std::fabs(miss.x), std::fabs(miss.y), std::fabs(miss.z), 1e-6F});
        const float relative = std::max({difference.x, difference.y, difference.z}) / scale;
        if (relative > 0.0F) {
            ++mismatches;
            worstRelative = std::max(worstRelative, relative);
        }
    }

    ctx.plan(2);
    PT_EXPECT(ctx, drawn > 0, "LightSet::sample returned nothing, so the comparison below is vacuous");
    char detail[224];
    std::snprintf(detail, sizeof(detail),
                  "NEE and the miss path disagree on Le at %d of %d sampled directions, worst relative %.3e",
                  mismatches, drawn, static_cast<double>(worstRelative));
    PT_EXPECT(ctx, mismatches == 0, detail);
}

// MIS estimator vs brute-force reference, both noisy: differenced as independent estimators, variances adding (Welch 1947).
PT_CHECK(mis_agreement_environment, Slow, Statistical) {
    constexpr int kTotalSamples = 100000;
    constexpr int kPerReplicate = kTotalSamples / tools::stats::kReplicates;
    // Excludes roughness 0.05: uniform-hemisphere sampling misses a sharp GGX peak, biasing the reference low; this check is two-sided.
    const std::array<float, 3> roughnesses = {0.25F, 0.5F, 1.0F};
    const std::array<float, 2> metallics = {0.0F, 1.0F};
    const std::array<float, 3> ndotVs = {0.2F, 0.6F, 1.0F};

    ctx.plan(static_cast<int>(roughnesses.size() * metallics.size() * ndotVs.size()));
    const EnvironmentMap env = makeUniformEnvironment();

    for (float roughness : roughnesses) {
        for (float metallic : metallics) {
            for (float ndotV : ndotVs) {
                char label[96];
                std::snprintf(label, sizeof(label), "env r=%g m=%g n=%g", static_cast<double>(roughness),
                              static_cast<double>(metallic), static_cast<double>(ndotV));
                const OpenPbrInputs<Constant> params = makeParams(roughness, metallic);
                const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);

                const std::uint64_t rowSeed = ctx.subSeed(label);
                std::mt19937 referenceRng(static_cast<std::mt19937::result_type>(rowSeed));
                tools::stats::Welford reference;
                tools::stats::Welford combined;
                for (int r = 0; r < tools::stats::kReplicates; ++r) {
                    reference.add(referenceLo(params, wo, kPerReplicate, referenceRng));
                    // A fresh scramble seed per replicate: that independence is the whole basis of the band.
                    combined.add(misCombinedLo(params, wo, env, kPerReplicate,
                                                static_cast<std::uint32_t>(rowSeed + r)));
                }

                const double difference = combined.mean() - reference.mean();
                const tools::stats::Band band = tools::stats::differenceBand(reference, combined, ctx.alpha());
                char detail[256];
                std::snprintf(detail, sizeof(detail),
                              "%s: reference %.5f, MIS %.5f, difference %+.3e vs +/-%.3e", label,
                              reference.mean(), combined.mean(), difference, band.halfWidth());
                PT_EXPECT(ctx, band.contains(difference), detail);
            }
        }
    }
}

// Uniform over the FULL sphere: a quad can subtend solid angle either side of the tangent plane, so the oracle must search all of it.
glm::vec3 sampleUniformSphere(std::mt19937& rng) {
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    const float z = 1.0F - (2.0F * unit(rng));
    const float r = std::sqrt(std::max(0.0F, 1.0F - (z * z)));
    const float phi = 2.0F * kPi * unit(rng);
    return {r * std::cos(phi), r * std::sin(phi), z};
}

// Where a ray from p along unit dir first meets an area light, from its geometry alone: no code shared with the samplers under test.
struct LightHit {
    glm::vec3 point;
    glm::vec3 normal;  // geometric: the emitting side's for a quad, disk or mesh face, outward for a sphere
};

// Re-derives ray/rectangle from world geometry (edge0 perpendicular to edge1), so a buildSphericalRectangle frame bug cannot repeat.
std::optional<LightHit> hitQuad(const QuadLight& quad, const glm::vec3& p, const glm::vec3& dir) {
    const glm::vec3 normal = glm::normalize(glm::cross(quad.edge0, quad.edge1));
    const float denom = glm::dot(dir, normal);
    if (std::fabs(denom) < 1e-8F) {
        return std::nullopt;  // parallel to the plane
    }
    const float t = glm::dot(quad.origin - p, normal) / denom;
    if (t <= 0.0F) {
        return std::nullopt;  // plane is behind p
    }
    const glm::vec3 point = p + (t * dir);
    const glm::vec3 hit = point - quad.origin;
    const float u = glm::dot(hit, quad.edge0) / glm::dot(quad.edge0, quad.edge0);
    const float v = glm::dot(hit, quad.edge1) / glm::dot(quad.edge1, quad.edge1);
    if (u < 0.0F || u > 1.0F || v < 0.0F || v > 1.0F) {
        return std::nullopt;
    }
    return LightHit{point, normal};
}

std::optional<LightHit> hitDisk(const DiskLight& disk, const glm::vec3& p, const glm::vec3& dir) {
    const glm::vec3 normal = disk.frame[2];
    const float t = glm::dot(disk.centre - p, normal) / glm::dot(dir, normal);
    const glm::vec3 point = p + (t * dir);
    if (!(t > 0.0F) || glm::length(point - disk.centre) > disk.radius) {
        return std::nullopt;
    }
    return LightHit{point, normal};
}

// The nearer positive root of |p + t dir - c| = r, so a ray from inside meets the far wall.
std::optional<LightHit> hitSphere(const SphereLight& sphere, const glm::vec3& p, const glm::vec3& dir) {
    const glm::vec3 fromCentre = p - sphere.centre;
    const float b = glm::dot(dir, fromCentre);
    const float discriminant = (b * b) - (glm::dot(fromCentre, fromCentre) - (sphere.radius * sphere.radius));
    if (discriminant < 0.0F) {
        return std::nullopt;
    }
    const float root = std::sqrt(discriminant);
    const float t = -b - root > 0.0F ? -b - root : -b + root;
    if (!(t > 0.0F)) {
        return std::nullopt;
    }
    const glm::vec3 point = p + (t * dir);
    return LightHit{point, glm::normalize(point - sphere.centre)};
}

// The closest face by plain Moller-Trumbore, its normal cross(v1 - v0, v2 - v0) as glTF's front.
std::optional<LightHit> hitMesh(const MeshLight& mesh, const glm::vec3& p, const glm::vec3& dir) {
    std::optional<LightHit> closest;
    float closestT = std::numeric_limits<float>::infinity();
    for (const Triangle& tri : mesh.triangles) {
        const glm::vec3 e1 = tri.v1 - tri.v0;
        const glm::vec3 e2 = tri.v2 - tri.v0;
        const glm::vec3 pvec = glm::cross(dir, e2);
        const float det = glm::dot(e1, pvec);
        if (det == 0.0F) {
            continue;
        }
        const glm::vec3 tvec = p - tri.v0;
        const float u = glm::dot(tvec, pvec) / det;
        const glm::vec3 qvec = glm::cross(tvec, e1);
        const float v = glm::dot(dir, qvec) / det;
        const float t = glm::dot(e2, qvec) / det;
        if (u >= 0.0F && v >= 0.0F && u + v <= 1.0F && t > 0.0F && t < closestT) {
            closestT = t;
            closest = LightHit{p + (t * dir), glm::normalize(glm::cross(e1, e2))};
        }
    }
    return closest;
}

std::optional<LightHit> hitLight(const AreaLight& light, const glm::vec3& p, const glm::vec3& dir) {
    return std::visit(
        [&](const auto& shape) {
            using Shape = std::decay_t<decltype(shape)>;
            if constexpr (std::is_same_v<Shape, QuadLight>) {
                return hitQuad(shape, p, dir);
            } else if constexpr (std::is_same_v<Shape, DiskLight>) {
                return hitDisk(shape, p, dir);
            } else if constexpr (std::is_same_v<Shape, SphereLight>) {
                return hitSphere(shape, p, dir);
            } else {
                return hitMesh(shape, p, dir);
            }
        },
        light);
}

// Le arriving along dir off a hit, re-derived: the front face only, unless two-sided; a sphere's outward face is all it shows.
glm::vec3 emittedAlong(const AreaLight& light, const glm::vec3& dir, const glm::vec3& normal) {
    return std::visit(
        [&](const auto& shape) {
            bool twoSided = false;
            if constexpr (!std::is_same_v<std::decay_t<decltype(shape)>, SphereLight>) {
                twoSided = shape.twoSided;
            }
            return twoSided || glm::dot(dir, normal) < 0.0F ? shape.radiance : glm::vec3(0.0F);
        },
        light);
}

// Girard solid angle against a uniform-sphere + hitQuad oracle sharing no code, and every sample() direction against it too.
PT_CHECK(quad_light_solid_angle, Slow, Statistical) {
    const QuadLight quad{glm::vec3(-0.5F, 1.0F, -0.5F), glm::vec3(1.0F, 0.0F, 0.0F),
                          glm::vec3(0.0F, 0.0F, 1.0F), glm::vec3(1.0F), false};
    const glm::vec3 p(0.0F, 0.0F, 0.0F);
    const std::optional<pathtracer::scene::SphericalRectangle> rect =
        pathtracer::scene::buildSphericalRectangle(quad, p);
    ctx.plan(3);
    if (!rect.has_value()) {
        PT_EXPECT(ctx, false, "buildSphericalRectangle returned nullopt for a valid configuration");
        ctx.plan(1);
        return;
    }

    constexpr int kSampleCount = 2000000;
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    long long hits = 0;
    for (int i = 0; i < kSampleCount; ++i) {
        hits += hitQuad(quad, p, sampleUniformSphere(rng)).has_value() ? 1 : 0;
    }
    // Clopper-Pearson, not Wald: Wald's width collapses near 0 or 1, where a small light lands. Scaled to solid angle by the same 4*pi.
    const tools::stats::Band fraction = tools::stats::clopperPearsonBand(hits, kSampleCount, ctx.alpha());
    const tools::stats::Band solidAngle{4.0 * kPi * fraction.lo, 4.0 * kPi * fraction.hi};
    char detail[224];
    std::snprintf(detail, sizeof(detail), "analytic %.6f vs Monte Carlo interval [%.6f, %.6f] from %lld/%d hits",
                  static_cast<double>(rect->solidAngle), solidAngle.lo, solidAngle.hi, hits, kSampleCount);
    PT_EXPECT(ctx, solidAngle.contains(static_cast<double>(rect->solidAngle)), detail);

    // Every direction sample() draws must land on the rectangle: an exact assertion on the xu/yv inversion the solid angle cannot make.
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    constexpr int kDrawCount = 20000;
    int misses = 0;
    glm::vec2 firstMiss(0.0F);
    for (int i = 0; i < kDrawCount; ++i) {
        const glm::vec2 u(unit(rng), unit(rng));
        const glm::vec3 dir = glm::normalize(rect->sample(u) - p);
        if (!hitQuad(quad, p, dir)) {
            if (misses == 0) {
                firstMiss = u;
            }
            ++misses;
        }
    }
    char missDetail[192];
    std::snprintf(missDetail, sizeof(missDetail), "%d of %d sample() directions missed the rectangle (first at u=%g,%g)",
                  misses, kDrawCount, static_cast<double>(firstMiss.x), static_cast<double>(firstMiss.y));
    PT_EXPECT(ctx, misses == 0, missDetail);
    PT_EXPECT(ctx, rect->solidAngle > 0.0F, "a valid rectangle must subtend positive solid angle");
}

// A disk light facing -z at `centre`, frame (+x, -y, -z) as buildLights places an unrotated one.
DiskLight makeDisk(const glm::vec3& centre, float radius, bool twoSided = false) {
    return DiskLight{centre, glm::mat3(glm::vec3(1.0F, 0.0F, 0.0F), glm::vec3(0.0F, -1.0F, 0.0F), glm::vec3(0.0F, 0.0F, -1.0F)), radius,
                     glm::vec3(3.0F), twoSided};
}

// A disk tilted 35 degrees about x, so no symmetry plane is a coordinate plane and the ellipse frame is fully exercised.
DiskLight makeTiltedDisk(const glm::vec3& centre, float radius) {
    const glm::mat3 tilt = pathtracer::scene::rotationXyz(glm::vec3(35.0F, 0.0F, 0.0F));
    const DiskLight flat = makeDisk(centre, radius);
    return DiskLight{centre, tilt * flat.frame, radius, flat.radiance, false};
}

// Omega = the integral over the disk of h / d^3 dA, in polar coordinates about its centre, by nested adaptive Gauss-Kronrod in double.
template <typename Weight>
double integrateOverDisk(const DiskLight& disk, const glm::dvec3& p, Weight weight) {
    const glm::dvec3 centre(disk.centre);
    const glm::dvec3 tangent(disk.frame[0]);
    const glm::dvec3 bitangent(disk.frame[1]);
    const double height = std::abs(glm::dot(centre - p, glm::dvec3(disk.frame[2])));
    constexpr unsigned kMaxDepth = 20;
    constexpr double kTolerance = 1e-12;
    const auto ring = [&](double s) {
        const auto around = [&](double phi) {
            const glm::dvec3 toPoint = centre + (s * ((std::cos(phi) * tangent) + (std::sin(phi) * bitangent))) - p;
            const double distance = glm::length(toPoint);
            return weight(toPoint / distance) * height * s / (distance * distance * distance);
        };
        return boost::math::quadrature::gauss_kronrod<double, 61>::integrate(around, 0.0, 2.0 * std::numbers::pi, kMaxDepth, kTolerance);
    };
    return boost::math::quadrature::gauss_kronrod<double, 61>::integrate(ring, 0.0, static_cast<double>(disk.radius), kMaxDepth, kTolerance);
}

// Axis-aligned, with every coordinate dyadic and the offset along x: height, offset and radius reach the ellipse unrounded.
struct DiskPose {
    const char* label;
    glm::vec3 p;
};
const std::array<DiskPose, 6> kDiskPoses = {{{"on axis", {0.0F, 0.0F, 0.0F}},
                                             {"oblique", {1.5F, 0.0F, 0.0F}},
                                             {"far", {0.25F, 0.0F, -62.0F}},
                                             {"near the rim, across major", {0.75F, 0.0F, 1.875F}},
                                             {"over the face, in-plane major", {0.25F, 0.0F, 1.9375F}},
                                             {"almost touching", {0.125F, 0.0F, 1.9990234375F}}}};

// Eq. 19's closed form against the area integral: inputs exact, so the bound is the float result's rounding plus the oracle's own.
PT_CHECK(disk_light_solid_angle, Fast, Exact) {
    const DiskLight disk = makeDisk(glm::vec3(0.0F, 0.0F, 2.0F), 0.5F);
    ctx.plan(static_cast<int>(kDiskPoses.size()));
    for (const DiskPose& pose : kDiskPoses) {
        const std::optional<pathtracer::scene::SphericalEllipse> ellipse = pathtracer::scene::buildSphericalEllipse(disk, pose.p);
        const double oracle = integrateOverDisk(disk, glm::dvec3(pose.p), [](const glm::dvec3&) { return 1.0; });
        const double bound = oracle * (std::numeric_limits<float>::epsilon() + 1e-10);
        const double analytic = ellipse ? static_cast<double>(ellipse->solidAngle) : -1.0;
        char detail[192];
        std::snprintf(detail, sizeof(detail), "%s: closed form %.9g sr, quadrature %.9g sr, |difference| %.3g > bound %.3g", pose.label, analytic,
                      oracle, std::abs(analytic - oracle), bound);
        PT_EXPECT(ctx, std::abs(analytic - oracle) <= bound, detail);
    }
}

// Every sample lands on the disk: in its plane and within its radius, to a few float roundings of the coordinates that place it.
PT_CHECK(disk_light_samples_land_on_the_disk, Fast, Exact) {
    const std::array<DiskLight, 2> disks = {makeDisk(glm::vec3(0.0F, 0.0F, 2.0F), 0.5F), makeTiltedDisk(glm::vec3(0.1F, 0.2F, 2.0F), 0.7F)};
    constexpr int kDraws = 4096;
    // Float error scales with the magnitudes summed, |p| + |q - p| + |c|: 8 roundings, one per float operation between u and the offset.
    constexpr float kRoundings = 8.0F * std::numeric_limits<float>::epsilon();
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    ctx.plan(static_cast<int>(disks.size() * kDiskPoses.size()));
    for (const DiskLight& disk : disks) {
        for (const DiskPose& pose : kDiskPoses) {
            const std::optional<pathtracer::scene::SphericalEllipse> ellipse = pathtracer::scene::buildSphericalEllipse(disk, pose.p);
            int off = 0;
            for (int i = 0; ellipse && i < kDraws; ++i) {
                const glm::vec3 q = ellipse->sample(glm::vec2(unit(rng), unit(rng)));
                const float slack = kRoundings * (glm::length(pose.p) + glm::length(q - pose.p) + glm::length(disk.centre));
                const glm::vec3 offset = q - disk.centre;
                off += std::abs(glm::dot(offset, disk.frame[2])) <= slack && glm::length(offset) <= disk.radius + slack ? 0 : 1;
            }
            PT_EXPECT(ctx, ellipse && off == 0, std::string(pose.label) + ": " + std::to_string(off) + " samples off the disk");
        }
    }
}

// The two-sided t band about a replicated estimator's mean: its replicate means are iid, so their standard error carries it.
tools::stats::Band meanBand(const tools::stats::Welford& estimate, double alpha) {
    const double half = tools::stats::studentTTwoSided(alpha, static_cast<double>(estimate.count() - 1)) * estimate.standardError();
    return {estimate.mean() - half, estimate.mean() + half};
}

// A smooth, asymmetric test function of direction: uniform sampling's mean of it must equal its solid-angle average.
double probe(const glm::dvec3& dir) {
    const glm::dvec3 e1 = glm::normalize(glm::dvec3(0.3, -0.5, 0.81));
    const glm::dvec3 e2 = glm::normalize(glm::dvec3(0.7, 0.2, -0.68));
    const double along = glm::dot(dir, e2);
    return glm::dot(dir, e1) + (along * along);
}

// Replicated means of probe over draws of `draw`, against the oracle average: a non-uniform map biases the mean whatever the support.
template <typename Draw>
void expectUniform(tools::check::Context& ctx, const char* label, double oracle, Draw draw) {
    constexpr int kPerReplicate = 8192;
    const std::uint64_t seed = ctx.subSeed(label);
    tools::stats::Welford replicates;
    // Owen-scrambled replicates are iid (Owen 1997), so the t band over their means is exact in distribution.
    for (int r = 0; r < tools::stats::kReplicates; ++r) {
        double sum = 0.0;
        for (int i = 0; i < kPerReplicate; ++i) {
            Sampler sampler(0, 0, i, kPerReplicate, static_cast<std::uint32_t>(seed + static_cast<std::uint64_t>(r)));
            sum += probe(glm::dvec3(draw(sampler)));
        }
        replicates.add(sum / kPerReplicate);
    }
    const tools::stats::Band band = meanBand(replicates, ctx.alpha());
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%s: sampled mean %.6f, band [%.6f, %.6f], solid-angle average %.6f", label, replicates.mean(), band.lo,
                  band.hi, oracle);
    PT_EXPECT(ctx, band.contains(oracle), detail);
}

// The radial map is area-preserving: E[g] over its directions is g's solid-angle average over the disk, from the quadrature oracle.
PT_CHECK(disk_light_sampling_is_uniform_in_solid_angle, Slow, Statistical) {
    const DiskLight disk = makeTiltedDisk(glm::vec3(0.1F, 0.2F, 2.0F), 0.7F);
    ctx.plan(static_cast<int>(kDiskPoses.size()));
    for (const DiskPose& pose : kDiskPoses) {
        const std::optional<pathtracer::scene::SphericalEllipse> ellipse = pathtracer::scene::buildSphericalEllipse(disk, pose.p);
        if (!ellipse) {
            PT_EXPECT(ctx, false, std::string(pose.label) + ": no ellipse for a disk off its plane");
            continue;
        }
        const glm::dvec3 p(pose.p);
        const double oracle = integrateOverDisk(disk, p, probe) / integrateOverDisk(disk, p, [](const glm::dvec3&) { return 1.0; });
        expectUniform(ctx, pose.label, oracle, [&](Sampler& sampler) { return glm::normalize(ellipse->sample(sampler.next2D()) - pose.p); });
    }
}

// Cone sampling against g's average over the cap, integrated in the cone's own frame: the samples' law-of-cosines step is not reused.
PT_CHECK(sphere_light_sampling_is_uniform_in_solid_angle, Slow, Statistical) {
    const SphereLight sphere{glm::vec3(0.3F, -0.2F, 2.0F), glm::mat3(1.0F), 0.6F, glm::vec3(3.0F)};
    const std::array<DiskPose, 3> poses = {{{"distant", {0.0F, 0.0F, -6.0F}}, {"near", {0.0F, 0.0F, 0.0F}}, {"grazing", {1.0F, -0.2F, 2.0F}}}};
    const std::vector<AreaLight> lights{sphere};
    const LightSet set(nullptr, glm::vec3(0.0F), 1.0F, lights);
    ctx.plan(static_cast<int>(poses.size()) * 2);
    for (const DiskPose& pose : poses) {
        const glm::dvec3 axis = glm::normalize(glm::dvec3(sphere.centre - pose.p));
        const double sinMax = static_cast<double>(sphere.radius) / glm::length(glm::dvec3(sphere.centre - pose.p));
        const double thetaMax = std::asin(sinMax);
        const glm::dvec3 a = glm::normalize(glm::cross(axis, std::abs(axis.x) < 0.9 ? glm::dvec3(1.0, 0.0, 0.0) : glm::dvec3(0.0, 1.0, 0.0)));
        const glm::dvec3 b = glm::cross(axis, a);
        const auto ring = [&](double theta) {
            const auto around = [&](double phi) {
                return probe((std::cos(theta) * axis) + (std::sin(theta) * ((std::cos(phi) * a) + (std::sin(phi) * b)))) * std::sin(theta);
            };
            return boost::math::quadrature::gauss_kronrod<double, 61>::integrate(around, 0.0, 2.0 * std::numbers::pi, 20, 1e-12);
        };
        const double capArea = 2.0 * std::numbers::pi * (1.0 - std::cos(thetaMax));
        const double oracle = boost::math::quadrature::gauss_kronrod<double, 61>::integrate(ring, 0.0, thetaMax, 20, 1e-12) / capArea;
        // The pdf is 1/Omega exactly: the cap's solid angle, here in double from the oracle's own angle.
        Sampler pdfSampler(0, 0, 0, 1, 1U);
        const std::optional<LightSample> one = set.sample(pose.p, pdfSampler);
        const double relative = one ? std::abs((1.0 / static_cast<double>(one->pdf)) - capArea) / capArea : 1.0;
        // Eight float roundings separate it from the exact cap: d^2 (3), r^2, the ratio, the root, the sum and the 2 pi product.
        PT_EXPECT(ctx, relative <= 8.0 * std::numeric_limits<float>::epsilon(),
                  std::string(pose.label) + ": pdf is not 1 / the cap's solid angle, relative error " + std::to_string(relative));
        expectUniform(ctx, pose.label, oracle, [&](Sampler& sampler) { return set.sample(pose.p, sampler)->direction; });
    }
}


// A closed tetrahedron above the origin, faces wound outward: its lower faces show their fronts to p, its upper ones their backs.
MeshLight makeTetrahedronLight() {
    const std::array<glm::vec3, 4> v = {glm::vec3(0.0F, 0.0F, 1.5F), glm::vec3(0.6F, 0.0F, 2.4F), glm::vec3(-0.3F, 0.52F, 2.4F),
                                        glm::vec3(-0.3F, -0.52F, 2.4F)};
    const glm::vec3 centroid = 0.25F * (v[0] + v[1] + v[2] + v[3]);
    std::vector<Triangle> faces;
    for (const auto& [a, b, c] : {std::tuple{0, 1, 2}, std::tuple{0, 2, 3}, std::tuple{0, 3, 1}, std::tuple{1, 3, 2}}) {
        Triangle face{v[static_cast<std::size_t>(a)], v[static_cast<std::size_t>(b)], v[static_cast<std::size_t>(c)]};
        if (glm::dot(glm::cross(face.v1 - face.v0, face.v2 - face.v0), face.v0 - centroid) < 0.0F) {
            std::swap(face.v1, face.v2);
        }
        faces.push_back(face);
    }
    return MeshLight(std::move(faces), glm::vec3(3.0F), false);
}

// One light of each shape above the origin, each subtending enough for BSDF sampling alone to converge: the MIS rows' fixtures.
std::vector<std::pair<const char*, AreaLight>> areaLightCases() {
    return {{"quad", QuadLight{glm::vec3(-0.5F, -0.5F, 2.0F), glm::vec3(0.0F, 1.0F, 0.0F), glm::vec3(1.0F, 0.0F, 0.0F), glm::vec3(3.0F), false}},
            {"disk", makeTiltedDisk(glm::vec3(0.2F, -0.1F, 1.8F), 0.6F)},
            {"sphere", SphereLight{glm::vec3(0.3F, 0.0F, 2.0F), glm::mat3(1.0F), 0.5F, glm::vec3(3.0F)}},
            {"mesh", makeTetrahedronLight()}};
}

// The pdf NEE draws a direction with is the pdf the emitter-hit branch recomputes there: MIS weights sum to 1 only if both agree.
PT_CHECK(area_light_sample_and_hit_pdfs_agree, Fast, Exact) {
    const std::vector<std::pair<const char*, AreaLight>> cases = areaLightCases();
    const std::array<glm::vec3, 3> points = {glm::vec3(0.0F), glm::vec3(1.25F, 0.5F, 0.25F), glm::vec3(-0.75F, 0.25F, 1.0F)};
    constexpr int kDraws = 2048;
    ctx.plan(static_cast<int>(cases.size() * points.size()));
    for (const auto& [label, light] : cases) {
        const std::vector<AreaLight> lights{light};
        const LightSet set(nullptr, glm::vec3(0.0F), 1.0F, lights);
        for (const glm::vec3& p : points) {
            int disagreements = 0;
            for (int i = 0; i < kDraws; ++i) {
                Sampler sampler(0, 0, i, kDraws, static_cast<std::uint32_t>(ctx.subSeed(label)));
                const std::optional<LightSample> drawn = set.sample(p, sampler);
                if (!drawn) {
                    continue;
                }
                const std::optional<LightHit> hit = hitLight(light, p, drawn->direction);
                // A mesh pdf rereads the point through p + t dir, roundings of the path that cos_l amplifies; the rest reread p alone.
                const float roundings = 16.0F * std::numeric_limits<float>::epsilon();
                const float cosLight = hit ? std::abs(glm::dot(drawn->direction, hit->normal)) : 0.0F;
                const bool mesh = std::holds_alternative<MeshLight>(light);
                // A closed mesh's far face is drawn too, behind its near one: a different point, its NEE sample self-occluded in a render.
                if (mesh && hit && std::abs(glm::length(hit->point - p) - drawn->distance) > roundings * drawn->distance / cosLight) {
                    continue;
                }
                const float slack = mesh ? roundings / cosLight : 0.0F;
                const float recomputed = hit ? set.pdfArea(0, p, hit->point, hit->normal) : 0.0F;
                disagreements += hit && std::abs(recomputed - drawn->pdf) <= slack * drawn->pdf ? 0 : 1;
            }
            PT_EXPECT(ctx, disagreements == 0, std::string(label) + ": " + std::to_string(disagreements) + " draws whose hit pdf differs");
        }
    }
}

// referenceLo's method over the directions the light subtends: Lo = integral of the cosine-weighted evaluateBsdf * Le.
float referenceLoArea(const OpenPbrInputs<Constant>& params, const glm::vec3& wo, const AreaLight& light, const glm::vec3& p,
                      int sampleCount, std::mt19937& rng) {
    constexpr float kUniformPdf = 1.0F / (2.0F * kPi);
    glm::vec3 accum(0.0F);
    for (int i = 0; i < sampleCount; ++i) {
        const glm::vec3 wi = sampleUniformHemisphere(rng);
        if (const std::optional<LightHit> hit = hitLight(light, p, wi)) {
            accum += pathtracer::scene::evaluateBsdf(params, wo, wi) * emittedAlong(light, wi, hit->normal) / kUniformPdf;
        }
    }
    return std::max({accum.x, accum.y, accum.z}) / static_cast<float>(sampleCount);
}

// Mirrors misCombinedLo with LightSet({light}) and hitLight for "always reaches it": a BSDF ray can miss a finite light.
float misCombinedLoArea(const OpenPbrInputs<Constant>& params, const glm::vec3& wo, const AreaLight& light, const glm::vec3& p,
                        int sampleCount, std::uint32_t seed) {
    const std::vector<AreaLight> lights{light};
    const LightSet set(nullptr, glm::vec3(0.0F), 1.0F, lights);
    glm::vec3 accum(0.0F);
    for (int i = 0; i < sampleCount; ++i) {
        Sampler sampler(0, 0, i, sampleCount, seed);

        // NEE.
        const std::optional<LightSample> lightSample = set.sample(p, sampler);
        if (lightSample.has_value() && lightSample->direction.z > 0.0F) {
            const glm::vec3 bsdfValue = pathtracer::scene::evaluateBsdf(params, wo, lightSample->direction);
            const float bsdfPdf = pathtracer::scene::pdfBsdf(params, wo, lightSample->direction);
            if (bsdfPdf > 0.0F && (bsdfValue.x > 0.0F || bsdfValue.y > 0.0F || bsdfValue.z > 0.0F)) {
                const float lightPdf2 = lightSample->pdf * lightSample->pdf;
                const float misWeight = lightPdf2 / (lightPdf2 + (bsdfPdf * bsdfPdf));
                accum += bsdfValue * misWeight * lightSample->radiance / lightSample->pdf;
            }
        }

        // BSDF-sampled.
        const std::optional<pathtracer::scene::BsdfSample> sample = pathtracer::scene::sampleBsdf(params, wo, sampler);
        if (sample.has_value() && sample->type != LobeType::Transmission) {
            if (const std::optional<LightHit> hit = hitLight(light, p, sample->wiLocal)) {
                const float bsdfPdf = pathtracer::scene::pdfBsdf(params, wo, sample->wiLocal);
                const float lightPdf = set.pdfArea(0, p, hit->point, hit->normal);
                const float bsdfPdf2 = bsdfPdf * bsdfPdf;
                const float misWeight = bsdfPdf2 / (bsdfPdf2 + (lightPdf * lightPdf));
                accum += sample->throughputWeight * set.radianceToward(0, sample->wiLocal, hit->normal) * misWeight;
            }
        }
    }
    return std::max({accum.x, accum.y, accum.z}) / static_cast<float>(sampleCount);
}

// mis_agreement_environment's identity against one area light, converted the same way and for the same reason: both sides are estimators.
void expectMisAgreement(tools::check::Context& ctx, const char* shape, const AreaLight& light) {
    constexpr int kTotalSamples = 200000;
    constexpr int kPerReplicate = kTotalSamples / tools::stats::kReplicates;
    const std::array<float, 3> roughnesses = {0.25F, 0.5F, 1.0F};
    const std::array<float, 2> metallics = {0.0F, 1.0F};
    const std::array<float, 3> ndotVs = {0.2F, 0.6F, 1.0F};
    const glm::vec3 p(0.0F);

    ctx.plan(static_cast<int>(roughnesses.size() * metallics.size() * ndotVs.size()));
    for (float roughness : roughnesses) {
        for (float metallic : metallics) {
            for (float ndotV : ndotVs) {
                char label[96];
                std::snprintf(label, sizeof(label), "%s r=%g m=%g n=%g", shape, static_cast<double>(roughness), static_cast<double>(metallic),
                              static_cast<double>(ndotV));
                const OpenPbrInputs<Constant> params = makeParams(roughness, metallic);
                const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);

                const std::uint64_t rowSeed = ctx.subSeed(label);
                std::mt19937 referenceRng(static_cast<std::mt19937::result_type>(rowSeed));
                tools::stats::Welford reference;
                tools::stats::Welford combined;
                for (int r = 0; r < tools::stats::kReplicates; ++r) {
                    reference.add(referenceLoArea(params, wo, light, p, kPerReplicate, referenceRng));
                    combined.add(misCombinedLoArea(params, wo, light, p, kPerReplicate, static_cast<std::uint32_t>(rowSeed + r)));
                }

                const double difference = combined.mean() - reference.mean();
                const tools::stats::Band band = tools::stats::differenceBand(reference, combined, ctx.alpha());
                char detail[256];
                std::snprintf(detail, sizeof(detail), "%s: reference %.5f, MIS %.5f, difference %+.3e vs +/-%.3e", label, reference.mean(),
                              combined.mean(), difference, band.halfWidth());
                PT_EXPECT(ctx, band.contains(difference), detail);
            }
        }
    }
}

PT_CHECK(mis_agreement_quad_light, Slow, Statistical) {
    expectMisAgreement(ctx, "quad", areaLightCases()[0].second);
}

PT_CHECK(mis_agreement_disk_light, Slow, Statistical) {
    expectMisAgreement(ctx, "disk", areaLightCases()[1].second);
}

PT_CHECK(mis_agreement_sphere_light, Slow, Statistical) {
    expectMisAgreement(ctx, "sphere", areaLightCases()[2].second);
}

PT_CHECK(mis_agreement_mesh_light, Slow, Statistical) {
    expectMisAgreement(ctx, "mesh", areaLightCases()[3].second);
}

}  // namespace

// latLongUvJacobian against the chart's own tangents: the u and v tangents map to (1, 0) and (0, 1), the radial direction to 0.
PT_CHECK(lat_long_jacobian_inverts_the_chart_tangents, Fast, Exact) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    // Clear of the poles, where u is singular; any radius, since uv reads only the direction.
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    std::uniform_real_distribution<float> band(0.1F, 0.9F);
    std::uniform_real_distribution<float> radius(0.5F, 2.0F);
    float worst = 0.0F;
    for (int i = 0; i < 1024; ++i) {
        const glm::vec2 uv(unit(rng), band(rng));
        const float r = radius(rng);
        const glm::vec3 local = r * pathtracer::scene::latLongDirection(uv);
        const float longitude = (uv.x - 0.5F) * glm::two_pi<float>();
        const float colatitude = uv.y * glm::pi<float>();
        const float sinColatitude = std::sin(colatitude);
        const float cosColatitude = std::cos(colatitude);
        const glm::vec3 perU = r * glm::two_pi<float>() * sinColatitude * glm::vec3(std::cos(longitude), 0.0F, -std::sin(longitude));
        const glm::vec3 perV =
            r * glm::pi<float>() * glm::vec3(cosColatitude * std::sin(longitude), -sinColatitude, cosColatitude * std::cos(longitude));
        const glm::mat3x2 jacobian = pathtracer::scene::latLongUvJacobian(local);
        // Entries scale as 1/(r sin) and tangents as r sin, so each product is O(1); the radial row cancels terms of size 1/sin.
        const float conditioning = 1.0F / sinColatitude;
        // The radial rate against the terms it cancels: |J l| / (|J|_F |l|), 0 exactly in real arithmetic.
        const float frobenius =
            std::sqrt(glm::dot(jacobian[0], jacobian[0]) + glm::dot(jacobian[1], jacobian[1]) + glm::dot(jacobian[2], jacobian[2]));
        const float error = std::max({glm::length((jacobian * perU) - glm::vec2(1.0F, 0.0F)),
                                      glm::length((jacobian * perV) - glm::vec2(0.0F, 1.0F)),
                                      glm::length(jacobian * local) / (frobenius * glm::length(local))});
        worst = std::max(worst, error / conditioning);
    }
    // 3 terms of (6 + 2) roundings is 24u = 12 eps per unit conditioning; 16 eps adds the 1-ulp sin and cos the tangents are built from.
    const float bound = 16.0F * std::numeric_limits<float>::epsilon();
    ctx.plan(1);
    char detail[160];
    std::snprintf(detail, sizeof(detail), "the Jacobian misses the chart's tangents by %.3g per unit conditioning, over %.3g",
                  static_cast<double>(worst), static_cast<double>(bound));
    PT_EXPECT(ctx, worst <= bound, detail);
}

PT_CHECK_MAIN("nee")
