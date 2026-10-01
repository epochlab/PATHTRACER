// NEE+MIS (Veach & Guibas 1995 power heuristic) against EnvironmentMap and LightSet.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/epsilon.hpp>

#include "pathtracer/gfx/hdr_image.h"
#include "check.h"
#include "fixtures.h"
#include "stats.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/environment_map.h"
#include "pathtracer/scene/light.h"
#include "pathtracer/scene/sampler.h"

namespace {

using pathtracer::scene::BsdfParams;
using pathtracer::scene::EnvironmentMap;
using pathtracer::scene::LightSample;
using pathtracer::scene::LightSet;
using pathtracer::scene::LobeType;
using pathtracer::scene::QuadLight;
using pathtracer::scene::Sampler;

using tools::fixtures::kPi;
using tools::fixtures::makeUniformEnvironment;
using tools::fixtures::referenceLo;
using tools::fixtures::sampleUniformHemisphere;

BsdfParams makeParams(float roughness, float metallic) {
    const glm::vec3 baseColor(1.0F);  // worst case: full white albedo
    const glm::vec3 f0 = glm::mix(glm::vec3(0.04F), baseColor, metallic);
    return BsdfParams{baseColor,   metallic, roughness, f0, /*edgeTint=*/glm::vec3(1.0F),
                       /*ior=*/1.5F, /*transmissionFactor=*/0.0F, /*diffuseRoughness=*/0.0F,
                       pathtracer::scene::eonAlbedoInversion(baseColor, 0.0F),
                       /*transmissionTint=*/glm::vec3(1.0F)};
}

// Dim background with one bright patch: a uniform map gives linear CDFs, where a mis-scaled Jacobian or an off-by-one bin would pass.
EnvironmentMap makeStructuredEnvironment(pathtracer::gfx::ScalarType type) {
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
    return EnvironmentMap(tools::fixtures::makeImageTexture(kWidth, kHeight, kChannels, rgb, type));
}

// sample returns a density, pdf() recovers it from a direction; MIS uses both, so disagreement corrupts every MIS weight in the renderer.
PT_CHECK(environment_pdf_consistency, Fast, Exact) {
    constexpr int kSampleCount = 20000;
    constexpr float kTolerance = 1e-3F;
    // Non-zero rotation: sampling rotates by +angle and querying by -angle, so a sign slip cancels at 0 and shows only here.
    constexpr float kRotation = 0.7F;
    constexpr std::array<pathtracer::gfx::ScalarType, 2> kTypes = {pathtracer::gfx::ScalarType::Float32,
                                                               pathtracer::gfx::ScalarType::Float16};
    ctx.plan(static_cast<int>(kTypes.size()));
    for (const pathtracer::gfx::ScalarType type : kTypes) {
        const EnvironmentMap env = makeStructuredEnvironment(type);
        std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
        std::uniform_real_distribution<float> unit(0.0F, 1.0F);
        int worstIndex = -1;
        float worstRelative = 0.0F;
        for (int i = 0; i < kSampleCount; ++i) {
            const EnvironmentMap::EnvSample sample =
                env.importanceSampleDirection(glm::vec2(unit(rng), unit(rng)), pathtracer::scene::YRotation::of(kRotation));
            const float queried = env.pdf(sample.direction, pathtracer::scene::YRotation::of(kRotation));
            const float relative = std::fabs(queried - sample.pdf) / std::max(sample.pdf, 1e-6F);
            if (relative > worstRelative) {
                worstRelative = relative;
                worstIndex = i;
            }
        }
        // Two code paths over one direction: exactness with a float tolerance, not statistics -- they agree or every MIS weight is wrong.
        char detail[224];
        std::snprintf(detail, sizeof(detail),
                      "%s: worst relative mismatch %.3e at sample %d, between importanceSampleDirection's own pdf and pdf()",
                      pathtracer::gfx::scalarTypeName(type), static_cast<double>(worstRelative), worstIndex);
        PT_EXPECT(ctx, worstRelative <= kTolerance, detail);
    }
}

// CDFs must come from the values the map returns: 1 + 2^-11 stores as 1.0 in binary16.
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

    // Direction through a texel's centre, inverting equirectTexelOf's (u, v) = (phi / 2pi + 1/2, theta / pi).
    const auto centre = [](int x, int y) {
        const float phi = ((((static_cast<float>(x) + 0.5F) / kWidth) - 0.5F) * 2.0F * glm::pi<float>());
        const float theta = ((static_cast<float>(y) + 0.5F) / kHeight) * glm::pi<float>();
        return glm::vec3(std::sin(theta) * std::sin(phi), std::cos(theta), std::sin(theta) * std::cos(phi));
    };
    ctx.plan(2);
    for (const pathtracer::gfx::ScalarType type : {pathtracer::gfx::ScalarType::Float16, pathtracer::gfx::ScalarType::Float32}) {
        const pathtracer::gfx::ImageTexture image = tools::fixtures::makeImageTexture(kWidth, kHeight, kChannels, rgb, type);
        const float storedRatio = image.texel(kPatchX, kPatchY).g / image.texel(kBackgroundX, kPatchY).g;
        const float sourceRatio = midpoint;
        const EnvironmentMap env(image);
        // Same row, so sin(theta) cancels and the solid-angle pdf ratio is the luminance ratio.
        const float pdfRatio = env.pdf(centre(kPatchX, kPatchY)) / env.pdf(centre(kBackgroundX, kPatchY));
        const bool tracksStored = type == pathtracer::gfx::ScalarType::Float16
                                      ? std::fabs(pdfRatio - storedRatio) < std::fabs(pdfRatio - sourceRatio)
                                      : std::fabs(pdfRatio - sourceRatio) < std::fabs(pdfRatio - 1.0F);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "%s: pdf ratio %.9g, stored luminance ratio %.9g, source ratio %.9g",
                      pathtracer::gfx::scalarTypeName(type), static_cast<double>(pdfRatio),
                      static_cast<double>(storedRatio), static_cast<double>(sourceRatio));
        PT_EXPECT(ctx, tracksStored, detail);
    }
}

// MIS-combined NEE + BSDF estimator mirroring tracePath: power heuristic, and with no geometry both strategies reach the environment.
float misCombinedLo(const BsdfParams& params, const glm::vec3& wo, const EnvironmentMap& env,
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
                accum += bsdfValue * lightSample.direction.z * misWeight / lightSample.pdf;
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
    const EnvironmentMap env(
        tools::fixtures::makeImageTexture(kWidth, kHeight, kChannels, rgb, pathtracer::gfx::ScalarType::Float32));

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

// MIS weights sum to 1, so both strategies must evaluate ONE Le. Structured, not uniform: on a uniform map every lookup agrees.
PT_CHECK(environment_nee_and_miss_share_one_radiance, Fast, Exact) {
    const EnvironmentMap env = makeStructuredEnvironment(pathtracer::gfx::ScalarType::Float32);
    const std::vector<QuadLight> noQuads;
    const LightSet lights(&env, /*envRotationRadians=*/0.0F, /*envExposure=*/1.0F, noQuads);

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
                const BsdfParams params = makeParams(roughness, metallic);
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

// Re-derives ray/rectangle from world geometry (edge0 perpendicular to edge1), so a buildSphericalRectangle frame bug cannot repeat.
bool directionHitsQuad(const QuadLight& quad, const glm::vec3& p, const glm::vec3& dir) {
    const glm::vec3 normal = glm::normalize(glm::cross(quad.edge0, quad.edge1));
    const float denom = glm::dot(dir, normal);
    if (std::fabs(denom) < 1e-8F) {
        return false;  // parallel to the plane
    }
    const float t = glm::dot(quad.origin - p, normal) / denom;
    if (t <= 0.0F) {
        return false;  // plane is behind p
    }
    const glm::vec3 hit = (p + (t * dir)) - quad.origin;
    const float u = glm::dot(hit, quad.edge0) / glm::dot(quad.edge0, quad.edge0);
    const float v = glm::dot(hit, quad.edge1) / glm::dot(quad.edge1, quad.edge1);
    return u >= 0.0F && u <= 1.0F && v >= 0.0F && v <= 1.0F;
}

// Girard solid angle against a uniform-sphere + directionHitsQuad oracle sharing no code, and every sample() direction against it too.
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
        hits += directionHitsQuad(quad, p, sampleUniformSphere(rng)) ? 1 : 0;
    }
    // Wilson interval, not Wald: Wald's width collapses near 0 or 1, where a small light lands. Scaled to solid angle by the same 4*pi.
    const tools::stats::Band fraction = tools::stats::wilsonBand(hits, kSampleCount, ctx.alpha());
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
        if (!directionHitsQuad(quad, p, dir)) {
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

// referenceLo's method over the directions the quad subtends: Lo = integral of evaluateBsdf * Le * cos, with Le constant where visible.
float referenceLoQuad(const BsdfParams& params, const glm::vec3& wo, const QuadLight& quad,
                       const glm::vec3& p, int sampleCount, std::mt19937& rng) {
    constexpr float kUniformPdf = 1.0F / (2.0F * kPi);
    glm::vec3 accum(0.0F);
    for (int i = 0; i < sampleCount; ++i) {
        const glm::vec3 wi = sampleUniformHemisphere(rng);
        if (directionHitsQuad(quad, p, wi)) {
            accum += pathtracer::scene::evaluateBsdf(params, wo, wi) * wi.z * quad.radiance / kUniformPdf;
        }
    }
    return std::max({accum.x, accum.y, accum.z}) / static_cast<float>(sampleCount);
}

// Mirrors misCombinedLo with LightSet({quad}) and directionHitsQuad for "always reaches it": a BSDF ray can miss a finite rectangle.
float misCombinedLoQuad(const BsdfParams& params, const glm::vec3& wo, const QuadLight& quad,
                         const glm::vec3& p, int sampleCount, std::uint32_t seed) {
    const std::vector<QuadLight> quads{quad};
    const LightSet lights(nullptr, 0.0F, 1.0F, quads);
    glm::vec3 accum(0.0F);
    for (int i = 0; i < sampleCount; ++i) {
        Sampler sampler(0, 0, i, sampleCount, seed);

        // NEE.
        const std::optional<LightSample> lightSample = lights.sample(p, sampler);
        if (lightSample.has_value() && lightSample->direction.z > 0.0F) {
            const glm::vec3 bsdfValue = pathtracer::scene::evaluateBsdf(params, wo, lightSample->direction);
            const float bsdfPdf = pathtracer::scene::pdfBsdf(params, wo, lightSample->direction);
            if (bsdfPdf > 0.0F && (bsdfValue.x > 0.0F || bsdfValue.y > 0.0F || bsdfValue.z > 0.0F)) {
                const float lightPdf2 = lightSample->pdf * lightSample->pdf;
                const float bsdfPdf2 = bsdfPdf * bsdfPdf;
                const float misWeight = lightPdf2 / (lightPdf2 + bsdfPdf2);
                accum += bsdfValue * lightSample->direction.z * misWeight * lightSample->radiance /
                         lightSample->pdf;
            }
        }

        // BSDF-sampled.
        const std::optional<pathtracer::scene::BsdfSample> sample =
            pathtracer::scene::sampleBsdf(params, wo, sampler);
        if (sample.has_value() && sample->type != LobeType::Transmission &&
            directionHitsQuad(quad, p, sample->wiLocal)) {
            const float bsdfPdf = pathtracer::scene::pdfBsdf(params, wo, sample->wiLocal);
            const float lightPdf = lights.pdfQuad(0, p);
            const float bsdfPdf2 = bsdfPdf * bsdfPdf;
            const float lightPdf2 = lightPdf * lightPdf;
            const float misWeight = bsdfPdf2 / (bsdfPdf2 + lightPdf2);
            accum += sample->throughputWeight * lights.quadRadianceToward(0, sample->wiLocal) * misWeight;
        }
    }
    return std::max({accum.x, accum.y, accum.z}) / static_cast<float>(sampleCount);
}

// mis_agreement_environment's identity against an area light, converted the same way and for the same reason: both sides are estimators.
PT_CHECK(mis_agreement_quad_light, Slow, Statistical) {
    constexpr int kTotalSamples = 200000;
    constexpr int kPerReplicate = kTotalSamples / tools::stats::kReplicates;
    const std::array<float, 3> roughnesses = {0.25F, 0.5F, 1.0F};
    const std::array<float, 2> metallics = {0.0F, 1.0F};
    const std::array<float, 3> ndotVs = {0.2F, 0.6F, 1.0F};

    // A 1x1 rectangle at height 2: big enough for BSDF sampling alone to converge, small enough that light sampling matters at grazing wo.
    const QuadLight quad{glm::vec3(-0.5F, -0.5F, 2.0F), glm::vec3(0.0F, 1.0F, 0.0F),
                          glm::vec3(1.0F, 0.0F, 0.0F), glm::vec3(3.0F), false};
    const glm::vec3 p(0.0F);

    ctx.plan(static_cast<int>(roughnesses.size() * metallics.size() * ndotVs.size()));
    for (float roughness : roughnesses) {
        for (float metallic : metallics) {
            for (float ndotV : ndotVs) {
                char label[96];
                std::snprintf(label, sizeof(label), "quad r=%g m=%g n=%g", static_cast<double>(roughness),
                              static_cast<double>(metallic), static_cast<double>(ndotV));
                const BsdfParams params = makeParams(roughness, metallic);
                const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);

                const std::uint64_t rowSeed = ctx.subSeed(label);
                std::mt19937 referenceRng(static_cast<std::mt19937::result_type>(rowSeed));
                tools::stats::Welford reference;
                tools::stats::Welford combined;
                for (int r = 0; r < tools::stats::kReplicates; ++r) {
                    reference.add(referenceLoQuad(params, wo, quad, p, kPerReplicate, referenceRng));
                    combined.add(misCombinedLoQuad(params, wo, quad, p, kPerReplicate,
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

}  // namespace

PT_CHECK_MAIN("nee")
