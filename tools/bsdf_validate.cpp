// Upper-bound checks on scene::bsdf: the combined pdf never exceeds the lobe-selection mass, no furnace returns more energy than it got.

#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

#include <glm/glm.hpp>

#include "check.h"
#include "conductor_reference.h"
#include "fixtures.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/fresnel_dielectric.h"
#include "pathtracer/scene/sampler.h"

#include "stats.h"

namespace {

using pathtracer::scene::Constant;
using pathtracer::scene::OpenPbrInputs;
using Surface = OpenPbrInputs<Constant>;
using tools::stats::simpson;

using tools::fixtures::kPi;
using tools::fixtures::sampleUniformHemisphere;
using tools::reference::cosineAverageFresnel;
using tools::reference::kMuBar;
using tools::reference::referenceF82;

// Each check keeps its own `ok` accumulator and per-row diagnostics carrying parameters, measurement and reference, reporting one verdict.
void finish(tools::check::Context& ctx, bool ok, const char* what) {
    ctx.plan(1);
    PT_EXPECT(ctx, ok, what);
}

// OpenPBR inputs over a white base, worst case for albedo, edited by the caller and resolved as a hit resolves them.
template <typename Edit>
Surface paramsWith(Edit edit) {
    Surface inputs;
    inputs.baseColor = glm::vec3(1.0F);
    edit(inputs);
    return inputs;
}

// The slabs' reflected Fresnel at a view cosine, through a closure facing the macro normal, as the Fresnel AOV reads it.
glm::vec3 viewFresnel(const Surface& inputs, float cosTheta) {
    return pathtracer::scene::fresnelAtViewAngle(pathtracer::scene::makeBsdfClosure(inputs, glm::vec3(0.0F, 0.0F, 1.0F)), cosTheta);
}

// OpenPBR's thin film over the inputs: thin_film_weight, thin_film_thickness in micrometres, thin_film_ior.
Surface filmed(Surface inputs, float weight, float thicknessUm, float ior) {
    inputs.thinFilmWeight = weight;
    inputs.thinFilmThickness = thicknessUm;
    inputs.thinFilmIor = ior;
    return inputs;
}

// specular_color defaults to white, F82's no-dip Schlick edge, so a tinted metal is a strict extension of the swept coverage.
Surface makeParams(float roughness, float metallic, float transmission, float diffuseRoughness = 0.0F,
                      glm::vec3 specularColor = glm::vec3(1.0F)) {
    return paramsWith([&](OpenPbrInputs<Constant>& inputs) {
        inputs.specularRoughness = roughness;
        inputs.baseMetalness = metallic;
        inputs.transmissionWeight = transmission;
        inputs.baseDiffuseRoughness = diffuseRoughness;
        inputs.specularColor = specularColor;
    });
}

// A coloured dark metal (base_color 0.5): F0 below 1, so its Kulla-Conty lobe and F82 correction both carry weight.
Surface makeColoredMetalParams(float roughness, glm::vec3 specularColor = glm::vec3(1.0F)) {
    return paramsWith([&](OpenPbrInputs<Constant>& inputs) {
        inputs.baseColor = glm::vec3(0.5F);
        inputs.baseMetalness = 1.0F;
        inputs.specularRoughness = roughness;
        inputs.specularColor = specularColor;
    });
}

// Uniform-solid-angle hemisphere samples (z=u1, r=sqrt(1-u1^2)) integrate pdfBsdf, MIS-combined with sampleBsdf's density (Veach 1997 9.2).
PT_CHECK(pdf_normalization, Slow, Statistical) {
    std::mt19937 rng(7);
    constexpr int kUniformSamples = 50000;
    constexpr int kBsdfSamples = 50000;
    constexpr std::uint32_t kBsdfSeed = 7;
    constexpr float kTolerance = 0.05F;
    constexpr double kUniformPdf = 1.0 / (2.0 * kPi);
    // The balance-heuristic denominator, shared by both proposals: N1*q1 + N2*q2 with q2 == pdfBsdf.
    const auto combinedDensity = [](double p) {
        return (kUniformSamples * kUniformPdf) + (kBsdfSamples * p);
    };
    const std::array<float, 4> roughnesses = {0.05F, 0.25F, 0.5F, 1.0F};
    const std::array<float, 2> metallics = {0.0F, 1.0F};
    const std::array<float, 4> ndotVs = {0.2F, 0.6F, 1.0F, -0.6F};
    const std::array<float, 3> diffuseRoughnesses = {0.0F, 0.5F, 1.0F};

    bool ok = true;
    double worstIntegral = 0.0;
    for (float roughness : roughnesses) {
        for (float metallic : metallics) {
            for (float diffuseRoughness : diffuseRoughnesses) {
                for (float ndotV : ndotVs) {
                    const Surface params = makeParams(roughness, metallic, 0.0F, diffuseRoughness);
                    const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);
                    double integral = 0.0;
                    for (int i = 0; i < kUniformSamples; ++i) {
                        glm::vec3 wi = sampleUniformHemisphere(rng);
                        // pdfBsdf mirrors wi into wo's hemisphere, so a below-surface wo has zero +z density; the sample flips to match.
                        if (ndotV < 0.0F) {
                            wi.z = -wi.z;
                        }
                        const double p = pathtracer::scene::pdfBsdf(params, wo, wi);
                        integral += p / combinedDensity(p);
                    }
                    // sampleBsdf returns wi in woLocal's convention with the density it drew, so no second pdfBsdf call and no mirroring.
                    for (int i = 0; i < kBsdfSamples; ++i) {
                        pathtracer::scene::Sampler sampler(0, 0, i, kBsdfSamples, kBsdfSeed);
                        const std::optional<pathtracer::scene::BsdfSample> sample =
                            pathtracer::scene::sampleBsdf(params, wo, sampler);
                        if (sample.has_value()) {
                            integral += sample->pdf / combinedDensity(sample->pdf);
                        }
                    }
                    worstIntegral = std::max(worstIntegral, integral);
                    if (!(integral <= 1.0 + kTolerance)) {
                        std::cerr << "bsdf_validate: FAILED pdf normalization UPPER bound at roughness="
                                  << roughness << " metallic=" << metallic
                                  << " diffuseRoughness=" << diffuseRoughness << " ndotV=" << ndotV
                                  << " integral=" << integral << " (expected <= 1.0)\n";
                        ok = false;
                    }
                }
            }
        }
    }
    std::cout << "bsdf_validate: pdf normalization, worst integral " << worstIntegral << " (must be <= "
              << 1.0 + kTolerance << ")\n";
    finish(ctx, ok, "pdf_normalization failed; see the rows above");
    return;
}

// sampleBsdf's density must equal pdfBsdf at the direction returned: exact, both being the same arithmetic over one closure.
PT_CHECK(sample_density_consistency, Slow, Exact) {
    constexpr int kSampleCount = 8000;
    constexpr std::uint32_t kSeed = 11;
    const std::array<float, 4> roughnesses = {0.05F, 0.25F, 0.5F, 1.0F};
    const std::array<float, 3> metallics = {0.0F, 0.5F, 1.0F};
    const std::array<float, 3> transmissions = {0.0F, 0.5F, 1.0F};
    const std::array<float, 3> diffuseRoughnesses = {0.0F, 0.5F, 1.0F};
    const std::array<float, 4> ndotVs = {0.2F, 0.6F, 1.0F, -0.6F};

    bool ok = true;
    long long compared = 0;
    for (float roughness : roughnesses) {
        for (float metallic : metallics) {
            for (float transmission : transmissions) {
                for (float diffuseRoughness : diffuseRoughnesses) {
                    for (float ndotV : ndotVs) {
                        const Surface params =
                            makeParams(roughness, metallic, transmission, diffuseRoughness);
                        const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F,
                                            ndotV);
                        for (int i = 0; i < kSampleCount && ok; ++i) {
                            pathtracer::scene::Sampler sampler(0, 0, i, kSampleCount, kSeed);
                            const std::optional<pathtracer::scene::BsdfSample> sample =
                                pathtracer::scene::sampleBsdf(params, wo, sampler);
                            if (!sample.has_value()) {
                                continue;
                            }
                            ++compared;
                            const float reevaluated =
                                pathtracer::scene::pdfBsdf(params, wo, sample->wiLocal);
                            if (reevaluated == sample->pdf) {
                                continue;
                            }
                            std::cerr << "bsdf_validate: FAILED sample/pdf consistency at roughness="
                                      << roughness << " metallic=" << metallic
                                      << " transmission=" << transmission
                                      << " diffuseRoughness=" << diffuseRoughness
                                      << " ndotV=" << ndotV << " sample->pdf=" << sample->pdf
                                      << " pdfBsdf=" << reevaluated << '\n';
                            ok = false;
                        }
                    }
                }
            }
        }
    }
    std::cout << "bsdf_validate: sample/pdf consistency, " << compared
              << " sampled directions re-evaluated (exact equality)\n";
    finish(ctx, ok, "sample_density_consistency failed; see the rows above");
    return;
}

glm::vec3 furnaceLo(const Surface& params, const glm::vec3& wo, int sampleCount, std::uint32_t seed) {
    glm::vec3 accum(0.0F);
    for (int i = 0; i < sampleCount; ++i) {
        pathtracer::scene::Sampler sampler(0, 0, i, sampleCount, seed);
        const std::optional<pathtracer::scene::BsdfSample> sample =
            pathtracer::scene::sampleBsdf(params, wo, sampler);
        if (sample.has_value()) {
            accum += sample->throughputWeight;  // L0=1
        }
    }
    return accum / static_cast<float>(sampleCount);
}

float maxChannel(const glm::vec3& v) { return std::max({v.x, v.y, v.z}); }
float minChannel(const glm::vec3& v) { return std::min({v.x, v.y, v.z}); }

// Asserts the pass condition, not the failure one: a `min < lo || max > hi` form would be satisfied by a NaN and report success.
bool withinBand(const glm::vec3& value, float centre, float tolerance) {
    return minChannel(value) >= centre - tolerance && maxChannel(value) <= centre + tolerance;
}

// Support coverage: every direction with BSDF value must carry mixture density (Veach 1997 9.2 unbiasedness); exact, not a tolerance.
PT_CHECK(strategy_coverage, Fast, Exact) {
    constexpr int kMuNodes = 16;
    constexpr int kPhiNodes = 8;
    constexpr float kBandStep = 1.0F / 248.0F;
    std::vector<float> roughnesses = {0.05F, 0.25F, 0.5F, 1.0F};
    for (float roughness = 0.12F; roughness <= 0.18F; roughness += kBandStep) {
        roughnesses.push_back(roughness);
    }
    const std::array<float, 7> iors = {1.063F, 1.2F, 1.33F, 1.5F, 1.5168F, 2.0F, 2.4F};
    const std::array<float, 6> ndotVs = {0.95F, 0.6F, 0.2F, -0.2F, -0.6F, -0.95F};
    const std::array<float, 2> transmissions = {0.5F, 1.0F};

    bool ok = true;
    long long valued = 0;
    long long farValued = 0;
    long long uncovered = 0;
    for (float roughness : roughnesses) {
        for (float ior : iors) {
            for (float transmission : transmissions) {
                const Surface params = paramsWith([&](OpenPbrInputs<Constant>& inputs) {
                    inputs.specularRoughness = roughness;
                    inputs.specularIor = ior;
                    inputs.transmissionWeight = transmission;
                });
                for (float ndotV : ndotVs) {
                    const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);
                    for (int mi = 0; mi < 2 * kMuNodes; ++mi) {
                        const float z = -1.0F + ((static_cast<float>(mi) + 0.5F) / kMuNodes);
                        const float r = std::sqrt(std::max(0.0F, 1.0F - (z * z)));
                        for (int pi = 0; pi < kPhiNodes; ++pi) {
                            const float phi = 2.0F * kPi * (static_cast<float>(pi) + 0.5F) / kPhiNodes;
                            const glm::vec3 wi(r * std::cos(phi), r * std::sin(phi), z);
                            if (!(maxChannel(pathtracer::scene::evaluateBsdf(params, wo, wi)) > 0.0F)) {
                                continue;
                            }
                            ++valued;
                            farValued += (wi.z * wo.z < 0.0F) ? 1 : 0;
                            if (pathtracer::scene::pdfBsdf(params, wo, wi) > 0.0F) {
                                continue;
                            }
                            if (uncovered++ < 8) {
                                std::cerr << "bsdf_validate: FAILED strategy coverage at roughness=" << roughness
                                          << " ior=" << ior << " transmission=" << transmission
                                          << " ndotV=" << ndotV << " wi.z=" << wi.z << " phi=" << phi
                                          << " -- the BSDF has value here and no strategy samples it\n";
                            }
                            ok = false;
                        }
                    }
                }
            }
        }
    }
    // Anti-vacuity: the far hemisphere is where the gated lobe lives, so a sweep that never found value there tested nothing.
    if (farValued == 0) {
        std::cerr << "bsdf_validate: FAILED strategy coverage found no far-side direction with value\n";
        ok = false;
    }
    std::cout << "  strategy coverage: " << valued << " valued directions (" << farValued << " far-side), "
              << uncovered << " with zero mixture density\n";
    finish(ctx, ok, "strategy_coverage failed; see the rows above");
    return;
}

// Furnace through sampleBsdf over both hemispheres: bound 1.0, except on the exiting side below the critical angle where eta^2 is correct.
PT_CHECK(furnace_energy_bound, Slow, Statistical) {
    constexpr int kSampleCount = 50000;
    constexpr float kTolerance = 0.1F;
    constexpr float kIor = 1.5F;  // matches makeParams/makeColoredMetalParams
    const std::array<float, 4> roughnesses = {0.05F, 0.25F, 0.5F, 1.0F};
    const std::array<float, 2> metallics = {0.0F, 1.0F};
    const std::array<float, 3> transmissions = {0.0F, 0.5F, 1.0F};
    const std::array<float, 5> ndotVs = {0.2F, 0.6F, 1.0F, -0.9F, -0.3F};  // last two: exiting/TIR

    bool ok = true;
    std::uint32_t seed = 0;
    for (float roughness : roughnesses) {
        for (float metallic : metallics) {
            for (float transmission : transmissions) {
                for (float ndotV : ndotVs) {
                    ++seed;
                    const Surface params = makeParams(roughness, metallic, transmission);
                    const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F,
                                        ndotV);
                    const float maxLo = maxChannel(furnaceLo(params, wo, kSampleCount, seed));
                    const bool exitingTransmissive = ndotV < 0.0F && transmission > 0.0F;
                    const float energyBound = exitingTransmissive ? kIor * kIor : 1.0F;
                    if (!(maxLo <= energyBound + kTolerance)) {
                        std::cerr << "bsdf_validate: FAILED furnace test at roughness=" << roughness
                                  << " metallic=" << metallic << " transmission=" << transmission
                                  << " ndotV=" << ndotV << " Lo=" << maxLo
                                  << " (expected <= " << energyBound << ")\n";
                        ok = false;
                    }
                }
            }
        }
    }

    // Coloured metal (base_color 0.5): below a white F0's perfect mirror, where the furnace has energy to lose or gain.
    for (float roughness : roughnesses) {
        for (float ndotV : ndotVs) {
            ++seed;
            const Surface params = makeColoredMetalParams(roughness);
            const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);
            const float maxLo = maxChannel(furnaceLo(params, wo, kSampleCount, seed));
            if (!(maxLo <= 1.0F + kTolerance)) {
                std::cerr << "bsdf_validate: FAILED colored-metal furnace test at roughness="
                          << roughness << " ndotV=" << ndotV << " Lo=" << maxLo
                          << " (expected <= 1.0)\n";
                ok = false;
            }
        }
    }
    finish(ctx, ok, "furnace_energy_bound failed; see the rows above");
    return;
}

// Two-sided white furnace must return exactly 1.0: single-scatter GGX loses what Smith G2 masks (0.307 at roughness 1, Heitz 2016).
struct WhiteFurnaceCase {
    float roughness;
    float ndotV;
    bool offGrid;
};

PT_CHECK(white_furnace_two_sided, Slow, Statistical) {
    constexpr int kSampleCount = 400000;
    constexpr float kTolerance = 0.02F;
    const std::array<WhiteFurnaceCase, 16> cases = {{
        {0.0F, 1.0F, false},
        {0.0F, 0.4F, false},
        {0.05F, 1.0F, false},
        {0.05F, 0.4F, false},
        {0.25F, 1.0F, false},
        {0.25F, 0.4F, false},
        {0.50F, 1.0F, false},
        {0.50F, 0.4F, false},
        {1.00F, 1.0F, false},
        {1.00F, 0.4F, false},
        // Off-grid values, each the exact float at index k+0.5; deriving them from albedoGridRes() would follow the grid and never fail.
        {0.3666667F, 0.5464398F, true},
        {0.3666667F, 0.2949981F, true},
        {0.6333333F, 0.5464398F, true},
        {0.6333333F, 0.2949981F, true},
        {0.8215686F, 0.5464398F, true},
        {0.8215686F, 0.2949981F, true},
    }};

    bool ok = true;
    std::uint32_t seed = 9000;
    std::cout << "bsdf_validate: white furnace energy (1.0 = perfectly energy-conserving)\n";
    std::cout << "  roughness  ndotV  metal      dielectric\n";
    for (const WhiteFurnaceCase& entry : cases) {
        ++seed;
        const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (entry.ndotV * entry.ndotV))), 0.0F,
                            entry.ndotV);
        const glm::vec3 conductor =
            furnaceLo(makeParams(entry.roughness, 1.0F, 0.0F), wo, kSampleCount, seed);
        const glm::vec3 dielectric =
            furnaceLo(makeParams(entry.roughness, 0.0F, 0.0F), wo, kSampleCount, seed + 500U);
        std::cout << "  " << entry.roughness << "       " << entry.ndotV << "    "
                  << minChannel(conductor) << "   " << minChannel(dielectric)
                  << (entry.offGrid ? "   (off-grid)" : "") << '\n';

        const std::array<std::pair<const char*, glm::vec3>, 2> measured = {
            {{"metal", conductor}, {"dielectric", dielectric}}};
        for (const auto& [label, value] : measured) {
            if (!withinBand(value, 1.0F, kTolerance)) {
                std::cerr << "bsdf_validate: FAILED white-" << label
                          << " furnace energy conservation at roughness=" << entry.roughness
                          << " ndotV=" << entry.ndotV << " Lo=[" << minChannel(value) << ", "
                          << maxChannel(value) << "] (expected 1.0 +/- " << kTolerance << ")\n";
                ok = false;
            }
        }
    }
    finish(ctx, ok, "white_furnace_two_sided failed; see the rows above");
    return;
}

    // EON rough-diffuse energy preservation over diffuseRoughness: its multi-scatter term holds 1.0 where Oren-Nayar variants lose energy.
PT_CHECK(eon_diffuse_furnace, Slow, Statistical) {
    constexpr int kSampleCount = 400000;
    constexpr float kTolerance = 0.02F;
    constexpr float kRoughness = 0.5F;
    const std::array<float, 5> diffuseRoughnesses = {0.0F, 0.25F, 0.5F, 0.75F, 1.0F};
    const std::array<float, 3> ndotVs = {1.0F, 0.6F, 0.2F};
    const std::array<float, 2> metallics = {0.0F, 1.0F};

    bool ok = true;
    std::cout << "bsdf_validate: EON diffuse-roughness furnace energy (1.0 = perfectly energy-conserving)\n";
    std::cout << "  metallic  diffuseRoughness  ndotV  Lo\n";
    for (std::size_t m = 0; m < metallics.size(); ++m) {
        const float metallic = metallics[m];
        // Per-ndotV reading at diffuseRoughness 0, the reference the metal rows must reproduce bit for bit.
        std::array<glm::vec3, 3> baseline{};
        for (float diffuseRoughness : diffuseRoughnesses) {
            for (std::size_t v = 0; v < ndotVs.size(); ++v) {
                const float ndotV = ndotVs[v];
                // Seeded by (metallic, ndotV) only, not diffuseRoughness: the exact assertion needs both readings to draw one sequence.
                const std::uint32_t seed = 20000 + static_cast<std::uint32_t>((m * ndotVs.size()) + v);
                const Surface params = makeParams(kRoughness, metallic, 0.0F, diffuseRoughness);
                const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);
                const glm::vec3 lo = furnaceLo(params, wo, kSampleCount, seed);
                std::cout << "  " << metallic << "         " << diffuseRoughness << "              "
                          << ndotV << "    " << minChannel(lo) << '\n';
                if (!withinBand(lo, 1.0F, kTolerance)) {
                    std::cerr << "bsdf_validate: FAILED EON diffuse furnace energy at metallic="
                              << metallic << " diffuseRoughness=" << diffuseRoughness
                              << " ndotV=" << ndotV << " Lo=[" << minChannel(lo) << ", "
                              << maxChannel(lo) << "] (expected 1.0 +/- " << kTolerance << ")\n";
                    ok = false;
                }
                if (diffuseRoughness == 0.0F) {
                    baseline[v] = lo;
                } else if (metallic == 1.0F && lo != baseline[v]) {
                    std::cerr << "bsdf_validate: FAILED metal diffuseRoughness invariance at ndotV="
                              << ndotV << " diffuseRoughness=" << diffuseRoughness << " Lo="
                              << minChannel(lo) << " vs " << minChannel(baseline[v])
                              << " (a metal has no diffuse lobe; diffuseRoughness must reach nothing)\n";
                    ok = false;
                }
            }
        }
    }
    finish(ctx, ok, "eon_diffuse_furnace failed; see the rows above");
    return;
}

// Listing 1's E_EON at normal incidence, rho*E_F + rho_ms*(1-E_F), transcribed independently of evaluateEon so neither shares a mistake.
glm::vec3 referenceEonAlbedo(const glm::vec3& rho, float r) {
    const float c1 = 0.5F - (2.0F / (3.0F * kPi));
    const float c2 = (2.0F / 3.0F) - (28.0F / (15.0F * kPi));
    const float eFon = 1.0F / (1.0F + (c1 * r));
    const float avgEFon = eFon * (1.0F + (c2 * r));
    const glm::vec3 rhoMs = (rho * rho) * avgEFon / (glm::vec3(1.0F) - (rho * (1.0F - avgEFon)));
    return (rho * eFon) + (rhoMs * (1.0F - eFon));
}

// A bare EON diffuse surface; ior=1 zeroes dielectric Fresnel, so the glossy layer is absent and roughness is swept to show it inert.
Surface makeDiffuseParams(const glm::vec3& baseColor, float diffuseRoughness, float roughness = 0.5F, float ior = 1.0F) {
    return paramsWith([&](OpenPbrInputs<Constant>& inputs) {
        inputs.baseColor = baseColor;
        inputs.baseDiffuseRoughness = diffuseRoughness;
        inputs.specularRoughness = roughness;
        inputs.specularIor = ior;
    });
}

struct AlbedoEstimate {
    glm::vec3 mean;
    glm::vec3 stdError;
};

// Cosine-weighted integral of the shipped diffuse lobe at normal incidence; the second moment makes the band the estimator's own error.
AlbedoEstimate measureDiffuseAlbedo(const Surface& params, int sampleCount, std::mt19937& rng) {
    const glm::vec3 wo(0.0F, 0.0F, 1.0F);
    glm::vec3 sum(0.0F);
    glm::vec3 sumSq(0.0F);
    for (int i = 0; i < sampleCount; ++i) {
        const glm::vec3 wi = sampleUniformHemisphere(rng);
        const glm::vec3 sample =
            pathtracer::scene::evaluateBsdfSplit(params, wo, wi).diffuse * 2.0F * kPi;
        sum += sample;
        sumSq += sample * sample;
    }
    const auto n = static_cast<float>(sampleCount);
    const glm::vec3 mean = sum / n;
    const glm::vec3 variance = glm::max((sumSq / n) - (mean * mean), glm::vec3(0.0F));
    return {mean, glm::sqrt(variance / n)};
}

// The observed albedo must equal the authored one, EON Appendix A's purpose: closed form against closed form, then the shipped lobe.
PT_CHECK(eon_albedo_inversion, Slow, Statistical) {
    constexpr int kSampleCount = 400000;
    // Two named bounded residuals: kSigmaBand is a confidence level on the measured standard error, kFitTolerance the model residual.
    constexpr float kSigmaBand = 5.0F;
    constexpr float kFitTolerance = 0.001F;
    constexpr float kAnalyticTolerance = 1e-5F;
    const std::array<float, 5> diffuseRoughnesses = {0.0F, 0.25F, 0.5F, 0.75F, 1.0F};
    const std::array<glm::vec3, 3> albedos = {glm::vec3(1.0F), glm::vec3(0.5F),
                                               glm::vec3(0.8F, 0.3F, 0.1F)};

    bool ok = true;
    std::mt19937 rng(31337);
    std::cout << "bsdf_validate: EON albedo inversion, observed vs authored albedo at normal incidence\n";
    std::cout << "  diffuseRoughness  authored              rho                   observed              worst err\n";
    for (const glm::vec3& authored : albedos) {
        for (float diffuseRoughness : diffuseRoughnesses) {
            const Surface params = makeDiffuseParams(authored, diffuseRoughness);
            const glm::vec3 rho = pathtracer::scene::eonAlbedoInversion(authored, diffuseRoughness);
            const glm::vec3 analytic = referenceEonAlbedo(rho, diffuseRoughness);
            const AlbedoEstimate measured = measureDiffuseAlbedo(params, kSampleCount, rng);

            const glm::vec3 analyticErr = glm::abs(analytic - authored);
            const glm::vec3 band =
                (kSigmaBand * measured.stdError) + (kFitTolerance * glm::max(authored, 0.01F));
            const glm::vec3 measuredErr = glm::abs(measured.mean - authored);

            std::cout << "  " << diffuseRoughness << "               [" << authored.x << ", "
                       << authored.y << ", " << authored.z << "]   [" << rho.x << ", " << rho.y << ", " << rho.z << "]   ["
                       << measured.mean.x << ", " << measured.mean.y << ", " << measured.mean.z
                       << "]   " << maxChannel(measuredErr) << '\n';

            if (maxChannel(analyticErr) > kAnalyticTolerance) {
                std::cerr << "bsdf_validate: FAILED EON albedo inversion (analytic) at diffuseRoughness="
                           << diffuseRoughness << " authored=[" << authored.x << ", " << authored.y
                           << ", " << authored.z << "] E_EON=[" << analytic.x << ", " << analytic.y
                           << ", " << analytic.z << "] err=" << maxChannel(analyticErr)
                           << " (expected <= " << kAnalyticTolerance << ")\n";
                ok = false;
            }
            if (!glm::all(glm::lessThanEqual(measuredErr, band))) {
                std::cerr << "bsdf_validate: FAILED EON albedo inversion (measured) at diffuseRoughness="
                           << diffuseRoughness << " authored=[" << authored.x << ", " << authored.y
                           << ", " << authored.z << "] observed=[" << measured.mean.x << ", "
                           << measured.mean.y << ", " << measured.mean.z
                           << "] err=" << maxChannel(measuredErr) << " (expected <= "
                           << maxChannel(band) << ")\n";
                ok = false;
            }
        }
    }
    finish(ctx, ok, "eon_albedo_inversion failed; see the rows above");
    return;
}

// EON BRDF value (eq. 16-19) in double, transcribed independently with c1/c2 re-derived; the 1e-7 floors are the model's guard at r=0.
glm::vec3 referenceEon(const glm::vec3& rho, float r, const glm::vec3& wi, const glm::vec3& wo) {
    const double c1 = 0.5 - (2.0 / (3.0 * kPi));
    const double c2 = (2.0 / 3.0) - (28.0 / (15.0 * kPi));
    const double muI = wi.z;
    const double muO = wo.z;
    const double s = static_cast<double>(glm::dot(wi, wo)) - (muI * muO);
    const double sOverT = s > 0.0 ? s / std::max(muI, muO) : s;
    const double af = 1.0 / (1.0 + (c1 * r));
    const double avgEFon = af * (1.0 + (c2 * r));
    // Paper eq. 14's quartic in (1 - mu), evaluated as an explicit polynomial rather than eon.cpp's Horner nesting.
    const auto eFon = [&](double mu) {
        const double m = 1.0 - mu;
        const double gOverPi = (0.0571085289 * m) + (0.491881867 * m * m) +
                                (-0.332181442 * m * m * m) + (0.0714429953 * m * m * m * m);
        return (1.0 + (r * gOverPi)) * af;
    };
    constexpr double kEps = 1e-7;
    const double shadow = (std::max(kEps, 1.0 - eFon(muO)) * std::max(kEps, 1.0 - eFon(muI))) /
                           std::max(kEps, 1.0 - avgEFon);
    glm::vec3 result(0.0F);
    for (int c = 0; c < 3; ++c) {
        const double rhoC = rho[c];
        const double rhoMs = (rhoC * rhoC * avgEFon) / (1.0 - (rhoC * (1.0 - avgEFon)));
        result[c] = static_cast<float>(((rhoC * af * (1.0 + (r * sOverT))) + (rhoMs * shadow)) / kPi);
    }
    return result;
}

// At ior 1 the glossy layer is optically absent: it reflects nothing, so the albedo scaling is exactly 1 and the diffuse is EON alone.
PT_CHECK(index_matched_glossy_layer, Fast, Exact) {
    // Exact from the collapse above; the second is a float32-vs-double residual, worst 2.03e-7.
    constexpr float kInvarianceTolerance = 0.0F;
    constexpr float kValueTolerance = 1e-6F;
    // 0.0 is the reference row every other is compared against; 0.3661 and 0.92 sit deliberately off the table's grid.
    const std::array<float, 8> roughnesses = {0.0F, 0.05F, 0.25F, 0.3661F, 0.5F, 0.75F, 0.92F, 1.0F};
    // The tail below 2.44e-4 (2^-12, where 1.0F-mu*mu rounds to 1.0F) is where the old Snell transcription falsely reported TIR.
    const std::array<float, 11> cosines = {1.0F,   0.8F,         0.6F,    0.4F,    0.2F, 0.05F,
                                           1e-2F, 1e-3F, 2.44e-4F, 1.7263349e-4F, 1e-5F};
    const std::array<float, 3> diffuseRoughnesses = {0.0F, 0.5F, 1.0F};
    // The chromatic row carries the value assertion: at baseColor 1 the inversion is the identity and proves nothing.
    const std::array<glm::vec3, 2> albedos = {glm::vec3(1.0F), glm::vec3(0.8F, 0.3F, 0.1F)};

    bool ok = true;
    int rowsChecked = 0;
    float worstOverall = 0.0F;
    float worstValueErr = 0.0F;
    std::cout << "bsdf_validate: index-matched glossy layer, diffuse channel vs specular roughness (ior 1)\n";
    std::cout << "  baseColor              diffuseRoughness  worst |delta|  at roughness/mu_o/mu_i\n";
    for (const glm::vec3& albedo : albedos) {
        for (float diffuseRoughness : diffuseRoughnesses) {
            float worst = 0.0F;
            float worstRoughness = 0.0F;
            float worstMuO = 0.0F;
            float worstMuI = 0.0F;
            for (float muO : cosines) {
                for (float muI : cosines) {
                    // Non-coplanar pair, checkReciprocity's construction: a shared azimuth would leave a swapped-phi bug invisible.
                    const float sinO = std::sqrt(std::max(0.0F, 1.0F - (muO * muO)));
                    const float sinI = std::sqrt(std::max(0.0F, 1.0F - (muI * muI)));
                    const glm::vec3 wo(sinO, 0.0F, muO);
                    const glm::vec3 wi(sinI * std::cos(1.1F), sinI * std::sin(1.1F), muI);
                    const glm::vec3 reference =
                        pathtracer::scene::evaluateBsdfSplit(
                            makeDiffuseParams(albedo, diffuseRoughness, roughnesses[0]), wo, wi)
                            .diffuse;
                    const glm::vec3 analytic = referenceEon(
                        pathtracer::scene::eonAlbedoInversion(albedo, diffuseRoughness),
                        diffuseRoughness, wi, wo) * muI;
                    const float scale = std::max(maxChannel(analytic), 1e-6F);
                    const float valueErr = maxChannel(glm::abs(reference - analytic)) / scale;
                    worstValueErr = std::max(worstValueErr, valueErr);
                    if (!(valueErr <= kValueTolerance)) {
                        std::cerr << "bsdf_validate: FAILED index-matched glossy layer (value) at baseColor=["
                                   << albedo.x << ", " << albedo.y << ", " << albedo.z
                                   << "] diffuseRoughness=" << diffuseRoughness << " mu_o=" << muO
                                   << " mu_i=" << muI << " diffuse=[" << reference.x << ", "
                                   << reference.y << ", " << reference.z << "] reference EON=["
                                   << analytic.x << ", " << analytic.y << ", " << analytic.z
                                   << "] relative err=" << valueErr << " (expected <= "
                                   << kValueTolerance << ")\n";
                        ok = false;
                    }
                    for (std::size_t i = 1; i < roughnesses.size(); ++i) {
                        const glm::vec3 value =
                            pathtracer::scene::evaluateBsdfSplit(
                                makeDiffuseParams(albedo, diffuseRoughness, roughnesses[i]), wo, wi)
                                .diffuse;
                        const float delta = maxChannel(glm::abs(value - reference)) / scale;
                        ++rowsChecked;
                        if (delta > worst) {
                            worst = delta;
                            worstRoughness = roughnesses[i];
                            worstMuO = muO;
                            worstMuI = muI;
                        }
                        if (!(delta <= kInvarianceTolerance)) {
                            std::cerr << "bsdf_validate: FAILED index-matched glossy layer at baseColor=["
                                       << albedo.x << ", " << albedo.y << ", " << albedo.z
                                       << "] diffuseRoughness=" << diffuseRoughness
                                       << " roughness=" << roughnesses[i] << " mu_o=" << muO
                                       << " mu_i=" << muI << " diffuse=[" << value.x << ", "
                                       << value.y << ", " << value.z << "] vs roughness 0 ["
                                       << reference.x << ", " << reference.y << ", " << reference.z
                                       << "] relative delta=" << delta
                                       << " (expected exactly 0: at ior 1 the glossy layer is absent, so its roughness cannot reach the diffuse channel)\n";
                            ok = false;
                        }
                    }
                }
            }
            worstOverall = std::max(worstOverall, worst);
            std::cout << "  [" << albedo.x << ", " << albedo.y << ", " << albedo.z << "]         "
                       << diffuseRoughness << "               " << worst << "            "
                       << worstRoughness << " / " << worstMuO << " / " << worstMuI << '\n';
        }
    }
    std::cout << "  " << rowsChecked << " roughness comparisons, worst |delta| " << worstOverall
               << ", worst value error " << worstValueErr << '\n';
    // Anti-vacuity: a sweep that asserted nothing would print zeros too.
    if (rowsChecked == 0) {
        std::cerr << "bsdf_validate: FAILED index-matched glossy layer -- no rows asserted\n";
        ok = false;
    }
    finish(ctx, ok, "index_matched_glossy_layer failed; see the rows above");
    return;
}

// Mean throughput with transmitted draws divided by eta^2, putting every sample in one domain; in double, a float ulp being 0.008 at 200k.
glm::vec3 transmissiveEnergyLo(const Surface& params, const glm::vec3& wo, int sampleCount,
                                std::uint32_t seed) {
    const float eta = wo.z < 0.0F ? params.specularIor : 1.0F / params.specularIor;  // etaI/etaT, exiting vs entering
    const float etaSq = eta * eta;
    glm::dvec3 accum(0.0);
    for (int i = 0; i < sampleCount; ++i) {
        pathtracer::scene::Sampler sampler(0, 0, i, sampleCount, seed);
        const std::optional<pathtracer::scene::BsdfSample> sample =
            pathtracer::scene::sampleBsdf(params, wo, sampler);
        if (!sample.has_value()) {
            continue;
        }
        accum += glm::dvec3(sample->type == pathtracer::scene::LobeType::Transmission
                                 ? sample->throughputWeight / etaSq
                                 : sample->throughputWeight);
    }
    return glm::vec3(accum / static_cast<double>(sampleCount));
}

// Two-sided energy balance, 1.0 in the energy domain: catches compensation over the wrong hemisphere and a double-counted escape budget.
PT_CHECK(transmissive_energy_balance, Slow, Statistical) {
    constexpr int kSampleCount = 200000;
    // Same tolerance as the opaque white furnace; the residual is model error, measured worst 0.0036 at transmission_weight 0.5 entering.
    constexpr float kTolerance = 0.02F;
    const std::array<float, 4> roughnesses = {0.05F, 0.4F, 0.7F, 1.0F};
    const std::array<float, 4> ndotVs = {1.0F, 0.6F, -0.9F, -0.4F};  // entering, entering, exiting, TIR
    const std::array<float, 2> transmissions = {0.5F, 1.0F};
    const std::array<float, 2> metallics = {0.0F, 1.0F};

    bool ok = true;
    std::uint32_t seed = 12000;
    std::cout << "bsdf_validate: transmissive energy balance (1.0 = perfectly energy-conserving)\n";
    std::cout << "  roughness  ndotV  transmission  metallic  Lo\n";
    for (float roughness : roughnesses) {
        for (float transmission : transmissions) {
            for (float metallic : metallics) {
                for (float ndotV : ndotVs) {
                    ++seed;
                    const Surface params = makeParams(roughness, metallic, transmission);
                    const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F,
                                        ndotV);
                    const glm::vec3 lo = transmissiveEnergyLo(params, wo, kSampleCount, seed);
                    std::cout << "  " << roughness << "        " << ndotV << "     " << transmission
                              << "           " << metallic << "       " << minChannel(lo) << '\n';
                    if (!withinBand(lo, 1.0F, kTolerance)) {
                        std::cerr << "bsdf_validate: FAILED transmissive energy balance at roughness="
                                  << roughness << " ndotV=" << ndotV
                                  << " transmission=" << transmission << " metallic=" << metallic
                                  << " Lo=[" << minChannel(lo) << ", " << maxChannel(lo)
                                  << "] (expected 1.0 +/- " << kTolerance << ")\n";
                        ok = false;
                    }
                }
            }
        }
    }
    finish(ctx, ok, "transmissive_energy_balance failed; see the rows above");
    return;
}

// Round-trip closure of a rough dielectric slab, 1.0 by BSDF sampling alone; the floor is the table's 1e-3 accuracy over a worst 6.2e-4.
PT_CHECK(transmissive_slab_walk, Slow, Statistical) {
    // Sized so the band resolves the smallest shortfall the old table left, 0.6% at roughness 0.7; the band stays under it.
    constexpr int kPathsPerReplicate = 1 << 18;
    constexpr double kEscapeInterpolationFloor = 1e-3;
    const std::array<float, 3> roughnesses = {0.4F, 0.7F, 1.0F};

    ctx.plan(static_cast<int>(2 * roughnesses.size()));
    std::cout << "bsdf_validate: white rough-glass slab by BSDF sampling alone (1.0 = energy closes)\n";
    for (float roughness : roughnesses) {
        char label[64];
        std::snprintf(label, sizeof(label), "slab walk r=%g", static_cast<double>(roughness));
        const std::uint64_t rowSeed = ctx.subSeed(label);
        const Surface params = makeParams(roughness, /*metallic=*/0.0F, /*transmission=*/1.0F);
        std::array<tools::fixtures::SlabWalk, tools::stats::kReplicates> replicates{};
        std::atomic<int> next{0};
        std::vector<std::thread> workers;
        for (int w = 0; w < std::min(ctx.threads(), tools::stats::kReplicates); ++w) {
            workers.emplace_back([&] {
                for (int r = next++; r < tools::stats::kReplicates; r = next++) {
                    replicates[static_cast<std::size_t>(r)] =
                        tools::fixtures::slabWalkLo(params, kPathsPerReplicate, static_cast<std::uint32_t>(rowSeed + r));
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
        tools::stats::Welford lo;
        tools::stats::Welford vertices;
        long long truncated = 0;
        for (const tools::fixtures::SlabWalk& replicate : replicates) {
            lo.add(replicate.mean);
            vertices.add(replicate.verticesPerPath);
            truncated += replicate.truncated;
        }
        const double half = (tools::stats::studentTTwoSided(ctx.alpha(), tools::stats::kReplicates - 1) * lo.standardError()) +
                            (kEscapeInterpolationFloor * vertices.mean());
        char detail[256];
        std::snprintf(detail, sizeof(detail), "%s: Lo %.5f vs 1 +/- %.5f over %.2f vertices per path", label, lo.mean(), half, vertices.mean());
        std::cout << "  " << detail << '\n';
        PT_EXPECT(ctx, std::abs(lo.mean() - 1.0) <= half, detail);
        std::snprintf(detail, sizeof(detail), "%s: %lld paths reached the depth cap, which would truncate the estimate", label, truncated);
        PT_EXPECT(ctx, truncated == 0, detail);
    }
}

// A non-absorbing transmissive dielectric with explicit base_color and transmission_color, at depth 0 so the colour tints the surface.
Surface makeTransmissiveTintParams(float roughness, const glm::vec3& baseColor, const glm::vec3& transmissionColor) {
    return paramsWith([&](OpenPbrInputs<Constant>& inputs) {
        inputs.baseColor = baseColor;
        inputs.specularRoughness = roughness;
        inputs.transmissionWeight = 1.0F;
        inputs.transmissionColor = transmissionColor;
    });
}

// Snell refraction of wo about +z, transcribed independently of microfacet.cpp, so a sign or eta error in either shows as a disagreement.
glm::vec3 refractAboutZ(const glm::vec3& wo, float eta) {
    const float sin2ThetaT = eta * eta * std::max(0.0F, 1.0F - (wo.z * wo.z));
    const float cosThetaT = std::sqrt(std::max(0.0F, 1.0F - sin2ThetaT));
    return {-eta * wo.x, -eta * wo.y, -cosThetaT};
}

// Throughput of sampleBsdf's smooth delta transmission branch, identified by pdf == 0, a rough sample returning a real density.
std::optional<glm::vec3> deltaTransmitThroughput(const Surface& params, const glm::vec3& wo,
                                                  std::uint32_t seed) {
    constexpr int kAttempts = 64;
    for (int i = 0; i < kAttempts; ++i) {
        pathtracer::scene::Sampler sampler(0, 0, i, kAttempts, seed);
        const std::optional<pathtracer::scene::BsdfSample> sample =
            pathtracer::scene::sampleBsdf(params, wo, sampler);
        if (sample.has_value() && sample->type == pathtracer::scene::LobeType::Transmission &&
            sample->pdf == 0.0F) {
            return sample->throughputWeight;
        }
    }
    return std::nullopt;
}

// The transmission-tint convention, the only instrument that sees it: every other transmissive case runs at baseColor 1, where they agree.
PT_CHECK(transmission_tint, Fast, Exact) {
    constexpr float kUlpBand = 1e-6F;
    constexpr float kSmoothRoughness = 0.0F;   // the smooth interface, so transmission is the delta branch
    const glm::vec3 baseColour(0.2F, 0.5F, 0.9F);
    const glm::vec3 tint(0.3F, 0.6F, 0.9F);
    const glm::vec3 white(1.0F);
    const std::array<float, 3> roughnesses = {0.3F, 0.6F, 1.0F};
    const std::array<float, 3> ndotVs = {0.95F, 0.7F, 0.35F};

    bool ok = true;
    int measured = 0;
    std::uint32_t seed = 31000;

    // One row's pair of assertions for both paths: T must not move with baseColor at all, and must scale exactly with the tint.
    const auto assertRow = [&](const char* lobe, float roughness, float ndotV,
                                const glm::vec3& tWhite, const glm::vec3& tBaseColoured,
                                const glm::vec3& tTinted, const glm::vec3& tBoth) {
        if (maxChannel(tWhite) <= 0.0F) {
            return;   // no transmission at this configuration; the backstop below catches an all-skipped run
        }
        ++measured;
        const glm::vec3 expected = tint * tWhite;
        std::cout << "  " << lobe << "  roughness " << roughness << "  ndotV " << ndotV
                  << "   T(white) [" << tWhite.x << ", " << tWhite.y << ", " << tWhite.z
                  << "]   tinted/expected [" << tTinted.x / expected.x << ", "
                  << tTinted.y / expected.y << ", " << tTinted.z / expected.z << "]\n";
        for (int c = 0; c < 3; ++c) {
            if (tBaseColoured[c] != tWhite[c] || tBoth[c] != tTinted[c]) {
                std::cerr << "bsdf_validate: FAILED transmission tint at " << lobe << " roughness="
                          << roughness << " ndotV=" << ndotV << " channel " << c
                          << " -- baseColor moved the transmitted value from " << tWhite[c]
                          << " to " << tBaseColoured[c]
                          << ". baseColor is a reflection quantity (OpenPBR base_color) and must not "
                             "reach the transmission lobe; transmission_color is what tints it.\n";
                ok = false;
            }
            if (std::fabs(tTinted[c] - expected[c]) > kUlpBand * std::fabs(expected[c])) {
                std::cerr << "bsdf_validate: FAILED transmission tint linearity at " << lobe
                          << " roughness=" << roughness << " ndotV=" << ndotV << " channel " << c
                          << " -- measured " << tTinted[c] << ", expected " << expected[c]
                          << ". transmission_color multiplies the transmitted value once, so the lobe "
                             "must be exactly linear in it.\n";
                ok = false;
            }
        }
    };

    std::cout << "bsdf_validate: transmission tint convention (transmissionColor tints transmission, baseColor does not)\n";
    for (float ndotV : ndotVs) {
        ++seed;
        const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);
        const glm::vec3 wi = refractAboutZ(wo, 1.0F / 1.5F);
        for (float roughness : roughnesses) {
            const auto rough = [&](const glm::vec3& bc, const glm::vec3& tn) {
                return pathtracer::scene::evaluateBsdfSplit(makeTransmissiveTintParams(roughness, bc, tn),
                                                         wo, wi)
                    .transmission;
            };
            assertRow("rough ", roughness, ndotV, rough(white, white), rough(baseColour, white),
                       rough(white, tint), rough(baseColour, tint));
        }
        // The delta branch is the smooth interface, not a point of the roughness sweep, so it is measured once.
        const auto smooth = [&](const glm::vec3& bc, const glm::vec3& tn) {
            return deltaTransmitThroughput(makeTransmissiveTintParams(kSmoothRoughness, bc, tn), wo,
                                            seed)
                .value_or(glm::vec3(0.0F));
        };
        assertRow("smooth", kSmoothRoughness, ndotV, smooth(white, white), smooth(baseColour, white),
                   smooth(white, tint), smooth(baseColour, tint));
    }

    // Backstop: every assertion above is skipped where the interface transmits nothing, so a zeroed lobe would pass them all vacuously.
    if (measured == 0) {
        std::cerr << "bsdf_validate: FAILED transmission tint -- no row transmitted anything, so "
                     "nothing was asserted\n";
        ok = false;
    }
    finish(ctx, ok, "transmission_tint failed; see the rows above");
    return;
}

// Unpolarized dielectric Fresnel, entering orientation, in double: the reference the interface's Fresnel is measured against.
double referenceDielectricFresnel(double cosTheta, double ior) {
    const double c = std::clamp(cosTheta, 0.0, 1.0);
    const double sinT2 = (1.0 - (c * c)) / (ior * ior);
    if (sinT2 >= 1.0) {
        return 1.0;
    }
    const double cosT = std::sqrt(1.0 - sinT2);
    const double rs = (c - (ior * cosT)) / (c + (ior * cosT));
    const double rp = ((ior * c) - cosT) / ((ior * c) + cosT);
    return 0.5 * ((rs * rs) + (rp * rp));
}

// F_avg attenuates every repeated Kulla-Conty bounce and no furnace resolves an error in it; the metal's is closed-form, so float-tight.
PT_CHECK(average_fresnel, Fast, Exact) {
    // F0 + (1-F0)/21 - k/126 against the double quadrature: three float operations on terms below 2, a few ulps of 1.
    constexpr double kMetalTolerance = 2e-6;
    const std::array<double, 12> reflectivities = {1e-4, 0.01, 0.1,  0.25, 0.4,  0.48,
                                                    0.555, 0.7, 0.85, 0.95, 0.99, 1.0};
    const std::array<double, 8> tints = {0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0};

    bool ok = true;
    std::cout << "bsdf_validate: average Fresnel vs quadrature (F_avg = 2*int F(mu)*mu dmu)\n";
    std::cout << "  metal: worst |metalFresnelAvg - truth| over specular_color, per F0\n";
    for (double reflectivity : reflectivities) {
        double worst = 0.0;
        double worstTint = 0.0;
        for (double tint : tints) {
            const double truth = cosineAverageFresnel([&](double mu) { return referenceF82(reflectivity, tint, mu); });
            const glm::vec3 rule = pathtracer::scene::metalFresnelAvg(glm::vec3(static_cast<float>(reflectivity)),
                                                                      glm::vec3(static_cast<float>(tint)));
            const double error = std::abs(static_cast<double>(rule.x) - truth);
            if (error > worst) {
                worst = error;
                worstTint = tint;
            }
            if (!(error <= kMetalTolerance)) {
                std::cerr << "bsdf_validate: FAILED metal F_avg at F0=" << reflectivity << " specular_color=" << tint << " rule=" << rule.x
                          << " vs quadrature " << truth << " (error " << error << ", tolerance " << kMetalTolerance << ")\n";
                ok = false;
            }
        }
        std::cout << "    F0 " << reflectivity << "   worst " << worst << " at specular_color " << worstTint << '\n';
    }
    // The dielectric's closed form against the same quadrature, entering and (eta < 1) leaving, where Simpson meets the TIR kink.
    constexpr double kEnteringTolerance = 2e-6;
    constexpr double kLeavingTolerance = 1e-5;
    std::cout << "  dielectric: |fresnelAverage - truth| per relative index\n";
    for (double eta : {0.5, 0.8, 0.95, 1.0, 1.05, 1.33, 1.5, 1.6, 2.0, 3.0}) {
        const double truth = cosineAverageFresnel([&](double mu) { return referenceDielectricFresnel(mu, eta); });
        const double error = std::abs(static_cast<double>(pathtracer::scene::fresnelAverage(static_cast<float>(eta))) - truth);
        const double tolerance = eta < 1.0 ? kLeavingTolerance : kEnteringTolerance;
        std::cout << "    eta " << eta << "   error " << error << '\n';
        if (!(error <= tolerance)) {
            std::cerr << "bsdf_validate: FAILED dielectric F_avg at eta=" << eta << " error " << error << " (tolerance " << tolerance << ")\n";
            ok = false;
        }
    }

    finish(ctx, ok, "average_fresnel failed; see the rows above");
    return;
}

// --- Independent reference for the reflect-side albedo table, on the generator's domain and measure; Gauss-Legendre, Simpson being slower.
double referenceSmithG2OverCosO(double cosO, double cosI, double alpha) {
    const double alpha2 = alpha * alpha;
    const auto radical = [&](double c) { return std::sqrt(alpha2 + ((1.0 - alpha2) * c * c)); };
    return 2.0 * cosI / ((cosI * radical(cosO)) + (cosO * radical(cosI)));
}

constexpr double kPiDouble = 3.14159265358979324;

// Gauss-Legendre nodes/weights on [0,1] by Newton on P_n via Bonnet's recurrence (Numerical Recipes 3rd ed.), built once per node count.
struct GaussLegendreRule {
    std::vector<double> node;
    std::vector<double> weight;
};

template <int N>
const GaussLegendreRule& gaussLegendreRule() {
    static const GaussLegendreRule rule = [] {
        GaussLegendreRule built{std::vector<double>(N), std::vector<double>(N)};
        for (int i = 0; i < N; ++i) {
            double x = std::cos(kPiDouble * (i + 0.75) / (N + 0.5));
            double derivative = 0.0;
            for (int iteration = 0; iteration < 100; ++iteration) {
                double p0 = 1.0;
                double p1 = 0.0;
                for (int k = 0; k < N; ++k) {
                    const double p2 = p1;
                    p1 = p0;
                    p0 = ((((2.0 * k) + 1.0) * x * p1) - (k * p2)) / (k + 1.0);
                }
                derivative = N * ((x * p0) - p1) / ((x * x) - 1.0);
                const double step = p0 / derivative;
                x -= step;
                if (std::abs(step) <= 1e-16) {
                    break;
                }
            }
            built.node[static_cast<std::size_t>(i)] = 0.5 * (1.0 - x);
            built.weight[static_cast<std::size_t>(i)] = 1.0 / ((1.0 - (x * x)) * derivative * derivative);
        }
        return built;
    }();
    return rule;
}

// Gauss-Legendre over [lower, upper], on any value type with + and scalar *, matching simpson's shape.
template <int N, typename F>
auto gaussLegendre(double lower, double upper, F f) -> decltype(f(lower)) {
    const GaussLegendreRule& rule = gaussLegendreRule<N>();
    decltype(f(lower)) sum = f(lower) * 0.0;
    for (std::size_t i = 0; i < rule.node.size(); ++i) {
        sum = sum + (rule.weight[i] * f(lower + ((upper - lower) * rule.node[i])));
    }
    return (upper - lower) * sum;
}

// Composite Simpson over [lower, upper] with an even panel count, on any value type with + and scalar *.

// OpenPBR's alpha = r^2, mirrored so every reference evaluates the alpha the lobe ships at.
double alphaAt(double roughness) { return roughness * roughness; }

// F82-split directional albedo (a, b, c): E(F0, k) = F0*a + b - k*c, and a + b = E.
glm::dvec3 referenceDirectionalAlbedo(double mu, double alpha) {
    constexpr int kPanels = 48;     // phi, even for Simpson; doubling it moves no worst by over 1%, far inside the tolerances
    constexpr int kPsiNodes = 192;  // psi, twice the generator's, so the control row compares two rules and not one
    const double sinTv = std::sqrt(std::max(0.0, 1.0 - (mu * mu)));
    const auto azimuth = [&](double phi) {
        const double horizontal = sinTv * std::cos(phi);
        const double radius = std::sqrt((horizontal * horizontal) + (mu * mu));
        const double delta = std::atan2(horizontal, mu);
        const double psiMax = std::atan(std::tan(0.5 * (delta + (0.5 * kPiDouble))) / alpha);
        return gaussLegendre<kPsiNodes>(0.0, psiMax, [&](double psi) {
            const double thetaH = std::atan(alpha * std::tan(psi));
            const double woDotH = radius * std::cos(thetaH - delta);
            const double wiZ = radius * std::cos((2.0 * thetaH) - delta);
            const double weight = (woDotH / std::cos(thetaH)) *
                                   referenceSmithG2OverCosO(mu, wiZ, alpha) * std::sin(psi) * std::cos(psi);
            const double fc = std::pow(std::clamp(1.0 - woDotH, 0.0, 1.0), 5.0);
            return glm::dvec3(weight * (1.0 - fc), weight * fc, weight * woDotH * std::pow(std::clamp(1.0 - woDotH, 0.0, 1.0), 6.0));
        });
    };
    const glm::dvec3 half = simpson(0.0, 0.5 * kPiDouble, kPanels, azimuth) +
                             simpson(0.5 * kPiDouble, kPiDouble, kPanels, azimuth);
    return (2.0 / kPiDouble) * half;
}

// Cosine-weighted mean, 2*int_0^1 E(mu)*mu dmu; the mu=0 endpoint contributes exactly 0, the mu weight killing a bounded E.
glm::dvec3 referenceAverageAlbedo(double alpha) {
    constexpr int kPanels = 64;
    return 2.0 * simpson(0.0, 1.0, kPanels,
                          [&](double mu) { return referenceDirectionalAlbedo(mu, alpha) * mu; });
}


// --- Instrument for albedo_table.inc's interpolation error, which only the 2% furnace otherwise bounds: four measurements, reported apart.
struct InterpolationError {
    double worst;
    double roughness;
    double mu;
};

void recordWorst(InterpolationError& error, double delta, double roughness, double mu) {
    if (delta > error.worst) {
        error = {delta, roughness, mu};
    }
}

// Worst over the three F82 channels of the shipped lookup against the reference, at one (mu, roughness).
double directionalAlbedoError(double mu, double roughness) {
    const glm::dvec4 shipped(pathtracer::scene::directionalAlbedoSplit(static_cast<float>(mu), static_cast<float>(roughness)));
    const glm::dvec3 exact = referenceDirectionalAlbedo(mu, alphaAt(roughness));
    const glm::dvec3 error = glm::abs(glm::dvec3(shipped) - exact);
    // The deficit channel is 1 - E as baked, clipped at 0, so it is measured against the same quantity.
    const double deficitError = std::abs(shipped.w - std::max(1.0 - (exact.x + exact.y), 0.0));
    return std::max({error.x, error.y, error.z, deficitError});
}

    // Evenly spread node indices over [first, last]: the "held on exact nodes" coordinate, where that axis adds no interpolation error.
std::vector<int> spreadNodes(int first, int last, int count) {
    std::vector<int> nodes(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
        nodes[static_cast<std::size_t>(k)] = first + ((k * (last - first)) / (count - 1));
    }
    return nodes;
}

// Row-parallel over the swept axis: rows share no accumulator and combine in index order, so the reported worst matches the serial one.
template <typename Row>
void parallelRows(int rows, int threads, Row row) {
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    for (int w = 0; w < std::min(threads, rows); ++w) {
        workers.emplace_back([&] {
            for (int i = next++; i < rows; i = next++) {
                row(i);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
}

PT_CHECK(albedo_table_interpolation, Slow, Exact) {
    // Each bound is the measured worst plus ~1.7x; the three directional worsts sit at roughness <= 0.022 and mu <= 4.4e-3.
    constexpr double kControlTolerance = 5e-5;
    constexpr double kRoughnessAxisTolerance = 1e-3;
    constexpr double kMuAxisTolerance = 3.7e-3;
    // In E*mu: measured worst 7.1e-8 at roughness 1/255, where the grazing layer (width ~alpha) is narrower than the cell.
    constexpr double kFirstMuCellTolerance = 1.2e-7;
    constexpr double kAverageAlbedoTolerance = 8e-6;
    // Fractions across the first mu cell: the layer sits at the mu = 0 edge, so where the worst falls depends on its width vs the cell's.
    const std::array<double, 4> firstCellFractions = {0.2, 0.4, 0.6, 0.8};
    constexpr int kSpread = 16;

    const glm::ivec2 res = pathtracer::scene::albedoGridRes();
    const std::vector<int> muNodes = spreadNodes(0, res.y - 1, kSpread);
    const std::vector<int> roughnessNodes = spreadNodes(0, res.x - 1, kSpread);

    // Control first: both axes on exact nodes, so no interpolation happens and what is left is this file's Simpson against the generator.
    std::vector<InterpolationError> controlRows(static_cast<std::size_t>(res.x));
    parallelRows(res.x, ctx.threads(), [&](int ri) {
        const double roughness = pathtracer::scene::albedoGridRoughness(static_cast<float>(ri));
        InterpolationError row{0.0, roughness, 0.0};
        for (int mi : muNodes) {
            const double mu = pathtracer::scene::albedoGridMu(static_cast<float>(mi));
            recordWorst(row, directionalAlbedoError(mu, roughness), roughness, mu);
        }
        controlRows[static_cast<std::size_t>(ri)] = row;
    });

    // Roughness axis: every cell midpoint on that axis, held on exact mu nodes.
    std::vector<InterpolationError> roughnessRows(static_cast<std::size_t>(res.x - 1));
    parallelRows(res.x - 1, ctx.threads(), [&](int ri) {
        const double roughness = pathtracer::scene::albedoGridRoughness(static_cast<float>(ri) + 0.5F);
        InterpolationError row{0.0, roughness, 0.0};
        for (int mi : muNodes) {
            const double mu = pathtracer::scene::albedoGridMu(static_cast<float>(mi));
            recordWorst(row, directionalAlbedoError(mu, roughness), roughness, mu);
        }
        roughnessRows[static_cast<std::size_t>(ri)] = row;
    });

    // mu axis: every cell midpoint but the first, on exact roughness nodes; folding it in would let one boundary layer set the whole bound.
    std::vector<InterpolationError> muRows(roughnessNodes.size());
    parallelRows(static_cast<int>(roughnessNodes.size()), ctx.threads(), [&](int k) {
        const int ri = roughnessNodes[static_cast<std::size_t>(k)];
        const double roughness = pathtracer::scene::albedoGridRoughness(static_cast<float>(ri));
        InterpolationError row{0.0, roughness, 0.0};
        for (int mi = 1; mi + 1 < res.y; ++mi) {
            const double mu = pathtracer::scene::albedoGridMu(static_cast<float>(mi) + 0.5F);
            recordWorst(row, directionalAlbedoError(mu, roughness), roughness, mu);
        }
        muRows[static_cast<std::size_t>(k)] = row;
    });

    // First mu cell: dense in roughness, on exact nodes there, so what is left is the cell alone, measured as energy sees it.
    std::vector<InterpolationError> firstCellRows(static_cast<std::size_t>(res.x));
    parallelRows(res.x, ctx.threads(), [&](int ri) {
        const double roughness = pathtracer::scene::albedoGridRoughness(static_cast<float>(ri));
        InterpolationError row{0.0, roughness, 0.0};
        for (double fraction : firstCellFractions) {
            const double mu = pathtracer::scene::albedoGridMu(static_cast<float>(fraction));
            // In E*mu, the projected-solid-angle integrand every albedo integral sums, as the cell carries energy.
            recordWorst(row, directionalAlbedoError(mu, roughness) * mu, roughness, mu);
        }
        firstCellRows[static_cast<std::size_t>(ri)] = row;
    });

    // Eavg's own 1-D lerp, reaching every Kulla-Conty tint: a different route in, so its own number.
    std::vector<InterpolationError> averageRows(static_cast<std::size_t>(res.x - 1));
    parallelRows(res.x - 1, ctx.threads(), [&](int ri) {
        const double roughness = pathtracer::scene::albedoGridRoughness(static_cast<float>(ri) + 0.5F);
        const glm::dvec4 shipped(pathtracer::scene::averageAlbedoSplit(static_cast<float>(roughness)));
        const glm::dvec3 exact = referenceAverageAlbedo(alphaAt(roughness));
        const glm::dvec3 error = glm::abs(glm::dvec3(shipped) - exact);
        const double delta = std::max({error.x, error.y, error.z, std::abs(shipped.w - std::max(1.0 - (exact.x + exact.y), 0.0))});
        averageRows[static_cast<std::size_t>(ri)] = {delta, roughness, -1.0};
    });

    const auto reduce = [](const std::vector<InterpolationError>& rows) {
        InterpolationError worst{0.0, 0.0, 0.0};
        for (const InterpolationError& row : rows) {
            recordWorst(worst, row.worst, row.roughness, row.mu);
        }
        return worst;
    };

    const std::array<std::tuple<const char*, InterpolationError, double>, 5> measurements = {{
        {"control: both axes on nodes    ", reduce(controlRows), kControlTolerance},
        {"roughness axis, mu on nodes    ", reduce(roughnessRows), kRoughnessAxisTolerance},
        {"mu axis cells 1..n, r on nodes ", reduce(muRows), kMuAxisTolerance},
        {"first mu cell in E*mu, r on nodes", reduce(firstCellRows), kFirstMuCellTolerance},
        {"Eavg lerp, 1-D in roughness    ", reduce(averageRows), kAverageAlbedoTolerance},
    }};

    bool ok = true;
    std::cout << "bsdf_validate: albedo table interpolation error vs independent quadrature, worst F82 channel\n";
    std::cout << "  measurement                                 worst        at roughness   mu\n";
    for (const auto& [name, worst, tolerance] : measurements) {
        std::cout << "  " << name << "   " << worst.worst << "   " << worst.roughness << "   "
                  << worst.mu << '\n';
        if (!(worst.worst <= tolerance)) {
            std::cerr << "bsdf_validate: FAILED albedo table interpolation on the " << name
                       << " -- worst " << worst.worst << " at roughness " << worst.roughness << " mu "
                       << worst.mu << " exceeds " << tolerance
                       << ". The committed src/scene/albedo_table.inc lost accuracy on this axis.\n";
            ok = false;
        }
    }
    finish(ctx, ok, "albedo_table_interpolation failed; see the rows above");
    return;
}

// Exact-Fresnel dielectric reflection albedo R(mu) in double, the escape table's reflect channel on the reflect side's measure.
double referenceDielectricReflectAlbedo(double mu, double alpha, double ior) {
    constexpr int kPanels = 48;
    constexpr int kPsiNodes = 192;
    const double sinTv = std::sqrt(std::max(0.0, 1.0 - (mu * mu)));
    const auto azimuth = [&](double phi) {
        const double horizontal = sinTv * std::cos(phi);
        const double radius = std::sqrt((horizontal * horizontal) + (mu * mu));
        const double delta = std::atan2(horizontal, mu);
        const double psiMax = std::atan(std::tan(0.5 * (delta + (0.5 * kPiDouble))) / alpha);
        return gaussLegendre<kPsiNodes>(0.0, psiMax, [&](double psi) {
            const double thetaH = std::atan(alpha * std::tan(psi));
            const double woDotH = radius * std::cos(thetaH - delta);
            const double wiZ = radius * std::cos((2.0 * thetaH) - delta);
            return (woDotH / std::cos(thetaH)) * referenceSmithG2OverCosO(mu, wiZ, alpha) * std::sin(psi) * std::cos(psi) *
                   referenceDielectricFresnel(woDotH, ior);
        });
    };
    return (2.0 / kPiDouble) * (simpson(0.0, 0.5 * kPiDouble, kPanels, azimuth) + simpson(0.5 * kPiDouble, kPiDouble, kPanels, azimuth));
}

// OpenPBR's glossy-diffuse is albedo scaling, f_spec + (1 - E_spec(mu_o)) f_diffuse: the diffuse weight is a function of wo alone.
PT_CHECK(glossy_diffuse_albedo_scaling, Slow, Exact) {
    // The escape table's trilinear read over (roughness, mu, eta) at 32 x 64 x 64 dominates; the references are exact to ~1e-9.
    constexpr double kTableTolerance = 2e-3;
    // Simpson over the hemisphere against the lobe's own normalisation: the GGX peak at roughness 0.25 is the hardest panel.
    constexpr double kClosureTolerance = 2e-3;
    constexpr int kPanels = 256;
    constexpr double kWiInvarianceTolerance = 1e-6;
    const std::array<double, 4> iors = {1.33, 1.5, 2.0, 2.5};
    const std::array<double, 4> roughnesses = {0.25, 0.5, 0.75, 1.0};
    const std::array<double, 3> cosines = {0.4, 0.7, 1.0};
    const glm::vec3 albedo(0.8F, 0.3F, 0.1F);
    constexpr float kDiffuseRoughness = 0.5F;
    const glm::vec3 rho = pathtracer::scene::eonAlbedoInversion(albedo, kDiffuseRoughness);

    struct Worst {
        double table = 0.0;
        double closure = 0.0;
        double invariance = 0.0;
    };
    std::vector<Worst> worstByIor(iors.size());
    parallelRows(static_cast<int>(iors.size()), ctx.threads(), [&](int k) {
        const double ior = iors[static_cast<std::size_t>(k)];
        Worst& worst = worstByIor[static_cast<std::size_t>(k)];
        for (double roughness : roughnesses) {
            const Surface params = makeDiffuseParams(albedo, kDiffuseRoughness, static_cast<float>(roughness), static_cast<float>(ior));
            for (double muO : cosines) {
                const float sinO = std::sqrt(std::max(0.0F, 1.0F - static_cast<float>(muO * muO)));
                const glm::vec3 wo(sinO, 0.0F, static_cast<float>(muO));
                const pathtracer::scene::BsdfClosure closure = pathtracer::scene::makeBsdfClosure(params, wo);
                const pathtracer::scene::DielectricSlab& slab = closure.dielectric;
                // The single-scattering albedo the scaling reads, against an exact-Fresnel quadrature of the same lobe.
                worst.table = std::max(worst.table, std::abs(slab.reflectSingle.x - referenceDielectricReflectAlbedo(muO, alphaAt(roughness), ior)));
                // The reflection the closure evaluates must carry the energy the scaling removes: int f_spec = E_spec.
                const double reflected = simpson(0.0, 1.0, kPanels, [&](double mu) {
                    const double sine = std::sqrt(std::max(0.0, 1.0 - (mu * mu)));
                    return simpson(0.0, 2.0 * kPiDouble, kPanels, [&](double phi) {
                        const glm::vec3 wi(static_cast<float>(sine * std::cos(phi)), static_cast<float>(sine * std::sin(phi)), static_cast<float>(mu));
                        return static_cast<double>(pathtracer::scene::evaluateBsdfSplit(closure, wi).specular.x);
                    });
                });
                worst.closure = std::max(worst.closure, std::abs(reflected - (slab.reflectSingle.x + slab.multiReflect)));
                // diffuse / (EON * mu_i) is the weight itself, the same at every wi; the reference EON is independent of the shipped one.
                for (double muI : cosines) {
                    const float sinI = std::sqrt(std::max(0.0F, 1.0F - static_cast<float>(muI * muI)));
                    const glm::vec3 wi(sinI * std::cos(1.1F), sinI * std::sin(1.1F), static_cast<float>(muI));
                    const double measured = pathtracer::scene::evaluateBsdfSplit(closure, wi).diffuse.x /
                                            (static_cast<double>(referenceEon(rho, kDiffuseRoughness, wi, wo).x) * muI);
                    const double expected = 1.0 - (slab.reflectSingle.x + slab.multiReflect);
                    worst.invariance = std::max(worst.invariance, std::abs(measured - expected) / expected);
                }
            }
        }
    });
    bool ok = true;
    std::cout << "bsdf_validate: glossy-diffuse albedo scaling, worst |R_ss - reference|, |int f_spec - E_spec|, wi-invariance\n";
    for (std::size_t k = 0; k < iors.size(); ++k) {
        const Worst& worst = worstByIor[k];
        std::cout << "    ior " << iors[k] << "   " << worst.table << "   " << worst.closure << "   " << worst.invariance << '\n';
        if (!(worst.table <= kTableTolerance && worst.closure <= kClosureTolerance && worst.invariance <= kWiInvarianceTolerance)) {
            std::cerr << "bsdf_validate: FAILED glossy-diffuse albedo scaling at ior=" << iors[k] << '\n';
            ok = false;
        }
    }
    finish(ctx, ok, "glossy_diffuse_albedo_scaling failed; see the rows above");
}

// Cauchy dispersion against its contract: (ior, abbe) means n_d with V_d = (n_d-1)/(n_F-n_C), so those identities are the specification.
PT_CHECK(cauchy_dispersion, Fast, Exact) {
    // Both bands are rounding headroom, not fit error: the Abbe band is relative for n_F-n_C's ~187x cancellation, the d-line absolute.
    constexpr float kDLineTolerance = 1e-6F;
    constexpr float kAbbeRelativeTolerance = 1e-4F;

    struct Glass {
        const char* name;
        float iorD;
        float abbe;
    };
    // Real catalogue materials spanning the range, including the two OpenPBR names as mid-dispersion; published constants, not fits.
    const std::array<Glass, 4> glasses{{
        {"Schott N-BK7 (crown)", 1.5168F, 64.17F},
        {"Schott SF10 (dense flint)", 1.72825F, 28.53F},
        {"water, 20C", 1.333F, 55.4F},
        {"diamond", 2.417F, 55.3F},
    }};
    constexpr float kLambdaDNm = 587.56F;
    constexpr float kLambdaFNm = 486.13F;
    constexpr float kLambdaCNm = 656.27F;

    bool ok = true;
    std::cout << "bsdf_validate: Cauchy dispersion inverted from (ior, abbe)\n";
    for (const Glass& glass : glasses) {
        const float nD = pathtracer::scene::cauchyIor(glass.iorD, glass.abbe, kLambdaDNm);
        const float nF = pathtracer::scene::cauchyIor(glass.iorD, glass.abbe, kLambdaFNm);
        const float nC = pathtracer::scene::cauchyIor(glass.iorD, glass.abbe, kLambdaCNm);
        const float measuredAbbeDifference = nF - nC;
        // The Abbe number's definition, rearranged. Computed here from the authored inputs alone.
        const float expectedAbbeDifference = (glass.iorD - 1.0F) / glass.abbe;

        const float nRed = pathtracer::scene::cauchyIor(glass.iorD, glass.abbe,
                                                     pathtracer::scene::kRgbWavelengthsNm.x);
        const float nGreen = pathtracer::scene::cauchyIor(glass.iorD, glass.abbe,
                                                       pathtracer::scene::kRgbWavelengthsNm.y);
        const float nBlue = pathtracer::scene::cauchyIor(glass.iorD, glass.abbe,
                                                      pathtracer::scene::kRgbWavelengthsNm.z);

        std::cout << "  " << glass.name;
        for (std::size_t pad = std::string(glass.name).size(); pad < 28; ++pad) {
            std::cout << ' ';
        }
        std::cout << "n_d " << nD << "   n_F-n_C " << measuredAbbeDifference << " (expected "
                  << expectedAbbeDifference << ")   RGB [" << nRed << ", " << nGreen << ", " << nBlue
                  << "]   dn(B-R) " << (nBlue - nRed) << '\n';

        if (!(std::fabs(nD - glass.iorD) <= kDLineTolerance)) {
            std::cerr << "bsdf_validate: FAILED d-line index for " << glass.name << " -- n(lambda_d) "
                      << nD << ", authored ior " << glass.iorD
                      << ". The authored ior IS the index at the d line; a dispersion curve that does not"
                         " pass through it has changed the material, not just spread it.\n";
            ok = false;
        }
        if (!(std::fabs(measuredAbbeDifference - expectedAbbeDifference) <=
              kAbbeRelativeTolerance * expectedAbbeDifference)) {
            std::cerr << "bsdf_validate: FAILED Abbe difference for " << glass.name << " -- n_F - n_C "
                      << measuredAbbeDifference << ", expected " << expectedAbbeDifference
                      << " = (ior-1)/abbe. That equation is the definition of the Abbe number, so the"
                         " authored abbe does not mean what it says.\n";
            ok = false;
        }
        // Normal dispersion: index falls with wavelength, so blue bends most, pinning B's sign and kRgbWavelengthsNm's ordering together.
        if (!(nBlue > nGreen && nGreen > nRed)) {
            std::cerr << "bsdf_validate: FAILED normal dispersion ordering for " << glass.name
                      << " -- RGB indices [" << nRed << ", " << nGreen << ", " << nBlue
                      << "] are not increasing toward blue. A transparent dielectric has no anomalous"
                         " dispersion in the visible band.\n";
            ok = false;
        }
    }

    // V_d infinite, dispersion scale 0, is the off switch every non-dispersive material relies on: B = 0, so the ior returns bit-exact.
    std::cout << "  infinite V_d returns the authored ior unchanged at every wavelength\n";
    for (const Glass& glass : glasses) {
        for (float lambda : {kLambdaFNm, pathtracer::scene::kRgbWavelengthsNm.z, kLambdaDNm,
                             pathtracer::scene::kRgbWavelengthsNm.x, kLambdaCNm}) {
            const float n = pathtracer::scene::cauchyIor(glass.iorD, std::numeric_limits<float>::infinity(), lambda);
            if (n != glass.iorD) {
                std::cerr << "bsdf_validate: FAILED infinite-V_d no-op at ior " << glass.iorD << " lambda "
                          << lambda << " -- returned " << n
                          << ", expected the authored ior bit-for-bit.\n";
                ok = false;
            }
        }
    }
    finish(ctx, ok, "cauchy_dispersion failed; see the rows above");
    return;
}

// Metal F82 via the public API at the mirrored pair, smooth enough that multiple scattering is negligible: f = K * F82, so ratios are F's.
PT_CHECK(metal_fresnel, Fast, Exact) {
    // A quotient of float BSDF values against a double reference: relative, a few float roundings of the lobe and the ratio.
    constexpr double kRelativeTolerance = 2e-5;
    constexpr float kSmoothRoughness = 0.05F;
    // The property Schlick structurally cannot have: at grazing a black specular_color must sit well below a white one.
    constexpr double kMinGrazingSeparation = 0.05;
    const std::array<float, 4> reflectivities = {0.1F, 0.5F, 0.95F, 1.0F};
    const std::array<float, 5> tints = {0.0F, 0.25F, 0.5F, 0.75F, 1.0F};
    const std::array<float, 4> cosines = {1.0F, 0.6F, static_cast<float>(kMuBar), 0.05F};
    const std::array<float, 3> weights = {0.5F, 1.0F, 1.5F};

    bool ok = true;
    int rowsChecked = 0;
    std::cout << "bsdf_validate: metal F82-tint Fresnel against the specification's formula\n";
    for (float reflectivity : reflectivities) {
        for (float tint : tints) {
            for (float weight : weights) {
                const Surface params = paramsWith([&](OpenPbrInputs<Constant>& inputs) {
                    inputs.baseColor = glm::vec3(reflectivity);
                    inputs.baseMetalness = 1.0F;
                    inputs.specularRoughness = kSmoothRoughness;
                    inputs.specularColor = glm::vec3(tint);
                    inputs.specularWeight = weight;
                });
                for (float cosine : cosines) {
                    // viewFresnel is the macro Fresnel the lobe evaluates at its own facet: the curve itself, not a BSDF reading.
                    const double measured = viewFresnel(params, cosine).x;
                    const double expected = weight * referenceF82(reflectivity, tint, cosine);
                    ++rowsChecked;
                    // Floored at 1e-2 of a unit reflectance: where F82 is 0 by definition, float rounding leaves a few 1e-8 either side.
                    if (!(std::abs(measured - expected) <= kRelativeTolerance * std::max(std::abs(expected), 1e-2))) {
                        std::cerr << "bsdf_validate: FAILED metal F82 at F0=" << reflectivity << " specular_color=" << tint
                                  << " specular_weight=" << weight << " cos=" << cosine << " measured " << measured << " vs " << expected << '\n';
                        ok = false;
                    }
                }
                // F(1) = specular_weight * F0 and F(1/7) = specular_weight * tint * Schlick(1/7): the two points the tint is defined by.
                const double schlickAtMuBar = reflectivity + ((1.0 - reflectivity) * std::pow(1.0 - kMuBar, 5.0));
                const double atMuBar = viewFresnel(params, static_cast<float>(kMuBar)).x;
                if (!(std::abs(atMuBar - (weight * tint * schlickAtMuBar)) <= kRelativeTolerance * std::max(weight * schlickAtMuBar, 1e-2))) {
                    std::cerr << "bsdf_validate: FAILED F82 definition at F0=" << reflectivity << " specular_color=" << tint
                              << ": F(1/7) " << atMuBar << " is not specular_color * Schlick(1/7)\n";
                    ok = false;
                }
            }
        }
        // The dip Schlick cannot express: below a perfect mirror a black tint reflects less at grazing than a white one.
        if (reflectivity < 1.0F) {
            const auto atGrazing = [&](float tint) {
                return viewFresnel(makeColoredMetalParams(kSmoothRoughness, glm::vec3(tint)), static_cast<float>(kMuBar)).x;
            };
            const double separation = (atGrazing(1.0F) - atGrazing(0.0F)) / atGrazing(1.0F);
            if (!(separation >= kMinGrazingSeparation)) {
                std::cerr << "bsdf_validate: FAILED F82 grazing dip at F0 0.5: separation " << separation << '\n';
                ok = false;
            }
        }
    }
    std::cout << "  " << rowsChecked << " points against the formula\n";
    finish(ctx, ok, "metal_fresnel failed; see the rows above");
}

// base_metalness mixes two complete bases, so every lobe value is exactly linear in it: metal is weighted once, not twice.
PT_CHECK(fractional_metal_is_linear, Fast, Exact) {
    constexpr float kRelativeTolerance = 1e-5F;
    const std::array<float, 3> roughnesses = {0.1F, 0.4F, 0.9F};
    const std::array<float, 3> metalnesses = {0.25F, 0.5F, 0.75F};
    const std::array<float, 2> transmissions = {0.0F, 0.6F};
    const std::array<float, 4> cosines = {1.0F, 0.7F, 0.4F, 0.15F};
    const glm::vec3 baseColor(0.9F, 0.6F, 0.3F);
    const glm::vec3 tint(0.8F, 0.7F, 0.95F);

    bool ok = true;
    int rowsChecked = 0;
    for (float roughness : roughnesses) {
        for (float transmission : transmissions) {
            const auto at = [&](float metalness) {
                return paramsWith([&](OpenPbrInputs<Constant>& inputs) {
                    inputs.baseColor = baseColor;
                    inputs.specularColor = tint;
                    inputs.specularRoughness = roughness;
                    inputs.transmissionWeight = transmission;
                    inputs.baseMetalness = metalness;
                });
            };
            const Surface dielectric = at(0.0F);
            const Surface metal = at(1.0F);
            for (float metalness : metalnesses) {
                const Surface mixed = at(metalness);
                for (float muA : cosines) {
                    for (float muB : cosines) {
                        const float sinA = std::sqrt(std::max(0.0F, 1.0F - (muA * muA)));
                        const float sinB = std::sqrt(std::max(0.0F, 1.0F - (muB * muB)));
                        const glm::vec3 wo(sinA, 0.0F, muA);
                        const glm::vec3 wi(sinB * std::cos(1.1F), sinB * std::sin(1.1F), muB);
                        const glm::vec3 expected = ((1.0F - metalness) * pathtracer::scene::evaluateBsdf(dielectric, wo, wi)) +
                                                   (metalness * pathtracer::scene::evaluateBsdf(metal, wo, wi));
                        const glm::vec3 measured = pathtracer::scene::evaluateBsdf(mixed, wo, wi);
                        ++rowsChecked;
                        if (!(maxChannel(glm::abs(measured - expected)) <= kRelativeTolerance * std::max(maxChannel(expected), 1e-4F))) {
                            std::cerr << "bsdf_validate: FAILED fractional metal linearity at roughness=" << roughness
                                      << " transmission=" << transmission << " metalness=" << metalness << " muO=" << muA << " muI=" << muB
                                      << " f=" << measured.x << " vs mix " << expected.x << '\n';
                            ok = false;
                        }
                    }
                }
            }
        }
    }
    std::cout << "bsdf_validate: fractional metal, " << rowsChecked << " pairs exactly linear in base_metalness\n";
    finish(ctx, ok, "fractional_metal_is_linear failed; see the rows above");
}

// specular_weight scales the dielectric's normal-incidence reflectance, F0 = xi * F_s, through the modulated ior OpenPBR specifies.
PT_CHECK(specular_weight_modulates_f0, Fast, Exact) {
    constexpr double kRelativeTolerance = 4e-6;
    const std::array<float, 4> iors = {1.33F, 1.5F, 2.0F, 2.5F};
    bool ok = true;
    for (float ior : iors) {
        const double reflectance = std::pow((ior - 1.0) / (ior + 1.0), 2.0);
        for (double weight : {0.0, 0.25, 0.5, 1.0, 1.5, 0.5 / reflectance}) {
            const Surface params = paramsWith([&](OpenPbrInputs<Constant>& inputs) {
                inputs.specularIor = ior;
                inputs.specularWeight = static_cast<float>(weight);
            });
            const double measured = viewFresnel(params, 1.0F).x;
            const double expected = std::min(weight * reflectance, 1.0);
            if (!(std::abs(measured - expected) <= kRelativeTolerance * std::max(expected, 1e-6))) {
                std::cerr << "bsdf_validate: FAILED specular_weight at ior=" << ior << " weight=" << weight << " F(1) " << measured
                          << " vs xi*F_s " << expected << '\n';
                ok = false;
            }
        }
    }
    finish(ctx, ok, "specular_weight_modulates_f0 failed; see the rows above");
}

// specular_color tints the dielectric's reflection from above, OpenPBR's non-physical highlight tint; transmission stays untinted.
PT_CHECK(specular_color_tints_reflection_only, Fast, Exact) {
    constexpr float kRelativeTolerance = 1e-6F;
    const glm::vec3 tint(0.9F, 0.5F, 0.2F);
    bool ok = true;
    for (float roughness : {0.2F, 0.6F}) {
        const auto make = [&](glm::vec3 colour) {
            return paramsWith([&](OpenPbrInputs<Constant>& inputs) {
                inputs.specularRoughness = roughness;
                inputs.transmissionWeight = 1.0F;
                inputs.specularColor = colour;
            });
        };
        const Surface white = make(glm::vec3(1.0F));
        const Surface tinted = make(tint);
        const glm::vec3 wo(0.6F, 0.0F, 0.8F);
        const glm::vec3 reflected(-0.6F, 0.0F, 0.8F);
        const glm::vec3 refracted = glm::refract(-wo, glm::vec3(0.0F, 0.0F, 1.0F), 1.0F / 1.5F);
        const glm::vec3 specularWhite = pathtracer::scene::evaluateBsdfSplit(white, wo, reflected).specular;
        const glm::vec3 specularTinted = pathtracer::scene::evaluateBsdfSplit(tinted, wo, reflected).specular;
        const glm::vec3 transmitWhite = pathtracer::scene::evaluateBsdfSplit(white, wo, refracted).transmission;
        const glm::vec3 transmitTinted = pathtracer::scene::evaluateBsdfSplit(tinted, wo, refracted).transmission;
        if (!(maxChannel(glm::abs(specularTinted - (tint * specularWhite))) <= kRelativeTolerance * maxChannel(specularWhite))) {
            std::cerr << "bsdf_validate: FAILED specular_color does not scale the reflection at roughness " << roughness << '\n';
            ok = false;
        }
        if (transmitTinted != transmitWhite || !(maxChannel(transmitWhite) > 0.0F)) {
            std::cerr << "bsdf_validate: FAILED specular_color reached the transmission at roughness " << roughness << '\n';
            ok = false;
        }
    }
    finish(ctx, ok, "specular_color_tints_reflection_only failed; see the rows above");
}

// The specular lobe's closed form at the mirrored pair in double: nh is exactly +z, so D = 1/(pi*alpha^2) exactly and G2 uses one cosine.
double specularGeometry(double alpha, double cosine) {
    const double alpha2 = alpha * alpha;
    const double d = 1.0 / (kPiDouble * alpha2);
    const double tan2 = (1.0 - (cosine * cosine)) / (cosine * cosine);
    const double lambda = 0.5 * (-1.0 + std::sqrt(1.0 + (alpha2 * tan2)));
    const double g2 = 1.0 / (1.0 + (2.0 * lambda));
    return (d * g2) / (4.0 * cosine * cosine);
}

// The only instrument reading the specular lobe's absolute magnitude, sampleBsdf's f/pdf cancelling a common factor; grazing to cos 1e-7.
PT_CHECK(dielectric_fresnel, Fast, Exact) {
    // Float32 round-off against a double reference, relative to F since rounding is; M is not resolvable at these roughnesses.
    constexpr double kFresnelTolerance = 4.7e-7;
    // Normal incidence is an exact identity, not a fit: referenceDielectricFresnel(1, n) is ((n-1)/(n+1))^2, both polarisations equal.
    constexpr double kNormalIncidenceTolerance = 3e-8;
    const std::array<double, 4> iors = {1.1, 1.5, 1.5168, 2.5};   // 1.5168 is glass.json's own N-BK7 value
    const std::array<float, 3> roughnesses = {0.02F, 0.05F, 0.1F};
    const std::array<float, 14> cosines = {1.0F,  0.9F,  0.7F,  0.5F,  0.35F, 0.25F, 0.15F,
                                           0.08F, 0.02F, 4e-4F, 2e-4F, 1e-4F, 1e-5F, 1e-7F};

    bool ok = true;
    int rowsChecked = 0;
    std::cout << "bsdf_validate: dielectric Fresnel, absolute lobe magnitude vs exact unpolarized\n";
    std::cout << "  ior      rough   worst |err|/tol   F(cos=1e-7) measured / exact\n";
    for (double ior : iors) {
        for (float roughness : roughnesses) {
            const double alpha =
                static_cast<double>(roughness) * roughness;
            const Surface params = paramsWith([&](OpenPbrInputs<Constant>& inputs) {
                inputs.specularRoughness = roughness;
                inputs.specularIor = static_cast<float>(ior);
            });
            double worstError = 0.0;
            double previous = -1.0;
            double atGrazing = 0.0;
            for (float cosine : cosines) {
                const float sine = std::sqrt(std::max(0.0F, 1.0F - (cosine * cosine)));
                const glm::vec3 wo(sine, 0.0F, cosine);
                const glm::vec3 wi(-sine, 0.0F, cosine);
                // The single-scattering lobe alone: the closure's multiple scattering is switched off, its energy below this tolerance.
                pathtracer::scene::BsdfClosure closure = pathtracer::scene::makeBsdfClosure(params, wo);
                closure.dielectric.multiReflect = 0.0F;
                const double measured = static_cast<double>(pathtracer::scene::evaluateBsdfSplit(closure, wi).specular.x) /
                                        (specularGeometry(alpha, cosine) * cosine);
                const double expected = referenceDielectricFresnel(cosine, ior);
                const double tolerance = cosine == 1.0F ? kNormalIncidenceTolerance : kFresnelTolerance * expected;
                ++rowsChecked;
                worstError = std::max(worstError, std::abs(measured - expected) / tolerance);
                atGrazing = measured;
                if (!(std::abs(measured - expected) <= tolerance)) {
                    std::cerr << "bsdf_validate: FAILED dielectric Fresnel at ior=" << ior
                              << " roughness=" << roughness << " cos=" << cosine << " measured " << measured
                              << " vs reference " << expected << " (tolerance " << tolerance
                              << "); the lobe's absolute magnitude is D*G2*F/(4*muO), so this fires on an "
                                 "error in any of the three\n";
                    ok = false;
                }
                // Strict, no epsilon: unpolarized external reflection is monotone in theta for n > 1, and the smallest step clears noise.
                if (previous >= 0.0 && !(measured > previous)) {
                    std::cerr << "bsdf_validate: FAILED dielectric Fresnel monotonicity at ior=" << ior
                              << " roughness=" << roughness << " cos=" << cosine << " gave " << measured
                              << " after " << previous
                              << "; reflectance must rise strictly as the view approaches grazing\n";
                    ok = false;
                }
                previous = measured;
            }
            std::cout << "  " << ior << "      " << roughness << "    " << worstError << "    " << atGrazing
                      << " / " << referenceDielectricFresnel(cosines.back(), ior) << '\n';
        }
    }
    // Index match, out of the sweep: at ior 1 the curve is identically zero; the cosines run to 1e-5, where the old Snell form hit TIR.
    const std::array<float, 8> indexMatchedCosines = {1.0F,     0.5F,          0.02F,  1e-3F,
                                                      2.44e-4F, 1.7263349e-4F, 1e-4F, 1e-5F};
    int indexMatchedRows = 0;
    for (float roughness : roughnesses) {
        const Surface params = paramsWith([&](OpenPbrInputs<Constant>& inputs) {
            inputs.specularRoughness = roughness;
            inputs.specularIor = 1.0F;
        });
        for (float cosine : indexMatchedCosines) {
            const float sine = std::sqrt(std::max(0.0F, 1.0F - (cosine * cosine)));
            const glm::vec3 wo(sine, 0.0F, cosine);
            const glm::vec3 wi(-sine, 0.0F, cosine);
            const float measured = maxChannel(pathtracer::scene::evaluateBsdfSplit(params, wo, wi).specular);
            ++indexMatchedRows;
            if (!(measured == 0.0F)) {
                std::cerr << "bsdf_validate: FAILED dielectric Fresnel at index match, roughness="
                          << roughness << " cos=" << cosine << " gave " << measured
                          << "; an ior-1 interface has no critical angle and reflects nothing, so the "
                             "specular lobe must be exactly zero at every angle\n";
                ok = false;
            }
        }
    }
    // No anti-vacuity guard here: both arrays are non-empty at compile time, every row is compared, and the count is fixed.
    std::cout << "  dielectric Fresnel: " << rowsChecked << " points vs reference, "
              << indexMatchedRows << " index-matched rows at exactly zero\n";
    finish(ctx, ok, "dielectric_fresnel failed; see the rows above");
    return;
}

// Helmholtz reciprocity per slab: the reflections and EON are symmetric; OpenPBR's albedo-scaling weight on the diffuse is wo's alone.
PT_CHECK(reciprocity, Fast, Exact) {
    constexpr float kRelativeTolerance = 1e-4F;
    const std::array<float, 4> roughnesses = {0.05F, 0.25F, 0.5F, 1.0F};
    const std::array<float, 3> metallics = {0.0F, 0.5F, 1.0F};
    const std::array<float, 4> cosines = {1.0F, 0.7F, 0.4F, 0.15F};
    // At 0 the diffuse lobe is Lambertian and reciprocal for free, so the sweep is what puts EON under this check.
    const std::array<float, 3> diffuseRoughnesses = {0.0F, 0.5F, 1.0F};

    bool ok = true;
    for (float roughness : roughnesses) {
        for (float metallic : metallics) {
            for (float diffuseRoughness : diffuseRoughnesses) {
                const Surface params = makeParams(roughness, metallic, 0.0F, diffuseRoughness);
                for (float muA : cosines) {
                    for (float muB : cosines) {
                        // Non-coplanar pair: a shared azimuth would leave a swapped-phi bug invisible.
                        const float sinA = std::sqrt(std::max(0.0F, 1.0F - (muA * muA)));
                        const float sinB = std::sqrt(std::max(0.0F, 1.0F - (muB * muB)));
                        const glm::vec3 wo(sinA, 0.0F, muA);
                        const glm::vec3 wi(sinB * std::cos(1.1F), sinB * std::sin(1.1F), muB);
                        const pathtracer::scene::BsdfClosure out = pathtracer::scene::makeBsdfClosure(params, wo);
                        const pathtracer::scene::BsdfClosure back = pathtracer::scene::makeBsdfClosure(params, wi);
                        const pathtracer::scene::BsdfEval forward = pathtracer::scene::evaluateBsdfSplit(out, wi);
                        const pathtracer::scene::BsdfEval reverse = pathtracer::scene::evaluateBsdfSplit(back, wo);
                        const auto check = [&](const glm::vec3& f, const glm::vec3& g, const char* slab) {
                            const float scale = std::max(maxChannel(f), maxChannel(g));
                            if (!(maxChannel(glm::abs(f - g)) <= kRelativeTolerance * std::max(scale, 1e-4F))) {
                                std::cerr << "bsdf_validate: FAILED " << slab << " reciprocity at roughness=" << roughness
                                          << " metallic=" << metallic << " diffuseRoughness=" << diffuseRoughness << " muO=" << muA
                                          << " muI=" << muB << " f(wo->wi)=" << f.x << " f(wi->wo)=" << g.x << '\n';
                                ok = false;
                            }
                        };
                        // Bare f from the cosine-weighted values; the diffuse also divides out the weight albedo scaling puts on it at wo.
                        check(forward.specular / muB, reverse.specular / muA, "specular");
                        if (metallic < 1.0F) {
                            check(forward.diffuse / (out.diffuseWeight * muB), reverse.diffuse / (back.diffuseWeight * muA), "diffuse");
                        }
                    }
                }
            }
        }
    }
    finish(ctx, ok, "reciprocity failed; see the rows above");
    return;
}

// eta^2-corrected transmission reciprocity, single scatter only: multiple-scatter transmission is not reciprocal (0.40 fails 5x).
PT_CHECK(transmission_reciprocity, Fast, Exact) {
    // Not checkReciprocity's 1e-4: D is sharply peaked at these alphas and the two queries build ht from differently scaled sums.
    constexpr float kRelativeTolerance = 1e-2F;
    const std::array<float, 2> roughnesses = {0.05F, 0.10F};
    const std::array<float, 3> iors = {1.2F, 1.5F, 2.0F};
    const std::array<float, 4> cosines = {1.0F, 0.9F, 0.7F, 0.5F};
    const std::array<float, 5> offsets = {0.0F, 0.5F, 1.0F, 2.0F, 4.0F};  // multiples of alpha
    // Perturbation axes: out-of-plane and diagonal put wo and wi at different azimuths, so a swapped-phi error cannot hide.
    const std::array<glm::vec3, 3> axes = {
        {{0.0F, 1.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {0.70710678F, 0.70710678F, 0.0F}}};

    bool ok = true;
    int pairsSeen = 0;
    for (float roughness : roughnesses) {
        const float alpha = roughness * roughness;
        for (float ior : iors) {
            const Surface params = paramsWith([&](OpenPbrInputs<Constant>& inputs) {
                inputs.specularRoughness = roughness;
                inputs.specularIor = ior;
                inputs.transmissionWeight = 1.0F;
            });
            for (float mu : cosines) {
                const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (mu * mu))), 0.0F, mu);
                const glm::vec3 refracted =
                    glm::refract(-wo, glm::vec3(0.0F, 0.0F, 1.0F), 1.0F / ior);
                for (float offset : offsets) {
                    const float angle = offset * alpha;
                    for (std::size_t axisIndex = 0; axisIndex < axes.size(); ++axisIndex) {
                        // A zero offset lands on the refracted direction whatever the axis, so only the first pass over it is distinct.
                        if (offset == 0.0F && axisIndex > 0) {
                            continue;
                        }
                        const glm::vec3& axis = axes[axisIndex];
                        const glm::vec3 tangent =
                            glm::normalize(axis - (refracted * glm::dot(axis, refracted)));
                        const glm::vec3 wi = glm::normalize((std::cos(angle) * refracted) +
                                                             (std::sin(angle) * tangent));
                        // Bare f from the cosine-weighted values: each direction's own |cos| divides out.
                        const glm::vec3 forward = pathtracer::scene::evaluateBsdf(params, wo, wi) * ior * ior / std::abs(wi.z);
                        const glm::vec3 reverse = pathtracer::scene::evaluateBsdf(params, wi, wo) / std::abs(wo.z);
                        const float scale = std::max(maxChannel(forward), maxChannel(reverse));
                        if (scale <= 0.0F) {
                            continue;
                        }
                        ++pairsSeen;
                        if (!(maxChannel(glm::abs(forward - reverse)) <= kRelativeTolerance * scale)) {
                            std::cerr << "bsdf_validate: FAILED transmission reciprocity at roughness="
                                      << roughness << " ior=" << ior << " muO=" << mu
                                      << " offset=" << offset << "*alpha"
                                      << " f(wo->wi)*ior^2=" << maxChannel(forward)
                                      << " f(wi->wo)=" << maxChannel(reverse) << '\n';
                            ok = false;
                        }
                    }
                }
            }
        }
    }
    if (pairsSeen == 0) {
        std::cerr << "bsdf_validate: FAILED transmission reciprocity observed zero non-zero pairs -- "
                     "every constructed wi missed the lobe, so nothing was tested\n";
        ok = false;
    }
    std::cout << "  transmission reciprocity: " << pairsSeen << " non-zero pairs\n";
    finish(ctx, ok, "transmission_reciprocity failed; see the rows above");
    return;
}

// Round trip: the eta^2 compression (Veach 1997 5.2) applies entering and exiting and must cancel exactly, so this asserts the product.
PT_CHECK(transmission_round_trip, Slow, Statistical) {
    constexpr int kSampleCount = 200000;
    // Not tight to 1.0: each side's furnace contains its reflected lobe too, so a Fresnel cross-term stays; compounding lands near ior^2.
    constexpr float kTolerance = 0.08F;
    constexpr float kIorRoundTrip = 1.5F;  // matches makeParams
    const std::array<float, 3> ndotVs = {1.0F, 0.8F, 0.5F};

    bool ok = true;
    std::uint32_t seed = 4000;
    for (float ndotV : ndotVs) {
        ++seed;
        // Snell-correct pairing: the exit angle is not the entry angle, and reusing thetaI would put the exit past the critical angle.
        const Surface params = makeParams(0.05F, 0.0F, 1.0F);
        const float sinThetaI = std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV)));
        const float sinThetaT = sinThetaI / kIorRoundTrip;
        const float cosThetaT = std::sqrt(std::max(0.0F, 1.0F - (sinThetaT * sinThetaT)));
        const glm::vec3 woIn(sinThetaI, 0.0F, ndotV);
        const glm::vec3 woOut(sinThetaT, 0.0F, -cosThetaT);
        const float entering = maxChannel(furnaceLo(params, woIn, kSampleCount, seed));
        const float exiting = maxChannel(furnaceLo(params, woOut, kSampleCount, seed + 700U));
        const float roundTrip = entering * exiting;
        std::cout << "  transmission round trip ndotV=" << ndotV << ": " << entering << " x "
                  << exiting << " = " << roundTrip << '\n';
        // Two-sided deliberately: an upper bound alone is blind to the factors under-cancelling, which loses energy just as wrongly.
        if (!(roundTrip >= 1.0F - kTolerance && roundTrip <= 1.0F + kTolerance)) {
            std::cerr << "bsdf_validate: FAILED transmission round trip at ndotV=" << ndotV
                      << " -- entering " << entering << " x exiting " << exiting << " = " << roundTrip
                      << "; the eta^2 radiance-compression factors must cancel over a round trip "
                         "(expected 1.0 +/- " << kTolerance << ").\n";
            ok = false;
        }
    }
    finish(ctx, ok, "transmission_round_trip failed; see the rows above");
    return;
}

// Transmission at ior 1, needing no reference: the only check reaching the Snell block's cos^2 TIR predicate there, at every roughness.
PT_CHECK(index_matched_transmission, Fast, Exact) {
    // Straight-through is algebraic at r == 1, but it is reached through the shipped code path rather than asserted of it.
    constexpr float kDirectionTolerance = 1.2e-7F;
    // Not throughput == tint: sampleBsdf returns f/pdf, so a draw carries tint/P; chromaticity is the noise-free invariant instead.
    constexpr float kChromaticityTolerance = 1.2e-7F;
    constexpr int kRoughDraws = 4096;
    constexpr float kSmoothRoughness = 0.0F;   // the smooth interface: the delta branch
    constexpr float kRoughRoughness = 0.3F;     // alpha 0.09, comfortably above it: the refractAbout branch
    // Reaches 1e-5 as checkIndexMatchedCoat does: 2.44e-4 is 2^-12, and every row at or below it returned nullopt on the pre-fix code.
    const std::array<float, 8> cosines = {1.0F,     0.7F,          0.1F,   1e-3F,
                                          2.44e-4F, 1.7263349e-4F, 1e-4F, 1e-5F};
    // Chromatic on purpose: a white tint cannot distinguish a preserved throughput from one that merely kept its brightness.
    const glm::vec3 tint(0.8F, 0.5F, 0.2F);

    bool ok = true;
    int rowsChecked = 0;
    float worstDirection = 0.0F;
    float worstChromaticity = 0.0F;
    int roughRejections = 0;
    int transmittedAtNormal = -1;   // set by the first row, cos 1, where no formulation can report TIR
    std::cout << "bsdf_validate: index-matched transmission, straight through at every angle (ior 1)\n";
    for (float cosine : cosines) {
        const float sine = std::sqrt(std::max(0.0F, 1.0F - (cosine * cosine)));
        const glm::vec3 wo(sine, 0.0F, cosine);
        Surface smoothParams = makeTransmissiveTintParams(kSmoothRoughness, glm::vec3(1.0F), tint);
        smoothParams.specularIor = 1.0F;
        pathtracer::scene::Sampler sampler(0, 0, 0, 1, 9100U);
        const std::optional<pathtracer::scene::BsdfSample> smooth =
            pathtracer::scene::sampleBsdf(smoothParams, wo, sampler);
        ++rowsChecked;
        if (!smooth.has_value()) {
            std::cerr << "bsdf_validate: FAILED index-matched transmission at cos=" << cosine
                      << " -- the smooth branch returned no sample; an ior-1 interface has no critical "
                         "angle, so refraction cannot fail at any angle and the energy is simply lost\n";
            ok = false;
            continue;
        }
        const float directionErr = maxChannel(glm::abs(smooth->wiLocal + wo));
        const glm::vec3 ratio = smooth->throughputWeight / tint;
        const float chromaticityErr = maxChannel(ratio) - minChannel(ratio);
        worstDirection = std::max(worstDirection, directionErr);
        worstChromaticity = std::max(worstChromaticity, chromaticityErr);
        if (!(directionErr <= kDirectionTolerance)) {
            std::cerr << "bsdf_validate: FAILED index-matched transmission direction at cos=" << cosine
                      << " -- got (" << smooth->wiLocal.x << ", " << smooth->wiLocal.y << ", "
                      << smooth->wiLocal.z << "), expected -wo; an index-matched interface cannot bend a ray\n";
            ok = false;
        }
        if (!(chromaticityErr <= kChromaticityTolerance)) {
            std::cerr << "bsdf_validate: FAILED index-matched transmission chromaticity at cos=" << cosine
                      << " -- throughput/tint is (" << ratio.x << ", " << ratio.y << ", " << ratio.z
                      << "), not one scalar; at ior 1 nothing but the tint can colour a transmitted ray\n";
            ok = false;
        }

        // The rough draws must select that same delta branch and carry that same throughput; every draw is asserted, not merely counted.
        Surface roughParams = smoothParams;
        roughParams.specularRoughness = kRoughRoughness;
        int rejected = 0;
        int transmitted = 0;
        int notDelta = 0;
        int notFinite = 0;
        float roughDirection = 0.0F;
        float roughChromaticity = 0.0F;
        for (int i = 0; i < kRoughDraws; ++i) {
            pathtracer::scene::Sampler roughSampler(0, 0, i, kRoughDraws, 9200U);
            const std::optional<pathtracer::scene::BsdfSample> rough =
                pathtracer::scene::sampleBsdf(roughParams, wo, roughSampler);
            if (!rough.has_value()) {
                ++rejected;
                continue;
            }
            if (rough->type != pathtracer::scene::LobeType::Transmission) {
                continue;
            }
            ++transmitted;
            const glm::vec3 weight = rough->throughputWeight;
            if (!std::isfinite(weight.x) || !std::isfinite(weight.y) || !std::isfinite(weight.z)) {
                ++notFinite;
                continue;
            }
            if (rough->pdf != 0.0F) {
                ++notDelta;
            }
            const glm::vec3 roughRatio = weight / tint;
            roughDirection = std::max(roughDirection, maxChannel(glm::abs(rough->wiLocal + wo)));
            roughChromaticity =
                std::max(roughChromaticity, maxChannel(roughRatio) - minChannel(roughRatio));
        }
        worstDirection = std::max(worstDirection, roughDirection);
        worstChromaticity = std::max(worstChromaticity, roughChromaticity);
        roughRejections += rejected;
        if (transmittedAtNormal < 0) {
            transmittedAtNormal = transmitted;
        }
        std::cout << "  cos " << cosine << ": rough draws " << kRoughDraws << ", transmission "
                  << transmitted << ", rejected " << rejected << ", non-delta " << notDelta
                  << ", non-finite " << notFinite << '\n';
        if (notFinite != 0) {
            std::cerr << "bsdf_validate: FAILED index-matched rough transmission at cos=" << cosine
                      << " -- " << notFinite << " of " << transmitted
                      << " transmission draws carried a non-finite throughput; at ior 1 the lobe is a "
                         "delta and every draw must carry tint/P, so a NaN here is the zero half-vector "
                         "an index-matched rough refraction forms\n";
            ok = false;
        }
        if (notDelta != 0) {
            std::cerr << "bsdf_validate: FAILED index-matched rough transmission at cos=" << cosine
                      << " -- " << notDelta << " of " << transmitted
                      << " transmission draws reported a continuous pdf; an index-matched interface has "
                         "no rough lobe to sample at any roughness\n";
            ok = false;
        }
        // Nothing may reflect: exact dielectric Fresnel is identically zero at ior 1, so the reflection lobe has no value to carry.
        for (float wiZ : {0.9F, 0.5F, 0.15F}) {
            const glm::vec3 wi(std::sqrt(std::max(0.0F, 1.0F - (wiZ * wiZ))), 0.0F, wiZ);
            const glm::vec3 reflected = pathtracer::scene::evaluateBsdfSplit(roughParams, wo, wi).specular;
            if (maxChannel(reflected) != 0.0F) {
                std::cerr << "bsdf_validate: FAILED index-matched reflection at cos=" << cosine
                          << " wi.z=" << wiZ << " -- reflected (" << reflected.x << ", " << reflected.y
                          << ", " << reflected.z
                          << "), expected exactly 0; an index-matched interface reflects nothing at any "
                             "angle, so this is multiple-scattering compensation returning a deficit that "
                             "does not exist\n";
                ok = false;
            }
        }

        // The same two backstops the smooth row carries: existing and finite is not enough, the sample must point the right way.
        if (!(roughDirection <= kDirectionTolerance)) {
            std::cerr << "bsdf_validate: FAILED index-matched rough transmission direction at cos="
                      << cosine << " -- worst |wi + wo| " << roughDirection
                      << " over the rough draws; an index-matched interface cannot bend a ray whatever "
                         "microfacet normal it draws\n";
            ok = false;
        }
        if (!(roughChromaticity <= kChromaticityTolerance)) {
            std::cerr << "bsdf_validate: FAILED index-matched rough transmission chromaticity at cos="
                      << cosine << " -- worst throughput/tint spread " << roughChromaticity
                      << " over the rough draws; at ior 1 nothing but the tint can colour a transmitted "
                         "ray\n";
            ok = false;
        }
        // The count of transmission samples, not of rejections: at ior 1 the reflection lobe is zero, so every draw is a transmission.
        if (transmitted != transmittedAtNormal) {
            std::cerr << "bsdf_validate: FAILED index-matched rough transmission at cos=" << cosine
                      << " -- " << transmitted << " transmission samples against "
                      << transmittedAtNormal
                      << " at normal incidence; the lobe selection is angle-independent at ior 1, so a "
                         "deficit is refraction failing at an interface whose critical angle does not exist\n";
            ok = false;
        }
    }
    // The rough assertions are all per-draw, so a change that stopped the interface transmitting would satisfy every one vacuously.
    if (transmittedAtNormal <= 0) {
        std::cerr << "bsdf_validate: FAILED index-matched transmission -- no rough draw transmitted, so "
                     "every rough assertion above was made about nothing\n";
        ok = false;
    }
    // No anti-vacuity guard on the angle count: cosines is non-empty at compile time and the counter increments unconditionally.
    std::cout << "  " << rowsChecked << " angles asserted, worst direction error " << worstDirection
              << ", worst chromaticity spread " << worstChromaticity << ", rough rejections "
              << roughRejections << " of " << kRoughDraws * static_cast<int>(cosines.size()) << '\n';
    finish(ctx, ok, "index_matched_transmission failed; see the rows above");
    return;
}


// --- Total internal reflection past the critical angle: the interface reflects all the energy, no transmitted direction existing.
namespace {

// cos of the critical angle leaving the denser medium, sqrt(1 - (etaT/etaI)^2), stated independently of cos2Transmitted.
double criticalCosine(double iorDense) {
    const double ratio = 1.0 / iorDense;
    return std::sqrt(std::max(0.0, 1.0 - (ratio * ratio)));
}

}  // namespace

    // Inside the TIR cone there is no transmitted direction at all, so a draw reporting one is energy arriving where Snell cannot reach.
PT_CHECK(critical_angle_onset, Fast, Exact) {
    // Spans the shipped range: glass.json 1.5, clay.json 1.55, with 1.33/2.4 bracketing it so the cone width varies by over 2x.
    const std::array<double, 5> iors = {1.33, 1.5, 1.5168, 1.55, 2.4};
    // Offsets in cos straddling the critical cosine; the two smallest are below 2^-12, where the old 1 - cos^2 form lost the distinction.
    const std::array<double, 6> offsets = {1e-5, 2.44e-4, 1e-3, 1e-2, 0.1, 0.3};

    bool ok = true;
    int rowsChecked = 0;
    float worstOutside = 0.0F;
    std::cout << "bsdf_validate: total internal reflection onset at the critical angle (ior > 1)\n";
    for (double ior : iors) {
        const double muC = criticalCosine(ior);
        for (double offset : offsets) {
            // Exiting orientation: etaI is the dense medium. Inside the cone means a SMALLER cosine than critical.
            const double inside = muC - offset;
            if (inside > 0.0) {
                ++rowsChecked;
                const float f = pathtracer::scene::fresnelDielectric(static_cast<float>(inside),
                                                                  static_cast<float>(ior), 1.0F);
                if (!(f == 1.0F)) {
                    std::cerr << "bsdf_validate: FAILED TIR onset at ior=" << ior << " cos=" << inside
                              << " (critical " << muC << ", inside the cone by " << offset << ") -- reflectance "
                              << f << ", must be exactly 1.0: past the critical angle cosThetaT is zero and both "
                                 "polarisation terms are exactly one\n";
                    ok = false;
                }
            }
            const double outside = muC + offset;
            if (outside <= 1.0) {
                ++rowsChecked;
                const float f = pathtracer::scene::fresnelDielectric(static_cast<float>(outside),
                                                                  static_cast<float>(ior), 1.0F);
                worstOutside = std::max(worstOutside, f);
                if (!(f < 1.0F)) {
                    std::cerr << "bsdf_validate: FAILED TIR onset at ior=" << ior << " cos=" << outside
                              << " (critical " << muC << ", outside the cone by " << offset << ") -- reflectance "
                              << f << ", must be below 1.0: this direction refracts, so reporting total internal "
                                 "reflection deletes the transmitted energy entirely\n";
                    ok = false;
                }
            }
        }
    }
    // A sweep that checked nothing would pass vacuously, and the bracket above is conditional on both sides.
    if (rowsChecked == 0) {
        std::cerr << "bsdf_validate: FAILED TIR onset -- no rows checked\n";
        ok = false;
    }
    std::cout << "  rows " << rowsChecked << ", worst reflectance just outside the cone " << worstOutside << '\n';
    finish(ctx, ok, "critical_angle_onset failed; see the rows above");
    return;
}

// The two macro-normal TIR sites share one cos2Transmitted predicate; not asserted on the rough branch, which decides per microfacet.
PT_CHECK(tir_predicate_agreement, Fast, Exact) {
    // Enough draws that reachability is not one lucky selection: just outside the cone ~26% transmit (1068/4096 at ior 1.33).
    constexpr int kDraws = 4096;
    // Angular resolution is set by the finest offset below: a divergence moving the critical cosine less would fall between rows.
    constexpr float kSmoothRoughness = 0.0F;  // the smooth interface: the delta branch
    const std::array<double, 3> iors = {1.33, 1.5, 2.4};
    const std::array<double, 4> offsets = {1e-3, 1e-2, 0.1, 0.25};

    bool ok = true;
    int rowsChecked = 0;
    std::cout << "bsdf_validate: TIR predicate agreement, fresnelDielectric vs the smooth refraction branch\n";
    for (double ior : iors) {
        const double muC = criticalCosine(ior);
        for (double offset : offsets) {
            for (const bool insideCone : {true, false}) {
                const double mu = insideCone ? muC - offset : muC + offset;
                if (mu <= 0.0 || mu > 1.0) {
                    continue;
                }
                ++rowsChecked;
                const float muF = static_cast<float>(mu);
                const bool fresnelSaysTir =
                    pathtracer::scene::fresnelDielectric(muF, static_cast<float>(ior), 1.0F) == 1.0F;

                // Exiting side: woLocal.z < 0 makes the dense medium incident, the only orientation where a critical angle exists.
                const float sine = std::sqrt(std::max(0.0F, 1.0F - (muF * muF)));
                const glm::vec3 wo(sine, 0.0F, -muF);
                // ior from the row, not makeParams' fixed 1.5: the critical angle above derives from this same value.
                Surface params = makeParams(kSmoothRoughness, 0.0F, 1.0F);
                params.specularIor = static_cast<float>(ior);

                int transmitted = 0;
                for (int i = 0; i < kDraws; ++i) {
                    pathtracer::scene::Sampler sampler(0, 0, i, kDraws, 5100U);
                    const std::optional<pathtracer::scene::BsdfSample> sample =
                        pathtracer::scene::sampleBsdf(params, wo, sampler);
                    transmitted +=
                        sample.has_value() && sample->type == pathtracer::scene::LobeType::Transmission ? 1 : 0;
                }

                // Stated as the agreement itself rather than two independent thresholds, so the check cannot pass by both being wrong.
                if (!(fresnelSaysTir == (transmitted == 0))) {
                    std::cerr << "bsdf_validate: FAILED TIR agreement at ior=" << ior << " cos=" << mu
                              << " (critical " << muC << ") -- fresnelDielectric "
                              << (fresnelSaysTir ? "reports" : "does not report")
                              << " total internal reflection, but the smooth branch transmitted " << transmitted
                              << " of " << kDraws << " draws; the two sites decide from the same cos2Transmitted and "
                                 "must agree\n";
                    ok = false;
                }
                // Inside the cone the count must be exactly zero: a delta lobe has one Snell direction and there it does not exist.
                if (fresnelSaysTir && !(transmitted == 0)) {
                    std::cerr << "bsdf_validate: FAILED TIR agreement at ior=" << ior << " cos=" << mu
                              << " -- " << transmitted << " transmitted draws inside the critical cone\n";
                    ok = false;
                }
            }
        }
    }
    if (rowsChecked == 0) {
        std::cerr << "bsdf_validate: FAILED TIR agreement -- no rows checked\n";
        ok = false;
    }
    std::cout << "  rows " << rowsChecked << '\n';
    finish(ctx, ok, "tir_predicate_agreement failed; see the rows above");
    return;
}

}  // namespace

// --- Pearson chi-square over equal-solid-angle bins between sampleBsdf's draws and pdfBsdf, rejected draws their own cell; iid scrambles.
struct ChiSquareCase {
    float roughness;
    float metallic;
    float transmission;
    float ndotV;
    float coat = 0.0F;
    float coatRoughness = 0.0F;
    float fuzz = 0.0F;
    float fuzzRoughness = 0.5F;
    float filmThickness = 0.0F;  // micrometres; a film of weight 1 where positive
    float anisotropy = 0.0F;     // specular_roughness_anisotropy
    float coatAnisotropy = 0.0F;
    float woAzimuth = 0.0F;      // wo's azimuth from the tangent, which only an anisotropic lobe distinguishes
};

PT_CHECK(sampling_chi_square, Slow, Statistical) {
    constexpr int kCosBins = 16;
    constexpr int kPhiBins = 8;
    // Even, for Simpson, per axis per bin. Measured: the peaked refraction lobe at roughness 0.2 still fails at 64 panels, passes from 96.
    constexpr int kPanels = 256;
    // Sized for power at the suite's family-wise rate: 200000 draws scaled by the noncentrality ratio, so power is unchanged.
    constexpr int kSampleCount = 280000;
    // Cochran's rule: the count below which a cell's chi-square term is not trustworthy and must be pooled.
    constexpr double kMinExpected = 5.0;
    constexpr std::uint32_t kSeed = 0x9E3779B9U;

    // Transmissive rows sit either side of the interface; coated, fuzzed and filmed rows put each layer's lobe and masses in the draw.
    const std::array<ChiSquareCase, 24> cases = {{
        {0.2F, 0.0F, 1.0F, 0.8F},  {0.2F, 0.0F, 1.0F, -0.6F},
        {0.4F, 0.0F, 1.0F, 0.8F},  {0.4F, 0.0F, 1.0F, -0.6F},
        {0.7F, 0.0F, 1.0F, 0.8F},  {0.7F, 0.0F, 1.0F, -0.6F},
        {1.0F, 0.0F, 1.0F, 0.8F},  {1.0F, 0.0F, 1.0F, -0.6F},
        {0.3F, 1.0F, 0.0F, 0.7F},  {0.8F, 1.0F, 0.0F, 0.7F},
        {0.5F, 0.0F, 0.0F, 0.5F},  {1.0F, 0.0F, 0.0F, 0.5F},
        {0.5F, 0.0F, 0.0F, 0.6F, 1.0F, 0.3F}, {0.3F, 1.0F, 0.0F, 0.7F, 0.5F, 0.6F}, {0.4F, 0.0F, 1.0F, 0.8F, 1.0F, 0.2F},
        {0.5F, 0.0F, 0.0F, 0.6F, 0.0F, 0.0F, 1.0F, 0.5F}, {0.3F, 1.0F, 0.0F, 0.4F, 0.5F, 0.3F, 0.7F, 0.8F},
        {0.4F, 0.0F, 1.0F, 0.7F, 0.0F, 0.0F, 0.0F, 0.5F, 0.5F}, {0.3F, 1.0F, 0.0F, 0.6F, 0.0F, 0.0F, 0.0F, 0.5F, 0.4F},
        {0.6F, 1.0F, 0.0F, 0.7F, 0.0F, 0.0F, 0.0F, 0.5F, 0.0F, 0.8F, 0.0F, 0.6F}, {0.5F, 0.0F, 0.0F, 0.6F, 0.0F, 0.0F, 0.0F, 0.5F, 0.0F, 0.9F, 0.0F, 1.0F},
        {0.6F, 0.0F, 1.0F, 0.8F, 0.0F, 0.0F, 0.0F, 0.5F, 0.0F, 0.6F, 0.0F, 0.6F}, {0.6F, 0.0F, 1.0F, -0.6F, 0.0F, 0.0F, 0.0F, 0.5F, 0.0F, 0.6F, 0.0F, 0.6F},
        {0.5F, 1.0F, 0.0F, 0.7F, 1.0F, 0.5F, 0.0F, 0.5F, 0.0F, 0.0F, 0.8F, 0.6F},
    }};
    // The suite's corrected significance, split across the grid by Sidak so the independent cases share it.
    ctx.plan(1);
    const double perCase = tools::stats::sidak(ctx.alpha(), static_cast<int>(cases.size()));

    bool ok = true;
    double worstP = 1.0;
    for (const ChiSquareCase& testCase : cases) {
        Surface params = makeParams(testCase.roughness, testCase.metallic, testCase.transmission);
        params.coatWeight = testCase.coat;
        params.coatRoughness = testCase.coatRoughness;
        params.fuzzWeight = testCase.fuzz;
        params.fuzzRoughness = testCase.fuzzRoughness;
        params = filmed(params, testCase.filmThickness > 0.0F ? 1.0F : 0.0F, testCase.filmThickness, 1.33F);
        params.specularRoughnessAnisotropy = testCase.anisotropy;
        params.coatRoughnessAnisotropy = testCase.coatAnisotropy;
        const float sinV = std::sqrt(std::max(0.0F, 1.0F - (testCase.ndotV * testCase.ndotV)));
        const glm::vec3 wo(sinV * std::cos(testCase.woAzimuth), sinV * std::sin(testCase.woAzimuth), testCase.ndotV);

        std::vector<double> expected(static_cast<std::size_t>(kCosBins) * kPhiBins + 1, 0.0);
        double mass = 0.0;
        for (int ci = 0; ci < kCosBins; ++ci) {
            const double c0 = -1.0 + (2.0 * ci / kCosBins);
            const double c1 = -1.0 + (2.0 * (ci + 1.0) / kCosBins);
            for (int pi = 0; pi < kPhiBins; ++pi) {
                const double p0 = 2.0 * kPiDouble * pi / kPhiBins;
                const double p1 = 2.0 * kPiDouble * (pi + 1.0) / kPhiBins;
                const double integral = simpson(c0, c1, kPanels, [&](double cosTheta) {
                    const double sinTheta = std::sqrt(std::max(0.0, 1.0 - (cosTheta * cosTheta)));
                    return simpson(p0, p1, kPanels, [&](double phi) {
                        const glm::vec3 wi(static_cast<float>(sinTheta * std::cos(phi)),
                                            static_cast<float>(sinTheta * std::sin(phi)),
                                            static_cast<float>(cosTheta));
                        return static_cast<double>(pathtracer::scene::pdfBsdf(params, wo, wi));
                    });
                });
                expected[static_cast<std::size_t>((ci * kPhiBins) + pi)] = integral * kSampleCount;
                mass += integral;
            }
        }
        expected.back() = std::max(1.0 - mass, 0.0) * kSampleCount;

        std::vector<double> observed(expected.size(), 0.0);
        std::mt19937 rng(kSeed);
        for (int i = 0; i < kSampleCount; ++i) {
            pathtracer::scene::Sampler sampler(0, 0, 0, 1, rng());
            const std::optional<pathtracer::scene::BsdfSample> sample = pathtracer::scene::sampleBsdf(params, wo, sampler);
            if (!sample.has_value() || sample->pdf <= 0.0F) {
                observed.back() += 1.0;
                continue;
            }
            const glm::vec3 wi = sample->wiLocal;
            const int ci = std::min(static_cast<int>((wi.z + 1.0F) * 0.5F * kCosBins), kCosBins - 1);
            float phi = std::atan2(wi.y, wi.x);
            if (phi < 0.0F) { phi += static_cast<float>(2.0 * kPiDouble); }
            const int pi = std::min(static_cast<int>(phi / static_cast<float>(2.0 * kPiDouble) * kPhiBins), kPhiBins - 1);
            observed[static_cast<std::size_t>((ci * kPhiBins) + pi)] += 1.0;
        }

        // Cells too sparse to trust individually are merged into one, which keeps the statistic chi-square distributed, not merely shaped.
        double chiSquare = 0.0;
        int cells = 0;
        double pooledExpected = 0.0;
        double pooledObserved = 0.0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (expected[i] < kMinExpected) {
                pooledExpected += expected[i];
                pooledObserved += observed[i];
                continue;
            }
            const double delta = observed[i] - expected[i];
            chiSquare += (delta * delta) / expected[i];
            ++cells;
        }
        if (pooledExpected >= kMinExpected) {
            const double delta = pooledObserved - pooledExpected;
            chiSquare += (delta * delta) / pooledExpected;
            ++cells;
        }
        const int dof = cells - 1;
        const double p = tools::stats::chiSquareUpperTail(chiSquare, dof);
        worstP = std::min(worstP, p);
        if (p >= perCase) {
            continue;
        }
        std::cerr << "bsdf_validate: FAILED sampling chi-square at roughness=" << testCase.roughness
                  << " metallic=" << testCase.metallic << " transmission=" << testCase.transmission << " coat=" << testCase.coat << " fuzz=" << testCase.fuzz
                  << " anisotropy=" << testCase.anisotropy << " coatAnisotropy=" << testCase.coatAnisotropy
                  << " ndotV=" << testCase.ndotV << " chi2=" << chiSquare << " dof=" << dof << " p=" << p
                  << " (threshold " << perCase << ", sampled mass " << mass << ")\n";
        ok = false;
    }
    std::cout << "bsdf_validate: sampling chi-square over " << cases.size() << " configurations, "
              << kSampleCount << " draws each, worst p-value " << worstP << " against a Sidak threshold of "
              << perCase << "\n";
    finish(ctx, ok, "sampling_chi_square failed; see the rows above");
    return;
}

// Roughness 0 reflects as a delta (OpenPBR's smooth limit): every reflection draw is the mirror at pdf 0, and there is no continuous value.
PT_CHECK(smooth_reflection_is_a_delta, Fast, Exact) {
    constexpr int kDraws = 256;
    const std::array<float, 3> ndotVs = {1.0F, 0.6F, 0.2F};
    struct SmoothCase {
        const char* name;
        Surface params;
        bool onlyStrategy;  // a lone strategy is drawn with probability 1, so its throughput is the Fresnel itself
    };
    const std::array<SmoothCase, 4> cases{{
        {"white metal", makeParams(0.0F, 1.0F, 0.0F), true},
        {"tinted metal", makeColoredMetalParams(0.0F, glm::vec3(0.9F, 0.6F, 0.3F)), true},
        {"glossy dielectric", makeParams(0.0F, 0.0F, 0.0F), false},
        {"glass", makeParams(0.0F, 0.0F, 1.0F), false},
    }};
    bool ok = true;
    int reflections = 0;
    for (const SmoothCase& testCase : cases) {
        for (float ndotV : ndotVs) {
            const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);
            const glm::vec3 mirror(-wo.x, -wo.y, wo.z);
            const glm::vec3 fresnel = viewFresnel(testCase.params, ndotV);
            if (!(maxChannel(pathtracer::scene::evaluateBsdfSplit(testCase.params, wo, mirror).specular) == 0.0F)) {
                std::cerr << "bsdf_validate: FAILED " << testCase.name << " at ndotV=" << ndotV << " has a continuous mirror value\n";
                ok = false;
            }
            for (int i = 0; i < kDraws; ++i) {
                pathtracer::scene::Sampler sampler(0, 0, i, kDraws, 4242U);
                const std::optional<pathtracer::scene::BsdfSample> sample = pathtracer::scene::sampleBsdf(testCase.params, wo, sampler);
                if (!sample.has_value() || sample->type != pathtracer::scene::LobeType::SpecularReflection) {
                    continue;
                }
                ++reflections;
                const bool exactFresnel = !testCase.onlyStrategy || sample->throughputWeight == fresnel;
                if (!sample->delta || sample->pdf != 0.0F || sample->wiLocal != mirror || !exactFresnel) {
                    std::cerr << "bsdf_validate: FAILED " << testCase.name << " at ndotV=" << ndotV << ": delta " << sample->delta << " pdf "
                              << sample->pdf << " throughput " << sample->throughputWeight.x << " vs F " << fresnel.x << '\n';
                    ok = false;
                    break;
                }
            }
        }
    }
    if (reflections == 0) {
        std::cerr << "bsdf_validate: FAILED smooth reflection -- no reflection was drawn, so nothing was asserted\n";
        ok = false;
    }
    finish(ctx, ok, "smooth_reflection_is_a_delta failed; see the rows above");
}

// fresnelAtMicrofacet, the Fresnel AOV's estimator: the half-vector must come from sampleBsdf's VNDF, and E[F(wo.wh)] varies with alpha.
PT_CHECK(microfacet_fresnel, Slow, Statistical) {
    ctx.plan(3);
    constexpr int kDraws = 200000;
    constexpr std::uint32_t kSeed = 7919U;
    // Expectation of F over D_vis at this roughness and view angle, by the same draws the renderer makes.
    const auto expectation = [](const Surface& params, float ndotV) {
        const pathtracer::scene::BsdfClosure closure =
            pathtracer::scene::makeBsdfClosure(params, glm::vec3(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV));
        glm::dvec3 sum(0.0);
        for (int i = 0; i < kDraws; ++i) {
            pathtracer::scene::Sampler sampler(0, 0, i, kDraws, kSeed);
            sum += glm::dvec3(pathtracer::scene::fresnelAtMicrofacet(closure, sampler.next2D()));
        }
        return glm::vec3(sum / static_cast<double>(kDraws));
    };

    // Roughness 0, the smooth surface: its visible normals are the macro normal, so E[F] is the macro Fresnel exactly.
    bool smoothOk = true;
    float worstSmooth = 0.0F;
    for (const float ndotV : {0.1F, 0.4F, 0.7F, 1.0F}) {
        for (const float metallic : {0.0F, 1.0F}) {
            const Surface params = makeParams(0.0F, metallic, 0.0F);
            const glm::vec3 mean = expectation(params, ndotV);
            const glm::vec3 macro = viewFresnel(params, ndotV);
            for (int c = 0; c < 3; ++c) {
                worstSmooth = std::max(worstSmooth, std::abs(mean[c] - macro[c]));
            }
        }
    }
    // Exact: every draw returns the macro Fresnel itself, and a double sum of identical floats divides back to it.
    smoothOk = worstSmooth == 0.0F;
    char smoothDetail[192];
    std::snprintf(smoothDetail, sizeof(smoothDetail), "worst |E[F(wo.wh)] - F(n.wo)| on the smooth surface is %.3e, which must be 0",
                  static_cast<double>(worstSmooth));
    PT_EXPECT(ctx, smoothOk, smoothDetail);

    // Grazing, where the macro ramp is steepest and the lobe average therefore departs from it most.
    constexpr float kGrazing = 0.1F;
    const Surface rough = makeParams(0.6F, 0.0F, 0.0F);
    const glm::vec3 roughMean = expectation(rough, kGrazing);
    const glm::vec3 roughMacro = viewFresnel(rough, kGrazing);
    char flatDetail[192];
    std::snprintf(flatDetail, sizeof(flatDetail),
                  "at roughness 0.6, grazing: E[F] = %.4f against a macro F of %.4f, which it must fall below",
                  static_cast<double>(roughMean.x), static_cast<double>(roughMacro.x));
    PT_EXPECT(ctx, roughMean.x < roughMacro.x, flatDetail);

    // A coloured metal, the reason the lane carries RGB: F82's specular_color tints per channel, so the mean is chromatic.
    const Surface tinted = makeColoredMetalParams(0.2F, glm::vec3(0.9F, 0.6F, 0.3F));
    const glm::vec3 tintedMean = expectation(tinted, kGrazing);
    const float spread = std::max({tintedMean.x, tintedMean.y, tintedMean.z}) -
                          std::min({tintedMean.x, tintedMean.y, tintedMean.z});
    char chromaDetail[192];
    std::snprintf(chromaDetail, sizeof(chromaDetail),
                  "F82-tinted metal spans %.4f across RGB (%.4f, %.4f, %.4f); a greyscale lane would report one",
                  static_cast<double>(spread), static_cast<double>(tintedMean.x),
                  static_cast<double>(tintedMean.y), static_cast<double>(tintedMean.z));
    PT_EXPECT(ctx, spread > 1e-3F, chromaDetail);
}


// --- OpenPBR's coat: a dielectric slab over the whole base, absorbing by coat_color and darkening it.
Surface coated(Surface inputs, float weight, const glm::vec3& color, float roughness, float ior, float darkening) {
    inputs.coatWeight = weight;
    inputs.coatColor = color;
    inputs.coatRoughness = roughness;
    inputs.coatIor = ior;
    inputs.coatDarkening = darkening;
    return inputs;
}

// Non-coplanar (wo, wi) at two cosines, checkReciprocity's construction: a shared azimuth would leave a swapped-phi bug invisible.
std::pair<glm::vec3, glm::vec3> pairAt(float muO, float muI) {
    const float sinO = std::sqrt(std::max(0.0F, 1.0F - (muO * muO)));
    const float sinI = std::sqrt(std::max(0.0F, 1.0F - (muI * muI)));
    return {glm::vec3(sinO, 0.0F, muO), glm::vec3(sinI * std::cos(1.1F), sinI * std::sin(1.1F), muI)};
}

// coat_weight 0 is no coat whatever its other inputs: every value, density and draw bit-identical to the uncoated surface.
PT_CHECK(coat_weight_zero_is_identity, Fast, Exact) {
    constexpr int kDraws = 64;
    const std::array<Surface, 4> bases = {makeParams(0.4F, 0.0F, 0.0F), makeParams(0.0F, 1.0F, 0.0F), makeParams(0.3F, 0.0F, 1.0F),
                                          makeColoredMetalParams(0.6F)};
    const std::array<float, 4> cosines = {1.0F, 0.6F, 0.2F, -0.6F};
    bool ok = true;
    for (const Surface& base : bases) {
        const Surface inert = coated(base, 0.0F, glm::vec3(0.3F, 0.6F, 0.9F), 0.5F, 2.0F, 1.0F);
        for (float muO : cosines) {
            for (float muI : cosines) {
                const auto [wo, wi] = pairAt(muO, muI);
                const pathtracer::scene::BsdfEval a = pathtracer::scene::evaluateBsdfSplit(base, wo, wi);
                const pathtracer::scene::BsdfEval b = pathtracer::scene::evaluateBsdfSplit(inert, wo, wi);
                ok = ok && a.total() == b.total() && a.pdf == b.pdf;
            }
            const auto [wo, unused] = pairAt(muO, 1.0F);
            for (int i = 0; i < kDraws; ++i) {
                pathtracer::scene::Sampler first(0, 0, i, kDraws, 77U);
                pathtracer::scene::Sampler second(0, 0, i, kDraws, 77U);
                const auto a = pathtracer::scene::sampleBsdf(base, wo, first);
                const auto b = pathtracer::scene::sampleBsdf(inert, wo, second);
                ok = ok && a.has_value() == b.has_value() &&
                     (!a || (a->wiLocal == b->wiLocal && a->throughputWeight == b->throughputWeight && a->pdf == b->pdf));
            }
        }
    }
    finish(ctx, ok, "coat_weight 0 changed a value, density or draw: an absent coat must not reach the base");
}

// An index-matched coat (coat_ior 1) reflects and darkens nothing: the base under it is the base times (1-C) + C T(mu_o) T(mu_i).
PT_CHECK(index_matched_coat_is_absorption, Fast, Exact) {
    // Base roughening at coat_roughness 0 is lerp(r, (r^4)^(1/4), C): r up to the two square roots' rounding.
    constexpr float kRelativeTolerance = 1e-5F;
    const glm::vec3 color(0.3F, 0.6F, 0.9F);
    const std::array<Surface, 2> bases = {makeParams(0.4F, 0.0F, 0.0F), makeColoredMetalParams(0.5F)};
    const std::array<float, 3> cosines = {1.0F, 0.6F, 0.25F};
    bool ok = true;
    for (const Surface& base : bases) {
        for (float weight : {0.5F, 1.0F}) {
            const Surface coat = coated(base, weight, color, 0.0F, 1.0F, 1.0F);
            for (float muO : cosines) {
                for (float muI : cosines) {
                    const auto [wo, wi] = pairAt(muO, muI);
                    const glm::vec3 plain = pathtracer::scene::evaluateBsdf(base, wo, wi);
                    const glm::vec3 measured = pathtracer::scene::evaluateBsdf(coat, wo, wi);
                    const glm::vec3 absorbed = glm::vec3(1.0F - weight) + (weight * glm::pow(color, glm::vec3((0.5F / muO) + (0.5F / muI))));
                    const glm::vec3 expected = plain * absorbed;
                    if (!(maxChannel(glm::abs(measured - expected)) <= kRelativeTolerance * maxChannel(expected))) {
                        std::cerr << "bsdf_validate: FAILED index-matched coat at C=" << weight << " muO=" << muO << " muI=" << muI << " f "
                                  << measured.x << " vs " << expected.x << '\n';
                        ok = false;
                    }
                }
            }
        }
    }
    finish(ctx, ok, "index_matched_coat_is_absorption failed; see the rows above");
}

// Over a base that reflects all, a clear coat conserves energy exactly: E_coat + (1 - E_coat) * 1, darkening being 1 at E_b = 1.
PT_CHECK(coated_white_furnace, Slow, Statistical) {
    constexpr int kSampleCount = 200000;
    constexpr float kTolerance = 0.02F;
    const std::array<Surface, 3> bases = {makeParams(0.2F, 1.0F, 0.0F), makeParams(0.6F, 1.0F, 0.0F), makeParams(0.5F, 0.0F, 0.0F)};
    const std::array<float, 3> coatRoughnesses = {0.0F, 0.3F, 0.7F};
    const std::array<float, 3> ndotVs = {1.0F, 0.6F, 0.25F};
    bool ok = true;
    std::uint32_t seed = 61000;
    std::cout << "bsdf_validate: clear coat over white bases (1.0 = energy conserved)\n";
    for (const Surface& base : bases) {
        for (float weight : {0.5F, 1.0F}) {
            for (float coatRoughness : coatRoughnesses) {
                for (float ndotV : ndotVs) {
                    const Surface coat = coated(base, weight, glm::vec3(1.0F), coatRoughness, 1.5F, 1.0F);
                    const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);
                    const glm::vec3 lo = furnaceLo(coat, wo, kSampleCount, ++seed);
                    if (!withinBand(lo, 1.0F, kTolerance)) {
                        std::cerr << "bsdf_validate: FAILED coated furnace at metal=" << base.baseMetalness << " C=" << weight
                                  << " coat_roughness=" << coatRoughness << " ndotV=" << ndotV << " Lo=" << minChannel(lo) << '\n';
                        ok = false;
                    }
                }
            }
        }
    }
    finish(ctx, ok, "coated_white_furnace failed; see the rows above");
}

// Over a base that reflects nothing, the coat alone remains: a GGX interface whose multiple scattering is Kulla-Conty's symmetric form.
PT_CHECK(coat_reciprocity, Fast, Exact) {
    constexpr float kRelativeTolerance = 1e-4F;
    const Surface black = paramsWith([](Surface& inputs) {
        inputs.baseWeight = 0.0F;
        inputs.specularWeight = 0.0F;
    });
    const std::array<float, 4> cosines = {1.0F, 0.7F, 0.4F, 0.15F};
    bool ok = true;
    int rows = 0;
    for (float coatRoughness : {0.1F, 0.3F, 0.6F, 1.0F}) {
        for (float ior : {1.3F, 1.6F, 2.0F}) {
            const Surface coat = coated(black, 0.7F, glm::vec3(1.0F), coatRoughness, ior, 1.0F);
            for (float muA : cosines) {
                for (float muB : cosines) {
                    const auto [wo, wi] = pairAt(muA, muB);
                    const glm::vec3 f = pathtracer::scene::evaluateBsdfSplit(coat, wo, wi).specular / muB;
                    const glm::vec3 g = pathtracer::scene::evaluateBsdfSplit(coat, wi, wo).specular / muA;
                    const float scale = std::max(maxChannel(f), maxChannel(g));
                    rows += scale > 0.0F ? 1 : 0;
                    if (!(maxChannel(glm::abs(f - g)) <= kRelativeTolerance * std::max(scale, 1e-4F))) {
                        std::cerr << "bsdf_validate: FAILED coat reciprocity at coat_roughness=" << coatRoughness << " ior=" << ior
                                  << " muO=" << muA << " muI=" << muB << " f=" << f.x << " vs " << g.x << '\n';
                        ok = false;
                    }
                }
            }
        }
    }
    // Anti-vacuity: a coat reflecting nothing would pass every pair at zero.
    finish(ctx, ok && rows > 0, "coat_reciprocity failed; see the rows above");
}

// A tilted geometry_coat_normal: every draw's density is pdfBsdf's exactly, and the mixture integrates to at most one.
PT_CHECK(tilted_coat_normal_sampling, Slow, Statistical) {
    constexpr int kDraws = 20000;
    constexpr int kUniform = 50000;
    constexpr float kTolerance = 0.05F;
    const std::array<glm::vec3, 2> normals = {glm::normalize(glm::vec3(0.0F, std::sin(0.3F), std::cos(0.3F))),
                                              glm::normalize(glm::vec3(0.25F, 0.25F, 1.0F))};
    const std::array<Surface, 2> bases = {makeParams(0.4F, 0.0F, 0.0F), makeColoredMetalParams(0.3F)};
    std::mt19937 rng(97);
    bool ok = true;
    long long compared = 0;
    double worstIntegral = 0.0;
    for (const glm::vec3& normal : normals) {
        for (const Surface& base : bases) {
            for (float ndotV : {0.9F, 0.5F}) {
                const Surface coat = coated(base, 0.8F, glm::vec3(0.9F, 0.8F, 0.7F), 0.3F, 1.5F, 1.0F);
                const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);
                const pathtracer::scene::BsdfClosure closure = pathtracer::scene::makeBsdfClosure(coat, wo, std::nullopt, normal);
                double integral = 0.0;
                const auto combined = [&](double p) { return (kUniform / (2.0 * kPi)) + (kDraws * p); };
                for (int i = 0; i < kDraws; ++i) {
                    pathtracer::scene::Sampler sampler(0, 0, i, kDraws, 4100U);
                    const auto sample = pathtracer::scene::sampleBsdf(closure, sampler);
                    if (!sample || sample->delta) {
                        continue;
                    }
                    ++compared;
                    ok = ok && pathtracer::scene::evaluateBsdfSplit(closure, sample->wiLocal).pdf == sample->pdf;
                    integral += sample->pdf / combined(sample->pdf);
                }
                for (int i = 0; i < kUniform; ++i) {
                    const double p = pathtracer::scene::evaluateBsdfSplit(closure, sampleUniformHemisphere(rng)).pdf;
                    integral += p / combined(p);
                }
                worstIntegral = std::max(worstIntegral, integral);
                ok = ok && integral <= 1.0 + kTolerance;
            }
        }
    }
    std::cout << "bsdf_validate: tilted coat normal, " << compared << " draws re-evaluated, worst pdf integral " << worstIntegral << '\n';
    finish(ctx, ok && compared > 0, "a tilted coat normal broke sample/pdf consistency or normalisation");
}

// OpenPBR's darkening over a smooth metal, where E_b = F0 and K = F_coat(mu_o) exactly: Delta = (1-K)/(1-F0 K), lerp(1, Delta, C delta).
PT_CHECK(coat_darkening, Fast, Exact) {
    constexpr double kRelativeTolerance = 1e-5;
    constexpr double kF0 = 0.5;
    constexpr double kCoatIor = 1.5;
    bool ok = true;
    for (float weight : {0.5F, 1.0F}) {
        for (float darkening : {0.0F, 0.5F, 1.0F}) {
            for (float mu : {1.0F, 0.6F, 0.2F}) {
                const Surface coat = coated(makeParams(0.0F, 1.0F, 0.0F), weight, glm::vec3(1.0F), 0.0F, static_cast<float>(kCoatIor), darkening);
                Surface metal = coat;
                metal.baseColor = glm::vec3(static_cast<float>(kF0));
                const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (mu * mu))), 0.0F, mu);
                const pathtracer::scene::BsdfClosure closure = pathtracer::scene::makeBsdfClosure(metal, wo);
                const double fresnel = referenceDielectricFresnel(mu, kCoatIor);
                const double delta = (1.0 - fresnel) / (1.0 - (kF0 * fresnel));
                const double factor = 1.0 + (weight * darkening * (delta - 1.0));
                const double bare = (1.0 - weight) * factor;
                const double under = weight * (1.0 - fresnel) * factor;
                const auto close = [&](double measured, double expected) {
                    return std::abs(measured - expected) <= kRelativeTolerance * std::max(std::abs(expected), 1e-6);
                };
                if (!close(closure.baseBare.x, bare) || !close(closure.baseUnder.x, under)) {
                    std::cerr << "bsdf_validate: FAILED coat darkening at C=" << weight << " delta=" << darkening << " mu=" << mu << " bare "
                              << closure.baseBare.x << " vs " << bare << ", under " << closure.baseUnder.x << " vs " << under << '\n';
                    ok = false;
                }
            }
        }
    }
    finish(ctx, ok, "coat_darkening failed; see the rows above");
}

// The coat's one-pass transmittance: a white coat is no absorber, normal incidence crosses sqrt(coat_color), grazing paths are longer.
PT_CHECK(coat_transmittance, Fast, Exact) {
    const glm::vec3 color(0.25F, 0.5F, 0.81F);
    bool ok = true;
    for (float ior : {1.0F, 1.5F, 2.5F}) {
        for (float mu : {1.0F, 0.5F, 0.1F}) {
            ok = ok && pathtracer::scene::coatTransmittance(glm::vec3(1.0F), ior, mu) == glm::vec3(1.0F);
        }
        ok = ok && maxChannel(glm::abs(pathtracer::scene::coatTransmittance(color, ior, 1.0F) - glm::sqrt(color))) <= 1e-6F;
        ok = ok && glm::all(glm::lessThan(pathtracer::scene::coatTransmittance(color, ior, 0.3F),
                                          pathtracer::scene::coatTransmittance(color, ior, 0.9F)));
    }
    finish(ctx, ok, "coat_transmittance broke an identity: white is clear, mu = 1 gives sqrt(coat_color), grazing absorbs more");
}


// --- OpenPBR's fuzz: the sheen LTC of Zeltner, Burley and Chiang 2022 over the coated base, withholding what it reflects.
Surface fuzzed(Surface inputs, float weight, const glm::vec3& color, float roughness) {
    inputs.fuzzWeight = weight;
    inputs.fuzzColor = color;
    inputs.fuzzRoughness = roughness;
    return inputs;
}

// The published table read through the same two-name shim the renderer compiles it with: the oracle is the file, not the lookup.
namespace published {
struct Vector3f {
    float aInv;
    float bInv;
    float albedo;
    constexpr Vector3f(float a, float b, float r) : aInv(a), bInv(b), albedo(r) {}
};
struct SheenLTC {
    static const Vector3f _ltcParamTableVolume[32][32];
};
#include "ltc-sheen/ltc_table_sheen_volume.cpp"
}  // namespace published

// fuzz_weight 0 is no fuzz whatever its other inputs: every value, density and draw bit-identical to the unfuzzed surface.
PT_CHECK(fuzz_weight_zero_is_identity, Fast, Exact) {
    constexpr int kDraws = 64;
    const std::array<Surface, 3> bases = {makeParams(0.4F, 0.0F, 0.0F), makeParams(0.3F, 0.0F, 1.0F), coated(makeColoredMetalParams(0.6F), 1.0F,
                                                                                                             glm::vec3(0.8F), 0.2F, 1.5F, 1.0F)};
    bool ok = true;
    for (const Surface& base : bases) {
        const Surface inert = fuzzed(base, 0.0F, glm::vec3(0.3F, 0.6F, 0.9F), 0.7F);
        for (float muO : {1.0F, 0.5F, -0.6F}) {
            for (float muI : {1.0F, 0.3F, -0.6F}) {
                const auto [wo, wi] = pairAt(muO, muI);
                const pathtracer::scene::BsdfEval a = pathtracer::scene::evaluateBsdfSplit(base, wo, wi);
                const pathtracer::scene::BsdfEval b = pathtracer::scene::evaluateBsdfSplit(inert, wo, wi);
                ok = ok && a.total() == b.total() && a.pdf == b.pdf;
            }
            const auto [wo, unused] = pairAt(muO, 1.0F);
            for (int i = 0; i < kDraws; ++i) {
                pathtracer::scene::Sampler first(0, 0, i, kDraws, 79U);
                pathtracer::scene::Sampler second(0, 0, i, kDraws, 79U);
                const auto a = pathtracer::scene::sampleBsdf(base, wo, first);
                const auto b = pathtracer::scene::sampleBsdf(inert, wo, second);
                ok = ok && a.has_value() == b.has_value() &&
                     (!a || (a->wiLocal == b->wiLocal && a->throughputWeight == b->throughputWeight && a->pdf == b->pdf));
            }
        }
    }
    finish(ctx, ok, "fuzz_weight 0 changed a value, density or draw: an absent fuzz must not reach the layers beneath");
}

// E_fuzz at every node of the published 32x32 table is the table's own entry: [alpha][cos theta], neither axis transposed.
PT_CHECK(fuzz_table_matches_published, Fast, Exact) {
    // At a node the bilinear weights are 0 and 1 up to the index's float rounding; the entries are given to 1e-5.
    constexpr float kTolerance = 1e-6F;
    float worst = 0.0F;
    for (int r = 0; r < 32; ++r) {
        for (int m = 0; m < 32; ++m) {
            const float measured = pathtracer::scene::fuzzAlbedo(static_cast<float>(r) / 31.0F, static_cast<float>(m) / 31.0F);
            worst = std::max(worst, std::abs(measured - published::SheenLTC::_ltcParamTableVolume[r][m].albedo));
        }
    }
    std::cout << "bsdf_validate: fuzz albedo against the published table, worst node error " << worst << '\n';
    finish(ctx, worst <= kTolerance, "fuzzAlbedo does not reproduce the published sheen table at its nodes");
}

// Over a base that reflects nothing the fuzz alone remains: its density integrates to one and its value to E_fuzz(mu_o).
PT_CHECK(fuzz_ltc_normalisation, Slow, Exact) {
    // Simpson over the hemisphere, panels resolving the LTC's grazing lobe at the smallest roughness swept.
    constexpr int kPanels = 256;
    constexpr double kTolerance = 2e-3;
    const Surface black = paramsWith([](Surface& inputs) {
        inputs.baseWeight = 0.0F;
        inputs.specularWeight = 0.0F;
    });
    bool ok = true;
    double worst = 0.0;
    for (float roughness : {0.2F, 0.5F, 1.0F}) {
        for (float mu : {1.0F, 0.6F, 0.2F}) {
            const Surface fuzz = fuzzed(black, 1.0F, glm::vec3(1.0F), roughness);
            const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (mu * mu))), 0.0F, mu);
            const pathtracer::scene::BsdfClosure closure = pathtracer::scene::makeBsdfClosure(fuzz, wo);
            const auto integrate = [&](auto read) {
                return simpson(0.0, 1.0, kPanels, [&](double z) {
                    const double sine = std::sqrt(std::max(0.0, 1.0 - (z * z)));
                    return simpson(0.0, 2.0 * kPiDouble, kPanels, [&](double phi) {
                        const glm::vec3 wi(static_cast<float>(sine * std::cos(phi)), static_cast<float>(sine * std::sin(phi)), static_cast<float>(z));
                        return static_cast<double>(read(pathtracer::scene::evaluateBsdfSplit(closure, wi)));
                    });
                });
            };
            const double pdf = integrate([](const pathtracer::scene::BsdfEval& eval) { return eval.pdf; });
            const double albedo = integrate([](const pathtracer::scene::BsdfEval& eval) { return eval.specular.x; });
            const double expected = pathtracer::scene::fuzzAlbedo(roughness, mu);
            worst = std::max({worst, std::abs(pdf - 1.0), std::abs(albedo - expected)});
            if (!(std::abs(pdf - 1.0) <= kTolerance && std::abs(albedo - expected) <= kTolerance)) {
                std::cerr << "bsdf_validate: FAILED fuzz LTC at roughness=" << roughness << " mu=" << mu << ": pdf integral " << pdf
                          << ", value integral " << albedo << " vs E_fuzz " << expected << '\n';
                ok = false;
            }
        }
    }
    std::cout << "bsdf_validate: fuzz LTC normalisation, worst |integral - expected| " << worst << '\n';
    finish(ctx, ok, "fuzz_ltc_normalisation failed; see the rows above");
}

// White fuzz over a base that reflects all conserves energy exactly: F E_fuzz + (1 - F E_fuzz) * 1.
PT_CHECK(fuzzed_white_furnace, Slow, Statistical) {
    constexpr int kSampleCount = 200000;
    constexpr float kTolerance = 0.02F;
    const std::array<Surface, 3> bases = {makeParams(0.5F, 0.0F, 0.0F), makeParams(0.3F, 1.0F, 0.0F),
                                          coated(makeParams(0.5F, 0.0F, 0.0F), 1.0F, glm::vec3(1.0F), 0.2F, 1.5F, 1.0F)};
    bool ok = true;
    std::uint32_t seed = 63000;
    for (const Surface& base : bases) {
        for (float weight : {0.5F, 1.0F}) {
            for (float roughness : {0.3F, 0.8F}) {
                for (float ndotV : {1.0F, 0.5F, 0.15F}) {
                    const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);
                    const glm::vec3 lo = furnaceLo(fuzzed(base, weight, glm::vec3(1.0F), roughness), wo, kSampleCount, ++seed);
                    if (!withinBand(lo, 1.0F, kTolerance)) {
                        std::cerr << "bsdf_validate: FAILED fuzzed furnace at F=" << weight << " roughness=" << roughness << " ndotV=" << ndotV
                                  << " Lo=" << minChannel(lo) << '\n';
                        ok = false;
                    }
                }
            }
        }
    }
    finish(ctx, ok, "fuzzed_white_furnace failed; see the rows above");
}


// --- OpenPBR's thin film: Belcour & Barla 2017's Airy reflectance in the base's Fresnel, its spectrum integrated in Rec.709.

// Gulbrandsen 2014's (n, k) from reflectance r and edge tint g, transcribed apart from the shading path's for the reference below.
std::complex<double> referenceConductorIor(double r, double g) {
    const double n = (g * (1.0 - r) / (1.0 + r)) + ((1.0 - g) * (1.0 + std::sqrt(r)) / (1.0 - std::sqrt(r)));
    return {n, std::sqrt(std::max(((r * (n + 1.0) * (n + 1.0)) - ((n - 1.0) * (n - 1.0))) / (1.0 - r), 0.0))};
}

// The exact film: per wavelength the all-order Airy amplitude (r12 + r23 e^{i d}) / (1 + r12 r23 e^{i d}), integrated by CIE 015 in D65.
glm::dvec3 referenceFilm(double cosTheta, double filmIor, double thicknessNm, const std::array<std::complex<double>, 3>& substrate) {
    using complex = std::complex<double>;
    const double sin2 = 1.0 - (cosTheta * cosTheta);
    const complex cosFilm = std::sqrt(complex(1.0 - (sin2 / (filmIor * filmIor))));
    glm::dvec3 rec709(0.0);
    for (int c = 0; c < 3; ++c) {
        const complex n3 = substrate[static_cast<std::size_t>(c)];
        const complex cos3 = std::sqrt(1.0 - (sin2 / (n3 * n3)));
        const auto amplitudes = [](complex na, complex ca, complex nb, complex cb) {
            return std::pair{((na * ca) - (nb * cb)) / ((na * ca) + (nb * cb)), ((nb * ca) - (na * cb)) / ((nb * ca) + (na * cb))};
        };
        const auto [r12s, r12p] = amplitudes(1.0, cosTheta, filmIor, cosFilm);
        const auto [r23s, r23p] = amplitudes(filmIor, cosFilm, n3, cos3);
        pathtracer::scene::cie::Spectrum reflectance{};
        for (int i = 0; i < pathtracer::scene::cie::kSampleCount; ++i) {
            const complex phase = std::exp(complex(0.0, 4.0 * kPiDouble * filmIor * thicknessNm / pathtracer::scene::cie::wavelengthNm(i)) * cosFilm);
            const complex rs = (r12s + (r23s * phase)) / (1.0 + (r12s * r23s * phase));
            const complex rp = (r12p + (r23p * phase)) / (1.0 + (r12p * r23p * phase));
            reflectance[static_cast<std::size_t>(i)] = 0.5 * (std::norm(rs) + std::norm(rp));
        }
        rec709[c] = pathtracer::scene::cie::reflectanceToRec709(reflectance)[c];
    }
    return rec709;
}

// The film's reflectance against the exact spectral reference: the converged series and the tabulated transforms against every order.
PT_CHECK(thin_film_against_spectral_reference, Fast, Exact) {
    // The series converges below float resolution; the 2 nm transform table's interpolation is what remains: measured 3.3e-6 and 1.7e-5.
    constexpr double kDielectricTolerance = 1e-4;
    constexpr double kConductorTolerance = 1e-4;
    bool ok = true;
    double worstDielectric = 0.0;
    double worstConductor = 0.0;
    for (double filmIor : {1.33, 1.8}) {
        for (double thickness : {0.1, 0.3, 0.5, 1.0}) {
            for (float cosine : {1.0F, 0.7F, 0.3F}) {
                const Surface glass = filmed(makeParams(0.0F, 0.0F, 0.0F), 1.0F, static_cast<float>(thickness), static_cast<float>(filmIor));
                const glm::dvec3 measured(viewFresnel(glass, cosine));
                const glm::dvec3 expected = referenceFilm(cosine, filmIor, thickness * 1000.0, {1.5, 1.5, 1.5});
                const double error = maxChannel(glm::vec3(glm::abs(measured - expected)));
                worstDielectric = std::max(worstDielectric, error);
                const glm::vec3 r(0.9F, 0.6F, 0.3F);
                const glm::vec3 g(0.8F, 0.5F, 0.9F);
                Surface metal = filmed(makeParams(0.0F, 1.0F, 0.0F), 1.0F, static_cast<float>(thickness), static_cast<float>(filmIor));
                metal.baseColor = r;
                metal.specularColor = g;
                const glm::dvec3 metallic(viewFresnel(metal, cosine));
                const glm::dvec3 reference = referenceFilm(cosine, filmIor, thickness * 1000.0,
                                                           {referenceConductorIor(r.x, g.x), referenceConductorIor(r.y, g.y), referenceConductorIor(r.z, g.z)});
                worstConductor = std::max(worstConductor, static_cast<double>(maxChannel(glm::vec3(glm::abs(metallic - reference)))));
                if (!(error <= kDielectricTolerance) || !(maxChannel(glm::vec3(glm::abs(metallic - reference))) <= kConductorTolerance)) {
                    std::cerr << "bsdf_validate: FAILED thin film at n=" << filmIor << " d=" << thickness << "um cos=" << cosine << ": glass "
                              << measured.x << " vs " << expected.x << ", metal " << metallic.x << " vs " << reference.x << '\n';
                    ok = false;
                }
            }
        }
    }
    std::cout << "bsdf_validate: thin film vs spectral reference, worst over glass " << worstDielectric << ", over metal " << worstConductor << '\n';
    finish(ctx, ok, "thin_film_against_spectral_reference failed; see the rows above");
}

// thin_film_weight 0, and a film of zero thickness, are no film: every value, density and draw bit-identical to the bare base.
PT_CHECK(thin_film_identity, Fast, Exact) {
    constexpr int kDraws = 64;
    const std::array<Surface, 3> bases = {makeParams(0.4F, 0.0F, 0.0F), makeParams(0.3F, 1.0F, 0.0F), makeParams(0.2F, 0.0F, 1.0F)};
    bool ok = true;
    for (const Surface& base : bases) {
        for (const Surface& inert : {filmed(base, 0.0F, 0.5F, 1.8F), filmed(base, 1.0F, 0.0F, 1.8F)}) {
            for (float muO : {1.0F, 0.5F, -0.6F}) {
                for (float muI : {1.0F, 0.3F, -0.6F}) {
                    const auto [wo, wi] = pairAt(muO, muI);
                    const pathtracer::scene::BsdfEval a = pathtracer::scene::evaluateBsdfSplit(base, wo, wi);
                    const pathtracer::scene::BsdfEval b = pathtracer::scene::evaluateBsdfSplit(inert, wo, wi);
                    ok = ok && a.total() == b.total() && a.pdf == b.pdf;
                }
                const auto [wo, unused] = pairAt(muO, 1.0F);
                for (int i = 0; i < kDraws; ++i) {
                    pathtracer::scene::Sampler first(0, 0, i, kDraws, 81U);
                    pathtracer::scene::Sampler second(0, 0, i, kDraws, 81U);
                    const auto a = pathtracer::scene::sampleBsdf(base, wo, first);
                    const auto b = pathtracer::scene::sampleBsdf(inert, wo, second);
                    ok = ok && a.has_value() == b.has_value() &&
                         (!a || (a->wiLocal == b->wiLocal && a->throughputWeight == b->throughputWeight && a->pdf == b->pdf));
                }
            }
        }
    }
    finish(ctx, ok, "a weightless or zero-thickness film changed the base");
}

// A non-absorbing film over a perfect conductor reflects everything: the Airy series' incoherent term is 1 and its orders vanish.
PT_CHECK(thin_film_over_perfect_conductor, Fast, Exact) {
    constexpr float kTolerance = 1e-6F;
    float worst = 0.0F;
    for (float thickness : {0.1F, 0.5F, 2.0F}) {
        for (float cosine : {1.0F, 0.6F, 0.2F, 0.0F}) {
            const glm::vec3 f = viewFresnel(filmed(makeParams(0.0F, 1.0F, 0.0F), 1.0F, thickness, 1.5F), cosine);
            worst = std::max(worst, maxChannel(glm::abs(f - 1.0F)));
        }
    }
    finish(ctx, worst <= kTolerance, "a film over a perfect mirror did not reflect exactly all the light");
}

// A filmed base over a white substrate conserves energy: the film only moves light between reflection and transmission.
PT_CHECK(filmed_white_furnace, Slow, Statistical) {
    constexpr int kSampleCount = 200000;
    constexpr float kTolerance = 0.02F;
    const std::array<Surface, 3> bases = {makeParams(0.4F, 0.0F, 0.0F), makeParams(0.8F, 0.0F, 0.0F), makeParams(0.3F, 1.0F, 0.0F)};
    bool ok = true;
    std::uint32_t seed = 65000;
    for (const Surface& base : bases) {
        for (float thickness : {0.3F, 0.7F}) {
            for (float ndotV : {1.0F, 0.5F, 0.2F}) {
                const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (ndotV * ndotV))), 0.0F, ndotV);
                const glm::vec3 lo = furnaceLo(filmed(base, 1.0F, thickness, 1.33F), wo, kSampleCount, ++seed);
                if (!withinBand(lo, 1.0F, kTolerance)) {
                    std::cerr << "bsdf_validate: FAILED filmed furnace at metal=" << base.baseMetalness << " roughness=" << base.specularRoughness
                              << " d=" << thickness << "um ndotV=" << ndotV << " Lo=[" << minChannel(lo) << ", " << maxChannel(lo) << "]\n";
                    ok = false;
                }
            }
        }
    }
    finish(ctx, ok, "filmed_white_furnace failed; see the rows above");
}

// The film's reflected energy, the kernel's Gauss rule over its Fresnel, is what the lobe integrates to: albedo scaling stays exact.
PT_CHECK(filmed_albedo_matches_lobe, Slow, Exact) {
    // The rule's own order-8 residual plus its bilinear blend across the 32 x 32 grid, at Simpson's resolution of the lobe.
    constexpr double kTolerance = 3e-3;
    constexpr int kPanels = 256;
    double worst = 0.0;
    for (float roughness : {0.3F, 0.6F, 1.0F}) {
        for (float thickness : {0.3F, 0.8F}) {
            for (float mu : {1.0F, 0.6F, 0.25F}) {
                const Surface surface = filmed(makeParams(roughness, 0.0F, 0.0F), 1.0F, thickness, 1.33F);
                const glm::vec3 wo(std::sqrt(std::max(0.0F, 1.0F - (mu * mu))), 0.0F, mu);
                const pathtracer::scene::BsdfClosure closure = pathtracer::scene::makeBsdfClosure(surface, wo);
                const glm::dvec3 integral = simpson(0.0, 1.0, kPanels, [&](double z) {
                    const double sine = std::sqrt(std::max(0.0, 1.0 - (z * z)));
                    return simpson(0.0, 2.0 * kPiDouble, kPanels, [&](double phi) {
                        const glm::vec3 wi(static_cast<float>(sine * std::cos(phi)), static_cast<float>(sine * std::sin(phi)), static_cast<float>(z));
                        return glm::dvec3(pathtracer::scene::evaluateBsdfSplit(closure, wi).specular);
                    });
                });
                const glm::dvec3 albedo(closure.dielectric.reflectSingle + closure.dielectric.multiReflect);
                worst = std::max(worst, static_cast<double>(maxChannel(glm::vec3(glm::abs(integral - albedo)))));
            }
        }
    }
    std::cout << "bsdf_validate: filmed albedo vs integrated lobe, worst " << worst << '\n';
    finish(ctx, worst <= kTolerance, "a filmed interface's albedo does not match the energy its lobe reflects");
}

// specular_roughness_anisotropy and coat_roughness_anisotropy over the inputs.
Surface anisotropic(Surface inputs, float anisotropy, float coatAnisotropy) {
    inputs.specularRoughnessAnisotropy = anisotropy;
    inputs.coatRoughnessAnisotropy = coatAnisotropy;
    return inputs;
}

// The direction at cosine mu and azimuth phi from the tangent.
glm::vec3 directionAt(float mu, float phi) {
    const float sine = std::sqrt(std::max(0.0F, 1.0F - (mu * mu)));
    return {sine * std::cos(phi), sine * std::sin(phi), mu};
}

// White metal, glossy-diffuse and a coat over white metal at every anisotropy regime, a = 1 the one-dimensional lobe, across azimuth.
PT_CHECK(anisotropic_white_furnace, Slow, Statistical) {
    constexpr int kSampleCount = 200000;
    constexpr float kTolerance = 0.02F;
    bool ok = true;
    std::uint32_t seed = 67000;
    for (float anisotropy : {0.25F, 0.5F, 0.9F, 1.0F}) {
        for (float roughness : {0.4F, 1.0F}) {
            const std::array<std::pair<const char*, Surface>, 3> surfaces = {{
                {"metal", anisotropic(makeParams(roughness, 1.0F, 0.0F), anisotropy, 0.0F)},
                {"glossy-diffuse", anisotropic(makeParams(roughness, 0.0F, 0.0F), anisotropy, 0.0F)},
                {"coated metal", anisotropic(coated(makeParams(0.4F, 1.0F, 0.0F), 1.0F, glm::vec3(1.0F), roughness, 1.5F, 1.0F), 0.0F, anisotropy)},
            }};
            for (const auto& [label, surface] : surfaces) {
                for (float phi : {0.0F, 0.25F * kPi, 0.5F * kPi}) {
                    for (float mu : {0.8F, 0.35F}) {
                        const glm::vec3 lo = furnaceLo(surface, directionAt(mu, phi), kSampleCount, ++seed);
                        if (!withinBand(lo, 1.0F, kTolerance)) {
                            std::cerr << "bsdf_validate: FAILED anisotropic " << label << " furnace at roughness=" << roughness
                                      << " anisotropy=" << anisotropy << " phi=" << phi << " mu=" << mu << " Lo=[" << minChannel(lo) << ", "
                                      << maxChannel(lo) << "]\n";
                            ok = false;
                        }
                    }
                }
            }
        }
    }
    finish(ctx, ok, "anisotropic_white_furnace failed; see the rows above");
}

// Anisotropic GGX and its Kulla-Conty lobe are symmetric in (wo, wi) at any azimuth: f(wo, wi) = f(wi, wo) for metal and dielectric.
PT_CHECK(anisotropic_reciprocity, Fast, Exact) {
    constexpr float kRelativeTolerance = 1e-4F;
    bool ok = true;
    for (float anisotropy : {0.5F, 0.9F, 1.0F}) {
        for (float roughness : {0.3F, 0.7F, 1.0F}) {
            for (float metallic : {0.0F, 1.0F}) {
                const Surface params = anisotropic(makeParams(roughness, metallic, 0.0F), anisotropy, 0.0F);
                for (float muA : {0.9F, 0.5F, 0.2F}) {
                    for (float muB : {0.8F, 0.4F, 0.15F}) {
                        const glm::vec3 wo = directionAt(muA, 0.3F);
                        const glm::vec3 wi = directionAt(muB, 1.9F);
                        const glm::vec3 f = pathtracer::scene::evaluateBsdfSplit(params, wo, wi).specular / muB;
                        const glm::vec3 g = pathtracer::scene::evaluateBsdfSplit(params, wi, wo).specular / muA;
                        const float scale = std::max(maxChannel(f), maxChannel(g));
                        if (!(maxChannel(glm::abs(f - g)) <= kRelativeTolerance * std::max(scale, 1e-4F))) {
                            std::cerr << "bsdf_validate: FAILED anisotropic reciprocity at roughness=" << roughness << " anisotropy=" << anisotropy
                                      << " metallic=" << metallic << " muO=" << muA << " muI=" << muB << " f(wo->wi)=" << f.x
                                      << " f(wi->wo)=" << g.x << '\n';
                            ok = false;
                        }
                    }
                }
            }
        }
    }
    finish(ctx, ok, "anisotropic_reciprocity failed; see the rows above");
}

// Every anisotropic draw re-evaluates to the density it was drawn at, a = 1's groove included, which no binned test can integrate.
PT_CHECK(anisotropic_sample_density_consistency, Fast, Exact) {
    constexpr int kSampleCount = 2000;
    bool ok = true;
    long long compared = 0;
    for (float anisotropy : {0.5F, 0.9F, 1.0F}) {
        for (float roughness : {0.3F, 0.8F}) {
            for (float metallic : {0.0F, 1.0F}) {
                for (float transmission : {0.0F, 1.0F}) {
                    const Surface params = anisotropic(coated(makeParams(roughness, metallic, transmission), 0.5F, glm::vec3(1.0F), roughness,
                                                              1.5F, 1.0F),
                                                       anisotropy, anisotropy);
                    for (float mu : {0.7F, 0.2F, -0.5F}) {
                        const glm::vec3 wo = directionAt(mu, 0.7F);
                        for (int i = 0; i < kSampleCount; ++i) {
                            pathtracer::scene::Sampler sampler(0, 0, i, kSampleCount, 23U);
                            const std::optional<pathtracer::scene::BsdfSample> sample = pathtracer::scene::sampleBsdf(params, wo, sampler);
                            if (!sample.has_value() || sample->delta) {
                                continue;
                            }
                            ++compared;
                            const float reevaluated = pathtracer::scene::pdfBsdf(params, wo, sample->wiLocal);
                            if (reevaluated != sample->pdf) {
                                std::cerr << "bsdf_validate: FAILED anisotropic sample/pdf consistency at anisotropy=" << anisotropy
                                          << " roughness=" << roughness << " metallic=" << metallic << " transmission=" << transmission
                                          << " mu=" << mu << " sample->pdf=" << sample->pdf << " pdfBsdf=" << reevaluated << '\n';
                                ok = false;
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "bsdf_validate: anisotropic sample/pdf consistency, " << compared << " draws re-evaluated\n";
    finish(ctx, ok && compared > 0, "an anisotropic draw's density differs from its re-evaluation");
}

// The albedo each anisotropic slab reports, its selection mass and albedo-scaling weight, is the energy its lobe reflects.
PT_CHECK(anisotropic_albedo_matches_lobe, Slow, Exact) {
    // The 4D kernel rule's order-8 residual plus its quadrilinear blend over (r, a, sqrt(mu), phi), at Simpson's resolution of the lobe.
    constexpr double kTolerance = 3e-3;
    constexpr int kPanels = 512;
    double worst = 0.0;
    for (float anisotropy : {0.5F, 0.9F}) {
        for (float roughness : {0.6F, 1.0F}) {
            for (float metallic : {0.0F, 1.0F}) {
                const Surface surface = anisotropic(metallic > 0.0F ? makeColoredMetalParams(roughness, glm::vec3(0.9F, 0.6F, 0.3F))
                                                                    : makeParams(roughness, 0.0F, 0.0F),
                                                    anisotropy, 0.0F);
                for (float phi : {0.0F, 0.25F * kPi, 0.5F * kPi}) {
                    for (float mu : {1.0F, 0.6F, 0.25F}) {
                        const pathtracer::scene::BsdfClosure closure = pathtracer::scene::makeBsdfClosure(surface, directionAt(mu, phi));
                        const glm::dvec3 integral = simpson(0.0, 1.0, kPanels, [&](double z) {
                            const double sine = std::sqrt(std::max(0.0, 1.0 - (z * z)));
                            return simpson(0.0, 2.0 * kPiDouble, kPanels, [&](double azimuth) {
                                const glm::vec3 wi(static_cast<float>(sine * std::cos(azimuth)), static_cast<float>(sine * std::sin(azimuth)),
                                                   static_cast<float>(z));
                                return glm::dvec3(pathtracer::scene::evaluateBsdfSplit(closure, wi).specular);
                            });
                        });
                        const glm::dvec3 albedo(metallic > 0.0F ? closure.metal.albedo
                                                                : closure.dielectric.reflectSingle + closure.dielectric.multiReflect);
                        const double delta = static_cast<double>(maxChannel(glm::vec3(glm::abs(integral - albedo))));
                        if (delta > worst) {
                            std::cout << "  roughness=" << roughness << " anisotropy=" << anisotropy << " metallic=" << metallic << " phi=" << phi
                                      << " mu=" << mu << " |lobe - albedo|=" << delta << '\n';
                        }
                        worst = std::max(worst, delta);
                    }
                }
            }
        }
    }
    std::cout << "bsdf_validate: anisotropic albedo vs integrated lobe, worst " << worst << '\n';
    finish(ctx, worst <= kTolerance, "an anisotropic slab's albedo does not match the energy its lobe reflects");
}

PT_CHECK_MAIN("bsdf")
