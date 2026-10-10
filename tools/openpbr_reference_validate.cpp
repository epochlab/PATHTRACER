// Adobe's openpbr-bsdf @ c91aad1 as an oracle, asserted where it is OpenPBR-exact; openpbr_reference_report measures its heuristics.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <glm/glm.hpp>

#include "openpbr.h"

#include "check.h"
#include "eon.h"
#include "fuzz.h"
#include "microfacet.h"
#include "microfacet_quadrature.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/fresnel_dielectric.h"
#include "stats.h"
#include "thin_film.h"

namespace {

namespace scene = pathtracer::scene;
namespace quadrature = tools::quadrature;
using Surface = scene::OpenPbrInputs<scene::Constant>;

// Each check keeps its worst row and reports one verdict, as bsdf_validate does.
void expectWithin(tools::check::Context& ctx, const char* what, double worst, double tolerance, const std::string& where) {
    char detail[384];
    std::snprintf(detail, sizeof(detail), "%s: worst %.3g (tolerance %.3g) at %s", what, worst, tolerance, where.c_str());
    std::cout << "  " << detail << '\n';
    PT_EXPECT(ctx, worst <= tolerance, detail);
}

// Relative difference against the larger magnitude, absolute below 1 where both formulas resolve to float epsilon.
double relativeGap(double ours, double reference) { return std::abs(ours - reference) / std::max({std::abs(ours), std::abs(reference), 1.0}); }

// Pure relative difference, for quantities with no natural unit scale (alphas, densities).
double relative(double ours, double reference) { return std::abs(ours - reference) / std::max(std::abs(ours), std::abs(reference)); }

// Two float evaluations of one formula, each within a few rounding steps of the real value: 16 ulps of the larger covers both chains.
constexpr double kFloatAgreement = 16.0 * 1.1920929e-7;

std::string format(const char* pattern, double a, double b = 0.0, double c = 0.0) {
    char text[128];
    std::snprintf(text, sizeof(text), pattern, a, b, c);
    return text;
}

// A uniform float strictly below 1, as Adobe's VNDF sampler asserts its rand.y is.
float uniformBelowOne(std::mt19937_64& rng) {
    return std::min(static_cast<float>(std::uniform_real_distribution<double>(0.0, 1.0)(rng)), 0x1.fffffep-1F);
}

PT_CHECK(dielectric_fresnel_matches_reference, Fast, Exact) {
    ctx.plan(1);
    double worst = 0.0;
    std::string where;
    for (const float ior : {0.4F, 0.667F, 0.95F, 1.05F, 1.5F, 2.5F}) {
        for (int i = 0; i <= 1024; ++i) {
            const float mu = static_cast<float>(i) / 1024.0F;
            // The transmitted cosine's sqrt amplifies cos^2's rounding by 1/(2 cos_t): the two float chains agree to that conditioning.
            const float cos2T = scene::cos2Transmitted(mu, 1.0F / ior);
            const double conditioning = 1.0 + (cos2T > 0.0F ? 1.0 / std::sqrt(cos2T) : 0.0);
            const double gap = std::abs(scene::fresnelDielectric(mu, 1.0F, ior) - openpbr_fresnel(ior, mu)) / conditioning;
            if (gap > worst) {
                worst = gap;
                where = format("ior %.3f mu %.4f", ior, mu);
            }
        }
    }
    expectWithin(ctx, "fresnelDielectric vs openpbr_fresnel, per unit conditioning", worst, kFloatAgreement, where);
}

// The F82-tint Fresnel at every angle, and its hemispherical average wherever F82 stays non-negative, where Adobe's closed form is exact.
PT_CHECK(f82_tint_matches_reference, Fast, Exact) {
    ctx.plan(3);
    double worstCurve = 0.0;
    double worstAverage = 0.0;
    double worstClamped = 0.0;
    std::string curveAt;
    std::string averageAt;
    std::string clampedAt;
    for (const float f0 : {0.0F, 0.04F, 0.3F, 0.7F, 0.95F, 1.0F}) {
        for (const float tint : {0.0F, 0.3F, 0.8F, 1.0F}) {
            const glm::vec3 f0v(f0);
            const glm::vec3 tintv(tint);
            const scene::ConductorSlab slab = scene::makeConductorSlab(0.0F, 0.0F, f0v, tintv, 1.0F, scene::FilmLayer{}, glm::vec3(0.0F, 0.0F, 1.0F));
            float rawMin = 1.0F;
            const float b = openpbr_compute_metal_schlick_b_factor(f0v, tintv).x;
            for (int i = 0; i <= 512; ++i) {
                const float mu = static_cast<float>(i) / 512.0F;
                const double gap = std::abs(scene::conductorFresnel(slab, mu).x - openpbr_metal_schlick_with_f82_tint(f0v, tintv, mu).x);
                if (gap > worstCurve) {
                    worstCurve = gap;
                    curveAt = format("f0 %.2f tint %.2f mu %.4f", f0, tint, mu);
                }
                const float c = 1.0F - mu;
                rawMin = std::min(rawMin, f0 + (((1.0F - f0) - (b * mu * c)) * c * c * c * c * c));
            }
            const double ours = scene::metalFresnelAvg(f0v, tintv).x;
            const double reference = openpbr_metal_average_fresnel_with_f82_tint(f0v, tintv).x;
            if (rawMin >= 0.0F) {
                if (std::abs(ours - reference) > worstAverage) {
                    worstAverage = std::abs(ours - reference);
                    averageAt = format("f0 %.2f tint %.2f", f0, tint);
                }
            } else if (reference - ours > worstClamped) {
                // Adobe averages F82 before saturating; the clamped curve's average can only be larger.
                worstClamped = reference - ours;
                clampedAt = format("f0 %.2f tint %.2f", f0, tint);
            }
        }
    }
    expectWithin(ctx, "conductorFresnel vs openpbr_metal_schlick_with_f82_tint", worstCurve, kFloatAgreement, curveAt);
    expectWithin(ctx, "metalFresnelAvg vs openpbr_metal_average_fresnel_with_f82_tint (F82 >= 0)", worstAverage, kFloatAgreement, averageAt);
    expectWithin(ctx, "Adobe's unclamped average above ours where F82 dips negative", worstClamped, kFloatAgreement, clampedAt);
}

// specular_weight's IOR modulation, identical below Adobe's F0 cap of 0.9999.
PT_CHECK(specular_weight_ior_matches_reference, Fast, Exact) {
    ctx.plan(1);
    double worst = 0.0;
    std::string where;
    for (const float ior : {0.5F, 0.8F, 1.2F, 1.5F, 2.5F, 3.0F}) {
        for (const float weight : {0.0F, 0.25F, 0.5F, 1.0F, 2.0F, 10.0F}) {
            const float f0 = ((ior - 1.0F) / (ior + 1.0F)) * ((ior - 1.0F) / (ior + 1.0F));
            if (!(f0 * weight < 0.9999F)) {
                continue;
            }
            Surface inputs;
            inputs.specularIor = ior;
            inputs.specularWeight = weight;
            const double gap = relativeGap(scene::modulatedIor(inputs, std::nullopt), openpbr_apply_specular_weight_to_ior(ior, weight));
            if (gap > worst) {
                worst = gap;
                where = format("ior %.2f weight %.2f", ior, weight);
            }
        }
    }
    expectWithin(ctx, "modulatedIor vs openpbr_apply_specular_weight_to_ior", worst, kFloatAgreement, where);
}

// OpenPBR's anisotropic alpha map, and GGX D times Smith G1 through the conductor's VNDF density at the same alphas.
PT_CHECK(anisotropic_ggx_matches_reference, Fast, Exact) {
    ctx.plan(2);
    double worstAlpha = 0.0;
    double worstDensity = 0.0;
    std::string alphaAt;
    std::string densityAt;
    std::mt19937_64 rng(ctx.seed());
    for (const float roughness : {0.2F, 0.5F, 0.9F}) {
        for (const float anisotropy : {0.0F, 0.3F, 0.7F, 0.95F}) {
            const glm::vec2 ours = scene::alphaForRoughness(roughness, anisotropy);
            const vec2 reference = openpbr_compute_anisotropic_alpha(roughness * roughness, anisotropy, true, 0.0F);
            const double gap = std::max(relative(ours.x, reference.x), relative(ours.y, reference.y));
            if (gap > worstAlpha) {
                worstAlpha = gap;
                alphaAt = format("r %.2f a %.2f", roughness, anisotropy);
            }
            for (int i = 0; i < 64; ++i) {
                const glm::vec3 wo = glm::normalize(glm::vec3(uniformBelowOne(rng) - 0.5F, uniformBelowOne(rng) - 0.5F, uniformBelowOne(rng) + 0.05F));
                const glm::vec3 wi = glm::normalize(glm::vec3(uniformBelowOne(rng) - 0.5F, uniformBelowOne(rng) - 0.5F, uniformBelowOne(rng) + 0.05F));
                const scene::ConductorSlab slab = scene::makeConductorSlab(roughness, anisotropy, glm::vec3(1.0F), glm::vec3(1.0F), 1.0F, scene::FilmLayer{}, wo);
                const glm::vec3 h = glm::normalize(wo + wi);
                const double expected = openpbr_eval_aniso_ggx(h, reference) * openpbr_eval_aniso_smith_g1(wo, reference) / (4.0 * wo.z);
                const double density = relative(scene::evaluateConductor(slab, wo, wi).pdfSingle, expected);
                if (density > worstDensity) {
                    worstDensity = density;
                    densityAt = format("r %.2f a %.2f mu_o %.3f", roughness, anisotropy, wo.z);
                }
            }
        }
    }
    expectWithin(ctx, "alphaForRoughness vs openpbr_compute_anisotropic_alpha", worstAlpha, kFloatAgreement, alphaAt);
    // D's 1/(pi d^2) and G1's radical each round a few times in float: 64 ulps relative covers the product's chain.
    expectWithin(ctx, "VNDF density vs openpbr_eval_aniso_ggx * G1 / (4 mu_o)", worstDensity, 4.0 * kFloatAgreement, densityAt);
}

// Cauchy dispersion from (n_d, V_d), on either side of the surround, at the hero wavelengths.
PT_CHECK(dispersion_matches_reference, Fast, Exact) {
    ctx.plan(1);
    double worst = 0.0;
    std::string where;
    for (const float ior : {0.45F, 0.7F, 1.2F, 1.5F, 2.4F}) {
        for (const float abbe : {20.0F, 40.0F, 90.0F}) {
            Surface inputs;
            inputs.specularIor = ior;
            inputs.transmissionDispersionAbbeNumber = abbe;
            inputs.transmissionDispersionScale = 1.0F;
            for (int channel = 0; channel < 3; ++channel) {
                const double gap = relativeGap(scene::modulatedIor(inputs, channel),
                                               openpbr_dispersion_adjusted_ior(ior, 20.0F / abbe, scene::kRgbWavelengthsNm[channel]));
                if (gap > worst) {
                    worst = gap;
                    where = format("ior %.2f abbe %.0f channel %.0f", ior, abbe, channel);
                }
            }
        }
    }
    expectWithin(ctx, "dispersed specular_ior vs openpbr_dispersion_adjusted_ior", worst, kFloatAgreement, where);
}

// Gulbrandsen 2014's (n, k) against its formula in double, and against Adobe's, whose float k^2 cancels as r -> 1, to Adobe's own rounding.
PT_CHECK(gulbrandsen_matches_reference, Fast, Exact) {
    ctx.plan(2);
    double worstExact = 0.0;
    double worstReference = 0.0;
    std::string exactAt;
    std::string referenceAt;
    for (const float r : {0.02F, 0.2F, 0.5F, 0.8F, 0.95F, 0.985F}) {
        for (const float g : {0.0F, 0.25F, 0.5F, 0.75F, 1.0F}) {
            const double sqrtR = std::sqrt(static_cast<double>(r));
            const double n = (g * (1.0 - r) / (1.0 + r)) + ((1.0 - g) * (1.0 + sqrtR) / (1.0 - sqrtR));
            const double k = std::sqrt(std::max(((r * (n + 1.0) * (n + 1.0)) - ((n - 1.0) * (n - 1.0))) / (1.0 - r), 0.0));
            const scene::ComplexIor ours = scene::conductorIor(glm::vec3(r), glm::vec3(g));
            const vec2 reference = openpbr_compute_gulbrandsen_n_and_k(r, g);
            const double exact = std::max(relativeGap(ours.eta.x, n), relativeGap(ours.kappa.x, k));
            if (exact > worstExact) {
                worstExact = exact;
                exactAt = format("r %.3f g %.2f", r, g);
            }
            const double excess = std::max(relativeGap(ours.eta.x, reference.x) - relativeGap(reference.x, n),
                                           relativeGap(ours.kappa.x, reference.y) - relativeGap(reference.y, k));
            if (excess > worstReference) {
                worstReference = excess;
                referenceAt = format("r %.3f g %.2f", r, g);
            }
        }
    }
    expectWithin(ctx, "conductorIor vs Gulbrandsen in double", worstExact, kFloatAgreement, exactAt);
    expectWithin(ctx, "conductorIor's gap to openpbr_compute_gulbrandsen_n_and_k beyond Adobe's own rounding", worstReference, kFloatAgreement,
                 referenceAt);
}

// EON's value with rho = C (OpenPBR's albedo-scaling form), the FON quartic albedo fit both implementations ship.
PT_CHECK(eon_matches_reference, Fast, Exact) {
    ctx.plan(1);
    double worst = 0.0;
    std::string where;
    std::mt19937_64 rng(ctx.seed());
    const glm::vec3 rho(0.9F, 0.5F, 0.1F);
    for (const float roughness : {0.0F, 0.3F, 0.7F, 1.0F}) {
        for (int i = 0; i < 128; ++i) {
            const glm::vec3 wo = glm::normalize(glm::vec3(uniformBelowOne(rng) - 0.5F, uniformBelowOne(rng) - 0.5F, uniformBelowOne(rng) + 0.02F));
            const glm::vec3 wi = glm::normalize(glm::vec3(uniformBelowOne(rng) - 0.5F, uniformBelowOne(rng) - 0.5F, uniformBelowOne(rng) + 0.02F));
            const glm::vec3 ours = scene::evaluateDiffuse(scene::makeDiffuseSlab(rho, roughness, wo), wo, wi).value;
            const glm::vec3 reference = openpbr_f_EON(rho, roughness, wi, wo, false) * wi.z;
            for (int c = 0; c < 3; ++c) {
                const double gap = relativeGap(ours[c], reference[c]);
                if (gap > worst) {
                    worst = gap;
                    where = format("r %.2f mu_o %.3f mu_i %.3f", roughness, wo.z, wi.z);
                }
            }
        }
    }
    expectWithin(ctx, "evaluateDiffuse vs openpbr_f_EON * mu_i", worst, kFloatAgreement, where);
}

// The fuzz's sheen LTC (Zeltner, Burley, Chiang 2022): one published 32x32 table, read at every node by both implementations.
PT_CHECK(fuzz_ltc_matches_reference, Fast, Exact) {
    ctx.plan(1);
    double worst = 0.0;
    std::string where;
    for (int row = 0; row < OpenPBR_LTCTableSize; ++row) {
        for (int column = 0; column < OpenPBR_LTCTableSize; ++column) {
            const float roughness = static_cast<float>(row) / (OpenPBR_LTCTableSize - 1);
            const float mu = static_cast<float>(column) / (OpenPBR_LTCTableSize - 1);
            const vec3 reference = OpenPBR_LTC_Array[(row * OpenPBR_LTCTableSize) + column];
            const scene::FuzzSlab slab = scene::makeFuzzSlab(glm::vec3(1.0F), roughness, glm::vec3(std::sqrt(1.0F - (mu * mu)), 0.0F, mu));
            const double gap = std::max({relativeGap(slab.aInv, reference.x), relativeGap(slab.bInv, reference.y), relativeGap(slab.albedo, reference.z)});
            if (gap > worst) {
                worst = gap;
                where = format("row %.0f column %.0f", row, column);
            }
        }
    }
    // Adobe's table prints five decimals of the published one: half a unit in the fifth place bounds the difference.
    expectWithin(ctx, "fuzz LTC (aInv, bInv, albedo) vs OpenPBR_LTC_Array", worst, 5e-6, where);
}

// The coat's one-pass transmittance coat_color^(1/(2 mu_t)), mu_t refracted into the coat.
PT_CHECK(coat_transmittance_matches_reference, Fast, Exact) {
    ctx.plan(1);
    double worst = 0.0;
    std::string where;
    for (const float coatIor : {1.2F, 1.5F, 2.2F}) {
        OpenPBR_CoatingLobe_AggregateLobe lobe{};
        lobe.tint = vec3(0.2F, 0.6F, 0.95F);
        lobe.presence = 1.0F;
        lobe.coat_reflection_lobe.refl_trans_coeff.eta_t_over_eta_i = coatIor;
        for (int i = 1; i <= 64; ++i) {
            const float mu = static_cast<float>(i) / 64.0F;
            const glm::vec3 ours = scene::coatTransmittance(lobe.tint, coatIor, mu);
            const vec3 reference = openpbr_coat_passage_color_multiplier(lobe, mu);
            for (int c = 0; c < 3; ++c) {
                const double gap = relativeGap(ours[c], reference[c]);
                if (gap > worst) {
                    worst = gap;
                    where = format("coat_ior %.2f mu %.4f", coatIor, mu);
                }
            }
        }
    }
    expectWithin(ctx, "coatTransmittance vs openpbr_coat_passage_color_multiplier", worst, kFloatAgreement, where);
}

// The coat's E_F, 2 int F mu dmu in closed form, against Gauss-Legendre over Adobe's exact Fresnel, panelled at the critical cosine.
PT_CHECK(fresnel_average_matches_reference, Fast, Exact) {
    ctx.plan(1);
    const quadrature::GaussLegendre rule = quadrature::gaussLegendre(64);
    double worst = 0.0;
    std::string where;
    for (const float eta : {0.4F, 0.667F, 0.9F, 0.99F, 1.01F, 1.1F, 1.5F, 2.0F, 3.0F}) {
        const double critical = eta < 1.0F ? std::sqrt(1.0 - (static_cast<double>(eta) * eta)) : 0.0;
        double average = 0.0;
        for (const auto& [lo, hi] : {std::array<double, 2>{0.0, critical}, std::array<double, 2>{critical, 1.0}}) {
            for (std::size_t i = 0; i < rule.node.size(); ++i) {
                const double mu = lo + ((hi - lo) * rule.node[i]);
                average += 2.0 * mu * (hi - lo) * rule.weight[i] * openpbr_fresnel(eta, static_cast<float>(mu));
            }
        }
        const double gap = std::abs(scene::fresnelAverage(eta) - average);
        if (gap > worst) {
            worst = gap;
            where = format("eta %.3f", eta);
        }
    }
    // Fresnel's sqrt edge at grazing (and the critical cosine) holds Gauss-Legendre to ~mu^(3/2) convergence: 64 nodes reach 1e-5.
    expectWithin(ctx, "fresnelAverage vs 2 int openpbr_fresnel mu dmu", worst, 1e-5, where);
}

// Rec.709 of a reflectance spectrum under D65, normalised so a flat unit spectrum is white and clamped, as filmReflectance integrates.
template <typename Reflectance>
glm::vec3 spectralRec709(Reflectance reflectance) {
    glm::dvec3 xyz(0.0);
    double whiteY = 0.0;
    for (int i = 0; i < scene::cie::kSampleCount; ++i) {
        const scene::cie::TableRow row = scene::cie::tableRow(i);
        xyz += row.colourMatching * row.d65 * static_cast<double>(reflectance(static_cast<float>(scene::cie::wavelengthNm(i))));
        whiteY += row.colourMatching.y * row.d65;
    }
    return glm::clamp(glm::vec3(scene::cie::xyzToRec709() * (xyz / whiteY)), 0.0F, 1.0F);
}

// The thin film: Adobe's per-wavelength Airy reflectance integrated against CIE 1931 x D65, against filmReflectance's exact transform.
PT_CHECK(thin_film_matches_reference, Fast, Exact) {
    ctx.plan(1);
    const glm::vec3 metalF0(0.9F, 0.6F, 0.3F);
    const glm::vec3 metalTint(0.95F, 0.9F, 0.8F);
    const scene::ComplexIor metal = scene::conductorIor(metalF0, metalTint);
    double worst = 0.0;
    std::string where;
    for (const float baseIor : {1.5F, 2.4F, 0.0F}) {
        for (const float filmIor : {1.3F, 1.8F}) {
            for (const float thickness : {120.0F, 400.0F, 900.0F}) {
                for (const float mu : {1.0F, 0.7F, 0.3F}) {
                    glm::vec3 reference;
                    glm::vec3 ours;
                    if (baseIor > 0.0F) {
                        reference = spectralRec709([&](float lambda) {
                            return openpbr_thin_film_and_base_reflectance(mu, 1.0F, filmIor, vec3(baseIor), false, vec3(0.0F), vec3(1.0F), true,
                                                                          false, thickness, vec3(lambda))
                                .reflectance_dielectric.x;
                        });
                        ours = scene::filmReflectance({thickness, filmIor}, mu, 1.0F, glm::vec3(baseIor), glm::vec3(0.0F));
                    } else {
                        // Per channel the conductor is grey across the spectrum: each Rec.709 channel integrates its own (F0, tint).
                        for (int c = 0; c < 3; ++c) {
                            reference[c] = spectralRec709([&](float lambda) {
                                return openpbr_thin_film_and_base_reflectance(mu, 1.0F, filmIor, vec3(1.0F), false, vec3(metalF0[c]),
                                                                              vec3(metalTint[c]), false, true, thickness, vec3(lambda))
                                    .reflectance_metal.x;
                            })[c];
                        }
                        ours = scene::filmReflectance({thickness, filmIor}, mu, 1.0F, metal.eta, metal.kappa);
                    }
                    for (int c = 0; c < 3; ++c) {
                        const double gap = std::abs(ours[c] - reference[c]);
                        if (gap > worst) {
                            worst = gap;
                            where = format("base %.1f film %.1f thickness %.0f", baseIor, filmIor, thickness) + format(" mu %.2f channel %.0f", mu, c);
                        }
                    }
                }
            }
        }
    }
    // filmReflectance lerps its CIE transforms at 2 nm of path: (h^2/8)(2 pi/380 nm)^2 = 1.4e-4 per term, twice that over 2|amplitude| < 2.
    expectWithin(ctx, "filmReflectance vs spectrally integrated openpbr_thin_film_and_base_reflectance", worst, 3e-4, where);
}

// --- Energy integrals: the tables' quadrature, switched to Adobe's separable shadowing, against Monte Carlo through Adobe's own sampler.

struct Probe {
    float ior;  // eta_t/eta_i as Adobe reads it; 0 for the white conductor
    float roughness;
    float mu;
};

// Adobe's single-scattering albedo estimator: VNDF-sampled facets weighted F G1(wi), so E = E[F G1(r) + (1 - F) G1(t)].
double adobeAlbedoSample(const Probe& probe, const vec3& wo, std::mt19937_64& rng) {
    const vec2 alpha(probe.roughness * probe.roughness);
    const vec3 h = openpbr_sample_aniso_ggx_smith_vndf(alpha, wo, vec2(uniformBelowOne(rng), uniformBelowOne(rng)));
    const float woDotH = dot(wo, h);
    if (!(woDotH > 0.0F)) {
        return 0.0;
    }
    const float fresnel = probe.ior == 0.0F ? 1.0F : openpbr_fresnel(probe.ior, woDotH);
    double value = 0.0;
    const vec3 reflected = (2.0F * woDotH * h) - wo;
    if (reflected.z > 0.0F) {
        value += fresnel * openpbr_eval_aniso_smith_g1(reflected, alpha);
    }
    vec3 refracted;
    if (fresnel < 1.0F && openpbr_refract(wo, h, woDotH, probe.ior, refracted) && refracted.z < 0.0F) {
        value += (1.0F - fresnel) * openpbr_eval_aniso_smith_g1(refracted, alpha);
    }
    return value;
}

// The tables' integrator under a shadowing model: the conductor's E, or the interface's R + T.
template <typename Shadowing>
double integratedAlbedo(const Probe& probe, int nodes) {
    const quadrature::GaussLegendre rule = quadrature::gaussLegendre(nodes);
    const double alpha = static_cast<double>(probe.roughness) * probe.roughness;
    if (probe.ior == 0.0F) {
        double albedo = 0.0;
        quadrature::forEachReflectNode<Shadowing>(probe.mu, alpha, rule, rule, [&](double weight, double) { albedo += weight; });
        return albedo;
    }
    const quadrature::EscapeSums sums = quadrature::escapeAlbedo<Shadowing>(probe.mu, alpha, 1.0 / probe.ior, rule);
    return sums.reflect + sums.transmit;
}

PT_CHECK(separable_integrator_matches_reference_sampler, Slow, Statistical) {
    const std::array<Probe, 8> probes{{{0.0F, 0.25F, 0.65F}, {0.0F, 0.6F, 0.3F}, {0.0F, 1.0F, 0.8F}, {2.0F, 1.0F, 0.8F},
                                       {2.0F, 0.25F, 0.65F}, {0.5F, 1.0F, 0.8F}, {0.5F, 0.6F, 0.3F}, {1.5F, 0.6F, 0.5F}}};
    constexpr int kSamplesPerReplicate = 1 << 18;
    ctx.plan(static_cast<int>(probes.size()));
    std::cout << "openpbr_reference_validate: separable-G2 quadrature vs Monte Carlo through Adobe's VNDF sampler, Fresnel and G1\n";
    for (const Probe& probe : probes) {
        char label[96];
        std::snprintf(label, sizeof(label), "ior %.2f r %.2f mu %.2f", static_cast<double>(probe.ior), static_cast<double>(probe.roughness),
                      static_cast<double>(probe.mu));
        const std::uint64_t seed = ctx.subSeed(label);
        const vec3 wo(std::sqrt(1.0F - (probe.mu * probe.mu)), 0.0F, probe.mu);
        std::array<double, tools::stats::kReplicates> replicates{};
        std::atomic<int> next{0};
        std::vector<std::thread> workers;
        for (int w = 0; w < std::min(ctx.threads(), tools::stats::kReplicates); ++w) {
            workers.emplace_back([&] {
                for (int r = next++; r < tools::stats::kReplicates; r = next++) {
                    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(r));
                    double sum = 0.0;
                    for (int s = 0; s < kSamplesPerReplicate; ++s) {
                        sum += adobeAlbedoSample(probe, wo, rng);
                    }
                    replicates[static_cast<std::size_t>(r)] = sum / kSamplesPerReplicate;
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
        tools::stats::Welford estimate;
        for (const double replicate : replicates) {
            estimate.add(replicate);
        }
        // The shipped rule's own error is its distance from a doubled one, as albedo_table reports it.
        const double shipped = integratedAlbedo<quadrature::Separable>(probe, 48);
        const double doubled = integratedAlbedo<quadrature::Separable>(probe, 96);
        const double half = (tools::stats::studentTTwoSided(ctx.alpha(), tools::stats::kReplicates - 1) * estimate.standardError()) +
                            std::abs(shipped - doubled);
        char detail[256];
        std::snprintf(detail, sizeof(detail), "%s: quadrature %.6f vs sampler %.6f +/- %.2g", label, doubled, estimate.mean(), half);
        std::cout << "  " << detail << '\n';
        PT_EXPECT(ctx, std::abs(doubled - estimate.mean()) <= half, detail);
    }
}

// The same integrator under height-correlated shadowing reproduces the shipped tables at their nodes: the .inc is current.
PT_CHECK(height_correlated_integrator_matches_tables, Fast, Exact) {
    ctx.plan(2);
    const quadrature::GaussLegendre reflectRule = quadrature::gaussLegendre(96);
    const glm::ivec2 grid = scene::albedoGridRes();
    double worstReflect = 0.0;
    std::string reflectAt;
    for (int ri = 1; ri < grid.x; ri += 17) {
        for (int mi = 1; mi < grid.y; mi += 17) {
            const double roughness = static_cast<double>(ri) / (grid.x - 1);
            const double sqrtMu = static_cast<double>(mi) / (grid.y - 1);
            double albedo = 0.0;
            quadrature::forEachReflectNode<quadrature::HeightCorrelated>(sqrtMu * sqrtMu, roughness * roughness, reflectRule, reflectRule,
                                                                         [&](double weight, double) { albedo += weight; });
            const double table = 1.0 - scene::directionalAlbedoSplit(scene::albedoGridMu(static_cast<float>(mi)),
                                                                    scene::albedoGridRoughness(static_cast<float>(ri))).w;
            if (std::abs(table - albedo) > worstReflect) {
                worstReflect = std::abs(table - albedo);
                reflectAt = format("roughness %.4f mu %.4f", roughness, sqrtMu * sqrtMu);
            }
        }
    }
    const quadrature::GaussLegendre escapeRule = quadrature::gaussLegendre(48);
    double worstEscape = 0.0;
    std::string escapeAt;
    for (const int ri : {4, 12, 20, 31}) {
        for (const int ei : {8, 24, 40, 56}) {
            for (const int mi : {8, 24, 40, 63}) {
                const double roughness = ri / 31.0;
                const double eta = std::exp(std::log(1.0 / 3.0) + ((ei / 63.0) * std::log(9.0)));
                const double mu = (mi / 63.0) * (mi / 63.0);
                const quadrature::EscapeSums sums =
                    quadrature::escapeAlbedo<quadrature::HeightCorrelated>(mu, roughness * roughness, eta, escapeRule);
                const auto etaF = static_cast<float>(eta);
                const scene::InterfaceInputs interface{static_cast<float>(roughness), 0.0F, etaF, 1.0F, etaF, 1.0F, glm::vec3(1.0F), 1.0F,
                                                       glm::vec3(1.0F), scene::FilmLayer{}, 1.0F, false};
                const auto muF = static_cast<float>(mu);
                const scene::DielectricSlab slab = scene::makeDielectricSlab(interface, glm::vec3(std::sqrt(1.0F - (muF * muF)), 0.0F, muF));
                const double gap = std::max(std::abs(slab.reflectSingle.x - sums.reflect), std::abs(slab.transmitSingle.x - sums.transmit));
                if (gap > worstEscape) {
                    worstEscape = gap;
                    escapeAt = format("roughness %.3f eta %.3f mu %.4f", roughness, eta, mu);
                }
            }
        }
    }
    // Float storage and the lookups' float axis coordinates, a node read as a 1e-6 blend with its neighbour: both below 1e-5.
    expectWithin(ctx, "kAlbedo* at their nodes vs height-correlated quadrature", worstReflect, 1e-5, reflectAt);
    expectWithin(ctx, "kEscapeReflect/Transmit at their nodes vs height-correlated quadrature", worstEscape, 1e-5, escapeAt);
}

}  // namespace

PT_CHECK_MAIN("openpbr_reference_validate")
