// Markdown on stdout for results/: where Adobe's openpbr-bsdf departs by design, measured; openpbr_reference_validate asserts the rest.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <thread>
#include <vector>

#include <glm/glm.hpp>

#include "openpbr.h"

#include "microfacet.h"
#include "microfacet_quadrature.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/sampler.h"
#include "stats.h"

namespace {

namespace scene = pathtracer::scene;
namespace quadrature = tools::quadrature;
using Surface = scene::OpenPbrInputs<scene::Constant>;

constexpr int kTableSize = OpenPBR_EnergyTableSize;

// Adobe's table axes: roughness k/31, mu j/31, and ior piecewise linear in 1/ior below 1 and ior above, nodes 15 and 16 both ior 1.
double adobeIor(int index) {
    const double half = kTableSize / 2;
    return index < half ? 1.0 / (1.0 + ((half - 1 - index) / (half - 1) * (OpenPBR_IorMax - 1.0)))
                        : 1.0 + ((index - half) / (half - 1) * (OpenPBR_IorMax - 1.0));
}

double adobeTable(const OpenPBR_EnergyTableElement* table, int flat) { return table[flat] / 65535.0; }

// Rows run in parallel, each writing only its own slots.
void parallelFor(int count, const std::function<void(int)>& body) {
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    for (unsigned w = 0; w < std::max(1U, std::thread::hardware_concurrency()); ++w) {
        workers.emplace_back([&] {
            for (int i = next++; i < count; i = next++) {
                body(i);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
}

struct Summary {
    double sum = 0.0;
    double sumSquares = 0.0;
    double worst = 0.0;
    int count = 0;
    void add(double residual) {
        sum += residual;
        sumSquares += residual * residual;
        worst = std::max(worst, std::abs(residual));
        ++count;
    }
};

void printRow(const char* table, const Summary& separable, const Summary& correlated) {
    std::printf("| %s | %d | %+.2e | %.2e | %.2e | %+.2e | %.2e |\n", table, separable.count, separable.sum / separable.count,
                std::sqrt(separable.sumSquares / separable.count), separable.worst, correlated.sum / correlated.count, correlated.worst);
}

// The integrator's E (white conductor) or R and T (interface) under a shadowing model at one node.
template <typename Shadowing>
quadrature::EscapeSums albedoAt(double ior, double alpha, double mu, const quadrature::GaussLegendre& rule) {
    if (ior == 0.0) {
        double albedo = 0.0;
        quadrature::forEachReflectNode<Shadowing>(mu, alpha, rule, rule, [&](double weight, double) { albedo += weight; });
        return {albedo, 0.0};
    }
    return quadrature::escapeAlbedo<Shadowing>(mu, alpha, 1.0 / ior, rule);
}

// 2 int E mu dmu by Gauss-Legendre panelled at the critical cosine, where an interface's E kinks.
template <typename Shadowing>
quadrature::EscapeSums averageAt(double ior, double alpha, const quadrature::GaussLegendre& rule, const quadrature::GaussLegendre& muRule) {
    const double critical = ior > 0.0 && ior < 1.0 ? std::sqrt(1.0 - (ior * ior)) : 0.0;
    quadrature::EscapeSums average{};
    for (const auto& [lo, hi] : {std::array<double, 2>{0.0, critical}, std::array<double, 2>{critical, 1.0}}) {
        for (std::size_t i = 0; i < muRule.node.size(); ++i) {
            const double mu = lo + ((hi - lo) * muRule.node[i]);
            const quadrature::EscapeSums at = albedoAt<Shadowing>(ior, alpha, mu, rule);
            const double weight = 2.0 * mu * (hi - lo) * muRule.weight[i];
            average.reflect += weight * at.reflect;
            average.transmit += weight * at.transmit;
        }
    }
    return average;
}

void reportTables() {
    const quadrature::GaussLegendre rule = quadrature::gaussLegendre(48);
    const quadrature::GaussLegendre muRule = quadrature::gaussLegendre(16);
    std::printf("## Energy tables\n\n");
    std::printf("Adobe's 32^3 16-bit tables against the renderer's quadrature (tools/microfacet_quadrature.h) at Adobe's nodes, mu > 0, ior != 1.\n");
    std::printf("Separable: the quadrature under Adobe's own G1(o)G1(i), which openpbr_reference_validate pins to Adobe's sampler by Monte Carlo,\n");
    std::printf("so its residual is the table's own error. Height-correlated: the renderer's model; the gap is the shadowing model's.\n\n");
    std::printf("| Table | Nodes | Separable mean | RMS | Max | Height-correlated mean | Max |\n|---|---|---|---|---|---|---|\n");

    // Rows by roughness for the conductor; by (ior, roughness) for the interface, each slot written once.
    std::vector<std::array<double, 4>> metal(static_cast<std::size_t>(kTableSize * kTableSize));
    parallelFor(kTableSize, [&](int k) {
        const double alpha = std::pow(k / 31.0, 2.0);
        for (int j = 1; j < kTableSize && k > 0; ++j) {
            const double adobe = 1.0 - adobeTable(OpenPBR_IdealMetalEnergyComplement_Array, (k * kTableSize) + j);
            metal[static_cast<std::size_t>((k * kTableSize) + j)] = {albedoAt<quadrature::Separable>(0.0, alpha, j / 31.0, rule).reflect - adobe,
                                                                      albedoAt<quadrature::HeightCorrelated>(0.0, alpha, j / 31.0, rule).reflect - adobe, 0.0, 0.0};
        }
        if (k > 0) {
            const double adobe = 1.0 - adobeTable(OpenPBR_IdealMetalAverageEnergyComplement_Array, k);
            metal[static_cast<std::size_t>(k * kTableSize)] = {averageAt<quadrature::Separable>(0.0, alpha, rule, muRule).reflect - adobe,
                                                               averageAt<quadrature::HeightCorrelated>(0.0, alpha, rule, muRule).reflect - adobe, 0.0, 0.0};
        }
    });
    Summary metalSeparable;
    Summary metalCorrelated;
    Summary metalAverageSeparable;
    Summary metalAverageCorrelated;
    for (int k = 1; k < kTableSize; ++k) {
        for (int j = 1; j < kTableSize; ++j) {
            metalSeparable.add(metal[static_cast<std::size_t>((k * kTableSize) + j)][0]);
            metalCorrelated.add(metal[static_cast<std::size_t>((k * kTableSize) + j)][1]);
        }
        metalAverageSeparable.add(metal[static_cast<std::size_t>(k * kTableSize)][0]);
        metalAverageCorrelated.add(metal[static_cast<std::size_t>(k * kTableSize)][1]);
    }
    printRow("Ideal metal E", metalSeparable, metalCorrelated);
    printRow("Ideal metal E_avg", metalAverageSeparable, metalAverageCorrelated);

    // Per (ior, roughness) row: [R+T sep, R+T hc, R sep, R hc] at each mu, then the two averages.
    std::vector<std::array<double, 4>> interface(static_cast<std::size_t>(kTableSize * kTableSize * kTableSize));
    std::vector<std::array<double, 4>> averages(static_cast<std::size_t>(kTableSize * kTableSize));
    parallelFor(kTableSize * kTableSize, [&](int row) {
        const int i = row / kTableSize;
        const int k = row % kTableSize;
        if (i == (kTableSize / 2) - 1 || i == kTableSize / 2 || k == 0) {
            return;
        }
        const double ior = adobeIor(i);
        const double alpha = std::pow(k / 31.0, 2.0);
        for (int j = 1; j < kTableSize; ++j) {
            const int flat = (((i * kTableSize) + k) * kTableSize) + j;
            const quadrature::EscapeSums separable = albedoAt<quadrature::Separable>(ior, alpha, j / 31.0, rule);
            const quadrature::EscapeSums correlated = albedoAt<quadrature::HeightCorrelated>(ior, alpha, j / 31.0, rule);
            const double ideal = 1.0 - adobeTable(OpenPBR_IdealDielectricEnergyComplement_Array, flat);
            const double opaque = 1.0 - adobeTable(OpenPBR_OpaqueDielectricEnergyComplement_Array, flat);
            interface[static_cast<std::size_t>(flat)] = {separable.reflect + separable.transmit - ideal, correlated.reflect + correlated.transmit - ideal,
                                                         separable.reflect - opaque, correlated.reflect - opaque};
        }
        const quadrature::EscapeSums separable = averageAt<quadrature::Separable>(ior, alpha, rule, muRule);
        const quadrature::EscapeSums correlated = averageAt<quadrature::HeightCorrelated>(ior, alpha, rule, muRule);
        const double ideal = 1.0 - adobeTable(OpenPBR_IdealDielectricAverageEnergyComplement_Array, row);
        const double opaque = 1.0 - adobeTable(OpenPBR_OpaqueDielectricAverageEnergyComplement_Array, row);
        averages[static_cast<std::size_t>(row)] = {separable.reflect + separable.transmit - ideal, correlated.reflect + correlated.transmit - ideal,
                                                   separable.reflect - opaque, correlated.reflect - opaque};
    });
    std::array<Summary, 4> directional;
    std::array<Summary, 4> mean;
    for (int i = 0; i < kTableSize; ++i) {
        for (int k = 1; k < kTableSize; ++k) {
            if (i == (kTableSize / 2) - 1 || i == kTableSize / 2) {
                continue;
            }
            for (int j = 1; j < kTableSize; ++j) {
                for (int t = 0; t < 4; ++t) {
                    directional[static_cast<std::size_t>(t)].add(interface[static_cast<std::size_t>((((i * kTableSize) + k) * kTableSize) + j)][static_cast<std::size_t>(t)]);
                }
            }
            for (int t = 0; t < 4; ++t) {
                mean[static_cast<std::size_t>(t)].add(averages[static_cast<std::size_t>((i * kTableSize) + k)][static_cast<std::size_t>(t)]);
            }
        }
    }
    printRow("Ideal dielectric R+T", directional[0], directional[1]);
    printRow("Ideal dielectric (R+T)_avg", mean[0], mean[1]);
    printRow("Opaque dielectric R", directional[2], directional[3]);
    printRow("Opaque dielectric R_avg", mean[2], mean[3]);
}

void reportReflectedShare() {
    std::printf("\n## Multiple scattering's reflected share\n\n");
    std::printf("The share of an interface's 2+ bounce energy leaving on the incident side: Adobe's IdealDielectricReflectionRatio against the\n");
    std::printf("renderer's Smith random walk (Heitz et al. 2016), which replaced the single-scatter stand-in R_avg/(R_avg + T_avg).\n\n");
    std::printf("| ior | roughness | Adobe | Random walk |\n|---|---|---|---|\n");
    for (const int i : {2, 8, 13, 18, 23, 31}) {
        for (const int k : {8, 16, 24, 31}) {
            const auto eta = static_cast<float>(1.0 / adobeIor(i));
            const float roughness = static_cast<float>(k) / 31.0F;
            const scene::InterfaceInputs interface{roughness, 0.0F, eta, 1.0F, eta, 1.0F, glm::vec3(1.0F), 1.0F, glm::vec3(1.0F), scene::FilmLayer{}, 1.0F, false};
            const scene::DielectricSlab slab = scene::makeDielectricSlab(interface, glm::vec3(std::sqrt(0.75F), 0.0F, 0.5F));
            const double share = slab.multiReflect / (slab.multiReflect + slab.multiTransmit);
            std::printf("| %.3f | %.3f | %.4f | %.4f |\n", adobeIor(i), static_cast<double>(roughness),
                        adobeTable(OpenPBR_IdealDielectricReflectionRatio_Array, (i * kTableSize) + k), share);
        }
    }
}

void reportAverageFresnel() {
    std::printf("\n## Average Fresnel\n\n");
    std::printf("Adobe's fitted openpbr_average_fresnel against the exact 2 int F mu dmu (fresnelAverage, asserted by openpbr_reference_validate).\n\n");
    std::printf("| eta | Adobe fit | Exact | Relative error |\n|---|---|---|---|\n");
    for (const float eta : {0.5F, 0.8F, 0.95F, 1.05F, 1.2F, 1.5F, 2.0F, 3.0F}) {
        const double fit = openpbr_average_fresnel(eta);
        const double exact = scene::fresnelAverage(eta);
        std::printf("| %.2f | %.5f | %.5f | %+.1f%% |\n", static_cast<double>(eta), fit, exact, 100.0 * (fit - exact) / exact);
    }
}

void reportThinWall() {
    std::printf("\n## Thin-walled sheet\n\n");
    std::printf("The smooth sheet's R' and T' summed per polarisation (exact) against Adobe's window form 2R/(1+R), (1-R)/(1+R) of the\n");
    std::printf("unpolarised R, its tint applied once and its reflection free of absorption. Clear, they agree at normal incidence\n");
    std::printf("(asserted); off normal the polarisations part, and a tinted sheet's back-face reflections cross it twice, absorbed.\n\n");
    std::printf("| ior | colour | mu | R' | Adobe R | T' | Adobe T |\n|---|---|---|---|---|---|---|\n");
    for (const float ior : {1.5F, 2.4F}) {
        for (const float color : {1.0F, 0.6F}) {
            const scene::SheetInterfaces sheet{glm::vec3(1.0F), glm::vec3(color), ior, ior, scene::FilmLayer{}};
            for (const float mu : {1.0F, 0.8F, 0.5F, 0.2F, 0.05F}) {
                const scene::SheetLadder ours = scene::sheetLadder(sheet, mu);
                const double adobeReflect = openpbr_thin_wall_fresnel(ior, mu);
                const double adobeTransmit = (1.0 - adobeReflect) * openpbr_compute_final_transmission_tint(vec3(color), true, mu, ior).x;
                std::printf("| %.2f | %.2f | %.2f | %.4f | %.4f | %.4f | %.4f |\n", static_cast<double>(ior), static_cast<double>(color),
                            static_cast<double>(mu), static_cast<double>(ours.reflect.x), adobeReflect, static_cast<double>(ours.transmit.x),
                            adobeTransmit);
            }
        }
    }
}

// One OpenPBR material for both BSDFs, every input the sweep varies set on each side, and Adobe's departures that reach it.
struct Material {
    const char* name;
    const char* departures;
    Surface inputs;
};

OpenPBR_ResolvedInputs adobeInputs(const Surface& s) {
    OpenPBR_ResolvedInputs a = openpbr_make_default_resolved_inputs();
    a.base_weight = s.baseWeight;
    a.base_color = s.baseColor;
    a.base_metalness = s.baseMetalness;
    a.base_diffuse_roughness = s.baseDiffuseRoughness;
    a.specular_weight = s.specularWeight;
    a.specular_color = s.specularColor;
    a.specular_roughness = s.specularRoughness;
    a.specular_roughness_anisotropy = s.specularRoughnessAnisotropy;
    a.specular_ior = s.specularIor;
    a.transmission_weight = s.transmissionWeight;
    a.coat_weight = s.coatWeight;
    a.coat_color = s.coatColor;
    a.coat_roughness = s.coatRoughness;
    a.coat_ior = s.coatIor;
    a.coat_darkening = s.coatDarkening;
    a.fuzz_weight = s.fuzzWeight;
    a.fuzz_color = s.fuzzColor;
    a.fuzz_roughness = s.fuzzRoughness;
    a.thin_film_weight = s.thinFilmWeight;
    a.thin_film_thickness = s.thinFilmThickness;
    a.thin_film_ior = s.thinFilmIor;
    a.transmission_color = s.transmissionColor;
    a.subsurface_weight = s.subsurfaceWeight;
    a.subsurface_color = s.subsurfaceColor;
    a.subsurface_scatter_anisotropy = s.subsurfaceScatterAnisotropy;
    a.geometry_thin_walled = s.geometryThinWalled;
    return a;
}

template <typename Edit>
Material material(const char* name, const char* departures, Edit edit) {
    Material m{name, departures, Surface{}};
    edit(m.inputs);
    return m;
}

// Directional albedo of both BSDFs at wo by their own importance sampling: the mean of RGB throughput weights.
void reportAlbedoSweep() {
    const std::array<Material, 12> materials{{
        material("white metal r 0.5", "separable G2", [](Surface& s) {
            s.baseMetalness = 1.0F;
            s.baseColor = glm::vec3(1.0F);
            s.specularRoughness = 0.5F;
        }),
        material("gold r 0.3", "separable G2; multiple-scattering colour F_avg^2 for Kulla-Conty's F_ms", [](Surface& s) {
            s.baseMetalness = 1.0F;
            s.baseColor = glm::vec3(1.0F, 0.78F, 0.34F);
            s.specularRoughness = 0.3F;
        }),
        material("dark metal tint 0 r 0.4", "separable G2; F82's average saturated after averaging, not averaged clamped", [](Surface& s) {
            s.baseMetalness = 1.0F;
            s.baseColor = glm::vec3(0.1F);
            s.specularColor = glm::vec3(0.0F);
            s.specularRoughness = 0.4F;
        }),
        material("glossy diffuse r 0.4", "diffuse scaled by (1 - E(mu_o))(1 - E(mu_i))/(1 - E_avg), not the spec's 1 - E(mu_o); separable G2 in E", [](Surface& s) { s.specularRoughness = 0.4F; }),
        material("rough diffuse", "diffuse scaled by (1 - E(mu_o))(1 - E(mu_i))/(1 - E_avg), not the spec's 1 - E(mu_o); separable G2 in E", [](Surface& s) {
            s.baseDiffuseRoughness = 1.0F;
            s.specularRoughness = 0.6F;
        }),
        material("rough glass r 0.4", "separable G2; tabulated reflected share; multiple scattering clamped to 1/mu", [](Surface& s) {
            s.transmissionWeight = 1.0F;
            s.baseColor = glm::vec3(1.0F);
            s.specularRoughness = 0.4F;
        }),
        material("coated metal", "darkening's K_s the fitted average Fresnel, not the spec's F(mu_o) (darkening off: within 0.3%); view-side attenuation only",
                 [](Surface& s) {
                     s.baseMetalness = 1.0F;
                     s.coatWeight = 1.0F;
                     s.coatRoughness = 0.1F;
                     s.specularRoughness = 0.5F;
                 }),
        material("fuzz over diffuse", "fuzz roughens the layers beneath; the diffuse beneath scaled as in the glossy-diffuse rows", [](Surface& s) {
            s.fuzzWeight = 1.0F;
            s.fuzzColor = glm::vec3(0.9F);
            s.fuzzRoughness = 0.5F;
        }),
        material("film over metal", "Airy reflectance at three RGB wavelengths, not integrated over CIE x D65", [](Surface& s) {
            s.baseMetalness = 1.0F;
            s.thinFilmWeight = 1.0F;
            s.thinFilmThickness = 0.4F;
            s.specularRoughness = 0.3F;
        }),
        material("anisotropic metal a 0.8", "multiple scattering from the isotropic tables at the isotropic alpha", [](Surface& s) {
            s.baseMetalness = 1.0F;
            s.baseColor = glm::vec3(0.95F);
            s.specularRoughness = 0.4F;
            s.specularRoughnessAnisotropy = 0.8F;
        }),
        material("thin-walled sheet r 0.3", "window form 2R/(1+R), unpolarised and absorption-free; separable G2 in the mirrored lobe", [](Surface& s) {
            s.geometryThinWalled = true;
            s.transmissionWeight = 1.0F;
            s.transmissionColor = glm::vec3(0.8F);
            s.specularRoughness = 0.3F;
        }),
        material("thin-walled subsurface g 0.4", "subsurface under the interface scaled by Adobe's average-Fresnel albedo", [](Surface& s) {
            s.geometryThinWalled = true;
            s.subsurfaceWeight = 1.0F;
            s.subsurfaceScatterAnisotropy = 0.4F;
            s.specularRoughness = 0.4F;
        }),
    }};
    constexpr int kSamples = 1 << 20;
    std::printf("\n## Directional albedo through both BSDFs\n\n");
    std::printf("Each BSDF's own importance sampling, %d samples per row; channel-mean albedo, its standard error, and the difference in\n", kSamples);
    std::printf("units of the combined standard error. Energy, not radiance: the renderer's refraction carries 1/eta^2, undone here.\n\n");
    std::printf("| Material | mu | Renderer | Adobe | Difference | sigma | Adobe's departures |\n|---|---|---|---|---|---|---|\n");
    for (const Material& m : materials) {
        for (const float mu : {1.0F, 0.5F, 0.15F}) {
            const glm::vec3 wo(std::sqrt(1.0F - (mu * mu)), 0.0F, mu);
            constexpr int kBatches = 64;
            std::array<double, kBatches> ours{};
            std::array<double, kBatches> adobe{};
            const OpenPBR_ResolvedInputs reference = adobeInputs(m.inputs);
            parallelFor(kBatches, [&](int batch) {
                const int per = kSamples / kBatches;
                double sumOurs = 0.0;
                double sumAdobe = 0.0;
                std::mt19937_64 rng(0x9E3779B97F4A7C15ULL + static_cast<std::uint64_t>(batch));
                const auto uniform = [&] { return std::min(static_cast<float>(std::uniform_real_distribution<double>(0.0, 1.0)(rng)), 0x1.fffffep-1F); };
                const OpenPBR_PreparedBsdf prepared = openpbr_prepare(reference, vec3(1.0F), OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, wo);
                for (int s = 0; s < per; ++s) {
                    scene::Sampler sampler(0, 0, (batch * per) + s, kSamples, 0x5EEDU);
                    if (const std::optional<scene::BsdfSample> sample = scene::sampleBsdf(m.inputs, wo, sampler)) {
                        // Entering from the ambient the radiance compression is (1/specular_ior)^2; a thin wall returns to the ambient.
                        const float compression = m.inputs.geometryThinWalled ? 1.0F : m.inputs.specularIor * m.inputs.specularIor;
                        const glm::vec3 energy = sample->throughputWeight + (sample->transmitWeight * (compression - 1.0F));
                        sumOurs += (energy.x + energy.y + energy.z) / 3.0;
                    }
                    vec3 direction;
                    OpenPBR_DiffuseSpecular weight;
                    float pdf = 0.0F;
                    OpenPBR_BsdfLobeType type;
                    const float u0 = uniform();
                    const float u1 = uniform();
                    const float u2 = uniform();
                    openpbr_sample(prepared, vec3(u0, u1, u2), direction, weight, pdf, type);
                    if (pdf > 0.0F) {
                        const vec3 sum = openpbr_get_sum_of_diffuse_specular(weight);
                        sumAdobe += (sum.x + sum.y + sum.z) / 3.0;
                    }
                }
                ours[static_cast<std::size_t>(batch)] = sumOurs / per;
                adobe[static_cast<std::size_t>(batch)] = sumAdobe / per;
            });
            tools::stats::Welford o;
            tools::stats::Welford a;
            for (int b = 0; b < kBatches; ++b) {
                o.add(ours[static_cast<std::size_t>(b)]);
                a.add(adobe[static_cast<std::size_t>(b)]);
            }
            const double sigma = std::sqrt((o.standardError() * o.standardError()) + (a.standardError() * a.standardError()));
            std::printf("| %s | %.2f | %.4f +/- %.1e | %.4f +/- %.1e | %+.4f | %.1f | %s |\n", m.name, static_cast<double>(mu), o.mean(),
                        o.standardError(), a.mean(), a.standardError(), o.mean() - a.mean(), (o.mean() - a.mean()) / sigma, m.departures);
        }
    }
}

}  // namespace

int main() {
    std::printf("# OpenPBR reference comparison: renderer vs Adobe openpbr-bsdf @ c91aad1\n\n");
    reportTables();
    reportReflectedShare();
    reportAverageFresnel();
    reportThinWall();
    reportAlbedoSweep();
    return 0;
}
