// Offline generator for src/scene/albedo_table.inc, the Kulla-Conty energy tables.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <limits>
#include <atomic>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "microfacet_quadrature.h"
#include "pathtracer/scene/fresnel_dielectric.h"

namespace {

// The shading path's own dielectric interface, so the table is baked against exactly what reads it.
using pathtracer::scene::fresnelDielectric;
using tools::quadrature::EscapeSums;
using tools::quadrature::GaussLegendre;
using tools::quadrature::HeightCorrelated;
using tools::quadrature::gaussLegendre;
using tools::quadrature::kPi;

// Reflect side, three resolutions each sized by what checkAlbedoTableInterpolation measures on that axis; the bilinear read dominates.
constexpr int kAlbedoRoughnessRes = 256;
constexpr int kAlbedoMuRes = 256;

// The reflected MS lobe's sampling grid, uniform in mu and its own constant: the runtime inversion needs one step width, not that warp.
constexpr int kMsReflectMuRes = 128;

// Transmit side, sized by the energy closure it buys: 64 mu and 64 eta nodes close to 3e-4, where 32 each lost 2% at mu 0.02.
constexpr int kTransmitRoughnessRes = 32;
constexpr int kTransmitMuRes = 64;
constexpr int kEtaRes = 64;
constexpr double kEtaMin = 1.0 / 3.0;  // OpenPBR's normalised specular_ior range [1, 3], exiting; the reciprocal end is entering
constexpr double kEtaMax = 3.0;

// Gauss-Legendre nodes per transmit panel, in phi and each psi panel; verifyTransmit reports the residual against a doubled rule.
constexpr int kTransmitNodes = 48;
// Gauss-Legendre nodes per mu panel of the escape means; verifyTransmit doubles them with the angular rule.
constexpr int kTransmitMeanNodes = 16;
// Jittered incidence strata per axis for the random walks: 2^20 walks per (roughness, eta) cell.
constexpr int kWalkStrata = 1024;

// The reflection kernel's Gauss rules over x = dot(wo, h): E[F] for any Fresnel of x, the thin film's among them, on a coarse grid.
constexpr int kKernelRoughnessRes = 32;
constexpr int kKernelMuRes = 32;
constexpr int kKernelOrder = 8;

// The anisotropic kernel over (r, 1 - sqrt(1 - a), sqrt(mu), sqrt(alpha_o)), each axis sized by its measured mid-cell error.
constexpr int kAnisoRoughnessRes = 16;
constexpr int kAnisoAnisotropyRes = 8;
constexpr int kAnisoMuRes = 32;
constexpr int kAnisoPhiRes = 8;
// Gauss-Legendre nodes per axis over the VNDF's unit square and the mean deficit's (mu, phi) rules; verifyAnisotropic doubles all three.
constexpr int kAnisoSquareNodes = 128;
constexpr int kAnisoMeanMuNodes = 16;
constexpr int kAnisoMeanPhiNodes = 8;

// Reflect side, exact-domain Gauss-Legendre: E = F0*a + b - k*c for F82's F0 + (1-F0)(1-x)^5 - k*x(1-x)^6, linear in (F0, k).
struct Split {
    double a;
    double b;
    double c;
};

// The deficit 1 - E in double, where it is resolved: formed in float from a table's E it cancels to nothing as alpha -> 0.
double deficitOf(const Split& split) {
    // A passive microsurface reflects no more than it receives; the quadrature's own residual above E = 1 is no energy.
    return std::max(1.0 - (split.a + split.b), 0.0);
}

Split reflectAlbedo(double mu, double alpha, const GaussLegendre& phiRule, const GaussLegendre& psiRule) {
    // alpha = 0 is the smooth mirror, every facet the macro normal: E(F) = F(mu) exactly, where the measure below degenerates.
    if (alpha == 0.0) {
        const double fc = std::pow(1.0 - mu, 5.0);
        return {1.0 - fc, fc, mu * fc * (1.0 - mu)};
    }
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    tools::quadrature::forEachReflectNode<HeightCorrelated>(mu, alpha, phiRule, psiRule, [&](double weight, double woDotH) {
        const double fc = std::pow(std::clamp(1.0 - woDotH, 0.0, 1.0), 5.0);
        a += weight * (1.0 - fc);
        b += weight * fc;
        c += weight * woDotH * std::pow(std::clamp(1.0 - woDotH, 0.0, 1.0), 6.0);
    });
    // The 1/mu is inside the G2 over cosO, so mu = 0 is a node; never 0/0: psiMax is unreachable on panel one and 0 on panel two.
    return {a, b, c};
}

// --- Transmit side: the reflect measure panelled at the interface's boundaries; a VNDF midpoint rule lumps the slope tail in, 3.6e-3.

// log-spaced so eta and 1/eta are symmetric about index kEtaRes/2.
double etaAtIndex(int index) {
    const double u = static_cast<double>(index) / static_cast<double>(kEtaRes - 1);
    return std::exp(std::log(kEtaMin) + (u * (std::log(kEtaMax) - std::log(kEtaMin))));
}

// Rows are independent and each writes only its own slice, so the split is a pure speedup with no effect on the values.
template <typename Row>
void parallelRows(int rows, Row row) {
    const unsigned workers = std::max(1U, std::thread::hardware_concurrency());
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned w = 0; w < workers; ++w) {
        pool.emplace_back([&] {
            for (int i = next++; i < rows; i = next++) {
                row(i);
            }
        });
    }
    for (std::thread& worker : pool) {
        worker.join();
    }
}

// --- Table assembly.

struct AlbedoTable {
    std::vector<float> a;  // [roughnessIndex][muIndex], kAlbedoRoughnessRes * kAlbedoMuRes
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> d;  // 1 - E in double
    std::vector<float> davg;  // the deficit's cosine-weighted mean, 2*integral((1 - E(mu))*mu dmu)
    std::vector<float> r;  // [roughnessIndex][muIndex][etaIndex], kTransmitRoughnessRes * kTransmitMuRes * kEtaRes
    std::vector<float> t;
    std::vector<float> escapeDeficit;  // 1 - R - T in double, at the physical bound 0 where the quadrature lands past unity
    std::vector<float> ravg;
    std::vector<float> tavg;
    std::vector<float> escapeAvgDeficit;  // 1 - Ravg - Tavg in double, at the physical bound 0
    std::vector<float> walkReflect;   // [roughness][eta], the cosine-weighted energy escaping on the incident side after 2+ bounces
    std::vector<float> walkTransmit;  // the same escaping across the interface
    std::vector<float> msDensity;  // [roughnessIndex][muIndex], the reflected multiple-scattering lobe's own shape
    std::vector<float> msCdf;
    std::vector<float> escapeShapeDensity;  // [roughnessIndex][muIndex][etaIndex], the transmitted twin, unnormalised
    std::vector<float> escapeShapeCdf;
    std::vector<float> kernelNode;    // [roughnessIndex][muIndex][order], the reflect kernel's Gauss nodes in x = dot(wo, h)
    std::vector<float> kernelWeight;
    std::vector<float> averageNode;   // the Gauss rule of 2 mu dmu on [0, 1], the hemispherical average's
    std::vector<float> averageWeight;
    std::vector<float> anisoNode;     // [roughness][anisotropy][mu][phi][order]
    std::vector<float> anisoWeight;
    std::vector<float> anisoDeficit;  // [roughness][anisotropy][mu][phi], 1 - E for unit Fresnel in double
    std::vector<float> anisoB;        // F82's split E = F0 a + b - k c at each cell, a = 1 - deficit - b: the metal reads no rule
    std::vector<float> anisoC;
    std::vector<float> anisoDeficitAvg;  // [roughness][anisotropy], its cosine-weighted hemispherical mean
    std::vector<double> anisoProbe;   // [cell][probe], each kernel probe's expectation under the raw quadrature measure, for verification
    double anisoExcess = 0.0;         // the largest E - 1 the quadrature reached, which verifyAnisotropic bounds by its residual
};

// The reflect table's mu axis, uniform in sqrt(mu); node 0 is mu = 0 itself, where E = 1 exactly and buildReflect asserts it on every row.
double reflectMu(int index) {
    const double t = static_cast<double>(index) / static_cast<double>(kAlbedoMuRes - 1);
    return t * t;
}

// The escape tables' mu axis, uniform in sqrt(mu), node 0 being mu = 0 itself: the grazing limit G2 over cosO holds to.
double escapeMu(int index) {
    const double t = static_cast<double>(index) / static_cast<double>(kTransmitMuRes - 1);
    return t * t;
}

// alpha = r^2 exactly, microfacet.cpp's alphaForRoughness at zero anisotropy: row 0 is the smooth surface itself.
double gridAlpha(int index, int resolution) {
    const double roughness = static_cast<double>(index) / static_cast<double>(resolution - 1);
    return roughness * roughness;
}

// Directional tables at the stored grid plus cosine-weighted means, each mean a Gauss-Legendre integral in mu, not a trapezoid.
void buildReflect(AlbedoTable& table, int phiNodes, int psiNodes, int muNodes) {
    const GaussLegendre phiRule = gaussLegendre(phiNodes);
    const GaussLegendre psiRule = gaussLegendre(psiNodes);
    const GaussLegendre muRule = gaussLegendre(muNodes);
    table.a.assign(static_cast<std::size_t>(kAlbedoRoughnessRes) * kAlbedoMuRes, 0.0F);
    table.b.assign(static_cast<std::size_t>(kAlbedoRoughnessRes) * kAlbedoMuRes, 0.0F);
    table.c.assign(static_cast<std::size_t>(kAlbedoRoughnessRes) * kAlbedoMuRes, 0.0F);
    table.d.assign(static_cast<std::size_t>(kAlbedoRoughnessRes) * kAlbedoMuRes, 0.0F);
    table.davg.assign(kAlbedoRoughnessRes, 0.0F);
    // One roughness row per worker, sharing no accumulator, so the result is identical to serial order: the artifact stays deterministic.
    parallelRows(kAlbedoRoughnessRes, [&](int ri) {
        const double alpha = gridAlpha(ri, kAlbedoRoughnessRes);
        for (int mi = 0; mi < kAlbedoMuRes; ++mi) {
            const Split split = reflectAlbedo(reflectMu(mi), alpha, phiRule, psiRule);
            table.a[static_cast<std::size_t>((ri * kAlbedoMuRes) + mi)] = static_cast<float>(split.a);
            table.b[static_cast<std::size_t>((ri * kAlbedoMuRes) + mi)] = static_cast<float>(split.b);
            table.c[static_cast<std::size_t>((ri * kAlbedoMuRes) + mi)] = static_cast<float>(split.c);
            table.d[static_cast<std::size_t>((ri * kAlbedoMuRes) + mi)] = static_cast<float>(deficitOf(split));
        }
        // E(0, alpha) = 1 for every alpha, analytic at the axis' endpoint; a bake-time abort, since a miss means the quadrature is wrong.
        const double grazing = static_cast<double>(table.a[static_cast<std::size_t>(ri * kAlbedoMuRes)]) +
                                static_cast<double>(table.b[static_cast<std::size_t>(ri * kAlbedoMuRes)]);
        if (!(std::abs(grazing - 1.0) < 1e-6)) {
            std::cerr << "albedo_table: roughness row " << ri << " has E(mu=0) = " << grazing
                      << ", not 1 -- the reflect quadrature does not reach its own grazing limit\n";
            std::exit(EXIT_FAILURE);
        }
        double aMean = 0.0;
        double bMean = 0.0;
        for (std::size_t k = 0; k < muRule.node.size(); ++k) {
            const double mu = muRule.node[k];
            const Split split = reflectAlbedo(mu, alpha, phiRule, psiRule);
            aMean += muRule.weight[k] * 2.0 * split.a * mu;
            bMean += muRule.weight[k] * 2.0 * split.b * mu;
        }
        table.davg[static_cast<std::size_t>(ri)] = static_cast<float>(deficitOf({aMean, bMean, 0.0}));
    });
}

// The directional tables on the escapeMu grid; each mean is its own Gauss rule in mu, panelled at the critical cosine where E kinks.
void buildTransmit(AlbedoTable& table, int nodes, int meanNodes) {
    const GaussLegendre rule = gaussLegendre(nodes);
    const GaussLegendre meanRule = gaussLegendre(meanNodes);
    const auto cells = static_cast<std::size_t>(kTransmitRoughnessRes) * kTransmitMuRes * kEtaRes;
    table.r.assign(cells, 0.0F);
    table.t.assign(cells, 0.0F);
    table.escapeDeficit.assign(cells, 0.0F);
    table.ravg.assign(static_cast<std::size_t>(kTransmitRoughnessRes) * kEtaRes, 0.0F);
    table.tavg.assign(static_cast<std::size_t>(kTransmitRoughnessRes) * kEtaRes, 0.0F);
    table.escapeAvgDeficit.assign(static_cast<std::size_t>(kTransmitRoughnessRes) * kEtaRes, 0.0F);
    parallelRows(kTransmitRoughnessRes, [&](int ri) {
        const double alpha = gridAlpha(ri, kTransmitRoughnessRes);
        for (int ei = 0; ei < kEtaRes; ++ei) {
            const double eta = etaAtIndex(ei);
            for (int mi = 0; mi < kTransmitMuRes; ++mi) {
                const EscapeSums sums = tools::quadrature::escapeAlbedo<HeightCorrelated>(escapeMu(mi), alpha, eta, rule);
                const auto cell = static_cast<std::size_t>((((ri * kTransmitMuRes) + mi) * kEtaRes) + ei);
                table.r[cell] = static_cast<float>(sums.reflect);
                table.t[cell] = static_cast<float>(sums.transmit);
                table.escapeDeficit[cell] = static_cast<float>(std::max(1.0 - sums.reflect - sums.transmit, 0.0));
            }
            // 2 int E mu dmu over [0, mu_c] and [mu_c, 1]: past the critical cosine sqrt(1 - 1/eta^2) a smooth interface stops TIR.
            const double critical = eta > 1.0 ? std::sqrt(1.0 - (1.0 / (eta * eta))) : 0.0;
            double reflect = 0.0;
            double transmit = 0.0;
            for (const auto& [lo, hi] : {std::pair{0.0, critical}, std::pair{critical, 1.0}}) {
                for (std::size_t k = 0; k < meanRule.node.size() && hi > lo; ++k) {
                    const double mu = lo + ((hi - lo) * meanRule.node[k]);
                    const EscapeSums sums = tools::quadrature::escapeAlbedo<HeightCorrelated>(mu, alpha, eta, rule);
                    reflect += meanRule.weight[k] * (hi - lo) * 2.0 * mu * sums.reflect;
                    transmit += meanRule.weight[k] * (hi - lo) * 2.0 * mu * sums.transmit;
                }
            }
            const auto mean = static_cast<std::size_t>((ri * kEtaRes) + ei);
            table.ravg[mean] = static_cast<float>(reflect);
            table.tavg[mean] = static_cast<float>(transmit);
            table.escapeAvgDeficit[mean] = static_cast<float>(std::max(1.0 - reflect - transmit, 0.0));
        }
    });
}

// Reflect-side 1 - E at a uniform-mu density node, read through the sqrt(mu) axis as directionalAlbedo does, float arithmetic included.
float deficitAtUniformMu(const AlbedoTable& table, int ri, int mi) {
    const float mf = std::sqrt(static_cast<float>(mi) / static_cast<float>(kMsReflectMuRes - 1)) * (kAlbedoMuRes - 1);
    const int m0 = std::min(static_cast<int>(mf), kAlbedoMuRes - 2);
    const float mt = mf - static_cast<float>(m0);
    const auto at = [&](int m) { return table.d[static_cast<std::size_t>((ri * kAlbedoMuRes) + m)]; };
    return at(m0) + (mt * (at(m0 + 1) - at(m0)));
}

// Sampling shape for the reflected multiple-scattering lobe, the exact (1-E)cos sampler; a smooth row takes its alpha -> 0 limit.
void buildMultipleScatteringShape(AlbedoTable& table) {
    const double step = 1.0 / (kMsReflectMuRes - 1);
    table.msDensity.assign(static_cast<std::size_t>(kAlbedoRoughnessRes) * kMsReflectMuRes, 0.0F);
    table.msCdf.assign(static_cast<std::size_t>(kAlbedoRoughnessRes) * kMsReflectMuRes, 0.0F);
    // Top down, so a row with no deficit (the mirror, whose lobe carries no energy) inherits the shape of the nearest row that has one.
    int shapeRow = -1;
    for (int ri = kAlbedoRoughnessRes - 1; ri >= 0; --ri) {
        std::vector<double> raw(kMsReflectMuRes);
        double norm = 0.0;
        for (int mi = 0; mi < kMsReflectMuRes; ++mi) {
            raw[static_cast<std::size_t>(mi)] = static_cast<double>(deficitAtUniformMu(table, ri, mi)) * mi * step;
            norm += mi > 0 ? 0.5 * (raw[static_cast<std::size_t>(mi) - 1] + raw[static_cast<std::size_t>(mi)]) * step : 0.0;
        }
        const auto row = [&](int r, int mi) { return static_cast<std::size_t>((r * kMsReflectMuRes) + mi); };
        if (!(norm > 0.0)) {
            if (shapeRow < 0) {
                std::cerr << "albedo_table: roughness row " << ri << " and every row above have no energy deficit -- the reflect table is wrong\n";
                std::exit(EXIT_FAILURE);
            }
            for (int mi = 0; mi < kMsReflectMuRes; ++mi) {
                table.msDensity[row(ri, mi)] = table.msDensity[row(shapeRow, mi)];
                table.msCdf[row(ri, mi)] = table.msCdf[row(shapeRow, mi)];
            }
            continue;
        }
        shapeRow = ri;
        double cdf = 0.0;
        for (int mi = 0; mi < kMsReflectMuRes; ++mi) {
            const double density = raw[static_cast<std::size_t>(mi)] / norm;
            if (mi > 0) {
                cdf += 0.5 * (table.msDensity[row(ri, mi) - 1] + density) * step;
            }
            table.msDensity[row(ri, mi)] = static_cast<float>(density);
            table.msCdf[row(ri, mi)] = static_cast<float>(mi == kMsReflectMuRes - 1 ? 1.0 : cdf);
        }
    }
}

// Escape deficit at a uniform-mu density node, read through the sqrt(mu) axis as microfacet.cpp's escapeAt does, float arithmetic included.
float escapeDeficitAtUniformMu(const AlbedoTable& table, int ri, int mi, int ei) {
    const float mf = std::sqrt(static_cast<float>(mi) / static_cast<float>(kTransmitMuRes - 1)) * (kTransmitMuRes - 1);
    const int m0 = std::min(static_cast<int>(mf), kTransmitMuRes - 2);
    const float mt = mf - static_cast<float>(m0);
    const auto at = [&](int m) { return table.escapeDeficit[static_cast<std::size_t>((((ri * kTransmitMuRes) + m) * kEtaRes) + ei)]; };
    return at(m0) + (mt * (at(m0 + 1) - at(m0)));
}

// Escape-deficit shape for the transmissive MS lobes, stored UNNORMALISED and uniform in mu.
void buildTransmitMultipleScatteringShape(AlbedoTable& table) {
    const double step = 1.0 / (kTransmitMuRes - 1);
    const auto size = static_cast<std::size_t>(kTransmitRoughnessRes) * kTransmitMuRes * kEtaRes;
    table.escapeShapeDensity.assign(size, 0.0F);
    table.escapeShapeCdf.assign(size, 0.0F);
    for (int ri = 0; ri < kTransmitRoughnessRes; ++ri) {
        for (int ei = 0; ei < kEtaRes; ++ei) {
            double cdf = 0.0;
            for (int mi = 0; mi < kTransmitMuRes; ++mi) {
                const auto index = static_cast<std::size_t>((((ri * kTransmitMuRes) + mi) * kEtaRes) + ei);
                const auto density = static_cast<float>(static_cast<double>(escapeDeficitAtUniformMu(table, ri, mi, ei)) * mi * step);
                if (mi > 0) {
                    // Trapezoid over the float density as emitted, not the double behind it, so the stored pair agrees at read precision.
                    cdf += 0.5 * (table.escapeShapeDensity[index - kEtaRes] + density) * step;
                }
                table.escapeShapeDensity[index] = density;
                table.escapeShapeCdf[index] = static_cast<float>(cdf);
            }
        }
    }
}


// --- Gauss rules of positive measures: E[F] = sum w_i F(x_i), exact for polynomials of degree 2n - 1 (Golub & Welsch 1969).
struct GaussRule {
    std::array<double, kKernelOrder> node{};
    std::array<double, kKernelOrder> weight{};
};

// Eigen-decomposition of a small symmetric matrix by cyclic Jacobi rotations (Golub & Van Loan 8.5), converged to double precision.
void symmetricEigen(std::array<std::array<double, kKernelOrder>, kKernelOrder>& a, std::array<std::array<double, kKernelOrder>, kKernelOrder>& v) {
    for (int i = 0; i < kKernelOrder; ++i) {
        for (int j = 0; j < kKernelOrder; ++j) {
            v[i][j] = i == j ? 1.0 : 0.0;
        }
    }
    for (int sweep = 0; sweep < 64; ++sweep) {
        double off = 0.0;
        double norm = 0.0;
        for (int i = 0; i < kKernelOrder; ++i) {
            for (int j = 0; j < kKernelOrder; ++j) {
                (j > i ? off : norm) += a[i][j] * a[i][j];
            }
        }
        // Converged once the off-diagonal mass is below double precision of the matrix's own: rotations past it only rotate rounding.
        const double epsilon = std::numeric_limits<double>::epsilon();
        if (off <= epsilon * epsilon * norm) {
            return;
        }
        for (int p = 0; p < kKernelOrder; ++p) {
            for (int q = p + 1; q < kKernelOrder; ++q) {
                if (a[p][q] == 0.0) {
                    continue;
                }
                const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                const double t = std::copysign(1.0, theta) / (std::abs(theta) + std::sqrt((theta * theta) + 1.0));
                const double c = 1.0 / std::sqrt((t * t) + 1.0);
                const double sn = t * c;
                for (int k = 0; k < kKernelOrder; ++k) {
                    const double akp = a[k][p];
                    const double akq = a[k][q];
                    a[k][p] = (c * akp) - (sn * akq);
                    a[k][q] = (sn * akp) + (c * akq);
                }
                for (int k = 0; k < kKernelOrder; ++k) {
                    const double apk = a[p][k];
                    const double aqk = a[q][k];
                    a[p][k] = (c * apk) - (sn * aqk);
                    a[q][k] = (sn * apk) + (c * aqk);
                }
                for (int k = 0; k < kKernelOrder; ++k) {
                    const double vkp = v[k][p];
                    const double vkq = v[k][q];
                    v[k][p] = (c * vkp) - (sn * vkq);
                    v[k][q] = (sn * vkp) + (c * vkq);
                }
            }
        }
    }
    std::cerr << "albedo_table: Jacobi rotations did not converge\n";
    std::exit(EXIT_FAILURE);
}

// A discrete positive measure's Gauss rule: Lanczos (Gragg & Harrod 1984) to the Jacobi matrix, nodes its eigenvalues, weights v_0^2.
GaussRule gaussRuleOf(const std::vector<double>& x, const std::vector<double>& w) {
    double total = 0.0;
    for (const double weight : w) {
        total += weight;
    }
    const std::size_t n = x.size();
    std::vector<std::vector<double>> q(1, std::vector<double>(n));
    for (std::size_t j = 0; j < n; ++j) {
        q[0][j] = std::sqrt(w[j] / total);
    }
    std::array<double, kKernelOrder> alpha{};
    std::array<double, kKernelOrder> beta{};
    int order = 0;
    for (; order < kKernelOrder; ++order) {
        std::vector<double> v(n);
        for (std::size_t j = 0; j < n; ++j) {
            v[j] = x[j] * q[static_cast<std::size_t>(order)][j];
        }
        // Two passes of Gram-Schmidt against every earlier vector: classical Lanczos alone loses orthogonality as nodes converge.
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t k = 0; k < q.size(); ++k) {
                double dot = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    dot += q[k][j] * v[j];
                }
                if (pass == 0 && k == static_cast<std::size_t>(order)) {
                    alpha[static_cast<std::size_t>(order)] = dot;
                }
                for (std::size_t j = 0; j < n; ++j) {
                    v[j] -= dot * q[k][j];
                }
            }
        }
        double norm = 0.0;
        for (const double value : v) {
            norm += value * value;
        }
        norm = std::sqrt(norm);
        // The measure's support is exhausted to double precision: the rule is exact with the nodes found so far.
        if (order + 1 == kKernelOrder || !(norm > 1e-14 * std::max(1.0, std::abs(alpha[static_cast<std::size_t>(order)])))) {
            ++order;
            break;
        }
        beta[static_cast<std::size_t>(order) + 1] = norm;
        for (double& value : v) {
            value /= norm;
        }
        q.push_back(v);
    }
    std::array<std::array<double, kKernelOrder>, kKernelOrder> jacobi{};
    std::array<std::array<double, kKernelOrder>, kKernelOrder> vectors{};
    for (int i = 0; i < order; ++i) {
        jacobi[i][i] = alpha[static_cast<std::size_t>(i)];
        if (i + 1 < order) {
            jacobi[i][i + 1] = jacobi[i + 1][i] = beta[static_cast<std::size_t>(i) + 1];
        }
    }
    symmetricEigen(jacobi, vectors);
    GaussRule rule;
    std::vector<std::pair<double, double>> nodes;
    for (int i = 0; i < order; ++i) {
        nodes.emplace_back(jacobi[i][i], total * vectors[0][i] * vectors[0][i]);
    }
    std::sort(nodes.begin(), nodes.end());
    for (int i = 0; i < kKernelOrder; ++i) {
        const auto& [node, weight] = nodes[static_cast<std::size_t>(std::min(i, order - 1))];
        rule.node[static_cast<std::size_t>(i)] = node;
        rule.weight[static_cast<std::size_t>(i)] = i < order ? weight : 0.0;
    }
    return rule;
}

// The reflect kernel's rule at (mu, alpha): alpha = 0 is the mirror, its measure the single point x = mu, E[F] = F(mu).
GaussRule kernelRule(double mu, double alpha, const GaussLegendre& phiRule, const GaussLegendre& psiRule) {
    if (alpha == 0.0) {
        GaussRule rule;
        rule.node.fill(mu);
        rule.weight[0] = 1.0;
        return rule;
    }
    std::vector<double> x;
    std::vector<double> w;
    tools::quadrature::forEachReflectNode<HeightCorrelated>(mu, alpha, phiRule, psiRule, [&](double weight, double woDotH) {
        x.push_back(woDotH);
        w.push_back(weight);
    });
    return gaussRuleOf(x, w);
}

// The kernel rules' probes: a dielectric Fresnel, smooth, and cos(6 pi x), a film's fringes.
double probeFresnel(double x) { return fresnelDielectric(static_cast<float>(x), 1.0F, 1.5F); }
double probeFringe(double x) { return std::cos(6.0 * kPi * x); }
constexpr std::array<double (*)(double), 2> kKernelProbes{probeFresnel, probeFringe};

// The worst |rule - measure| over the kernel probes at every node of the grid.
double buildKernel(AlbedoTable& table, int nodes) {
    const GaussLegendre rule = gaussLegendre(nodes);
    const auto cells = static_cast<std::size_t>(kKernelRoughnessRes) * kKernelMuRes * kKernelOrder;
    table.kernelNode.assign(cells, 0.0F);
    table.kernelWeight.assign(cells, 0.0F);
    std::vector<double> worstByRow(kKernelRoughnessRes, 0.0);
    parallelRows(kKernelRoughnessRes, [&](int ri) {
        const double alpha = gridAlpha(ri, kKernelRoughnessRes);
        for (int mi = 0; mi < kKernelMuRes; ++mi) {
            const double t = static_cast<double>(mi) / (kKernelMuRes - 1);
            const double mu = t * t;
            const GaussRule gauss = kernelRule(mu, alpha, rule, rule);
            const auto base = static_cast<std::size_t>(((ri * kKernelMuRes) + mi) * kKernelOrder);
            for (int k = 0; k < kKernelOrder; ++k) {
                table.kernelNode[base + static_cast<std::size_t>(k)] = static_cast<float>(gauss.node[static_cast<std::size_t>(k)]);
                table.kernelWeight[base + static_cast<std::size_t>(k)] = static_cast<float>(gauss.weight[static_cast<std::size_t>(k)]);
            }
            if (alpha == 0.0) {
                continue;
            }
            for (const auto probe : kKernelProbes) {
                double exact = 0.0;
                tools::quadrature::forEachReflectNode<HeightCorrelated>(mu, alpha, rule, rule, [&](double weight, double woDotH) { exact += weight * probe(woDotH); });
                double ruled = 0.0;
                for (int k = 0; k < kKernelOrder; ++k) {
                    ruled += gauss.weight[static_cast<std::size_t>(k)] * probe(gauss.node[static_cast<std::size_t>(k)]);
                }
                worstByRow[static_cast<std::size_t>(ri)] = std::max(worstByRow[static_cast<std::size_t>(ri)], std::abs(ruled - exact));
            }
        }
    });
    // The hemispherical average 2 int F(mu) mu dmu as a rule of its own, from a fine Gauss-Legendre discretisation of 2 mu dmu.
    std::vector<double> x(rule.node.begin(), rule.node.end());
    std::vector<double> w(rule.node.size());
    for (std::size_t j = 0; j < w.size(); ++j) {
        w[j] = 2.0 * rule.weight[j] * rule.node[j];
    }
    const GaussRule average = gaussRuleOf(x, w);
    table.averageNode.assign(average.node.begin(), average.node.end());
    table.averageWeight.assign(average.weight.begin(), average.weight.end());
    return *std::max_element(worstByRow.begin(), worstByRow.end());
}

// --- The anisotropic reflect kernel, integrated over the VNDF's unit square (Heitz 2018), a measure-preserving map onto D_wo.

// The anisotropy axis, uniform in 1 - sqrt(alpha_b/alpha_t), perceptual in alpha_b: nodes crowd toward a = 1, where the lobe is a groove.
double anisoGridAnisotropy(int index) {
    const double v = 1.0 - (static_cast<double>(index) / (kAnisoAnisotropyRes - 1));
    return 1.0 - (v * v);
}

// OpenPBR's anisotropic roughness: alpha_t = r^2 sqrt(2 / (1 + (1 - a)^2)), alpha_b = (1 - a) alpha_t.
glm::dvec2 anisotropicAlpha(double roughness, double anisotropy) {
    const double tangent = roughness * roughness * std::sqrt(2.0 / (1.0 + ((1.0 - anisotropy) * (1.0 - anisotropy))));
    return {tangent, (1.0 - anisotropy) * tangent};
}

// The azimuth axis node's phi: uniform in sqrt(alpha_o), alpha_o^2 = alpha_t^2 cos^2 phi + alpha_b^2 sin^2 phi, the roughness Smith sees.
double anisoNodeAzimuth(const glm::dvec2& alpha, int index) {
    const double v = static_cast<double>(index) / (kAnisoPhiRes - 1);
    const double span = std::sqrt(alpha.x) - std::sqrt(alpha.y);
    // An isotropic row's E has no azimuth: any spacing serves, and phi uniform is the one with no 0/0.
    if (!(span > 0.0)) {
        return 0.5 * kPi * v;
    }
    const double projected = std::pow(std::sqrt(alpha.x) - (v * span), 2.0);
    const double sine2 = ((alpha.x * alpha.x) - (projected * projected)) / ((alpha.x * alpha.x) - (alpha.y * alpha.y));
    return std::asin(std::sqrt(std::clamp(sine2, 0.0, 1.0)));
}

// |(ax wx, ay wy, wz)|, the Smith radical: 1 + Lambda(w) = (wz + radical) / (2 wz), anisotropic and division-free.
double anisoRadical(const glm::dvec3& w, const glm::dvec2& alpha) {
    return std::sqrt((alpha.x * alpha.x * w.x * w.x) + (alpha.y * alpha.y * w.y * w.y) + (w.z * w.z));
}

// Heitz 2018's VNDF draw in double; the stretch (ax wx, ay wy, wz) degenerates gracefully at alpha_b = 0, a one-dimensional GGX.
glm::dvec3 anisoVndf(const glm::dvec3& wo, const glm::dvec2& alpha, double u1, double u2) {
    const glm::dvec3 vh = glm::normalize(glm::dvec3(alpha.x * wo.x, alpha.y * wo.y, wo.z));
    const double lensq = (vh.x * vh.x) + (vh.y * vh.y);
    const glm::dvec3 t1 = lensq > 0.0 ? glm::dvec3(-vh.y, vh.x, 0.0) / std::sqrt(lensq) : glm::dvec3(1.0, 0.0, 0.0);
    const glm::dvec3 t2 = glm::cross(vh, t1);
    const double r = std::sqrt(u1);
    const double phi = 2.0 * kPi * u2;
    const double t1p = r * std::cos(phi);
    const double s = 0.5 * (1.0 + vh.z);
    const double t2p = ((1.0 - s) * std::sqrt(std::max(0.0, 1.0 - (t1p * t1p)))) + (s * r * std::sin(phi));
    const glm::dvec3 nh = (t1p * t1) + (t2p * t2) + (std::sqrt(std::max(0.0, 1.0 - (t1p * t1p) - (t2p * t2p))) * vh);
    return glm::normalize(glm::dvec3(alpha.x * nh.x, alpha.y * nh.y, std::max(0.0, nh.z)));
}

// The anisotropic kernel's nodes at wo: x = dot(wo, h) under weight G2/G1 = wi.z (wo.z + R_o) / (wi.z R_o + wo.z R_i), finite at wo.z = 0.
template <typename Visit>
void forEachAnisoNode(const glm::dvec3& wo, const glm::dvec2& alpha, const GaussLegendre& rule, Visit visit) {
    const double radicalO = anisoRadical(wo, alpha);
    for (std::size_t i = 0; i < rule.node.size(); ++i) {
        for (std::size_t j = 0; j < rule.node.size(); ++j) {
            const glm::dvec3 h = anisoVndf(wo, alpha, rule.node[i], rule.node[j]);
            const double woDotH = glm::dot(wo, h);
            const glm::dvec3 wi = (2.0 * woDotH * h) - wo;
            if (!(wi.z > 0.0)) {
                continue;
            }
            const double ratio = wi.z * (wo.z + radicalO) / ((wi.z * radicalO) + (wo.z * anisoRadical(wi, alpha)));
            visit(rule.weight[i] * rule.weight[j] * ratio, woDotH);
        }
    }
}

// The anisotropic tables at squareNodes^2 VNDF nodes per cell, and each (roughness, anisotropy) mean by (muNodes, phiNodes) Gauss rules.
void buildAnisotropicKernel(AlbedoTable& table, int squareNodes, int muNodes, int phiNodes) {
    const GaussLegendre square = gaussLegendre(squareNodes);
    const GaussLegendre muRule = gaussLegendre(muNodes);
    const GaussLegendre phiRule = gaussLegendre(phiNodes);
    const auto cellCount = static_cast<std::size_t>(kAnisoRoughnessRes) * kAnisoAnisotropyRes * kAnisoMuRes * kAnisoPhiRes;
    table.anisoNode.assign(cellCount * kKernelOrder, 0.0F);
    table.anisoWeight.assign(cellCount * kKernelOrder, 0.0F);
    table.anisoDeficit.assign(cellCount, 0.0F);
    table.anisoB.assign(cellCount, 0.0F);
    table.anisoC.assign(cellCount, 0.0F);
    table.anisoDeficitAvg.assign(static_cast<std::size_t>(kAnisoRoughnessRes) * kAnisoAnisotropyRes, 0.0F);
    table.anisoProbe.assign(cellCount * kKernelProbes.size(), 0.0);
    std::vector<double> excessByRow(static_cast<std::size_t>(kAnisoRoughnessRes) * kAnisoAnisotropyRes, 0.0);
    parallelRows(kAnisoRoughnessRes * kAnisoAnisotropyRes, [&](int row) {
        const int ri = row / kAnisoAnisotropyRes;
        const int ai = row % kAnisoAnisotropyRes;
        const glm::dvec2 alpha = anisotropicAlpha(static_cast<double>(ri) / (kAnisoRoughnessRes - 1), anisoGridAnisotropy(ai));
        double& excess = excessByRow[static_cast<std::size_t>(row)];
        // E's raw measure at (mu, phi): its total, and, where asked, its Gauss rule, F82's split (b, c) and the probes' expectations.
        const auto measureAt = [&](double mu, double phi, GaussRule* rule, double* split, double* probes) {
            const double sine = std::sqrt(std::max(0.0, 1.0 - (mu * mu)));
            const glm::dvec3 wo(sine * std::cos(phi), sine * std::sin(phi), mu);
            // The smooth surface is its mirror: one node at mu, E[F] = F(mu), no deficit.
            if (alpha.x == 0.0) {
                if (rule != nullptr) {
                    rule->node.fill(mu);
                    rule->weight.fill(0.0);
                    rule->weight[0] = 1.0;
                    split[0] = std::pow(1.0 - mu, 5.0);
                    split[1] = mu * std::pow(1.0 - mu, 6.0);
                    for (std::size_t p = 0; p < kKernelProbes.size(); ++p) {
                        probes[p] = kKernelProbes[p](mu);
                    }
                }
                return 1.0;
            }
            std::vector<double> x;
            std::vector<double> w;
            double total = 0.0;
            forEachAnisoNode(wo, alpha, square, [&](double weight, double woDotH) {
                total += weight;
                if (rule != nullptr) {
                    x.push_back(woDotH);
                    w.push_back(weight);
                    const double m = std::clamp(1.0 - woDotH, 0.0, 1.0);
                    split[0] += weight * std::pow(m, 5.0);
                    split[1] += weight * woDotH * std::pow(m, 6.0);
                    for (std::size_t p = 0; p < kKernelProbes.size(); ++p) {
                        probes[p] += weight * kKernelProbes[p](woDotH);
                    }
                }
            });
            if (rule != nullptr) {
                *rule = gaussRuleOf(x, w);
            }
            excess = std::max(excess, total - 1.0);
            return total;
        };
        // A passive microsurface reflects at most what it receives: E past 1 is quadrature error, asserted within the residual by main.
        const auto deficitOfTotal = [](double total) { return std::max(1.0 - total, 0.0); };
        for (int mi = 0; mi < kAnisoMuRes; ++mi) {
            const double t = static_cast<double>(mi) / (kAnisoMuRes - 1);
            for (int pi = 0; pi < kAnisoPhiRes; ++pi) {
                const double phi = anisoNodeAzimuth(alpha, pi);
                const auto cell = static_cast<std::size_t>((((((ri * kAnisoAnisotropyRes) + ai) * kAnisoMuRes) + mi) * kAnisoPhiRes) + pi);
                GaussRule rule;
                std::array<double, 2> split{};
                table.anisoDeficit[cell] =
                    static_cast<float>(deficitOfTotal(measureAt(t * t, phi, &rule, split.data(), &table.anisoProbe[cell * kKernelProbes.size()])));
                table.anisoB[cell] = static_cast<float>(split[0]);
                table.anisoC[cell] = static_cast<float>(split[1]);
                for (int k = 0; k < kKernelOrder; ++k) {
                    table.anisoNode[(cell * kKernelOrder) + static_cast<std::size_t>(k)] = static_cast<float>(rule.node[static_cast<std::size_t>(k)]);
                    table.anisoWeight[(cell * kKernelOrder) + static_cast<std::size_t>(k)] = static_cast<float>(rule.weight[static_cast<std::size_t>(k)]);
                }
            }
        }
        // (1/pi) int (1 - E) cos dw over the hemisphere: four symmetric quadrants, each 2 int_0^{pi/2} int_0^1 (1 - E) mu dmu dphi / pi.
        double mean = 0.0;
        for (std::size_t p = 0; p < phiRule.node.size(); ++p) {
            for (std::size_t m = 0; m < muRule.node.size(); ++m) {
                const double mu = muRule.node[m];
                mean += phiRule.weight[p] * muRule.weight[m] * 2.0 * mu * deficitOfTotal(measureAt(mu, 0.5 * kPi * phiRule.node[p], nullptr, nullptr, nullptr));
            }
        }
        table.anisoDeficitAvg[static_cast<std::size_t>(row)] = static_cast<float>(mean);
    });
    table.anisoExcess = *std::max_element(excessByRow.begin(), excessByRow.end());
}

// The anisotropic tables against a rebake at doubled rules: deficits, means, and each cell's rule against the doubled measure's probes.
double verifyAnisotropic(const AlbedoTable& table) {
    AlbedoTable reference;
    buildAnisotropicKernel(reference, 2 * kAnisoSquareNodes, 2 * kAnisoMeanMuNodes, 2 * kAnisoMeanPhiNodes);
    const auto worstOf = [](const std::vector<float>& shipped, const std::vector<float>& exact) {
        double worst = 0.0;
        for (std::size_t i = 0; i < shipped.size(); ++i) {
            worst = std::max(worst, std::abs(static_cast<double>(shipped[i]) - static_cast<double>(exact[i])));
        }
        return worst;
    };
    double rule = 0.0;
    for (std::size_t cell = 0; cell < table.anisoDeficit.size(); ++cell) {
        for (std::size_t p = 0; p < kKernelProbes.size(); ++p) {
            double ruled = 0.0;
            for (std::size_t k = 0; k < static_cast<std::size_t>(kKernelOrder); ++k) {
                const std::size_t at = (cell * kKernelOrder) + k;
                ruled += static_cast<double>(table.anisoWeight[at]) * kKernelProbes[p](static_cast<double>(table.anisoNode[at]));
            }
            rule = std::max(rule, std::abs(ruled - reference.anisoProbe[(cell * kKernelProbes.size()) + p]));
        }
    }
    const double deficit = worstOf(table.anisoDeficit, reference.anisoDeficit);
    const double split = std::max(worstOf(table.anisoB, reference.anisoB), worstOf(table.anisoC, reference.anisoC));
    const double mean = worstOf(table.anisoDeficitAvg, reference.anisoDeficitAvg);
    std::cout << "albedo_table: anisotropic deficit residual " << deficit << ", F82 split " << split << ", mean " << mean << ", rule probes "
              << rule << " against " << 2 * kAnisoSquareNodes << "^2 nodes; largest E - 1 " << table.anisoExcess << "\n";
    return std::max({deficit, split, mean, rule});
}

// Largest disagreement between the shipped reflect rule and one at doubled order, over the grid and means: the rule's own measured error.
struct Residual {
    double value;
    const char* channel;
    int roughnessIndex;
    int muIndex;
};

Residual verifyReflect(const AlbedoTable& table, int phiNodes, int psiNodes, int muNodes) {
    AlbedoTable reference;
    buildReflect(reference, phiNodes * 2, psiNodes * 2, muNodes * 2);
    Residual worst{0.0, "a", 0, 0};
    const std::array<std::tuple<const char*, const std::vector<float>*, const std::vector<float>*>, 5>
        channels = {{{"a", &table.a, &reference.a},
                     {"b", &table.b, &reference.b},
                     {"c", &table.c, &reference.c},
                     {"d", &table.d, &reference.d},
                     {"davg", &table.davg, &reference.davg}}};
    for (const auto& [name, shipped, exact] : channels) {
        Residual channelWorst{0.0, name, 0, 0};
        for (std::size_t i = 0; i < shipped->size(); ++i) {
            const double delta = std::abs(static_cast<double>((*shipped)[i]) -
                                           static_cast<double>((*exact)[i]));
            if (delta > channelWorst.value) {
                const int stride = shipped->size() == table.davg.size() ? 1 : kAlbedoMuRes;
                channelWorst = {delta, name, static_cast<int>(i) / stride,
                                 stride == 1 ? -1 : static_cast<int>(i) % kAlbedoMuRes};
            }
        }
        std::cout << "albedo_table: " << name << " residual " << channelWorst.value
                  << " at roughnessIndex " << channelWorst.roughnessIndex << " muIndex "
                  << channelWorst.muIndex << "\n";
        if (channelWorst.value > worst.value) {
            worst = channelWorst;
        }
    }
    return worst;
}

// Largest disagreement between the shipped transmit tables and a rebake at twice the nodes per axis: the quadrature's own error.
double verifyTransmit(const AlbedoTable& table, int nodes, int meanNodes) {
    AlbedoTable reference;
    buildTransmit(reference, nodes * 2, meanNodes * 2);
    double worst = 0.0;
    const std::array<std::pair<const char*, std::pair<const std::vector<float>*, const std::vector<float>*>>, 6>
        channels = {{{"r", {&table.r, &reference.r}},
                     {"t", {&table.t, &reference.t}},
                     {"deficit", {&table.escapeDeficit, &reference.escapeDeficit}},
                     {"ravg", {&table.ravg, &reference.ravg}},
                     {"tavg", {&table.tavg, &reference.tavg}},
                     {"avg deficit", {&table.escapeAvgDeficit, &reference.escapeAvgDeficit}}}};
    for (const auto& [name, pair] : channels) {
        double channelWorst = 0.0;
        std::size_t at = 0;
        for (std::size_t i = 0; i < pair.first->size(); ++i) {
            const double delta = std::abs(static_cast<double>((*pair.first)[i]) - static_cast<double>((*pair.second)[i]));
            if (delta > channelWorst) {
                channelWorst = delta;
                at = i;
            }
        }
        // Directional channels are [roughness][mu][eta]; the means drop the mu axis.
        const bool directional = pair.first->size() == table.r.size();
        const std::size_t muStride = directional ? kTransmitMuRes : 1;
        std::cout << "albedo_table: " << name << " residual " << channelWorst << " against " << nodes * 2
                  << " nodes at roughnessIndex " << at / (muStride * kEtaRes) << " muIndex "
                  << (directional ? static_cast<long>((at / kEtaRes) % kTransmitMuRes) : -1L) << " etaIndex "
                  << at % kEtaRes << "\n";
        worst = std::max(worst, channelWorst);
    }
    return worst;
}

// --- The interface's multiple scattering by Smith random walks (Heitz, Hanika, d'Eon & Dachsbacher 2016), height-uniform microsurface.

// Lambda(w) = (radical / w.z - 1) / 2 for any w.z != 0: negative below the horizon, where the walk descends.
double walkLambda(const glm::dvec3& w, double alpha) {
    return 0.5 * ((std::sqrt((w.z * w.z) + (alpha * alpha * ((w.x * w.x) + (w.y * w.y)))) / w.z) - 1.0);
}

// The next intersection height along w from h, C1(h) = (h + 1)/2; infinity where the ray leaves, G1 = C1(h)^Lambda (Heitz 2016 eq. 9).
double walkHeight(const glm::dvec3& w, double h, double alpha, double u) {
    // A horizontal ray keeps its height, the Smith microsurface being statistically flat.
    if (w.z == 0.0) {
        return h;
    }
    const double c1 = std::clamp(0.5 * (h + 1.0), 0.0, 1.0);
    const double lambda = walkLambda(w, alpha);
    // Only a rising ray can leave, with probability G1; a falling one always meets the surface below.
    if (w.z > 0.0 && u > 1.0 - std::pow(c1, lambda)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::clamp((2.0 * c1 * std::pow(1.0 - u, -1.0 / lambda)) - 1.0, -1.0, 1.0);
}

// Visible normals for wi anywhere off the downward pole (Dupuy & Benyoub 2023): a spherical cap in the stretched configuration.
glm::dvec3 walkVisibleNormal(const glm::dvec3& wi, double alpha, double u1, double u2) {
    const glm::dvec3 stretched = glm::normalize(glm::dvec3(alpha * wi.x, alpha * wi.y, wi.z));
    const double phi = 2.0 * kPi * u1;
    const double z = ((1.0 - u2) * (1.0 + stretched.z)) - stretched.z;
    const double sine = std::sqrt(std::clamp(1.0 - (z * z), 0.0, 1.0));
    const glm::dvec3 h = glm::dvec3(sine * std::cos(phi), sine * std::sin(phi), z) + stretched;
    return glm::normalize(glm::dvec3(alpha * h.x, alpha * h.y, h.z));
}

// One walk from incident wi (z > 0) through an interface of ratio eta = etaI/etaT: its exit side and scattering order.
struct WalkExit {
    bool reflected;
    int order;
};

WalkExit randomWalk(const glm::dvec3& wi, double alpha, double eta, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    glm::dvec3 ray = -wi;
    double height = 1.0;
    bool outside = true;
    int order = 0;
    while (true) {
        // Inside, the walk is the outside one mirrored through the mean plane: heights and directions negate.
        const double next = outside ? walkHeight(ray, height, alpha, uniform(rng)) : -walkHeight(-ray, -height, alpha, uniform(rng));
        if (std::isinf(next)) {
            return {outside, order};
        }
        height = next;
        ++order;
        const glm::dvec3 toward = -ray;
        const glm::dvec3 normal = outside ? walkVisibleNormal(toward, alpha, uniform(rng), uniform(rng))
                                          : -walkVisibleNormal(-toward, alpha, uniform(rng), uniform(rng));
        const double ratio = outside ? eta : 1.0 / eta;
        const double cosine = glm::dot(toward, normal);
        const double fresnel = fresnelDielectric(static_cast<float>(cosine), static_cast<float>(ratio), 1.0F);
        if (uniform(rng) < fresnel) {
            ray = (2.0 * cosine * normal) - toward;
        } else {
            const double cos2T = (1.0 - ((1.0 - (cosine * cosine)) * ratio * ratio));
            ray = glm::normalize((((ratio * cosine) - std::sqrt(std::max(cos2T, 0.0))) * normal) - (ratio * toward));
            outside = !outside;
        }
    }
}

// Cosine-weighted walks per (roughness, eta) cell on jittered incidence: order 1 must reproduce ravg/tavg, 2+ give the interface's split.
double buildEscapeWalk(AlbedoTable& table, int strata) {
    const auto cells = static_cast<std::size_t>(kTransmitRoughnessRes) * kEtaRes;
    table.walkReflect.assign(cells, 0.0F);
    table.walkTransmit.assign(cells, 0.0F);
    std::vector<double> worstZ(kTransmitRoughnessRes, 0.0);
    std::vector<double> worstNoise(kTransmitRoughnessRes, 0.0);
    parallelRows(kTransmitRoughnessRes, [&](int ri) {
        const double alpha = gridAlpha(ri, kTransmitRoughnessRes);
        for (int ei = 0; ei < kEtaRes; ++ei) {
            const auto cell = static_cast<std::size_t>((ri * kEtaRes) + ei);
            std::mt19937_64 rng((static_cast<std::uint64_t>(ri) << 32U) | static_cast<std::uint64_t>(ei));
            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            std::array<double, 4> counts{};  // order-1 reflect, order-1 transmit, 2+ reflect, 2+ transmit
            for (int i = 0; i < strata; ++i) {
                for (int j = 0; j < strata; ++j) {
                    const double u1 = (i + uniform(rng)) / strata;
                    const double u2 = (j + uniform(rng)) / strata;
                    const double radius = std::sqrt(u1);
                    const glm::dvec3 wi(radius * std::cos(2.0 * kPi * u2), radius * std::sin(2.0 * kPi * u2), std::sqrt(1.0 - u1));
                    const WalkExit exit = randomWalk(wi, alpha, etaAtIndex(ei), rng);
                    counts[static_cast<std::size_t>((exit.order > 1 ? 2 : 0) + (exit.reflected ? 0 : 1))] += 1.0;
                }
            }
            const double walks = static_cast<double>(strata) * strata;
            table.walkReflect[cell] = static_cast<float>(counts[2] / walks);
            table.walkTransmit[cell] = static_cast<float>(counts[3] / walks);
            // Binomial standard errors, conservative under stratification, for the order-1 means against the quadrature's.
            for (const auto& [count, exact] : {std::pair{counts[0], table.ravg[cell]}, std::pair{counts[1], table.tavg[cell]}}) {
                const double p = count / walks;
                const double sigma = std::sqrt(std::max(p * (1.0 - p), 1.0 / walks) / walks);
                worstZ[static_cast<std::size_t>(ri)] = std::max(worstZ[static_cast<std::size_t>(ri)], std::abs(p - exact) / sigma);
            }
            for (const double count : {counts[2], counts[3]}) {
                const double p = count / walks;
                worstNoise[static_cast<std::size_t>(ri)] = std::max(worstNoise[static_cast<std::size_t>(ri)], std::sqrt(p * (1.0 - p) / walks));
            }
        }
    });
    const double z = *std::max_element(worstZ.begin(), worstZ.end());
    const double noise = *std::max_element(worstNoise.begin(), worstNoise.end());
    std::cout << "albedo_table: random walks, " << strata * strata << " per cell; order-1 vs quadrature worst |z| " << z
              << ", 2+ order standard error at most " << noise << "\n";
    return z;
}

// %.9g is FLT_DECIMAL_DIG, round-tripping float32 exactly; it drops the point on a whole number, so "1" becomes "1.0F".
std::string floatLiteral(float value) {
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.9g", static_cast<double>(value));
    const std::string literal(buffer.data());
    return literal + (literal.find_first_of(".e") == std::string::npos ? ".0F" : "F");
}

void writeArray(std::ofstream& out, const char* name, const std::vector<float>& values) {
    out << "\nconstexpr std::array<float, " << values.size() << "> " << name << " = {{";
    for (std::size_t i = 0; i < values.size(); ++i) {
        out << (i % 8 == 0 ? "\n    " : " ") << floatLiteral(values[i]) << ",";
    }
    out << "\n}};\n";
}

bool writeInc(const std::string& path, const AlbedoTable& table, double residual, int transmitNodes, double transmitResidual,
              double walkZ, double kernelResidual, double anisoResidual) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "albedo_table: cannot write " << path << "\n";
        return false;
    }
    const std::string etaMin = floatLiteral(static_cast<float>(kEtaMin));
    const std::string etaMax = floatLiteral(static_cast<float>(kEtaMax));
    out << "// Generated by tools/albedo_table.cpp, do not edit; regenerate with ./build/albedo_table --out src/scene/albedo_table.inc\n"
           "// Kulla-Conty energy tables (Kulla & Conty 2017) over perceptual roughness, edge-aligned so r = 0 and mu = 1 are exact nodes.\n"
           "// Every directional mu axis is uniform in sqrt(mu), nodes crowding where E climbs over mu ~ alpha; the lookups index sqrt(mu).\n"
           "// The multiple-scattering shapes are uniform in mu, where their piecewise-linear inversion has one step width.\n"
           "// Roughness 0 is the smooth surface, tabulated analytically: E = F(mu), R = F, T = 1 - F.\n"
           "// kAlbedo*: F82's E = F0 a + b - k c, exact in (F0, k); node mu = 0 has E = 1, asserted per row. Residual "
        << residual << " vs a doubled rule.\n"
           "// kAlbedoDeficit, kEscapeDeficit and their means are 1 - E and 1 - R - T formed in double, never cancelled in float.\n"
           "// kEscape*: the dielectric's R, T and deficit over (roughness, mu, log eta) at "
        << transmitNodes << " nodes per panel, residual " << transmitResidual << " vs a doubled rule.\n"
           "// Refraction's shadowing is height-correlated Smith's B(1 + Lambda_o, 1 + Lambda_i) (Heitz 2014 sec. 6), not reflection's.\n"
           "// kEscapeWalkReflect/Transmit: 2+ bounce energy leaving each side by Smith random walks (Heitz et al. 2016), order 1 at |z| "
        << walkZ << ".\n"
           "// kMsReflectDensity/Cdf: the conductor's (1 - E(mu)) mu lobe as a normalised piecewise-linear density with exact prefix sums.\n"
           "// kEscapeShapeDensity/Cdf: the escape deficit's shape, unnormalised so a blend of four rows divides by its own blended total.\n"
           "// kKernelNode/Weight: Gauss rules of order " << kKernelOrder << " for E[F] over x = dot(wo, h), any Fresnel; probe residual "
        << kernelResidual << ".\n"
           "// kAverageNode/Weight: the Gauss rule of 2 mu dmu on [0, 1], the hemispherical average of a Fresnel with no closed form.\n"
           "// kAniso*: rules, deficit, F82 split and mean over (r, 1 - sqrt(1 - a), sqrt(mu), sqrt(alpha_o)); residual "
        << anisoResidual << " vs a doubled rule.\n";
    out << "\nconstexpr int kAlbedoRoughnessRes = " << kAlbedoRoughnessRes << ";\n"
        << "constexpr int kAlbedoMuRes = " << kAlbedoMuRes << ";\n"
        << "constexpr int kMsReflectMuRes = " << kMsReflectMuRes << ";\n"
        << "constexpr int kTransmitRoughnessRes = " << kTransmitRoughnessRes << ";\n"
        << "constexpr int kTransmitMuRes = " << kTransmitMuRes << ";\n"
        << "constexpr int kEtaRes = " << kEtaRes << ";\n"
        << "constexpr float kEtaMin = " << etaMin << ";\n"
        << "constexpr float kEtaMax = " << etaMax << ";\n"
        << "constexpr int kKernelRoughnessRes = " << kKernelRoughnessRes << ";\n"
        << "constexpr int kKernelMuRes = " << kKernelMuRes << ";\n"
        << "constexpr int kKernelOrder = " << kKernelOrder << ";\n"
        << "constexpr int kAnisoRoughnessRes = " << kAnisoRoughnessRes << ";\n"
        << "constexpr int kAnisoAnisotropyRes = " << kAnisoAnisotropyRes << ";\n"
        << "constexpr int kAnisoMuRes = " << kAnisoMuRes << ";\n"
        << "constexpr int kAnisoPhiRes = " << kAnisoPhiRes << ";\n";
    writeArray(out, "kAlbedoA", table.a);
    writeArray(out, "kAlbedoB", table.b);
    writeArray(out, "kAlbedoC", table.c);
    writeArray(out, "kAlbedoDeficit", table.d);
    writeArray(out, "kAlbedoAvgDeficit", table.davg);
    writeArray(out, "kEscapeReflect", table.r);
    writeArray(out, "kEscapeTransmit", table.t);
    writeArray(out, "kEscapeDeficit", table.escapeDeficit);
    writeArray(out, "kEscapeWalkReflect", table.walkReflect);
    writeArray(out, "kEscapeWalkTransmit", table.walkTransmit);
    writeArray(out, "kEscapeAvgDeficit", table.escapeAvgDeficit);
    writeArray(out, "kMsReflectDensity", table.msDensity);
    writeArray(out, "kMsReflectCdf", table.msCdf);
    writeArray(out, "kEscapeShapeDensity", table.escapeShapeDensity);
    writeArray(out, "kEscapeShapeCdf", table.escapeShapeCdf);
    writeArray(out, "kKernelNode", table.kernelNode);
    writeArray(out, "kKernelWeight", table.kernelWeight);
    writeArray(out, "kAverageNode", table.averageNode);
    writeArray(out, "kAverageWeight", table.averageWeight);
    writeArray(out, "kAnisoNode", table.anisoNode);
    writeArray(out, "kAnisoWeight", table.anisoWeight);
    writeArray(out, "kAnisoDeficit", table.anisoDeficit);
    writeArray(out, "kAnisoB", table.anisoB);
    writeArray(out, "kAnisoC", table.anisoC);
    writeArray(out, "kAnisoDeficitAvg", table.anisoDeficitAvg);
    return out.good();
}

}  // namespace

int main(int argc, char** argv) {
    std::string outPath = "src/scene/albedo_table.inc";
    // At 96 the mean channels agree to 1.1e-6 and the directional ones to 3.0e-5 against the doubled rule, worst at the mu = 0 column.
    int phiNodes = 96;
    int psiNodes = 96;
    int muNodes = 96;
    int transmitNodes = kTransmitNodes;
    for (int i = 1; i < argc; ++i) {
        const bool hasValue = i + 1 < argc;
        if (std::strcmp(argv[i], "--out") == 0 && hasValue) {
            outPath = argv[++i];
        } else if (std::strcmp(argv[i], "--nodes") == 0 && hasValue) {
            phiNodes = psiNodes = muNodes = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--transmit-nodes") == 0 && hasValue) {
            transmitNodes = std::atoi(argv[++i]);
        } else {
            std::cerr << "albedo_table: unknown or incomplete argument '" << argv[i]
                      << "'\nusage: albedo_table [--out path.inc] [--nodes N] [--transmit-nodes N]\n";
            return EXIT_FAILURE;
        }
    }
    if (phiNodes < 2 || transmitNodes < 2) {
        std::cerr << "albedo_table: --nodes and --transmit-nodes must be at least 2\n";
        return EXIT_FAILURE;
    }

    AlbedoTable table;
    buildReflect(table, phiNodes, psiNodes, muNodes);
    const Residual residual = verifyReflect(table, phiNodes, psiNodes, muNodes);
    buildTransmit(table, transmitNodes, kTransmitMeanNodes);
    const double transmitResidual = verifyTransmit(table, transmitNodes, kTransmitMeanNodes);
    // Order-1 walks must match the quadrature: a two-sided Bonferroni bound at a 1-in-1000 false alarm over every cell's two means.
    const double walkZ = buildEscapeWalk(table, kWalkStrata);
    const double perTest = 1e-3 / (2.0 * kTransmitRoughnessRes * kEtaRes);
    double zLow = 0.0;
    double zHigh = 40.0;
    while (zHigh - zLow > 1e-9) {
        const double mid = 0.5 * (zLow + zHigh);
        (std::erfc(mid / std::sqrt(2.0)) > perTest ? zLow : zHigh) = mid;
    }
    if (walkZ > zHigh) {
        std::cerr << "albedo_table: the random walk's single scattering departs from the quadrature at |z| " << walkZ << " > " << zHigh << "\n";
        return EXIT_FAILURE;
    }
    buildMultipleScatteringShape(table);
    buildTransmitMultipleScatteringShape(table);
    const double kernelResidual = buildKernel(table, phiNodes);
    std::cout << "albedo_table: kernel Gauss rules of order " << kKernelOrder << ", worst probe residual " << kernelResidual << "\n";
    buildAnisotropicKernel(table, kAnisoSquareNodes, kAnisoMeanMuNodes, kAnisoMeanPhiNodes);
    const double anisoResidual = verifyAnisotropic(table);
    if (table.anisoExcess > anisoResidual) {
        std::cerr << "albedo_table: anisotropic E exceeds 1 by " << table.anisoExcess << ", past the quadrature residual " << anisoResidual << "\n";
        return EXIT_FAILURE;
    }
    if (!writeInc(outPath, table, residual.value, transmitNodes, transmitResidual, walkZ, kernelResidual, anisoResidual)) {
        return EXIT_FAILURE;
    }
    std::cout << "albedo_table: wrote " << outPath << " (reflect " << kAlbedoRoughnessRes << "x" << kAlbedoMuRes
              << " at " << phiNodes << " nodes, residual " << residual.value << " in " << residual.channel
              << " at roughnessIndex " << residual.roughnessIndex << " muIndex " << residual.muIndex
              << "; transmit " << kTransmitRoughnessRes << "x" << kTransmitMuRes << "x" << kEtaRes << " at "
              << transmitNodes << " nodes, residual " << transmitResidual << ")\n";
    return EXIT_SUCCESS;
}
