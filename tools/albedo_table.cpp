// Offline generator for src/scene/albedo_table.inc, the Kulla-Conty energy tables.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <atomic>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/scene/fresnel_dielectric.h"

namespace {

// The shading path's own dielectric interface, so the table is baked against exactly what reads it.
using pathtracer::scene::cos2Transmitted;
using pathtracer::scene::fresnelDielectric;

constexpr double kPi = 3.14159265358979323846;

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

// The reflection kernel's Gauss rules over x = dot(wo, h): E[F] for any Fresnel of x, the thin film's among them, on a coarse grid.
constexpr int kKernelRoughnessRes = 32;
constexpr int kKernelMuRes = 32;
constexpr int kKernelOrder = 8;

double smithRadical(double cosTheta, double alpha) {
    const double alpha2 = alpha * alpha;
    return std::sqrt(alpha2 + ((1.0 - alpha2) * cosTheta * cosTheta));
}

// Height-correlated G2 over cosO, 2 cosI/(cosI s(cosO) + cosO s(cosI)): no cosine divides, so it holds to cosO = 0, where it is 2/alpha.
double smithG2OverCosO(double cosO, double cosI, double alpha) {
    return 2.0 * cosI / ((cosI * smithRadical(cosO, alpha)) + (cosO * smithRadical(cosI, alpha)));
}

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

// Gauss-Legendre nodes/weights on [0,1] by Newton on P_n through Bonnet's recurrence (Numerical Recipes 3rd ed. 4.6.1); weights sum to 1.
struct GaussLegendre {
    std::vector<double> node;
    std::vector<double> weight;
};

GaussLegendre gaussLegendre(int n) {
    GaussLegendre quadrature{std::vector<double>(static_cast<std::size_t>(n)),
                              std::vector<double>(static_cast<std::size_t>(n))};
    for (int i = 0; i < n; ++i) {
        double x = std::cos(kPi * (i + 0.75) / (n + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double p0 = 1.0;
            double p1 = 0.0;
            for (int k = 0; k < n; ++k) {
                const double p2 = p1;
                p1 = p0;
                p0 = ((((2.0 * k) + 1.0) * x * p1) - (k * p2)) / (k + 1.0);
            }
            derivative = n * ((x * p0) - p1) / ((x * x) - 1.0);
            const double step = p0 / derivative;
            x -= step;
            if (std::abs(step) <= 1e-16) {
                break;
            }
        }
        quadrature.node[static_cast<std::size_t>(i)] = 0.5 * (1.0 - x);
        quadrature.weight[static_cast<std::size_t>(i)] =
            1.0 / ((1.0 - (x * x)) * derivative * derivative);
    }
    return quadrature;
}

// Every node of the reflect measure at (mu, alpha > 0): visit(weight, dot(wo, h)), the weight carrying D, G2 and the Jacobian.
template <typename Visit>
void forEachReflectNode(double mu, double alpha, const GaussLegendre& phiRule, const GaussLegendre& psiRule, Visit visit) {
    const double sinTv = std::sqrt(std::max(0.0, 1.0 - (mu * mu)));
    // phi is even about 0, so half the circle is integrated and doubled; the panels meet at pi/2, resolving the |cos phi| < mu layer.
    for (int panel = 0; panel < 2; ++panel) {
        const double phiBase = 0.5 * kPi * panel;
        for (std::size_t p = 0; p < phiRule.node.size(); ++p) {
            const double horizontal = sinTv * std::cos(phiBase + (0.5 * kPi * phiRule.node[p]));
            const double radius = std::sqrt((horizontal * horizontal) + (mu * mu));
            const double delta = std::atan2(horizontal, mu);
            const double psiMax = std::atan(std::tan(0.5 * (delta + (0.5 * kPi))) / alpha);
            for (std::size_t s = 0; s < psiRule.node.size(); ++s) {
                const double psi = psiMax * psiRule.node[s];
                const double thetaH = std::atan(alpha * std::tan(psi));
                const double woDotH = radius * std::cos(thetaH - delta);
                const double wiZ = radius * std::cos((2.0 * thetaH) - delta);
                visit(phiRule.weight[p] * psiRule.weight[s] * psiMax * (woDotH / std::cos(thetaH)) * smithG2OverCosO(mu, wiZ, alpha) *
                          std::sin(psi) * std::cos(psi),
                      woDotH);
            }
        }
    }
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
    forEachReflectNode(mu, alpha, phiRule, psiRule, [&](double weight, double woDotH) {
        const double fc = std::pow(std::clamp(1.0 - woDotH, 0.0, 1.0), 5.0);
        a += weight * (1.0 - fc);
        b += weight * fc;
        c += weight * woDotH * std::pow(std::clamp(1.0 - woDotH, 0.0, 1.0), 6.0);
    });
    // The 1/mu is inside smithG2OverCosO, which lets mu = 0 be a node; never 0/0, psiMax being unreachable on panel one and 0 on panel two.
    return {a, b, c};
}

// --- Transmit side: the reflect measure panelled at the interface's boundaries; a VNDF midpoint rule lumps the slope tail in, 3.6e-3.

// log-spaced so eta and 1/eta are symmetric about index kEtaRes/2.
double etaAtIndex(int index) {
    const double u = static_cast<double>(index) / static_cast<double>(kEtaRes - 1);
    return std::exp(std::log(kEtaMin) + (u * (std::log(kEtaMax) - std::log(kEtaMin))));
}

struct EscapeSums {
    std::array<double, kEtaRes> reflect;
    std::array<double, kEtaRes> transmit;
};

// Escaping fraction of a dielectric interface, reflected and transmitted shares: exact Fresnel, 1.0 inside TIR where Schlick reads ~0.1.
EscapeSums escapeAlbedo(double mu, double alpha, const GaussLegendre& rule) {
    EscapeSums sums{};
    // alpha = 0 is the smooth interface: it reflects F(mu) and transmits the rest, F being 1 past the critical angle.
    if (alpha == 0.0) {
        for (int ei = 0; ei < kEtaRes; ++ei) {
            const double fresnel = fresnelDielectric(static_cast<float>(mu), static_cast<float>(etaAtIndex(ei)), 1.0F);
            sums.reflect[static_cast<std::size_t>(ei)] = fresnel;
            sums.transmit[static_cast<std::size_t>(ei)] = 1.0 - fresnel;
        }
        return sums;
    }
    const double sinTv = std::sqrt(std::max(0.0, 1.0 - (mu * mu)));
    const glm::dvec3 wo(sinTv, 0.0, mu);
    for (int ei = 0; ei < kEtaRes; ++ei) {
        const auto e = static_cast<std::size_t>(ei);
        const auto eta = static_cast<float>(etaAtIndex(ei));
        // wo.h below which a facet totally internally reflects; zero when entering, where there is no cone.
        const double criticalCos = eta > 1.0F ? std::sqrt(1.0 - (1.0 / (static_cast<double>(eta) * eta))) : 0.0;
        // phi even about 0, so half the circle is doubled; split at pi/2 and the TIR tangency, where 1-F is sqrt-singular (3.9e-3 without).
        std::array<double, 5> phiBreaks{0.0, 0.5 * kPi, kPi, 0.0, 0.0};
        int phiCount = 3;
        if (criticalCos > mu && sinTv > 0.0) {
            const double tangent = std::acos(std::sqrt((criticalCos * criticalCos) - (mu * mu)) / sinTv);
            phiBreaks[3] = tangent;
            phiBreaks[4] = kPi - tangent;
            phiCount = 5;
        }
        std::sort(phiBreaks.begin(), phiBreaks.begin() + phiCount);
        for (int phiPanel = 0; phiPanel + 1 < phiCount; ++phiPanel) {
            const double phiLo = phiBreaks[static_cast<std::size_t>(phiPanel)];
            const double phiHi = phiBreaks[static_cast<std::size_t>(phiPanel) + 1];
            for (std::size_t p = 0; p < rule.node.size(); ++p) {
                const double phi = phiLo + ((phiHi - phiLo) * rule.node[p]);
                const double horizontal = sinTv * std::cos(phi);
                const double radius = std::hypot(horizontal, mu);
                const double delta = std::atan2(horizontal, mu);
                const double visible = std::min(0.5 * kPi, delta + (0.5 * kPi));
                std::array<double, 5> breaks{0.0, visible, std::min(visible, 0.5 * (delta + (0.5 * kPi))), 0.0, 0.0};
                int count = 3;
                if (criticalCos > 0.0 && criticalCos < radius) {
                    const double half = std::acos(criticalCos / radius);
                    for (const double at : {delta - half, delta + half}) {
                        if (at > 0.0 && at < visible) {
                            breaks[static_cast<std::size_t>(count++)] = at;
                        }
                    }
                }
                std::sort(breaks.begin(), breaks.begin() + count);
                for (int panel = 0; panel + 1 < count; ++panel) {
                    const double psiLo = std::atan(std::tan(breaks[static_cast<std::size_t>(panel)]) / alpha);
                    const double psiHi = std::atan(std::tan(breaks[static_cast<std::size_t>(panel) + 1]) / alpha);
                    for (std::size_t q = 0; q < rule.node.size(); ++q) {
                        const double psi = psiLo + ((psiHi - psiLo) * rule.node[q]);
                        const double thetaH = std::atan(alpha * std::tan(psi));
                        const glm::dvec3 h(std::sin(thetaH) * std::cos(phi), std::sin(thetaH) * std::sin(phi), std::cos(thetaH));
                        const double woDotH = glm::dot(wo, h);
                        // phi and psi panel widths, measure sin(psi)cos(psi)/pi doubled; the 1/mu is in smithG2OverCosO.
                        const double weight = rule.weight[p] * rule.weight[q] * (phiHi - phiLo) * (psiHi - psiLo) *
                                              (2.0 * std::sin(psi) * std::cos(psi) / kPi) * (woDotH / h.z);
                        const double fresnel = fresnelDielectric(static_cast<float>(woDotH), eta, 1.0F);
                        const double wiZ = (2.0 * woDotH * h.z) - mu;
                        if (wiZ > 0.0) {
                            sums.reflect[e] += weight * fresnel * smithG2OverCosO(mu, wiZ, alpha);
                        }
                        const double cos2T = cos2Transmitted(static_cast<float>(woDotH), eta);
                        if (cos2T >= 0.0) {
                            const double wtZ = (((eta * woDotH) - std::sqrt(cos2T)) * h.z) - (eta * mu);
                            if (wtZ < 0.0) {
                                sums.transmit[e] += weight * (1.0 - fresnel) * smithG2OverCosO(mu, -wtZ, alpha);
                            }
                        }
                    }
                }
            }
        }
    }
    return sums;
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
    std::vector<float> aavg;  // cosine-weighted means, 2*integral(.(mu)*mu dmu)
    std::vector<float> bavg;
    std::vector<float> cavg;
    std::vector<float> davg;
    std::vector<float> r;  // [roughnessIndex][muIndex][etaIndex], kTransmitRoughnessRes * kTransmitMuRes * kEtaRes
    std::vector<float> t;
    std::vector<float> escapeDeficit;  // 1 - R - T in double, at the physical bound 0 where the quadrature lands past unity
    std::vector<float> ravg;
    std::vector<float> tavg;
    std::vector<float> escapeAvgDeficit;  // 1 - Ravg - Tavg in double, at the physical bound 0
    std::vector<float> msDensity;  // [roughnessIndex][muIndex], the reflected multiple-scattering lobe's own shape
    std::vector<float> msCdf;
    std::vector<float> msTransmitDensity;  // [roughnessIndex][muIndex][etaIndex], the transmitted twin, unnormalised
    std::vector<float> msTransmitCdf;
    std::vector<float> kernelNode;    // [roughnessIndex][muIndex][order], the reflect kernel's Gauss nodes in x = dot(wo, h)
    std::vector<float> kernelWeight;
    std::vector<float> averageNode;   // the Gauss rule of 2 mu dmu on [0, 1], the hemispherical average's
    std::vector<float> averageWeight;
};

// The reflect table's mu axis, uniform in sqrt(mu); node 0 is mu = 0 itself, where E = 1 exactly and buildReflect asserts it on every row.
double reflectMu(int index) {
    const double t = static_cast<double>(index) / static_cast<double>(kAlbedoMuRes - 1);
    return t * t;
}

// The escape tables' mu axis, uniform in sqrt(mu), node 0 being mu = 0 itself: the grazing limit smithG2OverCosO holds to.
double escapeMu(int index) {
    const double t = static_cast<double>(index) / static_cast<double>(kTransmitMuRes - 1);
    return t * t;
}

// alpha = r^2 exactly, as microfacet.cpp's alphaForRoughness: row 0 is the smooth surface itself.
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
    table.aavg.assign(kAlbedoRoughnessRes, 0.0F);
    table.bavg.assign(kAlbedoRoughnessRes, 0.0F);
    table.cavg.assign(kAlbedoRoughnessRes, 0.0F);
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
        double cMean = 0.0;
        for (std::size_t k = 0; k < muRule.node.size(); ++k) {
            const double mu = muRule.node[k];
            const Split split = reflectAlbedo(mu, alpha, phiRule, psiRule);
            aMean += muRule.weight[k] * 2.0 * split.a * mu;
            bMean += muRule.weight[k] * 2.0 * split.b * mu;
            cMean += muRule.weight[k] * 2.0 * split.c * mu;
        }
        table.aavg[static_cast<std::size_t>(ri)] = static_cast<float>(aMean);
        table.bavg[static_cast<std::size_t>(ri)] = static_cast<float>(bMean);
        table.cavg[static_cast<std::size_t>(ri)] = static_cast<float>(cMean);
        table.davg[static_cast<std::size_t>(ri)] = static_cast<float>(deficitOf({aMean, bMean, cMean}));
    });
}

void buildTransmit(AlbedoTable& table, int nodes) {
    const GaussLegendre rule = gaussLegendre(nodes);
    const auto cells = static_cast<std::size_t>(kTransmitRoughnessRes) * kTransmitMuRes * kEtaRes;
    table.r.assign(cells, 0.0F);
    table.t.assign(cells, 0.0F);
    table.escapeDeficit.assign(cells, 0.0F);
    table.ravg.assign(static_cast<std::size_t>(kTransmitRoughnessRes) * kEtaRes, 0.0F);
    table.tavg.assign(static_cast<std::size_t>(kTransmitRoughnessRes) * kEtaRes, 0.0F);
    table.escapeAvgDeficit.assign(static_cast<std::size_t>(kTransmitRoughnessRes) * kEtaRes, 0.0F);
    parallelRows(kTransmitRoughnessRes, [&](int ri) {
        const double alpha = gridAlpha(ri, kTransmitRoughnessRes);
        std::array<double, kEtaRes> rWeighted{};
        std::array<double, kEtaRes> tWeighted{};
        EscapeSums previous{};
        for (int mi = 0; mi < kTransmitMuRes; ++mi) {
            const double mu = escapeMu(mi);
            const EscapeSums sums = escapeAlbedo(mu, alpha, rule);
            // Trapezoid of 2*E*mu over the node spacing, which sqrt spacing makes non-uniform. First order.
            const double previousMu = mi > 0 ? escapeMu(mi - 1) : 0.0;
            const double width = mu - previousMu;
            for (int ei = 0; ei < kEtaRes; ++ei) {
                const auto e = static_cast<std::size_t>(ei);
                const double r = sums.reflect[e];
                const double t = sums.transmit[e];
                table.r[static_cast<std::size_t>((((ri * kTransmitMuRes) + mi) * kEtaRes) + ei)] =
                    static_cast<float>(r);
                table.t[static_cast<std::size_t>((((ri * kTransmitMuRes) + mi) * kEtaRes) + ei)] =
                    static_cast<float>(t);
                table.escapeDeficit[static_cast<std::size_t>((((ri * kTransmitMuRes) + mi) * kEtaRes) + ei)] =
                    static_cast<float>(std::max(1.0 - r - t, 0.0));
                rWeighted[e] += width * ((r * mu) + (previous.reflect[e] * previousMu));
                tWeighted[e] += width * ((t * mu) + (previous.transmit[e] * previousMu));
            }
            previous = sums;
        }
        for (int ei = 0; ei < kEtaRes; ++ei) {
            const auto e = static_cast<std::size_t>(ei);
            table.ravg[static_cast<std::size_t>((ri * kEtaRes) + ei)] = static_cast<float>(rWeighted[e]);
            table.tavg[static_cast<std::size_t>((ri * kEtaRes) + ei)] = static_cast<float>(tWeighted[e]);
            table.escapeAvgDeficit[static_cast<std::size_t>((ri * kEtaRes) + ei)] =
                static_cast<float>(std::max(1.0 - rWeighted[e] - tWeighted[e], 0.0));
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
    table.msTransmitDensity.assign(size, 0.0F);
    table.msTransmitCdf.assign(size, 0.0F);
    for (int ri = 0; ri < kTransmitRoughnessRes; ++ri) {
        for (int ei = 0; ei < kEtaRes; ++ei) {
            double cdf = 0.0;
            for (int mi = 0; mi < kTransmitMuRes; ++mi) {
                const auto index = static_cast<std::size_t>((((ri * kTransmitMuRes) + mi) * kEtaRes) + ei);
                const auto density = static_cast<float>(static_cast<double>(escapeDeficitAtUniformMu(table, ri, mi, ei)) * mi * step);
                if (mi > 0) {
                    // Trapezoid over the float density as emitted, not the double behind it, so the stored pair agrees at read precision.
                    cdf += 0.5 * (table.msTransmitDensity[index - kEtaRes] + density) * step;
                }
                table.msTransmitDensity[index] = density;
                table.msTransmitCdf[index] = static_cast<float>(cdf);
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
    forEachReflectNode(mu, alpha, phiRule, psiRule, [&](double weight, double woDotH) {
        x.push_back(woDotH);
        w.push_back(weight);
    });
    return gaussRuleOf(x, w);
}

// The worst |rule - measure| over two probes per node of the grid: a dielectric Fresnel, smooth, and cos(6 pi x), a film's fringes.
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
            for (const auto& probe : {std::function<double(double)>([](double x) { return fresnelDielectric(static_cast<float>(x), 1.0F, 1.5F); }),
                                      std::function<double(double)>([](double x) { return std::cos(6.0 * kPi * x); })}) {
                double exact = 0.0;
                forEachReflectNode(mu, alpha, rule, rule, [&](double weight, double woDotH) { exact += weight * probe(woDotH); });
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
    const std::array<std::tuple<const char*, const std::vector<float>*, const std::vector<float>*>, 8>
        channels = {{{"a", &table.a, &reference.a},
                     {"b", &table.b, &reference.b},
                     {"c", &table.c, &reference.c},
                     {"d", &table.d, &reference.d},
                     {"aavg", &table.aavg, &reference.aavg},
                     {"bavg", &table.bavg, &reference.bavg},
                     {"cavg", &table.cavg, &reference.cavg},
                     {"davg", &table.davg, &reference.davg}}};
    for (const auto& [name, shipped, exact] : channels) {
        Residual channelWorst{0.0, name, 0, 0};
        for (std::size_t i = 0; i < shipped->size(); ++i) {
            const double delta = std::abs(static_cast<double>((*shipped)[i]) -
                                           static_cast<double>((*exact)[i]));
            if (delta > channelWorst.value) {
                const int stride = shipped->size() == table.aavg.size() ? 1 : kAlbedoMuRes;
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
double verifyTransmit(const AlbedoTable& table, int nodes) {
    AlbedoTable reference;
    buildTransmit(reference, nodes * 2);
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

bool writeInc(const std::string& path, const AlbedoTable& table, double residual, int transmitNodes,
              double transmitResidual, double kernelResidual) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "albedo_table: cannot write " << path << "\n";
        return false;
    }
    const std::string etaMin = floatLiteral(static_cast<float>(kEtaMin));
    const std::string etaMax = floatLiteral(static_cast<float>(kEtaMax));
    out << "// Generated by tools/albedo_table.cpp -- do not edit. Regenerate with:\n"
           "//   ./build/albedo_table --out src/scene/albedo_table.inc\n"
           "// Kulla-Conty energy tables, indexed by perceptual roughness rather than alpha: E is far better\n"
           "// distributed in sqrt(alpha), and it is what callers already hold. Every grid is edge-aligned, so\n"
           "// roughness 0 and mu 1 are exact table entries and microfacet.cpp's lookups can interpolate on k/(res-1).\n"
           "// BOTH mu axes are uniform in sqrt(mu), mu = (k/(res-1))^2, so nodes crowd where E climbs from its\n"
           "// grazing limit over mu ~ alpha; microfacet.cpp indexes each of them by sqrt(mu). The reflect side's node\n"
           "// 0 is mu = 0 itself, where E = 1 exactly for every alpha and the bake asserts that identity. The\n"
           "// two multiple-scattering shapes below stay uniform in mu, where their piecewise-linear inversion\n"
           "// has one step width.\n"
           "// The reflect side splits E = F0*a + b - k*c for the F82-tint Fresnel F0 + (1-F0)(1-x)^5 - k*x(1-x)^6\n"
           "// (OpenPBR's metal, Hoffman 2023): linear in (F0, k), so the three Fresnel-free integrals are exact.\n"
           "// kAlbedoDeficit is 1 - E formed in double, so the Kulla-Conty lobe never cancels it in float as alpha -> 0.\n"
           "// Roughness 0 is alpha = 0, the smooth surface, tabulated analytically: E = F(mu), R = F, T = 1 - F.\n"
           "// Reflect side (a, b, c and their means) is exact-domain Gauss-Legendre, residual "
        << residual << " against a doubled rule.\n"
           "// kEscapeDeficit and kEscapeAvgDeficit are 1 - R - T and its mean formed in double: no float difference turns negative.\n"
           "// Transmit side (r, t, deficit and their means) is Gauss-Legendre in the NDF measure at "
        << transmitNodes << " nodes per panel, residual " << transmitResidual << " against a doubled rule.\n"
           "// kMsReflectDensity/kMsReflectCdf are the reflected multiple-scattering lobe's sampling shape: a\n"
           "// piecewise-linear density over mu, proportional to (1-E(mu))*mu and normalised to 1, with its exact\n"
           "// prefix integrals. microfacet.cpp inverts the first and evaluates it for the matching pdf.\n"
           "// kMsTransmitDensity/kMsTransmitCdf are the same shape for the transmitted twin, carrying the escape\n"
           "// table's eta axis and stored UNNORMALISED: microfacet.cpp blends four rows over (roughness, eta) and\n"
           "// divides by the blended total, so the sampled shape is the raw-deficit interpolation escapeAlbedo\n"
           "// performs and a numerically zero row cannot contribute a unit-mass shape of amplified noise.\n"
           "// kKernelNode/kKernelWeight are the reflect kernel's Gauss rules over x = dot(wo, h) on a roughness x sqrt(mu) grid:\n"
           "// E[F] = sum w F(x) for any Fresnel F, exact to degree 2n - 1; worst probe residual "
        << kernelResidual << ". kAverageNode/kAverageWeight are the rule of 2 mu dmu.\n";
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
        << "constexpr int kKernelOrder = " << kKernelOrder << ";\n";
    writeArray(out, "kAlbedoA", table.a);
    writeArray(out, "kAlbedoB", table.b);
    writeArray(out, "kAlbedoC", table.c);
    writeArray(out, "kAlbedoDeficit", table.d);
    writeArray(out, "kAlbedoAvgA", table.aavg);
    writeArray(out, "kAlbedoAvgB", table.bavg);
    writeArray(out, "kAlbedoAvgC", table.cavg);
    writeArray(out, "kAlbedoAvgDeficit", table.davg);
    writeArray(out, "kEscapeReflect", table.r);
    writeArray(out, "kEscapeTransmit", table.t);
    writeArray(out, "kEscapeDeficit", table.escapeDeficit);
    writeArray(out, "kEscapeAvgReflect", table.ravg);
    writeArray(out, "kEscapeAvgTransmit", table.tavg);
    writeArray(out, "kEscapeAvgDeficit", table.escapeAvgDeficit);
    writeArray(out, "kMsReflectDensity", table.msDensity);
    writeArray(out, "kMsReflectCdf", table.msCdf);
    writeArray(out, "kMsTransmitDensity", table.msTransmitDensity);
    writeArray(out, "kMsTransmitCdf", table.msTransmitCdf);
    writeArray(out, "kKernelNode", table.kernelNode);
    writeArray(out, "kKernelWeight", table.kernelWeight);
    writeArray(out, "kAverageNode", table.averageNode);
    writeArray(out, "kAverageWeight", table.averageWeight);
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
    buildTransmit(table, transmitNodes);
    const double transmitResidual = verifyTransmit(table, transmitNodes);
    buildMultipleScatteringShape(table);
    buildTransmitMultipleScatteringShape(table);
    const double kernelResidual = buildKernel(table, phiNodes);
    std::cout << "albedo_table: kernel Gauss rules of order " << kKernelOrder << ", worst probe residual " << kernelResidual << "\n";
    if (!writeInc(outPath, table, residual.value, transmitNodes, transmitResidual, kernelResidual)) {
        return EXIT_FAILURE;
    }
    std::cout << "albedo_table: wrote " << outPath << " (reflect " << kAlbedoRoughnessRes << "x" << kAlbedoMuRes
              << " at " << phiNodes << " nodes, residual " << residual.value << " in " << residual.channel
              << " at roughnessIndex " << residual.roughnessIndex << " muIndex " << residual.muIndex
              << "; transmit " << kTransmitRoughnessRes << "x" << kTransmitMuRes << "x" << kEtaRes << " at "
              << transmitNodes << " nodes, residual " << transmitResidual << ")\n";
    return EXIT_SUCCESS;
}
