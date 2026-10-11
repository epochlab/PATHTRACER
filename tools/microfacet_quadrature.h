#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/scene/fresnel_dielectric.h"
#include "pathtracer/scene/smith_transmission.h"

// GGX single-scattering quadrature shared by the table generator and the reference validator, generic in the Smith shadowing model.
namespace tools::quadrature {

constexpr double kPi = 3.14159265358979323846;

// Gauss-Legendre nodes/weights on [0,1] by Newton on P_n through Bonnet's recurrence (Numerical Recipes 3rd ed. 4.6.1); weights sum to 1.
struct GaussLegendre {
    std::vector<double> node;
    std::vector<double> weight;
};

inline GaussLegendre gaussLegendre(int n) {
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

inline double smithRadical(double cosTheta, double alpha) {
    const double alpha2 = alpha * alpha;
    return std::sqrt(alpha2 + ((1.0 - alpha2) * cosTheta * cosTheta));
}

// Height-correlated Smith (Heitz 2014), each G2 over cosO so cosO = 0 is a node: the renderer's model, the one the tables bake.
struct HeightCorrelated {
    // 2 cosI/(cosI s(cosO) + cosO s(cosI)) = 1/((1 + Lambda_o + Lambda_i) cosO): no cosine divides; 2/alpha at cosO = 0.
    static double reflect(double cosO, double cosI, double alpha) {
        return 2.0 * cosI / ((cosI * smithRadical(cosO, alpha)) + (cosO * smithRadical(cosI, alpha)));
    }
    // Refraction's B(1 + Lambda_o, 1 + Lambda_i) / cosO (Heitz 2014 sec. 6): 0 at cosO = 0, its limit.
    static double transmit(double cosO, double cosI, double alpha) {
        if (cosO == 0.0 || cosI == 0.0) {
            return 0.0;
        }
        const double lambdaO = 0.5 * ((smithRadical(cosO, alpha) / cosO) - 1.0);
        const double lambdaI = 0.5 * ((smithRadical(cosI, alpha) / cosI) - 1.0);
        return pathtracer::scene::smithTransmitG2(lambdaO, lambdaI) / cosO;
    }
};

// Separable Smith G1(o) G1(i) over cosO, either side: 4 cosI/((cosO + s(cosO))(cosI + s(cosI))), the form Adobe's openpbr-bsdf ships.
struct Separable {
    static double reflect(double cosO, double cosI, double alpha) {
        return 4.0 * cosI / ((cosO + smithRadical(cosO, alpha)) * (cosI + smithRadical(cosI, alpha)));
    }
    static double transmit(double cosO, double cosI, double alpha) { return reflect(cosO, cosI, alpha); }
};

// Every node of the reflect measure at (mu, alpha > 0): visit(weight, dot(wo, h)), the weight carrying D, G2 and the Jacobian.
template <typename Shadowing, typename Visit>
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
                visit(phiRule.weight[p] * psiRule.weight[s] * psiMax * (woDotH / std::cos(thetaH)) * Shadowing::reflect(mu, wiZ, alpha) *
                          std::sin(psi) * std::cos(psi),
                      woDotH);
            }
        }
    }
}

// --- Transmit side: the reflect measure panelled at the interface's boundaries; a VNDF midpoint rule lumps the slope tail in, 3.6e-3.

struct EscapeSums {
    double reflect;
    double transmit;
};

// Escaping fraction of a dielectric interface, reflected and transmitted shares: exact Fresnel, 1.0 inside TIR where Schlick reads ~0.1.
template <typename Shadowing>
EscapeSums escapeAlbedo(double mu, double alpha, double etaRatio, const GaussLegendre& rule) {
    using pathtracer::scene::cos2Transmitted;
    using pathtracer::scene::fresnelDielectric;
    EscapeSums sums{};
    const auto eta = static_cast<float>(etaRatio);
    // alpha = 0 is the smooth interface: it reflects F(mu) and transmits the rest, F being 1 past the critical angle.
    if (alpha == 0.0) {
        const double fresnel = fresnelDielectric(static_cast<float>(mu), eta, 1.0F);
        return {fresnel, 1.0 - fresnel};
    }
    const double sinTv = std::sqrt(std::max(0.0, 1.0 - (mu * mu)));
    const glm::dvec3 wo(sinTv, 0.0, mu);
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
                    // phi and psi panel widths, measure sin(psi)cos(psi)/pi doubled; the 1/mu is in the shadowing's G2 over cosO.
                    const double weight = rule.weight[p] * rule.weight[q] * (phiHi - phiLo) * (psiHi - psiLo) *
                                          (2.0 * std::sin(psi) * std::cos(psi) / kPi) * (woDotH / h.z);
                    const double fresnel = fresnelDielectric(static_cast<float>(woDotH), eta, 1.0F);
                    const double wiZ = (2.0 * woDotH * h.z) - mu;
                    if (wiZ > 0.0) {
                        sums.reflect += weight * fresnel * Shadowing::reflect(mu, wiZ, alpha);
                    }
                    const double cos2T = cos2Transmitted(static_cast<float>(woDotH), eta);
                    if (cos2T >= 0.0) {
                        const double wtZ = (((eta * woDotH) - std::sqrt(cos2T)) * h.z) - (eta * mu);
                        if (wtZ < 0.0) {
                            sums.transmit += weight * (1.0 - fresnel) * Shadowing::transmit(mu, -wtZ, alpha);
                        }
                    }
                }
            }
        }
    }
    return sums;
}

}  // namespace tools::quadrature
