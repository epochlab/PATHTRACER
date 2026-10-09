#pragma once

#include <algorithm>
#include <cmath>
#include <complex>

// Double-precision conductor Fresnel references shared by tools. Never include src/scene/bsdf.* here: the oracle-independence rule.
namespace tools::reference {

// OpenPBR's F82-tint metal Fresnel in double, transcribed from the specification: Schlick less a mu(1-mu)^6 term fit at mu-bar = 1/7.
inline constexpr double kMuBar = 1.0 / 7.0;

inline double referenceF82(double f0, double tint, double mu) {
    const auto schlick = [f0](double c) { return f0 + ((1.0 - f0) * std::pow(1.0 - c, 5.0)); };
    const double correction = (mu * std::pow(1.0 - mu, 6.0)) / (kMuBar * std::pow(1.0 - kMuBar, 6.0));
    return schlick(mu) - (correction * (schlick(kMuBar) - (tint * schlick(kMuBar))));
}

// Textbook complex Fresnel for a conductor of complex index eta, unpolarized, in double.
inline double referenceConductorFresnelAt(const std::complex<double>& eta, double cosTheta) {
    const double c = std::clamp(cosTheta, 0.0, 1.0);
    const std::complex<double> cosThetaT = std::sqrt(1.0 - ((1.0 - (c * c)) / (eta * eta)));
    const std::complex<double> rParallel = ((eta * c) - cosThetaT) / ((eta * c) + cosThetaT);
    const std::complex<double> rPerpendicular = (c - (eta * cosThetaT)) / (c + (eta * cosThetaT));
    return 0.5 * (std::norm(rParallel) + std::norm(rPerpendicular));
}

// Cosine-weighted average Fresnel by composite Simpson: the integrand is analytic on [0,1], so the O(h^4) error here is ~1e-13.
template <typename Fresnel>
double cosineAverageFresnel(Fresnel fresnel) {
    constexpr int kPanels = 4000;   // even, for Simpson
    const double h = 1.0 / kPanels;
    double sum = fresnel(1.0);
    for (int i = 1; i < kPanels; ++i) {
        const double mu = i * h;
        sum += (i % 2 == 1 ? 4.0 : 2.0) * fresnel(mu) * mu;
    }
    return 2.0 * (h / 3.0) * sum;
}

}  // namespace tools::reference
