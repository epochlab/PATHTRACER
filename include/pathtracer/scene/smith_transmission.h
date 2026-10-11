#pragma once

#include <array>
#include <cmath>

namespace pathtracer::scene {

// ln Gamma(x) for x >= 1 by Lanczos' series (Numerical Recipes 3rd ed. 6.1, g = 5): relative error below 2e-10, and no global signgam.
[[nodiscard]] inline double logGamma(double x) {
    constexpr std::array<double, 6> kCoefficients{76.18009172947146,  -86.50532032941677,     24.01409824083091,
                                                  -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};
    double denominator = x;
    double series = 1.000000000190015;
    for (const double coefficient : kCoefficients) {
        denominator += 1.0;
        series += coefficient / denominator;
    }
    const double shifted = x + 5.5;
    return ((x + 0.5) * std::log(shifted)) - shifted + std::log(2.5066282746310005 * series / x);
}

// Height-correlated Smith masking-shadowing for refraction, B(1 + Lambda_o, 1 + Lambda_i) (Heitz 2014 sec. 6); reflection's is 1/(1 + sum).
[[nodiscard]] inline double smithTransmitG2(double lambdaO, double lambdaI) {
    return std::exp(logGamma(1.0 + lambdaO) + logGamma(1.0 + lambdaI) - logGamma(2.0 + lambdaO + lambdaI));
}

}  // namespace pathtracer::scene
