#include "pathtracer/scene/lens.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace pathtracer::scene {

namespace {

// theta_d' expressed in u = theta^2 is a quartic, so its sign on an interval is exactly decidable without root-finding.
constexpr int kSlopeDegree = 4;

// Depth at which a slope that only grazes zero is rejected: 2^-24 of the interval is past float's ability to separate it from zero.
constexpr int kMaxSubdivisionDepth = 24;

// Bracket width that resolves the root: 4 ulps of the domain, the angular resolution a unit float direction carries.
constexpr float kBracketUlps = 4.0F;

// Halvings from thetaMax to kBracketUlps * eps * thetaMax: log2(1 / (4 * 2^-23)) = 21, independent of the lens.
constexpr int kBracketHalvings = 21;

// A Newton step that fails to halve the bracket forces a bisection next, so every two iterations halve it at worst.
constexpr int kMaxIterations = 2 * kBracketHalvings;

// Pascal's triangle to degree 4, the power-to-Bernstein basis change's only inputs.
constexpr std::array<std::array<double, kSlopeDegree + 1>, kSlopeDegree + 1> kBinomial{
    {{1.0, 0.0, 0.0, 0.0, 0.0},
     {1.0, 1.0, 0.0, 0.0, 0.0},
     {1.0, 2.0, 1.0, 0.0, 0.0},
     {1.0, 3.0, 3.0, 1.0, 0.0},
     {1.0, 4.0, 6.0, 4.0, 1.0}}};

using SlopeCoefficients = std::array<double, kSlopeDegree + 1>;

// theta_d' = 1 + 3*k1*u + 5*k2*u^2 + 7*k3*u^3 + 9*k4*u^4 in Bernstein form on u in [0, uMax], b_i = sum_j C(i,j)/C(n,j) * a_j.
SlopeCoefficients slopeBernstein(const std::array<float, 4>& k, double uMax) {
    // Power coefficients pre-scaled by uMax^j, so the Bernstein basis is the standard one in s = u / uMax on [0, 1].
    SlopeCoefficients a{1.0, 0.0, 0.0, 0.0, 0.0};
    double scale = 1.0;
    for (int j = 1; j <= kSlopeDegree; ++j) {
        scale *= uMax;
        a[static_cast<std::size_t>(j)] = ((2.0 * j) + 1.0) * static_cast<double>(k[static_cast<std::size_t>(j) - 1]) * scale;
    }
    SlopeCoefficients b{};
    for (int i = 0; i <= kSlopeDegree; ++i) {
        double sum = 0.0;
        for (int j = 0; j <= i; ++j) {
            const auto ji = static_cast<std::size_t>(j);
            sum += (kBinomial[static_cast<std::size_t>(i)][ji] / kBinomial[kSlopeDegree][ji]) * a[ji];
        }
        b[static_cast<std::size_t>(i)] = sum;
    }
    return b;
}

// de Casteljau at the midpoint: the left half keeps b^level_0 and the right half b^(n-i)_i, each the exact Bernstein form of its half.
void subdivide(const SlopeCoefficients& b, SlopeCoefficients& left, SlopeCoefficients& right) {
    SlopeCoefficients work = b;
    left.front() = work.front();
    right.back() = work.back();
    for (int level = 1; level <= kSlopeDegree; ++level) {
        for (int i = 0; i + level <= kSlopeDegree; ++i) {
            const auto index = static_cast<std::size_t>(i);
            work[index] = 0.5 * (work[index] + work[index + 1]);
        }
        const auto tail = static_cast<std::size_t>(kSlopeDegree - level);
        left[static_cast<std::size_t>(level)] = work.front();
        right[tail] = work[tail];
    }
}

// Positivity by the convex-hull property: all-positive coefficients prove it on that half, so only undecided halves are refined.
bool slopeIsProvenPositive(const SlopeCoefficients& coefficients) {
    // Depth-first, so the stack holds one entry per level plus the sibling pushed at each: bounded by the depth cap, never allocated.
    std::array<std::pair<SlopeCoefficients, int>, kMaxSubdivisionDepth + 2> stack{};
    int top = 0;
    stack[static_cast<std::size_t>(top++)] = {coefficients, 0};
    while (top > 0) {
        const auto [b, depth] = stack[static_cast<std::size_t>(--top)];
        // The end coefficients are theta_d' at the subinterval's endpoints, so a non-positive one is a witness rather than a loose bound.
        if (b.front() <= 0.0 || b.back() <= 0.0) {
            return false;
        }
        if (std::all_of(b.begin(), b.end(), [](double c) { return c > 0.0; })) {
            continue;
        }
        if (depth >= kMaxSubdivisionDepth) {
            return false;
        }
        SlopeCoefficients left{};
        SlopeCoefficients right{};
        subdivide(b, left, right);
        const int childDepth = depth + 1;
        stack[static_cast<std::size_t>(top++)] = {left, childDepth};
        stack[static_cast<std::size_t>(top++)] = {right, childDepth};
    }
    return true;
}

}  // namespace

float kannalaBrandtRadius(const std::array<float, 4>& k, float thetaRadians) {
    const float u = thetaRadians * thetaRadians;
    return thetaRadians * (1.0F + (u * (k[0] + (u * (k[1] + (u * (k[2] + (u * k[3]))))))));
}

float kannalaBrandtRadiusSlope(const std::array<float, 4>& k, float thetaRadians) {
    const float u = thetaRadians * thetaRadians;
    return 1.0F + (u * ((3.0F * k[0]) + (u * ((5.0F * k[1]) + (u * ((7.0F * k[2]) + (u * (9.0F * k[3]))))))));
}

float kannalaBrandtTheta(const std::array<float, 4>& k, float radius, float thetaMax) {
    float lo = 0.0F;
    float hi = thetaMax;
    // theta_d's leading term is theta itself, so the radius is the first-order root and the exact answer for an equidistant lens.
    float theta = std::clamp(radius, lo, hi);
    const float tolerance = kBracketUlps * std::numeric_limits<float>::epsilon() * thetaMax;
    float previousWidth = hi - lo;
    for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
        const float residual = kannalaBrandtRadius(k, theta) - radius;
        if (residual == 0.0F) {
            return theta;
        }
        // The bracket only ever narrows: a radius past theta_d(thetaMax) converges on thetaMax.
        if (residual > 0.0F) {
            hi = theta;
        } else {
            lo = theta;
        }
        if ((hi - lo) <= tolerance) {
            break;
        }
        const bool halved = (hi - lo) <= 0.5F * previousWidth;
        previousWidth = hi - lo;
        const float next = theta - (residual / kannalaBrandtRadiusSlope(k, theta));
        // Outside the bracket, non-finite from a vanishing slope, or after a step that failed to halve it: bisect; Newton only accelerates.
        theta = (halved && std::isfinite(next) && next > lo && next < hi) ? next : 0.5F * (lo + hi);
    }
    return theta;
}

bool kannalaBrandtIsInvertible(const std::array<float, 4>& k, float thetaMax) {
    if (!std::isfinite(thetaMax) || thetaMax <= 0.0F) {
        return false;
    }
    if (!std::all_of(k.begin(), k.end(), [](float c) { return std::isfinite(c); })) {
        return false;
    }
    return slopeIsProvenPositive(slopeBernstein(k, static_cast<double>(thetaMax) * static_cast<double>(thetaMax)));
}

}  // namespace pathtracer::scene
