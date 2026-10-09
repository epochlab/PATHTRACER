#pragma once

#include <algorithm>
#include <cmath>
#include <utility>

// The dielectric interface predicate, shared by microfacet.cpp and the albedo-table generator, which cannot link it. Header-only.
namespace pathtracer::scene {

// Snell in cos^2 (Walter 2007 eq. 40); never forms 1-cos^2, which rounds to 1.0F below cos 2^-12 and falsely reports TIR. r<=1 cannot TIR.
[[nodiscard]] inline float cos2Transmitted(float cosThetaI, float etaRatio) {
    const float r2 = etaRatio * etaRatio;
    return (1.0F - r2) + (r2 * cosThetaI * cosThetaI);
}

// Exact unpolarized dielectric Fresnel (PBRT's FrDielectric); 1.0 from the critical angle inward, where both polarisations are 1.
[[nodiscard]] inline float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = std::clamp(cosThetaI, -1.0F, 1.0F);
    if (cosThetaI < 0.0F) {
        std::swap(etaI, etaT);
        cosThetaI = -cosThetaI;
    }
    const float cos2ThetaT = cos2Transmitted(cosThetaI, etaI / etaT);
    if (cos2ThetaT < 0.0F) {
        return 1.0F;
    }
    const float cosThetaT = std::sqrt(cos2ThetaT);
    const float rParallel =
        ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    const float rPerpendicular =
        ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return ((rParallel * rParallel) + (rPerpendicular * rPerpendicular)) * 0.5F;
}

}  // namespace pathtracer::scene
