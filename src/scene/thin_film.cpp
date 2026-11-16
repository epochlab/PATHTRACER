#include "thin_film.h"

#include <cmath>
#include <limits>
#include <numbers>

namespace pathtracer::scene {

namespace {

// R_p and R_s from index 1 onto relative complex index (n, k), per lane (Born & Wolf 14.2), with a^2 + b^2 = |n^2 - k^2 - s^2 + 2ink|.
void polarisedFresnel(float cosTheta, const SampledSpectrum& n, const SampledSpectrum& k, SampledSpectrum& rp, SampledSpectrum& rs) {
    const float cos2 = cosTheta * cosTheta;
    const float sin2 = 1.0F - cos2;
    const SampledSpectrum t0 = (n * n) - (k * k) - sin2;
    const SampledSpectrum a2b2 = sqrt((t0 * t0) + (4.0F * n * n * k * k));
    const SampledSpectrum a = sqrt(max(0.5F * (a2b2 + t0), SampledSpectrum(0.0F)));
    const SampledSpectrum b = sqrt(max(0.5F * (a2b2 - t0), SampledSpectrum(0.0F)));
    // Sums of squares, not differences: exactly non-negative, where the expanded form rounds below zero at Brewster's angle.
    const SampledSpectrum sCos = a - cosTheta;
    const SampledSpectrum sCosPlus = a + cosTheta;
    rs = ((sCos * sCos) + (b * b)) / ((sCosPlus * sCosPlus) + (b * b));
    const SampledSpectrum pMinus = (a * cosTheta) - sin2;
    const SampledSpectrum pPlus = (a * cosTheta) + sin2;
    const SampledSpectrum bCos = b * cosTheta;
    rp = rs * ((pMinus * pMinus) + (bCos * bCos)) / ((pPlus * pPlus) + (bCos * bCos));
}

// Reflection phase from a medium of index eta1 onto (eta2, kappa2), per polarisation (Belcour & Barla 2017, eq. 12); kappa 0 is dielectric.
void reflectionPhase(float cosTheta, float eta1, const SampledSpectrum& eta2, const SampledSpectrum& kappa2, SampledSpectrum& phiP, SampledSpectrum& phiS) {
    const SampledSpectrum k2 = kappa2 / eta2;
    const float sin2 = 1.0F - (cosTheta * cosTheta);
    const SampledSpectrum a = (eta2 * eta2 * (1.0F - (k2 * k2))) - (eta1 * eta1 * sin2);
    const SampledSpectrum twoEtaK = 2.0F * eta2 * eta2 * k2;
    const SampledSpectrum b = sqrt((a * a) + (twoEtaK * twoEtaK));
    const SampledSpectrum u = sqrt(0.5F * (a + b));
    const SampledSpectrum v = sqrt(max(0.5F * (b - a), SampledSpectrum(0.0F)));
    const float c = eta1 * cosTheta;
    for (int i = 0; i < kSpectrumSamples; ++i) {
        phiS[i] = std::atan2(2.0F * eta1 * v[i] * cosTheta, (u[i] * u[i]) + (v[i] * v[i]) - (c * c));
        const float denominatorRoot = eta2[i] * eta2[i] * (1.0F + (k2[i] * k2[i])) * cosTheta;
        phiP[i] = std::atan2(2.0F * eta1 * eta2[i] * eta2[i] * cosTheta * ((2.0F * k2[i] * u[i]) - ((1.0F - (k2[i] * k2[i])) * v[i])),
                             (denominatorRoot * denominatorRoot) - (eta1 * eta1 * ((u[i] * u[i]) + (v[i] * v[i]))));
    }
}

// Belcour & Barla 2017 eq. 10 at one wavelength, closed: sum_m>=1 rho^m cos(m Phi) = (rho cos Phi - rho^2) / (1 - 2 rho cos Phi + rho^2).
SampledSpectrum airy(float r12, float t121, const SampledSpectrum& r23, float opd, const SampledSpectrum& phase, const glm::vec4& lambdaNm) {
    // At grazing the outer interface reflects all and nothing enters the film to interfere.
    if (t121 == 0.0F) {
        return SampledSpectrum(r12);
    }
    const SampledSpectrum transmitted = (t121 * t121 * r23) / (1.0F - (r12 * r23));
    const SampledSpectrum rho = sqrt(r12 * r23);
    const SampledSpectrum cosPhi(glm::cos((2.0F * std::numbers::pi_v<float> * opd / lambdaNm) + phase.lanes));
    const SampledSpectrum series = ((rho * cosPhi) - (rho * rho)) / (1.0F - (2.0F * rho * cosPhi) + (rho * rho));
    return r12 + transmitted + (2.0F * (transmitted - t121) * series);
}

// Each polarisation's Airy reflectance at the film's lane wavelengths.
PolarisedReflectance airyReflectance(const ThinFilm& film, float cosTheta, float outerIor, const SampledSpectrum& substrateIor,
                                     const SampledSpectrum& substrateKappa) {
    // Snell into the film; past its critical angle no light enters it, and the outer interface reflects all.
    const float ratio = outerIor / film.ior;
    const float cos2Film = 1.0F - ((1.0F - (cosTheta * cosTheta)) * ratio * ratio);
    if (!(cos2Film > 0.0F)) {
        return {SampledSpectrum(1.0F), SampledSpectrum(1.0F)};
    }
    const float cosFilm = std::sqrt(cos2Film);
    SampledSpectrum r12p;
    SampledSpectrum r12s;
    polarisedFresnel(cosTheta, SampledSpectrum(film.ior / outerIor), SampledSpectrum(0.0F), r12p, r12s);
    SampledSpectrum r23p;
    SampledSpectrum r23s;
    polarisedFresnel(cosFilm, substrateIor / film.ior, substrateKappa / film.ior, r23p, r23s);
    SampledSpectrum phi21p;
    SampledSpectrum phi21s;
    reflectionPhase(cosFilm, film.ior, SampledSpectrum(outerIor), SampledSpectrum(0.0F), phi21p, phi21s);
    SampledSpectrum phi23p;
    SampledSpectrum phi23s;
    reflectionPhase(cosFilm, film.ior, substrateIor, substrateKappa, phi23p, phi23s);
    // A perfect conductor (Gulbrandsen's r = 1) reflects all with phase pi, the limit both formulas reach at kappa -> infinity.
    for (int i = 0; i < kSpectrumSamples; ++i) {
        if (std::isinf(substrateKappa[i])) {
            r23p[i] = r23s[i] = 1.0F;
            phi23p[i] = phi23s[i] = std::numbers::pi_v<float>;
        }
    }
    const float opd = 2.0F * film.ior * cosFilm * film.thicknessNm;
    return {airy(r12p[0], 1.0F - r12p[0], r23p, opd, phi23p + phi21p, film.wavelengthNm),
            airy(r12s[0], 1.0F - r12s[0], r23s, opd, phi23s + phi21s, film.wavelengthNm)};
}

}  // namespace

// Each wavelength reflects within [0, 1]; the clamp only absorbs the closed form's rounding at its bounds.
SampledSpectrum filmReflectance(const ThinFilm& film, float cosTheta, float outerIor, const SampledSpectrum& substrateIor,
                          const SampledSpectrum& substrateKappa) {
    const PolarisedReflectance r = airyReflectance(film, cosTheta, outerIor, substrateIor, substrateKappa);
    return clamp(0.5F * (r.parallel + r.perpendicular), 0.0F, 1.0F);
}

PolarisedReflectance filmReflectancePolarised(const ThinFilm& film, float cosTheta, float outerIor, const SampledSpectrum& substrateIor,
                                              const SampledSpectrum& substrateKappa) {
    const PolarisedReflectance r = airyReflectance(film, cosTheta, outerIor, substrateIor, substrateKappa);
    return {clamp(r.parallel, 0.0F, 1.0F), clamp(r.perpendicular, 0.0F, 1.0F)};
}

ComplexIor conductorIor(const SampledSpectrum& reflectance, const SampledSpectrum& edgeTint) {
    ComplexIor ior;
    for (int i = 0; i < kSpectrumSamples; ++i) {
        const float r = reflectance[i];
        // r = 1 is a perfect conductor, n and k unbounded: kappa infinite marks the limit filmReflectance takes exactly.
        if (!(r < 1.0F)) {
            ior.eta[i] = 1.0F;
            ior.kappa[i] = std::numeric_limits<float>::infinity();
            continue;
        }
        const float sqrtR = std::sqrt(r);
        const float rise = 1.0F + sqrtR;
        const float g = edgeTint[i];
        // n_max = (1 + sqrt(r))/(1 - sqrt(r)) as (1 + sqrt(r))^2/(1 - r): 1 - r is exact near 1, where 1 - sqrt(r) cancels.
        const float n = (g * (1.0F - r) / (1.0F + r)) + ((1.0F - g) * rise * rise / (1.0F - r));
        ior.eta[i] = n;
        // k^2 = (r(n+1)^2 - (n-1)^2)/(1-r) as (sqrt(r)(n+1) - (n-1))(sqrt(r)(n+1) + (n-1)), each factor rewritten free of cancellation.
        const float vanishing = g * rise * 2.0F * sqrtR / (1.0F + r);
        const float growing = (n * rise) - ((1.0F - r) / rise);
        ior.kappa[i] = std::sqrt(vanishing * growing / (1.0F - r));
    }
    return ior;
}

}  // namespace pathtracer::scene
