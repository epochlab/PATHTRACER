#include "thin_film.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include "pathtracer/scene/cie.h"

namespace pathtracer::scene {

namespace {

// Path differences tabulated to 40 um at 2 nm: past it the CIE x D65 transforms sit under 5e-3, a film incoherent there.
constexpr double kOpdStepNm = 2.0;
constexpr int kOpdSamples = 20001;
constexpr float kOpdLimitNm = static_cast<float>(kOpdStepNm * (kOpdSamples - 1));

// Scene RGB of the spectrum cos(2 pi o / lambda + phi) under D65 is cos(phi) C(o) - sin(phi) S(o): its exact cosine and sine transforms.
struct Sensitivity {
    std::vector<Rgb> cosine;
    std::vector<Rgb> sine;
    float bound;  // the largest |cos(phi) C - sin(phi) S| = sqrt(C^2 + S^2) over the table, the series' tail bound per unit amplitude
};

// Built once, on first use, from the CIE 015:2018 tables, normalised so a flat unit spectrum is scene (1, 1, 1) at o = 0.
const Sensitivity& sensitivity() {
    static const Sensitivity table = [] {
        double whiteY = 0.0;
        std::array<glm::dvec3, cie::kSampleCount> weight{};
        for (int i = 0; i < cie::kSampleCount; ++i) {
            const cie::TableRow row = cie::tableRow(i);
            weight[static_cast<std::size_t>(i)] = cie::xyzToScene() * (row.colourMatching * row.d65);
            whiteY += row.colourMatching.y * row.d65;
        }
        Sensitivity built{std::vector<Rgb>(kOpdSamples), std::vector<Rgb>(kOpdSamples), 0.0F};
        for (int s = 0; s < kOpdSamples; ++s) {
            glm::dvec3 cosine(0.0);
            glm::dvec3 sine(0.0);
            for (int i = 0; i < cie::kSampleCount; ++i) {
                const double phase = 2.0 * std::numbers::pi * (s * kOpdStepNm) / cie::wavelengthNm(i);
                cosine += weight[static_cast<std::size_t>(i)] * std::cos(phase);
                sine += weight[static_cast<std::size_t>(i)] * std::sin(phase);
            }
            built.cosine[static_cast<std::size_t>(s)] = Rgb(glm::vec3(cosine / whiteY));
            built.sine[static_cast<std::size_t>(s)] = Rgb(glm::vec3(sine / whiteY));
            const glm::dvec3 magnitude = sqrt((cosine * cosine) + (sine * sine)) / whiteY;
            built.bound = std::max({built.bound, static_cast<float>(magnitude.x), static_cast<float>(magnitude.y), static_cast<float>(magnitude.z)});
        }
        return built;
    }();
    return table;
}

// The interference term's scene RGB at path difference opd (nm) and phase shift per channel; zero past the table, the film incoherent.
Rgb interference(float opd, const Rgb& shift) {
    const float index = opd / static_cast<float>(kOpdStepNm);
    if (!(opd < kOpdLimitNm)) {
        return Rgb(0.0F);
    }
    const Sensitivity& table = sensitivity();
    const auto i0 = static_cast<std::size_t>(index);
    const float t = index - static_cast<float>(i0);
    const Rgb cosine = table.cosine[i0] + (t * (table.cosine[i0 + 1] - table.cosine[i0]));
    const Rgb sine = table.sine[i0] + (t * (table.sine[i0 + 1] - table.sine[i0]));
    return (cos(shift) * cosine) - (sin(shift) * sine);
}

// R_p and R_s from index 1 onto relative complex index (n, k), per channel (Born & Wolf 14.2), with a^2 + b^2 = |n^2 - k^2 - s^2 + 2ink|.
void polarisedFresnel(float cosTheta, const Rgb& n, const Rgb& k, Rgb& rp, Rgb& rs) {
    const float cos2 = cosTheta * cosTheta;
    const float sin2 = 1.0F - cos2;
    const Rgb t0 = (n * n) - (k * k) - sin2;
    const Rgb a2b2 = sqrt((t0 * t0) + (4.0F * n * n * k * k));
    const Rgb a = sqrt(max(0.5F * (a2b2 + t0), Rgb(0.0F)));
    const Rgb b = sqrt(max(0.5F * (a2b2 - t0), Rgb(0.0F)));
    // Sums of squares, not differences: exactly non-negative, where the expanded form rounds below zero at Brewster's angle.
    const Rgb sCos = a - cosTheta;
    const Rgb sCosPlus = a + cosTheta;
    rs = ((sCos * sCos) + (b * b)) / ((sCosPlus * sCosPlus) + (b * b));
    const Rgb pMinus = (a * cosTheta) - sin2;
    const Rgb pPlus = (a * cosTheta) + sin2;
    const Rgb bCos = b * cosTheta;
    rp = rs * ((pMinus * pMinus) + (bCos * bCos)) / ((pPlus * pPlus) + (bCos * bCos));
}

// Reflection phase from a medium of index eta1 onto (eta2, kappa2), per polarisation (Belcour & Barla 2017, eq. 12); kappa 0 is dielectric.
void reflectionPhase(float cosTheta, float eta1, const Rgb& eta2, const Rgb& kappa2, Rgb& phiP, Rgb& phiS) {
    const Rgb k2 = kappa2 / eta2;
    const float sin2 = 1.0F - (cosTheta * cosTheta);
    const Rgb a = (eta2 * eta2 * (1.0F - (k2 * k2))) - (eta1 * eta1 * sin2);
    const Rgb twoEtaK = 2.0F * eta2 * eta2 * k2;
    const Rgb b = sqrt((a * a) + (twoEtaK * twoEtaK));
    const Rgb u = sqrt(0.5F * (a + b));
    const Rgb v = sqrt(max(0.5F * (b - a), Rgb(0.0F)));
    const float c = eta1 * cosTheta;
    for (int i = 0; i < 3; ++i) {
        phiS[i] = std::atan2(2.0F * eta1 * v[i] * cosTheta, (u[i] * u[i]) + (v[i] * v[i]) - (c * c));
        const float denominatorRoot = eta2[i] * eta2[i] * (1.0F + (k2[i] * k2[i])) * cosTheta;
        phiP[i] = std::atan2(2.0F * eta1 * eta2[i] * eta2[i] * cosTheta * ((2.0F * k2[i] * u[i]) - ((1.0F - (k2[i] * k2[i])) * v[i])),
                             (denominatorRoot * denominatorRoot) - (eta1 * eta1 * ((u[i] * u[i]) + (v[i] * v[i]))));
    }
}

// One polarisation's Airy series (Belcour & Barla 2017, eq. 10), to its tail below float resolution or past the table, where terms are 0.
Rgb airy(float r12, float t121, const Rgb& r23, float opd, const Rgb& phase) {
    // At grazing the outer interface reflects all and nothing enters the film to interfere.
    if (t121 == 0.0F) {
        return Rgb(r12);
    }
    const Rgb transmitted = (t121 * t121 * r23) / (1.0F - (r12 * r23));
    const Rgb r123 = sqrt(r12 * r23);
    const float ratio = maxChannel(r123);
    // Each order m adds C_m S_m with |S_m| <= 2 bound and |C_m| = |C_0| r^m, so past order m the rest sums to at most that over 1 - r.
    const float tailScale = 2.0F * sensitivity().bound * ratio / (1.0F - ratio);
    Rgb amplitude = transmitted - t121;
    Rgb reflectance = r12 + transmitted;
    for (int m = 1; static_cast<float>(m) * opd < kOpdLimitNm &&
                    tailScale * maxChannel(abs(amplitude)) >=
                        std::numeric_limits<float>::epsilon();
         ++m) {
        amplitude *= r123;
        reflectance += amplitude * (2.0F * interference(static_cast<float>(m) * opd, static_cast<float>(m) * phase));
    }
    return reflectance;
}

// Each polarisation's Airy reflectance in scene space, unclamped: a fringe may lie outside the primaries' gamut.
PolarisedReflectance airyReflectance(const ThinFilm& film, float cosTheta, float outerIor, const Rgb& substrateIor,
                                     const Rgb& substrateKappa) {
    // Snell into the film; past its critical angle no light enters it, and the outer interface reflects all.
    const float ratio = outerIor / film.ior;
    const float cos2Film = 1.0F - ((1.0F - (cosTheta * cosTheta)) * ratio * ratio);
    if (!(cos2Film > 0.0F)) {
        return {Rgb(1.0F), Rgb(1.0F)};
    }
    const float cosFilm = std::sqrt(cos2Film);
    Rgb r12p;
    Rgb r12s;
    polarisedFresnel(cosTheta, Rgb(film.ior / outerIor), Rgb(0.0F), r12p, r12s);
    Rgb r23p;
    Rgb r23s;
    polarisedFresnel(cosFilm, substrateIor / film.ior, substrateKappa / film.ior, r23p, r23s);
    Rgb phi21p;
    Rgb phi21s;
    reflectionPhase(cosFilm, film.ior, Rgb(outerIor), Rgb(0.0F), phi21p, phi21s);
    Rgb phi23p;
    Rgb phi23s;
    reflectionPhase(cosFilm, film.ior, substrateIor, substrateKappa, phi23p, phi23s);
    // A perfect conductor (Gulbrandsen's r = 1) reflects all with phase pi, the limit both formulas reach at kappa -> infinity.
    for (int i = 0; i < 3; ++i) {
        if (std::isinf(substrateKappa[i])) {
            r23p[i] = r23s[i] = 1.0F;
            phi23p[i] = phi23s[i] = std::numbers::pi_v<float>;
        }
    }
    const float opd = 2.0F * film.ior * cosFilm * film.thicknessNm;
    return {airy(r12p.r, 1.0F - r12p.r, r23p, opd, phi23p + phi21p), airy(r12s.r, 1.0F - r12s.r, r23s, opd, phi23s + phi21s)};
}

}  // namespace

// Each wavelength reflects within [0, 1]; a fringe more saturated than the scene primaries is clipped to the RGB the renderer carries.
Rgb filmReflectance(const ThinFilm& film, float cosTheta, float outerIor, const Rgb& substrateIor,
                          const Rgb& substrateKappa) {
    const PolarisedReflectance r = airyReflectance(film, cosTheta, outerIor, substrateIor, substrateKappa);
    return clamp(0.5F * (r.parallel + r.perpendicular), 0.0F, 1.0F);
}

PolarisedReflectance filmReflectancePolarised(const ThinFilm& film, float cosTheta, float outerIor, const Rgb& substrateIor,
                                              const Rgb& substrateKappa) {
    const PolarisedReflectance r = airyReflectance(film, cosTheta, outerIor, substrateIor, substrateKappa);
    return {clamp(r.parallel, 0.0F, 1.0F), clamp(r.perpendicular, 0.0F, 1.0F)};
}

ComplexIor conductorIor(const Rgb& reflectance, const Rgb& edgeTint) {
    ComplexIor ior;
    for (int i = 0; i < 3; ++i) {
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
