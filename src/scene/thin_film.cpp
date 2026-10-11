#include "thin_film.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "pathtracer/scene/cie.h"

namespace pathtracer::scene {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Path differences tabulated to 40 um at 2 nm: past it the CIE x D65 transforms sit under 5e-3, a film incoherent there.
constexpr double kOpdStepNm = 2.0;
constexpr int kOpdSamples = 20001;

// Rec.709 of the spectrum cos(2 pi o / lambda + phi) under D65 is cos(phi) C(o) - sin(phi) S(o): its exact cosine and sine transforms.
struct Sensitivity {
    std::vector<glm::vec3> cosine;
    std::vector<glm::vec3> sine;
    float bound;  // the largest |cos(phi) C - sin(phi) S| = sqrt(C^2 + S^2) over the table, the series' tail bound per unit amplitude
};

// Built once, on first use, from the CIE 015:2018 tables, normalised so a flat unit spectrum is Rec.709 (1, 1, 1) at o = 0.
const Sensitivity& sensitivity() {
    static const Sensitivity table = [] {
        double whiteY = 0.0;
        std::array<glm::dvec3, cie::kSampleCount> weight{};
        for (int i = 0; i < cie::kSampleCount; ++i) {
            const cie::TableRow row = cie::tableRow(i);
            weight[static_cast<std::size_t>(i)] = cie::xyzToRec709() * (row.colourMatching * row.d65);
            whiteY += row.colourMatching.y * row.d65;
        }
        Sensitivity built{std::vector<glm::vec3>(kOpdSamples), std::vector<glm::vec3>(kOpdSamples), 0.0F};
        for (int s = 0; s < kOpdSamples; ++s) {
            glm::dvec3 cosine(0.0);
            glm::dvec3 sine(0.0);
            for (int i = 0; i < cie::kSampleCount; ++i) {
                const double phase = 2.0 * kPi * (s * kOpdStepNm) / cie::wavelengthNm(i);
                cosine += weight[static_cast<std::size_t>(i)] * std::cos(phase);
                sine += weight[static_cast<std::size_t>(i)] * std::sin(phase);
            }
            built.cosine[static_cast<std::size_t>(s)] = glm::vec3(cosine / whiteY);
            built.sine[static_cast<std::size_t>(s)] = glm::vec3(sine / whiteY);
            const glm::dvec3 magnitude = glm::sqrt((cosine * cosine) + (sine * sine)) / whiteY;
            built.bound = std::max({built.bound, static_cast<float>(magnitude.x), static_cast<float>(magnitude.y), static_cast<float>(magnitude.z)});
        }
        return built;
    }();
    return table;
}

// The interference term's Rec.709 at path difference opd (nm) and phase shift per channel; zero past the table, the film incoherent.
glm::vec3 interference(float opd, const glm::vec3& shift) {
    const float index = opd / static_cast<float>(kOpdStepNm);
    if (!(index < static_cast<float>(kOpdSamples - 1))) {
        return glm::vec3(0.0F);
    }
    const Sensitivity& table = sensitivity();
    const auto i0 = static_cast<std::size_t>(index);
    const float t = index - static_cast<float>(i0);
    const glm::vec3 cosine = table.cosine[i0] + (t * (table.cosine[i0 + 1] - table.cosine[i0]));
    const glm::vec3 sine = table.sine[i0] + (t * (table.sine[i0 + 1] - table.sine[i0]));
    return (glm::cos(shift) * cosine) - (glm::sin(shift) * sine);
}

// R_p and R_s from index 1 onto relative complex index (n, k), per channel (Born & Wolf 14.2), with a^2 + b^2 = |n^2 - k^2 - s^2 + 2ink|.
void polarisedFresnel(float cosTheta, const glm::vec3& n, const glm::vec3& k, glm::vec3& rp, glm::vec3& rs) {
    const float cos2 = cosTheta * cosTheta;
    const float sin2 = 1.0F - cos2;
    const glm::vec3 t0 = (n * n) - (k * k) - sin2;
    const glm::vec3 a2b2 = glm::sqrt((t0 * t0) + (4.0F * n * n * k * k));
    const glm::vec3 a = glm::sqrt(glm::max(0.5F * (a2b2 + t0), glm::vec3(0.0F)));
    const glm::vec3 b = glm::sqrt(glm::max(0.5F * (a2b2 - t0), glm::vec3(0.0F)));
    // Sums of squares, not differences: exactly non-negative, where the expanded form rounds below zero at Brewster's angle.
    const glm::vec3 sCos = a - cosTheta;
    const glm::vec3 sCosPlus = a + cosTheta;
    rs = ((sCos * sCos) + (b * b)) / ((sCosPlus * sCosPlus) + (b * b));
    const glm::vec3 pMinus = (a * cosTheta) - sin2;
    const glm::vec3 pPlus = (a * cosTheta) + sin2;
    const glm::vec3 bCos = b * cosTheta;
    rp = rs * ((pMinus * pMinus) + (bCos * bCos)) / ((pPlus * pPlus) + (bCos * bCos));
}

// Reflection phase from a medium of index eta1 onto (eta2, kappa2), per polarisation (Belcour & Barla 2017, eq. 12); kappa 0 is dielectric.
void reflectionPhase(float cosTheta, float eta1, const glm::vec3& eta2, const glm::vec3& kappa2, glm::vec3& phiP, glm::vec3& phiS) {
    const glm::vec3 k2 = kappa2 / eta2;
    const float sin2 = 1.0F - (cosTheta * cosTheta);
    const glm::vec3 a = (eta2 * eta2 * (1.0F - (k2 * k2))) - (eta1 * eta1 * sin2);
    const glm::vec3 twoEtaK = 2.0F * eta2 * eta2 * k2;
    const glm::vec3 b = glm::sqrt((a * a) + (twoEtaK * twoEtaK));
    const glm::vec3 u = glm::sqrt(0.5F * (a + b));
    const glm::vec3 v = glm::sqrt(glm::max(0.5F * (b - a), glm::vec3(0.0F)));
    const float c = eta1 * cosTheta;
    for (int i = 0; i < 3; ++i) {
        phiS[i] = std::atan2(2.0F * eta1 * v[i] * cosTheta, (u[i] * u[i]) + (v[i] * v[i]) - (c * c));
        const float denominatorRoot = eta2[i] * eta2[i] * (1.0F + (k2[i] * k2[i])) * cosTheta;
        phiP[i] = std::atan2(2.0F * eta1 * eta2[i] * eta2[i] * cosTheta * ((2.0F * k2[i] * u[i]) - ((1.0F - (k2[i] * k2[i])) * v[i])),
                             (denominatorRoot * denominatorRoot) - (eta1 * eta1 * ((u[i] * u[i]) + (v[i] * v[i]))));
    }
}

// One polarisation's Airy series (Belcour & Barla 2017, eq. 10), summed until its geometric tail lies below float resolution.
glm::vec3 airy(float r12, float t121, const glm::vec3& r23, float opd, const glm::vec3& phase) {
    // At grazing the outer interface reflects all and nothing enters the film to interfere.
    if (t121 == 0.0F) {
        return glm::vec3(r12);
    }
    const glm::vec3 transmitted = (t121 * t121 * r23) / (1.0F - (r12 * r23));
    const glm::vec3 r123 = glm::sqrt(r12 * r23);
    const float ratio = std::max({r123.x, r123.y, r123.z});
    // Each order m adds C_m S_m with |S_m| <= 2 bound and |C_m| = |C_0| r^m, so past order m the rest sums to at most that over 1 - r.
    const float tailScale = 2.0F * sensitivity().bound * ratio / (1.0F - ratio);
    glm::vec3 amplitude = transmitted - t121;
    glm::vec3 reflectance = r12 + transmitted;
    for (int m = 1; tailScale * std::max({std::abs(amplitude.x), std::abs(amplitude.y), std::abs(amplitude.z)}) >=
                    std::numeric_limits<float>::epsilon();
         ++m) {
        amplitude *= r123;
        reflectance += amplitude * (2.0F * interference(static_cast<float>(m) * opd, static_cast<float>(m) * phase));
    }
    return reflectance;
}

}  // namespace

glm::vec3 filmReflectance(const ThinFilm& film, float cosTheta, float outerIor, const glm::vec3& substrateIor,
                          const glm::vec3& substrateKappa) {
    // Snell into the film; past its critical angle no light enters it, and the outer interface reflects all.
    const float ratio = outerIor / film.ior;
    const float cos2Film = 1.0F - ((1.0F - (cosTheta * cosTheta)) * ratio * ratio);
    if (!(cos2Film > 0.0F)) {
        return glm::vec3(1.0F);
    }
    const float cosFilm = std::sqrt(cos2Film);
    glm::vec3 r12p;
    glm::vec3 r12s;
    polarisedFresnel(cosTheta, glm::vec3(film.ior / outerIor), glm::vec3(0.0F), r12p, r12s);
    glm::vec3 r23p;
    glm::vec3 r23s;
    polarisedFresnel(cosFilm, substrateIor / film.ior, substrateKappa / film.ior, r23p, r23s);
    glm::vec3 phi21p;
    glm::vec3 phi21s;
    reflectionPhase(cosFilm, film.ior, glm::vec3(outerIor), glm::vec3(0.0F), phi21p, phi21s);
    glm::vec3 phi23p;
    glm::vec3 phi23s;
    reflectionPhase(cosFilm, film.ior, substrateIor, substrateKappa, phi23p, phi23s);
    // A perfect conductor (Gulbrandsen's r = 1) reflects all with phase pi, the limit both formulas reach at kappa -> infinity.
    for (int i = 0; i < 3; ++i) {
        if (std::isinf(substrateKappa[i])) {
            r23p[i] = r23s[i] = 1.0F;
            phi23p[i] = phi23s[i] = static_cast<float>(kPi);
        }
    }
    const float opd = 2.0F * film.ior * cosFilm * film.thicknessNm;
    const glm::vec3 parallel = airy(r12p.x, 1.0F - r12p.x, r23p, opd, phi23p + phi21p);
    const glm::vec3 perpendicular = airy(r12s.x, 1.0F - r12s.x, r23s, opd, phi23s + phi21s);
    // Each wavelength reflects within [0, 1]; a fringe more saturated than Rec.709's primaries is clipped to the RGB the renderer carries.
    return glm::clamp(0.5F * (parallel + perpendicular), glm::vec3(0.0F), glm::vec3(1.0F));
}

ComplexIor conductorIor(const glm::vec3& reflectance, const glm::vec3& edgeTint) {
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
        const float n = (edgeTint[i] * (1.0F - r) / (1.0F + r)) + ((1.0F - edgeTint[i]) * (1.0F + sqrtR) / (1.0F - sqrtR));
        ior.eta[i] = n;
        // Zero at g's bounds, where n meets a root of the numerator; rounding there can leave it an ulp negative.
        ior.kappa[i] = std::sqrt(std::max(((r * (n + 1.0F) * (n + 1.0F)) - ((n - 1.0F) * (n - 1.0F))) / (1.0F - r), 0.0F));
    }
    return ior;
}

}  // namespace pathtracer::scene
