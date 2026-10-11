#include "eon.h"

#include <algorithm>
#include <cmath>

namespace pathtracer::scene {

namespace {

constexpr float kPi = 3.14159265F;

constexpr float kConstant1Fon = 0.5F - (2.0F / (3.0F * kPi));
constexpr float kConstant2Fon = (2.0F / 3.0F) - (28.0F / (15.0F * kPi));

float lerp1(float a, float b, float t) { return a + ((b - a) * t); }

// FON directional albedo, quartic fit (paper eq. 14): within 0.1% of the exact form and ~5x cheaper, so used exclusively.
float evalFonAlbedoApprox(float mu, float r) {
    const float muComplement = 1.0F - mu;
    constexpr float g1 = 0.0571085289F;
    constexpr float g2 = 0.491881867F;
    constexpr float g3 = -0.332181442F;
    constexpr float g4 = 0.0714429953F;
    const float gOverPi = muComplement * (g1 + (muComplement * (g2 + (muComplement * (g3 + (muComplement * g4))))));
    const float af = 1.0F / (1.0F + (kConstant1Fon * r));
    return (1.0F + (r * gOverPi)) * af;
}

// EON value (paper eq. 16-19): FON single scatter plus the analytic multiple-scattering lobe, identically zero at r = 0 (Lambertian).
glm::vec3 evaluateEon(const glm::vec3& rho, float r, const glm::vec3& wi, const glm::vec3& wo) {
    const float muI = wi.z;
    const float muO = wo.z;
    const float s = glm::dot(wi, wo) - (muI * muO);
    const float sOverT = s > 0.0F ? s / std::max(muI, muO) : s;
    const float af = 1.0F / (1.0F + (kConstant1Fon * r));
    const glm::vec3 singleScatter = (rho / kPi) * af * (1.0F + (r * sOverT));
    if (r == 0.0F) {
        return singleScatter;
    }
    const float avgEFon = af * (1.0F + (kConstant2Fon * r));
    const glm::vec3 rhoMs = (rho * rho) * avgEFon / (glm::vec3(1.0F) - (rho * (1.0F - avgEFon)));
    // The paper GLSL's eps floor on each 1 - E_FON: its quartic fit overshoots 1 at mu = 0, by 3.6e-4 at r = 1.
    constexpr float kEps = 1e-7F;
    return singleScatter + ((rhoMs / kPi) * std::max(kEps, 1.0F - evalFonAlbedoApprox(muO, r)) *
                            std::max(kEps, 1.0F - evalFonAlbedoApprox(muI, r)) / (1.0F - avgEFon));
}

// Uniform hemisphere direction, pdf = 1/(2*pi): EON's defensive-sampling companion to CLTC (Owen & Zhou 2000 one-sample MIS).
glm::vec3 sampleUniformHemisphere(glm::vec2 u) {
    const float sinTheta = std::sqrt(std::max(0.0F, 1.0F - (u.x * u.x)));
    const float phi = 2.0F * kPi * u.y;
    return {sinTheta * std::cos(phi), sinTheta * std::sin(phi), u.x};
}

// Fitted LTC matrix coefficients (paper Listing 2) matching EON's cosine-weighted backscattering lobe at a view angle and roughness.
glm::vec4 eonLtcCoeffs(float mu, float r) {
    const float a = 1.0F + (r * (0.303392F + (((-0.518982F + (0.111709F * mu)) * mu) + ((-0.276266F + (0.335918F * mu)) * r))));
    const float b = (r * (-1.16407F + (1.15859F * mu) + ((0.150815F - (0.150105F * mu)) * r))) / ((mu * mu * mu) - 1.43545F);
    const float c = 1.0F + (r * (0.20013F + ((-0.506373F + (0.261777F * mu)) * mu)));
    const float d = (r * (0.540852F + ((-1.01625F + (0.475392F * mu)) * mu))) / (-1.0743F + ((0.0725628F + mu) * mu));
    return {a, b, c, d};
}

// Orthonormal frame aligning wLocal's azimuth to the x-axis, to move into and out of the space the LTC fit is expressed in.
glm::mat3 orthonormalBasisLtc(const glm::vec3& wLocal) {
    const float lenSq = (wLocal.x * wLocal.x) + (wLocal.y * wLocal.y);
    const glm::vec3 x = lenSq > 0.0F ? glm::vec3(wLocal.x, wLocal.y, 0.0F) * (1.0F / std::sqrt(lenSq)) : glm::vec3(1.0F, 0.0F, 0.0F);
    const glm::vec3 y(-x.y, x.x, 0.0F);
    return glm::mat3(x, y, glm::vec3(0.0F, 0.0F, 1.0F));
}

// Clipped-LTC sample (paper Sec. 4, Listing 3): cosine sampling of the hemisphere clipped to the lobe, so no sample lands below.
glm::vec3 cltcSample(const glm::vec4& m, const glm::mat3& basisT, float s, glm::vec2 u) {
    const float radius = std::sqrt(u.x);
    const float phi = 2.0F * kPi * u.y;
    const float y = radius * std::sin(phi);
    const float x = -lerp1(std::sqrt(std::max(0.0F, 1.0F - (y * y))), radius * std::cos(phi), s);
    const glm::vec3 wh(x, y, std::sqrt(std::max(0.0F, 1.0F - (x * x) - (y * y))));
    const glm::vec3 wiUnnormalized((m.x * wh.x) + (m.y * wh.z), m.z * wh.y, (m.w * wh.x) + wh.z);
    // Transposing back costs no arithmetic and keeps one stored basis for both directions of the transform.
    return glm::normalize(glm::transpose(basisT) * wiUnnormalized);
}

// pdf of cltcSample's distribution at wiLocal (paper Listing 3's cltc_pdf). M is invertible (det > 0 over the fit), so |M^-1 wi| > 0.
float cltcPdf(const glm::vec4& m, const glm::mat3& basisT, float s, const glm::vec3& wiLocal) {
    const glm::vec3 wi = basisT * wiLocal;
    const glm::vec3 wh(m.z * (wi.x - (m.y * wi.z)), (m.x - (m.y * m.w)) * wi.y, -m.z * ((m.w * wi.x) - (m.x * wi.z)));
    const float lenSq = glm::dot(wh, wh);
    const float detM = m.z * (m.x - (m.y * m.w));
    return (detM * detM) / (lenSq * lenSq) * std::max(wh.z, 0.0F) / (kPi * s);
}

// Mixing weight between the CLTC and uniform lobes (paper Sec. 4): CLTC alone has a variance spike the uniform term corrects.
float eonUniformMixWeight(float mu, float r) {
    const float inner = 0.538233F - (0.290822F * mu);
    const float mid = -0.372058F + (inner * mu);
    return std::pow(r, 0.1F) * (0.162925F + (mid * mu));
}

}  // namespace

// Paper Appendix A: rho for a desired observed albedo, by the stable root not eq. 30.
glm::vec3 eonAlbedoInversion(const glm::vec3& albedo, float r) {
    const float eFonNormal = 1.0F / (1.0F + (kConstant1Fon * r));
    const float avgEFon = eFonNormal * (1.0F + (kConstant2Fon * r));
    const float a = avgEFon - eFonNormal;
    const glm::vec3 b = glm::vec3(eFonNormal) + (albedo * (1.0F - avgEFon));
    return (2.0F * albedo) / (b + glm::sqrt((b * b) + (4.0F * a * albedo)));
}

DiffuseSlab makeDiffuseSlab(const glm::vec3& rho, float r, const glm::vec3& wo) {
    const glm::vec4 m = eonLtcCoeffs(wo.z, r);
    return {rho,
            r,
            eonUniformMixWeight(wo.z, r),
            m,
            glm::transpose(orthonormalBasisLtc(wo)),
            0.5F * (1.0F + (1.0F / std::sqrt((m.w * m.w) + 1.0F)))};
}

DiffuseEval evaluateDiffuse(const DiffuseSlab& slab, const glm::vec3& wo, const glm::vec3& wi) {
    if (wi.z <= 0.0F) {
        return {glm::vec3(0.0F), 0.0F};
    }
    constexpr float kUniformHemispherePdf = 1.0F / (2.0F * kPi);
    return {evaluateEon(slab.rho, slab.roughness, wi, wo) * wi.z,
            (slab.uniformMix * kUniformHemispherePdf) + ((1.0F - slab.uniformMix) * cltcPdf(slab.ltcM, slab.ltcBasisT, slab.ltcS, wi))};
}

glm::vec3 sampleDiffuse(const DiffuseSlab& slab, glm::vec2 u) {
    // Strict: uniformMix is exactly 0 at r=0, where an inclusive test admits u.x==0 and reshuffles it as 0/0.
    if (u.x < slab.uniformMix) {
        return sampleUniformHemisphere({u.x / slab.uniformMix, u.y});
    }
    return cltcSample(slab.ltcM, slab.ltcBasisT, slab.ltcS, {(u.x - slab.uniformMix) / (1.0F - slab.uniformMix), u.y});
}

}  // namespace pathtracer::scene
