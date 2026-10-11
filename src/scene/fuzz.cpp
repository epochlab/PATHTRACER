#include "fuzz.h"

#include <algorithm>
#include <cmath>

#include "eon.h"
#include "shading_math.h"

namespace pathtracer::scene {

namespace {

// The two names the published table is written against, so third_party/ltc-sheen compiles exactly as its authors ship it.
struct Vector3f {
    float aInv;
    float bInv;
    float albedo;
    constexpr Vector3f(float a, float b, float r) : aInv(a), bInv(b), albedo(r) {}
};

struct SheenLTC {
    static const Vector3f _ltcParamTableVolume[32][32];
};

#include "ltc_table_sheen_volume.cpp"  // NOLINT(bugprone-suspicious-include) -- vendored data, compiled as published

constexpr int kTableRes = 32;

// The reference's bilinear fetch over [alpha][cos theta], both on [0, 1] edge-aligned, so the last row and column are exact nodes.
Vector3f coefficients(float roughness, float mu) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kTableRes - 1);
    const float mf = std::clamp(mu, 0.0F, 1.0F) * (kTableRes - 1);
    const int r0 = std::min(static_cast<int>(rf), kTableRes - 2);
    const int m0 = std::min(static_cast<int>(mf), kTableRes - 2);
    const float rt = rf - static_cast<float>(r0);
    const float mt = mf - static_cast<float>(m0);
    const auto& table = SheenLTC::_ltcParamTableVolume;
    const auto blend = [&](float Vector3f::*field) {
        return lerp1(lerp1(table[r0][m0].*field, table[r0][m0 + 1].*field, mt), lerp1(table[r0 + 1][m0].*field, table[r0 + 1][m0 + 1].*field, mt),
                     rt);
    };
    return {blend(&Vector3f::aInv), blend(&Vector3f::bInv), blend(&Vector3f::albedo)};
}

}  // namespace

float fuzzAlbedo(float roughness, float mu) { return coefficients(roughness, mu).albedo; }

FuzzSlab makeFuzzSlab(const glm::vec3& color, float roughness, const glm::vec3& wo) {
    const Vector3f fit = coefficients(roughness, wo.z);
    return {color, fit.aInv, fit.bInv, fit.albedo, toAzimuthFrame(wo)};
}

// D(w) = D_o(M^-1 w / |M^-1 w|) |M^-1| / |M^-1 w|^3 (Heitz et al. 2016) with |M^-1| = aInv^2 and D_o the clamped cosine.
FuzzEval evaluateFuzz(const FuzzSlab& slab, const glm::vec3& wi) {
    const glm::vec3 w = slab.basisT * wi;
    if (w.z <= 0.0F) {
        return {glm::vec3(0.0F), 0.0F};
    }
    const glm::vec3 original((slab.aInv * w.x) + (slab.bInv * w.z), slab.aInv * w.y, w.z);
    const float lenSq = glm::dot(original, original);
    const float pdf = (w.z / kPi) * (slab.aInv * slab.aInv) / (lenSq * lenSq);
    return {slab.color * (slab.albedo * pdf), pdf};
}

// The clamped cosine sampled, then M = [[1/aInv, 0, -bInv/aInv], [0, 1/aInv, 0], [0, 0, 1]] applied and the result normalised.
glm::vec3 sampleFuzz(const FuzzSlab& slab, glm::vec2 u) {
    const glm::vec3 original = sampleCosineHemisphere(u);
    const glm::vec3 w((original.x - (original.z * slab.bInv)) / slab.aInv, original.y / slab.aInv, original.z);
    return glm::transpose(slab.basisT) * glm::normalize(w);
}

}  // namespace pathtracer::scene
