#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/fresnel_dielectric.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace pathtracer::scene {

namespace {

constexpr float kPi = 3.14159265F;
constexpr float kMinAlpha = 0.02F * 0.02F;  // roughness floor, avoids a degenerate GGX delta lobe

// Perceptual roughness to GGX alpha, in one place: the three consumers must agree, or the Fresnel AOV reports an unevaluated term.
float alphaForRoughness(float roughness) { return std::max(roughness * roughness, kMinAlpha); }

// GGX D, cancellation-free (Filament 4.4.2); no denominator floor, kPi*d*d >= 8e-14. A floor here once cut D by 124340x at roughness 0.02.
float distributionGGX(const glm::vec3& nh, float alpha) {
    const float alpha2 = alpha * alpha;
    const float d = (alpha2 * nh.z * nh.z) + (nh.x * nh.x) + (nh.y * nh.y);
    return alpha2 / (kPi * d * d);
}

// cos*sqrt(1 + alpha^2*tan^2), the Smith Lambda radical scaled by cosine: 1 + Lambda(c) = (c + radical)/(2c), finite at c = 0.
float smithRadical(float cosTheta, float alpha) {
    const float alpha2 = alpha * alpha;
    return std::sqrt(alpha2 + ((1.0F - alpha2) * cosTheta * cosTheta));
}

// G2/(4*cosO*cosI), height-correlated Smith (Heitz 2014): cosines multiply rather than divide, so it is exact to the silhouette.
float smithVisibility(float cosO, float cosI, float alpha) {
    return 0.5F / ((cosI * smithRadical(cosO, alpha)) + (cosO * smithRadical(cosI, alpha)));
}

// G1(c)/c, the VNDF pdf's projected-area factor, in the same division-free form: 2/alpha at grazing rather than 0/0.
float smithG1OverCos(float cosTheta, float alpha) { return 2.0F / (cosTheta + smithRadical(cosTheta, alpha)); }

// --- Conductor Fresnel (Gulbrandsen 2014, JCGT 3(4)), replacing Schlick.

// Reflectivity is clamped: f0 is an unbounded product. 0.9999 needs the factored k^2 below; the literal form needs 0.99, 1% at f0=1.
constexpr float kMinReflectivity = 1e-4F;
constexpr float kMaxReflectivity = 0.9999F;

struct ConductorIor {
    glm::vec3 n;
    glm::vec3 k;
};

// Paper eq 12 for n, eq 2 for k, factored: the literal form gives k^2 = -1.28e6 at r=0.9999, g=0. Inputs come from tools/metal_fit.
ConductorIor conductorIorFromReflectivity(const glm::vec3& reflectivity, const glm::vec3& edgeTint) {
    const glm::vec3 r = glm::clamp(reflectivity, kMinReflectivity, kMaxReflectivity);
    const glm::vec3 g = glm::clamp(edgeTint, 0.0F, 1.0F);  // the paper's stated domain for g
    const glm::vec3 sqrtR = glm::sqrt(r);
    const glm::vec3 nMin = (1.0F - r) / (1.0F + r);
    const glm::vec3 nMax = (1.0F + sqrtR) / (1.0F - sqrtR);
    const glm::vec3 nLow = (1.0F - sqrtR) / (1.0F + sqrtR);
    const glm::vec3 n = (g * nMin) + ((1.0F - g) * nMax);
    // Both factors are non-negative over the clamped domain; the max absorbs float rounding on nMax - n at g -> 0.
    return {n, glm::sqrt(glm::max((nMax - n) * (n - nLow), 0.0F))};
}

// Exact unpolarized conductor Fresnel (Born & Wolf), not the paper's large-|eta| approximation, which is off by 0.094 at r~0.25.
float fresnelConductorChannel(float cosTheta, float n, float k) {
    const float c2 = cosTheta * cosTheta;
    const float s2 = 1.0F - c2;
    const float nk2 = (n * n) * (k * k);
    const float t0 = (n * n) - (k * k) - s2;
    const float a2b2 = std::sqrt((t0 * t0) + (4.0F * nk2));
    // a^2 through whichever equal form is a sum: the direct one collapses a to zero in float32 when t0 < 0, pinning F at exactly 1.0.
    const float a2 = t0 >= 0.0F ? (a2b2 + t0) * 0.5F : (2.0F * nk2) / (a2b2 - t0);
    const float a = std::sqrt(a2);
    const float t1 = a2b2 + c2;
    const float t2 = 2.0F * a * cosTheta;
    const float rPerpendicular = (t1 - t2) / (t1 + t2);
    const float t3 = (c2 * a2b2) + (s2 * s2);
    const float t4 = t2 * s2;
    // R = (rPerp + rPara)/2 with rPara = rPerp*(t3-t4)/(t3+t4), factored to drop a multiply.
    return 0.5F * rPerpendicular * (1.0F + ((t3 - t4) / (t3 + t4)));
}

// cosTheta clamped to [0,1] rather than sign-swapped: a conductor has no far side, and grazing normal maps push wo.z negative.
glm::vec3 fresnelConductor(float cosTheta, const glm::vec3& n, const glm::vec3& k) {
    const float c = std::clamp(cosTheta, 0.0F, 1.0F);
    return {fresnelConductorChannel(c, n.x, k.x), fresnelConductorChannel(c, n.y, k.y),
             fresnelConductorChannel(c, n.z, k.z)};
}

// Heitz 2018 VNDF sampling. wo.z > 0 required (caller pre-flips into the +z hemisphere).
glm::vec3 sampleGGXVNDF(const glm::vec3& wo, float alpha, glm::vec2 u) {
    const glm::vec3 vh = glm::normalize(glm::vec3(alpha * wo.x, alpha * wo.y, wo.z));
    const float lensq = (vh.x * vh.x) + (vh.y * vh.y);
    const glm::vec3 t1 = lensq > 0.0F ? glm::vec3(-vh.y, vh.x, 0.0F) * (1.0F / std::sqrt(lensq))
                                       : glm::vec3(1.0F, 0.0F, 0.0F);
    const glm::vec3 t2 = glm::cross(vh, t1);
    const float r = std::sqrt(u.x);
    const float phi = 2.0F * kPi * u.y;
    const float t1p = r * std::cos(phi);
    float t2p = r * std::sin(phi);
    const float s = 0.5F * (1.0F + vh.z);
    t2p = ((1.0F - s) * std::sqrt(std::max(0.0F, 1.0F - (t1p * t1p)))) + (s * t2p);
    const glm::vec3 nh = (t1p * t1) + (t2p * t2) +
                          (std::sqrt(std::max(0.0F, 1.0F - (t1p * t1p) - (t2p * t2p))) * vh);
    return glm::normalize(glm::vec3(alpha * nh.x, alpha * nh.y, std::max(0.0F, nh.z)));
}

// Kulla-Conty energy tables, baked by tools/albedo_table.cpp (Kulla & Conty 2017).
#include "albedo_table.inc"

// Refract wo about microfacet normal ht. Returns false on total internal reflection at that facet.
bool refractAbout(const glm::vec3& wo, const glm::vec3& ht, float eta, glm::vec3& wi) {
    const float cosI = glm::dot(wo, ht);
    if (cosI <= 0.0F) {
        return false;
    }
    const float cos2T = cos2Transmitted(cosI, eta);
    if (cos2T < 0.0F) {
        return false;
    }
    wi = ((eta * cosI) - std::sqrt(cos2T)) * ht - (eta * wo);
    return true;
}

float lerp1(float a, float b, float t) { return a + ((b - a) * t); }

// Schlick-split albedo pair: Ess(f0) = f0*a + b, and a+b = E (the f0=1 case).
struct AlbedoSplit {
    float a;
    float b;
    [[nodiscard]] float total() const { return a + b; }                 // E, Fresnel-free
    [[nodiscard]] float at(float f0) const { return (f0 * a) + b; }
};

// Bilinear lookup indexed by sqrt(mu), matching the generator's grid: E reaches its plateau over mu ~ alpha, under a cell if uniform.
AlbedoSplit directionalAlbedo(float mu, float roughness) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kAlbedoRoughnessRes - 1);
    const float mf = std::sqrt(std::clamp(mu, 0.0F, 1.0F)) * (kAlbedoMuRes - 1);
    const int r0 = std::min(static_cast<int>(rf), kAlbedoRoughnessRes - 2);
    const int m0 = std::min(static_cast<int>(mf), kAlbedoMuRes - 2);
    const float rt = rf - static_cast<float>(r0);
    const float mt = mf - static_cast<float>(m0);
    const int i0 = (r0 * kAlbedoMuRes) + m0;
    const int i1 = ((r0 + 1) * kAlbedoMuRes) + m0;
    return {lerp1(lerp1(kAlbedoA[i0], kAlbedoA[i0 + 1], mt),
                   lerp1(kAlbedoA[i1], kAlbedoA[i1 + 1], mt), rt),
             lerp1(lerp1(kAlbedoB[i0], kAlbedoB[i0 + 1], mt),
                   lerp1(kAlbedoB[i1], kAlbedoB[i1 + 1], mt), rt)};
}

AlbedoSplit averageAlbedo(float roughness) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kAlbedoRoughnessRes - 1);
    const int r0 = std::min(static_cast<int>(rf), kAlbedoRoughnessRes - 2);
    const float rt = rf - static_cast<float>(r0);
    return {lerp1(kAlbedoAvgA[r0], kAlbedoAvgA[r0 + 1], rt),
             lerp1(kAlbedoAvgB[r0], kAlbedoAvgB[r0 + 1], rt)};
}

// --- Reflected multiple-scattering lobe; cosine sampling costs up to +17.3 relative variance at low roughness.
struct MsReflectRow {
    int base;
    float blend;
};

MsReflectRow msReflectRow(float roughness) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kAlbedoRoughnessRes - 1);
    const int r0 = std::min(static_cast<int>(rf), kAlbedoRoughnessRes - 2);
    return {r0 * kMsReflectMuRes, rf - static_cast<float>(r0)};
}

float msReflectDensity(const MsReflectRow& row, int index) {
    return lerp1(kMsReflectDensity[row.base + index],
                  kMsReflectDensity[row.base + kMsReflectMuRes + index], row.blend);
}

float msReflectCdf(const MsReflectRow& row, int index) {
    return lerp1(kMsReflectCdf[row.base + index], kMsReflectCdf[row.base + kMsReflectMuRes + index],
                  row.blend);
}

// Solid-angle density: the mu density over 2*pi of azimuth. Uniform in mu, not sqrt(mu), this being the sampling grid.
float msReflectPdf(float mu, float roughness) {
    const MsReflectRow row = msReflectRow(roughness);
    const float mf = std::clamp(mu, 0.0F, 1.0F) * (kMsReflectMuRes - 1);
    const int m0 = std::min(static_cast<int>(mf), kMsReflectMuRes - 2);
    const float mt = mf - static_cast<float>(m0);
    return lerp1(msReflectDensity(row, m0), msReflectDensity(row, m0 + 1), mt) / (2.0F * kPi);
}

// Exact inversion of a tabulated piecewise-linear density over mu.
template <typename Density, typename Cdf>
float invertPiecewiseLinearDensity(Density density, Cdf cdf, int resolution, float u) {
    int low = 0;
    int high = resolution - 1;
    while (high - low > 1) {
        const int mid = (low + high) / 2;
        (cdf(mid) <= u ? low : high) = mid;
    }
    const float step = 1.0F / static_cast<float>(resolution - 1);
    const float q0 = density(low);
    const float dq = density(low + 1) - q0;
    const float c = (u - cdf(low)) / step;
    const float root = std::sqrt(std::max((q0 * q0) + (2.0F * dq * c), 0.0F));
    const float t = std::clamp((2.0F * c) / std::max(q0 + root, 1e-9F), 0.0F, 1.0F);
    return (static_cast<float>(low) + t) * step;
}

glm::vec3 sampleMsReflect(float roughness, glm::vec2 u) {
    const MsReflectRow row = msReflectRow(roughness);
    const float mu = invertPiecewiseLinearDensity([&](int i) { return msReflectDensity(row, i); },
                                                   [&](int i) { return msReflectCdf(row, i); },
                                                   kMsReflectMuRes, u.x);
    const float r = std::sqrt(std::max(0.0F, 1.0F - (mu * mu)));
    const float phi = 2.0F * kPi * u.y;
    return {r * std::cos(phi), r * std::sin(phi), mu};
}

// Fractional index into the log-spaced eta axis, clamped to the tabulated range.
float etaAxisCoord(float eta) {
    const float logMin = std::log(kEtaMin);
    const float u = (std::log(std::clamp(eta, kEtaMin, kEtaMax)) - logMin) / (std::log(kEtaMax) - logMin);
    return u * (kEtaRes - 1);
}

// Reflected and transmitted escaping shares of a dielectric interface, exact Fresnel already applied.
struct EscapeSplit {
    float reflect;
    float transmit;
    [[nodiscard]] float total() const { return reflect + transmit; }
};

// Trilinear over (roughness, mu, eta).
EscapeSplit escapeAlbedo(float mu, float roughness, float eta) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kTransmitRoughnessRes - 1);
    // The generator's escapeMu axis: nodes uniform in sqrt(mu), dense where E climbs from its grazing limit.
    const float mf = std::sqrt(std::clamp(mu, 0.0F, 1.0F)) * (kTransmitMuRes - 1);
    const float ef = etaAxisCoord(eta);
    const int r0 = std::min(static_cast<int>(rf), kTransmitRoughnessRes - 2);
    const int m0 = std::min(static_cast<int>(mf), kTransmitMuRes - 2);
    const int e0 = std::min(static_cast<int>(ef), kEtaRes - 2);
    const float rt = rf - static_cast<float>(r0);
    const float mt = mf - static_cast<float>(m0);
    const float et = ef - static_cast<float>(e0);
    const auto fetch = [&](const auto& channel, int r, int m) {
        const int base = (((r * kTransmitMuRes) + m) * kEtaRes) + e0;
        return lerp1(channel[base], channel[base + 1], et);
    };
    const auto bilinear = [&](const auto& channel) {
        return lerp1(lerp1(fetch(channel, r0, m0), fetch(channel, r0, m0 + 1), mt),
                      lerp1(fetch(channel, r0 + 1, m0), fetch(channel, r0 + 1, m0 + 1), mt), rt);
    };
    return {bilinear(kEscapeReflect), bilinear(kEscapeTransmit)};
}

EscapeSplit averageEscapeAlbedo(float roughness, float eta) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kTransmitRoughnessRes - 1);
    const float ef = etaAxisCoord(eta);
    const int r0 = std::min(static_cast<int>(rf), kTransmitRoughnessRes - 2);
    const int e0 = std::min(static_cast<int>(ef), kEtaRes - 2);
    const float rt = rf - static_cast<float>(r0);
    const float et = ef - static_cast<float>(e0);
    const auto fetch = [&](const auto& channel, int r) {
        const int base = (r * kEtaRes) + e0;
        return lerp1(channel[base], channel[base + 1], et);
    };
    return {lerp1(fetch(kEscapeAvgReflect, r0), fetch(kEscapeAvgReflect, r0 + 1), rt),
             lerp1(fetch(kEscapeAvgTransmit, r0), fetch(kEscapeAvgTransmit, r0 + 1), rt)};
}

// --- Transmitted multiple-scattering lobe, the far-hemisphere twin: bilinear over (roughness, eta) of four rows, mu stride kEtaRes.
template <typename Table>
float msTransmitBlend(const Table& table, const MsTransmitRow& row, int index) {
    const int offset = index * kEtaRes;
    return (row.weight[0] * table[row.base[0] + offset]) + (row.weight[1] * table[row.base[1] + offset]) +
           (row.weight[2] * table[row.base[2] + offset]) + (row.weight[3] * table[row.base[3] + offset]);
}

// Stored unnormalised, divided by its blended total here: integration is linear, so blended prefix integrals are the blend's integral.
MsTransmitRow msTransmitRow(float roughness, float eta) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kTransmitRoughnessRes - 1);
    const float ef = etaAxisCoord(eta);
    const int r0 = std::min(static_cast<int>(rf), kTransmitRoughnessRes - 2);
    const int e0 = std::min(static_cast<int>(ef), kEtaRes - 2);
    const float rt = rf - static_cast<float>(r0);
    const float et = ef - static_cast<float>(e0);
    const int base0 = (r0 * kTransmitMuRes * kEtaRes) + e0;
    const int base1 = base0 + (kTransmitMuRes * kEtaRes);
    MsTransmitRow row{{base0, base0 + 1, base1, base1 + 1},
                       {(1.0F - rt) * (1.0F - et), (1.0F - rt) * et, rt * (1.0F - et), rt * et},
                       0.0F};
    // Last prefix integral is the row's energy deficit; zero means no energy to carry, so it reports zero density rather than dividing.
    const float total = msTransmitBlend(kMsTransmitCdf, row, kTransmitMuRes - 1);
    row.scale = total > 0.0F ? 1.0F / total : 0.0F;
    return row;
}

float msTransmitDensity(const MsTransmitRow& row, int index) {
    return msTransmitBlend(kMsTransmitDensity, row, index) * row.scale;
}

float msTransmitCdf(const MsTransmitRow& row, int index) {
    return msTransmitBlend(kMsTransmitCdf, row, index) * row.scale;
}

// Solid-angle density: the mu density spread over 2*pi of azimuth, mu measured from the far-side normal.
float msTransmitPdf(float mu, const MsTransmitRow& row) {
    const float mf = std::clamp(mu, 0.0F, 1.0F) * (kTransmitMuRes - 1);
    const int m0 = std::min(static_cast<int>(mf), kTransmitMuRes - 2);
    const float mt = mf - static_cast<float>(m0);
    return lerp1(msTransmitDensity(row, m0), msTransmitDensity(row, m0 + 1), mt) / (2.0F * kPi);
}

// Returns the near-hemisphere direction; the caller mirrors z, as the cosine draw it replaces did.
glm::vec3 sampleMsTransmit(const MsTransmitRow& row, glm::vec2 u) {
    const float mu = invertPiecewiseLinearDensity([&](int i) { return msTransmitDensity(row, i); },
                                                   [&](int i) { return msTransmitCdf(row, i); }, kTransmitMuRes, u.x);
    const float r = std::sqrt(std::max(0.0F, 1.0F - (mu * mu)));
    const float phi = 2.0F * kPi * u.y;
    return {r * std::cos(phi), r * std::sin(phi), mu};
}

}  // namespace

// Malley's method, a uniform disk point lifted to the hemisphere (PBR 4th ed. 13.6.3). External for the AO lane, where the pdf cancels.
glm::vec3 sampleCosineHemisphere(glm::vec2 u) {
    const float r = std::sqrt(u.x);
    const float phi = 2.0F * kPi * u.y;
    return {r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0F, 1.0F - u.x))};
}

// Cosine-weighted average Fresnel, 2*int_0^1 F(mu)*mu dmu, one 3-node rule for both.
constexpr float kFresnelAvgNodes[3] = {0.105319802F, 0.382154433F, 0.796427281F};
constexpr float kFresnelAvgWeights[3] = {0.038972482F, 0.280518736F, 0.680508783F};

// The dielectric interface. Conductor nodes carried unrefitted: worst 5.5e-5 over ior [1.05, 3.0]. Exactly +0 at ior 1, structurally.
float dielectricFresnelAvg(float ior) {
    float sum = 0.0F;
    for (int i = 0; i < 3; ++i) {
        sum += kFresnelAvgWeights[i] * fresnelDielectric(kFresnelAvgNodes[i], 1.0F, ior);
    }
    return sum;
}

// External linkage under the same rule: checkAlbedoTableInterpolation is the only instrument that sees the .inc's interpolation error.
glm::vec2 directionalAlbedoSplit(float mu, float roughness) {
    const AlbedoSplit split = directionalAlbedo(mu, roughness);
    return {split.a, split.b};
}

glm::vec2 averageAlbedoSplit(float roughness) {
    const AlbedoSplit split = averageAlbedo(roughness);
    return {split.a, split.b};
}

// The grid the two lookups index, described rather than transcribed. Both axes edge-aligned, so 0 and res-1 are exact endpoints.
glm::ivec2 albedoGridRes() { return {kAlbedoRoughnessRes, kAlbedoMuRes}; }

float albedoGridRoughness(float index) { return index / static_cast<float>(kAlbedoRoughnessRes - 1); }

// Inverts directionalAlbedo's sqrt(mu) index, so an instrument's fractional index lands where that lookup interpolates.
float albedoGridMu(float index) {
    const float t = index / static_cast<float>(kAlbedoMuRes - 1);
    return t * t;
}

// The conductor interface. Karis' mean is exact for Schlick, so it averages a different function once the scatter is complex-IOR.
glm::vec3 conductorFresnelAvg(const glm::vec3& n, const glm::vec3& k) {
    glm::vec3 sum(0.0F);
    for (int i = 0; i < 3; ++i) {
        sum += kFresnelAvgWeights[i] * fresnelConductor(kFresnelAvgNodes[i], n, k);
    }
    return sum;
}

// Fraunhofer d, F and C lines, where V_d = (n_d-1)/(n_F-n_C) is defined: physical constants of the definition, not tuning.
constexpr float kLambdaDNm = 587.56F;
constexpr float kLambdaFNm = 486.13F;
constexpr float kLambdaCNm = 656.27F;

// Cauchy n(lambda) = A + B/lambda^2, (A,B) from (n_d, V_d) per KHR_materials_dispersion.
float cauchyIor(float iorD, float abbe, float lambdaNm) {
    if (abbe <= 0.0F) {
        return iorD;
    }
    const float b = (iorD - 1.0F) /
                    (abbe * ((1.0F / (kLambdaFNm * kLambdaFNm)) - (1.0F / (kLambdaCNm * kLambdaCNm))));
    const float a = iorD - (b / (kLambdaDNm * kLambdaDNm));
    return a + (b / (lambdaNm * lambdaNm));
}

namespace {

glm::vec3 schlickFresnelAvg(const glm::vec3& f0) { return f0 + ((glm::vec3(1.0F) - f0) / 21.0F); }

// Normal-incidence reflectance implied by the ior: the coat's own f0, independent of the f0 texture the conductor path uses.
float dielectricF0(float ior) {
    const float r = (ior - 1.0F) / (ior + 1.0F);
    return r * r;
}

// Kulla-Conty tint: the share of (1-E) energy surviving repeated bounces, each attenuated by Favg. Exactly 1 at Favg=1.
float multiScatterTint(float fresnelAvg, float albedoAvg) {
    return (fresnelAvg * fresnelAvg * albedoAvg) /
           std::max(1.0F - (fresnelAvg * (1.0F - albedoAvg)), 1e-4F);
}

float schlickScalar(float cosTheta, float f0) {
    return f0 + ((1.0F - f0) * std::pow(std::clamp(1.0F - cosTheta, 0.0F, 1.0F), 5.0F));
}

// etaI/etaT rather than a bare ior: fresnelDielectric returns exactly 1.0 past the critical angle, and Schlick cannot express TIR.
float coatFresnelRatio(float cosTheta, float etaI, float etaT, float f0) {
    return fresnelDielectric(cosTheta, etaI, etaT) / std::max(schlickScalar(cosTheta, f0), 1e-6F);
}

// Coat albedo, not the macro-facet F(mu_o): 4x at roughness 1, mu 0.4, 10% energy lost.
float coatAlbedo(const AlbedoSplit& split, float albedoAvg, float f0, float fresnelRatio,
                  float fresnelAvg) {
    return (split.at(f0) * fresnelRatio) +
           (multiScatterTint(fresnelAvg, albedoAvg) * (1.0F - split.total()));
}

// Below this the GGX transmission lobe is a delta (PBRT's EffectivelySmooth); kMinAlpha sits inside it, so smooth glass stays exact.
constexpr float kSmoothAlpha = 1e-3F;

// ior == 1 is a delta at every roughness: the half-vector normalizes zero, NaN on 7783 of 7783 transmission draws at roughness 0.1.
bool transmissionIsRough(const BsdfParams& params, float alpha) {
    return params.transmissionFactor > 0.0F && alpha >= kSmoothAlpha && params.ior != 1.0F;
}

struct LobeEval {
    glm::vec3 f;
    float pdf;
};

// The transmitted share of the multiple-scattering energy for any far-side wi. K*pdf/mu integrates to exactly K, so no gate is needed.
glm::vec3 transmitMultiScatter(const BsdfParams& params, float mu, float msPdf, const LobeProbabilities& lobes) {
    return params.transmissionTint * lobes.transmitWeight * lobes.transmitShare * lobes.etaSq *
           (std::max(1.0F - lobes.escapeWo, 0.0F) * msPdf / mu);
}

// The full reciprocal coupling at wi: the wo half is in lobes.diffuseKd, the wi half the same (1 - coatAlbedo) evaluated here.
float diffuseKdAt(const glm::vec3& wi, const LobeProbabilities& lobes, const AlbedoSplit& splitWi) {
    const float coat = coatAlbedo(splitWi, lobes.albedoAvg, lobes.coatF0,
                                   coatFresnelRatio(wi.z, lobes.etaI, lobes.etaT, lobes.coatF0),
                                   lobes.coatFresnelAvg);
    return std::max(lobes.diffuseKd, 0.0F) * (1.0F - coat);
}

// --- EON rough-diffuse BRDF (Portsmouth, Kutz, Hill 2025, JCGT 14(1)); ported from the paper's GLSL, never hand-derived.

constexpr float kConstant1Fon = 0.5F - (2.0F / (3.0F * kPi));
constexpr float kConstant2Fon = (2.0F / 3.0F) - (28.0F / (15.0F * kPi));

// FON directional albedo, quartic fit (paper eq. 14): within 0.1% of the exact form and ~5x cheaper, so used exclusively.
float evalFonAlbedoApprox(float mu, float r) {
    const float muComplement = 1.0F - mu;
    constexpr float g1 = 0.0571085289F;
    constexpr float g2 = 0.491881867F;
    constexpr float g3 = -0.332181442F;
    constexpr float g4 = 0.0714429953F;
    const float gOverPi =
        muComplement * (g1 + (muComplement * (g2 + (muComplement * (g3 + (muComplement * g4))))));
    const float af = 1.0F / (1.0F + (kConstant1Fon * r));
    return (1.0F + (r * gOverPi)) * af;
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

namespace {

// EON BRDF value (paper eq. 16-19): FON single scatter plus an analytic multiple-scattering lobe. rho is not the authored colour.
glm::vec3 evaluateEon(const glm::vec3& rho, float r, const glm::vec3& wi, const glm::vec3& wo) {
    const float muI = wi.z;
    const float muO = wo.z;
    const float s = glm::dot(wi, wo) - (muI * muO);
    const float sOverT = s > 0.0F ? s / std::max(muI, muO) : s;
    const float af = 1.0F / (1.0F + (kConstant1Fon * r));
    const glm::vec3 singleScatter = (rho / kPi) * af * (1.0F + (r * sOverT));

    const float eFonO = evalFonAlbedoApprox(muO, r);
    const float eFonI = evalFonAlbedoApprox(muI, r);
    const float avgEFon = af * (1.0F + (kConstant2Fon * r));
    const glm::vec3 rhoMs = (rho * rho) * avgEFon / (glm::vec3(1.0F) - (rho * (1.0F - avgEFon)));
    constexpr float kEps = 1e-7F;
    const glm::vec3 multiScatter = (rhoMs / kPi) * std::max(kEps, 1.0F - eFonO) *
                                    std::max(kEps, 1.0F - eFonI) / std::max(kEps, 1.0F - avgEFon);
    return singleScatter + multiScatter;
}

// Uniform hemisphere direction, pdf = 1/(2*pi): EON's defensive-sampling companion to CLTC (Owen & Zhou 2000 one-sample MIS).
glm::vec3 sampleUniformHemisphereEon(glm::vec2 u) {
    const float sinTheta = std::sqrt(std::max(0.0F, 1.0F - (u.x * u.x)));
    const float phi = 2.0F * kPi * u.y;
    return {sinTheta * std::cos(phi), sinTheta * std::sin(phi), u.x};
}

struct EonLtcCoeffs {
    float a;
    float b;
    float c;
    float d;
};

// Fitted LTC matrix coefficients (paper Listing 2) matching EON's cosine-weighted backscattering lobe at a view angle and roughness.
EonLtcCoeffs eonLtcCoeffs(float mu, float r) {
    const float a = 1.0F + (r * (0.303392F + (((-0.518982F + (0.111709F * mu)) * mu) +
                                                ((-0.276266F + (0.335918F * mu)) * r))));
    const float b = (r * (-1.16407F + (1.15859F * mu) + ((0.150815F - (0.150105F * mu)) * r))) /
                    ((mu * mu * mu) - 1.43545F);
    const float c = 1.0F + (r * (0.20013F + ((-0.506373F + (0.261777F * mu)) * mu)));
    const float d = (r * (0.540852F + ((-1.01625F + (0.475392F * mu)) * mu))) /
                    (-1.0743F + ((0.0725628F + mu) * mu));
    return {a, b, c, d};
}

// Orthonormal frame aligning wLocal's azimuth to the x-axis, to move into and out of the space the LTC fit is expressed in.
glm::mat3 orthonormalBasisLtc(const glm::vec3& wLocal) {
    const float lenSq = (wLocal.x * wLocal.x) + (wLocal.y * wLocal.y);
    const glm::vec3 x = lenSq > 0.0F ? glm::vec3(wLocal.x, wLocal.y, 0.0F) * (1.0F / std::sqrt(lenSq))
                                       : glm::vec3(1.0F, 0.0F, 0.0F);
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

// pdf of cltcSample's distribution at an arbitrary wiLocal (paper Listing 3's cltc_pdf), evaluated independently of how wiLocal arose.
float cltcPdf(const glm::vec4& m, const glm::mat3& basisT, float s, const glm::vec3& wiLocal) {
    const glm::vec3 wi = basisT * wiLocal;
    const glm::vec3 wh(m.z * (wi.x - (m.y * wi.z)), (m.x - (m.y * m.w)) * wi.y,
                        -m.z * ((m.w * wi.x) - (m.x * wi.z)));
    const float lenSq = glm::dot(wh, wh);
    const float detM = m.z * (m.x - (m.y * m.w));
    return (detM * detM) / std::max(lenSq * lenSq, 1e-12F) * std::max(wh.z, 0.0F) / (kPi * s);
}

// Mixing weight between the CLTC and uniform lobes (paper Sec. 4): CLTC alone has a variance spike the uniform term corrects.
float eonUniformMixWeight(float mu, float r) {
    const float inner = 0.538233F - (0.290822F * mu);
    const float mid = -0.372058F + (inner * mu);
    return std::pow(r, 0.1F) * (0.162925F + (mid * mu));
}

// Samples EON's distribution (paper Sec. 4), one-sample MIS between CLTC and uniform. Direction only; pdfEon owns the density.
glm::vec3 sampleEon(const LobeProbabilities& lobes, glm::vec2 u) {
    const float pUniform = lobes.eonUniformMix;
    // Strict: pUniform is exactly 0 at r=0, where an inclusive test admits u.x==0 and reshuffles it as 0/0, poisoning the pixel with NaN.
    if (u.x < pUniform) {
        u.x /= pUniform;
        return sampleUniformHemisphereEon(u);
    }
    u.x = (u.x - pUniform) / (1.0F - pUniform);
    return cltcSample(lobes.eonLtcM, lobes.eonLtcBasisT, lobes.eonLtcS, u);
}

// pdf of sampleEon's distribution at an arbitrary wiLocal.
float pdfEon(const glm::vec3& wiLocal, const LobeProbabilities& lobes) {
    const float pUniform = lobes.eonUniformMix;
    constexpr float kUniformHemispherePdf = 1.0F / (2.0F * kPi);
    return (pUniform * kUniformHemispherePdf) +
           ((1.0F - pUniform) * cltcPdf(lobes.eonLtcM, lobes.eonLtcBasisT, lobes.eonLtcS, wiLocal));
}

LobeEval evaluateDiffuseLobe(const BsdfParams& params, const glm::vec3& wo, const glm::vec3& wi,
                              const LobeProbabilities& lobes, const AlbedoSplit& splitWi) {
    if (wi.z <= 0.0F || wo.z <= 0.0F) {
        return {glm::vec3(0.0F), 0.0F};
    }
    const glm::vec3 f = evaluateEon(params.diffuseRho, params.diffuseRoughness, wi, wo);
    return {f * diffuseKdAt(wi, lobes, splitWi), pdfEon(wi, lobes)};
}

// Channel mean of the Fresnel evaluateSpecularLobe applies at a facet, metallic blend included.
float facetReflectance(const BsdfParams& params, float cosTheta, float fDielectric, const LobeProbabilities& lobes) {
    if (params.metallic <= 0.0F) {
        return fDielectric;
    }
    const glm::vec3 conductor = fresnelConductor(cosTheta, lobes.conductorN, lobes.conductorK);
    return glm::mix(fDielectric, (conductor.x + conductor.y + conductor.z) / 3.0F, params.metallic);
}

// Probability the VNDF strategy reflects about a facet (Walter 2007 5.3); callers gate facetTransmit > 0, so the denominator is positive.
float facetReflectProbability(const BsdfParams& params, float cosTheta, float fDielectric, const LobeProbabilities& lobes) {
    const float reflect = facetReflectance(params, cosTheta, fDielectric, lobes);
    return reflect / (reflect + ((1.0F - fDielectric) * lobes.facetTransmit));
}

// Single scatter D*G2*F/(4*ndotV*ndotL) plus the Kulla-Conty lobe, and the VNDF pdf (Heitz 2018 eq.3) times its Jacobian.
LobeEval evaluateSpecularLobe(const BsdfParams& params, const glm::vec3& wo, const glm::vec3& wi,
                               float alpha, const LobeProbabilities& lobes, const AlbedoSplit& splitWi) {
    if (wo.z <= 0.0F || wi.z <= 0.0F) {
        return {glm::vec3(0.0F), 0.0F};
    }
    const glm::vec3 nh = glm::normalize(wo + wi);
    const float woDotNh = std::max(glm::dot(wo, nh), 0.0F);
    if (woDotNh <= 0.0F) {
        return {glm::vec3(0.0F), 0.0F};
    }
    const float d = distributionGGX(nh, alpha);
    const float fDielectric = fresnelDielectric(woDotNh, lobes.etaI, lobes.etaT);
    // Gated, not mixed away at weight 0: glm::mix is a + t*(b-a), so a non-finite conductor term would survive t=0 as NaN.
    const glm::vec3 f =
        params.metallic > 0.0F
            ? glm::mix(glm::vec3(fDielectric),
                        fresnelConductor(woDotNh, lobes.conductorN, lobes.conductorK), params.metallic)
            : glm::vec3(fDielectric);
    const glm::vec3 singleScatter = d * smithVisibility(wo.z, wi.z, alpha) * f;

    // Kulla & Conty 2017. Integrates to (1-E(mu_o)) at Favg=1, so a white conductor conserves within the 1% the furnace bounds.
    const glm::vec3& fms = lobes.multiScatterFms;
    const float albedoWi = splitWi.total();
    const glm::vec3 opaqueMs = fms * ((1.0F - lobes.albedoWo) * (1.0F - albedoWi)) /
                                (kPi * std::max(1.0F - lobes.albedoAvg, 1e-4F));
    // transmitWeight is exactly zero for every opaque material, so skip the escape-shape row rather than mix it away at weight 0.
    const glm::vec3 multiScatter =
        lobes.transmitWeight > 0.0F
            ? glm::mix(opaqueMs,
                        glm::vec3((1.0F - lobes.transmitShare) * std::max(1.0F - lobes.escapeWo, 0.0F) *
                                  msTransmitPdf(wi.z, lobes.reflectShape) /
                                  wi.z),
                        lobes.transmitWeight)
            : opaqueMs;

    const float vndfPdf = 0.25F * d * smithG1OverCos(wo.z, alpha);
    return {singleScatter + multiScatter,
            lobes.facetTransmit > 0.0F ? vndfPdf * facetReflectProbability(params, woDotNh, fDielectric, lobes) : vndfPdf};
}

// The conductor's Fresnel state at wo; index-matched (1, 0) with zero luma and average off the metal path, as evaluateSpecularLobe gates.
struct ConductorAtWo {
    ConductorIor ior;
    float luma;  // channel mean of the conductor Fresnel at wo
    glm::vec3 avg;  // cosine-weighted mean Fresnel, the Kulla-Conty tint input
};

ConductorAtWo conductorAtWo(const BsdfParams& params, float cosWo) {
    if (params.metallic <= 0.0F) {
        return {{glm::vec3(1.0F), glm::vec3(0.0F)}, 0.0F, glm::vec3(0.0F)};
    }
    const ConductorIor conductor = conductorIorFromReflectivity(params.f0, params.edgeTint);
    const glm::vec3 conductorF = fresnelConductor(cosWo, conductor.n, conductor.k);
    return {conductor, (conductorF.x + conductorF.y + conductorF.z) / 3.0F, conductorFresnelAvg(conductor.n, conductor.k)};
}

// The dielectric coat over the diffuse substrate: its f0, its cosine-mean Fresnel, and the diffuse lobe's reciprocal coupling at wo.
struct CoatTerms {
    float f0;
    float fresnelAvg;
    float diffuseCoupling;
};

// Reciprocal diffuse coupling, renormalised by 1/(1-coatAlbedoAvg) and symmetric in wo/wi, which the bare (1-F(mu_o)) form was not.
CoatTerms coatTerms(const BsdfParams& params, const AlbedoSplit& splitWo, const AlbedoSplit& splitAvg, float cosWo, float etaI,
                    float etaT) {
    const float coatF0 = dielectricF0(params.ior);
    const float dielectricAvg = dielectricFresnelAvg(params.ior);
    const float coatAlbedoAvg =
        coatAlbedo(splitAvg, splitAvg.total(), coatF0,
                    dielectricAvg / std::max(schlickFresnelAvg(glm::vec3(coatF0)).x, 1e-6F),
                    dielectricAvg);
    const float diffuseCoupling = (1.0F - coatAlbedo(splitWo, splitAvg.total(), coatF0,
                                                       coatFresnelRatio(cosWo, etaI, etaT, coatF0),
                                                       dielectricAvg)) /
                                   std::max(1.0F - coatAlbedoAvg, 1e-4F);
    return {coatF0, dielectricAvg, diffuseCoupling};
}

// Opaque: specular is the Fresnel probability times E, the rest split between diffuse and msReflect by the energy each carries.
void selectOpaqueLobes(LobeProbabilities& lobes, float specularFresnel, float msReflectEnergy, float diffuseEnergy) {
    // Scaled by E: VNDF draws only the single-scattering part, the (1-E)cos-shaped rest belonging to msReflect's selection mass.
    const float specularProb = std::clamp(specularFresnel * lobes.albedoWo, 0.05F, 0.95F);
    const float diffuseProb = 1.0F - specularProb;
    // Splits the non-specular mass by the energy each strategy carries; without it the Kulla-Conty lobe borrows the diffuse slot.
    const float msReflectProb =
        diffuseProb *
        (msReflectEnergy + diffuseEnergy > 1e-6F ? msReflectEnergy / (msReflectEnergy + diffuseEnergy) : 1.0F);
    lobes.specular = specularProb;
    lobes.diffuse = diffuseProb - msReflectProb;
    lobes.msReflect = msReflectProb;
}

// Transmissive: each strategy's mass is proportional to its energy at wo, as flux without the eta^2 compression (PBRT-v4).
void selectTransmissiveLobes(LobeProbabilities& lobes, const BsdfParams& params, float alpha, float eta, float cosWo,
                             float conductorLuma, float msReflectEnergy, float diffuseEnergy) {
    float reflectSs = 0.0F;
    float transmitSs = 1.0F;
    if (params.ior == 1.0F) {
        lobes.escapeWo = 1.0F;
        lobes.transmitShare = 1.0F;
    } else {
        // R + T, unweighted by transmissionFactor: energy it withholds from the transmit lobe enters the diffuse substrate instead.
        const EscapeSplit escapeWo = escapeAlbedo(cosWo, params.roughness, eta);
        const EscapeSplit escapeMean = averageEscapeAlbedo(params.roughness, eta);
        lobes.escapeWo = escapeWo.total();
        lobes.transmitShape = msTransmitRow(params.roughness, 1.0F / eta);
        lobes.reflectShape = msTransmitRow(params.roughness, eta);
        lobes.transmitShare = lobes.transmitWeight * escapeMean.transmit / std::max(escapeMean.total(), 1e-4F);
        reflectSs = escapeWo.reflect;
        transmitSs = escapeWo.transmit;
    }
    const float transmitWeight = lobes.transmitWeight;
    const float deficit = std::max(1.0F - lobes.escapeWo, 0.0F);
    const bool rough = transmissionIsRough(params, alpha);
    // The conductor's single scatter is macro Fresnel times E, as in the opaque split; the dielectric's is the table's R_ss.
    const float specularEnergy = glm::mix(reflectSs, conductorLuma * lobes.albedoWo, params.metallic);
    const float opaqueMsEnergy = (1.0F - transmitWeight) * msReflectEnergy;
    // A zero-scale row has no density to draw from, and its value reads the same zero.
    const float msReflectTransmissiveEnergy =
        lobes.reflectShape.scale > 0.0F ? transmitWeight * (1.0F - lobes.transmitShare) * deficit : 0.0F;
    // A rough interface's refraction is the VNDF strategy's other branch, so its energy joins the specular mass; a delta keeps its own.
    const float transmitEnergy = rough ? transmitWeight * transmitSs : lobes.transmitPhysicalValue;
    // evaluateContinuousLobes drops the far-side multiple scattering of a delta interface, so it has no energy to select for.
    const float msTransmitEnergy = rough && lobes.transmitShape.scale > 0.0F
                                       ? transmitWeight * lobes.transmitShare * deficit
                                       : 0.0F;
    const float total = specularEnergy + diffuseEnergy + opaqueMsEnergy + msReflectTransmissiveEnergy +
                        transmitEnergy + msTransmitEnergy;
    // Zero only where every lobe's value is zero -- a fully metallic interface with conductor Fresnel 0 -- leaving sampleBsdf nothing.
    const float inverseTotal = total > 0.0F ? 1.0F / total : 0.0F;
    lobes.specular = (rough ? specularEnergy + transmitEnergy : specularEnergy) * inverseTotal;
    lobes.diffuse = diffuseEnergy * inverseTotal;
    lobes.msReflect = opaqueMsEnergy * inverseTotal;
    lobes.msReflectTransmissive = msReflectTransmissiveEnergy * inverseTotal;
    lobes.transmit = rough ? 0.0F : transmitEnergy * inverseTotal;
    lobes.msTransmit = msTransmitEnergy * inverseTotal;
    lobes.facetTransmit = rough ? transmitWeight : 0.0F;
}

// The wo-side state every lobe shares, then each strategy's selection mass: by Fresnel and E if opaque, by energy if transmissive.
LobeProbabilities computeLobeProbabilities(const BsdfParams& params, const glm::vec3& wo, float sign,
                                            float alpha) {
    const bool exiting = sign < 0.0F && params.transmissionFactor > 0.0F;
    const float etaI = exiting ? params.ior : 1.0F;
    const float etaT = exiting ? 1.0F : params.ior;
    const float fresnelAtNormal = fresnelDielectric(wo.z, etaI, etaT);
    const ConductorAtWo conductor = conductorAtWo(params, wo.z);
    const AlbedoSplit splitWo = directionalAlbedo(wo.z, params.roughness);
    const AlbedoSplit splitAvg = averageAlbedo(params.roughness);
    const float transmittance = (1.0F - fresnelAtNormal) * (1.0F - params.metallic);
    const CoatTerms coat = coatTerms(params, splitWo, splitAvg, wo.z, etaI, etaT);
    const float diffuseKd = exiting ? 0.0F : coat.diffuseCoupling * (1.0F - params.metallic) * (1.0F - params.transmissionFactor);
    const float transmitPhysicalValue = exiting ? transmittance : transmittance * params.transmissionFactor;
    // Each interface's own cosine mean over the Fresnel its single scatter evaluates; conductor.avg is 0 off the metal path.
    const glm::vec3 fresnelAvg = glm::mix(glm::vec3(coat.fresnelAvg), conductor.avg, params.metallic);
    // msEnergy is exact: opaqueMs integrates to fms*(1-E(mu_o)), since int (1-E(mu_i)) cos = pi*(1-Eavg). Mass need only be proportional.
    const glm::vec3 fms(multiScatterTint(fresnelAvg.x, splitAvg.total()), multiScatterTint(fresnelAvg.y, splitAvg.total()),
                         multiScatterTint(fresnelAvg.z, splitAvg.total()));
    const float msReflectEnergy = ((fms.x + fms.y + fms.z) / 3.0F) * std::max(1.0F - splitWo.total(), 0.0F);
    const float diffuseEnergy =
        diffuseKd * (params.diffuseRho.x + params.diffuseRho.y + params.diffuseRho.z) / 3.0F;
    // R_ss uses the coat's Schlick split with exact-Fresnel rescale, T_ss is (1-fc) scaled by (1-f0). transmitWeight gates entry only.
    const float eta = etaI / etaT;
    const float effectiveTransmission = exiting ? 1.0F : params.transmissionFactor;
    const EonLtcCoeffs eonLtc = eonLtcCoeffs(wo.z, params.diffuseRoughness);
    LobeProbabilities lobes{.specular = 0.0F,
                            .diffuse = 0.0F,
                            .msReflect = 0.0F,
                            .msReflectTransmissive = 0.0F,
                            .transmit = 0.0F,
                            .msTransmit = 0.0F,
                            .etaI = etaI,
                            .etaT = etaT,
                            .diffuseKd = diffuseKd,
                            .transmitPhysicalValue = transmitPhysicalValue,
                            .albedoWo = splitWo.total(),
                            .albedoAvg = splitAvg.total(),
                            .coatF0 = coat.f0,
                            .coatFresnelAvg = coat.fresnelAvg,
                            .fresnelAvg = fresnelAvg,
                            .multiScatterFms = fms,
                            .eonUniformMix = eonUniformMixWeight(wo.z, params.diffuseRoughness),
                            .eonLtcM = glm::vec4(eonLtc.a, eonLtc.b, eonLtc.c, eonLtc.d),
                            .eonLtcBasisT = glm::transpose(orthonormalBasisLtc(wo)),
                            .eonLtcS = 0.5F * (1.0F + (1.0F / std::sqrt((eonLtc.d * eonLtc.d) + 1.0F))),
                            .conductorN = conductor.ior.n,
                            .conductorK = conductor.ior.k,
                            .escapeWo = 0.0F,
                            .transmitShape = {},
                            .reflectShape = {},
                            .transmitShare = 0.0F,
                            .etaSq = eta * eta,
                            .transmitWeight = effectiveTransmission * (1.0F - params.metallic),
                            .facetTransmit = 0.0F};
    if (params.transmissionFactor <= 0.0F) {
        selectOpaqueLobes(lobes, glm::mix(fresnelAtNormal, conductor.luma, params.metallic), msReflectEnergy, diffuseEnergy);
    } else {
        selectTransmissiveLobes(lobes, params, alpha, eta, wo.z, conductor.luma, msReflectEnergy, diffuseEnergy);
    }
    return lobes;
}

// Walter 2007 (eq. 21, 16, 17), PBRT-v3 radiance form: eta^2 (Veach 1997 5.2) is already folded in, hence etaR^2 in the pdf only.
LobeEval evaluateTransmissionLobe(const BsdfParams& params, const glm::vec3& wo, const glm::vec3& wi,
                                   float alpha, const LobeProbabilities& lobes) {
    if (wo.z <= 0.0F || wi.z >= 0.0F) {
        return {glm::vec3(0.0F), 0.0F};
    }
    const float etaR = lobes.etaT / lobes.etaI;
    glm::vec3 ht = glm::normalize(wo + (etaR * wi));
    if (ht.z < 0.0F) {
        ht = -ht;
    }
    const float woDotH = glm::dot(wo, ht);
    const float wiDotH = glm::dot(wi, ht);
    // Both on the same side of the microfacet means this pair is a reflection about ht, not a refraction.
    if (woDotH <= 0.0F || wiDotH >= 0.0F) {
        return {glm::vec3(0.0F), 0.0F};
    }
    const float denom = woDotH + (etaR * wiDotH);
    const float denom2 = denom * denom;
    if (denom2 < 1e-12F) {
        return {glm::vec3(0.0F), 0.0F};
    }
    const float d = distributionGGX(ht, alpha);
    const float fresnel = fresnelDielectric(woDotH, lobes.etaI, lobes.etaT);
    // D*G2*|wi.h|*(wo.h)/(wo.z*|wi.z|*denom^2), with G2/(wo.z*|wi.z|) taken as 4*smithVisibility.
    const float common = (4.0F * d * smithVisibility(wo.z, -wi.z, alpha) * std::abs(wiDotH) * woDotH) / denom2;
    const float vndfPdf = d * woDotH * smithG1OverCos(wo.z, alpha);
    // transmitWeight carries the delta branch's factors: (1-metallic), a conductor transmitting nothing, and the entering side's factor.
    const float refractProbability =
        lobes.facetTransmit > 0.0F ? 1.0F - facetReflectProbability(params, woDotH, fresnel, lobes) : 0.0F;
    return {params.transmissionTint * (1.0F - fresnel) * lobes.transmitWeight * common,
             refractProbability * vndfPdf * etaR * etaR * std::abs(wiDotH) / denom2};
}

BsdfEval evaluateContinuousLobes(const BsdfParams& params, const glm::vec3& wo, const glm::vec3& wi,
                                  float alpha, const LobeProbabilities& lobes) {
    // Reflection and transmission occupy disjoint hemispheres, so the mixture is piecewise: no overlap between the two to double-count.
    if (wi.z < 0.0F) {
        // The multiple-scattering term stays inside this gate: a delta keeps pdf=0 on the far side, where a non-zero value would be lost.
        if (!transmissionIsRough(params, alpha)) {
            return {};
        }
        const LobeEval transmission = evaluateTransmissionLobe(params, wo, wi, alpha, lobes);
        const float msPdf = msTransmitPdf(-wi.z, lobes.transmitShape);
        return {glm::vec3(0.0F), glm::vec3(0.0F), transmission.f + transmitMultiScatter(params, -wi.z, msPdf, lobes),
                (lobes.specular * transmission.pdf) + (lobes.msTransmit * msPdf)};
    }
    // One wi-side table lookup for both lobes: the specular energy compensation and the diffuse coat coupling read the same split.
    const AlbedoSplit splitWi = directionalAlbedo(wi.z, params.roughness);
    const LobeEval specular = evaluateSpecularLobe(params, wo, wi, alpha, lobes, splitWi);
    const LobeEval diffuse = evaluateDiffuseLobe(params, wo, wi, lobes, splitWi);
    float pdf = (lobes.specular * specular.pdf) + (lobes.diffuse * diffuse.pdf) +
                (lobes.msReflect * msReflectPdf(wi.z, params.roughness));
    // Gated so an opaque material skips the row lookup; the term is 0 there either way.
    if (lobes.msReflectTransmissive > 0.0F) {
        pdf += lobes.msReflectTransmissive * msTransmitPdf(wi.z, lobes.reflectShape);
    }
    return {diffuse.f, specular.f, glm::vec3(0.0F), pdf};
}

// One-sample MIS: whichever strategy drew wi, the throughput divides by the whole mixture's density at it.
std::optional<BsdfSample> weighSample(const BsdfClosure& closure, const glm::vec3& wi, LobeType type) {
    const BsdfEval eval = evaluateContinuousLobes(closure.params, closure.wo, wi, closure.alpha, closure.lobes);
    if (eval.pdf <= 1e-8F) {
        return std::nullopt;
    }
    return BsdfSample{glm::vec3(wi.x, wi.y, wi.z * closure.sign), (eval.total() * std::abs(wi.z)) / eval.pdf, type, eval.pdf};
}

// VNDF single scatter, refracting about the sampled facet too (Walter 2007), split by facetReflectProbability without another draw.
std::optional<BsdfSample> sampleVndfStrategy(const BsdfClosure& closure, float lobeU, Sampler& sampler) {
    const glm::vec3& wo = closure.wo;
    const LobeProbabilities& lobes = closure.lobes;
    const glm::vec3 nh = sampleGGXVNDF(wo, closure.alpha, sampler.next2D());
    if (lobes.facetTransmit > 0.0F) {
        const float woDotNh = glm::dot(wo, nh);
        const float fDielectric = fresnelDielectric(woDotNh, lobes.etaI, lobes.etaT);
        if (lobeU >= lobes.specular * facetReflectProbability(closure.params, woDotNh, fDielectric, lobes)) {
            glm::vec3 wi;
            if (!refractAbout(wo, nh, lobes.etaI / lobes.etaT, wi) || wi.z >= 0.0F) {
                return std::nullopt;
            }
            return weighSample(closure, wi, LobeType::Transmission);
        }
    }
    const glm::vec3 wi = glm::reflect(-wo, nh);
    if (wi.z <= 0.0F) {
        return std::nullopt;
    }
    return weighSample(closure, wi, LobeType::SpecularReflection);
}

// Diffuse (EON), opaque multiple-scattering and a transmissive interface's reflected multiple scattering, by lobeU's sub-range.
std::optional<BsdfSample> sampleReflectionStrategies(const BsdfClosure& closure, float lobeU, Sampler& sampler) {
    const LobeProbabilities& lobes = closure.lobes;
    const bool sampledDiffuse = lobeU < lobes.specular + lobes.diffuse;
    glm::vec3 wi;
    if (sampledDiffuse) {
        wi = sampleEon(lobes, sampler.next2D());
    } else if (lobeU < lobes.specular + lobes.diffuse + lobes.msReflect) {
        wi = sampleMsReflect(closure.params.roughness, sampler.next2D());
    } else {
        wi = sampleMsTransmit(lobes.reflectShape, sampler.next2D());
    }
    if (wi.z <= 0.0F) {
        return std::nullopt;
    }
    // The multiple-scattering branches report SpecularReflection: repeated GGX bounces are specular however broad their exitant lobe.
    return weighSample(closure, wi, sampledDiffuse ? LobeType::Diffuse : LobeType::SpecularReflection);
}

// Smooth specular transmission: Snell, TIR already folded into lobes.transmit, whose (1-F) energy is exactly 0 past the critical angle.
std::optional<BsdfSample> sampleDeltaTransmission(const BsdfClosure& closure) {
    const glm::vec3& wo = closure.wo;
    const LobeProbabilities& lobes = closure.lobes;
    const float eta = lobes.etaI / lobes.etaT;
    const float cos2ThetaT = cos2Transmitted(wo.z, eta);
    if (cos2ThetaT < 0.0F) {
        return std::nullopt;
    }
    const float cosThetaT = std::sqrt(cos2ThetaT);
    const glm::vec3 wt(-eta * wo.x, -eta * wo.y, -cosThetaT);
    // Non-symmetric radiance compression for camera-originated transport (Veach 1997 sec. 5.2): eta^2 = (etaI/etaT)^2.
    const glm::vec3 throughput =
        closure.params.transmissionTint * (lobes.transmitPhysicalValue / lobes.transmit) * (eta * eta);
    // pdf 0: a delta lobe has no density for NEE to double-count against, which is exactly the test path_tracer.cpp's MIS weighting makes.
    return BsdfSample{glm::vec3(wt.x, wt.y, wt.z * closure.sign), throughput, LobeType::Transmission, 0.0F};
}

}  // namespace

glm::vec3 fresnelAtViewAngle(const BsdfParams& params, float cosTheta) {
    // Entering orientation (etaI=1): a primary-hit view-angle value is always outside the surface, so there is no exiting side to swap.
    const float fDielectric = fresnelDielectric(cosTheta, 1.0F, params.ior);
    if (params.metallic <= 0.0F) {
        return glm::vec3(fDielectric);
    }
    const ConductorIor conductor = conductorIorFromReflectivity(params.f0, params.edgeTint);
    return glm::mix(glm::vec3(fDielectric), fresnelConductor(cosTheta, conductor.n, conductor.k),
                     params.metallic);
}

glm::vec3 fresnelAtMicrofacet(const BsdfParams& params, const glm::vec3& woLocal, glm::vec2 u) {
    // sampleGGXVNDF's +z-hemisphere precondition. Only dot(wo, wh) is read and reflecting leaves it unchanged, so the flip needs no undo.
    const glm::vec3 wo(woLocal.x, woLocal.y, std::abs(woLocal.z));
    const glm::vec3 wh = sampleGGXVNDF(wo, alphaForRoughness(params.roughness), u);
    // Clamped, not raw: fresnelDielectric swaps etaI/etaT below zero, so a negative dot would report the exiting-side term.
    return fresnelAtViewAngle(params, std::max(glm::dot(wo, wh), 0.0F));
}

BsdfClosure makeBsdfClosure(const BsdfParams& params, const glm::vec3& woLocal) {
    const float sign = woLocal.z >= 0.0F ? 1.0F : -1.0F;
    const glm::vec3 wo(woLocal.x, woLocal.y, woLocal.z * sign);
    const float alpha = alphaForRoughness(params.roughness);
    return {params, wo, sign, alpha, computeLobeProbabilities(params, wo, sign, alpha)};
}

BsdfEval evaluateBsdfSplit(const BsdfClosure& closure, const glm::vec3& wiLocal) {
    const glm::vec3 wi(wiLocal.x, wiLocal.y, wiLocal.z * closure.sign);
    return evaluateContinuousLobes(closure.params, closure.wo, wi, closure.alpha, closure.lobes);
}

// The params/wo form, for the validators and any caller with a single direction to answer for: one closure, used once.
BsdfEval evaluateBsdfSplit(const BsdfParams& params, const glm::vec3& woLocal,
                            const glm::vec3& wiLocal) {
    return evaluateBsdfSplit(makeBsdfClosure(params, woLocal), wiLocal);
}

float pdfBsdf(const BsdfParams& params, const glm::vec3& woLocal, const glm::vec3& wiLocal) {
    return evaluateBsdfSplit(params, woLocal, wiLocal).pdf;
}

glm::vec3 evaluateBsdf(const BsdfParams& params, const glm::vec3& woLocal, const glm::vec3& wiLocal) {
    return evaluateBsdfSplit(params, woLocal, wiLocal).total();
}

std::optional<BsdfSample> sampleBsdf(const BsdfClosure& closure, Sampler& sampler) {
    const LobeProbabilities& lobes = closure.lobes;
    const float lobeU = sampler.next1D();
    if (lobeU < lobes.specular) {
        return sampleVndfStrategy(closure, lobeU, sampler);
    }
    // The reflection sub-ranges are prefix sums of one expression, so they partition it exactly: a massless strategy is never reached.
    if (lobeU < lobes.specular + lobes.diffuse + lobes.msReflect + lobes.msReflectTransmissive) {
        return sampleReflectionStrategies(closure, lobeU, sampler);
    }
    // Tested first: the probabilities below sum to 1.0 only in float, and this lobe reaches directions no microfacet could refract into.
    if (lobes.msTransmit > 0.0F &&
        lobeU >= lobes.specular + lobes.diffuse + lobes.msReflect + lobes.msReflectTransmissive + lobes.transmit) {
        glm::vec3 wi = sampleMsTransmit(lobes.transmitShape, sampler.next2D());
        wi.z = -wi.z;
        return weighSample(closure, wi, LobeType::Transmission);
    }
    if (lobes.transmit <= 0.0F) {
        return std::nullopt;
    }
    return sampleDeltaTransmission(closure);
}

std::optional<BsdfSample> sampleBsdf(const BsdfParams& params, const glm::vec3& woLocal,
                                      Sampler& sampler) {
    return sampleBsdf(makeBsdfClosure(params, woLocal), sampler);
}

}  // namespace pathtracer::scene
