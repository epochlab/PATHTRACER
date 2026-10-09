#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/fresnel_dielectric.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace pathtracer::scene {

namespace {

constexpr float kPi = 3.14159265F;

// Perceptual roughness to GGX alpha, in one place: the three consumers must agree, or the Fresnel AOV reports an unevaluated term.
float alphaForRoughness(float roughness) { return roughness * roughness; }

// GGX's denominator pi*d^2 falls to pi*alpha^4 at its peak: below FLT_MIN it underflows and D overflows, so the lobe is a delta.
bool isSmooth(float alpha) { return alpha * alpha * alpha * alpha < std::numeric_limits<float>::min(); }

// GGX D, cancellation-free (Filament 4.4.2); no denominator floor, isSmooth keeping pi*d^2 >= pi*alpha^4 a normal float.
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


// --- Metal Fresnel: OpenPBR's F82-tint (Hoffman 2023; Kutz et al. 2021), Schlick less k*mu*(1-mu)^6, fit at mu-bar = 1/7 (~82 deg).
constexpr float kOneMinusMuBar = 6.0F / 7.0F;
constexpr float kSchlickAtMuBar = kOneMinusMuBar * kOneMinusMuBar * kOneMinusMuBar * kOneMinusMuBar * kOneMinusMuBar;
constexpr float kCorrectionAtMuBar = (1.0F / 7.0F) * kSchlickAtMuBar * kOneMinusMuBar;

// F82's correction weight k fit so F(1/7) = tint * Schlick(1/7) (OpenPBR's metal; Hoffman 2023); linear in F0.
glm::vec3 f82Weight(const glm::vec3& f0, const glm::vec3& tint) {
    return (f0 + ((1.0F - f0) * kSchlickAtMuBar)) * (1.0F - tint) / kCorrectionAtMuBar;
}

// The metal's F82-tint Fresnel (OpenPBR; Hoffman 2023) at cosTheta in [0,1], before specular_weight scales it.
glm::vec3 fresnelF82(float cosTheta, const glm::vec3& f0, const glm::vec3& tint) {
    const float m = 1.0F - cosTheta;
    const float m5 = (m * m) * (m * m) * m;
    return f0 + ((1.0F - f0) * m5) - (f82Weight(f0, tint) * (cosTheta * m5 * m));
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

// F82-split albedo: E(F0, k) = F0*a + b - k*c for the metal's F82-tint Fresnel, a+b = E, and the deficit 1 - E as baked in double.
struct AlbedoSplit {
    float a;
    float b;
    float c;
    float deficit;
    [[nodiscard]] glm::vec3 at(const glm::vec3& f0, const glm::vec3& k) const { return (f0 * a) + b - (k * c); }
};

// The bilinear cell of the reflect tables, indexed by sqrt(mu): E reaches its plateau over mu ~ alpha, under a cell if uniform.
struct AlbedoCell {
    int i0;
    int i1;
    float mt;
    float rt;
    template <typename Table>
    [[nodiscard]] float read(const Table& table) const {
        return lerp1(lerp1(table[i0], table[i0 + 1], mt), lerp1(table[i1], table[i1 + 1], mt), rt);
    }
};

AlbedoCell albedoCell(float mu, float roughness) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kAlbedoRoughnessRes - 1);
    const float mf = std::sqrt(std::clamp(mu, 0.0F, 1.0F)) * (kAlbedoMuRes - 1);
    const int r0 = std::min(static_cast<int>(rf), kAlbedoRoughnessRes - 2);
    const int m0 = std::min(static_cast<int>(mf), kAlbedoMuRes - 2);
    return {(r0 * kAlbedoMuRes) + m0, ((r0 + 1) * kAlbedoMuRes) + m0, mf - static_cast<float>(m0), rf - static_cast<float>(r0)};
}

AlbedoSplit directionalAlbedo(float mu, float roughness) {
    const AlbedoCell cell = albedoCell(mu, roughness);
    return {cell.read(kAlbedoA), cell.read(kAlbedoB), cell.read(kAlbedoC), cell.read(kAlbedoDeficit)};
}

// The wi side needs the deficit alone: one table read, not four.
float directionalDeficit(float mu, float roughness) { return albedoCell(mu, roughness).read(kAlbedoDeficit); }

AlbedoSplit averageAlbedo(float roughness) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kAlbedoRoughnessRes - 1);
    const int r0 = std::min(static_cast<int>(rf), kAlbedoRoughnessRes - 2);
    const float rt = rf - static_cast<float>(r0);
    const auto lerpRow = [&](const auto& table) { return lerp1(table[r0], table[r0 + 1], rt); };
    return {lerpRow(kAlbedoAvgA), lerpRow(kAlbedoAvgB), lerpRow(kAlbedoAvgC), lerpRow(kAlbedoAvgDeficit)};
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

// Reflected and transmitted shares of a dielectric interface, exact Fresnel already applied.
struct EscapeSplit {
    float reflect;
    float transmit;
    [[nodiscard]] float total() const { return reflect + transmit; }
};

// The (roughness, eta) bilinear weights of the escape tables, so each directional read is one mu interpolation over four rows.
EscapeRow escapeRow(float roughness, float eta) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kTransmitRoughnessRes - 1);
    const float ef = etaAxisCoord(eta);
    const int r0 = std::min(static_cast<int>(rf), kTransmitRoughnessRes - 2);
    const int e0 = std::min(static_cast<int>(ef), kEtaRes - 2);
    const float rt = rf - static_cast<float>(r0);
    const float et = ef - static_cast<float>(e0);
    const int base0 = (r0 * kTransmitMuRes * kEtaRes) + e0;
    const int base1 = base0 + (kTransmitMuRes * kEtaRes);
    return {{base0, base0 + 1, base1, base1 + 1}, {(1.0F - rt) * (1.0F - et), (1.0F - rt) * et, rt * (1.0F - et), rt * et}};
}

// A row's blend of one table at mu node `index`; the mu stride is kEtaRes.
template <typename Table>
float escapeBlend(const Table& table, const EscapeRow& row, int index) {
    const int offset = index * kEtaRes;
    return (row.weight[0] * table[row.base[0] + offset]) + (row.weight[1] * table[row.base[1] + offset]) +
           (row.weight[2] * table[row.base[2] + offset]) + (row.weight[3] * table[row.base[3] + offset]);
}

// The generator's escapeMu axis: nodes uniform in sqrt(mu), dense where E climbs from its grazing limit.
template <typename Table>
float escapeAt(const Table& table, const EscapeRow& row, float mu) {
    const float mf = std::sqrt(std::clamp(mu, 0.0F, 1.0F)) * (kTransmitMuRes - 1);
    const int m0 = std::min(static_cast<int>(mf), kTransmitMuRes - 2);
    return lerp1(escapeBlend(table, row, m0), escapeBlend(table, row, m0 + 1), mf - static_cast<float>(m0));
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

// Transmitted MS lobe's shape, stored unnormalised and divided by its blended total: blended prefix integrals are the blend's integral.
MsTransmitRow msTransmitRow(float roughness, float eta) {
    const EscapeRow row = escapeRow(roughness, eta);
    // Last prefix integral is the row's energy deficit; zero means no energy to carry, so it reports zero density rather than dividing.
    const float total = escapeBlend(kMsTransmitCdf, row, kTransmitMuRes - 1);
    return {row, total > 0.0F ? 1.0F / total : 0.0F};
}

float msTransmitDensity(const MsTransmitRow& row, int index) {
    return escapeBlend(kMsTransmitDensity, row.row, index) * row.scale;
}

float msTransmitCdf(const MsTransmitRow& row, int index) {
    return escapeBlend(kMsTransmitCdf, row.row, index) * row.scale;
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

// Malley's method, a uniform disk point lifted to the hemisphere (PBR 4th ed. 13.6.3). External for the AO lane, where the pdf cancels.
glm::vec3 sampleCosineHemisphere(glm::vec2 u) {
    const float r = std::sqrt(u.x);
    const float phi = 2.0F * kPi * u.y;
    return {r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0F, 1.0F - u.x))};
}

// Cosine-weighted average Fresnel, 2*int_0^1 F(mu)*mu dmu, a 3-node Gauss rule: worst 5.5e-5 over ior [1.05, 3.0], +0 at ior 1.
constexpr float kFresnelAvgNodes[3] = {0.105319802F, 0.382154433F, 0.796427281F};
constexpr float kFresnelAvgWeights[3] = {0.038972482F, 0.280518736F, 0.680508783F};

float dielectricFresnelAvg(float ior) {
    float sum = 0.0F;
    for (int i = 0; i < 3; ++i) {
        sum += kFresnelAvgWeights[i] * fresnelDielectric(kFresnelAvgNodes[i], 1.0F, ior);
    }
    return sum;
}

// 2*int mu*F82 dmu in closed form: Schlick's mean F0 + (1-F0)/21 less k * 2*B(3, 7) = k/126 for the correction.
glm::vec3 metalFresnelAvg(const glm::vec3& f0, const glm::vec3& tint) {
    return f0 + ((1.0F - f0) / 21.0F) - (f82Weight(f0, tint) / 126.0F);
}

// External linkage: checkAlbedoTableInterpolation is the only instrument that sees the .inc's interpolation error.
glm::vec4 directionalAlbedoSplit(float mu, float roughness) {
    const AlbedoSplit split = directionalAlbedo(mu, roughness);
    return {split.a, split.b, split.c, split.deficit};
}

glm::vec4 averageAlbedoSplit(float roughness) {
    const AlbedoSplit split = averageAlbedo(roughness);
    return {split.a, split.b, split.c, split.deficit};
}

// The grid the two lookups index, described rather than transcribed. Both axes edge-aligned, so 0 and res-1 are exact endpoints.
glm::ivec2 albedoGridRes() { return {kAlbedoRoughnessRes, kAlbedoMuRes}; }

float albedoGridRoughness(float index) { return index / static_cast<float>(kAlbedoRoughnessRes - 1); }

// Inverts directionalAlbedo's sqrt(mu) index, so an instrument's fractional index lands where that lookup interpolates.
float albedoGridMu(float index) {
    const float t = index / static_cast<float>(kAlbedoMuRes - 1);
    return t * t;
}

// Fraunhofer d, F and C lines, where V_d = (n_d-1)/(n_F-n_C) is defined: physical constants of the definition, not tuning.
constexpr float kLambdaDNm = 587.56F;
constexpr float kLambdaFNm = 486.13F;
constexpr float kLambdaCNm = 656.27F;

// Cauchy n(lambda) = A + B/lambda^2, (A,B) from (n_d, V_d); V_d infinite, dispersion scale 0, gives B = 0 and the index itself.
float cauchyIor(float iorD, float abbe, float lambdaNm) {
    const float b = (iorD - 1.0F) /
                    (abbe * ((1.0F / (kLambdaFNm * kLambdaFNm)) - (1.0F / (kLambdaCNm * kLambdaCNm))));
    const float a = iorD - (b / (kLambdaDNm * kLambdaDNm));
    return a + (b / (lambdaNm * lambdaNm));
}

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

// Kulla-Conty tint from the mean deficit 1 - Eavg: the share of (1-E) energy surviving repeated bounces. Exactly 1 at Favg=1.
float multiScatterTint(float fresnelAvg, float deficitAvg) {
    return (fresnelAvg * fresnelAvg * (1.0F - deficitAvg)) / (1.0F - (fresnelAvg * deficitAvg));
}

glm::vec3 multiScatterTint(const glm::vec3& fresnelAvg, float deficitAvg) {
    return {multiScatterTint(fresnelAvg.x, deficitAvg), multiScatterTint(fresnelAvg.y, deficitAvg),
            multiScatterTint(fresnelAvg.z, deficitAvg)};
}

float channelMean(const glm::vec3& v) { return (v.x + v.y + v.z) / 3.0F; }

// ior == 1 is a delta at every roughness: the half-vector normalizes zero, NaN on 7783 of 7783 transmission draws at roughness 0.1.
bool transmissionIsRough(const BsdfParams& params, float alpha) {
    return params.transmissionWeight > 0.0F && !isSmooth(alpha) && params.ior != 1.0F;
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

// The glossy-diffuse layer's reflection albedo at mu: its tinted single scattering plus Kulla-Conty, the energy the diffuse loses.
glm::vec3 glossyAlbedo(const LobeProbabilities& lobes, float mu, float deficit) {
    return lobes.dielectricTint * (escapeAt(kEscapeReflect, lobes.dielectricRow, mu) + (lobes.glossyFms * deficit));
}

LobeEval evaluateDiffuseLobe(const BsdfParams& params, const glm::vec3& wo, const glm::vec3& wi, const LobeProbabilities& lobes,
                             float deficitWi) {
    if (wi.z <= 0.0F || wo.z <= 0.0F) {
        return {glm::vec3(0.0F), 0.0F};
    }
    const glm::vec3 f = evaluateEon(params.diffuseRho, params.diffuseRoughness, wi, wo);
    return {f * lobes.diffuseCouplingWo * (1.0F - glossyAlbedo(lobes, wi.z, deficitWi)), pdfEon(wi, lobes)};
}

// Every reflection Fresnel at a facet: the metal's F82 and the tinted dielectric, mixed by metalness (OpenPBR's base substrate).
glm::vec3 reflectedFresnel(const LobeProbabilities& lobes, float cosTheta, float fDielectric) {
    const float m = 1.0F - cosTheta;
    const float m5 = (m * m) * (m * m) * m;
    const glm::vec3 metal = lobes.metalF0 + ((1.0F - lobes.metalF0) * m5) - (lobes.metalK * (cosTheta * m5 * m));
    return (lobes.metalWeight * metal) + (lobes.dielectricWeight * fDielectric * lobes.dielectricTint);
}

// Probability the VNDF strategy reflects about a facet (Walter 2007 5.3); callers gate facetTransmit > 0, so the denominator is positive.
float facetReflectProbability(float cosTheta, float fDielectric, const LobeProbabilities& lobes) {
    const float reflect = channelMean(reflectedFresnel(lobes, cosTheta, fDielectric));
    return reflect / (reflect + ((1.0F - fDielectric) * lobes.facetTransmit));
}

// Single scatter D*G2*F/(4*ndotV*ndotL) plus multiple scattering, and the VNDF pdf (Heitz 2018 eq.3) times its Jacobian.
LobeEval evaluateSpecularLobe(const glm::vec3& wo, const glm::vec3& wi, float alpha, const LobeProbabilities& lobes, float deficitWi) {
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
    const glm::vec3 singleScatter = d * smithVisibility(wo.z, wi.z, alpha) * reflectedFresnel(lobes, woDotNh, fDielectric);
    // Kulla & Conty 2017: (1-E(mu_o))(1-E(mu_i))/(pi(1-Eavg)), symmetric in wo and wi, so the lobe stays reciprocal.
    glm::vec3 multiScatter = lobes.msReflectTint * (lobes.msReflectScaleWo * deficitWi);
    if (lobes.reflectShape.scale > 0.0F) {
        multiScatter += lobes.dielectricTint * (lobes.transmitWeight * (1.0F - lobes.transmitShare) *
                                                std::max(1.0F - lobes.escapeWo, 0.0F) * msTransmitPdf(wi.z, lobes.reflectShape) / wi.z);
    }
    const float vndfPdf = 0.25F * d * smithG1OverCos(wo.z, alpha);
    return {singleScatter + multiScatter,
            lobes.facetTransmit > 0.0F ? vndfPdf * facetReflectProbability(woDotNh, fDielectric, lobes) : vndfPdf};
}

// Each strategy's energy at wo (channel mean), as flux without the eta^2 compression (PBRT-v4): the selection masses before normalising.
struct StrategyEnergies {
    float specular = 0.0F;
    float diffuse = 0.0F;
    float msReflect = 0.0F;
    float msReflectTransmissive = 0.0F;
    float transmit = 0.0F;
    float msTransmit = 0.0F;
};

// The wo-side frame of the base substrate: orientation, mix weights, the metal's F82 state and the Kulla-Conty wo half.
LobeProbabilities baseState(const BsdfParams& params, float sign, const AlbedoSplit& splitWo, float deficitAvg) {
    const bool exiting = sign < 0.0F && params.transmissionWeight > 0.0F;
    LobeProbabilities lobes{};
    lobes.etaI = exiting ? params.ior : 1.0F;
    lobes.etaT = exiting ? 1.0F : params.ior;
    lobes.etaSq = (lobes.etaI / lobes.etaT) * (lobes.etaI / lobes.etaT);
    lobes.dielectricWeight = 1.0F - params.metalness;
    // Only the interface faces a ray from inside: the translucent share is whole there, with no glossy-diffuse and no tint.
    lobes.transmitWeight = lobes.dielectricWeight * (exiting ? 1.0F : params.transmissionWeight);
    lobes.dielectricTint = exiting ? glm::vec3(1.0F) : params.specularColor;
    lobes.metalWeight = params.metalness * params.specularWeight;
    lobes.metalF0 = params.metalF0;
    lobes.metalK = f82Weight(params.metalF0, params.specularColor);
    // A zero mean deficit, the smooth row, has every deficit zero: the lobe vanishes, its exact limit, with no 0/0 to form.
    lobes.msReflectScaleWo = deficitAvg > 0.0F ? splitWo.deficit / (kPi * deficitAvg) : 0.0F;
    return lobes;
}

// The reflecting strategies: metal and dielectric single scattering, their Kulla-Conty tints, and the glossy layer's diffuse coupling.
void addReflection(LobeProbabilities& lobes, StrategyEnergies& energies, const BsdfParams& params, const glm::vec3& wo,
                   const AlbedoSplit& splitWo, float deficitAvg) {
    const float eta = lobes.etaI / lobes.etaT;
    if (lobes.metalWeight > 0.0F) {
        energies.specular += channelMean(lobes.metalWeight * splitWo.at(lobes.metalF0, lobes.metalK));
        lobes.msReflectTint +=
            params.metalness * multiScatterTint(params.specularWeight * metalFresnelAvg(params.metalF0, params.specularColor), deficitAvg);
    }
    // Index-matched, the interface reflects exactly nothing: an all-zero row, so no table's interpolation residual reads as reflection.
    if (lobes.dielectricWeight > 0.0F && params.ior != 1.0F) {
        lobes.dielectricRow = escapeRow(params.roughness, eta);
        energies.specular += lobes.dielectricWeight * channelMean(lobes.dielectricTint) * escapeAt(kEscapeReflect, lobes.dielectricRow, wo.z);
    }
    const float glossyWeight = lobes.dielectricWeight - lobes.transmitWeight;
    if (glossyWeight > 0.0F) {
        lobes.glossyFms = multiScatterTint(dielectricFresnelAvg(params.ior), deficitAvg);
        lobes.msReflectTint += glossyWeight * lobes.dielectricTint * lobes.glossyFms;
        // Kelemen and Szirmay-Kalos 2001: the reciprocal coupling, whose mean is 1 - Eavg; a layer reflecting all of it leaves none.
        const float meanReflect = params.ior != 1.0F ? averageEscapeAlbedo(params.roughness, eta).reflect : 0.0F;
        const glm::vec3 meanAlbedo = lobes.dielectricTint * (meanReflect + (lobes.glossyFms * deficitAvg));
        const glm::vec3 transmitted = 1.0F - glossyAlbedo(lobes, wo.z, splitWo.deficit);
        for (int c = 0; c < 3; ++c) {
            lobes.diffuseCouplingWo[c] = meanAlbedo[c] < 1.0F ? glossyWeight * transmitted[c] / (1.0F - meanAlbedo[c]) : 0.0F;
        }
    }
    energies.msReflect = channelMean(lobes.msReflectTint) * splitWo.deficit;
    energies.diffuse = channelMean(params.diffuseRho * lobes.diffuseCouplingWo);
}

// The translucent base's strategies: delta or rough refraction, and the interface's multiple scattering on either side.
void addTransmission(LobeProbabilities& lobes, StrategyEnergies& energies, const BsdfParams& params, const glm::vec3& wo, bool rough) {
    const float eta = lobes.etaI / lobes.etaT;
    lobes.transmitPhysicalValue = lobes.transmitWeight * (1.0F - fresnelDielectric(wo.z, lobes.etaI, lobes.etaT));
    float transmitSs = 1.0F;
    if (params.ior == 1.0F) {
        lobes.escapeWo = 1.0F;
        lobes.transmitShare = 1.0F;
    } else {
        // The interface's own single-scattering escape R + T at wo; multiple scattering returns the deficit 1 - (R + T).
        const EscapeSplit escapeMean = averageEscapeAlbedo(params.roughness, eta);
        transmitSs = escapeAt(kEscapeTransmit, lobes.dielectricRow, wo.z);
        lobes.escapeWo = escapeAt(kEscapeReflect, lobes.dielectricRow, wo.z) + transmitSs;
        lobes.transmitShape = msTransmitRow(params.roughness, 1.0F / eta);
        lobes.reflectShape = msTransmitRow(params.roughness, eta);
        // A property of the interface alone: OpenPBR mixes whole BSDFs, so every lobe stays linear in its weight.
        lobes.transmitShare = escapeMean.transmit / escapeMean.total();
    }
    const float deficit = std::max(1.0F - lobes.escapeWo, 0.0F);
    // A zero-scale row has no density to draw from, and its value reads the same zero.
    if (lobes.reflectShape.scale > 0.0F) {
        energies.msReflectTransmissive = lobes.transmitWeight * (1.0F - lobes.transmitShare) * deficit * channelMean(lobes.dielectricTint);
    }
    // A rough interface's refraction is the VNDF strategy's other branch, so its energy joins the specular mass; a delta keeps its own.
    if (rough) {
        energies.specular += lobes.transmitWeight * transmitSs;
        lobes.facetTransmit = lobes.transmitWeight;
        // evaluateContinuousLobes drops the far-side multiple scattering of a delta interface, so it has no energy to select for.
        energies.msTransmit = lobes.transmitShape.scale > 0.0F ? lobes.transmitWeight * lobes.transmitShare * deficit : 0.0F;
    } else {
        energies.transmit = lobes.transmitPhysicalValue;
    }
}

// Each strategy's mass is its energy at wo over the total; a massless strategy is never drawn, and a black vertex draws nothing.
LobeProbabilities computeLobeProbabilities(const BsdfParams& params, const glm::vec3& wo, float sign, float alpha) {
    const AlbedoSplit splitWo = directionalAlbedo(wo.z, params.roughness);
    const float deficitAvg = averageAlbedo(params.roughness).deficit;
    LobeProbabilities lobes = baseState(params, sign, splitWo, deficitAvg);
    StrategyEnergies energies;
    addReflection(lobes, energies, params, wo, splitWo, deficitAvg);
    if (lobes.transmitWeight > 0.0F) {
        addTransmission(lobes, energies, params, wo, transmissionIsRough(params, alpha));
    }
    const float total = energies.specular + energies.diffuse + energies.msReflect + energies.msReflectTransmissive + energies.transmit +
                        energies.msTransmit;
    const float inverseTotal = total > 0.0F ? 1.0F / total : 0.0F;
    lobes.specular = energies.specular * inverseTotal;
    lobes.diffuse = energies.diffuse * inverseTotal;
    lobes.msReflect = energies.msReflect * inverseTotal;
    lobes.msReflectTransmissive = energies.msReflectTransmissive * inverseTotal;
    lobes.transmit = energies.transmit * inverseTotal;
    lobes.msTransmit = energies.msTransmit * inverseTotal;
    // EON's wo-side sampling state, only where the diffuse lobe can be drawn: its pow() and LTC fit are not free.
    if (lobes.diffuse > 0.0F) {
        const EonLtcCoeffs eonLtc = eonLtcCoeffs(wo.z, params.diffuseRoughness);
        lobes.eonUniformMix = eonUniformMixWeight(wo.z, params.diffuseRoughness);
        lobes.eonLtcM = glm::vec4(eonLtc.a, eonLtc.b, eonLtc.c, eonLtc.d);
        lobes.eonLtcBasisT = glm::transpose(orthonormalBasisLtc(wo));
        lobes.eonLtcS = 0.5F * (1.0F + (1.0F / std::sqrt((eonLtc.d * eonLtc.d) + 1.0F)));
    }
    return lobes;
}

// Walter 2007 (eq. 21, 16, 17), PBRT-v3 radiance form: eta^2 (Veach 1997 5.2) is already folded in, hence etaR^2 in the pdf only.
LobeEval evaluateTransmissionLobe(const BsdfParams& params, const glm::vec3& wo, const glm::vec3& wi, float alpha,
                                  const LobeProbabilities& lobes) {
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
    // |wo + etaR*wi| >= |1 - etaR| > 0: etaR is never 1 here, index-matched interfaces being deltas (transmissionIsRough).
    const float denom = woDotH + (etaR * wiDotH);
    const float denom2 = denom * denom;
    const float d = distributionGGX(ht, alpha);
    const float fresnel = fresnelDielectric(woDotH, lobes.etaI, lobes.etaT);
    // D*G2*|wi.h|*(wo.h)/(wo.z*|wi.z|*denom^2), with G2/(wo.z*|wi.z|) taken as 4*smithVisibility.
    const float common = (4.0F * d * smithVisibility(wo.z, -wi.z, alpha) * std::abs(wiDotH) * woDotH) / denom2;
    const float vndfPdf = d * woDotH * smithG1OverCos(wo.z, alpha);
    const float refractProbability = lobes.facetTransmit > 0.0F ? 1.0F - facetReflectProbability(woDotH, fresnel, lobes) : 0.0F;
    return {params.transmissionTint * (1.0F - fresnel) * lobes.transmitWeight * common,
            refractProbability * vndfPdf * etaR * etaR * std::abs(wiDotH) / denom2};
}

BsdfEval evaluateContinuousLobes(const BsdfParams& params, const glm::vec3& wo, const glm::vec3& wi, float alpha,
                                 const LobeProbabilities& lobes) {
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
    // One wi-side table read serves the Kulla-Conty lobe and the diffuse coupling; with neither drawable it is never needed.
    const float deficitWi = lobes.msReflect > 0.0F || lobes.diffuse > 0.0F ? directionalDeficit(wi.z, params.roughness) : 0.0F;
    // A smooth surface's reflection is a delta: no continuous value for NEE or a continuous strategy to meet.
    const LobeEval specular = isSmooth(alpha) ? LobeEval{glm::vec3(0.0F), 0.0F} : evaluateSpecularLobe(wo, wi, alpha, lobes, deficitWi);
    const LobeEval diffuse = lobes.diffuse > 0.0F ? evaluateDiffuseLobe(params, wo, wi, lobes, deficitWi) : LobeEval{glm::vec3(0.0F), 0.0F};
    float pdf = (lobes.specular * specular.pdf) + (lobes.diffuse * diffuse.pdf);
    if (lobes.msReflect > 0.0F) {
        pdf += lobes.msReflect * msReflectPdf(wi.z, params.roughness);
    }
    // Gated so an opaque material skips the row lookup; the term is 0 there either way.
    if (lobes.msReflectTransmissive > 0.0F) {
        pdf += lobes.msReflectTransmissive * msTransmitPdf(wi.z, lobes.reflectShape);
    }
    return {diffuse.f, specular.f, glm::vec3(0.0F), pdf};
}

// One-sample MIS: whichever strategy drew wi, the throughput divides by the whole mixture's density at it.
std::optional<BsdfSample> weighSample(const BsdfClosure& closure, const glm::vec3& wi, LobeType type) {
    const BsdfEval eval = evaluateContinuousLobes(closure.params, closure.wo, wi, closure.alpha, closure.lobes);
    if (!(eval.pdf > 0.0F)) {
        return std::nullopt;
    }
    return BsdfSample{glm::vec3(wi.x, wi.y, wi.z * closure.sign), (eval.total() * std::abs(wi.z)) / eval.pdf, type, eval.pdf, false};
}

// VNDF single scatter, refracting about the sampled facet too (Walter 2007), split by facetReflectProbability without another draw.
std::optional<BsdfSample> sampleVndfStrategy(const BsdfClosure& closure, float lobeU, Sampler& sampler) {
    const glm::vec3& wo = closure.wo;
    const LobeProbabilities& lobes = closure.lobes;
    const glm::vec3 nh = sampleGGXVNDF(wo, closure.alpha, sampler.next2D());
    if (lobes.facetTransmit > 0.0F) {
        const float woDotNh = glm::dot(wo, nh);
        const float fDielectric = fresnelDielectric(woDotNh, lobes.etaI, lobes.etaT);
        if (lobeU >= lobes.specular * facetReflectProbability(woDotNh, fDielectric, lobes)) {
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
    const glm::vec3 throughput = closure.params.transmissionTint * (lobes.transmitPhysicalValue / lobes.transmit) * (eta * eta);
    // pdf 0: a delta lobe has no density for NEE to double-count against, which is exactly the test path_tracer.cpp's MIS weighting makes.
    return BsdfSample{glm::vec3(wt.x, wt.y, wt.z * closure.sign), throughput, LobeType::Transmission, 0.0F, true};
}

// Smooth reflection: the mirror carries every reflection Fresnel at mu_o, metal and tinted dielectric, divided by its selection mass.
std::optional<BsdfSample> sampleDeltaReflection(const BsdfClosure& closure) {
    const glm::vec3& wo = closure.wo;
    const LobeProbabilities& lobes = closure.lobes;
    const glm::vec3 fresnel = reflectedFresnel(lobes, wo.z, fresnelDielectric(wo.z, lobes.etaI, lobes.etaT));
    return BsdfSample{glm::vec3(-wo.x, -wo.y, wo.z * closure.sign), fresnel / lobes.specular, LobeType::SpecularReflection, 0.0F, true};
}

}  // namespace

glm::vec3 fresnelAtViewAngle(const BsdfParams& params, float cosTheta) {
    // Entering orientation (etaI=1): a primary-hit view-angle value is always outside the surface, so there is no exiting side to swap.
    const float fDielectric = fresnelDielectric(cosTheta, 1.0F, params.ior);
    return (params.metalness * params.specularWeight * fresnelF82(cosTheta, params.metalF0, params.specularColor)) +
           ((1.0F - params.metalness) * fDielectric * params.specularColor);
}

glm::vec3 fresnelAtMicrofacet(const BsdfParams& params, const glm::vec3& woLocal, glm::vec2 u) {
    // sampleGGXVNDF's +z-hemisphere precondition. Only dot(wo, wh) is read and reflecting leaves it unchanged, so the flip needs no undo.
    const glm::vec3 wo(woLocal.x, woLocal.y, std::abs(woLocal.z));
    // A smooth surface's only facet is the macro normal, where the visible-normal distribution collapses to a delta.
    if (isSmooth(alphaForRoughness(params.roughness))) {
        return fresnelAtViewAngle(params, wo.z);
    }
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
BsdfEval evaluateBsdfSplit(const BsdfParams& params, const glm::vec3& woLocal, const glm::vec3& wiLocal) {
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
        return isSmooth(closure.alpha) ? sampleDeltaReflection(closure) : sampleVndfStrategy(closure, lobeU, sampler);
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

std::optional<BsdfSample> sampleBsdf(const BsdfParams& params, const glm::vec3& woLocal, Sampler& sampler) {
    return sampleBsdf(makeBsdfClosure(params, woLocal), sampler);
}

}  // namespace pathtracer::scene
