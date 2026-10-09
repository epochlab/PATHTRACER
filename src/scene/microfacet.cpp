#include "microfacet.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "pathtracer/scene/fresnel_dielectric.h"

namespace pathtracer::scene {

namespace {

constexpr float kPi = 3.14159265F;

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

// --- F82-tint constants: Schlick less k*mu*(1-mu)^6, fit at mu-bar = 1/7 (~82 deg).
constexpr float kOneMinusMuBar = 6.0F / 7.0F;
constexpr float kSchlickAtMuBar = kOneMinusMuBar * kOneMinusMuBar * kOneMinusMuBar * kOneMinusMuBar * kOneMinusMuBar;
constexpr float kCorrectionAtMuBar = (1.0F / 7.0F) * kSchlickAtMuBar * kOneMinusMuBar;

// F82's correction weight k fit so F(1/7) = tint * Schlick(1/7) (OpenPBR's metal; Hoffman 2023); linear in F0.
glm::vec3 f82Weight(const glm::vec3& f0, const glm::vec3& tint) {
    return (f0 + ((1.0F - f0) * kSchlickAtMuBar)) * (1.0F - tint) / kCorrectionAtMuBar;
}

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

float channelMean(const glm::vec3& v) { return (v.x + v.y + v.z) / 3.0F; }

// Kulla-Conty energy tables, baked by tools/albedo_table.cpp (Kulla & Conty 2017).
#include "albedo_table.inc"

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

// --- The conductor's Kulla-Conty lobe, drawn from its own (1-E)cos shape; cosine sampling costs up to +17.3 relative variance.
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
    return lerp1(kMsReflectDensity[row.base + index], kMsReflectDensity[row.base + kMsReflectMuRes + index], row.blend);
}

float msReflectCdf(const MsReflectRow& row, int index) {
    return lerp1(kMsReflectCdf[row.base + index], kMsReflectCdf[row.base + kMsReflectMuRes + index], row.blend);
}

// Solid-angle density: the mu density over 2*pi of azimuth. Uniform in mu, not sqrt(mu), this being the sampling grid.
float msReflectPdf(float mu, float roughness) {
    const MsReflectRow row = msReflectRow(roughness);
    const float mf = std::clamp(mu, 0.0F, 1.0F) * (kMsReflectMuRes - 1);
    const int m0 = std::min(static_cast<int>(mf), kMsReflectMuRes - 2);
    const float mt = mf - static_cast<float>(m0);
    return lerp1(msReflectDensity(row, m0), msReflectDensity(row, m0 + 1), mt) / (2.0F * kPi);
}

// Exact inversion of a tabulated piecewise-linear density over mu; u lands in a cell of positive mass, so only c = 0 needs its limit.
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
    // The stable root 2c/(q0 + root) of dq/2 t^2 + q0 t = c; clamped for the cell edge's rounding, not for a missing root.
    const float t = c > 0.0F ? std::min((2.0F * c) / (q0 + root), 1.0F) : 0.0F;
    return (static_cast<float>(low) + t) * step;
}

glm::vec3 directionAbout(float mu, float phiU) {
    const float r = std::sqrt(std::max(0.0F, 1.0F - (mu * mu)));
    const float phi = 2.0F * kPi * phiU;
    return {r * std::cos(phi), r * std::sin(phi), mu};
}

// --- The dielectric interface's escape tables over (roughness, eta, mu).

// Fractional index into the log-spaced eta axis, clamped to the tabulated range.
float etaAxisCoord(float eta) {
    const float logMin = std::log(kEtaMin);
    const float u = (std::log(std::clamp(eta, kEtaMin, kEtaMax)) - logMin) / (std::log(kEtaMax) - logMin);
    return u * (kEtaRes - 1);
}

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

// The interface's cosine-weighted mean escape: the reflected share Ravg/(Ravg + Tavg) and the mean deficit 1 - Ravg - Tavg.
struct EscapeMean {
    float reflectShare;
    float deficit;
};

EscapeMean escapeMean(float roughness, float eta) {
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
    const float reflect = lerp1(fetch(kEscapeAvgReflect, r0), fetch(kEscapeAvgReflect, r0 + 1), rt);
    const float transmit = lerp1(fetch(kEscapeAvgTransmit, r0), fetch(kEscapeAvgTransmit, r0 + 1), rt);
    return {reflect / (reflect + transmit), lerp1(fetch(kEscapeAvgDeficit, r0), fetch(kEscapeAvgDeficit, r0 + 1), rt)};
}

// The escape-deficit shape, stored unnormalised and divided by its blended total: blended prefix integrals are the blend's integral.
MsTransmitRow msTransmitRow(float roughness, float eta) {
    const EscapeRow row = escapeRow(roughness, eta);
    const float total = escapeBlend(kMsTransmitCdf, row, kTransmitMuRes - 1);
    return {row, total > 0.0F ? 1.0F / total : 0.0F};
}

float msTransmitDensity(const MsTransmitRow& row, int index) { return escapeBlend(kMsTransmitDensity, row.row, index) * row.scale; }

float msTransmitCdf(const MsTransmitRow& row, int index) { return escapeBlend(kMsTransmitCdf, row.row, index) * row.scale; }

// Solid-angle density: the mu density spread over 2*pi of azimuth, mu measured from the side the lobe leaves on.
float msTransmitPdf(float mu, const MsTransmitRow& row) {
    const float mf = std::clamp(mu, 0.0F, 1.0F) * (kTransmitMuRes - 1);
    const int m0 = std::min(static_cast<int>(mf), kTransmitMuRes - 2);
    const float mt = mf - static_cast<float>(m0);
    return lerp1(msTransmitDensity(row, m0), msTransmitDensity(row, m0 + 1), mt) / (2.0F * kPi);
}

// Kulla-Conty tint from the mean deficit 1 - Eavg: the share of (1-E) energy surviving repeated bounces. Exactly 1 at Favg=1.
float multiScatterTint(float fresnelAvg, float deficitAvg) {
    return (fresnelAvg * fresnelAvg * (1.0F - deficitAvg)) / (1.0F - (fresnelAvg * deficitAvg));
}

glm::vec3 multiScatterTint(const glm::vec3& fresnelAvg, float deficitAvg) {
    return {multiScatterTint(fresnelAvg.x, deficitAvg), multiScatterTint(fresnelAvg.y, deficitAvg),
            multiScatterTint(fresnelAvg.z, deficitAvg)};
}

// Probability the interface's VNDF technique reflects about a facet (Walter 2007 5.3), by the facet's tinted and refracting energies.
float facetReflectProbability(const DielectricSlab& slab, float fresnel) {
    if (slab.refractWeight == 0.0F) {
        return 1.0F;
    }
    const float reflect = slab.tintMean * fresnel;
    // Zero only for a black specular_color at a TIR facet, where neither branch carries energy and reflection is the facet's outcome.
    const float total = reflect + ((1.0F - fresnel) * slab.refractWeight);
    return total > 0.0F ? reflect / total : 1.0F;
}

}  // namespace

float alphaForRoughness(float roughness) { return roughness * roughness; }

bool isSmooth(float alpha) { return alpha * alpha * alpha * alpha < std::numeric_limits<float>::min(); }

glm::vec3 sampleGGXVNDF(const glm::vec3& wo, float alpha, glm::vec2 u) {
    const glm::vec3 vh = glm::normalize(glm::vec3(alpha * wo.x, alpha * wo.y, wo.z));
    const float lensq = (vh.x * vh.x) + (vh.y * vh.y);
    const glm::vec3 t1 = lensq > 0.0F ? glm::vec3(-vh.y, vh.x, 0.0F) * (1.0F / std::sqrt(lensq)) : glm::vec3(1.0F, 0.0F, 0.0F);
    const glm::vec3 t2 = glm::cross(vh, t1);
    const float r = std::sqrt(u.x);
    const float phi = 2.0F * kPi * u.y;
    const float t1p = r * std::cos(phi);
    float t2p = r * std::sin(phi);
    const float s = 0.5F * (1.0F + vh.z);
    t2p = ((1.0F - s) * std::sqrt(std::max(0.0F, 1.0F - (t1p * t1p)))) + (s * t2p);
    const glm::vec3 nh = (t1p * t1) + (t2p * t2) + (std::sqrt(std::max(0.0F, 1.0F - (t1p * t1p) - (t2p * t2p))) * vh);
    return glm::normalize(glm::vec3(alpha * nh.x, alpha * nh.y, std::max(0.0F, nh.z)));
}

glm::vec3 fresnelF82(float cosTheta, const glm::vec3& f0, const glm::vec3& k) {
    const float m = 1.0F - cosTheta;
    const float m5 = (m * m) * (m * m) * m;
    return f0 + ((1.0F - f0) * m5) - (k * (cosTheta * m5 * m));
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

ConductorSlab makeConductorSlab(float roughness, const glm::vec3& f0, const glm::vec3& tint, float scale, float muO) {
    ConductorSlab slab{};
    slab.roughness = roughness;
    slab.alpha = alphaForRoughness(roughness);
    slab.f0 = f0;
    slab.k = f82Weight(f0, tint);
    slab.scale = scale;
    // A smooth metal is its mirror: the albedo is F82 at mu_o exactly, and no multiple scattering exists.
    if (isSmooth(slab.alpha)) {
        slab.singleEnergy = channelMean(scale * fresnelF82(muO, f0, slab.k));
        return slab;
    }
    const AlbedoSplit split = directionalAlbedo(muO, roughness);
    const float deficitAvg = averageAlbedo(roughness).deficit;
    slab.singleEnergy = channelMean(scale * split.at(f0, slab.k));
    // A zero mean deficit has every deficit zero: the lobe vanishes, its exact limit, with no 0/0 to form.
    if (deficitAvg > 0.0F) {
        slab.msTint = multiScatterTint(scale * metalFresnelAvg(f0, tint), deficitAvg);
        slab.msScaleWo = split.deficit / (kPi * deficitAvg);
        slab.multiEnergy = channelMean(slab.msTint) * split.deficit;
    }
    return slab;
}

DielectricSlab makeDielectricSlab(float roughness, float etaI, float etaT, const glm::vec3& tint, float refractWeight,
                                  const glm::vec3& transmitTint, float muO) {
    DielectricSlab slab{};
    slab.alpha = alphaForRoughness(roughness);
    slab.etaI = etaI;
    slab.etaT = etaT;
    slab.tint = tint;
    slab.tintMean = channelMean(tint);
    slab.refractWeight = refractWeight;
    slab.transmitTint = transmitTint;
    slab.etaSq = (etaI / etaT) * (etaI / etaT);
    // Index-matched, the interface is invisible: nothing reflects and everything passes undeviated, at every roughness.
    if (etaI == etaT) {
        slab.transmitSingle = 1.0F;
        return slab;
    }
    if (isSmooth(slab.alpha)) {
        slab.reflectSingle = fresnelDielectric(muO, etaI, etaT);
        slab.transmitSingle = 1.0F - slab.reflectSingle;
        return slab;
    }
    const float eta = etaI / etaT;
    slab.row = escapeRow(roughness, eta);
    slab.reflectSingle = escapeAt(kEscapeReflect, slab.row, muO);
    slab.transmitSingle = escapeAt(kEscapeTransmit, slab.row, muO);
    slab.reflectShape = msTransmitRow(roughness, eta);
    slab.transmitShape = msTransmitRow(roughness, 1.0F / eta);
    // The interface's own multiple scattering, a property of the interface alone: OpenPBR mixes whole BSDFs, so each stays linear.
    const float deficit = escapeAt(kEscapeDeficit, slab.row, muO);
    const EscapeMean mean = escapeMean(roughness, eta);
    // A zero mean deficit or a shape with no density has no energy to carry: the lobe vanishes, its exact limit, with no 0/0 to form.
    if (mean.deficit > 0.0F && slab.reflectShape.scale > 0.0F) {
        slab.multiReflect = deficit * mean.reflectShare;
        slab.multiReflectScaleWo = slab.multiReflect / (kPi * mean.deficit);
    }
    slab.multiTransmit = slab.transmitShape.scale > 0.0F ? deficit * (1.0F - mean.reflectShare) : 0.0F;
    return slab;
}

ConductorEval evaluateConductor(const ConductorSlab& slab, const glm::vec3& wo, const glm::vec3& wi) {
    ConductorEval eval{};
    if (!isSmooth(slab.alpha)) {
        // wo and wi share +z, so dot(wo, h) = |wo + wi|/2 > 0: every half-vector here is a visible facet.
        const glm::vec3 h = glm::normalize(wo + wi);
        const float d = distributionGGX(h, slab.alpha);
        eval.value = (d * smithVisibility(wo.z, wi.z, slab.alpha) * slab.scale * wi.z) * fresnelF82(glm::dot(wo, h), slab.f0, slab.k);
        eval.pdfSingle = 0.25F * d * smithG1OverCos(wo.z, slab.alpha);
    }
    // Kulla & Conty 2017: (1-E(mu_o))(1-E(mu_i))/(pi(1-Eavg)), symmetric in wo and wi, so the lobe stays reciprocal.
    if (slab.msScaleWo > 0.0F) {
        eval.value += slab.msTint * (slab.msScaleWo * directionalDeficit(wi.z, slab.roughness) * wi.z);
        eval.pdfMulti = msReflectPdf(wi.z, slab.roughness);
    }
    return eval;
}

DielectricEval evaluateDielectric(const DielectricSlab& slab, const glm::vec3& wo, const glm::vec3& wi) {
    DielectricEval eval{};
    // An invisible or smooth interface scatters by deltas alone, and a smooth one has no multiple scattering.
    if (isIndexMatched(slab) || isSmooth(slab.alpha)) {
        return eval;
    }
    if (wi.z > 0.0F) {
        const glm::vec3 h = glm::normalize(wo + wi);
        const float woDotH = glm::dot(wo, h);
        const float d = distributionGGX(h, slab.alpha);
        const float fresnel = fresnelDielectric(woDotH, slab.etaI, slab.etaT);
        eval.reflect = slab.tint * (d * smithVisibility(wo.z, wi.z, slab.alpha) * fresnel * wi.z);
        eval.pdfSingle = 0.25F * d * smithG1OverCos(wo.z, slab.alpha) * facetReflectProbability(slab, fresnel);
        // Kulla-Conty's form over the escape deficit, share_R (1-E(mu_o))(1-E(mu_i))/(pi(1-Eavg)): symmetric, so reflection is reciprocal.
        if (slab.multiReflect > 0.0F) {
            eval.reflect += slab.tint * (slab.multiReflectScaleWo * escapeAt(kEscapeDeficit, slab.row, wi.z) * wi.z);
            eval.pdfMultiReflect = msTransmitPdf(wi.z, slab.reflectShape);
        }
        return eval;
    }
    // A zero refractWeight passes the interface's transmission to the diffuse below: nothing leaves on the far side.
    if (slab.refractWeight == 0.0F) {
        return eval;
    }
    // The transmitted share of multiple scattering for any far-side wi: its value is its energy times its own density, eta^2-compressed.
    if (slab.multiTransmit > 0.0F) {
        eval.pdfMultiTransmit = msTransmitPdf(-wi.z, slab.transmitShape);
        eval.transmit = slab.transmitTint * (slab.refractWeight * slab.etaSq * slab.multiTransmit * eval.pdfMultiTransmit);
    }
    // Walter 2007 (eq. 21, 16, 17), PBRT-v3 radiance form: eta^2 (Veach 1997 5.2) is already folded in, hence etaR^2 in the pdf only.
    const float etaR = slab.etaT / slab.etaI;
    glm::vec3 ht = glm::normalize(wo + (etaR * wi));
    if (ht.z < 0.0F) {
        ht = -ht;
    }
    const float woDotH = glm::dot(wo, ht);
    const float wiDotH = glm::dot(wi, ht);
    // Both on the same side of the microfacet means this pair is a reflection about ht, not a refraction.
    if (woDotH <= 0.0F || wiDotH >= 0.0F) {
        return eval;
    }
    // |wo + etaR*wi| >= |1 - etaR| > 0: etaR is never 1 here, index-matched interfaces being deltas.
    const float denom = woDotH + (etaR * wiDotH);
    const float denom2 = denom * denom;
    const float d = distributionGGX(ht, slab.alpha);
    const float fresnel = fresnelDielectric(woDotH, slab.etaI, slab.etaT);
    // D*G2*|wi.h|*(wo.h)/(wo.z*denom^2), with G2/(wo.z*|wi.z|) taken as 4*smithVisibility and |wi.z| the cosine weight.
    const float value = (4.0F * d * smithVisibility(wo.z, -wi.z, slab.alpha) * -wiDotH * woDotH * -wi.z) / denom2;
    const float vndfPdf = d * woDotH * smithG1OverCos(wo.z, slab.alpha);
    eval.transmit += slab.transmitTint * ((1.0F - fresnel) * slab.refractWeight * value);
    eval.pdfSingle = (1.0F - facetReflectProbability(slab, fresnel)) * vndfPdf * etaR * etaR * -wiDotH / denom2;
    return eval;
}

std::optional<glm::vec3> sampleConductorSingle(const ConductorSlab& slab, const glm::vec3& wo, glm::vec2 u) {
    const glm::vec3 wi = glm::reflect(-wo, sampleGGXVNDF(wo, slab.alpha, u));
    return wi.z > 0.0F ? std::optional(wi) : std::nullopt;
}

glm::vec3 sampleConductorMulti(const ConductorSlab& slab, glm::vec2 u) {
    const MsReflectRow row = msReflectRow(slab.roughness);
    const float mu = invertPiecewiseLinearDensity([&](int i) { return msReflectDensity(row, i); },
                                                   [&](int i) { return msReflectCdf(row, i); }, kMsReflectMuRes, u.x);
    return directionAbout(mu, u.y);
}

std::optional<InterfaceSample> sampleDielectricSingle(const DielectricSlab& slab, const glm::vec3& wo, glm::vec2 u, float uSplit) {
    const glm::vec3 h = sampleGGXVNDF(wo, slab.alpha, u);
    const float woDotH = glm::dot(wo, h);
    if (uSplit >= facetReflectProbability(slab, fresnelDielectric(woDotH, slab.etaI, slab.etaT))) {
        glm::vec3 wi;
        if (!refractAbout(wo, h, slab.etaI / slab.etaT, wi) || wi.z >= 0.0F) {
            return std::nullopt;
        }
        return InterfaceSample{wi, true};
    }
    const glm::vec3 wi = glm::reflect(-wo, h);
    return wi.z > 0.0F ? std::optional(InterfaceSample{wi, false}) : std::nullopt;
}

glm::vec3 sampleEscapeShape(const MsTransmitRow& shape, glm::vec2 u) {
    const float mu = invertPiecewiseLinearDensity([&](int i) { return msTransmitDensity(shape, i); },
                                                   [&](int i) { return msTransmitCdf(shape, i); }, kTransmitMuRes, u.x);
    return directionAbout(mu, u.y);
}

}  // namespace pathtracer::scene
