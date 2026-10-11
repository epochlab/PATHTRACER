#include "microfacet.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "pathtracer/scene/fresnel_dielectric.h"
#include "pathtracer/scene/smith_transmission.h"
#include "shading_math.h"
#include "thin_film.h"

namespace pathtracer::scene {

namespace {

// GGX D, cancellation-free (Filament 4.4.2) and anisotropic (Burley 2012): ax ay/(pi d^2), d = ax ay hz^2 + (ay/ax) hx^2 + (ax/ay) hy^2.
float distributionGGX(const glm::vec3& h, const glm::vec2& alpha) {
    const float d = (alpha.x * alpha.y * h.z * h.z) + ((alpha.y / alpha.x) * h.x * h.x) + ((alpha.x / alpha.y) * h.y * h.y);
    return (alpha.x * alpha.y) / (kPi * d * d);
}

// |(ax wx, ay wy, wz)|, the Smith Lambda radical scaled by cosine (Heitz 2014): 1 + Lambda(w) = (|wz| + radical)/(2|wz|), finite at wz = 0.
float smithRadical(const glm::vec3& w, const glm::vec2& alpha) {
    return std::sqrt((alpha.x * alpha.x * w.x * w.x) + (alpha.y * alpha.y * w.y * w.y) + (w.z * w.z));
}

// G2/(4|cosO cosI|), height-correlated Smith (Heitz 2014): cosines multiply rather than divide, so it is exact to the silhouette.
float smithVisibility(const glm::vec3& wo, const glm::vec3& wi, const glm::vec2& alpha) {
    return 0.5F / ((std::abs(wi.z) * smithRadical(wo, alpha)) + (wo.z * smithRadical(wi, alpha)));
}

// Refraction's height-correlated Smith G2 = B(1 + Lambda_o, 1 + Lambda_i), Lambda = (radical/|cos| - 1)/2; 0 at grazing, its limit.
float smithTransmitG2(const glm::vec3& wo, const glm::vec3& wi, const glm::vec2& alpha) {
    const float cosO = std::abs(wo.z);
    const float cosI = std::abs(wi.z);
    if (!(cosO > 0.0F && cosI > 0.0F)) {
        return 0.0F;
    }
    const double lambdaO = 0.5 * ((smithRadical(wo, alpha) / cosO) - 1.0F);
    const double lambdaI = 0.5 * ((smithRadical(wi, alpha) / cosI) - 1.0F);
    return static_cast<float>(pathtracer::scene::smithTransmitG2(lambdaO, lambdaI));
}

// G1(wo)/cosO, the VNDF pdf's projected-area factor, in the same division-free form: 2/alpha_o at grazing rather than 0/0.
float smithG1OverCos(const glm::vec3& wo, const glm::vec2& alpha) { return 2.0F / (wo.z + smithRadical(wo, alpha)); }

// The least alpha whose GGX peak denominator pi*alpha^4 is a normal float: below it D overflows and that axis is a delta.
const float kLeastAlpha = std::sqrt(std::sqrt(std::numeric_limits<float>::min()));

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

// The cosine-weighted mean deficit 1 - Eavg, the Kulla-Conty lobe's normaliser, linear in roughness.
float averageDeficit(float roughness) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kAlbedoRoughnessRes - 1);
    const int r0 = std::min(static_cast<int>(rf), kAlbedoRoughnessRes - 2);
    return lerp1(kAlbedoAvgDeficit[r0], kAlbedoAvgDeficit[r0 + 1], rf - static_cast<float>(r0));
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

// The interface's mean escape: the share of 2+ bounce energy leaving reflected, from the random walks, and the mean deficit.
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
    const float reflect = lerp1(fetch(kEscapeWalkReflect, r0), fetch(kEscapeWalkReflect, r0 + 1), rt);
    const float transmit = lerp1(fetch(kEscapeWalkTransmit, r0), fetch(kEscapeWalkTransmit, r0 + 1), rt);
    return {reflect / (reflect + transmit), lerp1(fetch(kEscapeAvgDeficit, r0), fetch(kEscapeAvgDeficit, r0 + 1), rt)};
}

// The escape-deficit shape, stored unnormalised and divided by its blended total: blended prefix integrals are the blend's integral.
EscapeShape escapeShape(float roughness, float eta) {
    const EscapeRow row = escapeRow(roughness, eta);
    const float total = escapeBlend(kEscapeShapeCdf, row, kTransmitMuRes - 1);
    return {row, total > 0.0F ? 1.0F / total : 0.0F};
}

float escapeShapeDensity(const EscapeShape& row, int index) { return escapeBlend(kEscapeShapeDensity, row.row, index) * row.scale; }

float escapeShapeCdf(const EscapeShape& row, int index) { return escapeBlend(kEscapeShapeCdf, row.row, index) * row.scale; }

// Solid-angle density: the mu density spread over 2*pi of azimuth, mu measured from the side the lobe leaves on.
float escapeShapePdf(float mu, const EscapeShape& row) {
    const float mf = std::clamp(mu, 0.0F, 1.0F) * (kTransmitMuRes - 1);
    const int m0 = std::min(static_cast<int>(mf), kTransmitMuRes - 2);
    const float mt = mf - static_cast<float>(m0);
    return lerp1(escapeShapeDensity(row, m0), escapeShapeDensity(row, m0 + 1), mt) / (2.0F * kPi);
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
float facetReflectProbability(const DielectricSlab& slab, const glm::vec3& fresnel) {
    if (slab.refractWeight == 0.0F || slab.deltaRefraction) {
        return 1.0F;
    }
    const float reflect = channelMean(slab.tint * fresnel);
    // Zero only for a black specular_color at a TIR facet, where neither branch carries energy and reflection is the facet's outcome.
    const float total = reflect + (channelMean(1.0F - fresnel) * slab.refractWeight);
    return total > 0.0F ? reflect / total : 1.0F;
}

// F82 before its clamp: Schlick less k mu (1 - mu)^6.
glm::vec3 unclampedF82(float cosTheta, const glm::vec3& f0, const glm::vec3& k) {
    const float m = 1.0F - cosTheta;
    const float m5 = (m * m) * (m * m) * m;
    return f0 + ((1.0F - f0) * m5) - (k * (cosTheta * m5 * m));
}

// The metal's F82-tint Fresnel (OpenPBR; Hoffman 2023) at cosTheta in [0,1]: a reflectance, so its dip below 0 at dark tints is 0.
glm::vec3 fresnelF82(float cosTheta, const glm::vec3& f0, const glm::vec3& k) { return glm::max(unclampedF82(cosTheta, f0, k), glm::vec3(0.0F)); }

// F82's dip below 0 as its zeros: F' = -(1-mu)^4 (7k mu^2 - 8k mu + k + 5(1-f0)) is 0 at mu_a < mu_b; zeros in [0,mu_a], [mu_a,mu_b].
std::optional<glm::vec2> f82Dip(float f0, float k) {
    const float discriminant = 4.0F * k * ((9.0F * k) - (35.0F * (1.0F - f0)));
    if (!(k > 0.0F) || discriminant < 0.0F) {
        return std::nullopt;
    }
    const float muA = ((8.0F * k) - std::sqrt(discriminant)) / (14.0F * k);
    const float muB = ((8.0F * k) + std::sqrt(discriminant)) / (14.0F * k);
    const auto f82 = [&](float mu) { return unclampedF82(mu, glm::vec3(f0), glm::vec3(k)).x; };
    if (!(f82(muA) < 0.0F)) {
        return std::nullopt;
    }
    // Bisection to float resolution on a monotone bracket; the dip's moment errs only to second order in a zero's error.
    const auto zero = [&](float lo, float hi) {
        const bool negativeLo = f82(lo) < 0.0F;
        for (int i = 0; i < std::numeric_limits<float>::digits; ++i) {
            const float mid = 0.5F * (lo + hi);
            (f82(mid) < 0.0F) == negativeLo ? lo = mid : hi = mid;
        }
        return 0.5F * (lo + hi);
    };
    return glm::vec2(zero(0.0F, muA), zero(muA, muB));
}

// F82 >= 0 on [0, 1] in every channel: the split E = F0 a + b - k c holds exactly.
bool f82IsNonnegative(const glm::vec3& f0, const glm::vec3& k) {
    return !f82Dip(f0.x, k.x) && !f82Dip(f0.y, k.y) && !f82Dip(f0.z, k.z);
}

// The antiderivative of 2 mu F82(mu) in m = 1 - mu: f0 mu^2 - 2(1 - f0)(m^6/6 - m^7/7) + 2k(m^7/7 - m^8/4 + m^9/9).
float f82Moment(float mu, float f0, float k) {
    const float m = 1.0F - mu;
    const float m6 = (m * m * m) * (m * m * m);
    return (f0 * mu * mu) - (2.0F * (1.0F - f0) * m6 * ((1.0F / 6.0F) - (m / 7.0F))) +
           (2.0F * k * m6 * m * ((1.0F / 7.0F) - (m / 4.0F) + (m * m / 9.0F)));
}

// E[F] under the reflect kernel at (roughness, mu): its Gauss rule blended bilinearly between the four grid rules about the point.
template <typename Fresnel>
glm::vec3 kernelExpectation(float roughness, float mu, Fresnel fresnel) {
    const float rf = std::clamp(roughness, 0.0F, 1.0F) * (kKernelRoughnessRes - 1);
    const float mf = std::sqrt(std::clamp(mu, 0.0F, 1.0F)) * (kKernelMuRes - 1);
    const int r0 = std::min(static_cast<int>(rf), kKernelRoughnessRes - 2);
    const int m0 = std::min(static_cast<int>(mf), kKernelMuRes - 2);
    const float rt = rf - static_cast<float>(r0);
    const float mt = mf - static_cast<float>(m0);
    const auto blend = [&](const auto& table, int k) {
        const auto at = [&](int r, int m) { return table[(((r * kKernelMuRes) + m) * kKernelOrder) + k]; };
        return lerp1(lerp1(at(r0, m0), at(r0, m0 + 1), mt), lerp1(at(r0 + 1, m0), at(r0 + 1, m0 + 1), mt), rt);
    };
    glm::vec3 expectation(0.0F);
    for (int k = 0; k < kKernelOrder; ++k) {
        expectation += blend(kKernelWeight, k) * fresnel(blend(kKernelNode, k));
    }
    return expectation;
}

// The hemispherical average 2 int F(mu) mu dmu by its Gauss rule, the Kulla-Conty tint's input for a Fresnel with no closed form.
template <typename Fresnel>
glm::vec3 averageExpectation(Fresnel fresnel) {
    glm::vec3 average(0.0F);
    for (int k = 0; k < kKernelOrder; ++k) {
        average += kAverageWeight[k] * fresnel(kAverageNode[k]);
    }
    return average;
}

// The anisotropy axis coordinate 1 - sqrt(alpha_b/alpha_t), the generator's: nodes crowd toward a = 1, where E curves as alpha_b -> 0.
float anisoAxis(float anisotropy) { return 1.0F - std::sqrt(1.0F - std::clamp(anisotropy, 0.0F, 1.0F)); }

// The table's (alpha_t, alpha_b) at a (roughness, anisotropy) node, unfloored as the generator bakes it.
glm::vec2 anisoNodeAlpha(int roughnessIndex, int anisotropyIndex) {
    const float r = static_cast<float>(roughnessIndex) / static_cast<float>(kAnisoRoughnessRes - 1);
    const float v = 1.0F - (static_cast<float>(anisotropyIndex) / static_cast<float>(kAnisoAnisotropyRes - 1));
    const float tangent = r * r * std::sqrt(2.0F / (1.0F + (v * v * v * v)));
    return {tangent, v * v * tangent};
}

// The azimuth axis: sqrt(alpha_o) from the row's tangent end to its bitangent end, alpha_o = |(alpha_t cos phi, alpha_b sin phi)|.
float projectedAxis(const glm::vec2& alpha, float cos2) {
    const float span = std::sqrt(alpha.x) - std::sqrt(alpha.y);
    // An isotropic row's E has no azimuth, so any coordinate reads it; the generator spaces it uniformly in phi, which 0 matches.
    if (!(span > 0.0F)) {
        return 0.0F;
    }
    const float projected = std::sqrt(std::sqrt((alpha.x * alpha.x * cos2) + (alpha.y * alpha.y * (1.0F - cos2))));
    return (std::sqrt(alpha.x) - projected) / span;
}

// The anisotropic tables' quadrilinear cell over (roughness, anisoAxis, sqrt(mu), projectedAxis at each corner row's own alphas).
struct AnisoCell {
    static constexpr int kCorners = 16;
    std::array<int, kCorners> index;
    std::array<float, kCorners> weight;
    template <typename Table>
    [[nodiscard]] float read(const Table& table) const {
        float sum = 0.0F;
        for (int c = 0; c < kCorners; ++c) {
            sum += weight[c] * table[index[c]];
        }
        return sum;
    }
};

// The axis position of x on a uniform grid of res nodes over [0, 1].
struct Axis {
    int i0;
    float t;
};

Axis axisAt(float x, int res) {
    const float f = std::clamp(x, 0.0F, 1.0F) * static_cast<float>(res - 1);
    const int i0 = std::min(static_cast<int>(f), res - 2);
    return {i0, f - static_cast<float>(i0)};
}

float lerpWeight(const Axis& axis, int upper) { return upper != 0 ? axis.t : 1.0F - axis.t; }

// The tables' rows about (roughness, anisotropy): per slab, so each direction's lookup computes only its mu and azimuth axes.
AnisoRows anisoRows(float roughness, float anisotropy) {
    const Axis r = axisAt(roughness, kAnisoRoughnessRes);
    const Axis a = axisAt(anisoAxis(anisotropy), kAnisoAnisotropyRes);
    AnisoRows rows{};
    for (std::size_t c = 0; c < rows.row.size(); ++c) {
        const int dr = static_cast<int>(c / 2);
        const int da = static_cast<int>(c % 2);
        rows.row[c] = ((r.i0 + dr) * kAnisoAnisotropyRes) + a.i0 + da;
        rows.weight[c] = lerpWeight(r, dr) * lerpWeight(a, da);
        rows.alpha[c] = anisoNodeAlpha(r.i0 + dr, a.i0 + da);
    }
    return rows;
}

// Anisotropic GGX is even in wx and in wy, so only cos^2 of w's azimuth enters; normal incidence has none and reads the tangent end.
AnisoCell anisoCell(const AnisoRows& rows, const glm::vec3& w) {
    const Axis m = axisAt(std::sqrt(std::max(w.z, 0.0F)), kAnisoMuRes);
    const float planar = (w.x * w.x) + (w.y * w.y);
    const float cos2 = planar > 0.0F ? (w.x * w.x) / planar : 1.0F;
    AnisoCell cell{};
    int c = 0;
    for (std::size_t r = 0; r < rows.row.size(); ++r) {
        const Axis p = axisAt(projectedAxis(rows.alpha[r], cos2), kAnisoPhiRes);
        for (int dm = 0; dm < 2; ++dm) {
            for (int dp = 0; dp < 2; ++dp) {
                cell.index[c] = (((rows.row[r] * kAnisoMuRes) + m.i0 + dm) * kAnisoPhiRes) + p.i0 + dp;
                cell.weight[c] = rows.weight[r] * lerpWeight(m, dm) * lerpWeight(p, dp);
                ++c;
            }
        }
    }
    return cell;
}

// E[F] under the anisotropic kernel, each corner's rule applied then blended: near a = 1 adjacent rules differ too much to pair nodes.
template <typename Fresnel>
glm::vec3 anisoKernelExpectation(const AnisoCell& cell, Fresnel fresnel) {
    glm::vec3 expectation(0.0F);
    for (int c = 0; c < AnisoCell::kCorners; ++c) {
        if (cell.weight[c] == 0.0F) {
            continue;
        }
        const int base = cell.index[c] * kKernelOrder;
        glm::vec3 corner(0.0F);
        for (int k = 0; k < kKernelOrder; ++k) {
            corner += kAnisoWeight[base + k] * fresnel(kAnisoNode[base + k]);
        }
        expectation += cell.weight[c] * corner;
    }
    return expectation;
}

// E[F] under the reflect kernel at wo: the isotropic (roughness, mu) rules, or the anisotropic ones at wo's azimuth from the tangent.
template <typename Fresnel>
glm::vec3 reflectExpectation(float roughness, float anisotropy, const glm::vec3& wo, Fresnel fresnel) {
    return anisotropy > 0.0F ? anisoKernelExpectation(anisoCell(anisoRows(roughness, anisotropy), wo), fresnel)
                             : kernelExpectation(roughness, wo.z, fresnel);
}

// The anisotropic microsurface's cosine-weighted mean deficit 1 - Eavg, bilinear over its rows.
float anisoDeficitAvg(const AnisoRows& rows) {
    float mean = 0.0F;
    for (std::size_t r = 0; r < rows.row.size(); ++r) {
        mean += rows.weight[r] * kAnisoDeficitAvg[rows.row[r]];
    }
    return mean;
}

// The film's reflectance over its ambient side, a statistical mix (OpenPBR): ambient (index 1) with 1 - C, the coat's n_c with C.
template <typename At>
glm::vec3 overAmbient(const FilmLayer& film, At at) {
    glm::vec3 reflectance(0.0F);
    if (film.coatWeight < 1.0F) {
        reflectance += (1.0F - film.coatWeight) * at(1.0F);
    }
    if (film.coatWeight > 0.0F) {
        reflectance += film.coatWeight * at(film.coatIor);
    }
    return reflectance;
}

glm::vec3 conductorFilm(const ConductorSlab& slab, float cosTheta) {
    const ThinFilm film{slab.film.thicknessNm, slab.film.ior};
    return overAmbient(slab.film, [&](float ambient) { return filmReflectance(film, cosTheta, ambient, slab.filmEta, slab.filmKappa); });
}

// From outside the film lies on the base; from inside, the base is the film's incident medium and the ambient its substrate.
glm::vec3 interfaceFilm(const DielectricSlab& slab, float cosTheta) {
    const ThinFilm film{slab.film.thicknessNm, slab.film.ior};
    return overAmbient(slab.film, [&](float ambient) {
        return slab.fromBase ? filmReflectance(film, cosTheta, slab.baseIor, glm::vec3(ambient), glm::vec3(0.0F))
                             : filmReflectance(film, cosTheta, ambient, glm::vec3(slab.baseIor), glm::vec3(0.0F));
    });
}

// The conductor's deficit 1 - E at w for unit Fresnel: the isotropic table by mu, the anisotropic one by mu and azimuth.
float conductorDeficit(const ConductorSlab& slab, const glm::vec3& w) {
    return slab.anisotropy > 0.0F ? anisoCell(slab.rows, w).read(kAnisoDeficit)
                                  : directionalDeficit(w.z, slab.roughness);
}

}  // namespace

glm::vec3 conductorFresnel(const ConductorSlab& slab, float cosTheta) {
    const glm::vec3 bare = fresnelF82(cosTheta, slab.f0, slab.k);
    return slab.scale * (slab.film.weight > 0.0F ? glm::mix(bare, conductorFilm(slab, cosTheta), slab.film.weight) : bare);
}

glm::vec3 interfaceFresnel(const DielectricSlab& slab, float cosTheta) {
    const glm::vec3 bare(fresnelDielectric(cosTheta, slab.fresnelEtaI, slab.fresnelEtaT));
    return slab.film.weight > 0.0F ? glm::mix(bare, interfaceFilm(slab, cosTheta), slab.film.weight) : bare;
}

glm::vec2 alphaForRoughness(float roughness, float anisotropy) {
    const float tangent = roughness * roughness * std::sqrt(2.0F / (1.0F + ((1.0F - anisotropy) * (1.0F - anisotropy))));
    // alpha_b = 0 at full anisotropy is a delta across the grain, which no float D represents: it is held at the least alpha that is.
    return {tangent, std::max((1.0F - anisotropy) * tangent, kLeastAlpha)};
}

bool isSmooth(float alpha) { return alpha < kLeastAlpha; }

glm::vec3 sampleGGXVNDF(const glm::vec3& wo, const glm::vec2& alpha, glm::vec2 u) {
    const glm::vec3 vh = glm::normalize(glm::vec3(alpha.x * wo.x, alpha.y * wo.y, wo.z));
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
    return glm::normalize(glm::vec3(alpha.x * nh.x, alpha.y * nh.y, std::max(0.0F, nh.z)));
}

// 2*int mu*F82 dmu of the clamped F82: Schlick's mean F0 + (1-F0)/21 less k * 2*B(3, 7) = k/126, less any dip's moment, exactly.
glm::vec3 metalFresnelAvg(const glm::vec3& f0, const glm::vec3& tint) {
    const glm::vec3 k = f82Weight(f0, tint);
    glm::vec3 average = f0 + ((1.0F - f0) / 21.0F) - (k / 126.0F);
    for (int c = 0; c < 3; ++c) {
        if (const std::optional<glm::vec2> dip = f82Dip(f0[c], k[c])) {
            average[c] -= f82Moment(dip->y, f0[c], k[c]) - f82Moment(dip->x, f0[c], k[c]);
        }
    }
    return average;
}

// External linkage: checkAlbedoTableInterpolation is the only instrument that sees the .inc's interpolation error.
glm::vec4 directionalAlbedoSplit(float mu, float roughness) {
    const AlbedoSplit split = directionalAlbedo(mu, roughness);
    return {split.a, split.b, split.c, split.deficit};
}

float averageAlbedoDeficit(float roughness) { return averageDeficit(roughness); }

// The grid the two lookups index, described rather than transcribed. Both axes edge-aligned, so 0 and res-1 are exact endpoints.
glm::ivec2 albedoGridRes() { return {kAlbedoRoughnessRes, kAlbedoMuRes}; }

float albedoGridRoughness(float index) { return index / static_cast<float>(kAlbedoRoughnessRes - 1); }

// Inverts directionalAlbedo's sqrt(mu) index, so an instrument's fractional index lands where that lookup interpolates.
float albedoGridMu(float index) {
    const float t = index / static_cast<float>(kAlbedoMuRes - 1);
    return t * t;
}

ConductorSlab makeConductorSlab(float roughness, float anisotropy, const glm::vec3& f0, const glm::vec3& tint, float scale,
                                const FilmLayer& film, const glm::vec3& wo) {
    // Written field by field, not zeroed: the multiple-scattering terms start at their zero, the film's index only where a film is.
    ConductorSlab slab;
    slab.msScaleWo = 0.0F;
    slab.multiEnergy = 0.0F;
    slab.roughness = roughness;
    slab.anisotropy = anisotropy;
    slab.alpha = alphaForRoughness(roughness, anisotropy);
    slab.f0 = f0;
    slab.k = f82Weight(f0, tint);
    slab.scale = scale;
    slab.film = film;
    // OpenPBR leaves F82 without a complex index; the film needs one, so Gulbrandsen's map takes (base_color, specular_color) to it.
    if (film.weight > 0.0F) {
        const ComplexIor ior = conductorIor(f0, tint);
        slab.filmEta = ior.eta;
        slab.filmKappa = ior.kappa;
    }
    // A smooth metal is its mirror: the albedo is its Fresnel at mu_o exactly, and no multiple scattering exists.
    if (isSmooth(slab.alpha.x)) {
        slab.albedo = conductorFresnel(slab, wo.z);
        slab.singleEnergy = channelMean(slab.albedo);
        return slab;
    }
    glm::vec3 single;
    float deficit;
    float deficitAvg;
    AnisoCell cell{};
    // A Fresnel with no split takes the reflect kernel's Gauss rule over it, isotropic or at wo's anisotropic cell.
    const auto kernel = [&](auto fresnel) {
        return anisotropy > 0.0F ? anisoKernelExpectation(cell, fresnel) : kernelExpectation(roughness, wo.z, fresnel);
    };
    // F82's split E = F0 a + b - k c is exact wherever F82 >= 0; where the clamp binds, the clamped F82 has no split.
    const bool split = f82IsNonnegative(f0, slab.k);
    const auto bare = [&](float x) { return scale * fresnelF82(x, f0, slab.k); };
    if (anisotropy > 0.0F) {
        slab.rows = anisoRows(roughness, anisotropy);
        cell = anisoCell(slab.rows, wo);
        deficit = cell.read(kAnisoDeficit);
        // The 4D tables carry b, c and the deficit, a = 1 - deficit - b.
        const float b = cell.read(kAnisoB);
        single = split ? scale * ((f0 * (1.0F - deficit - b)) + b - (slab.k * cell.read(kAnisoC))) : kernel(bare);
        deficitAvg = anisoDeficitAvg(slab.rows);
    } else {
        const AlbedoSplit albedo = directionalAlbedo(wo.z, roughness);
        single = split ? scale * albedo.at(f0, slab.k) : kernel(bare);
        deficit = albedo.deficit;
        deficitAvg = averageDeficit(roughness);
    }
    glm::vec3 average = scale * metalFresnelAvg(f0, tint);
    // The film's Fresnel has no split: its albedo is the reflect kernel's Gauss rule over it, its average the 2 mu dmu rule's.
    if (film.weight > 0.0F) {
        const auto filmed = [&](float x) { return scale * conductorFilm(slab, x); };
        single = glm::mix(single, kernel(filmed), film.weight);
        average = glm::mix(average, averageExpectation(filmed), film.weight);
    }
    slab.singleEnergy = channelMean(single);
    slab.albedo = single;
    // A zero mean deficit has every deficit zero: the lobe vanishes, its exact limit, with no 0/0 to form.
    if (deficitAvg > 0.0F) {
        slab.msTint = multiScatterTint(average, deficitAvg);
        slab.msScaleWo = deficit / (kPi * deficitAvg);
        slab.multiEnergy = channelMean(slab.msTint) * deficit;
        slab.albedo += slab.msTint * deficit;
    }
    return slab;
}

namespace {

// The interface's reflection at mu by the escape tables at the Fresnel ratio: single scattering, and the reflected multiple scattering.
struct InterfaceReflection {
    float single;
    float multi;
    float deficit;
    EscapeMean mean;
};

InterfaceReflection interfaceReflection(const EscapeRow& row, float roughness, float eta, float mu) {
    const float deficit = escapeAt(kEscapeDeficit, row, mu);
    const EscapeMean mean = escapeMean(roughness, eta);
    return {escapeAt(kEscapeReflect, row, mu), mean.deficit > 0.0F ? deficit * mean.reflectShare : 0.0F, deficit, mean};
}

}  // namespace

DielectricSlab makeDielectricSlab(const InterfaceInputs& interface, const glm::vec3& wo) {
    // Written field by field, not zeroed: the escape rows are read only behind the multiple-scattering energies, which start at zero.
    DielectricSlab slab;
    slab.reflectSingle = glm::vec3(0.0F);
    slab.multiReflect = 0.0F;
    slab.multiTransmit = 0.0F;
    slab.alpha = alphaForRoughness(interface.roughness, interface.anisotropy);
    slab.etaI = interface.etaI;
    slab.etaT = interface.etaT;
    slab.fresnelEtaI = interface.fresnelEtaI;
    slab.fresnelEtaT = interface.fresnelEtaT;
    slab.tint = interface.tint;
    slab.tintMean = channelMean(interface.tint);
    slab.refractWeight = interface.refractWeight;
    slab.transmitTint = interface.transmitTint;
    slab.etaSq = (interface.etaI / interface.etaT) * (interface.etaI / interface.etaT);
    slab.film = interface.film;
    slab.baseIor = interface.baseIor;
    slab.fromBase = interface.fromBase;
    // Index-matched media refract undeviated at every roughness: no facet can bend the ray, so refraction is a delta.
    slab.deltaRefraction = isSmooth(slab.alpha.x) || interface.etaI == interface.etaT;
    if (isSmooth(slab.alpha.x)) {
        // A mirror past the geometric critical angle reflects all, even Fresnel-matched: no refracted direction exists to carry 1 - F.
        const bool tir = cos2Transmitted(wo.z, interface.etaI / interface.etaT) < 0.0F;
        slab.reflectSingle = tir ? glm::vec3(1.0F) : isIndexMatched(slab) ? glm::vec3(0.0F) : interfaceFresnel(slab, wo.z);
        slab.transmitSingle = 1.0F - slab.reflectSingle;
        return slab;
    }
    // A Fresnel-matched, unfilmed interface reflects exactly nothing: no table's interpolation residual may read as reflection.
    if (isIndexMatched(slab)) {
        slab.transmitSingle = glm::vec3(1.0F);
        return slab;
    }
    const auto kernel = [&](auto fresnel) { return reflectExpectation(interface.roughness, interface.anisotropy, wo, fresnel); };
    const auto filmed = [&](float x) { return interfaceFilm(slab, x); };
    // Fresnel-matched under a film: the bare interface reflects nothing and has no multiple scattering; the film alone reflects.
    if (interface.fresnelEtaI == interface.fresnelEtaT) {
        slab.reflectSingle = slab.film.weight * kernel(filmed);
        slab.transmitSingle = glm::vec3(1.0F);
        return slab;
    }
    const float eta = interface.fresnelEtaI / interface.fresnelEtaT;
    // The escape tables have no anisotropy axis: OpenPBR's map keeps alpha_t^2 + alpha_b^2 = 2 r^4, so r is the RMS-equivalent node.
    slab.row = escapeRow(interface.roughness, eta);
    slab.reflectShape = escapeShape(interface.roughness, eta);
    slab.transmitShape = escapeShape(interface.roughness, 1.0F / eta);
    // The interface's own multiple scattering, a property of the interface alone: OpenPBR mixes whole BSDFs, so each stays linear.
    const InterfaceReflection reflection = interfaceReflection(slab.row, interface.roughness, eta, wo.z);
    // Anisotropic single scattering is the 4D kernel's rule over the bare Fresnel, exact in azimuth; a film reshapes it the same way.
    const auto bare = [&](float x) { return glm::vec3(fresnelDielectric(x, interface.fresnelEtaI, interface.fresnelEtaT)); };
    slab.reflectSingle = interface.anisotropy > 0.0F ? kernel(bare) : glm::vec3(reflection.single);
    if (slab.film.weight > 0.0F) {
        slab.reflectSingle = glm::mix(slab.reflectSingle, kernel(filmed), slab.film.weight);
    }
    slab.transmitSingle = glm::vec3(escapeAt(kEscapeTransmit, slab.row, wo.z));
    // A zero mean deficit or a shape with no density has no energy to carry: the lobe vanishes, its exact limit, with no 0/0 to form.
    if (reflection.multi > 0.0F && slab.reflectShape.scale > 0.0F) {
        slab.multiReflect = reflection.multi;
        slab.multiReflectScaleWo = slab.multiReflect / (kPi * reflection.mean.deficit);
    }
    // Undeviated refraction (index-matched media) is a delta: every facet transmits along it, multiple scattering included.
    if (slab.transmitShape.scale > 0.0F) {
        const float transmitted = reflection.deficit * (1.0F - reflection.mean.reflectShare);
        if (slab.deltaRefraction) {
            slab.transmitSingle += transmitted;
        } else {
            slab.multiTransmit = transmitted;
        }
    }
    return slab;
}

SheetLadder sheetLadder(const SheetInterfaces& sheet, float mu) {
    const float cos2Inside = cos2Transmitted(mu, 1.0F / sheet.ior);
    // Past the critical angle no refracted direction exists, and at grazing both faces reflect all: either way nothing enters.
    if (!(cos2Inside > 0.0F) || mu == 0.0F) {
        return {glm::vec3(1.0F), glm::vec3(0.0F)};
    }
    // One crossing at angle theta_t is 1/cos(theta_t) normal crossings, Beer-Lambert's power of the normal-incidence colour.
    const glm::vec3 crossing = glm::pow(sheet.color, glm::vec3(1.0F / std::sqrt(cos2Inside)));
    const glm::vec3 roundTrip = crossing * crossing;
    const auto [farParallel, farPerpendicular] = fresnelDielectricPolarised(mu, 1.0F, sheet.ior);
    const auto [nearParallel, nearPerpendicular] = fresnelDielectricPolarised(mu, 1.0F, sheet.fresnelIor);
    glm::vec3 nearP(nearParallel);
    glm::vec3 nearS(nearPerpendicular);
    // A lossless film reflects alike from either side (Stokes), so one reflectance serves the entry and every internal bounce.
    if (sheet.film.weight > 0.0F) {
        const ThinFilm film{sheet.film.thicknessNm, sheet.film.ior};
        PolarisedReflectance filmed{glm::vec3(0.0F), glm::vec3(0.0F)};
        const auto addAmbient = [&](float ambient, float share) {
            const PolarisedReflectance r = filmReflectancePolarised(film, mu, ambient, glm::vec3(sheet.ior), glm::vec3(0.0F));
            filmed.parallel += share * r.parallel;
            filmed.perpendicular += share * r.perpendicular;
        };
        if (sheet.film.coatWeight < 1.0F) {
            addAmbient(1.0F, 1.0F - sheet.film.coatWeight);
        }
        if (sheet.film.coatWeight > 0.0F) {
            addAmbient(sheet.film.coatIor, sheet.film.coatWeight);
        }
        nearP = glm::mix(nearP, filmed.parallel, sheet.film.weight);
        nearS = glm::mix(nearS, filmed.perpendicular, sheet.film.weight);
    }
    // R' = R1 + T1^2 R2 t^2/(1 - R1 R2 t^2), T' = T1 T2 t/(1 - R1 R2 t^2): the geometric series of bounces between the faces.
    const auto ladder = [&](const glm::vec3& near, float far) {
        const glm::vec3 nearT = 1.0F - near;
        const glm::vec3 loss = 1.0F - (near * far * roundTrip);
        return SheetLadder{near + (nearT * nearT * far * roundTrip / loss), nearT * (1.0F - far) * crossing / loss};
    };
    const SheetLadder parallel = ladder(nearP, farParallel);
    const SheetLadder perpendicular = ladder(nearS, farPerpendicular);
    return {0.5F * (parallel.reflect + perpendicular.reflect), 0.5F * (parallel.transmit + perpendicular.transmit)};
}

SheetSlab makeSheetSlab(const SheetInterfaces& interfaces, float weight, float roughness, float anisotropy, const glm::vec3& wo) {
    SheetSlab slab{interfaces, weight, makeConductorSlab(roughness, anisotropy, glm::vec3(1.0F), glm::vec3(1.0F), 1.0F, FilmLayer{}, wo), {}, {}, 0.0F};
    const SheetLadder ladder = sheetLadder(interfaces, wo.z);
    slab.reflect = weight * interfaces.tint * ladder.reflect;
    slab.transmit = weight * ladder.transmit;
    const float total = channelMean(slab.reflect) + channelMean(slab.transmit);
    slab.reflectShare = total > 0.0F ? channelMean(slab.reflect) / total : 0.0F;
    return slab;
}

float reflectionAlbedo(float roughness, float anisotropy, float etaI, float etaT, const glm::vec3& w) {
    if (etaI == etaT) {
        return 0.0F;
    }
    if (isSmooth(alphaForRoughness(roughness, anisotropy).x)) {
        return fresnelDielectric(w.z, etaI, etaT);
    }
    const float eta = etaI / etaT;
    const InterfaceReflection reflection = interfaceReflection(escapeRow(roughness, eta), roughness, eta, w.z);
    const auto bare = [&](float x) { return glm::vec3(fresnelDielectric(x, etaI, etaT)); };
    const float single = anisotropy > 0.0F ? anisoKernelExpectation(anisoCell(anisoRows(roughness, anisotropy), w), bare).x : reflection.single;
    return single + (escapeShape(roughness, eta).scale > 0.0F ? reflection.multi : 0.0F);
}

namespace {

// The classical closed form for unpolarised light entering n > 1 from outside, in double: its terms cancel to O((n-1)^2) as n -> 1.
double externalFresnelAverage(double n) {
    const double n2 = n * n;
    const double n4 = n2 * n2;
    return 0.5 + ((n - 1.0) * ((3.0 * n) + 1.0) / (6.0 * (n + 1.0) * (n + 1.0))) +
           ((n2 * (n2 - 1.0) * (n2 - 1.0)) / ((n2 + 1.0) * (n2 + 1.0) * (n2 + 1.0)) * std::log((n - 1.0) / (n + 1.0))) -
           ((2.0 * n * n2 * (n2 + (2.0 * n) - 1.0)) / ((n2 + 1.0) * (n4 - 1.0))) +
           ((8.0 * n4 * (n4 + 1.0)) / ((n2 + 1.0) * (n4 - 1.0) * (n4 - 1.0)) * std::log(n));
}

}  // namespace

float fresnelAverage(float eta) {
    if (eta == 1.0F) {
        return 0.0F;
    }
    // Entering a rarer medium is the denser side's internal reflection: 1 - F_int = (1 - F_ext) / n^2 by etendue, n = 1/eta.
    const double n = eta;
    return static_cast<float>(n > 1.0 ? externalFresnelAverage(n) : 1.0 - ((1.0 - externalFresnelAverage(1.0 / n)) * n * n));
}

ConductorEval evaluateConductor(const ConductorSlab& slab, const glm::vec3& wo, const glm::vec3& wi) {
    ConductorEval eval{};
    if (!isSmooth(slab.alpha.x)) {
        // wo and wi share +z, so dot(wo, h) = |wo + wi|/2 > 0: every half-vector here is a visible facet.
        const glm::vec3 h = glm::normalize(wo + wi);
        const float d = distributionGGX(h, slab.alpha);
        eval.value = (d * smithVisibility(wo, wi, slab.alpha) * wi.z) * conductorFresnel(slab, glm::dot(wo, h));
        eval.pdfSingle = 0.25F * d * smithG1OverCos(wo, slab.alpha);
    }
    // Kulla & Conty 2017: (1-E(wo))(1-E(wi))/(pi(1-Eavg)), symmetric in wo and wi, so the lobe stays reciprocal at any anisotropy.
    if (slab.msScaleWo > 0.0F) {
        eval.value += slab.msTint * (slab.msScaleWo * conductorDeficit(slab, wi) * wi.z);
        eval.pdfMulti = msReflectPdf(wi.z, slab.roughness);
    }
    return eval;
}

DielectricEval evaluateDielectric(const DielectricSlab& slab, const glm::vec3& wo, const glm::vec3& wi) {
    DielectricEval eval{};
    // A smooth interface scatters by deltas alone; a Fresnel-matched one reflects nothing and refracts by a delta.
    if (isSmooth(slab.alpha.x)) {
        return eval;
    }
    if (wi.z > 0.0F) {
        if (isIndexMatched(slab)) {
            return eval;
        }
        const glm::vec3 h = glm::normalize(wo + wi);
        const float woDotH = glm::dot(wo, h);
        const float d = distributionGGX(h, slab.alpha);
        const glm::vec3 fresnel = interfaceFresnel(slab, woDotH);
        eval.reflect = slab.tint * fresnel * (d * smithVisibility(wo, wi, slab.alpha) * wi.z);
        eval.pdfSingle = 0.25F * d * smithG1OverCos(wo, slab.alpha) * facetReflectProbability(slab, fresnel);
        // Kulla-Conty's form over the escape deficit, share_R (1-E(mu_o))(1-E(mu_i))/(pi(1-Eavg)): symmetric, so reflection is reciprocal.
        if (slab.multiReflect > 0.0F) {
            eval.reflect += slab.tint * (slab.multiReflectScaleWo * escapeAt(kEscapeDeficit, slab.row, wi.z) * wi.z);
            eval.pdfMultiReflect = escapeShapePdf(wi.z, slab.reflectShape);
        }
        return eval;
    }
    // A zero refractWeight passes the interface's transmission to the diffuse below: nothing leaves on the far side.
    if (slab.refractWeight == 0.0F || slab.deltaRefraction) {
        return eval;
    }
    // The transmitted share of multiple scattering for any far-side wi: its value is its energy times its own density, eta^2-compressed.
    if (slab.multiTransmit > 0.0F) {
        eval.pdfMultiTransmit = escapeShapePdf(-wi.z, slab.transmitShape);
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
    const glm::vec3 fresnel = interfaceFresnel(slab, woDotH);
    // D G2 |wi.h| (wo.h) / (wo.z denom^2), the |wi.z| of Walter's 1/|wi.z| cancelled by the cosine weight; G2 refraction's own.
    const float value = (d * smithTransmitG2(wo, wi, slab.alpha) * -wiDotH * woDotH) / (wo.z * denom2);
    const float vndfPdf = d * woDotH * smithG1OverCos(wo, slab.alpha);
    eval.transmit += slab.transmitTint * (1.0F - fresnel) * (slab.refractWeight * value);
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
    if (uSplit >= facetReflectProbability(slab, interfaceFresnel(slab, woDotH))) {
        glm::vec3 wi;
        if (!refractAbout(wo, h, slab.etaI / slab.etaT, wi) || wi.z >= 0.0F) {
            return std::nullopt;
        }
        return InterfaceSample{wi, true};
    }
    const glm::vec3 wi = glm::reflect(-wo, h);
    return wi.z > 0.0F ? std::optional(InterfaceSample{wi, false}) : std::nullopt;
}

glm::vec3 sampleEscapeShape(const EscapeShape& shape, glm::vec2 u) {
    const float mu = invertPiecewiseLinearDensity([&](int i) { return escapeShapeDensity(shape, i); },
                                                   [&](int i) { return escapeShapeCdf(shape, i); }, kTransmitMuRes, u.x);
    return directionAbout(mu, u.y);
}

}  // namespace pathtracer::scene
