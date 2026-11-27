#include "pathtracer/scene/path_tracer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/gbuffer_shading.h"
#include "pathtracer/scene/sampler.h"
#include "pathtracer/scene/shading_scene.h"
#include "medium.h"
#include "path_graph.h"
#include "shading_math.h"

namespace pathtracer::scene {

namespace {

// Offsets the AO sampler's seed from the path sampler's so the two streams are independent. 2^32/phi, Knuth's decorrelating constant.
constexpr std::uint32_t kAoSeedOffset = 0x9E3779B9U;
// The same for the Fresnel AOV's stream: frac(sqrt 2)*2^32, kAoSeedOffset's sibling. Distinct and non-zero is the whole requirement.
constexpr std::uint32_t kFresnelSeedOffset = 0x6A09E667U;
// And for volume tracking, whose draw count varies per path: frac(sqrt 3)*2^32, so no BSDF dimension shifts with it.
constexpr std::uint32_t kVolumeSeedOffset = 0xBB67AE85U;
// And for the microsurface walk, whose length varies per vertex: frac(sqrt 11)*2^32, the next odd one of the series.
constexpr std::uint32_t kWalkSeedOffset = 0x510E527FU;
// And for the path's hero wavelength, so spectral sampling shifts no dimension: frac(sqrt 5)*2^32, the series' unused member.
constexpr std::uint32_t kWavelengthSeedOffset = 0x3C6EF372U;

// pbrt's ShadowEpsilon (PBR 6.8.6): a relative back-off on tMax, or a light's own front face occludes it. A no-op for the environment.
constexpr float kShadowDistanceEpsilon = 1e-3F;

// The window's cosine-series coefficients: w(t) = a0 - a1 cos(2 pi t) + a2 cos(4 pi t) - a3 cos(6 pi t) over t in [0, 1].
constexpr double kBlackmanHarrisA0 = 0.35875;
constexpr double kBlackmanHarrisA1 = 0.48829;
constexpr double kBlackmanHarrisA2 = 0.14128;
constexpr double kBlackmanHarrisA3 = 0.01168;
// Knots of the half-filter CDF over |x| in [0, kFilterRadius]: exact at each knot, linear between, so 63 bins of 0.024 px each.
constexpr int kFilterCdfKnots = 64;

// CDF of |x| / kFilterRadius from the window's closed-form integral: the window spans t in [0, 1] centred at 0.5, so |x| = 0 maps there.
std::array<float, kFilterCdfKnots> buildFilterCdf() {
    constexpr double kTwoPi = 2.0 * std::numbers::pi;
    const auto integral = [](double t) {
        return (kBlackmanHarrisA0 * t) - (kBlackmanHarrisA1 * std::sin(kTwoPi * t) / kTwoPi) +
               (kBlackmanHarrisA2 * std::sin(2.0 * kTwoPi * t) / (2.0 * kTwoPi)) -
               (kBlackmanHarrisA3 * std::sin(3.0 * kTwoPi * t) / (3.0 * kTwoPi));
    };
    // Differenced against the same centre and normalised by the same span, so the end knots are exactly 0 and 1.
    const double centre = integral(0.5);
    const double half = integral(1.0) - centre;
    std::array<float, kFilterCdfKnots> cdf{};
    for (int i = 0; i < kFilterCdfKnots; ++i) {
        const double t = 0.5 + (0.5 * static_cast<double>(i) / static_cast<double>(kFilterCdfKnots - 1));
        cdf[static_cast<std::size_t>(i)] = static_cast<float>((integral(t) - centre) / half);
    }
    return cdf;
}

// CDF of r / kFilterRadius for the radially symmetric window on the disc, int_0^u f(s) s ds: the area element weights f by s.
std::array<float, kFilterCdfKnots> buildRadialFilterCdf() {
    // f(u) = w(1/2 + u/2) = a0 + sum a_k cos(k pi u), each odd term's sign flipped by the half-period shift.
    const std::array<double, 3> coefficients{kBlackmanHarrisA1, kBlackmanHarrisA2, kBlackmanHarrisA3};
    // int_0^u s cos(w s) ds = u sin(w u) / w + (cos(w u) - 1) / w^2, by parts.
    const auto integral = [&coefficients](double u) {
        double sum = 0.5 * kBlackmanHarrisA0 * u * u;
        for (std::size_t k = 0; k < coefficients.size(); ++k) {
            const double w = static_cast<double>(k + 1) * std::numbers::pi;
            sum += coefficients[k] * ((u * std::sin(w * u) / w) + ((std::cos(w * u) - 1.0) / (w * w)));
        }
        return sum;
    };
    // Normalised by the same closed form, so the end knots are exactly 0 and 1.
    const double total = integral(1.0);
    std::array<float, kFilterCdfKnots> cdf{};
    for (int i = 0; i < kFilterCdfKnots; ++i) {
        const double u = static_cast<double>(i) / static_cast<double>(kFilterCdfKnots - 1);
        cdf[static_cast<std::size_t>(i)] = static_cast<float>(integral(u) / total);
    }
    return cdf;
}

const std::array<float, kFilterCdfKnots> kFilterCdf = buildFilterCdf();
const std::array<float, kFilterCdfKnots> kRadialFilterCdf = buildRadialFilterCdf();

// Film offset from the pixel centre, filter-distributed: the half-CDF inverted at |2u-1|, signed by 2u-1, is monotone so u's strata hold.
float sampleFilterOffset(float u) {
    const float signedU = (2.0F * u) - 1.0F;
    const CdfSample bin = invertCdf(kFilterCdf.data(), kFilterCdfKnots - 1, std::abs(signedU));
    const float radiusFraction = (static_cast<float>(bin.index) + bin.fraction) / static_cast<float>(kFilterCdfKnots - 1);
    return std::copysign(kFilterRadius * radiusFraction, signedU);
}

// Shirley-Chiu's concentric map (1997) as a signed radius and an angle: u's strata hold and rho^2 is uniform on [0, 1].
struct ConcentricPolar {
    float rho;
    float phi;
};

// Nullopt at the map's centre, where its angle is undefined and every radial sample is the axis itself.
std::optional<ConcentricPolar> concentricPolar(glm::vec2 u) {
    const glm::vec2 s = (2.0F * u) - 1.0F;
    if (s.x == 0.0F && s.y == 0.0F) {
        return std::nullopt;
    }
    const bool xMajor = std::abs(s.x) > std::abs(s.y);
    const float rho = xMajor ? s.x : s.y;
    const float phi = xMajor ? 0.25F * std::numbers::pi_v<float> * (s.y / s.x)
                             : (0.5F * std::numbers::pi_v<float>) - (0.25F * std::numbers::pi_v<float> * (s.x / s.y));
    return ConcentricPolar{rho, phi};
}

// Radial offset: the concentric map's uniform rho^2 inverts the radial CDF.
glm::vec2 sampleRadialFilterOffset(glm::vec2 u) {
    const std::optional<ConcentricPolar> polar = concentricPolar(u);
    if (!polar) {
        return glm::vec2(0.0F);
    }
    const CdfSample bin = invertCdf(kRadialFilterCdf.data(), kFilterCdfKnots - 1, polar->rho * polar->rho);
    const float radius = kFilterRadius * (static_cast<float>(bin.index) + bin.fraction) / static_cast<float>(kFilterCdfKnots - 1);
    return std::copysign(radius, polar->rho) * glm::vec2(std::cos(polar->phi), std::sin(polar->phi));
}

// The square lattice's separable filter keeps its per-axis strata; the hexagonal lattice's six equidistant neighbours need it radial.
glm::vec2 sampleFilmOffset(pathtracer::gfx::PixelLattice lattice, glm::vec2 u) {
    if (lattice == pathtracer::gfx::PixelLattice::Hexagonal) {
        return sampleRadialFilterOffset(u);
    }
    return {sampleFilterOffset(u.x), sampleFilterOffset(u.y)};
}

}  // namespace

AcceptanceSample sampleAcceptance(const glm::vec3& axis, float kappa, glm::vec2 u) {
    // 1 - e^-2k, the lobe's mass over the sphere, without cancellation at small kappa; peak is the density on the axis.
    const float mass = -std::expm1(-2.0F * kappa);
    const float peak = kappa / (2.0F * std::numbers::pi_v<float> * mass);
    const std::optional<ConcentricPolar> polar = concentricPolar(u);
    if (!polar) {
        return {axis, peak};
    }
    // CDF (1 - e^-kv) / mass of v = 1 - cos(theta) set to rho^2; capped at the antipode v = 2, where rounding of log1p(-1) overshoots.
    const float v = std::min(2.0F, -std::log1p(-(polar->rho * polar->rho) * mass) / kappa);
    const float sinTheta = std::copysign(std::sqrt(v * (2.0F - v)), polar->rho);
    const glm::vec3 local(sinTheta * std::cos(polar->phi), sinTheta * std::sin(polar->phi), 1.0F - v);
    return {glm::transpose(toFrameAbout(axis)) * local, peak * std::exp(-kappa * v)};
}

float acceptanceSolidAngle(float kappa) {
    return 2.0F * std::numbers::pi_v<float> * -std::expm1(-2.0F * kappa) / kappa;
}

float lobeSpread(float spread, float share) {
    // Variances add under the lobe's convolution (Belcour et al. 2013); no density, an infinite share, spans the sphere.
    return std::sqrt(std::min(4.0F * std::numbers::pi_v<float>, (spread * spread) + share));
}

float scatteredSpread(float spread, float width, const glm::vec3& d, const glm::vec3& wi, const glm::vec3& n, float h, float eta,
                      float share) {
    const float cosI = -glm::dot(d, n);
    const float wiDotN = glm::dot(wi, n);
    const float cosO = std::abs(wiDotN);
    const float sphere = std::sqrt(4.0F * std::numbers::pi_v<float>);
    // Grazing in or out, etendue and the curvature term diverge: the limit is the whole sphere.
    if (!(cosI * cosO > 0.0F)) {
        return sphere;
    }
    // Igehy 1999: wi = eta d + mu n, so a normal turned by h width turns wi by mu h width; mu = 2 cos on reflection.
    const float mu = (eta * cosI) + wiDotN;
    // Etendue n^2 cos dOmega is invariant; in- and out-of-plane terms differ by cosI/cosO, the isotropic cone takes their geometric mean.
    const float transferred = (eta * std::sqrt(cosI / cosO) * spread) + (mu * h * width / std::sqrt(cosI * cosO));
    return share > 0.0F ? lobeSpread(transferred, share) : std::clamp(transferred, -sphere, sphere);
}

// Factoring, x = 2 (Hummel, Schonberg and Flynn 1992): batches of P chunks of ceil(R / 2P), so each batch hands out half of what remains.
std::vector<int> pathTraceChunkBounds(int texels, unsigned int threadCount) {
    const int threads = static_cast<int>(threadCount);
    std::vector<int> bounds{0};
    while (bounds.back() < texels) {
        const int chunk = (texels - bounds.back() + (2 * threads) - 1) / (2 * threads);
        for (int i = 0; i < threads && bounds.back() < texels; ++i) {
            bounds.push_back(std::min(bounds.back() + chunk, texels));
        }
    }
    return bounds;
}

std::vector<std::uint32_t> allTexels(int width, int height) {
    std::vector<std::uint32_t> texels(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    std::iota(texels.begin(), texels.end(), 0U);
    return texels;
}

namespace {

// Two offsets of length |spread| across unit dir, a direction footprint whose solid angle is spread^2.
glm::mat2x3 spreadAcross(const glm::vec3& dir, float spread) {
    const glm::mat3 across = glm::transpose(toFrameAbout(dir));
    return {across[0] * std::abs(spread), across[1] * std::abs(spread)};
}

// The index ratio etaI/etaT a crossing bends by, from wo's side: the base dielectric's, 1 through a thin wall.
float bendRatio(const NormalMappedBsdf& bsdf) {
    const BsdfClosure& base = !bsdf.perturbed || bsdf.lambdaP > 0.0F ? bsdf.facet : bsdf.viaWall;
    return base.thinWalled ? 1.0F : base.dielectric.etaI / base.dielectric.etaT;
}

struct TraceResult {
    Rgb radiance{0.0F};
    int terminationBounce = 0;  // bounce index the path stopped at (== maxBounces + 1 if depth-capped)
    float shadowLost = 0.0F;        // bounce-0 direct luminance the occluders remove, light-sampled without MIS; 0 with no shadow ray
    float shadowUnoccluded = 0.0F;  // the same estimate with visibility 1: Shadow is the ratio of their means
    float ao = 0.0F;  // 1.0 = unoccluded, 0.0 = occluded within aoMaxDistance; 0 with no ray at all
    // One VNDF draw's Fresnel at the primary hit; 0 where there is no BSDF vertex at bounce 0 (a miss, or an emitter hit).
    Rgb fresnel{0.0F};
    float alpha = 0.0F;  // PathTraceResult::alpha's per-sample estimate; 0 with no ray at all

    // Transport-component breakdown -- see PathTraceResult's doc comment for the bucketing rule.
    Rgb directDiffuse{0.0F};
    Rgb indirectDiffuse{0.0F};
    Rgb directSpecular{0.0F};
    Rgb indirectSpecular{0.0F};
    Rgb refraction{0.0F};
    Rgb directVolume{0.0F};
    Rgb indirectVolume{0.0F};
    Rgb directSubsurface{0.0F};
    Rgb indirectSubsurface{0.0F};

    // OIDN's auxiliary features at the path's first rough vertex: its one-sample directional albedo and its shading normal.
    Rgb denoiseAlbedo{0.0F};
    glm::vec3 denoiseNormal{0.0F};
};

// Each TraceResult member beside the PathTraceResult image its pixel mean writes: one table sums the samples and writes the means.
constexpr std::tuple kTraceLanes{
    std::pair{&TraceResult::radiance, &PathTraceResult::beauty},
    std::pair{&TraceResult::terminationBounce, &PathTraceResult::bounceHeatmap},
    std::pair{&TraceResult::shadowLost, &PathTraceResult::shadowLost},
    std::pair{&TraceResult::shadowUnoccluded, &PathTraceResult::shadowUnoccluded},
    std::pair{&TraceResult::ao, &PathTraceResult::ao},
    std::pair{&TraceResult::fresnel, &PathTraceResult::fresnel},
    std::pair{&TraceResult::alpha, &PathTraceResult::alpha},
    std::pair{&TraceResult::directDiffuse, &PathTraceResult::directDiffuse},
    std::pair{&TraceResult::indirectDiffuse, &PathTraceResult::indirectDiffuse},
    std::pair{&TraceResult::directSpecular, &PathTraceResult::directSpecular},
    std::pair{&TraceResult::indirectSpecular, &PathTraceResult::indirectSpecular},
    std::pair{&TraceResult::refraction, &PathTraceResult::refraction},
    std::pair{&TraceResult::directVolume, &PathTraceResult::directVolume},
    std::pair{&TraceResult::indirectVolume, &PathTraceResult::indirectVolume},
    std::pair{&TraceResult::directSubsurface, &PathTraceResult::directSubsurface},
    std::pair{&TraceResult::indirectSubsurface, &PathTraceResult::indirectSubsurface},
    std::pair{&TraceResult::denoiseAlbedo, &PathTraceResult::denoiseAlbedo},
    std::pair{&TraceResult::denoiseNormal, &PathTraceResult::denoiseNormal},
};

// A lane's mean over n samples: divided, not scaled by fl(1/n), so all-0 or all-1 samples read exactly.
void writeMean(pathtracer::gfx::HdrImage& image, int x, int y, const Rgb& sum, float n) { writeTexel(image, x, y, (sum / n).vec3()); }
void writeMean(pathtracer::gfx::HdrImage& image, int x, int y, const glm::vec3& sum, float n) { writeTexel(image, x, y, sum / n); }
void writeMean(pathtracer::gfx::HdrImage& image, int x, int y, float sum, float n) { writeTexel(image, x, y, sum / n); }
void writeMean(pathtracer::gfx::HdrImage& image, int x, int y, int sum, float n) { writeTexel(image, x, y, static_cast<float>(sum) / n); }

// Routes a contribution into bucket's direct or indirect lane; unset is background radiance, deliberately unbucketed.
void addToBucket(TraceResult& result, std::optional<PathBucket> bucket, const Rgb& contribution, bool isDirect) {
    if (!bucket.has_value()) {
        return;
    }
    switch (*bucket) {
        case PathBucket::Refraction:
            result.refraction += contribution;
            break;
        case PathBucket::Diffuse:
            (isDirect ? result.directDiffuse : result.indirectDiffuse) += contribution;
            break;
        case PathBucket::SpecularReflection:
            (isDirect ? result.directSpecular : result.indirectSpecular) += contribution;
            break;
        case PathBucket::Volume:
            (isDirect ? result.directVolume : result.indirectVolume) += contribution;
            break;
        case PathBucket::Subsurface:
            (isDirect ? result.directSubsurface : result.indirectSubsurface) += contribution;
            break;
    }
}

// The path's MIS after construction: each node weighed from its cached densities, added to Beauty and its route in insertion order.
void evaluatePathGraph(const PathGraph& graph, const SampledWavelengths& lambda, TraceResult& result) {
    for (const LightNode& node : graph.lights) {
        BsdfEval weighted{};
        Rgb rgb{0.0F};
        switch (node.technique) {
            case LightTechnique::BsdfHit: {
                // Power heuristic (Veach 1997): full weight after the camera or a delta draw, neither having a density to balance.
                const PathVertex& parent = graph.vertices[static_cast<std::size_t>(node.parent)];
                const float weight = parent.delta ? 1.0F : powerHeuristic(parent.pdf, node.lightPdf);
                rgb = toScene(node.radiance * weight, lambda);
                break;
            }
            case LightTechnique::Emission:
                rgb = toScene(node.radiance, lambda);
                break;
            case LightTechnique::SurfaceNee:
                weighted = weighOrders({graph.orders.data() + node.ordersBegin, node.ordersEnd - node.ordersBegin}, node.lightPdf);
                rgb = toScene((node.farSide ? weighted.transmission : weighted.diffuse + weighted.specular) * node.radiance, lambda);
                break;
            case LightTechnique::VolumeNee:
                rgb = toScene(node.radiance * (node.phase * powerHeuristic(node.lightPdf, node.phase) / node.lightPdf), lambda);
                break;
        }
        result.radiance += rgb;
        switch (node.route) {
            case Route::Bucket:
                addToBucket(result, node.bucket, rgb, node.direct);
                break;
            case Route::DirectVolume:
                result.directVolume += rgb;
                break;
            case Route::DirectRefraction:
                result.refraction += rgb;
                break;
            case Route::DirectLobes:
                // Bounce 0 splits NEE by the lobe that carried it, not the continuation's lobe.
                result.directDiffuse += toScene(weighted.diffuse * node.radiance, lambda);
                result.directSpecular += toScene(weighted.specular * node.radiance, lambda);
                break;
        }
    }
}

// One interior the path is inside: its instance, its medium and the bulk index its boundary refracts against.
struct MediumEntry {
    int instance;
    HomogeneousMedium medium;
    BulkIndex index;
};

// The innermost interior other than instance's own: the medium beyond that instance's boundary; nullptr is vacuum.
const MediumEntry* enclosingMedium(const std::vector<MediumEntry>& stack, int instance) {
    for (auto entry = stack.rbegin(); entry != stack.rend(); ++entry) {
        if (entry->instance != instance) {
            return &*entry;
        }
    }
    return nullptr;
}

// A medium's index at the path's hero wavelength; vacuum's is 1.
float indexOf(const MediumEntry* entry, std::optional<float> heroNm) {
    return entry != nullptr ? indexAt(entry->index, heroNm) : 1.0F;
}

// What a ray may cross without scattering: an instance whose opacity may fall below 1, or a thin wall that may transmit undeviated.
struct PassThroughScene {
    const EmbreeAccel& accel;
    const std::vector<ShadingTriangle>& shadingTriangles;
    const std::vector<MeshInstance>& instances;
    const std::vector<std::uint8_t>& mayPassThrough;  // per instance
    bool any;  // no instance may: every shadow ray is a plain occlusion query
    const VolumeSet& volumes;
};

template <typename T>
bool isConstant(const MaterialInput<T>& input, T value) {
    const T* constant = std::get_if<T>(&input);
    return constant != nullptr && *constant == value;
}

std::vector<std::uint8_t> passThroughInstances(const std::vector<MeshInstance>& instances) {
    std::vector<std::uint8_t> may(instances.size());
    for (std::size_t i = 0; i < instances.size(); ++i) {
        const Material& material = instances[i].material;
        may[i] = !isConstant(material.geometryOpacity, 1.0F) ||
                 (material.geometryThinWalled && !isConstant(material.transmissionWeight, 0.0F));
    }
    return may;
}

// The vertex's BSDF toward woWorld in its frame, Schussler's microsurface where a map bends it; the coat follows its own input.
NormalMappedBsdf closureAt(const Material& material, const ShadingTriangle& triangle, const ShadingVertex& shading,
                           const OpenPbrInputs<Spectral>& inputs, const ShadingFrame& frame, const glm::vec3& woWorld,
                           const SampledWavelengths& lambda, std::optional<int> heroLane, float ambientIor) {
    const ShadingFrame coatFrame = inputs.coatWeight > 0.0F ? buildShadingFrame(triangle, shading, material.geometryCoatNormal) : frame;
    const bool perturbed = material.geometryNormal.map || material.geometryNormal.height;
    return makeNormalMappedBsdf(inputs, woWorld * frame, perturbed ? std::optional(glm::normalize(shading.normal) * frame) : std::nullopt,
                                lambda.nm, heroLane, coatFrame[2] * frame, coatFrame[0] * frame, ambientIor);
}

// The medium a shadow ray travels in, and its index for the thin walls it may cross: no bulk surface lets it reach another.
struct ShadowMedium {
    const HomogeneousMedium* medium;
    float ior;
};

// The media's transmittance over the whole shadow ray, exact: its interior's in closed form, the VDB media's by regular tracking.
SampledSpectrum mediaTransmittance(const Ray& ray, const PassThroughScene& scene, const ShadowMedium& in, const SampledWavelengths& lambda) {
    const SampledSpectrum interior = in.medium != nullptr ? transmittance(*in.medium, ray.tMax - ray.tMin) : SampledSpectrum(1.0F);
    if (scene.volumes.empty()) {
        return interior;
    }
    return interior * scene.volumes.transmittance(ray, ray.tMin, ray.tMax, lambda);
}

// The share of light a shadow ray carries to its end: (1 - alpha) + alpha * passThrough at each crossable surface, 0 at any other.
SampledSpectrum transmittanceAlong(Ray ray, const PassThroughScene& scene, const SampledWavelengths& lambda, std::optional<int> heroLane,
                                   const ShadowMedium& in, std::uint64_t& shadowRays) {
    if (!scene.any) {
        return scene.accel.occluded(ray) ? SampledSpectrum(0.0F) : mediaTransmittance(ray, scene, in, lambda);
    }
    const Ray whole = ray;
    SampledSpectrum transmittance(1.0F);
    // Closest hit first, so each surface is met in order along the same ray: exact for presence and for an undeviated sheet.
    while (const std::optional<Hit> hit = scene.accel.intersect(ray)) {
        // A disk or sphere is a light's own surface, as opaque as any emitter.
        if (hit->shape >= 0) {
            return SampledSpectrum(0.0F);
        }
        const ShadingTriangle& triangle = scene.shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)];
        const auto instance = static_cast<std::size_t>(triangle.instanceIndex);
        if (!scene.mayPassThrough[instance]) {
            return SampledSpectrum(0.0F);
        }
        const Material& material = scene.instances[instance].material;
        ShadingVertex shading = interpolateShading(triangle, hit->u, hit->v);
        shading.normal = viewAdaptedNormal(shading.normal, geometricNormalOf(triangle), -ray.dir);
        const float opacity = opacityAt(material, shading.uv, {});
        SampledSpectrum crossing(1.0F - opacity);
        if (opacity > 0.0F) {
            const OpenPbrInputs<Spectral> inputs = toSpectral(resolveInputs(material, shading.uv, {}, shading.colour), lambda);
            const ShadingFrame frame = buildShadingFrame(triangle, shading, material.geometryNormal);
            crossing += opacity * passThrough(closureAt(material, triangle, shading, inputs, frame, -ray.dir, lambda, heroLane, in.ior));
        }
        transmittance *= crossing;
        if (isBlack(transmittance)) {
            return transmittance;
        }
        ray.tMin = std::nextafter(hit->t, std::numeric_limits<float>::infinity());
        ++shadowRays;
    }
    return transmittance * mediaTransmittance(whole, scene, in, lambda);
}

// Obscurance through cutouts, Sum T_i alpha_i rho(t_i) + T_end; closest-hit, as rho needs the distance (under 1.3% of frame time).
float obscuranceAlong(Ray ray, const PassThroughScene& scene, float maxDistance) {
    // rho(x) = 1 - (1-x)^2 is the lowest-degree polynomial with rho(0)=0, rho(1)=1 and rho'(1)=0, so nothing steps at D.
    const auto rho = [&](float t) {
        const float k = 1.0F - (t / maxDistance);
        return 1.0F - (k * k);
    };
    float ao = 0.0F;
    float transmittance = 1.0F;
    while (const std::optional<Hit> hit = scene.accel.intersect(ray)) {
        // A disk or sphere is a light's own surface, as opaque as any emitter.
        float opacity = 1.0F;
        if (hit->shape < 0) {
            const ShadingTriangle& triangle = scene.shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)];
            const auto instance = static_cast<std::size_t>(triangle.instanceIndex);
            if (scene.mayPassThrough[instance]) {
                opacity = opacityAt(scene.instances[instance].material, interpolateShading(triangle, hit->u, hit->v).uv, {});
            }
        }
        ao += transmittance * opacity * rho(hit->t);
        transmittance *= 1.0F - opacity;
        if (transmittance == 0.0F) {
            return ao;
        }
        ray.tMin = std::nextafter(hit->t, std::numeric_limits<float>::infinity());
    }
    return ao + transmittance;
}

// dirFootprint: the primary ray's direction offsets across its stratum, filtering bounce 0; a ray cone carries it on over N = 1/invSamples.
TraceResult tracePath(const Ray& primaryRay, const glm::mat2x3& dirFootprint, float invSamples, float pixelSolidAngle,
                       const PassThroughScene& scene, const std::vector<int>& instanceLightIndex, const LightSet& lights,
                       bool showSky, const PathTraceSettings& settings, const SampledWavelengths& lambda,
                       Sampler& sampler, Sampler& volumeSampler, Sampler& walkSampler, glm::vec2 aoSample, glm::vec2 fresnelSample,
                       std::vector<MediumEntry>& mediumStack, PathGraph& graph, pathtracer::debug::RayCounts& __restrict rays) {
    const EmbreeAccel& accel = scene.accel;
    TraceResult result;
    result.ao = 1.0F;  // background is fully unoccluded, matching the polarity in PathTraceResult
    // Radiance per lane at the path's wavelengths; every lane is projected onto scene RGB through the sensor as it is recorded.
    SampledSpectrum throughput(1.0F);
    Ray ray = primaryRay;
    int bounce = 0;
    // The ray cone (Amanatides 1984): width at ray.origin, spread^2 one sample's solid angle; a pinhole's starts at width 0.
    float coneWidth = 0.0F;
    float coneSpread = std::sqrt(glm::length(glm::cross(dirFootprint[0], dirFootprint[1])));
    // Where the ray's current segment began: a ray continued past an absent surface or through a window keeps its origin.
    float segmentStart = 0.0F;
    // Past an absent surface or through a window's undeviated share: the same ray strictly past the hit, with no change to MIS state.
    const auto continuePast = [&](float t) {
        segmentStart = t;
        ray.tMin = std::nextafter(t, std::numeric_limits<float>::infinity());
    };
    std::optional<PathBucket> pathBucket;  // unset until bounce 0 successfully samples a lobe
    // The interiors the path is inside, innermost last: a bulk crossing removes its instance if present, else pushes it.
    mediumStack.clear();
    // The lane this path committed to at a dispersive interface, and its wavelength; unset, every lane follows the path.
    std::optional<int> heroLane;
    std::optional<float> heroNm;
    // The instance whose walk bounce 0 entered, -1 once left: NEE at that walk's boundary is direct subsurface light.
    int primaryWalk = -1;
    // The ray leaving that walk, until its next vertex: its emitter or sky hit is direct subsurface light too.
    bool leftPrimaryWalk = false;
    // A hit ending a segment is direct when its segment leaves bounce 0's vertex, or for Subsurface the primary walk.
    const auto directHit = [&] { return pathBucket == PathBucket::Subsurface ? leftPrimaryWalk : bounce == 1; };
    // Shadow's lanes take a bounce-0 light estimator without its MIS weight: alone it estimates all direct light, occluded or not.
    const auto recordShadow = [&](const SampledSpectrum& direct, const SampledSpectrum& visibility) {
        result.shadowUnoccluded = luminance(toScene(direct, lambda));
        result.shadowLost = luminance(toScene(direct * (1.0F - visibility), lambda));
    };
    // Techniques append to the graph and weigh nothing; the camera is vertex 0, a delta with no density for a hit to balance.
    graph.clear();
    graph.vertices.push_back({glm::vec3(0.0F), 0.0F, true});
    // The vertex whose drawn continuation the current ray is: a hit's MIS weighs against NEE from its shading position.
    int misParent = 0;
    const auto addVertex = [&](const glm::vec3& position, float pdf, bool delta) {
        graph.vertices.push_back({position, pdf, delta});
        return static_cast<int>(graph.vertices.size()) - 1;
    };
    // A light the continuation met: its density as NEE from misParent would have drawn it, needed only when misParent has a density.
    const auto addHit = [&](LightTechnique technique, const SampledSpectrum& radiance, float lightPdf) {
        graph.lights.push_back({technique, misParent, radiance, lightPdf, 0.0F, false, 0, 0, Route::Bucket, pathBucket, directHit()});
    };
    const auto parentDelta = [&] { return graph.vertices[static_cast<std::size_t>(misParent)].delta; };

    // Denoiser features (OIDN): followed through deltas and sub-pixel lobes to the first rough vertex, apart from `throughput`.
    SampledSpectrum featureWeight(1.0F);
    float featureSurvival = 1.0F;  // 1 / Russian roulette's continuation odds while the chain is open, so both features stay unbiased
    bool featurePending = settings.denoiseFeatures;
    // Closes the chain at a surface: no ray, no draw and no Beauty float changes, so the lanes ride along at no cost to the image.
    const auto recordFeatures = [&](const SampledSpectrum& albedo, const glm::vec3& normal) {
        result.denoiseAlbedo = featureSurvival * reflectanceToScene(albedo, lambda);
        result.denoiseNormal = featureSurvival * normal;
        featurePending = false;
    };
    // Russian roulette from its start bounce, and at every walk event, the walk's sole termination as one bounce: false ends the path.
    const auto survivesRoulette = [&](bool walkEvent) {
        if (!walkEvent && bounce < settings.russianRouletteStartBounce) {
            return true;
        }
        const float continueProb =
            std::clamp(maxChannel(throughput), settings.rrMinProb, settings.rrMaxProb);
        if (sampler.next1D() >= continueProb) {
            return false;
        }
        throughput /= continueProb;
        if (featurePending) {
            featureSurvival /= continueProb;
        }
        return true;
    };

    // bounce 0 always traces: maxBounces counts secondary bounces, so maxBounces==0 is direct lighting with no continuation rays.
    for (; bounce <= settings.maxBounces + 1; ++bounce) {
        (bounce == 0 ? rays.primary : rays.bounce) += 1;
        const std::optional<Hit> hit = accel.intersect(ray);

        // The interior this segment ran through: a scattering one draws its first collision in closed form, a clear one attenuates.
        const HomogeneousMedium* medium = mediumStack.empty() ? nullptr : &mediumStack.back().medium;
        const bool inWalk = medium != nullptr && scatters(*medium);
        const float segmentEnd = hit.has_value() ? hit->t : ray.tMax;
        SampledSpectrum odds(0.0F);
        std::optional<float> walkCollision;
        if (inWalk) {
            odds = channelOdds(*medium, throughput);
            const float t = segmentStart + sampleCollision(*medium, odds, volumeSampler.next2D());
            if (t < segmentEnd) {
                walkCollision = t;
            }
        }

        // Free flight through the VDB media to that collision or the segment's end: the earlier of two independent processes wins.
        std::optional<FreeFlight> flight;
        if (!scene.volumes.empty()) {
            flight = scene.volumes.sampleFreeFlight(ray, segmentStart, walkCollision.value_or(segmentEnd), throughput, lambda, volumeSampler);
            throughput *= flight->weight;
            if (featurePending) {
                featureWeight *= flight->weight;
            }
        }
        const bool scattered = flight.has_value() && flight->event == FreeFlightEvent::Scattered;
        const bool walkScattered = walkCollision.has_value() && !scattered;
        if (bounce == 0) {
            // E[mean(1 - w) 1{escape}] = 1 - mean(T): opacity projected, white's E[projection] = 1 a control variate; 0 exactly when clear.
            result.alpha =
                hit.has_value() || showSky || scattered ? 1.0F : channelMean(reflectanceToScene(1.0F - throughput, lambda));
        }

        // The interior's weight to the event that ended the flight, or its own collision's; a miss is unbounded, t = inf.
        if (medium != nullptr) {
            const float travelled = (scattered       ? flight->t
                                     : walkScattered ? *walkCollision
                                     : hit.has_value() ? hit->t
                                                       : std::numeric_limits<float>::infinity()) -
                                    segmentStart;
            const SampledSpectrum surviving = transmittance(*medium, travelled);
            const SampledSpectrum weight =
                !inWalk         ? surviving
                : walkScattered ? collisionWeight(odds, surviving, medium->scattering, medium->absorption + medium->scattering)
                                : passWeight(odds, surviving);
            throughput *= weight;
            // A walk's weights are its albedo estimate, so the feature chain carries them; a clear interior's colour it never did.
            if (featurePending && inWalk) {
                featureWeight *= weight;
            }
        }

        if (scattered) {
            // The depth cap's extra iteration only collects the final ray's miss or emitter hit, which this scattering stops short of.
            if (bounce > settings.maxBounces) {
                break;
            }
            const glm::vec3 position = ray.origin + (flight->t * ray.dir);
            const glm::vec3 dir = glm::normalize(ray.dir);
            if (bounce == 0) {
                pathBucket = PathBucket::Volume;
            }
            // The phase function integrates to 1, so the albedo estimate is the tracking weight already carried; no surface, no normal.
            if (featurePending) {
                recordFeatures(featureWeight, glm::vec3(0.0F));
            }
            const int vertex = addVertex(position, 0.0F, false);
            // NEE from the scattering point, against the phase-sampled continuation by the power heuristic on directional pdfs.
            if (const std::optional<LightSample> lightSample = lights.sample(position, sampler, lambda)) {
                const float phase = henyeyGreenstein(glm::dot(dir, lightSample->direction), flight->g);
                const Ray shadowRay{position, lightSample->direction, 0.0F, lightSample->distance * (1.0F - kShadowDistanceEpsilon)};
                ++rays.shadow;
                const ShadowMedium in{medium, indexOf(mediumStack.empty() ? nullptr : &mediumStack.back(), heroNm)};
                const SampledSpectrum visibility = transmittanceAlong(shadowRay, scene, lambda, heroLane, in, rays.shadow);
                if (bounce == 0) {
                    recordShadow(throughput * lightSample->radiance * (phase / lightSample->pdf), visibility);
                }
                if (!isBlack(visibility)) {
                    graph.lights.push_back({LightTechnique::VolumeNee, vertex, visibility * throughput * lightSample->radiance,
                                            lightSample->pdf, phase, false, 0, 0, bounce == 0 ? Route::DirectVolume : Route::Bucket,
                                            pathBucket, false});
                }
            }
            // The phase sample weighs f/pdf = 1; its pdf is what this ray's miss or emitter hit is MIS-weighted against.
            const glm::vec3 wi = sampleHenyeyGreenstein(dir, flight->g, sampler.next2D());
            const float phasePdf = henyeyGreenstein(glm::dot(dir, wi), flight->g);
            graph.vertices[static_cast<std::size_t>(vertex)].pdf = phasePdf;
            misParent = vertex;
            leftPrimaryWalk = false;
            if (!survivesRoulette(false)) {
                break;
            }
            coneWidth += coneSpread * flight->t;
            coneSpread = lobeSpread(coneSpread, invSamples / phasePdf);
            ray = Ray{position, wi, 0.0F, std::numeric_limits<float>::max()};
            segmentStart = 0.0F;
            continue;
        }

        // A walk collision, past an entry vertex as every path starts outside: its shadow ray would meet the bulk boundary, so no NEE.
        if (walkScattered) {
            // The depth cap's extra iteration only collects the final ray's miss or emitter hit, which this collision stops short of.
            if (bounce > settings.maxBounces) {
                break;
            }
            const glm::vec3 position = ray.origin + (*walkCollision * ray.dir);
            // The phase sample weighs f/pdf = 1; with no NEE beside it, its miss or emitter hit takes full weight.
            const glm::vec3 dir = glm::normalize(ray.dir);
            const glm::vec3 wi = sampleHenyeyGreenstein(dir, medium->anisotropy, sampler.next2D());
            misParent = addVertex(position, 0.0F, true);
            if (!survivesRoulette(true)) {
                break;
            }
            coneWidth += coneSpread * *walkCollision;
            coneSpread = lobeSpread(coneSpread, invSamples / henyeyGreenstein(glm::dot(dir, wi), medium->anisotropy));
            ray = Ray{position, wi, 0.0F, std::numeric_limits<float>::max()};
            segmentStart = 0.0F;
            // The walk is one bounce, counted at its entry: its collisions consume none of maxBounces.
            --bounce;
            continue;
        }

        if (!hit.has_value()) {
            // showSky gates only the primary ray's miss; indirect bounces and NEE always sample real environment radiance.
            if (bounce == 0 && !showSky) {
                break;
            }
            // Filtered only at full MIS weight: NEE reads the sky at a point, and the strategies MIS blends must estimate one integrand.
            const SampledSpectrum envRadiance =
                bounce == 0          ? lights.environmentRadiance(ray.dir, dirFootprint, lambda)
                : parentDelta()      ? lights.environmentRadiance(ray.dir, spreadAcross(glm::normalize(ray.dir), coneSpread), lambda)
                                     : lights.environmentRadiance(ray.dir, lambda);
            addHit(LightTechnique::BsdfHit, throughput * envRadiance, parentDelta() ? 0.0F : lights.pdfEnvironment(ray.dir));
            break;
        }

        // An emitter hit, every light's surface in the BVH: Le MIS-weighted like a miss, then terminate. Only lights have shapes.
        const int hitInstance = hit->shape >= 0 ? accel.shapes()[static_cast<std::size_t>(hit->shape)].instanceIndex
                                                : scene.shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)].instanceIndex;
        const int lightIndex = instanceLightIndex[static_cast<std::size_t>(hitInstance)];
        if (lightIndex >= 0) {
            const glm::vec3 point = ray.origin + (hit->t * ray.dir);
            const glm::vec3 lightNormal = hit->shape >= 0 ? shadingAt(accel.shapes()[static_cast<std::size_t>(hit->shape)], point).normal
                                                          : geometricNormalOf(scene.shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)]);
            // An emitter is a surface with no BSDF: its normal, and no albedo.
            if (featurePending) {
                recordFeatures(SampledSpectrum(0.0F), lightNormal);
            }
            const SampledSpectrum emitted = lights.radianceToward(lightIndex, ray.dir, lightNormal, lambda);
            const glm::vec3& from = graph.vertices[static_cast<std::size_t>(misParent)].position;
            addHit(LightTechnique::BsdfHit, throughput * emitted, parentDelta() ? 0.0F : lights.pdfArea(lightIndex, from, point, lightNormal));
            break;
        }

        const ShadingTriangle& triangle = scene.shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)];
        const auto instance = static_cast<std::size_t>(triangle.instanceIndex);
        const Material& material = scene.instances[instance].material;
        // Bounce 0 keeps the stratum's anisotropic differentials; later vertices read the cone, |width| across the ray at the hit.
        const float hitWidth = coneWidth + (coneSpread * hit->t);
        const pathtracer::gfx::TextureFootprint footprint =
            bounce == 0 ? primaryHitFootprint(triangle, ray, hit->t, dirFootprint) : coneHitFootprint(triangle, ray, hitWidth);
        // True flat plane normal, for light-leak rejection and ray-origin offsets: both need geometry, not the shading normal.
        const glm::vec3 geoNormal = geometricNormalOf(triangle);
        ShadingVertex shading = interpolateShading(triangle, hit->u, hit->v);
        shading.normal = viewAdaptedNormal(shading.normal, geoNormal, -ray.dir);

        // OpenPBR's presence, mix(ambient, surface, alpha) (spec eq. 23): absent, the same ray continues, no bounce and no vertex.
        if (scene.mayPassThrough[instance]) {
            const float opacity = opacityAt(material, shading.uv, footprint);
            if (opacity < 1.0F && sampler.next1D() >= opacity) {
                continuePast(hit->t);
                --bounce;
                continue;
            }
        }

        // Surface emission, absent from LightSet, so the hit takes MIS weight 1. Before the depth cap, as an emitter hit.
        const SampledSpectrum emitted = emittedRadiance(material, triangle, shading, -ray.dir, footprint, lambda);
        if (!isBlack(emitted)) {
            addHit(LightTechnique::Emission, throughput * emitted, 0.0F);
        }

        // Depth cap. The extra iteration exists only so the final BSDF ray can collect its MIS-weighted miss or emitter hit.
        const bool capped = bounce > settings.maxBounces;
        if (capped && !scene.mayPassThrough[instance]) {
            break;
        }

        const OpenPbrInputs<Constant> authored = resolveInputs(material, shading.uv, footprint, shading.colour);
        const OpenPbrInputs<Spectral> inputs = toSpectral(authored, lambda);
        // The medium beyond this boundary as seen from its inside: its index is the ambient one this interface refracts against.
        const MediumEntry* ambient = enclosingMedium(mediumStack, triangle.instanceIndex);
        // Commit to one lane at the first dispersive interface (Wilkie et al. 2014's limit at a delta); rrMinProb keeps zero sums alive.
        if (!heroLane.has_value() && isDispersive(authored, ambient != nullptr && disperses(ambient->index))) {
            const float sum = channelSum(throughput);
            if (sum > 0.0F) {
                const float u = sampler.next1D() * sum;
                int lane = 0;
                for (float below = throughput[0]; lane < kSpectrumSamples - 1 && u >= below; below += throughput[++lane]) {
                }
                // The one-sample spectral estimator: the drawn lane over its selection odds throughput[lane] / sum.
                if (featurePending) {
                    const float feature = featureWeight[lane] * sum / throughput[lane];
                    featureWeight = SampledSpectrum(0.0F);
                    featureWeight[lane] = feature;
                }
                throughput = SampledSpectrum(0.0F);
                throughput[lane] = sum;
                heroLane = lane;
                heroNm = lambda.nm[lane];
            }
        }

        const ShadingFrame frame = buildShadingFrame(triangle, shading, material.geometryNormal);
        const glm::vec3 woWorld = -ray.dir;

        // Built once for both estimators below: the continuation draw and NEE's evaluation share every wo-side lookup it holds.
        const float ambientIor = indexOf(ambient, heroNm);
        const NormalMappedBsdf closure = closureAt(material, triangle, shading, inputs, frame, woWorld, lambda, heroLane, ambientIor);

        // Past the cap a window still passes the final ray on to its emitter, by its expected undeviated share, as NEE's shadow rays do.
        if (capped) {
            const SampledSpectrum through = passThrough(closure);
            if (isBlack(through)) {
                break;
            }
            throughput *= through;
            continuePast(hit->t);
            --bounce;
            continue;
        }
        leftPrimaryWalk = false;

        if (bounce == 0) {
            // The Fresnel the microfacet lobe evaluates here, one VNDF draw per sample, on its own stream. No ray and no BVH query.
            result.fresnel = reflectanceToScene(fresnelAtMicrofacet(closure, fresnelSample), lambda);
            // Obscurance (Zhukov 1998; Iones 2003): sampling at pdf = cos/pi cancels both factors, so the estimator is the mean of rho.
            const bool frontSide = glm::dot(geoNormal, woWorld) > 0.0F;
            const glm::vec3 aoDir =
                frame * sampleCosineHemisphere(aoSample) * (frontSide ? 1.0F : -1.0F);
            const glm::vec3 aoOrigin = spawnOrigin(triangle, hit->u, hit->v, geoNormal, frontSide);
            ++rays.ao;
            result.ao = obscuranceAlong(Ray{aoOrigin, aoDir, 0.0F, settings.aoMaxDistance}, scene, settings.aoMaxDistance);
        }

        // A zero BSDF, emission alone, carries no light onward: the path ends here, its AO and Fresnel lanes already written.
        if (!scatters(closure)) {
            if (featurePending) {
                recordFeatures(SampledSpectrum(0.0F), frame[2]);
            }
            break;
        }

        // A failed sample must not skip the NEE block: the two are independent estimators, sharing only this vertex's params and frame.
        std::optional<WalkVertex> revisit;  // where the draw's walk first returned to the facet: NEE resumes its later orders there
        const std::optional<BsdfSample> sample = sampleBsdf(closure, sampler, walkSampler, revisit);

        // One side rule for NEE and continuation alike: reflection lobes count on wo's geometric side, transmission lobes across it.
        const bool woAbove = glm::dot(woWorld, geoNormal) > 0.0F;
        const glm::vec3 wiWorld = !sample.has_value() ? glm::vec3(0.0F) : sample->passThrough ? ray.dir : frame * sample->wiLocal;
        // The side rule NEE applied: Schussler's microsurface keeps lobes on omega_g's sides, this the flat face's (his Fig. 18 residual).
        const bool crosses = sample.has_value() && (glm::dot(wiWorld, geoNormal) > 0.0F) != woAbove;
        // A bulk boundary leaves the interior the path is inside, or enters its own; the medium on each side follows.
        const bool bulk = hasInterior(inputs);
        const bool inside = std::any_of(mediumStack.begin(), mediumStack.end(),
                                        [&](const MediumEntry& entry) { return entry.instance == triangle.instanceIndex; });
        const std::optional<HomogeneousMedium> interior = bulk && !inside ? std::optional(interiorMedium(inputs)) : std::nullopt;
        const bool entersWalk = crosses && interior.has_value() && scatters(*interior);
        // A walk's boundary reflection back into its scattering interior is part of the same one bounce.
        const bool walkReflects = !crosses && inWalk && mediumStack.back().instance == triangle.instanceIndex;

        // Bucket assignment: bounce 0 sets it from scratch, a later transmission overrides it to Refraction unless it is Subsurface.
        if (sample.has_value()) {
            if (bounce == 0) {
                pathBucket = entersWalk                                ? PathBucket::Subsurface
                             : sample->type == LobeType::Transmission ? PathBucket::Refraction
                             : sample->type == LobeType::Diffuse      ? PathBucket::Diffuse
                                                                      : PathBucket::SpecularReflection;
                primaryWalk = entersWalk ? triangle.instanceIndex : -1;
            } else if (sample->type == LobeType::Transmission && pathBucket != PathBucket::Subsurface) {
                pathBucket = PathBucket::Refraction;
            }
        }

        // A passed-through or failed draw continues no density, so the vertex is NEE's parent only; a drawn one is the next hit's too.
        const bool continues = sample.has_value() && !sample->passThrough;
        const int vertex = addVertex(shading.position, continues ? sample->pdf : 0.0F, !continues || sample->delta);

        // NEE: sample a light, evaluate the BSDF toward it, append it unweighted if unoccluded. Fired whatever lobe `sample` drew.
        const std::optional<LightSample> lightSample = lights.sample(shading.position, sampler, lambda);
        if (lightSample.has_value()) {
            const float geoCos = glm::dot(lightSample->direction, geoNormal);
            // A light across the surface reaches wo only through a refracting interface: only then is the far side worth a shadow ray.
            const bool farSide = (geoCos > 0.0F) != woAbove;
            if (!farSide || transmits(closure)) {
                const glm::vec3 wiLocalLight = lightSample->direction * frame;
                // One evaluation for the value and per-lobe split, each lobe cosine-weighted about its own normal, its orders kept for MIS.
                const std::size_t nodes = graph.lights.size();
                const std::size_t ordersBegin = graph.orders.size();
                const BsdfEval value = evaluateForLight(closure, wiLocalLight, revisit, walkSampler, graph.orders);
                const SampledSpectrum bsdfValue = farSide ? value.transmission : value.diffuse + value.specular;
                if (maxChannel(bsdfValue) > 0.0F) {
                    const glm::vec3 shadowOrigin = spawnOrigin(triangle, hit->u, hit->v, geoNormal, geoCos > 0.0F);
                    // pbrt's SpawnRayTo: re-formed from the offset origin to the sampled point, which a back-off along wi cannot reach.
                    const bool finiteLight = lightSample->distance < std::numeric_limits<float>::max();
                    const glm::vec3 toLight =
                        finiteLight ? (shading.position - shadowOrigin) +
                                           (lightSample->direction * lightSample->distance)
                                     : lightSample->direction;  // the environment's point is at infinity
                    const float shadowDistance = finiteLight ? glm::length(toLight) : lightSample->distance;
                    const Ray shadowRay{shadowOrigin, finiteLight ? toLight / shadowDistance : toLight,
                                         0.0F, shadowDistance * (1.0F - kShadowDistanceEpsilon)};
                    ++rays.shadow;
                    // Across a bulk boundary the shadow ray is in the interior it enters or the one enclosing it; else in this side's.
                    const MediumEntry* here = mediumStack.empty() ? nullptr : &mediumStack.back();
                    const ShadowMedium in = !farSide || !bulk ? ShadowMedium{medium, indexOf(here, heroNm)}
                                            : inside          ? ShadowMedium{ambient != nullptr ? &ambient->medium : nullptr, ambientIor}
                                                              : ShadowMedium{&*interior, indexAt(bulkIndexOf(inputs), heroNm)};
                    const SampledSpectrum visibility = transmittanceAlong(shadowRay, scene, lambda, heroLane, in, rays.shadow);
                    if (bounce == 0) {
                        recordShadow(bsdfValue * throughput * lightSample->radiance / lightSample->pdf, visibility);
                    }
                    if (!isBlack(visibility)) {
                        // Deeper bounces keep the sticky bucket; only the primary walk's boundary takes direct light past bounce 0.
                        const Route route = bounce > 0 ? Route::Bucket : farSide ? Route::DirectRefraction : Route::DirectLobes;
                        graph.lights.push_back({LightTechnique::SurfaceNee, vertex, visibility * throughput * lightSample->radiance / lightSample->pdf,
                                                lightSample->pdf, 0.0F, farSide, ordersBegin, graph.orders.size(), route, pathBucket,
                                                primaryWalk >= 0});
                    }
                }
                // Orders no node took: dropped, so the buffer holds only what the evaluation reads.
                if (graph.lights.size() == nodes) {
                    graph.orders.resize(ordersBegin);
                }
            }
        }

        // No continuation direction could be sampled: terminate, NEE having already taken this vertex's direct lighting.
        if (!sample.has_value()) {
            // A failed draw is the albedo estimator's zero, not a skipped sample; the surface's normal stands.
            if (featurePending) {
                recordFeatures(SampledSpectrum(0.0F), frame[2]);
            }
            break;
        }

        // A window's undeviated transmission is a null vertex: MIS keeps the last scattering vertex, whose density this ray still carries.
        if (continues) {
            misParent = vertex;
        }

        const SampledSpectrum weight = crosses ? sample->transmitWeight : sample->throughputWeight - sample->transmitWeight;
        // E[f cos / pdf] is the albedo; a lobe denser than 1/pixel solid angle stays open as a delta, as does a walk until its exit.
        if (featurePending) {
            if (sample->delta || sample->lobePdf * pixelSolidAngle >= 1.0F || entersWalk || walkReflects) {
                featureWeight *= weight;
            } else {
                recordFeatures(featureWeight * weight, frame[2]);
            }
        }
        if (isBlack(weight)) {
            break;
        }
        // A bulk ray enters or leaves its interior exactly when it crosses the surface; a thin wall has no interior to enter.
        if (crosses && bulk) {
            if (inside) {
                std::erase_if(mediumStack, [&](const MediumEntry& entry) { return entry.instance == triangle.instanceIndex; });
                if (primaryWalk == triangle.instanceIndex) {
                    primaryWalk = -1;
                    leftPrimaryWalk = true;
                }
            } else {
                mediumStack.push_back({triangle.instanceIndex, *interior, bulkIndexOf(inputs)});
            }
        }

        throughput *= weight;

        if (!survivesRoulette(walkReflects)) {
            break;
        }
        // A boundary reflection inside a walk is part of that walk's one bounce.
        if (walkReflects) {
            --bounce;
        }

        if (sample->passThrough) {
            continuePast(hit->t);
            continue;
        }

        // The cone through the shading normal facing the ray, curvature signed on the ray's side: convex spreads it, concave focuses it.
        const glm::vec3 incident = glm::normalize(ray.dir);
        const glm::vec3 facing = glm::dot(incident, frame[2]) < 0.0F ? frame[2] : -frame[2];
        const float curvature = glm::dot(incident, shading.normal) < 0.0F ? meanCurvature(triangle) : -meanCurvature(triangle);
        coneSpread = scatteredSpread(coneSpread, hitWidth, incident, wiWorld, facing, curvature, crosses ? bendRatio(closure) : 1.0F,
                                     sample->delta ? 0.0F : invSamples / sample->pdf);
        coneWidth = hitWidth;

        // On wi's side for every lobe: an unmirrored terminator lift desynced the medium stack on 0.93% of cornell's glass entries.
        const glm::vec3 origin = spawnOrigin(triangle, hit->u, hit->v, geoNormal, glm::dot(wiWorld, geoNormal) > 0.0F);
        ray = Ray{origin, wiWorld, 0.0F, std::numeric_limits<float>::max()};
        segmentStart = 0.0F;
    }

    result.terminationBounce = bounce;
    evaluatePathGraph(graph, lambda, result);
    return result;
}

}  // namespace

PathTraceResult makePathTraceResult(int width, int height, int replicates) {
    PathTraceResult result;
    for (int i = 0; i < static_cast<int>(pathtracer::debug::AovId::Count); ++i) {
        const auto aov = static_cast<pathtracer::debug::AovId>(i);
        if (const pathtracer::debug::PathTracedLane lane = pathtracer::debug::pathTracedLane(aov)) {
            result.*lane = pathtracer::gfx::makeImage(width, height, pathtracer::debug::aovChannels(aov));
        }
    }
    for (const pathtracer::debug::PathTracedLane lane : pathtracer::debug::kDenoiseFeatureLanes) {
        result.*lane = pathtracer::gfx::makeImage(width, height, pathtracer::gfx::kRgbChannels);
    }
    for (const pathtracer::debug::PathTracedLane lane : pathtracer::debug::kShadowLanes) {
        result.*lane = pathtracer::gfx::makeImage(width, height, pathtracer::gfx::kScalarChannels);
    }
    result.replicateColour.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * replicateStride(replicates), 0.0F);
    result.replicates = replicates;
    result.texelSamples.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
    return result;
}

void renderPathTraced(const Camera& camera, const EmbreeAccel& accel,
                       const std::vector<ShadingTriangle>& shadingTriangles,
                       const std::vector<MeshInstance>& instances,
                       const std::vector<int>& instanceLightIndex, const LightSet& lights, const VolumeSet& volumes,
                       int width, int height, std::span<const std::uint32_t> texels, bool showSky, const PathTraceSettings& settings,
                       std::uint32_t scrambleSeed, int sampleBase, int sampleCount,
                       const std::atomic<std::uint64_t>& generation,
                       std::uint64_t requestedGeneration, ThreadPool& threadPool,
                       pathtracer::debug::PassStats& stats, PathTraceResult& out) {
    const pathtracer::gfx::PixelLattice lattice = camera.pixelLattice();
    // The film in column pitches: its ratio, not width / height, is the aspect once rows sit closer than columns.
    const glm::vec2 frame = pathtracer::gfx::frameExtent(lattice, width, height);
    const float aspect = frame.x / frame.y;
    // Constant for the whole pass, so built once: the aspect-taking primaryRay rebuilds it on every one of millions of rays.
    const Camera::ViewBasis basis = camera.viewBasis(aspect);
    out.wrapsHorizontally = wrapsHorizontally(basis.lens.projection);
    out.pixelLattice = lattice;
    const float pixelSide = pathtracer::gfx::pixelSide(lattice);
    // Samples in a pixel's estimate; unbounded N as Sampler's.
    const auto estimateSamples = static_cast<float>(std::max(sampleCount, sampleBase + settings.samplesPerPass));
    // Sample spacing pixelSide/sqrt(N) (pbrt-v4's differential scale), so prefilter blur vanishes as N grows.
    const float strataPerAxis = std::sqrt(estimateSamples) / pixelSide;
    // d(ndc) across that spacing: two NDC units span the film, and +Y flips because row 0 is the top.
    const glm::mat2 ndcPerStratum(2.0F / (frame.x * strataPerAxis), 0.0F, 0.0F, -2.0F / (frame.y * strataPerAxis));
    // A pixel's solid angle is what a mirror lobe must undercut to stay open in the feature chain.
    const glm::mat2 ndcPerPixel = pathtracer::gfx::ndcPerPixel(lattice, frame);
    // A compound-eye facet's angular sensitivity replaces the film filter: each sample is drawn from its lobe about the pixel's axis.
    const float acceptance = camera.acceptanceAngleDegrees();
    const float kappa = acceptance > 0.0F ? acceptanceConcentration(acceptance) : 0.0F;
    const float lobeSolidAngle = kappa > 0.0F ? acceptanceSolidAngle(kappa) : 0.0F;
    // Replicated RQMC: pass p draws from replicate p mod K at its position p div K, so each replicate is a prefix of its own sequence.
    const int pass = sampleBase / settings.samplesPerPass;
    const std::uint32_t seed = replicateSeed(scrambleSeed, pass % settings.replicates);
    const int replicateBase = (pass / settings.replicates) * settings.samplesPerPass;
    const int replicateCount = sampleCount > 0 ? (sampleCount + settings.replicates - 1) / settings.replicates : sampleCount;
    // Chunks of the list, not of the frame: as texels converge the work left stays split over the pool, wherever it lies.
    const std::vector<int> chunkBounds = pathTraceChunkBounds(static_cast<int>(texels.size()), threadPool.threadCount());
    const std::vector<std::uint8_t> mayPassThrough = passThroughInstances(instances);
    const PassThroughScene scene{accel, shadingTriangles, instances, mayPassThrough,
                                 std::any_of(mayPassThrough.begin(), mayPassThrough.end(), [](std::uint8_t may) { return may != 0; }),
                                 volumes};
    // Each instance is in a path's medium stack at most once, so the instances that may bound an interior are its exact capacity.
    const auto interiors = static_cast<std::size_t>(std::count_if(instances.begin(), instances.end(), [](const MeshInstance& instance) {
        const Material& material = instance.material;
        return !material.geometryThinWalled &&
               (!isConstant(material.transmissionWeight, 0.0F) || !isConstant(material.subsurfaceWeight, 0.0F));
    }));

    const auto samples = static_cast<float>(settings.samplesPerPass);
    // One worker owns one chunk: filter importance sampling keeps each sample in its own pixel's estimate, so chunks share no work.
    const auto renderChunk = [&](int chunkIndex) {
        const auto chunkBegin = static_cast<std::size_t>(chunkBounds[static_cast<std::size_t>(chunkIndex)]);
        const auto chunkEnd = static_cast<std::size_t>(chunkBounds[static_cast<std::size_t>(chunkIndex) + 1]);

        // Stack-local, not thread_local: zeroed by construction, so the reset boundary is the chunk boundary with no bookkeeping.
        pathtracer::debug::RayCounts chunkRays;
        // Reserved once per chunk: a path's pushes never exceed it, so no path allocates.
        std::vector<MediumEntry> mediumStack;
        mediumStack.reserve(interiors);
        // Reused by every path in the chunk: its buffers keep their capacity, so a path allocates only past the chunk's deepest.
        PathGraph graph;

        for (std::size_t i = chunkBegin; i < chunkEnd; ++i) {
            // Checked per texel, so a stale pass stops within one path however large the chunk; the caller discards the whole pass.
            if (generation.load(std::memory_order_relaxed) != requestedGeneration) {
                stats.addCancelledChunk(chunkRays);  // its rays were paid for, so a cancelled pass must not read as a short one
                return;
            }
            const int x = static_cast<int>(texels[i] % static_cast<std::uint32_t>(width));
            const int y = static_cast<int>(texels[i] / static_cast<std::uint32_t>(width));
            const glm::vec2 centre = pathtracer::gfx::pixelCentre(lattice, x, y);
            // The facet's optical axis, once per pixel; nullopt past a fisheye's image circle, where the facet sees nothing.
            const glm::vec2 centreNdc = pathtracer::gfx::filmToNdc(centre, frame);
            const std::optional<Camera::RayDifferential> axis =
                kappa > 0.0F ? camera.primaryRayDifferential(basis, centreNdc.x, centreNdc.y) : std::nullopt;
            TraceResult sum;
            for (int s = 0; s < settings.samplesPerPass; ++s) {
                // replicateBase + s is this sample's position in its replicate's sequence: it must advance across passes to stratify.
                const int sampleIndex = replicateBase + s;
                Sampler sampler(x, y, sampleIndex, replicateCount, seed);
                // Dimension set 0 places the sample: on film through the filter, or on the sphere through the facet's lobe.
                const glm::vec2 placement = sampler.next2D();
                std::optional<Camera::RayDifferential> primary;
                glm::mat2x3 dirFootprint(0.0F);
                if (kappa > 0.0F) {
                    if (axis) {
                        const AcceptanceSample facet = sampleAcceptance(axis->ray.dir, kappa, placement);
                        primary = axis;
                        primary->ray.dir = facet.dir;
                        // Each sample's share of the lobe, 1 / (N pdf) sr (Suykens & Willems 2001), at most the whole sphere.
                        dirFootprint = spreadAcross(facet.dir, lobeSpread(0.0F, 1.0F / (estimateSamples * facet.pdf)));
                    }
                } else {
                    // Reaches past the frame: rectilinear extrapolates, fisheye is rayless past its circle, lat-long wraps seam, pole.
                    const glm::vec2 ndc = pathtracer::gfx::filmToNdc(centre + sampleFilmOffset(lattice, placement), frame);
                    primary = camera.primaryRayDifferential(basis, ndc.x, ndc.y);
                    dirFootprint = primary ? primary->dirPerNdc * ndcPerStratum : glm::mat2x3(0.0F);
                }
                // AO and Fresnel draw from their own stream: taking dimensions from `sampler` would shift every later dimension.
                Sampler aoSampler = sampler.stream(kAoSeedOffset);
                const glm::vec2 aoSample = aoSampler.next2D();
                Sampler fresnelSampler = sampler.stream(kFresnelSeedOffset);
                const glm::vec2 fresnelSample = fresnelSampler.next2D();
                Sampler volumeSampler = sampler.stream(kVolumeSeedOffset);
                Sampler walkSampler = sampler.stream(kWalkSeedOffset);
                // The hero's CDF position in the scene's wavelength density, stratified over the pixel's samples like any first dimension.
                Sampler wavelengthSampler = sampler.stream(kWavelengthSeedOffset);
                const SampledWavelengths lambda = sampleWavelengths(wavelengthSampler.next1D(), lights.wavelengthPdf());
                // The pixel's footprint on the unit sphere, read only by the feature chain where a ray exists; a facet's is its lobe's.
                const auto footprintSolidAngle = [&] {
                    return kappa > 0.0F ? lobeSolidAngle : pixelSolidAngle(primary->dirPerNdc, ndcPerPixel);
                };
                // Nullopt is a fisheye sample outside the image circle: no ray exists, so every lane reads zero for it.
                const TraceResult trace =
                    primary ? tracePath(primary->ray, dirFootprint, 1.0F / estimateSamples,
                                        settings.denoiseFeatures ? footprintSolidAngle() : 0.0F, scene, instanceLightIndex, lights,
                                        showSky, settings, lambda, sampler, volumeSampler, walkSampler, aoSample, fresnelSample, mediumStack,
                                        graph, chunkRays)
                            : TraceResult{};
                std::apply([&](const auto&... lane) { ((sum.*lane.first += trace.*lane.first), ...); }, kTraceLanes);
            }
            // Unit weights make the estimate the plain mean.
            std::apply([&](const auto&... lane) { (writeMean(out.*lane.second, x, y, sum.*lane.first, samples), ...); }, kTraceLanes);
        }

        stats.addChunk(chunkRays);
    };

    threadPool.parallelFor(static_cast<int>(chunkBounds.size()) - 1, renderChunk);
    stats.setCoreUse(threadPool.lastCoreUse());
}

}  // namespace pathtracer::scene
