#include "pathtracer/scene/path_tracer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/gbuffer_shading.h"
#include "pathtracer/scene/sampler.h"
#include "pathtracer/scene/shading_scene.h"
#include "shading_math.h"

namespace pathtracer::scene {

namespace {

constexpr float kRayEpsilon = 1e-4F;

// Offsets the AO sampler's seed from the path sampler's so the two streams are independent. 2^32/phi, Knuth's decorrelating constant.
constexpr std::uint32_t kAoSeedOffset = 0x9E3779B9U;
// The same for the Fresnel AOV's stream: frac(sqrt 2)*2^32, kAoSeedOffset's sibling. Distinct, non-zero and odd is the whole requirement.
constexpr std::uint32_t kFresnelSeedOffset = 0x6A09E667U;
// And for volume tracking, whose draw count varies per path: frac(sqrt 3)*2^32, so no BSDF dimension shifts with it.
constexpr std::uint32_t kVolumeSeedOffset = 0xBB67AE85U;
// And for the microsurface walk, whose length varies per vertex: frac(sqrt 11)*2^32, the next odd one of the series.
constexpr std::uint32_t kWalkSeedOffset = 0x510E527FU;

// pbrt's ShadowEpsilon (PBR 6.8.6): a relative back-off on tMax, or a light's own front face occludes it. A no-op for the environment.
constexpr float kShadowDistanceEpsilon = 1e-3F;

// An interior in OpenPBR's form: transmittance reaches transmission_color at transmission_depth, T(t) = color^(t/depth).
struct Medium {
    glm::vec3 color;
    float depth;
    // Depth 0 is no medium (the colour tints the surface instead); pow is exact at colour 0 and t = inf, with no log of zero to guard.
    [[nodiscard]] glm::vec3 transmittance(float t) const { return depth > 0.0F ? glm::pow(color, glm::vec3(t / depth)) : glm::vec3(1.0F); }
};

// Transmission needs a curvature-scaled offset, ~1e-3 on a 500-triangle sphere; raw edge length has no zero and blew up on a flat slab.
float transmissionOffsetEpsilon(const ShadingTriangle& tri) {
    const glm::vec3 n0 = glm::normalize(tri.v0.normal);
    const glm::vec3 n1 = glm::normalize(tri.v1.normal);
    const glm::vec3 n2 = glm::normalize(tri.v2.normal);
    const float curvature = std::max({glm::length(glm::cross(n0, n1)), glm::length(glm::cross(n1, n2)),
                                       glm::length(glm::cross(n2, n0))});
    const float e0 = glm::length(tri.v1.position - tri.v0.position);
    const float e1 = glm::length(tri.v2.position - tri.v1.position);
    const float e2 = glm::length(tri.v0.position - tri.v2.position);
    return std::max(kRayEpsilon, std::max({e0, e1, e2}) * curvature);
}

// Blackman-Harris (Harris 1978) at Arnold's default 1.5 px radius, importance-sampled (Ernst et al. 2006): pdf proportional to f, weight 1.
constexpr float kFilterRadius = 1.5F;
// The window's cosine-series coefficients: w(t) = a0 - a1 cos(2 pi t) + a2 cos(4 pi t) - a3 cos(6 pi t) over t in [0, 1].
constexpr double kBlackmanHarrisA0 = 0.35875;
constexpr double kBlackmanHarrisA1 = 0.48829;
constexpr double kBlackmanHarrisA2 = 0.14128;
constexpr double kBlackmanHarrisA3 = 0.01168;
// Knots of the half-filter CDF over |x| in [0, kFilterRadius]: exact at each knot, linear between, so 63 bins of 0.024 px each.
constexpr int kFilterCdfKnots = 64;
// Per-pixel lanes: beauty.rgb, bounce, shadow lost, surface buckets, AO, Fresnel, alpha, denoise features, volume buckets, unoccluded.
constexpr int kSampleLanes = 38;

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

// Radial offset: Shirley-Chiu's concentric map (1997) keeps u's strata and leaves |rho|^2 uniform, which inverts the radial CDF.
glm::vec2 sampleRadialFilterOffset(glm::vec2 u) {
    const glm::vec2 s = (2.0F * u) - 1.0F;
    // The map's centre, where its angle is undefined, is the zero offset.
    if (s.x == 0.0F && s.y == 0.0F) {
        return glm::vec2(0.0F);
    }
    const bool xMajor = std::abs(s.x) > std::abs(s.y);
    const float rho = xMajor ? s.x : s.y;
    const float phi = xMajor ? 0.25F * std::numbers::pi_v<float> * (s.y / s.x)
                             : (0.5F * std::numbers::pi_v<float>) - (0.25F * std::numbers::pi_v<float> * (s.x / s.y));
    const CdfSample bin = invertCdf(kRadialFilterCdf.data(), kFilterCdfKnots - 1, rho * rho);
    const float radius = kFilterRadius * (static_cast<float>(bin.index) + bin.fraction) / static_cast<float>(kFilterCdfKnots - 1);
    return std::copysign(radius, rho) * glm::vec2(std::cos(phi), std::sin(phi));
}

// The square lattice's separable filter keeps its per-axis strata; the hexagonal lattice's six equidistant neighbours need it radial.
glm::vec2 sampleFilmOffset(pathtracer::gfx::PixelLattice lattice, glm::vec2 u) {
    if (lattice == pathtracer::gfx::PixelLattice::Hexagonal) {
        return sampleRadialFilterOffset(u);
    }
    return {sampleFilterOffset(u.x), sampleFilterOffset(u.y)};
}

}  // namespace

// Largest tile at or below kPathTraceTileSize still giving the pool kTilesPerThread each: at interactive scale 96 leaves most workers idle.
int pathTraceTileSize(int width, int height, unsigned int threadCount) {
    const auto tileCount = [width, height](int size) {
        return ((width + size - 1) / size) * ((height + size - 1) / size);
    };
    const int wanted = static_cast<int>(threadCount) * kTilesPerThread;
    int size = kPathTraceTileSize;
    while (size > 1 && tileCount(size) < wanted) {
        size /= 2;
    }
    return size;
}

namespace {

struct TraceResult {
    glm::vec3 radiance;
    int terminationBounce;  // bounce index the path stopped at (== maxBounces + 1 if depth-capped)
    float shadowLost;        // bounce-0 direct luminance the occluders remove, light-sampled without MIS; 0 with no shadow ray
    float shadowUnoccluded;  // the same estimate with visibility 1: Shadow is the ratio of their means
    float ao;  // 1.0 = unoccluded, 0.0 = occluded within aoMaxDistance
    // One VNDF draw's Fresnel at the primary hit; 0 where there is no BSDF vertex at bounce 0.
    glm::vec3 fresnel{0.0F};
    float alpha = 0.0F;  // PathTraceResult::alpha's per-sample estimate; 0 with no ray at all

    // Transport-component breakdown -- see PathTraceResult's doc comment for the bucketing rule.
    glm::vec3 directDiffuse{0.0F};
    glm::vec3 indirectDiffuse{0.0F};
    glm::vec3 directSpecular{0.0F};
    glm::vec3 indirectSpecular{0.0F};
    glm::vec3 refraction{0.0F};
    glm::vec3 directVolume{0.0F};
    glm::vec3 indirectVolume{0.0F};

    // OIDN's auxiliary features at the path's first rough vertex: its one-sample directional albedo and its shading normal.
    glm::vec3 denoiseAlbedo{0.0F};
    glm::vec3 denoiseNormal{0.0F};
};

// Which transport bucket a path belongs to: set at bounce 0's lobe, stickily overridden to Refraction by any transmission sample.
enum class PathBucket { Diffuse, SpecularReflection, Refraction, Volume };

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
                           const OpenPbrInputs<Constant>& inputs, const ShadingFrame& frame, const glm::vec3& woWorld, std::optional<int> hero) {
    const ShadingFrame coatFrame = inputs.coatWeight > 0.0F ? buildShadingFrame(triangle, shading, material.geometryCoatNormal) : frame;
    const bool perturbed = material.geometryNormal.map || material.geometryNormal.height;
    return makeNormalMappedBsdf(inputs, woWorld * frame, perturbed ? std::optional(glm::normalize(shading.normal) * frame) : std::nullopt, hero,
                                coatFrame[2] * frame, coatFrame[0] * frame);
}

// The VDB media's transmittance over the whole shadow ray, by ratio tracking, once its surfaces have let it through.
glm::vec3 volumeTransmittance(const Ray& ray, const PassThroughScene& scene, Sampler& volumeSampler) {
    if (scene.volumes.empty()) {
        return glm::vec3(1.0F);
    }
    return scene.volumes.transmittance(ray, ray.tMin, ray.tMax, volumeSampler);
}

// The share of light a shadow ray carries to its end: (1 - alpha) + alpha * passThrough at each crossable surface, 0 at any other.
glm::vec3 transmittanceAlong(Ray ray, const PassThroughScene& scene, std::optional<int> hero, std::uint64_t& shadowRays,
                             Sampler& volumeSampler) {
    if (!scene.any) {
        return scene.accel.occluded(ray) ? glm::vec3(0.0F) : volumeTransmittance(ray, scene, volumeSampler);
    }
    const Ray whole = ray;
    glm::vec3 transmittance(1.0F);
    // Closest hit first, so each surface is met in order along the same ray: exact for presence and for an undeviated sheet.
    while (const std::optional<Hit> hit = scene.accel.intersect(ray)) {
        const ShadingTriangle& triangle = scene.shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)];
        const auto instance = static_cast<std::size_t>(triangle.instanceIndex);
        if (!scene.mayPassThrough[instance]) {
            return glm::vec3(0.0F);
        }
        const Material& material = scene.instances[instance].material;
        const ShadingVertex shading = interpolateShading(triangle, hit->u, hit->v);
        const float opacity = opacityAt(material, shading.uv, {});
        glm::vec3 crossing(1.0F - opacity);
        if (opacity > 0.0F) {
            const OpenPbrInputs<Constant> inputs = resolveInputs(material, shading.uv, {}, shading.colour);
            const ShadingFrame frame = buildShadingFrame(triangle, shading, material.geometryNormal);
            crossing += opacity * passThrough(closureAt(material, triangle, shading, inputs, frame, -ray.dir, hero));
        }
        transmittance *= crossing;
        if (transmittance == glm::vec3(0.0F)) {
            return transmittance;
        }
        ray.tMin = std::nextafter(hit->t, std::numeric_limits<float>::infinity());
        ++shadowRays;
    }
    return transmittance * volumeTransmittance(whole, scene, volumeSampler);
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
        const ShadingTriangle& triangle = scene.shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)];
        const auto instance = static_cast<std::size_t>(triangle.instanceIndex);
        const float opacity = scene.mayPassThrough[instance]
                                  ? opacityAt(scene.instances[instance].material, interpolateShading(triangle, hit->u, hit->v).uv, {})
                                  : 1.0F;
        ao += transmittance * opacity * rho(hit->t);
        transmittance *= 1.0F - opacity;
        if (transmittance == 0.0F) {
            return ao;
        }
        ray.tMin = std::nextafter(hit->t, std::numeric_limits<float>::infinity());
    }
    return ao + transmittance;
}

// dirFootprint: the primary ray's direction offsets across its sample's stratum, filtering bounce 0's lookups; deeper ones point-sample.
TraceResult tracePath(const Ray& primaryRay, const glm::mat2x3& dirFootprint, float pixelSolidAngle, const PassThroughScene& scene,
                       const std::vector<int>& instanceLightIndex, const LightSet& lights,
                       bool showSky, const PathTraceSettings& settings,
                       Sampler& sampler, Sampler& volumeSampler, Sampler& walkSampler, glm::vec2 aoSample, glm::vec2 fresnelSample,
                       pathtracer::debug::RayCounts& __restrict rays) {
    const EmbreeAccel& accel = scene.accel;
    glm::vec3 radiance(0.0F);
    glm::vec3 throughput(1.0F);
    Ray ray = primaryRay;
    int bounce = 0;
    // Where the ray's current segment began: a ray continued past an absent surface or through a window keeps its origin.
    float segmentStart = 0.0F;
    // Past an absent surface or through a window's undeviated share: the same ray strictly past the hit, with no change to MIS state.
    const auto continuePast = [&](float t) {
        segmentStart = t;
        ray.tMin = std::nextafter(t, std::numeric_limits<float>::infinity());
    };
    std::optional<PathBucket> pathBucket;  // unset until bounce 0 successfully samples a lobe
    // Single-level medium stack: nullopt = vacuum, set = the medium the ray is inside. Enough for one glass object, not two overlapping.
    std::optional<Medium> medium;
    // The RGB channel this path committed to at a dispersive interface; unset means full RGB transport. See the selection block below.
    std::optional<int> heroChannel;
    glm::vec3 directDiffuseAccum(0.0F);
    glm::vec3 indirectDiffuseAccum(0.0F);
    glm::vec3 directSpecularAccum(0.0F);
    glm::vec3 indirectSpecularAccum(0.0F);
    glm::vec3 refractionAccum(0.0F);
    glm::vec3 directVolumeAccum(0.0F);
    glm::vec3 indirectVolumeAccum(0.0F);
    // Routes a contribution into the path's bucket; a no-op when unset, that background radiance being deliberately unbucketed.
    const auto addToBucket = [&](const glm::vec3& contribution, bool isDirect) {
        if (!pathBucket.has_value()) {
            return;
        }
        switch (*pathBucket) {
            case PathBucket::Refraction:
                refractionAccum += contribution;
                break;
            case PathBucket::Diffuse:
                (isDirect ? directDiffuseAccum : indirectDiffuseAccum) += contribution;
                break;
            case PathBucket::SpecularReflection:
                (isDirect ? directSpecularAccum : indirectSpecularAccum) += contribution;
                break;
            case PathBucket::Volume:
                (isDirect ? directVolumeAccum : indirectVolumeAccum) += contribution;
                break;
        }
    };
    float gShadowLost = 0.0F;  // default: no bounce-0 shadow ray, so no direct light to lose
    float gShadowUnoccluded = 0.0F;
    float gAo = 1.0F;      // default: background is fully unoccluded, matching the polarity in PathTraceResult
    // Default: no BSDF vertex at bounce 0 -- the camera ray missed, or hit an emitter, which returns before shading.
    glm::vec3 gFresnel(0.0F);
    float gAlpha = 0.0F;

    // MIS state for the previous bounce's BSDF sample: reweights this bounce's miss against NEE's pdf so neither double-counts.
    float lastBsdfPdf = 0.0F;
    // A delta lobe has no density for NEE to double-count, so its miss takes full weight: smooth reflection, refraction, the camera.
    bool lastSampleWasDelta = true;
    // The previous vertex's shading position, not ray.origin: MIS needs the pdf NEE would have had, taken from shading.position.
    glm::vec3 lastShadingPosition(0.0F);

    // Denoiser features (OIDN): followed through deltas and sub-pixel lobes to the first rough vertex, apart from `throughput`.
    glm::vec3 featureWeight(1.0F);
    float featureSurvival = 1.0F;  // 1 / Russian roulette's continuation odds while the chain is open, so both features stay unbiased
    bool featurePending = settings.denoiseFeatures;
    glm::vec3 featureAlbedo(0.0F);
    glm::vec3 featureNormal(0.0F);
    // Closes the chain at a surface: no ray, no draw and no Beauty float changes, so the lanes ride along at no cost to the image.
    const auto recordFeatures = [&](const glm::vec3& albedo, const glm::vec3& normal) {
        featureAlbedo = featureSurvival * albedo;
        featureNormal = featureSurvival * normal;
        featurePending = false;
    };
    // Russian roulette from its start bounce: false ends the path, else throughput and the feature survival carry 1/continuation.
    const auto survivesRoulette = [&] {
        if (bounce < settings.russianRouletteStartBounce) {
            return true;
        }
        const float continueProb =
            std::clamp(std::max({throughput.x, throughput.y, throughput.z}), settings.rrMinProb, settings.rrMaxProb);
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

        // Free flight through the VDB media over the segment just travelled, which a miss runs to the ray's own bound (a far clip).
        std::optional<FreeFlight> flight;
        if (!scene.volumes.empty()) {
            flight = scene.volumes.sampleFreeFlight(ray, segmentStart, hit.has_value() ? hit->t : ray.tMax, throughput, volumeSampler);
            throughput *= flight->weight;
            if (featurePending) {
                featureWeight *= flight->weight;
            }
        }
        const bool scattered = flight.has_value() && flight->event == FreeFlightEvent::Scattered;
        const bool absorbed = flight.has_value() && flight->event == FreeFlightEvent::Absorbed;
        if (bounce == 0) {
            // E[1 - mean(w) 1{escape}] = 1 - mean(T): unbiased, though one sample may read below 0 where spectral weights exceed 1.
            gAlpha = hit.has_value() || showSky || scattered || absorbed ? 1.0F : 1.0F - channelMean(throughput);
        }

        // Beer-Lambert for the segment just travelled, to the scattering if one ended it; a miss is unbounded, t = inf.
        if (medium.has_value()) {
            throughput *= medium->transmittance(scattered ? flight->t - segmentStart
                                                : hit.has_value() ? hit->t - segmentStart
                                                                  : std::numeric_limits<float>::infinity());
        }

        // Absorbed carries nothing onward, emission being out of scope: the feature chain closes on a zero albedo and no normal.
        if (absorbed) {
            if (featurePending) {
                recordFeatures(glm::vec3(0.0F), glm::vec3(0.0F));
            }
            break;
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
            // NEE from the scattering point, against the phase-sampled continuation by the power heuristic on directional pdfs.
            if (const std::optional<LightSample> lightSample = lights.sample(position, sampler)) {
                const float phase = henyeyGreenstein(glm::dot(dir, lightSample->direction), flight->g);
                const Ray shadowRay{position, lightSample->direction, 0.0F, lightSample->distance * (1.0F - kShadowDistanceEpsilon)};
                ++rays.shadow;
                const glm::vec3 visibility = transmittanceAlong(shadowRay, scene, heroChannel, rays.shadow, volumeSampler);
                // Shadow's lanes take the light estimator without its MIS weight: alone it estimates all direct light, occluded or not.
                if (bounce == 0) {
                    const glm::vec3 direct = throughput * lightSample->radiance * (phase / lightSample->pdf);
                    const glm::vec3& luminance = cie::sceneLuminanceWeights();
                    gShadowUnoccluded = glm::dot(luminance, direct);
                    gShadowLost = glm::dot(luminance, direct * (1.0F - visibility));
                }
                if (visibility != glm::vec3(0.0F)) {
                    const float lightPdf2 = lightSample->pdf * lightSample->pdf;
                    const float misWeightLight = lightPdf2 / (lightPdf2 + (phase * phase));
                    const glm::vec3 neeContribution =
                        visibility * throughput * lightSample->radiance * (phase * misWeightLight / lightSample->pdf);
                    radiance += neeContribution;
                    if (bounce == 0) {
                        directVolumeAccum += neeContribution;
                    } else {
                        addToBucket(neeContribution, /*isDirect=*/false);
                    }
                }
            }
            // The phase sample weighs f/pdf = 1; its pdf is what this ray's miss or emitter hit is MIS-weighted against.
            const glm::vec3 wi = sampleHenyeyGreenstein(dir, flight->g, sampler.next2D());
            lastBsdfPdf = henyeyGreenstein(glm::dot(dir, wi), flight->g);
            lastSampleWasDelta = false;
            lastShadingPosition = position;
            if (!survivesRoulette()) {
                break;
            }
            ray = Ray{position, wi, 0.0F, std::numeric_limits<float>::max()};
            segmentStart = 0.0F;
            continue;
        }

        if (!hit.has_value()) {
            // showSky gates only the primary ray's miss; indirect bounces and NEE always sample real environment radiance.
            if (bounce == 0 && !showSky) {
                break;
            }
            const glm::vec3 envRadiance =
                bounce == 0 ? lights.environmentRadiance(ray.dir, dirFootprint) : lights.environmentRadiance(ray.dir);
            // Power heuristic (Veach 1997): full weight after the camera or a delta sample, neither having a density to balance.
            float misWeight = 1.0F;
            if (!lastSampleWasDelta) {
                const float lightPdf = lights.pdfEnvironment(ray.dir);
                const float bsdfPdf2 = lastBsdfPdf * lastBsdfPdf;
                misWeight = bsdfPdf2 / (bsdfPdf2 + (lightPdf * lightPdf));
            }
            const glm::vec3 missRadiance = throughput * envRadiance * misWeight;
            radiance += missRadiance;
            addToBucket(missRadiance, /*isDirect=*/bounce == 1);
            break;
        }

        const ShadingTriangle& triangle =
            scene.shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)];

        // An emitter hit, a quad light's triangles being in the BVH: Le, MIS-weighted like a miss, then terminate. Before the depth cap.
        const int lightIndex = instanceLightIndex[static_cast<std::size_t>(triangle.instanceIndex)];
        if (lightIndex >= 0) {
            // An emitter is a surface with no BSDF: its normal, and no albedo.
            if (featurePending) {
                recordFeatures(glm::vec3(0.0F), geometricNormalOf(triangle));
            }
            const glm::vec3 emitted = lights.quadRadianceToward(lightIndex, ray.dir);
            float misWeight = 1.0F;
            if (!lastSampleWasDelta) {
                const float lightPdf = lights.pdfQuad(lightIndex, lastShadingPosition);
                const float bsdfPdf2 = lastBsdfPdf * lastBsdfPdf;
                misWeight = bsdfPdf2 / (bsdfPdf2 + (lightPdf * lightPdf));
            }
            const glm::vec3 hitRadiance = throughput * emitted * misWeight;
            radiance += hitRadiance;
            addToBucket(hitRadiance, /*isDirect=*/bounce == 1);
            break;
        }

        const auto instance = static_cast<std::size_t>(triangle.instanceIndex);
        const Material& material = scene.instances[instance].material;
        // Secondary vertices lack propagated differentials (Igehy's BSDF transfer), so they read level 0 as a zero footprint does.
        const pathtracer::gfx::TextureFootprint footprint =
            bounce == 0 ? primaryHitFootprint(triangle, ray, hit->t, dirFootprint) : pathtracer::gfx::TextureFootprint{};
        const ShadingVertex shading = interpolateShading(triangle, hit->u, hit->v);

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
        const glm::vec3 emitted = emittedRadiance(material, triangle, shading, -ray.dir, footprint);
        if (emitted != glm::vec3(0.0F)) {
            const glm::vec3 hitRadiance = throughput * emitted;
            radiance += hitRadiance;
            addToBucket(hitRadiance, /*isDirect=*/bounce == 1);
        }

        // Depth cap. The extra iteration exists only so the final BSDF ray can collect its MIS-weighted miss or emitter hit.
        const bool capped = bounce > settings.maxBounces;
        if (capped && !scene.mayPassThrough[instance]) {
            break;
        }

        const OpenPbrInputs<Constant> inputs = resolveInputs(material, shading.uv, footprint, shading.colour);
        // Commit to one RGB channel at the first dispersive interface; sum == 0 is reachable, rrMinProb keeps zero-throughput paths alive.
        if (!heroChannel.has_value() && isDispersive(inputs)) {
            const float sum = throughput.x + throughput.y + throughput.z;
            if (sum > 0.0F) {
                const float u = sampler.next1D() * sum;
                const int channel = u < throughput.x ? 0 : (u < throughput.x + throughput.y ? 1 : 2);
                // The same one-sample spectral estimator: the drawn channel over its selection odds throughput[c] / sum.
                if (featurePending) {
                    const float feature = featureWeight[channel] * sum / throughput[channel];
                    featureWeight = glm::vec3(0.0F);
                    featureWeight[channel] = feature;
                }
                throughput = glm::vec3(0.0F);
                throughput[channel] = sum;
                heroChannel = channel;
            }
        }

        const ShadingFrame frame = buildShadingFrame(triangle, shading, material.geometryNormal);
        const glm::vec3 woWorld = -ray.dir;
        // True flat plane normal, for light-leak rejection and ray-origin offsets: both need geometry, not the shading normal.
        const glm::vec3 geoNormal = geometricNormalOf(triangle);

        // Built once for both estimators below: the continuation draw and NEE's evaluation share every wo-side lookup it holds.
        const NormalMappedBsdf closure = closureAt(material, triangle, shading, inputs, frame, woWorld, heroChannel);

        // Past the cap a window still passes the final ray on to its emitter, by its expected undeviated share, as NEE's shadow rays do.
        if (capped) {
            const glm::vec3 through = passThrough(closure);
            if (through == glm::vec3(0.0F)) {
                break;
            }
            throughput *= through;
            continuePast(hit->t);
            --bounce;
            continue;
        }

        if (bounce == 0) {
            // The Fresnel the microfacet lobe evaluates here, one VNDF draw per sample, on its own stream. No ray and no BVH query.
            gFresnel = fresnelAtMicrofacet(closure, fresnelSample);
            // Obscurance (Zhukov 1998; Iones 2003): sampling at pdf = cos/pi cancels both factors, so the estimator is the mean of rho.
            const bool frontSide = glm::dot(geoNormal, woWorld) > 0.0F;
            const glm::vec3 aoDir =
                frame * sampleCosineHemisphere(aoSample) * (frontSide ? 1.0F : -1.0F);
            // NEE's near-side origin verbatim: the shading-terminator offset plus a geometric-normal back-off, on whichever side wo is.
            const glm::vec3 aoOrigin = shadowTerminatorOffset(triangle, hit->u, hit->v, frontSide) +
                                        (geoNormal * kRayEpsilon * (frontSide ? 1.0F : -1.0F));
            ++rays.ao;
            gAo = obscuranceAlong(Ray{aoOrigin, aoDir, kRayEpsilon, settings.aoMaxDistance}, scene, settings.aoMaxDistance);
        }

        // A zero BSDF, emission alone, carries no light onward: the path ends here, its AO and Fresnel lanes already written.
        if (!scatters(closure)) {
            if (featurePending) {
                recordFeatures(glm::vec3(0.0F), frame[2]);
            }
            break;
        }

        // A failed sample must not skip the NEE block: the two are independent estimators, sharing only this vertex's params and frame.
        std::optional<WalkVertex> revisit;  // where the draw's walk first returned to the facet: NEE resumes its later orders there
        const std::optional<BsdfSample> sample = sampleBsdf(closure, sampler, walkSampler, revisit);

        // Bucket assignment: bounce 0 sets it from scratch, any later bounce only ever overrides it to Refraction, stickily.
        if (sample.has_value()) {
            if (bounce == 0) {
                pathBucket = sample->type == LobeType::Transmission ? PathBucket::Refraction
                             : sample->type == LobeType::Diffuse            ? PathBucket::Diffuse
                                                                             : PathBucket::SpecularReflection;
            } else if (sample->type == LobeType::Transmission) {
                pathBucket = PathBucket::Refraction;
            }
        }

        // One side rule for NEE and continuation alike: reflection lobes count on wo's geometric side, transmission lobes across it.
        const bool woAbove = glm::dot(woWorld, geoNormal) > 0.0F;

        // NEE: sample a light, evaluate the BSDF toward it, add it MIS-weighted if unoccluded. Fired whatever lobe `sample` drew.
        const std::optional<LightSample> lightSample = lights.sample(shading.position, sampler);
        if (lightSample.has_value()) {
            const float geoCos = glm::dot(lightSample->direction, geoNormal);
            // A light across the surface reaches wo only through a refracting interface: only then is the far side worth a shadow ray.
            const bool farSide = (geoCos > 0.0F) != woAbove;
            if (!farSide || transmits(closure)) {
                const glm::vec3 wiLocalLight = lightSample->direction * frame;
                // One evaluation for the value, pdf and per-lobe split, each lobe cosine-weighted about its own normal.
                const BsdfEval eval = evaluateBsdfSplit(closure, wiLocalLight, revisit, walkSampler);
                const glm::vec3 bsdfValue = farSide ? eval.transmission : eval.diffuse + eval.specular;
                // A zero pdf is light only the walk's later orders reach: the BSDF draw's MIS weight is 0 there, so NEE takes it whole.
                if (bsdfValue.x > 0.0F || bsdfValue.y > 0.0F || bsdfValue.z > 0.0F) {
                    // Offset along geoNormal toward the light's side: the far side crosses the interface, so it takes the curvature offset.
                    const float shadowEpsilon =
                        farSide ? transmissionOffsetEpsilon(triangle) : kRayEpsilon;
                    const glm::vec3 shadowOrigin =
                        shadowTerminatorOffset(triangle, hit->u, hit->v, geoCos > 0.0F) +
                        (geoNormal * shadowEpsilon * (geoCos > 0.0F ? 1.0F : -1.0F));
                    // pbrt's SpawnRayTo: re-formed from the offset origin to the sampled point, which a back-off along wi cannot reach.
                    const bool finiteLight = lightSample->distance < std::numeric_limits<float>::max();
                    const glm::vec3 toLight =
                        finiteLight ? (shading.position - shadowOrigin) +
                                           (lightSample->direction * lightSample->distance)
                                     : lightSample->direction;  // the environment's point is at infinity
                    const float shadowDistance = finiteLight ? glm::length(toLight) : lightSample->distance;
                    const Ray shadowRay{shadowOrigin, finiteLight ? toLight / shadowDistance : toLight,
                                         shadowEpsilon, shadowDistance * (1.0F - kShadowDistanceEpsilon)};
                    ++rays.shadow;
                    const glm::vec3 visibility = transmittanceAlong(shadowRay, scene, heroChannel, rays.shadow, volumeSampler);
                    // As at a scattering: the unweighted light estimator, its luminance and the share the occluders remove of it.
                    if (bounce == 0) {
                        const glm::vec3 direct = bsdfValue * throughput * lightSample->radiance / lightSample->pdf;
                        const glm::vec3& luminance = cie::sceneLuminanceWeights();
                        gShadowUnoccluded = glm::dot(luminance, direct);
                        gShadowLost = glm::dot(luminance, direct * (1.0F - visibility));
                    }
                    if (visibility != glm::vec3(0.0F)) {
                        const float lightPdf2 = lightSample->pdf * lightSample->pdf;
                        const float bsdfPdf2 = eval.pdf * eval.pdf;
                        const float misWeightLight = lightPdf2 / (lightPdf2 + bsdfPdf2);
                        const glm::vec3 common = visibility * throughput * lightSample->radiance * misWeightLight / lightSample->pdf;
                        const glm::vec3 neeContribution = bsdfValue * common;
                        radiance += neeContribution;
                        if (bounce == 0) {
                            // Bounce 0 splits NEE by the lobe that carried it, not sample->type, which names the continuation.
                            if (farSide) {
                                refractionAccum += eval.transmission * common;
                            } else {
                                directDiffuseAccum += eval.diffuse * common;
                                directSpecularAccum += eval.specular * common;
                            }
                        } else {
                            // Deeper bounces keep the sticky bucket: which lobe carries light here no longer names the transport type.
                            addToBucket(neeContribution, /*isDirect=*/false);
                        }
                    }
                }
            }
        }

        // No continuation direction could be sampled: terminate, NEE having already taken this vertex's direct lighting.
        if (!sample.has_value()) {
            // A failed draw is the albedo estimator's zero, not a skipped sample; the surface's normal stands.
            if (featurePending) {
                recordFeatures(glm::vec3(0.0F), frame[2]);
            }
            break;
        }

        // A window's undeviated transmission is a null vertex: MIS keeps the last scattering vertex, whose density this ray still carries.
        if (!sample->passThrough) {
            lastBsdfPdf = sample->pdf;
            lastSampleWasDelta = sample->delta;
            lastShadingPosition = shading.position;
        }

        const glm::vec3 wiWorld = sample->passThrough ? ray.dir : frame * sample->wiLocal;

        // The side rule NEE applied: Schussler's microsurface keeps lobes on omega_g's sides, this the flat face's (his Fig. 18 residual).
        const bool crosses = (glm::dot(wiWorld, geoNormal) > 0.0F) != woAbove;
        const glm::vec3 weight = crosses ? sample->transmitWeight : sample->throughputWeight - sample->transmitWeight;
        // E[f cos / pdf] is the directional albedo; a lobe denser than 1/pixel solid angle images the next surface sharply, as a delta.
        if (featurePending) {
            if (sample->delta || sample->lobePdf * pixelSolidAngle >= 1.0F) {
                featureWeight *= weight;
            } else {
                recordFeatures(featureWeight * weight, frame[2]);
            }
        }
        if (weight == glm::vec3(0.0F)) {
            break;
        }
        // A bulk ray enters or leaves the medium exactly when it crosses the surface; a thin wall has no interior to enter.
        if (crosses && !inputs.geometryThinWalled) {
            medium = medium.has_value() ? std::nullopt : std::make_optional(Medium{inputs.transmissionColor, inputs.transmissionDepth});
        }

        throughput *= weight;

        if (!survivesRoulette()) {
            break;
        }

        if (sample->passThrough) {
            continuePast(hit->t);
            continue;
        }

        // Chiang/Li/Burley origin nudged toward wi's side: unmirrored, it desynced the medium stack on 0.93% of cornell's glass entries.
        const bool leavingOnNormalSide = glm::dot(wiWorld, geoNormal) > 0.0F;
        const float offsetEpsilon = sample->type == LobeType::Transmission
                                         ? transmissionOffsetEpsilon(triangle)
                                         : kRayEpsilon;
        const glm::vec3 offsetOrigin =
            shadowTerminatorOffset(triangle, hit->u, hit->v, leavingOnNormalSide) +
            (geoNormal * offsetEpsilon * (leavingOnNormalSide ? 1.0F : -1.0F));
        ray = Ray{offsetOrigin, wiWorld, offsetEpsilon, std::numeric_limits<float>::max()};
        segmentStart = 0.0F;
    }

    return {radiance,           bounce,               gShadowLost,          gShadowUnoccluded,    gAo,
            gFresnel,           gAlpha,               directDiffuseAccum,   indirectDiffuseAccum,
            directSpecularAccum, indirectSpecularAccum, refractionAccum,    directVolumeAccum,
            indirectVolumeAccum, featureAlbedo,         featureNormal};
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
    result.replicateLuminance.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                                         static_cast<std::size_t>(replicates),
                                     0.0F);
    result.replicates = replicates;
    return result;
}

void renderPathTraced(const Camera& camera, const EmbreeAccel& accel,
                       const std::vector<ShadingTriangle>& shadingTriangles,
                       const std::vector<MeshInstance>& instances,
                       const std::vector<int>& instanceLightIndex, const LightSet& lights, const VolumeSet& volumes,
                       int width, int height, bool showSky, const PathTraceSettings& settings,
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
    // Sample spacing pixelSide/sqrt(N) (pbrt-v4's differential scale), so prefilter blur vanishes as N grows; unbounded N as Sampler's.
    const float strataPerAxis = std::sqrt(static_cast<float>(std::max(sampleCount, sampleBase + settings.samplesPerPixel))) / pixelSide;
    // d(ndc) across that spacing: two NDC units span the film, and +Y flips because row 0 is the top.
    const glm::mat2 ndcPerStratum(2.0F / (frame.x * strataPerAxis), 0.0F, 0.0F, -2.0F / (frame.y * strataPerAxis));
    // d(ndc) across a pixel side, a square of the cell's area: its image on the unit sphere is the solid angle a mirror lobe must undercut.
    const glm::mat2 ndcPerPixel((2.0F * pixelSide) / frame.x, 0.0F, 0.0F, (-2.0F * pixelSide) / frame.y);
    // Replicated RQMC: pass p draws from replicate p mod K at its position p div K, so each replicate is a prefix of its own sequence.
    const int pass = sampleBase / settings.samplesPerPixel;
    const std::uint32_t seed = replicateSeed(scrambleSeed, pass % settings.replicates);
    const int replicateBase = (pass / settings.replicates) * settings.samplesPerPixel;
    const int replicateCount = sampleCount > 0 ? (sampleCount + settings.replicates - 1) / settings.replicates : sampleCount;
    // Derived from the target, not fixed: the interactive scale renders a fraction of the frame, where a 96 px grid is only a few tiles.
    const int tileSize = pathTraceTileSize(width, height, threadPool.threadCount());
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;
    const std::vector<std::uint8_t> mayPassThrough = passThroughInstances(instances);
    const PassThroughScene scene{accel, shadingTriangles, instances, mayPassThrough,
                                 std::any_of(mayPassThrough.begin(), mayPassThrough.end(), [](std::uint8_t may) { return may != 0; }),
                                 volumes};

    // One worker owns one tile: filter importance sampling keeps each sample in its own pixel's estimate, so tiles share no work.
    const auto renderTile = [&](int tileIndex) {
        const int tileX0 = (tileIndex % tilesX) * tileSize;
        const int tileY0 = (tileIndex / tilesX) * tileSize;
        const int tileX1 = std::min(tileX0 + tileSize, width);
        const int tileY1 = std::min(tileY0 + tileSize, height);

        // Stack-local, not thread_local: zeroed by construction, so the reset boundary is the tile boundary with no bookkeeping.
        pathtracer::debug::RayCounts tileRays;

        const auto samples = static_cast<float>(settings.samplesPerPixel);
        for (int y = tileY0; y < tileY1; ++y) {
            for (int x = tileX0; x < tileX1; ++x) {
                const glm::vec2 centre = pathtracer::gfx::pixelCentre(lattice, x, y);
                std::array<float, kSampleLanes> sum{};
                for (int s = 0; s < settings.samplesPerPixel; ++s) {
                    // replicateBase + s is this sample's position in its replicate's sequence: it must advance across passes to stratify.
                    const int sampleIndex = replicateBase + s;
                    Sampler sampler(x, y, sampleIndex, replicateCount, seed, lattice);
                    // Reaches past the frame: rectilinear extrapolates, fisheye is rayless past its circle, lat-long wraps seam, pole.
                    const glm::vec2 film = centre + sampleFilmOffset(lattice, sampler.next2D());
                    const float ndcX = ((film.x / frame.x) * 2.0F) - 1.0F;
                    // HdrImage row 0 is the top (EXR/glTF convention); NDC +Y is up -- flip.
                    const float ndcY = 1.0F - ((film.y / frame.y) * 2.0F);
                    const std::optional<Camera::RayDifferential> primary = camera.primaryRayDifferential(basis, ndcX, ndcY);
                    // AO and Fresnel draw from their own stream: taking dimensions from `sampler` would shift every later dimension.
                    Sampler aoSampler(x, y, sampleIndex, replicateCount, seed ^ kAoSeedOffset, lattice);
                    const glm::vec2 aoSample = aoSampler.next2D();
                    Sampler fresnelSampler(x, y, sampleIndex, replicateCount, seed ^ kFresnelSeedOffset, lattice);
                    const glm::vec2 fresnelSample = fresnelSampler.next2D();
                    Sampler volumeSampler(x, y, sampleIndex, replicateCount, seed ^ kVolumeSeedOffset, lattice);
                    Sampler walkSampler(x, y, sampleIndex, replicateCount, seed ^ kWalkSeedOffset, lattice);
                    // The pixel's footprint on the unit sphere, read only by the feature chain and only where a ray exists.
                    const auto pixelSolidAngle = [&] {
                        const glm::mat2x3 dirPerPixel = primary->dirPerNdc * ndcPerPixel;
                        return glm::length(glm::cross(dirPerPixel[0], dirPerPixel[1]));
                    };
                    // Nullopt is a fisheye sample outside the image circle: no ray exists, so every lane reads zero for it.
                    const TraceResult trace =
                        primary ? tracePath(primary->ray, primary->dirPerNdc * ndcPerStratum,
                                            settings.denoiseFeatures ? pixelSolidAngle() : 0.0F, scene, instanceLightIndex, lights,
                                            showSky, settings, sampler, volumeSampler, walkSampler, aoSample, fresnelSample, tileRays)
                                : TraceResult{};
                    const std::array<float, kSampleLanes> values{
                        trace.radiance.x,          trace.radiance.y,
                        trace.radiance.z,          static_cast<float>(trace.terminationBounce),
                        trace.shadowLost,          trace.directDiffuse.x,
                        trace.directDiffuse.y,     trace.directDiffuse.z,
                        trace.indirectDiffuse.x,   trace.indirectDiffuse.y,
                        trace.indirectDiffuse.z,   trace.directSpecular.x,
                        trace.directSpecular.y,    trace.directSpecular.z,
                        trace.indirectSpecular.x,  trace.indirectSpecular.y,
                        trace.indirectSpecular.z,  trace.refraction.x,
                        trace.refraction.y,        trace.refraction.z,
                        trace.ao,                  trace.fresnel.x,
                        trace.fresnel.y,           trace.fresnel.z,
                        trace.alpha,               trace.denoiseAlbedo.x,
                        trace.denoiseAlbedo.y,     trace.denoiseAlbedo.z,
                        trace.denoiseNormal.x,     trace.denoiseNormal.y,
                        trace.denoiseNormal.z,     trace.directVolume.x,
                        trace.directVolume.y,      trace.directVolume.z,
                        trace.indirectVolume.x,    trace.indirectVolume.y,
                        trace.indirectVolume.z,    trace.shadowUnoccluded};
                    for (int lane = 0; lane < kSampleLanes; ++lane) {
                        sum[lane] += values[lane];
                    }
                }
                // Unit weights make the estimate the plain mean; divided, not scaled by fl(1/n), so all-0 or all-1 samples read exactly.
                writeTexel(out.beauty, x, y, glm::vec3(sum[0], sum[1], sum[2]) / samples);
                writeTexel(out.bounceHeatmap, x, y, sum[3] / samples);
                writeTexel(out.shadowLost, x, y, sum[4] / samples);
                writeTexel(out.directDiffuse, x, y, glm::vec3(sum[5], sum[6], sum[7]) / samples);
                writeTexel(out.indirectDiffuse, x, y, glm::vec3(sum[8], sum[9], sum[10]) / samples);
                writeTexel(out.directSpecular, x, y, glm::vec3(sum[11], sum[12], sum[13]) / samples);
                writeTexel(out.indirectSpecular, x, y, glm::vec3(sum[14], sum[15], sum[16]) / samples);
                writeTexel(out.refraction, x, y, glm::vec3(sum[17], sum[18], sum[19]) / samples);
                writeTexel(out.ao, x, y, sum[20] / samples);
                writeTexel(out.fresnel, x, y, glm::vec3(sum[21], sum[22], sum[23]) / samples);
                writeTexel(out.alpha, x, y, sum[24] / samples);
                writeTexel(out.denoiseAlbedo, x, y, glm::vec3(sum[25], sum[26], sum[27]) / samples);
                writeTexel(out.denoiseNormal, x, y, glm::vec3(sum[28], sum[29], sum[30]) / samples);
                writeTexel(out.directVolume, x, y, glm::vec3(sum[31], sum[32], sum[33]) / samples);
                writeTexel(out.indirectVolume, x, y, glm::vec3(sum[34], sum[35], sum[36]) / samples);
                writeTexel(out.shadowUnoccluded, x, y, sum[37] / samples);
            }
        }

        stats.addTile(tileRays);
    };

    threadPool.parallelFor(tilesX * tilesY, [&renderTile, &generation, requestedGeneration,
                                              &stats](int tileIndex) {
        if (generation.load(std::memory_order_relaxed) != requestedGeneration) {
            stats.addCancelledTile();  // counted, not ignored: a cancelled pass must not read as a short one
            return;  // stale -- caller discards this pass's result entirely
        }
        renderTile(tileIndex);
    });
}

}  // namespace pathtracer::scene
