#include "pathtracer/scene/path_tracer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/gbuffer_shading.h"
#include "pathtracer/scene/sampler.h"
#include "pathtracer/scene/shading_scene.h"

namespace pathtracer::scene {

namespace {

constexpr float kRayEpsilon = 1e-4F;

// Offsets the AO sampler's seed from the path sampler's so the two streams are independent. 2^32/phi, Knuth's decorrelating constant.
constexpr std::uint32_t kAoSeedOffset = 0x9E3779B9U;
// The same for the Fresnel AOV's stream: frac(sqrt 2)*2^32, kAoSeedOffset's sibling. Distinct, non-zero and odd is the whole requirement.
constexpr std::uint32_t kFresnelSeedOffset = 0x6A09E667U;

// pbrt's ShadowEpsilon (PBR 6.8.6): a relative back-off on tMax, or a light's own front face occludes it. A no-op for the environment.
constexpr float kShadowDistanceEpsilon = 1e-3F;

// Beer-Lambert (Arnold/OpenPBR): sigma_a = -ln(color)/depth. The colour floor stays, since -log(0) would meet t = inf as 0*inf.
glm::vec3 sigmaAFromTransmission(const glm::vec3& color, float depth) {
    if (depth <= 0.0F) {
        return glm::vec3(0.0F);
    }
    return -glm::log(glm::max(color, glm::vec3(1e-6F))) / depth;
}

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

// Blackman-Harris at Arnold's default 1.5px radius: support wider than one pixel, so each sample reconstructs several pixels.
constexpr float kFilterRadius = 1.5F;
// How far a splat reaches: a sample sits at most 1.0 past its own pixel's far centre, so two pixels away weighs exactly zero.
constexpr int kFilterExtent = 1;
constexpr int kFilterTableSize = 64;
// Per-tile lanes: beauty.rgb, termination bounce, shadow, five transport buckets, AO, Fresnel. Scalars broadcast to RGB at write-out.
constexpr int kSampleLanes = 24;
constexpr int kTileLanes = kSampleLanes + 1;  // plus the per-pixel filter weight the lanes above are normalised by

// Sampled at |x| = i/(N-1) * kFilterRadius, read by truncating lookup: PBRT's table trick, replacing three cos() per tap.
std::array<float, kFilterTableSize> buildFilterTable() {
    constexpr float kA0 = 0.35875F;
    constexpr float kA1 = 0.48829F;
    constexpr float kA2 = 0.14128F;
    constexpr float kA3 = 0.01168F;
    constexpr float kPi = 3.14159265F;
    std::array<float, kFilterTableSize> table{};
    for (int i = 0; i < kFilterTableSize; ++i) {
        // Blackman-Harris is defined over [0,1] centred at t = 0.5, so |x| = 0 maps there and the radius to the zero end.
        const float t = 0.5F + (0.5F * static_cast<float>(i) / static_cast<float>(kFilterTableSize - 1));
        table[static_cast<std::size_t>(i)] = kA0 - (kA1 * std::cos(2.0F * kPi * t)) +
                                              (kA2 * std::cos(4.0F * kPi * t)) -
                                              (kA3 * std::cos(6.0F * kPi * t));
    }
    return table;
}

const std::array<float, kFilterTableSize> kFilterTable = buildFilterTable();

float filterWeight(float distance) {
    const float t = std::abs(distance) / kFilterRadius;
    if (t >= 1.0F) {
        return 0.0F;
    }
    return kFilterTable[static_cast<std::size_t>(t * static_cast<float>(kFilterTableSize - 1))];
}

}  // namespace

// Largest tile at or below kPathTraceTileSize still giving the pool kTilesPerThread each: at interactive scale 96 leaves most workers idle.
int pathTraceTileSize(int width, int height, unsigned int threadCount) {
    const auto tileCount = [width, height](int size) {
        return ((width + size - 1) / size) * ((height + size - 1) / size);
    };
    const int wanted = static_cast<int>(threadCount) * kTilesPerThread;
    int size = kPathTraceTileSize;
    while (size > kMinPathTraceTileSize && tileCount(size) < wanted) {
        size = std::max(size / 2, kMinPathTraceTileSize);
    }
    return size;
}

namespace {

struct TraceResult {
    glm::vec3 radiance;
    int terminationBounce;  // bounce index the path stopped at (== maxBounces + 1 if depth-capped)
    float shadow;           // 1.0 = shadowed/occluded, 0.0 = lit or no primary hit at all (background)
    float ao;  // 1.0 = unoccluded, 0.0 = occluded within aoMaxDistance -- inverted relative to shadow above
    // One VNDF draw's Fresnel at the primary hit; 0 where there is no BSDF vertex at bounce 0.
    glm::vec3 fresnel{0.0F};

    // Transport-component breakdown -- see PathTraceResult's doc comment for the bucketing rule.
    glm::vec3 directDiffuse{0.0F};
    glm::vec3 indirectDiffuse{0.0F};
    glm::vec3 directSpecular{0.0F};
    glm::vec3 indirectSpecular{0.0F};
    glm::vec3 refraction{0.0F};
};

// Which transport bucket a path belongs to: set at bounce 0's lobe, stickily overridden to Refraction by any transmission sample.
enum class PathBucket { Diffuse, SpecularReflection, Refraction };

TraceResult tracePath(const Ray& primaryRay, const EmbreeAccel& accel,
                       const std::vector<ShadingTriangle>& shadingTriangles,
                       const std::vector<MeshInstance>& instances,
                       const std::vector<int>& instanceLightIndex, const LightSet& lights,
                       bool showSky, const PathTraceSettings& settings,
                       const std::vector<PathTraceSettings>& perInstanceSettings,
                       Sampler& sampler, glm::vec2 aoSample, glm::vec2 fresnelSample,
                       pathtracer::debug::RayCounts& __restrict rays) {
    glm::vec3 radiance(0.0F);
    glm::vec3 throughput(1.0F);
    Ray ray = primaryRay;
    int bounce = 0;
    std::optional<PathBucket> pathBucket;  // unset until bounce 0 successfully samples a lobe
    // Single-level medium stack: nullopt = vacuum, set = the sigmaA the ray is inside. Enough for one glass object, not two overlapping.
    std::optional<glm::vec3> mediumSigmaA;
    // The RGB channel this path committed to at a dispersive interface; unset means full RGB transport. See the selection block below.
    std::optional<int> heroChannel;
    glm::vec3 directDiffuseAccum(0.0F);
    glm::vec3 indirectDiffuseAccum(0.0F);
    glm::vec3 directSpecularAccum(0.0F);
    glm::vec3 indirectSpecularAccum(0.0F);
    glm::vec3 refractionAccum(0.0F);
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
        }
    };
    float gShadow = 0.0F;  // default: no surface hit at all -- not "shadowed", just background
    float gAo = 1.0F;      // default: background is fully unoccluded, matching the polarity in PathTraceResult
    // Default: no BSDF vertex at bounce 0 -- the camera ray missed, or hit an emitter, which returns before shading.
    glm::vec3 gFresnel(0.0F);

    // MIS state for the previous bounce's BSDF sample: reweights this bounce's miss against NEE's pdf so neither double-counts.
    float lastBsdfPdf = 0.0F;
    // A delta lobe has no density for NEE to double-count, so its miss takes full weight; only the smooth-glass branch gives pdf 0.
    bool lastSampleWasDelta = false;
    // The previous vertex's shading position, not ray.origin: MIS needs the pdf NEE would have had, taken from shading.position.
    glm::vec3 lastShadingPosition(0.0F);

    // bounce 0 always traces: maxBounces counts secondary bounces, so maxBounces==0 is direct lighting with no continuation rays.
    for (; bounce <= settings.maxBounces + 1; ++bounce) {
        (bounce == 0 ? rays.primary : rays.bounce) += 1;
        const std::optional<Hit> hit = accel.intersect(ray);

        // Beer-Lambert for the segment just travelled. A miss is unbounded, so it is written per channel: exp(-0 * inf) is NaN.
        if (mediumSigmaA.has_value()) {
            const glm::vec3& sigmaA = *mediumSigmaA;
            throughput *= hit.has_value()
                               ? glm::exp(-sigmaA * hit->t)
                               : glm::vec3(sigmaA.x > 0.0F ? 0.0F : 1.0F,
                                            sigmaA.y > 0.0F ? 0.0F : 1.0F,
                                            sigmaA.z > 0.0F ? 0.0F : 1.0F);
        }

        if (!hit.has_value()) {
            // showSky gates only the primary ray's miss; indirect bounces and NEE always sample real environment radiance.
            if (bounce == 0 && !showSky) {
                break;
            }
            const glm::vec3 envRadiance = lights.environmentRadiance(ray.dir);
            // Power heuristic (Veach 1997): full weight at bounce 0 and after a delta sample, neither having a density to balance.
            float misWeight = 1.0F;
            if (bounce > 0 && !lastSampleWasDelta) {
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
            shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)];

        // An emitter hit, a quad light's triangles being in the BVH: Le, MIS-weighted like a miss, then terminate. Before the depth cap.
        const int lightIndex = instanceLightIndex[static_cast<std::size_t>(triangle.instanceIndex)];
        if (lightIndex >= 0) {
            const glm::vec3 emitted = lights.quadRadianceToward(lightIndex, ray.dir);
            float misWeight = 1.0F;
            if (bounce > 0 && !lastSampleWasDelta) {
                const float lightPdf = lights.pdfQuad(lightIndex, lastShadingPosition);
                const float bsdfPdf2 = lastBsdfPdf * lastBsdfPdf;
                misWeight = bsdfPdf2 / (bsdfPdf2 + (lightPdf * lightPdf));
            }
            const glm::vec3 hitRadiance = throughput * emitted * misWeight;
            radiance += hitRadiance;
            addToBucket(hitRadiance, /*isDirect=*/bounce == 1);
            break;
        }

        const Material& material =
            instances[static_cast<std::size_t>(triangle.instanceIndex)].material;
        const PathTraceSettings& instanceSettings =
            perInstanceSettings[static_cast<std::size_t>(triangle.instanceIndex)];

        // A constant surface emits its base colour two-sided and scatters nothing; absent from LightSet, so the hit takes MIS weight 1.
        if (instanceSettings.shadingModel == ShadingModel::Constant) {
            const ShadingVertex shading = interpolateShading(triangle, hit->u, hit->v);
            const glm::vec3 hitRadiance =
                throughput * resolveBaseColor(material, shading.uv, shading.colour, instanceSettings);
            radiance += hitRadiance;
            addToBucket(hitRadiance, /*isDirect=*/bounce == 1);
            break;
        }

        // Depth cap. The extra iteration exists only so the final BSDF ray can collect its MIS-weighted miss or emitter hit.
        if (bounce > settings.maxBounces) {
            break;
        }

        // Commit to one RGB channel at the first dispersive interface; sum == 0 is reachable, rrMinProb keeps zero-throughput paths alive.
        if (!heroChannel.has_value() && instanceSettings.abbe > 0.0F &&
            instanceSettings.transmissionFactor > 0.0F) {
            const float sum = throughput.x + throughput.y + throughput.z;
            if (sum > 0.0F) {
                const float u = sampler.next1D() * sum;
                const int channel = u < throughput.x ? 0 : (u < throughput.x + throughput.y ? 1 : 2);
                throughput = glm::vec3(0.0F);
                throughput[channel] = sum;
                heroChannel = channel;
            }
        }

        const ShadingVertex shading = interpolateShading(triangle, hit->u, hit->v);
        const ShadingFrame frame = buildShadingFrame(triangle, shading, material, instanceSettings);
        const BsdfParams params =
            resolveBsdfParams(material, shading.uv, shading.colour, instanceSettings, heroChannel);
        const glm::vec3 woWorld = -ray.dir;
        // True flat plane normal, for light-leak rejection and ray-origin offsets: both need geometry, not the shading normal.
        const glm::vec3 geoNormal = geometricNormalOf(triangle);

        const glm::vec3 woLocal = woWorld * frame;
        // Built once for both estimators below: the continuation draw and NEE's evaluation share every wo-side lookup it holds.
        const BsdfClosure closure = makeBsdfClosure(params, woLocal);

        if (bounce == 0) {
            gShadow = 1.0F;  // assume shadowed once we know there's a real surface; the NEE check below may clear this
            // The Fresnel the microfacet lobe evaluates here, one VNDF draw per sample, on its own stream. No ray and no BVH query.
            gFresnel = fresnelAtMicrofacet(params, woLocal, fresnelSample);
            // Obscurance (Zhukov 1998; Iones 2003): sampling at pdf = cos/pi cancels both factors, so the estimator is the mean of rho.
            const bool frontSide = glm::dot(geoNormal, woWorld) > 0.0F;
            const glm::vec3 aoDir =
                frame * sampleCosineHemisphere(aoSample) * (frontSide ? 1.0F : -1.0F);
            // NEE's near-side origin verbatim: the shading-terminator offset plus a geometric-normal back-off, on whichever side wo is.
            const glm::vec3 aoOrigin = shadowTerminatorOffset(triangle, hit->u, hit->v, frontSide) +
                                        (geoNormal * kRayEpsilon * (frontSide ? 1.0F : -1.0F));
            ++rays.ao;
            // Closest-hit, not any-hit, because rho needs the distance; measured under 1.3% of frame time at this ray length.
            const std::optional<Hit> aoHit =
                accel.intersect(Ray{aoOrigin, aoDir, kRayEpsilon, settings.aoMaxDistance});
            // rho(x) = 1 - (1-x)^2 is the lowest-degree polynomial with rho(0)=0, rho(1)=1 and rho'(1)=0, so nothing steps at D.
            if (aoHit.has_value()) {
                const float k = 1.0F - (aoHit->t / settings.aoMaxDistance);
                gAo = 1.0F - (k * k);
            } else {
                gAo = 1.0F;
            }
        }

        // A failed sample must not skip the NEE block: the two are independent estimators, sharing only this vertex's params and frame.
        const std::optional<BsdfSample> sample = sampleBsdf(closure, sampler);

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

        // Medium toggle, co-located with the bucket since both key off a Transmission sample. Reflection and TIR leave it untouched.
        if (sample.has_value() && sample->type == LobeType::Transmission) {
            mediumSigmaA = mediumSigmaA.has_value()
                               ? std::nullopt
                               : std::make_optional(sigmaAFromTransmission(
                                     instanceSettings.transmissionColor,
                                     instanceSettings.transmissionDepth));
        }

        // NEE: sample a light, evaluate the BSDF toward it, add it MIS-weighted if unoccluded. Fired whatever lobe `sample` drew.
        const std::optional<LightSample> lightSample = lights.sample(shading.position, sampler);
        if (lightSample.has_value()) {
            const float geoCos = glm::dot(lightSample->direction, geoNormal);
            const float shadingCos = glm::dot(lightSample->direction, frame[2]);
            // Both sides, not just wo's: on a transmissive surface a light behind the vertex reaches the eye through the transmission lobe.
            const bool nearSide = geoCos > 0.0F && shadingCos > 0.0F;
            const bool farSide = geoCos < 0.0F && shadingCos < 0.0F && params.transmissionFactor > 0.0F;
            if (nearSide || farSide) {
                const glm::vec3 wiLocalLight = lightSample->direction * frame;
                const float lightCos = std::abs(shadingCos);  // far-side samples carry a negative cosine
                // One evaluation for the value, the pdf and the per-lobe split: four separate calls recomputed the same lookups.
                const BsdfEval eval = evaluateBsdfSplit(closure, wiLocalLight);
                const glm::vec3 bsdfValue = eval.total();
                if (eval.pdf > 0.0F &&
                    (bsdfValue.x > 0.0F || bsdfValue.y > 0.0F || bsdfValue.z > 0.0F)) {
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
                    if (!accel.occluded(shadowRay)) {
                        if (bounce == 0) {
                            gShadow = 0.0F;
                        }
                        const float lightPdf2 = lightSample->pdf * lightSample->pdf;
                        const float bsdfPdf2 = eval.pdf * eval.pdf;
                        const float misWeightLight = lightPdf2 / (lightPdf2 + bsdfPdf2);
                        const glm::vec3 common = throughput * lightSample->radiance * lightCos *
                                                  misWeightLight / lightSample->pdf;
                        const glm::vec3 neeContribution = bsdfValue * common;
                        radiance += neeContribution;
                        if (bounce == 0) {
                            // Bounce 0 splits NEE by the lobe that carried it, not sample->type, which names the continuation.
                            directDiffuseAccum += eval.diffuse * common;
                            directSpecularAccum += eval.specular * common;
                            refractionAccum += eval.transmission * common;
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
            break;
        }

        lastBsdfPdf = sample->pdf;
        lastSampleWasDelta = lastBsdfPdf <= 0.0F;
        lastShadingPosition = shading.position;

        const glm::vec3 wiWorld = frame * sample->wiLocal;

        // Geometric-normal-consistency rejection, a stand-in for Schussler et al. 2017: a sample crossing to the wrong side is rejected.
        if (sample->type != LobeType::Transmission) {
            const bool woAbove = glm::dot(woWorld, geoNormal) > 0.0F;
            const bool wiAbove = glm::dot(wiWorld, geoNormal) > 0.0F;
            if (woAbove != wiAbove) {
                break;
            }
        }

        throughput *= sample->throughputWeight;

        if (bounce >= settings.russianRouletteStartBounce) {
            const float continueProb = std::clamp(
                std::max({throughput.x, throughput.y, throughput.z}), settings.rrMinProb,
                settings.rrMaxProb);
            if (sampler.next1D() >= continueProb) {
                break;
            }
            throughput /= continueProb;
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
    }

    return {radiance,           bounce,               gShadow,             gAo,
            gFresnel,           directDiffuseAccum,   indirectDiffuseAccum,
            directSpecularAccum, indirectSpecularAccum, refractionAccum};
}

}  // namespace

PathTraceResult makePathTraceResult(int width, int height) {
    PathTraceResult result;
    for (int i = 0; i < static_cast<int>(pathtracer::debug::AovId::Count); ++i) {
        const auto aov = static_cast<pathtracer::debug::AovId>(i);
        if (const pathtracer::debug::PathTracedLane lane = pathtracer::debug::pathTracedLane(aov)) {
            result.*lane = pathtracer::gfx::makeImage(width, height, pathtracer::debug::aovChannels(aov));
        }
    }
    result.beautyLuminanceM2.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0.0F);
    return result;
}

void renderPathTraced(const Camera& camera, const EmbreeAccel& accel,
                       const std::vector<ShadingTriangle>& shadingTriangles,
                       const std::vector<MeshInstance>& instances,
                       const std::vector<int>& instanceLightIndex, const LightSet& lights,
                       int width, int height, bool showSky, const PathTraceSettings& settings,
                       const std::vector<PathTraceSettings>& perInstanceSettings,
                       std::uint32_t scrambleSeed, int sampleBase, int sampleCount,
                       const std::atomic<std::uint64_t>& generation,
                       std::uint64_t requestedGeneration, ThreadPool& threadPool,
                       pathtracer::debug::PassStats& stats, PathTraceResult& out) {
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    // Constant for the whole pass, so built once: the aspect-taking primaryRay rebuilds it on every one of millions of rays.
    const Camera::ViewBasis basis = camera.viewBasis(aspect);
    const bool wraps = wrapsHorizontally(basis.lens.projection);
    out.wrapsHorizontally = wraps;
    // Derived from the target, not fixed: the interactive scale renders a fraction of the frame, where a 96 px grid is only a few tiles.
    const int tileSize = pathTraceTileSize(width, height, threadPool.threadCount());
    const int tilesX = (width + tileSize - 1) / tileSize;
    const int tilesY = (height + tileSize - 1) / tileSize;

    // One worker owns every output pixel of one tile and traces every pixel within the filter radius: the halo is traced twice.
    const auto renderTile = [&](int tileIndex) {
        const int tileX0 = (tileIndex % tilesX) * tileSize;
        const int tileY0 = (tileIndex / tilesX) * tileSize;
        const int tileX1 = std::min(tileX0 + tileSize, width);
        const int tileY1 = std::min(tileY0 + tileSize, height);

        // Stack-local, not thread_local: zeroed by construction, so the reset boundary is the tile boundary with no bookkeeping.
        pathtracer::debug::RayCounts tileRays;

        // Reused for the worker's life, so a pass allocates nothing; sized for a full tile so the stride stays tileSize.

        // NOLINTNEXTLINE(misc-use-internal-linkage)
        thread_local std::vector<float> accumulator;
        accumulator.assign(static_cast<std::size_t>(tileSize) * tileSize * kTileLanes, 0.0F);

        // Past a wrapping image's left or right edge the halo is the far edge's pixels: on a lat-long both edges are one meridian.
        const int haloX0 = wraps ? tileX0 - kFilterExtent : std::max(tileX0 - kFilterExtent, 0);
        const int haloX1 = wraps ? tileX1 + kFilterExtent : std::min(tileX1 + kFilterExtent, width);
        for (int y = std::max(tileY0 - kFilterExtent, 0);
             y < std::min(tileY1 + kFilterExtent, height); ++y) {
            for (int x = haloX0; x < haloX1; ++x) {
                // The owning pixel: its sample stream and its ray, so a wrapped halo sample is bit-identical to the one its owner traces.
                const int pixelX = (x + width) % width;
                for (int s = 0; s < settings.samplesPerPixel; ++s) {
                    // sampleBase + s is this sample's position in the accumulated sequence: it must advance across passes to stratify.
                    const int sampleIndex = sampleBase + s;
                    Sampler sampler(pixelX, y, sampleIndex, sampleCount, scrambleSeed);
                    const glm::vec2 jitter = sampler.next2D();
                    // Unwrapped, so the splat below measures filter distance across the seam as across any interior pixel boundary.
                    const float filmX = static_cast<float>(x) + jitter.x;
                    const float filmY = static_cast<float>(y) + jitter.y;
                    const float ndcX = (((static_cast<float>(pixelX) + jitter.x) / static_cast<float>(width)) * 2.0F) - 1.0F;
                    // HdrImage row 0 is the top (EXR/glTF convention); NDC +Y is up -- flip.
                    const float ndcY = 1.0F - ((filmY / static_cast<float>(height)) * 2.0F);
                    const std::optional<Ray> primary = camera.primaryRay(basis, ndcX, ndcY);
                    // AO and Fresnel draw from their own stream: taking dimensions from `sampler` would shift every later dimension.
                    Sampler aoSampler(pixelX, y, sampleIndex, sampleCount, scrambleSeed ^ kAoSeedOffset);
                    const glm::vec2 aoSample = aoSampler.next2D();
                    Sampler fresnelSampler(pixelX, y, sampleIndex, sampleCount, scrambleSeed ^ kFresnelSeedOffset);
                    const glm::vec2 fresnelSample = fresnelSampler.next2D();
                    // Nullopt is a fisheye sample outside the image circle: no ray exists, so every lane reads zero for it.
                    const TraceResult trace =
                        primary ? tracePath(*primary, accel, shadingTriangles, instances, instanceLightIndex,
                                            lights, showSky, settings, perInstanceSettings, sampler,
                                            aoSample, fresnelSample, tileRays)
                                : TraceResult{};
                    const std::array<float, kSampleLanes> values{
                        trace.radiance.x,          trace.radiance.y,
                        trace.radiance.z,          static_cast<float>(trace.terminationBounce),
                        trace.shadow,              trace.directDiffuse.x,
                        trace.directDiffuse.y,     trace.directDiffuse.z,
                        trace.indirectDiffuse.x,   trace.indirectDiffuse.y,
                        trace.indirectDiffuse.z,   trace.directSpecular.x,
                        trace.directSpecular.y,    trace.directSpecular.z,
                        trace.indirectSpecular.x,  trace.indirectSpecular.y,
                        trace.indirectSpecular.z,  trace.refraction.x,
                        trace.refraction.y,        trace.refraction.z,
                        trace.ao,                  trace.fresnel.x,
                        trace.fresnel.y,           trace.fresnel.z};

                    // Clipped to this tile: taps outside belong to a neighbouring tile, which traces this same sample itself.
                    const int splatX0 = std::max(tileX0, static_cast<int>(std::ceil(filmX - 0.5F - kFilterRadius)));
                    const int splatX1 = std::min(tileX1 - 1, static_cast<int>(std::floor(filmX - 0.5F + kFilterRadius)));
                    const int splatY0 = std::max(tileY0, static_cast<int>(std::ceil(filmY - 0.5F - kFilterRadius)));
                    const int splatY1 = std::min(tileY1 - 1, static_cast<int>(std::floor(filmY - 0.5F + kFilterRadius)));
                    for (int splatY = splatY0; splatY <= splatY1; ++splatY) {
                        // Separable: the 2D weight is the product of two 1D lookups, so a row's factor hoists out of the inner loop.
                        const float weightY = filterWeight(filmY - (static_cast<float>(splatY) + 0.5F));
                        if (weightY <= 0.0F) {
                            continue;
                        }
                        for (int splatX = splatX0; splatX <= splatX1; ++splatX) {
                            const float weight =
                                weightY * filterWeight(filmX - (static_cast<float>(splatX) + 0.5F));
                            if (weight <= 0.0F) {
                                continue;
                            }
                            float* lanes =
                                accumulator.data() +
                                ((static_cast<std::size_t>(splatY - tileY0) * tileSize) +
                                 static_cast<std::size_t>(splatX - tileX0)) * kTileLanes;
                            for (int lane = 0; lane < kSampleLanes; ++lane) {
                                lanes[lane] += weight * values[lane];
                            }
                            lanes[kSampleLanes] += weight;
                        }
                    }
                }
            }
        }

        for (int y = tileY0; y < tileY1; ++y) {
            for (int x = tileX0; x < tileX1; ++x) {
                const float* lanes = accumulator.data() +
                                      ((static_cast<std::size_t>(y - tileY0) * tileSize) +
                                       static_cast<std::size_t>(x - tileX0)) * kTileLanes;
                // Always positive: a film sample carries weight even where the lens formed no ray, and lands inside the 1.5px support.
                const float invWeight = 1.0F / lanes[kSampleLanes];
                writeTexel(out.beauty, x, y, glm::vec3(lanes[0], lanes[1], lanes[2]) * invWeight);
                writeTexel(out.bounceHeatmap, x, y, lanes[3] * invWeight);
                writeTexel(out.shadow, x, y, lanes[4] * invWeight);
                writeTexel(out.directDiffuse, x, y, glm::vec3(lanes[5], lanes[6], lanes[7]) * invWeight);
                writeTexel(out.indirectDiffuse, x, y, glm::vec3(lanes[8], lanes[9], lanes[10]) * invWeight);
                writeTexel(out.directSpecular, x, y, glm::vec3(lanes[11], lanes[12], lanes[13]) * invWeight);
                writeTexel(out.indirectSpecular, x, y, glm::vec3(lanes[14], lanes[15], lanes[16]) * invWeight);
                writeTexel(out.refraction, x, y, glm::vec3(lanes[17], lanes[18], lanes[19]) * invWeight);
                writeTexel(out.ao, x, y, lanes[20] * invWeight);
                writeTexel(out.fresnel, x, y, glm::vec3(lanes[21], lanes[22], lanes[23]) * invWeight);
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
