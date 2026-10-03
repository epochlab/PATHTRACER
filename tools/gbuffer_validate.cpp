// renderGBuffer against a pixel-centre ray oracle through the same sampling functions, and its line metric against exact edge distances.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/false_color.h"
#include "pathtracer/scene/gbuffer.h"
#include "pathtracer/scene/gbuffer_shading.h"
#include "pathtracer/scene/gltf_loader.h"
#include "check.h"
#include "pathtracer/scene/ray_types.h"
#include "pathtracer/scene/shading_scene.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using namespace pathtracer::scene;  // NOLINT(google-build-using-namespace) -- tool-local, mirrors embree_validate.cpp

constexpr int kWidth = 96;
constexpr int kHeight = 96;
constexpr float kAspect = static_cast<float>(kWidth) / static_cast<float>(kHeight);
constexpr int kTriangleCount = 150;
constexpr int kMaterialCount = 4;
// Depth span the triangle centres are drawn over; the lookahead check derives its horizon from it, so one outside would blind the check.
constexpr float kSceneNearZ = 1.0F;
constexpr float kSceneFarZ = 16.0F;
constexpr float kEps = std::numeric_limits<float>::epsilon();
// camera_validate's budget for a dozen rounded operations, applied to each stage of a reprojection or a line distance.
constexpr float kClosedFormUlps = 16.0F;
// The AOV's documented on-screen line width (gbuffer.cpp), the threshold every line classification compares against.
constexpr float kLineThicknessPx = 1.0F;
// Samples per edge for the exact-distance oracle: a 96 px image keeps chords under 0.1 px, whose sag is far inside the excluded band.
constexpr int kEdgeSamples = 1024;
constexpr Camera::FilmBack kFilmBack{36.0F, 24.0F};

// The neutral default material, varying only the two slots the G-buffer AOVs under test read.
Material makeMaterial(glm::vec3 baseColor, float roughness) {
    return Material{.baseColor = baseColor, .roughness = roughness};
}

glm::vec4 tangentFor(const glm::vec3& normal) {
    const glm::vec3 up = std::fabs(normal.y) < 0.99F ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::vec3(1.0F, 0.0F, 0.0F);
    return glm::vec4(glm::normalize(glm::cross(up, normal)), 1.0F);
}

// Random triangles overlapping in depth, dense enough near z=-1 that a camera inside the cluster has some behind its near clip.
std::vector<ShadingTriangle> makeSyntheticTriangles(std::mt19937& rng) {
    std::uniform_real_distribution<float> centerXY(-8.0F, 8.0F);
    std::uniform_real_distribution<float> centerZ(-kSceneFarZ, -kSceneNearZ);
    std::uniform_real_distribution<float> offset(-1.5F, 1.5F);

    std::vector<ShadingTriangle> triangles;
    triangles.reserve(kTriangleCount);
    for (int i = 0; i < kTriangleCount; ++i) {
        const glm::vec3 center(centerXY(rng), centerXY(rng), centerZ(rng));
        const glm::vec3 p0 = center + glm::vec3(offset(rng), offset(rng), offset(rng) * 0.1F);
        const glm::vec3 p1 = center + glm::vec3(offset(rng), offset(rng), offset(rng) * 0.1F);
        const glm::vec3 p2 = center + glm::vec3(offset(rng), offset(rng), offset(rng) * 0.1F);
        const glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
        if (!std::isfinite(normal.x)) {
            continue;  // degenerate (near-collinear) draw -- skip, next iteration redraws a fresh center
        }
        const glm::vec4 tangent = tangentFor(normal);
        ShadingTriangle tri;
        tri.v0 = ShadingVertex{p0, normal, glm::vec2(0.0F, 0.0F), tangent};
        tri.v1 = ShadingVertex{p1, normal, glm::vec2(1.0F, 0.0F), tangent};
        tri.v2 = ShadingVertex{p2, normal, glm::vec2(0.0F, 1.0F), tangent};
        tri.instanceIndex = i % kMaterialCount;
        triangles.push_back(tri);
    }
    return triangles;
}

std::vector<Triangle> worldTrianglesOf(const std::vector<ShadingTriangle>& shadingTriangles) {
    std::vector<Triangle> triangles;
    triangles.reserve(shadingTriangles.size());
    for (const ShadingTriangle& tri : shadingTriangles) {
        triangles.push_back(Triangle{tri.v0.position, tri.v1.position, tri.v2.position});
    }
    return triangles;
}

glm::vec3 texelAt(const pathtracer::gfx::HdrImage& image, int x, int y) {
    return image.rgb((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) + static_cast<std::size_t>(x));
}

// renderGBuffer's own pixel-centre expressions, so the oracle's ray is the producer's bit for bit.
glm::vec2 pixelNdc(int x, int y) {
    const glm::vec2 pixelsPerNdc(0.5F * static_cast<float>(kWidth), -0.5F * static_cast<float>(kHeight));
    return {((static_cast<float>(x) + 0.5F) / pixelsPerNdc.x) - 1.0F, 1.0F + ((static_cast<float>(y) + 0.5F) / pixelsPerNdc.y)};
}

glm::dvec2 ndcToPixel(glm::dvec2 ndc) {
    return {(ndc.x + 1.0) * 0.5 * kWidth, (1.0 - ndc.y) * 0.5 * kHeight};
}

// The material-independent settings every check renders with: the AOVs under test are geometric, so these need only be valid, not varied.
PathTraceSettings makeTestSettings() {
    PathTraceSettings settings{};
    settings.samplesPerPixel = 1;
    settings.maxBounces = 0;
    settings.russianRouletteStartBounce = 1;
    // Derived from the scene's depth span, not profile.json: the horizon must fall inside it or one remap branch carries no pixels.
    settings.lookaheadDistance = 0.5F * (kSceneNearZ + kSceneFarZ);
    settings.bumpStrength = 1.0F;
    settings.roughnessMin = 0.045F;
    settings.roughnessMax = 1.0F;
    settings.diffuseColour = glm::vec3(1.0F);
    settings.ior = 1.5F;
    settings.transmissionFactor = 0.0F;
    settings.metallicFactor = 0.2F;
    settings.roughnessFactor = 1.0F;
    return settings;
}

// One scene and its accel, rebuilt per check: construction is milliseconds, and a check building its own world orders no other.
struct Fixture {
    std::vector<ShadingTriangle> shadingTriangles;
    std::optional<EmbreeAccel> accel;
    std::vector<MeshInstance> instances;
    std::vector<PathTraceSettings> perInstanceSettings;
    std::vector<AabbBounds> instanceBounds;
    ThreadPool threadPool;

    [[nodiscard]] GBuffer render(const Camera& camera, const Camera& previous) {
        GBuffer gbuffer;
        renderGBuffer(camera, previous, *accel, shadingTriangles, instances, perInstanceSettings, instanceBounds, kWidth, kHeight,
                      threadPool, gbuffer);
        return gbuffer;
    }
};

std::unique_ptr<Fixture> makeFixture(std::vector<ShadingTriangle> triangles, int instanceCount) {
    auto fixture = std::make_unique<Fixture>();
    fixture->shadingTriangles = std::move(triangles);
    fixture->accel = EmbreeAccel::build(worldTrianglesOf(fixture->shadingTriangles));
    const std::array<glm::vec3, kMaterialCount> colours{
        {{0.8F, 0.2F, 0.2F}, {0.2F, 0.8F, 0.2F}, {0.2F, 0.2F, 0.8F}, {0.8F, 0.8F, 0.2F}}};
    for (int i = 0; i < instanceCount; ++i) {
        const auto slot = static_cast<std::size_t>(i % kMaterialCount);
        fixture->instances.push_back(
            MeshInstance{makeMaterial(colours[slot], 0.2F + (0.25F * static_cast<float>(slot))), glm::mat4(1.0F), ""});
    }
    fixture->perInstanceSettings.assign(fixture->instances.size(), makeTestSettings());
    fixture->instanceBounds = computeInstanceBounds(fixture->shadingTriangles, instanceCount);
    return fixture;
}

std::unique_ptr<Fixture> makeClusterFixture(const tools::check::Context& ctx) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    return makeFixture(makeSyntheticTriangles(rng), kMaterialCount);
}

Camera straightOnCamera() {
    return Camera(glm::vec3(0.0F), 0.0F, 0.0F, kFilmBack, 35.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F, 100.0F);
}

// The angled pose: off every axis and inside the cluster's depth span, so a frame holds both hits and misses.
Camera angledCamera(glm::vec3 offset = glm::vec3(0.0F), float yawOffset = 0.0F, float pitchOffset = 0.0F, float focalMm = 35.0F) {
    return Camera(glm::vec3(3.0F, 2.0F, 1.0F) + offset, 20.0F + yawOffset, -10.0F + pitchOffset, kFilmBack, focalMm, 0.1F,
                  100.0F, 2.8F, 1.0F / 125.0F, 100.0F);
}

// An equidistant fisheye inside the cluster: at 220 degrees it images triangles behind its own optical plane, where planar z < 0.
Camera fisheyeCamera(float fieldOfViewDegrees, glm::vec3 offset = glm::vec3(0.0F), float yawOffset = 0.0F) {
    return Camera(glm::vec3(0.5F, -0.5F, -8.0F) + offset, 15.0F + yawOffset, 5.0F, kFilmBack, 8.0F, 0.1F, 100.0F, 2.8F,
                  1.0F / 125.0F, 100.0F, Lens{LensProjection::FisheyePolynomial, {}, fieldOfViewDegrees});
}

// The lat-long from the same spot: it sees the whole cluster around it, and behind it the environment through the seam.
Camera omnidirectionalCamera(glm::vec3 offset = glm::vec3(0.0F), float yawOffset = 0.0F) {
    return Camera(glm::vec3(0.5F, -0.5F, -8.0F) + offset, 15.0F + yawOffset, 5.0F, kFilmBack, 8.0F, 0.1F, 100.0F, 2.8F,
                  1.0F / 125.0F, 100.0F, Lens{LensProjection::Omnidirectional});
}

// Smallest singular value of the per-pixel direction map: radians per pixel along its least-stretched axis.
float minDirPerPixel(const glm::mat2x3& dirPerPixel) {
    const float a = glm::dot(dirPerPixel[0], dirPerPixel[0]);
    const float b = glm::dot(dirPerPixel[0], dirPerPixel[1]);
    const float c = glm::dot(dirPerPixel[1], dirPerPixel[1]);
    const float half = 0.5F * (a + c);
    return std::sqrt(half - std::sqrt((0.25F * (a - c) * (a - c)) + (b * b)));
}

std::optional<glm::mat2x3> dirPerPixelAt(const Camera& camera, const Camera::ViewBasis& basis, glm::vec2 ndc) {
    const std::optional<Camera::RayDifferential> differential = camera.primaryRayDifferential(basis, ndc.x, ndc.y);
    if (!differential) {
        return std::nullopt;
    }
    const glm::vec2 pixelsPerNdc(0.5F * static_cast<float>(kWidth), -0.5F * static_cast<float>(kHeight));
    return differential->dirPerNdc * glm::mat2(1.0F / pixelsPerNdc.x, 0.0F, 0.0F, 1.0F / pixelsPerNdc.y);
}

struct PoseCounts {
    int hits = 0;
    int misses = 0;
    int unimaged = 0;
    int fieldMismatches = 0;
    int reprojectionFailures = 0;
    double worstReprojection = 0.0;  // in units of the per-pixel budget
};

// Every lane bar wireframe at a pixel without a surface: zero, IOR's -1 background, and zero motion between identical views.
bool isBackground(const GBuffer& g, int x, int y) {
    const std::array<const pathtracer::gfx::HdrImage*, 13> zeroLanes{
        &g.depth, &g.lookahead, &g.worldPos, &g.uv, &g.motionVector, &g.normal, &g.geomNormal, &g.albedo,
        &g.metallic, &g.roughness, &g.tangent, &g.objectId, &g.alpha};
    return std::all_of(zeroLanes.begin(), zeroLanes.end(), [&](const auto* lane) { return texelAt(*lane, x, y) == glm::vec3(0.0F); }) &&
           texelAt(g.iorAov, x, y).x == -1.0F;
}

// The hit's lanes against the oracle's own sampling calls, compared bitwise: both resolve the same ray through the same functions.
bool matchesOracle(const GBuffer& g, int x, int y, const Hit& hit, const Fixture& fixture) {
    const ShadingTriangle& triangle = fixture.shadingTriangles[static_cast<std::size_t>(hit.triangleIndex)];
    const auto instance = static_cast<std::size_t>(triangle.instanceIndex);
    const Material& material = fixture.instances[instance].material;
    const PathTraceSettings& settings = fixture.perInstanceSettings[instance];
    const ShadingVertex shading = interpolateShading(triangle, hit.u, hit.v);
    const ShadingFrame frame = buildShadingFrame(shading, material, settings);
    const BsdfParams params = resolveBsdfParams(material, shading.uv, shading.colour, settings, std::nullopt);
    return texelAt(g.depth, x, y).x == hit.t &&
           texelAt(g.lookahead, x, y).x == std::clamp(1.0F - (hit.t / settings.lookaheadDistance), 0.0F, 1.0F) &&
           texelAt(g.worldPos, x, y) == shading.position && glm::vec2(texelAt(g.uv, x, y)) == glm::fract(shading.uv) &&
           texelAt(g.normal, x, y) == frame[2] && texelAt(g.geomNormal, x, y) == glm::normalize(shading.normal) &&
           texelAt(g.albedo, x, y) == params.baseColor && texelAt(g.metallic, x, y).x == params.metallic &&
           texelAt(g.roughness, x, y).x == params.roughness && texelAt(g.tangent, x, y) == frame[0] &&
           texelAt(g.objectId, x, y) == falseColorForId(triangle.instanceIndex) && texelAt(g.alpha, x, y).x == 1.0F &&
           texelAt(g.iorAov, x, y).x == settings.ior && glm::vec2(texelAt(g.motionVector, x, y)) == glm::vec2(0.0F);
}

// worldPos reprojected through Camera::project lands on its own pixel centre, within the hit's rounding over the incidence it is seen at.
double reprojectionRatio(const Camera& camera, const Camera::ViewBasis& basis, const ShadingTriangle& triangle, const Ray& ray,
                         glm::vec3 worldPos, glm::vec2 ndc) {
    const std::optional<glm::vec2> back = camera.project(basis, glm::vec4(worldPos, 1.0F));
    const std::optional<glm::mat2x3> dirPerPixel = dirPerPixelAt(camera, basis, ndc);
    if (!back || !dirPerPixel) {
        return std::numeric_limits<double>::infinity();
    }
    const glm::vec3 face =
        glm::normalize(glm::cross(triangle.v1.position - triangle.v0.position, triangle.v2.position - triangle.v0.position));
    const float extent = std::max({glm::length(triangle.v0.position - ray.origin), glm::length(triangle.v1.position - ray.origin),
                                   glm::length(triangle.v2.position - ray.origin)});
    const float range = glm::length(worldPos - ray.origin);
    // A barycentric rounding of k ulps of the extent moves the point along the surface by 1/cos(incidence), off the ray by its sine.
    const float angularBudget = (kClosedFormUlps * kEps * extent / (range * std::abs(glm::dot(face, ray.dir)))) +
                                (kClosedFormUlps * kEps * std::max(1.0F, glm::length(ndc)));
    const double errorPx = glm::length(ndcToPixel(glm::dvec2(*back)) - ndcToPixel(glm::dvec2(ndc)));
    return errorPx / (angularBudget / minDirPerPixel(*dirPerPixel));
}

PoseCounts checkPose(Fixture& fixture, const Camera& camera) {
    const GBuffer g = fixture.render(camera, camera);
    const Camera::ViewBasis basis = camera.viewBasis(kAspect);
    PoseCounts counts;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const glm::vec2 ndc = pixelNdc(x, y);
            const std::optional<Ray> ray = camera.primaryRay(basis, ndc.x, ndc.y);
            const std::optional<Hit> hit = ray ? fixture.accel->intersect(*ray) : std::nullopt;
            if (!hit) {
                ++(ray ? counts.misses : counts.unimaged);
                counts.fieldMismatches += isBackground(g, x, y) ? 0 : 1;
                continue;
            }
            ++counts.hits;
            counts.fieldMismatches += matchesOracle(g, x, y, *hit, fixture) ? 0 : 1;
            const ShadingTriangle& triangle = fixture.shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)];
            const double ratio = reprojectionRatio(camera, basis, triangle, *ray, texelAt(g.worldPos, x, y), ndc);
            counts.worstReprojection = std::max(counts.worstReprojection, ratio);
            counts.reprojectionFailures += ratio <= 1.0 ? 0 : 1;
        }
    }
    return counts;
}

void runPose(tools::check::Context& ctx, const char* name, const Camera& camera, bool expectUnimaged) {
    const std::unique_ptr<Fixture> fixture = makeClusterFixture(ctx);
    ctx.plan(4);
    const PoseCounts c = checkPose(*fixture, camera);
    std::cout << "gbuffer_validate: " << name << " -- " << c.hits << " hits, " << c.misses << " misses, " << c.unimaged
              << " outside the image circle, " << c.fieldMismatches << " lane mismatches, worst reprojection "
              << c.worstReprojection << " of budget\n";
    PT_EXPECT(ctx, c.hits > 0 && c.misses > 0, std::string(name) + ": the pose must show both surface and environment");
    PT_EXPECT(ctx, (c.unimaged > 0) == expectUnimaged, std::string(name) + ": unexpected image-circle coverage");
    PT_EXPECT(ctx, c.fieldMismatches == 0,
              std::string(name) + ": " + std::to_string(c.fieldMismatches) + " pixels departed from the ray oracle");
    PT_EXPECT(ctx, c.reprojectionFailures == 0,
              std::string(name) + ": " + std::to_string(c.reprojectionFailures) + " worldPos texels reprojected off their pixel");
}

PT_CHECK(gbuffer_pose_straight_on, Fast, Exact) {
    runPose(ctx, "straightOn", straightOnCamera(), false);
}

// Yawed and pitched off-axis, separating correct interpolation from one that works only when screen gradients are axis-aligned.
PT_CHECK(gbuffer_pose_angled, Fast, Exact) {
    runPose(ctx, "angled", angledCamera(), false);
}

// Near clip 5.0 puts it mid-cluster, so the ray's tMin skips surfaces closer than it.
PT_CHECK(gbuffer_pose_near_clip, Fast, Exact) {
    runPose(ctx, "nearClip",
            Camera(glm::vec3(0.0F), 5.0F, 5.0F, kFilmBack, 35.0F, 5.0F, 100.0F, 2.8F, 1.0F / 125.0F, 100.0F), false);
}

// A 180-degree circle wider than the gate's height: the corners lie outside it and must read as background.
PT_CHECK(gbuffer_pose_fisheye_180, Fast, Exact) {
    runPose(ctx, "fisheye180", fisheyeCamera(180.0F), true);
}

// Past 180 degrees the lens sees behind its optical plane: depth stays positive because it is the ray distance.
PT_CHECK(gbuffer_pose_fisheye_220, Fast, Exact) {
    runPose(ctx, "fisheye220", fisheyeCamera(220.0F), true);
}

// Every pixel is imaged and the pole rows' differentials are the most anisotropic of any lens, so reprojection is tested hardest there.
PT_CHECK(gbuffer_pose_omnidirectional, Fast, Exact) {
    runPose(ctx, "omnidirectional", omnidirectionalCamera(), false);
}

bool isBoxColorOf(glm::vec3 c, int instanceIndex) {
    return c == falseColorForId(instanceIndex);
}

bool isWireframeColor(glm::vec3 c) {
    return c == glm::vec3(1.0F);
}

// One 3D segment sampled and projected to pixels with each sample's range; a sample with no image breaks the polyline there.
struct ProjectedSample {
    std::optional<glm::dvec2> pixel;
    double range;
};

std::vector<ProjectedSample> projectSegment(const Camera& camera, const Camera::ViewBasis& basis, glm::vec3 a, glm::vec3 b) {
    std::vector<ProjectedSample> samples(kEdgeSamples + 1);
    std::optional<double> previousX;
    for (int i = 0; i <= kEdgeSamples; ++i) {
        const glm::vec3 point = glm::mix(a, b, static_cast<float>(i) / static_cast<float>(kEdgeSamples));
        const std::optional<glm::vec2> ndc = camera.project(basis, glm::vec4(point, 1.0F));
        std::optional<glm::dvec2> pixel = ndc ? std::optional(ndcToPixel(glm::dvec2(*ndc))) : std::nullopt;
        // On a lat-long, each sample joins its predecessor the short way round, so the polyline lives on the universal cover.
        if (pixel && previousX && wrapsHorizontally(basis.lens.projection)) {
            pixel->x = *previousX + std::remainder(pixel->x - *previousX, static_cast<double>(kWidth));
        }
        previousX = pixel ? std::optional(pixel->x) : std::nullopt;
        samples[static_cast<std::size_t>(i)] = {pixel, glm::length(glm::dvec3(point - camera.position()))};
    }
    return samples;
}

// The x offsets a pixel stands for: on a lat-long, its copies one turn either side meet any unwrapped polyline that winds past the seam.
std::vector<double> periodicShiftsPx(const Camera::ViewBasis& basis) {
    const auto turn = static_cast<double>(kWidth);
    return wrapsHorizontally(basis.lens.projection) ? std::vector<double>{-turn, 0.0, turn} : std::vector<double>{0.0};
}

// Exact pixel distance from p to the polyline's chords whose nearer end lies within [rangeMin, rangeMax], infinity if none does.
double polylineDistancePx(const std::vector<ProjectedSample>& samples, glm::dvec2 p, double rangeMin, double rangeMax) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i + 1 < samples.size(); ++i) {
        if (!samples[i].pixel || !samples[i + 1].pixel) {
            continue;
        }
        const glm::dvec2 a = *samples[i].pixel;
        const glm::dvec2 ab = *samples[i + 1].pixel - a;
        const double t = glm::dot(ab, ab) > 0.0 ? std::clamp(glm::dot(p - a, ab) / glm::dot(ab, ab), 0.0, 1.0) : 0.0;
        const double range = glm::mix(samples[i].range, samples[i + 1].range, t);
        if (range >= rangeMin && range <= rangeMax) {
            best = std::min(best, glm::length(p - (a + (t * ab))));
        }
    }
    return best;
}

// The first-order model's relative error at distance d, d/2 * |dJ/dp| / sigma_min(J), by a one-pixel central difference of J.
double modelBand(const Camera& camera, const Camera::ViewBasis& basis, int x, int y) {
    const std::optional<glm::mat2x3> centre = dirPerPixelAt(camera, basis, pixelNdc(x, y));
    const glm::vec2 stepX = pixelNdc(x + 1, y) - pixelNdc(x, y);
    const glm::vec2 stepY = pixelNdc(x, y + 1) - pixelNdc(x, y);
    const std::optional<glm::mat2x3> px = dirPerPixelAt(camera, basis, pixelNdc(x, y) + stepX);
    const std::optional<glm::mat2x3> mx = dirPerPixelAt(camera, basis, pixelNdc(x, y) - stepX);
    const std::optional<glm::mat2x3> py = dirPerPixelAt(camera, basis, pixelNdc(x, y) + stepY);
    const std::optional<glm::mat2x3> my = dirPerPixelAt(camera, basis, pixelNdc(x, y) - stepY);
    if (!centre || !px || !mx || !py || !my) {
        return std::numeric_limits<double>::infinity();
    }
    const auto norm = [](const glm::mat2x3& m) { return std::sqrt(glm::dot(m[0], m[0]) + glm::dot(m[1], m[1])); };
    const double rate = 0.5 * (norm(*px - *mx) + norm(*py - *my));
    return (0.5 * kLineThicknessPx * rate / minDirPerPixel(*centre)) + (kClosedFormUlps * kEps);
}

struct LineCounts {
    int lines = 0;
    int agreed = 0;
    int excluded = 0;
    int mismatches = 0;
};

// Classification against the exact distance, pixels inside the model's error band of the threshold excluded, and counted.
void classify(LineCounts& counts, bool drawn, double exactPx, double band) {
    if (std::abs(exactPx - kLineThicknessPx) <= kLineThicknessPx * band) {
        ++counts.excluded;
        return;
    }
    const bool expected = exactPx < kLineThicknessPx;
    counts.lines += expected ? 1 : 0;
    (drawn == expected ? counts.agreed : counts.mismatches) += 1;
}

// Every hit pixel's wire flag against the exact distance to its triangle's projected boundary. No boxes, so white is the mesh alone.
LineCounts checkWireframeDistances(const Camera& camera, tools::check::Context& ctx) {
    std::unique_ptr<Fixture> fixture = makeClusterFixture(ctx);
    fixture->instanceBounds.clear();
    const GBuffer g = fixture->render(camera, camera);
    const Camera::ViewBasis basis = camera.viewBasis(kAspect);
    const std::vector<double> shifts = periodicShiftsPx(basis);
    std::vector<std::array<std::vector<ProjectedSample>, 3>> edges(fixture->shadingTriangles.size());
    LineCounts counts;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const glm::vec2 ndc = pixelNdc(x, y);
            const std::optional<Ray> ray = camera.primaryRay(basis, ndc.x, ndc.y);
            const std::optional<Hit> hit = ray ? fixture->accel->intersect(*ray) : std::nullopt;
            if (!hit) {
                continue;
            }
            const auto index = static_cast<std::size_t>(hit->triangleIndex);
            const ShadingTriangle& tri = fixture->shadingTriangles[index];
            if (edges[index][0].empty()) {
                edges[index] = {projectSegment(camera, basis, tri.v0.position, tri.v1.position),
                                projectSegment(camera, basis, tri.v1.position, tri.v2.position),
                                projectSegment(camera, basis, tri.v2.position, tri.v0.position)};
            }
            const glm::dvec2 p(x + 0.5, y + 0.5);
            const double unbounded = std::numeric_limits<double>::infinity();
            double exactPx = unbounded;
            for (const std::vector<ProjectedSample>& edge : edges[index]) {
                for (const double shift : shifts) {
                    exactPx = std::min(exactPx, polylineDistancePx(edge, p + glm::dvec2(shift, 0.0), 0.0, unbounded));
                }
            }
            classify(counts, isWireframeColor(texelAt(g.wireframe, x, y)), exactPx, modelBand(camera, basis, x, y));
        }
    }
    return counts;
}

void expectLines(tools::check::Context& ctx, const std::string& name, const LineCounts& c) {
    std::cout << "gbuffer_validate: " << name << " -- " << c.lines << " line pixels, " << c.agreed << " agreed, " << c.excluded
              << " inside the model band, " << c.mismatches << " mismatches\n";
    PT_EXPECT(ctx, c.lines > 0, name + ": no pixel lay within the line thickness, so the classification went unexercised");
    PT_EXPECT(ctx, c.mismatches == 0, name + ": " + std::to_string(c.mismatches) + " pixels disagreed with the exact edge distance");
}

PT_CHECK(wireframe_matches_the_exact_edge_distance, Fast, Exact) {
    ctx.plan(10);
    expectLines(ctx, "wireframe straightOn", checkWireframeDistances(straightOnCamera(), ctx));
    expectLines(ctx, "wireframe angled", checkWireframeDistances(angledCamera(), ctx));
    expectLines(ctx, "wireframe fisheye180", checkWireframeDistances(fisheyeCamera(180.0F), ctx));
    expectLines(ctx, "wireframe fisheye220", checkWireframeDistances(fisheyeCamera(220.0F), ctx));
    expectLines(ctx, "wireframe omnidirectional", checkWireframeDistances(omnidirectionalCamera(), ctx));
}

// A single tiny (non-degenerate) triangle near `center` -- plants a known point into its instance's bounds without a real visible surface.
ShadingTriangle makeTinyTriangle(glm::vec3 center, float eps, int instanceIndex) {
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    const glm::vec4 tangent(1.0F, 0.0F, 0.0F, 1.0F);
    ShadingTriangle tri;
    tri.v0 = ShadingVertex{center, normal, glm::vec2(0.0F, 0.0F), tangent};
    tri.v1 = ShadingVertex{center + glm::vec3(eps, 0.0F, 0.0F), normal, glm::vec2(1.0F, 0.0F), tangent};
    tri.v2 = ShadingVertex{center + glm::vec3(0.0F, eps, 0.0F), normal, glm::vec2(0.0F, 1.0F), tangent};
    tri.instanceIndex = instanceIndex;
    return tri;
}

// The box's 12 edges in gbuffer.cpp's corner order.
std::array<std::array<glm::vec3, 2>, 12> boxEdgesOf(const AabbBounds& box) {
    const std::array<glm::vec3, 8> c{glm::vec3(box.min.x, box.min.y, box.min.z), glm::vec3(box.max.x, box.min.y, box.min.z),
                                     glm::vec3(box.max.x, box.max.y, box.min.z), glm::vec3(box.min.x, box.max.y, box.min.z),
                                     glm::vec3(box.min.x, box.min.y, box.max.z), glm::vec3(box.max.x, box.min.y, box.max.z),
                                     glm::vec3(box.max.x, box.max.y, box.max.z), glm::vec3(box.min.x, box.max.y, box.max.z)};
    return {{{c[0], c[1]}, {c[1], c[2]}, {c[2], c[3]}, {c[3], c[0]}, {c[4], c[5]}, {c[5], c[6]},
             {c[6], c[7]}, {c[7], c[4]}, {c[0], c[4]}, {c[1], c[5]}, {c[2], c[6]}, {c[3], c[7]}}};
}

// Largest range change per pixel along any sampled edge: how far the range moves when the nearest point slides one pixel along it.
double rangePerPixel(const std::vector<std::vector<ProjectedSample>>& edges) {
    double rate = 0.0;
    for (const std::vector<ProjectedSample>& edge : edges) {
        for (std::size_t i = 0; i + 1 < edge.size(); ++i) {
            if (edge[i].pixel && edge[i + 1].pixel) {
                const double px = glm::length(*edge[i + 1].pixel - *edge[i].pixel);
                rate = px > 0.0 ? std::max(rate, std::abs(edge[i + 1].range - edge[i].range) / px) : rate;
            }
        }
    }
    return rate;
}

// Every pixel's box flag against the exact distance to the 12 projected edges, visible from the ray's tMin to its first surface.
LineCounts checkBoxDistances(const Camera& camera, const AabbBounds& box, std::vector<ShadingTriangle> occluder) {
    std::unique_ptr<Fixture> fixture = makeFixture(std::move(occluder), 1);
    fixture->instanceBounds = {box};
    const GBuffer g = fixture->render(camera, camera);
    const Camera::ViewBasis basis = camera.viewBasis(kAspect);
    std::vector<std::vector<ProjectedSample>> edges;
    for (const std::array<glm::vec3, 2>& edge : boxEdgesOf(box)) {
        edges.push_back(projectSegment(camera, basis, edge[0], edge[1]));
    }
    // The model's nearest point may sit up to a line width along the edge from the exact one, moving its range by this much.
    const double rangeBand = kLineThicknessPx * rangePerPixel(edges);
    // Each polyline's pixel bounds: a pixel farther than the line width outside them cannot be near that edge, an exact skip.
    std::vector<std::pair<glm::dvec2, glm::dvec2>> bounds;
    for (const std::vector<ProjectedSample>& edge : edges) {
        glm::dvec2 lo(std::numeric_limits<double>::infinity());
        glm::dvec2 hi(-std::numeric_limits<double>::infinity());
        for (const ProjectedSample& sample : edge) {
            lo = sample.pixel ? glm::min(lo, *sample.pixel) : lo;
            hi = sample.pixel ? glm::max(hi, *sample.pixel) : hi;
        }
        bounds.emplace_back(lo - glm::dvec2(kLineThicknessPx), hi + glm::dvec2(kLineThicknessPx));
    }
    const std::vector<double> shifts = periodicShiftsPx(basis);
    const auto nearestPx = [&edges, &bounds, &shifts](glm::dvec2 p, double rangeMin, double rangeMax) {
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t e = 0; e < edges.size(); ++e) {
            for (const double shift : shifts) {
                const glm::dvec2 q = p + glm::dvec2(shift, 0.0);
                if (glm::all(glm::greaterThanEqual(q, bounds[e].first)) && glm::all(glm::lessThanEqual(q, bounds[e].second))) {
                    best = std::min(best, polylineDistancePx(edges[e], q, rangeMin, rangeMax));
                }
            }
        }
        return best;
    };
    LineCounts counts;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const glm::vec2 ndc = pixelNdc(x, y);
            const std::optional<Ray> ray = camera.primaryRay(basis, ndc.x, ndc.y);
            if (!ray) {
                continue;
            }
            const std::optional<Hit> hit = fixture->accel->intersect(*ray);
            const double rangeMax = hit ? hit->t : ray->tMax;
            const glm::dvec2 p(x + 0.5, y + 0.5);
            // A pixel whose verdict flips within the range band is decided by the occluder's depth, not by the line metric under test.
            if (hit && (nearestPx(p, ray->tMin, rangeMax - rangeBand) < kLineThicknessPx) !=
                           (nearestPx(p, ray->tMin, rangeMax + rangeBand) < kLineThicknessPx)) {
                ++counts.excluded;
                continue;
            }
            classify(counts, isBoxColorOf(texelAt(g.wireframe, x, y), 0), nearestPx(p, ray->tMin, rangeMax),
                     modelBand(camera, basis, x, y));
        }
    }
    return counts;
}

// A plane through the box, tilted so its depth crosses the edges' depths: edges show only up to where it occludes them.
std::vector<ShadingTriangle> obliqueOccluder() {
    const glm::vec3 normal = glm::normalize(glm::vec3(0.6F, 0.0F, 1.0F));
    const glm::vec4 tangent = tangentFor(normal);
    // The plane 0.6 x + z = -9, two triangles spanning the view.
    const auto vertex = [&](float x, float y) {
        return ShadingVertex{glm::vec3(x, y, -9.0F - (0.6F * x)), normal, glm::vec2(x, y), tangent};
    };
    return {ShadingTriangle{vertex(-20.0F, -20.0F), vertex(20.0F, -20.0F), vertex(20.0F, 20.0F), 0},
            ShadingTriangle{vertex(-20.0F, -20.0F), vertex(20.0F, 20.0F), vertex(-20.0F, 20.0F), 0}};
}

// Outside the box, inside it (edges crossing behind the eye image as half-lines), through a fisheye and a lat-long, and half occluded.
PT_CHECK(box_edges_match_the_exact_edge_distance, Fast, Exact) {
    ctx.plan(10);
    const AabbBounds outside{glm::vec3(-2.0F, -1.5F, -12.0F), glm::vec3(2.5F, 1.0F, -6.0F)};
    expectLines(ctx, "box outside", checkBoxDistances(angledCamera(), outside, {}));
    expectLines(ctx, "box around the eye",
                checkBoxDistances(straightOnCamera(), {glm::vec3(-1.0F, -0.5F, -6.0F), glm::vec3(0.8F, 1.0F, 3.0F)}, {}));
    expectLines(ctx, "box around a fisheye",
                checkBoxDistances(fisheyeCamera(220.0F), {glm::vec3(-3.0F, -4.0F, -12.0F), glm::vec3(4.0F, 2.0F, -5.0F)}, {}));
    // The eye inside the box, so its edges wrap the whole sphere, through the seam and past both poles' rows.
    expectLines(ctx, "box around a lat-long",
                checkBoxDistances(omnidirectionalCamera(), {glm::vec3(-3.0F, -4.0F, -12.0F), glm::vec3(4.0F, 2.0F, -5.0F)}, {}));
    expectLines(ctx, "box cut by an oblique plane", checkBoxDistances(straightOnCamera(), outside, obliqueOccluder()));
}

// A flat quad (2 triangles) in the XY plane, centered at `center`.
std::array<ShadingTriangle, 2> makeQuad(float halfExtent, glm::vec3 center, int instanceIndex) {
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    const glm::vec4 tangent(1.0F, 0.0F, 0.0F, 1.0F);
    const glm::vec3 a = center + glm::vec3(-halfExtent, -halfExtent, 0.0F);
    const glm::vec3 b = center + glm::vec3(halfExtent, -halfExtent, 0.0F);
    const glm::vec3 c = center + glm::vec3(halfExtent, halfExtent, 0.0F);
    const glm::vec3 d = center + glm::vec3(-halfExtent, halfExtent, 0.0F);
    ShadingTriangle t0;
    t0.v0 = ShadingVertex{a, normal, glm::vec2(0.0F, 0.0F), tangent};
    t0.v1 = ShadingVertex{b, normal, glm::vec2(1.0F, 0.0F), tangent};
    t0.v2 = ShadingVertex{c, normal, glm::vec2(1.0F, 1.0F), tangent};
    t0.instanceIndex = instanceIndex;
    ShadingTriangle t1;
    t1.v0 = ShadingVertex{a, normal, glm::vec2(0.0F, 0.0F), tangent};
    t1.v1 = ShadingVertex{c, normal, glm::vec2(1.0F, 1.0F), tangent};
    t1.v2 = ShadingVertex{d, normal, glm::vec2(0.0F, 1.0F), tangent};
    t1.instanceIndex = instanceIndex;
    return {t0, t1};
}

// Screen-space extent of one instance's box pixels: count plus the x range they occupy, inverted when the count is 0.
struct BoxPixels {
    int count;
    int minX;
    int maxX;
};

BoxPixels boxPixelsOf(const GBuffer& g, int instanceIndex) {
    BoxPixels found{0, kWidth, -1};
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            if (!isBoxColorOf(texelAt(g.wireframe, x, y), instanceIndex)) {
                continue;
            }
            ++found.count;
            found.minX = std::min(found.minX, x);
            found.maxX = std::max(found.maxX, x);
        }
    }
    return found;
}

// Box edges are range-tested against scene geometry, so a solid occluder in front should hide the edges behind it.
PT_CHECK(bounding_box_occlusion, Fast, Exact) {
    ctx.plan(2);
    const ShadingTriangle farMarker = makeTinyTriangle(glm::vec3(-0.5F, -0.5F, -15.0F), 0.05F, 0);
    const ShadingTriangle nearMarker = makeTinyTriangle(glm::vec3(0.5F, 0.5F, -5.0F), 0.05F, 0);
    // Box x/y in [-0.5,0.5], z in [-15,-5] with nothing solid: every edge is eligible.
    const std::unique_ptr<Fixture> open = makeFixture({farMarker, nearMarker}, 1);
    const int unoccluded = boxPixelsOf(open->render(straightOnCamera(), straightOnCamera()), 0).count;
    // An occluder at z=-4, nearer than the near corner at z=-5: its surface becomes the new near face, so those 4 edges stay visible.
    const std::array<ShadingTriangle, 2> occluder = makeQuad(0.5F, glm::vec3(0.0F, 0.0F, -4.0F), 0);
    const std::unique_ptr<Fixture> blocked = makeFixture({farMarker, occluder[0], occluder[1]}, 1);
    const int occluded = boxPixelsOf(blocked->render(straightOnCamera(), straightOnCamera()), 0).count;
    std::cout << "gbuffer_validate: boundingBox -- " << unoccluded << " unoccluded, " << occluded << " occluded pixels\n";
    PT_EXPECT(ctx, unoccluded > 0, "no box edges visible with nothing in front of the box");
    PT_EXPECT(ctx, occluded * 2 < unoccluded, "the occluder did not hide the box edges behind it");
}

// One box per instance: two separated quads must draw disjoint boxes, where a single fused box would span both.
PT_CHECK(per_instance_boxes, Fast, Exact) {
    ctx.plan(2);
    // Both at z=-10, where the 35mm lens on 36mm film sees x in about +-5.1, leaving both fully on screen with a clear gap between them.
    const std::array<ShadingTriangle, 2> left = makeQuad(0.5F, glm::vec3(-2.0F, 0.0F, -10.0F), 0);
    const std::array<ShadingTriangle, 2> right = makeQuad(0.5F, glm::vec3(2.0F, 0.0F, -10.0F), 1);
    const std::unique_ptr<Fixture> fixture = makeFixture({left[0], left[1], right[0], right[1]}, 2);
    const GBuffer g = fixture->render(straightOnCamera(), straightOnCamera());
    const BoxPixels leftBox = boxPixelsOf(g, 0);
    const BoxPixels rightBox = boxPixelsOf(g, 1);
    std::cout << "gbuffer_validate: perInstanceBoxes -- instance 0 " << leftBox.count << " px [x " << leftBox.minX << ","
              << leftBox.maxX << "], instance 1 " << rightBox.count << " px [x " << rightBox.minX << "," << rightBox.maxX << "]\n";
    PT_EXPECT(ctx, leftBox.count > 0 && rightBox.count > 0, "an instance drew no box edges in its own false colour");
    PT_EXPECT(ctx, leftBox.maxX < rightBox.minX, "the two boxes overlap in x, expected one box per instance");
}

// Must draw something (not silently empty) but only a thin fraction of hit pixels, not most of the mesh.
void expectThinWireframe(tools::check::Context& ctx, const char* name, const Camera& camera) {
    const std::unique_ptr<Fixture> fixture = makeClusterFixture(ctx);
    const GBuffer g = fixture->render(camera, camera);
    int hitPixels = 0;
    int wirePixels = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            if (texelAt(g.alpha, x, y).x > 0.5F) {
                ++hitPixels;
                wirePixels += isWireframeColor(texelAt(g.wireframe, x, y)) ? 1 : 0;
            }
        }
    }
    std::cout << "gbuffer_validate: wireframe " << name << " -- " << wirePixels << " / " << hitPixels << " hit pixels\n";
    PT_EXPECT(ctx, wirePixels > 0, std::string(name) + ": no wireframe pixels drawn");
    PT_EXPECT(ctx, wirePixels * 2 <= hitPixels, std::string(name) + ": wireframe covers over half the hit pixels");
}

// Pinhole poses only: a 220-degree fisheye shrinks these triangles to a few pixels, where most of each lies within a pixel of an edge.
PT_CHECK(wireframe_sanity, Fast, Exact) {
    ctx.plan(4);
    expectThinWireframe(ctx, "straightOn", straightOnCamera());
    expectThinWireframe(ctx, "angled", angledCamera());
}

// A closed cube, each face an n x n grid; each lattice point is shared, so neighbours hold bitwise-identical positions.
std::vector<ShadingTriangle> makeClosedCube(int n, float halfExtent, float jitter, std::mt19937& rng) {
    const int side = n + 1;
    const float cell = 2.0F * halfExtent / static_cast<float>(n);
    std::uniform_real_distribution<float> offset(-jitter, jitter);
    const auto index = [side](glm::ivec3 c) {
        return static_cast<std::size_t>((((c.x * side) + c.y) * side) + c.z);
    };
    std::vector<glm::vec3> lattice(static_cast<std::size_t>(side) * side * side);
    for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; ++j) {
            for (int k = 0; k < side; ++k) {
                const glm::vec3 jitterOffset(offset(rng), offset(rng), offset(rng));
                lattice[index({i, j, k})] = ((glm::vec3(i, j, k) - (static_cast<float>(n) * 0.5F)) * cell) + jitterOffset;
            }
        }
    }
    std::vector<ShadingTriangle> triangles;
    triangles.reserve(static_cast<std::size_t>(12 * n * n));
    for (int axis = 0; axis < 3; ++axis) {
        const int u = (axis + 1) % 3;
        const int v = (axis + 2) % 3;
        for (const int layer : {0, n}) {
            glm::vec3 normal(0.0F);
            normal[axis] = layer == 0 ? 1.0F : -1.0F;  // facing inward, toward the camera
            const glm::vec4 tangent = tangentFor(normal);
            for (int p = 0; p < n; ++p) {
                for (int q = 0; q < n; ++q) {
                    const auto corner = [&](int du, int dv) {
                        glm::ivec3 c(0);
                        c[axis] = layer;
                        c[u] = p + du;
                        c[v] = q + dv;
                        return ShadingVertex{lattice[index(c)], normal, glm::vec2(du, dv), tangent};
                    };
                    triangles.push_back(ShadingTriangle{corner(0, 0), corner(1, 0), corner(1, 1), 0});
                    triangles.push_back(ShadingTriangle{corner(0, 0), corner(1, 1), corner(0, 1), 0});
                }
            }
        }
    }
    return triangles;
}

int uncoveredPixels(const GBuffer& g) {
    int uncovered = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            uncovered += texelAt(g.alpha, x, y).x > 0.5F ? 0 : 1;
        }
    }
    return uncovered;
}

// Pixels past a fisheye's image circle: no ray, so no surface, and no crack either.
int unimagedPixels(const Camera& camera) {
    const Camera::ViewBasis basis = camera.viewBasis(kAspect);
    int unimaged = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const glm::vec2 ndc = pixelNdc(x, y);
            unimaged += camera.primaryRay(basis, ndc.x, ndc.y) ? 0 : 1;
        }
    }
    return unimaged;
}

// Watertightness, exact: from inside a closed mesh every imaged pixel must hit, with no oracle and no tolerance.
PT_CHECK(watertight_closed_mesh, Fast, Exact) {
    constexpr int kCells = 16;
    constexpr float kHalfExtent = 5.0F;
    constexpr int kSeededPoses = 8;
    // Any jitter that keeps the origin inside leaves the surface closed around the camera; a quarter cell keeps triangles well-shaped.
    constexpr float kJitter = 0.25F * (2.0F * kHalfExtent / static_cast<float>(kCells));
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    ctx.plan(2 + kSeededPoses);

    const auto expectWatertight = [&](const std::string& pose, const Camera& camera, Fixture& fixture) {
        const int uncovered = uncoveredPixels(fixture.render(camera, camera)) - unimagedPixels(camera);
        std::cout << "gbuffer_validate: watertight " << pose << " -- " << uncovered << " uncovered pixels\n";
        PT_EXPECT(ctx, uncovered == 0, pose + ": pixels uncovered from inside a closed mesh (a crack)");
    };

    const std::unique_ptr<Fixture> regular = makeFixture(makeClosedCube(kCells, kHalfExtent, 0.0F, rng), 1);
    expectWatertight("straightOn", straightOnCamera(), *regular);
    const std::unique_ptr<Fixture> jittered = makeFixture(makeClosedCube(kCells, kHalfExtent, kJitter, rng), 1);
    // A 360-degree fisheye images every direction, so a single frame sweeps the whole enclosing surface.
    const Camera allRound(glm::vec3(0.0F), 0.0F, 0.0F, kFilmBack, 4.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F, 100.0F,
                          Lens{LensProjection::FisheyePolynomial, {}, 360.0F});
    expectWatertight("fisheye360", allRound, *jittered);
    std::uniform_real_distribution<float> yaw(0.0F, 360.0F);
    std::uniform_real_distribution<float> pitch(-85.0F, 85.0F);
    for (int i = 0; i < kSeededPoses; ++i) {
        const float poseYaw = yaw(rng);
        const float posePitch = pitch(rng);
        expectWatertight("yaw " + std::to_string(poseYaw) + " pitch " + std::to_string(posePitch),
                         Camera(glm::vec3(0.0F), poseYaw, posePitch, kFilmBack, 35.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F, 100.0F),
                         *jittered);
    }
}

// A triangle far larger than the view covers every pixel yet shows no wireframe: its edges lie far off-screen.
PT_CHECK(wireframe_ignores_offscreen_edges, Fast, Exact) {
    // Plane z = -10 - 0.3y: every frustum ray meets it in front of the camera, while its y = -1000 vertices sit at z = +290, behind it.
    const glm::vec3 normal = glm::normalize(glm::vec3(0.0F, 0.3F, 1.0F));
    const glm::vec4 tangent = tangentFor(normal);
    const std::unique_ptr<Fixture> fixture = makeFixture(
        {ShadingTriangle{ShadingVertex{glm::vec3(-1000.0F, -1000.0F, 290.0F), normal, glm::vec2(0.0F, 0.0F), tangent},
                         ShadingVertex{glm::vec3(1000.0F, -1000.0F, 290.0F), normal, glm::vec2(1.0F, 0.0F), tangent},
                         ShadingVertex{glm::vec3(0.0F, 1000.0F, -310.0F), normal, glm::vec2(0.0F, 1.0F), tangent}, 0}},
        1);
    fixture->instanceBounds.clear();
    const GBuffer g = fixture->render(straightOnCamera(), straightOnCamera());
    int wirePixels = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            wirePixels += isWireframeColor(texelAt(g.wireframe, x, y)) ? 1 : 0;
        }
    }
    const int uncovered = uncoveredPixels(g);
    std::cout << "gbuffer_validate: off-screen edges -- " << uncovered << " uncovered, " << wirePixels << " wireframe pixels\n";
    ctx.plan(2);
    PT_EXPECT(ctx, uncovered == 0, "the screen-covering triangle left pixels uncovered");
    PT_EXPECT(ctx, wirePixels == 0, "edges far outside the view were drawn as wireframe");
}

bool sameBsdfParams(const BsdfParams& a, const BsdfParams& b) {
    return a.baseColor == b.baseColor && a.metallic == b.metallic && a.roughness == b.roughness && a.f0 == b.f0 &&
           a.edgeTint == b.edgeTint && a.ior == b.ior && a.transmissionFactor == b.transmissionFactor &&
           a.diffuseRoughness == b.diffuseRoughness && a.diffuseRho == b.diffuseRho && a.transmissionTint == b.transmissionTint;
}

// An unbound slot's constant shades bit-identically to the 1x1 texture of that value it replaced: no tolerance, any uv/frame/settings.
PT_CHECK(constant_inputs_match_unit_textures, Fast, Exact) {
    constexpr int kCases = 256;
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    std::uniform_real_distribution<float> wrappedUv(-4.0F, 4.0F);
    const auto randomVec3 = [&] { return glm::vec3(unit(rng), unit(rng), unit(rng)); };
    const auto unitTexture = [](const float* value, int channels) {
        return std::make_shared<const pathtracer::gfx::ImageTexture>(
            pathtracer::gfx::ImageTexture{1, 1, channels, std::vector<float>(value, value + channels)});
    };
    ctx.plan(3);
    int paramMismatches = 0;
    int frameMismatches = 0;
    int bumpMismatches = 0;
    for (int i = 0; i < kCases; ++i) {
        const glm::vec3 baseColor = randomVec3();
        const glm::vec3 encodedNormal = randomVec3();
        const float height = unit(rng);
        const float roughness = unit(rng);
        const glm::vec3 specular = randomVec3();
        const Material constant{.baseColor = baseColor, .normal = encodedNormal, .bump = height, .roughness = roughness,
                                .specular = specular};
        // Bump stays constant here so the frames compare exactly; the bound bump texture is the third material's alone.
        const Material textured{.baseColor = unitTexture(&baseColor.x, pathtracer::gfx::kRgbChannels),
                                .normal = unitTexture(&encodedNormal.x, pathtracer::gfx::kRgbChannels),
                                .bump = height,
                                .roughness = unitTexture(&roughness, pathtracer::gfx::kScalarChannels),
                                .specular = unitTexture(&specular.x, pathtracer::gfx::kRgbChannels)};
        Material bumped = constant;
        bumped.bump = unitTexture(&height, pathtracer::gfx::kScalarChannels);

        PathTraceSettings settings = makeTestSettings();
        settings.bumpStrength = unit(rng);
        settings.diffuseColour = randomVec3();
        settings.roughnessFactor = unit(rng);
        settings.metallicFactor = unit(rng);
        settings.transmissionFactor = unit(rng);
        settings.diffuseRoughness = unit(rng);
        settings.ior = 1.0F + unit(rng);
        const glm::vec3 normal = glm::normalize(randomVec3() - glm::vec3(0.5F));
        glm::vec4 tangent = tangentFor(normal);
        tangent.w = unit(rng) < 0.5F ? -1.0F : 1.0F;
        const ShadingVertex vertex{glm::vec3(0.0F), normal, glm::vec2(wrappedUv(rng), wrappedUv(rng)), tangent, randomVec3()};

        const BsdfParams constantParams = resolveBsdfParams(constant, vertex.uv, vertex.colour, settings, std::nullopt);
        const BsdfParams texturedParams = resolveBsdfParams(textured, vertex.uv, vertex.colour, settings, std::nullopt);
        paramMismatches += sameBsdfParams(constantParams, texturedParams) ? 0 : 1;
        const ShadingFrame constantFrame = buildShadingFrame(vertex, constant, settings);
        frameMismatches += constantFrame == buildShadingFrame(vertex, textured, settings) ? 0 : 1;
        // Four taps of a constant height differ by exactly 0, so the texture path only renormalises the unbumped normal.
        bumpMismatches += buildShadingFrame(vertex, bumped, settings)[2] == glm::normalize(constantFrame[2]) ? 0 : 1;
    }
    std::cout << "gbuffer_validate: constant vs 1x1 texture over " << kCases << " cases -- " << paramMismatches << " BsdfParams, "
              << frameMismatches << " frame, " << bumpMismatches << " bump mismatches\n";
    PT_EXPECT(ctx, paramMismatches == 0, "a constant input resolved BsdfParams differently from its 1x1 texture");
    PT_EXPECT(ctx, frameMismatches == 0, "a constant normal built a different shading frame from its 1x1 texture");
    PT_EXPECT(ctx, bumpMismatches == 0, "a constant bump texture tilted the normal: its gradient is not exactly zero");
}

// Double-precision pinhole K [R | -R c] (Hartley & Zisserman 2004, eq. 6.8) on homogeneous (p, w): shares no arithmetic with project.
std::optional<glm::dvec2> pinholeNdc(const Camera& camera, const glm::dvec4& point) {
    const Camera::ViewBasis basis = camera.viewBasis(kAspect);
    const glm::dmat3 worldToView =
        glm::transpose(glm::dmat3(glm::dvec3(basis.right), glm::dvec3(basis.up), glm::dvec3(basis.forward)));
    const glm::dvec3 view = worldToView * (glm::dvec3(point) - (point.w * glm::dvec3(camera.position())));
    if (!(view.z > 0.0)) {
        return std::nullopt;
    }
    return glm::dvec2(view.x / (view.z * static_cast<double>(basis.halfWidth)), view.y / (view.z * static_cast<double>(basis.halfHeight)));
}

// Double-precision equidistant fisheye, r = f * theta (Kannala & Brandt 2006 with k = 0): the closed form, independent of project.
std::optional<glm::dvec2> equidistantNdc(const Camera& camera, const glm::dvec4& point) {
    const Camera::ViewBasis basis = camera.viewBasis(kAspect);
    const glm::dvec3 view = glm::dvec3(point) - (point.w * glm::dvec3(camera.position()));
    const glm::dvec2 lateral(glm::dot(view, glm::dvec3(basis.right)), glm::dot(view, glm::dvec3(basis.up)));
    const double theta = std::atan2(glm::length(lateral), glm::dot(view, glm::dvec3(basis.forward)));
    if (theta > static_cast<double>(basis.maxThetaRadians)) {
        return std::nullopt;
    }
    const double radiusMm = static_cast<double>(basis.focalLengthMm) * theta;
    return radiusMm * glm::normalize(lateral) / glm::dvec2(basis.halfWidthMm, basis.halfHeightMm);
}

// Double-precision lat-long: longitude pi x and latitude pi y / 2 on the camera frame, by atan2 alone, independent of lat_long.h.
std::optional<glm::dvec2> latLongNdc(const Camera& camera, const glm::dvec4& point) {
    const Camera::ViewBasis basis = camera.viewBasis(kAspect);
    const glm::dvec3 view = glm::dvec3(point) - (point.w * glm::dvec3(camera.position()));
    if (view == glm::dvec3(0.0)) {
        return std::nullopt;
    }
    const double right = glm::dot(view, glm::dvec3(basis.right));
    const double forward = glm::dot(view, glm::dvec3(basis.forward));
    const double latitude = std::atan2(glm::dot(view, glm::dvec3(basis.up)), std::hypot(right, forward));
    return glm::dvec2(std::atan2(right, forward) / std::numbers::pi, latitude / (0.5 * std::numbers::pi));
}

std::optional<glm::dvec2> oracleNdc(const Camera& camera, const glm::dvec4& point) {
    switch (camera.lens().projection) {
        case LensProjection::FisheyePolynomial:
            return equidistantNdc(camera, point);
        case LensProjection::Omnidirectional:
            return latLongNdc(camera, point);
        case LensProjection::Rectilinear:
        case LensProjection::Count:
            break;
    }
    return pinholeNdc(camera, point);
}

// Relative condition number: the pinhole divides by a depth formed by cancellation, |view| / |depth|; the fisheye's atan2 is 1.
double projectionCondition(const Camera& camera, const glm::dvec4& point) {
    const glm::dvec3 view = glm::dvec3(point) - (point.w * glm::dvec3(camera.position()));
    switch (camera.lens().projection) {
        case LensProjection::FisheyePolynomial:
            return 1.0;
        // Longitude is atan2 of the horizontal components, whose length is |view| sin(colatitude): it degrades as 1/sin at a pole.
        case LensProjection::Omnidirectional:
            return glm::length(view) / std::hypot(glm::dot(view, glm::dvec3(camera.viewBasis(kAspect).right)),
                                                  glm::dot(view, glm::dvec3(camera.forward())));
        case LensProjection::Rectilinear:
        case LensProjection::Count:
            break;
    }
    return glm::length(view) / std::abs(glm::dot(view, glm::dvec3(camera.forward())));
}

// The oracle's motion at (x, y) and its float budget: two projections, each 16 ulps of its NDC magnitude times its conditioning.
struct OracleMotion {
    glm::dvec2 motion;
    double budgetPx;
};

// From the producer's own hit point, else its pixel-centre ray direction at infinity, in double.
OracleMotion oracleMotion(const GBuffer& g, const Camera& camera, const Camera& previous, int x, int y) {
    const double unitBudgetPx = kClosedFormUlps * kEps * 0.5 * kWidth;
    const glm::vec2 ndc = pixelNdc(x, y);
    const std::optional<Ray> ray = camera.primaryRay(camera.viewBasis(kAspect), ndc.x, ndc.y);
    if (!ray) {
        return {glm::dvec2(0.0), 0.0};
    }
    const bool hit = texelAt(g.alpha, x, y).x > 0.5F;
    const glm::dvec4 point = hit ? glm::dvec4(glm::dvec3(texelAt(g.worldPos, x, y)), 1.0) : glm::dvec4(glm::dvec3(ray->dir), 0.0);
    const std::optional<glm::dvec2> now = oracleNdc(camera, point);
    const std::optional<glm::dvec2> before = oracleNdc(previous, point);
    if (!now || !before) {
        return {glm::dvec2(0.0), 0.0};
    }
    // A pinhole far off its axis images at |ndc| >> 1, and its float rounding grows with it: the budget is relative, floored at 1.
    const auto magnitude = [](glm::dvec2 v) { return std::max({1.0, std::abs(v.x), std::abs(v.y)}); };
    const double budget = unitBudgetPx * ((magnitude(*now) * projectionCondition(camera, point)) +
                                          (magnitude(*before) * projectionCondition(previous, point)));
    glm::dvec2 motion = *now - *before;
    // Both views lat-long: ndc x lives on a circle of circumference 2, and the motion is the shorter arc of it.
    if (wrapsHorizontally(camera.lens().projection) && wrapsHorizontally(previous.lens().projection)) {
        motion.x = std::remainder(motion.x, 2.0);
    }
    return {motion * glm::dvec2(0.5 * kWidth, -0.5 * kHeight), budget};
}

glm::vec2 motionAt(const GBuffer& g, int x, int y) {
    return glm::vec2(texelAt(g.motionVector, x, y));
}

int nonZeroMotion(const GBuffer& g) {
    int nonZero = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            nonZero += motionAt(g, x, y) == glm::vec2(0.0F) ? 0 : 1;
        }
    }
    return nonZero;
}

// Same camera both sides: one projection of one point twice, so every pixel, hit or miss, reads zero bitwise rather than nearly.
PT_CHECK(motion_vector_is_exactly_zero_for_an_unmoved_camera, Fast, Exact) {
    const std::unique_ptr<Fixture> fixture = makeClusterFixture(ctx);
    ctx.plan(9);
    for (const Camera& camera : {angledCamera(), fisheyeCamera(220.0F), omnidirectionalCamera()}) {
        const GBuffer g = fixture->render(camera, camera);
        const int misses = uncoveredPixels(g);
        PT_EXPECT(ctx, misses > 0, "the pose shows no environment, so the w = 0 path went unexercised");
        PT_EXPECT(ctx, misses < kWidth * kHeight, "the pose shows no geometry, so the w = 1 path went unexercised");
        PT_EXPECT(ctx, nonZeroMotion(g) == 0, std::to_string(nonZeroMotion(g)) + " pixels moved although the camera did not");
    }
}

// The worst error in units of each pixel's own budget, so 1 is the bound wherever the motion's magnitude puts it.
double worstMotionError(const GBuffer& g, const Camera& camera, const Camera& previous) {
    double worst = 0.0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const OracleMotion oracle = oracleMotion(g, camera, previous, x, y);
            const glm::dvec2 error = glm::abs(glm::dvec2(motionAt(g, x, y)) - oracle.motion);
            const double largest = std::max(error.x, error.y);
            worst = std::max(worst, largest == 0.0 ? 0.0 : largest / oracle.budgetPx);
        }
    }
    return worst;
}

// Rotation, translation, both, zoom and lens toggles against the double oracle at every pixel, from a pinhole and from a fisheye.
PT_CHECK(motion_vector_matches_the_oracle, Fast, Exact) {
    const std::unique_ptr<Fixture> fixture = makeClusterFixture(ctx);
    const Lens equidistant{LensProjection::FisheyePolynomial, {}, 180.0F};
    const std::array<std::tuple<const char*, Camera, Camera>, 11> cases{{
        {"pinhole rotation", angledCamera(), angledCamera(glm::vec3(0.0F), 3.0F, -2.0F)},
        {"pinhole translation", angledCamera(), angledCamera(glm::vec3(0.3F, -0.2F, 0.5F))},
        {"pinhole rotation+translation", angledCamera(), angledCamera(glm::vec3(-0.4F, 0.1F, 0.25F), -4.0F, 1.5F)},
        {"pinhole zoom", angledCamera(), angledCamera(glm::vec3(0.0F), 0.0F, 0.0F, 50.0F)},
        {"pinhole from an equidistant fisheye", angledCamera(),
         Camera(glm::vec3(3.0F, 2.0F, 1.0F), 20.0F, -10.0F, kFilmBack, 8.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F, 100.0F, equidistant)},
        {"fisheye rotation", fisheyeCamera(220.0F), fisheyeCamera(220.0F, glm::vec3(0.0F), 4.0F)},
        {"fisheye translation", fisheyeCamera(220.0F), fisheyeCamera(220.0F, glm::vec3(0.2F, 0.1F, -0.3F))},
        {"fisheye from a pinhole", fisheyeCamera(220.0F),
         Camera(glm::vec3(0.5F, -0.5F, -8.0F), 15.0F, 5.0F, kFilmBack, 35.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F, 100.0F)},
        {"lat-long rotation", omnidirectionalCamera(), omnidirectionalCamera(glm::vec3(0.0F), 4.0F)},
        {"lat-long translation", omnidirectionalCamera(), omnidirectionalCamera(glm::vec3(0.2F, 0.1F, -0.3F))},
        {"lat-long from a fisheye", omnidirectionalCamera(), fisheyeCamera(220.0F)},
    }};
    ctx.plan(static_cast<int>(cases.size()));
    for (const auto& [name, camera, previous] : cases) {
        const double worst = worstMotionError(fixture->render(camera, previous), camera, previous);
        std::cout << "gbuffer_validate: motion " << name << " -- worst " << worst << " of budget\n";
        PT_EXPECT(ctx, worst <= 1.0, std::string(name) + ": worst motion error " + std::to_string(worst) + " of budget");
    }
}

// A level camera's yaw shifts every lat-long direction by one longitude, so each miss moves by the same pixels, the seam included.
PT_CHECK(motion_vector_takes_the_short_way_across_the_seam, Fast, Exact) {
    const std::unique_ptr<Fixture> fixture = makeClusterFixture(ctx);
    ctx.plan(3);
    constexpr float kYawStepDegrees = 4.0F;
    const Lens latLong{LensProjection::Omnidirectional};
    const Camera camera(glm::vec3(0.5F, -0.5F, -8.0F), 15.0F, 0.0F, kFilmBack, 8.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F, 100.0F, latLong);
    const Camera previous(glm::vec3(0.5F, -0.5F, -8.0F), 15.0F - kYawStepDegrees, 0.0F, kFilmBack, 8.0F, 0.1F, 100.0F, 2.8F,
                          1.0F / 125.0F, 100.0F, latLong);
    const GBuffer g = fixture->render(camera, previous);
    // Turning left by the step moves a fixed direction right by step / 360 of the width; float yaw and atan2 round to 16 ulps of a turn.
    const double expected = static_cast<double>(kYawStepDegrees) / 360.0 * kWidth;
    const double budget = kClosedFormUlps * kEps * kWidth;
    int seamMisses = 0;
    int departures = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            if (texelAt(g.alpha, x, y).x > 0.5F) {
                continue;
            }
            seamMisses += x < expected + 1.0 ? 1 : 0;
            departures += std::abs(static_cast<double>(motionAt(g, x, y).x) - expected) > budget ? 1 : 0;
        }
    }
    PT_EXPECT(ctx, seamMisses > 0, "no environment pixel lay where its previous image was across the seam");
    PT_EXPECT(ctx, departures == 0, std::to_string(departures) + " environment pixels moved other than by the yaw step");
    PT_EXPECT(ctx, g.wrapsHorizontally, "the G-buffer does not record that its lens wraps, so its display would clamp at the seam");
}

// Translation leaves the plane at infinity fixed (w = 0 drops the camera centre), so every miss reads zero bitwise while hits move.
PT_CHECK(motion_vector_holds_the_sky_still_under_translation, Fast, Exact) {
    const std::unique_ptr<Fixture> fixture = makeClusterFixture(ctx);
    ctx.plan(4);
    for (const auto& [camera, previous] : {std::pair{angledCamera(), angledCamera(glm::vec3(0.3F, -0.2F, 0.5F))},
                                           std::pair{fisheyeCamera(220.0F), fisheyeCamera(220.0F, glm::vec3(0.2F, 0.1F, -0.3F))}}) {
        const GBuffer g = fixture->render(camera, previous);
        int movedMisses = 0;
        int movedHits = 0;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const bool moved = motionAt(g, x, y) != glm::vec2(0.0F);
                (texelAt(g.alpha, x, y).x > 0.5F ? movedHits : movedMisses) += moved ? 1 : 0;
            }
        }
        PT_EXPECT(ctx, movedMisses == 0, std::to_string(movedMisses) + " environment pixels moved under a pure translation");
        PT_EXPECT(ctx, movedHits > 0, "no surface moved under a translation");
    }
}

// Turned half a revolution, the previous view faces away from everything this one sees: no image point exists, so motion reads zero.
PT_CHECK(motion_vector_is_zero_where_the_previous_view_has_no_image, Fast, Exact) {
    const std::unique_ptr<Fixture> fixture = makeClusterFixture(ctx);
    ctx.plan(1);
    const Camera previous(glm::vec3(0.0F), 180.0F, 0.0F, kFilmBack, 35.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F, 100.0F);
    const int nonZero = nonZeroMotion(fixture->render(straightOnCamera(), previous));
    PT_EXPECT(ctx, nonZero == 0, std::to_string(nonZero) + " pixels were given motion from a view that cannot see them");
}

}  // namespace

PT_CHECK_MAIN("gbuffer")
