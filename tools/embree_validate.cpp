// Correctness check for EmbreeAccel: random rays over synthetic soup against bruteForceIntersect, a dependency-free O(n) oracle.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "check.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/ray_types.h"

namespace {

using pathtracer::scene::AnalyticShape;
using pathtracer::scene::EmbreeAccel;
using pathtracer::scene::Hit;
using pathtracer::scene::Ray;
using pathtracer::scene::Triangle;

constexpr int kTriangleCount = 2000;
constexpr int kRayCount = 20000;
constexpr int kShapeCount = 300;  // per kind
// Embree solves a sphere in float about the ray origin, so its t errs relative to t; the integrator's shadow rays back off 1e-3 of it.
constexpr double kShadowBackOff = 1e-3;
constexpr float kTEpsilon = 1e-3F;

// Triangles clustered around random centres: exercises ordinary BVH splitting as well as tightly-packed clusters.
std::vector<Triangle> makeSyntheticTriangles(std::mt19937& rng) {
    std::uniform_real_distribution<float> centerDist(-50.0F, 50.0F);
    std::uniform_real_distribution<float> offsetDist(-1.0F, 1.0F);

    std::vector<Triangle> triangles;
    triangles.reserve(kTriangleCount);
    for (int i = 0; i < kTriangleCount; ++i) {
        const glm::vec3 center(centerDist(rng), centerDist(rng), centerDist(rng));
        const glm::vec3 v0 =
            center + glm::vec3(offsetDist(rng), offsetDist(rng), offsetDist(rng));
        const glm::vec3 v1 =
            center + glm::vec3(offsetDist(rng), offsetDist(rng), offsetDist(rng));
        const glm::vec3 v2 =
            center + glm::vec3(offsetDist(rng), offsetDist(rng), offsetDist(rng));
        triangles.push_back(Triangle{v0, v1, v2});
    }
    return triangles;
}

Ray makeRandomRay(std::mt19937& rng) {
    std::uniform_real_distribution<float> originDist(-80.0F, 80.0F);
    std::uniform_real_distribution<float> dirDist(-1.0F, 1.0F);
    glm::vec3 dir(dirDist(rng), dirDist(rng), dirDist(rng));
    while (glm::length(dir) < 1e-4F) {
        dir = glm::vec3(dirDist(rng), dirDist(rng), dirDist(rng));
    }
    return Ray{glm::vec3(originDist(rng), originDist(rng), originDist(rng)), glm::normalize(dir),
               0.0F, 1000.0F};
}

// Builds the scene and traces every ray once into `compare`; each check pays its own trace, which keeps them independently runnable.
template <typename Compare>
void crossCheck(tools::check::Context& ctx, Compare compare) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    const std::vector<Triangle> triangles = makeSyntheticTriangles(rng);
    std::optional<EmbreeAccel> accel = EmbreeAccel::build(triangles);
    ctx.plan(1);
    if (!accel) {
        PT_EXPECT(ctx, false, "EmbreeAccel::build failed");
        return;
    }
    int mismatches = 0;
    for (int i = 0; i < kRayCount; ++i) {
        const Ray ray = makeRandomRay(rng);
        mismatches += compare(*accel, triangles, ray) ? 0 : 1;
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "%d of %d rays mismatched over %zu triangles", mismatches, kRayCount,
                  triangles.size());
    PT_EXPECT(ctx, mismatches == 0, detail);
}

// Watertightness and traversal: Embree and the O(n) reference must agree on whether a ray hits anything at all.
PT_CHECK(intersect_hit_agreement, Fast, Exact) {
    crossCheck(ctx, [](const EmbreeAccel& accel, const std::vector<Triangle>& triangles, const Ray& ray) {
        return accel.intersect(ray).has_value() ==
               pathtracer::scene::bruteForceIntersect(triangles, ray).has_value();
    });
}

// Intersector precision: where both agree there is a hit, they must agree WHERE. Misses are skipped, not counted as agreement.
PT_CHECK(intersect_distance_agreement, Fast, Exact) {
    crossCheck(ctx, [](const EmbreeAccel& accel, const std::vector<Triangle>& triangles, const Ray& ray) {
        const std::optional<Hit> accelHit = accel.intersect(ray);
        const std::optional<Hit> bruteHit = pathtracer::scene::bruteForceIntersect(triangles, ray);
        if (!accelHit.has_value() || !bruteHit.has_value()) {
            return true;
        }
        return std::fabs(accelHit->t - bruteHit->t) <= kTEpsilon;
    });
}

// The any-hit path is a separate Embree entry point, so it can diverge unnoticed: a shadow ray reporting clear through geometry is a leak.
PT_CHECK(occluded_agreement, Fast, Exact) {
    crossCheck(ctx, [](const EmbreeAccel& accel, const std::vector<Triangle>& triangles, const Ray& ray) {
        return accel.occluded(ray) == pathtracer::scene::bruteForceIntersect(triangles, ray).has_value();
    });
}

// Random spheres and discs, discs in random orientations: the native point geometries the disk and sphere lights trace as.
std::vector<AnalyticShape> makeSyntheticShapes(std::mt19937& rng) {
    std::uniform_real_distribution<float> centreDist(-50.0F, 50.0F);
    std::uniform_real_distribution<float> radiusDist(0.5F, 2.0F);
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    std::vector<AnalyticShape> shapes;
    for (int i = 0; i < 2 * kShapeCount; ++i) {
        glm::vec3 normal(unit(rng), unit(rng), unit(rng));
        while (glm::length(normal) < 1e-2F) {
            normal = glm::vec3(unit(rng), unit(rng), unit(rng));
        }
        normal = glm::normalize(normal);
        const glm::vec3 tangent = glm::normalize(glm::cross(std::abs(normal.x) < 0.9F ? glm::vec3(1.0F, 0.0F, 0.0F) : glm::vec3(0.0F, 1.0F, 0.0F), normal));
        shapes.push_back(AnalyticShape{i < kShapeCount ? AnalyticShape::Kind::Sphere : AnalyticShape::Kind::Disc,
                                       glm::vec3(centreDist(rng), centreDist(rng), centreDist(rng)), radiusDist(rng),
                                       glm::mat3(tangent, glm::cross(normal, tangent), normal), i});
    }
    return shapes;
}

// The nearest t in (tMin, tMax) of an analytic ray-sphere or ray-disc solve in double, with how far the ray passes from the silhouette.
struct AnalyticHit {
    double t;
    int shape;
    double margin;  // distance from the nearest shape boundary along the hit's own measure, relative to its radius
};

std::optional<AnalyticHit> nearestShape(const std::vector<AnalyticShape>& shapes, const Ray& ray, double& closestMargin) {
    std::optional<AnalyticHit> nearest;
    const glm::dvec3 o(ray.origin);
    const glm::dvec3 d(ray.dir);
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        const AnalyticShape& shape = shapes[i];
        const glm::dvec3 c(shape.centre);
        const double r = shape.radius;
        std::optional<double> t;
        double margin = 0.0;
        if (shape.kind == AnalyticShape::Kind::Sphere) {
            const double b = glm::dot(d, o - c);
            const double cc = glm::dot(o - c, o - c) - (r * r);
            const double discriminant = (b * b) - cc;
            // The ray's closest approach to the centre against r: near zero is a tangent ray either intersector may call either way.
            margin = std::abs(std::sqrt(std::max(0.0, glm::dot(o - c, o - c) - (b * b))) - r) / r;
            if (discriminant >= 0.0) {
                const double root = std::sqrt(discriminant);
                for (const double candidate : {-b - root, -b + root}) {
                    if (!t && candidate > ray.tMin && candidate < ray.tMax) {
                        t = candidate;
                    }
                }
            }
        } else {
            const glm::dvec3 n(shape.frame[2]);
            const double candidate = glm::dot(c - o, n) / glm::dot(d, n);
            const double rim = glm::length(o + (candidate * d) - c);
            margin = std::abs(rim - r) / r;
            if (rim <= r && candidate > ray.tMin && candidate < ray.tMax) {
                t = candidate;
            }
        }
        closestMargin = std::min(closestMargin, margin);
        if (t && (!nearest || *t < nearest->t)) {
            nearest = AnalyticHit{*t, static_cast<int>(i), margin};
        }
    }
    return nearest;
}

// Spheres and discs against the analytic solve: hit, which shape and where, rays from inside spheres included; tangent rays excused.
PT_CHECK(shape_intersections_match_the_analytic_solve, Fast, Exact) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    const std::vector<AnalyticShape> shapes = makeSyntheticShapes(rng);
    std::optional<EmbreeAccel> accel = EmbreeAccel::build({}, shapes);
    ctx.plan(2);
    if (!accel) {
        PT_EXPECT(ctx, false, "EmbreeAccel::build failed");
        ctx.plan(1);
        return;
    }
    // Float single-precision intersection near a silhouette: within 1e-3 of the radius either answer is a correct rounding.
    constexpr double kTangentMargin = 1e-3;
    int mismatches = 0;
    int excused = 0;
    for (int i = 0; i < kRayCount; ++i) {
        Ray ray = makeRandomRay(rng);
        // Every tenth ray starts at a sphere's centre, so it must meet that sphere's far wall from inside.
        if (i % 10 == 0) {
            ray.origin = shapes[static_cast<std::size_t>(i / 10) % kShapeCount].centre;
        }
        double closestMargin = std::numeric_limits<double>::infinity();
        const std::optional<AnalyticHit> reference = nearestShape(shapes, ray, closestMargin);
        if (closestMargin <= kTangentMargin) {
            ++excused;
            continue;
        }
        const std::optional<Hit> hit = accel->intersect(ray);
        const bool agree = hit.has_value() == reference.has_value() &&
                           (!hit || (hit->shape == reference->shape && hit->triangleIndex == -1 &&
                                     std::abs(hit->t - reference->t) <= kShadowBackOff * reference->t));
        const bool occlusionAgrees = accel->occluded(ray) == reference.has_value();
        mismatches += agree && occlusionAgrees ? 0 : 1;
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "%d of %d rays mismatched over %zu shapes (%d tangent rays excused)", mismatches, kRayCount - excused,
                  shapes.size(), excused);
    PT_EXPECT(ctx, mismatches == 0, detail);
    // The excusal is for rounding at silhouettes, a sliver: it must not grow to hide a systematic miss.
    PT_EXPECT(ctx, excused * 100 < kRayCount, std::string(detail) + ": the tangent excusal swallowed over 1% of the rays");
}

}  // namespace

PT_CHECK_MAIN("embree")
