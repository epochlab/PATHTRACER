#include "pathtracer/scene/shading_scene.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include "pathtracer/scene/lat_long.h"

namespace pathtracer::scene {

bool isEmpty(const AabbBounds& box) {
    return box.min.x > box.max.x || box.min.y > box.max.y || box.min.z > box.max.z;
}

std::vector<AabbBounds> computeInstanceBounds(const std::vector<ShadingTriangle>& triangles, int instanceCount,
                                               const std::vector<AnalyticShape>& shapes) {
    constexpr float kInf = std::numeric_limits<float>::infinity();
    std::vector<AabbBounds> bounds(static_cast<std::size_t>(instanceCount),
                                    AabbBounds{glm::vec3(kInf), glm::vec3(-kInf)});
    for (const ShadingTriangle& tri : triangles) {
        AabbBounds& box = bounds[static_cast<std::size_t>(tri.instanceIndex)];
        const auto expand = [&box](const glm::vec3& position) {
            box.min = glm::min(box.min, position);
            box.max = glm::max(box.max, position);
        };
        expand(tri.v0.position);
        expand(tri.v1.position);
        expand(tri.v2.position);
    }
    for (const AnalyticShape& shape : shapes) {
        AabbBounds& box = bounds[static_cast<std::size_t>(shape.instanceIndex)];
        // A disc of normal n spans r sqrt(1 - n_i^2) along each axis i, the exact extent of a circle; a sphere spans r.
        const glm::vec3 extent = shape.kind == AnalyticShape::Kind::Disc
                                     ? shape.radius * glm::sqrt(glm::max(glm::vec3(0.0F), 1.0F - (shape.frame[2] * shape.frame[2])))
                                     : glm::vec3(shape.radius);
        box.min = glm::min(box.min, shape.centre - extent);
        box.max = glm::max(box.max, shape.centre + extent);
    }
    return bounds;
}

ShadingVertex shadingAt(const AnalyticShape& shape, const glm::vec3& point) {
    const glm::vec3 offset = point - shape.centre;
    if (shape.kind == AnalyticShape::Kind::Disc) {
        const glm::vec2 uv = 0.5F + (0.5F * glm::vec2(glm::dot(offset, shape.frame[0]), glm::dot(offset, shape.frame[1])) / shape.radius);
        return ShadingVertex{point, shape.frame[2], uv, glm::vec4(shape.frame[0], 1.0F)};
    }
    const glm::vec3 normal = glm::normalize(offset);
    const glm::vec3 local = glm::transpose(shape.frame) * normal;
    // The tangent runs along increasing longitude, atan2(x, z): (z, 0, -x) in the frame, cross(n, t) toward its +y pole.
    const glm::vec3 tangent = shape.frame * glm::normalize(glm::vec3(local.z, 0.0F, -local.x));
    return ShadingVertex{point, normal, latLongUv(local), glm::vec4(tangent, 1.0F)};
}

ShadingVertex interpolateShading(const ShadingTriangle& tri, float u, float v) {
    const float w = 1.0F - u - v;
    ShadingVertex result{};
    result.position = (w * tri.v0.position) + (u * tri.v1.position) + (v * tri.v2.position);
    result.normal = glm::normalize((w * tri.v0.normal) + (u * tri.v1.normal) + (v * tri.v2.normal));
    result.uv = (w * tri.v0.uv) + (u * tri.v1.uv) + (v * tri.v2.uv);
    const glm::vec3 tangentXyz =
        glm::normalize((w * glm::vec3(tri.v0.tangent)) + (u * glm::vec3(tri.v1.tangent)) +
                        (v * glm::vec3(tri.v2.tangent)));
    result.tangent = glm::vec4(tangentXyz, tri.v0.tangent.w);  // handedness: +-1 constant, glTF requires it consistent per-triangle
    result.colour = (w * tri.v0.colour) + (u * tri.v1.colour) + (v * tri.v2.colour);
    return result;
}

namespace {

// Higham's gamma_n = n u / (1 - n u), u = eps/2: n chained float roundings move a value by at most this fraction (PBR 4e 6.8.1).
constexpr float gamma(int n) {
    constexpr float kUnitRoundoff = std::numeric_limits<float>::epsilon() * 0.5F;
    return (static_cast<float>(n) * kUnitRoundoff) / (1.0F - (static_cast<float>(n) * kUnitRoundoff));
}

// pbrt's NextFloatUp (PBR 4e 6.8.2): positive floats order as their bit patterns, so one ulp is one integer step; -0 steps as +0.
float nextFloatUp(float v) {
    const auto bits = std::bit_cast<std::uint32_t>(v == 0.0F ? 0.0F : v);
    return std::bit_cast<float>(v >= 0.0F ? bits + 1U : bits - 1U);
}

// One float further along each component's sign in dir, so the rounded sum before it can never sit nearer the plane than the exact one.
glm::vec3 roundAway(glm::vec3 p, const glm::vec3& dir) {
    for (int i = 0; i < 3; ++i) {
        if (dir[i] > 0.0F) {
            p[i] = nextFloatUp(p[i]);
        } else if (dir[i] < 0.0F) {
            p[i] = -nextFloatUp(-p[i]);
        }
    }
    return p;
}

// Chiang/Li/Burley 2019 as a displacement: each vertex's tangent-plane projection of a hit below it, along +-n_i, so never toward the face.
glm::vec3 terminatorLift(const ShadingTriangle& tri, float w, float u, float v, const glm::vec3& flatPosition, bool normalSide) {
    const auto lift = [&](const ShadingVertex& vertex) {
        const float d = glm::dot(flatPosition - vertex.position, vertex.normal);
        const bool below = normalSide ? (d < 0.0F) : (d > 0.0F);
        return below ? -d * vertex.normal : glm::vec3(0.0F);
    };
    return (w * lift(tri.v0)) + (u * lift(tri.v1)) + (v * lift(tri.v2));
}

}  // namespace

glm::vec3 spawnOrigin(const ShadingTriangle& tri, float u, float v, const glm::vec3& geoNormal, bool normalSide) {
    const float w = 1.0F - u - v;
    const glm::vec3 flatPosition = (w * tri.v0.position) + (u * tri.v1.position) + (v * tri.v2.position);
    // A 3-term dot product's gamma_3 (Higham 3.1), plus w's two roundings |1-u| u + |w| u carried by p0 since w + u + v drifts off 1.
    const glm::vec3 error = (gamma(3) * (glm::abs(w * tri.v0.position) + glm::abs(u * tri.v1.position) + glm::abs(v * tri.v2.position))) +
                            (gamma(1) * (std::abs(1.0F - u) + std::abs(w)) * glm::abs(tri.v0.position));
    const glm::vec3 side = normalSide ? geoNormal : -geoNormal;
    // The lift projects non-negatively on side, its rounding a sliver of that: offset plus lift clears the bound; a round-away seats it.
    const glm::vec3 displacement = (glm::dot(glm::abs(geoNormal), error) * side) + terminatorLift(tri, w, u, v, flatPosition, normalSide);
    return roundAway(flatPosition + displacement, side);
}

}  // namespace pathtracer::scene
