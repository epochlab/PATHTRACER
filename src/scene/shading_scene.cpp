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

// Hanika 2021's lift projected on side: each vertex's tangent plane over a hit below it, by n_i . side; each term at most |p - p_i| / 2.
float terminatorHeight(const ShadingTriangle& tri, float w, float u, float v, const glm::vec3& flatPosition, float sign,
                       const glm::vec3& side) {
    const auto height = [&](const ShadingVertex& vertex) {
        const glm::vec3 normal = sign * vertex.normal;
        return std::max(-glm::dot(flatPosition - vertex.position, normal), 0.0F) * glm::dot(normal, side);
    };
    // A folded n_i (n_i . side < 0) weighs negative: the clamp keeps the lift off the rounding bound, so it only adds clearance.
    return std::max((w * height(tri.v0)) + (u * height(tri.v1)) + (v * height(tri.v2)), 0.0F);
}

}  // namespace

glm::vec3 spawnOrigin(const ShadingTriangle& tri, float u, float v, const glm::vec3& geoNormal, bool normalSide) {
    const float w = 1.0F - u - v;
    const glm::vec3 flatPosition = (w * tri.v0.position) + (u * tri.v1.position) + (v * tri.v2.position);
    // A 3-term dot product's gamma_3 (Higham 3.1), plus w's two roundings |1-u| u + |w| u carried by p0 since w + u + v drifts off 1.
    const glm::vec3 error = (gamma(3) * (glm::abs(w * tri.v0.position) + glm::abs(u * tri.v1.position) + glm::abs(v * tri.v2.position))) +
                            (gamma(1) * (std::abs(1.0F - u) + std::abs(w)) * glm::abs(tri.v0.position));
    const float sign = normalSide ? 1.0F : -1.0F;
    const glm::vec3 side = sign * geoNormal;
    // Bound and lift are both >= 0 along side, so the lift only adds clearance and the round-away seats the sum past the bound.
    const float offset = glm::dot(glm::abs(geoNormal), error) + terminatorHeight(tri, w, u, v, flatPosition, sign, side);
    return roundAway(flatPosition + (offset * side), side);
}

glm::vec3 viewAdaptedNormal(const glm::vec3& normal, const glm::vec3& geoNormal, const glm::vec3& wo) {
    // On wo's side of the face: a back-face view keeps its normal opposing wo as the face's does, the side a dielectric reads.
    const float sign = glm::dot(wo, geoNormal) < 0.0F ? -1.0F : 1.0F;
    const glm::vec3 up = sign * geoNormal;
    const glm::vec3 shading = sign * normal;
    const glm::vec3 view = glm::normalize(wo);
    const float cosView = glm::dot(view, shading);
    const glm::vec3 mirror = (2.0F * cosView * shading) - view;
    if (cosView > 0.0F && glm::dot(mirror, up) >= 0.0F) {
        return normal;
    }
    // Keller et al. 2017: the mirror dropped onto the face's horizon and the normal re-bisected, equal to n where the mirror meets it.
    const glm::vec3 horizon = glm::normalize(mirror - (glm::dot(mirror, up) * up));
    return sign * glm::normalize(view + horizon);
}

}  // namespace pathtracer::scene
