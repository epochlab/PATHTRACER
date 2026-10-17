#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

namespace pathtracer::scene {

struct Ray {
    glm::vec3 origin;
    glm::vec3 dir;  // not required to be unit length
    float tMin;
    float tMax;
};

struct Triangle {
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
};

// A disk or sphere light's own surface, intersected as Embree's native primitive: exact, where a tessellation would bias NEE.
struct AnalyticShape {
    enum class Kind : std::uint8_t { Sphere, Disc };
    Kind kind;
    glm::vec3 centre;
    float radius;
    glm::mat3 frame;  // columns tangent, bitangent, normal: a disc's normal is its emitting side, a sphere's frame orients its uv
    int instanceIndex;  // the MeshInstance it belongs to, as ShadingTriangle::instanceIndex
};

struct Hit {
    float t;
    int triangleIndex;  // index into the Triangle list the intersection was built from; -1 on a shape
    float u;  // barycentric weight on v1 (Moller-Trumbore convention, w=1-u-v on v0)
    float v;  // barycentric weight on v2
    int shape = -1;  // index into the accel's AnalyticShape list on a shape hit, else -1
};

// Brute-force O(n) reference intersection: the correctness oracle for embree_validate, simple enough to be obviously right.
[[nodiscard]] std::optional<Hit> bruteForceIntersect(const std::vector<Triangle>& triangles,
                                                       const Ray& ray);

}  // namespace pathtracer::scene
