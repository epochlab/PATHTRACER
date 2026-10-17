#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "pathtracer/scene/ray_types.h"

using RTCDeviceTy = struct RTCDeviceTy;
using RTCSceneTy = struct RTCSceneTy;

namespace pathtracer::scene {

// Ray-scene intersection over Embree's SIMD BVH: a static triangle soup plus native spheres and discs, built once at load.
class EmbreeAccel {
public:
    ~EmbreeAccel();

    EmbreeAccel(const EmbreeAccel&) = delete;
    EmbreeAccel& operator=(const EmbreeAccel&) = delete;
    EmbreeAccel(EmbreeAccel&& other) noexcept;
    EmbreeAccel& operator=(EmbreeAccel&& other) noexcept;

    // nullopt if the Embree device or scene fails to initialize, logged to stderr: a real failure mode, surfaced rather than aborting.
    static std::optional<EmbreeAccel> build(std::vector<Triangle> triangles, std::vector<AnalyticShape> shapes = {});

    [[nodiscard]] std::optional<Hit> intersect(const Ray& ray) const;

    // Any-hit query for shadow rays: true if anything blocks [tMin, tMax], without finding the closest -- all NEE needs, and cheaper.
    [[nodiscard]] bool occluded(const Ray& ray) const;

    [[nodiscard]] int triangleCount() const { return triangleCount_; }

    // Indexed by Hit::shape.
    [[nodiscard]] const std::vector<AnalyticShape>& shapes() const { return shapes_; }

private:
    EmbreeAccel(RTCDeviceTy* device, RTCSceneTy* scene, std::vector<Triangle> triangles, std::vector<AnalyticShape> shapes,
                unsigned int triangleGeometry, std::vector<std::vector<int>> shapeOfPrimitive);
    void release();

    RTCDeviceTy* device_ = nullptr;
    RTCSceneTy* scene_ = nullptr;
    std::vector<Triangle> triangles_;  // kept alive: the vertex buffer shares this memory with Embree
    std::vector<AnalyticShape> shapes_;
    unsigned int triangleGeometry_;  // the triangles' geomID, RTC_INVALID_GEOMETRY_ID when there are none
    std::vector<std::vector<int>> shapeOfPrimitive_;  // [geomID][primID] -> index into shapes_ for the point geometries
    int triangleCount_ = 0;
};

// Bytes Embree has allocated for BVH and geometry, through its own device memory monitor rather than an estimate from triangle count.
[[nodiscard]] std::size_t embreeAllocatedBytes();

}  // namespace pathtracer::scene
