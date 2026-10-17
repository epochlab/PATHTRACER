#include "pathtracer/scene/embree_accel.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>

#include <embree4/rtcore.h>

namespace pathtracer::scene {

namespace {

void logEmbreeError(void* /*userPtr*/, RTCError code, const char* str) {
    std::cerr << "EmbreeAccel: " << rtcGetErrorString(code) << ": " << str << "\n";
}

// Signed: Embree passes a negative delta on free. Relaxed, being read only after rtcCommitScene joins the threads that wrote it.
std::atomic<std::int64_t> gEmbreeBytes{0};

// Embree's own BVH accounting, not an estimate. Called concurrently from its build threads, hence the atomic; true permits the alloc.
bool embreeMemoryMonitor(void* /*userPtr*/, ssize_t bytes, bool /*post*/) {
    gEmbreeBytes.fetch_add(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
    return true;
}

// Every shape of one kind as one point geometry, centre and radius as FLOAT4 and a disc's normal beside it; its geomID and members.
std::pair<unsigned int, std::vector<int>> attachShapes(RTCDevice device, RTCScene scene, const std::vector<AnalyticShape>& shapes,
                                                       AnalyticShape::Kind kind) {
    std::vector<int> members;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        if (shapes[i].kind == kind) {
            members.push_back(static_cast<int>(i));
        }
    }
    if (members.empty()) {
        return {RTC_INVALID_GEOMETRY_ID, std::move(members)};
    }
    const bool disc = kind == AnalyticShape::Kind::Disc;
    RTCGeometry geometry = rtcNewGeometry(device, disc ? RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT : RTC_GEOMETRY_TYPE_SPHERE_POINT);
    auto* points = static_cast<glm::vec4*>(
        rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(glm::vec4), members.size()));
    glm::vec3* normals =
        disc ? static_cast<glm::vec3*>(rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_NORMAL, 0, RTC_FORMAT_FLOAT3, sizeof(glm::vec3), members.size()))
             : nullptr;
    for (std::size_t i = 0; i < members.size(); ++i) {
        const AnalyticShape& shape = shapes[static_cast<std::size_t>(members[i])];
        points[i] = glm::vec4(shape.centre, shape.radius);
        if (disc) {
            normals[i] = shape.frame[2];
        }
    }
    rtcCommitGeometry(geometry);
    const unsigned int geomID = rtcAttachGeometry(scene, geometry);
    rtcReleaseGeometry(geometry);
    return {geomID, std::move(members)};
}

}  // namespace

std::size_t embreeAllocatedBytes() {
    const std::int64_t bytes = gEmbreeBytes.load(std::memory_order_relaxed);
    return bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
}

EmbreeAccel::EmbreeAccel(RTCDeviceTy* device, RTCSceneTy* scene, std::vector<Triangle> triangles, std::vector<AnalyticShape> shapes,
                         unsigned int triangleGeometry, std::vector<std::vector<int>> shapeOfPrimitive)
    : device_(device), scene_(scene), triangles_(std::move(triangles)), shapes_(std::move(shapes)), triangleGeometry_(triangleGeometry),
      shapeOfPrimitive_(std::move(shapeOfPrimitive)), triangleCount_(static_cast<int>(triangles_.size())) {}

EmbreeAccel::EmbreeAccel(EmbreeAccel&& other) noexcept
    : device_(std::exchange(other.device_, nullptr)),
      scene_(std::exchange(other.scene_, nullptr)),
      triangles_(std::move(other.triangles_)),
      shapes_(std::move(other.shapes_)),
      triangleGeometry_(other.triangleGeometry_),
      shapeOfPrimitive_(std::move(other.shapeOfPrimitive_)),
      triangleCount_(std::exchange(other.triangleCount_, 0)) {}

EmbreeAccel& EmbreeAccel::operator=(EmbreeAccel&& other) noexcept {
    if (this != &other) {
        release();
        device_ = std::exchange(other.device_, nullptr);
        scene_ = std::exchange(other.scene_, nullptr);
        triangles_ = std::move(other.triangles_);
        shapes_ = std::move(other.shapes_);
        triangleGeometry_ = other.triangleGeometry_;
        shapeOfPrimitive_ = std::move(other.shapeOfPrimitive_);
        triangleCount_ = std::exchange(other.triangleCount_, 0);
    }
    return *this;
}

EmbreeAccel::~EmbreeAccel() { release(); }

void EmbreeAccel::release() {
    if (scene_ != nullptr) {
        rtcReleaseScene(scene_);
        scene_ = nullptr;
    }
    if (device_ != nullptr) {
        rtcReleaseDevice(device_);
        device_ = nullptr;
    }
}

std::optional<EmbreeAccel> EmbreeAccel::build(std::vector<Triangle> triangles, std::vector<AnalyticShape> shapes) {
    RTCDevice device = rtcNewDevice(nullptr);
    if (device == nullptr) {
        std::cerr << "EmbreeAccel::build: rtcNewDevice failed\n";
        return std::nullopt;
    }
    rtcSetDeviceErrorFunction(device, logEmbreeError, nullptr);
    rtcSetDeviceMemoryMonitorFunction(device, embreeMemoryMonitor, nullptr);

    RTCScene scene = rtcNewScene(device);
    const int triangleCount = static_cast<int>(triangles.size());
    unsigned int triangleGeometry = RTC_INVALID_GEOMETRY_ID;
    std::vector<std::vector<int>> shapeOfPrimitive;

    if (triangleCount > 0) {
        RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);

        // triangles is non-indexed soup, three unique vertices each, shared directly as the vertex buffer with no copy.
        static_assert(sizeof(Triangle) == 3 * sizeof(glm::vec3),
                      "Triangle must be tightly packed for the shared vertex buffer stride below");
        // Embree reads the last vertex of a shared buffer with a 16-byte SSE load, so it needs a float of padding past it.
        triangles.reserve(triangles.size() + 1);
        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                    triangles.data(), 0, sizeof(glm::vec3),
                                    static_cast<std::size_t>(triangleCount) * 3);

        auto* indices = static_cast<unsigned int*>(rtcSetNewGeometryBuffer(
            geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned int),
            static_cast<std::size_t>(triangleCount)));
        for (int i = 0; i < triangleCount; ++i) {
            indices[(3 * i) + 0] = static_cast<unsigned int>((3 * i) + 0);
            indices[(3 * i) + 1] = static_cast<unsigned int>((3 * i) + 1);
            indices[(3 * i) + 2] = static_cast<unsigned int>((3 * i) + 2);
        }

        rtcCommitGeometry(geometry);
        triangleGeometry = rtcAttachGeometry(scene, geometry);
        rtcReleaseGeometry(geometry);
    }

    for (const AnalyticShape::Kind kind : {AnalyticShape::Kind::Sphere, AnalyticShape::Kind::Disc}) {
        if (auto [geomID, members] = attachShapes(device, scene, shapes, kind); !members.empty()) {
            shapeOfPrimitive.resize(std::max<std::size_t>(shapeOfPrimitive.size(), geomID + 1));
            shapeOfPrimitive[geomID] = std::move(members);
        }
    }

    rtcCommitScene(scene);

    if (rtcGetDeviceError(device) != RTC_ERROR_NONE) {
        rtcReleaseScene(scene);
        rtcReleaseDevice(device);
        return std::nullopt;
    }

    return EmbreeAccel(device, scene, std::move(triangles), std::move(shapes), triangleGeometry, std::move(shapeOfPrimitive));
}

std::optional<Hit> EmbreeAccel::intersect(const Ray& ray) const {
    RTCRayHit rayHit{};
    rayHit.ray.org_x = ray.origin.x;
    rayHit.ray.org_y = ray.origin.y;
    rayHit.ray.org_z = ray.origin.z;
    rayHit.ray.dir_x = ray.dir.x;
    rayHit.ray.dir_y = ray.dir.y;
    rayHit.ray.dir_z = ray.dir.z;
    rayHit.ray.tnear = ray.tMin;
    rayHit.ray.tfar = ray.tMax;
    rayHit.ray.mask = 0xFFFFFFFFU;
    rayHit.ray.flags = 0;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene_, &rayHit, nullptr);

    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return std::nullopt;
    }
    if (rayHit.hit.geomID == triangleGeometry_) {
        return Hit{rayHit.ray.tfar, static_cast<int>(rayHit.hit.primID), rayHit.hit.u, rayHit.hit.v};
    }
    return Hit{rayHit.ray.tfar, -1, 0.0F, 0.0F, shapeOfPrimitive_[rayHit.hit.geomID][rayHit.hit.primID]};
}

bool EmbreeAccel::occluded(const Ray& ray) const {
    RTCRay embreeRay{};
    embreeRay.org_x = ray.origin.x;
    embreeRay.org_y = ray.origin.y;
    embreeRay.org_z = ray.origin.z;
    embreeRay.dir_x = ray.dir.x;
    embreeRay.dir_y = ray.dir.y;
    embreeRay.dir_z = ray.dir.z;
    embreeRay.tnear = ray.tMin;
    embreeRay.tfar = ray.tMax;
    embreeRay.mask = 0xFFFFFFFFU;
    embreeRay.flags = 0;

    rtcOccluded1(scene_, &embreeRay, nullptr);

    // rtcOccluded1 signals a hit by setting tfar to -inf, per Embree convention: no hit record, occlusion being a boolean query.
    return embreeRay.tfar < 0.0F;
}

}  // namespace pathtracer::scene
