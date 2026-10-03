#include "pathtracer/scene/gbuffer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <optional>

#include "pathtracer/debug/aov_routing.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/false_color.h"
#include "pathtracer/scene/gbuffer_shading.h"

namespace pathtracer::scene {

namespace {

// The cube's 12 edges as corner-index pairs: the 4 of the min-z face, the 4 of the max-z face, then the 4 pillars.
constexpr std::array<std::array<int, 2>, 12> kBoxEdges{
    {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}}};
constexpr std::size_t kBoxCornerCount = 8;

// Fixed on-screen line thickness in pixels, constant regardless of triangle/box size or distance.
constexpr float kLineThicknessPx = 1.0F;

const glm::vec3 kWireframeColor(1.0F, 1.0F, 1.0F);

// One lens's forward model, chosen once per call: a pinhole stays an inline matrix product, a fisheye goes through Camera::project.
struct Projector {
    const Camera& camera;
    Camera::ViewBasis basis;
    std::optional<Camera::PinholeMatrix> pinhole;

    [[nodiscard]] std::optional<glm::vec2> operator()(const glm::vec4& point) const {
        return pinhole ? Camera::project(*pinhole, point) : camera.project(basis, point);
    }
};

// Both views on the current aspect: a resize or render-scale change re-frames the image, it moves nothing.
Projector projectorFor(const Camera& camera, float aspect) {
    const Camera::ViewBasis basis = camera.viewBasis(aspect);
    const bool isPinhole = basis.lens.projection == LensProjection::Rectilinear;
    return Projector{camera, basis, isPinhole ? std::optional(camera.pinholeMatrix(basis)) : std::nullopt};
}

// An instance box relative to the eye with its bounding cone (centre direction, half-angle), which rejects a pixel before any edge.
struct ViewBox {
    std::array<glm::vec3, kBoxCornerCount> corners;
    // Unit normals of the edges' eye planes as x, y, z rows, zero for an edge on a line through the eye: the reject vectorises over them.
    std::array<std::array<float, kBoxEdges.size()>, 3> edgeNormals;
    glm::vec3 centreDir;
    float cosHalfAngle;
    float sinHalfAngle;
    bool containsEye;
    glm::vec3 color;
};

// Corner order matches kBoxEdges. An eye inside the bounding sphere sees the box in every direction, so its cone never rejects.
ViewBox viewBox(const AabbBounds& box, const glm::vec3& eye, const glm::vec3& color) {
    ViewBox view{};
    view.corners = {glm::vec3(box.min.x, box.min.y, box.min.z) - eye, glm::vec3(box.max.x, box.min.y, box.min.z) - eye,
                    glm::vec3(box.max.x, box.max.y, box.min.z) - eye, glm::vec3(box.min.x, box.max.y, box.min.z) - eye,
                    glm::vec3(box.min.x, box.min.y, box.max.z) - eye, glm::vec3(box.max.x, box.min.y, box.max.z) - eye,
                    glm::vec3(box.max.x, box.max.y, box.max.z) - eye, glm::vec3(box.min.x, box.max.y, box.max.z) - eye};
    const glm::vec3 centre = (0.5F * (box.min + box.max)) - eye;
    const float radius = 0.5F * glm::length(box.max - box.min);
    const float distance = glm::length(centre);
    view.containsEye = distance <= radius;
    view.centreDir = view.containsEye ? glm::vec3(0.0F) : centre / distance;
    view.sinHalfAngle = view.containsEye ? 1.0F : radius / distance;
    view.cosHalfAngle = std::sqrt(1.0F - (view.sinHalfAngle * view.sinHalfAngle));
    for (std::size_t e = 0; e < kBoxEdges.size(); ++e) {
        const glm::vec3 normal = glm::cross(view.corners[static_cast<std::size_t>(kBoxEdges[e][0])],
                                            view.corners[static_cast<std::size_t>(kBoxEdges[e][1])]);
        const float length = glm::length(normal);
        const glm::vec3 unit = length > 0.0F ? normal / length : glm::vec3(0.0F);
        for (int axis = 0; axis < 3; ++axis) {
            view.edgeNormals[static_cast<std::size_t>(axis)][e] = unit[axis];
        }
    }
    view.color = color;
    return view;
}

// Everything a row needs that is constant across the call, built once rather than per pixel.
struct GBufferView {
    const Camera& camera;
    Camera::ViewBasis basis;
    Projector current;
    Projector previous;
    glm::vec2 pixelsPerNdc;  // (W/2, -H/2): NDC +Y up to pixels with row 0 at the top
    std::vector<ViewBox> boxes;
};

// A pixel-centre ray in the local linear image model d(p + dp) = d + J dp (Igehy 1999), J's columns d(dir)/d(pixel x, y).
struct PixelRay {
    Ray ray;
    glm::mat2x3 dirPerPixel;
};

// J's pseudo-inverse (J^T J)^-1 J^T: a tangent-plane offset at d back to pixels, built only for a pixel an edge survives at.
glm::mat3x2 pixelPerDir(const PixelRay& pixel) {
    const glm::mat3x2 transposed = glm::transpose(pixel.dirPerPixel);
    return glm::inverse(transposed * pixel.dirPerPixel) * transposed;
}

std::optional<PixelRay> pixelRay(const GBufferView& view, int x, int y) {
    const float ndcX = ((static_cast<float>(x) + 0.5F) / view.pixelsPerNdc.x) - 1.0F;
    const float ndcY = 1.0F + ((static_cast<float>(y) + 0.5F) / view.pixelsPerNdc.y);
    const std::optional<Camera::RayDifferential> differential = view.camera.primaryRayDifferential(view.basis, ndcX, ndcY);
    if (!differential) {
        return std::nullopt;
    }
    const glm::mat2x3 dirPerPixel = differential->dirPerNdc * glm::mat2(1.0F / view.pixelsPerNdc.x, 0.0F, 0.0F,
                                                                        1.0F / view.pixelsPerNdc.y);
    return PixelRay{differential->ray, dirPerPixel};
}

// Pixel distance to the image of the eye plane with normal n, |n.d| / |J^T n| (Baerentzen et al. 2006); NaN for n = 0 fails every test.
float planeDistancePx(const glm::vec3& normal, const PixelRay& pixel) {
    return std::abs(glm::dot(normal, pixel.ray.dir)) / glm::length(glm::transpose(pixel.dirPerPixel) * normal);
}

// Inside the hit triangle's image the nearest boundary point lies on one of its three edge planes: a spherical triangle is convex.
bool nearTriangleEdge(const ShadingTriangle& triangle, const PixelRay& pixel) {
    const glm::vec3 a = triangle.v0.position - pixel.ray.origin;
    const glm::vec3 b = triangle.v1.position - pixel.ray.origin;
    const glm::vec3 c = triangle.v2.position - pixel.ray.origin;
    return planeDistancePx(glm::cross(a, b), pixel) < kLineThicknessPx ||
           planeDistancePx(glm::cross(b, c), pixel) < kLineThicknessPx ||
           planeDistancePx(glm::cross(c, a), pixel) < kLineThicknessPx;
}

// The pixel's angular reach, atan(thickness * |J|_F) >= the angle any point within the line thickness subtends, against the cone.
bool coneRejects(const ViewBox& box, const PixelRay& pixel, float cosReach, float sinReach) {
    return !box.containsEye &&
           glm::dot(pixel.ray.dir, box.centreDir) < (box.cosHalfAngle * cosReach) - (box.sinHalfAngle * sinReach);
}

// Edges whose eye plane lies within the reach of d, as a bit mask: (n.d)^2 = sin^2 of the angle to the plane, against sin^2(reach).
unsigned edgesWithinReach(const ViewBox& box, const glm::vec3& d, float sinReachSquared) {
    unsigned mask = 0;
    for (std::size_t e = 0; e < kBoxEdges.size(); ++e) {
        const float dot = (box.edgeNormals[0][e] * d.x) + (box.edgeNormals[1][e] * d.y) + (box.edgeNormals[2][e] * d.z);
        mask |= static_cast<unsigned>(dot * dot <= sinReachSquared) << e;
    }
    return mask;
}

// A box edge in the gnomonic plane at d, where a segment images as a segment: its nearest point in pixels and that point's 3D range.
std::optional<float> edgeRange(const glm::vec3& a, const glm::vec3& b, float da, float db, glm::vec2 qa, glm::vec2 qb,
                               const glm::mat3x2& toPixels) {
    if (da <= 0.0F && db <= 0.0F) {
        return std::nullopt;
    }
    // A front endpoint first: the other is either in front too, imaging a segment, or behind, imaging a half-line to infinity.
    const bool swap = da <= 0.0F;
    const glm::vec3& front = swap ? b : a;
    const glm::vec3& back = swap ? a : b;
    const float dFront = swap ? db : da;
    const float dBack = swap ? da : db;
    const glm::vec2 qFront = swap ? qb : qa;
    float s = 0.0F;
    if (dBack > 0.0F) {
        const LineProximity prox = nearLineSegmentPx(glm::vec2(0.0F), qFront, swap ? qa : qb, kLineThicknessPx);
        if (!prox.near) {
            return std::nullopt;
        }
        // The gnomonic map is projective along the edge: image parameter t is edge parameter s = t dF / ((1 - t) dB + t dF).
        s = (prox.t * dFront) / (((1.0F - prox.t) * dBack) + (prox.t * dFront));
    } else {
        // The edge leaves d's hemisphere at s0, X(s0) perpendicular to d: the image's direction at infinity, and lambda maps back to s.
        const float s0 = dFront / (dFront - dBack);
        const glm::vec2 direction = toPixels * (front + (s0 * (back - front)));
        const float lambda = std::max(0.0F, -glm::dot(qFront, direction) / glm::dot(direction, direction));
        if (!(glm::length(qFront + (lambda * direction)) < kLineThicknessPx)) {
            return std::nullopt;
        }
        s = (lambda * dFront * s0) / (1.0F + (lambda * dFront));
    }
    return glm::length(front + (s * (back - front)));
}

// The colour of the last box whose edge lies within the line thickness and before the first surface; geometry occludes edges only.
std::optional<glm::vec3> nearBoxEdge(const std::vector<ViewBox>& boxes, const PixelRay& pixel, float rangeLimit) {
    const glm::mat2x3& dirPerPixel = pixel.dirPerPixel;
    const float reachTanSquared = kLineThicknessPx * kLineThicknessPx *
                                  (glm::dot(dirPerPixel[0], dirPerPixel[0]) + glm::dot(dirPerPixel[1], dirPerPixel[1]));
    const float cosReach = 1.0F / std::sqrt(1.0F + reachTanSquared);
    const float sinReach = std::sqrt(reachTanSquared) * cosReach;
    const float sinReachSquared = reachTanSquared / (1.0F + reachTanSquared);
    std::optional<glm::vec3> color;
    std::optional<glm::mat3x2> toPixels;
    const glm::vec3& d = pixel.ray.dir;
    for (const ViewBox& box : boxes) {
        if (coneRejects(box, pixel, cosReach, sinReach)) {
            continue;
        }
        // Any point within the reach of d lies within it of the edge's eye plane too, so only edges in the mask can be near.
        for (unsigned mask = edgesWithinReach(box, d, sinReachSquared); mask != 0; mask &= mask - 1) {
            const auto e = static_cast<std::size_t>(std::countr_zero(mask));
            if (!toPixels) {
                toPixels = pixelPerDir(pixel);
            }
            const glm::vec3& a = box.corners[static_cast<std::size_t>(kBoxEdges[e][0])];
            const glm::vec3& b = box.corners[static_cast<std::size_t>(kBoxEdges[e][1])];
            const float da = glm::dot(d, a);
            const float db = glm::dot(d, b);
            // Each endpoint's gnomonic image C/(d.C) - d in pixel offsets; meaningful only where d.C > 0.
            const glm::vec2 qa = da > 0.0F ? *toPixels * ((a / da) - d) : glm::vec2(0.0F);
            const glm::vec2 qb = db > 0.0F ? *toPixels * ((b / db) - d) : glm::vec2(0.0F);
            const std::optional<float> range = edgeRange(a, b, da, db, qa, qb, *toPixels);
            if (range && *range >= pixel.ray.tMin && *range <= rangeLimit) {
                color = box.color;
                break;
            }
        }
    }
    return color;
}

// Resolves every surface field for one hit pixel: tracePath's bounce-0 sampling calls, never lighting or BSDF. Returns worldPos.
glm::vec3 shadeHit(GBuffer& result, int x, int y, const Hit& hit, const PixelRay& pixel, const ShadingTriangle& triangle,
              const Material& material, const PathTraceSettings& settings) {
    const ShadingVertex shading = interpolateShading(triangle, hit.u, hit.v);
    const ShadingFrame frame = buildShadingFrame(shading, material, settings);
    const BsdfParams params = resolveBsdfParams(material, shading.uv, shading.colour, settings, std::nullopt);
    writeTexel(result.depth, x, y, hit.t);
    writeTexel(result.lookahead, x, y, std::clamp(1.0F - (hit.t / settings.lookaheadDistance), 0.0F, 1.0F));
    writeTexel(result.worldPos, x, y, shading.position);
    writeTexel(result.uv, x, y, glm::fract(shading.uv));
    writeTexel(result.normal, x, y, frame[2]);
    writeTexel(result.geomNormal, x, y, glm::normalize(shading.normal));
    writeTexel(result.albedo, x, y, params.baseColor);
    writeTexel(result.metallic, x, y, params.metallic);
    writeTexel(result.roughness, x, y, params.roughness);
    writeTexel(result.tangent, x, y, frame[0]);
    writeTexel(result.objectId, x, y, falseColorForId(triangle.instanceIndex));
    writeTexel(result.alpha, x, y, 1.0F);
    writeTexel(result.iorAov, x, y, settings.ior);
    if (nearTriangleEdge(triangle, pixel)) {
        writeTexel(result.wireframe, x, y, kWireframeColor);
    }
    return shading.position;
}

// The scene a G-buffer call reads, bundled so a row's signature stays short.
struct GBufferScene {
    const EmbreeAccel& accel;
    const std::vector<ShadingTriangle>& shadingTriangles;
    const std::vector<MeshInstance>& instances;
    const std::vector<PathTraceSettings>& perInstanceSettings;
};

// x_now - x_previous by one projection of one point on both sides, so identical cameras give exactly zero; a miss moves by rotation.
void writeMotion(GBuffer& result, const GBufferView& view, int x, int y, const glm::vec4& point) {
    const std::optional<glm::vec2> now = view.current(point);
    const std::optional<glm::vec2> before = view.previous(point);
    writeTexel(result.motionVector, x, y, now && before ? (*now - *before) * view.pixelsPerNdc : glm::vec2(0.0F));
}

void renderPixel(GBuffer& result, const GBufferView& view, const GBufferScene& scene, int x, int y) {
    // Past a fisheye's image circle the lens images nothing: every lane keeps its cleared background value.
    const std::optional<PixelRay> pixel = pixelRay(view, x, y);
    if (!pixel) {
        return;
    }
    const std::optional<Hit> hit = scene.accel.intersect(pixel->ray);
    // A miss sees the environment, the plane at infinity, along the ray direction (w = 0); a hit is its worldPos (w = 1).
    glm::vec4 seen(pixel->ray.dir, 0.0F);
    if (hit) {
        const ShadingTriangle& triangle = scene.shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)];
        const auto instance = static_cast<std::size_t>(triangle.instanceIndex);
        seen = glm::vec4(shadeHit(result, x, y, *hit, *pixel, triangle, scene.instances[instance].material,
                                  scene.perInstanceSettings[instance]),
                         1.0F);
    }
    writeMotion(result, view, x, y, seen);
    if (const std::optional<glm::vec3> boxColor = nearBoxEdge(view.boxes, *pixel, hit ? hit->t : pixel->ray.tMax)) {
        writeTexel(result.wireframe, x, y, *boxColor);
    }
}

// Reallocated only on a resolution change; every other call reuses the storage and relies on the per-row clear.
void ensureLanes(GBuffer& result, int width, int height) {
    if (result.depth.width == width && result.depth.height == height) {
        return;
    }
    for (int i = 0; i < static_cast<int>(pathtracer::debug::AovId::Count); ++i) {
        const auto aov = static_cast<pathtracer::debug::AovId>(i);
        if (const pathtracer::debug::GBufferLane lane = pathtracer::debug::gbufferLane(aov)) {
            result.*lane = pathtracer::gfx::makeImage(width, height, pathtracer::debug::aovChannels(aov));
        }
    }
}

// Clearing a row of every lane is what makes the buffers reusable: zeroed while cache-warm, IOR's background -1 included.
void clearRow(GBuffer& result, int y) {
    const int width = result.depth.width;
    const std::size_t rowStart = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
    for (const pathtracer::debug::GBufferLane lane : pathtracer::debug::gbufferLanes()) {
        pathtracer::gfx::HdrImage& image = result.*lane;
        const auto channels = static_cast<std::size_t>(image.channels);
        float* row = image.texels.data() + (rowStart * channels);
        std::fill(row, row + (static_cast<std::size_t>(width) * channels), 0.0F);
    }
    for (int x = 0; x < width; ++x) {
        writeTexel(result.iorAov, x, y, -1.0F);
    }
}

}  // namespace

void renderGBuffer(const Camera& camera, const Camera& previousCamera, const EmbreeAccel& accel,
                   const std::vector<ShadingTriangle>& shadingTriangles, const std::vector<MeshInstance>& instances,
                   const std::vector<PathTraceSettings>& perInstanceSettings,
                   const std::vector<AabbBounds>& instanceBounds, int width, int height, ThreadPool& threadPool,
                   GBuffer& result) {
    ensureLanes(result, width, height);
    ++result.generation;

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    GBufferView view{camera, camera.viewBasis(aspect), projectorFor(camera, aspect), projectorFor(previousCamera, aspect),
                     glm::vec2(0.5F * static_cast<float>(width), -0.5F * static_cast<float>(height)), {}};
    // One box per instance in its ObjectID false colour; an instance that contributed no triangles has an empty box and no edges.
    view.boxes.reserve(instanceBounds.size());
    for (std::size_t i = 0; i < instanceBounds.size(); ++i) {
        if (!isEmpty(instanceBounds[i])) {
            view.boxes.push_back(viewBox(instanceBounds[i], camera.position(), falseColorForId(static_cast<int>(i))));
        }
    }
    const GBufferScene scene{accel, shadingTriangles, instances, perInstanceSettings};
    threadPool.parallelFor(height, [&](int y) {
        clearRow(result, y);
        for (int x = 0; x < width; ++x) {
            renderPixel(result, view, scene, x, y);
        }
    });
}

}  // namespace pathtracer::scene
