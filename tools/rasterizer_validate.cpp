// renderRasterGBuffer against a pixel-centre Embree ray through the same sampling functions; silhouette-edge mismatches are inherent.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/false_color.h"
#include "pathtracer/scene/gbuffer_shading.h"
#include "pathtracer/scene/gltf_loader.h"
#include "check.h"
#include "pathtracer/scene/rasterizer.h"
#include "pathtracer/scene/ray_types.h"
#include "pathtracer/scene/shading_scene.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using namespace pathtracer::scene;  // NOLINT(google-build-using-namespace) -- tool-local, mirrors embree_validate.cpp

constexpr int kWidth = 96;
constexpr int kHeight = 96;
constexpr int kTriangleCount = 150;
constexpr int kMaterialCount = 4;
// Depth span the triangle centres are drawn over; the lookahead check derives its horizon from it, so one outside would blind the check.
constexpr float kSceneNearZ = 1.0F;
constexpr float kSceneFarZ = 16.0F;
constexpr float kPosEpsilon = 5e-2F;    // world-space units (worldPos, depth)
constexpr float kUnitEpsilon = 1e-2F;   // unit-vector/[0,1]-range fields (normal, uv, albedo, ...)
constexpr float kMaxCoverageMismatchFraction = 0.02F;
constexpr float kMaxValueMismatchFraction = 0.02F;
// Two float projections, each within camera_validate's 16-ulp closed-form budget of an NDC scaled by kWidth/2 pixels.
constexpr double kMotionEpsilonPx = 16.0 * kWidth * std::numeric_limits<float>::epsilon();

// The neutral default material, varying only the two slots the G-buffer AOVs under test read.
Material makeMaterial(glm::vec3 baseColor, float roughness) {
    return Material{.baseColor = baseColor, .roughness = roughness};
}

glm::vec4 tangentFor(const glm::vec3& normal) {
    const glm::vec3 up = std::fabs(normal.y) < 0.99F ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::vec3(1.0F, 0.0F, 0.0F);
    return glm::vec4(glm::normalize(glm::cross(up, normal)), 1.0F);
}

// Random triangles overlapping in depth, dense enough near z=-1 that a camera inside the cluster clips some at the near plane.
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

struct FieldCheck {
    const char* name;
    glm::vec3 raster;
    glm::vec3 oracle;
    float epsilon;
};

int checkFields(const std::vector<FieldCheck>& fields, int x, int y, const char* poseName) {
    int mismatches = 0;
    for (const FieldCheck& f : fields) {
        if (glm::any(glm::greaterThan(glm::abs(f.raster - f.oracle), glm::vec3(f.epsilon)))) {
            std::cerr << "rasterizer_validate: " << poseName << " (" << x << "," << y << ") " << f.name
                      << " mismatch (raster=" << f.raster.x << "," << f.raster.y << "," << f.raster.z
                      << " oracle=" << f.oracle.x << "," << f.oracle.y << "," << f.oracle.z << ")\n";
            ++mismatches;
        }
    }
    return mismatches;
}

// Oracle: unjittered pixel-centre Embree intersection through the same gbuffer_shading.h functions, independent of scan-conversion.
bool checkPose(const char* poseName, const Camera& camera, const EmbreeAccel& accel,
               const std::vector<ShadingTriangle>& shadingTriangles,
               const std::vector<MeshInstance>& instances,
               const std::vector<PathTraceSettings>& perInstanceSettings,
               const std::vector<AabbBounds>& instanceBounds, ThreadPool& threadPool) {
    RasterGBuffer raster;
    renderRasterGBuffer(camera, camera, shadingTriangles, instances, perInstanceSettings, instanceBounds, kWidth,
                         kHeight, threadPool, raster);
    const float aspect = static_cast<float>(kWidth) / static_cast<float>(kHeight);
    const glm::vec3 camPos = camera.position();
    const glm::vec3 camForward = camera.forward();

    int coverageMismatches = 0;
    int valueMismatches = 0;
    int hitPixels = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const float ndcX = (((static_cast<float>(x) + 0.5F) / static_cast<float>(kWidth)) * 2.0F) - 1.0F;
            const float ndcY = 1.0F - (((static_cast<float>(y) + 0.5F) / static_cast<float>(kHeight)) * 2.0F);
            // The rasterizer's own oracle, so the camera is rectilinear by construction and every ndc point images.
            const Ray ray = camera.primaryRay(ndcX, ndcY, aspect).value();
            const std::optional<Hit> hit = accel.intersect(ray);
            const float rasterAlpha = texelAt(raster.alpha, x, y).x;
            const bool rasterHit = rasterAlpha > 0.5F;

            if (rasterHit != hit.has_value()) {
                ++coverageMismatches;
                continue;
            }
            if (!hit.has_value()) {
                continue;
            }
            ++hitPixels;

            const ShadingTriangle& triangle = shadingTriangles[static_cast<std::size_t>(hit->triangleIndex)];
            const Material& material = instances[static_cast<std::size_t>(triangle.instanceIndex)].material;
            const PathTraceSettings& settings =
                perInstanceSettings[static_cast<std::size_t>(triangle.instanceIndex)];
            const ShadingVertex shading = interpolateShading(triangle, hit->u, hit->v);
            const ShadingFrame frame = buildShadingFrame(shading, material, settings);
            const BsdfParams params =
                resolveBsdfParams(material, shading.uv, shading.colour, settings, std::nullopt);
            const float depth = glm::dot(shading.position - camPos, camForward);

            const std::vector<FieldCheck> fields{
                {"depth", texelAt(raster.depth, x, y), glm::vec3(depth), kPosEpsilon},
                {"lookahead", texelAt(raster.lookahead, x, y),
                 glm::vec3(std::clamp(1.0F - (depth / settings.lookaheadDistance), 0.0F, 1.0F)),
                 kUnitEpsilon},
                {"worldPos", texelAt(raster.worldPos, x, y), shading.position, kPosEpsilon},
                {"uv", texelAt(raster.uv, x, y), glm::vec3(glm::fract(shading.uv), 0.0F), kUnitEpsilon},
                {"normal", texelAt(raster.normal, x, y), frame[2], kUnitEpsilon},
                {"geomNormal", texelAt(raster.geomNormal, x, y), glm::normalize(shading.normal), kUnitEpsilon},
                {"albedo", texelAt(raster.albedo, x, y), params.baseColor, kUnitEpsilon},
                {"metallic", texelAt(raster.metallic, x, y), glm::vec3(params.metallic), kUnitEpsilon},
                {"roughness", texelAt(raster.roughness, x, y), glm::vec3(params.roughness), kUnitEpsilon},
                {"tangent", texelAt(raster.tangent, x, y), frame[0], kUnitEpsilon},
                {"objectId", texelAt(raster.objectId, x, y), falseColorForId(triangle.instanceIndex), kUnitEpsilon},
                {"iorAov", texelAt(raster.iorAov, x, y), glm::vec3(settings.ior), kUnitEpsilon},
            };
            valueMismatches += checkFields(fields, x, y, poseName);
        }
    }

    const int totalPixels = kWidth * kHeight;
    const float coverageFraction = static_cast<float>(coverageMismatches) / static_cast<float>(totalPixels);
    const float valueFraction =
        hitPixels > 0 ? static_cast<float>(valueMismatches) / static_cast<float>(hitPixels) : 0.0F;
    std::cout << "rasterizer_validate: " << poseName << " -- " << hitPixels << " hit pixels, "
              << coverageMismatches << " coverage mismatches, " << valueMismatches << " value mismatches\n";

    if (coverageFraction > kMaxCoverageMismatchFraction || valueFraction > kMaxValueMismatchFraction) {
        std::cerr << "rasterizer_validate: " << poseName << " FAILED (coverage " << coverageFraction * 100.0F
                  << "%, value " << valueFraction * 100.0F << "%)\n";
        return false;
    }
    return true;
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

// Wireframe is combined: white mesh edges, each instance's falseColorForId hue on its boxes. The exact hue says which instance drew it.
bool isBoxColorOf(glm::vec3 c, int instanceIndex) {
    return glm::all(glm::lessThan(glm::abs(c - falseColorForId(instanceIndex)), glm::vec3(kUnitEpsilon)));
}

bool isWireframeColor(glm::vec3 c) {
    return c.z > 0.5F;
}

// Screen-space extent of one instance's box pixels: count plus the bounding box they occupy, inverted when the count is 0.
struct BoxPixels {
    int count;
    int minX;
    int maxX;
};

BoxPixels boxPixelsOf(const RasterGBuffer& raster, int instanceIndex) {
    BoxPixels found{0, kWidth, -1};
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            if (!isBoxColorOf(texelAt(raster.wireframe, x, y), instanceIndex)) {
                continue;
            }
            ++found.count;
            found.minX = std::min(found.minX, x);
            found.maxX = std::max(found.maxX, x);
        }
    }
    return found;
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

// Regression check: box edges are z-tested against scene geometry, so a solid occluder in front should hide the edges behind it.
bool checkBoundingBoxOcclusion(ThreadPool& threadPool) {
    const Camera::FilmBack filmBack{36.0F, 24.0F};
    const Camera camera(glm::vec3(0.0F), 0.0F, 0.0F, filmBack, 35.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F,
                        100.0F);
    const std::vector<MeshInstance> instances{
        MeshInstance{makeMaterial(glm::vec3(0.5F), 0.5F), glm::mat4(1.0F), ""}};
    const std::vector<PathTraceSettings> perInstanceSettings(instances.size(), makeTestSettings());
    const int instanceCount = static_cast<int>(instances.size());

    const ShadingTriangle farMarker = makeTinyTriangle(glm::vec3(-0.5F, -0.5F, -15.0F), 0.05F, 0);
    const ShadingTriangle nearMarker = makeTinyTriangle(glm::vec3(0.5F, 0.5F, -5.0F), 0.05F, 0);

    // Case 1: box spans x/y in [-0.5,0.5], z in [-15,-5], nothing solid anywhere -- all edges eligible.
    const std::vector<ShadingTriangle> scene{farMarker, nearMarker};
    RasterGBuffer raster;
    renderRasterGBuffer(camera, camera, scene, instances, perInstanceSettings,
                         computeInstanceBounds(scene, instanceCount), kWidth, kHeight, threadPool, raster);
    const int unoccluded = boxPixelsOf(raster, 0).count;
    std::cout << "rasterizer_validate: boundingBox unoccluded -- " << unoccluded << " pixels\n";
    if (unoccluded == 0) {
        std::cerr << "rasterizer_validate: checkBoundingBoxOcclusion FAILED -- no box edges "
                     "visible with nothing in front of the box\n";
        return false;
    }

    // Case 2: occluder at z=-4, nearer than the near corner at z=-5: its surface becomes the new near face, so those 4 edges stay visible.
    const std::array<ShadingTriangle, 2> occluder = makeQuad(0.5F, glm::vec3(0.0F, 0.0F, -4.0F), 0);
    const std::vector<ShadingTriangle> occludedScene{farMarker, occluder[0], occluder[1]};
    RasterGBuffer occludedRaster;
    renderRasterGBuffer(camera, camera, occludedScene, instances, perInstanceSettings,
                         computeInstanceBounds(occludedScene, instanceCount), kWidth, kHeight, threadPool,
                         occludedRaster);
    const int occluded = boxPixelsOf(occludedRaster, 0).count;
    std::cout << "rasterizer_validate: boundingBox occluded -- " << occluded << " pixels\n";
    if (occluded * 2 >= unoccluded) {
        std::cerr << "rasterizer_validate: checkBoundingBoxOcclusion FAILED -- occluded pixel count ("
                  << occluded << ") not substantially lower than unoccluded (" << unoccluded << ")\n";
        return false;
    }
    return true;
}

// Regression check for one box per instance: two separated quads must draw disjoint boxes, where a single fused box would span both.
bool checkPerInstanceBoxes(ThreadPool& threadPool) {
    const Camera::FilmBack filmBack{36.0F, 24.0F};
    const Camera camera(glm::vec3(0.0F), 0.0F, 0.0F, filmBack, 35.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F,
                        100.0F);
    const std::vector<MeshInstance> instances{
        MeshInstance{makeMaterial(glm::vec3(0.5F), 0.5F), glm::mat4(1.0F), ""},
        MeshInstance{makeMaterial(glm::vec3(0.5F), 0.5F), glm::mat4(1.0F), ""}};
    const std::vector<PathTraceSettings> perInstanceSettings(instances.size(), makeTestSettings());

    // Both at z=-10, where the 35mm lens on 36mm film sees x in about +-5.1, leaving both fully on screen with a clear gap between them.
    const std::array<ShadingTriangle, 2> left = makeQuad(0.5F, glm::vec3(-2.0F, 0.0F, -10.0F), 0);
    const std::array<ShadingTriangle, 2> right = makeQuad(0.5F, glm::vec3(2.0F, 0.0F, -10.0F), 1);
    const std::vector<ShadingTriangle> scene{left[0], left[1], right[0], right[1]};

    RasterGBuffer raster;
    renderRasterGBuffer(camera, camera, scene, instances, perInstanceSettings,
                         computeInstanceBounds(scene, static_cast<int>(instances.size())), kWidth, kHeight,
                         threadPool, raster);
    const BoxPixels leftBox = boxPixelsOf(raster, 0);
    const BoxPixels rightBox = boxPixelsOf(raster, 1);
    std::cout << "rasterizer_validate: perInstanceBoxes -- instance 0 " << leftBox.count << " px [x "
              << leftBox.minX << "," << leftBox.maxX << "], instance 1 " << rightBox.count << " px [x "
              << rightBox.minX << "," << rightBox.maxX << "]\n";
    if (leftBox.count == 0 || rightBox.count == 0) {
        std::cerr << "rasterizer_validate: checkPerInstanceBoxes FAILED -- an instance drew no box "
                     "edges in its own false colour\n";
        return false;
    }
    if (leftBox.maxX >= rightBox.minX) {
        std::cerr << "rasterizer_validate: checkPerInstanceBoxes FAILED -- the two boxes overlap in x ("
                  << leftBox.maxX << " >= " << rightBox.minX << "), expected one box per instance\n";
        return false;
    }
    return true;
}

// Regression check: must draw something (not silently empty) but only a thin fraction of hit pixels, not most of the mesh.
bool checkWireframeSanity(const Camera& camera, const std::vector<ShadingTriangle>& shadingTriangles,
                           const std::vector<MeshInstance>& instances,
                           const std::vector<PathTraceSettings>& perInstanceSettings,
                           const std::vector<AabbBounds>& instanceBounds, ThreadPool& threadPool) {
    RasterGBuffer raster;
    renderRasterGBuffer(camera, camera, shadingTriangles, instances, perInstanceSettings, instanceBounds, kWidth,
                         kHeight, threadPool, raster);
    int hitPixels = 0;
    int wirePixels = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            if (texelAt(raster.alpha, x, y).x > 0.5F) {
                ++hitPixels;
                if (isWireframeColor(texelAt(raster.wireframe, x, y))) {
                    ++wirePixels;
                }
            }
        }
    }
    std::cout << "rasterizer_validate: wireframe -- " << wirePixels << " / " << hitPixels << " hit pixels\n";
    if (wirePixels == 0) {
        std::cerr << "rasterizer_validate: checkWireframeSanity FAILED -- no wireframe pixels drawn\n";
        return false;
    }
    if (hitPixels > 0 && (wirePixels * 2) > hitPixels) {
        std::cerr << "rasterizer_validate: checkWireframeSanity FAILED -- wireframe covers over "
                     "half of all hit pixels, expected thin edges only\n";
        return false;
    }
    return true;
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
                lattice[index({i, j, k})] =
                    ((glm::vec3(i, j, k) - (static_cast<float>(n) * 0.5F)) * cell) + jitterOffset;
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

int uncoveredPixels(const RasterGBuffer& raster) {
    int uncovered = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            uncovered += texelAt(raster.alpha, x, y).x > 0.5F ? 0 : 1;
        }
    }
    return uncovered;
}

// Everything the pose checks share, rebuilt per check: construction is milliseconds, and a check building its own world orders no other.
struct RasterFixture {
    std::vector<ShadingTriangle> shadingTriangles;
    std::vector<Triangle> worldTriangles;
    std::optional<EmbreeAccel> accel;
    std::vector<MeshInstance> instances;
    std::vector<PathTraceSettings> perInstanceSettings;
    std::vector<AabbBounds> instanceBounds;
    ThreadPool threadPool;
};

std::unique_ptr<RasterFixture> makeFixture(const tools::check::Context& ctx) {
    auto fixture = std::make_unique<RasterFixture>();
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    fixture->shadingTriangles = makeSyntheticTriangles(rng);
    fixture->worldTriangles = worldTrianglesOf(fixture->shadingTriangles);
    fixture->accel = EmbreeAccel::build(fixture->worldTriangles);
    fixture->instances.push_back(MeshInstance{makeMaterial(glm::vec3(0.8F, 0.2F, 0.2F), 0.2F), glm::mat4(1.0F), ""});
    fixture->instances.push_back(MeshInstance{makeMaterial(glm::vec3(0.2F, 0.8F, 0.2F), 0.5F), glm::mat4(1.0F), ""});
    fixture->instances.push_back(MeshInstance{makeMaterial(glm::vec3(0.2F, 0.2F, 0.8F), 0.8F), glm::mat4(1.0F), ""});
    fixture->instances.push_back(MeshInstance{makeMaterial(glm::vec3(0.8F, 0.8F, 0.2F), 1.0F), glm::mat4(1.0F), ""});
    fixture->perInstanceSettings.assign(fixture->instances.size(), makeTestSettings());
    fixture->instanceBounds =
        computeInstanceBounds(fixture->shadingTriangles, static_cast<int>(fixture->instances.size()));
    return fixture;
}

constexpr Camera::FilmBack kFilmBack{36.0F, 24.0F};

Camera straightOnCamera() {
    return Camera(glm::vec3(0.0F, 0.0F, 0.0F), 0.0F, 0.0F, kFilmBack, 35.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F, 100.0F);
}

// Runs one pose's full G-buffer cross-check against the independent per-pixel Embree oracle.
void runPose(tools::check::Context& ctx, const char* name, const Camera& camera) {
    const std::unique_ptr<RasterFixture> fixture = makeFixture(ctx);
    ctx.plan(1);
    if (!fixture->accel) {
        PT_EXPECT(ctx, false, "EmbreeAccel::build failed");
        return;
    }
    const bool passed =
        checkPose(name, camera, *fixture->accel, fixture->shadingTriangles, fixture->instances,
                   fixture->perInstanceSettings, fixture->instanceBounds, fixture->threadPool);
    PT_EXPECT(ctx, passed, "G-buffer disagreed with the per-pixel Embree oracle; see the rows above");
}

// Axis-aligned, near clip 0.1: the ordinary path, no triangle clipped, every field compared against a ray through the same pixel centre.
PT_CHECK(gbuffer_pose_straight_on, Fast, Statistical) {
    runPose(ctx, "straightOn", straightOnCamera());
}

// Yawed and pitched off-axis, separating correct interpolation from one that works only when screen gradients are axis-aligned.
PT_CHECK(gbuffer_pose_angled, Fast, Statistical) {
    runPose(ctx, "angled",
            Camera(glm::vec3(3.0F, 2.0F, 1.0F), 20.0F, -10.0F, kFilmBack, 35.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F,
                    100.0F));
}

// Near clip 5.0 puts it mid-cluster, so some centres straddle or sit behind: exercises Sutherland-Hodgman instead of skipping it.
PT_CHECK(gbuffer_pose_near_clip, Fast, Statistical) {
    runPose(ctx, "clipTest",
            Camera(glm::vec3(0.0F, 0.0F, 0.0F), 5.0F, 5.0F, kFilmBack, 35.0F, 5.0F, 100.0F, 2.8F, 1.0F / 125.0F,
                    100.0F));
}

PT_CHECK(wireframe_sanity, Fast, Exact) {
    const std::unique_ptr<RasterFixture> fixture = makeFixture(ctx);
    ctx.plan(1);
    const bool passed =
        checkWireframeSanity(straightOnCamera(), fixture->shadingTriangles, fixture->instances,
                              fixture->perInstanceSettings, fixture->instanceBounds, fixture->threadPool);
    PT_EXPECT(ctx, passed, "wireframe was empty or covered more than half the hit pixels");
}

PT_CHECK(bounding_box_occlusion, Fast, Exact) {
    ThreadPool threadPool;
    ctx.plan(1);
    PT_EXPECT(ctx, checkBoundingBoxOcclusion(threadPool), "box edges were not depth-tested against the occluder");
}

PT_CHECK(per_instance_boxes, Fast, Exact) {
    ThreadPool threadPool;
    ctx.plan(1);
    PT_EXPECT(ctx, checkPerInstanceBoxes(threadPool), "per-instance boxes were not drawn in disjoint hues/spans");
}

// Watertightness, exact: from inside a closed mesh every pixel must be covered, with no oracle and no tolerance.
PT_CHECK(watertight_closed_mesh, Fast, Exact) {
    constexpr int kCells = 16;
    constexpr float kHalfExtent = 5.0F;
    constexpr int kSeededPoses = 8;
    // Any jitter that keeps the origin inside leaves the surface closed around the camera; a quarter cell keeps triangles well-shaped.
    constexpr float kJitter = 0.25F * (2.0F * kHalfExtent / static_cast<float>(kCells));
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    ThreadPool threadPool;
    const std::vector<MeshInstance> instances{
        MeshInstance{makeMaterial(glm::vec3(0.5F), 0.5F), glm::mat4(1.0F), ""}};
    const std::vector<PathTraceSettings> perInstanceSettings(instances.size(), makeTestSettings());
    ctx.plan(1 + kSeededPoses);

    const auto expectWatertight = [&](const std::string& pose, const Camera& camera,
                                      const std::vector<ShadingTriangle>& mesh) {
        RasterGBuffer raster;
        renderRasterGBuffer(camera, camera, mesh, instances, perInstanceSettings, computeInstanceBounds(mesh, 1), kWidth,
                             kHeight, threadPool, raster);
        const int uncovered = uncoveredPixels(raster);
        std::cout << "rasterizer_validate: watertight " << pose << " -- " << uncovered << " uncovered pixels\n";
        PT_EXPECT(ctx, uncovered == 0, pose + ": pixels uncovered from inside a closed mesh (a crack)");
    };

    expectWatertight("straightOn", straightOnCamera(), makeClosedCube(kCells, kHalfExtent, 0.0F, rng));
    const std::vector<ShadingTriangle> jittered = makeClosedCube(kCells, kHalfExtent, kJitter, rng);
    std::uniform_real_distribution<float> yaw(0.0F, 360.0F);
    std::uniform_real_distribution<float> pitch(-85.0F, 85.0F);
    for (int i = 0; i < kSeededPoses; ++i) {
        const float poseYaw = yaw(rng);
        const float posePitch = pitch(rng);
        expectWatertight("yaw " + std::to_string(poseYaw) + " pitch " + std::to_string(posePitch),
                         Camera(glm::vec3(0.0F), poseYaw, posePitch, kFilmBack, 35.0F, 0.1F, 100.0F, 2.8F,
                                1.0F / 125.0F, 100.0F),
                         jittered);
    }
}

// Clip-plane and fan edges are not mesh edges: a triangle larger than the view covers every pixel yet shows no wireframe.
PT_CHECK(wireframe_ignores_clip_edges, Fast, Exact) {
    ThreadPool threadPool;
    const std::vector<MeshInstance> instances{
        MeshInstance{makeMaterial(glm::vec3(0.5F), 0.5F), glm::mat4(1.0F), ""}};
    const std::vector<PathTraceSettings> perInstanceSettings(instances.size(), makeTestSettings());
    // Plane z = -10 - 0.3y: every frustum ray meets it in front of the camera, while its y = -1000 vertices sit at z = +290, behind it.
    const glm::vec3 normal = glm::normalize(glm::vec3(0.0F, 0.3F, 1.0F));
    const glm::vec4 tangent = tangentFor(normal);
    const std::vector<ShadingTriangle> scene{ShadingTriangle{
        ShadingVertex{glm::vec3(-1000.0F, -1000.0F, 290.0F), normal, glm::vec2(0.0F, 0.0F), tangent},
        ShadingVertex{glm::vec3(1000.0F, -1000.0F, 290.0F), normal, glm::vec2(1.0F, 0.0F), tangent},
        ShadingVertex{glm::vec3(0.0F, 1000.0F, -310.0F), normal, glm::vec2(0.0F, 1.0F), tangent}, 0}};
    RasterGBuffer raster;
    const Camera camera = straightOnCamera();
    renderRasterGBuffer(camera, camera, scene, instances, perInstanceSettings, computeInstanceBounds(scene, 1),
                         kWidth, kHeight, threadPool, raster);
    int wirePixels = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            wirePixels += isWireframeColor(texelAt(raster.wireframe, x, y)) ? 1 : 0;
        }
    }
    const int uncovered = uncoveredPixels(raster);
    std::cout << "rasterizer_validate: clip edges -- " << uncovered << " uncovered, " << wirePixels
              << " wireframe pixels\n";
    ctx.plan(2);
    PT_EXPECT(ctx, uncovered == 0, "the screen-covering triangle left pixels uncovered");
    PT_EXPECT(ctx, wirePixels == 0, "clip-plane or fan edges were drawn as mesh wireframe");
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
        const ShadingVertex vertex{glm::vec3(0.0F), normal, glm::vec2(wrappedUv(rng), wrappedUv(rng)), tangent,
                                   randomVec3()};

        const BsdfParams constantParams = resolveBsdfParams(constant, vertex.uv, vertex.colour, settings, std::nullopt);
        const BsdfParams texturedParams = resolveBsdfParams(textured, vertex.uv, vertex.colour, settings, std::nullopt);
        paramMismatches += sameBsdfParams(constantParams, texturedParams) ? 0 : 1;
        const ShadingFrame constantFrame = buildShadingFrame(vertex, constant, settings);
        frameMismatches += constantFrame == buildShadingFrame(vertex, textured, settings) ? 0 : 1;
        // Four taps of a constant height differ by exactly 0, so the texture path only renormalises the unbumped normal.
        bumpMismatches += buildShadingFrame(vertex, bumped, settings)[2] == glm::normalize(constantFrame[2]) ? 0 : 1;
    }
    std::cout << "rasterizer_validate: constant vs 1x1 texture over " << kCases << " cases -- " << paramMismatches
              << " BsdfParams, " << frameMismatches << " frame, " << bumpMismatches << " bump mismatches\n";
    PT_EXPECT(ctx, paramMismatches == 0, "a constant input resolved BsdfParams differently from its 1x1 texture");
    PT_EXPECT(ctx, frameMismatches == 0, "a constant normal built a different shading frame from its 1x1 texture");
    PT_EXPECT(ctx, bumpMismatches == 0, "a constant bump texture tilted the normal: its gradient is not exactly zero");
}

}  // namespace

// Double-precision pinhole K [R | -R c] (Hartley & Zisserman 2004, eq. 6.8) on homogeneous (p, w): shares no arithmetic with project.
std::optional<glm::dvec2> pinholeNdc(const Camera& camera, float aspect, const glm::dvec4& point) {
    const Camera::ViewBasis basis = camera.viewBasis(aspect);
    const glm::dmat3 worldToView =
        glm::transpose(glm::dmat3(glm::dvec3(basis.right), glm::dvec3(basis.up), glm::dvec3(basis.forward)));
    const glm::dvec3 view = worldToView * (glm::dvec3(point) - (point.w * glm::dvec3(camera.position())));
    if (!(view.z > 0.0)) {
        return std::nullopt;
    }
    return glm::dvec2(view.x / (view.z * static_cast<double>(basis.halfWidth)), view.y / (view.z * static_cast<double>(basis.halfHeight)));
}

// Double-precision equidistant fisheye, r = f * theta (Kannala & Brandt 2006 with k = 0): the closed form, independent of project.
std::optional<glm::dvec2> equidistantNdc(const Camera& camera, float aspect, const glm::dvec4& point) {
    const Camera::ViewBasis basis = camera.viewBasis(aspect);
    const glm::dvec3 view = glm::dvec3(point) - (point.w * glm::dvec3(camera.position()));
    const glm::dvec2 lateral(glm::dot(view, glm::dvec3(basis.right)), glm::dot(view, glm::dvec3(basis.up)));
    const double theta = std::atan2(glm::length(lateral), glm::dot(view, glm::dvec3(basis.forward)));
    if (theta > static_cast<double>(basis.maxThetaRadians)) {
        return std::nullopt;
    }
    const double radiusMm = static_cast<double>(basis.focalLengthMm) * theta;
    return radiusMm * glm::normalize(lateral) / glm::dvec2(basis.halfWidthMm, basis.halfHeightMm);
}

// The motion the oracle predicts at (x, y): the rasterizer's own hit point, else the pixel-centre direction at infinity, in double.
glm::dvec2 oracleMotion(const RasterGBuffer& raster, const Camera& camera, const Camera& previous, int x, int y) {
    const float aspect = static_cast<float>(kWidth) / static_cast<float>(kHeight);
    const Camera::ViewBasis basis = camera.viewBasis(aspect);
    const glm::dvec2 ndc((((x + 0.5) / kWidth) * 2.0) - 1.0, 1.0 - (((y + 0.5) / kHeight) * 2.0));
    const glm::dvec3 direction = glm::dvec3(basis.forward) + (ndc.x * static_cast<double>(basis.halfWidth) * glm::dvec3(basis.right)) +
                                 (ndc.y * static_cast<double>(basis.halfHeight) * glm::dvec3(basis.up));
    const bool hit = texelAt(raster.alpha, x, y).x > 0.5F;
    const glm::dvec4 point = hit ? glm::dvec4(glm::dvec3(texelAt(raster.worldPos, x, y)), 1.0) : glm::dvec4(direction, 0.0);
    const std::optional<glm::dvec2> now = pinholeNdc(camera, aspect, point);
    const bool fisheye = previous.lens().projection == LensProjection::FisheyePolynomial;
    const std::optional<glm::dvec2> before = fisheye ? equidistantNdc(previous, aspect, point) : pinholeNdc(previous, aspect, point);
    if (!now || !before) {
        return glm::dvec2(0.0);
    }
    return (*now - *before) * glm::dvec2(0.5 * kWidth, -0.5 * kHeight);
}

glm::vec2 motionAt(const RasterGBuffer& raster, int x, int y) {
    return glm::vec2(texelAt(raster.motionVector, x, y));
}

// The angled pose: off every axis and inside the cluster's depth span, so a frame holds both hits and misses.
Camera angledCamera(glm::vec3 offset = glm::vec3(0.0F), float yawOffset = 0.0F, float pitchOffset = 0.0F, float focalMm = 35.0F) {
    return Camera(glm::vec3(3.0F, 2.0F, 1.0F) + offset, 20.0F + yawOffset, -10.0F + pitchOffset, kFilmBack, focalMm, 0.1F,
                  100.0F, 2.8F, 1.0F / 125.0F, 100.0F);
}

// Same camera both sides: one projection of one point twice, so every pixel, hit or miss, reads zero bitwise rather than nearly.
PT_CHECK(motion_vector_is_exactly_zero_for_an_unmoved_camera, Fast, Exact) {
    const std::unique_ptr<RasterFixture> fixture = makeFixture(ctx);
    ctx.plan(3);
    const Camera camera = angledCamera();
    RasterGBuffer raster;
    renderRasterGBuffer(camera, camera, fixture->shadingTriangles, fixture->instances, fixture->perInstanceSettings,
                         fixture->instanceBounds, kWidth, kHeight, fixture->threadPool, raster);
    int nonZero = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            nonZero += motionAt(raster, x, y) == glm::vec2(0.0F) ? 0 : 1;
        }
    }
    const int misses = uncoveredPixels(raster);
    PT_EXPECT(ctx, misses > 0, "the pose shows no environment, so the w = 0 path went unexercised");
    PT_EXPECT(ctx, misses < kWidth * kHeight, "the pose shows no geometry, so the w = 1 path went unexercised");
    PT_EXPECT(ctx, nonZero == 0, std::to_string(nonZero) + " pixels moved although the camera did not");
}

// Rotation, translation, both, zoom and a lens toggle against the double oracle at every pixel; misses ignore translation.
PT_CHECK(motion_vector_matches_a_pinhole_oracle, Fast, Exact) {
    const std::unique_ptr<RasterFixture> fixture = makeFixture(ctx);
    const Camera camera = angledCamera();
    const std::array<std::pair<const char*, Camera>, 5> previousViews{{
        {"rotation", angledCamera(glm::vec3(0.0F), 3.0F, -2.0F)},
        {"translation", angledCamera(glm::vec3(0.3F, -0.2F, 0.5F))},
        {"rotation+translation", angledCamera(glm::vec3(-0.4F, 0.1F, 0.25F), -4.0F, 1.5F)},
        {"zoom", angledCamera(glm::vec3(0.0F), 0.0F, 0.0F, 50.0F)},
        {"lens toggle from an equidistant fisheye", Camera(glm::vec3(3.0F, 2.0F, 1.0F), 20.0F, -10.0F, kFilmBack, 8.0F, 0.1F, 100.0F,
                                                           2.8F, 1.0F / 125.0F, 100.0F,
                                                           Lens{LensProjection::FisheyePolynomial, {}, 180.0F})},
    }};
    ctx.plan(static_cast<int>(previousViews.size()));
    for (const auto& [name, previous] : previousViews) {
        RasterGBuffer raster;
        renderRasterGBuffer(camera, previous, fixture->shadingTriangles, fixture->instances, fixture->perInstanceSettings,
                             fixture->instanceBounds, kWidth, kHeight, fixture->threadPool, raster);
        double worst = 0.0;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const glm::dvec2 error = glm::dvec2(motionAt(raster, x, y)) - oracleMotion(raster, camera, previous, x, y);
                worst = std::max(worst, std::max(std::abs(error.x), std::abs(error.y)));
            }
        }
        std::cout << "rasterizer_validate: motion " << name << " -- worst " << worst << " px\n";
        PT_EXPECT(ctx, worst <= kMotionEpsilonPx, std::string(name) + ": worst motion error " + std::to_string(worst) + " px");
    }
}

// Translation leaves the plane at infinity fixed (w = 0 drops the camera centre), so every miss reads zero bitwise while hits move.
PT_CHECK(motion_vector_holds_the_sky_still_under_translation, Fast, Exact) {
    const std::unique_ptr<RasterFixture> fixture = makeFixture(ctx);
    ctx.plan(2);
    const Camera camera = angledCamera();
    const Camera previous = angledCamera(glm::vec3(0.3F, -0.2F, 0.5F));
    RasterGBuffer raster;
    renderRasterGBuffer(camera, previous, fixture->shadingTriangles, fixture->instances, fixture->perInstanceSettings,
                         fixture->instanceBounds, kWidth, kHeight, fixture->threadPool, raster);
    int movedMisses = 0;
    int movedHits = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const bool moved = motionAt(raster, x, y) != glm::vec2(0.0F);
            (texelAt(raster.alpha, x, y).x > 0.5F ? movedHits : movedMisses) += moved ? 1 : 0;
        }
    }
    PT_EXPECT(ctx, movedMisses == 0, std::to_string(movedMisses) + " environment pixels moved under a pure translation");
    PT_EXPECT(ctx, movedHits > 0, "no surface moved under a translation");
}

// Turned half a revolution, the previous view faces away from everything this one sees: no image point exists, so motion reads zero.
PT_CHECK(motion_vector_is_zero_where_the_previous_view_has_no_image, Fast, Exact) {
    const std::unique_ptr<RasterFixture> fixture = makeFixture(ctx);
    ctx.plan(1);
    const Camera camera = straightOnCamera();
    const Camera previous(glm::vec3(0.0F), 180.0F, 0.0F, kFilmBack, 35.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F, 100.0F);
    RasterGBuffer raster;
    renderRasterGBuffer(camera, previous, fixture->shadingTriangles, fixture->instances, fixture->perInstanceSettings,
                         fixture->instanceBounds, kWidth, kHeight, fixture->threadPool, raster);
    int nonZero = 0;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            nonZero += motionAt(raster, x, y) == glm::vec2(0.0F) ? 0 : 1;
        }
    }
    PT_EXPECT(ctx, nonZero == 0, std::to_string(nonZero) + " pixels were given motion from a view that cannot see them");
}

PT_CHECK_MAIN("rasterizer")
