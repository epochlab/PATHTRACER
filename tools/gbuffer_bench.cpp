// Timing harness for renderGBuffer, not a correctness gate.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/debug/bench_log.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/gbuffer.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/shading_scene.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using namespace pathtracer::scene;  // NOLINT(google-build-using-namespace) -- tool-local convenience, mirrors gbuffer_validate.cpp

constexpr int kMaterialCount = 4;
constexpr float kNearestLayerZ = 4.0F;   // world units in front of the camera; > nearClip so no layer is clipped away
constexpr float kLayerSpacing = 1.0F;
constexpr float kLayerCoverage = 1.05F;  // centers spread slightly past the frustum cross-section so triangles reach the screen edges
constexpr float kVertexAngleStep = 2.0943951F;  // 2*pi/3, the three vertices of an equilateral triangle

struct Options {
    int triangleCount = 20561;  // scene.json's rkswd_tier_2.gltf, the shipped default
    int width = pathtracer::debug::kBenchWidth;  // --width/--height override
    int height = pathtracer::debug::kBenchHeight;
    int frames = 5;
    int layers = 1;
    unsigned int seed = 42;
    std::string benchLogPath;  // appends the run to this JSON Lines benchmark log (bench_log.h); empty = no log
};

// The neutral default material, varying only the two slots the G-buffer AOVs under test read.
Material makeMaterial(glm::vec3 baseColor, float roughness) {
    return Material{.baseColor = baseColor, .roughness = roughness};
}

glm::vec4 tangentFor(const glm::vec3& normal) {
    const glm::vec3 up = std::fabs(normal.y) < 0.99F ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::vec3(1.0F, 0.0F, 0.0F);
    return glm::vec4(glm::normalize(glm::cross(up, normal)), 1.0F);
}

// `layers` screen-filling shells sharing the triangle budget, emitted furthest first, so a pixel accumulates `layers` depth records.
std::vector<ShadingTriangle> makeLayeredTriangles(const Options& options, const Camera& camera) {
    const float aspect = static_cast<float>(options.width) / static_cast<float>(options.height);
    const Camera::ViewBasis basis = camera.viewBasis(aspect);
    const int perLayer = options.triangleCount / options.layers;  // >= 1: parseOptions rejects layers > triangleCount

    std::mt19937 rng(options.seed);
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);

    std::vector<ShadingTriangle> triangles;
    triangles.reserve(static_cast<std::size_t>(options.triangleCount));
    for (int layer = options.layers - 1; layer >= 0; --layer) {
        const float depth = kNearestLayerZ + (static_cast<float>(layer) * kLayerSpacing);
        const float halfWidth = depth * basis.halfWidth * kLayerCoverage;
        const float halfHeight = depth * basis.halfHeight * kLayerCoverage;
        // Circumradius for perLayer triangles at ~1.3x the cross-section, from the equilateral area (3*sqrt(3)/4)r^2.
        const float radius = std::sqrt((4.0F * halfWidth * halfHeight) / static_cast<float>(perLayer));
        // Furthest shell (layer layers-1, emitted first) absorbs the integer-division remainder so the total matches --triangles exactly.
        const int count = layer == options.layers - 1
                              ? options.triangleCount - (perLayer * (options.layers - 1))
                              : perLayer;

        for (int i = 0; i < count; ++i) {
            const glm::vec3 center(unit(rng) * halfWidth, unit(rng) * halfHeight, -depth);
            std::array<glm::vec3, 3> p{};
            for (int k = 0; k < 3; ++k) {
                const float angle = kVertexAngleStep * static_cast<float>(k);
                // Per-vertex depth jitter tilts triangles off screen-parallel, so the BVH and barycentric interpolation do real work.
                p[static_cast<std::size_t>(k)] =
                    center + glm::vec3(radius * std::cos(angle), radius * std::sin(angle), unit(rng) * radius * 0.25F);
            }
            const glm::vec3 normal = glm::normalize(glm::cross(p[1] - p[0], p[2] - p[0]));
            if (!std::isfinite(normal.x)) {
                continue;  // degenerate draw -- skipped, leaving the total marginally under --triangles
            }
            const glm::vec4 tangent = tangentFor(normal);
            ShadingTriangle tri;
            tri.v0 = ShadingVertex{p[0], normal, glm::vec2(0.0F, 0.0F), tangent};
            tri.v1 = ShadingVertex{p[1], normal, glm::vec2(1.0F, 0.0F), tangent};
            tri.v2 = ShadingVertex{p[2], normal, glm::vec2(0.0F, 1.0F), tangent};
            tri.instanceIndex = i % kMaterialCount;
            triangles.push_back(tri);
        }
    }
    return triangles;
}

// nullopt on an unrecognized flag, missing value, or out-of-range value: argv is a boundary, so a bad value surfaces rather than clamps.
std::optional<Options> parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) {
            std::cerr << "gbuffer_bench: " << argv[i] << " expects a value\n";
            return std::nullopt;
        }
        const char* flag = argv[i];
        const char* text = argv[++i];
        if (std::strcmp(flag, "--bench-log") == 0) {
            options.benchLogPath = text;
            continue;
        }
        char* end = nullptr;
        const long value = std::strtol(text, &end, 10);
        // strtol reports non-numeric input as 0, so the terminator check is what makes "--seed foo" an error not a silent seed of 0.
        if (end == text || *end != '\0') {
            std::cerr << "gbuffer_bench: " << flag << " expects an integer, got " << text << '\n';
            return std::nullopt;
        }
        // Bounded before the narrowing casts below, so an out-of-range argument is an error rather than an implementation-defined wrap.
        if (value < 0 || value > std::numeric_limits<int>::max()) {
            std::cerr << "gbuffer_bench: " << flag << " out of range: " << text << '\n';
            return std::nullopt;
        }
        if (std::strcmp(flag, "--triangles") == 0) {
            options.triangleCount = static_cast<int>(value);
        } else if (std::strcmp(flag, "--width") == 0) {
            options.width = static_cast<int>(value);
        } else if (std::strcmp(flag, "--height") == 0) {
            options.height = static_cast<int>(value);
        } else if (std::strcmp(flag, "--frames") == 0) {
            options.frames = static_cast<int>(value);
        } else if (std::strcmp(flag, "--layers") == 0) {
            options.layers = static_cast<int>(value);
        } else if (std::strcmp(flag, "--seed") == 0) {
            options.seed = static_cast<unsigned int>(value);
        } else {
            std::cerr << "gbuffer_bench: unknown flag " << flag
                       << "\n  usage: gbuffer_bench [--triangles N] [--width N] [--height N] [--frames N] [--layers N] [--seed N] [--bench-log log.jsonl]\n";
            return std::nullopt;
        }
    }
    if (options.triangleCount < 1 || options.width < 1 || options.height < 1 || options.frames < 1 ||
        options.layers < 1) {
        std::cerr << "gbuffer_bench: --triangles/--width/--height/--frames/--layers must all be >= 1\n";
        return std::nullopt;
    }
    if (options.layers > options.triangleCount) {
        std::cerr << "gbuffer_bench: --layers cannot exceed --triangles\n";
        return std::nullopt;
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    const std::optional<Options> options = parseOptions(argc, argv);
    if (!options) {
        return EXIT_FAILURE;
    }

    const Camera::FilmBack filmBack{36.0F, 24.0F};
    const Camera camera(glm::vec3(0.0F), 0.0F, 0.0F, filmBack, 35.0F, 0.1F, 100.0F, 2.8F, 1.0F / 125.0F,
                         100.0F);

    const std::vector<ShadingTriangle> shadingTriangles = makeLayeredTriangles(*options, camera);
    std::vector<MeshInstance> instances;
    instances.push_back(MeshInstance{makeMaterial(glm::vec3(0.8F, 0.2F, 0.2F), 0.2F), glm::mat4(1.0F), ""});
    instances.push_back(MeshInstance{makeMaterial(glm::vec3(0.2F, 0.8F, 0.2F), 0.5F), glm::mat4(1.0F), ""});
    instances.push_back(MeshInstance{makeMaterial(glm::vec3(0.2F, 0.2F, 0.8F), 0.8F), glm::mat4(1.0F), ""});
    instances.push_back(MeshInstance{makeMaterial(glm::vec3(0.8F, 0.8F, 0.2F), 1.0F), glm::mat4(1.0F), ""});

    PathTraceSettings settings{};
    settings.samplesPerPixel = 1;
    settings.maxBounces = 0;
    settings.russianRouletteStartBounce = 1;
    settings.bumpStrength = 1.0F;
    settings.roughnessMin = 0.045F;
    settings.roughnessMax = 1.0F;
    settings.diffuseColour = glm::vec3(1.0F);
    settings.ior = 1.5F;
    settings.transmissionFactor = 0.0F;
    settings.metallicFactor = 0.2F;
    settings.roughnessFactor = 1.0F;
    const std::vector<PathTraceSettings> perInstanceSettings(instances.size(), settings);
    // Once outside the timed loop, as the app does at load: the measured cost is testing the boxes' edges, not deriving them.
    const std::vector<AabbBounds> instanceBounds =
        computeInstanceBounds(shadingTriangles, static_cast<int>(instances.size()));

    // Built once at load as the app does, so the timed frames measure tracing and shading, not the BVH build.
    std::vector<Triangle> worldTriangles;
    worldTriangles.reserve(shadingTriangles.size());
    for (const ShadingTriangle& tri : shadingTriangles) {
        worldTriangles.push_back(Triangle{tri.v0.position, tri.v1.position, tri.v2.position});
    }
    const std::optional<EmbreeAccel> accel = EmbreeAccel::build(std::move(worldTriangles));
    if (!accel) {
        return EXIT_FAILURE;
    }

    ThreadPool threadPool;
    // One buffer for the whole run as the app owns it: renderGBuffer reuses it in place, so timed frames measure steady state.
    GBuffer gbuffer;
    // Discarded warm-up pass absorbing the once-only costs: spinning up and parking the pool's workers, and the buffer's only allocation.
    renderGBuffer(camera, camera, *accel, shadingTriangles, instances, perInstanceSettings, instanceBounds, options->width,
                  options->height, threadPool, gbuffer);
    if (gbuffer.depth.width != options->width) {
        std::cerr << "gbuffer_bench: warm-up produced a " << gbuffer.depth.width << "px-wide buffer\n";
        return EXIT_FAILURE;
    }

    std::vector<double> milliseconds;
    milliseconds.reserve(static_cast<std::size_t>(options->frames));
    for (int frame = 0; frame < options->frames; ++frame) {
        const auto start = std::chrono::steady_clock::now();
        renderGBuffer(camera, camera, *accel, shadingTriangles, instances, perInstanceSettings, instanceBounds,
                      options->width, options->height, threadPool, gbuffer);
        const auto end = std::chrono::steady_clock::now();
        milliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        // Reading one texel keeps the optimizer from treating the whole call as dead; the result is otherwise unused.
        if (!std::isfinite(gbuffer.depth.texels[0])) {
            std::cerr << "gbuffer_bench: non-finite depth at texel 0\n";
            return EXIT_FAILURE;
        }
    }

    const auto [best, worst] = std::minmax_element(milliseconds.begin(), milliseconds.end());
    double total = 0.0;
    for (const double ms : milliseconds) {
        total += ms;
    }

    std::cout << "gbuffer_bench: " << shadingTriangles.size() << " tris, " << options->width << "x"
              << options->height << ", " << options->layers << " layer(s), best-of-" << options->frames
              << ": " << *best << " ms  (mean " << total / static_cast<double>(options->frames)
              << ", worst " << *worst << ")\n";

    if (!options->benchLogPath.empty()) {
        const pathtracer::debug::BenchRecord record{
            .tool = "gbuffer_bench",
            .argv = std::vector<std::string>(argv, argv + argc),
            .config = {{"triangles", options->triangleCount},
                       {"width", options->width},
                       {"height", options->height},
                       {"frames", options->frames},
                       {"layers", options->layers},
                       {"seed", options->seed}},
            .samples = {{"frame_ms", milliseconds}},
            .work = {{"triangles_emitted", shadingTriangles.size()},
                     {"crc32", pathtracer::debug::floatCrc32(gbuffer.depth.texels)}},
        };
        if (!pathtracer::debug::appendBenchRecord(options->benchLogPath, record)) {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
