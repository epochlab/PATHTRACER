// Correctness check for path_tracer.cpp's integrator (renderPathTraced/tracePath).

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <variant>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "pathtracer/config/scene_config.h"
#include "pathtracer/gfx/hdr_image.h"
#include "check.h"
#include "fixtures.h"
#include "stats.h"
#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/gbuffer_shading.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/environment_map.h"
#include "pathtracer/scene/gbuffer.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/light.h"
#include "pathtracer/scene/material_binding.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/shading_scene.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using pathtracer::scene::Camera;
using pathtracer::scene::EmbreeAccel;
using pathtracer::scene::Constant;
using pathtracer::scene::EnvironmentMap;
using pathtracer::scene::forEachInput;
using pathtracer::scene::InputSpec;
using pathtracer::scene::Material;
using pathtracer::scene::OpenPbrInputs;
using pathtracer::scene::MeshInstance;
using pathtracer::scene::PathTraceSettings;
using pathtracer::scene::ShadingTriangle;
using pathtracer::scene::ShadingVertex;
using pathtracer::scene::TextureHandle;
using pathtracer::scene::Triangle;

using tools::fixtures::kPi;
using tools::fixtures::makeMaterial;
using tools::fixtures::makeUniformEnvironment;
using tools::fixtures::referenceLo;

// One pool for the whole binary, sized from the runner's --threads so concurrent ctest jobs do not oversubscribe the machine.
pathtracer::scene::ThreadPool& sharedPool(int threads) {
    static pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(threads));
    return pool;
}

// Each check keeps its own `ok` accumulator and per-row diagnostics, reporting one verdict: per-row assertions would lose that context.
void finish(tools::check::Context& ctx, bool ok, const char* what) {
    ctx.plan(1);
    PT_EXPECT(ctx, ok, what);
}

// Quad half-extent: large enough that every primary ray in the narrow test FOV lands on it, so no pixel sees the environment directly.
constexpr float kQuadExtent = 1000.0F;
// Sphere geometry shared by its two checks so tessellations cannot drift: checkBeerLambert's transmittance depends on this exact mesh.
constexpr float kSphereRadius = 1.0F;
// 64x32 sets transmissionOffsetEpsilon to 1.93e-2 on a unit sphere, backing each origin in and shortening every measured chord.
constexpr int kSphereSlices = 64;
constexpr int kSphereStacks = 32;
// Narrow FOV (200mm on a 36x24 gate, ~6.9 degrees vertical) so every pixel direction is within a fraction of a degree of the quad normal.
constexpr float kFocalLengthMm = 200.0F;
constexpr int kImageSize = 16;
constexpr int kSamplesPerPixel = 512;

struct TestScene {
    std::vector<Triangle> worldTriangles;
    std::vector<ShadingTriangle> shadingTriangles;
    std::vector<MeshInstance> instances;
};

// One quad in the z=0 plane facing +Z, wound counter-clockwise as seen from +Z so geometricNormalOf gives (0,0,1).
TestScene makeQuadScene(float roughness) {
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    const glm::vec4 tangent(1.0F, 0.0F, 0.0F, 1.0F);
    const auto vertex = [&](float x, float y) {
        return ShadingVertex{glm::vec3(x, y, 0.0F), normal, glm::vec2(0.5F, 0.5F), tangent};
    };
    const ShadingVertex v0 = vertex(-kQuadExtent, -kQuadExtent);
    const ShadingVertex v1 = vertex(kQuadExtent, -kQuadExtent);
    const ShadingVertex v2 = vertex(kQuadExtent, kQuadExtent);
    const ShadingVertex v3 = vertex(-kQuadExtent, kQuadExtent);

    TestScene scene;
    scene.worldTriangles = {Triangle{v0.position, v1.position, v2.position},
                             Triangle{v0.position, v2.position, v3.position}};
    scene.shadingTriangles = {ShadingTriangle{v0, v1, v2, 0}, ShadingTriangle{v0, v2, v3, 0}};
    scene.instances = {MeshInstance{makeMaterial(roughness), glm::mat4(1.0F), ""}};
    return scene;
}

// Two parallel quads with opposing geometric normals, front at +Z and back at -Z, so a camera ray enters the front and exits the back.
TestScene makeSlabScene(float roughness, float thickness) {
    const glm::vec4 tangent(1.0F, 0.0F, 0.0F, 1.0F);
    const auto vertex = [&](float x, float y, float z, float nz) {
        return ShadingVertex{glm::vec3(x, y, z), glm::vec3(0.0F, 0.0F, nz), glm::vec2(0.5F, 0.5F),
                              tangent};
    };
    // Front wound counter-clockwise as seen from +Z, back clockwise, so geometricNormalOf gives +Z and -Z.
    const ShadingVertex fv0 = vertex(-kQuadExtent, -kQuadExtent, 0.0F, 1.0F);
    const ShadingVertex fv1 = vertex(kQuadExtent, -kQuadExtent, 0.0F, 1.0F);
    const ShadingVertex fv2 = vertex(kQuadExtent, kQuadExtent, 0.0F, 1.0F);
    const ShadingVertex fv3 = vertex(-kQuadExtent, kQuadExtent, 0.0F, 1.0F);
    const ShadingVertex bv0 = vertex(-kQuadExtent, -kQuadExtent, -thickness, -1.0F);
    const ShadingVertex bv1 = vertex(-kQuadExtent, kQuadExtent, -thickness, -1.0F);
    const ShadingVertex bv2 = vertex(kQuadExtent, kQuadExtent, -thickness, -1.0F);
    const ShadingVertex bv3 = vertex(kQuadExtent, -kQuadExtent, -thickness, -1.0F);

    TestScene scene;
    scene.worldTriangles = {Triangle{fv0.position, fv1.position, fv2.position},
                             Triangle{fv0.position, fv2.position, fv3.position},
                             Triangle{bv0.position, bv1.position, bv2.position},
                             Triangle{bv0.position, bv2.position, bv3.position}};
    scene.shadingTriangles = {ShadingTriangle{fv0, fv1, fv2, 0}, ShadingTriangle{fv0, fv2, fv3, 0},
                               ShadingTriangle{bv0, bv1, bv2, 0}, ShadingTriangle{bv0, bv2, bv3, 0}};
    scene.instances = {MeshInstance{makeMaterial(roughness), glm::mat4(1.0F), ""}};
    return scene;
}

// The quad above plus an opaque wall at wallX facing -X, the only scene here where a non-transmissive path reaches a second surface.
TestScene makeCornerScene(float roughness, float wallX = 1.0F) {
    TestScene scene = makeQuadScene(roughness);
    const glm::vec3 wallNormal(-1.0F, 0.0F, 0.0F);
    const glm::vec4 wallTangent(0.0F, 1.0F, 0.0F, 1.0F);
    const auto vertex = [&](float y, float z) {
        return ShadingVertex{glm::vec3(wallX, y, z), wallNormal, glm::vec2(0.5F, 0.5F), wallTangent};
    };
    // Wound so geometricNormalOf gives -X, i.e. facing back across the floor rather than away from it.
    const ShadingVertex w0 = vertex(-kQuadExtent, 0.0F);
    const ShadingVertex w1 = vertex(kQuadExtent, 0.0F);
    const ShadingVertex w2 = vertex(kQuadExtent, kQuadExtent);
    const ShadingVertex w3 = vertex(-kQuadExtent, kQuadExtent);

    scene.worldTriangles.push_back(Triangle{w0.position, w3.position, w2.position});
    scene.worldTriangles.push_back(Triangle{w0.position, w2.position, w1.position});
    scene.shadingTriangles.push_back(ShadingTriangle{w0, w3, w2, 0});
    scene.shadingTriangles.push_back(ShadingTriangle{w0, w2, w1, 0});
    return scene;
}

// UV sphere, poles on Y so the camera reads the tessellated equator; curvature is the point, as flat quads zero transmissionOffsetEpsilon.
TestScene makeSphereScene(float roughness, int slices = kSphereSlices, int stacks = kSphereStacks) {
    const auto vertexAt = [&](int stack, int slice) {
        const float phi = kPi * static_cast<float>(stack) / static_cast<float>(stacks);
        const float theta = 2.0F * kPi * static_cast<float>(slice) / static_cast<float>(slices);
        const glm::vec3 normal(std::sin(phi) * std::sin(theta), std::cos(phi),
                                std::sin(phi) * std::cos(theta));
        return ShadingVertex{normal * kSphereRadius, normal, glm::vec2(0.5F, 0.5F),
                              glm::vec4(std::cos(theta), 0.0F, -std::sin(theta), 1.0F)};
    };

    TestScene scene;
    for (int stack = 0; stack < stacks; ++stack) {
        for (int slice = 0; slice < slices; ++slice) {
            const ShadingVertex v00 = vertexAt(stack, slice);
            const ShadingVertex v10 = vertexAt(stack + 1, slice);
            const ShadingVertex v11 = vertexAt(stack + 1, slice + 1);
            const ShadingVertex v01 = vertexAt(stack, slice + 1);
            // Wound so geometricNormalOf points outward; each pole row contributes one triangle, the collapsed half-quad being degenerate.
            if (stack + 1 < stacks) {
                scene.worldTriangles.push_back(Triangle{v00.position, v10.position, v11.position});
                scene.shadingTriangles.push_back(ShadingTriangle{v00, v10, v11, 0});
            }
            if (stack > 0) {
                scene.worldTriangles.push_back(Triangle{v00.position, v11.position, v01.position});
                scene.shadingTriangles.push_back(ShadingTriangle{v00, v11, v01, 0});
            }
        }
    }
    scene.instances = {MeshInstance{makeMaterial(roughness), glm::mat4(1.0F), ""}};
    return scene;
}

// Two coplanar quads meeting at x=0, one MeshInstance each with identical Materials, which a check then edits apart.
TestScene makeTwoInstanceScene(float roughness) {
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    const glm::vec4 tangent(1.0F, 0.0F, 0.0F, 1.0F);
    const auto vertex = [&](float x, float y) {
        return ShadingVertex{glm::vec3(x, y, 0.0F), normal, glm::vec2(0.5F, 0.5F), tangent};
    };
    // Wound counter-clockwise as seen from +Z so geometricNormalOf gives (0,0,1), matching makeQuadScene.
    const auto addQuad = [&](TestScene& scene, float xMin, float xMax, int instanceIndex) {
        const ShadingVertex v0 = vertex(xMin, -kQuadExtent);
        const ShadingVertex v1 = vertex(xMax, -kQuadExtent);
        const ShadingVertex v2 = vertex(xMax, kQuadExtent);
        const ShadingVertex v3 = vertex(xMin, kQuadExtent);
        scene.worldTriangles.push_back(Triangle{v0.position, v1.position, v2.position});
        scene.worldTriangles.push_back(Triangle{v0.position, v2.position, v3.position});
        scene.shadingTriangles.push_back(ShadingTriangle{v0, v1, v2, instanceIndex});
        scene.shadingTriangles.push_back(ShadingTriangle{v0, v2, v3, instanceIndex});
    };

    TestScene scene;
    addQuad(scene, -kQuadExtent, 0.0F, 0);
    addQuad(scene, 0.0F, kQuadExtent, 1);
    scene.instances = {MeshInstance{makeMaterial(roughness), glm::mat4(1.0F), ""},
                        MeshInstance{makeMaterial(roughness), glm::mat4(1.0F), ""}};
    return scene;
}

// The integrator's limits alone: a check's surface is its scene's per-instance materials, as a scene file's is.
PathTraceSettings makeSettings(int maxBounces, int rrStartBounce) {
    PathTraceSettings settings{};
    settings.samplesPerPixel = kSamplesPerPixel;
    settings.maxBounces = maxBounces;
    settings.russianRouletteStartBounce = rrStartBounce;
    return settings;
}

// Every instance's material edited alike, on a copy: the geometry, and the accel built from it, stay shared with the source.
template <typename Edit>
TestScene withSurface(TestScene scene, Edit edit) {
    for (MeshInstance& instance : scene.instances) {
        edit(instance.material);
    }
    return scene;
}

// The surfaces the checks share: a metal of the scene's roughness, and a clear dielectric that transmits everything it does not reflect.
const auto metallic = [](float metalness) { return [metalness](Material& material) { material.baseMetalness = metalness; }; };
const auto transmissive = [](Material& material) { material.transmissionWeight = 1.0F; };

// Index-matched, so fresnelDielectric is exactly 0 and a non-metallic, non-transmissive surface is EON alone: f = base_color/pi at r = 0.
const auto indexMatched = [](Material& material) { material.specularIor = 1.0F; };

Camera makeCamera() {
    return Camera(glm::vec3(0.0F, 0.0F, 5.0F), glm::vec3(0.0F), Camera::FilmBack{36.0F, 24.0F},
                   kFocalLengthMm, 0.01F, 1000.0F, 2.8F, 1.0F / 125.0F, 100.0F);
}

// Mean radiance over the half-open pixel block [x0,x1) x [y0,y1).
glm::vec3 regionMean(const pathtracer::gfx::HdrImage& image, int x0, int y0, int x1, int y1) {
    glm::vec3 sum(0.0F);
    int count = 0;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            sum += image.rgb((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) +
                             static_cast<std::size_t>(x));
            ++count;
        }
    }
    return sum / static_cast<float>(count);
}

// Mean radiance over the centre 4x4 block: averaging tightens the estimate without widening the view-direction spread enough to matter.
glm::vec3 centreMean(const pathtracer::gfx::HdrImage& image) {
    return regionMean(image, (kImageSize / 2) - 2, (kImageSize / 2) - 2, (kImageSize / 2) + 2,
                       (kImageSize / 2) + 2);
}

// Runs one renderPathTraced pass over the scene's own per-instance materials.
pathtracer::scene::PathTraceResult renderPass(const TestScene& scene, const EnvironmentMap& env, const PathTraceSettings& settings,
                                              EmbreeAccel& accel, pathtracer::scene::ThreadPool& pool, bool showSky,
                                              std::uint32_t scrambleSeed = 7U) {
    const std::atomic<std::uint64_t> generation{1};
    pathtracer::scene::PathTraceResult result =
        pathtracer::scene::makePathTraceResult(kImageSize, kImageSize);
    // No test scene here authors an emitter -- every instance is ordinary geometry.
    const std::vector<int> instanceLightIndex(scene.instances.size(), -1);
    const std::vector<pathtracer::scene::QuadLight> noQuads;
    const pathtracer::scene::LightSet lights(&env, /*envRotationDegrees=*/glm::vec3(0.0F), /*envExposure=*/1.0F, noQuads);
    pathtracer::debug::PassStats stats;  // required by renderPathTraced; this tool checks radiance, not throughput
    pathtracer::scene::renderPathTraced(makeCamera(), accel, scene.shadingTriangles, scene.instances,
                                     instanceLightIndex, lights, kImageSize, kImageSize, showSky,
                                     settings, scrambleSeed, /*sampleBase=*/0,
                                     /*sampleCount=*/settings.samplesPerPixel, generation,
                                     /*requestedGeneration=*/1U, pool, stats, result);
    return result;
}

// Centre-region mean radiance of one pass.
float renderCentre(const TestScene& scene, const EnvironmentMap& env, const PathTraceSettings& settings,
                    EmbreeAccel& accel, pathtracer::scene::ThreadPool& pool) {
    const glm::vec3 mean = centreMean(renderPass(scene, env, settings, accel, pool, true).beauty);
    return std::max({mean.x, mean.y, mean.z});
}

struct Case {
    const char* name;
    float roughness;
    float metallic;
};

PT_CHECK(depth_and_russian_roulette_invariance, Slow, Statistical) {
    // Roughness restricted to values where the uniform-hemisphere reference converges, the same limitation nee_validate documents.
    const std::array<Case, 4> cases{{
        {"diffuse (metallic 0, rough 1.0)", 1.0F, 0.0F},
        {"glossy dielectric (rough 0.35)", 0.35F, 0.0F},
        {"rough metal (rough 0.5)", 0.5F, 1.0F},
        {"rough metal (rough 0.25)", 0.25F, 1.0F},
    }};

    // Depth invariance is exact up to Monte Carlo noise, so it gets the tighter bound; absolute agreement is looser, its reference noisier.
    constexpr float kDepthInvarianceTolerance = 0.02F;
    constexpr float kReferenceTolerance = 0.06F;
    constexpr int kReferenceSamples = 400000;

    const EnvironmentMap env = makeUniformEnvironment();
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    std::mt19937 referenceRng(99);
    bool ok = true;

    std::cout << "integrator_validate: quad under uniform L0=1 environment, wo = surface normal\n";
    std::cout << "  case                              maxB=0    maxB=1    RR on    reference\n";

    for (const Case& testCase : cases) {
        const TestScene scene = withSurface(makeQuadScene(testCase.roughness), metallic(testCase.metallic));
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree scene for " << testCase.name << '\n';
            finish(ctx, false, "depth_and_russian_roulette_invariance failed; see the rows above");
            return;
        }

        // rrStartBounce far above maxBounces disables Russian roulette for the first two renders, so depth invariance sheds that variance.
        const float loDepth0 = renderCentre(scene, env, makeSettings(0, 999), *accel, pool);
        const float loDepth1 = renderCentre(scene, env, makeSettings(1, 999), *accel, pool);
        // RR from bounce 0, same scene: reweighting by 1/p must leave the expectation unchanged.
        const float loRoulette = renderCentre(scene, env, makeSettings(1, 0), *accel, pool);

        OpenPbrInputs<Constant> inputs;
        inputs.baseColor = glm::vec3(1.0F);
        inputs.baseMetalness = testCase.metallic;
        inputs.specularRoughness = testCase.roughness;
        const float reference = referenceLo(inputs, glm::vec3(0.0F, 0.0F, 1.0F), kReferenceSamples,
                                             referenceRng);

        std::cout << "  " << testCase.name;
        for (std::size_t pad = std::string(testCase.name).size(); pad < 34; ++pad) {
            std::cout << ' ';
        }
        std::cout << loDepth0 << "  " << loDepth1 << "  " << loRoulette << "  " << reference << '\n';

        const float scale = std::max(reference, 0.05F);
        if (std::fabs(loDepth0 - loDepth1) > kDepthInvarianceTolerance * scale) {
            std::cerr << "integrator_validate: FAILED depth invariance at " << testCase.name
                      << " -- maxBounces=0 gave " << loDepth0 << ", maxBounces=1 gave " << loDepth1
                      << ". With no indirect light these must match; a gap means the terminal "
                         "BSDF-sampled ray is not being traced, so NEE's MIS weight is never "
                         "complemented.\n";
            ok = false;
        }
        if (std::fabs(loRoulette - loDepth1) > kDepthInvarianceTolerance * scale) {
            std::cerr << "integrator_validate: FAILED Russian roulette invariance at " << testCase.name
                      << " -- RR on gave " << loRoulette << ", RR off gave " << loDepth1
                      << ". RR reweights by 1/p and must not change the expectation.\n";
            ok = false;
        }
        if (std::fabs(loDepth1 - reference) > kReferenceTolerance * scale) {
            std::cerr << "integrator_validate: FAILED reference agreement at " << testCase.name
                      << " -- rendered " << loDepth1 << ", analytic " << reference << '\n';
            ok = false;
        }
    }
    finish(ctx, ok, "depth_and_russian_roulette_invariance failed; see the rows above");
    return;
}

// A white non-absorbing dielectric slab is invisible under a uniform environment, the only case reaching a transmissive exiting vertex.
PT_CHECK(transmissive_slab_energy, Slow, Statistical) {
    // Enough depth for internally reflected paths to converge; truncation only ever darkens.
    constexpr int kSlabBounces = 12;
    constexpr float kThickness = 0.5F;
    constexpr int kWalkPaths = 1 << 16;
    // 0 is the smooth interface, the delta transmission path; the rest take the Walter lobe.
    const std::array<float, 5> roughnesses = {0.0F, 0.05F, 0.4F, 0.7F, 1.0F};

    ctx.plan(static_cast<int>(2 * roughnesses.size()));
    const EnvironmentMap env = makeUniformEnvironment();
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    PathTraceSettings settings = makeSettings(kSlabBounces, 999);
    settings.samplesPerPixel = kSamplesPerPixel / tools::stats::kReplicates;

    std::cout << "integrator_validate: white non-absorbing slab, uniform L0=1, Embree render vs BSDF-only walk\n";
    for (float roughness : roughnesses) {
        char label[64];
        std::snprintf(label, sizeof(label), "slab r=%g", static_cast<double>(roughness));
        const std::uint64_t rowSeed = ctx.subSeed(label);
        const TestScene scene = withSurface(makeSlabScene(roughness, kThickness), transmissive);
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        PT_EXPECT(ctx, accel.has_value(), "failed to build the Embree slab scene");
        if (!accel.has_value()) {
            continue;
        }
        // The inputs the integrator resolves for this material, through the same resolution.
        const OpenPbrInputs<Constant> inputs = resolveInputs(scene.instances[0].material, glm::vec2(0.5F), {}, glm::vec3(1.0F));
        tools::stats::Welford render;
        tools::stats::Welford walk;
        for (int r = 0; r < tools::stats::kReplicates; ++r) {
            const auto seed = static_cast<std::uint32_t>(rowSeed + r);
            const glm::vec3 mean =
                regionMean(renderPass(scene, env, settings, *accel, pool, true, seed).beauty, 0, 0, kImageSize, kImageSize);
            render.add(std::max({mean.x, mean.y, mean.z}));
            // Disjoint seeds keep the two estimators independent, as Welch's band assumes.
            walk.add(tools::fixtures::slabWalkLo(inputs, kWalkPaths, seed + tools::stats::kReplicates).mean);
        }
        const double difference = render.mean() - walk.mean();
        // The render's float32 film resolves its mean to one ulp: a smooth slab's exact-Fresnel vertices leave no variance above that.
        const double filmUlp = std::numeric_limits<float>::epsilon() * render.mean();
        const tools::stats::Band statistical = tools::stats::differenceBand(walk, render, ctx.alpha());
        const tools::stats::Band band{statistical.lo - filmUlp, statistical.hi + filmUlp};
        char detail[320];
        std::snprintf(detail, sizeof(detail),
                      "%s: render %.5f, walk %.5f, difference %+.3e vs +/-%.3e -- above means NEE and the BSDF-sampled "
                      "miss double-count the transmission lobe, below means a transmissive vertex loses energy",
                      label, render.mean(), walk.mean(), difference, band.halfWidth());
        std::cout << "  " << detail << '\n';
        PT_EXPECT(ctx, band.contains(difference), detail);
    }
}

// The curved counterpart on the same invariant: a sphere traps far more light, internal hits past the critical angle reflecting totally.
PT_CHECK(transmissive_sphere_energy, Slow, Statistical) {
    // Measured convergence point, not a guess: 32 and 96 bounces are bit-identical to this, and 12 is not.
    constexpr int kSphereBounces = 16;
    constexpr float kTolerance = 0.03F;
    // 0 is the smooth interface and takes the delta path; the rest take the Walter lobe with far-side NEE.
    const std::array<float, 4> roughnesses = {0.0F, 0.2F, 0.4F, 0.7F};

    const EnvironmentMap env = makeUniformEnvironment();
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    bool ok = true;

    std::cout << "integrator_validate: white non-absorbing glass sphere, uniform L0=1 (1.0 = invisible)\n";
    for (float roughness : roughnesses) {
        const TestScene scene = withSurface(makeSphereScene(roughness), transmissive);
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree sphere scene\n";
            finish(ctx, false, "transmissive_sphere_energy failed; see the rows above");
            return;
        }
        const float lo = renderCentre(scene, env, makeSettings(kSphereBounces, 999), *accel, pool);
        std::cout << "  roughness " << roughness << "   Lo " << lo << '\n';
        if (std::fabs(lo - 1.0F) > kTolerance) {
            std::cerr << "integrator_validate: FAILED glass sphere transparency at roughness=" << roughness
                      << " -- rendered " << lo
                      << ", expected 1.0. Same invariant as the flat slab above, so a flat slab that "
                         "passes while this fails localises the fault to a curvature-driven term: the "
                         "transmission offset epsilon, or the shadow-terminator projection's side.\n";
            ok = false;
        }
    }

    // The same invariant with a dispersive index, gating the one-sample channel estimator.
    struct DispersiveGlass {
        const char* name;
        float ior;
        float abbe;
    };
    const std::array<DispersiveGlass, 2> dispersive{{
        {"Schott N-BK7", 1.5168F, 64.17F},
        {"Schott SF10", 1.72825F, 28.53F},
    }};
    std::cout << "  dispersive (per channel, the one-sample hero-channel estimator)\n";
    for (const DispersiveGlass& glass : dispersive) {
        const TestScene scene = withSurface(makeSphereScene(0.0F), [&](Material& material) {
            transmissive(material);
            material.specularIor = glass.ior;
            material.transmissionDispersionAbbeNumber = glass.abbe;
            material.transmissionDispersionScale = 1.0F;
        });
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree sphere scene\n";
            finish(ctx, false, "transmissive_sphere_energy failed; see the rows above");
            return;
        }
        const glm::vec3 lo = centreMean(renderPass(scene, env, makeSettings(kSphereBounces, 999), *accel, pool, true).beauty);
        std::cout << "    " << glass.name << " (ior " << glass.ior << ", abbe " << glass.abbe << ")   Lo ["
                  << lo.x << ", " << lo.y << ", " << lo.z << "]\n";
        for (int c = 0; c < 3; ++c) {
            if (std::fabs(lo[c] - 1.0F) > kTolerance) {
                std::cerr << "integrator_validate: FAILED dispersive glass sphere transparency for "
                          << glass.name << " channel " << c << " -- rendered " << lo[c]
                          << ", expected 1.0. Energy conservation does not depend on wavelength, so this "
                             "is the channel estimator losing or gaining energy, not the optics.\n";
                ok = false;
            }
        }
    }
    finish(ctx, ok, "transmissive_sphere_energy failed; see the rows above");
    return;
}

// Beer-Lambert: transmissionDepth is the distance at which transmittance reaches transmissionColor, so that identity is the specification.
PT_CHECK(beer_lambert_absorption, Slow, Statistical) {
    constexpr int kBounces = 8;
    constexpr float kSlabThickness = 0.5F;
    // Relative, the squared row's green being 0.0625; the sphere row reads high from the tessellated chord and the offset epsilon.
    constexpr float kRelativeTolerance = 0.03F;
    const glm::vec3 colour(0.5F, 0.25F, 0.75F);

    struct AbsorptionCase {
        const char* name;
        bool sphere;
        float depth;
        glm::vec3 expected;
        float abbe;
    };
    // The dispersive row crosses the hero channel with a per-channel sigmaA, asserting the two do not interact.
    const std::array<AbsorptionCase, 4> cases{{
        {"flat slab, depth == thickness", false, kSlabThickness, colour, 0.0F},
        {"flat slab, depth == half thickness", false, kSlabThickness * 0.5F, colour * colour, 0.0F},
        {"sphere, depth == chord", true, 2.0F * kSphereRadius, colour, 0.0F},
        {"flat slab, dispersive (abbe 64.17)", false, kSlabThickness, colour, 64.17F},
    }};

    const EnvironmentMap env = makeUniformEnvironment();
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    bool ok = true;

    std::cout << "integrator_validate: Beer-Lambert absorption through an index-matched medium\n";
    for (const AbsorptionCase& testCase : cases) {
        const TestScene scene = withSurface(testCase.sphere ? makeSphereScene(0.0F) : makeSlabScene(0.0F, kSlabThickness),
                                            [&](Material& material) {
                                                transmissive(material);
                                                indexMatched(material);
                                                material.transmissionColor = colour;
                                                material.transmissionDepth = testCase.depth;
                                                material.transmissionDispersionAbbeNumber = testCase.abbe > 0.0F ? testCase.abbe : 20.0F;
                                                material.transmissionDispersionScale = testCase.abbe > 0.0F ? 1.0F : 0.0F;
                                            });
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree scene for " << testCase.name << '\n';
            finish(ctx, false, "beer_lambert_absorption failed; see the rows above");
            return;
        }
        const glm::vec3 lo = centreMean(renderPass(scene, env, makeSettings(kBounces, 999), *accel, pool, true).beauty);

        std::cout << "  " << testCase.name;
        for (std::size_t pad = std::string(testCase.name).size(); pad < 36; ++pad) {
            std::cout << ' ';
        }
        std::cout << "measured [" << lo.x << ", " << lo.y << ", " << lo.z << "]   expected ["
                  << testCase.expected.x << ", " << testCase.expected.y << ", "
                  << testCase.expected.z << "]\n";
        for (int c = 0; c < 3; ++c) {
            if (std::fabs(lo[c] - testCase.expected[c]) > kRelativeTolerance * testCase.expected[c]) {
                std::cerr << "integrator_validate: FAILED Beer-Lambert at " << testCase.name
                          << " channel " << c << " -- measured " << lo[c] << ", expected "
                          << testCase.expected[c]
                          << ". transmissionDepth is the distance at which transmittance reaches "
                             "transmissionColor, so with the medium index-matched to vacuum nothing "
                             "but exp(-sigma_a*d) can move this reading.\n";
                ok = false;
            }
        }
    }
    finish(ctx, ok, "beer_lambert_absorption failed; see the rows above");
    return;
}

// The other half of the convention: transmissionDepth == 0 is a tint applied once per crossing, so distance must not matter at all.
PT_CHECK(on_surface_transmission_tint, Slow, Statistical) {
    constexpr int kBounces = 8;
    constexpr float kSlabThickness = 0.5F;
    // Relative and tight: with no absorption and no refraction there is no distance bias, so both rows carry only estimator noise.
    constexpr float kRelativeTolerance = 0.002F;
    const glm::vec3 colour(0.5F, 0.25F, 0.75F);
    const glm::vec3 expected = colour * colour;   // two interfaces crossed, one factor each

    struct TintCase {
        const char* name;
        bool sphere;
    };
    const std::array<TintCase, 2> cases{{{"flat slab, depth 0", false}, {"sphere, depth 0", true}}};

    const EnvironmentMap env = makeUniformEnvironment();
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    bool ok = true;

    std::cout << "integrator_validate: on-surface transmission tint at transmissionDepth 0\n";
    for (const TintCase& testCase : cases) {
        const TestScene scene = withSurface(testCase.sphere ? makeSphereScene(0.0F) : makeSlabScene(0.0F, kSlabThickness),
                                            [&](Material& material) {
                                                transmissive(material);
                                                indexMatched(material);
                                                material.transmissionColor = colour;
                                            });
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree scene for " << testCase.name
                      << '\n';
            finish(ctx, false, "on_surface_transmission_tint failed; see the rows above");
            return;
        }
        const glm::vec3 lo = centreMean(renderPass(scene, env, makeSettings(kBounces, 999), *accel, pool, true).beauty);

        std::cout << "  " << testCase.name;
        for (std::size_t pad = std::string(testCase.name).size(); pad < 36; ++pad) {
            std::cout << ' ';
        }
        std::cout << "measured [" << lo.x << ", " << lo.y << ", " << lo.z << "]   expected ["
                  << expected.x << ", " << expected.y << ", " << expected.z << "]\n";
        for (int c = 0; c < 3; ++c) {
            if (std::fabs(lo[c] - expected[c]) > kRelativeTolerance * expected[c]) {
                std::cerr << "integrator_validate: FAILED on-surface transmission tint at "
                          << testCase.name << " channel " << c << " -- measured " << lo[c]
                          << ", expected " << expected[c]
                          << ". At transmissionDepth 0 there is no medium to absorb in, so "
                             "transmissionColor tints each interface crossing once and nothing "
                             "else can move this reading.\n";
                ok = false;
            }
        }
    }
    finish(ctx, ok, "on_surface_transmission_tint failed; see the rows above");
    return;
}

// Per-instance material binding, integrator side: two coplanar quads distinguished only by each instance's base_color.
PT_CHECK(per_instance_materials, Slow, Statistical) {
    constexpr float kRoughness = 1.0F;
    constexpr float kDominance = 4.0F;   // the off-channel is the white glossy layer, not zero, so this is a ratio test, not an equality
    constexpr float kSwapTolerance = 0.02F;
    const glm::vec3 red(1.0F, 0.0F, 0.0F);
    const glm::vec3 blue(0.0F, 0.0F, 1.0F);

    const TestScene scene = makeTwoInstanceScene(kRoughness);
    std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
    if (!accel.has_value()) {
        std::cerr << "integrator_validate: FAILED to build Embree two-instance scene\n";
        finish(ctx, false, "per_instance_materials failed; see the rows above");
        return;
    }

    const EnvironmentMap env = makeUniformEnvironment();
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    const auto render = [&](const glm::vec3& left, const glm::vec3& right) {
        TestScene coloured = scene;
        coloured.instances[0].material.baseColor = left;
        coloured.instances[1].material.baseColor = right;
        const pathtracer::gfx::HdrImage beauty = renderPass(coloured, env, makeSettings(1, 999), *accel, pool, /*showSky=*/true).beauty;
        return std::pair{regionMean(beauty, 2, 6, 6, 10), regionMean(beauty, 10, 6, 14, 10)};
    };

    bool ok = true;
    const auto [leftRed, rightBlue] = render(red, blue);
    const auto [leftBlue, rightRed] = render(blue, red);

    std::cout << "integrator_validate: per-instance material binding (two instances, one material each)\n";
    std::cout << "  entry0=red   entry1=blue   left [" << leftRed.x << ", " << leftRed.z
              << "]  right [" << rightBlue.x << ", " << rightBlue.z << "]  (r, b)\n";
    std::cout << "  entry0=blue  entry1=red    left [" << leftBlue.x << ", " << leftBlue.z
              << "]  right [" << rightRed.x << ", " << rightRed.z << "]  (r, b)\n";

    if (!(leftRed.x > kDominance * leftRed.z) || !(rightBlue.z > kDominance * rightBlue.x)) {
        std::cerr << "integrator_validate: FAILED per-instance material binding -- with entry 0 red "
                     "and entry 1 blue, the left quad (instanceIndex 0) read ["
                  << leftRed.x << ", " << leftRed.z << "] and the right (instanceIndex 1) read ["
                  << rightBlue.x << ", " << rightBlue.z
                  << "] in (r, b). Each instance's triangles must resolve to its own "
                     "material; a swap or a constant index lands here.\n";
        ok = false;
    }
    // Swapping the materials must swap the picture: each block is compared against the other assignment's opposite block.
    if (std::fabs(leftRed.x - rightRed.x) > kSwapTolerance ||
        std::fabs(rightBlue.z - leftBlue.z) > kSwapTolerance) {
        std::cerr << "integrator_validate: FAILED per-instance material binding under swap -- "
                     "exchanging the two instances' base_color must exchange the two blocks' "
                     "readings, but red measured " << leftRed.x << " on the left and " << rightRed.x
                  << " on the right, blue " << rightBlue.z << " then " << leftBlue.z << ".\n";
        ok = false;
    }
    finish(ctx, ok, "per_instance_materials failed; see the rows above");
    return;
}

// An emission-only surface returns emission_luminance x emission_color to a camera ray whatever lights the scene: the authored value.
PT_CHECK(emission_returns_luminance_times_colour, Fast, Exact) {
    constexpr int kTexelSamples = 16;
    constexpr int kEnvWidth = 64;
    constexpr int kEnvHeight = 32;
    const glm::vec3 colour(0.2F, 0.5F, 0.8F);
    constexpr float kLuminance = 2.5F;
    const glm::vec3 expected = kLuminance * colour;

    TestScene scene = makeTwoInstanceScene(1.0F);
    Material& emitter = scene.instances[0].material;
    emitter.baseWeight = 0.0F;
    emitter.specularWeight = 0.0F;
    emitter.emissionLuminance = kLuminance;
    emitter.emissionColor = colour;
    std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
    if (!accel.has_value()) {
        finish(ctx, false, "failed to build the Embree two-instance scene");
        return;
    }
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    PathTraceSettings base = makeSettings(1, 999);
    base.samplesPerPixel = kTexelSamples;

    const EnvironmentMap black = tools::fixtures::makeEnvironment(
        kEnvWidth, kEnvHeight, std::vector<float>(static_cast<std::size_t>(kEnvWidth) * kEnvHeight * pathtracer::gfx::kRgbChannels, 0.0F));
    const EnvironmentMap uniform = makeUniformEnvironment();
    // Identical samples average to themselves up to one rounding per accumulation: kTexelSamples per pixel, then the 16-pixel block.
    const float tolerance = static_cast<float>(kTexelSamples + 16) * std::numeric_limits<float>::epsilon() *
                            std::max({expected.x, expected.y, expected.z});

    ctx.plan(4);
    // A clear coat absorbs nothing, so it leaves emission bit for bit; a tinted one can only take light away, never add it.
    TestScene clear = scene;
    clear.instances[0].material.coatWeight = 1.0F;
    TestScene tinted = scene;
    tinted.instances[0].material.coatWeight = 0.5F;
    tinted.instances[0].material.coatColor = glm::vec3(0.25F, 0.5F, 0.81F);
    const glm::vec3 open = regionMean(renderPass(scene, black, base, *accel, pool, true).beauty, 2, 6, 6, 10);
    const glm::vec3 throughClear = regionMean(renderPass(clear, black, base, *accel, pool, true).beauty, 2, 6, 6, 10);
    const glm::vec3 throughTint = regionMean(renderPass(tinted, black, base, *accel, pool, true).beauty, 2, 6, 6, 10);
    PT_EXPECT(ctx, throughClear == open, "a clear coat changed the emission beneath it");
    PT_EXPECT(ctx, glm::all(glm::lessThan(throughTint, open)) && glm::all(glm::greaterThan(throughTint, 0.5F * open)),
              "a half-weight tinted coat must dim emission, by at most its weight: (1-C) + C*T with 0 < T < 1");
    for (const auto& [name, env] : {std::pair{"black", &black}, std::pair{"uniform L0=1", &uniform}}) {
        const glm::vec3 mean =
            regionMean(renderPass(scene, *env, base, *accel, pool, true).beauty, 2, 6, 6, 10);
        const float maxError = std::max({std::fabs(mean.x - expected.x), std::fabs(mean.y - expected.y),
                                         std::fabs(mean.z - expected.z)});
        char detail[288];
        std::snprintf(detail, sizeof(detail),
                      "%s environment: emitter read (%.7f, %.7f, %.7f), expected luminance x colour (%.7f, %.7f, %.7f), "
                      "max error %.3e vs %.3e -- lighting reached a surface with no BSDF",
                      name, mean.x, mean.y, mean.z, expected.x, expected.y, expected.z, maxError, tolerance);
        std::cout << "  " << detail << '\n';
        PT_EXPECT(ctx, maxError <= tolerance, detail);
    }
}

// Furnace equivalence: a floor under an emitting ceiling of radiance L0 and a uniform-L0 sky sees L0 everywhere, as under the sky alone.
PT_CHECK(emission_reaches_indirect, Slow, Statistical) {
    // Above the camera's z=5, so primary rays reach the floor; the floor then sees the ceiling over all but its grazing horizon.
    constexpr float kCeilingHeight = 10.0F;
    TestScene enclosed = makeQuadScene(1.0F);
    const glm::vec3 down(0.0F, 0.0F, -1.0F);
    const glm::vec4 tangent(1.0F, 0.0F, 0.0F, 1.0F);
    const auto vertex = [&](float x, float y) {
        return ShadingVertex{glm::vec3(x, y, kCeilingHeight), down, glm::vec2(0.5F, 0.5F), tangent};
    };
    // Wound clockwise as seen from +Z so geometricNormalOf gives (0,0,-1), facing the floor.
    const ShadingVertex c0 = vertex(-kQuadExtent, -kQuadExtent);
    const ShadingVertex c1 = vertex(kQuadExtent, -kQuadExtent);
    const ShadingVertex c2 = vertex(kQuadExtent, kQuadExtent);
    const ShadingVertex c3 = vertex(-kQuadExtent, kQuadExtent);
    enclosed.worldTriangles.push_back(Triangle{c0.position, c3.position, c2.position});
    enclosed.worldTriangles.push_back(Triangle{c0.position, c2.position, c1.position});
    enclosed.shadingTriangles.push_back(ShadingTriangle{c0, c3, c2, 1});
    enclosed.shadingTriangles.push_back(ShadingTriangle{c0, c2, c1, 1});
    // Emission alone, white at luminance 1, matching the sky's L0 = 1.
    Material ceiling;
    ceiling.baseWeight = 0.0F;
    ceiling.specularWeight = 0.0F;
    ceiling.emissionLuminance = 1.0F;
    enclosed.instances.push_back(MeshInstance{ceiling, glm::mat4(1.0F), ""});
    const TestScene open = makeQuadScene(1.0F);

    std::optional<EmbreeAccel> enclosedAccel = EmbreeAccel::build(enclosed.worldTriangles);
    std::optional<EmbreeAccel> openAccel = EmbreeAccel::build(open.worldTriangles);
    if (!enclosedAccel.has_value() || !openAccel.has_value()) {
        finish(ctx, false, "failed to build the Embree furnace scenes");
        return;
    }
    const EnvironmentMap env = makeUniformEnvironment();
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    PathTraceSettings settings = makeSettings(1, 999);
    settings.samplesPerPixel = kSamplesPerPixel / tools::stats::kReplicates;

    const std::uint64_t rowSeed = ctx.subSeed("emitting ceiling");
    tools::stats::Welford enclosedMean;
    tools::stats::Welford openMean;
    for (int r = 0; r < tools::stats::kReplicates; ++r) {
        // Disjoint seeds keep the two estimators independent, as Welch's band assumes.
        const auto seed = static_cast<std::uint32_t>(rowSeed + r);
        const glm::vec3 a = regionMean(
            renderPass(enclosed, env, settings, *enclosedAccel, pool, true, seed).beauty, 0, 0,
            kImageSize, kImageSize);
        const glm::vec3 b = regionMean(renderPass(open, env, settings, *openAccel, pool, true,
                                                  seed + tools::stats::kReplicates).beauty,
                                       0, 0, kImageSize, kImageSize);
        enclosedMean.add(std::max({a.x, a.y, a.z}));
        openMean.add(std::max({b.x, b.y, b.z}));
    }
    const double difference = enclosedMean.mean() - openMean.mean();
    const tools::stats::Band band = tools::stats::differenceBand(enclosedMean, openMean, ctx.alpha());
    char detail[320];
    std::snprintf(detail, sizeof(detail),
                  "emitting ceiling %.5f, open sky %.5f, difference %+.3e vs +/-%.3e -- below means indirect rays miss the "
                  "surface emission, above means NEE double-counts it",
                  enclosedMean.mean(), openMean.mean(), difference, band.halfWidth());
    std::cout << "integrator_validate: " << detail << '\n';
    ctx.plan(1);
    PT_EXPECT(ctx, band.contains(difference), detail);
}

// Every OpenPBR input of two materials equal, compared bitwise as the binding copies them.
bool sameMaterial(const Material& a, const Material& b) {
    bool same = true;
    forEachInput([&](const InputSpec&, const auto& x, const auto& y) { same = same && x == y; }, a, b);
    return same;
}

// Material binding: applySceneMaterials gives every instance materialPath's constants and a named node its override, atomically.
PT_CHECK(material_binding_resolution, Fast, Exact) {
    const std::string assetRoot = ASSET_ROOT_DIR;
    const std::optional<pathtracer::config::MaterialConfig> clay = pathtracer::config::loadMaterialConfig(assetRoot + "/materials/clay.json");
    const std::optional<pathtracer::config::MaterialConfig> glass = pathtracer::config::loadMaterialConfig(assetRoot + "/materials/glass.json");
    ctx.plan(7);
    PT_EXPECT(ctx, clay.has_value() && glass.has_value(), "materials/clay.json or materials/glass.json failed to load");
    if (!clay.has_value() || !glass.has_value()) {
        return;
    }
    const Material clayMaterial = pathtracer::scene::materialOf(*clay);
    const Material glassMaterial = pathtracer::scene::materialOf(*glass);
    // Non-vacuity: two shipped materials that resolved alike would let a dropped override pass unseen.
    PT_EXPECT(ctx, !sameMaterial(clayMaterial, glassMaterial), "clay.json and glass.json resolve to the same inputs");

    const auto makeInstances = [] {
        return std::vector<MeshInstance>{MeshInstance{makeMaterial(1.0F), glm::mat4(1.0F), "alpha"},
                                         MeshInstance{makeMaterial(1.0F), glm::mat4(1.0F), "beta"},
                                         MeshInstance{makeMaterial(1.0F), glm::mat4(1.0F), "gamma"}};
    };
    std::vector<MeshInstance> instances = makeInstances();
    const bool applied =
        pathtracer::scene::applySceneMaterials(instances, "materials/clay.json", {{"beta", "materials/glass.json"}}, assetRoot);
    PT_EXPECT(ctx, applied && sameMaterial(instances[1].material, glassMaterial), "the override keyed on 'beta' did not reach instance 1");
    PT_EXPECT(ctx, applied && sameMaterial(instances[0].material, clayMaterial) && sameMaterial(instances[2].material, clayMaterial),
              "an instance not named by the override lost materialPath's constants");

    // Every rejection path, each leaving the instances untouched: a scene that renders the wrong material is worse than one that refuses.
    std::cout << "  the stderr diagnostics below are expected: they are the function under test refusing a bad scene\n";
    std::vector<MeshInstance> untouched = makeInstances();
    PT_EXPECT(ctx, !pathtracer::scene::applySceneMaterials(untouched, "materials/clay.json", {{"delta", "materials/glass.json"}}, assetRoot),
              "an override key matching no instance name was accepted");
    PT_EXPECT(ctx, !pathtracer::scene::applySceneMaterials(untouched, "materials/clay.json", {{"beta", "materials/does_not_exist.json"}},
                                                           assetRoot),
              "an override naming an unloadable material file was accepted");
    PT_EXPECT(ctx,
              !pathtracer::scene::applySceneMaterials(untouched, "materials/does_not_exist.json", {}, assetRoot) &&
                  sameMaterial(untouched[0].material, makeMaterial(1.0F)),
              "an unloadable materialPath was accepted, or a rejected scene still changed an instance");
}

// buildQuadLights places a centred quad emitting local -Z by sceneTransform * T * Rz * Ry * Rx, its edges w and h, perpendicular.
PT_CHECK(quad_light_placement, Fast, Exact) {
    struct Case {
        const char* name;
        pathtracer::config::QuadLightConfig light;
        glm::mat4 sceneTransform;
        glm::vec3 expectedNormal;
    };
    const auto light = [](glm::vec3 position, glm::vec3 rotation, glm::vec2 size) {
        return pathtracer::config::QuadLightConfig{position, rotation, size, glm::vec3(1.0F), 1.0F};
    };
    const glm::mat4 identity(1.0F);
    const glm::mat4 sceneMoved = glm::translate(identity, glm::vec3(1.0F, 2.0F, 3.0F)) *
                                 glm::rotate(identity, glm::radians(90.0F), glm::vec3(0.0F, 1.0F, 0.0F));
    const std::vector<Case> cases = {
        {"rest pose", light({0.1F, 0.2F, 0.3F}, {0.0F, 0.0F, 0.0F}, {0.4F, 0.2F}), identity, {0.0F, 0.0F, -1.0F}},
        {"ceiling, Rx(-90)", light({0.0F, 0.49F, 0.0F}, {-90.0F, 0.0F, 0.0F}, {0.3F, 0.3F}), identity, {0.0F, -1.0F, 0.0F}},
        // X before Y gives +Y; Y before X would give -X, so this row pins the Euler order.
        {"Rx(90) then Ry(90)", light({0.0F, 0.0F, 0.0F}, {90.0F, 90.0F, 0.0F}, {0.5F, 0.25F}), identity, {0.0F, 1.0F, 0.0F}},
        // Ry(90) takes -Z to -X, then Rz(90) to -Y; Z before Y would give -X, so this row pins Y's and Z's sense and their order.
        {"Ry(90) then Rz(90)", light({0.0F, 0.0F, 0.0F}, {0.0F, 90.0F, 90.0F}, {0.5F, 0.25F}), identity, {0.0F, -1.0F, 0.0F}},
        {"under a moved scene root", light({0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.3F, 0.6F}), sceneMoved, {-1.0F, 0.0F, 0.0F}},
    };
    // Higham's gamma_n, n = 8 chained 4-term products, at coordinate magnitude <= 4; eps, not u = eps/2, also covers sin/cos's ulp.
    constexpr float kTolerance = 8.0F * 4.0F * std::numeric_limits<float>::epsilon() * 4.0F;

    ctx.plan(static_cast<int>(cases.size()) + 1);
    for (const Case& testCase : cases) {
        const pathtracer::scene::QuadLight quad = pathtracer::scene::buildQuadLights({testCase.light}, testCase.sceneTransform)[0];
        const glm::vec3 centre = quad.origin + (0.5F * quad.edge0) + (0.5F * quad.edge1);
        const glm::vec3 expectedCentre = glm::vec3(testCase.sceneTransform * glm::vec4(testCase.light.position, 1.0F));
        const float centreError = glm::length(centre - expectedCentre);
        const float normalError = glm::length(quad.normal - testCase.expectedNormal);
        const float widthError = std::fabs(glm::length(quad.edge0) - testCase.light.size.x);
        const float heightError = std::fabs(glm::length(quad.edge1) - testCase.light.size.y);
        const float skew = std::fabs(glm::dot(quad.edge0, quad.edge1));
        const float maxError = std::max({centreError, normalError, widthError, heightError, skew});
        char detail[288];
        std::snprintf(detail, sizeof(detail),
                      "%s: centre error %.3e, normal (%.6f, %.6f, %.6f) error %.3e, width/height error %.3e/%.3e, edge dot %.3e "
                      "vs %.3e",
                      testCase.name, centreError, quad.normal.x, quad.normal.y, quad.normal.z, normalError, widthError,
                      heightError, skew, kTolerance);
        std::cout << "  " << detail << '\n';
        PT_EXPECT(ctx, maxError <= kTolerance, detail);
    }

    // The shipped cornell light must reproduce the retired corner-authored quad it replaced, corner and both edges.
    const pathtracer::scene::QuadLight ceiling = pathtracer::scene::buildQuadLights({cases[1].light}, identity)[0];
    const float migrationError = std::max({glm::length(ceiling.origin - glm::vec3(-0.15F, 0.49F, -0.15F)),
                                           glm::length(ceiling.edge0 - glm::vec3(0.3F, 0.0F, 0.0F)),
                                           glm::length(ceiling.edge1 - glm::vec3(0.0F, 0.0F, 0.3F))});
    char detail[160];
    std::snprintf(detail, sizeof(detail), "ceiling light vs the retired corner (-0.15, 0.49, -0.15) + edges: max error %.3e vs %.3e",
                  migrationError, kTolerance);
    std::cout << "  " << detail << '\n';
    PT_EXPECT(ctx, migrationError <= kTolerance, detail);
}

// Texture binding: bindSceneTextures fills exactly the named nodes' named inputs, atomically, and rejects every bad entry.
PT_CHECK(texture_binding_resolution, Fast, Exact) {
    const std::filesystem::path root = std::filesystem::temp_directory_path();
    const std::string exr = "engine_integrator_scene_texture.exr";
    const glm::vec3 texel(0.25F, 0.5F, 0.75F);
    const bool written = pathtracer::gfx::writeExr((root / exr).string(),
                                                   {1, 1, pathtracer::gfx::kRgbChannels, {texel.x, texel.y, texel.z}},
                                                   pathtracer::gfx::ImageRole::Colour);
    // "beta" owns two primitives, as a multi-primitive glTF node does: an override must reach both.
    const auto makeInstances = [] {
        return std::vector<MeshInstance>{MeshInstance{makeMaterial(1.0F), glm::mat4(1.0F), "alpha"},
                                         MeshInstance{makeMaterial(1.0F), glm::mat4(1.0F), "beta"},
                                         MeshInstance{makeMaterial(1.0F), glm::mat4(1.0F), "beta"}};
    };
    using pathtracer::config::TextureConfig;
    const auto apply = [&](std::vector<MeshInstance>& instances,
                           const std::map<std::string, std::map<std::string, TextureConfig>>& textures) {
        return pathtracer::scene::bindSceneTextures(instances, textures, root.string());
    };
    const TextureConfig file{exr, std::nullopt, 0, std::nullopt};
    const auto texelOf = [](const TextureHandle* handle) { return pathtracer::gfx::sampleTexture(**handle, glm::vec2(0.5F)); };
    const auto channelsOf = [](const TextureHandle* handle) { return pathtracer::gfx::readTexels(**handle).value().channels; };
    const glm::vec3 rgb = texel;
    // The scalar input holds R alone: had the cache handed it the RGB lookup, its texel would read (0.25, 0.5, 0.75).
    const glm::vec3 red(texel.r, 0.0F, 0.0F);
    // A packed glTF metallic-roughness map binds a scalar input to G: the channel offset reaches the lookup as R.
    const glm::vec3 green(texel.g, 0.0F, 0.0F);

    ctx.plan(22);
    PT_EXPECT(ctx, written, "could not write the override EXR");
    std::vector<MeshInstance> instances = makeInstances();
    const TextureConfig packed{exr, std::nullopt, 1, std::nullopt};
    PT_EXPECT(ctx, apply(instances, {{"beta", {{"base_color", file}, {"specular_roughness", file}, {"base_metalness", packed}}}}),
              "a valid override was rejected");
    const TextureHandle* baseColor1 = std::get_if<TextureHandle>(&instances[1].material.baseColor);
    const TextureHandle* baseColor2 = std::get_if<TextureHandle>(&instances[2].material.baseColor);
    const TextureHandle* roughness1 = std::get_if<TextureHandle>(&instances[1].material.specularRoughness);
    const TextureHandle* roughness2 = std::get_if<TextureHandle>(&instances[2].material.specularRoughness);
    const TextureHandle* metalness1 = std::get_if<TextureHandle>(&instances[1].material.baseMetalness);
    PT_EXPECT(ctx, baseColor1 && baseColor2 && texelOf(baseColor1) == rgb && texelOf(baseColor2) == rgb,
              "base_color did not reach both of beta's primitives");
    PT_EXPECT(ctx, roughness1 && roughness2 && texelOf(roughness1) == red && texelOf(roughness2) == red,
              "specular_roughness did not reach both of beta's primitives as R alone");
    PT_EXPECT(ctx, metalness1 && texelOf(metalness1) == green, "base_metalness bound to channel g did not read G");
    // One file in an RGB and a scalar slot: each must hold its own channel count, not whichever open the cache kept first.
    PT_EXPECT(ctx, baseColor1 && roughness1 && channelsOf(baseColor1) == pathtracer::gfx::kRgbChannels &&
                       channelsOf(roughness1) == pathtracer::gfx::kScalarChannels,
              "a file shared by an RGB and a scalar input was stored at one channel count for both");
    // One open per (file, channel count, role): beta's two primitives must share one texture, not open a 4K file each.
    PT_EXPECT(ctx, baseColor1 && baseColor2 && roughness1 && roughness2 && *baseColor1 == *baseColor2 &&
                       *roughness1 == *roughness2,
              "beta's primitives hold separate copies of one decoded texture");
    // An unbound input stays its constant, not a texture: shading then filters nothing for it.
    PT_EXPECT(ctx, instances[1].material.specularIor == makeMaterial(1.0F).specularIor && !instances[1].material.geometryNormal.map,
              "an unnamed input on an overridden node changed");
    PT_EXPECT(ctx, instances[0].material.baseColor == makeMaterial(1.0F).baseColor, "the override leaked onto alpha, which it does not name");
    // geometry_coat_normal is the coat's own normal input: a bump on it reaches the coat's sources and leaves the base's alone.
    std::vector<MeshInstance> coatBumped = makeInstances();
    PT_EXPECT(ctx, apply(coatBumped, {{"alpha", {{"geometry_coat_normal", TextureConfig{std::nullopt, std::nullopt, 0, pathtracer::config::BumpConfig{exr, 0.01F}}}}}}),
              "a bump on geometry_coat_normal was rejected");
    PT_EXPECT(ctx, coatBumped[0].material.geometryCoatNormal.height && coatBumped[0].material.geometryCoatNormal.heightMetres == 0.01F &&
                       !coatBumped[0].material.geometryNormal.height,
              "geometry_coat_normal's bump did not reach the coat's normal input alone");

    std::cout << "  the stderr diagnostics below are expected: they are the function under test refusing a bad scene\n";
    std::vector<MeshInstance> untouched = makeInstances();
    PT_EXPECT(ctx, !apply(untouched, {{"delta", {{"base_color", file}}}}), "a key naming no node was accepted");
    // A pre-OpenPBR slot name binds nothing: rejected rather than loading a map no input reads.
    PT_EXPECT(ctx, !apply(untouched, {{"beta", {{"baseColorTexture", file}}}}), "a retired slot name was accepted");
    PT_EXPECT(ctx, !apply(untouched, {{"beta", {{"base_color", TextureConfig{"does_not_exist.exr", std::nullopt, 0, std::nullopt}}}}}),
              "a missing texture file was accepted");
    // A normal map is data: naming a colour space for it asks for a conversion that would corrupt every direction it encodes.
    PT_EXPECT(ctx, !apply(untouched, {{"beta", {{"geometry_normal", TextureConfig{exr, "ACEScg", 0, std::nullopt}}}}}),
              "a colour space on a data input was accepted");
    // A channel picks one of a scalar input's R, G, B; a colour input reads all three, and only a normal input takes a bump.
    PT_EXPECT(ctx, !apply(untouched, {{"beta", {{"base_color", TextureConfig{exr, std::nullopt, 1, std::nullopt}}}}}),
              "a channel on a colour input was accepted");
    PT_EXPECT(ctx, !apply(untouched, {{"beta", {{"base_weight", TextureConfig{exr, std::nullopt, 0, pathtracer::config::BumpConfig{exr, 1.0F}}}}}}),
              "a bump on an input other than a normal input was accepted");
    // Volumetric switches need transport this renderer lacks: binding one must fail, not render as if it were zero.
    PT_EXPECT(ctx, !apply(untouched, {{"beta", {{"subsurface_weight", file}}}}), "a texture on a volumetric switch was accepted");
    // alpha is valid and sorts first, so a non-atomic implementation would have bound it before beta's missing file failed.
    PT_EXPECT(ctx, !apply(untouched, {{"alpha", {{"base_color", file}}}, {"beta", {{"base_color", TextureConfig{"missing.exr", std::nullopt, 0, std::nullopt}}}}}),
              "a scene with one bad entry was accepted");
    PT_EXPECT(ctx, untouched[0].material.baseColor == makeMaterial(1.0F).baseColor,
              "a rejected scene still rebound alpha: validation must finish before any instance changes");

    // A texel past base_color's [0, 1] is an encoding excursion: bound with a warning, then clamped into the range at every lookup.
    const std::string excursion = "engine_integrator_scene_excursion.exr";
    const std::string nonFinite = "engine_integrator_scene_nonfinite.exr";
    const bool excursionWritten =
        pathtracer::gfx::writeExr((root / excursion).string(), {1, 1, pathtracer::gfx::kRgbChannels, {1.5F, -0.25F, 0.5F}},
                                  pathtracer::gfx::ImageRole::Colour);
    const bool nonFiniteWritten = pathtracer::gfx::writeExr(
        (root / nonFinite).string(), {1, 1, pathtracer::gfx::kRgbChannels, {std::numeric_limits<float>::infinity(), 0.0F, 0.0F}},
        pathtracer::gfx::ImageRole::Colour);
    std::vector<MeshInstance> clamped = makeInstances();
    const bool excursionBound =
        excursionWritten && apply(clamped, {{"alpha", {{"base_color", TextureConfig{excursion, std::nullopt, 0, std::nullopt}}}}});
    const glm::vec3 resolved = resolveInputs(clamped[0].material, glm::vec2(0.5F), {}, glm::vec3(1.0F)).baseColor;
    PT_EXPECT(ctx, excursionBound && resolved == glm::vec3(1.0F, 0.0F, 0.5F), "an out-of-range base_color texel was not clamped into [0, 1]");
    PT_EXPECT(ctx, nonFiniteWritten && !apply(untouched, {{"alpha", {{"base_color", TextureConfig{nonFinite, std::nullopt, 0, std::nullopt}}}}}),
              "a texture with a non-finite texel was bound");
    std::filesystem::remove(root / excursion);
    std::filesystem::remove(root / nonFinite);
    std::filesystem::remove(root / exr);
}

// The five transport buckets are a partition of beauty, not related-looking images: their sum must equal Beauty at every pixel.
PT_CHECK(transport_aov_partition, Slow, Exact) {
    // Relative to beauty, since the absolute scale differs by case; float error over kSamplesPerPixel accumulations sits orders below this.
    constexpr float kTolerance = 1e-4F;

    enum class Geometry { Quad, Corner, Slab };
    struct PartitionCase {
        const char* name;
        Geometry geometry;
        float roughness;
        float metallic;
        float transmission;
        int maxBounces;
        float abbe;  // 0: no dispersion
    };
    // The dispersive row is here because the hero channel is the one mechanism writing different values into channels of one throughput.
    const std::array<PartitionCase, 7> cases{{
        {"quad diffuse (rough 1.0)", Geometry::Quad, 1.0F, 0.0F, 0.0F, 1, 0.0F},
        {"quad glossy dielectric (rough 0.35)", Geometry::Quad, 0.35F, 0.0F, 0.0F, 1, 0.0F},
        {"quad rough metal (rough 0.5)", Geometry::Quad, 0.5F, 1.0F, 0.0F, 2, 0.0F},
        {"corner diffuse (rough 1.0)", Geometry::Corner, 1.0F, 0.0F, 0.0F, 4, 0.0F},
        {"slab smooth glass (rough 0)", Geometry::Slab, 0.0F, 0.0F, 1.0F, 8, 0.0F},
        {"slab rough glass (rough 0.4)", Geometry::Slab, 0.4F, 0.0F, 1.0F, 8, 0.0F},
        {"slab dispersive glass (rough 0.4)", Geometry::Slab, 0.4F, 0.0F, 1.0F, 8, 64.17F},
    }};

    const EnvironmentMap env = makeUniformEnvironment();
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    bool ok = true;

    std::cout << "integrator_validate: transport buckets partition beauty (showSky off)\n";
    for (const PartitionCase& testCase : cases) {
        const TestScene geometry = testCase.geometry == Geometry::Slab     ? makeSlabScene(testCase.roughness, 0.5F)
                                   : testCase.geometry == Geometry::Corner ? makeCornerScene(testCase.roughness)
                                                                           : makeQuadScene(testCase.roughness);
        const TestScene scene = withSurface(geometry, [&](Material& material) {
            material.baseMetalness = testCase.metallic;
            material.transmissionWeight = testCase.transmission;
            material.transmissionDispersionScale = testCase.abbe > 0.0F ? 1.0F : 0.0F;
            material.transmissionDispersionAbbeNumber = testCase.abbe > 0.0F ? testCase.abbe : 20.0F;
        });
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree scene for " << testCase.name << '\n';
            finish(ctx, false, "transport_aov_partition failed; see the rows above");
            return;
        }
        const pathtracer::scene::PathTraceResult result =
            renderPass(scene, env, makeSettings(testCase.maxBounces, 999), *accel, pool, /*showSky=*/false);

        float worstGap = 0.0F;
        float worstBeauty = 0.0F;
        float maxIndirect = 0.0F;
        // All six lanes are RGB, so one index walks every channel of every texel in step.
        for (std::size_t i = 0; i < result.beauty.texels.size(); ++i) {
            const float beauty = result.beauty.texels[i];
            const float sum = result.directDiffuse.texels[i] + result.indirectDiffuse.texels[i] +
                               result.directSpecular.texels[i] + result.indirectSpecular.texels[i] +
                               result.refraction.texels[i];
            const float gap = std::fabs(beauty - sum) / std::max(std::fabs(beauty), 1e-3F);
            if (gap > worstGap) {
                worstGap = gap;
                worstBeauty = beauty;
            }
            maxIndirect = std::max({maxIndirect, result.indirectDiffuse.texels[i], result.indirectSpecular.texels[i]});
        }

        std::cout << "  " << testCase.name;
        for (std::size_t pad = std::string(testCase.name).size(); pad < 38; ++pad) {
            std::cout << ' ';
        }
        std::cout << "worst relative gap " << worstGap << "   max indirect " << maxIndirect << '\n';
        // Without the corner a flat quad escapes at bounce 1 and the slab's paths stay transmission-sticky, so both Indirect buckets are 0.
        if (testCase.geometry == Geometry::Corner && maxIndirect <= 0.0F) {
            std::cerr << "integrator_validate: FAILED transport partition coverage -- the corner scene "
                         "produced no Indirect bucket energy, so the identity below is not testing "
                         "them. Check that the wall is being hit by bounce rays.\n";
            ok = false;
        }
        if (worstGap > kTolerance) {
            std::cerr << "integrator_validate: FAILED transport partition at " << testCase.name
                      << " -- worst pixel is off by " << (worstGap * 100.0F) << "% of its beauty value "
                      << worstBeauty
                      << ". The five buckets must sum to beauty exactly once the background term is "
                         "zeroed; a gap means a contribution is unbucketed, double-bucketed or "
                         "rescaled.\n";
            ok = false;
        }
    }
    finish(ctx, ok, "transport_aov_partition failed; see the rows above");
    return;
}

// Appends a QuadLight's two emitting triangles as appendQuadLights does, so these checks meet a real scene's self-occlusion regime.
void appendLightGeometry(TestScene& scene, const pathtracer::scene::QuadLight& light, int quadIndex,
                          std::vector<int>& instanceLightIndex) {
    const glm::vec3& normal = light.normal;
    const glm::vec4 tangent(glm::normalize(light.edge0), 1.0F);
    const auto vertex = [&](const glm::vec3& position, glm::vec2 uv) {
        return ShadingVertex{position, normal, uv, tangent};
    };
    const ShadingVertex p00 = vertex(light.origin, glm::vec2(0.0F, 0.0F));
    const ShadingVertex p10 = vertex(light.origin + light.edge0, glm::vec2(1.0F, 0.0F));
    const ShadingVertex p01 = vertex(light.origin + light.edge1, glm::vec2(0.0F, 1.0F));
    const ShadingVertex p11 = vertex(light.origin + light.edge0 + light.edge1, glm::vec2(1.0F, 1.0F));
    const int instanceIndex = static_cast<int>(scene.instances.size());
    scene.worldTriangles.push_back(Triangle{p00.position, p10.position, p11.position});
    scene.worldTriangles.push_back(Triangle{p00.position, p11.position, p01.position});
    scene.shadingTriangles.push_back(ShadingTriangle{p00, p10, p11, instanceIndex});
    scene.shadingTriangles.push_back(ShadingTriangle{p00, p11, p01, instanceIndex});
    scene.instances.push_back(
        MeshInstance{makeMaterial(1.0F), glm::mat4(1.0F), "light"});
    instanceLightIndex.push_back(quadIndex);
}

// Like renderPass, but for scenes carrying real light-set state: an explicit environment pointer and emitter instances/quads.
pathtracer::scene::PathTraceResult renderPassWithLights(const TestScene& scene,
                                                     const std::vector<int>& instanceLightIndex,
                                                     const std::vector<pathtracer::scene::QuadLight>& quads,
                                                     const EnvironmentMap* env,
                                                     const PathTraceSettings& settings,
                                                     EmbreeAccel& accel, pathtracer::scene::ThreadPool& pool,
                                                     bool showSky) {
    const std::atomic<std::uint64_t> generation{1};
    pathtracer::scene::PathTraceResult result = pathtracer::scene::makePathTraceResult(kImageSize, kImageSize);
    const pathtracer::scene::LightSet lights(env, /*envRotationDegrees=*/glm::vec3(0.0F), /*envExposure=*/1.0F, quads);
    pathtracer::debug::PassStats stats;  // required by renderPathTraced; this tool checks radiance, not throughput
    pathtracer::scene::renderPathTraced(makeCamera(), accel, scene.shadingTriangles, scene.instances,
                                     instanceLightIndex, lights, kImageSize, kImageSize, showSky,
                                     settings, /*scrambleSeed=*/7U, /*sampleBase=*/0,
                                     /*sampleCount=*/settings.samplesPerPixel, generation,
                                     /*requestedGeneration=*/1U, pool, stats, result);
    return result;
}

// The rough lobe's half of the transmissionDepth 0 convention: Lo is exactly linear in transmissionColor.
PT_CHECK(rough_transmission_tint, Slow, Exact) {
    constexpr int kBounces = 4;
    // Numerical, not physical: the paths are identical, so the only slack is fl(t*x) summed vs fl(t * sum(x)); measured worst 1.37e-06.
    constexpr float kUlpBand = 1e-5F;
    const glm::vec3 tint(0.5F, 0.25F, 0.75F);
    // Both far-hemisphere strategies at their split's ends: msTransmit takes 2.6e-5 of the selection mass at roughness 0.15, 0.0263 at 0.6.
    const std::array<float, 2> roughnesses = {0.15F, 0.6F};
    // Behind the interface, emitting face pointing back at it, so the only way to the camera is through; the quad hides it from the camera.
    const pathtracer::scene::QuadLight light{glm::vec3(-0.5F, -0.5F, -1.5F), glm::vec3(1.0F, 0.0F, 0.0F),
                                          glm::vec3(0.0F, 1.0F, 0.0F), glm::vec3(3.0F)};
    const std::vector<pathtracer::scene::QuadLight> quads{light};

    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    bool ok = true;
    float worstRelative = 0.0F;

    std::cout << "integrator_validate: rough transmission tint at transmissionDepth 0 (ior 1.5, one interface)\n";
    for (float roughness : roughnesses) {
        TestScene scene = withSurface(makeQuadScene(roughness), transmissive);
        std::vector<int> instanceLightIndex(scene.instances.size(), -1);
        appendLightGeometry(scene, light, /*quadIndex=*/0, instanceLightIndex);
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree scene at roughness " << roughness
                      << '\n';
            finish(ctx, false, "rough_transmission_tint failed; see the rows above");
            return;
        }
        const auto render = [&](const glm::vec3& colour) {
            TestScene tinted = scene;
            tinted.instances[0].material.transmissionColor = colour;
            return centreMean(renderPassWithLights(tinted, instanceLightIndex, quads, /*env=*/nullptr, makeSettings(kBounces, 999), *accel,
                                                    pool, /*showSky=*/false)
                                   .beauty);
        };
        const glm::vec3 opaqueTint = render(glm::vec3(0.0F));
        const glm::vec3 white = render(glm::vec3(1.0F));
        const glm::vec3 tinted = render(tint);

        // Raw triples rather than a ratio: the ratio's denominator is what the guard below allows to be zero, so it would print nan.
        std::cout << "  roughness " << roughness << "   Lo(1) [" << white.x << ", " << white.y << ", "
                  << white.z << "]   Lo(tint) [" << tinted.x << ", " << tinted.y << ", " << tinted.z
                  << "]   Lo(0) [" << opaqueTint.x << ", " << opaqueTint.y << ", " << opaqueTint.z
                  << "]\n";
        for (int c = 0; c < 3; ++c) {
            // Anti-vacuity, per channel: a change that stopped the interface transmitting would send every row to 0 and satisfy both.
            if (!(white[c] > 0.0F)) {
                std::cerr << "integrator_validate: FAILED rough transmission tint at roughness="
                          << roughness << " channel " << c
                          << " -- an untinted interface transmitted nothing, so the rows below assert "
                             "about nothing\n";
                ok = false;
                continue;
            }
            if (opaqueTint[c] != 0.0F) {
                std::cerr << "integrator_validate: FAILED rough transmission tint at roughness="
                          << roughness << " channel " << c << " -- transmissionColor 0 read "
                          << opaqueTint[c]
                          << ", expected exactly 0. The only light in this scene is behind the "
                             "interface, so anything arriving that a black tint does not extinguish is "
                             "transmitted energy the tint never reached.\n";
                ok = false;
            }
            const float expected = tint[c] * white[c];
            worstRelative = std::max(worstRelative, std::fabs(tinted[c] - expected) / expected);
            if (std::fabs(tinted[c] - expected) > kUlpBand * expected) {
                std::cerr << "integrator_validate: FAILED rough transmission tint linearity at roughness="
                          << roughness << " channel " << c << " -- measured " << tinted[c]
                          << ", expected " << expected
                          << ". Every path here crosses exactly one interface, so transmissionColor "
                             "multiplies the reading once and the rough lobe must be exactly linear in "
                             "it.\n";
                ok = false;
            }
        }
    }
    std::cout << "  worst relative departure from linearity " << worstRelative << '\n';
    finish(ctx, ok, "rough_transmission_tint failed; see the rows above");
    return;
}

// Lambert (1760) / Baum, Rushmeier & Winget 1989 polygon form factor, summing over EDGES rather than Girard's internal angles.
float lambertPolygonIrradiance(const std::array<glm::vec3, 4>& vertices, const glm::vec3& p,
                                const glm::vec3& n, float radiance) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const glm::vec3 r0 = glm::normalize(vertices[i] - p);
        const glm::vec3 r1 = glm::normalize(vertices[(i + 1) % vertices.size()] - p);
        const float gamma = std::acos(glm::clamp(glm::dot(r0, r1), -1.0F, 1.0F));
        // cross(r1, r0), not (r0, r1): the sign convention matching Baum/Rushmeier/Winget's vertex ordering, fixed against the renderer.
        const glm::vec3 axis = glm::cross(r1, r0);
        const float axisLen = glm::length(axis);
        if (axisLen > 1e-8F) {
            sum += gamma * glm::dot(axis / axisLen, n);
        }
    }
    return radiance * 0.5F * sum;
}

// The receiver: makeQuadScene's own z=0 quad facing +Z, reused so every analytic reference below describes the same surface.
TestScene makeLambertianFloor() { return withSurface(makeQuadScene(/*roughness=*/1.0F), indexMatched); }

// A one-sided quad light facing -Z at the receiver, offset in x so its non-emitting back face stays out of the camera's view cone.
pathtracer::scene::QuadLight makeOverheadLight(bool twoSided = false) {
    return pathtracer::scene::QuadLight{glm::vec3(0.5F, -0.5F, 1.5F), glm::vec3(0.0F, 1.0F, 0.0F),
                                     glm::vec3(1.0F, 0.0F, 0.0F), glm::vec3(3.0F), twoSided};
}

// Irradiance against the closed form, a one-sided face reading 0 and an occluded light 0.
PT_CHECK(quad_light_irradiance_and_occlusion, Slow, Statistical) {
    constexpr float kTolerance = 0.02F;  // Monte Carlo NEE noise at kSamplesPerPixel, not a formula slop
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    bool ok = true;

    struct LightCase {
        const char* name;
        pathtracer::scene::QuadLight light;
    };
    const std::array<LightCase, 3> cases{{
        {"close, off to one side, unit quad", makeOverheadLight()},
        {"off-axis, unit quad",
         pathtracer::scene::QuadLight{glm::vec3(0.3F, 0.2F, 3.0F), glm::vec3(0.0F, 0.8F, 0.0F),
                                   glm::vec3(0.8F, 0.0F, 0.0F), glm::vec3(5.0F), false}},
        // Offset in x for the same camera-sightline reason, far enough that its near edge (x=0.5) clears the view cone at this depth.
        {"large, grazing", pathtracer::scene::QuadLight{glm::vec3(0.5F, -3.0F, 1.2F),
                                                      glm::vec3(0.0F, 6.0F, 0.0F),
                                                      glm::vec3(6.0F, 0.0F, 0.0F), glm::vec3(2.0F), false}},
    }};

    std::cout << "integrator_validate: quad light irradiance vs Lambert's closed-form polygon form factor\n";
    for (const LightCase& testCase : cases) {
        TestScene scene = makeLambertianFloor();
        std::vector<int> instanceLightIndex(scene.instances.size(), -1);
        appendLightGeometry(scene, testCase.light, /*quadIndex=*/0, instanceLightIndex);
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree scene for " << testCase.name << '\n';
            finish(ctx, false, "quad_light_irradiance_and_occlusion failed; see the rows above");
            return;
        }
        const std::vector<pathtracer::scene::QuadLight> quads{testCase.light};
        const glm::vec3 lo = centreMean(renderPassWithLights(scene, instanceLightIndex, quads,
                                                              /*env=*/nullptr,
                                                              makeSettings(0, 999), *accel, pool,
                                                              /*showSky=*/false)
                                             .beauty);

        const std::array<glm::vec3, 4> corners{
            testCase.light.origin, testCase.light.origin + testCase.light.edge0,
            testCase.light.origin + testCase.light.edge0 + testCase.light.edge1,
            testCase.light.origin + testCase.light.edge1};
        const float irradiance =
            lambertPolygonIrradiance(corners, glm::vec3(0.0F), glm::vec3(0.0F, 0.0F, 1.0F),
                                      testCase.light.radiance.x);
        const float reference = irradiance / kPi;

        std::cout << "  " << testCase.name << "   rendered " << lo.x << "   reference " << reference
                  << '\n';
        if (std::fabs(lo.x - reference) > kTolerance * std::max(reference, 0.05F)) {
            std::cerr << "integrator_validate: FAILED quad light irradiance at " << testCase.name
                      << " -- rendered " << lo.x << ", Lambert polygon reference " << reference << '\n';
            ok = false;
        }
    }

    // The same position with edge0/edge1 swapped, so only quadRadianceToward's front/back-face test stands between this and nonzero.
    {
        TestScene scene = makeLambertianFloor();
        std::vector<int> instanceLightIndex(scene.instances.size(), -1);
        const pathtracer::scene::QuadLight light{glm::vec3(0.5F, -0.5F, 1.5F), glm::vec3(1.0F, 0.0F, 0.0F),
                                              glm::vec3(0.0F, 1.0F, 0.0F), glm::vec3(3.0F), false};
        appendLightGeometry(scene, light, /*quadIndex=*/0, instanceLightIndex);
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree scene for one-sided check\n";
            finish(ctx, false, "quad_light_irradiance_and_occlusion failed; see the rows above");
            return;
        }
        const std::vector<pathtracer::scene::QuadLight> quads{light};
        const glm::vec3 lo = centreMean(renderPassWithLights(scene, instanceLightIndex, quads,
                                                              /*env=*/nullptr, makeSettings(0, 999),
                                                              *accel, pool, /*showSky=*/false)
                                             .beauty);
        std::cout << "  one-sided (light behind, facing away)   rendered [" << lo.x << ", " << lo.y
                  << ", " << lo.z << "]\n";
        if (lo.x > kTolerance || lo.y > kTolerance || lo.z > kTolerance) {
            std::cerr << "integrator_validate: FAILED quad light one-sidedness -- rendered [" << lo.x
                      << ", " << lo.y << ", " << lo.z
                      << "], expected exactly 0. A one-sided emitter's back face must emit nothing.\n";
            ok = false;
        }
    }

    // The first case's light plus a wall sized to its projected footprint, not kQuadExtent, which would swallow the camera sightline.
    {
        TestScene scene = makeLambertianFloor();
        std::vector<int> instanceLightIndex(scene.instances.size(), -1);
        const pathtracer::scene::QuadLight light = makeOverheadLight();
        appendLightGeometry(scene, light, /*quadIndex=*/0, instanceLightIndex);

        const glm::vec3 wallNormal(0.0F, 0.0F, 1.0F);
        const glm::vec4 wallTangent(1.0F, 0.0F, 0.0F, 1.0F);
        const auto wallVertex = [&](float x, float y) {
            return ShadingVertex{glm::vec3(x, y, 1.0F), wallNormal, glm::vec2(0.5F, 0.5F), wallTangent};
        };
        const ShadingVertex w0 = wallVertex(0.15F, -0.5F);
        const ShadingVertex w1 = wallVertex(1.25F, -0.5F);
        const ShadingVertex w2 = wallVertex(1.25F, 0.5F);
        const ShadingVertex w3 = wallVertex(0.15F, 0.5F);
        const int wallInstance = static_cast<int>(scene.instances.size());
        scene.worldTriangles.push_back(Triangle{w0.position, w1.position, w2.position});
        scene.worldTriangles.push_back(Triangle{w0.position, w2.position, w3.position});
        scene.shadingTriangles.push_back(ShadingTriangle{w0, w1, w2, wallInstance});
        scene.shadingTriangles.push_back(ShadingTriangle{w0, w2, w3, wallInstance});
        scene.instances.push_back(MeshInstance{makeLambertianFloor().instances[0].material, glm::mat4(1.0F), "wall"});
        instanceLightIndex.push_back(-1);

        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree scene for occlusion check\n";
            finish(ctx, false, "quad_light_irradiance_and_occlusion failed; see the rows above");
            return;
        }
        const std::vector<pathtracer::scene::QuadLight> quads{light};
        const glm::vec3 lo = centreMean(renderPassWithLights(scene, instanceLightIndex, quads,
                                                              /*env=*/nullptr, makeSettings(0, 999),
                                                              *accel, pool, /*showSky=*/false)
                                             .beauty);
        std::cout << "  occluded (opaque wall between light and receiver)   rendered [" << lo.x << ", "
                  << lo.y << ", " << lo.z << "]\n";
        if (lo.x > kTolerance || lo.y > kTolerance || lo.z > kTolerance) {
            std::cerr << "integrator_validate: FAILED quad light occlusion -- rendered [" << lo.x << ", "
                      << lo.y << ", " << lo.z
                      << "], expected exactly 0. An opaque wall between light and receiver must fully "
                         "block NEE; a non-zero reading means kShadowDistanceEpsilon's back-off (or the "
                         "shadow ray itself) is not reaching the blocker.\n";
            ok = false;
        }
    }
    finish(ctx, ok, "quad_light_irradiance_and_occlusion failed; see the rows above");
    return;
}

// A convex receiver cannot occlude itself from a light it faces, so Shadow is exactly 0 however coarsely the sphere is tessellated.
PT_CHECK(quad_light_visibility_on_curved_receiver, Slow, Exact) {
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    bool ok = true;

    // At z=3, offset in x: clear of the view cone (half-width 0.18 there), yet every visible fragment keeps geoCos > 0.44 -- no terminator.
    const pathtracer::scene::QuadLight light{glm::vec3(0.5F, -0.5F, 3.0F), glm::vec3(0.0F, 1.0F, 0.0F),
                                              glm::vec3(1.0F, 0.0F, 0.0F), glm::vec3(3.0F), false};

    // Chiang's origin offset is a tangent-plane distance, ~e^2/2R, so refining the mesh shrinks it: both rows must read 0 regardless.
    struct Row {
        const char* name;
        int slices;
        int stacks;
    };
    const std::array<Row, 2> rows{{{"fine   (64x32, equator edge 0.098)", kSphereSlices, kSphereStacks},
                                    {"coarse (32x16, equator edge 0.196)", 32, 16}}};

    for (const Row& row : rows) {
        TestScene scene = withSurface(makeSphereScene(/*roughness=*/1.0F, row.slices, row.stacks), indexMatched);
        std::vector<int> instanceLightIndex(scene.instances.size(), -1);
        appendLightGeometry(scene, light, /*quadIndex=*/0, instanceLightIndex);

        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree scene for curved-receiver check\n";
            finish(ctx, false, "quad_light_visibility_on_curved_receiver failed; see the rows above");
            return;
        }
        const std::vector<pathtracer::scene::QuadLight> quads{light};
        // Whole frame, not centreMean: (0,0,1) is a vertex, where the terminator offset vanishes and the failure is weakest.
        const pathtracer::scene::PathTraceResult result = renderPassWithLights(
            scene, instanceLightIndex, quads, /*env=*/nullptr, makeSettings(0, 999), *accel, pool,
            /*showSky=*/false);
        const float shadow = regionMean(result.shadow, 0, 0, kImageSize, kImageSize).x;
        std::cout << "  " << row.name << "   shadow " << shadow << "\n";
        if (shadow != 0.0F) {
            std::cerr << "integrator_validate: FAILED curved-receiver visibility -- " << row.name
                      << " reads shadow " << shadow
                      << ", expected exactly 0. Nothing lies between a convex receiver and a light it "
                         "faces, so a non-zero reading means the shadow ray is stopped by the light's "
                         "own front face: tMax measured at shading.position while the ray leaves from "
                         "the Chiang-offset origin.\n";
            ok = false;
        }
    }
    finish(ctx, ok, "quad_light_visibility_on_curved_receiver failed; see the rows above");
    return;
}

// Quadratic falloff is implicit: NEE divides by selectionPdf/solidAngle, and nothing asserted the point-source limit until this.
PT_CHECK(quad_light_inverse_square, Slow, Statistical) {
    constexpr float kTolerance = 0.02F;  // same Monte Carlo NEE noise band as checkQuadLightIrradiance above, at the same kSamplesPerPixel
    // The far row must reach the point-source limit; 1e-2 sits above the MC noise and far below the near-field deviation demanded below.
    constexpr float kPointLimitTolerance = 1e-2F;
    constexpr float kMinNearFieldDeviation = 0.25F;
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    bool ok = true;

    // Small square light offset in x to clear the view cone, held FIXED across the sweep, so the true distance is sqrt(kOffsetX^2 + z^2).
    constexpr float kOffsetX = 0.5F;
    constexpr float kSide = 0.2F;
    constexpr float kRadiance = 4.0F;
    const auto makeLight = [](float centreX, float z, float side, float radiance) {
        return pathtracer::scene::QuadLight{glm::vec3(centreX - (side * 0.5F), -side * 0.5F, z),
                                         glm::vec3(0.0F, side, 0.0F), glm::vec3(side, 0.0F, 0.0F),
                                         glm::vec3(radiance), false};
    };

    struct Row {
        const char* name;
        pathtracer::scene::QuadLight light;
        bool farField;  // true: must reach the point-source limit. false: the near-field conditioning guard.
    };
    const std::array<Row, 5> rows{{
        {"z=0.5  small", makeLight(kOffsetX, 0.5F, kSide, kRadiance), false},
        {"z=1.0  small", makeLight(kOffsetX, 1.0F, kSide, kRadiance), false},
        {"z=2.0  small", makeLight(kOffsetX, 2.0F, kSide, kRadiance), true},
        {"z=4.0  small", makeLight(kOffsetX, 4.0F, kSide, kRadiance), true},
        // Near field: a light 30x wider at half the nearest distance, subtending nearly the whole hemisphere, where irradiance flattens.
        {"z=0.5  large (near field)", makeLight(3.5F, 0.5F, 6.0F, 1.0F), false},
    }};

    std::cout << "integrator_validate: quad light inverse-square limit (E*d^2 / (L*A*cos_r*cos_l))\n";
    float previousRatioError = std::numeric_limits<float>::max();
    bool sawNearFieldDeviation = false;
    for (const Row& row : rows) {
        TestScene scene = makeLambertianFloor();
        std::vector<int> instanceLightIndex(scene.instances.size(), -1);
        appendLightGeometry(scene, row.light, /*quadIndex=*/0, instanceLightIndex);
        std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
        if (!accel.has_value()) {
            std::cerr << "integrator_validate: FAILED to build Embree scene for " << row.name << '\n';
            finish(ctx, false, "quad_light_inverse_square failed; see the rows above");
            return;
        }
        const std::vector<pathtracer::scene::QuadLight> quads{row.light};
        const glm::vec3 lo = centreMean(renderPassWithLights(scene, instanceLightIndex, quads,
                                                              /*env=*/nullptr,
                                                              makeSettings(0, 999), *accel, pool,
                                                              /*showSky=*/false)
                                             .beauty);
        const float renderedIrradiance = lo.x * kPi;

        const std::array<glm::vec3, 4> corners{
            row.light.origin, row.light.origin + row.light.edge0,
            row.light.origin + row.light.edge0 + row.light.edge1, row.light.origin + row.light.edge1};
        const glm::vec3 receiverNormal(0.0F, 0.0F, 1.0F);
        const float exactIrradiance = lambertPolygonIrradiance(corners, glm::vec3(0.0F), receiverNormal,
                                                                row.light.radiance.x);

        // Point-source limit: E = L * A * cos(theta_r) * cos(theta_l) / d^2, evaluated at the light's centre.
        const glm::vec3 centre = row.light.origin + (row.light.edge0 * 0.5F) + (row.light.edge1 * 0.5F);
        const float distance = glm::length(centre);
        const glm::vec3 toLight = centre / distance;
        const float area = glm::length(row.light.edge0) * glm::length(row.light.edge1);
        const float pointIrradiance = row.light.radiance.x * area * glm::dot(toLight, receiverNormal) *
                                       glm::dot(-toLight, row.light.normal) / (distance * distance);
        const float ratio = renderedIrradiance / pointIrradiance;

        std::cout << "  " << row.name << "   d " << distance << "   rendered E " << renderedIrradiance
                   << "   exact E " << exactIrradiance << "   E/E_point " << ratio << '\n';

        // The renderer must track the EXACT polygon irradiance at every distance, near field included.
        if (std::fabs(renderedIrradiance - exactIrradiance) > kTolerance * std::max(exactIrradiance, 0.05F)) {
            std::cerr << "integrator_validate: FAILED inverse-square row " << row.name
                       << " -- rendered E " << renderedIrradiance << ", Lambert polygon reference "
                       << exactIrradiance << '\n';
            ok = false;
        }
        const float ratioError = std::fabs(ratio - 1.0F);
        if (row.farField) {
            // Assert the PASS condition so a NaN ratio fails rather than slipping through an ordered compare.
            if (!(ratioError <= kPointLimitTolerance)) {
                std::cerr << "integrator_validate: FAILED inverse-square limit at " << row.name
                           << " -- E/E_point " << ratio << ", expected 1 within " << kPointLimitTolerance << '\n';
                ok = false;
            }
            // Monotone approach, so a renderer that merely happened to land on 1 at one distance cannot pass.
            if (!(ratioError <= previousRatioError)) {
                std::cerr << "integrator_validate: FAILED inverse-square convergence at " << row.name
                           << " -- |E/E_point - 1| rose to " << ratioError << " from " << previousRatioError << '\n';
                ok = false;
            }
            previousRatioError = ratioError;
        } else if (ratioError >= kMinNearFieldDeviation) {
            sawNearFieldDeviation = true;
        }
    }
    // Keeps the guard honest: if no row ever deviated from the point model, the far-field rows proved nothing.
    if (!sawNearFieldDeviation) {
        std::cerr << "integrator_validate: FAILED inverse-square conditioning -- no near-field row deviated from the point-source model by " << kMinNearFieldDeviation << ", so the far-field rows are vacuous\n";
        ok = false;
    }
    finish(ctx, ok, "quad_light_inverse_square failed; see the rows above");
    return;
}

// --- Ray-traced ambient occlusion (path_tracer.cpp's AO lane) ---------------------------------------

// Wall distance for the AO checks, 10x makeCornerScene's default: holds AO(c) curvature bias to 3.3e-5 and keeps c > 1 at every pixel.
constexpr float kAoWallDistance = 10.0F;
// One rho draw per sample, so error falls as 1/sqrt(N); this count puts the 6-sigma tolerance 8x below every curve the sweep excludes.
constexpr int kAoSamplesPerPixel = 1024;
// Every pixel, not centreMean's 4x4: AO ignores view direction, so the whole frame gives the same N for a sixteenth of the paths.
constexpr float kAoMeasuredPixels = static_cast<float>(kImageSize) * static_cast<float>(kImageSize);

// Closed-form cosine-weighted obscurance for the corner scene, in double: three O(1) terms whose sum vanishes as (1-c)^(7/2).
float analyticAmbientOcclusion(float c) {
    if (c >= 1.0F) {
        return 1.0F;
    }
    const double x = static_cast<double>(c);
    const double s = std::sqrt(1.0 - (x * x));
    const double deficit =
        ((1.0 - (2.0 * x * x)) * std::acos(x)) + (5.0 * x * s) - (4.0 * x * std::log((1.0 + s) / x));
    return static_cast<float>(1.0 - (deficit / std::numbers::pi));
}

// Standard error on the mean at ~6 sigma, from the estimator's own statistics, conservative by the Bhatia-Davis bound
float aoTolerance(float expected) {
    const float n = static_cast<float>(kAoSamplesPerPixel) * kAoMeasuredPixels;
    return 6.0F * std::sqrt(expected * (1.0F - expected) / n);
}

// One AO render over the whole frame at maxBounces=0, since AO is written at bounce 0 and reads no material at all.
bool checkAoRow(const char* label, const TestScene& scene, const EnvironmentMap& env,
                 float aoMaxDistance, float expected, EmbreeAccel& accel,
                 pathtracer::scene::ThreadPool& pool) {
    PathTraceSettings settings = makeSettings(/*maxBounces=*/0, 999);
    settings.samplesPerPixel = kAoSamplesPerPixel;
    settings.aoMaxDistance = aoMaxDistance;
    const pathtracer::gfx::HdrImage& ao = renderPass(scene, env, settings, accel, pool, true).ao;
    const float measured = regionMean(ao, 0, 0, kImageSize, kImageSize).x;
    const float tolerance = aoTolerance(expected);
    std::cout << "  " << label;
    for (std::size_t pad = std::string(label).size(); pad < 38; ++pad) {
        std::cout << ' ';
    }
    std::cout << "measured " << measured << "   analytic " << expected << "   tolerance " << tolerance
              << '\n';
    if (std::fabs(measured - expected) > tolerance) {
        std::cerr << "integrator_validate: FAILED ambient occlusion at " << label << " -- measured "
                  << measured << " vs analytic " << expected << " (tolerance " << tolerance << ")\n";
        return false;
    }
    return true;
}

struct AoCase {
    const char* name;
    float c;  // d / aoMaxDistance
};

// Ray-traced AO against its closed form: an unoccluded plane, then a sweep over c excluding three wrong integrators
PT_CHECK(ambient_occlusion_analytic, Slow, Statistical) {
    std::cout << "integrator_validate: ambient occlusion vs analytic cosine-weighted visibility\n";
    const EnvironmentMap env = makeUniformEnvironment();
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    const TestScene openScene = makeQuadScene(1.0F);
    std::optional<EmbreeAccel> openAccel = EmbreeAccel::build(openScene.worldTriangles);
    if (!openAccel.has_value()) {
        std::cerr << "integrator_validate: FAILED to build Embree scene for the unoccluded AO plane\n";
        finish(ctx, false, "ambient_occlusion_analytic failed; see the rows above");
        return;
    }
    // The first row initialises the accumulator; later rows keep `check(...) && ok`, so no row is ever short-circuited away.
    bool ok = checkAoRow("unoccluded plane", openScene, env, kAoWallDistance, 1.0F, *openAccel, pool);

    const TestScene corner = makeCornerScene(1.0F, kAoWallDistance);
    std::optional<EmbreeAccel> cornerAccel = EmbreeAccel::build(corner.worldTriangles);
    if (!cornerAccel.has_value()) {
        std::cerr << "integrator_validate: FAILED to build Embree scene for the AO corner\n";
        finish(ctx, false, "ambient_occlusion_analytic failed; see the rows above");
        return;
    }
    const std::array<AoCase, 4> sweep{{{"corner c=0.05 (half-space limit)", 0.05F},
                                        {"corner c=0.15", 0.15F},
                                        {"corner c=0.25", 0.25F},
                                        {"corner c=0.50", 0.5F}}};
    for (const AoCase& testCase : sweep) {
        ok = checkAoRow(testCase.name, corner, env, kAoWallDistance / testCase.c,
                         analyticAmbientOcclusion(testCase.c), *cornerAccel, pool) &&
             ok;
    }
    finish(ctx, ok, "ambient_occlusion_analytic failed; see the rows above");
    return;
}

// aoMaxDistance bracketed on one geometry: inside, the wall darkens by 11 tolerances; outside, every ray escapes and the lane reads 1.0.
PT_CHECK(ambient_occlusion_distance_bound, Slow, Statistical) {
    std::cout << "integrator_validate: ambient occlusion respects aoMaxDistance\n";
    const EnvironmentMap env = makeUniformEnvironment();
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());

    const TestScene corner = makeCornerScene(1.0F, kAoWallDistance);
    std::optional<EmbreeAccel> accel = EmbreeAccel::build(corner.worldTriangles);
    if (!accel.has_value()) {
        std::cerr << "integrator_validate: FAILED to build Embree scene for the AO distance bound\n";
        finish(ctx, false, "ambient_occlusion_distance_bound failed; see the rows above");
        return;
    }
    const std::array<AoCase, 2> cases{{{"occluder inside range (c=0.50)", 0.5F},
                                        {"occluder just outside range (c=1.10)", 1.1F}}};
    bool ok = true;
    for (const AoCase& testCase : cases) {
        ok = checkAoRow(testCase.name, corner, env, kAoWallDistance / testCase.c,
                         analyticAmbientOcclusion(testCase.c), *accel, pool) &&
             ok;
    }
    finish(ctx, ok, "ambient_occlusion_distance_bound failed; see the rows above");
    return;
}

}  // namespace

// Alpha is filtered primary coverage: the half-plane x < 0 images onto columns left of the centre line, the frame's x = 8.
PT_CHECK(alpha_is_filtered_primary_coverage, Fast, Exact) {
    // A pixel's support reaches kFilterRadius = 1.5 px from its centre c + 0.5: columns 0-6 see only the plane, 9-15 only the sky.
    constexpr int kLastOpaque = 6;
    constexpr int kFirstClear = 9;
    TestScene scene = makeTwoInstanceScene(1.0F);
    scene.worldTriangles.resize(2);
    scene.shadingTriangles.resize(2);
    scene.instances.resize(1);
    std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
    const EnvironmentMap env = makeUniformEnvironment();
    PathTraceSettings settings = makeSettings(1, 1);
    settings.samplesPerPixel = 16;
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    const pathtracer::scene::PathTraceResult hidden = renderPass(scene, env, settings, *accel, pool, /*showSky=*/false);
    const pathtracer::scene::PathTraceResult shown = renderPass(scene, env, settings, *accel, pool, /*showSky=*/true);
    // The lane is (sum w) * fl(1 / sum w), two roundings: an all-opaque support reads within 2u = FLT_EPSILON of 1.
    const auto opaque = [](float alpha) { return std::abs(alpha - 1.0F) <= std::numeric_limits<float>::epsilon(); };
    bool interior = true;
    bool edge = true;
    bool monotone = true;
    bool premultiplied = true;
    bool opaqueSky = true;
    for (int y = 0; y < kImageSize; ++y) {
        for (int x = 0; x < kImageSize; ++x) {
            const std::size_t pixel = (static_cast<std::size_t>(y) * kImageSize) + static_cast<std::size_t>(x);
            const float alpha = hidden.alpha.texels[pixel];
            interior = interior && (x > kLastOpaque || opaque(alpha)) && (x < kFirstClear || alpha == 0.0F);
            edge = edge && (x <= kLastOpaque || x >= kFirstClear || (alpha > 0.0F && !opaque(alpha)));
            monotone = monotone && (x == 0 || alpha <= hidden.alpha.texels[pixel - 1] || opaque(alpha));
            premultiplied = premultiplied && (alpha > 0.0F || hidden.beauty.rgb(pixel) == glm::vec3(0.0F));
            opaqueSky = opaqueSky && opaque(shown.alpha.texels[pixel]);
        }
    }
    ctx.plan(5);
    PT_EXPECT(ctx, interior, "a pixel whose whole filter support sees only plane or only sky is not exactly 1 or 0");
    PT_EXPECT(ctx, edge, "a pixel whose support straddles the edge is not strictly between 0 and 1");
    PT_EXPECT(ctx, monotone, "coverage rises moving off the half-plane");
    PT_EXPECT(ctx, premultiplied, "a transparent pixel carries radiance with the sky hidden");
    PT_EXPECT(ctx, opaqueSky, "a shown sky is not opaque");
}

// A checker minified 64:1 reads its mean wherever bounce 0 filters: Beauty by each sample's stratum, the G-buffer's Albedo by its pixel.
PT_CHECK(primary_hits_filter_minified_textures, Fast, Exact) {
    constexpr int kChecker = 64;
    constexpr float kTexelsPerPixel = 64.0F;
    std::vector<float> checker;
    for (int y = 0; y < kChecker; ++y) {
        for (int x = 0; x < kChecker; ++x) {
            const auto texel = static_cast<float>((x + y) % 2);
            checker.insert(checker.end(), {texel, texel, texel});
        }
    }
    const Camera camera = makeCamera();
    const Camera::ViewBasis basis = camera.viewBasis(1.0F);
    // The z = 0 plane faces the camera along its axis: one pixel spans 2 z tan(half-angle) / kImageSize metres, the narrower axis set.
    const float pixelMetres = 2.0F * camera.position().z * std::min(basis.halfWidth, basis.halfHeight) / static_cast<float>(kImageSize);
    const float uvPerMetre = kTexelsPerPixel / (static_cast<float>(kChecker) * pixelMetres);
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    const glm::vec4 tangent(1.0F, 0.0F, 0.0F, 1.0F);
    // Half a texel over, so a pixel centre, a whole number of texels plus a half from the axis, lands on a texel centre: 0 or 1 to a point.
    const glm::vec2 halfTexel(0.5F / static_cast<float>(kChecker));
    const auto vertex = [&](float x, float y) {
        return ShadingVertex{glm::vec3(x, y, 0.0F), normal, (glm::vec2(x, y) * uvPerMetre) + halfTexel, tangent};
    };
    // A metre each way covers the frame and every filter tap past it, so no pixel sees the sky.
    const ShadingVertex v0 = vertex(-1.0F, -1.0F);
    const ShadingVertex v1 = vertex(1.0F, -1.0F);
    const ShadingVertex v2 = vertex(1.0F, 1.0F);
    const ShadingVertex v3 = vertex(-1.0F, 1.0F);
    TestScene scene;
    scene.worldTriangles = {Triangle{v0.position, v1.position, v2.position}, Triangle{v0.position, v2.position, v3.position}};
    scene.shadingTriangles = {ShadingTriangle{v0, v1, v2, 0}, ShadingTriangle{v0, v2, v3, 0}};
    const TextureHandle checkerTexture = tools::fixtures::makeTexture(kChecker, kChecker, pathtracer::gfx::kRgbChannels, checker);
    // Emission alone, so Beauty is the filtered emission_color lookup itself, splatted; the G-buffer's Albedo reads base_color.
    Material material = makeMaterial(1.0F);
    material.baseWeight = 0.0F;
    material.specularWeight = 0.0F;
    material.emissionLuminance = 1.0F;
    material.emissionColor = checkerTexture;
    scene.instances = {MeshInstance{material, glm::mat4(1.0F), ""}};
    TestScene albedoScene = scene;
    albedoScene.instances[0].material = makeMaterial(1.0F);
    albedoScene.instances[0].material.baseColor = checkerTexture;
    // The same checker on a lobe-shaping input, which the classification point-samples: each pixel reads one texel, 0 or 1.
    std::vector<float> scalarChecker;
    for (std::size_t texel = 0; texel < checker.size(); texel += pathtracer::gfx::kRgbChannels) {
        scalarChecker.push_back(checker[texel]);
    }
    albedoScene.instances[0].material.specularRoughness =
        tools::fixtures::makeTexture(kChecker, kChecker, pathtracer::gfx::kScalarChannels, std::move(scalarChecker));
    std::optional<EmbreeAccel> accel = EmbreeAccel::build(scene.worldTriangles);
    const EnvironmentMap env = makeUniformEnvironment();
    PathTraceSettings settings = makeSettings(0, 1);
    settings.samplesPerPixel = 16;
    pathtracer::scene::ThreadPool& pool = sharedPool(ctx.threads());
    const pathtracer::scene::PathTraceResult traced = renderPass(scene, env, settings, *accel, pool, /*showSky=*/false);
    pathtracer::scene::GBuffer gbuffer;
    pathtracer::scene::renderGBuffer(camera, camera, *accel, albedoScene.shadingTriangles, albedoScene.instances, settings.lookaheadDistance,
                                     pathtracer::scene::computeInstanceBounds(scene.shadingTriangles, 1), kImageSize, kImageSize, pool,
                                     gbuffer);
    // 16 texels per stratum select levels 3-4, every one exactly 0.5; the blend and the splat's (sum w) fl(1/sum w) round a few ulp.
    const float bound = 8.0F * std::numeric_limits<float>::epsilon() * 0.5F;
    float worstBeauty = 0.0F;
    float worstAlbedo = 0.0F;
    int unfilteredRoughness = 0;
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(kImageSize) * kImageSize; ++pixel) {
        const glm::vec3 beauty = traced.beauty.rgb(pixel);
        const glm::vec3 albedo = gbuffer.albedo.rgb(pixel);
        worstBeauty = std::max({worstBeauty, std::abs(beauty.r - 0.5F), std::abs(beauty.g - 0.5F), std::abs(beauty.b - 0.5F)});
        worstAlbedo = std::max({worstAlbedo, std::abs(albedo.r - 0.5F), std::abs(albedo.g - 0.5F), std::abs(albedo.b - 0.5F)});
        // A point lookup near a texel centre reads within a sliver of 0 or 1; the footprint's mean is 0.5.
        unfilteredRoughness += std::abs(gbuffer.roughness.texels[pixel] - 0.5F) > 0.25F ? 1 : 0;
    }
    ctx.plan(3);
    char detail[160];
    // Point lookups of a 0/1 checker would leave 16 samples' mean about 1/8 from 0.5: these bounds hold only if bounce 0 filtered.
    std::snprintf(detail, sizeof(detail), "Beauty is %.3g from the checker's mean 0.5, over %.3g", static_cast<double>(worstBeauty),
                  static_cast<double>(bound));
    PT_EXPECT(ctx, worstBeauty <= bound, detail);
    std::snprintf(detail, sizeof(detail), "Albedo is %.3g from the checker's mean 0.5, over %.3g", static_cast<double>(worstAlbedo),
                  static_cast<double>(bound));
    PT_EXPECT(ctx, worstAlbedo <= bound, detail);
    std::snprintf(detail, sizeof(detail), "%d of %d pixels read specular_roughness as a single texel; a lobe-shaping input was filtered",
                  unfilteredRoughness, kImageSize * kImageSize);
    PT_EXPECT(ctx, unfilteredRoughness == kImageSize * kImageSize, detail);
}

// Radiance only inside one end column, 1/8 px clear of its edges: the column across the seam is lit if, and only if, the filter wraps.
PT_CHECK(omnidirectional_film_filter_reaches_across_the_seam, Fast, Exact) {
    constexpr int kWidth = 16;
    constexpr int kHeight = 8;
    // Four env texels per image column: the band is the middle two, so its bilinear support is the column's interior 3/4.
    constexpr int kTexelsPerColumn = 4;
    constexpr int kEnvWidth = kTexelsPerColumn * kWidth;
    constexpr int kEnvHeight = 4;
    std::optional<EmbreeAccel> accel = EmbreeAccel::build({});
    pathtracer::scene::ThreadPool pool;
    PathTraceSettings settings;
    settings.samplesPerPixel = 16;
    // Yaw 180 faces +Z with -X on the right, the map's own frame, so the image's u is the map's u.
    const Camera camera(glm::vec3(0.0F), glm::vec3(0.0F, 180.0F, 0.0F), Camera::FilmBack{36.0F, 24.0F}, kFocalLengthMm, 0.01F, 1000.0F,
                        2.8F, 1.0F / 125.0F, 100.0F, pathtracer::scene::Lens{pathtracer::scene::LensProjection::Omnidirectional});
    ctx.plan(6);
    for (const int litColumn : {kWidth - 1, 0}) {
        std::vector<float> rgb(static_cast<std::size_t>(kEnvWidth) * kEnvHeight * pathtracer::gfx::kRgbChannels, 0.0F);
        for (int y = 0; y < kEnvHeight; ++y) {
            for (int x = (litColumn * kTexelsPerColumn) + 1; x < ((litColumn + 1) * kTexelsPerColumn) - 1; ++x) {
                const std::size_t texel = ((static_cast<std::size_t>(y) * kEnvWidth) + static_cast<std::size_t>(x)) * 3;
                rgb[texel] = rgb[texel + 1] = rgb[texel + 2] = 1.0F;
            }
        }
        const EnvironmentMap env = tools::fixtures::makeEnvironment(kEnvWidth, kEnvHeight, rgb);
        const std::atomic<std::uint64_t> generation{1};
        pathtracer::scene::PathTraceResult result = pathtracer::scene::makePathTraceResult(kWidth, kHeight);
        const std::vector<pathtracer::scene::QuadLight> noQuads;
        const pathtracer::scene::LightSet lights(&env, /*envRotationDegrees=*/glm::vec3(0.0F), /*envExposure=*/1.0F, noQuads);
        pathtracer::debug::PassStats stats;
        pathtracer::scene::renderPathTraced(camera, *accel, {}, {}, {}, lights, kWidth, kHeight, /*showSky=*/true, settings, 7U,
                                            /*sampleBase=*/0, settings.samplesPerPixel, generation, /*requestedGeneration=*/1U, pool,
                                            stats, result);
        // The column across the seam, within the filter's 1.5 px of the band, and the next one in, beyond it.
        const int seamSide = litColumn == 0 ? kWidth - 1 : 0;
        const int inward = litColumn == 0 ? kWidth - 2 : 1;
        bool lit = true;
        bool dark = true;
        for (int y = 0; y < kHeight; ++y) {
            lit = lit && result.beauty.rgb((static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(seamSide)).x > 0.0F;
            dark = dark && result.beauty.rgb((static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(inward)).x == 0.0F;
        }
        PT_EXPECT(ctx, lit, "column " + std::to_string(seamSide) + " got nothing across the seam from " + std::to_string(litColumn));
        PT_EXPECT(ctx, dark, "column " + std::to_string(inward) + ", past the filter's reach of the band, was lit");
        PT_EXPECT(ctx, result.wrapsHorizontally, "the result does not record that its lens wraps, so its filters and display would clamp");
    }
}

PT_CHECK_MAIN("integrator")
