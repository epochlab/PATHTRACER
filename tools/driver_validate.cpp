// Correctness gate for PathTraceDriver, driven only through its public API; also pins the request invariant driverLoop relies on.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <glm/glm.hpp>

#include "check.h"
#include "fixtures.h"
#include "stats.h"
#include "pathtracer/debug/aov_filters.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/environment_map.h"
#include "pathtracer/scene/light.h"
#include "pathtracer/scene/path_trace_driver.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/shading_scene.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

using pathtracer::scene::Camera;
using pathtracer::scene::EmbreeAccel;
using pathtracer::scene::EnvironmentMap;
using pathtracer::scene::MeshInstance;
using pathtracer::scene::PathTraceDriver;
using pathtracer::scene::PathTraceResult;
using pathtracer::scene::PathTraceSettings;
using pathtracer::scene::AreaLight;
using pathtracer::scene::ShadingTriangle;
using pathtracer::scene::ShadingVertex;
using pathtracer::scene::Triangle;

// Small enough that a pass is milliseconds: this measures accumulation arithmetic and handshakes, so the image need only be non-trivial.
constexpr int kImageSize = 8;
constexpr float kQuadExtent = 50.0F;
// The camera sits at z = 5 looking down -z; the wall stands inside its field of view, right of centre.
constexpr float kFloorZ = -3.0F;
constexpr float kWallX = 1.0F;
constexpr float kWallTopZ = 3.0F;

// A floor and a transmissive wall, kept local because this scene must sit at a permanent address that outlives the driver.
struct TestScene {
    std::vector<Triangle> worldTriangles;
    std::vector<ShadingTriangle> shadingTriangles;
    std::vector<MeshInstance> instances;
    std::vector<int> instanceLightIndex;
    std::vector<AreaLight> lights;
};

// Corners wound counter-clockwise about `normal`, split along the 0-2 diagonal; the tangent is the first edge, so it lies in the plane.
void pushQuad(TestScene& scene, const std::array<glm::vec3, 4>& corners, glm::vec3 normal, int instance) {
    const std::array<int, 6> order = {0, 1, 2, 0, 2, 3};
    const glm::vec4 tangent(glm::normalize(corners[1] - corners[0]), 1.0F);
    for (int t = 0; t < 2; ++t) {
        Triangle world{};
        ShadingTriangle shading{};
        for (int v = 0; v < 3; ++v) {
            const glm::vec3 p = corners[order[(t * 3) + v]];
            (v == 0 ? world.v0 : v == 1 ? world.v1 : world.v2) = p;
            ShadingVertex vertex{};
            vertex.position = p;
            vertex.normal = normal;
            vertex.tangent = tangent;
            vertex.uv = glm::vec2(0.0F);
            (v == 0 ? shading.v0 : v == 1 ? shading.v1 : shading.v2) = vertex;
        }
        shading.instanceIndex = instance;
        scene.worldTriangles.push_back(world);
        scene.shadingTriangles.push_back(shading);
    }
}

PathTraceSettings makeSettings() {
    PathTraceSettings settings{};
    settings.samplesPerPixel = 1;  // one sample per pass: the progressive configuration the driver exists for
    settings.replicates = 4;  // interleaved, so every check below crosses replicate boundaries rather than one sequence
    settings.maxBounces = 1;
    settings.russianRouletteStartBounce = 8;
    settings.denoiseFeatures = true;  // the feature lanes accumulate too, so the lane checks below cover them
    return settings;
}

// Everything the driver holds by reference must outlive it and never move, so the fixture is heap-allocated before the driver exists.
struct DriverFixture {
    TestScene scene;
    std::optional<EmbreeAccel> accel;
    EnvironmentMap environment = tools::fixtures::makeUniformEnvironment();
    pathtracer::scene::VolumeSet volumes;  // none: these checks are of the driver, not of media
    std::optional<PathTraceDriver> driver;
    pathtracer::scene::ThreadPool pool;

    [[nodiscard]] bool valid() const { return accel.has_value() && driver.has_value(); }
};

std::unique_ptr<DriverFixture> makeFixture() {
    auto fixture = std::make_unique<DriverFixture>();
    TestScene& scene = fixture->scene;
    // Every image must vary across passes or the running-mean check is blind to it: the floor ends mid-frame, so Alpha varies there.
    pushQuad(scene,
             {glm::vec3(-kQuadExtent, -kQuadExtent, kFloorZ), glm::vec3(kQuadExtent, -kQuadExtent, kFloorZ),
              glm::vec3(kQuadExtent, 0.0F, kFloorZ), glm::vec3(-kQuadExtent, 0.0F, kFloorZ)},
             glm::vec3(0.0F, 0.0F, 1.0F), 0);
    pushQuad(scene,
             {glm::vec3(kWallX, -kQuadExtent, kFloorZ), glm::vec3(kWallX, -kQuadExtent, kWallTopZ),
              glm::vec3(kWallX, kQuadExtent, kWallTopZ), glm::vec3(kWallX, kQuadExtent, kFloorZ)},
             glm::vec3(-1.0F, 0.0F, 0.0F), 1);
    for (int i = 0; i < 2; ++i) {
        scene.instances.push_back(MeshInstance{tools::fixtures::makeMaterial(1.0F), glm::mat4(1.0F), ""});
    }
    scene.instances[1].material.transmissionWeight = 1.0F;
    scene.instanceLightIndex.assign(scene.instances.size(), -1);
    fixture->accel = EmbreeAccel::build(fixture->scene.worldTriangles);
    if (!fixture->accel.has_value()) {
        return fixture;
    }
    fixture->driver.emplace(*fixture->accel, fixture->scene.shadingTriangles, fixture->scene.instances,
                             fixture->scene.instanceLightIndex, fixture->environment,
                             fixture->scene.lights, fixture->volumes);
    return fixture;
}

Camera makeCamera() {
    return Camera(glm::vec3(0.0F, 0.0F, 5.0F), glm::vec3(0.0F), Camera::FilmBack{36.0F, 24.0F}, 50.0F, 0.01F, 1000.0F,
                   2.8F, 1.0F / 125.0F, 100.0F);
}

PathTraceDriver::Request makeRequest(int maxSamples, const Camera& camera) {
    PathTraceDriver::Request request{camera, 0, 0, glm::vec3(0.0F), true, true, 1.0F, makeSettings(), 0};
    request.width = kImageSize;
    request.height = kImageSize;
    request.envRotationDegrees = glm::vec3(0.0F);
    request.showSky = false;
    request.envLightEnabled = true;
    request.envExposure = 1.0F;
    request.settings = makeSettings();
    request.maxSamples = maxSamples;
    return request;
}

// Polls until the predicate holds or the deadline passes. The deadline is a HANG DETECTOR, never a performance assertion.
template <typename Predicate>
bool waitFor(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

// Polls until a result of `generation` averages >= `samples` passes, nullptr on timeout; keyed on fields published with its images.
std::shared_ptr<const PathTraceResult> waitForPublished(const PathTraceDriver& driver, std::uint64_t generation,
                                                        int samples) {
    std::shared_ptr<const PathTraceResult> result;
    const bool published = waitFor([&] {
        result = driver.latestResult();
        return result != nullptr && result->generation == generation && result->samples >= samples;
    });
    return published ? result : nullptr;
}

// Every image the driver averages, index-aligned with accumulateMean's lists; replicateLuminance is not one, see below.
struct Lane {
    pathtracer::gfx::HdrImage PathTraceResult::*image;
    const char* name;
};
constexpr std::array<Lane, 14> kLanes{{{&PathTraceResult::beauty, "beauty"},
                                      {&PathTraceResult::bounceHeatmap, "bounceHeatmap"},
                                      {&PathTraceResult::ao, "ao"},
                                      {&PathTraceResult::directDiffuse, "directDiffuse"},
                                      {&PathTraceResult::indirectDiffuse, "indirectDiffuse"},
                                      {&PathTraceResult::directSpecular, "directSpecular"},
                                      {&PathTraceResult::indirectSpecular, "indirectSpecular"},
                                      {&PathTraceResult::refraction, "refraction"},
                                      {&PathTraceResult::fresnel, "fresnel"},
                                      {&PathTraceResult::alpha, "alpha"},
                                      {&PathTraceResult::denoiseAlbedo, "denoiseAlbedo"},
                                      {&PathTraceResult::denoiseNormal, "denoiseNormal"},
                                      {&PathTraceResult::shadowLost, "shadowLost"},
                                      {&PathTraceResult::shadowUnoccluded, "shadowUnoccluded"}}};

// Per float: the batch mean rounded once to float and the bound on the driver's distance from it; per lane, whether any float moved.
struct OracleMean {
    std::array<std::vector<float>, kLanes.size()> mean;
    std::array<std::vector<double>, kLanes.size()> tolerance;
    std::array<bool, kLanes.size()> varies{};
    // HeadlessRenderer's calling pattern: the shared kernels folded in place over whole images, the first pass kept as drawn.
    std::array<std::vector<float>, kLanes.size()> folded;
    std::vector<float> foldedReplicates;
};

// Renders the driver's passes synchronously: the exact batch mean m_k in double, beside the float32 running mean's error bound E_k.
OracleMean oracleBatchMean(DriverFixture& fixture, const Camera& camera, int passes, std::uint32_t scrambleSeed) {
    constexpr double kUnitRoundoff = std::numeric_limits<float>::epsilon() / 2.0;
    constexpr double kGamma4 = (4.0 * kUnitRoundoff) / (1.0 - (4.0 * kUnitRoundoff));
    const pathtracer::scene::LightSet lights(&fixture.environment, glm::vec3(0.0F), 1.0F, fixture.scene.lights);
    const PathTraceSettings settings = makeSettings();
    PathTraceResult pass = pathtracer::scene::makePathTraceResult(kImageSize, kImageSize, settings.replicates);
    std::array<std::vector<double>, kLanes.size()> sum;
    OracleMean oracle;
    for (std::size_t lane = 0; lane < kLanes.size(); ++lane) {
        const std::size_t floats = (pass.*kLanes[lane].image).texels.size();
        sum[lane].assign(floats, 0.0);
        oracle.tolerance[lane].assign(floats, 0.0);
    }
    oracle.foldedReplicates.assign(pass.replicateLuminance.size(), 0.0F);
    const std::atomic<std::uint64_t> generation{scrambleSeed};
    pathtracer::debug::PassStats stats;
    for (int p = 0; p < passes; ++p) {
        stats.reset();
        pathtracer::scene::renderPathTraced(camera, *fixture.accel, fixture.scene.shadingTriangles,
                                         fixture.scene.instances, fixture.scene.instanceLightIndex, lights, pathtracer::scene::VolumeSet{},
                                         kImageSize, kImageSize, pathtracer::scene::allTexels(kImageSize, kImageSize),
                                         /*showSky=*/false, settings, scrambleSeed, /*sampleBase=*/p,
                                         /*sampleCount=*/passes, generation, scrambleSeed, fixture.pool, stats,
                                         pass);
        const double k = p + 1;
        const float invN = 1.0F / static_cast<float>(p + 1);
        const int count = (p / settings.replicates) + 1;
        pathtracer::scene::foldReplicateLuminance(oracle.foldedReplicates.data(), pass.beauty.texels.data(),
                                                  static_cast<std::size_t>(pass.beauty.channels), oracle.foldedReplicates.data(),
                                                  static_cast<std::size_t>(kImageSize) * kImageSize, settings.replicates,
                                                  p % settings.replicates, 1.0F / static_cast<float>(count));
        for (std::size_t lane = 0; lane < kLanes.size(); ++lane) {
            const std::vector<float>& drawn = (pass.*kLanes[lane].image).texels;
            if (p == 0) {
                oracle.folded[lane] = drawn;
            } else {
                std::vector<float>& mean = oracle.folded[lane];
                pathtracer::scene::foldRunningMean(mean.data(), drawn.data(), mean.data(), mean.size(), invN);
            }
        }
        for (std::size_t lane = 0; lane < kLanes.size(); ++lane) {
            const std::vector<float>& x = (pass.*kLanes[lane].image).texels;
            for (std::size_t i = 0; i < x.size(); ++i) {
                const double previousMean = p == 0 ? 0.0 : sum[lane][i] / (k - 1.0);
                sum[lane][i] += static_cast<double>(x[i]);
                // E_1 = 0: the driver publishes its first pass unaltered.
                if (p > 0) {
                    oracle.varies[lane] = oracle.varies[lane] || static_cast<double>(x[i]) != previousMean;
                    const double e = oracle.tolerance[lane][i];
                    oracle.tolerance[lane][i] = ((1.0 - (1.0 / k)) * (1.0 + kUnitRoundoff) * e) +
                                                (kGamma4 * (std::fabs(x[i] - previousMean) + e) / k) +
                                                (kUnitRoundoff * std::fabs(sum[lane][i] / k));
                }
            }
        }
    }
    for (std::size_t lane = 0; lane < kLanes.size(); ++lane) {
        oracle.mean[lane].resize(sum[lane].size());
        for (std::size_t i = 0; i < sum[lane].size(); ++i) {
            const double mean = sum[lane][i] / static_cast<double>(passes);
            oracle.mean[lane][i] = static_cast<float>(mean);
            // Plus the oracle's own single rounding of the exact mean to float.
            oracle.tolerance[lane][i] += kUnitRoundoff * std::fabs(mean);
        }
    }
    return oracle;
}

// --- Checks ------------------------------------------------------------------------------------------------------

// A task that throws must not strand parallelFor on its worker count. The pool stays usable afterwards, the dispatch being what failed.
PT_CHECK(thread_pool_rethrows_and_stays_usable, Fast, Exact) {
    pathtracer::scene::ThreadPool pool;
    ctx.plan(3);

    bool rethrown = false;
    try {
        // Thrown from one index only: the others still run, and the hang this guards against was the decrement being skipped.
        pool.parallelFor(256, [](int index) {
            if (index == 101) {
                throw std::runtime_error("task failure");
            }
        });
    } catch (const std::runtime_error&) {
        rethrown = true;
    }
    PT_EXPECT(ctx, rethrown, "parallelFor swallowed an exception escaping fn, or never returned at all");

    // The captured exception must not survive into the next dispatch, which would fail a caller that did nothing wrong.
    std::atomic<int> count{0};
    bool secondThrew = false;
    try {
        pool.parallelFor(256, [&count](int) { count.fetch_add(1, std::memory_order_relaxed); });
    } catch (...) {
        secondThrew = true;
    }
    PT_EXPECT(ctx, !secondThrew, "a later dispatch inherited the previous dispatch's exception");
    char detail[160];
    std::snprintf(detail, sizeof(detail), "the dispatch after a throwing one ran %d of 256 indices", count.load());
    PT_EXPECT(ctx, count.load() == 256, detail);
}

// The core accumulation property, on all ten images, under Higham's exact forward-error bound E_k for the running mean.
PT_CHECK(running_mean_matches_batch_mean, Slow, Exact) {
    constexpr int kPasses = 8;
    ctx.plan(3);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        for (int i = 0; i < 3; ++i) {
            PT_EXPECT(ctx, false, "scene/driver construction failed");
        }
        return;
    }
    const Camera camera = makeCamera();
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(kPasses, camera));
    const std::shared_ptr<const PathTraceResult> published =
        waitForPublished(*fixture->driver, generation, kPasses);
    if (published == nullptr) {
        for (int i = 0; i < 3; ++i) {
            PT_EXPECT(ctx, false, "driver never published its maxSamples cap");
        }
        return;
    }
    char countDetail[96];
    std::snprintf(countDetail, sizeof(countDetail), "published %d samples against a cap of %d", published->samples,
                  kPasses);
    PT_EXPECT(ctx, published->samples == kPasses, countDetail);

    // The driver's scramble seed is its generation (driverLoop).
    const OracleMean expected = oracleBatchMean(*fixture, camera, kPasses, static_cast<std::uint32_t>(generation));
    std::string constantLanes;
    for (std::size_t lane = 0; lane < kLanes.size(); ++lane) {
        if (!expected.varies[lane]) {
            constantLanes += std::string(" ") + kLanes[lane].name;
        }
    }
    const std::string coverageDetail = "lanes identical in every pass, so blind to accumulation:" + constantLanes;
    PT_EXPECT(ctx, constantLanes.empty(), coverageDetail);
    std::size_t violations = 0;
    std::size_t floats = 0;
    double worstRatio = 0.0;
    const char* worstLane = kLanes[0].name;
    for (std::size_t lane = 0; lane < kLanes.size(); ++lane) {
        const std::vector<float>& running = ((*published).*kLanes[lane].image).texels;
        for (std::size_t i = 0; i < running.size(); ++i) {
            const double error = std::fabs(static_cast<double>(running[i]) - expected.mean[lane][i]);
            const double tolerance = expected.tolerance[lane][i];
            // Negated so a NaN counts as a violation; a zero bound admits only an exact match.
            violations += !(error <= tolerance) ? 1 : 0;
            const double ratio =
                tolerance > 0.0 ? error / tolerance : (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
            if (ratio > worstRatio) {
                worstRatio = ratio;
                worstLane = kLanes[lane].name;
            }
        }
        floats += running.size();
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail),
                  "%zu of %zu floats over their forward-error bound; worst |running - batch|/bound = %.3e (%s)",
                  violations, floats, worstRatio, worstLane);
    PT_EXPECT(ctx, violations == 0, detail);
}

// Unification of the two accumulators: the driver's out-of-place row fold equals HeadlessRenderer's in-place one, every bit.
PT_CHECK(in_place_fold_is_bit_identical_to_driver, Slow, Exact) {
    constexpr int kPasses = 8;
    ctx.plan(2);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        for (int i = 0; i < 2; ++i) {
            PT_EXPECT(ctx, false, "scene/driver construction failed");
        }
        return;
    }
    const Camera camera = makeCamera();
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(kPasses, camera));
    const std::shared_ptr<const PathTraceResult> published = waitForPublished(*fixture->driver, generation, kPasses);
    if (published == nullptr) {
        for (int i = 0; i < 2; ++i) {
            PT_EXPECT(ctx, false, "driver never published its maxSamples cap");
        }
        return;
    }
    const OracleMean expected = oracleBatchMean(*fixture, camera, kPasses, static_cast<std::uint32_t>(generation));
    std::size_t differing = 0;
    std::size_t floats = 0;
    for (std::size_t lane = 0; lane < kLanes.size(); ++lane) {
        const std::vector<float>& image = ((*published).*kLanes[lane].image).texels;
        // Bitwise, not ==: a NaN or a signed zero that differs is a divergence too.
        for (std::size_t i = 0; i < image.size(); ++i) {
            differing += std::bit_cast<std::uint32_t>(image[i]) != std::bit_cast<std::uint32_t>(expected.folded[lane][i]) ? 1 : 0;
        }
        floats += image.size();
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu of %zu lane floats differ from the in-place fold", differing, floats);
    PT_EXPECT(ctx, differing == 0, detail);
    std::size_t differingReplicates = 0;
    for (std::size_t i = 0; i < published->replicateLuminance.size(); ++i) {
        differingReplicates += std::bit_cast<std::uint32_t>(published->replicateLuminance[i]) !=
                                       std::bit_cast<std::uint32_t>(expected.foldedReplicates[i])
                                   ? 1
                                   : 0;
    }
    std::snprintf(detail, sizeof(detail), "%zu of %zu replicate mean floats differ from the in-place fold", differingReplicates,
                  published->replicateLuminance.size());
    PT_EXPECT(ctx, differingReplicates == 0 && published->replicateLuminance.size() == expected.foldedReplicates.size(), detail);
}

// A driver holding sampleBase at 0 publishes the mean of N identical passes (sampler.h); pinned at maxSamples = 1, bit-identical.
PT_CHECK(first_pass_is_bit_identical_to_oracle, Slow, Exact) {
    ctx.plan(1);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    const Camera camera = makeCamera();
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(1, camera));
    const std::shared_ptr<const PathTraceResult> published = waitForPublished(*fixture->driver, generation, 1);
    if (published == nullptr) {
        PT_EXPECT(ctx, false, "driver never published its first pass");
        return;
    }
    const OracleMean expected = oracleBatchMean(*fixture, camera, 1, static_cast<std::uint32_t>(generation));
    std::size_t differing = 0;
    std::size_t floats = 0;
    for (std::size_t lane = 0; lane < kLanes.size(); ++lane) {
        const std::vector<float>& image = ((*published).*kLanes[lane].image).texels;
        for (std::size_t i = 0; i < image.size(); ++i) {
            differing += image[i] != expected.mean[lane][i] ? 1 : 0;
        }
        floats += image.size();
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu of %zu floats differ from a direct single-pass render", differing,
                  floats);
    PT_EXPECT(ctx, differing == 0, detail);
}

// The cap is a hard stop, not a target: accumulating past it moves a converged image under a caller that asked it to stop.
PT_CHECK(max_samples_cap_is_respected, Slow, Exact) {
    constexpr int kCap = 4;
    ctx.plan(1);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(kCap, makeCamera()));
    if (waitForPublished(*fixture->driver, generation, kCap) == nullptr) {
        PT_EXPECT(ctx, false, "driver never reached its maxSamples cap");
        return;
    }
    // Long enough that an uncapped driver would have run many further passes at this image size.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const int samples = fixture->driver->latestResult()->samples;
    char detail[128];
    std::snprintf(detail, sizeof(detail), "published %d samples against a cap of %d", samples, kCap);
    PT_EXPECT(ctx, samples == kCap, detail);
}

// Every texel is traced every pass without a stopping rule, so each published count is the pass count, carried across pool buffers.
PT_CHECK(texel_samples_count_every_pass, Slow, Exact) {
    constexpr int kPasses = 6;
    ctx.plan(1);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(kPasses, makeCamera()));
    const std::shared_ptr<const PathTraceResult> published = waitForPublished(*fixture->driver, generation, kPasses);
    const std::size_t off =
        published == nullptr ? 0 : static_cast<std::size_t>(std::count_if(published->texelSamples.begin(), published->texelSamples.end(),
                                                                           [](int count) { return count != kPasses; }));
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%zu texels count other than %d passes, or nothing was published", off, kPasses);
    PT_EXPECT(ctx, published != nullptr && published->texelSamples.size() == static_cast<std::size_t>(kImageSize * kImageSize) && off == 0,
              detail);
}

// The returned generation is the one its passes carry, so records are attributable; a request replaced before pickup never runs.
PT_CHECK(request_generation_labels_its_passes, Slow, Exact) {
    constexpr int kCap = 3;
    ctx.plan(2);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    const std::uint64_t superseded = fixture->driver->requestTrace(makeRequest(kCap, makeCamera()));
    const std::uint64_t latest = fixture->driver->requestTrace(makeRequest(kCap, makeCamera()));
    PT_EXPECT(ctx, latest == superseded + 1, "consecutive requests must return consecutive generations");
    const bool labelled = waitFor([&] {
        const pathtracer::debug::PassRecord pass = fixture->driver->lastPassRecord();
        return pass.generation == latest && pass.passIndex == kCap && !pass.cancelled;
    });
    PT_EXPECT(ctx, labelled, "the final pass never carried the generation requestTrace returned");
}

// Passes finish faster than frames on small images, so a per-frame read of the latest record drops some; the taken log must not.
PT_CHECK(every_pass_record_is_taken_once_in_order, Slow, Exact) {
    constexpr int kCap = 12;
    ctx.plan(2);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(kCap, makeCamera()));
    std::vector<int> indices;
    const bool converged = waitFor([&] {
        for (const pathtracer::debug::PassRecord& pass : fixture->driver->takePassRecords()) {
            if (pass.generation == generation && !pass.cancelled) {
                indices.push_back(pass.passIndex);
            }
        }
        return !indices.empty() && indices.back() == kCap;
    });
    std::vector<int> expected(kCap);
    std::iota(expected.begin(), expected.end(), 1);
    PT_EXPECT(ctx, converged && indices == expected, "taken pass indices must be exactly 1.." + std::to_string(kCap) + " in order");
    PT_EXPECT(ctx, fixture->driver->takePassRecords().empty(), "a converged driver must have nothing left to take");
}

// A new request must RESTART accumulation, never mix camera poses; pins the bump-implies-engaged-request invariant 16 suppressions rest on.
PT_CHECK(new_request_restarts_accumulation, Slow, Exact) {
    ctx.plan(2);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    const std::uint64_t first = fixture->driver->requestTrace(makeRequest(4, makeCamera()));
    if (waitForPublished(*fixture->driver, first, 4) == nullptr) {
        PT_EXPECT(ctx, false, "driver never reached the first cap");
        PT_EXPECT(ctx, false, "driver never reached the first cap");
        return;
    }

    // A different camera: a genuinely different image, which must not be averaged into the first.
    const Camera moved(glm::vec3(0.5F, 0.25F, 5.0F), glm::vec3(-5.0F, 10.0F, 0.0F), Camera::FilmBack{36.0F, 24.0F}, 50.0F, 0.01F,
                        1000.0F, 2.8F, 1.0F / 125.0F, 100.0F);
    const std::uint64_t second = fixture->driver->requestTrace(makeRequest(4, moved));
    // Keyed on the second generation: the first's result already holds 4 samples, so a count alone would pass unrestarted.
    const std::shared_ptr<const PathTraceResult> restarted = waitForPublished(*fixture->driver, second, 4);
    PT_EXPECT(ctx, restarted != nullptr, "driver never reached the second cap after the camera moved");
    // Restarted, the count returns to the new cap rather than continuing past it.
    const int samples = restarted != nullptr ? restarted->samples : 0;
    char detail[160];
    std::snprintf(detail, sizeof(detail), "published %d after restart, which must equal the new cap of 4", samples);
    PT_EXPECT(ctx, samples == 4, detail);
}

// Suspension parks the driver for a G-buffer AOV: the generation is the scramble seed, so bumping it to cancel would restart the image.
PT_CHECK(suspension_halts_and_resumes, Slow, Exact) {
    ctx.plan(3);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    // Uncapped, so only suspension can stop it.
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(0, makeCamera()));
    if (waitForPublished(*fixture->driver, generation, 2) == nullptr) {
        PT_EXPECT(ctx, false, "driver never started accumulating");
        PT_EXPECT(ctx, false, "driver never started accumulating");
        PT_EXPECT(ctx, false, "driver never started accumulating");
        return;
    }
    fixture->driver->setSuspended(true);
    // Let the pass already in flight finish and be accumulated before sampling what is parked.
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const std::shared_ptr<const PathTraceResult> parked = fixture->driver->latestResult();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const std::shared_ptr<const PathTraceResult> later = fixture->driver->latestResult();
    char haltDetail[160];
    std::snprintf(haltDetail, sizeof(haltDetail), "published %d samples while suspended, %d when parked",
                  later->samples, parked->samples);
    PT_EXPECT(ctx, later == parked, haltDetail);

    fixture->driver->setSuspended(false);
    const std::shared_ptr<const PathTraceResult> resumed =
        waitForPublished(*fixture->driver, generation, parked->samples + 1);
    PT_EXPECT(ctx, resumed != nullptr,
                  "driver did not resume accumulating the parked generation after suspension was lifted");
    // The count continues rather than returning to 1: anything else means the park discarded the mean it was holding.
    const int samples = resumed != nullptr ? resumed->samples : 0;
    char resumeDetail[160];
    std::snprintf(resumeDetail, sizeof(resumeDetail), "resumed at %d samples, which must exceed the %d parked",
                  samples, parked->samples);
    PT_EXPECT(ctx, samples > parked->samples, resumeDetail);
}

// Stronger: park and resume must leave images BIT-IDENTICAL to an uninterrupted run -- same Sobol sequence, no sample redrawn or skipped.
PT_CHECK(suspension_preserves_the_accumulation_exactly, Slow, Exact) {
    ctx.plan(1);
    constexpr int kSamples = 6;
    std::unique_ptr<DriverFixture> reference = makeFixture();
    std::unique_ptr<DriverFixture> parked = makeFixture();
    if (!reference->valid() || !parked->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    const std::uint64_t referenceGeneration = reference->driver->requestTrace(makeRequest(kSamples, makeCamera()));
    const std::shared_ptr<const PathTraceResult> uninterrupted =
        waitForPublished(*reference->driver, referenceGeneration, kSamples);

    // Same request on a second driver, parked partway: identical scene and generation 1 on each, so only the interruption differs.
    const std::uint64_t parkedGeneration = parked->driver->requestTrace(makeRequest(kSamples, makeCamera()));
    if (waitForPublished(*parked->driver, parkedGeneration, kSamples / 2) == nullptr || uninterrupted == nullptr) {
        PT_EXPECT(ctx, false, "a driver never reached its cap");
        return;
    }
    parked->driver->setSuspended(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    parked->driver->setSuspended(false);
    const std::shared_ptr<const PathTraceResult> continued =
        waitForPublished(*parked->driver, parkedGeneration, kSamples);
    if (continued == nullptr) {
        PT_EXPECT(ctx, false, "parked driver never reached its cap after resuming");
        return;
    }

    std::size_t differing = 0;
    const char* differingLane = "";
    for (const Lane& lane : kLanes) {
        const std::vector<float>& a = (uninterrupted.get()->*lane.image).texels;
        const std::vector<float>& b = (continued.get()->*lane.image).texels;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) {
                ++differing;
                differingLane = lane.name;
            }
        }
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu floats differ from the uninterrupted run (last in %s)", differing,
                  differing == 0 ? "none" : differingLane);
    PT_EXPECT(ctx, differing == 0, detail);
}

// Three pooled images, two driver-held and one for the reader: acquireFreeBuffer must STALL rather than recycle a held result.
PT_CHECK(published_results_are_not_overwritten, Slow, Exact) {
    ctx.plan(1);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(0, makeCamera()));
    // Pin a published result and snapshot it, then run many more passes: the mean moves every pass, so a recycled buffer would show it.
    const std::shared_ptr<const PathTraceResult> pinned = waitForPublished(*fixture->driver, generation, 1);
    if (pinned == nullptr) {
        PT_EXPECT(ctx, false, "driver published nothing");
        return;
    }
    const std::vector<float> snapshot = pinned->beauty.texels;
    const std::shared_ptr<const PathTraceResult> later =
        waitForPublished(*fixture->driver, generation, pinned->samples + 6);
    if (later == nullptr) {
        PT_EXPECT(ctx, false, "driver stalled while a single result was pinned");
        return;
    }
    std::size_t moved = 0;
    for (std::size_t i = 0; i < snapshot.size(); ++i) {
        moved += pinned->beauty.texels[i] != snapshot[i] ? 1 : 0;
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu of %zu floats in a pinned result changed while %d further passes ran",
                  moved, snapshot.size(), later->samples - pinned->samples);
    PT_EXPECT(ctx, moved == 0, detail);
}

// The over-range histogram is a parallel reduction with a serial definition, recomputed single-threaded from the published image.
PT_CHECK(over_range_stats_match_serial_scan, Slow, Exact) {
    ctx.plan(4);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        for (int i = 0; i < 4; ++i) {
            PT_EXPECT(ctx, false, "scene/driver construction failed");
        }
        return;
    }
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(2, makeCamera()));
    const std::shared_ptr<const PathTraceResult> published = waitForPublished(*fixture->driver, generation, 2);
    if (published == nullptr) {
        for (int i = 0; i < 4; ++i) {
            PT_EXPECT(ctx, false, "driver never reached its cap");
        }
        return;
    }

    const pathtracer::scene::OverRangeStats& stats = published->overRange;
    const std::size_t texels = static_cast<std::size_t>(kImageSize) * kImageSize;

    // aboveBin is a complementary CDF: bin 0 counts every texel and the last bin counts none.
    char zeroDetail[160];
    std::snprintf(zeroDetail, sizeof(zeroDetail), "aboveBin[0] = %u, must equal the texel count %zu",
                  stats.aboveBin[0], texels);
    PT_EXPECT(ctx, static_cast<std::size_t>(stats.aboveBin[0]) == texels, zeroDetail);
    PT_EXPECT(ctx, stats.aboveBin[pathtracer::scene::kOverRangeBinCount] == 0,
                  "the last complementary-CDF entry must be empty by construction");

    bool monotone = true;
    for (std::size_t b = 1; b < stats.aboveBin.size(); ++b) {
        monotone = monotone && stats.aboveBin[b] <= stats.aboveBin[b - 1];
    }
    PT_EXPECT(ctx, monotone, "a complementary CDF must be non-increasing in the bin index");

    float serialPeak = 0.0F;
    for (std::size_t t = 0; t < texels; ++t) {
        const glm::vec3 rgb = published->beauty.rgb(t);
        serialPeak = std::max(serialPeak, std::max(rgb.r, std::max(rgb.g, rgb.b)));
    }
    char peakDetail[176];
    std::snprintf(peakDetail, sizeof(peakDetail), "published rawPeak %.9g vs single-threaded scan %.9g",
                  static_cast<double>(stats.rawPeak), static_cast<double>(serialPeak));
    PT_EXPECT(ctx, stats.rawPeak == serialPeak, peakDetail);
}

// The chunking is a work split, so it must stay within its unit ceiling and never hand the pool fewer chunks than it can use.
PT_CHECK(chunk_length_bounds_the_unit_and_fills_the_pool, Fast, Exact) {
    using pathtracer::scene::kChunksPerThread;
    using pathtracer::scene::kPathTraceChunkTexels;
    using pathtracer::scene::pathTraceChunkTexels;
    constexpr std::array<int, 9> kTexels{0, 1, 8, 32, 96, 115, 9216, 1152 * 205, 2048 * 1024};
    constexpr std::array<unsigned int, 5> kThreads{1, 2, 8, 64, 4096};
    ctx.plan(static_cast<int>(kTexels.size() * kThreads.size() * 2));
    for (const int texels : kTexels) {
        for (const unsigned int threads : kThreads) {
            const int chunk = pathTraceChunkTexels(texels, threads);
            char detail[192];
            std::snprintf(detail, sizeof(detail), "%d texels on %u threads gave chunk %d, outside [1, %d]", texels, threads, chunk,
                          kPathTraceChunkTexels);
            PT_EXPECT(ctx, chunk >= 1 && chunk <= kPathTraceChunkTexels, detail);

            // Fewer chunks than the pool wants is justified only by the unit ceiling, or by a list shorter than the pool.
            const long long chunks = (static_cast<long long>(texels) + chunk - 1) / chunk;
            const long long wanted = static_cast<long long>(threads) * kChunksPerThread;
            std::snprintf(detail, sizeof(detail), "%d texels on %u threads split into %lld chunks of %d, short of %lld", texels, threads,
                          chunks, chunk, wanted);
            PT_EXPECT(ctx, chunks >= wanted || chunk == kPathTraceChunkTexels || texels < wanted, detail);
        }
    }
}

// Untraced texels are left as they were and traced ones match the full render: a subset pass is the full pass, restricted.
PT_CHECK(render_of_a_texel_subset_is_the_full_render_restricted, Slow, Exact) {
    ctx.plan(2);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene construction failed");
        return;
    }
    const Camera camera = makeCamera();
    const pathtracer::scene::LightSet lights(&fixture->environment, glm::vec3(0.0F), 1.0F, fixture->scene.lights);
    const std::atomic<std::uint64_t> generation{1};
    pathtracer::debug::PassStats stats;
    const auto renderOf = [&](std::span<const std::uint32_t> texels) {
        PathTraceResult out = pathtracer::scene::makePathTraceResult(kImageSize, kImageSize, makeSettings().replicates);
        // A sentinel no sample produces, so an untouched texel is told from a traced one.
        std::fill(out.beauty.texels.begin(), out.beauty.texels.end(), -1.0F);
        stats.reset();
        pathtracer::scene::renderPathTraced(camera, *fixture->accel, fixture->scene.shadingTriangles, fixture->scene.instances,
                                            fixture->scene.instanceLightIndex, lights, pathtracer::scene::VolumeSet{}, kImageSize, kImageSize,
                                            texels, /*showSky=*/true, makeSettings(), /*scrambleSeed=*/1U, /*sampleBase=*/0,
                                            /*sampleCount=*/1, generation, 1U, fixture->pool, stats, out);
        return out.beauty;
    };
    const std::vector<std::uint32_t> all = pathtracer::scene::allTexels(kImageSize, kImageSize);
    // Every third texel: runs of one with gaps, the shape an adaptive list takes, crossing rows.
    std::vector<std::uint32_t> subset;
    for (std::size_t i = 0; i < all.size(); i += 3) {
        subset.push_back(all[i]);
    }
    const pathtracer::gfx::HdrImage full = renderOf(all);
    const pathtracer::gfx::HdrImage partial = renderOf(subset);
    std::size_t tracedDiffering = 0;
    std::size_t untracedWritten = 0;
    for (std::size_t texel = 0; texel < all.size(); ++texel) {
        const bool traced = texel % 3 == 0;
        for (int c = 0; c < partial.channels; ++c) {
            const std::size_t i = (texel * static_cast<std::size_t>(partial.channels)) + static_cast<std::size_t>(c);
            tracedDiffering += traced && partial.texels[i] != full.texels[i] ? 1 : 0;
            untracedWritten += !traced && partial.texels[i] != -1.0F ? 1 : 0;
        }
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "%zu floats of traced texels differ from the full render", tracedDiffering);
    PT_EXPECT(ctx, tracedDiffering == 0, detail);
    std::snprintf(detail, sizeof(detail), "%zu floats of untraced texels were written", untracedWritten);
    PT_EXPECT(ctx, untracedWritten == 0, detail);
}

// The load-bearing property of a list-derived chunk length: it partitions the work, it must not change the result.
PT_CHECK(render_is_invariant_to_chunk_length, Slow, Exact) {
    ctx.plan(2);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene construction failed");
        return;
    }
    // Chosen so the two thread counts resolve to different chunks on any host: the counts are explicit, not hardware_concurrency.
    constexpr int kChunkedImageSize = 96;
    constexpr unsigned int kFewThreads = 1;
    constexpr unsigned int kManyThreads = 8;
    const int coarse = pathtracer::scene::pathTraceChunkTexels(kChunkedImageSize * kChunkedImageSize, kFewThreads);
    const int fine = pathtracer::scene::pathTraceChunkTexels(kChunkedImageSize * kChunkedImageSize, kManyThreads);
    char sizes[160];
    std::snprintf(sizes, sizeof(sizes), "both thread counts resolved to chunk %d, so this check would prove nothing", coarse);
    PT_EXPECT(ctx, coarse != fine, sizes);

    const Camera camera = makeCamera();
    const pathtracer::scene::LightSet lights(&fixture->environment, glm::vec3(0.0F), 1.0F, fixture->scene.lights);
    const std::atomic<std::uint64_t> generation{1};
    pathtracer::debug::PassStats stats;
    const auto renderWith = [&](unsigned int threads) {
        pathtracer::scene::ThreadPool pool(threads);
        PathTraceResult out = pathtracer::scene::makePathTraceResult(kChunkedImageSize, kChunkedImageSize, makeSettings().replicates);
        stats.reset();
        pathtracer::scene::renderPathTraced(camera, *fixture->accel, fixture->scene.shadingTriangles,
                                         fixture->scene.instances, fixture->scene.instanceLightIndex, lights, pathtracer::scene::VolumeSet{},
                                         kChunkedImageSize, kChunkedImageSize, pathtracer::scene::allTexels(kChunkedImageSize, kChunkedImageSize),
                                         /*showSky=*/true, makeSettings(), /*scrambleSeed=*/1U, /*sampleBase=*/0,
                                         /*sampleCount=*/1, generation, 1U, pool, stats, out);
        return out.beauty.texels;
    };
    const std::vector<float> coarseImage = renderWith(kFewThreads);
    const std::vector<float> fineImage = renderWith(kManyThreads);
    std::size_t differing = 0;
    for (std::size_t i = 0; i < coarseImage.size(); ++i) {
        differing += coarseImage[i] != fineImage[i] ? 1 : 0;
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu of %zu floats differ between %d- and %d-texel chunks", differing,
                  coarseImage.size(), coarse, fine);
    PT_EXPECT(ctx, differing == 0, detail);
}

// Thread-count invariance on renderPathTraced (the pool is private): bit-identical, as chunks are owned outright and sample order is fixed.
PT_CHECK(render_is_invariant_to_thread_count, Slow, Exact) {
    ctx.plan(1);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene construction failed");
        return;
    }
    const Camera camera = makeCamera();
    const pathtracer::scene::LightSet lights(&fixture->environment, glm::vec3(0.0F), 1.0F, fixture->scene.lights);
    const std::atomic<std::uint64_t> generation{1};
    pathtracer::debug::PassStats stats;

    const auto renderWith = [&](unsigned int threads) {
        pathtracer::scene::ThreadPool pool(threads);
        PathTraceResult out = pathtracer::scene::makePathTraceResult(kImageSize, kImageSize, makeSettings().replicates);
        stats.reset();
        pathtracer::scene::renderPathTraced(camera, *fixture->accel, fixture->scene.shadingTriangles,
                                         fixture->scene.instances, fixture->scene.instanceLightIndex, lights, pathtracer::scene::VolumeSet{},
                                         kImageSize, kImageSize, pathtracer::scene::allTexels(kImageSize, kImageSize),
                                         /*showSky=*/true, makeSettings(), /*scrambleSeed=*/1U, /*sampleBase=*/0,
                                         /*sampleCount=*/1, generation, 1U, pool, stats, out);
        return out.beauty.texels;
    };

    const std::vector<float> single = renderWith(1);
    const std::vector<float> many = renderWith(std::max(2U, std::thread::hardware_concurrency()));
    std::size_t differing = 0;
    for (std::size_t i = 0; i < single.size(); ++i) {
        differing += single[i] != many[i] ? 1 : 0;
    }
    char detail[176];
    std::snprintf(detail, sizeof(detail), "%zu of %zu floats differ between a 1-thread and an N-thread render",
                  differing, single.size());
    PT_EXPECT(ctx, differing == 0, detail);
}

}  // namespace

// The replicate means are not an averaged image, so they are absent from kLanes and from the two checks above; this is what covers them.
PT_CHECK(replicate_means_match_batch_means, Slow, Exact) {
    constexpr int kPasses = 8;
    ctx.plan(3);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        for (int i = 0; i < 3; ++i) {
            PT_EXPECT(ctx, false, "scene/driver construction failed");
        }
        return;
    }
    const Camera camera = makeCamera();
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(kPasses, camera));
    const std::shared_ptr<const PathTraceResult> published =
        waitForPublished(*fixture->driver, generation, kPasses);
    if (published == nullptr) {
        for (int i = 0; i < 3; ++i) {
            PT_EXPECT(ctx, false, "driver never published its maxSamples cap");
        }
        return;
    }

    constexpr double kUnitRoundoff = std::numeric_limits<float>::epsilon() / 2.0;
    constexpr double kGamma3 = (3.0 * kUnitRoundoff) / (1.0 - (3.0 * kUnitRoundoff));
    constexpr double kGamma4 = (4.0 * kUnitRoundoff) / (1.0 - (4.0 * kUnitRoundoff));
    const int replicates = makeSettings().replicates;
    const auto stride = static_cast<std::size_t>(replicates);
    const auto pixels = static_cast<std::size_t>(kImageSize) * kImageSize;
    const glm::dvec3 weights = glm::dvec3(pathtracer::scene::cie::sceneLuminanceWeights());

    // The same passes the driver ran, synchronously: per replicate slot the exact luminance sum and the float recurrence's error bound.
    const pathtracer::scene::LightSet lights(&fixture->environment, glm::vec3(0.0F), 1.0F, fixture->scene.lights);
    PathTraceResult pass = pathtracer::scene::makePathTraceResult(kImageSize, kImageSize, replicates);
    const auto scrambleSeed = static_cast<std::uint32_t>(generation);
    const std::atomic<std::uint64_t> oracleGeneration{scrambleSeed};
    pathtracer::debug::PassStats stats;
    std::vector<double> sum(pixels * stride, 0.0);
    // Rounding of the recurrence itself (E) and the mean of the float luminance dot's own errors it averages in (D).
    std::vector<double> recurrenceError(pixels * stride, 0.0);
    std::vector<double> inputError(pixels * stride, 0.0);
    for (int p = 0; p < kPasses; ++p) {
        stats.reset();
        pathtracer::scene::renderPathTraced(camera, *fixture->accel, fixture->scene.shadingTriangles,
                                         fixture->scene.instances, fixture->scene.instanceLightIndex, lights, pathtracer::scene::VolumeSet{},
                                         kImageSize, kImageSize, pathtracer::scene::allTexels(kImageSize, kImageSize),
                                         /*showSky=*/false, makeSettings(), scrambleSeed, /*sampleBase=*/p,
                                         /*sampleCount=*/kPasses, oracleGeneration, scrambleSeed, fixture->pool,
                                         stats, pass);
        const auto slot = static_cast<std::size_t>(p % replicates);
        const double k = (p / replicates) + 1;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const glm::dvec3 sample(pass.beauty.rgb(pixel));
            const double luminance = glm::dot(sample, weights);
            // The fold's three-term float dot: |fl(w.x) - w.x| <= gamma_3 sum |w_c x_c| (Higham 2002, eq. 3.5).
            const double dotError = kGamma3 * glm::dot(glm::abs(sample), weights);
            const std::size_t i = (pixel * stride) + slot;
            const double before = k == 1.0 ? 0.0 : sum[i] / (k - 1.0);
            sum[i] += luminance;
            // The first visit is 0 + (x - 0) * 1, exactly the float luminance; later ones are foldRunningMean's recurrence.
            if (k > 1.0) {
                const double e = recurrenceError[i];
                recurrenceError[i] = ((1.0 - (1.0 / k)) * (1.0 + kUnitRoundoff) * e) +
                                     (kGamma4 * (std::fabs(luminance - before) + dotError + inputError[i] + e) / k) +
                                     (kUnitRoundoff * std::fabs(sum[i] / k));
            }
            inputError[i] += (dotError - inputError[i]) / k;
        }
    }

    std::size_t violations = 0;
    double worstRatio = 0.0;
    double largestSpread = 0.0;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        double lowest = std::numeric_limits<double>::infinity();
        double highest = -lowest;
        for (std::size_t slot = 0; slot < stride; ++slot) {
            const std::size_t i = (pixel * stride) + slot;
            const double count = static_cast<double>((kPasses - static_cast<int>(slot) + replicates - 1) / replicates);
            const double expected = sum[i] / count;
            lowest = std::min(lowest, expected);
            highest = std::max(highest, expected);
            const double error = std::fabs(static_cast<double>(published->replicateLuminance[i]) - expected);
            // Plus the oracle's own single rounding of the exact mean to float, as the lanes' bound carries.
            const double tolerance = recurrenceError[i] + inputError[i] + (kUnitRoundoff * std::fabs(expected));
            violations += !(error <= tolerance) ? 1 : 0;
            const double ratio = tolerance > 0.0 ? error / tolerance
                                                 : (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
            worstRatio = std::max(worstRatio, ratio);
        }
        largestSpread = std::max(largestSpread, highest - lowest);
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail),
                  "%zu of %zu replicate means over their forward-error bound; worst |running - batch|/bound = %.3e", violations,
                  pixels * stride, worstRatio);
    PT_EXPECT(ctx, violations == 0, detail);
    PT_EXPECT(ctx, published->replicates == replicates && published->replicateLuminance.size() == pixels * stride,
              "the published result does not carry one luminance mean per replicate per texel");
    // Replicates that agree everywhere would satisfy the bound with copies of one lane, proving nothing about the interleaving.
    std::snprintf(detail, sizeof(detail), "largest spread between replicate means across the frame is %.6g", largestSpread);
    PT_EXPECT(ctx, largestSpread > 0.0, detail);
}

// A cap no check reaches before parking the driver, so offline accumulations are sampled mid-flight; a multiple of every replicate count.
constexpr int kLongCap = 1 << 16;

// Parks the driver once `generation` has run `passes` passes, then lets the pass in flight land; returns the parked index, 0 on timeout.
int parkAfter(PathTraceDriver& driver, std::uint64_t generation, int passes) {
    const bool reached = waitFor([&] {
        const pathtracer::debug::PassRecord pass = driver.lastPassRecord();
        return pass.generation == generation && pass.passIndex >= passes;
    });
    driver.setSuspended(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    return reached ? driver.lastPassRecord().passIndex : 0;
}

// Offline folds every pass but publishes none before the last: parked mid-accumulation, passes have run and nothing is on screen.
PT_CHECK(offline_publishes_nothing_mid_accumulation, Slow, Exact) {
    ctx.plan(1);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    fixture->driver->setPublishesFinalOnly(true);
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(kLongCap, makeCamera()));
    const int parked = parkAfter(*fixture->driver, generation, 2);
    const std::shared_ptr<const PathTraceResult> shown = fixture->driver->latestResult();
    char detail[160];
    std::snprintf(detail, sizeof(detail), "parked after %d of %d passes with %d published", parked, kLongCap,
                  shown == nullptr ? 0 : shown->samples);
    PT_EXPECT(ctx, parked >= 2 && parked < kLongCap && shown == nullptr, detail);
}

// The finished frame stays on screen while the next accumulation runs, and only it: a new request does not clear the hold.
PT_CHECK(offline_holds_the_previous_frame_until_the_next_finishes, Slow, Exact) {
    ctx.plan(2);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    fixture->driver->setPublishesFinalOnly(true);
    const std::uint64_t first = fixture->driver->requestTrace(makeRequest(4, makeCamera()));
    const std::shared_ptr<const PathTraceResult> finished = waitForPublished(*fixture->driver, first, 4);
    PT_EXPECT(ctx, finished != nullptr && finished->samples == 4, "offline never published the finished first accumulation");
    const std::uint64_t second = fixture->driver->requestTrace(makeRequest(kLongCap, makeCamera()));
    const int parked = parkAfter(*fixture->driver, second, 2);
    char detail[160];
    std::snprintf(detail, sizeof(detail), "parked %d passes into the second accumulation; the held frame was %s", parked,
                  fixture->driver->latestResult() == finished ? "kept" : "replaced");
    PT_EXPECT(ctx, finished != nullptr && parked >= 2 && fixture->driver->latestResult() == finished, detail);
}

// The mode is no part of the request: a switch either way neither restarts the accumulation nor loses a pass, it gates the next publish.
PT_CHECK(render_mode_switch_keeps_the_accumulation, Slow, Exact) {
    ctx.plan(3);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        for (int i = 0; i < 3; ++i) {
            PT_EXPECT(ctx, false, "scene/driver construction failed");
        }
        return;
    }
    const std::uint64_t generation = fixture->driver->requestTrace(makeRequest(kLongCap, makeCamera()));
    const int progressive = parkAfter(*fixture->driver, generation, 2);
    const std::shared_ptr<const PathTraceResult> held = fixture->driver->latestResult();
    PT_EXPECT(ctx, progressive >= 2 && held != nullptr && held->generation == generation && held->samples == progressive,
              "progressive did not publish its latest pass before the switch");
    // Switched while parked, so no pass straddles the switch and both halves are exact.
    fixture->driver->setPublishesFinalOnly(true);
    fixture->driver->setSuspended(false);
    const int offline = parkAfter(*fixture->driver, generation, progressive + 2);
    char detail[192];
    std::snprintf(detail, sizeof(detail), "offline continued %d -> %d passes and %s the frame", progressive, offline,
                  fixture->driver->latestResult() == held ? "held" : "replaced");
    PT_EXPECT(ctx, offline >= progressive + 2 && fixture->driver->latestResult() == held, detail);
    fixture->driver->setPublishesFinalOnly(false);
    fixture->driver->setSuspended(false);
    const std::shared_ptr<const PathTraceResult> resumed = waitForPublished(*fixture->driver, generation, offline + 1);
    std::snprintf(detail, sizeof(detail), "back in progressive, published %d samples after %d offline passes",
                  resumed == nullptr ? 0 : resumed->samples, offline);
    PT_EXPECT(ctx, resumed != nullptr && resumed->samples > offline, detail);
}

// Adaptive sampling on the driver fixture: a cap of 32 passes at K = 4 checks at 4, 8, 16 and 32, the first from minSamples.
constexpr int kAdaptiveCap = 32;
constexpr int kAdaptiveMinSamples = 4;

PathTraceDriver::Request makeAdaptiveRequest(float threshold, const Camera& camera) {
    PathTraceDriver::Request request = makeRequest(kAdaptiveCap, camera);
    request.settings.adaptive = {threshold, 0.95F, kAdaptiveMinSamples};
    return request;
}

// The driver's final result once its accumulation ends, at the cap or with every texel stopped; nullptr on timeout.
std::shared_ptr<const PathTraceResult> waitForSettled(const PathTraceDriver& driver, std::uint64_t generation) {
    std::shared_ptr<const PathTraceResult> result;
    const bool settled = waitFor([&] {
        result = driver.latestResult();
        return result != nullptr && result->generation == generation && (result->samples == kAdaptiveCap || result->activeTexels == 0);
    });
    return settled ? result : nullptr;
}

// HeadlessRenderer's adaptive loop on the shared kernels: trace the list, fold its runs in place, retire converged texels at checkpoints.
PathTraceResult sequentialAdaptiveFold(DriverFixture& fixture, const PathTraceDriver::Request& request, std::uint32_t scrambleSeed) {
    const pathtracer::scene::LightSet lights(&fixture.environment, glm::vec3(0.0F), 1.0F, fixture.scene.lights);
    const PathTraceSettings& settings = request.settings;
    PathTraceResult pass = pathtracer::scene::makePathTraceResult(kImageSize, kImageSize, settings.replicates);
    PathTraceResult mean = pathtracer::scene::makePathTraceResult(kImageSize, kImageSize, settings.replicates);
    std::vector<std::uint32_t> active = pathtracer::scene::allTexels(kImageSize, kImageSize);
    const std::atomic<std::uint64_t> generation{scrambleSeed};
    pathtracer::debug::PassStats stats;
    const auto stride = static_cast<std::size_t>(settings.replicates);
    int p = 0;
    for (; p < request.maxSamples && !active.empty(); ++p) {
        pathtracer::scene::renderPathTraced(request.camera, *fixture.accel, fixture.scene.shadingTriangles, fixture.scene.instances,
                                            fixture.scene.instanceLightIndex, lights, pathtracer::scene::VolumeSet{}, kImageSize, kImageSize,
                                            active, request.showSky, settings, scrambleSeed, /*sampleBase=*/p,
                                            /*sampleCount=*/request.maxSamples, generation, scrambleSeed, fixture.pool, stats, pass);
        const float invN = 1.0F / static_cast<float>(p + 1);
        const float invCount = 1.0F / static_cast<float>((p / settings.replicates) + 1);
        const auto fold = [&](std::uint32_t begin, std::uint32_t end) {
            const std::size_t run = end - begin;
            float* means = mean.replicateLuminance.data() + (begin * stride);
            pathtracer::scene::foldReplicateLuminance(means, pass.beauty.texels.data() + (begin * 3), 3, means, run, settings.replicates,
                                                      p % settings.replicates, invCount);
            std::fill(mean.texelSamples.data() + begin, mean.texelSamples.data() + end, p + 1);
            for (const Lane& lane : kLanes) {
                const auto channels = static_cast<std::size_t>((pass.*lane.image).channels);
                const float* drawn = (pass.*lane.image).texels.data() + (begin * channels);
                float* folded = (mean.*lane.image).texels.data() + (begin * channels);
                if (p == 0) {
                    std::copy(drawn, drawn + (run * channels), folded);
                } else {
                    pathtracer::scene::foldRunningMean(folded, drawn, folded, run * channels, invN);
                }
            }
        };
        for (int y = 0; y < kImageSize; ++y) {
            const auto rowBegin = static_cast<std::uint32_t>(y * kImageSize);
            pathtracer::scene::forEachTracedRun(active, rowBegin, rowBegin + kImageSize, fold, [](std::uint32_t, std::uint32_t) {});
        }
        if (pathtracer::scene::adaptiveCheckpoint(p + 1, settings)) {
            const pathtracer::scene::FoveatedThreshold fovea(request.camera, kImageSize, kImageSize, settings.adaptive.marSlope);
            pathtracer::scene::retireConvergedTexels(active, mean.replicateLuminance.data(), settings, fovea, fixture.pool);
        }
    }
    mean.samples = p;
    mean.activeTexels = active.size();
    return mean;
}

// Stopped texels are carried bit-exact through the pool's rotation, so the driver equals the in-place sequential fold everywhere.
PT_CHECK(adaptive_driver_matches_the_sequential_fold, Slow, Exact) {
    // Chosen on this fixture so texels stop at several checkpoints and some reach the cap; the last assertion holds it to that.
    constexpr float kThreshold = 0.5F;
    // Uniform, then foveated steeply enough that the 8-texel frame's 3.4-degree texels pool off-centre.
    constexpr std::array<float, 2> kMarSlopes{0.0F, 1.0F};
    ctx.plan(static_cast<int>(kMarSlopes.size()) * 4 + 1);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        for (int i = 0; i < static_cast<int>(kMarSlopes.size()) * 4 + 1; ++i) {
            PT_EXPECT(ctx, false, "scene/driver construction failed");
        }
        return;
    }
    for (const float marSlope : kMarSlopes) {
        PathTraceDriver::Request request = makeAdaptiveRequest(kThreshold, makeCamera());
        request.settings.adaptive.marSlope = marSlope;
        const std::uint64_t generation = fixture->driver->requestTrace(request);
        const std::shared_ptr<const PathTraceResult> published = waitForSettled(*fixture->driver, generation);
        if (published == nullptr) {
            for (int i = 0; i < (marSlope > 0.0F ? 5 : 4); ++i) {
                PT_EXPECT(ctx, false, "driver never settled");
            }
            continue;
        }
        const PathTraceResult expected = sequentialAdaptiveFold(*fixture, request, static_cast<std::uint32_t>(generation));
        std::size_t differing = 0;
        for (const Lane& lane : kLanes) {
            const std::vector<float>& image = ((*published).*lane.image).texels;
            for (std::size_t i = 0; i < image.size(); ++i) {
                const float folded = (expected.*lane.image).texels[i];
                differing += std::bit_cast<std::uint32_t>(image[i]) != std::bit_cast<std::uint32_t>(folded) ? 1 : 0;
            }
        }
        for (std::size_t i = 0; i < published->replicateLuminance.size(); ++i) {
            differing += std::bit_cast<std::uint32_t>(published->replicateLuminance[i]) !=
                                 std::bit_cast<std::uint32_t>(expected.replicateLuminance[i])
                             ? 1
                             : 0;
        }
        char detail[192];
        std::snprintf(detail, sizeof(detail), "%zu lane and replicate floats differ from the sequential adaptive fold", differing);
        PT_EXPECT(ctx, differing == 0, detail);
        std::snprintf(detail, sizeof(detail), "driver ended at %d passes with %zu active, the fold at %d with %zu", published->samples,
                      published->activeTexels, expected.samples, expected.activeTexels);
        PT_EXPECT(ctx, published->samples == expected.samples && published->activeTexels == expected.activeTexels &&
                           published->texelSamples == expected.texelSamples,
                  detail);
        // Every count is a checkpoint a texel stopped at, or the final pass of one still sampling.
        const auto checkpoint = [&](int n) { return pathtracer::scene::adaptiveCheckpoint(n, request.settings); };
        const bool countsValid = std::all_of(published->texelSamples.begin(), published->texelSamples.end(),
                                             [&](int n) { return checkpoint(n) || n == published->samples; });
        PT_EXPECT(ctx, countsValid, "a texel stopped between checkpoints");
        // Non-vacuous: stops at two or more checkpoints, and texels still sampling at the cap, or the check exercised no carry or no fold.
        std::vector<int> stops;
        for (const int n : published->texelSamples) {
            if (n < published->samples && std::find(stops.begin(), stops.end(), n) == stops.end()) {
                stops.push_back(n);
            }
        }
        std::snprintf(detail, sizeof(detail), "texels stopped at %zu distinct checkpoints with %zu still active at %d passes", stops.size(),
                      published->activeTexels, published->samples);
        PT_EXPECT(ctx, stops.size() >= 2 && published->activeTexels > 0 && published->samples == kAdaptiveCap, detail);
        if (marSlope > 0.0F) {
            // The same scramble uniform: generation seeds the sampler, so only the same seed isolates foveation's effect.
            PathTraceDriver::Request uniform = request;
            uniform.settings.adaptive.marSlope = 0.0F;
            const PathTraceResult unfoveated = sequentialAdaptiveFold(*fixture, uniform, static_cast<std::uint32_t>(generation));
            PT_EXPECT(ctx, unfoveated.texelSamples != published->texelSamples, "foveation left every texel's sample count unchanged");
        }
    }
}

// Once every texel has stopped there is no pass left to run, so the driver idles below its cap and says so in its records.
PT_CHECK(adaptive_driver_idles_once_every_texel_stops, Slow, Exact) {
    ctx.plan(2);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    // Over the floor alone, so every texel's samples vary: replicates agreeing exactly never stop, as a black miss would.
    const Camera floorOnly(glm::vec3(-10.0F, -10.0F, 5.0F), glm::vec3(0.0F), Camera::FilmBack{36.0F, 24.0F}, 50.0F, 0.01F, 1000.0F, 2.8F,
                           1.0F / 125.0F, 100.0F);
    // A threshold every estimate with spread meets, so the first checkpoint retires every texel.
    const std::uint64_t generation = fixture->driver->requestTrace(makeAdaptiveRequest(std::numeric_limits<float>::max(), floorOnly));
    const std::shared_ptr<const PathTraceResult> published = waitForSettled(*fixture->driver, generation);
    // Long enough for a driver that failed to idle to run past the checkpoint; a hang detector, not a timing assertion.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const std::shared_ptr<const PathTraceResult> later = fixture->driver->latestResult();
    char detail[160];
    std::snprintf(detail, sizeof(detail), "published %d passes, %d after waiting, against a first checkpoint of %d",
                  published == nullptr ? -1 : published->samples, later == nullptr ? -1 : later->samples, kAdaptiveMinSamples);
    PT_EXPECT(ctx, published != nullptr && later != nullptr && later->samples == kAdaptiveMinSamples && later->activeTexels == 0, detail);
    const pathtracer::debug::PassRecord record = fixture->driver->lastPassRecord();
    PT_EXPECT(ctx, record.passIndex == kAdaptiveMinSamples && record.activeTexels == 0,
              "the last pass record does not report the checkpoint that stopped every texel");
}

// Offline changes when a frame is published, never what it holds: the finished frame equals progressive's bit for bit, stats included.
PT_CHECK(offline_final_frame_is_bit_identical_to_progressive, Slow, Exact) {
    // The threshold the sequential-fold check uses, so texels stop at several checkpoints and the carry path is exercised.
    constexpr float kThreshold = 0.5F;
    ctx.plan(2);
    std::unique_ptr<DriverFixture> progressive = makeFixture();
    std::unique_ptr<DriverFixture> offline = makeFixture();
    if (!progressive->valid() || !offline->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    offline->driver->setPublishesFinalOnly(true);
    const PathTraceDriver::Request request = makeAdaptiveRequest(kThreshold, makeCamera());
    const std::shared_ptr<const PathTraceResult> a = waitForSettled(*progressive->driver, progressive->driver->requestTrace(request));
    const std::shared_ptr<const PathTraceResult> b = waitForSettled(*offline->driver, offline->driver->requestTrace(request));
    if (a == nullptr || b == nullptr) {
        PT_EXPECT(ctx, false, "a driver never settled");
        PT_EXPECT(ctx, false, "a driver never settled");
        return;
    }
    std::size_t differing = 0;
    const auto compare = [&](const std::vector<float>& x, const std::vector<float>& y) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            differing += std::bit_cast<std::uint32_t>(x[i]) != std::bit_cast<std::uint32_t>(y[i]) ? 1 : 0;
        }
    };
    for (const Lane& lane : kLanes) {
        compare(((*a).*lane.image).texels, ((*b).*lane.image).texels);
    }
    compare(a->replicateLuminance, b->replicateLuminance);
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu lane and replicate floats differ; %d vs %d passes", differing, a->samples, b->samples);
    PT_EXPECT(ctx, differing == 0 && a->samples == b->samples && a->texelSamples == b->texelSamples && a->activeTexels == b->activeTexels,
              detail);
    PT_EXPECT(ctx, a->overRange.rawPeak == b->overRange.rawPeak && a->overRange.aboveBin == b->overRange.aboveBin,
              "the finished frame's over-range statistics differ between the modes");
}

// Offline finishes on adaptive stopping as on the cap: with every texel retired at the first checkpoint, that pass is published.
PT_CHECK(offline_publishes_once_adaptive_stops_every_texel, Slow, Exact) {
    ctx.plan(1);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene/driver construction failed");
        return;
    }
    fixture->driver->setPublishesFinalOnly(true);
    // The floor-only view and saturating threshold of adaptive_driver_idles_once_every_texel_stops: all retire at the first checkpoint.
    const Camera floorOnly(glm::vec3(-10.0F, -10.0F, 5.0F), glm::vec3(0.0F), Camera::FilmBack{36.0F, 24.0F}, 50.0F, 0.01F, 1000.0F, 2.8F,
                           1.0F / 125.0F, 100.0F);
    const std::uint64_t generation = fixture->driver->requestTrace(makeAdaptiveRequest(std::numeric_limits<float>::max(), floorOnly));
    const std::shared_ptr<const PathTraceResult> published = waitForSettled(*fixture->driver, generation);
    char detail[160];
    std::snprintf(detail, sizeof(detail), "published %d passes with %zu active, against a first checkpoint of %d",
                  published == nullptr ? -1 : published->samples, published == nullptr ? 0 : published->activeTexels, kAdaptiveMinSamples);
    PT_EXPECT(ctx, published != nullptr && published->samples == kAdaptiveMinSamples && published->activeTexels == 0, detail);
}

PT_CHECK_MAIN("driver")
