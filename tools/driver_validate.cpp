// Correctness gate for PathTraceDriver, driven only through its public API; also pins the request invariant driverLoop relies on.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
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
using pathtracer::scene::QuadLight;
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
    std::vector<QuadLight> quadLights;
    std::vector<PathTraceSettings> perInstanceSettings;
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
    settings.maxBounces = 1;
    settings.russianRouletteStartBounce = 8;
    settings.bumpStrength = 0.0F;
    settings.roughnessMin = 0.0F;
    settings.roughnessMax = 1.0F;
    settings.diffuseColour = glm::vec3(1.0F);
    settings.ior = 1.5F;
    settings.transmissionFactor = 0.0F;
    settings.metallicFactor = 0.0F;
    settings.roughnessFactor = 1.0F;
    return settings;
}

// Everything the driver holds by reference must outlive it and never move, so the fixture is heap-allocated before the driver exists.
struct DriverFixture {
    TestScene scene;
    std::optional<EmbreeAccel> accel;
    EnvironmentMap environment = tools::fixtures::makeUniformEnvironment();
    std::optional<PathTraceDriver> driver;
    pathtracer::scene::ThreadPool pool;

    [[nodiscard]] bool valid() const { return accel.has_value() && driver.has_value(); }
};

std::unique_ptr<DriverFixture> makeFixture() {
    auto fixture = std::make_unique<DriverFixture>();
    TestScene& scene = fixture->scene;
    // Every accumulated image must vary across passes, or a missing accumulate entry for it is invisible to the running-mean check.
    pushQuad(scene,
             {glm::vec3(-kQuadExtent, -kQuadExtent, kFloorZ), glm::vec3(kQuadExtent, -kQuadExtent, kFloorZ),
              glm::vec3(kQuadExtent, kQuadExtent, kFloorZ), glm::vec3(-kQuadExtent, kQuadExtent, kFloorZ)},
             glm::vec3(0.0F, 0.0F, 1.0F), 0);
    pushQuad(scene,
             {glm::vec3(kWallX, -kQuadExtent, kFloorZ), glm::vec3(kWallX, -kQuadExtent, kWallTopZ),
              glm::vec3(kWallX, kQuadExtent, kWallTopZ), glm::vec3(kWallX, kQuadExtent, kFloorZ)},
             glm::vec3(-1.0F, 0.0F, 0.0F), 1);
    for (int i = 0; i < 2; ++i) {
        scene.instances.push_back(
            MeshInstance{tools::fixtures::makeMaterial(1.0F, glm::vec3(0.04F)), glm::mat4(1.0F), ""});
    }
    scene.instanceLightIndex.assign(scene.instances.size(), -1);
    scene.perInstanceSettings.assign(scene.instances.size(), makeSettings());
    scene.perInstanceSettings[1].transmissionFactor = 1.0F;
    fixture->accel = EmbreeAccel::build(fixture->scene.worldTriangles);
    if (!fixture->accel.has_value()) {
        return fixture;
    }
    fixture->driver.emplace(*fixture->accel, fixture->scene.shadingTriangles, fixture->scene.instances,
                             fixture->scene.instanceLightIndex, fixture->environment,
                             fixture->scene.quadLights, fixture->scene.perInstanceSettings);
    return fixture;
}

Camera makeCamera() {
    return Camera(glm::vec3(0.0F, 0.0F, 5.0F), 0.0F, 0.0F, Camera::FilmBack{36.0F, 24.0F}, 50.0F, 0.01F, 1000.0F,
                   2.8F, 1.0F / 125.0F, 100.0F);
}

PathTraceDriver::Request makeRequest(int maxSamples, const Camera& camera) {
    PathTraceDriver::Request request{camera, 0, 0, 0.0F, true, true, 1.0F, makeSettings(), 0};
    request.width = kImageSize;
    request.height = kImageSize;
    request.envRotationRadians = 0.0F;
    request.showSky = true;
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

// Every image the driver averages, index-aligned with accumulateMean's lists; beautyLuminanceM2 is not one, see below.
struct Lane {
    pathtracer::gfx::HdrImage PathTraceResult::*image;
    const char* name;
};
constexpr std::array<Lane, 10> kLanes{{{&PathTraceResult::beauty, "beauty"},
                                      {&PathTraceResult::bounceHeatmap, "bounceHeatmap"},
                                      {&PathTraceResult::ao, "ao"},
                                      {&PathTraceResult::shadow, "shadow"},
                                      {&PathTraceResult::directDiffuse, "directDiffuse"},
                                      {&PathTraceResult::indirectDiffuse, "indirectDiffuse"},
                                      {&PathTraceResult::directSpecular, "directSpecular"},
                                      {&PathTraceResult::indirectSpecular, "indirectSpecular"},
                                      {&PathTraceResult::refraction, "refraction"},
                                      {&PathTraceResult::fresnel, "fresnel"}}};

// Per float: the batch mean rounded once to float and the bound on the driver's distance from it; per lane, whether any float moved.
struct OracleMean {
    std::array<std::vector<float>, kLanes.size()> mean;
    std::array<std::vector<double>, kLanes.size()> tolerance;
    std::array<bool, kLanes.size()> varies{};
};

// Renders the driver's passes synchronously: the exact batch mean m_k in double, beside the float32 running mean's error bound E_k.
OracleMean oracleBatchMean(DriverFixture& fixture, const Camera& camera, int passes, std::uint32_t scrambleSeed) {
    constexpr double kUnitRoundoff = std::numeric_limits<float>::epsilon() / 2.0;
    constexpr double kGamma4 = (4.0 * kUnitRoundoff) / (1.0 - (4.0 * kUnitRoundoff));
    const pathtracer::scene::LightSet lights(&fixture.environment, 0.0F, 1.0F, fixture.scene.quadLights);
    PathTraceResult pass = pathtracer::scene::makePathTraceResult(kImageSize, kImageSize);
    const std::size_t floats = static_cast<std::size_t>(kImageSize) * kImageSize * 4;
    std::array<std::vector<double>, kLanes.size()> sum;
    OracleMean oracle;
    for (std::size_t lane = 0; lane < kLanes.size(); ++lane) {
        sum[lane].assign(floats, 0.0);
        oracle.tolerance[lane].assign(floats, 0.0);
    }
    const std::atomic<std::uint64_t> generation{scrambleSeed};
    pathtracer::debug::PassStats stats;
    for (int p = 0; p < passes; ++p) {
        stats.reset();
        pathtracer::scene::renderPathTraced(camera, *fixture.accel, fixture.scene.shadingTriangles,
                                         fixture.scene.instances, fixture.scene.instanceLightIndex, lights,
                                         kImageSize, kImageSize, /*showSky=*/true, makeSettings(),
                                         fixture.scene.perInstanceSettings, scrambleSeed, /*sampleBase=*/p,
                                         /*sampleCount=*/passes, generation, scrambleSeed, fixture.pool, stats,
                                         pass);
        const double k = p + 1;
        for (std::size_t lane = 0; lane < kLanes.size(); ++lane) {
            const std::vector<float>& x = (pass.*kLanes[lane].image).rgba;
            for (std::size_t i = 0; i < floats; ++i) {
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
        oracle.mean[lane].resize(floats);
        for (std::size_t i = 0; i < floats; ++i) {
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
        const std::vector<float>& running = ((*published).*kLanes[lane].image).rgba;
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
        const std::vector<float>& image = ((*published).*kLanes[lane].image).rgba;
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
    const Camera moved(glm::vec3(0.5F, 0.25F, 5.0F), 10.0F, -5.0F, Camera::FilmBack{36.0F, 24.0F}, 50.0F, 0.01F,
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

// Suspension parks the driver for a rasterizer AOV: the generation is the scramble seed, so bumping it to cancel would restart the image.
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
        const std::vector<float>& a = (uninterrupted.get()->*lane.image).rgba;
        const std::vector<float>& b = (continued.get()->*lane.image).rgba;
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

// Four pooled images for three pinnable results: acquireFreeBuffer must STALL rather than return a buffer still being read.
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
    const std::vector<float> snapshot = pinned->beauty.rgba;
    const std::shared_ptr<const PathTraceResult> later =
        waitForPublished(*fixture->driver, generation, pinned->samples + 6);
    if (later == nullptr) {
        PT_EXPECT(ctx, false, "driver stalled while a single result was pinned");
        return;
    }
    std::size_t moved = 0;
    for (std::size_t i = 0; i < snapshot.size(); ++i) {
        moved += pinned->beauty.rgba[i] != snapshot[i] ? 1 : 0;
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
        const float r = published->beauty.rgba[(t * 4) + 0];
        const float g = published->beauty.rgba[(t * 4) + 1];
        const float b = published->beauty.rgba[(t * 4) + 2];
        serialPeak = std::max(serialPeak, std::max(r, std::max(g, b)));
    }
    char peakDetail[176];
    std::snprintf(peakDetail, sizeof(peakDetail), "published rawPeak %.9g vs single-threaded scan %.9g",
                  static_cast<double>(stats.rawPeak), static_cast<double>(serialPeak));
    PT_EXPECT(ctx, stats.rawPeak == serialPeak, peakDetail);
}

// The tiling is a work split, so it must cover the target exactly and never hand the pool fewer tiles than it can use.
PT_CHECK(tile_size_covers_the_target_and_fills_the_pool, Fast, Exact) {
    using pathtracer::scene::kMinPathTraceTileSize;
    using pathtracer::scene::kPathTraceTileSize;
    using pathtracer::scene::kTilesPerThread;
    using pathtracer::scene::pathTraceTileSize;
    constexpr std::array<int, 9> kExtents{0, 1, 8, 32, 96, 115, 205, 1152, 2048};
    constexpr std::array<unsigned int, 5> kThreads{1, 2, 8, 64, 4096};
    ctx.plan(static_cast<int>(kExtents.size() * kExtents.size() * kThreads.size() * 3));
    for (const int width : kExtents) {
        for (const int height : kExtents) {
            for (const unsigned int threads : kThreads) {
                const int size = pathTraceTileSize(width, height, threads);
                char detail[192];
                std::snprintf(detail, sizeof(detail), "%dx%d on %u threads gave tile %d, outside [%d, %d]",
                              width, height, threads, size, kMinPathTraceTileSize, kPathTraceTileSize);
                PT_EXPECT(ctx, size >= kMinPathTraceTileSize && size <= kPathTraceTileSize, detail);

                // Ceiling division must reach every pixel: a grid short of the target would silently drop a strip.
                const int tilesX = (width + size - 1) / size;
                const int tilesY = (height + size - 1) / size;
                std::snprintf(detail, sizeof(detail), "%dx%d tile %d leaves %dx%d uncovered", width, height, size,
                              width - (tilesX * size), height - (tilesY * size));
                PT_EXPECT(ctx, tilesX * size >= width && tilesY * size >= height, detail);

                // Largest such tile: shrinking is only justified where the next size up could not fill the pool.
                const bool wantedMet = static_cast<long long>(tilesX) * tilesY >=
                                        static_cast<long long>(threads) * kTilesPerThread;
                const bool atFloor = size == kMinPathTraceTileSize;
                const bool atCeiling = size == kPathTraceTileSize;
                std::snprintf(detail, sizeof(detail), "%dx%d on %u threads shrank to %d without needing to", width,
                              height, threads, size);
                PT_EXPECT(ctx, atCeiling || atFloor || wantedMet, detail);
            }
        }
    }
}

// The load-bearing property of a resolution-derived tile size: it partitions the work, it must not change the result.
PT_CHECK(render_is_invariant_to_tile_size, Slow, Exact) {
    ctx.plan(2);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene construction failed");
        return;
    }
    // Chosen so the two thread counts resolve to different tiles on any host: the counts are explicit, not hardware_concurrency.
    constexpr int kTiledImageSize = 96;
    constexpr unsigned int kFewThreads = 1;
    constexpr unsigned int kManyThreads = 8;
    const int coarse = pathtracer::scene::pathTraceTileSize(kTiledImageSize, kTiledImageSize, kFewThreads);
    const int fine = pathtracer::scene::pathTraceTileSize(kTiledImageSize, kTiledImageSize, kManyThreads);
    char sizes[160];
    std::snprintf(sizes, sizeof(sizes), "both thread counts resolved to tile %d, so this check would prove nothing", coarse);
    PT_EXPECT(ctx, coarse != fine, sizes);

    const Camera camera = makeCamera();
    const pathtracer::scene::LightSet lights(&fixture->environment, 0.0F, 1.0F, fixture->scene.quadLights);
    const std::atomic<std::uint64_t> generation{1};
    pathtracer::debug::PassStats stats;
    const auto renderWith = [&](unsigned int threads) {
        pathtracer::scene::ThreadPool pool(threads);
        PathTraceResult out = pathtracer::scene::makePathTraceResult(kTiledImageSize, kTiledImageSize);
        stats.reset();
        pathtracer::scene::renderPathTraced(camera, *fixture->accel, fixture->scene.shadingTriangles,
                                         fixture->scene.instances, fixture->scene.instanceLightIndex, lights,
                                         kTiledImageSize, kTiledImageSize, /*showSky=*/true, makeSettings(),
                                         fixture->scene.perInstanceSettings, /*scrambleSeed=*/1U, /*sampleBase=*/0,
                                         /*sampleCount=*/1, generation, 1U, pool, stats, out);
        return out.beauty.rgba;
    };
    const std::vector<float> coarseImage = renderWith(kFewThreads);
    const std::vector<float> fineImage = renderWith(kManyThreads);
    std::size_t differing = 0;
    for (std::size_t i = 0; i < coarseImage.size(); ++i) {
        differing += coarseImage[i] != fineImage[i] ? 1 : 0;
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu of %zu floats differ between a %d px and a %d px tile grid", differing,
                  coarseImage.size(), coarse, fine);
    PT_EXPECT(ctx, differing == 0, detail);
}

// Thread-count invariance on renderPathTraced (the pool is private): bit-identical, as tiles are owned outright and sample order is fixed.
PT_CHECK(render_is_invariant_to_thread_count, Slow, Exact) {
    ctx.plan(1);
    std::unique_ptr<DriverFixture> fixture = makeFixture();
    if (!fixture->valid()) {
        PT_EXPECT(ctx, false, "scene construction failed");
        return;
    }
    const Camera camera = makeCamera();
    const pathtracer::scene::LightSet lights(&fixture->environment, 0.0F, 1.0F, fixture->scene.quadLights);
    const std::atomic<std::uint64_t> generation{1};
    pathtracer::debug::PassStats stats;

    const auto renderWith = [&](unsigned int threads) {
        pathtracer::scene::ThreadPool pool(threads);
        PathTraceResult out = pathtracer::scene::makePathTraceResult(kImageSize, kImageSize);
        stats.reset();
        pathtracer::scene::renderPathTraced(camera, *fixture->accel, fixture->scene.shadingTriangles,
                                         fixture->scene.instances, fixture->scene.instanceLightIndex, lights,
                                         kImageSize, kImageSize, /*showSky=*/true, makeSettings(),
                                         fixture->scene.perInstanceSettings, /*scrambleSeed=*/1U, /*sampleBase=*/0,
                                         /*sampleCount=*/1, generation, 1U, pool, stats, out);
        return out.beauty.rgba;
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

// The second moment is not an averaged image, so it is absent from kLanes and from the two checks above; this is what covers it.
PT_CHECK(running_m2_matches_batch_variance, Slow, Exact) {
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
    const auto pixels = static_cast<std::size_t>(kImageSize) * kImageSize;
    const std::size_t floats = pixels * 4;
    const glm::dvec3 weights(pathtracer::debug::kRec709LuminanceWeights);

    // The same passes the driver ran, synchronously, with every per-texel luminance kept in double for an exact two-pass variance.
    const pathtracer::scene::LightSet lights(&fixture->environment, 0.0F, 1.0F, fixture->scene.quadLights);
    PathTraceResult pass = pathtracer::scene::makePathTraceResult(kImageSize, kImageSize);
    const auto scrambleSeed = static_cast<std::uint32_t>(generation);
    const std::atomic<std::uint64_t> oracleGeneration{scrambleSeed};
    pathtracer::debug::PassStats stats;
    std::vector<std::vector<double>> luminance(kPasses, std::vector<double>(pixels, 0.0));
    std::vector<double> sum(floats, 0.0);
    // The driver reads its previous mean out of the float RGB lane, so the luminance it differences carries that lane's own error.
    std::vector<double> meanError(floats, 0.0);
    std::vector<double> previousLuminanceError(pixels, 0.0);
    std::vector<double> bound(pixels, 0.0);

    for (int p = 0; p < kPasses; ++p) {
        stats.reset();
        pathtracer::scene::renderPathTraced(camera, *fixture->accel, fixture->scene.shadingTriangles,
                                         fixture->scene.instances, fixture->scene.instanceLightIndex, lights,
                                         kImageSize, kImageSize, /*showSky=*/true, makeSettings(),
                                         fixture->scene.perInstanceSettings, scrambleSeed, /*sampleBase=*/p,
                                         /*sampleCount=*/kPasses, oracleGeneration, scrambleSeed, fixture->pool,
                                         stats, pass);
        const double k = p + 1;
        const std::vector<float>& drawn = pass.beauty.rgba;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::size_t texel = pixel * 4;
            glm::dvec3 before(0.0);
            glm::dvec3 sample(0.0);
            double beforeError = 0.0;
            for (int c = 0; c < 3; ++c) {
                before[c] = p == 0 ? 0.0 : sum[texel + static_cast<std::size_t>(c)] / (k - 1.0);
                sample[c] = drawn[texel + static_cast<std::size_t>(c)];
                beforeError += weights[c] * meanError[texel + static_cast<std::size_t>(c)];
            }
            const double sampleLuminance = glm::dot(sample, weights);
            luminance[static_cast<std::size_t>(p)][pixel] = sampleLuminance;
            if (p > 0) {
                // Welford's own step in exact arithmetic, against which the float recurrence's departure is bounded below.
                const double beforeLuminance = glm::dot(before, weights);
                const double afterLuminance = beforeLuminance + ((sampleLuminance - beforeLuminance) / k);
                const double term = (sampleLuminance - beforeLuminance) * (sampleLuminance - afterLuminance);
                // The three-term dot rounds too, so the float mean's luminance is off by the lane errors plus that.
                beforeError += kGamma3 * std::fabs(beforeLuminance);
                const double afterError = previousLuminanceError[pixel] + ((beforeError + previousLuminanceError[pixel]) / k);
                bound[pixel] = (bound[pixel] * (1.0 + kUnitRoundoff)) +
                               (std::fabs(sampleLuminance - afterLuminance) * beforeError) +
                               (std::fabs(sampleLuminance - beforeLuminance) * afterError) +
                               (beforeError * afterError) + (kGamma4 * std::fabs(term));
                previousLuminanceError[pixel] = beforeError;
            }
            for (int c = 0; c < 3; ++c) {
                const std::size_t i = texel + static_cast<std::size_t>(c);
                sum[i] += sample[c];
                if (p > 0) {
                    const double e = meanError[i];
                    meanError[i] = ((1.0 - (1.0 / k)) * (1.0 + kUnitRoundoff) * e) +
                                   (kGamma4 * (std::fabs(sample[c] - before[c]) + e) / k) +
                                   (kUnitRoundoff * std::fabs(sum[i] / k));
                }
            }
        }
    }

    std::size_t violations = 0;
    double worstRatio = 0.0;
    double largest = 0.0;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        // Two-pass sum of squared deviations, the definition, in double: the reference the incremental form must reproduce.
        double mean = 0.0;
        for (int p = 0; p < kPasses; ++p) {
            mean += luminance[static_cast<std::size_t>(p)][pixel];
        }
        mean /= static_cast<double>(kPasses);
        double expected = 0.0;
        for (int p = 0; p < kPasses; ++p) {
            const double deviation = luminance[static_cast<std::size_t>(p)][pixel] - mean;
            expected += deviation * deviation;
        }
        largest = std::max(largest, expected);
        const double error = std::fabs(static_cast<double>(published->beautyLuminanceM2[pixel]) - expected);
        // Plus the oracle's own single rounding of the exact moment to float, as the mean's bound carries.
        const double tolerance = bound[pixel] + (kUnitRoundoff * expected);
        violations += !(error <= tolerance) ? 1 : 0;
        const double ratio = tolerance > 0.0 ? error / tolerance
                                             : (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
        worstRatio = std::max(worstRatio, ratio);
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail),
                  "%zu of %zu texels over their forward-error bound; worst |running - batch|/bound = %.3e", violations,
                  pixels, worstRatio);
    PT_EXPECT(ctx, violations == 0, detail);
    // A frame the passes agree on everywhere would satisfy the bound with an all-zero lane, proving nothing about the recurrence.
    std::snprintf(detail, sizeof(detail), "largest batch second moment across the frame is %.6g", largest);
    PT_EXPECT(ctx, largest > 0.0, detail);
    const bool nonNegative = std::all_of(published->beautyLuminanceM2.begin(), published->beautyLuminanceM2.end(),
                                          [](float value) { return value >= 0.0F; });
    PT_EXPECT(ctx, nonNegative, "a sum of squared deviations came back negative");
}

PT_CHECK_MAIN("driver")
