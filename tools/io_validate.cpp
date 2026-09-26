// Correctness gate for the engine's file boundaries: the EXR round trip and the JSON scene, profile and bench-log parsers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include "check.h"
#include "pathtracer/config/profile_config.h"
#include "pathtracer/config/scene_config.h"
#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/bench_log.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/texture.h"

namespace {

// Written under the system temp directory: a validator must not need a writable checkout, nor leave anything for the next run.
std::filesystem::path scratchPath(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

// Values hostile to a lossy or narrowing round trip: denormal-scale, exact halves, far outside display range, and a negative.
pathtracer::gfx::HdrImage makeProbeImage() {
    pathtracer::gfx::HdrImage image;
    image.width = 7;   // deliberately not a power of two or a multiple of any tile size
    image.height = 5;
    image.rgba.resize(static_cast<std::size_t>(image.width) * image.height * 4);
    for (std::size_t i = 0; i < image.rgba.size(); ++i) {
        const std::size_t channel = i % 4;
        const auto t = static_cast<float>(i);
        switch (channel) {
            case 0: image.rgba[i] = t * 1.5F; break;              // exact halves
            case 1: image.rgba[i] = 1.0e-20F * (t + 1.0F); break;  // far below display range
            case 2: image.rgba[i] = 65504.0F - t; break;           // near the half-float maximum, if one were used
            default: image.rgba[i] = 1.0F; break;                  // alpha
        }
    }
    return image;
}

// The losslessness hdr_image.h claims in prose, bit-exact: the 1e-20 and 65504 rows are what would expose a half-float channel.
PT_CHECK(exr_round_trip_is_lossless, Fast, Exact) {
    ctx.plan(4);
    const pathtracer::gfx::HdrImage original = makeProbeImage();
    const std::filesystem::path path = scratchPath("engine_io_validate_roundtrip.exr");
    std::filesystem::remove(path);

    PT_EXPECT(ctx, pathtracer::gfx::writeExr(path.string(), original), "writeExr failed on a valid image");
    const std::optional<pathtracer::gfx::HdrImage> loaded = pathtracer::gfx::loadExr(path.string());
    if (!loaded.has_value()) {
        PT_EXPECT(ctx, false, "loadExr returned nullopt for a file writeExr had just written");
        PT_EXPECT(ctx, false, "dimensions unavailable");
        PT_EXPECT(ctx, false, "contents unavailable");
        std::filesystem::remove(path);
        return;
    }

    char dimDetail[160];
    std::snprintf(dimDetail, sizeof(dimDetail), "round trip returned %dx%d, wrote %dx%d", loaded->width,
                  loaded->height, original.width, original.height);
    PT_EXPECT(ctx, loaded->width == original.width && loaded->height == original.height, dimDetail);
    PT_EXPECT(ctx, loaded->rgba.size() == original.rgba.size(), "round trip changed the channel count");

    std::size_t differing = 0;
    float worst = 0.0F;
    if (loaded->rgba.size() == original.rgba.size()) {
        for (std::size_t i = 0; i < original.rgba.size(); ++i) {
            if (loaded->rgba[i] != original.rgba[i]) {
                ++differing;
                worst = std::max(worst, std::fabs(loaded->rgba[i] - original.rgba[i]));
            }
        }
    }
    char detail[192];
    std::snprintf(detail, sizeof(detail), "%zu of %zu floats changed across the round trip, worst delta %.9g",
                  differing, original.rgba.size(), static_cast<double>(worst));
    PT_EXPECT(ctx, differing == 0, detail);
    std::filesystem::remove(path);
}

// A missing file must be reported, not treated as an empty image: a caller given a zero-sized image renders black and never knows.
PT_CHECK(exr_load_rejects_bad_input, Fast, Exact) {
    ctx.plan(2);
    const std::filesystem::path missing = scratchPath("engine_io_validate_does_not_exist.exr");
    std::filesystem::remove(missing);
    PT_EXPECT(ctx, !pathtracer::gfx::loadExr(missing.string()).has_value(),
                  "loadExr accepted a path that does not exist");

    // A file that exists but is not an EXR: the realistic corruption, which a magic-number check catches and a header check does not.
    const std::filesystem::path garbage = scratchPath("engine_io_validate_garbage.exr");
    {
        std::ofstream out(garbage, std::ios::binary);
        out << "this is not an OpenEXR file, but it is definitely a file";
    }
    PT_EXPECT(ctx, !pathtracer::gfx::loadExr(garbage.string()).has_value(),
                  "loadExr accepted a file whose contents are not EXR");
    std::filesystem::remove(garbage);
}

// Writes `text` to a scratch .json and returns the path, so each rejection row states its own malformation inline.
std::filesystem::path writeJson(const char* name, const std::string& text) {
    const std::filesystem::path path = scratchPath(name);
    std::ofstream out(path);
    out << text;
    return path;
}

// The shipped scene must load: without this row, every rejection row below could pass by rejecting everything.
PT_CHECK(scene_config_accepts_the_shipped_scene, Fast, Exact) {
    ctx.plan(1);
    const std::filesystem::path scene = std::filesystem::path(ASSET_ROOT_DIR) / "scenes" / "cornell.json";
    const std::optional<pathtracer::config::SceneConfig> loaded = pathtracer::config::loadSceneConfig(scene.string());
    char detail[256];
    std::snprintf(detail, sizeof(detail), "loadSceneConfig rejected the shipped scene at %s", scene.string().c_str());
    PT_EXPECT(ctx, loaded.has_value(), detail);
}

// The rejection half of the contract: each row is a malformation a real authoring mistake produces, and each must be reported.
PT_CHECK(scene_config_rejects_malformed_input, Fast, Exact) {
    struct Case {
        const char* name;
        const char* file;
        std::string text;
    };
    // Built by mutating a base that loads, so each row fails for the reason it names rather than passing vacuously on an earlier error.
    const auto scene = [](const std::string& lights) {
        return std::string(
                   "{\"model\":{\"gltfPath\":\"geometry/cornell/cornell_v001.gltf\",\"texturePath\":\"\","
                   "\"position\":[0,0,0],\"rotation\":[0,0,0]},"
                   "\"environment\":{\"hdriPath\":\"textures/republiqueHDR_2k.exr\"},"
                   "\"materialPath\":\"materials/clay.json\"") +
               lights + "}";
    };
    const std::string validLights =
        ",\"lights\":[{\"type\":\"quad\",\"origin\":[0,0,0],\"edge0\":[1,0,0],\"edge1\":[0,0,1],"
        "\"color\":[1,1,1],\"intensity\":5.0,\"twoSided\":false}]";

    const std::vector<Case> cases = {
        {"not JSON at all", "engine_io_scene_notjson.json", "{ this is not json"},
        {"empty file", "engine_io_scene_empty.json", ""},
        {"JSON array where an object is required", "engine_io_scene_array.json", "[1, 2, 3]"},
        {"missing the model section entirely", "engine_io_scene_nomodel.json",
         "{\"environment\":{\"hdriPath\":\"x.exr\"},\"materialPath\":\"materials/clay.json\"}"},
        // Negative radiance is not a scene, and would propagate as negative energy through every estimator.
        {"negative light intensity", "engine_io_scene_negintensity.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"origin\":[0,0,0],\"edge0\":[1,0,0],\"edge1\":[0,0,1],"
               "\"color\":[1,1,1],\"intensity\":-5.0,\"twoSided\":false}]")},
        // The spherical-rectangle sampler (Urena et al. 2013) is exact only for a RECTANGLE, so skewed edges sample geometry it lacks.
        {"quad light with non-perpendicular edges", "engine_io_scene_skewlight.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"origin\":[0,0,0],\"edge0\":[1,0,0],\"edge1\":[1,1,0],"
               "\"color\":[1,1,1],\"intensity\":5.0,\"twoSided\":false}]")},
        {"negative light colour", "engine_io_scene_negcolor.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"origin\":[0,0,0],\"edge0\":[1,0,0],\"edge1\":[0,0,1],"
               "\"color\":[1,-1,1],\"intensity\":5.0,\"twoSided\":false}]")},
    };

    // Anti-vacuity: the base the three light rows are built from must itself LOAD, or they would prove nothing.
    const std::filesystem::path basePath = writeJson("engine_io_scene_base.json", scene(validLights));
    const bool baseLoads = pathtracer::config::loadSceneConfig(basePath.string()).has_value();
    std::filesystem::remove(basePath);

    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, baseLoads,
                  "the unmutated base scene must load, or every mutated row below passes vacuously");
    for (const Case& testCase : cases) {
        const std::filesystem::path path = writeJson(testCase.file, testCase.text);
        const bool accepted = pathtracer::config::loadSceneConfig(path.string()).has_value();
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadSceneConfig accepted a scene with %s", testCase.name);
        PT_EXPECT(ctx, !accepted, detail);
        std::filesystem::remove(path);
    }
}

// Anti-vacuity for the rejection rows below: every material the repo ships must still load once loadMaterialConfig validates.
PT_CHECK(material_config_accepts_the_shipped_materials, Fast, Exact) {
    const std::vector<const char*> shipped = {"chrome.json", "clay.json", "glass.json", "principled.json"};
    ctx.plan(static_cast<int>(shipped.size()));
    for (const char* name : shipped) {
        const std::filesystem::path path = std::filesystem::path(ASSET_ROOT_DIR) / "materials" / name;
        char detail[256];
        std::snprintf(detail, sizeof(detail), "loadMaterialConfig rejected the shipped material %s", name);
        PT_EXPECT(ctx, pathtracer::config::loadMaterialConfig(path.string()).has_value(), detail);
    }
}

// Each row names the BSDF failure it prevents; without the loader's bounds these reach the integrator as NaN or negative energy.
PT_CHECK(material_config_rejects_malformed_input, Fast, Exact) {
    struct Case {
        const char* name;
        const char* file;
        std::string text;
    };
    // Mutations of one base that loads, so a row fails for the reason it names rather than vacuously on an earlier parse error.
    const auto material = [](const std::string& overrides) {
        return std::string("{\"diffuseColour\":[1,1,1],\"roughnessFactor\":0.5,\"roughnessMin\":0.045,"
                           "\"roughnessMax\":1.0,\"bumpStrength\":0.0") +
               overrides + "}";
    };

    const std::vector<Case> cases = {
        // std::clamp(v, lo, hi) has undefined behaviour when lo > hi, and resolveRoughness clamps with exactly these two.
        {"roughnessMin above roughnessMax", "engine_io_material_roughrange.json",
         "{\"diffuseColour\":[1,1,1],\"roughnessFactor\":0.5,\"roughnessMin\":0.9,\"roughnessMax\":0.2,\"bumpStrength\":0.0}"},
        // eonUniformMixWeight raises r to the 0.1 power, which is NaN for r < 0, and the NaN reaches the pixel through sampleEon.
        {"negative diffuseRoughness", "engine_io_material_negdiffrough.json", material(",\"diffuseRoughness\":-0.2")},
        // Outside [0,1] the EON quartic albedo fit is extrapolated, where 1 - E can go negative and the BRDF with it.
        {"diffuseRoughness above 1", "engine_io_material_diffroughhigh.json", material(",\"diffuseRoughness\":1.5")},
        // transmissionColor is a transmittance; above 1 sigma_a = -ln(colour)/depth is negative and Beer-Lambert amplifies without bound.
        {"transmissionColor above 1", "engine_io_material_transcolour.json",
         material(",\"transmissionColor\":[1.4,1.0,1.0],\"transmissionDepth\":0.4")},
        // Lobe selection probabilities are mixed by these; outside [0,1] glm::mix extrapolates and the prefix-sum partition goes negative.
        {"metallicFactor above 1", "engine_io_material_metallic.json", material(",\"metallicFactor\":1.5")},
        {"negative transmissionFactor", "engine_io_material_negtrans.json", material(",\"transmissionFactor\":-0.5")},
        // dielectricF0 divides by ior + 1, and every dielectric lobe by the etaI/etaT ratio.
        {"non-positive ior", "engine_io_material_zeroior.json", material(",\"ior\":0.0")},
        // The EON albedo inversion leaves rho unbounded above an albedo of 1, where its multiple-scatter denominator can reach zero.
        {"diffuseColour above 1", "engine_io_material_colour.json",
         "{\"diffuseColour\":[1,2,1],\"roughnessFactor\":0.5,\"roughnessMin\":0.045,\"roughnessMax\":1.0,\"bumpStrength\":0.0}"},
        {"negative transmissionDepth", "engine_io_material_negdepth.json", material(",\"transmissionDepth\":-1.0")},
    };

    const std::filesystem::path basePath = writeJson("engine_io_material_base.json", material(""));
    const bool baseLoads = pathtracer::config::loadMaterialConfig(basePath.string()).has_value();
    std::filesystem::remove(basePath);

    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, baseLoads, "the unmutated base material must load, or every mutated row below passes vacuously");
    for (const Case& testCase : cases) {
        const std::filesystem::path path = writeJson(testCase.file, testCase.text);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadMaterialConfig accepted a material with %s", testCase.name);
        PT_EXPECT(ctx, !pathtracer::config::loadMaterialConfig(path.string()).has_value(), detail);
        std::filesystem::remove(path);
    }
}

PT_CHECK(profile_config_accepts_the_shipped_profile, Fast, Exact) {
    ctx.plan(1);
    const std::filesystem::path profile = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    char detail[256];
    std::snprintf(detail, sizeof(detail), "loadProfileConfig rejected the shipped profile at %s",
                  profile.string().c_str());
    PT_EXPECT(ctx, pathtracer::config::loadProfileConfig(profile.string()).has_value(), detail);
}

// The boundary values profile_config.cpp is responsible for: a resolution divides an aspect ratio, a film-back dimension the FOV.
PT_CHECK(profile_config_rejects_malformed_input, Fast, Exact) {
    struct Case {
        const char* name;
        const char* file;
        std::string text;
    };
    const std::vector<Case> cases = {
        {"not JSON at all", "engine_io_profile_notjson.json", "{ nope"},
        {"empty file", "engine_io_profile_empty.json", ""},
        {"JSON array where an object is required", "engine_io_profile_array.json", "[]"},
    };

    ctx.plan(static_cast<int>(cases.size()) + 1);
    for (const Case& testCase : cases) {
        const std::filesystem::path path = writeJson(testCase.file, testCase.text);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadProfileConfig accepted a profile with %s", testCase.name);
        PT_EXPECT(ctx, !pathtracer::config::loadProfileConfig(path.string()).has_value(), detail);
        std::filesystem::remove(path);
    }

    const std::filesystem::path missing = scratchPath("engine_io_profile_absent.json");
    std::filesystem::remove(missing);
    PT_EXPECT(ctx, !pathtracer::config::loadProfileConfig(missing.string()).has_value(),
                  "loadProfileConfig accepted a path that does not exist");
}

// render.vsync and the two bit depths, each varied alone: every accepted value maps to its own setting, every other is refused.
PT_CHECK(profile_config_render_display_settings, Fast, Exact) {
    using pathtracer::gfx::ScalarType;
    struct Case {
        std::string name;
        const char* key;
        nlohmann::json value;  // null = key removed
        bool accepted;
    };
    std::vector<Case> cases = {
        {"displayBitDepth 16", "displayBitDepth", 16, true},
        {"displayBitDepth 32", "displayBitDepth", 32, true},
        {"textureBitDepth 16", "textureBitDepth", 16, true},
        {"textureBitDepth 32", "textureBitDepth", 32, true},
        {"vsync false", "vsync", false, true},
        {"vsync true", "vsync", true, true},
        {"vsync 1", "vsync", 1, false},
        {"vsync \"true\"", "vsync", "true", false},
        {"vsync missing", "vsync", nullptr, false},
    };
    for (const char* key : {"displayBitDepth", "textureBitDepth"}) {
        for (const nlohmann::json& bad : {nlohmann::json(8), nlohmann::json(24), nlohmann::json(16.5), nlohmann::json("16"),
                                          nlohmann::json(nullptr)}) {
            cases.push_back({std::string(key) + " " + (bad.is_null() ? "missing" : bad.dump()), key, bad, false});
        }
    }

    // Each accepted row varies one key against the shipped profile, so the expectation assumes nothing about what the shipped depths are.
    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    const std::optional<pathtracer::config::ProfileConfig> shippedConfig = pathtracer::config::loadProfileConfig(shippedPath.string());
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, shippedConfig.has_value(), "the shipped profile does not load, so no row below means anything");
    if (!shippedConfig) {
        return;
    }
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        if (testCase.value.is_null()) {
            edited["render"].erase(testCase.key);
        } else {
            edited["render"][testCase.key] = testCase.value;
        }
        const std::filesystem::path path = writeJson("engine_io_profile_render.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        if (testCase.accepted) {
            const bool isDisplay = std::string(testCase.key) == "displayBitDepth";
            const bool isTexture = std::string(testCase.key) == "textureBitDepth";
            const std::optional<ScalarType> varied =
                testCase.value.is_number_integer() ? pathtracer::gfx::scalarTypeFromBitDepth(testCase.value.get<int>())
                                                   : std::nullopt;
            const ScalarType display = isDisplay ? *varied : shippedConfig->render.displayFormat;
            const ScalarType texture = isTexture ? *varied : shippedConfig->render.textureType;
            const bool vsync = isDisplay || isTexture ? shippedConfig->render.vsync : testCase.value.get<bool>();
            std::snprintf(detail, sizeof(detail), "loadProfileConfig rejected or mis-mapped %s", testCase.name.c_str());
            PT_EXPECT(ctx,
                          loaded.has_value() && loaded->render.displayFormat == display &&
                              loaded->render.textureType == texture && loaded->render.vsync == vsync,
                          detail);
        } else {
            std::snprintf(detail, sizeof(detail), "loadProfileConfig accepted %s", testCase.name.c_str());
            PT_EXPECT(ctx, !loaded.has_value(), detail);
        }
    }
}

// camera.lens selects the projection and carries the polynomial; both are validated whichever projection the profile selects.
PT_CHECK(profile_config_camera_lens, Fast, Exact) {
    struct Case {
        std::string name;
        nlohmann::json lens;
        bool accepted;
    };
    // theta_d' = 1 + 3*k1*theta^2 with k1 = -1/3 vanishes at theta = 1 rad, so a 180-degree circle (thetaMax = pi/2) crosses it.
    const nlohmann::json nonMonotone = {{"projection", "fisheyePolynomial"},
                                        {"maxFieldOfViewDegrees", 180.0},
                                        {"radialCoefficients", {-1.0 / 3.0, 0.0, 0.0, 0.0}}};
    const nlohmann::json spherical = {{"projection", "spherical"},
                                      {"maxFieldOfViewDegrees", 180.0},
                                      {"radialCoefficients", {0.0, 0.0, 0.0, 0.0}}};
    nlohmann::json fisheye = spherical;
    fisheye["projection"] = "fisheyePolynomial";
    nlohmann::json threeCoefficients = spherical;
    threeCoefficients["radialCoefficients"] = {0.0, 0.0, 0.0};
    nlohmann::json zeroFov = spherical;
    zeroFov["maxFieldOfViewDegrees"] = 0.0;
    nlohmann::json overFov = spherical;
    overFov["maxFieldOfViewDegrees"] = 360.5;
    nlohmann::json unknownProjection = spherical;
    unknownProjection["projection"] = "pinhole";
    nlohmann::json missingProjection = spherical;
    missingProjection.erase("projection");

    const std::vector<Case> cases = {
        {"spherical with a zero polynomial", spherical, true},
        {"fisheyePolynomial with an equidistant polynomial", fisheye, true},
        {"an unknown projection name", unknownProjection, false},
        {"no projection at all", missingProjection, false},
        {"three coefficients instead of four", threeCoefficients, false},
        {"a zero field of view", zeroFov, false},
        {"a field of view past 360 degrees", overFov, false},
        // Rejected under a spherical projection too: the HUD can switch to the fisheye at runtime, so the polynomial must hold either way.
        {"a non-monotone polynomial", nonMonotone, false},
    };

    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()));
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        edited["camera"]["lens"] = testCase.lens;
        const std::filesystem::path path = writeJson("engine_io_profile_lens.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded =
            pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadProfileConfig %s %s",
                      testCase.accepted ? "rejected" : "accepted", testCase.name.c_str());
        PT_EXPECT(ctx, loaded.has_value() == testCase.accepted, detail);
    }
}

// render.defaultAOV is a raw index main.cpp dereferences unchecked, so the bound holds at load: both ends plus the first past the top.
PT_CHECK(profile_config_default_aov_is_in_range, Fast, Exact) {
    const int aovCount = static_cast<int>(pathtracer::debug::AovId::Count);
    const std::vector<std::pair<nlohmann::json, bool>> cases = {
        {0, true}, {aovCount - 1, true}, {aovCount, false}, {-1, false}, {nlohmann::json(nullptr), false},
    };

    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, pathtracer::config::loadProfileConfig(shippedPath.string()).has_value(),
                  "the shipped profile does not load, so no row below means anything");
    for (const auto& [value, accepted] : cases) {
        nlohmann::json edited = shipped;
        if (value.is_null()) {
            edited["render"].erase("defaultAOV");
        } else {
            edited["render"]["defaultAOV"] = value;
        }
        const std::filesystem::path path = writeJson("engine_io_profile_defaultaov.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadProfileConfig %s defaultAOV %s",
                      accepted ? "rejected" : "accepted", value.dump().c_str());
        PT_EXPECT(ctx, loaded.has_value() == accepted && (!accepted || loaded->render.defaultAov == value.get<int>()),
                      detail);
    }
}

// The two scene-scale distances, each a divisor at its point of use: at or below zero the lane is inf/NaN, not a bounded gradient.
PT_CHECK(profile_config_scene_scale_distances, Fast, Exact) {
    struct Case {
        std::string name;
        const char* key;
        nlohmann::json value;  // null = key removed
        bool accepted;
    };
    std::vector<Case> cases;
    for (const char* key : {"aoMaxDistance", "lookaheadDistance"}) {
        cases.push_back({std::string(key) + " 0.5", key, 0.5, true});
        for (const nlohmann::json& bad : {nlohmann::json(0.0), nlohmann::json(-1.0), nlohmann::json("1.0"),
                                          nlohmann::json(nullptr)}) {
            cases.push_back({std::string(key) + " " + (bad.is_null() ? "missing" : bad.dump()), key, bad, false});
        }
    }

    const auto distanceOf = [](const pathtracer::config::PathTracerConfig& config, const char* key) {
        return std::string(key) == "aoMaxDistance" ? config.aoMaxDistance : config.lookaheadDistance;
    };
    const auto otherKey = [](const char* key) {
        return std::string(key) == "aoMaxDistance" ? "lookaheadDistance" : "aoMaxDistance";
    };

    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    const std::optional<pathtracer::config::ProfileConfig> shippedConfig =
        pathtracer::config::loadProfileConfig(shippedPath.string());
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, shippedConfig.has_value(), "the shipped profile does not load, so no row below means anything");
    if (!shippedConfig) {
        return;
    }
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        if (testCase.value.is_null()) {
            edited["pathTracer"].erase(testCase.key);
        } else {
            edited["pathTracer"][testCase.key] = testCase.value;
        }
        const std::filesystem::path path = writeJson("engine_io_profile_distances.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        if (testCase.accepted) {
            std::snprintf(detail, sizeof(detail), "loadProfileConfig rejected or mis-mapped %s",
                          testCase.name.c_str());
            PT_EXPECT(ctx,
                          loaded.has_value() &&
                              distanceOf(loaded->pathTracer, testCase.key) == testCase.value.get<float>() &&
                              distanceOf(loaded->pathTracer, otherKey(testCase.key)) ==
                                  distanceOf(shippedConfig->pathTracer, otherKey(testCase.key)),
                          detail);
        } else {
            std::snprintf(detail, sizeof(detail), "loadProfileConfig accepted %s", testCase.name.c_str());
            PT_EXPECT(ctx, !loaded.has_value(), detail);
        }
    }
}

// The integer counts nothing downstream re-checks. samplesPerPixel 0 is the sharp one: it divides by a zero filter weight, writing NaN.
PT_CHECK(profile_config_integer_counts, Fast, Exact) {
    struct Case {
        std::string name;
        const char* section;
        const char* key;
        nlohmann::json value;  // null = key removed
        bool accepted;
    };
    std::vector<Case> cases;
    // Lowest legal value accepted, then each way of going under it. maxBounces/maxSamples 0 are meaningful: direct-only and unbounded.
    const std::vector<std::pair<const char*, int>> floors = {
        {"samplesPerPixel", 1}, {"maxBounces", 0}, {"russianRouletteStartBounce", 0}, {"maxSamples", 0}};
    for (const auto& [key, floor] : floors) {
        cases.push_back({std::string(key) + " " + std::to_string(floor), "pathTracer", key, floor, true});
        cases.push_back({std::string(key) + " " + std::to_string(floor - 1), "pathTracer", key, floor - 1, false});
        cases.push_back({std::string(key) + " missing", "pathTracer", key, nlohmann::json(nullptr), false});
    }
    // The traced image renderScale multiplies and the denominator of the primary ray's aspect ratio.
    for (const char* key : {"width", "height"}) {
        cases.push_back({std::string("render.") + key + " 1", "render", key, 1, true});
        cases.push_back({std::string("render.") + key + " 0", "render", key, 0, false});
        cases.push_back({std::string("render.") + key + " -1", "render", key, -1, false});
        cases.push_back({std::string("render.") + key + " missing", "render", key, nlohmann::json(nullptr), false});
    }

    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    const std::optional<pathtracer::config::ProfileConfig> shippedConfig =
        pathtracer::config::loadProfileConfig(shippedPath.string());
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, shippedConfig.has_value(), "the shipped profile does not load, so no row below means anything");
    if (!shippedConfig) {
        return;
    }
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        if (testCase.value.is_null()) {
            edited[testCase.section].erase(testCase.key);
        } else {
            edited[testCase.section][testCase.key] = testCase.value;
        }
        const std::filesystem::path path = writeJson("engine_io_profile_counts.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadProfileConfig %s %s", testCase.accepted ? "rejected" : "accepted",
                      testCase.name.c_str());
        PT_EXPECT(ctx, loaded.has_value() == testCase.accepted, detail);
    }
}

// loadImageTexture's typed read: Float16 must equal the IEEE round-to-nearest-even cast, and a source above kHalfMax is rejected there.
PT_CHECK(image_texture_half_load, Fast, Exact) {
    using pathtracer::gfx::ScalarType;
    const auto writeProbe = [](const char* name, const std::vector<float>& values) {
        pathtracer::gfx::HdrImage image{static_cast<int>(values.size()), 1, {}};
        for (const float v : values) {
            image.rgba.insert(image.rgba.end(), {v, v, v, 1.0F});
        }
        const std::filesystem::path path = scratchPath(name);
        return pathtracer::gfx::writeExr(path.string(), image) ? std::optional(path) : std::nullopt;
    };
    const auto sameTexels = [](const pathtracer::gfx::ImageTexture& a, const std::vector<float>& expected) {
        for (int x = 0; x < a.width; ++x) {
            const glm::vec4 texel = a.texel(x, 0);
            if (texel.r != expected[static_cast<std::size_t>(x)] || texel.b != expected[static_cast<std::size_t>(x)]) {
                return false;
            }
        }
        return true;
    };
    ctx.plan(5);

    // Exact in binary16: small integers, dyadic fractions, the largest finite value, the smallest normal.
    const std::vector<float> exact = {0.0F, 1.0F, 1.5F, 0.25F, 2048.0F, -3.0F, pathtracer::gfx::kHalfMax, 1.0F / 16384.0F};
    const std::optional<std::filesystem::path> exactPath = writeProbe("engine_io_half_exact.exr", exact);
    const std::optional<pathtracer::gfx::ImageTexture> exact16 =
        exactPath ? pathtracer::gfx::loadImageTexture(exactPath->string(), ScalarType::Float16) : std::nullopt;
    const std::optional<pathtracer::gfx::ImageTexture> exact32 =
        exactPath ? pathtracer::gfx::loadImageTexture(exactPath->string(), ScalarType::Float32) : std::nullopt;
    PT_EXPECT(ctx, exact16 && exact32 && std::holds_alternative<std::vector<pathtracer::gfx::Half>>(exact16->texels) && sameTexels(*exact16, exact) &&
                           sameTexels(*exact32, exact),
                  "binary16-exact values did not load bit-identically at Float16 and Float32");

    // Arbitrary: needs rounding, including the exact midpoint above 1.0 (ties to even), a subnormal, and one below the smallest subnormal.
    const std::vector<float> arbitrary = {0.1F, 1.0F + pathtracer::gfx::kHalfUnitRoundoff, 3.14159265F, 1.0e-6F, 1.0e-20F, 60000.5F};
    std::vector<float> rounded;
    for (const float v : arbitrary) {
        rounded.push_back(static_cast<float>(static_cast<pathtracer::gfx::Half>(v)));
    }
    const std::optional<std::filesystem::path> arbitraryPath = writeProbe("engine_io_half_arbitrary.exr", arbitrary);
    const std::optional<pathtracer::gfx::ImageTexture> arbitrary16 =
        arbitraryPath ? pathtracer::gfx::loadImageTexture(arbitraryPath->string(), ScalarType::Float16) : std::nullopt;
    PT_EXPECT(ctx, arbitrary16 && sameTexels(*arbitrary16, rounded),
                  "Float16 load differs from static_cast<Half> (round to nearest even)");
    const std::optional<pathtracer::gfx::ImageTexture> arbitrary32 =
        arbitraryPath ? pathtracer::gfx::loadImageTexture(arbitraryPath->string(), ScalarType::Float32) : std::nullopt;
    PT_EXPECT(ctx, arbitrary32 && sameTexels(*arbitrary32, arbitrary), "Float32 load is not an exact copy");

    // Overflow: 70000 is finite in float and beyond kHalfMax, so Float16 must reject it and Float32 must not.
    const std::optional<std::filesystem::path> overPath = writeProbe("engine_io_half_overflow.exr", {1.0F, 70000.0F});
    PT_EXPECT(ctx, overPath && !pathtracer::gfx::loadImageTexture(overPath->string(), ScalarType::Float16),
                  "Float16 accepted a texel above binary16's finite max");
    PT_EXPECT(ctx, overPath && pathtracer::gfx::loadImageTexture(overPath->string(), ScalarType::Float32),
                  "Float32 rejected a finite texel");
    for (const std::optional<std::filesystem::path>& path : {exactPath, arbitraryPath, overPath}) {
        if (path) {
            std::filesystem::remove(*path);
        }
    }
}

// sampleBilinear at Float16 against Float32: the bound is derived from u = 2^-11 and gamma_6 (Higham 2002, 3.1), never fitted.
PT_CHECK(image_texture_bilinear_half_bound, Fast, Exact) {
    using pathtracer::gfx::ScalarType;
    constexpr int kWidth = 13;
    constexpr int kHeight = 7;
    constexpr int kSamples = 20000;
    const double unitRoundoff = std::numeric_limits<float>::epsilon() / 2.0;
    const double gamma6 = 6.0 * unitRoundoff / (1.0 - (6.0 * unitRoundoff));
    const double bound = pathtracer::gfx::kHalfUnitRoundoff + (2.0 * gamma6);

    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    // Normal range of binary16: [2^-14, kHalfMax], log-uniform so every binade is exercised.
    std::uniform_real_distribution<float> logValue(-14.0F, std::log2(pathtracer::gfx::kHalfMax));
    std::vector<float> rgba(static_cast<std::size_t>(kWidth) * kHeight * 4);
    for (float& v : rgba) {
        v = std::exp2(logValue(rng));
    }
    const pathtracer::gfx::ImageTexture full{kWidth, kHeight, rgba};
    const pathtracer::gfx::ImageTexture half{kWidth, kHeight, std::vector<pathtracer::gfx::Half>(rgba.begin(), rgba.end())};

    std::uniform_real_distribution<float> unit(-1.0F, 2.0F);  // beyond [0,1] so both wrap directions are covered
    double worst = 0.0;
    double largest = 0.0;
    for (int i = 0; i < kSamples; ++i) {
        const glm::vec2 uv(unit(rng), unit(rng));
        const glm::vec4 s32 = pathtracer::gfx::sampleBilinear(full, uv);
        const glm::vec4 s16 = pathtracer::gfx::sampleBilinear(half, uv);
        for (int c = 0; c < 4; ++c) {
            const double relative = std::fabs(static_cast<double>(s16[c]) - s32[c]) / s32[c];
            worst = std::max(worst, relative / bound);
            largest = std::max(largest, relative);
        }
    }
    ctx.plan(2);
    char detail[192];
    std::snprintf(detail, sizeof(detail), "worst |s16 - s32| / s32 is %.4g of the derived bound %.4g", worst, bound);
    PT_EXPECT(ctx, worst <= 1.0, detail);
    PT_EXPECT(ctx, largest > 0.0, "Float16 and Float32 samples never differed: the half path was not exercised");
}

// The film-back catalogue's contract: every preset's dimensions feed verticalFovRadians() as a denominator and an aspect ratio.
PT_CHECK(film_back_presets_are_physically_valid, Fast, Exact) {
    ctx.plan(2);
    const std::filesystem::path sensor = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "sensor.json";
    const std::optional<std::vector<pathtracer::scene::Camera::FilmBackPreset>> presets =
        pathtracer::config::loadFilmBackPresets(sensor.string());
    if (!presets.has_value()) {
        PT_EXPECT(ctx, false, "loadFilmBackPresets rejected the shipped sensor.json");
        PT_EXPECT(ctx, false, "presets unavailable");
        return;
    }
    PT_EXPECT(ctx, !presets->empty(), "the shipped film-back catalogue is empty");
    bool allPositive = true;
    for (const pathtracer::scene::Camera::FilmBackPreset& preset : *presets) {
        allPositive = allPositive && preset.filmBack.widthMm > 0.0F && preset.filmBack.heightMm > 0.0F;
    }
    PT_EXPECT(ctx, allPositive, "a film-back preset has a non-positive dimension");
}

// Every appended record is exactly one line that parses back with every schema field, and appending never rewrites earlier lines.
PT_CHECK(bench_log_appends_one_parseable_line_per_record, Fast, Exact) {
    ctx.plan(5);
    const std::filesystem::path path = scratchPath("engine_io_validate_bench.jsonl");
    std::filesystem::remove(path);
    const pathtracer::debug::BenchRecord first{"io_validate", {"io_validate", "--flag"}, {{"width", 7}}, {{"ms", {1.5, 2.5}}}, {{"crc32", 1}}};
    const pathtracer::debug::BenchRecord second{"io_validate", {"io_validate"}, {{"width", 9}}, {{"ms", {3.0}}}, {{"crc32", 2}}};
    PT_EXPECT(ctx, pathtracer::debug::appendBenchRecord(path.string(), first) && pathtracer::debug::appendBenchRecord(path.string(), second), "appendBenchRecord failed on a writable scratch path");

    std::vector<nlohmann::json> records;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        records.push_back(nlohmann::json::parse(line, nullptr, /*allow_exceptions=*/false));
    }
    PT_EXPECT(ctx, records.size() == 2, "expected exactly two lines after two appends");

    const auto complete = [](const nlohmann::json& r) {
        return !r.is_discarded() && r.value("schema", 0) == pathtracer::debug::kBenchLogSchema && r.contains("time_utc") && r.contains("pid") &&
               !r["build"].value("git", std::string()).empty() && r["build"].value("uuid", std::string()).size() == 32 &&
               r["host"].value("logical_cpus", 0) > 0 && r["rusage"].contains("user_s") && r["rusage"].contains("nivcsw");
    };
    PT_EXPECT(ctx, records.size() == 2 && complete(records[0]) && complete(records[1]), "a record is missing a provenance or rusage field");
    PT_EXPECT(ctx, records.size() == 2 && records[0]["config"] == first.config && records[0]["samples"] == first.samples && records[0]["argv"] == first.argv,
                  "first record's caller-supplied content did not round-trip");
    PT_EXPECT(ctx, records.size() == 2 && records[1]["config"] == second.config && records[1]["work"] == second.work,
                  "second record's caller-supplied content did not round-trip");
    std::filesystem::remove(path);
}

// A path that cannot be opened is reported as a failure, never as a silently skipped record.
PT_CHECK(bench_log_rejects_unwritable_path, Fast, Exact) {
    ctx.plan(1);
    const std::filesystem::path path = scratchPath("engine_io_validate_no_such_dir") / "bench.jsonl";
    std::filesystem::remove_all(path.parent_path());
    const pathtracer::debug::BenchRecord record{"io_validate", {}, nlohmann::json::object(), nlohmann::json::object(), nlohmann::json::object()};
    PT_EXPECT(ctx, !pathtracer::debug::appendBenchRecord(path.string(), record), "appendBenchRecord reported success writing into a missing directory");
}

// Known answers from an independent implementation (Python's zlib.crc32 over the same little-endian bytes).
PT_CHECK(float_crc32_matches_reference, Fast, Exact) {
    ctx.plan(2);
    const std::vector<float> zero{0.0F};
    const std::vector<float> mixed{1.0F, -2.5F, 0.1F};
    PT_EXPECT(ctx, pathtracer::debug::floatCrc32(zero) == 0x2144DF1CU, "CRC-32 of four zero bytes is not 0x2144DF1C");
    PT_EXPECT(ctx, pathtracer::debug::floatCrc32(mixed) == 2706677804U, "CRC-32 of {1, -2.5, 0.1} disagrees with zlib.crc32");
}

}  // namespace

PT_CHECK_MAIN("io")
