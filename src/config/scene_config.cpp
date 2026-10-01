#include "pathtracer/config/scene_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string_view>

#include <nlohmann/json.hpp>

#include "json_glm.h"

namespace pathtracer::config {

namespace {

// Closed key set: a misspelt or retired key (textureOverrides, texturePath) would otherwise load as its default, silently.
bool onlyKnownKeys(const nlohmann::json& object, std::initializer_list<std::string_view> known, const std::string& path,
                   const char* where) {
    for (const auto& item : object.items()) {
        if (std::find(known.begin(), known.end(), item.key()) == known.end()) {
            std::cerr << "loadSceneConfig: " << path << ": unknown " << where << " key '" << item.key() << "'\n";
            return false;
        }
    }
    return true;
}

// Parses and validates the optional "lights" array. Separate so each stays one screen, and every check is an authoring boundary.
std::optional<std::vector<QuadLightConfig>> parseQuadLights(const nlohmann::json& j, const std::string& path) {
    std::vector<QuadLightConfig> lights;
    const auto it = j.find("lights");
    if (it == j.end()) {
        return lights;
    }
    for (const nlohmann::json& light : *it) {
        const std::size_t index = lights.size();
        // Dispatched on rather than ignored: an unrecognised type must not silently load as a quad. A future light type branches here.
        if (const auto type = light.at("type").get<std::string>(); type != "quad") {
            std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] has unknown type '" << type << "', only 'quad' exists\n";
            return std::nullopt;
        }
        const QuadLightConfig quad{
            light.at("origin").get<glm::vec3>(),
            light.at("edge0").get<glm::vec3>(),
            light.at("edge1").get<glm::vec3>(),
            light.at("color").get<glm::vec3>(),
            light.at("intensity").get<float>(),
            light.value("twoSided", false),
        };
        const float length0 = glm::length(quad.edge0);
        const float length1 = glm::length(quad.edge1);
        // A zero-length edge subtends no solid angle: degenerate geometry in the BVH emitting nothing NEE could sample.
        if (!(length0 > 0.0F) || !(length1 > 0.0F)) {
            std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] has a zero-length edge0/edge1\n";
            return std::nullopt;
        }
        // The sampler's frame is normalize(edge0)/normalize(edge1); bound is its worst-case error, |cos| 6.1e-3 at 3dp and 9.9e-8 at 8dp.
        constexpr float kMaxEdgeCosine = 1e-4F;
        if (const float cosEdges = glm::dot(quad.edge0 / length0, quad.edge1 / length1); std::fabs(cosEdges) > kMaxEdgeCosine) {
            std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] has non-perpendicular edge0/edge1 (cos " << cosEdges << ")\n";
            return std::nullopt;
        }
        // Negative radiance is unrepresentable and would propagate through NEE as a permanent bias no downstream clamp removes.
        if (quad.intensity < 0.0F || glm::any(glm::lessThan(quad.color, glm::vec3(0.0F)))) {
            std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] has a negative intensity/color\n";
            return std::nullopt;
        }
        lights.push_back(quad);
    }
    return lights;
}

// Authored material bounds. Every field here reaches the BSDF unclamped, where out-of-range input is NaN or a negative lobe weight.
bool validMaterialConfig(const MaterialConfig& m, const std::string& path) {
    bool ok = true;
    // Negated comparisons throughout, as parseQuadLights uses: a NaN fails every one of them instead of slipping through.
    const auto unit = [&](const char* name, float v) {
        if (!(v >= 0.0F && v <= 1.0F)) {
            std::cerr << "loadMaterialConfig: " << path << ": " << name << " is " << v << ", expected [0,1]\n";
            ok = false;
        }
    };
    const auto unitRgb = [&](const char* name, const glm::vec3& v) {
        if (!glm::all(glm::greaterThanEqual(v, glm::vec3(0.0F))) || !glm::all(glm::lessThanEqual(v, glm::vec3(1.0F)))) {
            std::cerr << "loadMaterialConfig: " << path << ": " << name << " is (" << v.x << ", " << v.y << ", " << v.z
                      << "), expected [0,1] per channel\n";
            ok = false;
        }
    };
    const auto atLeast = [&](const char* name, float v, float low) {
        if (!(v >= low)) {
            std::cerr << "loadMaterialConfig: " << path << ": " << name << " is " << v << ", expected >= " << low << "\n";
            ok = false;
        }
    };

    // Energy fractions: metallic and transmissionFactor weight lobe probabilities, which glm::mix extrapolates negative outside [0,1].
    unit("roughnessMin", m.roughnessMin);
    unit("roughnessMax", m.roughnessMax);
    unit("metallicFactor", m.metallicFactor);
    unit("transmissionFactor", m.transmissionFactor);
    // EON's quartic albedo fit and eonUniformMixWeight's pow(r, 0.1) are defined on [0,1] only; a negative r makes that pow NaN.
    unit("diffuseRoughness", m.diffuseRoughness);
    // Reflectance, transmittance and Gulbrandsen edgetint are all fractions; above 1 the EON albedo inversion leaves rho unbounded.
    unitRgb("diffuseColour", m.diffuseColour);
    unitRgb("transmissionColor", m.transmissionColor);
    unitRgb("edgeTint", m.edgeTint);
    // Not bounded above: it multiplies the texture sample before the roughnessMin/Max clamp, which bounds the result anyway.
    atLeast("roughnessFactor", m.roughnessFactor, 0.0F);
    // A negative depth is rejected rather than treated as "no medium", which is what transmissionDepth == 0 already means.
    atLeast("transmissionDepth", m.transmissionDepth, 0.0F);
    // abbe <= 0 is the documented "no dispersion" case cauchyIor tests for, so only a non-finite value is wrong here.
    atLeast("abbe", m.abbe, 0.0F);
    // Denominator of dielectricF0's (ior-1)/(ior+1) and the etaI/etaT ratio every dielectric lobe divides by.
    if (!(m.ior > 0.0F)) {
        std::cerr << "loadMaterialConfig: " << path << ": ior is " << m.ior << ", expected > 0\n";
        ok = false;
    }
    // Scales a raw height difference either way, so sign is free; only a non-finite value would reach normalize() as NaN.
    if (!std::isfinite(m.bumpStrength)) {
        std::cerr << "loadMaterialConfig: " << path << ": bumpStrength is not finite\n";
        ok = false;
    }
    // resolveRoughness clamps with these as lo/hi, and std::clamp has undefined behaviour when lo > hi.
    if (!(m.roughnessMin <= m.roughnessMax)) {
        std::cerr << "loadMaterialConfig: " << path << ": roughnessMin " << m.roughnessMin << " exceeds roughnessMax "
                  << m.roughnessMax << "\n";
        ok = false;
    }
    return ok;
}

// A constant material scatters nothing, so any BSDF key would be silently dead: rejected rather than ignored.
std::optional<MaterialConfig> parseConstantMaterial(const nlohmann::json& j, const std::string& path) {
    for (const auto& item : j.items()) {
        if (item.key() != "shadingModel" && item.key() != "diffuseColour") {
            std::cerr << "loadMaterialConfig: " << path << ": '" << item.key() << "' has no effect on a constant material\n";
            return std::nullopt;
        }
    }
    // BSDF fields at identity: the rasterizer still resolves them for G-buffer AOVs, and resolveRoughness clamps by min <= max.
    MaterialConfig material{
        .bumpStrength = 0.0F,
        .roughnessMin = 0.0F,
        .roughnessMax = 1.0F,
        .diffuseColour = j.value("diffuseColour", glm::vec3(1.0F)),
        .roughnessFactor = 1.0F,
        .shadingModel = pathtracer::scene::ShadingModel::Constant,
    };
    if (!validMaterialConfig(material, path)) {
        return std::nullopt;
    }
    return material;
}

}  // namespace

std::optional<SceneConfig> loadSceneConfig(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "loadSceneConfig: could not read " << path << '\n';
        return std::nullopt;
    }

    try {
        nlohmann::json j;
        file >> j;

        const nlohmann::json& model = j.at("model");
        const nlohmann::json& environment = j.at("environment");
        if (!onlyKnownKeys(j, {"model", "environment", "materialPath", "materialOverrides", "textures", "lights"}, path,
                           "top-level") ||
            !onlyKnownKeys(model, {"gltfPath", "position", "rotation"}, path, "model") ||
            !onlyKnownKeys(environment, {"hdriPath", "lightEnabled"}, path, "environment")) {
            return std::nullopt;
        }

        std::map<std::string, std::string> materialOverrides;
        if (const auto it = j.find("materialOverrides"); it != j.end()) {
            materialOverrides = it->get<std::map<std::string, std::string>>();
        }
        std::map<std::string, std::map<std::string, std::string>> textures;
        if (const auto it = j.find("textures"); it != j.end()) {
            textures = it->get<std::map<std::string, std::map<std::string, std::string>>>();
        }

        std::optional<std::vector<QuadLightConfig>> lights = parseQuadLights(j, path);
        if (!lights) {
            return std::nullopt;
        }

        return SceneConfig{
            ModelConfig{
                model.at("gltfPath").get<std::string>(),
                model.at("position").get<glm::vec3>(),
                model.at("rotation").get<glm::vec3>(),
            },
            EnvironmentConfig{
                environment.at("hdriPath").get<std::string>(),
                environment.value("lightEnabled", true),
            },
            j.at("materialPath").get<std::string>(),
            std::move(materialOverrides),
            std::move(textures),
            std::move(*lights),
        };
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "loadSceneConfig: " << path << ": " << e.what() << '\n';
        return std::nullopt;
    }
}

std::optional<MaterialConfig> loadMaterialConfig(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "loadMaterialConfig: could not read " << path << '\n';
        return std::nullopt;
    }

    try {
        nlohmann::json j;
        file >> j;

        const auto shadingModel = j.value("shadingModel", std::string("standard"));
        if (shadingModel == "constant") {
            return parseConstantMaterial(j, path);
        }
        if (shadingModel != "standard") {
            std::cerr << "loadMaterialConfig: " << path << ": unknown shadingModel '" << shadingModel
                      << "', expected 'standard' or 'constant'\n";
            return std::nullopt;
        }

        const MaterialConfig material{
            j.at("bumpStrength").get<float>(),
            j.at("roughnessMin").get<float>(),
            j.at("roughnessMax").get<float>(),
            j.at("diffuseColour").get<glm::vec3>(),
            j.value("ior", 1.5F),
            j.value("abbe", 0.0F),
            j.value("transmissionFactor", 0.0F),
            j.value("metallicFactor", 0.0F),
            j.at("roughnessFactor").get<float>(),
            j.value("diffuseRoughness", 0.0F),
            j.value("transmissionColor", glm::vec3(1.0F)),
            j.value("transmissionDepth", 0.0F),
            j.value("edgeTint", glm::vec3(1.0F)),
        };
        if (!validMaterialConfig(material, path)) {
            return std::nullopt;
        }
        return material;
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "loadMaterialConfig: " << path << ": " << e.what() << '\n';
        return std::nullopt;
    }
}

}  // namespace pathtracer::config
