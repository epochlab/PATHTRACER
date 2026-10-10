#include "pathtracer/config/scene_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <vector>

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

// An optional string key: absent is nullopt, any other type a json type_error the caller reports.
std::optional<std::string> optionalString(const nlohmann::json& object, const char* key) {
    const auto it = object.find(key);
    return it == object.end() ? std::nullopt : std::optional(it->get<std::string>());
}

// "r", "g" or "b" as a channel offset; any other value a json error, a typo never silently reading R.
int channelIndex(const nlohmann::json& value) {
    const std::string name = value.get<std::string>();
    constexpr std::string_view kChannels = "rgb";
    const std::size_t index = kChannels.find(name);
    if (name.size() != 1 || index == std::string_view::npos) {
        throw nlohmann::json::type_error::create(302, "texture channel '" + name + "' is not one of r, g, b", &value);
    }
    return static_cast<int>(index);
}

// One binding: a path string, or {path, colorSpace, channel, bump: {path, height}} under a closed key set.
std::optional<TextureConfig> parseTexture(const nlohmann::json& binding, const std::string& path) {
    if (binding.is_string()) {
        return TextureConfig{binding.get<std::string>(), std::nullopt, 0, std::nullopt};
    }
    if (!onlyKnownKeys(binding, {"path", "colorSpace", "channel", "bump"}, path, "texture")) {
        return std::nullopt;
    }
    TextureConfig texture{optionalString(binding, "path"), optionalString(binding, "colorSpace"), 0, std::nullopt};
    if (const auto it = binding.find("channel"); it != binding.end()) {
        texture.channel = channelIndex(*it);
    }
    if (const auto it = binding.find("bump"); it != binding.end()) {
        if (!onlyKnownKeys(*it, {"path", "height"}, path, "bump")) {
            return std::nullopt;
        }
        texture.bump = BumpConfig{it->at("path").get<std::string>(), toFloat(it->at("height"))};
    }
    return texture;
}

// The optional "textures" block: node -> OpenPBR input name -> binding. Input names are checked where the inputs are known.
std::optional<std::map<std::string, std::map<std::string, TextureConfig>>> parseTextures(const nlohmann::json& j, const std::string& path) {
    std::map<std::string, std::map<std::string, TextureConfig>> textures;
    const auto it = j.find("textures");
    if (it == j.end()) {
        return textures;
    }
    for (const auto& [node, slots] : it->get<std::map<std::string, nlohmann::json>>()) {
        for (const auto& [slot, binding] : slots.get<std::map<std::string, nlohmann::json>>()) {
            std::optional<TextureConfig> texture = parseTexture(binding, path);
            if (!texture) {
                return std::nullopt;
            }
            textures[node][slot] = std::move(*texture);
        }
    }
    return textures;
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
        if (!onlyKnownKeys(light, {"type", "position", "rotation", "size", "color", "intensity", "twoSided"}, path, "light")) {
            return std::nullopt;
        }
        const QuadLightConfig quad{
            light.at("position").get<glm::vec3>(),
            light.at("rotation").get<glm::vec3>(),
            light.at("size").get<glm::vec2>(),
            light.at("color").get<glm::vec3>(),
            toFloat(light.at("intensity")),
            light.value("twoSided", false),
        };
        // A zero extent subtends no solid angle; a negative one mirrors the quad and flips which face emits.
        if (!(quad.size.x > 0.0F) || !(quad.size.y > 0.0F)) {
            std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] has a non-positive size\n";
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

// One input's constant: in its specification range per channel, and a bulk material's volumetric switch at its zero default.
template <typename T>
bool validInput(const pathtracer::scene::InputSpec& spec, const T& value, const T& fallback, bool thinWalled, const std::string& path) {
    const glm::vec3 channels(value);
    for (int c = 0; c < (std::is_same_v<T, float> ? 1 : 3); ++c) {
        if (!pathtracer::scene::inRange(channels[c], spec.range)) {
            std::cerr << "loadMaterialConfig: " << path << ": " << spec.name << " " << channels[c] << " is outside its OpenPBR range\n";
            return false;
        }
    }
    if (spec.requiresVolumes && !thinWalled && value != fallback) {
        std::cerr << "loadMaterialConfig: " << path << ": " << spec.name << " requires volumetric transport or geometry_thin_walled\n";
        return false;
    }
    return true;
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
            !onlyKnownKeys(environment, {"hdriPath", "lightEnabled", "colorSpace"}, path, "environment")) {
            return std::nullopt;
        }

        std::map<std::string, std::string> materialOverrides;
        if (const auto it = j.find("materialOverrides"); it != j.end()) {
            materialOverrides = it->get<std::map<std::string, std::string>>();
        }
        std::optional<std::map<std::string, std::map<std::string, TextureConfig>>> textures = parseTextures(j, path);
        std::optional<std::vector<QuadLightConfig>> lights = parseQuadLights(j, path);
        if (!textures || !lights) {
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
                optionalString(environment, "colorSpace"),
            },
            j.at("materialPath").get<std::string>(),
            std::move(materialOverrides),
            std::move(*textures),
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

        // Closed key set: a legacy or misspelt key would otherwise load as the input's default, silently.
        const MaterialConfig defaults;
        std::vector<std::string_view> names{"geometry_thin_walled"};
        pathtracer::scene::forEachInput([&](const pathtracer::scene::InputSpec& spec, const auto&) { names.push_back(spec.name); },
                                        defaults);
        for (const auto& item : j.items()) {
            if (std::find(names.begin(), names.end(), item.key()) == names.end()) {
                std::cerr << "loadMaterialConfig: " << path << ": '" << item.key() << "' is not an OpenPBR input\n";
                return std::nullopt;
            }
        }
        MaterialConfig material;
        // Read first: whether the volumetric switches are accepted depends on it.
        material.geometryThinWalled = j.value("geometry_thin_walled", false);
        bool ok = true;
        pathtracer::scene::forEachInput(
            [&](const pathtracer::scene::InputSpec& spec, auto& value, const auto& fallback) {
                const auto it = j.find(spec.name);
                if (it == j.end()) {
                    return;
                }
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>, float>) {
                    value = toFloat(*it);
                } else {
                    value = it->template get<glm::vec3>();
                }
                ok = validInput(spec, value, fallback, material.geometryThinWalled, path) && ok;
            },
            material, defaults);
        if (!ok) {
            return std::nullopt;
        }
        return material;
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "loadMaterialConfig: " << path << ": " << e.what() << '\n';
        return std::nullopt;
    }
}

}  // namespace pathtracer::config
