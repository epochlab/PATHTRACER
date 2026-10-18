#include "pathtracer/config/scene_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "json_glm.h"
#include "pathtracer/scene/cie.h"

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

// A JSON unsigned integer in [0, 2^32 - 1]; a float, negative or wider value is a json error, never truncated.
std::uint32_t uint32Of(const nlohmann::json& value) {
    const std::uint64_t integer = value.is_number_unsigned() ? value.get<std::uint64_t>() : 0U;
    if (!value.is_number_unsigned() || integer > std::numeric_limits<std::uint32_t>::max()) {
        throw nlohmann::json::type_error::create(302, value.dump() + " is not an integer in [0, 2^32 - 1]", &value);
    }
    return static_cast<std::uint32_t>(integer);
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

// Exactly one of an authored scene-space color or a Planckian temperature in kelvin, at unit luminance so luminance is the meter's.
std::optional<glm::vec3> lightColor(const nlohmann::json& light, const std::string& path, std::size_t index) {
    const auto color = light.find("color");
    const auto temperature = light.find("temperature");
    if ((color == light.end()) == (temperature == light.end())) {
        std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] needs exactly one of color or temperature\n";
        return std::nullopt;
    }
    if (color != light.end()) {
        const auto rgb = color->get<glm::vec3>();
        // Negative radiance is unrepresentable and would propagate through NEE as a permanent bias no downstream clamp removes.
        const float luminance = glm::dot(pathtracer::scene::cie::sceneLuminanceWeights(), rgb);
        if (glm::any(glm::lessThan(rgb, glm::vec3(0.0F))) || !(luminance > 0.0F)) {
            std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] color is negative or black\n";
            return std::nullopt;
        }
        return rgb / luminance;
    }
    const double kelvin = temperature->get<double>();
    const std::optional<glm::dvec3> rgb = pathtracer::scene::cie::blackbodyToScene(kelvin);
    if (!rgb) {
        std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] temperature " << kelvin
                  << " K is not positive or lies outside the scene gamut\n";
        return std::nullopt;
    }
    return glm::vec3(*rgb);
}

// Exactly one of a luminance in cd/m^2 or a luminous flux in lumens, positive: a zero light is an authoring error, not a dark one.
std::optional<std::variant<Luminance, LuminousFlux>> lightEmission(const nlohmann::json& light, const std::string& path,
                                                                   std::size_t index) {
    const auto luminance = light.find("luminance");
    const auto lumens = light.find("lumens");
    if ((luminance == light.end()) == (lumens == light.end())) {
        std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] needs exactly one of luminance or lumens\n";
        return std::nullopt;
    }
    const bool isLuminance = luminance != light.end();
    const float value = toFloat(isLuminance ? *luminance : *lumens);
    if (!(value > 0.0F)) {
        std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] " << (isLuminance ? "luminance " : "lumens ") << value
                  << " is not positive\n";
        return std::nullopt;
    }
    if (isLuminance) {
        return Luminance{value};
    }
    return LuminousFlux{value};
}

// The type's own shape, its keys closed per type; nullopt logs an unknown type, a foreign key or a non-positive extent.
std::optional<std::variant<QuadShape, DiskShape, SphereShape, MeshShape>> lightShape(const nlohmann::json& light, const std::string& path,
                                                                                 std::size_t index) {
    const auto type = light.at("type").get<std::string>();
    const auto reject = [&](const char* why) {
        std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] " << why << '\n';
        return std::nullopt;
    };
    // Each extent is a measure: zero subtends no solid angle, and a negative size would mirror a quad and flip which face emits.
    if (type == "quad") {
        if (!onlyKnownKeys(light, {"type", "position", "rotation", "size", "color", "temperature", "luminance", "lumens", "twoSided"}, path,
                           "light")) {
            return std::nullopt;
        }
        const QuadShape quad{light.at("size").get<glm::vec2>()};
        if (!(quad.size.x > 0.0F) || !(quad.size.y > 0.0F)) {
            return reject("has a non-positive size");
        }
        return quad;
    }
    if (type == "disk") {
        if (!onlyKnownKeys(light, {"type", "position", "rotation", "radius", "color", "temperature", "luminance", "lumens", "twoSided"}, path,
                           "light")) {
            return std::nullopt;
        }
        const DiskShape disk{toFloat(light.at("radius"))};
        if (!(disk.radius > 0.0F)) {
            return reject("has a non-positive radius");
        }
        return disk;
    }
    if (type == "sphere") {
        if (!onlyKnownKeys(light, {"type", "position", "rotation", "radius", "color", "temperature", "luminance", "lumens"}, path,
                           "light")) {
            return std::nullopt;
        }
        const SphereShape sphere{toFloat(light.at("radius"))};
        if (!(sphere.radius > 0.0F)) {
            return reject("has a non-positive radius");
        }
        return sphere;
    }
    if (type == "mesh") {
        if (!onlyKnownKeys(light, {"type", "position", "rotation", "gltfPath", "scale", "color", "temperature", "luminance", "lumens",
                                   "twoSided"},
                           path, "light")) {
            return std::nullopt;
        }
        const MeshShape mesh{light.at("gltfPath").get<std::string>(), light.contains("scale") ? toFloat(light.at("scale")) : 1.0F};
        if (!(mesh.scale > 0.0F)) {
            return reject("has a non-positive scale");
        }
        return mesh;
    }
    // Dispatched on rather than ignored: an unrecognised type must not silently load as another.
    std::cerr << "loadSceneConfig: " << path << ": lights[" << index << "] has unknown type '" << type
              << "', expected quad, disk, sphere or mesh\n";
    return std::nullopt;
}

// Parses and validates the optional "lights" array. Separate so each stays one screen, and every check is an authoring boundary.
std::optional<std::vector<LightConfig>> parseLights(const nlohmann::json& j, const std::string& path) {
    std::vector<LightConfig> lights;
    const auto it = j.find("lights");
    if (it == j.end()) {
        return lights;
    }
    for (const nlohmann::json& light : *it) {
        const std::size_t index = lights.size();
        const auto shape = lightShape(light, path, index);
        if (!shape) {
            return std::nullopt;
        }
        const std::optional<glm::vec3> color = lightColor(light, path, index);
        if (!color) {
            return std::nullopt;
        }
        const auto emission = lightEmission(light, path, index);
        if (!emission) {
            return std::nullopt;
        }
        lights.push_back(LightConfig{
            light.at("position").get<glm::vec3>(),
            light.at("rotation").get<glm::vec3>(),
            *shape,
            *color,
            *emission,
            light.value("twoSided", false),
        });
    }
    return lights;
}

// Parses and validates the optional "volumes" array, each entry a grid, its world placement and its volume material.
std::optional<std::vector<VolumeConfig>> parseVolumes(const nlohmann::json& j, const std::string& path) {
    std::vector<VolumeConfig> volumes;
    const auto it = j.find("volumes");
    if (it == j.end()) {
        return volumes;
    }
    for (const nlohmann::json& volume : *it) {
        if (!onlyKnownKeys(volume, {"vdbPath", "grid", "position", "rotation", "scale", "materialPath"}, path, "volume")) {
            return std::nullopt;
        }
        VolumeConfig config{
            volume.at("vdbPath").get<std::string>(),
            volume.at("grid").get<std::string>(),
            volume.at("position").get<glm::vec3>(),
            volume.at("rotation").get<glm::vec3>(),
            1.0F,
            volume.at("materialPath").get<std::string>(),
        };
        if (const auto scale = volume.find("scale"); scale != volume.end()) {
            config.scale = toFloat(*scale);
        }
        // A zero or negative scale has no inverse placement; NaN fails the comparison too.
        if (!(config.scale > 0.0F) || !std::isfinite(config.scale)) {
            std::cerr << "loadSceneConfig: " << path << ": volumes[" << volumes.size() << "] scale must be finite and positive\n";
            return std::nullopt;
        }
        volumes.push_back(std::move(config));
    }
    return volumes;
}

// One input's constant: in its specification range per channel.
template <typename T>
bool validInput(const pathtracer::scene::InputSpec& spec, const T& value, const std::string& path) {
    const glm::vec3 channels(value);
    for (int c = 0; c < (std::is_same_v<T, float> ? 1 : 3); ++c) {
        if (!pathtracer::scene::inRange(channels[c], spec.range)) {
            std::cerr << "loadMaterialConfig: " << path << ": " << spec.name << " " << channels[c] << " is outside its OpenPBR range\n";
            return false;
        }
    }
    return true;
}

// {"randomChecker": {"cellSize", "seed"}} in place of spec's constant; false, logged, where a constant 0 would be refused too.
bool addRandomChecker(const pathtracer::scene::InputSpec& spec, const nlohmann::json& binding, MaterialConfig& material,
                      const std::string& path) {
    if (!onlyKnownKeys(binding, {"randomChecker"}, path, "material input") ||
        !onlyKnownKeys(binding.at("randomChecker"), {"cellSize", "seed"}, path, "randomChecker")) {
        return false;
    }
    const nlohmann::json& checker = binding.at("randomChecker");
    const float cellSize = toFloat(checker.at("cellSize"));
    if (!(cellSize > 0.0F) || !std::isfinite(cellSize)) {
        std::cerr << "loadMaterialConfig: " << path << ": " << spec.name << " random checker cellSize must be positive and finite\n";
        return false;
    }
    // 1 lies in every OpenPBR range, so the pattern's 0 alone decides: a Positive input (an ior) cannot take it.
    if (!pathtracer::scene::inRange(0.0F, spec.range)) {
        std::cerr << "loadMaterialConfig: " << path << ": " << spec.name << " cannot take a random checker's 0\n";
        return false;
    }
    const auto seed = checker.find("seed");
    material.randomCheckers.emplace(spec.name, pathtracer::scene::RandomChecker{cellSize, seed == checker.end() ? 0U : uint32Of(*seed)});
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

        const nlohmann::json& environment = j.at("environment");
        if (!onlyKnownKeys(j, {"model", "environment", "materialPath", "materialOverrides", "textures", "lights", "volumes", "camera"},
                           path, "top-level") ||
            !onlyKnownKeys(environment, {"hdriPath", "lightEnabled", "colorSpace", "lux"}, path, "environment")) {
            return std::nullopt;
        }
        // No model is a scene of lights and volumes alone; the surface keys all name the model's nodes, so they come with it.
        ModelConfig model{std::nullopt, glm::vec3(0.0F), glm::vec3(0.0F)};
        if (const auto it = j.find("model"); it != j.end()) {
            if (!onlyKnownKeys(*it, {"gltfPath", "position", "rotation", "vertexColor"}, path, "model")) {
                return std::nullopt;
            }
            model = ModelConfig{it->at("gltfPath").get<std::string>(), it->at("position").get<glm::vec3>(),
                                it->at("rotation").get<glm::vec3>(), it->value("vertexColor", true)};
        } else if (j.contains("materialPath") || j.contains("materialOverrides") || j.contains("textures")) {
            std::cerr << "loadSceneConfig: " << path << ": materialPath, materialOverrides and textures need a model\n";
            return std::nullopt;
        }

        std::map<std::string, std::string> materialOverrides;
        if (const auto it = j.find("materialOverrides"); it != j.end()) {
            materialOverrides = it->get<std::map<std::string, std::string>>();
        }
        std::optional<std::map<std::string, std::map<std::string, TextureConfig>>> textures = parseTextures(j, path);
        std::optional<std::vector<LightConfig>> lights = parseLights(j, path);
        std::optional<std::vector<VolumeConfig>> volumes = parseVolumes(j, path);
        if (!textures || !lights || !volumes) {
            return std::nullopt;
        }

        // A calibration target: zero would scale the sky to black, which lightEnabled false already expresses.
        std::optional<float> illuminanceLux;
        if (const auto it = environment.find("lux"); it != environment.end()) {
            illuminanceLux = toFloat(*it);
            if (!(*illuminanceLux > 0.0F)) {
                std::cerr << "loadSceneConfig: " << path << ": environment lux " << *illuminanceLux
                          << " lx is not positive\n";
                return std::nullopt;
            }
        }

        std::optional<SceneCameraConfig> camera;
        if (const auto it = j.find("camera"); it != j.end()) {
            if (!onlyKnownKeys(*it, {"position", "rotation", "farClip"}, path, "camera")) {
                return std::nullopt;
            }
            camera = SceneCameraConfig{it->at("position").get<glm::vec3>(), it->at("rotation").get<glm::vec3>(), toFloat(it->at("farClip"))};
        }

        const bool hasModel = model.gltfPath.has_value();
        return SceneConfig{
            std::move(model),
            EnvironmentConfig{
                environment.at("hdriPath").get<std::string>(),
                environment.value("lightEnabled", true),
                optionalString(environment, "colorSpace"),
                illuminanceLux,
            },
            hasModel ? std::optional(j.at("materialPath").get<std::string>()) : std::nullopt,
            std::move(materialOverrides),
            std::move(*textures),
            std::move(*lights),
            std::move(*volumes),
            camera,
        };
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "loadSceneConfig: " << path << ": " << e.what() << '\n';
        return std::nullopt;
    }
}

void applySceneCamera(const SceneConfig& scene, ProfileConfig& profile) {
    if (scene.camera) {
        profile.camera.position = scene.camera->position;
        profile.camera.rotation = scene.camera->rotation;
        profile.camera.farClip = scene.camera->farClip;
    }
}

std::optional<VolumeMaterialConfig> loadVolumeMaterialConfig(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "loadVolumeMaterialConfig: could not read " << path << '\n';
        return std::nullopt;
    }

    try {
        nlohmann::json j;
        file >> j;
        if (!onlyKnownKeys(j, {"absorption", "scattering", "anisotropy", "density"}, path, "volume material")) {
            return std::nullopt;
        }
        VolumeMaterialConfig material{
            j.at("absorption").get<glm::vec3>(),
            j.at("scattering").get<glm::vec3>(),
            toFloat(j.at("anisotropy")),
        };
        if (const auto it = j.find("density"); it != j.end()) {
            material.density = toFloat(*it);
        }
        if (!(material.density >= 0.0F) || !std::isfinite(material.density)) {
            std::cerr << "loadVolumeMaterialConfig: " << path << ": density must be finite and non-negative\n";
            return std::nullopt;
        }
        // A negative coefficient is no medium; NaN fails both comparisons, and so does an infinite one, which tracking cannot bound.
        const auto validCoefficient = [](const glm::vec3& c) {
            const glm::vec3 infinity(std::numeric_limits<float>::infinity());
            return glm::all(glm::greaterThanEqual(c, glm::vec3(0.0F))) && glm::all(glm::lessThan(c, infinity));
        };
        if (!validCoefficient(material.absorption) || !validCoefficient(material.scattering)) {
            std::cerr << "loadVolumeMaterialConfig: " << path << ": absorption and scattering must be finite and non-negative\n";
            return std::nullopt;
        }
        // |g| = 1 is a delta lobe Henyey-Greenstein cannot normalise.
        if (!(std::abs(material.anisotropy) < 1.0F)) {
            std::cerr << "loadVolumeMaterialConfig: " << path << ": anisotropy must lie in (-1, 1)\n";
            return std::nullopt;
        }
        return material;
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "loadVolumeMaterialConfig: " << path << ": " << e.what() << '\n';
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
        material.geometryThinWalled = j.value("geometry_thin_walled", false);
        bool ok = true;
        pathtracer::scene::forEachInput(
            [&](const pathtracer::scene::InputSpec& spec, auto& value) {
                const auto it = j.find(spec.name);
                if (it == j.end()) {
                    return;
                }
                if (it->is_object()) {
                    ok = addRandomChecker(spec, *it, material, path) && ok;
                    return;
                }
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>, float>) {
                    value = toFloat(*it);
                } else {
                    value = it->template get<glm::vec3>();
                }
                ok = validInput(spec, value, path) && ok;
            },
            material);
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
