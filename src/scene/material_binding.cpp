#include "pathtracer/scene/material_binding.h"

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string_view>
#include <utility>
#include <variant>

#include <glm/gtc/matrix_transform.hpp>

#include "pathtracer/scene/rotation.h"

namespace pathtracer::scene {

namespace {

// A key naming no instance is a typo: rejected, so a scene never renders silently with the wrong material or texture.
template <typename Overrides>
bool everyKeyNamesAnInstance(const Overrides& overrides, const std::vector<MeshInstance>& instances, const char* caller,
                             const char* field) {
    std::set<std::string_view> instanceNames;
    for (const MeshInstance& instance : instances) {
        instanceNames.insert(instance.name);
    }
    for (const auto& entry : overrides) {
        if (!instanceNames.contains(entry.first)) {
            std::cerr << caller << ": " << field << " key '" << entry.first << "' matches no glTF node\n";
            return false;
        }
    }
    return true;
}

// translate * rotationXyz: the one placement convention shared by the model root and every light.
glm::mat4 placementTransform(const glm::vec3& position, const glm::vec3& rotationDegrees) {
    return glm::translate(glm::mat4(1.0F), position) * glm::mat4(rotationXyz(rotationDegrees));
}

}  // namespace

bool bindSceneTextures(std::vector<MeshInstance>& instances,
                       const std::map<std::string, std::map<std::string, std::string>>& textures,
                       const std::string& assetRoot, pathtracer::gfx::ScalarType textureType) {
    if (!everyKeyNamesAnInstance(textures, instances, "bindSceneTextures", "textures")) {
        return false;
    }
    using ScalarSlot = MaterialInput<float> Material::*;
    using ColorSlot = MaterialInput<glm::vec3> Material::*;
    static const std::map<std::string_view, std::variant<ScalarSlot, ColorSlot>> kSlots = {
        {"baseColorTexture", &Material::baseColor}, {"normalTexture", &Material::normal},
        {"bumpTexture", &Material::bump},           {"roughnessTexture", &Material::roughness},
        {"specularTexture", &Material::specular},
    };
    // A slot loads at its input type's channel count, so a file shared by an RGB and a scalar slot is decoded once per count.
    const auto channelsOf = [](const std::string& slot) {
        return std::holds_alternative<ScalarSlot>(kSlots.at(slot)) ? pathtracer::gfx::kScalarChannels
                                                                   : pathtracer::gfx::kRgbChannels;
    };
    // Everything validated and loaded before any instance changes, each distinct (file, channel count) once however many slots share it.
    std::map<std::pair<std::string, int>, TextureHandle> loaded;
    for (const auto& [nodeName, slots] : textures) {
        for (const auto& [slot, path] : slots) {
            if (!kSlots.contains(slot)) {
                std::cerr << "bindSceneTextures: '" << nodeName << "' names unknown slot '" << slot << "'\n";
                return false;
            }
            std::pair<std::string, int> key{path, channelsOf(slot)};
            if (loaded.contains(key)) {
                continue;
            }
            std::optional<pathtracer::gfx::ImageTexture> texture =
                pathtracer::gfx::loadImageTexture(assetRoot + "/" + path, textureType, key.second);
            if (!texture) {
                std::cerr << "bindSceneTextures: '" << nodeName << "' texture '" << path << "' failed to load\n";
                return false;
            }
            loaded.emplace(std::move(key), std::make_shared<const pathtracer::gfx::ImageTexture>(std::move(*texture)));
        }
    }
    for (MeshInstance& instance : instances) {
        if (const auto it = textures.find(instance.name); it != textures.end()) {
            for (const auto& [slot, path] : it->second) {
                const TextureHandle& texture = loaded.at({path, channelsOf(slot)});
                std::visit([&](auto member) { instance.material.*member = texture; }, kSlots.at(slot));
            }
        }
    }
    return true;
}

std::optional<std::vector<PathTraceSettings>> resolvePerInstanceSettings(
    const PathTraceSettings& base, const std::vector<MeshInstance>& instances,
    const std::map<std::string, std::string>& materialOverrides, const std::string& assetRoot) {
    if (!everyKeyNamesAnInstance(materialOverrides, instances, "resolvePerInstanceSettings", "materialOverrides")) {
        return std::nullopt;
    }

    std::map<std::string, pathtracer::config::MaterialConfig> overrideMaterialsByPath;
    for (const auto& [nodeName, path] : materialOverrides) {
        if (overrideMaterialsByPath.contains(path)) {
            continue;
        }
        std::optional<pathtracer::config::MaterialConfig> overrideMaterial =
            pathtracer::config::loadMaterialConfig(assetRoot + "/" + path);
        if (!overrideMaterial) {
            std::cerr << "resolvePerInstanceSettings: materialOverrides entry '" << path
                      << "' failed to load\n";
            return std::nullopt;
        }
        overrideMaterialsByPath.emplace(path, std::move(*overrideMaterial));
    }

    std::vector<PathTraceSettings> perInstanceSettings;
    perInstanceSettings.reserve(instances.size());
    for (const MeshInstance& instance : instances) {
        PathTraceSettings settings = base;
        if (const auto overrideIt = materialOverrides.find(instance.name);
            overrideIt != materialOverrides.end()) {
            const pathtracer::config::MaterialConfig& overrideMaterial =
                overrideMaterialsByPath.at(overrideIt->second);
            settings.bumpStrength = overrideMaterial.bumpStrength;
            settings.roughnessMin = overrideMaterial.roughnessMin;
            settings.roughnessMax = overrideMaterial.roughnessMax;
            settings.diffuseColour = overrideMaterial.diffuseColour;
            settings.ior = overrideMaterial.ior;
            settings.abbe = overrideMaterial.abbe;
            settings.transmissionFactor = overrideMaterial.transmissionFactor;
            settings.metallicFactor = overrideMaterial.metallicFactor;
            settings.roughnessFactor = overrideMaterial.roughnessFactor;
            settings.diffuseRoughness = overrideMaterial.diffuseRoughness;
            settings.transmissionColor = overrideMaterial.transmissionColor;
            settings.transmissionDepth = overrideMaterial.transmissionDepth;
            settings.edgeTint = overrideMaterial.edgeTint;
            settings.shadingModel = overrideMaterial.shadingModel;
        }
        perInstanceSettings.push_back(settings);
    }
    return perInstanceSettings;
}

std::vector<QuadLight> buildQuadLights(const std::vector<pathtracer::config::QuadLightConfig>& lights,
                                        const glm::mat4& sceneTransform) {
    std::vector<QuadLight> quadLights;
    quadLights.reserve(lights.size());
    for (const pathtracer::config::QuadLightConfig& light : lights) {
        const glm::mat4 lightToWorld = sceneTransform * placementTransform(light.position, light.rotation);
        const glm::vec2 halfSize = 0.5F * light.size;
        // edge0 = +X, edge1 = -Y: cross(edge0, edge1) is local -Z, the emitting side, from the (-x, +y) corner.
        quadLights.push_back(QuadLight{
            glm::vec3(lightToWorld * glm::vec4(-halfSize.x, halfSize.y, 0.0F, 1.0F)),
            glm::vec3(lightToWorld * glm::vec4(light.size.x, 0.0F, 0.0F, 0.0F)),
            glm::vec3(lightToWorld * glm::vec4(0.0F, -light.size.y, 0.0F, 0.0F)),
            light.color * light.intensity,
            light.twoSided,
        });
    }
    return quadLights;
}

glm::mat4 rootTransformOf(const pathtracer::config::ModelConfig& model) {
    return placementTransform(model.position, model.rotation);
}

PathTraceSettings baseSettingsOf(const pathtracer::config::ProfileConfig& profile,
                                 const pathtracer::config::MaterialConfig& material, int samplesPerPixel) {
    return PathTraceSettings{
        .samplesPerPixel = samplesPerPixel,
        .maxBounces = profile.pathTracer.maxBounces,
        .russianRouletteStartBounce = profile.pathTracer.russianRouletteStartBounce,
        .aoMaxDistance = profile.pathTracer.aoMaxDistance,
        .lookaheadDistance = profile.pathTracer.lookaheadDistance,
        .bumpStrength = material.bumpStrength,
        .roughnessMin = material.roughnessMin,
        .roughnessMax = material.roughnessMax,
        .diffuseColour = material.diffuseColour,
        .ior = material.ior,
        .abbe = material.abbe,
        .transmissionFactor = material.transmissionFactor,
        .metallicFactor = material.metallicFactor,
        .roughnessFactor = material.roughnessFactor,
        .diffuseRoughness = material.diffuseRoughness,
        .transmissionColor = material.transmissionColor,
        .transmissionDepth = material.transmissionDepth,
        .edgeTint = material.edgeTint,
        .shadingModel = material.shadingModel,
    };
}

}  // namespace pathtracer::scene
