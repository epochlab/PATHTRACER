#include "pathtracer/scene/material_binding.h"

#include <iostream>
#include <map>
#include <set>
#include <string_view>
#include <utility>

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

}  // namespace

bool bindSceneTextures(std::vector<MeshInstance>& instances,
                       const std::map<std::string, std::map<std::string, std::string>>& textures,
                       const std::string& assetRoot, pathtracer::gfx::ScalarType textureType) {
    if (!everyKeyNamesAnInstance(textures, instances, "bindSceneTextures", "textures")) {
        return false;
    }
    static const std::map<std::string_view, pathtracer::gfx::ImageTexture Material::*> kSlots = {
        {"baseColorTexture", &Material::baseColorTexture}, {"normalTexture", &Material::normalTexture},
        {"bumpTexture", &Material::bumpTexture},           {"roughnessTexture", &Material::roughnessTexture},
        {"specularTexture", &Material::specularTexture},
    };
    // Everything validated and loaded before any instance changes, each distinct file once however many slots share it.
    std::map<std::string, pathtracer::gfx::ImageTexture> loaded;
    for (const auto& [nodeName, slots] : textures) {
        for (const auto& [slot, path] : slots) {
            if (!kSlots.contains(slot)) {
                std::cerr << "bindSceneTextures: '" << nodeName << "' names unknown slot '" << slot << "'\n";
                return false;
            }
            if (loaded.contains(path)) {
                continue;
            }
            std::optional<pathtracer::gfx::ImageTexture> texture =
                pathtracer::gfx::loadImageTexture(assetRoot + "/" + path, textureType);
            if (!texture) {
                std::cerr << "bindSceneTextures: '" << nodeName << "' texture '" << path << "' failed to load\n";
                return false;
            }
            loaded.emplace(path, std::move(*texture));
        }
    }
    for (MeshInstance& instance : instances) {
        if (const auto it = textures.find(instance.name); it != textures.end()) {
            for (const auto& [slot, path] : it->second) {
                instance.material.*kSlots.at(slot) = loaded.at(path);
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
        quadLights.push_back(QuadLight{
            glm::vec3(sceneTransform * glm::vec4(light.origin, 1.0F)),
            glm::vec3(sceneTransform * glm::vec4(light.edge0, 0.0F)),
            glm::vec3(sceneTransform * glm::vec4(light.edge1, 0.0F)),
            light.color * light.intensity,
            light.twoSided,
        });
    }
    return quadLights;
}

}  // namespace pathtracer::scene
