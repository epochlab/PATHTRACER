#include "pathtracer/scene/material_binding.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string_view>
#include <type_traits>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

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

// The two normal inputs: each a tangent-space map, a bump, or both, rather than a texture replacing a constant.
const std::map<std::string_view, NormalInput Material::*> kNormalInputs = {{"geometry_normal", &Material::geometryNormal},
                                                                          {"geometry_coat_normal", &Material::geometryCoatNormal}};

// The two tangent inputs, each its frame's tangent map: RGB data like a normal map, giving the anisotropy's direction.
const std::map<std::string_view, NormalInput Material::*> kTangentInputs = {{"geometry_tangent", &Material::geometryNormal},
                                                                           {"geometry_coat_tangent", &Material::geometryCoatNormal}};

// Every input at its specification default, the instance inputNamed reads specifications and types from.
const OpenPbrInputs<Constant> kSpecificationDefaults{};

// A file opens once per (path, channel count, role, colour space, channel offset), however many inputs and nodes share it.
using TextureKey = std::tuple<std::string, int, pathtracer::gfx::ImageRole, std::optional<std::string>, int>;

// An input's specification and channel count by name; nullopt for a name that is no OpenPBR input.
struct NamedInput {
    InputSpec spec;
    int channels;
};

std::optional<NamedInput> inputNamed(std::string_view name) {
    std::optional<NamedInput> found;
    forEachInput(
        [&](const InputSpec& spec, const auto& value) {
            if (spec.name == name) {
                found = NamedInput{spec, std::is_same_v<std::decay_t<decltype(value)>, float> ? pathtracer::gfx::kScalarChannels
                                                                                               : pathtracer::gfx::kRgbChannels};
            }
        },
        kSpecificationDefaults);
    return found;
}

// One image a binding opens, and the range its texels are clamped into; none for a normal or height map, whose any finite value is data.
struct ImageKey {
    TextureKey key;
    std::optional<InputRange> range;
};

// A binding resolved against its input: the input's image (a normal input's map) and a normal input's height map.
struct ResolvedBinding {
    std::optional<ImageKey> image;
    std::optional<ImageKey> bump;
    float bumpHeightMetres = 0.0F;
};

// The images a binding opens, or why it cannot apply to its input, from the binding alone.
std::variant<ResolvedBinding, std::string> resolveBinding(const std::string& name, const pathtracer::config::TextureConfig& texture) {
    using pathtracer::gfx::ImageRole;
    if (kNormalInputs.contains(name)) {
        if (texture.channel != 0) {
            return "a normal map reads R, G and B, so it takes no channel";
        }
        ResolvedBinding resolved;
        if (texture.path) {
            resolved.image = ImageKey{{*texture.path, pathtracer::gfx::kRgbChannels, ImageRole::Data, texture.colorSpace, 0}, std::nullopt};
        }
        if (texture.bump) {
            resolved.bump = ImageKey{{texture.bump->path, pathtracer::gfx::kScalarChannels, ImageRole::Data, std::nullopt, 0}, std::nullopt};
            resolved.bumpHeightMetres = texture.bump->heightMetres;
        }
        if (!resolved.image && !resolved.bump) {
            return "binds neither a normal map nor a bump";
        }
        return resolved;
    }
    if (kTangentInputs.contains(name)) {
        if (!texture.path) {
            return "binds no path";
        }
        if (texture.bump || texture.channel != 0) {
            return "a tangent map reads R, G and B, so it takes no channel and no bump";
        }
        return ResolvedBinding{ImageKey{{*texture.path, pathtracer::gfx::kRgbChannels, ImageRole::Data, texture.colorSpace, 0}, std::nullopt},
                               std::nullopt};
    }
    const std::optional<NamedInput> input = inputNamed(name);
    if (!input) {
        return "is not an OpenPBR input";
    }
    if (input->spec.requiresVolumes) {
        return "requires volumetric transport";
    }
    if (!texture.path) {
        return "binds no path";
    }
    if (texture.bump) {
        return "only a normal input takes a bump";
    }
    if (input->channels != pathtracer::gfx::kScalarChannels && texture.channel != 0) {
        return "a colour input reads R, G and B, so it takes no channel";
    }
    return ResolvedBinding{ImageKey{{*texture.path, input->channels, input->spec.role, texture.colorSpace, texture.channel}, input->spec.range},
                           std::nullopt};
}

// A texture with the extremes of its finest level: every texel lies in an interval input's range iff both extremes do.
struct OpenedTexture {
    TextureHandle texture;
    float min;
    float max;
};

// Opens the texture and scans its finest level once, so a tiled file served in place is checked as a derived one is.
std::optional<OpenedTexture> openScanned(const std::string& path, const TextureKey& key) {
    const auto& [file, channels, role, colorSpace, offset] = key;
    TextureHandle texture = pathtracer::gfx::openTexture(path, channels, role, pathtracer::gfx::TextureWrap::Repeat, colorSpace, offset);
    const std::optional<pathtracer::gfx::HdrImage> texels = texture ? pathtracer::gfx::readTexels(*texture) : std::nullopt;
    if (!texels || !std::all_of(texels->texels.begin(), texels->texels.end(), [](float t) { return std::isfinite(t); })) {
        return std::nullopt;
    }
    const auto [min, max] = std::minmax_element(texels->texels.begin(), texels->texels.end());
    return OpenedTexture{std::move(texture), *min, *max};
}

// Opens an image once into `loaded`; false, logged, when it fails. A texel past the range is reported here and clamped at lookup.
bool openImage(std::map<TextureKey, OpenedTexture>& loaded, const ImageKey& image, const std::string& assetRoot, const std::string& where) {
    auto it = loaded.find(image.key);
    if (it == loaded.end()) {
        std::optional<OpenedTexture> opened = openScanned(assetRoot + "/" + std::get<0>(image.key), image.key);
        if (!opened) {
            std::cerr << "bindSceneTextures: " << where << " texture '" << std::get<0>(image.key) << "' failed to load or holds a non-finite texel\n";
            return false;
        }
        it = loaded.emplace(image.key, std::move(*opened)).first;
    }
    const OpenedTexture& opened = it->second;
    if (image.range && !(inRange(opened.min, *image.range) && inRange(opened.max, *image.range))) {
        std::cerr << "bindSceneTextures: " << where << " texture '" << std::get<0>(image.key) << "' spans [" << opened.min << ", " << opened.max
                  << "], outside the input's range; texels are clamped into it\n";
    }
    return true;
}

// Writes one resolved binding into the material: its input's texture, or a normal input's map and height map.
void assignBinding(Material& material, const std::string& name, const ResolvedBinding& binding,
                   const std::map<TextureKey, OpenedTexture>& loaded) {
    const auto textureOf = [&](const std::optional<ImageKey>& image) { return image ? loaded.at(image->key).texture : nullptr; };
    // Field by field: a frame's normal and tangent inputs bind independently, in either order.
    if (const auto normal = kNormalInputs.find(name); normal != kNormalInputs.end()) {
        NormalInput& input = material.*(normal->second);
        input.map = textureOf(binding.image);
        input.height = textureOf(binding.bump);
        input.heightMetres = binding.bumpHeightMetres;
        return;
    }
    if (const auto tangent = kTangentInputs.find(name); tangent != kTangentInputs.end()) {
        (material.*(tangent->second)).tangent = textureOf(binding.image);
        return;
    }
    forEachInput(
        [&](const InputSpec& spec, auto& input) {
            if (spec.name == name) {
                input = textureOf(binding.image);
            }
        },
        material);
}

}  // namespace

bool applySceneMaterials(std::vector<MeshInstance>& instances, const std::string& materialPath,
                         const std::map<std::string, std::string>& materialOverrides, const std::string& assetRoot) {
    if (!everyKeyNamesAnInstance(materialOverrides, instances, "applySceneMaterials", "materialOverrides")) {
        return false;
    }
    // Each file parsed once however many nodes name it; any failure leaves every instance untouched.
    std::map<std::string, Material> byPath;
    const auto load = [&](const std::string& path) {
        if (byPath.contains(path)) {
            return true;
        }
        const std::optional<pathtracer::config::MaterialConfig> constants = pathtracer::config::loadMaterialConfig(assetRoot + "/" + path);
        if (!constants) {
            std::cerr << "applySceneMaterials: material '" << path << "' failed to load\n";
            return false;
        }
        byPath.emplace(path, materialOf(*constants));
        return true;
    };
    if (!load(materialPath) ||
        !std::all_of(materialOverrides.begin(), materialOverrides.end(), [&](const auto& entry) { return load(entry.second); })) {
        return false;
    }
    for (MeshInstance& instance : instances) {
        const auto it = materialOverrides.find(instance.name);
        instance.material = byPath.at(it == materialOverrides.end() ? materialPath : it->second);
    }
    return true;
}

bool bindSceneTextures(std::vector<MeshInstance>& instances,
                       const std::map<std::string, std::map<std::string, pathtracer::config::TextureConfig>>& textures,
                       const std::string& assetRoot) {
    if (!everyKeyNamesAnInstance(textures, instances, "bindSceneTextures", "textures")) {
        return false;
    }
    // Every binding resolved and every image opened before any instance changes, each distinct image once.
    std::map<std::pair<std::string, std::string>, ResolvedBinding> resolved;
    std::map<TextureKey, OpenedTexture> loaded;
    for (const auto& [nodeName, inputs] : textures) {
        for (const auto& [name, texture] : inputs) {
            const std::string where = "'" + nodeName + "' " + name;
            std::variant<ResolvedBinding, std::string> binding = resolveBinding(name, texture);
            if (const std::string* fault = std::get_if<std::string>(&binding)) {
                std::cerr << "bindSceneTextures: " << where << ": " << *fault << "\n";
                return false;
            }
            const ResolvedBinding& images = std::get<ResolvedBinding>(binding);
            if ((images.image && !openImage(loaded, *images.image, assetRoot, where)) ||
                (images.bump && !openImage(loaded, *images.bump, assetRoot, where))) {
                return false;
            }
            resolved.emplace(std::pair{nodeName, name}, images);
        }
    }
    for (MeshInstance& instance : instances) {
        if (const auto it = textures.find(instance.name); it != textures.end()) {
            for (const auto& entry : it->second) {
                assignBinding(instance.material, entry.first, resolved.at({instance.name, entry.first}), loaded);
            }
        }
    }
    return true;
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

PathTraceSettings baseSettingsOf(const pathtracer::config::ProfileConfig& profile, int samplesPerPixel) {
    return PathTraceSettings{
        .samplesPerPixel = samplesPerPixel,
        .maxBounces = profile.pathTracer.maxBounces,
        .russianRouletteStartBounce = profile.pathTracer.russianRouletteStartBounce,
        .aoMaxDistance = profile.pathTracer.aoMaxDistance,
        .lookaheadDistance = profile.pathTracer.lookaheadDistance,
    };
}

}  // namespace pathtracer::scene
