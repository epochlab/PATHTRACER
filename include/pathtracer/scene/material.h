#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/openpbr.h"
#include "pathtracer/scene/random_checker.h"

namespace pathtracer::scene {

// Shared and immutable: one open serves every input, primitive and instance binding that file at that channel, count and role.
using TextureHandle = std::shared_ptr<const pathtracer::gfx::ImageTexture>;

// The constant an unbound input holds, the bound texture that replaces it, filtered per shading vertex, or a point-sampled pattern.
template <typename T>
using MaterialInput = std::variant<T, TextureHandle, RandomChecker>;

// A shading frame's sources: a tangent-space normal map, a height map at heightMetres metres per unit, bump last, and a tangent map.
struct NormalInput {
    TextureHandle map;
    TextureHandle height;
    float heightMetres = 0.0F;
    TextureHandle tangent;  // geometry_tangent: 2x - 1 in the mesh's (T, B) plane orients the anisotropy; unbound, the mesh's own T
};

// One instance's OpenPBR surface: every input a constant or a texture, plus the base's and the coat's shading-normal sources.
struct Material : OpenPbrInputs<MaterialInput> {
    NormalInput geometryNormal;
    NormalInput geometryCoatNormal;  // unbound, the coat follows the interpolated surface normal, not the base's map or bump
};

// A material file's random checkers by OpenPBR input name; each replaces that input's constant.
using RandomCheckerInputs = std::map<std::string, RandomChecker, std::less<>>;

// A material file's constants and random checkers as an unbound Material.
[[nodiscard]] inline Material materialOf(const OpenPbrInputs<Constant>& constants, const RandomCheckerInputs& randomCheckers) {
    Material material;
    forEachInput(
        [&](const InputSpec& spec, auto& input, const auto& value) {
            if (const auto checker = randomCheckers.find(spec.name); checker != randomCheckers.end()) {
                input = checker->second;
            } else {
                input = value;
            }
        },
        material, constants);
    material.geometryThinWalled = constants.geometryThinWalled;
    return material;
}

}  // namespace pathtracer::scene
