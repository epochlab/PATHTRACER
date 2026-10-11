#pragma once

#include <memory>
#include <variant>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/openpbr.h"

namespace pathtracer::scene {

// Shared and immutable: one open serves every input, primitive and instance binding that file at that channel, count and role.
using TextureHandle = std::shared_ptr<const pathtracer::gfx::ImageTexture>;

// The constant an unbound input holds, or the bound texture that replaces it, filtered per shading vertex.
template <typename T>
using MaterialInput = std::variant<T, TextureHandle>;

// geometry_normal's sources: a tangent-space normal map, a height map at heightMetres world metres per unit, or both, bump last.
struct NormalInput {
    TextureHandle map;
    TextureHandle height;
    float heightMetres = 0.0F;
};

// One instance's OpenPBR surface: every input a constant or a texture, plus the base's and the coat's shading-normal sources.
struct Material : OpenPbrInputs<MaterialInput> {
    NormalInput geometryNormal;
    NormalInput geometryCoatNormal;  // unbound, the coat follows the interpolated surface normal, not the base's map or bump
};

// A material file's constants as an unbound Material.
[[nodiscard]] inline Material materialOf(const OpenPbrInputs<Constant>& constants) {
    Material material;
    forEachInput([](const InputSpec&, auto& input, const auto& value) { input = value; }, material, constants);
    return material;
}

}  // namespace pathtracer::scene
