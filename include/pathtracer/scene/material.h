#pragma once

#include <memory>
#include <variant>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"

namespace pathtracer::scene {

// Standard: the metallic-roughness BSDF. Constant: emits its resolved base colour and scatters nothing, an unlit flat surface.
enum class ShadingModel { Standard, Constant };

// Shared and immutable: one decode serves every slot, primitive and instance binding that file at that channel count.
using TextureHandle = std::shared_ptr<const pathtracer::gfx::ImageTexture>;

// The constant an unbound slot holds, or the bound texture filtered per shading vertex. A float input decodes R, a vec3 one RGB.
template <typename T>
using MaterialInput = std::variant<T, TextureHandle>;

// Metallic-roughness material extended with Specular and Bump; default-constructed, every slot is its neutral constant.
struct Material {
    MaterialInput<glm::vec3> baseColor = glm::vec3(1.0F);           // white: diffuseColour alone sets the albedo
    MaterialInput<glm::vec3> normal = glm::vec3(0.5F, 0.5F, 1.0F);  // [0,1]-encoded as a normal map stores it: tangent-space +z
    MaterialInput<float> bump = 0.0F;                               // a constant height field has zero gradient, so no tilt
    MaterialInput<float> roughness = 1.0F;                          // roughnessFactor/min/max fully control the result
    MaterialInput<glm::vec3> specular = glm::vec3(0.04F);           // dielectric f0, inert whenever metallicFactor=0
};

}  // namespace pathtracer::scene
