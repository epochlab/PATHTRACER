#pragma once

#include <memory>

#include "pathtracer/gfx/hdr_image.h"

namespace pathtracer::scene {

// Standard: the metallic-roughness BSDF. Constant: emits its resolved base colour and scatters nothing, an unlit flat surface.
enum class ShadingModel { Standard, Constant };

// Shared and immutable: one decode serves every slot, primitive and instance binding that file at that channel count.
using TextureHandle = std::shared_ptr<const pathtracer::gfx::ImageTexture>;

// Metallic-roughness material extended with Specular and Bump: the texture slots a scene JSON binds per glTF node. Never null.
struct Material {
    TextureHandle baseColorTexture;
    TextureHandle normalTexture;
    TextureHandle bumpTexture;
    TextureHandle roughnessTexture;
    TextureHandle specularTexture;
};

// Neutral default, every slot a 1x1 identity texture; also the single definition of each slot's channel count bindSceneTextures loads at.
[[nodiscard]] Material makeDefaultMaterial();

}  // namespace pathtracer::scene
