#pragma once

#include "pathtracer/gfx/hdr_image.h"

namespace pathtracer::scene {

// Standard: the metallic-roughness BSDF. Constant: emits its resolved base colour and scatters nothing, an unlit flat surface.
enum class ShadingModel { Standard, Constant };

// Metallic-roughness material extended with Specular and Bump: the texture slots a scene JSON binds per glTF node.
struct Material {
    pathtracer::gfx::ImageTexture baseColorTexture;
    pathtracer::gfx::ImageTexture normalTexture;
    pathtracer::gfx::ImageTexture bumpTexture;
    pathtracer::gfx::ImageTexture roughnessTexture;
    pathtracer::gfx::ImageTexture specularTexture;
};

// Neutral default, every slot a 1x1 identity texture; also the single definition of each slot's channel count bindSceneTextures loads at.
[[nodiscard]] Material makeDefaultMaterial();

}  // namespace pathtracer::scene
