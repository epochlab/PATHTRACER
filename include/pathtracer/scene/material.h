#pragma once

#include "pathtracer/gfx/hdr_image.h"

namespace pathtracer::scene {

// Standard: the metallic-roughness BSDF. Constant: emits its resolved base colour and scatters nothing, an unlit flat surface.
enum class ShadingModel { Standard, Constant };

// Metallic-roughness material extended with Specular and AO: the raw texture set this project's assets ship.
struct Material {
    pathtracer::gfx::ImageTexture baseColorTexture;
    pathtracer::gfx::ImageTexture normalTexture;
    pathtracer::gfx::ImageTexture bumpTexture;
    pathtracer::gfx::ImageTexture roughnessTexture;
    pathtracer::gfx::ImageTexture specularTexture;
    // Loaded from glTF's occlusion_texture but read by nothing: AO is ray-traced per sample now, which a baked texture cannot match.
    pathtracer::gfx::ImageTexture aoTexture;
};

// Neutral default: every slot a 1x1 identity texture, which avoids a div-by-zero in buildShadingFrame and zeroes the bump difference.
[[nodiscard]] Material makeDefaultMaterial();

}  // namespace pathtracer::scene
