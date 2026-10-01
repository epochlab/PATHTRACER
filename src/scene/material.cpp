#include "pathtracer/scene/material.h"

#include <vector>

namespace pathtracer::scene {

Material makeDefaultMaterial() {
    // Float32 regardless of textureBitDepth: these are exact constants, not loaded scene data. Each slot's channel count is set here.
    using pathtracer::gfx::kRgbChannels;
    using pathtracer::gfx::kScalarChannels;
    return Material{
        {1, 1, kRgbChannels, std::vector<float>{1.0F, 1.0F, 1.0F}},  // baseColorTexture: white
        {1, 1, kRgbChannels, std::vector<float>{0.5F, 0.5F, 1.0F}},  // normalTexture: decodes to (0,0,1) tangent-space up
        {1, 1, kScalarChannels, std::vector<float>{0.5F}},  // bumpTexture: any constant -> zero finite-difference
        {1, 1, kScalarChannels, std::vector<float>{1.0F}},  // roughnessTexture: roughnessFactor/min/max fully control the result
        {1, 1, kRgbChannels, std::vector<float>{0.04F, 0.04F, 0.04F}},  // specularTexture: dielectric f0, inert whenever metallicFactor=0
    };
}

}  // namespace pathtracer::scene
