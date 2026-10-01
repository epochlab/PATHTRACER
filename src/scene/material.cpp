#include "pathtracer/scene/material.h"

#include <vector>

namespace pathtracer::scene {

Material makeDefaultMaterial() {
    // Float32 regardless of textureBitDepth: these are exact constants, not loaded scene data.
    return Material{
        {1, 1, std::vector<float>{1.0F, 1.0F, 1.0F, 1.0F}},  // baseColorTexture: white
        {1, 1, std::vector<float>{0.5F, 0.5F, 1.0F, 1.0F}},  // normalTexture: decodes to (0,0,1) tangent-space up
        {1, 1, std::vector<float>{0.5F, 0.5F, 0.5F, 1.0F}},  // bumpTexture: any constant -> zero finite-difference
        {1, 1, std::vector<float>{1.0F, 1.0F, 1.0F, 1.0F}},  // roughnessTexture: roughnessFactor/min/max fully control the result
        {1, 1, std::vector<float>{0.04F, 0.04F, 0.04F, 1.0F}},  // specularTexture: standard dielectric f0, inert whenever metallicFactor=0
    };
}

}  // namespace pathtracer::scene
