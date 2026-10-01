#include "pathtracer/scene/material.h"

#include <memory>
#include <utility>
#include <vector>

namespace pathtracer::scene {

namespace {

TextureHandle constantTexture(int channels, std::vector<float> value) {
    return std::make_shared<const pathtracer::gfx::ImageTexture>(
        pathtracer::gfx::ImageTexture{1, 1, channels, std::move(value)});
}

}  // namespace

Material makeDefaultMaterial() {
    // Float32 regardless of textureBitDepth: these are exact constants, not loaded scene data. Each slot's channel count is set here.
    using pathtracer::gfx::kRgbChannels;
    using pathtracer::gfx::kScalarChannels;
    // Built once and shared by every instance, as any other immutable texture is.
    static const Material kDefault{
        constantTexture(kRgbChannels, {1.0F, 1.0F, 1.0F}),  // baseColorTexture: white
        constantTexture(kRgbChannels, {0.5F, 0.5F, 1.0F}),  // normalTexture: decodes to (0,0,1) tangent-space up
        constantTexture(kScalarChannels, {0.5F}),  // bumpTexture: any constant -> zero finite-difference
        constantTexture(kScalarChannels, {1.0F}),  // roughnessTexture: roughnessFactor/min/max fully control the result
        constantTexture(kRgbChannels, {0.04F, 0.04F, 0.04F}),  // specularTexture: dielectric f0, inert whenever metallicFactor=0
    };
    return kDefault;
}

}  // namespace pathtracer::scene
