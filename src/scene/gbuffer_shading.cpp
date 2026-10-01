#include "pathtracer/scene/gbuffer_shading.h"

#include <algorithm>
#include <cstddef>

namespace pathtracer::scene {

glm::vec3 resolveBaseColor(const Material& material, glm::vec2 uv, const glm::vec3& vertexColour,
                            const PathTraceSettings& settings) {
    return pathtracer::gfx::sampleBilinear(material.baseColorTexture, uv) * settings.diffuseColour * vertexColour;
}

namespace {

float resolveRoughness(const Material& material, glm::vec2 uv, const PathTraceSettings& settings) {
    const float sample = pathtracer::gfx::sampleBilinear(material.roughnessTexture, uv).r;
    // Floor (UE4/Frostbite convention) avoids a near-zero-roughness GGX singularity.
    return std::clamp(sample * settings.roughnessFactor, settings.roughnessMin, settings.roughnessMax);
}

}  // namespace

LineProximity nearLineSegmentPx(glm::vec2 p, glm::vec2 a, glm::vec2 b, float thicknessPx) {
    const glm::vec2 ab = b - a;
    const float abLenSq = glm::dot(ab, ab);
    const float t = abLenSq > 1e-12F ? std::clamp(glm::dot(p - a, ab) / abLenSq, 0.0F, 1.0F) : 0.0F;
    const glm::vec2 closest = a + (t * ab);
    return LineProximity{glm::length(p - closest) < thicknessPx, t};
}

BsdfParams resolveBsdfParams(const Material& material, glm::vec2 uv, const glm::vec3& vertexColour,
                              const PathTraceSettings& settings,
                              std::optional<int> heroChannel) {
    const glm::vec3 baseColor = resolveBaseColor(material, uv, vertexColour, settings);
    const float roughness = resolveRoughness(material, uv, settings);
    const glm::vec3 specular = pathtracer::gfx::sampleBilinear(material.specularTexture, uv);
    const glm::vec3 f0 = glm::mix(specular, baseColor, settings.metallicFactor);
    // Dispersion enters here alone: every downstream ior consumer reads this one scalar, so the vertex stays spectrally consistent.
    const float ior = heroChannel.has_value()
                          ? cauchyIor(settings.ior, settings.abbe, kRgbWavelengthsNm[*heroChannel])
                          : settings.ior;
    // OpenPBR's two exclusive regimes: at transmissionDepth > 0 Beer-Lambert extinction carries it; at 0 it is the on-surface tint.
    const glm::vec3 transmissionTint =
        settings.transmissionDepth > 0.0F ? glm::vec3(1.0F) : settings.transmissionColor;
    return BsdfParams{baseColor,          settings.metallicFactor, roughness,
                       f0,                settings.edgeTint,       ior,
                       settings.transmissionFactor, settings.diffuseRoughness,
                       eonAlbedoInversion(baseColor, settings.diffuseRoughness), transmissionTint};
}

ShadingFrame buildShadingFrame(const ShadingVertex& shading, const Material& material,
                                const PathTraceSettings& settings) {
    const glm::vec3 normal = glm::normalize(shading.normal);
    glm::vec3 tangent = glm::vec3(shading.tangent);
    tangent = glm::normalize(tangent - (glm::dot(tangent, normal) * normal));
    const glm::vec3 bitangent = glm::cross(normal, tangent) * shading.tangent.w;

    const glm::vec3 normalSample = pathtracer::gfx::sampleBilinear(material.normalTexture, shading.uv);
    const glm::vec3 tangentSpaceNormal = glm::normalize((normalSample * 2.0F) - 1.0F);
    const glm::vec3 mappedNormal = glm::normalize(
        (tangentSpaceNormal.x * tangent) + (tangentSpaceNormal.y * bitangent) +
        (tangentSpaceNormal.z * normal));

    // Blinn 1978 bump mapping: the bump texture's height difference between adjacent texels becomes a shading-normal tilt.
    const glm::vec2 texel(1.0F / static_cast<float>(material.bumpTexture.width),
                           1.0F / static_cast<float>(material.bumpTexture.height));
    const float dHdu =
        pathtracer::gfx::sampleBilinear(material.bumpTexture, shading.uv + glm::vec2(texel.x, 0.0F)).r -
        pathtracer::gfx::sampleBilinear(material.bumpTexture, shading.uv - glm::vec2(texel.x, 0.0F)).r;
    const float dHdv =
        pathtracer::gfx::sampleBilinear(material.bumpTexture, shading.uv + glm::vec2(0.0F, texel.y)).r -
        pathtracer::gfx::sampleBilinear(material.bumpTexture, shading.uv - glm::vec2(0.0F, texel.y)).r;
    const glm::vec3 bumpedNormal = glm::normalize(
        mappedNormal - (settings.bumpStrength * dHdu * tangent) -
        (settings.bumpStrength * dHdv * bitangent));

    const glm::vec3 finalTangent =
        glm::normalize(tangent - (glm::dot(tangent, bumpedNormal) * bumpedNormal));
    const glm::vec3 finalBitangent = glm::cross(bumpedNormal, finalTangent) * shading.tangent.w;
    return ShadingFrame{finalTangent, finalBitangent, bumpedNormal};
}

glm::vec3 geometricNormalOf(const ShadingTriangle& tri) {
    return glm::normalize(
        glm::cross(tri.v1.position - tri.v0.position, tri.v2.position - tri.v0.position));
}

void writeTexel(pathtracer::gfx::HdrImage& image, int x, int y, glm::vec3 rgb) {
    const std::size_t idx = ((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) +
                              static_cast<std::size_t>(x)) *
                             4;
    image.rgba[idx + 0] = rgb.x;
    image.rgba[idx + 1] = rgb.y;
    image.rgba[idx + 2] = rgb.z;
    image.rgba[idx + 3] = 1.0F;
}

pathtracer::gfx::HdrImage makeImage(int width, int height) {
    pathtracer::gfx::HdrImage image;
    image.width = width;
    image.height = height;
    image.rgba.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4, 0.0F);
    return image;
}

}  // namespace pathtracer::scene
