#include "pathtracer/scene/gbuffer_shading.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <variant>

namespace pathtracer::scene {

namespace {

// A constant input is its own value at every uv: only a bound texture is filtered, so an unbound slot costs one load.
template <typename T>
T evaluate(const MaterialInput<T>& input, glm::vec2 uv) {
    if (const T* constant = std::get_if<T>(&input)) {
        return *constant;
    }
    const glm::vec3 texel = pathtracer::gfx::sampleBilinear(**std::get_if<TextureHandle>(&input), uv);
    if constexpr (std::is_same_v<T, float>) {
        return texel.r;
    } else {
        return texel;
    }
}

float resolveRoughness(const Material& material, glm::vec2 uv, const PathTraceSettings& settings) {
    const float sample = evaluate(material.roughness, uv);
    // Floor (UE4/Frostbite convention) avoids a near-zero-roughness GGX singularity.
    return std::clamp(sample * settings.roughnessFactor, settings.roughnessMin, settings.roughnessMax);
}

}  // namespace

glm::vec3 resolveBaseColor(const Material& material, glm::vec2 uv, const glm::vec3& vertexColour,
                            const PathTraceSettings& settings) {
    return evaluate(material.baseColor, uv) * settings.diffuseColour * vertexColour;
}

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
    const glm::vec3 specular = evaluate(material.specular, uv);
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

    const glm::vec3 normalSample = evaluate(material.normal, shading.uv);
    const glm::vec3 tangentSpaceNormal = glm::normalize((normalSample * 2.0F) - 1.0F);
    const glm::vec3 mappedNormal = glm::normalize(ShadingFrame(tangent, bitangent, normal) * tangentSpaceNormal);

    // Blinn 1978 bump mapping: adjacent-texel height differences tilt the normal. A constant height has zero gradient: no tilt.
    glm::vec3 bumpedNormal = mappedNormal;
    if (const TextureHandle* bumpTexture = std::get_if<TextureHandle>(&material.bump)) {
        const pathtracer::gfx::ImageTexture& bump = **bumpTexture;
        const glm::vec2 texel(1.0F / static_cast<float>(bump.width), 1.0F / static_cast<float>(bump.height));
        const float dHdu = pathtracer::gfx::sampleBilinear(bump, shading.uv + glm::vec2(texel.x, 0.0F)).r -
                           pathtracer::gfx::sampleBilinear(bump, shading.uv - glm::vec2(texel.x, 0.0F)).r;
        const float dHdv = pathtracer::gfx::sampleBilinear(bump, shading.uv + glm::vec2(0.0F, texel.y)).r -
                           pathtracer::gfx::sampleBilinear(bump, shading.uv - glm::vec2(0.0F, texel.y)).r;
        bumpedNormal = glm::normalize(
            mappedNormal - (settings.bumpStrength * dHdu * tangent) -
            (settings.bumpStrength * dHdv * bitangent));
    }

    const glm::vec3 finalTangent =
        glm::normalize(tangent - (glm::dot(tangent, bumpedNormal) * bumpedNormal));
    const glm::vec3 finalBitangent = glm::cross(bumpedNormal, finalTangent) * shading.tangent.w;
    return ShadingFrame(finalTangent, finalBitangent, bumpedNormal);
}

glm::vec3 geometricNormalOf(const ShadingTriangle& tri) {
    return glm::normalize(
        glm::cross(tri.v1.position - tri.v0.position, tri.v2.position - tri.v0.position));
}

}  // namespace pathtracer::scene
