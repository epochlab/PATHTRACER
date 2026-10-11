#include "pathtracer/scene/gbuffer_shading.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <variant>

namespace pathtracer::scene {

namespace {

// A constant input is its own value at every uv, validated at load; a texel is clamped into the input's specification range.
template <typename T>
T evaluate(const MaterialInput<T>& input, glm::vec2 uv, const pathtracer::gfx::TextureFootprint& footprint, InputRange range) {
    if (const T* constant = std::get_if<T>(&input)) {
        return *constant;
    }
    const glm::vec3 texel = pathtracer::gfx::sampleTexture(**std::get_if<TextureHandle>(&input), uv, footprint);
    if constexpr (std::is_same_v<T, float>) {
        return clampToRange(texel.r, range);
    } else {
        return {clampToRange(texel.r, range), clampToRange(texel.g, range), clampToRange(texel.b, range)};
    }
}

}  // namespace

pathtracer::gfx::TextureFootprint primaryHitFootprint(const ShadingTriangle& triangle, const Ray& ray, float t,
                                                      const glm::mat2x3& dirFootprint) {
    const glm::vec3 edge1 = triangle.v1.position - triangle.v0.position;
    const glm::vec3 edge2 = triangle.v2.position - triangle.v0.position;
    const glm::vec3 normal = glm::cross(edge1, edge2);
    // A pinhole's origin is fixed, so dP = t dD + dt D with dt = -t (dD.n)/(D.n): the offset ray meets the same plane.
    const float dirDotNormal = glm::dot(ray.dir, normal);
    const auto onPlane = [&](const glm::vec3& dirRate) {
        return t * (dirRate - ((glm::dot(dirRate, normal) / dirDotNormal) * ray.dir));
    };
    // dP in the edges' basis by its Gram system, whose determinant is |n|^2: the uvs then carry it, whatever their orientation.
    const float e11 = glm::dot(edge1, edge1);
    const float e12 = glm::dot(edge1, edge2);
    const float e22 = glm::dot(edge2, edge2);
    const float inverseDet = 1.0F / glm::dot(normal, normal);
    const glm::vec2 uv1 = triangle.v1.uv - triangle.v0.uv;
    const glm::vec2 uv2 = triangle.v2.uv - triangle.v0.uv;
    const auto toUv = [&](const glm::vec3& dP) {
        const float p1 = glm::dot(edge1, dP);
        const float p2 = glm::dot(edge2, dP);
        return ((((e22 * p1) - (e12 * p2)) * uv1) + (((e11 * p2) - (e12 * p1)) * uv2)) * inverseDet;
    };
    return {toUv(onPlane(dirFootprint[0])), toUv(onPlane(dirFootprint[1]))};
}

LineProximity nearLineSegmentPx(glm::vec2 p, glm::vec2 a, glm::vec2 b, float thicknessPx) {
    const glm::vec2 ab = b - a;
    const float abLenSq = glm::dot(ab, ab);
    const float t = abLenSq > 1e-12F ? std::clamp(glm::dot(p - a, ab) / abLenSq, 0.0F, 1.0F) : 0.0F;
    const glm::vec2 closest = a + (t * ab);
    return LineProximity{glm::length(p - closest) < thicknessPx, t};
}

OpenPbrInputs<Constant> resolveInputs(const Material& material, glm::vec2 uv, const pathtracer::gfx::TextureFootprint& footprint,
                                      const glm::vec3& vertexColour) {
    OpenPbrInputs<Constant> inputs;
    // Footprint filtering is exact in expectation only where radiance is linear; a lobe-shaping input is point-sampled.
    forEachInput(
        [&](const InputSpec& spec, auto& value, const auto& input) {
            if (spec.use == InputUse::Surface) {
                value = evaluate(input, uv, spec.filtered ? footprint : pathtracer::gfx::TextureFootprint{}, spec.range);
            }
        },
        inputs, material);
    inputs.baseColor *= vertexColour;
    return inputs;
}

bool isDispersive(const OpenPbrInputs<Constant>& inputs) {
    return inputs.transmissionDispersionScale > 0.0F && inputs.transmissionWeight > 0.0F && inputs.baseMetalness < 1.0F;
}

glm::vec3 emittedRadiance(const Material& material, glm::vec2 uv, const pathtracer::gfx::TextureFootprint& footprint) {
    // Most surfaces emit nothing: a zero luminance needs no colour lookup.
    const float luminance = evaluate(material.emissionLuminance, uv, footprint, InputRange::NonNegative);
    return luminance > 0.0F ? luminance * evaluate(material.emissionColor, uv, footprint, InputRange::NonNegative) : glm::vec3(0.0F);
}

ShadingFrame buildShadingFrame(const ShadingTriangle& triangle, const ShadingVertex& shading, const Material& material) {
    const glm::vec3 normal = glm::normalize(shading.normal);
    glm::vec3 tangent = glm::vec3(shading.tangent);
    tangent = glm::normalize(tangent - (glm::dot(tangent, normal) * normal));
    const glm::vec3 bitangent = glm::cross(normal, tangent) * shading.tangent.w;

    const NormalInput& geometryNormal = material.geometryNormal;
    glm::vec3 mappedNormal = normal;
    if (geometryNormal.map) {
        // Point-sampled like the bump below: a filtered normal shortens and shades flatter (Toksvig 2005), +13% on the minified stump.
        const glm::vec3 normalSample = pathtracer::gfx::sampleTexture(*geometryNormal.map, shading.uv);
        mappedNormal = glm::normalize(ShadingFrame(tangent, bitangent, normal) * glm::normalize((normalSample * 2.0F) - 1.0F));
    }

    // Bump (Blinn 1978) as Mikkelsen's surface gradient: height h = heightMetres*H in world units, n' = n - grad_s(h).
    glm::vec3 bumpedNormal = mappedNormal;
    if (geometryNormal.height) {
        // The lookup's own analytic dH/duv per unit uv: independent of texture resolution, and C1 across texel boundaries.
        const glm::vec2 dhduv = geometryNormal.heightMetres * pathtracer::gfx::sampleTextureGradient(*geometryNormal.height, shading.uv).dst;
        // Mikkelsen 2010, edges for screen derivatives: (dh1*R1 + dh2*R2)/det divides by dP/duv itself, so mirrored uvs keep their sign.
        const glm::vec3 edge1 = triangle.v1.position - triangle.v0.position;
        const glm::vec3 edge2 = triangle.v2.position - triangle.v0.position;
        const glm::vec3 r1 = glm::cross(edge2, normal);
        const glm::vec3 r2 = glm::cross(normal, edge1);
        const float dh1 = glm::dot(dhduv, triangle.v1.uv - triangle.v0.uv);
        const float dh2 = glm::dot(dhduv, triangle.v2.uv - triangle.v0.uv);
        const glm::vec3 surfaceGradient = ((dh1 * r1) + (dh2 * r2)) / glm::dot(edge1, r1);
        bumpedNormal = glm::normalize(mappedNormal - surfaceGradient);
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
