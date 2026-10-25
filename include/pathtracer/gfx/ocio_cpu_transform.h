#pragma once

#include <array>
#include <limits>
#include <span>

#include <glm/glm.hpp>

#include "pathtracer/gfx/ocio_display_transform.h"

// The viewer's display path on the CPU, from the same constants as the display shaders: one definition, so a comparison cannot drift.
namespace pathtracer::gfx {

// In place, RGB triples (no alpha), row-major, width*height*3 elements; Raw is the identity. Thread-safe, processors are built once.
void applyOcioDisplayTransform(std::span<float> rgb, int width, int height,
                               OcioDisplayTransform::Lut lut = OcioDisplayTransform::Lut::SRGB);

// The log stage's guard: log2 is undefined at and below zero, so texels clamp to the least positive normal float first.
inline constexpr float kLogFloor = std::numeric_limits<float>::min();

// The display shaders' log stage per lane, before the affine map: a receptor's response to radiance ratios, not differences.
[[nodiscard]] inline glm::vec3 logTransform(const glm::vec3& rgb) {
    return glm::log2(glm::max(rgb, glm::vec3(kLogFloor)));
}

// Edge of the display dither tile; the encode and the viewer both read texel (x, y), counted from the top-left, at (x, y) mod 128.
inline constexpr int kDitherMaskSize = 128;
inline constexpr int kDitherMaskTexels = kDitherMaskSize * kDitherMaskSize;

// Row-major dither offsets in display-referred units: a blue-noise mask remapped to triangular PDF on (-1, 1) codes of 255, built once.
[[nodiscard]] const std::array<float, kDitherMaskTexels>& displayDitherMask();

// Linear to display-referred 8-bit into out, presentFrame's order: affine map, transform, dither, quantize. rgb and out hold w*h*3.
void encodeForDisplay(std::span<const float> rgb, int width, int height, const glm::vec3& gain, bool applyDisplayTransform,
                      const glm::vec3& displayOffset, std::span<unsigned char> out);

// encodeForDisplay's affine map and transform without its dither or quantize, unclamped floats for an encoder of any bit depth.
void encodeForDisplay(std::span<const float> rgb, int width, int height, const glm::vec3& gain, bool applyDisplayTransform,
                      const glm::vec3& displayOffset, std::span<float> out);

// In place, RGB triples (no alpha): colours in a linear config colour space into the scene space. Builds its processor per call.
void convertToSceneSpace(std::span<float> rgb, const char* colorSpace);

// OpenEXR-order xy (R, G, B, white) of a config colour space via its CIE XYZ D65 interchange: exact for a D65-white space.
[[nodiscard]] std::array<float, 8> chromaticitiesOf(const char* colorSpace);

}  // namespace pathtracer::gfx
