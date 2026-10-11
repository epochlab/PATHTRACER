#pragma once

#include <array>
#include <span>

#include <glm/glm.hpp>

#include "pathtracer/gfx/ocio_display_transform.h"

// The viewer's display path on the CPU, from the same constants as the display shaders: one definition, so a comparison cannot drift.
namespace pathtracer::gfx {

// In place, RGB triples (no alpha), row-major, width*height*3 elements; Raw is the identity. Thread-safe, processors are built once.
void applyOcioDisplayTransform(std::span<float> rgb, int width, int height,
                               OcioDisplayTransform::Lut lut = OcioDisplayTransform::Lut::SRGB);

// Linear to display-referred 8-bit into out, presentFrame's order: affine map, transform, dither, quantize. rgb and out hold w*h*3.
void encodeForDisplay(std::span<const float> rgb, int width, int height, const glm::vec3& gain, bool applyDisplayTransform,
                      const glm::vec3& displayOffset, std::span<unsigned char> out);

// encodeForDisplay's affine map and transform without its dither or quantize, unclamped floats for an encoder of any bit depth.
void encodeForDisplay(std::span<const float> rgb, int width, int height, const glm::vec3& gain, bool applyDisplayTransform,
                      const glm::vec3& displayOffset, std::span<float> out);

// OpenEXR-order xy (R, G, B, white) of a config colour space via its CIE XYZ D65 interchange: exact for a D65-white space.
[[nodiscard]] std::array<float, 8> chromaticitiesOf(const char* colorSpace);

}  // namespace pathtracer::gfx
