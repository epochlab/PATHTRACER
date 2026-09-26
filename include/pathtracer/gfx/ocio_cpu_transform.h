#pragma once

#include <span>
#include <vector>

#include <glm/glm.hpp>

// The viewer's display path on the CPU, from the same constants as the display shaders: one definition, so a comparison cannot drift.
namespace pathtracer::gfx {

// In place, RGB triples (no alpha), row-major, width*height*3 elements.
void applyOcioDisplayTransform(std::vector<float>& rgb, int width, int height);

// Linear to display-referred 8-bit, presentFrame's order: affine map, transform, dither, quantize. rgb is width*height*3, gain per lane.
[[nodiscard]] std::vector<unsigned char> encodeForDisplay(std::span<const float> rgb, int width, int height,
                                                           const glm::vec3& gain, bool applyDisplayTransform,
                                                           const glm::vec3& displayOffset = glm::vec3(0.0F));

}  // namespace pathtracer::gfx
