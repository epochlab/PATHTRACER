#pragma once

#include <span>
#include <vector>

// The viewer's display path on the CPU, from the same constants as the display shaders: one definition, so a comparison cannot drift.
namespace pathtracer::gfx {

// In place, RGB triples (no alpha), row-major, width*height*3 elements.
void applyOcioDisplayTransform(std::vector<float>& rgb, int width, int height);

// Linear to display-referred 8-bit, presentFrame's order: gain, displayOffset, transform, dither, quantize. rgb is width*height*3.
[[nodiscard]] std::vector<unsigned char> encodeForDisplay(std::span<const float> rgb, int width, int height,
                                                           float exposureEv, bool applyDisplayTransform,
                                                           float displayOffset = 0.0F);

}  // namespace pathtracer::gfx
