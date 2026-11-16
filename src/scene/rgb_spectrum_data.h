#pragma once

#include <span>

namespace pathtracer::scene {

// The build-time table as rgb_spectrum_table wrote it: rgb_spectrum::kFloats native floats, read by memcpy.
[[nodiscard]] std::span<const unsigned char> rgbSpectrumTableBytes();

}  // namespace pathtracer::scene
