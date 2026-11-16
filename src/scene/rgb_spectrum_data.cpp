#include "rgb_spectrum_data.h"

#include "pathtracer/scene/rgb_spectrum_table.h"

namespace pathtracer::scene {

namespace {

// #embed is C++26 (P1967); clang takes it in C++20 as an extension, so the 9.4 MB table ships as bytes, not 25 MB of literal text.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
alignas(float) constexpr unsigned char kTable[] = {
#embed "rgb_spectrum.bin"
};
#pragma clang diagnostic pop

static_assert(sizeof(kTable) == rgb_spectrum::kFloats * sizeof(float), "the table is not the layout the reader expects");

}  // namespace

std::span<const unsigned char> rgbSpectrumTableBytes() { return kTable; }

}  // namespace pathtracer::scene
