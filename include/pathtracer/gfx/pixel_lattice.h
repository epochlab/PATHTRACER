#pragma once

#include <cmath>
#include <numbers>

#include <glm/glm.hpp>

namespace pathtracer::gfx {

// Square is the row-major grid; Hexagonal sets odd rows half a pitch right of even ones at sqrt(3)/2 row spacing (Mersereau 1979).
enum class PixelLattice { Square, Hexagonal, Count };

// Index-parallel with PixelLattice, so the HUD dropdown, the config parser and the bench log name a lattice from one table.
inline constexpr const char* kPixelLatticeNames[] = {"Square", "Hexagonal"};
static_assert(sizeof(kPixelLatticeNames) / sizeof(kPixelLatticeNames[0]) == static_cast<int>(PixelLattice::Count),
              "kPixelLatticeNames must stay index-parallel with PixelLattice");

// Row spacing in column pitches: sqrt(3)/2 puts all six hexagonal neighbours at distance 1, so every cell is a regular hexagon.
[[nodiscard]] constexpr float rowPitch(PixelLattice lattice) {
    return lattice == PixelLattice::Hexagonal ? 0.5F * std::numbers::sqrt3_v<float> : 1.0F;
}

// Side of the square with a cell's area 1 x rowPitch: the isotropic pixel size that footprints, strata and line widths are measured in.
[[nodiscard]] inline float pixelSide(PixelLattice lattice) {
    return std::sqrt(rowPitch(lattice));
}

// Pixel (x, y)'s film centre in column pitches, row 0 at the top; hexagonal rows sit 1/4 either side of x + 1/2, centring the lattice.
[[nodiscard]] inline glm::vec2 pixelCentre(PixelLattice lattice, int x, int y) {
    const float column = lattice == PixelLattice::Hexagonal ? ((y & 1) != 0 ? 0.75F : 0.25F) : 0.5F;
    return {static_cast<float>(x) + column, (static_cast<float>(y) + 0.5F) * rowPitch(lattice)};
}

// The film a width x height lattice spans in column pitches, whose ratio is the primary rays' aspect.
[[nodiscard]] inline glm::vec2 frameExtent(PixelLattice lattice, int width, int height) {
    return {static_cast<float>(width), static_cast<float>(height) * rowPitch(lattice)};
}

}  // namespace pathtracer::gfx
