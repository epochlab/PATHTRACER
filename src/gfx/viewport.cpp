#include "pathtracer/gfx/viewport.h"

#include <algorithm>
#include <cmath>

namespace pathtracer::gfx {

ViewportRect fitAspect(double imageWidth, double imageHeight, int viewportWidth, int viewportHeight) {
    if (imageWidth <= 0 || imageHeight <= 0 || viewportWidth <= 0 || viewportHeight <= 0) {
        return {0, 0, 0, 0};
    }
    // The single scale both axes take, so the aspect is preserved by construction rather than corrected afterwards.
    const double scale = std::min(static_cast<double>(viewportWidth) / imageWidth,
                                  static_cast<double>(viewportHeight) / imageHeight);
    // Clamped because rounding a scale that exactly fills one axis can land one pixel past it.
    const int width = std::min(viewportWidth, std::max(1, static_cast<int>(std::lround(imageWidth * scale))));
    const int height = std::min(viewportHeight, std::max(1, static_cast<int>(std::lround(imageHeight * scale))));
    return {(viewportWidth - width) / 2, (viewportHeight - height) / 2, width, height};
}

}  // namespace pathtracer::gfx
