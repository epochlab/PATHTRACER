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

ZoomLevel stepZoom(ZoomLevel level, int steps, double imageWidth, double imageHeight, int viewportWidth, int viewportHeight) {
    if (imageWidth <= 0 || imageHeight <= 0 || viewportWidth <= 0 || viewportHeight <= 0) {
        return level;
    }
    const double fit = std::min(viewportWidth / imageWidth, viewportHeight / imageHeight);
    // ilogb is floor(log2): one less when fit is itself a power of two, so 2^below < fit strictly.
    const int below = std::ilogb(fit) - (std::ldexp(1.0, std::ilogb(fit)) == fit ? 1 : 0);
    // 2^-ilogb(m) >= 1/m, so the shorter extent spans at least one pixel.
    const int lowest = -std::ilogb(std::min(imageWidth, imageHeight));
    // Fit is the step above below; a window too small for any level that spans a pixel leaves only fit.
    const int requested = (level.has_value() ? *level : below + 1) + steps;
    if (requested > below || below < lowest) {
        return std::nullopt;
    }
    return std::max(requested, lowest);
}

ViewportRect zoomedRect(double imageWidth, double imageHeight, int viewportWidth, int viewportHeight, ZoomLevel level) {
    if (!level.has_value()) {
        return fitAspect(imageWidth, imageHeight, viewportWidth, viewportHeight);
    }
    if (viewportWidth <= 0 || viewportHeight <= 0) {
        return {0, 0, 0, 0};
    }
    // stepZoom's bounds put each extent in [1, viewport) before rounding, so neither clamp nor a 1 px floor is needed.
    const double scale = std::ldexp(1.0, *level);
    const int width = static_cast<int>(std::lround(imageWidth * scale));
    const int height = static_cast<int>(std::lround(imageHeight * scale));
    return {(viewportWidth - width) / 2, (viewportHeight - height) / 2, width, height};
}

}  // namespace pathtracer::gfx
