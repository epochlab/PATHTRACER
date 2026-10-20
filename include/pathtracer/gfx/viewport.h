#pragma once

#include <optional>

namespace pathtracer::gfx {

// A rectangle of the default framebuffer, in pixels, origin bottom-left as GL addresses it.
struct ViewportRect {
    int x;
    int y;
    int width;
    int height;
};

// Largest rect of the image's film extent's aspect that fits the viewport, centred, so the authored framing is what the display shows.
[[nodiscard]] ViewportRect fitAspect(double imageWidth, double imageHeight, int viewportWidth, int viewportHeight);

// A display zoom: nullopt fits the viewport (fitAspect), a level scales by 2^level, uniform nearest blocks and exact 2x2 box minifies.
using ZoomLevel = std::optional<int>;

// Steps level by steps (+ in, - out) over powers of two strictly below fit, the shorter extent spanning >= 1 px; past fit is fit.
[[nodiscard]] ZoomLevel stepZoom(ZoomLevel level, int steps, double imageWidth, double imageHeight, int viewportWidth,
                                 int viewportHeight);

// The image at level's scale, centred; level must come from stepZoom for this viewport, so the rect lies inside it.
[[nodiscard]] ViewportRect zoomedRect(double imageWidth, double imageHeight, int viewportWidth, int viewportHeight, ZoomLevel level);

}  // namespace pathtracer::gfx
