#pragma once

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

}  // namespace pathtracer::gfx
