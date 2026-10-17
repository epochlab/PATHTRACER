#pragma once

#include "pathtracer/gfx/viewport.h"

namespace pathtracer::gfx {

class ShaderProgram;

// Draws a fullscreen triangle (the gl_VertexID trick, no VBO) into a framebuffer, sampling an HDR colour texture.
class PostProcessPass {
public:
    PostProcessPass();
    ~PostProcessPass();

    PostProcessPass(const PostProcessPass&) = delete;
    PostProcessPass& operator=(const PostProcessPass&) = delete;
    PostProcessPass(PostProcessPass&& other) noexcept;
    PostProcessPass& operator=(PostProcessPass&& other) noexcept;

    // Binds framebuffer (the viewer's default, 0), viewports imageRect, binds displayShader and hdrColorTexture on unit 0. Never clears.
    void draw(unsigned int hdrColorTexture, const ShaderProgram& displayShader, ViewportRect imageRect, unsigned int framebuffer) const;

private:
    unsigned int vao_ = 0;
};

}  // namespace pathtracer::gfx
