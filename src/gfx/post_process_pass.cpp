#include "pathtracer/gfx/post_process_pass.h"

#include <utility>

#include <GL/glew.h>

#include "pathtracer/gfx/gl_debug.h"
#include "pathtracer/gfx/shader_program.h"

namespace pathtracer::gfx {

PostProcessPass::PostProcessPass() {
    GL_CALL(glGenVertexArrays(1, &vao_));
}

PostProcessPass::~PostProcessPass() {
    if (vao_ != 0) {
        glDeleteVertexArrays(1, &vao_);
    }
}

PostProcessPass::PostProcessPass(PostProcessPass&& other) noexcept
    : vao_(std::exchange(other.vao_, 0)) {}

PostProcessPass& PostProcessPass::operator=(PostProcessPass&& other) noexcept {
    if (this != &other) {
        if (vao_ != 0) {
            glDeleteVertexArrays(1, &vao_);
        }
        vao_ = std::exchange(other.vao_, 0);
    }
    return *this;
}

// Not wrapped in GL_CALL: the entire body runs every frame.
void PostProcessPass::draw(unsigned int hdrColorTexture, const ShaderProgram& displayShader, ViewportRect imageRect,
                            unsigned int framebuffer) const {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glViewport(imageRect.x, imageRect.y, imageRect.width, imageRect.height);

    displayShader.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrColorTexture);

    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

// Not wrapped in GL_CALL: it runs every frame. One scissored clear of the grown rect; the image draw then overwrites its interior.
void drawImageBounds(ViewportRect imageRect, unsigned int framebuffer) {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glEnable(GL_SCISSOR_TEST);
    glScissor(imageRect.x - 1, imageRect.y - 1, imageRect.width + 2, imageRect.height + 2);
    glClearColor(kImageBoundsCode, kImageBoundsCode, kImageBoundsCode, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
}

}  // namespace pathtracer::gfx
