#include "pathtracer/gfx/bloom_pass.h"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <utility>

#include <GL/glew.h>

#include "pathtracer/debug/memory_tracker.h"
#include "pathtracer/gfx/display_optics.h"
#include "pathtracer/gfx/gl_debug.h"

namespace pathtracer::gfx {

namespace {

// Texture units the passes sample; presentFrame builds before OcioDisplayTransform::bind and PostProcessPass::draw rebind them.
constexpr GLint kSourceUnit = 0;
constexpr GLint kBaseUnit = 1;

// Lanes stored per texel: GL_RGBA, there being no sampled 3-lane half or float format on Metal, Vulkan or D3D (see texture.cpp).
constexpr std::size_t kStoredLanes = 4;

// Each pass addresses its target by gl_FragCoord, not the vertex shader's v-flipped vUv, so every level keeps the source's row order.
constexpr const char* kTargetUvGlsl =
    "#version 410 core\n"
    "out vec4 fragColor;\n"
    "uniform sampler2D uSource;\n"
    "uniform vec2 uSourceTexel;\n"
    "uniform vec2 uTargetTexel;\n"
    "vec3 tap(vec2 uv, vec2 offset) { return texture(uSource, uv + offset * uSourceTexel).rgb; }\n";

// The pass's taps as GLSL, one fetch each: the weights are dyadic, so max_digits10 prints each exactly as the CPU path holds it.
std::string tapSumGlsl(std::span<const BloomTap> taps) {
    std::ostringstream glsl;
    glsl << std::setprecision(std::numeric_limits<float>::max_digits10) << "    vec2 uv = gl_FragCoord.xy * uTargetTexel;\n"
         << "    vec3 sum = vec3(0.0);\n";
    for (const BloomTap& tap : taps) {
        glsl << "    sum += " << tap.weight << " * tap(uv, vec2(" << tap.dx << ", " << tap.dy << "));\n";
    }
    return glsl.str();
}

std::string downGlsl() {
    return "void main() {\n" + tapSumGlsl(kBloomDownTaps) + "    fragColor = vec4(sum, 1.0);\n}\n";
}

// Adds this octave's down level, so each octave enters the sum once at unit weight; uScale is 1/L on the last pass, else 1.
std::string upGlsl() {
    return "uniform sampler2D uBase;\n"
           "uniform bool uHasBase;\n"
           "uniform float uScale;\n"
           "void main() {\n" +
           tapSumGlsl(kBloomUpTaps) +
           "    if (uHasBase) { sum += texture(uBase, uv).rgb; }\n"
           "    fragColor = vec4(sum * uScale, 1.0);\n}\n";
}

GLint glInternalFormat(ScalarType format) {
    return format == ScalarType::Float16 ? GL_RGBA16F : GL_RGBA32F;
}

std::size_t bytesPerTexel(ScalarType format) {
    return kStoredLanes * (format == ScalarType::Float16 ? 2 : 4);
}

void setTexelSize(int location, int width, int height) {
    glUniform2f(location, 1.0F / static_cast<float>(width), 1.0F / static_cast<float>(height));
}

}  // namespace

std::optional<BloomPass> BloomPass::create() {
    const std::optional<std::string> vertSrc = ShaderProgram::readSource(ASSET_ROOT_DIR "/shaders/fullscreen_triangle.vert");
    if (!vertSrc) {
        return std::nullopt;
    }
    std::optional<ShaderProgram> downShader = ShaderProgram::loadFromSource(*vertSrc, kTargetUvGlsl + downGlsl());
    std::optional<ShaderProgram> upShader = ShaderProgram::loadFromSource(*vertSrc, kTargetUvGlsl + upGlsl());
    if (!downShader || !upShader) {
        return std::nullopt;
    }
    downShader->use();
    GL_CALL(glUniform1i(downShader->uniformLocation("uSource"), kSourceUnit));
    upShader->use();
    GL_CALL(glUniform1i(upShader->uniformLocation("uSource"), kSourceUnit));
    GL_CALL(glUniform1i(upShader->uniformLocation("uBase"), kBaseUnit));
    return BloomPass(std::move(*downShader), std::move(*upShader));
}

BloomPass::BloomPass(ShaderProgram downShader, ShaderProgram upShader)
    : downShader_(std::move(downShader)),
      upShader_(std::move(upShader)),
      downSourceTexelLoc_(downShader_.uniformLocation("uSourceTexel")),
      downTargetTexelLoc_(downShader_.uniformLocation("uTargetTexel")),
      upSourceTexelLoc_(upShader_.uniformLocation("uSourceTexel")),
      upTargetTexelLoc_(upShader_.uniformLocation("uTargetTexel")),
      upScaleLoc_(upShader_.uniformLocation("uScale")),
      upHasBaseLoc_(upShader_.uniformLocation("uHasBase")) {
    GL_CALL(glGenVertexArrays(1, &vao_));
    // A complete result from the start: an unbound sampler is incomplete, which the driver replaces with zero and logs, sampled or not.
    allocate(1, 1, format_);
}

BloomPass::~BloomPass() {
    destroy();
}

void BloomPass::destroy() {
    releaseLevels();
    if (vao_ != 0) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
}

BloomPass::BloomPass(BloomPass&& other) noexcept
    : downShader_(std::move(other.downShader_)),
      upShader_(std::move(other.upShader_)),
      downSourceTexelLoc_(other.downSourceTexelLoc_),
      downTargetTexelLoc_(other.downTargetTexelLoc_),
      upSourceTexelLoc_(other.upSourceTexelLoc_),
      upTargetTexelLoc_(other.upTargetTexelLoc_),
      upScaleLoc_(other.upScaleLoc_),
      upHasBaseLoc_(other.upHasBaseLoc_),
      vao_(std::exchange(other.vao_, 0)),
      down_(std::exchange(other.down_, {})),
      up_(std::exchange(other.up_, {})),
      format_(other.format_),
      byteSize_(std::exchange(other.byteSize_, 0)),
      wrapsHorizontally_(other.wrapsHorizontally_) {}

BloomPass& BloomPass::operator=(BloomPass&& other) noexcept {
    if (this != &other) {
        destroy();
        downShader_ = std::move(other.downShader_);
        upShader_ = std::move(other.upShader_);
        downSourceTexelLoc_ = other.downSourceTexelLoc_;
        downTargetTexelLoc_ = other.downTargetTexelLoc_;
        upSourceTexelLoc_ = other.upSourceTexelLoc_;
        upTargetTexelLoc_ = other.upTargetTexelLoc_;
        upScaleLoc_ = other.upScaleLoc_;
        upHasBaseLoc_ = other.upHasBaseLoc_;
        vao_ = std::exchange(other.vao_, 0);
        down_ = std::exchange(other.down_, {});
        up_ = std::exchange(other.up_, {});
        format_ = other.format_;
        byteSize_ = std::exchange(other.byteSize_, 0);
        wrapsHorizontally_ = other.wrapsHorizontally_;
    }
    return *this;
}

void BloomPass::releaseLevels() {
    for (const std::vector<Level>* chain : {&down_, &up_}) {
        for (const Level& level : *chain) {
            glDeleteFramebuffers(1, &level.fbo);
            glDeleteTextures(1, &level.texture);
        }
    }
    down_.clear();
    up_.clear();
    pathtracer::debug::trackGpuFree(byteSize_);
    byteSize_ = 0;
}

// bloomOctaves(width, height) octaves, the display_optics.h definition the CPU path shares.
void BloomPass::allocate(int width, int height, ScalarType format) {
    releaseLevels();
    format_ = format;
    const int octaves = bloomOctaves(width, height);
    // One framebuffer per level, attached once, so a build only binds them.
    const auto makeLevel = [&](int octave) {
        Level level{0, 0, std::max(1, width >> octave), std::max(1, height >> octave)};
        GL_CALL(glGenTextures(1, &level.texture));
        GL_CALL(glBindTexture(GL_TEXTURE_2D, level.texture));
        GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, glInternalFormat(format), level.width, level.height, 0, GL_RGBA, GL_FLOAT, nullptr));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapsHorizontally_ ? GL_REPEAT : GL_CLAMP_TO_EDGE));
        GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
        GL_CALL(glGenFramebuffers(1, &level.fbo));
        GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, level.fbo));
        GL_CALL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, level.texture, 0));
        byteSize_ += static_cast<std::size_t>(level.width) * static_cast<std::size_t>(level.height) * bytesPerTexel(format);
        return level;
    };
    for (int octave = 1; octave <= octaves; ++octave) {
        down_.push_back(makeLevel(octave));
    }
    for (int octave = 0; octave < octaves; ++octave) {
        up_.push_back(makeLevel(octave));
    }
    GL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_CALL(glBindTexture(GL_TEXTURE_2D, 0));
    pathtracer::debug::trackGpuAlloc(byteSize_);
}

// A lat-long's left and right edges are one meridian, so its glare wraps across the seam like the display texture's taps.
void BloomPass::setWrapsHorizontally(bool wrapsHorizontally) {
    wrapsHorizontally_ = wrapsHorizontally;
    for (const std::vector<Level>* chain : {&down_, &up_}) {
        for (const Level& level : *chain) {
            glBindTexture(GL_TEXTURE_2D, level.texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapsHorizontally ? GL_REPEAT : GL_CLAMP_TO_EDGE);
        }
    }
}

// One pass of the active program: source on the source unit, drawn over the whole of target.
void BloomPass::draw(const Level& source, const Level& target, int sourceTexelLoc, int targetTexelLoc) const {
    glActiveTexture(GL_TEXTURE0 + kSourceUnit);
    glBindTexture(GL_TEXTURE_2D, source.texture);
    setTexelSize(sourceTexelLoc, source.width, source.height);
    setTexelSize(targetTexelLoc, target.width, target.height);
    glBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
    glViewport(0, 0, target.width, target.height);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

// Not wrapped in GL_CALL: runs once per upload while bloom is on.
void BloomPass::build(unsigned int sourceTexture, int width, int height, ScalarType format, bool wrapsHorizontally) {
    if (up_.front().width != width || up_.front().height != height || format_ != format) {
        wrapsHorizontally_ = wrapsHorizontally;
        allocate(width, height, format);
    } else if (wrapsHorizontally_ != wrapsHorizontally) {
        setWrapsHorizontally(wrapsHorizontally);
    }
    glBindVertexArray(vao_);

    downShader_.use();
    const Level input{sourceTexture, 0, width, height};
    const Level* source = &input;
    for (const Level& level : down_) {
        draw(*source, level, downSourceTexelLoc_, downTargetTexelLoc_);
        source = &level;
    }

    // up_[k] = down octave k + tent(up_[k+1]) from the coarsest down; up_[0] = tent(up_[1]) / L, each octave weighted 1/L.
    upShader_.use();
    const auto octaves = static_cast<int>(down_.size());
    for (int k = octaves - 1; k >= 0; --k) {
        const bool hasBase = k > 0;
        if (hasBase) {
            glActiveTexture(GL_TEXTURE0 + kBaseUnit);
            glBindTexture(GL_TEXTURE_2D, down_[static_cast<std::size_t>(k - 1)].texture);
        }
        glUniform1i(upHasBaseLoc_, hasBase ? 1 : 0);
        glUniform1f(upScaleLoc_, hasBase ? 1.0F : 1.0F / static_cast<float>(octaves));
        const Level& target = up_[static_cast<std::size_t>(k)];
        draw(*source, target, upSourceTexelLoc_, upTargetTexelLoc_);
        source = &target;
    }
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void BloomPass::bind(unsigned int unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, up_.front().texture);
}

}  // namespace pathtracer::gfx
