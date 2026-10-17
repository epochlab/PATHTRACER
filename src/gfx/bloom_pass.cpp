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

// pixel_lattice.h's fold over a hexagonal level, texelFetch-addressed; every % takes non-negative operands, where GLSL 4.10 defines it.
constexpr const char* kHexagonalCellGlsl =
    "#version 410 core\n"
    "out vec4 fragColor;\n"
    "uniform sampler2D uSource;\n"
    "uniform bool uWraps;\n"
    "int positiveMod(int i, int n) { return i >= 0 ? i % n : n - 1 - ((-i - 1) % n); }\n"
    "int halfFloor(int i) { return (i - positiveMod(i, 2)) / 2; }\n"
    "int mirrorIndex(int i, int extent) {\n"
    "    if (extent <= 1) { return 0; }\n"
    "    int period = 2 * extent - 2;\n"
    "    int folded = positiveMod(i, period);\n"
    "    return folded < extent ? folded : period - folded;\n"
    "}\n"
    "vec3 cell(int x, int y) {\n"
    "    ivec2 size = textureSize(uSource, 0);\n"
    "    int row = mirrorIndex(y, size.y);\n"
    "    int column = 2 * x + positiveMod(y, 2);\n"
    "    int folded = uWraps ? positiveMod(column, 2 * size.x) : mirrorIndex(column, 2 * size.x);\n"
    "    return texelFetch(uSource, ivec2((folded - (row & 1)) >> 1, row), 0).rgb;\n"
    "}\n";

// Loop's adjoint at coarse (j, m), fine (2j + (m & 1), 2m): each mask entry over 4, at its column offset dq + floor(dr / 2).
std::string hexagonalDownGlsl() {
    std::ostringstream glsl;
    glsl << std::setprecision(std::numeric_limits<float>::max_digits10) << "void main() {\n"
         << "    ivec2 target = ivec2(gl_FragCoord.xy);\n"
         << "    int x = 2 * target.x + (target.y & 1);\n"
         << "    int y = 2 * target.y;\n"
         << "    vec3 sum = vec3(0.0);\n";
    for (const HexBloomTap& tap : kHexBloomMask) {
        glsl << "    sum += " << (0.25F * tap.weight) << " * cell(x + " << (tap.dq + (tap.dr >> 1)) << ", y + " << tap.dr << ");\n";
    }
    glsl << "    fragColor = vec4(sum, 1.0);\n}\n";
    return glsl.str();
}

// Loop subdivision at fine (x, y): its coset's mask entries, each at coarse axial (q - dq, r - dr) / 2; base and scale as upGlsl.
std::string hexagonalUpGlsl() {
    std::ostringstream glsl;
    glsl << std::setprecision(std::numeric_limits<float>::max_digits10) << "uniform sampler2D uBase;\n"
         << "uniform bool uHasBase;\n"
         << "uniform float uScale;\n"
         << "vec3 coarse(int q, int y, int dq, int dr) { int row = (y - dr) / 2; return cell(halfFloor(q - dq) + halfFloor(row), row); }\n"
         << "void main() {\n"
         << "    ivec2 fine = ivec2(gl_FragCoord.xy);\n"
         << "    int q = fine.x - (fine.y >> 1);\n"
         << "    int coset = positiveMod(q, 2) + 2 * (fine.y & 1);\n"
         << "    vec3 sum = vec3(0.0);\n";
    for (int coset = 0; coset < 4; ++coset) {
        glsl << "    " << (coset == 0 ? "if" : "else if") << " (coset == " << coset << ") {\n";
        for (const HexBloomTap& tap : hexBloomCoset(coset & 1, coset >> 1)) {
            glsl << "        sum += " << tap.weight << " * coarse(q, fine.y, " << tap.dq << ", " << tap.dr << ");\n";
        }
        glsl << "    }\n";
    }
    glsl << "    if (uHasBase) { sum += texelFetch(uBase, fine, 0).rgb; }\n"
         << "    fragColor = vec4(sum * uScale, 1.0);\n}\n";
    return glsl.str();
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
    // One lattice's two passes, compiled, bound to their units, and each uniform its programs declare located once.
    const auto pyramid = [&](const std::string& downSource, const std::string& upSource, bool hexagonal) -> std::optional<Pyramid> {
        std::optional<ShaderProgram> down = ShaderProgram::loadFromSource(*vertSrc, downSource);
        std::optional<ShaderProgram> up = ShaderProgram::loadFromSource(*vertSrc, upSource);
        if (!down || !up) {
            return std::nullopt;
        }
        down->use();
        GL_CALL(glUniform1i(down->uniformLocation("uSource"), kSourceUnit));
        up->use();
        GL_CALL(glUniform1i(up->uniformLocation("uSource"), kSourceUnit));
        GL_CALL(glUniform1i(up->uniformLocation("uBase"), kBaseUnit));
        // Square passes address by texel size, hexagonal ones by texelFetch and the seam flag: -1 marks the one a program lacks.
        const auto located = [](const ShaderProgram& program, const char* name, bool declared) { return declared ? program.uniformLocation(name) : -1; };
        // Located before the aggregate moves the programs in: its elements are evaluated in order.
        const int downSourceTexel = located(*down, "uSourceTexel", !hexagonal);
        const int downTargetTexel = located(*down, "uTargetTexel", !hexagonal);
        const int downWraps = located(*down, "uWraps", hexagonal);
        const int upSourceTexel = located(*up, "uSourceTexel", !hexagonal);
        const int upTargetTexel = located(*up, "uTargetTexel", !hexagonal);
        const int upWraps = located(*up, "uWraps", hexagonal);
        const int upScale = up->uniformLocation("uScale");
        const int upHasBase = up->uniformLocation("uHasBase");
        return Pyramid{std::move(*down), std::move(*up), downSourceTexel, downTargetTexel, downWraps, upSourceTexel, upTargetTexel,
                       upWraps,          upScale,        upHasBase};
    };

    std::optional<Pyramid> square = pyramid(kTargetUvGlsl + downGlsl(), kTargetUvGlsl + upGlsl(), false);
    std::optional<Pyramid> hexagonal = pyramid(kHexagonalCellGlsl + hexagonalDownGlsl(), kHexagonalCellGlsl + hexagonalUpGlsl(), true);
    if (!square || !hexagonal) {
        return std::nullopt;
    }
    return BloomPass(std::move(*square), std::move(*hexagonal));
}

BloomPass::BloomPass(Pyramid square, Pyramid hexagonal) : square_(std::move(square)), hexagonal_(std::move(hexagonal)) {
    GL_CALL(glGenVertexArrays(1, &vao_));
    // A complete result from the start: an unbound sampler is incomplete, which the driver replaces with zero and logs, sampled or not.
    allocate(1, 1, format_, lattice_);
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
    : square_(std::move(other.square_)),
      hexagonal_(std::move(other.hexagonal_)),
      vao_(std::exchange(other.vao_, 0)),
      down_(std::exchange(other.down_, {})),
      up_(std::exchange(other.up_, {})),
      format_(other.format_),
      lattice_(other.lattice_),
      byteSize_(std::exchange(other.byteSize_, 0)),
      wrapsHorizontally_(other.wrapsHorizontally_) {}

BloomPass& BloomPass::operator=(BloomPass&& other) noexcept {
    if (this != &other) {
        destroy();
        square_ = std::move(other.square_);
        hexagonal_ = std::move(other.hexagonal_);
        vao_ = std::exchange(other.vao_, 0);
        down_ = std::exchange(other.down_, {});
        up_ = std::exchange(other.up_, {});
        format_ = other.format_;
        lattice_ = other.lattice_;
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

// bloomOctaves(width, height) octaves of bloomLevelSize, the display_optics.h definitions the CPU path shares.
void BloomPass::allocate(int width, int height, ScalarType format, PixelLattice lattice) {
    releaseLevels();
    format_ = format;
    lattice_ = lattice;
    const int octaves = bloomOctaves(width, height);
    // One framebuffer per level, attached once, so a build only binds them.
    const auto makeLevel = [&](int octave) {
        const glm::ivec2 size = bloomLevelSize(lattice, width, height, octave, wrapsHorizontally_);
        Level level{0, 0, size.x, size.y};
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
void BloomPass::build(unsigned int sourceTexture, int width, int height, ScalarType format, bool wrapsHorizontally, PixelLattice lattice) {
    // A hexagonal level's width depends on the seam, so a wrap change there reshapes the pyramid rather than retuning its samplers.
    const bool reshaped = lattice == PixelLattice::Hexagonal && wrapsHorizontally_ != wrapsHorizontally;
    if (up_.front().width != width || up_.front().height != height || format_ != format || lattice_ != lattice || reshaped) {
        wrapsHorizontally_ = wrapsHorizontally;
        allocate(width, height, format, lattice);
    } else if (wrapsHorizontally_ != wrapsHorizontally) {
        setWrapsHorizontally(wrapsHorizontally);
    }
    glBindVertexArray(vao_);
    // texelFetch ignores GL_TEXTURE_WRAP_S, so the hexagonal programs fold or wrap in GLSL by uWraps; the square ones ignore it.
    const Pyramid& pyramid = lattice == PixelLattice::Hexagonal ? hexagonal_ : square_;

    pyramid.down.use();
    glUniform1i(pyramid.downWrapsLoc, wrapsHorizontally ? 1 : 0);
    const Level input{sourceTexture, 0, width, height};
    const Level* source = &input;
    for (const Level& level : down_) {
        draw(*source, level, pyramid.downSourceTexelLoc, pyramid.downTargetTexelLoc);
        source = &level;
    }

    // up_[k] = down octave k + up(up_[k+1]) from the coarsest down; up_[0] = up(up_[1]) / L, each octave weighted 1/L.
    pyramid.up.use();
    glUniform1i(pyramid.upWrapsLoc, wrapsHorizontally ? 1 : 0);
    const auto octaves = static_cast<int>(down_.size());
    for (int k = octaves - 1; k >= 0; --k) {
        const bool hasBase = k > 0;
        if (hasBase) {
            glActiveTexture(GL_TEXTURE0 + kBaseUnit);
            glBindTexture(GL_TEXTURE_2D, down_[static_cast<std::size_t>(k - 1)].texture);
        }
        glUniform1i(pyramid.upHasBaseLoc, hasBase ? 1 : 0);
        glUniform1f(pyramid.upScaleLoc, hasBase ? 1.0F : 1.0F / static_cast<float>(octaves));
        const Level& target = up_[static_cast<std::size_t>(k)];
        draw(*source, target, pyramid.upSourceTexelLoc, pyramid.upTargetTexelLoc);
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
