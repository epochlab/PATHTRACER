#include "pathtracer/gfx/ocio_display_transform.h"

#include <bit>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include <GL/glew.h>

#include <OpenColorIO/OpenColorIO.h>

#include "pathtracer/gfx/gl_debug.h"
#include "pathtracer/gfx/ocio_cpu_transform.h"

namespace OCIO = OCIO_NAMESPACE;

namespace pathtracer::gfx {

namespace {

// Declared in ocio_display_transform.h so CPU-side consumers reproduce this exact transform -- see the header.
constexpr const char* kBuiltinConfigName = kOcioConfigName;
constexpr const char* kSceneColorSpace = kOcioSceneColorSpace;
constexpr const char* kView = kOcioView;

// Channel isolation after the exposure affine and before the display curve; the CPU bake isolates first, equal while gain is one scalar.
constexpr const char* kChannelViewGlsl =
    "uniform int uChannelView;\n";

// Applied after the affine map, so isolating one lane of a per-lane map still broadcasts a true grey rather than three gains.
constexpr const char* kApplyChannelViewGlsl =
    "    if (uChannelView == 1) { hdrColor = vec3(hdrColor.r); }\n"
    "    else if (uChannelView == 2) { hdrColor = vec3(hdrColor.g); }\n"
    "    else if (uChannelView == 3) { hdrColor = vec3(hdrColor.b); }\n";

// 1.0 - rgb, applied to the final display-referred colour (after the display curve/Raw passthrough, before dither).
constexpr const char* kInvertGlsl = "uniform bool uInvert;\n";

// Radial per-channel UV offset, 0 = off: R toward centre, B away, scaling with distance from centre. Beauty only, so no separate pass.
constexpr const char* kAberrationGlsl =
    "uniform float uAberration;\n"
    "vec3 sampleAberrated(vec2 uv) {\n"
    "    vec2 dir = uv - vec2(0.5);\n"
    "    vec3 color;\n"
    "    color.r = texture(uHdrColor, uv - dir * uAberration).r;\n"
    "    color.g = texture(uHdrColor, uv).g;\n"
    "    color.b = texture(uHdrColor, uv + dir * uAberration).b;\n"
    "    return color;\n"
    "}\n";

// Texture unit of displayDitherMask(); unit 0 is uHdrColor, bound by PostProcessPass::draw.
constexpr unsigned int kDitherMaskUnit = 1;

// The shader wraps by AND, since integer modulo is emulated on GPUs; the CPU's % by a constant power of two compiles to the same.
static_assert(std::has_single_bit(static_cast<unsigned>(kDitherMaskSize)));

// Energy-conserving glare: (1 - k) I + k (I * PSF) in scene-linear, before the affine map, so exposure scales both terms alike.
constexpr const char* kBloomGlsl =
    "uniform sampler2D uBloom;\n"
    "uniform float uBloomStrength;\n";

constexpr const char* kApplyBloomGlsl =
    "    if (uBloomStrength > 0.0) { hdrColor = mix(hdrColor, texture(uBloom, vUv).rgb, uBloomStrength); }\n";

// TPDF dither before the 8-bit quantization, else gradients band: encodeForDisplay's table and tiling, so a 1:1 frame matches it.
constexpr const char* kDitherGlsl =
    "uniform sampler2D uDitherMask;\n"
    "uniform ivec2 uDitherOrigin;\n"
    "vec3 ditherOffset() {\n"
    "    ivec2 texel = ivec2(int(gl_FragCoord.x) - uDitherOrigin.x, uDitherOrigin.y - int(gl_FragCoord.y));\n"
    "    return vec3(texelFetch(uDitherMask, texel & (textureSize(uDitherMask, 0) - 1), 0).r);\n"
    "}\n";

// Wraps OCIO's generated GLSL in this project's fragment program; the affine gain and offset run before it, so it stays scene-referred.
std::string buildFragmentSource(const std::string& ocioShaderText, const std::string& functionName) {
    std::ostringstream src;
    src << "#version 410 core\n\n"
        << "in vec2 vUv;\n"
        << "out vec4 fragColor;\n\n"
        << "uniform sampler2D uHdrColor;\n"
        << "uniform vec3 uDisplayGain;\n"
        << "uniform vec3 uDisplayOffset;\n"
        << kChannelViewGlsl << kInvertGlsl << kAberrationGlsl << kBloomGlsl << "\n"
        << kDitherGlsl
        << ocioShaderText << "\n"
        << "void main() {\n"
        << "    vec3 hdrColor = sampleAberrated(vUv);\n"
        << kApplyBloomGlsl
        << "    hdrColor = hdrColor * uDisplayGain + uDisplayOffset;\n"
        << kApplyChannelViewGlsl
        << "    vec4 exposed = vec4(hdrColor, 1.0);\n"
        << "    vec3 displayColor = " << functionName << "(exposed).rgb;\n"
        << "    if (uInvert) { displayColor = 1.0 - displayColor; }\n"
        << "    fragColor = vec4(displayColor + ditherOffset(), 1.0);\n"
        << "}\n";
    return src.str();
}

// No OCIO at all: the affine map applied, then output with no display encode, so 'L' can cycle to a genuinely unencoded state.
std::string buildRawFragmentSource() {
    return std::string("#version 410 core\n\n"
                       "in vec2 vUv;\n"
                       "out vec4 fragColor;\n\n"
                       "uniform sampler2D uHdrColor;\n"
                       "uniform vec3 uDisplayGain;\n"
                       "uniform vec3 uDisplayOffset;\n") +
           kChannelViewGlsl + kInvertGlsl + kAberrationGlsl + kBloomGlsl + kDitherGlsl +
           "\nvoid main() {\n"
           "    vec3 hdrColor = sampleAberrated(vUv);\n" +
           kApplyBloomGlsl +
           "    hdrColor = hdrColor * uDisplayGain + uDisplayOffset;\n" +
           kApplyChannelViewGlsl +
           "    vec3 displayColor = hdrColor;\n"
           "    if (uInvert) { displayColor = 1.0 - displayColor; }\n"
           "    fragColor = vec4(displayColor + ditherOffset(), 1.0);\n"
           "}\n";
}

// One LUT's fragment source through OCIO's Display/View API. Both displays resolve to pure Matrix+Gamma, verified against OCIO source.
std::string buildOcioFragmentSource(const char* display, const char* functionName) {
    try {
        const OCIO::ConstConfigRcPtr config = OCIO::Config::CreateFromBuiltinConfig(kBuiltinConfigName);
        const OCIO::ConstProcessorRcPtr processor = config->getProcessor(
            kSceneColorSpace, display, kView, OCIO::TRANSFORM_DIR_FORWARD);
        const OCIO::ConstGPUProcessorRcPtr gpuProcessor = processor->getDefaultGPUProcessor();

        OCIO::GpuShaderDescRcPtr shaderDesc = OCIO::GpuShaderDesc::CreateShaderDesc();
        shaderDesc->setLanguage(OCIO::GPU_LANGUAGE_GLSL_4_0);
        shaderDesc->setFunctionName(functionName);
        gpuProcessor->extractGpuShaderInfo(shaderDesc);

        // Defensive, not documentary: this design assumes zero LUT textures. Skipping the upload would render wrong colours.
        if (shaderDesc->getNumTextures() != 0 || shaderDesc->getNum3DTextures() != 0) {
            std::cerr << "OcioDisplayTransform: " << display
                      << " processor unexpectedly requires LUT textures ("
                      << shaderDesc->getNumTextures() << " 1D/2D, " << shaderDesc->getNum3DTextures()
                      << " 3D): design assumption broken, aborting\n";
            std::exit(EXIT_FAILURE);
        }

        return buildFragmentSource(shaderDesc->getShaderText(), functionName);
    } catch (const OCIO::Exception& e) {
        std::cerr << "OcioDisplayTransform: OCIO error building " << display << ": " << e.what()
                   << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

std::optional<OcioDisplayTransform> OcioDisplayTransform::create() {
    // Same fullscreen-triangle vertex shader for all three; only the fragment side varies.
    const std::optional<std::string> vertSrc =
        ShaderProgram::readSource(ASSET_ROOT_DIR "/shaders/fullscreen_triangle.vert");
    if (!vertSrc) {
        return std::nullopt;
    }

    const std::string rawFragSrc = buildRawFragmentSource();
    const std::string srgbFragSrc = buildOcioFragmentSource(kOcioSrgbDisplay, "OCIODisplaySRGB");
    const std::string rec709FragSrc =
        buildOcioFragmentSource(kOcioRec709Display, "OCIODisplayRec709");

    std::optional<ShaderProgram> rawShader = ShaderProgram::loadFromSource(*vertSrc, rawFragSrc);
    std::optional<ShaderProgram> srgbShader = ShaderProgram::loadFromSource(*vertSrc, srgbFragSrc);
    std::optional<ShaderProgram> rec709Shader =
        ShaderProgram::loadFromSource(*vertSrc, rec709FragSrc);
    if (!rawShader || !srgbShader || !rec709Shader) {
        return std::nullopt;
    }

    // One-time texture-unit assignment, matching main.cpp's existing explicit-uniform convention for uBaseColor/uHdrColor.
    for (const ShaderProgram* shader : {&*rawShader, &*srgbShader, &*rec709Shader}) {
        shader->use();
        GL_CALL(glUniform1i(shader->uniformLocation("uHdrColor"), 0));
        GL_CALL(glUniform1i(shader->uniformLocation("uDitherMask"), static_cast<GLint>(kDitherMaskUnit)));
        GL_CALL(glUniform1i(shader->uniformLocation("uBloom"), static_cast<GLint>(kBloomTextureUnit)));
    }

    // Float32, so the shader adds exactly the offsets the CPU encode adds.
    Texture ditherMask = Texture::createFromFloatPixels(kDitherMaskSize, kDitherMaskSize, 1, displayDitherMask().data(),
                                                        ScalarType::Float32);
    return OcioDisplayTransform(std::move(*rawShader), std::move(*srgbShader),
                                 std::move(*rec709Shader), std::move(ditherMask));
}

OcioDisplayTransform::OcioDisplayTransform(ShaderProgram rawShader, ShaderProgram srgbShader,
                                            ShaderProgram rec709Shader, Texture ditherMask)
    : rawShader_(std::move(rawShader)),
      srgbShader_(std::move(srgbShader)),
      rec709Shader_(std::move(rec709Shader)),
      ditherMask_(std::move(ditherMask)),
      rawDisplayGainLoc_(rawShader_.uniformLocation("uDisplayGain")),
      srgbDisplayGainLoc_(srgbShader_.uniformLocation("uDisplayGain")),
      rec709DisplayGainLoc_(rec709Shader_.uniformLocation("uDisplayGain")),
      rawChannelViewLoc_(rawShader_.uniformLocation("uChannelView")),
      srgbChannelViewLoc_(srgbShader_.uniformLocation("uChannelView")),
      rec709ChannelViewLoc_(rec709Shader_.uniformLocation("uChannelView")),
      rawInvertLoc_(rawShader_.uniformLocation("uInvert")),
      srgbInvertLoc_(srgbShader_.uniformLocation("uInvert")),
      rec709InvertLoc_(rec709Shader_.uniformLocation("uInvert")),
      rawAberrationLoc_(rawShader_.uniformLocation("uAberration")),
      srgbAberrationLoc_(srgbShader_.uniformLocation("uAberration")),
      rec709AberrationLoc_(rec709Shader_.uniformLocation("uAberration")),
      rawDisplayOffsetLoc_(rawShader_.uniformLocation("uDisplayOffset")),
      srgbDisplayOffsetLoc_(srgbShader_.uniformLocation("uDisplayOffset")),
      rec709DisplayOffsetLoc_(rec709Shader_.uniformLocation("uDisplayOffset")),
      rawDitherOriginLoc_(rawShader_.uniformLocation("uDitherOrigin")),
      srgbDitherOriginLoc_(srgbShader_.uniformLocation("uDitherOrigin")),
      rec709DitherOriginLoc_(rec709Shader_.uniformLocation("uDitherOrigin")),
      rawBloomStrengthLoc_(rawShader_.uniformLocation("uBloomStrength")),
      srgbBloomStrengthLoc_(srgbShader_.uniformLocation("uBloomStrength")),
      rec709BloomStrengthLoc_(rec709Shader_.uniformLocation("uBloomStrength")) {}

// Not wrapped in GL_CALL: runs every frame.
void OcioDisplayTransform::bind() const {
    const ShaderProgram& shader = activeShader();
    int displayGainLoc = rawDisplayGainLoc_;
    int channelViewLoc = rawChannelViewLoc_;
    int invertLoc = rawInvertLoc_;
    int aberrationLoc = rawAberrationLoc_;
    int displayOffsetLoc = rawDisplayOffsetLoc_;
    int ditherOriginLoc = rawDitherOriginLoc_;
    int bloomStrengthLoc = rawBloomStrengthLoc_;
    if (activeLut_ == Lut::SRGB) {
        displayGainLoc = srgbDisplayGainLoc_;
        channelViewLoc = srgbChannelViewLoc_;
        invertLoc = srgbInvertLoc_;
        aberrationLoc = srgbAberrationLoc_;
        displayOffsetLoc = srgbDisplayOffsetLoc_;
        ditherOriginLoc = srgbDitherOriginLoc_;
        bloomStrengthLoc = srgbBloomStrengthLoc_;
    } else if (activeLut_ == Lut::Rec709) {
        displayGainLoc = rec709DisplayGainLoc_;
        channelViewLoc = rec709ChannelViewLoc_;
        invertLoc = rec709InvertLoc_;
        aberrationLoc = rec709AberrationLoc_;
        displayOffsetLoc = rec709DisplayOffsetLoc_;
        ditherOriginLoc = rec709DitherOriginLoc_;
        bloomStrengthLoc = rec709BloomStrengthLoc_;
    }
    shader.use();
    glUniform3f(displayGainLoc, gain_.r, gain_.g, gain_.b);
    glUniform1i(channelViewLoc, channelView_);
    glUniform1i(invertLoc, invert_ ? 1 : 0);
    glUniform1f(aberrationLoc, aberration_);
    glUniform3f(displayOffsetLoc, displayOffset_.r, displayOffset_.g, displayOffset_.b);
    glUniform1f(bloomStrengthLoc, bloomStrength_);
    // The image's top-left pixel in window coordinates, whose y axis points up.
    glUniform2i(ditherOriginLoc, imageRect_.x, imageRect_.y + imageRect_.height - 1);
    ditherMask_.bind(kDitherMaskUnit);
}

}  // namespace pathtracer::gfx
