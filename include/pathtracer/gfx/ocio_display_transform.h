#pragma once

#include <optional>

#include <glm/glm.hpp>

#include "pathtracer/gfx/shader_program.h"

namespace pathtracer::gfx {

// The colour pipeline's definition for CPU-side consumers. kOcioConfigName is pinned, not "-latest", so the transform cannot move.
inline constexpr const char* kOcioConfigName = "cg-config-v1.0.0_aces-v1.3_ocio-v2.1";
inline constexpr const char* kOcioSceneColorSpace = "Linear Rec.709 (sRGB)";
// Verified against the installed library: this view is a pure colorimetric pass (0->0, 1->1, zero crosstalk).
inline constexpr const char* kOcioView = "Un-tone-mapped";
inline constexpr const char* kOcioSrgbDisplay = "sRGB - Display";
inline constexpr const char* kOcioRec709Display = "Rec.1886 Rec.709 - Display";

// Owns three display shaders (sRGB LUT, Rec.1886/Rec.709 LUT, raw passthrough), compiled at startup, switched by the debug 'L' key.
class OcioDisplayTransform {
public:
    enum class Lut { Raw, SRGB, Rec709 };

    // Builds all three shaders. Returns nullopt only on a GLSL compile or link failure, per ShaderProgram's contract.
    [[nodiscard]] static std::optional<OcioDisplayTransform> create();

    void setActiveLut(Lut lut) { activeLut_ = lut; }
    [[nodiscard]] Lut activeLut() const { return activeLut_; }

    // The display map before the curve, per channel: gain * value + offset. Radiance passes exp2(ev) and zero; a signed AOV auto-ranges.
    void setDisplayAffine(const glm::vec3& gain, const glm::vec3& offset) { gain_ = gain; displayOffset_ = offset; }

    // 0 = off, 1/2/3 isolate R/G/B broadcast to grey, after the affine map so a per-lane gain cannot tint it. Uploaded by bind().
    void setChannelView(int channelView) { channelView_ = channelView; }

    // 1.0 - rgb on the final display-referred colour, after the display curve and before dither -- the 'I' toggle.
    void setInvert(bool invert) { invert_ = invert; }

    // 0 = off. Radial per-channel UV offset (R toward centre, B away) at the texture fetch, before exposure and the display curve.
    void setAberration(float aberration) { aberration_ = aberration; }

    [[nodiscard]] const ShaderProgram& activeShader() const {
        switch (activeLut_) {
            case Lut::SRGB:
                return srgbShader_;
            case Lut::Rec709:
                return rec709Shader_;
            case Lut::Raw:
            default:
                return rawShader_;
        }
    }

    // Uploads the display map to the active shader. Call once per frame, before PostProcessPass::draw consumes activeShader().
    void bind() const;

private:
    OcioDisplayTransform(ShaderProgram rawShader, ShaderProgram srgbShader,
                          ShaderProgram rec709Shader);

    ShaderProgram rawShader_;
    ShaderProgram srgbShader_;
    ShaderProgram rec709Shader_;
    int rawDisplayGainLoc_;
    int srgbDisplayGainLoc_;
    int rec709DisplayGainLoc_;
    int rawChannelViewLoc_;
    int srgbChannelViewLoc_;
    int rec709ChannelViewLoc_;
    int rawInvertLoc_;
    int srgbInvertLoc_;
    int rec709InvertLoc_;
    int rawAberrationLoc_;
    int srgbAberrationLoc_;
    int rec709AberrationLoc_;
    int rawDisplayOffsetLoc_;
    int srgbDisplayOffsetLoc_;
    int rec709DisplayOffsetLoc_;
    Lut activeLut_ = Lut::SRGB;
    glm::vec3 gain_{1.0F};
    int channelView_ = 0;
    bool invert_ = false;
    float aberration_ = 0.0F;
    glm::vec3 displayOffset_{0.0F};
};

}  // namespace pathtracer::gfx
