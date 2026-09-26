#pragma once

#include <optional>

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

    // ev is a stops adjustment; the GPU multiplier before the display curve is pow(2, ev). Seeded from DebugCameraController.
    void setExposureEv(float ev) { exposureEv_ = ev; }

    // 0 = off, 1/2/3 isolate R/G/B broadcast to grey, applied before exposure. Uploaded by bind(), so a switch is a uniform write.
    void setChannelView(int channelView) { channelView_ = channelView; }

    // 1.0 - rgb on the final display-referred colour, after the display curve and before dither -- the 'I' toggle.
    void setInvert(bool invert) { invert_ = invert; }

    // Added after the exposure multiply, making the display map affine: 0 keeps the pure gain, 0.5 lands a signed AOV's zero at mid-grey.
    void setDisplayOffset(float offset) { displayOffset_ = offset; }

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

    // Uploads the exposure uniform to the active shader. Call once per frame, before PostProcessPass::draw consumes activeShader().
    void bind() const;

private:
    OcioDisplayTransform(ShaderProgram rawShader, ShaderProgram srgbShader,
                          ShaderProgram rec709Shader);

    ShaderProgram rawShader_;
    ShaderProgram srgbShader_;
    ShaderProgram rec709Shader_;
    int rawExposureLoc_;
    int srgbExposureLoc_;
    int rec709ExposureLoc_;
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
    float exposureEv_ = 0.0F;
    int channelView_ = 0;
    bool invert_ = false;
    float aberration_ = 0.0F;
    float displayOffset_ = 0.0F;
};

}  // namespace pathtracer::gfx
