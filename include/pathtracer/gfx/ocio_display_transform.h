#pragma once

#include <optional>

#include <glm/glm.hpp>

#include "pathtracer/gfx/pixel_lattice.h"
#include "pathtracer/gfx/shader_program.h"
#include "pathtracer/gfx/texture.h"
#include "pathtracer/gfx/viewport.h"

namespace pathtracer::gfx {

// The colour pipeline's definition for CPU-side consumers. kOcioConfigName is pinned, not "-latest", so the transform cannot move.
inline constexpr const char* kOcioConfigName = "cg-config-v1.0.0_aces-v1.3_ocio-v2.1";
inline constexpr const char* kOcioSceneColorSpace = "ACEScg";
// glTF's linear colours (vertex colours) are BT.709 primaries at D65, the config's linear sRGB.
inline constexpr const char* kOcioLinearRec709 = "Linear Rec.709 (sRGB)";
// A colorimetric view with no tone curve: the scene space's gamut matrix to the display's primaries, then its curve.
inline constexpr const char* kOcioView = "Un-tone-mapped";
inline constexpr const char* kOcioSrgbDisplay = "sRGB - Display";
inline constexpr const char* kOcioRec709Display = "Rec.1886 Rec.709 - Display";

// Unit the display shaders sample uBloom from: the caller keeps BloomPass's result bound there, as the sampler is declared either way.
inline constexpr unsigned int kBloomTextureUnit = 2;

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

    // 0 = off; the fraction of energy scattered by the glare PSF, sampled from kBloomTextureUnit while nonzero.
    void setBloomStrength(float bloomStrength) { bloomStrength_ = bloomStrength; }

    // Per-lane log2 after bloom and before the affine map, which then carries the log window rather than the exposure.
    void setLogTransform(bool logTransform) { logTransform_ = logTransform; }

    // The displayed image's lattice and lens wrap: a hexagonal one is drawn cell by cell, its columns wrapping where the lens does.
    void setPixelLattice(PixelLattice pixelLattice, bool wrapsHorizontally) {
        pixelLattice_ = pixelLattice;
        wrapsHorizontally_ = wrapsHorizontally;
    }

    // The window rect the image is drawn into, which anchors the dither tile at the image's top-left as encodeForDisplay does.
    void setImageRect(ViewportRect imageRect) { imageRect_ = imageRect; }

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

    // Uploads the display map to the active shader and binds the dither mask. Call once per frame, before PostProcessPass::draw.
    void bind() const;

private:
    OcioDisplayTransform(ShaderProgram rawShader, ShaderProgram srgbShader,
                          ShaderProgram rec709Shader, Texture ditherMask);

    ShaderProgram rawShader_;
    ShaderProgram srgbShader_;
    ShaderProgram rec709Shader_;
    Texture ditherMask_;  // displayDitherMask() on kDitherMaskUnit, shared by all three shaders
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
    int rawDitherOriginLoc_;
    int srgbDitherOriginLoc_;
    int rec709DitherOriginLoc_;
    int rawBloomStrengthLoc_;
    int srgbBloomStrengthLoc_;
    int rec709BloomStrengthLoc_;
    int rawLogTransformLoc_;
    int srgbLogTransformLoc_;
    int rec709LogTransformLoc_;
    int rawHexagonalLoc_;
    int srgbHexagonalLoc_;
    int rec709HexagonalLoc_;
    int rawWrapsHorizontallyLoc_;
    int srgbWrapsHorizontallyLoc_;
    int rec709WrapsHorizontallyLoc_;
    Lut activeLut_ = Lut::SRGB;
    glm::vec3 gain_{1.0F};
    int channelView_ = 0;
    bool invert_ = false;
    float aberration_ = 0.0F;
    float bloomStrength_ = 0.0F;
    bool logTransform_ = false;
    glm::vec3 displayOffset_{0.0F};
    PixelLattice pixelLattice_ = PixelLattice::Square;
    bool wrapsHorizontally_ = false;
    ViewportRect imageRect_{};
};

}  // namespace pathtracer::gfx
