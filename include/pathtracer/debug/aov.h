#pragma once

#include <string_view>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"

namespace pathtracer::debug {

// Single source of truth for every selectable AOV. AppResources.aov stays int because ImGui::Combo needs int&.
enum class AovId : int {
    // Utility.
    Beauty = 0,
    Wireframe,  // combined AOV: white mesh-edge lines + one false-coloured bounding box per instance (rasterizer.h)
    Alpha,
    Depth,
    Lookahead,  // Depth remapped through profile.json's lookaheadDistance: 1 at the camera plane, 0 at that horizon and beyond
    HSV,
    Luminance,
    Sobel,
    Gabor,
    DoG,
    WorldPos,
    UV,
    MotionVector,  // geometric screen-space motion in pixels from the request's previous camera (rasterizer.h)
    // Perceptual: observer models over Beauty, as against the Utility block's image-space derivative operators.
    ColourOpponent,
    // Material.
    Normal,
    GeomNormal,
    Albedo,
    Metallic,
    Roughness,
    Tangent,
    ObjectID,
    AO,
    // Transport.
    Fresnel,
    IOR,
    BounceCount,
    SNR,
    // Lighting.
    DirectDiffuse,
    IndirectDiffuse,
    DirectSpecular,
    IndirectSpecular,
    Refraction,
    Shadow,
    Count  // sentinel, == array size, not itself a selectable value
};

// camelCase name, index-parallel to AovId (static_assert below): the one spelling HUD, CLI, profile.json and both APIs share.
inline constexpr const char* kAovNames[] = {
    "beauty", "wireframe", "alpha", "depth", "lookahead", "HSV", "luminance", "sobel",
    "gabor", "DoG", "worldPos", "UV", "motionVector", "colourOpponent",
    "normal", "geomNormal", "albedo", "metallic", "roughness", "tangent", "objectID", "AO",
    "fresnel", "IOR", "bounceCount", "SNR",
    "directDiffuse", "indirectDiffuse", "directSpecular", "indirectSpecular", "refraction", "shadow",
};
static_assert(sizeof(kAovNames) / sizeof(kAovNames[0]) == static_cast<int>(AovId::Count),
              "kAovNames must stay index-parallel with AovId");

// Which of the three producers computes each AOV: 10 accumulated path-traced lanes, 15 rasterizer lanes, 7 filters over Beauty.
enum class AovSource { PathTraced, GBuffer, BeautyFilter };

[[nodiscard]] AovSource aovSource(AovId aov);

// Channels the AOV carries: every producer allocates its lane at this count, so it is also that HdrImage's stride.
[[nodiscard]] int aovChannels(AovId aov);

// True where the AOV's value is proportional to scene radiance, so the display exposure is a gain on it rather than a distortion.
[[nodiscard]] bool aovCarriesRadiance(AovId aov);

// True where zero is the operator's own centre and both signs are meaningful, so the preview maps zero to mid-grey. Preview only.
[[nodiscard]] bool aovIsBipolar(AovId aov);

// Added after the preview's gain so a zero response lands exactly on mid-grey; the gain divides by it, so the two halves cannot drift.
inline constexpr float kBipolarDisplayOffset = 0.5F;

// The preview's affine display map, per lane: display = gain * value + offset. A lane the AOV does not define keeps zero, reading black.
struct BipolarDisplay {
    glm::vec3 gain;
    glm::vec3 offset;
};

// Auto-ranged per channel of `image`, because channels of one AOV can be different quantities in incomparable units.
[[nodiscard]] BipolarDisplay bipolarDisplay(const pathtracer::gfx::HdrImage& image);

// What the display needs that an AOV's own texels do not carry: the pass count anchoring SNR's log window, the path-depth ceiling.
struct AovDisplayContext {
    int samples = 0;
    int maxBounces = 0;
};

// Every display decision that has to read the values: a nonlinear pre-map where one applies, and the affine map for everything else.
struct AovDisplay {
    pathtracer::gfx::HdrImage mapped;  // the pre-mapped image; empty where the source passes through, so the common path copies nothing
    BipolarDisplay affine;
};

// One call per rebuilt pass, never per frame. The photographic exposure stays with the caller: it changes without a re-upload.
[[nodiscard]] AovDisplay aovDisplay(AovId aov, const pathtracer::gfx::HdrImage& image, const AovDisplayContext& context);

// True where the display's gain is the photographic exposure: degree one in radiance, and not already auto-ranged by aovDisplay.
[[nodiscard]] inline bool aovTakesDisplayExposure(AovId aov) {
    return aovCarriesRadiance(aov) && !aovIsBipolar(aov);
}

// True for AOVs needing light transport, false for the 15 primary-hit ones. Derived from aovSource, so the two cannot drift apart.
[[nodiscard]] inline bool aovNeedsLightTransport(AovId aov) {
    return aovSource(aov) != AovSource::GBuffer;
}

// Exact lookup against kAovNames, so "bounceCount" resolves and "Bounce Count" does not. AovId::Count doubles as "unknown".
[[nodiscard]] AovId aovIdFromName(std::string_view name);

}  // namespace pathtracer::debug
