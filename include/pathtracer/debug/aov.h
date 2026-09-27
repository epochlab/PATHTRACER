#pragma once

#include <span>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

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
    LoG,
    WorldPos,
    UV,
    // Perceptual: observer models over Beauty, as against the Utility block's image-space derivative operators.
    ColourOpponent,
    Retinex,
    HistogramAdjustment,
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

// Display name, index-parallel to AovId -- the array the HUD combo box binds to. The static_assert below gates that parallelism.
inline constexpr const char* kAovNames[] = {
    "Beauty",       "Wireframe",      "Alpha",           "Depth",
    "Lookahead",    "HSV",            "Luminance",       "Sobel",
    "Gabor",        "DoG",            "LoG",             "WorldPos",
    "UV",
    "Colour Opponent", "Retinex",   "Histogram Adjustment",
    "Normal",       "GeomNormal",     "Albedo",          "Metallic",
    "Roughness",    "Tangent",        "ObjectID",        "AO",
    "Fresnel",      "IOR",            "Bounce Count",    "SNR",
    "Direct Diffuse", "Indirect Diffuse", "Direct Specular",
    "Indirect Specular", "Refraction", "Shadow",
};
static_assert(sizeof(kAovNames) / sizeof(kAovNames[0]) == static_cast<int>(AovId::Count),
              "kAovNames must stay index-parallel with AovId");

// Which of the three producers computes each AOV: 10 accumulated path-traced lanes, 14 rasterizer lanes, 10 filters over Beauty.
enum class AovSource { PathTraced, GBuffer, BeautyFilter };

[[nodiscard]] AovSource aovSource(AovId aov);

// Channels the AOV means, not how it is stored: HdrImage is always 4 floats/texel, so this is what a packed consumer must allocate.
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

// Auto-ranged per lane over an interleaved RGBA buffer, because lanes of one AOV can be different quantities in incomparable units.
[[nodiscard]] BipolarDisplay bipolarDisplay(std::span<const float> rgba, int channels);

// What the display needs that an AOV's own texels do not carry: the pass count anchoring SNR's log window, the path-depth ceiling.
struct AovDisplayContext {
    int samples = 0;
    int maxBounces = 0;
};

// Every display decision that has to read the values: a nonlinear pre-map where one applies, and the affine map for everything else.
struct AovDisplay {
    std::vector<float> rgba;  // the pre-mapped texels; empty where the source passes through, so the common path copies nothing
    BipolarDisplay affine;
};

// One call per rebuilt pass, never per frame. The photographic exposure stays with the caller: it changes without a re-upload.
[[nodiscard]] AovDisplay aovDisplay(AovId aov, std::span<const float> rgba, const AovDisplayContext& context);

// True where the display's gain is the photographic exposure: degree one in radiance, and not already auto-ranged by aovDisplay.
[[nodiscard]] inline bool aovTakesDisplayExposure(AovId aov) {
    return aovCarriesRadiance(aov) && !aovIsBipolar(aov);
}

// True for AOVs needing light transport, false for the 14 primary-hit ones. Derived from aovSource, so the two cannot drift apart.
[[nodiscard]] inline bool aovNeedsLightTransport(AovId aov) {
    return aovSource(aov) != AovSource::GBuffer;
}

// Case- and separator-insensitive lookup against kAovNames, so "bounce-count" and "bouncecount" match. AovId::Count doubles as "unknown".
[[nodiscard]] AovId aovIdFromName(std::string_view name);

}  // namespace pathtracer::debug
