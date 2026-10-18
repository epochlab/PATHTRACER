#pragma once

#include <string_view>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/pixel_lattice.h"

namespace pathtracer::debug {

// Single source of truth for every selectable AOV. AppResources.aov stays int because ImGui::Combo needs int&.
enum class AovId : int {
    // Utility.
    Beauty = 0,
    Wireframe,  // combined AOV: white mesh-edge lines + one false-coloured bounding box per instance (gbuffer.h)
    Alpha,
    Depth,
    Lookahead,  // Depth remapped through profile.json's lookaheadDistance: 1 at the camera, 0 at that horizon and beyond
    HSV,
    Luminance,
    Sobel,
    Gabor,
    DoG,
    WorldPos,
    UV,
    MotionVector,  // geometric film-plane motion in column pitches from the request's previous camera (gbuffer.h)
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
    SampleCount,  // passes folded into each texel: below the published count where adaptive sampling stopped it
    // Lighting.
    DirectDiffuse,
    IndirectDiffuse,
    DirectSpecular,
    IndirectSpecular,
    Refraction,
    DirectVolume,
    IndirectVolume,
    DirectSubsurface,
    IndirectSubsurface,
    Shadow,
    // OIDN's estimate of Beauty, guided by albedo and normal lanes averaged over Beauty's own samples (kDenoiseFeatureLanes).
    Denoised,
    Count  // sentinel, == array size, not itself a selectable value
};

// camelCase call name, a standalone acronym lower case, index-parallel to AovId: the one spelling the CLI, bench logs and both APIs share.
inline constexpr const char* kAovNames[] = {
    "beauty", "wireframe", "alpha", "depth", "lookahead", "hsv", "luminance", "sobel",
    "gabor", "dog", "worldPos", "uv", "motionVector", "colourOpponent",
    "normal", "geomNormal", "albedo", "metallic", "roughness", "tangent", "objectID", "ao",
    "fresnel", "ior", "bounceCount", "snr", "sampleCount",
    "directDiffuse", "indirectDiffuse", "directSpecular", "indirectSpecular", "refraction", "directVolume", "indirectVolume",
    "directSubsurface", "indirectSubsurface", "shadow",
    "denoised",
};
static_assert(sizeof(kAovNames) / sizeof(kAovNames[0]) == static_cast<int>(AovId::Count),
              "kAovNames must stay index-parallel with AovId");

// Title-case HUD label, acronyms upper case, index-parallel to AovId: display only, never parsed.
inline constexpr const char* kAovLabels[] = {
    "Beauty", "Wireframe", "Alpha", "Depth", "Lookahead", "HSV", "Luminance", "Sobel",
    "Gabor", "DoG", "World Pos", "UV", "Motion Vector", "Colour Opponent",
    "Normal", "Geom Normal", "Albedo", "Metallic", "Roughness", "Tangent", "Object ID", "AO",
    "Fresnel", "IOR", "Bounce Count", "SNR", "Sample Count",
    "Direct Diffuse", "Indirect Diffuse", "Direct Specular", "Indirect Specular", "Refraction", "Direct Volume", "Indirect Volume",
    "Direct Subsurface", "Indirect Subsurface", "Shadow",
    "Denoised",
};
static_assert(sizeof(kAovLabels) / sizeof(kAovLabels[0]) == static_cast<int>(AovId::Count),
              "kAovLabels must stay index-parallel with AovId");

// Which of the three producers computes each AOV: 14 accumulated path-traced lanes, 14 G-buffer lanes, 10 derived after accumulation.
enum class AovSource { PathTraced, GBuffer, BeautyFilter };

[[nodiscard]] AovSource aovSource(AovId aov);

// Channels the AOV carries: every producer allocates its lane at this count, so it is also that HdrImage's stride.
[[nodiscard]] int aovChannels(AovId aov);

// True where the AOV's value is proportional to scene radiance, so the display exposure is a gain on it rather than a distortion.
[[nodiscard]] bool aovCarriesRadiance(AovId aov);

// True for the rendered picture, Beauty or its denoised estimate: the AOVs the user's LUT, lens effects and alpha apply to.
[[nodiscard]] bool aovIsPicture(AovId aov);

// The AOVs before Denoised: last in AovId, so a disabled denoiser truncates the selectable list rather than holing it.
inline constexpr int kAovCountWithoutDenoiser = static_cast<int>(AovId::Denoised);
static_assert(kAovCountWithoutDenoiser == static_cast<int>(AovId::Count) - 1, "Denoised must stay last in AovId");

// True where the AOV is RGB in the working space's primaries, so a file carries it as Colour and a change of space converts it.
[[nodiscard]] bool aovIsColour(AovId aov);

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

// What the display needs beyond the texels: SNR's pass count, the path-depth ceiling, and Depth's maximum the G-buffer pass reduced.
struct AovDisplayContext {
    int samples = 0;
    int maxBounces = 0;
    float depthMax = 0.0F;
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

// True for AOVs needing light transport, false for the 14 primary-hit ones. Derived from aovSource, so the two cannot drift apart.
[[nodiscard]] inline bool aovNeedsLightTransport(AovId aov) {
    return aovSource(aov) != AovSource::GBuffer;
}

// False where the AOV has no version on the lattice: Denoised on hexagonal, OIDN's network being trained on square images.
[[nodiscard]] bool aovSupportsLattice(AovId aov, pathtracer::gfx::PixelLattice lattice);

// Exact lookup against kAovNames, so "bounceCount" resolves and "Bounce Count" does not. AovId::Count doubles as "unknown".
[[nodiscard]] AovId aovIdFromName(std::string_view name);

}  // namespace pathtracer::debug
