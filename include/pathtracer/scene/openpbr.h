#pragma once

#include <algorithm>
#include <limits>
#include <string_view>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"

namespace pathtracer::scene {

// An input's hard range from the OpenPBR Surface specification v1.1.1: Unit [0,1], NonNegative [0,inf), Positive (0,inf), Signed [-1,1].
enum class InputRange { Unit, NonNegative, Positive, Signed };

// What reads an input: the surface BSDF at every vertex, emission at a hit, or in-volume scattering, inert without volumetric transport.
enum class InputUse { Surface, Emission, Volume };

// One OpenPBR input: its specification name, range, image role and reader; filtered means a primary hit prefilters it by its footprint.
struct InputSpec {
    std::string_view name;
    InputRange range;
    pathtracer::gfx::ImageRole role;
    // A mix weight or reflectance factor; an input shaping a lobe (roughness, ior, a power) is point-sampled, as the pixel integrates it.
    bool filtered;
    InputUse use = InputUse::Surface;
    // A switch for in-volume scattering, which needs volumetric transport: only its zero default is accepted.
    bool requiresVolumes = false;
};

// The identity wrapper: OpenPbrInputs<Constant> holds plain values, as a material file and a resolved shading point do.
template <typename T>
using Constant = T;

// OpenPBR Surface v1.1.1 inputs at their specification defaults; In<T> is T for constants or a constant-or-texture for a bound material.
template <template <typename> typename In>
struct OpenPbrInputs {
    In<float> baseWeight = 1.0F;
    In<glm::vec3> baseColor = glm::vec3(0.8F);
    In<float> baseMetalness = 0.0F;
    In<float> baseDiffuseRoughness = 0.0F;
    In<float> specularWeight = 1.0F;
    In<glm::vec3> specularColor = glm::vec3(1.0F);
    In<float> specularRoughness = 0.3F;
    In<float> specularIor = 1.5F;
    In<float> transmissionWeight = 0.0F;
    In<glm::vec3> transmissionColor = glm::vec3(1.0F);
    In<float> transmissionDepth = 0.0F;
    In<glm::vec3> transmissionScatter = glm::vec3(0.0F);
    In<float> transmissionScatterAnisotropy = 0.0F;
    In<float> transmissionDispersionScale = 0.0F;
    In<float> transmissionDispersionAbbeNumber = 20.0F;
    In<float> subsurfaceWeight = 0.0F;
    In<glm::vec3> subsurfaceColor = glm::vec3(0.8F);
    In<float> subsurfaceRadius = 1.0F;
    In<glm::vec3> subsurfaceRadiusScale = glm::vec3(1.0F, 0.5F, 0.25F);
    In<float> subsurfaceScatterAnisotropy = 0.0F;
    In<float> coatWeight = 0.0F;
    In<glm::vec3> coatColor = glm::vec3(1.0F);
    In<float> coatRoughness = 0.0F;
    In<float> coatIor = 1.6F;
    In<float> coatDarkening = 1.0F;
    In<float> fuzzWeight = 0.0F;
    In<glm::vec3> fuzzColor = glm::vec3(1.0F);
    In<float> fuzzRoughness = 0.5F;
    In<float> emissionLuminance = 0.0F;
    In<glm::vec3> emissionColor = glm::vec3(1.0F);
    In<float> thinFilmWeight = 0.0F;
    In<float> thinFilmThickness = 0.5F;
    In<float> thinFilmIor = 1.4F;
};

// Visits every input of one or more OpenPbrInputs in specification order, zipped: visit(spec, a.member, b.member, ...).
template <typename Visit, typename... Inputs>
void forEachInput(Visit&& visit, Inputs&... inputs) {
    using enum InputRange;
    constexpr auto kColour = pathtracer::gfx::ImageRole::Colour;
    constexpr auto kData = pathtracer::gfx::ImageRole::Data;
    visit(InputSpec{"base_weight", Unit, kData, true}, inputs.baseWeight...);
    visit(InputSpec{"base_color", Unit, kColour, true}, inputs.baseColor...);
    visit(InputSpec{"base_metalness", Unit, kData, true}, inputs.baseMetalness...);
    visit(InputSpec{"base_diffuse_roughness", Unit, kData, false}, inputs.baseDiffuseRoughness...);
    visit(InputSpec{"specular_weight", NonNegative, kData, false}, inputs.specularWeight...);
    visit(InputSpec{"specular_color", Unit, kColour, true}, inputs.specularColor...);
    visit(InputSpec{"specular_roughness", Unit, kData, false}, inputs.specularRoughness...);
    visit(InputSpec{"specular_ior", Positive, kData, false}, inputs.specularIor...);
    visit(InputSpec{"transmission_weight", Unit, kData, true}, inputs.transmissionWeight...);
    visit(InputSpec{"transmission_color", Unit, kColour, false}, inputs.transmissionColor...);
    visit(InputSpec{"transmission_depth", NonNegative, kData, false}, inputs.transmissionDepth...);
    visit(InputSpec{"transmission_scatter", NonNegative, kColour, false, InputUse::Volume, true}, inputs.transmissionScatter...);
    visit(InputSpec{"transmission_scatter_anisotropy", Signed, kData, false, InputUse::Volume}, inputs.transmissionScatterAnisotropy...);
    visit(InputSpec{"transmission_dispersion_scale", Unit, kData, false}, inputs.transmissionDispersionScale...);
    visit(InputSpec{"transmission_dispersion_abbe_number", Positive, kData, false}, inputs.transmissionDispersionAbbeNumber...);
    visit(InputSpec{"subsurface_weight", Unit, kData, false, InputUse::Volume, true}, inputs.subsurfaceWeight...);
    visit(InputSpec{"subsurface_color", Unit, kColour, false, InputUse::Volume}, inputs.subsurfaceColor...);
    visit(InputSpec{"subsurface_radius", NonNegative, kData, false, InputUse::Volume}, inputs.subsurfaceRadius...);
    visit(InputSpec{"subsurface_radius_scale", Unit, kData, false, InputUse::Volume}, inputs.subsurfaceRadiusScale...);
    visit(InputSpec{"subsurface_scatter_anisotropy", Signed, kData, false, InputUse::Volume}, inputs.subsurfaceScatterAnisotropy...);
    visit(InputSpec{"coat_weight", Unit, kData, false}, inputs.coatWeight...);
    visit(InputSpec{"coat_color", Unit, kColour, false}, inputs.coatColor...);
    visit(InputSpec{"coat_roughness", Unit, kData, false}, inputs.coatRoughness...);
    visit(InputSpec{"coat_ior", Positive, kData, false}, inputs.coatIor...);
    visit(InputSpec{"coat_darkening", Unit, kData, false}, inputs.coatDarkening...);
    visit(InputSpec{"fuzz_weight", Unit, kData, true}, inputs.fuzzWeight...);
    visit(InputSpec{"fuzz_color", Unit, kColour, true}, inputs.fuzzColor...);
    visit(InputSpec{"fuzz_roughness", Unit, kData, false}, inputs.fuzzRoughness...);
    visit(InputSpec{"emission_luminance", NonNegative, kData, true, InputUse::Emission}, inputs.emissionLuminance...);
    visit(InputSpec{"emission_color", NonNegative, kColour, true, InputUse::Emission}, inputs.emissionColor...);
    visit(InputSpec{"thin_film_weight", Unit, kData, true}, inputs.thinFilmWeight...);
    visit(InputSpec{"thin_film_thickness", NonNegative, kData, false}, inputs.thinFilmThickness...);
    visit(InputSpec{"thin_film_ior", Positive, kData, false}, inputs.thinFilmIor...);
}

// Membership in the range by direct comparisons, which every NaN fails; infinity fails too, past float max.
[[nodiscard]] constexpr bool inRange(float value, InputRange range) {
    switch (range) {
        case InputRange::Unit:
            return value >= 0.0F && value <= 1.0F;
        case InputRange::NonNegative:
            return value >= 0.0F && value <= std::numeric_limits<float>::max();
        case InputRange::Positive:
            return value > 0.0F && value <= std::numeric_limits<float>::max();
        case InputRange::Signed:
            return value >= -1.0F && value <= 1.0F;
    }
    return false;
}

// The nearest value in the range: a texture's texel is data, possibly a quantisation or gamut excursion past the specification.
[[nodiscard]] constexpr float clampToRange(float value, InputRange range) {
    switch (range) {
        case InputRange::Unit:
            return std::clamp(value, 0.0F, 1.0F);
        case InputRange::NonNegative:
            return std::max(value, 0.0F);
        case InputRange::Positive:
            return std::max(value, std::numeric_limits<float>::min());
        case InputRange::Signed:
            return std::clamp(value, -1.0F, 1.0F);
    }
    return value;
}

}  // namespace pathtracer::scene
