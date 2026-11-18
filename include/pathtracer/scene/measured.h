#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/openpbr.h"
#include "pathtracer/scene/rgb.h"
#include "pathtracer/scene/spectrum.h"

// Measured spectral data a material binds in place of its RGB colours and Abbe dispersion.
namespace pathtracer::scene {

// One measured spectrum, or a columns x rows grid of them over uv's unit square (a chart), row 0 at v = 0.
struct MeasuredColour {
    std::vector<TabulatedSpectrum> spectra;  // row-major
    std::vector<Rgb> rgb;                    // each spectrum's scene RGB under D65 in [0, 1], what RGB readers (AOVs, features) take
    int columns = 1;
    int rows = 1;
    [[nodiscard]] std::size_t cellAt(glm::vec2 uv) const;
    // Adds the next cell's spectrum, a reflectance or transmittance in [0, 1], with its scene RGB.
    void append(const cie::Spectrum& spectrum);
};

// One spectrum over the whole surface.
[[nodiscard]] MeasuredColour measuredColour(const cie::Spectrum& spectrum);

// The measured data one material file binds, shared by every instance and shading point it serves.
struct MeasuredMaterial {
    std::optional<MeasuredColour> baseColor;
    std::optional<MeasuredColour> specularColor;
    std::optional<MeasuredColour> transmissionColor;
    std::optional<TabulatedSpectrum> specularIor;
    std::optional<TabulatedSpectrum> conductorEta;
    std::optional<TabulatedSpectrum> conductorKappa;
};

// A colour input a measured spectrum may stand in for: its name, where its data lands, and the constant its scene RGB fills.
struct MeasuredColourInput {
    std::string_view name;
    std::optional<MeasuredColour> MeasuredMaterial::*data;
    Rgb OpenPbrInputs<Constant>::*constant;
};
inline constexpr std::array<MeasuredColourInput, 3> kMeasuredColourInputs{{
    {"base_color", &MeasuredMaterial::baseColor, &OpenPbrInputs<Constant>::baseColor},
    {"specular_color", &MeasuredMaterial::specularColor, &OpenPbrInputs<Constant>::specularColor},
    {"transmission_color", &MeasuredMaterial::transmissionColor, &OpenPbrInputs<Constant>::transmissionColor},
}};

// Whether input's value comes from measured data, which a texture cannot replace.
[[nodiscard]] bool measures(const MeasuredMaterial& measured, std::string_view input);

// The tables and single-spectrum colours, which hold at every uv; a grid colour stays null until resolveGrids resolves it.
[[nodiscard]] MeasuredInputs uniformView(const MeasuredMaterial& measured);

// Each grid colour's cell at uv: its spectrum into view, its scene RGB over the matching colour.
void resolveGrids(const MeasuredMaterial& measured, glm::vec2 uv, MeasuredInputs& view, Rgb& baseColor, Rgb& specularColor,
                  Rgb& transmissionColor);

// Unpolarised Fresnel reflectance of a conductor of complex index n + ik relative to the incident side, at cosTheta.
[[nodiscard]] double conductorFresnel(double n, double k, double cosTheta);

// F82-tint colours per wavelength: f0 = F(1), tint = F(1/7) / Schlick(f0, 1/7) capped at 1, so F82 meets F at both.
struct ConductorColours {
    cie::Spectrum f0;
    cie::Spectrum tint;
};
[[nodiscard]] ConductorColours conductorColours(const cie::Spectrum& n, const cie::Spectrum& k);

// Sellmeier n = sqrt(1 + sum B_i l^2 / (l^2 - C_i)), l in um, C_i in um^2 (glass catalogues' form), at the grid's samples.
[[nodiscard]] cie::Spectrum sellmeierIndex(const std::array<double, 3>& b, const std::array<double, 3>& c);

}  // namespace pathtracer::scene
