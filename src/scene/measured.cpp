#include "pathtracer/scene/measured.h"

#include <algorithm>
#include <cmath>
#include <complex>

namespace pathtracer::scene {

namespace {

// F82's second point, mu-bar = 1/7 (~82 deg), where OpenPBR's specular_color is defined, and Schlick's (1 - mu-bar)^5 there.
constexpr double kMuBar = 1.0 / 7.0;
const double kSchlickWeightAtMuBar = std::pow(1.0 - kMuBar, 5.0);

}  // namespace

std::size_t MeasuredColour::cellAt(glm::vec2 uv) const {
    // Repeat-wrapped as textures are, so uv 1 is the first cell again, not one past the last.
    const glm::vec2 unit = uv - glm::floor(uv);
    const int column = std::min(static_cast<int>(unit.x * static_cast<float>(columns)), columns - 1);
    const int row = std::min(static_cast<int>(unit.y * static_cast<float>(rows)), rows - 1);
    return (static_cast<std::size_t>(row) * static_cast<std::size_t>(columns)) + static_cast<std::size_t>(column);
}

void MeasuredColour::append(const cie::Spectrum& spectrum) {
    spectra.push_back(tabulate(spectrum));
    // Into OpenPBR's [0, 1] as a texel is: a saturated spectrum's scene RGB can leave the gamut.
    rgb.push_back(clamp(Rgb(glm::vec3(cie::reflectanceToScene(spectrum))), 0.0F, 1.0F));
}

MeasuredColour measuredColour(const cie::Spectrum& spectrum) {
    MeasuredColour colour;
    colour.append(spectrum);
    return colour;
}

bool measures(const MeasuredMaterial& measured, std::string_view input) {
    return (input == "specular_ior" && measured.specularIor) ||
           std::ranges::any_of(kMeasuredColourInputs, [&](const MeasuredColourInput& colour) {
               return colour.name == input && (measured.*colour.data).has_value();
           });
}

MeasuredInputs uniformView(const MeasuredMaterial& measured) {
    const auto single = [](const std::optional<MeasuredColour>& colour) -> const TabulatedSpectrum* {
        return colour && colour->spectra.size() == 1 ? colour->spectra.data() : nullptr;
    };
    const auto table = [](const std::optional<TabulatedSpectrum>& values) { return values ? &*values : nullptr; };
    return {.baseColor = single(measured.baseColor),
            .specularColor = single(measured.specularColor),
            .transmissionColor = single(measured.transmissionColor),
            .specularIor = table(measured.specularIor),
            .conductorEta = table(measured.conductorEta),
            .conductorKappa = table(measured.conductorKappa)};
}

void resolveGrids(const MeasuredMaterial& measured, glm::vec2 uv, MeasuredInputs& view, Rgb& baseColor, Rgb& specularColor,
                  Rgb& transmissionColor) {
    const auto resolve = [uv](const std::optional<MeasuredColour>& colour, const TabulatedSpectrum*& spectrum, Rgb& rgb) {
        if (colour && colour->spectra.size() > 1) {
            const std::size_t cell = colour->cellAt(uv);
            spectrum = &colour->spectra[cell];
            rgb = colour->rgb[cell];
        }
    };
    resolve(measured.baseColor, view.baseColor, baseColor);
    resolve(measured.specularColor, view.specularColor, specularColor);
    resolve(measured.transmissionColor, view.transmissionColor, transmissionColor);
}

double conductorFresnel(double n, double k, double cosTheta) {
    const std::complex<double> eta(n, k);
    const double c = std::clamp(cosTheta, 0.0, 1.0);
    const std::complex<double> cosThetaT = std::sqrt(1.0 - ((1.0 - (c * c)) / (eta * eta)));
    const std::complex<double> rParallel = ((eta * c) - cosThetaT) / ((eta * c) + cosThetaT);
    const std::complex<double> rPerpendicular = (c - (eta * cosThetaT)) / (c + (eta * cosThetaT));
    return 0.5 * (std::norm(rParallel) + std::norm(rPerpendicular));
}

ConductorColours conductorColours(const cie::Spectrum& n, const cie::Spectrum& k) {
    ConductorColours colours{};
    for (std::size_t i = 0; i < n.size(); ++i) {
        const double f0 = conductorFresnel(n[i], k[i], 1.0);
        colours.f0[i] = f0;
        colours.tint[i] = std::min(conductorFresnel(n[i], k[i], kMuBar) / (f0 + ((1.0 - f0) * kSchlickWeightAtMuBar)), 1.0);
    }
    return colours;
}

cie::Spectrum sellmeierIndex(const std::array<double, 3>& b, const std::array<double, 3>& c) {
    cie::Spectrum index{};
    for (int i = 0; i < cie::kSampleCount; ++i) {
        const double um = cie::wavelengthNm(i) / 1000.0;
        double sum = 1.0;
        for (std::size_t term = 0; term < b.size(); ++term) {
            sum += b[term] * um * um / ((um * um) - c[term]);
        }
        index[static_cast<std::size_t>(i)] = std::sqrt(sum);
    }
    return index;
}

}  // namespace pathtracer::scene
