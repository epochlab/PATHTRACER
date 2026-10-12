#pragma once

#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

// Measured conductor (lambda,n,k) to OpenPBR's F82-tint metal (base_color, specular_color) per scene-space channel, CIE 1931 projected.
namespace tools::metal_fit {

struct NkSample {
    double lambdaNm;
    double n;
    double k;
};

// Rows "lambda_um,n,k", '#' comments; rejects unsorted, unphysical (n<=0,k<0) or short of CIE's 360-830 nm: extrapolation is not data.
[[nodiscard]] std::optional<std::vector<NkSample>> loadNkTable(const std::string& path);

// Johnson & Christy tabulate in uniform photon energy, so both domains are defensible: the fit uses Wavelength, energy as a sensitivity.
enum class Interpolation { Wavelength, PhotonEnergy };

struct Fit {
    glm::dvec3 baseColor;          // F0: CIE projection of R(mu = 1, lambda), which F82 meets exactly
    glm::dvec3 specularColor;      // CIE R(mu-bar) / Schlick(F0, mu-bar), capped at 1: F82 meets the data at mu-bar = 1/7 unless capped
    glm::bvec3 specularCapped;     // channels whose tint exceeded OpenPBR's range: their F82 meets Schlick(F0) at mu-bar, short of the data
    glm::dvec3 atMuBar;            // CIE projection of R(1/7, lambda), the second point the fit passes through
    glm::dvec3 averageTarget;      // CIE projection of the spectral cosine-weighted average 2 int R(mu, lambda) mu dmu
    glm::dvec3 averageResidual;    // |F82 mean - averageTarget|: the energy the two-point fit misplaces
    glm::dvec3 angularResidual;    // max over mu of |R_fit(mu) - CIE projection of R(mu, lambda)|: the model's shape error
};

// nullopt (reason on stderr) when base_color leaves [0, 1] or the tint is below 0; a tint above 1 caps at 1, its misfit in angularResidual.
[[nodiscard]] std::optional<Fit> fitF82(const std::vector<NkSample>& table, Interpolation interpolation);

}  // namespace tools::metal_fit
