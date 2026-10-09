#pragma once

#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

// Measured conductor (lambda,n,k) to OpenPBR's F82-tint metal (base_color, specular_color) per Rec.709 channel, CIE 1931 projected.
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
    glm::dvec3 specularColor;      // CIE R(mu-bar) / Schlick(F0, mu-bar): F82 meets the measurement at mu-bar = 1/7 by its definition
    glm::dvec3 atMuBar;            // CIE projection of R(1/7, lambda), the second point the fit passes through
    glm::dvec3 averageTarget;      // CIE projection of the spectral cosine-weighted average 2 int R(mu, lambda) mu dmu
    glm::dvec3 averageResidual;    // |F82 mean - averageTarget|: the energy the two-point fit misplaces
    glm::dvec3 angularResidual;    // max over mu of |R_fit(mu) - CIE projection of R(mu, lambda)|: the model's shape error
};

// nullopt (with the reason on stderr) when a channel's base or specular colour leaves OpenPBR's [0, 1], which a clamp would misreport.
[[nodiscard]] std::optional<Fit> fitF82(const std::vector<NkSample>& table, Interpolation interpolation);

}  // namespace tools::metal_fit
