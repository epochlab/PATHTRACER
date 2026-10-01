#pragma once

#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/scene/cie.h"

// ColorChecker Classic spectral reflectance to linear Rec.709 albedo by CIE 1931 2-degree tristimulus integration under D65.
namespace tools::colorchecker {

// The Classic chart's 6x4 layout; patches are stored row-major from dark skin (top-left) to black (bottom-right).
inline constexpr int kColumns = 6;
inline constexpr int kRows = 4;
inline constexpr int kPatchCount = kColumns * kRows;

struct Patch {
    std::string name;
    glm::dvec3 rec709;  // unclamped: cyan lies outside the Rec.709 gamut, so its red is negative
};

// Sprague (1880) quintic through uniformly spaced samples (CIE 167:2005), on the CIE 1 nm grid; nearest value beyond the data's ends.
[[nodiscard]] pathtracer::scene::cie::Spectrum resampleSprague(double firstNm, double intervalNm,
                                                               const std::vector<double>& samples);

// Header "patch,<nm>,...", uniform >= 6 wavelengths; then kPatchCount rows "name,<reflectance>..." in [0,1]; '#' lines are comments.
[[nodiscard]] std::optional<std::vector<Patch>> loadColorChecker(const std::string& path);

}  // namespace tools::colorchecker
