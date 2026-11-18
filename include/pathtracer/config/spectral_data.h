#pragma once

#include <optional>
#include <string>
#include <vector>

#include "pathtracer/scene/cie.h"

// Measured spectral data files resampled onto the CIE 1 nm grid; each loader logs its first fault, prefixed by caller, and yields nullopt.
namespace pathtracer::config {

// Past a table's ends: zero, as the CIE emission datasets read; or the nearest value, as CIE 015:2018 extends a reflectance.
enum class Beyond { Zero, Nearest };

// Rows "nm,value", nm rising, values in [0, maxValue], '#' comments; linear between rows, beyond them as beyond says.
[[nodiscard]] std::optional<pathtracer::scene::cie::Spectrum> loadSpectrumCsv(const std::string& file, double maxValue, Beyond beyond,
                                                                              const std::string& caller);

// A conductor's complex index n + ik on the grid.
struct ComplexIndexSpectrum {
    pathtracer::scene::cie::Spectrum n;
    pathtracer::scene::cie::Spectrum k;
};

// Rows "lambda_um,n,k" (refractiveindex.info's), '#' comments, n > 0, k >= 0, spanning the grid: extrapolation is not data; linear in nm.
[[nodiscard]] std::optional<ComplexIndexSpectrum> loadNkCsv(const std::string& file, const std::string& caller);

// One named reflectance of a chart.
struct ChartPatch {
    std::string name;
    pathtracer::scene::cie::Spectrum reflectance;
};

// Header "patch,<nm>,..." on a uniform grid of 6 or more; rows "name,<reflectance>..." in [0, 1], '#' comments; Sprague-resampled.
[[nodiscard]] std::optional<std::vector<ChartPatch>> loadChartCsv(const std::string& file, const std::string& caller);

}  // namespace pathtracer::config
