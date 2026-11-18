#include "pathtracer/config/spectral_data.h"

#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string_view>

#include <glm/glm.hpp>

namespace pathtracer::config {

namespace {

namespace cie = pathtracer::scene::cie;

// A whole field, whitespace around it allowed, as a finite number; nullopt for anything else.
std::optional<double> number(std::string_view field) {
    const auto first = field.find_first_not_of(" \t");
    const auto last = field.find_last_not_of(" \t");
    if (first == std::string_view::npos) {
        return std::nullopt;
    }
    field = field.substr(first, last - first + 1);
    double value = 0.0;
    const auto [end, error] = std::from_chars(field.data(), field.data() + field.size(), value);
    if (error != std::errc{} || end != field.data() + field.size() || !std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

// A line's comma-separated numbers; nullopt if any field is not one.
std::optional<std::vector<double>> numbers(std::string_view line) {
    std::vector<double> values;
    for (std::size_t start = 0;;) {
        const std::size_t comma = line.find(',', start);
        const std::optional<double> value = number(line.substr(start, comma - start));
        if (!value) {
            return std::nullopt;
        }
        values.push_back(*value);
        if (comma == std::string_view::npos) {
            return values;
        }
        start = comma + 1;
    }
}

// row(lineNumber, line) per data line, CRLF trimmed, blank and '#' lines skipped; false if the file will not open or a row fails.
template <typename Row>
bool forEachRow(const std::string& file, const std::string& caller, Row row) {
    std::ifstream stream(file);
    if (!stream) {
        std::cerr << caller << ": " << file << " cannot be opened\n";
        return false;
    }
    std::string line;
    for (int lineNumber = 1; std::getline(stream, line); ++lineNumber) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty() && line.front() != '#' && !row(lineNumber, std::string_view(line))) {
            return false;
        }
    }
    return true;
}

// Logs file:line's fault and yields false, for a row callback to return.
template <typename... Why>
bool rejectRow(const std::string& caller, const std::string& file, int lineNumber, const Why&... why) {
    ((std::cerr << caller << ": " << file << ':' << lineNumber << ": ") << ... << why) << '\n';
    return false;
}

}  // namespace

std::optional<cie::Spectrum> loadSpectrumCsv(const std::string& file, double maxValue, Beyond beyond, const std::string& caller) {
    std::vector<glm::dvec2> samples;
    const bool read = forEachRow(file, caller, [&](int lineNumber, std::string_view line) {
        const std::optional<std::vector<double>> row = numbers(line);
        if (!row || row->size() != 2 || !((*row)[0] > 0.0)) {
            return rejectRow(caller, file, lineNumber, "is not a positive nm and a finite value, comma-separated");
        }
        const glm::dvec2 sample((*row)[0], (*row)[1]);
        if (!samples.empty() && !(sample.x > samples.back().x)) {
            return rejectRow(caller, file, lineNumber, "does not rise in nm");
        }
        if (!(sample.y >= 0.0 && sample.y <= maxValue)) {
            return rejectRow(caller, file, lineNumber, "value ", sample.y, " lies outside [0, ", maxValue, "]");
        }
        samples.push_back(sample);
        return true;
    });
    if (!read) {
        return std::nullopt;
    }
    if (samples.size() < 2) {
        std::cerr << caller << ": " << file << " needs two or more rows\n";
        return std::nullopt;
    }
    cie::Spectrum spectrum = cie::resample(samples);
    if (beyond == Beyond::Nearest) {
        for (int i = 0; i < cie::kSampleCount; ++i) {
            const double nm = cie::wavelengthNm(i);
            if (nm < samples.front().x || nm > samples.back().x) {
                spectrum[static_cast<std::size_t>(i)] = nm < samples.front().x ? samples.front().y : samples.back().y;
            }
        }
    }
    return spectrum;
}

std::optional<ComplexIndexSpectrum> loadNkCsv(const std::string& file, const std::string& caller) {
    std::vector<glm::dvec3> rows;  // (nm, n, k)
    const bool read = forEachRow(file, caller, [&](int lineNumber, std::string_view line) {
        const std::optional<std::vector<double>> row = numbers(line);
        if (!row || row->size() != 3) {
            return rejectRow(caller, file, lineNumber, "is not lambda_um,n,k");
        }
        const glm::dvec3 sample((*row)[0] * 1000.0, (*row)[1], (*row)[2]);
        if (!(sample.y > 0.0 && sample.z >= 0.0)) {
            return rejectRow(caller, file, lineNumber, "needs n > 0 and k >= 0");
        }
        if (!rows.empty() && !(sample.x > rows.back().x)) {
            return rejectRow(caller, file, lineNumber, "does not rise in wavelength");
        }
        rows.push_back(sample);
        return true;
    });
    if (!read) {
        return std::nullopt;
    }
    if (rows.size() < 2 || rows.front().x > cie::kLambdaMinNm || rows.back().x < cie::kLambdaMaxNm) {
        std::cerr << caller << ": " << file << " must span " << cie::kLambdaMinNm << "-" << cie::kLambdaMaxNm << " nm\n";
        return std::nullopt;
    }
    // n and k interpolated, never the reflectance, which is nonlinear in them.
    ComplexIndexSpectrum index{};
    std::size_t at = 0;
    for (int i = 0; i < cie::kSampleCount; ++i) {
        const double nm = cie::wavelengthNm(i);
        while (rows[at + 1].x < nm) {
            ++at;
        }
        const double t = (nm - rows[at].x) / (rows[at + 1].x - rows[at].x);
        const auto s = static_cast<std::size_t>(i);
        index.n[s] = std::lerp(rows[at].y, rows[at + 1].y, t);
        index.k[s] = std::lerp(rows[at].z, rows[at + 1].z, t);
    }
    return index;
}

std::optional<std::vector<ChartPatch>> loadChartCsv(const std::string& file, const std::string& caller) {
    std::vector<double> wavelengths;
    std::vector<ChartPatch> patches;
    const bool read = forEachRow(file, caller, [&](int lineNumber, std::string_view line) {
        const std::size_t comma = line.find(',');
        const std::optional<std::vector<double>> values =
            comma == std::string_view::npos ? std::nullopt : numbers(line.substr(comma + 1));
        if (!values) {
            return rejectRow(caller, file, lineNumber, "is not a name and comma-separated finite numbers");
        }
        if (wavelengths.empty()) {
            if (line.substr(0, comma) != "patch" || values->size() < cie::kSpragueStencil) {
                return rejectRow(caller, file, lineNumber, "is not the header patch,<nm>... with ", cie::kSpragueStencil, " or more wavelengths");
            }
            // Exact equality: integer-nm headers parse exactly, and Sprague's coefficients assume a uniform grid.
            const double interval = (*values)[1] - (*values)[0];
            for (std::size_t w = 1; w < values->size(); ++w) {
                if (!(interval > 0.0) || (*values)[w] - (*values)[w - 1] != interval) {
                    return rejectRow(caller, file, lineNumber, "wavelengths do not rise uniformly");
                }
            }
            wavelengths = *values;
            return true;
        }
        if (values->size() != wavelengths.size()) {
            return rejectRow(caller, file, lineNumber, values->size(), " reflectances for ", wavelengths.size(), " wavelengths");
        }
        for (const double reflectance : *values) {
            // A non-fluorescent reflector returns at most what it receives, and never less than nothing.
            if (!(reflectance >= 0.0 && reflectance <= 1.0)) {
                return rejectRow(caller, file, lineNumber, "reflectance ", reflectance, " lies outside [0, 1]");
            }
        }
        patches.push_back({std::string(line.substr(0, comma)),
                           cie::resampleSprague(wavelengths.front(), wavelengths[1] - wavelengths[0], *values)});
        return true;
    });
    if (!read) {
        return std::nullopt;
    }
    if (patches.empty()) {
        std::cerr << caller << ": " << file << " has no patches\n";
        return std::nullopt;
    }
    return patches;
}

}  // namespace pathtracer::config
