#include "colorchecker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

namespace tools::colorchecker {

namespace {

namespace cie = pathtracer::scene::cie;

// The quintic's six-point stencil spans two samples before and three after each interval, so fewer samples cannot seed it.
constexpr std::size_t kStencil = 6;

// CIE 167:2005 eqs 6-7 coefficients (over 209), synthesising the two points beyond each end that the stencil reads.
constexpr std::array<std::array<double, kStencil>, 4> kBoundary{{
    {884.0, -1960.0, 3033.0, -2648.0, 1080.0, -180.0},
    {508.0, -540.0, 488.0, -367.0, 144.0, -24.0},
    {-24.0, 144.0, -367.0, 488.0, -540.0, 508.0},
    {-180.0, 1080.0, -2648.0, 3033.0, -1960.0, 884.0},
}};

double boundaryPoint(const std::array<double, kStencil>& coefficients, const double* six) {
    double sum = 0.0;
    for (std::size_t i = 0; i < kStencil; ++i) {
        sum += coefficients[i] * six[i];
    }
    return sum / 209.0;
}

std::vector<std::string> splitFields(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

// Whole-field parse: a trailing character or a non-finite value is malformed input, not a number to truncate.
std::optional<double> parseNumber(const std::string& field) {
    char* end = nullptr;
    const double value = std::strtod(field.c_str(), &end);
    if (field.empty() || *end != '\0' || !std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

}  // namespace

cie::Spectrum resampleSprague(double firstNm, double intervalNm, const std::vector<double>& samples) {
    const std::size_t n = samples.size();
    // Padded so stencil index k reads samples k-2..k+3 for the interval [k, k+1].
    std::vector<double> padded(n + 4);
    padded[0] = boundaryPoint(kBoundary[0], samples.data());
    padded[1] = boundaryPoint(kBoundary[1], samples.data());
    std::copy(samples.begin(), samples.end(), padded.begin() + 2);
    padded[n + 2] = boundaryPoint(kBoundary[2], samples.data() + (n - kStencil));
    padded[n + 3] = boundaryPoint(kBoundary[3], samples.data() + (n - kStencil));

    const double lastNm = firstNm + (intervalNm * static_cast<double>(n - 1));
    cie::Spectrum out{};
    for (int i = 0; i < cie::kSampleCount; ++i) {
        const double nm = cie::wavelengthNm(i);
        // Nearest-value extension (CIE 015:2018, ASTM E308): the observer's 360-830 nm reaches past what a spectrophotometer measures.
        if (nm <= firstNm) {
            out[i] = samples.front();
            continue;
        }
        if (nm >= lastNm) {
            out[i] = samples.back();
            continue;
        }
        const double t = (nm - firstNm) / intervalNm;
        // Clamped: on a non-integer grid t can round up to n - 1 just below lastNm, whose stencil would read past the padding.
        const std::size_t k = std::min(static_cast<std::size_t>(t), n - 2);
        const double x = t - static_cast<double>(k);
        const double* r = &padded[k];
        // CIE 167:2005 eq 5: the quintic's coefficients over the stencil r[0..5] = y(k-2)..y(k+3), each over 24.
        const double a1 = ((2.0 * r[0]) - (16.0 * r[1]) + (16.0 * r[3]) - (2.0 * r[4])) / 24.0;
        const double a2 = (-r[0] + (16.0 * r[1]) - (30.0 * r[2]) + (16.0 * r[3]) - r[4]) / 24.0;
        const double a3 =
            ((-9.0 * r[0]) + (39.0 * r[1]) - (70.0 * r[2]) + (66.0 * r[3]) - (33.0 * r[4]) + (7.0 * r[5])) / 24.0;
        const double a4 =
            ((13.0 * r[0]) - (64.0 * r[1]) + (126.0 * r[2]) - (124.0 * r[3]) + (61.0 * r[4]) - (12.0 * r[5])) / 24.0;
        const double a5 =
            ((-5.0 * r[0]) + (25.0 * r[1]) - (50.0 * r[2]) + (50.0 * r[3]) - (25.0 * r[4]) + (5.0 * r[5])) / 24.0;
        out[i] = r[2] + (x * (a1 + (x * (a2 + (x * (a3 + (x * (a4 + (x * a5)))))))));
    }
    return out;
}

std::optional<std::vector<Patch>> loadColorChecker(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "loadColorChecker: cannot read " << path << '\n';
        return std::nullopt;
    }
    std::vector<double> wavelengths;
    std::vector<Patch> patches;
    std::string line;
    for (int lineNumber = 1; std::getline(in, line); ++lineNumber) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();  // CRLF tables from Windows-authored sources
        }
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const std::vector<std::string> fields = splitFields(line);
        std::vector<double> values;
        for (std::size_t f = 1; f < fields.size(); ++f) {
            const std::optional<double> value = parseNumber(fields[f]);
            if (!value) {
                std::cerr << "loadColorChecker: " << path << ':' << lineNumber << ": field " << f + 1
                          << " is not a finite number\n";
                return std::nullopt;
            }
            values.push_back(*value);
        }
        if (wavelengths.empty()) {
            if (fields.front() != "patch" || values.size() < kStencil) {
                std::cerr << "loadColorChecker: " << path << ':' << lineNumber << ": expected header patch,<nm>... with >= "
                          << kStencil << " wavelengths\n";
                return std::nullopt;
            }
            // Exact equality: integer-nm headers parse exactly, and Sprague's coefficients assume a uniform grid.
            for (std::size_t w = 1; w < values.size(); ++w) {
                if (!(values[w] - values[w - 1] == values[1] - values[0]) || !(values[1] > values[0])) {
                    std::cerr << "loadColorChecker: " << path << ':' << lineNumber
                              << ": wavelengths must increase uniformly\n";
                    return std::nullopt;
                }
            }
            wavelengths = std::move(values);
            continue;
        }
        if (values.size() != wavelengths.size()) {
            std::cerr << "loadColorChecker: " << path << ':' << lineNumber << ": " << values.size() << " reflectances for "
                      << wavelengths.size() << " wavelengths\n";
            return std::nullopt;
        }
        for (double reflectance : values) {
            // A non-fluorescent reflector returns at most what it receives, and never less than nothing.
            if (!(reflectance >= 0.0 && reflectance <= 1.0)) {
                std::cerr << "loadColorChecker: " << path << ':' << lineNumber << ": reflectance " << reflectance
                          << " is outside [0,1]\n";
                return std::nullopt;
            }
        }
        const cie::Spectrum spectrum = resampleSprague(wavelengths.front(), wavelengths[1] - wavelengths[0], values);
        patches.push_back(Patch{fields.front(), cie::reflectanceToRec709(spectrum)});
    }
    if (patches.size() != static_cast<std::size_t>(kPatchCount)) {
        std::cerr << "loadColorChecker: " << path << ": " << patches.size() << " patches, expected " << kPatchCount << '\n';
        return std::nullopt;
    }
    return patches;
}

}  // namespace tools::colorchecker
