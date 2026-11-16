// Build-time generator for the RGB-to-spectrum tables (Jakob & Hanika 2019), fitted against the renderer's own CIE colorimetry.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "parallel.h"
#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/rgb_spectrum_table.h"

namespace {

namespace cie = pathtracer::scene::cie;
namespace table = pathtracer::scene::rgb_spectrum;

// The fit's colorimetry under one white: scene RGB per unit spectrum at each 1 nm sample, and the CIELAB reference it is judged in.
struct Colorimetry {
    std::array<glm::dvec3, cie::kSampleCount> rgbPerSample;  // xyzToSceneFrom(white) * cmf * W / sum(W yBar): a flat 1 reads (1, 1, 1)
    glm::dmat3 sceneToXyz;
    glm::dvec3 whiteXyz;  // the perfect reflector under W at Y = 1, CIELAB's reference white
};

// Under D65, the scene white RGB (1, 1, 1) radiates: a reflectance renders as authored under it, and flat is equal RGB.
Colorimetry colorimetry() {
    std::array<double, cie::kSampleCount> power{};
    double whiteY = 0.0;
    glm::dvec3 white(0.0);
    for (int i = 0; i < cie::kSampleCount; ++i) {
        const cie::TableRow row = cie::tableRow(i);
        power[static_cast<std::size_t>(i)] = row.d65;
        white += power[static_cast<std::size_t>(i)] * row.colourMatching;
        whiteY += power[static_cast<std::size_t>(i)] * row.colourMatching.y;
    }
    white /= whiteY;
    const glm::dmat3 xyzToScene = cie::xyzToSceneFrom(white);
    Colorimetry c{{}, glm::inverse(xyzToScene), white};
    for (int i = 0; i < cie::kSampleCount; ++i) {
        const auto at = static_cast<std::size_t>(i);
        c.rgbPerSample[at] = xyzToScene * (cie::tableRow(i).colourMatching * power[at] / whiteY);
    }
    return c;
}

// CIE 015:2018's CIELAB f and its derivative: the cube root above (6/29)^3, the linear segment below.
constexpr double kLabDelta = 6.0 / 29.0;
double labF(double t) { return t > kLabDelta * kLabDelta * kLabDelta ? std::cbrt(t) : (t / (3.0 * kLabDelta * kLabDelta)) + (4.0 / 29.0); }
double labDf(double t) {
    return t > kLabDelta * kLabDelta * kLabDelta ? 1.0 / (3.0 * std::cbrt(t * t)) : 1.0 / (3.0 * kLabDelta * kLabDelta);
}

// CIELAB of scene RGB, and its Jacobian with respect to that RGB, columns d/dR, d/dG, d/dB.
struct Lab {
    glm::dvec3 value;
    glm::dmat3 jacobian;
};
Lab toLab(const Colorimetry& c, const glm::dvec3& rgb) {
    const glm::dvec3 t = (c.sceneToXyz * rgb) / c.whiteXyz;
    const glm::dvec3 f(labF(t.x), labF(t.y), labF(t.z));
    const glm::dvec3 df(labDf(t.x) / c.whiteXyz.x, labDf(t.y) / c.whiteXyz.y, labDf(t.z) / c.whiteXyz.z);
    // d(L, a, b)/d(X, Y, Z), rows L, a, b, in glm's column-major constructor.
    const glm::dmat3 dLabDXyz(0.0, 500.0 * df.x, 0.0, 116.0 * df.y, -500.0 * df.y, 200.0 * df.y, 0.0, 0.0, -200.0 * df.z);
    return {{(116.0 * f.y) - 16.0, 500.0 * (f.x - f.y), 200.0 * (f.y - f.z)}, dLabDXyz * c.sceneToXyz};
}

// The paper's spectrum: sigmoid(c0 x^2 + c1 x + c2) over x in [0, 1] across 360-830 nm, s = 1/2 + t / (2 sqrt(1 + t^2)).
struct Residual {
    glm::dvec3 value;     // Lab(spectrum) - Lab(target)
    glm::dmat3 jacobian;  // d value / d coefficients, exact
};
Residual residual(const Colorimetry& c, const glm::dvec3& coefficients, const glm::dvec3& targetLab) {
    glm::dvec3 rgb(0.0);
    glm::dmat3 dRgb(0.0);
    for (int i = 0; i < cie::kSampleCount; ++i) {
        const double x = static_cast<double>(i) / (cie::kSampleCount - 1);
        const double t = (((coefficients.x * x) + coefficients.y) * x) + coefficients.z;
        const double root = std::sqrt(1.0 + (t * t));
        const glm::dvec3& w = c.rgbPerSample[static_cast<std::size_t>(i)];
        rgb += w * (0.5 + (0.5 * t / root));
        // ds/dt = 1 / (2 (1 + t^2)^(3/2)); dt/dc = (x^2, x, 1).
        const double ds = 0.5 / (root * root * root);
        dRgb += glm::dmat3(w * (ds * x * x), w * (ds * x), w * ds);
    }
    const Lab lab = toLab(c, rgb);
    return {lab.value - targetLab, lab.jacobian * dRgb};
}

// Gauss-Newton with backtracking: halve a step until the residual falls; stop where a full step no longer moves a float coefficient.
glm::dvec3 fit(const Colorimetry& c, const glm::dvec3& target, glm::dvec3 coefficients) {
    const glm::dvec3 targetLab = toLab(c, target).value;
    Residual r = residual(c, coefficients, targetLab);
    double norm = glm::dot(r.value, r.value);
    // Quadratic convergence needs a handful of steps; the cap only bounds a target outside the object-colour solid, which never closes.
    constexpr int kMaxSteps = 64;
    for (int step = 0; step < kMaxSteps && norm > 0.0; ++step) {
        const glm::dvec3 delta = glm::inverse(r.jacobian) * r.value;
        if (!std::isfinite(delta.x) || !std::isfinite(delta.y) || !std::isfinite(delta.z)) {
            break;
        }
        double scale = 1.0;
        bool improved = false;
        // Down to the step that no longer changes any coefficient in float, the precision the table is stored at.
        while (true) {
            const glm::dvec3 trial = coefficients - (scale * delta);
            if (glm::dvec3(glm::vec3(trial)) == glm::dvec3(glm::vec3(coefficients))) {
                break;
            }
            const Residual tried = residual(c, trial, targetLab);
            const double triedNorm = glm::dot(tried.value, tried.value);
            if (triedNorm < norm) {
                coefficients = trial;
                r = tried;
                norm = triedNorm;
                improved = true;
                break;
            }
            scale *= 0.5;
        }
        if (!improved) {
            break;
        }
    }
    return coefficients;
}

// x = (lambda - 360) / 470 substituted into c0 x^2 + c1 x + c2: the polynomial in nm the renderer evaluates directly.
glm::dvec3 toNanometres(const glm::dvec3& c) {
    const double a = 1.0 / (cie::kLambdaMaxNm - cie::kLambdaMinNm);
    const double b = -cie::kLambdaMinNm * a;
    return {c.x * a * a, (2.0 * c.x * a * b) + (c.y * a), (c.x * b * b) + (c.y * b) + c.z};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: rgb_spectrum_table <out.bin>\n";
        return 2;
    }
    const Colorimetry c = colorimetry();
    std::vector<float> out(table::kFloats);
    for (int k = 0; k < table::kResolution; ++k) {
        out[static_cast<std::size_t>(k)] = static_cast<float>(table::scaleKnot(k));
    }
    // Zero coefficients are the flat 1/2, grey's exact fit at scale 1/2: each (l, j, i) column continues outward from the knot nearest it.
    const int start = table::kResolution / 2;
    tools::parallelFor(3 * table::kResolution, tools::hardwareThreads(), [&](int lj) {
        const int l = lj / table::kResolution;
        const int j = lj % table::kResolution;
        for (int i = 0; i < table::kResolution; ++i) {
            const double x = static_cast<double>(i) / (table::kResolution - 1);
            const double y = static_cast<double>(j) / (table::kResolution - 1);
            const auto solve = [&](int k, glm::dvec3& coefficients) {
                const double z = table::scaleKnot(k);
                glm::dvec3 rgb(0.0);
                rgb[l] = z;
                rgb[(l + 1) % 3] = x * z;
                rgb[(l + 2) % 3] = y * z;
                coefficients = fit(c, rgb, coefficients);
                const glm::dvec3 nm = toNanometres(coefficients);
                const std::size_t at = table::kResolution + table::nodeOffset(l, k, j, i);
                out[at] = static_cast<float>(nm.x);
                out[at + 1] = static_cast<float>(nm.y);
                out[at + 2] = static_cast<float>(nm.z);
            };
            glm::dvec3 upward(0.0);
            for (int k = start; k < table::kResolution; ++k) {
                solve(k, upward);
            }
            glm::dvec3 downward(0.0);
            for (int k = start; k >= 0; --k) {
                solve(k, downward);
            }
        }
    });
    std::ofstream file(argv[1], std::ios::binary);
    file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    if (!file) {
        std::cerr << "rgb_spectrum_table: cannot write " << argv[1] << "\n";
        return 1;
    }
    return 0;
}
