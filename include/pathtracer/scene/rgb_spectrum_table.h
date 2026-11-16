#pragma once

#include <cstddef>

// Jakob & Hanika 2019's RGB-to-spectrum tables: sigmoid-polynomial coefficients over the scene RGB cube, generated at build time.
namespace pathtracer::scene::rgb_spectrum {

// The paper's resolution: 64 nodes on each of the brightest channel's scale and the two ratios beside it.
inline constexpr int kResolution = 64;
inline constexpr std::size_t kCoefficients = 3;

// Scale knots, then per brightest channel l, scale node k, ratio nodes j and i, the three nm coefficients: floats in that order.
inline constexpr std::size_t kNodes = static_cast<std::size_t>(kResolution) * kResolution * kResolution;
inline constexpr std::size_t kFloats = kResolution + (3 * kNodes * kCoefficients);

// smoothstep(smoothstep(k / (res - 1))): knots dense near black and white, where the fitted spectra change fastest (the paper's axis).
[[nodiscard]] constexpr double scaleKnot(int k) {
    const auto smoothstep = [](double x) { return x * x * (3.0 - (2.0 * x)); };
    return smoothstep(smoothstep(static_cast<double>(k) / (kResolution - 1)));
}

// Offset of node (l, k, j, i)'s first coefficient into the data that follows the scale knots.
[[nodiscard]] constexpr std::size_t nodeOffset(int l, int k, int j, int i) {
    return ((((static_cast<std::size_t>(l) * kResolution + static_cast<std::size_t>(k)) * kResolution + static_cast<std::size_t>(j)) *
                 kResolution) +
            static_cast<std::size_t>(i)) *
           kCoefficients;
}

}  // namespace pathtracer::scene::rgb_spectrum
