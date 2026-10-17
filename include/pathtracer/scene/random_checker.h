#pragma once

#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>

namespace pathtracer::scene {

// Random binary checker (Bertrand et al. 2015, PLoS Comput Biol): fair, independent bits on square cells cellSize wide in UV.
struct RandomChecker {
    float cellSize;
    std::uint32_t seed;

    bool operator==(const RandomChecker&) const = default;  // MaterialInput's variant equality, which compares bound materials
};

// pcg3d (Jarzynski & Olano 2020, JCGT 9(3)): transcribed, never derived; their best-ranked 3D-to-3D hash for quality per cost.
[[nodiscard]] inline glm::uvec3 pcg3d(glm::uvec3 v) {
    v = (v * 1664525U) + 1013904223U;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16U;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// 0 or 1 at uv, point-sampled; cells are independent, so a UV seam is invisible exactly when it spans a whole number of cells.
[[nodiscard]] inline float randomCheckerAt(const RandomChecker& checker, glm::vec2 uv) {
    const auto cell = [size = static_cast<double>(checker.cellSize)](float coordinate) {
        // Floored mod 2^32, the hash's domain: defined for any finite uv / cellSize, negative cells included.
        constexpr double kDomain = 4294967296.0;
        const double index = std::fmod(std::floor(static_cast<double>(coordinate) / size), kDomain);
        return static_cast<std::uint32_t>(index < 0.0 ? index + kDomain : index);
    };
    return static_cast<float>(pcg3d(glm::uvec3(cell(uv.x), cell(uv.y), checker.seed)).x >> 31U);
}

}  // namespace pathtracer::scene
