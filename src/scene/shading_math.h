#pragma once

#include <cmath>
#include <numbers>

#include <glm/glm.hpp>

// The scalar helpers every shading source shares, defined once.
namespace pathtracer::scene {

inline constexpr float kPi = std::numbers::pi_v<float>;

inline float lerp1(float a, float b, float t) { return a + ((b - a) * t); }

// atan2(|a x b|, a.b), PBRT v4's AngleBetween: robust near 0 and pi, where acos of a dot product loses half its digits.
inline float angleBetween(const glm::vec3& a, const glm::vec3& b) { return std::atan2(glm::length(glm::cross(a, b)), glm::dot(a, b)); }

// Rows (t, b, n) of an orthonormal basis about unit n, branchless (Duff et al. 2017); the identity at n = +z.
inline glm::mat3 toFrameAbout(const glm::vec3& n) {
    const float s = std::copysign(1.0F, n.z);
    const float a = -1.0F / (s + n.z);
    const float b = n.x * n.y * a;
    return glm::transpose(glm::mat3(glm::vec3(1.0F + (s * n.x * n.x * a), s * b, -s * n.x), glm::vec3(b, s + (n.y * n.y * a), -n.y), n));
}

}  // namespace pathtracer::scene
