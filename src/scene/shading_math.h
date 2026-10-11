#pragma once

#include <cmath>

#include <glm/glm.hpp>

// The scalar helpers every shading source shares, defined once.
namespace pathtracer::scene {

inline constexpr float kPi = 3.14159265F;

inline float lerp1(float a, float b, float t) { return a + ((b - a) * t); }

// The equal-weight RGB mean: selection masses and energies compare channels on one scale.
inline float channelMean(const glm::vec3& v) { return (v.x + v.y + v.z) / 3.0F; }

// Rows (t, b, n) of an orthonormal basis about unit n, branchless (Duff et al. 2017); the identity at n = +z.
inline glm::mat3 toFrameAbout(const glm::vec3& n) {
    const float s = std::copysign(1.0F, n.z);
    const float a = -1.0F / (s + n.z);
    const float b = n.x * n.y * a;
    return glm::transpose(glm::mat3(glm::vec3(1.0F + (s * n.x * n.x * a), s * b, -s * n.x), glm::vec3(b, s + (n.y * n.y * a), -n.y), n));
}

}  // namespace pathtracer::scene
