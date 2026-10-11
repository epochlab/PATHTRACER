#pragma once

#include <glm/glm.hpp>

// The scalar helpers every shading source shares, defined once.
namespace pathtracer::scene {

inline constexpr float kPi = 3.14159265F;

inline float lerp1(float a, float b, float t) { return a + ((b - a) * t); }

// The equal-weight RGB mean: selection masses and energies compare channels on one scale.
inline float channelMean(const glm::vec3& v) { return (v.x + v.y + v.z) / 3.0F; }

}  // namespace pathtracer::scene
