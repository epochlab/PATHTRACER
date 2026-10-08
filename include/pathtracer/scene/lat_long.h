#pragma once

#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace pathtracer::scene {

// Plate carree (Snyder 1987) on a frame (right, up, forward): u = 1/2 + longitude/2pi from forward toward right, v = colatitude/pi.

// (u, v) of a direction of any non-zero length. atan2 for both angles: acos(up) keeps only half its digits near a pole.
[[nodiscard]] inline glm::vec2 latLongUv(const glm::vec3& local) {
    const float longitude = std::atan2(local.x, local.z);
    const float colatitude = std::atan2(std::hypot(local.x, local.z), local.y);
    return {0.5F + (longitude / glm::two_pi<float>()), colatitude / glm::pi<float>()};
}

// d(u, v)/d(local), columns d/dx, d/dy, d/dz: radial rates vanish, so it serves any length; singular at the poles, where u is.
[[nodiscard]] inline glm::mat3x2 latLongUvJacobian(const glm::vec3& local) {
    const float rhoSquared = (local.x * local.x) + (local.z * local.z);
    const float rho = std::sqrt(rhoSquared);
    const float uScale = 1.0F / (glm::two_pi<float>() * rhoSquared);
    const float vScale = 1.0F / (glm::pi<float>() * (rhoSquared + (local.y * local.y)));
    const float vRadial = vScale * local.y / rho;
    return {glm::vec2(uScale * local.z, vRadial * local.x), glm::vec2(0.0F, -vScale * rho), glm::vec2(-uScale * local.x, vRadial * local.z)};
}

// Unit direction at (u, v), latLongUv's inverse. Periodic in u, so a u outside [0, 1) is the same direction one turn over.
[[nodiscard]] inline glm::vec3 latLongDirection(glm::vec2 uv) {
    const float longitude = (uv.x - 0.5F) * glm::two_pi<float>();
    const float colatitude = uv.y * glm::pi<float>();
    const float sinColatitude = std::sin(colatitude);
    return {sinColatitude * std::sin(longitude), std::cos(colatitude), sinColatitude * std::cos(longitude)};
}

}  // namespace pathtracer::scene
