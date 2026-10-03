#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace pathtracer::scene {

// Rz * Ry * Rx of XYZ degrees, so X applies first, about fixed world axes: the one rotation convention of scene.json and the APIs.
[[nodiscard]] inline glm::mat3 rotationXyz(const glm::vec3& degrees) {
    const glm::vec3 radians = glm::radians(degrees);
    return glm::mat3(glm::rotate(glm::mat4(1.0F), radians.z, glm::vec3(0.0F, 0.0F, 1.0F)) *
                     glm::rotate(glm::mat4(1.0F), radians.y, glm::vec3(0.0F, 1.0F, 0.0F)) *
                     glm::rotate(glm::mat4(1.0F), radians.x, glm::vec3(1.0F, 0.0F, 0.0F)));
}

// rotationXyz's inverse, y in [-90, 90] and x, z in [-180, 180]: Day 2014's extraction, which still recomposes the matrix at y = +/-90.
[[nodiscard]] inline glm::vec3 eulerXyzDegrees(const glm::mat3& rotation) {
    glm::vec3 radians;
    glm::extractEulerAngleZYX(glm::mat4(rotation), radians.z, radians.y, radians.x);
    // + 0 turns atan2's -0 (from M[0][2] = -0 at rest) into +0, so an unturned pose reads, logs and rebuilds as the authored zeros.
    return glm::degrees(radians) + glm::vec3(0.0F);
}

}  // namespace pathtracer::scene
