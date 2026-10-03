#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace pathtracer::scene {

// Rz * Ry * Rx of XYZ degrees, so X applies first, about fixed world axes: the one rotation convention of scene.json and the APIs.
[[nodiscard]] inline glm::mat3 rotationXyz(const glm::vec3& degrees) {
    const glm::vec3 radians = glm::radians(degrees);
    return glm::mat3(glm::rotate(glm::mat4(1.0F), radians.z, glm::vec3(0.0F, 0.0F, 1.0F)) *
                     glm::rotate(glm::mat4(1.0F), radians.y, glm::vec3(0.0F, 1.0F, 0.0F)) *
                     glm::rotate(glm::mat4(1.0F), radians.x, glm::vec3(1.0F, 0.0F, 0.0F)));
}

}  // namespace pathtracer::scene
