#include "pathtracer/scene/debug_camera_controller.h"

#include <cmath>

#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>

#include "pathtracer/platform/window.h"

namespace pathtracer::scene {

namespace {

constexpr glm::vec3 kWorldUp{0.0F, 1.0F, 0.0F};

}  // namespace

DebugCameraController::DebugCameraController(const glm::vec3& position, float yawDegrees,
                                              float pitchDegrees, Camera::FilmBack filmBack,
                                              float focalLengthMm, float nearClip, float farClip,
                                              float aperture, float shutterSeconds, float iso, Lens lens,
                                              float flySpeedMetersPerSecond,
                                              float orbitSensitivityDegPerPixel)
    : position_(position),
      yawDegrees_(yawDegrees),
      pitchDegrees_(pitchDegrees),
      focalLengthMm_(focalLengthMm),
      defaultPosition_(position),
      defaultYawDegrees_(yawDegrees),
      defaultPitchDegrees_(pitchDegrees),
      filmBack_(filmBack),
      nearClip_(nearClip),
      farClip_(farClip),
      aperture_(aperture),
      shutterSeconds_(shutterSeconds),
      iso_(iso),
      lens_(lens),
      defaultAperture_(aperture),
      defaultShutterSeconds_(shutterSeconds),
      defaultIso_(iso),
      flySpeedMetersPerSecond_(flySpeedMetersPerSecond),
      orbitSensitivityDegPerPixel_(orbitSensitivityDegPerPixel) {}

Camera DebugCameraController::snapshot() const {
    return Camera(position_, yawDegrees_, pitchDegrees_, filmBack_, focalLengthMm_, nearClip_,
                  farClip_, aperture_, shutterSeconds_, iso_, lens_);
}

void DebugCameraController::applyFlyInput(const pathtracer::platform::Window& window,
                                           float dtSeconds) {
    if (orbiting_) {
        return;
    }

    // right is horizontal even when forward is pitched up/down: cross((fx,fy,fz), (0,1,0)) = (-fz, 0, fx), no y component.
    const glm::vec3 forward = snapshot().forward();
    const glm::vec3 right = glm::normalize(glm::cross(forward, kWorldUp));
    const float distance = flySpeedMetersPerSecond_ * dtSeconds;

    if (window.isKeyDown(GLFW_KEY_W)) {
        position_ += forward * distance;
    }
    if (window.isKeyDown(GLFW_KEY_S)) {
        position_ -= forward * distance;
    }
    if (window.isKeyDown(GLFW_KEY_D)) {
        position_ += right * distance;
    }
    if (window.isKeyDown(GLFW_KEY_A)) {
        position_ -= right * distance;
    }
    if (window.isKeyDown(GLFW_KEY_E)) {
        position_ += kWorldUp * distance;
    }
    if (window.isKeyDown(GLFW_KEY_Q)) {
        position_ -= kWorldUp * distance;
    }
}

void DebugCameraController::beginOrbit(const glm::vec3& pivot) {
    pivot_ = pivot;
    orbiting_ = true;
}

void DebugCameraController::applyOrbitDelta(float dxPixels, float dyPixels) {
    if (!orbiting_) {
        return;
    }

    glm::vec3 offset = position_ - pivot_;

    const float yawDeltaDegrees = -dxPixels * orbitSensitivityDegPerPixel_;
    offset = glm::vec3(glm::rotate(glm::mat4(1.0F), glm::radians(yawDeltaDegrees), kWorldUp) *
                        glm::vec4(offset, 0.0F));

    // The outer guard only stops cross(kWorldUp, offset) degenerating into a NaN at the pole; acceptance is re-checked below.
    const float offsetLength = glm::length(offset);
    if (offsetLength > 1e-5F && std::abs(offset.y / offsetLength) < 0.999F) {
        const float pitchDeltaDegrees = -dyPixels * orbitSensitivityDegPerPixel_;
        const glm::vec3 right = glm::normalize(glm::cross(kWorldUp, offset));
        const glm::vec3 candidate = glm::vec3(
            glm::rotate(glm::mat4(1.0F), glm::radians(pitchDeltaDegrees), right) *
            glm::vec4(offset, 0.0F));
        if (std::abs(candidate.y / offsetLength) < 0.99F) {
            offset = candidate;
        }
    }

    position_ = pivot_ + offset;

    // Re-derive yaw and pitch from the new look direction, the inverse of forwardFromEuler, so WASD fly resumes seamlessly.
    const glm::vec3 lookDir = glm::normalize(pivot_ - position_);
    yawDegrees_ = glm::degrees(std::atan2(-lookDir.x, -lookDir.z));
    pitchDegrees_ = glm::clamp(glm::degrees(std::asin(glm::clamp(lookDir.y, -1.0F, 1.0F))),
                                -89.0F, 89.0F);
}

void DebugCameraController::endOrbit() {
    orbiting_ = false;
}

void DebugCameraController::resetToDefault() {
    position_ = defaultPosition_;
    yawDegrees_ = defaultYawDegrees_;
    pitchDegrees_ = defaultPitchDegrees_;
}

}  // namespace pathtracer::scene
