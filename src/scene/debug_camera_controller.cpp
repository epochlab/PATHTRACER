#include "pathtracer/scene/debug_camera_controller.h"

#include <GLFW/glfw3.h>

#include "pathtracer/platform/window.h"
#include "pathtracer/scene/rotation.h"

namespace pathtracer::scene {

namespace {

constexpr glm::vec3 kWorldUp{0.0F, 1.0F, 0.0F};

}  // namespace

DebugCameraController::DebugCameraController(const glm::vec3& position, const glm::vec3& rotationDegrees,
                                              Camera::FilmBack filmBack,
                                              float focalLengthMm, float nearClip, float farClip,
                                              float aperture, float shutterSeconds, float iso, Lens lens,
                                              float flySpeedMetersPerSecond,
                                              float orbitSensitivityDegPerPixel)
    : position_(position),
      orientation_(glm::quat_cast(rotationXyz(rotationDegrees))),
      focalLengthMm_(focalLengthMm),
      defaultPosition_(position),
      defaultOrientation_(orientation_),
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
    return Camera(position_, eulerXyzDegrees(glm::mat3_cast(orientation_)), filmBack_, focalLengthMm_, nearClip_,
                  farClip_, aperture_, shutterSeconds_, iso_, lens_);
}

void DebugCameraController::applyFlyInput(const pathtracer::platform::Window& window,
                                           float dtSeconds) {
    if (orbiting_) {
        return;
    }

    // The view's own axes, so a rolled or inverted camera flies where it looks; Q/E stay on world up, the one axis independent of view.
    const glm::mat3 axes = glm::mat3_cast(orientation_);
    const glm::vec3 forward = -axes[2];
    const glm::vec3 right = axes[0];
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

    // Yaw multiplies on the left, about world up; pitch on the right, about the view's own X. Neither has a pole, so nothing is clamped.
    const glm::quat yaw = glm::angleAxis(glm::radians(-dxPixels * orbitSensitivityDegPerPixel_), kWorldUp);
    const glm::quat pitch = glm::angleAxis(glm::radians(-dyPixels * orbitSensitivityDegPerPixel_), glm::vec3(1.0F, 0.0F, 0.0F));
    const glm::quat turned = glm::normalize(yaw * orientation_ * pitch);
    // The same rigid turn carries the eye about the pivot, so a pivot on the view axis stays on it and the image orbits, not pans.
    position_ = pivot_ + ((turned * glm::inverse(orientation_)) * (position_ - pivot_));
    orientation_ = turned;
}

void DebugCameraController::endOrbit() {
    orbiting_ = false;
}

void DebugCameraController::resetToDefault() {
    position_ = defaultPosition_;
    orientation_ = defaultOrientation_;
}

}  // namespace pathtracer::scene
