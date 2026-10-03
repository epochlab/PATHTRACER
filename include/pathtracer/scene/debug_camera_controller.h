#pragma once

#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "pathtracer/scene/camera.h"

namespace pathtracer::platform {
class Window;
}

namespace pathtracer::scene {

// Mutable fly/orbit state producing a fresh immutable Camera each frame via snapshot(); this holds the debug rig that drives it.
class DebugCameraController {
public:
    // position/rotation are the initial pose and what resetToDefault() restores; the lens parameters pass through to every snapshot().
    DebugCameraController(const glm::vec3& position, const glm::vec3& rotationDegrees,
                           Camera::FilmBack filmBack, float focalLengthMm, float nearClip,
                           float farClip, float aperture, float shutterSeconds, float iso, Lens lens,
                           float flySpeedMetersPerSecond, float orbitSensitivityDegPerPixel);

    // Builds an immutable Camera from the current pose. Call once per frame: this is the only point where a Camera value exists.
    [[nodiscard]] Camera snapshot() const;

    // Polls WASD/QE: along the view's forward and right, Q/E along world up, scaled by dtSeconds. No-op while orbiting.
    void applyFlyInput(const pathtracer::platform::Window& window, float dtSeconds);

    void beginOrbit(const glm::vec3& pivot);

    // This frame's cursor delta turns the rig rigidly about pivot_: yaw about world up, pitch about the view's right. Roll survives.
    void applyOrbitDelta(float dxPixels, float dyPixels);

    void endOrbit();
    [[nodiscard]] bool isOrbiting() const { return orbiting_; }

    void resetToDefault();

    [[nodiscard]] float focalLengthMm() const { return focalLengthMm_; }
    [[nodiscard]] Camera::FilmBack filmBack() const { return filmBack_; }
    [[nodiscard]] float aperture() const { return aperture_; }
    [[nodiscard]] float shutterSeconds() const { return shutterSeconds_; }
    [[nodiscard]] float iso() const { return iso_; }
    [[nodiscard]] Lens lens() const { return lens_; }

    // Bound to the HUD's Camera section sliders/dropdown via setter, not a bare reference (controller-owned state).
    void setFocalLengthMm(float focalLengthMm) { focalLengthMm_ = focalLengthMm; }
    void setFilmBack(Camera::FilmBack filmBack) { filmBack_ = filmBack; }
    void setAperture(float aperture) { aperture_ = aperture; }
    void setShutterSeconds(float shutterSeconds) { shutterSeconds_ = shutterSeconds; }
    void setIso(float iso) { iso_ = iso; }
    // Projection only: the polynomial and its field of view are measured calibration data, authored in profile.json and immutable here.
    void setLensProjection(LensProjection projection) { lens_.projection = projection; }

    // EV100 delta against profile.json defaults, applied at the display stage as pow(2,ev): the scene is not photometrically calibrated.
    [[nodiscard]] float relativeExposureEv() const {
        const float defaultEv100 = Camera::ev100(defaultAperture_, defaultShutterSeconds_, defaultIso_);
        const float currentEv100 = Camera::ev100(aperture_, shutterSeconds_, iso_);
        return defaultEv100 - currentEv100;
    }

private:
    glm::vec3 position_;
    // The master orientation, unit and renormalised per turn: Euler angles are derived from it, never accumulated, so no angle is singular.
    glm::quat orientation_;
    float focalLengthMm_;

    const glm::vec3 defaultPosition_;
    const glm::quat defaultOrientation_;

    Camera::FilmBack filmBack_;
    const float nearClip_;
    const float farClip_;
    float aperture_;
    float shutterSeconds_;
    float iso_;
    Lens lens_;

    const float defaultAperture_;
    const float defaultShutterSeconds_;
    const float defaultIso_;

    const float flySpeedMetersPerSecond_;
    const float orbitSensitivityDegPerPixel_;

    bool orbiting_ = false;
    glm::vec3 pivot_{0.0F};
};

}  // namespace pathtracer::scene
