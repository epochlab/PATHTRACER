#include "pathtracer/scene/camera.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace pathtracer::scene {

namespace {

constexpr glm::vec3 kWorldUp{0.0F, 1.0F, 0.0F};

// Right-handed Euler forward vector, parameterised so yaw=0, pitch=0 points down -Z without the usual -90-degree yaw offset.
glm::vec3 forwardFromEuler(float yawRadians, float pitchRadians) {
    const float cosPitch = std::cos(pitchRadians);
    return glm::normalize(glm::vec3(-std::sin(yawRadians) * cosPitch, std::sin(pitchRadians),
                                     -std::cos(yawRadians) * cosPitch));
}

// Kannala-Brandt arm: the sensor point in mm sets the image radius, the model inverts it to a polar angle about the optical axis.
std::optional<Ray> fisheyeRay(const Camera::ViewBasis& basis, const glm::vec3& origin, float nearClip,
                               float farClip, float ndcX, float ndcY) {
    const float xMm = ndcX * basis.halfWidthMm;
    const float yMm = ndcY * basis.halfHeightMm;
    const float radiusMm = std::hypot(xMm, yMm);
    // Past the image circle the lens forms no image at all, which is the black corner of a real circular fisheye, not a clamp.
    if (radiusMm > basis.maxRadiusMm) {
        return std::nullopt;
    }
    if (radiusMm == 0.0F) {
        return Ray{origin, basis.forward, nearClip, farClip};
    }
    const float theta = kannalaBrandtTheta(basis.lens.radialCoefficients, radiusMm / basis.focalLengthMm,
                                            basis.maxThetaRadians);
    // The sensor offset normalised is the azimuth, so this is the polar reconstruction of the direction the lens imaged onto that point.
    const glm::vec3 azimuth = ((xMm * basis.right) + (yMm * basis.up)) / radiusMm;
    // Unit by construction: forward and the azimuth are orthonormal, and cos^2 + sin^2 = 1 to a rounding.
    const glm::vec3 dir = (std::cos(theta) * basis.forward) + (std::sin(theta) * azimuth);
    return Ray{origin, dir, nearClip, farClip};
}

}  // namespace

Camera::Camera(const glm::vec3& position, float yawDegrees, float pitchDegrees, FilmBack filmBack,
               float focalLengthMm, float nearClip, float farClip, float aperture,
               float shutterSeconds, float iso, Lens lens)
    : position_(position),
      yawRadians_(glm::radians(yawDegrees)),
      pitchRadians_(glm::radians(pitchDegrees)),
      filmBack_(filmBack),
      focalLengthMm_(focalLengthMm),
      nearClip_(nearClip),
      farClip_(farClip),
      aperture_(aperture),
      shutterSeconds_(shutterSeconds),
      iso_(iso),
      lens_(lens) {}

glm::vec3 Camera::forward() const {
    return forwardFromEuler(yawRadians_, pitchRadians_);
}

float Camera::verticalFovRadians() const {
    return 2.0F * std::atan(filmBack_.heightMm / (2.0F * focalLengthMm_));
}

float Camera::verticalAngularExtentRadians() const {
    if (lens_.projection == LensProjection::Rectilinear) {
        return verticalFovRadians();
    }
    // The angle imaged at the top of the gate, or the circle's edge where the circle falls inside it: the frame's real vertical extent.
    const float thetaMax = maxThetaRadians(lens_);
    const float halfHeightRadii = (0.5F * filmBack_.heightMm) / focalLengthMm_;
    return 2.0F * kannalaBrandtTheta(lens_.radialCoefficients, halfHeightRadii, thetaMax);
}

float Camera::pixelsPerRadian(int heightPixels) const {
    return focalLengthMm_ * static_cast<float>(heightPixels) / filmBack_.heightMm;
}

Camera::ViewBasis Camera::viewBasis(float aspect) const {
    const glm::vec3 fwd = forwardFromEuler(yawRadians_, pitchRadians_);
    const glm::vec3 right = glm::normalize(glm::cross(fwd, kWorldUp));
    const glm::vec3 up = glm::cross(right, fwd);
    const float halfHeight = std::tan(verticalFovRadians() * 0.5F);
    const float halfWidth = halfHeight * aspect;
    // Sensor width from the gate height and the render aspect, as the pinhole vfov is: widthMm stays display-only, so pixels stay square.
    const float halfHeightMm = 0.5F * filmBack_.heightMm;
    const float halfWidthMm = halfHeightMm * aspect;
    const float maxTheta = maxThetaRadians(lens_);
    // r_max = f * theta_d(thetaMax): the authored focal length and the gate alone decide whether the circle falls inside the frame.
    const float maxRadiusMm = focalLengthMm_ * kannalaBrandtRadius(lens_.radialCoefficients, maxTheta);
    return ViewBasis{fwd, right, up, halfWidth, halfHeight, halfWidthMm, halfHeightMm, maxTheta,
                     maxRadiusMm, focalLengthMm_, lens_};
}

std::optional<Ray> Camera::primaryRay(float ndcX, float ndcY, float aspect) const {
    return primaryRay(viewBasis(aspect), ndcX, ndcY);
}

std::optional<Ray> Camera::primaryRay(const ViewBasis& basis, float ndcX, float ndcY) const {
    if (basis.lens.projection == LensProjection::FisheyePolynomial) {
        return fisheyeRay(basis, position_, nearClip_, farClip_, ndcX, ndcY);
    }
    const glm::vec3 dir = glm::normalize(basis.forward + (ndcX * basis.halfWidth * basis.right) +
                                          (ndcY * basis.halfHeight * basis.up));
    return Ray{position_, dir, nearClip_, farClip_};
}

float Camera::ev100() const {
    return ev100(aperture_, shutterSeconds_, iso_);
}

float Camera::ev100(float aperture, float shutterSeconds, float iso) {
    return std::log2((aperture * aperture) / shutterSeconds * (100.0F / iso));
}

}  // namespace pathtracer::scene
