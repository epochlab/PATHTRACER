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

// A sensor point in polar form with the direction the lens images onto it: the one inverse solve a ray and its differential share.
struct FisheyeSample {
    float xMm;
    float yMm;
    float radiusMm;
    float theta;
    glm::vec3 dir;
};

// Kannala-Brandt arm: the sensor point in mm sets the image radius, the model inverts it to a polar angle about the optical axis.
std::optional<FisheyeSample> fisheyeSample(const Camera::ViewBasis& basis, float ndcX, float ndcY) {
    const float xMm = ndcX * basis.halfWidthMm;
    const float yMm = ndcY * basis.halfHeightMm;
    const float radiusMm = std::hypot(xMm, yMm);
    // Past the image circle the lens forms no image at all, which is the black corner of a real circular fisheye, not a clamp.
    if (radiusMm > basis.maxRadiusMm) {
        return std::nullopt;
    }
    if (radiusMm == 0.0F) {
        return FisheyeSample{xMm, yMm, radiusMm, 0.0F, basis.forward};
    }
    const float theta = kannalaBrandtTheta(basis.lens.radialCoefficients, radiusMm / basis.focalLengthMm,
                                            basis.maxThetaRadians);
    // The sensor offset normalised is the azimuth, so this is the polar reconstruction of the direction the lens imaged onto that point.
    const glm::vec3 azimuth = ((xMm * basis.right) + (yMm * basis.up)) / radiusMm;
    // Unit by construction: forward and the azimuth are orthonormal, and cos^2 + sin^2 = 1 to a rounding.
    const glm::vec3 dir = (std::cos(theta) * basis.forward) + (std::sin(theta) * azimuth);
    return FisheyeSample{xMm, yMm, radiusMm, theta, dir};
}

// d(dir)/d(ndc) of a fisheye sample: radial via d(theta)/dr = 1/(f theta_d'(theta)), azimuthal via sin(theta)/r, chained by mm/ndc.
glm::mat2x3 fisheyeDirPerNdc(const Camera::ViewBasis& basis, const FisheyeSample& sample) {
    const glm::vec2 mmPerNdc(basis.halfWidthMm, basis.halfHeightMm);
    // On the axis both rates tend to 1/f and the azimuth drops out: the lens is locally a pinhole of focal length f.
    if (sample.radiusMm == 0.0F) {
        return glm::mat2x3(basis.right * (mmPerNdc.x / basis.focalLengthMm), basis.up * (mmPerNdc.y / basis.focalLengthMm));
    }
    const glm::vec2 radial = glm::vec2(sample.xMm, sample.yMm) / sample.radiusMm;
    const glm::vec3 azimuth = (radial.x * basis.right) + (radial.y * basis.up);
    const glm::vec3 tangent = (radial.x * basis.up) - (radial.y * basis.right);
    const float sinTheta = std::sin(sample.theta);
    const float thetaPerMm =
        1.0F / (basis.focalLengthMm * kannalaBrandtRadiusSlope(basis.lens.radialCoefficients, sample.theta));
    const glm::vec3 dirPerRadiusMm = ((std::cos(sample.theta) * azimuth) - (sinTheta * basis.forward)) * thetaPerMm;
    const glm::vec3 dirPerArcMm = tangent * (sinTheta / sample.radiusMm);
    return glm::mat2x3((dirPerRadiusMm * radial.x) - (dirPerArcMm * radial.y),
                       (dirPerRadiusMm * radial.y) + (dirPerArcMm * radial.x)) *
           glm::mat2(mmPerNdc.x, 0.0F, 0.0F, mmPerNdc.y);
}

// Kannala & Brandt 2006 forward model: polar angle off the axis, image radius f * theta_d(theta) along the azimuth.
std::optional<glm::vec2> fisheyeProjection(const Camera::ViewBasis& basis, const glm::vec3& view) {
    const float depth = glm::dot(view, basis.forward);
    const glm::vec2 lateral(glm::dot(view, basis.right), glm::dot(view, basis.up));
    const float lateralLength = glm::length(lateral);
    const float theta = std::atan2(lateralLength, depth);
    if (theta > basis.maxThetaRadians) {
        return std::nullopt;
    }
    // On the axis the azimuth is undefined: ahead images the centre, behind (theta = pi under a 360-degree lens) the whole rim.
    if (lateralLength == 0.0F) {
        return depth > 0.0F ? std::optional<glm::vec2>(glm::vec2(0.0F)) : std::nullopt;
    }
    const float radiusMm = basis.focalLengthMm * kannalaBrandtRadius(basis.lens.radialCoefficients, theta);
    return (radiusMm / lateralLength) * lateral / glm::vec2(basis.halfWidthMm, basis.halfHeightMm);
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
        const std::optional<FisheyeSample> sample = fisheyeSample(basis, ndcX, ndcY);
        return sample ? std::optional(Ray{position_, sample->dir, nearClip_, farClip_}) : std::nullopt;
    }
    const glm::vec3 dir = glm::normalize(basis.forward + (ndcX * basis.halfWidth * basis.right) +
                                          (ndcY * basis.halfHeight * basis.up));
    return Ray{position_, dir, nearClip_, farClip_};
}

std::optional<Camera::RayDifferential> Camera::primaryRayDifferential(const ViewBasis& basis, float ndcX,
                                                                      float ndcY) const {
    if (basis.lens.projection == LensProjection::FisheyePolynomial) {
        const std::optional<FisheyeSample> sample = fisheyeSample(basis, ndcX, ndcY);
        if (!sample) {
            return std::nullopt;
        }
        return RayDifferential{Ray{position_, sample->dir, nearClip_, farClip_}, fisheyeDirPerNdc(basis, *sample)};
    }
    // The unnormalised pinhole direction is affine in ndc; normalising projects its rate off the ray: (I - d d^T) v / |v|.
    const glm::vec3 unnormalised =
        basis.forward + (ndcX * basis.halfWidth * basis.right) + (ndcY * basis.halfHeight * basis.up);
    const float length = glm::length(unnormalised);
    const glm::vec3 dir = glm::normalize(unnormalised);
    const auto offRay = [&dir, length](const glm::vec3& rate) { return (rate - (glm::dot(dir, rate) * dir)) / length; };
    return RayDifferential{Ray{position_, dir, nearClip_, farClip_},
                           glm::mat2x3(offRay(basis.halfWidth * basis.right), offRay(basis.halfHeight * basis.up))};
}

Camera::PinholeMatrix Camera::pinholeMatrix(const ViewBasis& basis) const {
    const glm::vec3 ndcRight = basis.right / basis.halfWidth;
    const glm::vec3 ndcUp = basis.up / basis.halfHeight;
    return PinholeMatrix{glm::vec4(ndcRight, -glm::dot(ndcRight, position_)), glm::vec4(ndcUp, -glm::dot(ndcUp, position_)),
                         glm::vec4(basis.forward, -glm::dot(basis.forward, position_))};
}

std::optional<glm::vec2> Camera::project(const ViewBasis& basis, const glm::vec4& point) const {
    if (basis.lens.projection == LensProjection::FisheyePolynomial) {
        return fisheyeProjection(basis, glm::vec3(point) - (point.w * position_));
    }
    return project(pinholeMatrix(basis), point);
}

float Camera::ev100() const {
    return ev100(aperture_, shutterSeconds_, iso_);
}

float Camera::ev100(float aperture, float shutterSeconds, float iso) {
    return std::log2((aperture * aperture) / shutterSeconds * (100.0F / iso));
}

}  // namespace pathtracer::scene
