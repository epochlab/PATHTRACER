#include "pathtracer/scene/camera.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>

#include <glm/gtc/constants.hpp>

#include "pathtracer/scene/lat_long.h"

namespace pathtracer::scene {

namespace {

constexpr glm::vec3 kWorldUp{0.0F, 1.0F, 0.0F};

// isfinite first: inf > 0 holds, and a NaN fails both, so every caller's rejection covers it.
bool finitePositive(float value) {
    return std::isfinite(value) && value > 0.0F;
}

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

// The omnidirectional image is the lat-long chart on the camera frame: ndc (0, 0) is forward, +x the camera's right, the top row up.
glm::vec2 omnidirectionalUv(float ndcX, float ndcY) {
    return {0.5F * (ndcX + 1.0F), 0.5F * (1.0F - ndcY)};
}

glm::vec3 omnidirectionalDirection(const Camera::ViewBasis& basis, float ndcX, float ndcY) {
    const glm::vec3 local = latLongDirection(omnidirectionalUv(ndcX, ndcY));
    return (local.x * basis.right) + (local.y * basis.up) + (local.z * basis.forward);
}

// d(longitude)/d(ndcX) = pi and d(colatitude)/d(ndcY) = -pi/2; the x column is pi sin(colatitude), zero only on the pole rows' edges.
glm::mat2x3 omnidirectionalDirPerNdc(const Camera::ViewBasis& basis, float ndcX, float ndcY) {
    const glm::vec2 uv = omnidirectionalUv(ndcX, ndcY);
    const float longitude = (uv.x - 0.5F) * glm::two_pi<float>();
    const float colatitude = uv.y * glm::pi<float>();
    const float sinColatitude = std::sin(colatitude);
    const glm::vec3 horizontal = (std::sin(longitude) * basis.right) + (std::cos(longitude) * basis.forward);
    const glm::vec3 east = (std::cos(longitude) * basis.right) - (std::sin(longitude) * basis.forward);
    return glm::mat2x3(east * (glm::pi<float>() * sinColatitude),
                       ((std::cos(colatitude) * horizontal) - (sinColatitude * basis.up)) * (-0.5F * glm::pi<float>()));
}

// Every direction has an image; only the eye itself, the zero vector, has none. A pole's whole row images it, atan2 picks column 0.
std::optional<glm::vec2> omnidirectionalProjection(const Camera::ViewBasis& basis, const glm::vec3& view) {
    if (view == glm::vec3(0.0F)) {
        return std::nullopt;
    }
    const glm::vec2 uv = latLongUv(glm::vec3(glm::dot(view, basis.right), glm::dot(view, basis.up), glm::dot(view, basis.forward)));
    return glm::vec2((2.0F * uv.x) - 1.0F, 1.0F - (2.0F * uv.y));
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

bool Camera::validate(std::string& error) const {
    std::ostringstream reason;
    if (!(std::isfinite(position_.x) && std::isfinite(position_.y) && std::isfinite(position_.z))) {
        reason << "position (" << position_.x << ", " << position_.y << ", " << position_.z << ") is not finite";
    } else if (!std::isfinite(yawRadians_)) {
        reason << "yaw " << yawDegrees() << " degrees is not finite";
    // right ~ cos(pitch): gimbal lock at +/-90, aliased to yaw + 180 past it. float(pi/2) is the first float past pi/2, so < is exact.
    } else if (!(std::abs(pitchRadians_) < 0.5F * std::numbers::pi_v<float>)) {
        reason << "pitch " << pitchDegrees() << " degrees is not strictly inside (-90, 90)";
    } else if (!validFilmBack(filmBack_)) {
        reason << "film back " << filmBack_.widthMm << " x " << filmBack_.heightMm << " mm is not finite and positive";
    } else if (!finitePositive(focalLengthMm_)) {
        reason << "focal length " << focalLengthMm_ << " mm is not finite and positive";
    // viewBasis's pinhole half-height h/2f, which pinholeMatrix divides by: finite inputs can still overflow or underflow it.
    } else if (!finitePositive((0.5F * filmBack_.heightMm) / focalLengthMm_)) {
        reason << "film back height " << filmBack_.heightMm << " mm over focal length " << focalLengthMm_
               << " mm leaves no finite positive view-plane extent";
    // The ray interval [near, far]; far may be +inf, the unbounded ray.
    } else if (!finitePositive(nearClip_) || !(nearClip_ < farClip_)) {
        reason << "clips near " << nearClip_ << ", far " << farClip_ << " do not satisfy 0 < near < far with near finite";
    } else if (!finitePositive(aperture_)) {
        reason << "aperture f/" << aperture_ << " is not finite and positive";
    } else if (!finitePositive(shutterSeconds_)) {
        reason << "shutter " << shutterSeconds_ << " s is not finite and positive";
    } else if (!finitePositive(iso_)) {
        reason << "ISO " << iso_ << " is not finite and positive";
    // aperture^2 / shutter * 100 / iso overflows or underflows at finite extremes; the display gain is 2^-ev.
    } else if (!std::isfinite(ev100())) {
        reason << "exposure f/" << aperture_ << ", " << shutterSeconds_ << " s, ISO " << iso_ << " has no finite EV100";
    // Checked whichever projection is active: the HUD switches projection at runtime over the same polynomial.
    } else if (!(lens_.maxFieldOfViewDegrees > 0.0F && lens_.maxFieldOfViewDegrees <= 360.0F)) {
        reason << "fisheye field of view " << lens_.maxFieldOfViewDegrees << " degrees is outside (0, 360]";
    } else if (!kannalaBrandtIsInvertible(lens_.radialCoefficients, maxThetaRadians(lens_))) {
        reason << "fisheye coefficients give an r(theta) that is not provably monotone over the field of view";
    }
    if (reason.tellp() == 0) {
        return true;
    }
    error = reason.str();
    return false;
}

bool Camera::validFilmBack(FilmBack filmBack) {
    return finitePositive(filmBack.widthMm) && finitePositive(filmBack.heightMm);
}

glm::vec3 Camera::forward() const {
    return forwardFromEuler(yawRadians_, pitchRadians_);
}

float Camera::verticalFovRadians() const {
    return 2.0F * std::atan(filmBack_.heightMm / (2.0F * focalLengthMm_));
}

float Camera::verticalAngularExtentRadians() const {
    switch (lens_.projection) {
        case LensProjection::FisheyePolynomial: {
            // The angle imaged at the top of the gate, or the circle's edge where the circle falls inside it: the frame's real extent.
            const float thetaMax = maxThetaRadians(lens_);
            const float halfHeightRadii = (0.5F * filmBack_.heightMm) / focalLengthMm_;
            return 2.0F * kannalaBrandtTheta(lens_.radialCoefficients, halfHeightRadii, thetaMax);
        }
        case LensProjection::Omnidirectional:
            return glm::pi<float>();
        // No default arm: -Werror then makes an unrouted new projection a compile error. Count is never a lens.
        case LensProjection::Rectilinear:
        case LensProjection::Count:
            break;
    }
    return verticalFovRadians();
}

Camera::ViewBasis Camera::viewBasis(float aspect) const {
    const glm::vec3 fwd = forwardFromEuler(yawRadians_, pitchRadians_);
    const glm::vec3 right = glm::normalize(glm::cross(fwd, kWorldUp));
    const glm::vec3 up = glm::cross(right, fwd);
    // Sensor width from the gate height and the render aspect, as the pinhole vfov is: widthMm stays display-only, so pixels stay square.
    const float halfHeightMm = 0.5F * filmBack_.heightMm;
    const float halfWidthMm = halfHeightMm * aspect;
    // tan(vfov/2) by similar triangles, exact; tan(atan(x)) caps or turns negative at small f, where atan rounds to pi/2.
    const float halfHeight = halfHeightMm / focalLengthMm_;
    const float halfWidth = halfHeight * aspect;
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
    switch (basis.lens.projection) {
        case LensProjection::FisheyePolynomial: {
            const std::optional<FisheyeSample> sample = fisheyeSample(basis, ndcX, ndcY);
            return sample ? std::optional(Ray{position_, sample->dir, nearClip_, farClip_}) : std::nullopt;
        }
        case LensProjection::Omnidirectional:
            return Ray{position_, omnidirectionalDirection(basis, ndcX, ndcY), nearClip_, farClip_};
        // No default arm: -Werror then makes an unrouted new projection a compile error. Count is never a lens.
        case LensProjection::Rectilinear:
        case LensProjection::Count:
            break;
    }
    const glm::vec3 dir = glm::normalize(basis.forward + (ndcX * basis.halfWidth * basis.right) +
                                          (ndcY * basis.halfHeight * basis.up));
    return Ray{position_, dir, nearClip_, farClip_};
}

std::optional<Camera::RayDifferential> Camera::primaryRayDifferential(const ViewBasis& basis, float ndcX,
                                                                      float ndcY) const {
    switch (basis.lens.projection) {
        case LensProjection::FisheyePolynomial: {
            const std::optional<FisheyeSample> sample = fisheyeSample(basis, ndcX, ndcY);
            if (!sample) {
                return std::nullopt;
            }
            return RayDifferential{Ray{position_, sample->dir, nearClip_, farClip_}, fisheyeDirPerNdc(basis, *sample)};
        }
        case LensProjection::Omnidirectional:
            return RayDifferential{Ray{position_, omnidirectionalDirection(basis, ndcX, ndcY), nearClip_, farClip_},
                                   omnidirectionalDirPerNdc(basis, ndcX, ndcY)};
        // No default arm: -Werror then makes an unrouted new projection a compile error. Count is never a lens.
        case LensProjection::Rectilinear:
        case LensProjection::Count:
            break;
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
    switch (basis.lens.projection) {
        case LensProjection::FisheyePolynomial:
            return fisheyeProjection(basis, glm::vec3(point) - (point.w * position_));
        case LensProjection::Omnidirectional:
            return omnidirectionalProjection(basis, glm::vec3(point) - (point.w * position_));
        // No default arm: -Werror then makes an unrouted new projection a compile error. Count is never a lens.
        case LensProjection::Rectilinear:
        case LensProjection::Count:
            break;
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
