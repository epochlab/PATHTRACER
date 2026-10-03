#pragma once

#include <optional>
#include <string>

#include <glm/glm.hpp>

#include "pathtracer/scene/lens.h"
#include "pathtracer/scene/ray_types.h"

namespace pathtracer::scene {

// Pose, lens and exposure, immutable once constructed. Right-handed, +Y up, -Z forward; yaw/pitch Euler, pitch bounded by validate().
class Camera {
public:
    // Sensor gate size in mm ({36.0F, 24.0F} for 35mm full-frame), paired with focal length to derive vertical FOV.
    struct FilmBack {
        float widthMm;
        float heightMm;
    };

    // A named, real-world FilmBack ("ARRI Alexa 65"): the HUD preset dropdown and assets/config/sensor.json key off name.
    struct FilmBackPreset {
        std::string name;
        FilmBack filmBack;
    };

    // Authored in degrees, more ergonomic at call sites, stored as radians because every consumer is trigonometric.
    Camera(const glm::vec3& position, float yawDegrees, float pitchDegrees, FilmBack filmBack,
           float focalLengthMm, float nearClip, float farClip, float aperture,
           float shutterSeconds, float iso, Lens lens = Lens{});

    // Every invariant the projections and exposure divide by or take trig of; false names the first violation. Run at each boundary.
    [[nodiscard]] bool validate(std::string& error) const;

    // Finite and positive in both dimensions, a FOV and aspect divisor; shared with sensor.json, whose presets the HUD swaps in.
    [[nodiscard]] static bool validFilmBack(FilmBack filmBack);

    [[nodiscard]] glm::vec3 position() const { return position_; }

    // Orientation in the degrees it was authored in, completing the accessors that return every constructor argument as given.
    [[nodiscard]] float yawDegrees() const { return glm::degrees(yawRadians_); }
    [[nodiscard]] float pitchDegrees() const { return glm::degrees(pitchRadians_); }

    // Unit-length view direction derived from yaw/pitch.
    [[nodiscard]] glm::vec3 forward() const;
    [[nodiscard]] FilmBack filmBack() const { return filmBack_; }
    [[nodiscard]] float focalLengthMm() const { return focalLengthMm_; }
    [[nodiscard]] float nearClip() const { return nearClip_; }
    [[nodiscard]] float farClip() const { return farClip_; }
    [[nodiscard]] float aperture() const { return aperture_; }
    [[nodiscard]] float shutterSeconds() const { return shutterSeconds_; }
    [[nodiscard]] float iso() const { return iso_; }
    [[nodiscard]] Lens lens() const { return lens_; }

    // Paraxial vertical FOV from focal length and film-back height, which viewBasis's half-extents are; other lenses' is below.
    [[nodiscard]] float verticalFovRadians() const;

    // Angle the frame's vertical extent subtends under the active projection: the fisheye saturates at its circle, the lat-long is pi.
    [[nodiscard]] float verticalAngularExtentRadians() const;

    // Orthonormal basis and view-plane half-extents, all primaryRay() needs bar the ndc weight, so a projector can share it.
    struct ViewBasis {
        glm::vec3 forward;
        glm::vec3 right;
        glm::vec3 up;
        // View-plane half-extents at unit depth, tangent-valued: the pinhole arm's ndc weights, which the other lenses leave unused.
        float halfWidth;
        float halfHeight;
        // Sensor half-extents in mm, the image circle and the model: everything the fisheye arm needs without reaching back to the Camera.
        float halfWidthMm;
        float halfHeightMm;
        float maxThetaRadians;
        float maxRadiusMm;
        float focalLengthMm;
        Lens lens;
    };
    [[nodiscard]] ViewBasis viewBasis(float aspect) const;

    // Ray for ndc in [-1,1] (+Y up), tMin/tMax nearClip()/farClip(); nullopt outside a fisheye's circle. The lat-long is 2-periodic in x.
    [[nodiscard]] std::optional<Ray> primaryRay(float ndcX, float ndcY, float aspect) const;

    // Same ray from a basis the caller already built; the aspect-taking overload rebuilds two sin, two cos, an atan and a tan every call.
    [[nodiscard]] std::optional<Ray> primaryRay(const ViewBasis& basis, float ndcX, float ndcY) const;

    // A primary ray and its exact differential (Igehy 1999): columns d(dir)/d(ndcX), d(dir)/d(ndcY) of the unit direction.
    struct RayDifferential {
        Ray ray;
        glm::mat2x3 dirPerNdc;
    };

    // primaryRay plus its closed-form differential, one inverse solve; nullopt where primaryRay is. Lat-long: rank 1 only at ndcY = +/-1.
    [[nodiscard]] std::optional<RayDifferential> primaryRayDifferential(const ViewBasis& basis, float ndcX, float ndcY) const;

    // Pinhole camera matrix P = K [R | -R c] (Hartley & Zisserman 2004, eq. 6.8) as NDC rows: ndc = (x . X, y . X) / (depth . X).
    struct PinholeMatrix {
        glm::vec4 x;
        glm::vec4 y;
        glm::vec4 depth;
    };
    [[nodiscard]] PinholeMatrix pinholeMatrix(const ViewBasis& basis) const;

    // primaryRay's inverse on (p, w), w = 0 at infinity; nullopt behind the pinhole, past a fisheye's thetaMax, or at the lat-long's eye.
    [[nodiscard]] std::optional<glm::vec2> project(const ViewBasis& basis, const glm::vec4& point) const;

    // Pinhole arm, inline for per-pixel use. Divides, not reciprocals: x * (1/z) - x' * (1/z') would contract to an FMA, losing exact 0.
    [[nodiscard]] static std::optional<glm::vec2> project(const PinholeMatrix& matrix, const glm::vec4& point) {
        const float depth = glm::dot(matrix.depth, point);
        // The pinhole images only the half-space ahead; depth <= 0 has no finite image-plane point.
        if (!(depth > 0.0F)) {
            return std::nullopt;
        }
        return glm::vec2(glm::dot(matrix.x, point) / depth, glm::dot(matrix.y, point) / depth);
    }

    // Exposure value at ISO 100 (log2): log2(aperture^2 / shutterSeconds * (100/iso)). The static overload is the formula's one definition.
    [[nodiscard]] float ev100() const;
    [[nodiscard]] static float ev100(float aperture, float shutterSeconds, float iso);

private:
    glm::vec3 position_;
    float yawRadians_;
    float pitchRadians_;
    FilmBack filmBack_;
    float focalLengthMm_;
    float nearClip_;
    float farClip_;
    float aperture_;
    float shutterSeconds_;
    float iso_;
    Lens lens_;
};

}  // namespace pathtracer::scene
