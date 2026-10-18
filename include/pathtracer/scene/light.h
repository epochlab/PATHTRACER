#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/scene/environment_map.h"
#include "pathtracer/scene/ray_types.h"
#include "pathtracer/scene/sampler.h"

namespace pathtracer::scene {

// Rectangular emitter, perpendicular edges from origin, emitting along normalize(cross(edge0, edge1)). Back face dark unless twoSided.
struct QuadLight {
    QuadLight(glm::vec3 origin, glm::vec3 edge0, glm::vec3 edge1, glm::vec3 radiance, bool twoSided = false)
        : origin(origin), edge0(edge0), edge1(edge1), radiance(radiance), twoSided(twoSided),
          normal(glm::normalize(glm::cross(edge0, edge1))) {}

    glm::vec3 origin;
    glm::vec3 edge0;
    glm::vec3 edge1;
    glm::vec3 radiance;  // constant Le over the emitting face, colour * luminance
    bool twoSided;
    // Derived from edge0/edge1 once here, not per query: every NEE sample of the quad reads it.
    glm::vec3 normal;
};

// Urena, Fajardo & King (EGSR 2013), as given in PBRT 4th ed. 12.5.3: an exact constant-pdf solid-angle sampler for a rectangle.
struct SphericalRectangle {
    glm::vec3 referencePoint;
    glm::vec3 x, y, z;  // local orthonormal frame, z chosen so the reference point has z0 < 0
    float z0;
    float x0, x1, y0, y1;
    float b0, b1;
    float k;
    float solidAngle;  // steradians subtended by the rectangle at referencePoint; pdf = 1/solidAngle

    // u in [0,1)^2 -> a world-space point on the rectangle, uniform in solid angle as seen from referencePoint.
    [[nodiscard]] glm::vec3 sample(glm::vec2 u) const;
};

// nullopt iff the rectangle subtends no solid angle at referencePoint: degenerate edges, or referencePoint in the rectangle's plane.
[[nodiscard]] std::optional<SphericalRectangle> buildSphericalRectangle(const QuadLight& quad,
                                                                         const glm::vec3& referencePoint);

// Circular emitter of `radius` about centre in the plane of frame[0], frame[1], emitting along frame[2]; back dark unless twoSided.
struct DiskLight {
    glm::vec3 centre;
    glm::mat3 frame;  // orthonormal columns tangent, bitangent, emitting normal: the placed local +X, -Y and -Z, as a quad's edges
    float radius;
    glm::vec3 radiance;  // constant Le over the emitting face, colour * luminance
    bool twoSided;
};

// Spherical emitter, Le leaving every point outward: closed, so a point inside sees only its dark inner face and is never lit by it.
struct SphereLight {
    glm::vec3 centre;
    glm::mat3 frame;  // the placed local axes, orienting only the surface's uv: the emission itself has no direction to turn
    float radius;
    glm::vec3 radiance;
};

// A triangle soup emitting along each face's cross(v1 - v0, v2 - v0), back faces dark unless twoSided; NEE samples it by area.
struct MeshLight {
    MeshLight(std::vector<Triangle> triangles, glm::vec3 radiance, bool twoSided);

    std::vector<Triangle> triangles;
    std::vector<float> areaCdf;  // triangles.size() + 1 knots from 0 to 1: the area-proportional, so power-proportional, pick
    float area;                  // sum over triangles; > 0 is the caller's to ensure
    glm::vec3 radiance;
    bool twoSided;
};

// Every light shape with a surface, so every one is hittable; point, spot and distant lights have none and are absent.
using AreaLight = std::variant<QuadLight, DiskLight, SphereLight, MeshLight>;

// Guillen et al. (EGSR 2017) radial map: an exact constant-pdf solid-angle sampler for a disk, the cone it subtends being elliptic.
struct SphericalEllipse {
    glm::vec3 referencePoint;
    glm::vec3 x, y, z;  // z the cone's axis (the ellipse's centre), x along the larger semi-arc alpha, y along beta
    glm::vec3 planeNormal;  // unit disk normal facing away from referencePoint, so the disk lies at planeDistance along it
    float planeDistance;
    double a, b;          // sin(alpha) >= sin(beta), the semi-axes of the ellipse projected along z (paper Fig. 5)
    double tangentRatio;  // a_t / b_t = tan(alpha) / tan(beta)
    double piScale;       // b (1 - a^2) / (a sqrt(1 - b^2)), eq. 19's factor on Pi
    double characteristic;  // n = (a^2 - b^2) / (a^2 (1 - b^2)), eq. 20
    double modulus;         // k = sqrt(m), m = (a^2 - b^2) / (1 - b^2), eq. 20
    double quadrantSolidAngle;  // Omega_r(pi/2)
    float solidAngle;           // 4 Omega_r(pi/2) steradians; pdf = 1/solidAngle

    // u in [0,1)^2 -> a world-space point on the disk, uniform in solid angle as seen from referencePoint.
    [[nodiscard]] glm::vec3 sample(glm::vec2 u) const;
};

// nullopt iff the disk subtends no solid angle at referencePoint: a zero radius, or referencePoint in the disk's plane.
[[nodiscard]] std::optional<SphericalEllipse> buildSphericalEllipse(const DiskLight& disk, const glm::vec3& referencePoint);

struct LightSample {
    glm::vec3 direction;  // unit, from the shading point toward the light
    glm::vec3 radiance;   // Le arriving from that direction (0 if the light doesn't emit that way)
    float pdf;            // solid-angle density of `direction`, INCLUDING the light-selection probability; > 0
    float distance;       // Euclidean distance to the sampled point; FLT_MAX for the environment
};

// The lights NEE can sample in one pass: the environment (or none, per the HUD toggle) plus zero or more area lights, picked uniformly.
class LightSet {
public:
    // environment == nullptr excludes it entirely. envRotationDegrees turns the map to world by rotationXyz. lights must outlive this.
    LightSet(const EnvironmentMap* environment, const glm::vec3& envRotationDegrees, float envExposure,
             const std::vector<AreaLight>& lights);

    [[nodiscard]] int count() const;

    // nullopt iff count() == 0, or the area light selected subtends no solid angle at p or meets the sample edge-on.
    [[nodiscard]] std::optional<LightSample> sample(const glm::vec3& p, Sampler& sampler) const;

    // MIS pdf of a BSDF-sampled ray having reached the environment in `dir` -- 0 if the environment is excluded from the set.
    [[nodiscard]] float pdfEnvironment(const glm::vec3& dir) const;

    // MIS pdf of a BSDF-sampled ray from p that hit area light `lightIndex` at `point`, whose geometric normal there is `normal`.
    [[nodiscard]] float pdfArea(int lightIndex, const glm::vec3& p, const glm::vec3& point, const glm::vec3& normal) const;

    // Le toward `direction` (from the light toward the viewer) off a surface of geometric normal `normal`: 0 on a dark back face.
    [[nodiscard]] glm::vec3 radianceToward(int lightIndex, const glm::vec3& direction, const glm::vec3& normal) const;

    // Radiance from the environment toward `direction`, 0 if excluded. The one Le in the renderer: NEE and the miss path share it.
    [[nodiscard]] glm::vec3 environmentRadiance(const glm::vec3& direction) const;

    // The same Le filtered over a camera ray's footprint, dirFootprint's columns the direction's offsets along its two axes.
    [[nodiscard]] glm::vec3 environmentRadiance(const glm::vec3& direction, const glm::mat2x3& dirFootprint) const;

private:
    const EnvironmentMap* environment_;
    glm::mat3 envRotation_;  // map to world, built once per LightSet, i.e. once per pass, rather than per environment query
    float envExposure_;
    const std::vector<AreaLight>& lights_;
    // Any mesh light draws a triangle dimension whichever light is picked, so the budget per vertex is the scene's, never the pick's.
    bool drawsTriangle_;
};

}  // namespace pathtracer::scene
