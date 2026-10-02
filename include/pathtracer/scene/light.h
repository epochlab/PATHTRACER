#pragma once

#include <optional>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/scene/environment_map.h"
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
    glm::vec3 radiance;  // constant Le over the emitting face, colour * intensity
    bool twoSided;
    // Derived from edge0/edge1 once here, not per query: quadRadianceToward runs on every NEE sample and every emitter hit.
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

struct LightSample {
    glm::vec3 direction;  // unit, from the shading point toward the light
    glm::vec3 radiance;   // Le arriving from that direction (0 if the light doesn't emit that way)
    float pdf;            // solid-angle density of `direction`, INCLUDING the light-selection probability; > 0
    float distance;       // Euclidean distance to the sampled point; FLT_MAX for the environment
};

// The lights NEE can sample in one pass: the environment (or none, per the HUD toggle) plus zero or more quads. Selection is uniform.
class LightSet {
public:
    // environment == nullptr excludes it entirely: no NEE, MIS or miss radiance. The quads vector must outlive this LightSet.
    LightSet(const EnvironmentMap* environment, float envRotationRadians, float envExposure,
             const std::vector<QuadLight>& quads);

    [[nodiscard]] int count() const;

    // nullopt iff count() == 0, or the one light selected (a quad) has zero solid angle at p.
    [[nodiscard]] std::optional<LightSample> sample(const glm::vec3& p, Sampler& sampler) const;

    // MIS pdf of a BSDF-sampled ray having reached the environment in `dir` -- 0 if the environment is excluded from the set.
    [[nodiscard]] float pdfEnvironment(const glm::vec3& dir) const;

    // MIS pdf of a BSDF-sampled ray from `p` that hit quad `quadIndex`; the hit is presumed, so the constant solid-angle pdf applies.
    [[nodiscard]] float pdfQuad(int quadIndex, const glm::vec3& p) const;

    // Le toward `direction` (unit, from the light toward the viewer) -- 0 on the non-emitting back face unless twoSided.
    [[nodiscard]] glm::vec3 quadRadianceToward(int quadIndex, const glm::vec3& direction) const;

    // Radiance from the environment toward `direction`, 0 if excluded. The one Le in the renderer: NEE and the miss path share it.
    [[nodiscard]] glm::vec3 environmentRadiance(const glm::vec3& direction) const;

private:
    const EnvironmentMap* environment_;
    glm::mat3 envRotation_;  // map to world about +Y, built once per LightSet, i.e. once per pass, rather than per environment query
    float envExposure_;
    const std::vector<QuadLight>& quads_;
};

}  // namespace pathtracer::scene
