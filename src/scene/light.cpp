#include "pathtracer/scene/light.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include <glm/gtc/constants.hpp>

#include "pathtracer/scene/rotation.h"

namespace pathtracer::scene {

namespace {

// atan2(|a x b|, a.b) -- PBRT v4's AngleBetween: robust near 0 and pi, where acos(dot) loses precision as gamma_i shrinks.
float angleBetween(const glm::vec3& a, const glm::vec3& b) {
    return std::atan2(glm::length(glm::cross(a, b)), glm::dot(a, b));
}

// Urena et al. 2013's local frame at referencePoint: unit edges x, y, normal z with z0 < 0, and the rectangle's extents in it.
std::optional<SphericalRectangle> sphericalRectangleFrame(const QuadLight& quad, const glm::vec3& referencePoint) {
    const float exl = glm::length(quad.edge0);
    const float eyl = glm::length(quad.edge1);
    if (!(exl > 0.0F) || !(eyl > 0.0F)) {
        return std::nullopt;
    }
    const glm::vec3 x = quad.edge0 / exl;
    const glm::vec3 y = quad.edge1 / eyl;
    glm::vec3 z = glm::cross(x, y);
    const float zLen = glm::length(z);
    if (!(zLen > 0.0F)) {
        return std::nullopt;  // edge0 parallel to edge1 -- degenerate quad
    }
    z /= zLen;

    // Rectangle-corner-to-referencePoint vector in world space; the local frame's origin is referencePoint (Urena's "o").
    const glm::vec3 d = quad.origin - referencePoint;
    float z0 = glm::dot(d, z);
    glm::vec3 zFrame = z;
    if (z0 > 0.0F) {
        // Keep z0 negative whichever way the quad's normal points: a numerical convention of the parametrization, not a front-face test.
        zFrame = -z;
        z0 = -z0;
    }
    if (!(z0 < 0.0F)) {
        return std::nullopt;  // referencePoint lies in the rectangle's own plane -- zero measure
    }

    const float x0 = glm::dot(d, x);
    const float y0 = glm::dot(d, y);
    return SphericalRectangle{referencePoint, x, y, zFrame, z0, x0, x0 + exl, y0, y0 + eyl, 0.0F, 0.0F, 0.0F, 0.0F};
}

// The spherical quadrilateral's edge-plane normals and interior angles, giving sample()'s b0, b1, k and the solid angle.
void sphericalRectangleAngles(SphericalRectangle& rect) {
    // Vectors from referencePoint to the four corners, in the local (x, y, zFrame) frame.
    const glm::vec3 v00(rect.x0, rect.y0, rect.z0);
    const glm::vec3 v01(rect.x0, rect.y1, rect.z0);
    const glm::vec3 v10(rect.x1, rect.y0, rect.z0);
    const glm::vec3 v11(rect.x1, rect.y1, rect.z0);
    const glm::vec3 n0 = glm::normalize(glm::cross(v00, v10));
    const glm::vec3 n1 = glm::normalize(glm::cross(v10, v11));
    const glm::vec3 n2 = glm::normalize(glm::cross(v11, v01));
    const glm::vec3 n3 = glm::normalize(glm::cross(v01, v00));

    // Internal angles of the spherical quadrilateral at each of its four vertices (gamma_i, paper notation).
    const float g0 = angleBetween(-n0, n1);
    const float g1 = angleBetween(-n1, n2);
    const float g2 = angleBetween(-n2, n3);
    const float g3 = angleBetween(-n3, n0);

    rect.b0 = n0.z;
    rect.b1 = n2.z;
    rect.k = (2.0F * glm::pi<float>()) - g2 - g3;
    // Girard's theorem: a spherical polygon's solid angle is its interior-angle sum minus (N-2)*pi.
    rect.solidAngle = g0 + g1 - rect.k;
}

}  // namespace

std::optional<SphericalRectangle> buildSphericalRectangle(const QuadLight& quad,
                                                            const glm::vec3& referencePoint) {
    std::optional<SphericalRectangle> rect = sphericalRectangleFrame(quad, referencePoint);
    if (!rect) {
        return std::nullopt;
    }
    sphericalRectangleAngles(*rect);
    if (!(rect->solidAngle > 0.0F) || !std::isfinite(rect->solidAngle)) {
        return std::nullopt;
    }
    return rect;
}

glm::vec3 SphericalRectangle::sample(glm::vec2 u) const {
    // Urena/Fajardo/King 2013's closed-form CDF inversion; au/fu/cu/xu/hv/yv keep the paper's own names, not renamed here.
    const float au = (u.x * solidAngle) + k;
    const float fu = ((std::cos(au) * b0) - b1) / std::sin(au);
    float cu = std::copysign(1.0F / std::sqrt((fu * fu) + (b0 * b0)), fu);
    cu = glm::clamp(cu, -1.0F, 1.0F);

    const float xu = glm::clamp(-(cu * z0) / std::sqrt(std::max(1.0F - (cu * cu), 0.0F)), x0, x1);
    const float d = std::sqrt((xu * xu) + (z0 * z0));
    const float h0 = y0 / std::sqrt((d * d) + (y0 * y0));
    const float h1 = y1 / std::sqrt((d * d) + (y1 * y1));
    const float hv = h0 + (u.y * (h1 - h0));
    const float hv2 = hv * hv;
    const float yv = hv2 < (1.0F - 1e-6F) ? (hv * d) / std::sqrt(1.0F - hv2) : y1;

    return referencePoint + (xu * x) + (yv * y) + (z0 * z);
}

LightSet::LightSet(const EnvironmentMap* environment, const glm::vec3& envRotationDegrees, float envExposure,
                    const std::vector<QuadLight>& quads)
    : environment_(environment),
      envRotation_(rotationXyz(envRotationDegrees)),
      envExposure_(envExposure),
      quads_(quads) {}

int LightSet::count() const {
    return (environment_ != nullptr ? 1 : 0) + static_cast<int>(quads_.size());
}

glm::vec3 LightSet::environmentRadiance(const glm::vec3& direction) const {
    if (environment_ == nullptr) {
        return glm::vec3(0.0F);
    }
    return environment_->sampleDirection(direction, envRotation_) * envExposure_;
}

glm::vec3 LightSet::environmentRadiance(const glm::vec3& direction, const glm::mat2x3& dirFootprint) const {
    if (environment_ == nullptr) {
        return glm::vec3(0.0F);
    }
    return environment_->sampleDirection(direction, dirFootprint, envRotation_) * envExposure_;
}

float LightSet::pdfEnvironment(const glm::vec3& dir) const {
    if (environment_ == nullptr) {
        return 0.0F;
    }
    return environment_->pdf(dir, envRotation_) / static_cast<float>(count());
}

glm::vec3 LightSet::quadRadianceToward(int quadIndex, const glm::vec3& direction) const {
    const QuadLight& quad = quads_[static_cast<std::size_t>(quadIndex)];
    // Front face: a ray travelling toward the light (direction) opposes its outward normal.
    if (glm::dot(direction, quad.normal) < 0.0F || quad.twoSided) {
        return quad.radiance;
    }
    return glm::vec3(0.0F);
}

float LightSet::pdfQuad(int quadIndex, const glm::vec3& p) const {
    const std::optional<SphericalRectangle> rect =
        buildSphericalRectangle(quads_[static_cast<std::size_t>(quadIndex)], p);
    if (!rect.has_value()) {
        return 0.0F;
    }
    return 1.0F / (rect->solidAngle * static_cast<float>(count()));
}

std::optional<LightSample> LightSet::sample(const glm::vec3& p, Sampler& sampler) const {
    const int n = count();
    if (n == 0) {
        return std::nullopt;
    }
    const bool envPresent = environment_ != nullptr;
    int index = 0;
    if (n > 1) {
        const float u = sampler.next1D();
        index = std::min(n - 1, static_cast<int>(u * static_cast<float>(n)));
    }
    const float selectionPdf = 1.0F / static_cast<float>(n);

    if (envPresent && index == 0) {
        const EnvironmentMap::EnvSample envSample =
            environment_->importanceSampleDirection(sampler.next2D(), envRotation_);
        // The miss path's lookup, not the nearest texel: MIS weights sum to 1 across strategies, so both must evaluate one Le.
        return LightSample{envSample.direction, environmentRadiance(envSample.direction),
                            envSample.pdf * selectionPdf, std::numeric_limits<float>::max()};
    }

    const int quadIndex = index - (envPresent ? 1 : 0);
    const QuadLight& quad = quads_[static_cast<std::size_t>(quadIndex)];
    // Drawn before the degeneracy check: returning early without consuming this 2D would shift every later dimension on that path.
    const glm::vec2 u = sampler.next2D();
    const std::optional<SphericalRectangle> rect = buildSphericalRectangle(quad, p);
    if (!rect.has_value()) {
        return std::nullopt;
    }
    const glm::vec3 pointOnQuad = rect->sample(u);
    const glm::vec3 toLight = pointOnQuad - p;
    const float distance = glm::length(toLight);
    if (!(distance > 0.0F)) {
        return std::nullopt;
    }
    const glm::vec3 direction = toLight / distance;
    return LightSample{direction, quadRadianceToward(quadIndex, direction),
                        selectionPdf / rect->solidAngle, distance};
}

}  // namespace pathtracer::scene
