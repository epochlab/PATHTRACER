#include "pathtracer/scene/light.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>

#include <boost/math/special_functions/ellint_3.hpp>
#include <glm/gtc/constants.hpp>

#include "pathtracer/scene/rotation.h"
#include "shading_math.h"

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

// The cone a sphere subtends at a point, PBR 4e 6.2.4: sin^2(theta_max) = r^2/d^2, and 1 - cos(theta_max) as sin^2/(1 + cos).
struct SphereCone {
    float sin2ThetaMax;
    float oneMinusCosThetaMax;  // the cap's solid angle over 2 pi, cancellation-free so no small-angle branch is needed
};

// nullopt iff the sphere subtends no solid angle at p: p on or inside it, or so far that sin^2(theta_max) underflows.
std::optional<SphereCone> sphereCone(const SphereLight& sphere, const glm::vec3& p) {
    const glm::vec3 toCentre = sphere.centre - p;
    const float sin2ThetaMax = (sphere.radius * sphere.radius) / glm::dot(toCentre, toCentre);
    if (!(sin2ThetaMax < 1.0F) || !(sin2ThetaMax > 0.0F)) {
        return std::nullopt;
    }
    return SphereCone{sin2ThetaMax, sin2ThetaMax / (1.0F + std::sqrt(1.0F - sin2ThetaMax))};
}

// A point on an area light with its geometric normal, before selection and Le.
struct AreaSample {
    glm::vec3 point;
    glm::vec3 normal;
    float inversePdf;  // 1 / solid-angle pdf at the shading point: a constant-pdf sampler's solid angle, or a mesh face's cos A / d^2
};

std::optional<AreaSample> sampleLight(const QuadLight& quad, const glm::vec3& p, glm::vec2 u) {
    const std::optional<SphericalRectangle> rect = buildSphericalRectangle(quad, p);
    if (!rect.has_value()) {
        return std::nullopt;
    }
    return AreaSample{rect->sample(u), quad.normal, rect->solidAngle};
}

std::optional<AreaSample> sampleLight(const DiskLight& disk, const glm::vec3& p, glm::vec2 u) {
    const std::optional<SphericalEllipse> ellipse = buildSphericalEllipse(disk, p);
    if (!ellipse.has_value()) {
        return std::nullopt;
    }
    return AreaSample{ellipse->sample(u), disk.frame[2], ellipse->solidAngle};
}

// PBR 4e 6.2.4: theta uniform in the cone's solid angle about the centre direction, then the near intersection by the law of cosines.
std::optional<AreaSample> sampleLight(const SphereLight& sphere, const glm::vec3& p, glm::vec2 u) {
    const std::optional<SphereCone> cone = sphereCone(sphere, p);
    if (!cone.has_value()) {
        return std::nullopt;
    }
    const float oneMinusCosTheta = u.x * cone->oneMinusCosThetaMax;
    const float cosTheta = 1.0F - oneMinusCosTheta;
    const float sin2Theta = oneMinusCosTheta * (2.0F - oneMinusCosTheta);
    const float cosAlpha = (sin2Theta / std::sqrt(cone->sin2ThetaMax)) +
                           (cosTheta * std::sqrt(std::max(0.0F, 1.0F - (sin2Theta / cone->sin2ThetaMax))));
    const float sinAlpha = std::sqrt(std::max(0.0F, 1.0F - (cosAlpha * cosAlpha)));
    const float phi = glm::two_pi<float>() * u.y;
    // alpha is the angle at the centre from the direction back toward p, so the frame's z is -wc.
    const glm::vec3 towardP = glm::normalize(p - sphere.centre);
    const glm::vec3 normal = glm::transpose(toFrameAbout(towardP)) * glm::vec3(sinAlpha * std::cos(phi), sinAlpha * std::sin(phi), cosAlpha);
    return AreaSample{sphere.centre + (sphere.radius * normal), normal, glm::two_pi<float>() * cone->oneMinusCosThetaMax};
}

// A mesh face's cos A / d^2 at the point toPoint away, cos = |toPoint . n| / d: NEE and the emitter hit share this one expression.
float meshInversePdf(const glm::vec3& toPoint, const glm::vec3& normal, float area) {
    const float distance2 = glm::dot(toPoint, toPoint);
    return std::abs(glm::dot(toPoint, normal)) * area / (distance2 * std::sqrt(distance2));
}

// Area sampling, Heitz 2019's low-distortion triangle map (PBR 4e SampleUniformTriangle) on an area-picked face: pdf_A = 1/area.
std::optional<AreaSample> sampleLight(const MeshLight& mesh, const glm::vec3& p, float uTriangle, glm::vec2 u) {
    const Triangle& tri = mesh.triangles[static_cast<std::size_t>(
        invertCdf(mesh.areaCdf.data(), static_cast<int>(mesh.triangles.size()), uTriangle).index)];
    glm::vec2 b;
    if (u.x < u.y) {
        b.x = 0.5F * u.x;
        b.y = u.y - b.x;
    } else {
        b.y = 0.5F * u.y;
        b.x = u.x - b.y;
    }
    const glm::vec3 point = (b.x * tri.v0) + (b.y * tri.v1) + ((1.0F - b.x - b.y) * tri.v2);
    const glm::vec3 normal = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
    const float inversePdf = meshInversePdf(point - p, normal, mesh.area);
    // Edge-on, the face subtends no solid angle: its density there is infinite and its contribution zero.
    if (!(inversePdf > 0.0F)) {
        return std::nullopt;
    }
    return AreaSample{point, normal, inversePdf};
}

// The radial map's quadrant area Omega_r(phi) (Guillen et al. 2017 eq. 19), phi in [0, pi/2] from the major axis, in double.
double quadrantArea(const SphericalEllipse& ellipse, double phi) {
    // eq. 20's tangent-ellipse angle: atan2 keeps it exact at phi = pi/2, where tan(phi) diverges.
    const double tangentAngle = std::atan2(ellipse.tangentRatio * std::sin(phi), std::cos(phi));
    return phi - (ellipse.piScale * boost::math::ellint_3(ellipse.modulus, ellipse.characteristic, tangentAngle));
}

// h_r(phi) = sqrt(1 - r(phi)^2) (eq. 17), r(phi) = ab / sqrt(a^2 sin^2 + b^2 cos^2) the projected ellipse's radius (eq. 18).
double boundaryAltitude(const SphericalEllipse& ellipse, double phi) {
    const double sinPhi = std::sin(phi);
    const double cosPhi = std::cos(phi);
    const double ab = ellipse.a * ellipse.b;
    const double radius2 = (ab * ab) / ((ellipse.a * ellipse.a * sinPhi * sinPhi) + (ellipse.b * ellipse.b * cosPhi * cosPhi));
    return std::sqrt(1.0 - radius2);
}

// eq. 21's root: Newton on Omega_r, whose derivative 1 - h_r is positive, bisecting whenever a step leaves the bracket.
double solveAzimuth(const SphericalEllipse& ellipse, double target) {
    // A unit direction's components resolve no finer than one float ulp at 1, so a root nearer than that is the root.
    constexpr double kTolerance = std::numeric_limits<float>::epsilon();
    double lo = 0.0;
    double hi = 0.5 * std::numbers::pi;
    // The tangent ellipse's own inverse, its sector area linear in eq. 20's angle: exact for a round cone and as the cone shrinks.
    const double parametric = hi * target / ellipse.quadrantSolidAngle;
    double phi = std::atan2(std::sin(parametric), ellipse.tangentRatio * std::cos(parametric));
    for (;;) {
        const double residual = quadrantArea(ellipse, phi) - target;
        (residual < 0.0 ? lo : hi) = phi;
        double next = phi - (residual / (1.0 - boundaryAltitude(ellipse, phi)));
        if (!(next > lo && next < hi)) {
            next = 0.5 * (lo + hi);
        }
        if (std::abs(next - phi) <= kTolerance) {
            return next;
        }
        phi = next;
    }
}

// The elliptic cone a disk subtends, unordered: its axis and its semi-arcs in and across the symmetry plane (paper Sec. 3, Fig. 2).
struct EllipseCone {
    glm::vec3 axis;     // the bisector of the rim's two generators in the symmetry plane: the spherical ellipse's centre
    glm::vec3 inPlane;  // unit, in the symmetry plane and normal to the axis
    glm::vec3 across;   // unit, normal to the symmetry plane
    double inPlaneArc;
    double acrossArc;
    glm::vec3 planeNormal;  // the disk's normal turned to face away from the reference point
    float planeDistance;
};

// nullopt iff the disk has no radius or the point lies in its plane.
std::optional<EllipseCone> ellipseCone(const DiskLight& disk, const glm::vec3& referencePoint) {
    const glm::vec3& normal = disk.frame[2];
    const glm::vec3 toCentre = disk.centre - referencePoint;
    const float height = glm::dot(toCentre, normal);
    if (!(disk.radius > 0.0F) || height == 0.0F) {
        return std::nullopt;
    }
    // Frame (u, v, w) at the point: w faces the disk, u lies in the symmetry plane of the normal and the centre, v across it.
    const glm::vec3 w = height > 0.0F ? normal : -normal;
    const glm::vec3 offset = toCentre - (height * normal);
    const float offsetLength = glm::length(offset);
    // On the disk's axis every plane through it is a symmetry plane, so any in-plane u serves.
    const glm::vec3 u = offsetLength > 0.0F ? offset / offsetLength : disk.frame[0];
    // The rim's generators in the symmetry plane as (u, w) pairs, in double: the arcs feed elliptic integrals that amplify error.
    const double h = std::abs(static_cast<double>(height));
    const double rho = offsetLength;
    const double r = disk.radius;
    const glm::dvec2 g0 = glm::normalize(glm::dvec2(rho + r, h));
    const glm::dvec2 g1 = glm::normalize(glm::dvec2(rho - r, h));
    const glm::dvec2 axis = glm::normalize(g0 + g1);
    // Across the plane: the chord through the axis's foot on the disk, parallel to v, half-length s at distance h / axis.w.
    const double foot = h * axis.x / axis.y;
    const double halfChord = std::sqrt(std::max(0.0, (r * r) - ((foot - rho) * (foot - rho))));
    return EllipseCone{(static_cast<float>(axis.x) * u) + (static_cast<float>(axis.y) * w),
                       (static_cast<float>(axis.y) * u) - (static_cast<float>(axis.x) * w),
                       glm::cross(w, u),
                       0.5 * std::atan2(std::abs((g0.x * g1.y) - (g0.y * g1.x)), glm::dot(g0, g1)),
                       std::atan2(halfChord, h / axis.y),
                       w,
                       std::abs(height)};
}

// Le toward `direction` off a surface of normal `normal`: a ray toward the light opposes the front face's normal; a sphere has no back.
template <typename Light>
glm::vec3 emittedToward(const Light& light, const glm::vec3& direction, const glm::vec3& normal) {
    if constexpr (!std::is_same_v<Light, SphereLight>) {
        if (light.twoSided) {
            return light.radiance;
        }
    }
    return glm::dot(direction, normal) < 0.0F ? light.radiance : glm::vec3(0.0F);
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

std::optional<SphericalEllipse> buildSphericalEllipse(const DiskLight& disk, const glm::vec3& referencePoint) {
    const std::optional<EllipseCone> cone = ellipseCone(disk, referencePoint);
    if (!cone.has_value()) {
        return std::nullopt;
    }
    const bool inPlaneMajor = cone->inPlaneArc >= cone->acrossArc;
    const double alpha = inPlaneMajor ? cone->inPlaneArc : cone->acrossArc;
    const double beta = inPlaneMajor ? cone->acrossArc : cone->inPlaneArc;
    SphericalEllipse ellipse{};
    ellipse.referencePoint = referencePoint;
    ellipse.x = inPlaneMajor ? cone->inPlane : cone->across;
    ellipse.y = glm::cross(cone->axis, ellipse.x);
    ellipse.z = cone->axis;
    ellipse.planeNormal = cone->planeNormal;
    ellipse.planeDistance = cone->planeDistance;
    ellipse.a = std::sin(alpha);
    ellipse.b = std::sin(beta);
    const double cosAlpha = std::cos(alpha);
    const double cosBeta = std::cos(beta);
    // a^2 - b^2 as sin(alpha - beta) sin(alpha + beta): exact where the ellipse is nearly round and the plain difference cancels.
    const double axisGap = std::sin(alpha - beta) * std::sin(alpha + beta);
    ellipse.tangentRatio = (ellipse.a * cosBeta) / (cosAlpha * ellipse.b);
    ellipse.piScale = (ellipse.b * cosAlpha * cosAlpha) / (ellipse.a * cosBeta);
    ellipse.characteristic = axisGap / (ellipse.a * ellipse.a * cosBeta * cosBeta);
    ellipse.modulus = std::sqrt(axisGap) / cosBeta;
    // n reaches 1, where Pi diverges, only as beta does: the disk seen edge-on, zero solid angle.
    if (!(ellipse.b > 0.0) || !(ellipse.characteristic < 1.0)) {
        return std::nullopt;
    }
    ellipse.quadrantSolidAngle =
        (0.5 * std::numbers::pi) - (ellipse.piScale * boost::math::ellint_3(ellipse.modulus, ellipse.characteristic));
    ellipse.solidAngle = static_cast<float>(4.0 * ellipse.quadrantSolidAngle);
    if (!(ellipse.solidAngle > 0.0F) || !std::isfinite(ellipse.solidAngle)) {
        return std::nullopt;
    }
    return ellipse;
}

glm::vec3 SphericalEllipse::sample(glm::vec2 u) const {
    // The four quadrants mirror one another (paper Sec. 3.3): u.x picks one, reversed in odd quadrants so the map stays continuous.
    const float scaled = 4.0F * u.x;
    const int quadrant = std::min(3, static_cast<int>(scaled));
    const double within = static_cast<double>(scaled) - quadrant;
    const double phi = solveAzimuth(*this, (quadrant % 2 == 0 ? within : 1.0 - within) * quadrantSolidAngle);
    // eq. 22: altitude uniform on the cylinder between the boundary and the pole, which Archimedes' hat-box makes uniform in area.
    const double altitude = ((1.0 - u.y) * boundaryAltitude(*this, phi)) + u.y;
    const double ring = std::sqrt((1.0 - altitude) * (1.0 + altitude));
    const float signX = quadrant == 1 || quadrant == 2 ? -1.0F : 1.0F;
    const float signY = quadrant >= 2 ? -1.0F : 1.0F;
    const glm::vec3 direction = (signX * static_cast<float>(ring * std::cos(phi)) * x) + (signY * static_cast<float>(ring * std::sin(phi)) * y) +
                                (static_cast<float>(altitude) * z);
    // Back-projected onto the disk's plane along the sampled direction, which lies inside the cone and so faces the plane.
    return referencePoint + ((planeDistance / glm::dot(direction, planeNormal)) * direction);
}

MeshLight::MeshLight(std::vector<Triangle> triangles, glm::vec3 radiance, bool twoSided)
    : triangles(std::move(triangles)), areaCdf(this->triangles.size() + 1, 0.0F), area(0.0F), radiance(radiance), twoSided(twoSided) {
    // Summed in double so the running total does not swamp a small face's area in a large mesh.
    std::vector<double> running(this->triangles.size() + 1, 0.0);
    for (std::size_t i = 0; i < this->triangles.size(); ++i) {
        const Triangle& tri = this->triangles[i];
        running[i + 1] = running[i] + (0.5 * glm::length(glm::dvec3(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0))));
    }
    const double total = running.back();
    for (std::size_t i = 1; i < running.size(); ++i) {
        areaCdf[i] = static_cast<float>(running[i] / total);
    }
    area = static_cast<float>(total);
}

LightSet::LightSet(const EnvironmentMap* environment, const glm::vec3& envRotationDegrees, float envExposure,
                    const std::vector<AreaLight>& lights)
    : environment_(environment),
      envRotation_(rotationXyz(envRotationDegrees)),
      envExposure_(envExposure),
      lights_(lights),
      drawsTriangle_(std::ranges::any_of(lights, [](const AreaLight& light) { return std::holds_alternative<MeshLight>(light); })) {}

int LightSet::count() const {
    return (environment_ != nullptr ? 1 : 0) + static_cast<int>(lights_.size());
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

glm::vec3 LightSet::radianceToward(int lightIndex, const glm::vec3& direction, const glm::vec3& normal) const {
    return std::visit([&](const auto& light) { return emittedToward(light, direction, normal); }, lights_[static_cast<std::size_t>(lightIndex)]);
}

float LightSet::pdfArea(int lightIndex, const glm::vec3& p, const glm::vec3& point, const glm::vec3& normal) const {
    // The hit is presumed, so each shape's density at p is its sampler's: constant over the solid angle, or area-converted for a mesh.
    const float inversePdf = std::visit(
        [&](const auto& light) {
            using Light = std::decay_t<decltype(light)>;
            if constexpr (std::is_same_v<Light, QuadLight>) {
                const std::optional<SphericalRectangle> rect = buildSphericalRectangle(light, p);
                return rect.has_value() ? rect->solidAngle : 0.0F;
            } else if constexpr (std::is_same_v<Light, DiskLight>) {
                const std::optional<SphericalEllipse> ellipse = buildSphericalEllipse(light, p);
                return ellipse.has_value() ? ellipse->solidAngle : 0.0F;
            } else if constexpr (std::is_same_v<Light, SphereLight>) {
                const std::optional<SphereCone> cone = sphereCone(light, p);
                return cone.has_value() ? glm::two_pi<float>() * cone->oneMinusCosThetaMax : 0.0F;
            } else {
                return meshInversePdf(point - p, normal, light.area);
            }
        },
        lights_[static_cast<std::size_t>(lightIndex)]);
    return inversePdf > 0.0F ? 1.0F / (inversePdf * static_cast<float>(count())) : 0.0F;
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
    // Drawn before the pick is resolved: a dimension only some lights consumed would shift every later one by the pick.
    const float uTriangle = drawsTriangle_ ? sampler.next1D() : 0.0F;

    if (envPresent && index == 0) {
        const EnvironmentMap::EnvSample envSample =
            environment_->importanceSampleDirection(sampler.next2D(), envRotation_);
        // The miss path's lookup, not the nearest texel: MIS weights sum to 1 across strategies, so both must evaluate one Le.
        return LightSample{envSample.direction, environmentRadiance(envSample.direction),
                            envSample.pdf * selectionPdf, std::numeric_limits<float>::max()};
    }

    // Drawn before the degeneracy check: returning early without consuming this 2D would shift every later dimension on that path.
    const glm::vec2 u = sampler.next2D();
    // One dispatch for the point, its pdf and its Le: NEE runs once per vertex, so a second indirect call would show in a pass.
    return std::visit(
        [&](const auto& light) -> std::optional<LightSample> {
            std::optional<AreaSample> area;
            if constexpr (std::is_same_v<std::decay_t<decltype(light)>, MeshLight>) {
                area = sampleLight(light, p, uTriangle, u);
            } else {
                area = sampleLight(light, p, u);
            }
            if (!area.has_value()) {
                return std::nullopt;
            }
            const glm::vec3 toLight = area->point - p;
            const float distance = glm::length(toLight);
            if (!(distance > 0.0F)) {
                return std::nullopt;
            }
            const glm::vec3 direction = toLight / distance;
            return LightSample{direction, emittedToward(light, direction, area->normal), selectionPdf / area->inversePdf, distance};
        },
        lights_[static_cast<std::size_t>(index - (envPresent ? 1 : 0))]);
}

}  // namespace pathtracer::scene
