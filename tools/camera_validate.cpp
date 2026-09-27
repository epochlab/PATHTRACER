// Correctness gate for the camera's projections: the spherical pinhole's closed form and the Kannala-Brandt polynomial fisheye.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "check.h"
#include "stats.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/lens.h"

namespace {

using pathtracer::scene::Camera;
using pathtracer::scene::kannalaBrandtIsInvertible;
using pathtracer::scene::kannalaBrandtRadius;
using pathtracer::scene::kannalaBrandtRadiusSlope;
using pathtracer::scene::kannalaBrandtTheta;
using pathtracer::scene::Lens;
using pathtracer::scene::LensProjection;
using pathtracer::scene::maxThetaRadians;
using pathtracer::scene::Ray;

constexpr float kEps = std::numeric_limits<float>::epsilon();
constexpr float kAspect = 16.0F / 9.0F;
constexpr Camera::FilmBack kFullFrame{36.0F, 24.0F};

// Normalize plus a tan and an atan is about a dozen rounded operations, each at most half an ulp: a 16-ulp relative budget bounds them all.
constexpr float kClosedFormUlps = 16.0F;

// theta_d is six rounded Horner operations, so its float value carries at most this many ulps before the inverse maps it to an angle.
constexpr float kRadiusEvalUlps = 6.0F;

// Bracket width kannalaBrandtTheta stops at, in ulps of the answer; the inverse cannot be asserted tighter than its own stopping rule.
constexpr float kBracketUlps = 4.0F;

// Degree-9 Taylor of equisolid angle 2*sin(theta/2) = theta - theta^3/24 + theta^5/1920 - theta^7/322560 + theta^9/92897280.
constexpr std::array<float, 4> kEquisolidTaylor{-1.0F / 24.0F, 1.0F / 1920.0F, -1.0F / 322560.0F,
                                                 1.0F / 92897280.0F};
// 2*tan(theta/2) = theta + theta^3/12 + theta^5/120 + 17*theta^7/20160 + 31*theta^9/362880 (stereographic).
constexpr std::array<float, 4> kStereographicTaylor{1.0F / 12.0F, 1.0F / 120.0F, 17.0F / 20160.0F,
                                                     31.0F / 362880.0F};
// sin(theta) = theta - theta^3/6 + theta^5/120 - theta^7/5040 + theta^9/362880 (orthographic), monotone only below pi/2.
constexpr std::array<float, 4> kOrthographicTaylor{-1.0F / 6.0F, 1.0F / 120.0F, -1.0F / 5040.0F,
                                                    1.0F / 362880.0F};

Camera makeCamera(float focalLengthMm, Lens lens) {
    return Camera{glm::vec3(0.0F, 0.0F, 0.0F), 0.0F,  0.0F,  kFullFrame, focalLengthMm,
                  0.01F,                       100.0F, 2.8F, 0.008F,      400.0F,
                  lens};
}

Lens fisheye(const std::array<float, 4>& coefficients, float fieldOfViewDegrees) {
    return Lens{LensProjection::FisheyePolynomial, coefficients, fieldOfViewDegrees};
}

// Forward-error budget on the inverse: the bracket it stops at, plus theta_d's own evaluation error divided through by the local slope.
float thetaTolerance(const std::array<float, 4>& k, float theta, float thetaMax) {
    const float radius = kannalaBrandtRadius(k, theta);
    const float slope = kannalaBrandtRadiusSlope(k, theta);
    return (kBracketUlps * kEps * thetaMax) + ((kRadiusEvalUlps * kEps * std::max(radius, 1.0F)) / slope);
}

// atan2 of the perpendicular against the axial component: acos(dot) loses half its digits near zero, where these angles live.
float angleBetween(const glm::vec3& a, const glm::vec3& b) {
    return std::atan2(glm::length(glm::cross(a, b)), glm::dot(a, b));
}

// Independent of the library: bisection on theta_d alone, which is a Horner evaluation and cannot share the inverse's logic.
float bisectTheta(const std::array<float, 4>& k, float radius, float thetaMax) {
    float lo = 0.0F;
    float hi = thetaMax;
    for (int i = 0; i < 80; ++i) {
        const float mid = 0.5F * (lo + hi);
        if (kannalaBrandtRadius(k, mid) < radius) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5F * (lo + hi);
}

}  // namespace

// The projection's closed form, from a basis built by glm rotations rather than the camera's own Euler formula.
PT_CHECK(spherical_primary_rays_match_the_closed_form, Fast, Exact) {
    ctx.plan(5);
    constexpr float kFocalMm = 35.0F;
    const Camera camera = makeCamera(kFocalMm, Lens{});
    const std::optional<Ray> centre = camera.primaryRay(0.0F, 0.0F, kAspect);
    PT_EXPECT(ctx, centre.has_value() && centre->dir == camera.forward(),
              "the optical axis must be the camera's forward direction bitwise");

    // A real lens and gate fix the half-angles: atan(h/2f) vertically, atan(aspect*h/2f) horizontally.
    const float expectedUp = std::atan(kFullFrame.heightMm / (2.0F * kFocalMm));
    const float expectedRight = std::atan((kAspect * kFullFrame.heightMm) / (2.0F * kFocalMm));
    const glm::vec3 forward = camera.forward();
    const std::array<std::pair<glm::vec2, float>, 4> cases{{{{0.0F, 1.0F}, expectedUp},
                                                             {{0.0F, -1.0F}, expectedUp},
                                                             {{1.0F, 0.0F}, expectedRight},
                                                             {{-1.0F, 0.0F}, expectedRight}}};
    for (const auto& [ndc, expected] : cases) {
        const std::optional<Ray> ray = camera.primaryRay(ndc.x, ndc.y, kAspect);
        const float angle = angleBetween(ray->dir, forward);
        PT_EXPECT(ctx, ray.has_value() && std::abs(angle - expected) <= kClosedFormUlps * kEps * expected,
                  "ndc (" + std::to_string(ndc.x) + ", " + std::to_string(ndc.y) + ") subtends " +
                      std::to_string(angle) + ", expected " + std::to_string(expected));
    }
}

// All coefficients zero is the equidistant family exactly, r = f*theta, which is the one case with a closed-form inverse.
PT_CHECK(kannala_brandt_reduces_to_equidistant, Fast, Exact) {
    ctx.plan(3);
    constexpr float kFocalMm = 12.0F;
    constexpr float kFovDegrees = 180.0F;
    const std::array<float, 4> k{};
    const float thetaMax = 0.5F * glm::radians(kFovDegrees);

    int inverseMismatches = 0;
    constexpr int kSamples = 4096;
    for (int i = 0; i <= kSamples; ++i) {
        const float radius = thetaMax * (static_cast<float>(i) / static_cast<float>(kSamples));
        // The seed is the radius itself, so the residual is zero on the first iteration and the inverse returns it unrounded.
        if (kannalaBrandtTheta(k, radius, thetaMax) != radius) {
            ++inverseMismatches;
        }
    }
    PT_EXPECT(ctx, inverseMismatches == 0,
              std::to_string(inverseMismatches) + " of " + std::to_string(kSamples + 1) +
                  " equidistant inversions were not bitwise the radius");

    const Camera camera = makeCamera(kFocalMm, fisheye(k, kFovDegrees));
    const Camera::ViewBasis basis = camera.viewBasis(kAspect);
    int angleMismatches = 0;
    int imaged = 0;
    constexpr int kGrid = 129;
    for (int iy = 0; iy < kGrid; ++iy) {
        for (int ix = 0; ix < kGrid; ++ix) {
            const float ndcX = ((static_cast<float>(ix) / static_cast<float>(kGrid - 1)) * 2.0F) - 1.0F;
            const float ndcY = ((static_cast<float>(iy) / static_cast<float>(kGrid - 1)) * 2.0F) - 1.0F;
            const std::optional<Ray> ray = camera.primaryRay(basis, ndcX, ndcY);
            if (!ray.has_value()) {
                continue;
            }
            ++imaged;
            const float radiusMm = std::hypot(ndcX * basis.halfWidthMm, ndcY * basis.halfHeightMm);
            const float expected = radiusMm / kFocalMm;
            const float angle = angleBetween(ray->dir, basis.forward);
            // Absolute below one radian: the equidistant angle is exact in theory, so the budget is the ndc arithmetic's rounding.
            if (std::abs(angle - expected) > kClosedFormUlps * kEps * std::max(expected, 1.0F)) {
                ++angleMismatches;
            }
        }
    }
    PT_EXPECT(ctx, imaged > 0, "no ndc sample on the grid was imaged at all");
    PT_EXPECT(ctx, angleMismatches == 0,
              std::to_string(angleMismatches) + " of " + std::to_string(imaged) +
                  " equidistant rays departed from r = f*theta");
}

// The inverse against a bisection oracle on the forward polynomial, over the three classic families' Taylor coefficients.
PT_CHECK(kannala_brandt_inverse_round_trips, Fast, Exact) {
    ctx.plan(4);
    const std::array<std::pair<const char*, std::array<float, 4>>, 4> lenses{
        {{"equidistant", std::array<float, 4>{}},
         {"equisolid", kEquisolidTaylor},
         {"stereographic", kStereographicTaylor},
         {"orthographic", kOrthographicTaylor}}};
    // Orthographic's sin(theta) turns over at pi/2, so its domain is the quarter circle; the rest hold across a 180-degree circle.
    constexpr float kHalfPi = 0.5F * std::numbers::pi_v<float>;
    for (const auto& [name, k] : lenses) {
        const float thetaMax = (name == std::string("orthographic")) ? kHalfPi * 0.99F : kHalfPi;
        int mismatches = 0;
        constexpr int kSamples = 8192;
        for (int i = 0; i <= kSamples; ++i) {
            const float theta = thetaMax * (static_cast<float>(i) / static_cast<float>(kSamples));
            const float radius = kannalaBrandtRadius(k, theta);
            const float recovered = kannalaBrandtTheta(k, radius, thetaMax);
            if (std::abs(recovered - theta) > thetaTolerance(k, theta, thetaMax)) {
                ++mismatches;
            }
            if (std::abs(recovered - bisectTheta(k, radius, thetaMax)) > thetaTolerance(k, theta, thetaMax)) {
                ++mismatches;
            }
        }
        PT_EXPECT(ctx, mismatches == 0,
                  std::string(name) + ": " + std::to_string(mismatches) +
                      " inversions left the derived forward-error bound");
    }
}

// The monotonicity gate against a slope whose root is known exactly by construction, plus one-sided soundness over random coefficients.
PT_CHECK(kannala_brandt_monotonicity_gate_is_conservative, Fast, Exact) {
    ctx.plan(4);
    // theta_d' = 1 + 3*k1*u with k1 = -1/(3*u0) is exactly 1 - u/u0, so the slope vanishes at theta = sqrt(u0) and nowhere before.
    constexpr float kRootTheta = 1.0F;
    const std::array<float, 4> constructed{-1.0F / (3.0F * kRootTheta * kRootTheta), 0.0F, 0.0F, 0.0F};
    PT_EXPECT(ctx, kannalaBrandtIsInvertible(constructed, kRootTheta * 0.95F),
              "a domain inside the known root must be accepted");
    PT_EXPECT(ctx, !kannalaBrandtIsInvertible(constructed, kRootTheta * 1.05F),
              "a domain past the known root must be rejected");
    // The exact orthographic family turns over at pi/2, and its Taylor polynomial inherits that: the gate must see it.
    PT_EXPECT(ctx, !kannalaBrandtIsInvertible(kOrthographicTaylor, std::numbers::pi_v<float>),
              "orthographic coefficients over a full 360-degree circle must be rejected");

    std::mt19937 rng(static_cast<std::uint32_t>(ctx.subSeed("random-coefficients")));
    std::uniform_real_distribution<float> coefficient(-0.5F, 0.5F);
    int unsound = 0;
    constexpr int kLenses = 2048;
    for (int lens = 0; lens < kLenses; ++lens) {
        const std::array<float, 4> k{coefficient(rng), coefficient(rng), coefficient(rng), coefficient(rng)};
        const float thetaMax = 0.5F * std::numbers::pi_v<float>;
        bool sweepFoundNonPositive = false;
        constexpr int kSweep = 4096;
        for (int i = 0; i <= kSweep; ++i) {
            const float theta = thetaMax * (static_cast<float>(i) / static_cast<float>(kSweep));
            sweepFoundNonPositive = sweepFoundNonPositive || kannalaBrandtRadiusSlope(k, theta) <= 0.0F;
        }
        // One-sided: the gate may reject a lens the sweep found nothing wrong with, but never accept one the sweep disproved.
        if (sweepFoundNonPositive && kannalaBrandtIsInvertible(k, thetaMax)) {
            ++unsound;
        }
    }
    PT_EXPECT(ctx, unsound == 0,
              std::to_string(unsound) + " of " + std::to_string(kLenses) +
                  " random lenses were accepted although their slope vanished");
}

// The test projects a known direction forward through the model; primaryRay must invert it. Forward here, inverse in the library.
PT_CHECK(fisheye_direction_inverts_the_forward_model, Fast, Exact) {
    ctx.plan(2);
    constexpr float kFocalMm = 10.0F;
    constexpr float kFovDegrees = 180.0F;
    const Camera camera = makeCamera(kFocalMm, fisheye(kEquisolidTaylor, kFovDegrees));
    const Camera::ViewBasis basis = camera.viewBasis(kAspect);
    std::mt19937 rng(static_cast<std::uint32_t>(ctx.subSeed("directions")));
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);

    int tested = 0;
    int mismatches = 0;
    constexpr int kDirections = 20000;
    for (int i = 0; i < kDirections; ++i) {
        // Uniform in theta rather than in solid angle: it spreads the sweep evenly across the image radius and keeps theta exact.
        const float theta = unit(rng) * basis.maxThetaRadians;
        const float cosTheta = std::cos(theta);
        const float phi = 2.0F * std::numbers::pi_v<float> * unit(rng);
        const glm::vec3 direction = glm::normalize((cosTheta * basis.forward) +
                                                   (std::sin(theta) * ((std::cos(phi) * basis.right) +
                                                                        (std::sin(phi) * basis.up))));
        const float radiusMm = kFocalMm * kannalaBrandtRadius(kEquisolidTaylor, theta);
        const float ndcX = (radiusMm * std::cos(phi)) / basis.halfWidthMm;
        const float ndcY = (radiusMm * std::sin(phi)) / basis.halfHeightMm;
        if (std::abs(ndcX) > 1.0F || std::abs(ndcY) > 1.0F) {
            continue;
        }
        ++tested;
        const std::optional<Ray> ray = camera.primaryRay(basis, ndcX, ndcY);
        if (!ray.has_value()) {
            ++mismatches;
            continue;
        }
        const float angleError = angleBetween(ray->dir, direction);
        // The angle is recovered through the inverse, so the budget is the inverse's own bound plus the ndc round trip's rounding.
        if (angleError > thetaTolerance(kEquisolidTaylor, theta, basis.maxThetaRadians) +
                             (kClosedFormUlps * kEps * std::max(theta, 1.0F))) {
            ++mismatches;
        }
    }
    PT_EXPECT(ctx, tested > kDirections / 4, "too few directions landed inside the frame to be a sweep");
    PT_EXPECT(ctx, mismatches == 0,
              std::to_string(mismatches) + " of " + std::to_string(tested) +
                  " directions were not recovered from their projected sensor point");
}

// The image circle is a property of the lens against the gate: it either falls inside the frame, leaving black corners, or covers it.
PT_CHECK(fisheye_image_circle_bounds_the_frame, Fast, Exact) {
    ctx.plan(6);
    constexpr float kFovDegrees = 180.0F;
    const Lens lens = fisheye(kEquisolidTaylor, kFovDegrees);
    const float thetaMax = 0.5F * glm::radians(kFovDegrees);
    const float cornerMm = std::hypot(kAspect * 0.5F * kFullFrame.heightMm, 0.5F * kFullFrame.heightMm);
    // r_max = f*theta_d(thetaMax) scales with focal length, so a long focal length puts the circle outside the gate and a short one inside.
    const float circleFocalMm = (0.5F * cornerMm) / kannalaBrandtRadius(lens.radialCoefficients, thetaMax);
    const float fullFrameFocalMm = (2.0F * cornerMm) / kannalaBrandtRadius(lens.radialCoefficients, thetaMax);

    const Camera circular = makeCamera(circleFocalMm, lens);
    const Camera::ViewBasis circularBasis = circular.viewBasis(kAspect);
    const Camera full = makeCamera(fullFrameFocalMm, lens);
    const Camera::ViewBasis fullBasis = full.viewBasis(kAspect);

    PT_EXPECT(ctx, circular.primaryRay(circularBasis, 0.0F, 0.0F).has_value(),
              "the optical axis must image whatever the circle's size");
    int corneredRays = 0;
    int fullCornerRays = 0;
    const std::array<glm::vec2, 4> corners{{{1.0F, 1.0F}, {-1.0F, 1.0F}, {1.0F, -1.0F}, {-1.0F, -1.0F}}};
    for (const glm::vec2& corner : corners) {
        corneredRays += circular.primaryRay(circularBasis, corner.x, corner.y).has_value() ? 1 : 0;
        fullCornerRays += full.primaryRay(fullBasis, corner.x, corner.y).has_value() ? 1 : 0;
    }
    PT_EXPECT(ctx, corneredRays == 0, "a circle inside the gate must leave all four corners unimaged");
    PT_EXPECT(ctx, fullCornerRays == 4, "a circle past the gate's corners must image all four of them");

    int outsidePredicateErrors = 0;
    int angleOverruns = 0;
    int nonMonotoneAzimuths = 0;
    constexpr int kGrid = 257;
    for (int iy = 0; iy < kGrid; ++iy) {
        bool imagedLast = true;
        for (int ix = 0; ix < kGrid; ++ix) {
            const float ndcX = ((static_cast<float>(ix) / static_cast<float>(kGrid - 1)) * 2.0F) - 1.0F;
            const float ndcY = ((static_cast<float>(iy) / static_cast<float>(kGrid - 1)) * 2.0F) - 1.0F;
            const std::optional<Ray> ray = circular.primaryRay(circularBasis, ndcX, ndcY);
            const float radiusMm = std::hypot(ndcX * circularBasis.halfWidthMm, ndcY * circularBasis.halfHeightMm);
            if (ray.has_value() != (radiusMm <= circularBasis.maxRadiusMm)) {
                ++outsidePredicateErrors;
            }
            if (ray.has_value()) {
                const float angle = angleBetween(ray->dir, circularBasis.forward);
                angleOverruns += angle > thetaMax + (kClosedFormUlps * kEps * thetaMax) ? 1 : 0;
                // Along a row the radius falls then rises, so an imaged sample after a gap would mean the unimaged set is not a disc.
                nonMonotoneAzimuths += (!imagedLast && ndcX > 0.0F) ? 1 : 0;
            }
            imagedLast = ray.has_value();
        }
    }
    PT_EXPECT(ctx, outsidePredicateErrors == 0,
              std::to_string(outsidePredicateErrors) + " samples disagreed with the image-circle radius test");
    PT_EXPECT(ctx, angleOverruns == 0, std::to_string(angleOverruns) + " imaged rays exceeded thetaMax");
    PT_EXPECT(ctx, nonMonotoneAzimuths == 0,
              std::to_string(nonMonotoneAzimuths) + " imaged samples followed an unimaged one across a row");
}

// What CLAHE's per-degree scale reads: the angle at the top of the gate, saturating at thetaMax once the circle falls inside it.
PT_CHECK(fisheye_vertical_extent_saturates_at_the_image_circle, Fast, Exact) {
    ctx.plan(3);
    constexpr float kFovDegrees = 180.0F;
    const Lens lens = fisheye(kEquisolidTaylor, kFovDegrees);
    const float thetaMax = maxThetaRadians(lens);
    const float halfHeightMm = 0.5F * kFullFrame.heightMm;
    // Focal length putting the circle exactly on the gate's top edge, so half of it leaves the circle inside and twice it outside.
    const float edgeFocalMm = halfHeightMm / kannalaBrandtRadius(lens.radialCoefficients, thetaMax);

    const Camera inside = makeCamera(0.5F * edgeFocalMm, lens);
    PT_EXPECT(ctx, inside.verticalAngularExtentRadians() == 2.0F * thetaMax,
              "a circle inside the gate must saturate the extent at the full field of view");

    // Outside, the gate cuts the circle at half its radius, so the extent is that radius' angle, by independent bisection.
    const Camera outside = makeCamera(2.0F * edgeFocalMm, lens);
    const float cutRadius = 0.5F * kannalaBrandtRadius(lens.radialCoefficients, thetaMax);
    const float expected = 2.0F * bisectTheta(lens.radialCoefficients, cutRadius, thetaMax);
    PT_EXPECT(ctx, std::abs(outside.verticalAngularExtentRadians() - expected) <=
                       2.0F * thetaTolerance(lens.radialCoefficients, 0.5F * expected, thetaMax),
              "a circle beyond the gate must give the angle imaged at the top edge, below the full field of view");

    // The CLAHE regression: under Spherical the extent is the pinhole vfov to the bit, so switching projections cannot move a filter.
    const Camera spherical = makeCamera(0.5F * edgeFocalMm, Lens{});
    PT_EXPECT(ctx, spherical.verticalAngularExtentRadians() == spherical.verticalFovRadians(),
              "Spherical must report the pinhole vertical field of view bitwise");
}

// The Jacobian no per-ray check can see: uniform sensor area maps to theta with CDF (theta_d(theta)/theta_d(thetaMax))^2.
PT_CHECK(fisheye_theta_distribution_matches_the_area_jacobian, Slow, Statistical) {
    ctx.plan(2);
    constexpr float kFovDegrees = 180.0F;
    const Lens lens = fisheye(kEquisolidTaylor, kFovDegrees);
    const float thetaMax = 0.5F * glm::radians(kFovDegrees);
    const float halfHeightMm = 0.5F * kFullFrame.heightMm;
    // Focal length chosen so the whole image circle sits inside the gate's shorter axis: every disc sample is then an ndc sample.
    const float focalMm = halfHeightMm / kannalaBrandtRadius(lens.radialCoefficients, thetaMax);
    const Camera camera = makeCamera(focalMm, lens);
    const Camera::ViewBasis basis = camera.viewBasis(kAspect);

    constexpr int kBins = 32;
    constexpr int kSamples = 400000;
    std::mt19937 rng(static_cast<std::uint32_t>(ctx.subSeed("area-samples")));
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    std::array<long long, kBins> observed{};
    int unimaged = 0;
    for (int i = 0; i < kSamples; ++i) {
        // Area-uniform on the disc: radius by the square root of a uniform, azimuth uniform, which is the sensor's own measure.
        const float radiusMm = basis.maxRadiusMm * std::sqrt(unit(rng));
        const float phi = 2.0F * std::numbers::pi_v<float> * unit(rng);
        const std::optional<Ray> ray = camera.primaryRay(basis, (radiusMm * std::cos(phi)) / basis.halfWidthMm,
                                                          (radiusMm * std::sin(phi)) / basis.halfHeightMm);
        if (!ray.has_value()) {
            ++unimaged;
            continue;
        }
        const float theta = angleBetween(ray->dir, basis.forward);
        const auto bin = static_cast<std::size_t>(std::min(
            static_cast<int>((theta / thetaMax) * static_cast<float>(kBins)), kBins - 1));
        ++observed[bin];
    }
    PT_EXPECT(ctx, unimaged == 0,
              std::to_string(unimaged) + " area samples inside the image circle were not imaged");

    // Expected mass per bin straight from the CDF, evaluated forward only: theta_d(b)^2 - theta_d(a)^2 over theta_d(thetaMax)^2.
    const double total = static_cast<double>(kSamples - unimaged);
    const double radiusMax = kannalaBrandtRadius(lens.radialCoefficients, thetaMax);
    double chiSquare = 0.0;
    for (int bin = 0; bin < kBins; ++bin) {
        const float lower = thetaMax * (static_cast<float>(bin) / static_cast<float>(kBins));
        const float upper = thetaMax * (static_cast<float>(bin + 1) / static_cast<float>(kBins));
        const double lowerRadius = kannalaBrandtRadius(lens.radialCoefficients, lower);
        const double upperRadius = kannalaBrandtRadius(lens.radialCoefficients, upper);
        const double expected =
            total * (((upperRadius * upperRadius) - (lowerRadius * lowerRadius)) / (radiusMax * radiusMax));
        const double residual = static_cast<double>(observed[static_cast<std::size_t>(bin)]) - expected;
        chiSquare += (residual * residual) / expected;
    }
    const double pValue = tools::stats::chiSquareUpperTail(chiSquare, kBins - 1);
    PT_EXPECT(ctx, pValue > ctx.alpha(),
              "chi-square " + std::to_string(chiSquare) + " on " + std::to_string(kBins - 1) +
                  " degrees of freedom gives p = " + std::to_string(pValue));
}

PT_CHECK_MAIN("camera_validate")
