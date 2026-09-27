#pragma once

#include <array>
#include <numbers>

namespace pathtracer::scene {

// Which projection primaryRay builds: Spherical is the rectilinear pinhole, FisheyePolynomial is Kannala & Brandt 2006.
enum class LensProjection { Spherical, FisheyePolynomial, Count };

// Index-parallel with LensProjection, so the HUD dropdown and the config error paths name a projection from one table.
inline constexpr const char* kLensProjectionNames[] = {"Spherical", "Fisheye Polynomial"};
static_assert(sizeof(kLensProjectionNames) / sizeof(kLensProjectionNames[0]) ==
                  static_cast<int>(LensProjection::Count),
              "kLensProjectionNames must stay index-parallel with LensProjection");

// Where the image circle lands on the sensor: Native is the authored focal length, the other two derive one from the gate.
enum class LensFit { Native, Circular, FullFrame, Count };

// Index-parallel with LensFit, the same one-table rule kLensProjectionNames follows.
inline constexpr const char* kLensFitNames[] = {"Native", "Circular", "Full Frame"};
static_assert(sizeof(kLensFitNames) / sizeof(kLensFitNames[0]) == static_cast<int>(LensFit::Count),
              "kLensFitNames must stay index-parallel with LensFit");

// r(theta) = focalLengthMm * theta_d(theta), theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9.
struct Lens {
    LensProjection projection = LensProjection::Spherical;
    // k1..k4 exactly as an OpenCV `fisheye` / COLMAP OPENCV_FISHEYE calibration reports them: dimensionless, theta in radians.
    std::array<float, 4> radialCoefficients{};
    // Full angle across the image circle, so a 180-degree fisheye authors 180 and thetaMax is half it. Authored either projection.
    float maxFieldOfViewDegrees = 180.0F;
    // Ignored under Spherical, whose framing is the authored focal length and the gate alone.
    LensFit fit = LensFit::Circular;
};

// Half the authored field of view, which is theta_d's domain: the one place the lens turns degrees into radians.
[[nodiscard]] constexpr float maxThetaRadians(const Lens& lens) {
    return 0.5F * lens.maxFieldOfViewDegrees * (std::numbers::pi_v<float> / 180.0F);
}

// theta_d(theta), the image radius in focal lengths. Horner in u = theta^2, so the odd powers cost no extra multiplies.
[[nodiscard]] float kannalaBrandtRadius(const std::array<float, 4>& k, float thetaRadians);

// d(theta_d)/d(theta). Positive on [0, thetaMax] for every lens kannalaBrandtIsInvertible accepted, which is what makes the inverse unique.
[[nodiscard]] float kannalaBrandtRadiusSlope(const std::array<float, 4>& k, float thetaRadians);

// The unique theta in [0, thetaMax] with theta_d(theta) == radius, by safeguarded Newton. A radius past theta_d(thetaMax) returns thetaMax.
[[nodiscard]] float kannalaBrandtTheta(const std::array<float, 4>& k, float radius, float thetaMax);

// True only where theta_d' > 0 is proven on [0, thetaMax] by Bernstein subdivision: conservative, so an unprovable lens is rejected.
[[nodiscard]] bool kannalaBrandtIsInvertible(const std::array<float, 4>& k, float thetaMax);

// f = extent / theta_d(thetaMax): Circular's extent is the frame's shorter half-axis, FullFrame's its half-diagonal, Native's authored.
[[nodiscard]] float fitFocalLengthMm(const Lens& lens, float authoredFocalLengthMm, float filmBackHeightMm,
                                      float aspect);

}  // namespace pathtracer::scene
