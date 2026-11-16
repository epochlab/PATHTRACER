#pragma once

#include <cstddef>

#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>

#include "pathtracer/scene/cie.h"

namespace pathtracer::scene {

// A per-channel ACEScg quantity, apart from geometry so mixing fails to compile; ops are glm's per channel, so bits match glm::vec3.
struct Rgb {
    float r;
    float g;
    float b;

    Rgb() = default;
    constexpr Rgb(float red, float green, float blue) : r(red), g(green), b(blue) {}
    constexpr explicit Rgb(float grey) : r(grey), g(grey), b(grey) {}
    constexpr explicit Rgb(const glm::vec3& v) : r(v.x), g(v.y), b(v.z) {}

    [[nodiscard]] constexpr glm::vec3 vec3() const { return {r, g, b}; }
    [[nodiscard]] constexpr float& operator[](std::size_t i) { return i == 0 ? r : (i == 1 ? g : b); }
    [[nodiscard]] constexpr float operator[](std::size_t i) const { return i == 0 ? r : (i == 1 ? g : b); }

    constexpr Rgb& operator+=(const Rgb& o) { r += o.r; g += o.g; b += o.b; return *this; }
    constexpr Rgb& operator*=(const Rgb& o) { r *= o.r; g *= o.g; b *= o.b; return *this; }
    constexpr Rgb& operator-=(float s) { r -= s; g -= s; b -= s; return *this; }
    constexpr Rgb& operator*=(float s) { r *= s; g *= s; b *= s; return *this; }
    constexpr Rgb& operator/=(float s) { r /= s; g /= s; b /= s; return *this; }

    friend constexpr bool operator==(const Rgb&, const Rgb&) = default;
};

[[nodiscard]] constexpr Rgb operator-(const Rgb& a) { return {-a.r, -a.g, -a.b}; }
[[nodiscard]] constexpr Rgb operator+(const Rgb& a, const Rgb& o) { return {a.r + o.r, a.g + o.g, a.b + o.b}; }
[[nodiscard]] constexpr Rgb operator-(const Rgb& a, const Rgb& o) { return {a.r - o.r, a.g - o.g, a.b - o.b}; }
[[nodiscard]] constexpr Rgb operator*(const Rgb& a, const Rgb& o) { return {a.r * o.r, a.g * o.g, a.b * o.b}; }
[[nodiscard]] constexpr Rgb operator/(const Rgb& a, const Rgb& o) { return {a.r / o.r, a.g / o.g, a.b / o.b}; }
[[nodiscard]] constexpr Rgb operator+(const Rgb& a, float s) { return {a.r + s, a.g + s, a.b + s}; }
[[nodiscard]] constexpr Rgb operator-(const Rgb& a, float s) { return {a.r - s, a.g - s, a.b - s}; }
[[nodiscard]] constexpr Rgb operator*(const Rgb& a, float s) { return {a.r * s, a.g * s, a.b * s}; }
[[nodiscard]] constexpr Rgb operator/(const Rgb& a, float s) { return {a.r / s, a.g / s, a.b / s}; }
[[nodiscard]] constexpr Rgb operator+(float s, const Rgb& a) { return {s + a.r, s + a.g, s + a.b}; }
[[nodiscard]] constexpr Rgb operator-(float s, const Rgb& a) { return {s - a.r, s - a.g, s - a.b}; }
[[nodiscard]] constexpr Rgb operator*(float s, const Rgb& a) { return {s * a.r, s * a.g, s * a.b}; }
[[nodiscard]] constexpr Rgb operator/(float s, const Rgb& a) { return {s / a.r, s / a.g, s / a.b}; }

// Photometric Y of a scene triple: glm::dot with the AP1 luminance row, summing to 1.
[[nodiscard]] inline float luminance(const Rgb& c) { return glm::dot(cie::sceneLuminanceWeights(), c.vec3()); }

// The equal-weight channel mean: selection masses and energies compare channels on one scale.
[[nodiscard]] constexpr float channelMean(const Rgb& c) { return (c.r + c.g + c.b) / 3.0F; }

[[nodiscard]] inline float maxChannel(const Rgb& c) { return glm::compMax(c.vec3()); }
[[nodiscard]] inline float minChannel(const Rgb& c) { return glm::compMin(c.vec3()); }

[[nodiscard]] inline Rgb max(const Rgb& a, const Rgb& o) { return Rgb(glm::max(a.vec3(), o.vec3())); }
[[nodiscard]] inline Rgb clamp(const Rgb& c, float lo, float hi) { return Rgb(glm::clamp(c.vec3(), lo, hi)); }

}  // namespace pathtracer::scene
