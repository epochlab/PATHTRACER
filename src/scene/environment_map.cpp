#include "pathtracer/scene/environment_map.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/constants.hpp>

namespace pathtracer::scene {

namespace {

// Matches sky.frag's rotateAboutY. Takes the resolved trig, the angle being a pass constant rather than a per-ray one.
glm::vec3 rotateAboutY(const glm::vec3& v, YRotation rotation) {
    const float c = rotation.cosAngle;
    const float s = rotation.sinAngle;
    return {(v.x * c) + (v.z * s), v.y, (-v.x * s) + (v.z * c)};
}

// Rec.709 weights, kRec709LuminanceWeights, floored at 0: a negative texel makes the CDF non-monotonic, which invertCdf's search needs.
float luminanceOf(const pathtracer::gfx::ImageTexture& image, int x, int y) {
    const glm::vec4 texel = image.texel(x, y);
    return std::max(0.0F, (0.2126F * texel.r) + (0.7152F * texel.g) + (0.0722F * texel.b));
}

// Inverts a piecewise-constant CDF slice at u, returning the bin and the fractional offset within its mass. Shared by both inversions.
struct CdfSample {
    int index;
    float fraction;  // [0,1) position within the selected bin
};
// The integer texel a direction falls in, plus that row's sin(theta) for the Jacobian, as pdf()'s piecewise-constant cell.
struct EquirectTexel {
    int x;
    int y;
    float sinTheta;
};

EquirectTexel equirectTexelOf(const pathtracer::gfx::ImageTexture& image, const glm::vec3& direction,
                               YRotation rotation) {
    const glm::vec3 rotated = rotateAboutY(direction, rotation.inverse());
    const float theta = std::acos(glm::clamp(rotated.y, -1.0F, 1.0F));
    const float phi = std::atan2(rotated.x, rotated.z);
    const float uCoord = (phi / (2.0F * glm::pi<float>())) + 0.5F;
    const float v = theta / glm::pi<float>();
    return {std::clamp(static_cast<int>(uCoord * static_cast<float>(image.width)), 0, image.width - 1),
            std::clamp(static_cast<int>(v * static_cast<float>(image.height)), 0, image.height - 1),
            std::max(std::sin(theta), 1e-6F)};
}

CdfSample invertCdf(const float* cdf, int count, float u) {
    // upper_bound finds the first entry > u; the bin before it is u's. cdf[0]==0 is never > u, so searching it is harmless.
    const float* it = std::upper_bound(cdf, cdf + count + 1, u);
    const int index = std::clamp(static_cast<int>(it - cdf) - 1, 0, count - 1);
    const float lo = cdf[index];
    const float hi = cdf[index + 1];
    const float fraction = hi > lo ? std::clamp((u - lo) / (hi - lo), 0.0F, 1.0F) : 0.5F;
    return {index, fraction};
}

}  // namespace

EnvironmentMap::EnvironmentMap(pathtracer::gfx::ImageTexture image) : image_(std::move(image)) {
    const int width = image_.width;
    const int height = image_.height;
    marginalCdf_.assign(static_cast<std::size_t>(height) + 1, 0.0F);
    conditionalCdf_.assign(static_cast<std::size_t>(height) * (static_cast<std::size_t>(width) + 1),
                            0.0F);

    float total = 0.0F;
    for (int y = 0; y < height; ++y) {
        // Row-centre theta, sin(theta)-weighted so density corrects for the equirect projection's polar over-representation.
        const float theta = glm::pi<float>() * (static_cast<float>(y) + 0.5F) / static_cast<float>(height);
        const float sinTheta = std::max(std::sin(theta), 1e-6F);

        float* row = &conditionalCdf_[static_cast<std::size_t>(y) * (static_cast<std::size_t>(width) + 1)];
        float rowSum = 0.0F;
        for (int x = 0; x < width; ++x) {
            rowSum += luminanceOf(image_, x, y) * sinTheta;
            row[x + 1] = rowSum;
        }
        if (rowSum > 0.0F) {
            for (int x = 0; x <= width; ++x) {
                row[x] /= rowSum;
            }
        } else {
            // Degenerate (fully black) row -- uniform fallback so later division/inversion stays well-defined.
            for (int x = 0; x <= width; ++x) {
                row[x] = static_cast<float>(x) / static_cast<float>(width);
            }
        }

        total += rowSum;
        marginalCdf_[static_cast<std::size_t>(y) + 1] = total;
    }

    if (total > 0.0F) {
        for (float& v : marginalCdf_) {
            v /= total;
        }
    } else {
        // Degenerate (fully black) map -- uniform fallback, same reasoning as the per-row case above.
        for (int y = 0; y <= height; ++y) {
            marginalCdf_[static_cast<std::size_t>(y)] = static_cast<float>(y) / static_cast<float>(height);
        }
    }
}

glm::vec3 EnvironmentMap::sampleDirection(const glm::vec3& direction, YRotation rotation) const {
    const glm::vec3 rotated = rotateAboutY(direction, rotation.inverse());
    const float theta = std::acos(glm::clamp(rotated.y, -1.0F, 1.0F));
    const float phi = std::atan2(rotated.x, rotated.z);
    const glm::vec2 uv((phi / (2.0F * glm::pi<float>())) + 0.5F, theta / glm::pi<float>());
    return glm::vec3(pathtracer::gfx::sampleBilinear(image_, uv, pathtracer::gfx::WrapMode::ClampV));
}

EnvironmentMap::EnvSample EnvironmentMap::importanceSampleDirection(glm::vec2 u,
                                                                     YRotation rotation) const {
    const int width = image_.width;
    const int height = image_.height;

    const CdfSample rowSample = invertCdf(marginalCdf_.data(), height, u.x);
    const float v = (static_cast<float>(rowSample.index) + rowSample.fraction) / static_cast<float>(height);

    const float* row =
        &conditionalCdf_[static_cast<std::size_t>(rowSample.index) * (static_cast<std::size_t>(width) + 1)];
    const CdfSample colSample = invertCdf(row, width, u.y);
    const float uCoord = (static_cast<float>(colSample.index) + colSample.fraction) / static_cast<float>(width);

    const float theta = v * glm::pi<float>();
    const float phi = (uCoord - 0.5F) * 2.0F * glm::pi<float>();
    const float sinTheta = std::sin(theta);
    const glm::vec3 rotated(sinTheta * std::sin(phi), std::cos(theta), sinTheta * std::cos(phi));
    const glm::vec3 direction = rotateAboutY(rotated, rotation);

    const float pdfV = (marginalCdf_[static_cast<std::size_t>(rowSample.index) + 1] -
                         marginalCdf_[static_cast<std::size_t>(rowSample.index)]) *
                        static_cast<float>(height);
    const float pdfU = (row[colSample.index + 1] - row[colSample.index]) * static_cast<float>(width);
    // Jacobian from (u,v) to solid angle: dw = sin(theta) * (pi dv) * (2pi du), so the pdf is pdf_uv / (2 * pi^2 * sin(theta)).
    const float pdfSolidAngle =
        (pdfU * pdfV) / std::max(2.0F * glm::pi<float>() * glm::pi<float>() * sinTheta, 1e-6F);
    return {direction, std::max(pdfSolidAngle, 1e-8F)};
}

float EnvironmentMap::pdf(const glm::vec3& direction, YRotation rotation) const {
    const int width = image_.width;
    const int height = image_.height;
    const EquirectTexel texel = equirectTexelOf(image_, direction, rotation);
    const float* row =
        &conditionalCdf_[static_cast<std::size_t>(texel.y) * (static_cast<std::size_t>(width) + 1)];

    const float pdfV = (marginalCdf_[static_cast<std::size_t>(texel.y) + 1] -
                         marginalCdf_[static_cast<std::size_t>(texel.y)]) *
                        static_cast<float>(height);
    const float pdfU = (row[texel.x + 1] - row[texel.x]) * static_cast<float>(width);
    const float pdfSolidAngle =
        (pdfU * pdfV) / std::max(2.0F * glm::pi<float>() * glm::pi<float>() * texel.sinTheta, 1e-6F);
    return std::max(pdfSolidAngle, 1e-8F);
}

}  // namespace pathtracer::scene
