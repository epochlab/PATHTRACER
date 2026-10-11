#pragma once

#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/gfx/hdr_image.h"

namespace pathtracer::scene {

// CPU-resident equirect env map for path-traced miss rays and NEE light sampling.
class EnvironmentMap {
public:
    // Builds the piecewise-constant importance-sampling CDFs (marginal over rows, conditional over columns) from the texture's texels.
    explicit EnvironmentMap(std::shared_ptr<const pathtracer::gfx::ImageTexture> texture);

    // Direction -> equirect UV -> filtered lookup. The renderer's only Le: pdf() below is merely its importance.
    [[nodiscard]] glm::vec3 sampleDirection(const glm::vec3& direction, const glm::mat3& rotation = glm::mat3(1.0F)) const;

    // The same lookup filtered over a footprint, dirFootprint's columns the direction's offsets along its two axes, mapped to st.
    [[nodiscard]] glm::vec3 sampleDirection(const glm::vec3& direction, const glm::mat2x3& dirFootprint, const glm::mat3& rotation) const;

    struct EnvSample {
        glm::vec3 direction;
        float pdf;  // solid-angle pdf, > 0
    };

    // Importance-samples a direction from the map's luminance distribution, for NEE. u is two independent uniforms.
    [[nodiscard]] EnvSample importanceSampleDirection(glm::vec2 u, const glm::mat3& rotation = glm::mat3(1.0F)) const;

    // Solid-angle pdf of importanceSampleDirection() producing this direction, for MIS-weighting a BSDF-sampled environment miss.
    [[nodiscard]] float pdf(const glm::vec3& direction, const glm::mat3& rotation = glm::mat3(1.0F)) const;

private:
    // The CDFs below are built from this texture's finest level, so sampling density matches the radiance returned.
    std::shared_ptr<const pathtracer::gfx::ImageTexture> texture_;
    int width_;
    int height_;
    std::vector<float> marginalCdf_;     // size height+1, marginalCdf_[0]=0, marginalCdf_[height]=1
    std::vector<float> conditionalCdf_;  // size height*(width+1), row-major, each row's slice sums to 1 at its last entry
};

}  // namespace pathtracer::scene
