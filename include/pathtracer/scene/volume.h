#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/config/scene_config.h"
#include "pathtracer/scene/ray_types.h"
#include "pathtracer/scene/sampler.h"

namespace pathtracer::scene {

// Henyey-Greenstein (1941) density per steradian of turning from `dir` to wi, cosTheta = dot(dir, wi): g > 0 scatters forward.
[[nodiscard]] float henyeyGreenstein(float cosTheta, float g);

// A unit wi about unit `dir` with density henyeyGreenstein(dot(dir, wi), g): the exact inverse CDF, in a form free of 1/g.
[[nodiscard]] glm::vec3 sampleHenyeyGreenstein(const glm::vec3& dir, float g, glm::vec2 u);

// How a free flight ended: at the segment's end, absorbed (the path carries nothing further), or at a real scattering.
enum class FreeFlightEvent { Escaped, Absorbed, Scattered };

struct FreeFlight {
    FreeFlightEvent event;
    float t;  // the scattering's ray parameter; unset otherwise
    float g;  // the scattering volume's asymmetry
    // Spectral tracking's weight (Kutz et al. 2017) for throughput and the feature chain alike: 1 for a grey medium.
    glm::vec3 weight;
};

// VDB density grids as heterogeneous media, read with OpenVDB and traced as NanoVDB. Empty by default; copies share the grids.
class VolumeSet {
public:
    VolumeSet() = default;

    // Every config's grid loaded and validated at this boundary; nullopt with `error` naming the file and the fault.
    [[nodiscard]] static std::optional<VolumeSet> load(const std::vector<pathtracer::config::VolumeConfig>& configs,
                                                       const std::string& assetRoot, std::string& error);

    [[nodiscard]] bool empty() const;

    // The first real collision along ray over [t0, t1) by spectral tracking with history-aware probabilities, `throughput` the history.
    [[nodiscard]] FreeFlight sampleFreeFlight(const Ray& ray, float t0, float t1, const glm::vec3& throughput, Sampler& sampler) const;

    // Unbiased transmittance over [t0, t1) by ratio tracking (Novak et al. 2014), the product of every volume's estimate.
    [[nodiscard]] glm::vec3 transmittance(const Ray& ray, float t0, float t1, Sampler& sampler) const;

    struct Volume;  // one grid's NanoVDB handles, placement and coefficients, defined in volume.cpp

private:
    std::shared_ptr<const std::vector<Volume>> volumes_;
};

}  // namespace pathtracer::scene
