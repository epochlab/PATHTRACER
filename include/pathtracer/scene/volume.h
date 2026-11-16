#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/config/scene_config.h"
#include "pathtracer/scene/ray_types.h"
#include "pathtracer/scene/spectrum.h"
#include "pathtracer/scene/sampler.h"

namespace pathtracer::scene {

// Henyey-Greenstein (1941) density per steradian of turning from `dir` to wi, cosTheta = dot(dir, wi): g > 0 scatters forward.
[[nodiscard]] float henyeyGreenstein(float cosTheta, float g);

// A unit wi about unit `dir` with density henyeyGreenstein(dot(dir, wi), g): the exact inverse CDF, in a form free of 1/g.
[[nodiscard]] glm::vec3 sampleHenyeyGreenstein(const glm::vec3& dir, float g, glm::vec2 u);

// How a free flight ended: at the segment's end or at a real scattering; absorption rides in the weight.
enum class FreeFlightEvent { Escaped, Scattered };

struct FreeFlight {
    FreeFlightEvent event;
    float t;  // the scattering's ray parameter; unset otherwise
    float g;  // the scattering volume's asymmetry
    // The lane mixture's one-sample weight (Wilkie et al. 2014) for throughput and the feature chain alike: the albedo in a grey medium.
    SampledSpectrum weight;
};

// VDB density grids as heterogeneous media, read with OpenVDB and traced as NanoVDB. Empty by default; copies share the grids.
class VolumeSet {
public:
    VolumeSet() = default;

    // Every config's grid loaded and validated at this boundary; nullopt with `error` naming the file and the fault.
    [[nodiscard]] static std::optional<VolumeSet> load(const std::vector<pathtracer::config::VolumeConfig>& configs,
                                                       const std::string& assetRoot, std::string& error);

    [[nodiscard]] bool empty() const;

    // The first scattering along ray over [t0, t1) by regular tracking (Sutton et al. 1999), its lane drawn by throughput-weighted odds.
    [[nodiscard]] FreeFlight sampleFreeFlight(const Ray& ray, float t0, float t1, const SampledSpectrum& throughput,
                                              const SampledWavelengths& lambda, Sampler& sampler) const;

    // Exact transmittance over [t0, t1): exp of each lane's optical depth, summed voxel by voxel over every volume.
    [[nodiscard]] SampledSpectrum transmittance(const Ray& ray, float t0, float t1, const SampledWavelengths& lambda) const;

    struct Volume;  // one grid's NanoVDB handles, placement and coefficients, defined in volume.cpp

private:
    std::shared_ptr<const std::vector<Volume>> volumes_;
};

}  // namespace pathtracer::scene
