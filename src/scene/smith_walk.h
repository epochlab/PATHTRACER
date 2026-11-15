#pragma once

#include <optional>

#include <glm/glm.hpp>

#include "pathtracer/scene/bsdf.h"

// The dielectric interface's scattering by random walks on its Smith microsurface (Heitz, Hanika, d'Eon & Dachsbacher 2016).
namespace pathtracer::scene {

// A walk's exit, untinted: its direction, the side it left on, its throughput, eta^2-compressed when it crossed.
struct WalkExit {
    glm::vec3 direction;
    Rgb weight;
    bool reflected;
};

// One whole walk from wo, every order, its weight unbiased for the untinted f |wi.z| at its exit; nullopt where a draw carries no measure.
[[nodiscard]] std::optional<WalkExit> sampleInterfaceWalk(const DielectricSlab& slab, const glm::vec3& wo, Sampler& walk);

// Untinted f_ms(wo, wi) |wi.z|: a walk from each end, every connection balance-weighted over both ends' path densities (Wang et al. 2022).
[[nodiscard]] Rgb evaluateInterfaceMulti(const DielectricSlab& slab, const glm::vec3& wo, const glm::vec3& wi, Sampler& walk);

}  // namespace pathtracer::scene
