#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

#include "pathtracer/scene/bsdf.h"
#include "pathtracer/scene/spectrum.h"

namespace pathtracer::scene {

// A path's transport bucket: set at bounce 0's lobe, then a transmission sample makes it Refraction unless it is Subsurface.
enum class PathBucket { Diffuse, SpecularReflection, Refraction, Volume, Subsurface };

// A scattering vertex: where the path turned, and the density its continuation was drawn with; a delta draw has none to weigh.
struct PathVertex {
    glm::vec3 position;
    float pdf;
    bool delta;
};

// The techniques that reach a light from a vertex (Fascione et al. 2018 sec. 3.3): each appends one node and weighs nothing itself.
enum class LightTechnique {
    BsdfHit,     // the continuation met a light or the sky: weighed against the light's density from its parent
    Emission,    // surface emission, absent from LightSet: no other technique draws it, so its weight is 1
    SurfaceNee,  // a light sample at a surface, its walk orders each weighed against their exit densities
    VolumeNee,   // a light sample at a volume scattering, weighed against the phase function
};

// Where a node's scene RGB goes besides Beauty: its path's bucket as of the node, or bounce 0's lobe for direct light.
enum class Route { Bucket, DirectVolume, DirectRefraction, DirectLobes };

// A light reached by one technique, with every quantity its MIS weight and contribution need, cached unweighted at construction.
struct LightNode {
    LightTechnique technique;
    int parent;  // the vertex it leaves: for BsdfHit the last whose continuation was drawn
    // BsdfHit and Emission: throughput * Le; SurfaceNee: visibility * throughput * Le / lightPdf; VolumeNee: visibility * throughput * Le.
    SampledSpectrum radiance;
    float lightPdf;  // the light's solid-angle density toward it: NEE's draw, or a BsdfHit's light as NEE would have drawn it
    float phase;     // VolumeNee: the phase function toward the light, its value and the competing density
    bool farSide;    // SurfaceNee: the light is across the surface, so only transmission carries it
    std::size_t ordersBegin;  // SurfaceNee: its WalkOrders in PathGraph::orders
    std::size_t ordersEnd;
    Route route;
    std::optional<PathBucket> bucket;  // Route::Bucket: the path's bucket when the node was added; unset is unbucketed
    bool direct;                       // Route::Bucket: direct or indirect within it
};

// One path's acyclic graph, reused across a chunk's paths so none allocates once warm: vertices, the lights they reach, NEE's orders.
struct PathGraph {
    std::vector<PathVertex> vertices;
    std::vector<LightNode> lights;
    std::vector<WalkOrder> orders;

    void clear() {
        vertices.clear();
        lights.clear();
        orders.clear();
    }
};

}  // namespace pathtracer::scene
