#pragma once

#include <cstdint>
#include <vector>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/embree_accel.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/shading_scene.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::scene {

// Primary-hit-only AOVs from one unjittered pixel-centre camera ray each: no lighting, BSDF sampling or recursion, just the first surface.
struct GBuffer {
    pathtracer::gfx::HdrImage iorAov;
    // Distance along the primary ray to the first hit, the one depth every projection defines, a fisheye's past 90 degrees included.
    pathtracer::gfx::HdrImage depth;
    // `depth` on a fixed scale: 1 at the camera falling linearly to 0 at lookaheadDistance, clamped, so no near/far pair is needed.
    pathtracer::gfx::HdrImage lookahead;
    pathtracer::gfx::HdrImage worldPos;
    pathtracer::gfx::HdrImage uv;
    // x_now - x_previous in current pixels of the point at the pixel centre, the shorter way round a lat-long; 0 where a view has no image.
    pathtracer::gfx::HdrImage motionVector;
    pathtracer::gfx::HdrImage normal;
    pathtracer::gfx::HdrImage geomNormal;
    pathtracer::gfx::HdrImage albedo;
    pathtracer::gfx::HdrImage metallic;
    pathtracer::gfx::HdrImage roughness;
    pathtracer::gfx::HdrImage tangent;
    pathtracer::gfx::HdrImage objectId;
    // Colour-coded, not a 0/1 mask: white near a mesh triangle edge, the instance's falseColorForId hue near the instance boundary.
    pathtracer::gfx::HdrImage wireframe;
    // Bumped by every renderGBuffer call: the buffer is reused in place, so a consumer caching by pointer needs this to see a change.
    std::uint64_t generation = 0;
    bool wrapsHorizontally = false;  // the lens it was cast through joins its left and right edges, so neighbour reads wrap in x
};

// Row-parallel over disjoint rows, through the path tracer's own accel and bounce-0 sampling; previousCamera sets motionVector's origin.
void renderGBuffer(const Camera& camera, const Camera& previousCamera, const EmbreeAccel& accel,
                   const std::vector<ShadingTriangle>& shadingTriangles, const std::vector<MeshInstance>& instances,
                   const std::vector<PathTraceSettings>& perInstanceSettings,
                   const std::vector<AabbBounds>& instanceBounds, int width, int height, ThreadPool& threadPool,
                   GBuffer& out);

}  // namespace pathtracer::scene
