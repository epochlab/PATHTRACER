#pragma once

#include <cstdint>
#include <vector>

#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/camera.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/shading_scene.h"
#include "pathtracer/scene/thread_pool.h"

namespace pathtracer::scene {

// Primary-hit-only G-buffer AOVs from a standalone CPU rasterizer: no lighting, BSDF sampling or recursion, just the first surface.
struct RasterGBuffer {
    pathtracer::gfx::HdrImage iorAov;
    pathtracer::gfx::HdrImage depth;
    // `depth` on a fixed scale: 1 at the camera plane falling linearly to 0 at lookaheadDistance, clamped, so no near/far pair is needed.
    pathtracer::gfx::HdrImage lookahead;
    pathtracer::gfx::HdrImage worldPos;
    pathtracer::gfx::HdrImage uv;
    // Screen-space motion in current pixels, x_now - x_previous of the point seen at the pixel centre; 0 where either view has no image.
    pathtracer::gfx::HdrImage motionVector;
    pathtracer::gfx::HdrImage normal;
    pathtracer::gfx::HdrImage geomNormal;
    pathtracer::gfx::HdrImage albedo;
    pathtracer::gfx::HdrImage metallic;
    pathtracer::gfx::HdrImage roughness;
    pathtracer::gfx::HdrImage tangent;
    pathtracer::gfx::HdrImage objectId;
    pathtracer::gfx::HdrImage alpha;
    // Colour-coded, not a 0/1 mask: white near a mesh triangle edge, the instance's falseColorForId hue near the instance boundary.
    pathtracer::gfx::HdrImage wireframe;
    // Bumped by every renderRasterGBuffer call: the buffer is reused in place, so a consumer caching by pointer needs this to see a change.
    std::uint64_t generation = 0;
};

// Watertight edge-function rasterization (Pineda 1988), row-parallel over disjoint rows; previousCamera sets motionVector's origin.
void renderRasterGBuffer(const Camera& camera, const Camera& previousCamera,
                          const std::vector<ShadingTriangle>& shadingTriangles,
                          const std::vector<MeshInstance>& instances,
                          const std::vector<PathTraceSettings>& perInstanceSettings,
                          const std::vector<AabbBounds>& instanceBounds, int width, int height,
                          ThreadPool& threadPool, RasterGBuffer& out);

}  // namespace pathtracer::scene
