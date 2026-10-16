#pragma once

#include <array>
#include <span>

#include "pathtracer/debug/aov.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/path_tracer.h"
#include "pathtracer/scene/gbuffer.h"

namespace pathtracer::debug {

// AovId to the buffer lane holding it. Member pointers, so both are total functions: null for an AovId the producer does not own.
using PathTracedLane = pathtracer::gfx::HdrImage pathtracer::scene::PathTraceResult::*;
using GBufferLane = pathtracer::gfx::HdrImage pathtracer::scene::GBuffer::*;

// Non-null exactly when aovSource(aov) == AovSource::PathTraced.
[[nodiscard]] PathTracedLane pathTracedLane(AovId aov);

// Non-null exactly when aovSource(aov) == AovSource::GBuffer.
[[nodiscard]] GBufferLane gbufferLane(AovId aov);

// Denoised's guides: path-traced lanes no AOV exposes, averaged over Beauty's samples and read by the denoiser alone.
inline constexpr std::array<PathTracedLane, 2> kDenoiseFeatureLanes{&pathtracer::scene::PathTraceResult::denoiseAlbedo,
                                                                    &pathtracer::scene::PathTraceResult::denoiseNormal};

// Shadow's numerator and denominator: a ratio does not average, so its two means accumulate and the ratio is taken after.
inline constexpr std::array<PathTracedLane, 2> kShadowLanes{&pathtracer::scene::PathTraceResult::shadowLost,
                                                            &pathtracer::scene::PathTraceResult::shadowUnoccluded};

// Every lane of each producer in AovId order, then kDenoiseFeatureLanes and kShadowLanes, so a loop over a buffer's lanes misses none.
[[nodiscard]] std::span<const PathTracedLane> pathTracedLanes();
[[nodiscard]] std::span<const GBufferLane> gbufferLanes();

}  // namespace pathtracer::debug
