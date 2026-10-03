#pragma once

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

// Every lane of each producer in AovId order, built once from the two maps above, so a loop over a buffer's lanes misses none.
[[nodiscard]] std::span<const PathTracedLane> pathTracedLanes();
[[nodiscard]] std::span<const GBufferLane> gbufferLanes();

}  // namespace pathtracer::debug
