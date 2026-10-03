#pragma once

#include <cstddef>

namespace pathtracer::debug {

struct GpuInfo;

// Everything the startup spec block reports that is not queryable from the host: the scene, its settings, and the cost of loading it.
struct EngineSpec {
    const char* scenePath;
    const char* hdriPath;
    const char* aovName;
    int imageWidth;
    int imageHeight;
    float renderScale;
    float interactiveRenderScale;
    int samplesPerPass;
    int maxBounces;
    int russianRouletteStartBounce;
    int maxSamples;  // 0 = unbounded
    float aoMaxDistance;
    unsigned int pathTraceThreads;
    unsigned int gbufferThreads;
    int tileSize;
    int instanceCount;
    int lightCount;
    int triangleCount;
    double modelLoadMs;
    double bvhBuildMs;
    std::size_t bvhBytes;
    bool khrDebugAvailable;
    bool gpuTimerAvailable;
    double refreshHz;  // the window's display, measured by DisplayLink
};

// One-shot provenance block on stdout, after the GL context and scene exist. Plain text, no ANSI, so it survives being piped to a log.
void printSpec(const EngineSpec& spec, const GpuInfo& gpu);

}  // namespace pathtracer::debug
