#include "pathtracer/debug/spec_report.h"

#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <string>

#include "pathtracer/debug/memory_tracker.h"
#include "pathtracer/debug/system_info.h"

namespace pathtracer::debug {

namespace {

constexpr double kMiB = 1024.0 * 1024.0;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

// printf, not iostream: a fixed label column on the 78-column grid; each row owns its format, an empty label continues the previous.
__attribute__((format(printf, 2, 3))) void row(const char* label, const char* format, ...) {
    std::printf("  %-12s", label);
    std::va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
    std::putchar('\n');
}

void section(const char* name) { std::printf("\n%s\n", name); }

// The error_code overload: relative() throws on a filesystem error, and shortening a path for display must not abort startup.
std::string displayPath(const char* path) {
    std::error_code ec;
    const std::filesystem::path relative = std::filesystem::relative(path, ec);
    // A scene outside the working directory relativises to a run of "..", which is longer and harder to read than the absolute path.
    if (ec || relative.empty() || relative.native().starts_with("..")) {
        return path;
    }
    return relative.string();
}

// Two levels reported separately: E-cores are several times slower, so a pool sized to hw.logicalcpu is not N equal workers.
void printHost(const HostInfo& host) {
    row("os", "macOS %s (build %s)", host.osProductVersion.c_str(), host.osBuild.c_str());
    if (!host.perfLevel1Name.empty()) {
        row("cpu", "%d logical cores", host.logicalCpuTotal);
        row("", "%d %s (L2 %.1f MiB) + %d %s (L2 %.1f MiB)", host.perfLevel0LogicalCpu,
            host.perfLevel0Name.c_str(), static_cast<double>(host.perfLevel0L2Bytes) / kMiB,
            host.perfLevel1LogicalCpu, host.perfLevel1Name.c_str(),
            static_cast<double>(host.perfLevel1L2Bytes) / kMiB);
    } else {
        row("cpu", "%d logical cores (L2 %.1f MiB)", host.logicalCpuTotal,
            static_cast<double>(host.perfLevel0L2Bytes) / kMiB);
    }
    row("memory", "%.2f GiB  %d B cache line", static_cast<double>(totalSystemBytes()) / kGiB,
        host.cacheLineBytes);
}

// Compiler alone on its row because __VERSION__ is long and unbounded; the flags and the SHA they applied to stay together.
void printBuild(const BuildInfo& build, const LibraryVersions& libs) {
    row("compiler", "%s", build.compiler);
    row("flags", "-march=%s  IPO %s", build.march, build.ipo ? "on" : "off");
    row("libraries", "embree %s  oiio %s  ocio %s  glew %s  glm %s", libs.embree.c_str(),
        libs.oiio.c_str(), libs.ocio.c_str(), libs.glew.c_str(), libs.glm.c_str());
    // glfw last and alone: glfwGetVersionString appends its backend list, making this field ~50 columns where the others are ~12.
    row("glfw", "%s", libs.glfw.c_str());
}

}  // namespace

void printSpec(const EngineSpec& spec, const GpuInfo& gpu) {
    const BuildInfo build = buildInfo();
    std::printf("pathtracer  %s  %s\n", build.buildType, build.gitSha);

    section("BUILD");
    printBuild(build, queryLibraryVersions());

    section("SYSTEM");
    printHost(queryHostInfo());
    // GL_RENDERER leads the gpu row: on a unified-memory part it names the machine as well as the GPU, hence no CPU brand row above.
    row("gpu", "%s  %s  %.2f Hz", gpu.renderer.c_str(), gpu.version.c_str(), spec.refreshHz);
    row("extensions", "GL_KHR_debug %s  GL_ARB_timer_query %s", spec.khrDebugAvailable ? "yes" : "no",
        spec.gpuTimerAvailable ? "yes" : "no");

    section("SCENE");
    // Paths stay alone on their rows: both are unbounded, so packing either beside a fixed field is what makes the block wrap.
    row("file", "%s", displayPath(spec.scenePath).c_str());
    row("geometry", "%d instances (%d light)  %d triangles", spec.instanceCount, spec.lightCount,
        spec.triangleCount);
    row("hdri", "%s", spec.hdriPath);
    row("load", "model %.1f ms  bvh %.1f ms  bvh %.1f MiB", spec.modelLoadMs, spec.bvhBuildMs,
        static_cast<double>(spec.bvhBytes) / kMiB);

    section("RENDER");
    row("resolution", "%dx%d  scale %.2f (interactive %.2f)  aov %s", spec.imageWidth,
        spec.imageHeight, spec.renderScale, spec.interactiveRenderScale, spec.aovName);
    row("path trace", "%u threads  %d px tiles  %d spp/pass  %d bounces  RR@%d",
        spec.pathTraceThreads, spec.tileSize, spec.samplesPerPass, spec.maxBounces,
        spec.russianRouletteStartBounce);
    if (spec.maxSamples > 0) {
        row("sampling", "max %d samples  ao range %.2f", spec.maxSamples,
            static_cast<double>(spec.aoMaxDistance));
    } else {
        row("sampling", "unlimited samples  ao range %.2f", static_cast<double>(spec.aoMaxDistance));
    }
    row("gbuffer", "%u threads", spec.gbufferThreads);

    std::fflush(stdout);
}

}  // namespace pathtracer::debug
