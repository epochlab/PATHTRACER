#pragma once

#include <cstddef>
#include <string>

namespace pathtracer::debug {

// GL_RENDERER and GL_VERSION, plus the window's display refresh from DisplayLink. No GL_VENDOR: GL_RENDERER already names the part.
struct GpuInfo {
    std::string renderer;
    std::string version;
};

[[nodiscard]] GpuInfo queryGpuInfo();

// Host CPU and memory topology from sysctl, for the startup spec block. No CPU brand string: GL_RENDERER already reads "Apple M1".
struct HostInfo {
    std::string perfLevel0Name;         // hw.perflevel0.name -- "Performance" on Apple Silicon
    int perfLevel0LogicalCpu = 0;       // hw.perflevel0.logicalcpu
    std::size_t perfLevel0L2Bytes = 0;  // hw.perflevel0.l2cachesize
    std::string perfLevel1Name;         // hw.perflevel1.name -- "Efficiency"; empty on a uniform-core machine
    int perfLevel1LogicalCpu = 0;
    std::size_t perfLevel1L2Bytes = 0;
    int logicalCpuTotal = 0;   // hw.logicalcpu
    int cacheLineBytes = 0;    // hw.cachelinesize
    std::string osProductVersion;  // kern.osproductversion
    std::string osBuild;           // kern.osversion
};

[[nodiscard]] HostInfo queryHostInfo();

// Compile-time build identity, from the PATHTRACER_* defines CMakeLists sets; nothing here can change after link.
struct BuildInfo {
    const char* compiler;
    const char* buildType;
    const char* gitSha;  // "+dirty" suffixed when the working tree had uncommitted tracked changes at configure time
    const char* march;
    bool ipo;
};

[[nodiscard]] BuildInfo buildInfo();

// The running executable's Mach-O LC_UUID as 32 lowercase hex digits, empty if absent. Identifies the exact binary on a dirty tree.
[[nodiscard]] std::string executableUuid();

// Dependency versions. GLFW and GLEW are queried at runtime, since a header constant could lie about what loaded; glm is header-only.
struct LibraryVersions {
    std::string embree;
    std::string oiio;
    std::string ocio;
    std::string glfw;
    std::string glew;
    std::string glm;
};

[[nodiscard]] LibraryVersions queryLibraryVersions();

}  // namespace pathtracer::debug
