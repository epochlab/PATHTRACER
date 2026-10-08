#include "pathtracer/debug/system_info.h"

#include <array>
#include <cstdio>

// GLEW before GLFW: see gl_debug.cpp for why.
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <OpenColorIO/OpenColorABI.h>
#include <OpenImageIO/oiioversion.h>
#include <embree4/rtcore_config.h>
#include <glm/detail/setup.hpp>

namespace pathtracer::debug {

namespace {
std::string glString(GLenum name) {
    const auto* str = reinterpret_cast<const char*>(glGetString(name));
    return str != nullptr ? std::string(str) : std::string();
}
}  // namespace

GpuInfo queryGpuInfo() {
    GpuInfo info;
    info.renderer = glString(GL_RENDERER);
    info.version = glString(GL_VERSION);
    return info;
}

LibraryVersions queryLibraryVersions() {
    LibraryVersions versions;
    versions.embree = RTC_VERSION_STRING;
    versions.oiio = OIIO_VERSION_STRING;
    versions.ocio = OCIO_VERSION_FULL_STR;
    // glfwGetVersionString carries the backend list too: which platform and timer backend were compiled in, not just the version.
    versions.glfw = glfwGetVersionString();
    versions.glew = reinterpret_cast<const char*>(glewGetString(GLEW_VERSION));
    std::array<char, 32> glmVersion{};
    std::snprintf(glmVersion.data(), glmVersion.size(), "%d.%d.%d", GLM_VERSION_MAJOR,
                   GLM_VERSION_MINOR, GLM_VERSION_PATCH);
    versions.glm = glmVersion.data();
    return versions;
}

}  // namespace pathtracer::debug
