#include "pathtracer/api/pathtracer_c.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "pathtracer/api/headless_renderer.h"
#include "pathtracer/debug/aov.h"
#include "pathtracer/gfx/ocio_cpu_transform.h"

namespace {

using pathtracer::api::HeadlessRenderer;
using pathtracer::debug::AovId;

// Every entry point is noexcept at the boundary: an exception crossing into ctypes is UB, so each is caught and reported through err.
void writeError(char* err, int errCap, const std::string& message) {
    if (err == nullptr || errCap <= 0) {
        return;
    }
    const auto length = std::min(message.size(), static_cast<std::size_t>(errCap - 1));
    std::memcpy(err, message.data(), length);
    err[length] = '\0';
}

[[nodiscard]] bool validAov(int aov) {
    return aov >= 0 && aov < static_cast<int>(AovId::Count);
}

// A boundary value, so anything outside the three the ABI defines is rejected rather than coerced to on or off.
[[nodiscard]] bool toOptionalBool(int triState, const char* field, std::optional<bool>& out, std::string& error) {
    if (triState == PT_DEFAULT) { out = std::nullopt; return true; }
    if (triState == 0 || triState == 1) { out = triState != 0; return true; }
    error = std::string(field) + " must be PT_DEFAULT, 0 or 1, got " + std::to_string(triState);
    return false;
}

// Decode only: an unknown projection has no enum value to coerce to; every range is Camera::validate's, run by HeadlessRenderer::render.
[[nodiscard]] std::optional<pathtracer::scene::Camera> toCamera(const PtCamera& camera, std::string& error) {
    pathtracer::scene::Lens lens;
    if (camera.lens_projection == PT_LENS_RECTILINEAR) {
        lens.projection = pathtracer::scene::LensProjection::Rectilinear;
    } else if (camera.lens_projection == PT_LENS_FISHEYE_POLYNOMIAL) {
        lens.projection = pathtracer::scene::LensProjection::FisheyePolynomial;
    } else if (camera.lens_projection == PT_LENS_OMNIDIRECTIONAL) {
        lens.projection = pathtracer::scene::LensProjection::Omnidirectional;
    } else {
        error = "lens_projection must be PT_LENS_RECTILINEAR, PT_LENS_FISHEYE_POLYNOMIAL or PT_LENS_OMNIDIRECTIONAL, got " +
                std::to_string(camera.lens_projection);
        return std::nullopt;
    }
    for (std::size_t i = 0; i < lens.radialCoefficients.size(); ++i) {
        lens.radialCoefficients[i] = camera.fisheye_coefficients[i];
    }
    lens.maxFieldOfViewDegrees = camera.fisheye_field_of_view_degrees;
    return pathtracer::scene::Camera{
        glm::vec3(camera.position[0], camera.position[1], camera.position[2]),
        camera.yaw_degrees,
        camera.pitch_degrees,
        pathtracer::scene::Camera::FilmBack{camera.film_back_mm[0], camera.film_back_mm[1]},
        camera.focal_length_mm,
        camera.near_clip,
        camera.far_clip,
        camera.aperture,
        camera.shutter_seconds,
        camera.iso,
        lens};
}

// NULL keeps the authored lights, so it decodes to nullopt; otherwise one XYZ triple per scene light, in scene order.
[[nodiscard]] std::optional<std::vector<glm::vec3>> toLightRotations(const float* xyz, std::size_t lightCount) {
    if (xyz == nullptr) {
        return std::nullopt;
    }
    std::vector<glm::vec3> rotations;
    rotations.reserve(lightCount);
    for (std::size_t i = 0; i < lightCount; ++i) {
        rotations.push_back(glm::make_vec3(xyz + (3 * i)));
    }
    return rotations;
}

// The request decoded in ABI order, error naming the first bad field; every range follows in render(), after every decode.
[[nodiscard]] std::optional<HeadlessRenderer::Request> decodeRequest(const PtRenderRequest& request, float* const* out,
                                                                     std::size_t lightCount, std::string& error) {
    if (request.aovs == nullptr || request.aov_count <= 0) {
        error = "no AOVs requested";
        return std::nullopt;
    }
    std::vector<AovId> aovs;
    aovs.reserve(static_cast<std::size_t>(request.aov_count));
    for (int i = 0; i < request.aov_count; ++i) {
        if (!validAov(request.aovs[i])) {
            error = "AOV id " + std::to_string(request.aovs[i]) + " is out of range";
            return std::nullopt;
        }
        aovs.push_back(static_cast<AovId>(request.aovs[i]));
    }
    std::optional<bool> envLightEnabled;
    std::optional<bool> showSky;
    if (!toOptionalBool(request.env_light_enabled, "env_light_enabled", envLightEnabled, error) ||
        !toOptionalBool(request.show_sky, "show_sky", showSky, error)) {
        return std::nullopt;
    }
    const std::optional<pathtracer::scene::Camera> camera = toCamera(request.camera, error);
    if (!camera.has_value()) {
        return std::nullopt;
    }
    std::optional<pathtracer::scene::Camera> previousCamera;
    if (request.previous_camera != nullptr) {
        previousCamera = toCamera(*request.previous_camera, error);
        if (!previousCamera.has_value()) {
            error = "previous_camera: " + error;
            return std::nullopt;
        }
    }
    for (int i = 0; i < request.aov_count; ++i) {
        if (out[i] == nullptr) {
            error = "output buffer " + std::to_string(i) + " is null";
            return std::nullopt;
        }
    }
    return HeadlessRenderer::Request{
        .camera = *camera,
        .previousCamera = previousCamera,
        .width = request.width,
        .height = request.height,
        .samples = request.samples,
        .scrambleSeed = request.seed,
        .aovs = std::move(aovs),
        .envLightEnabled = envLightEnabled,
        .showSky = showSky,
        .rootRotationDegrees = request.root_rotation_degrees != nullptr
                                   ? std::optional(glm::make_vec3(request.root_rotation_degrees))
                                   : std::nullopt,
        .lightRotationsDegrees = toLightRotations(request.light_rotation_degrees, lightCount),
        .envRotationDegrees = glm::make_vec3(request.env_rotation_degrees),
    };
}

}  // namespace

extern "C" {

PtRenderer* pt_renderer_open(const char* asset_root, const char* scene_path, char* err, int err_cap) {
    try {
        if (scene_path == nullptr) {
            writeError(err, err_cap, "scene_path is null");
            return nullptr;
        }
        const std::string assetRoot = asset_root != nullptr ? asset_root : ASSET_ROOT_DIR;
        std::string error;
        std::unique_ptr<HeadlessRenderer> renderer = HeadlessRenderer::open(assetRoot, scene_path, error);
        if (!renderer) {
            writeError(err, err_cap, error);
            return nullptr;
        }
        return reinterpret_cast<PtRenderer*>(renderer.release());
    } catch (const std::exception& e) {
        writeError(err, err_cap, e.what());
        return nullptr;
    } catch (...) {
        writeError(err, err_cap, "unknown error opening scene");
        return nullptr;
    }
}

void pt_renderer_close(PtRenderer* renderer) {
    // Raw delete is the C ABI's ownership contract: pt_renderer_open released a unique_ptr into the caller's hands.

    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete reinterpret_cast<HeadlessRenderer*>(renderer);
}

int pt_aov_count(void) { return static_cast<int>(AovId::Count); }

const char* pt_aov_name(int aov) {
    return validAov(aov) ? pathtracer::debug::kAovNames[aov] : nullptr;
}

int pt_aov_id(const char* name) {
    if (name == nullptr) {
        return -1;
    }
    const AovId aov = pathtracer::debug::aovIdFromName(name);
    return aov == AovId::Count ? -1 : static_cast<int>(aov);
}

int pt_aov_channels(int aov) {
    return validAov(aov) ? pathtracer::debug::aovChannels(static_cast<AovId>(aov)) : -1;
}

int pt_aov_needs_samples(int aov) {
    return validAov(aov) && pathtracer::debug::aovNeedsLightTransport(static_cast<AovId>(aov)) ? 1 : 0;
}

void pt_renderer_default_camera(const PtRenderer* renderer, PtCamera* out) {
    if (renderer == nullptr || out == nullptr) {
        return;
    }
    const pathtracer::scene::Camera& camera =
        reinterpret_cast<const HeadlessRenderer*>(renderer)->defaultCamera();
    const glm::vec3 position = camera.position();
    const pathtracer::scene::Camera::FilmBack filmBack = camera.filmBack();
    out->position[0] = position.x;
    out->position[1] = position.y;
    out->position[2] = position.z;
    out->yaw_degrees = camera.yawDegrees();
    out->pitch_degrees = camera.pitchDegrees();
    out->film_back_mm[0] = filmBack.widthMm;
    out->film_back_mm[1] = filmBack.heightMm;
    out->focal_length_mm = camera.focalLengthMm();
    out->near_clip = camera.nearClip();
    out->far_clip = camera.farClip();
    out->aperture = camera.aperture();
    out->shutter_seconds = camera.shutterSeconds();
    out->iso = camera.iso();
    const pathtracer::scene::Lens lens = camera.lens();
    // The outbound write is a cast, so the ABI's constants and the enum are asserted equal rather than re-mapped by hand.
    static_assert(static_cast<int>(pathtracer::scene::LensProjection::Rectilinear) == PT_LENS_RECTILINEAR &&
                      static_cast<int>(pathtracer::scene::LensProjection::FisheyePolynomial) == PT_LENS_FISHEYE_POLYNOMIAL &&
                      static_cast<int>(pathtracer::scene::LensProjection::Omnidirectional) == PT_LENS_OMNIDIRECTIONAL,
                  "PT_LENS_* must stay index-parallel with LensProjection");
    out->lens_projection = static_cast<int>(lens.projection);
    for (std::size_t i = 0; i < lens.radialCoefficients.size(); ++i) {
        out->fisheye_coefficients[i] = lens.radialCoefficients[i];
    }
    out->fisheye_field_of_view_degrees = lens.maxFieldOfViewDegrees;
}

int pt_abi_version(void) {
    return PT_ABI_VERSION;
}

int pt_renderer_default_width(const PtRenderer* renderer) {
    return renderer == nullptr ? 0 : reinterpret_cast<const HeadlessRenderer*>(renderer)->defaultWidth();
}

int pt_renderer_default_height(const PtRenderer* renderer) {
    return renderer == nullptr ? 0 : reinterpret_cast<const HeadlessRenderer*>(renderer)->defaultHeight();
}

int pt_renderer_light_count(const PtRenderer* renderer) {
    return renderer == nullptr ? 0 : static_cast<int>(reinterpret_cast<const HeadlessRenderer*>(renderer)->scene().lights.size());
}

void pt_renderer_default_root_rotation(const PtRenderer* renderer, float* out) {
    if (renderer == nullptr || out == nullptr) {
        return;
    }
    const glm::vec3& rotation = reinterpret_cast<const HeadlessRenderer*>(renderer)->scene().model.rotation;
    std::copy_n(glm::value_ptr(rotation), 3, out);
}

void pt_renderer_default_light_rotations(const PtRenderer* renderer, float* out) {
    if (renderer == nullptr || out == nullptr) {
        return;
    }
    for (const pathtracer::config::QuadLightConfig& light : reinterpret_cast<const HeadlessRenderer*>(renderer)->scene().lights) {
        out = std::copy_n(glm::value_ptr(light.rotation), 3, out);
    }
}

int pt_render(PtRenderer* renderer, const PtRenderRequest* request, float* const* out, char* err,
              int err_cap) {
    try {
        if (renderer == nullptr || request == nullptr || out == nullptr) {
            writeError(err, err_cap, "null renderer, request or output");
            return PT_ERROR;
        }
        std::string error;
        auto* const headless = reinterpret_cast<HeadlessRenderer*>(renderer);
        const std::optional<HeadlessRenderer::Request> internal =
            decodeRequest(*request, out, headless->scene().lights.size(), error);
        if (!internal.has_value()) {
            writeError(err, err_cap, error);
            return PT_ERROR;
        }
        const std::span<float* const> outputs(out, static_cast<std::size_t>(request->aov_count));
        if (!headless->render(*internal, outputs, error)) {
            writeError(err, err_cap, error);
            return PT_ERROR;
        }
        return PT_OK;
    } catch (const std::exception& e) {
        writeError(err, err_cap, e.what());
        return PT_ERROR;
    } catch (...) {
        writeError(err, err_cap, "unknown error during render");
        return PT_ERROR;
    }
}

int pt_display_encode(const float* rgb, int width, int height, float exposure_ev, int display_transform,
                      unsigned char* out, char* err, int err_cap) {
    try {
        if (rgb == nullptr || out == nullptr) {
            writeError(err, err_cap, "null input or output buffer");
            return PT_ERROR;
        }
        if (width <= 0 || height <= 0) {
            writeError(err, err_cap, "resolution must be positive");
            return PT_ERROR;
        }
        const auto count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3;
        // Caller-driven with no AOV context, so the map stays a scalar photographic gain with no offset, as this entry point always was.
        pathtracer::gfx::encodeForDisplay(std::span<const float>(rgb, count), width, height, glm::vec3(std::pow(2.0F, exposure_ev)),
                                          display_transform != 0, glm::vec3(0.0F), std::span<unsigned char>(out, count));
        return PT_OK;
    } catch (const std::exception& e) {
        writeError(err, err_cap, e.what());
        return PT_ERROR;
    } catch (...) {
        writeError(err, err_cap, "unknown error during display encode");
        return PT_ERROR;
    }
}

}  // extern "C"
