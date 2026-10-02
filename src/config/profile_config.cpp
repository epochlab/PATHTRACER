#include "pathtracer/config/profile_config.h"

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include "pathtracer/debug/aov.h"
#include "json_glm.h"

namespace pathtracer::config {

namespace {

std::optional<pathtracer::gfx::OcioDisplayTransform::Lut> parseLut(const std::string& name) {
    using Lut = pathtracer::gfx::OcioDisplayTransform::Lut;
    if (name == "sRGB") {
        return Lut::SRGB;
    }
    if (name == "Rec709") {
        return Lut::Rec709;
    }
    if (name == "Raw") {
        return Lut::Raw;
    }
    return std::nullopt;
}

// Integer-typed first: get<int>() would silently truncate 16.5 to 16.
std::optional<pathtracer::gfx::ScalarType> parseBitDepth(const nlohmann::json& bitDepth) {
    return bitDepth.is_number_integer() ? pathtracer::gfx::scalarTypeFromBitDepth(bitDepth.get<int>()) : std::nullopt;
}

// Camera lens block: projection, Kannala-Brandt k1..k4 and field of view, rejected unless r(theta) is invertible over it.
std::optional<pathtracer::scene::Lens> parseLens(const nlohmann::json& lens, const std::string& path) {
    const std::string projection = lens.at("projection").get<std::string>();
    pathtracer::scene::Lens parsed;
    if (projection == "rectilinear") {
        parsed.projection = pathtracer::scene::LensProjection::Rectilinear;
    } else if (projection == "fisheyePolynomial") {
        parsed.projection = pathtracer::scene::LensProjection::FisheyePolynomial;
    } else {
        std::cerr << "loadProfileConfig: " << path << " has lens.projection " << projection
                  << ", expected rectilinear or fisheyePolynomial\n";
        return std::nullopt;
    }
    const nlohmann::json& coefficients = lens.at("radialCoefficients");
    if (!coefficients.is_array() || coefficients.size() != parsed.radialCoefficients.size()) {
        std::cerr << "loadProfileConfig: " << path << " has lens.radialCoefficients " << coefficients.dump()
                  << ", expected four numbers k1..k4\n";
        return std::nullopt;
    }
    for (std::size_t i = 0; i < parsed.radialCoefficients.size(); ++i) {
        parsed.radialCoefficients[i] = coefficients[i].get<float>();
    }
    parsed.maxFieldOfViewDegrees = lens.at("maxFieldOfViewDegrees").get<float>();
    // Half of it is thetaMax, the polynomial's domain: at or below zero the lens images nothing, past 360 the circle wraps twice.
    if (parsed.maxFieldOfViewDegrees <= 0.0F || parsed.maxFieldOfViewDegrees > 360.0F) {
        std::cerr << "loadProfileConfig: " << path << " has a lens.maxFieldOfViewDegrees outside (0, 360]\n";
        return std::nullopt;
    }
    // A non-monotone r(theta) has no unique inverse, so primaryRay could not recover the angle a sensor radius came from.
    if (!pathtracer::scene::kannalaBrandtIsInvertible(parsed.radialCoefficients, maxThetaRadians(parsed))) {
        std::cerr << "loadProfileConfig: " << path
                  << " has lens.radialCoefficients whose r(theta) is not provably monotone over the field of view\n";
        return std::nullopt;
    }
    return parsed;
}

// The counts loadProfileConfig otherwise takes on trust. Nothing downstream re-checks them, and each has a concrete failure mode.
bool validCounts(const RenderConfig& render, const PathTracerConfig& pathTracer, const std::string& path) {
    bool ok = true;
    const auto atLeast = [&](const char* name, int v, int low) {
        if (v < low) {
            std::cerr << "loadProfileConfig: " << path << ": " << name << " is " << v << ", expected >= " << low << "\n";
            ok = false;
        }
    };
    // At zero the per-sample loop never runs, so renderPathTraced divides by a zero filter weight and writes NaN to every AOV texel.
    atLeast("samplesPerPixel", pathTracer.samplesPerPixel, 1);
    // Zero is direct lighting only, a legitimate render; negative makes the depth cap reject the primary hit before it is shaded.
    atLeast("maxBounces", pathTracer.maxBounces, 0);
    // Negative starts roulette before the primary ray, killing paths at full throughput.
    atLeast("russianRouletteStartBounce", pathTracer.russianRouletteStartBounce, 0);
    // Zero is the documented unbounded case; negative would cap accumulation below the first pass.
    atLeast("maxSamples", pathTracer.maxSamples, 0);
    // The traced image the render scale multiplies, and the denominator of the primary ray's aspect ratio.
    atLeast("render.width", render.width, 1);
    atLeast("render.height", render.height, 1);
    return ok;
}

// The render block's output fields: display LUT, the two bit depths and the image size. Scales, AOV and vsync are read after controls.
std::optional<RenderConfig> parseRenderOutput(const nlohmann::json& render, const std::string& path) {
    const std::string defaultLutName = render.at("defaultLUT").get<std::string>();
    const std::optional<pathtracer::gfx::OcioDisplayTransform::Lut> defaultLut =
        parseLut(defaultLutName);
    if (!defaultLut.has_value()) {
        std::cerr << "loadProfileConfig: " << path << " has an unrecognised defaultLUT \""
                   << defaultLutName << "\"\n";
        return std::nullopt;
    }

    const nlohmann::json& displayBitDepth = render.at("displayBitDepth");
    const nlohmann::json& textureBitDepth = render.at("textureBitDepth");
    const std::optional<pathtracer::gfx::ScalarType> displayFormat = parseBitDepth(displayBitDepth);
    const std::optional<pathtracer::gfx::ScalarType> textureType = parseBitDepth(textureBitDepth);
    if (!displayFormat.has_value() || !textureType.has_value()) {
        std::cerr << "loadProfileConfig: " << path << " has displayBitDepth " << displayBitDepth.dump()
                   << ", textureBitDepth " << textureBitDepth.dump() << ", each expected 16 or 32\n";
        return std::nullopt;
    }
    const int width = render.at("width").get<int>();
    const int height = render.at("height").get<int>();
    return RenderConfig{width, height, 0.0F, 0.0F, 0, *defaultLut, false, *displayFormat, *textureType};
}

// The camera block in field order, then its lens. Ranges are validCamera's, run once every block has been read.
std::optional<CameraConfig> parseCamera(const nlohmann::json& camera, const std::string& path) {
    CameraConfig config{
        camera.at("position").get<glm::vec3>(),
        camera.at("yawDegrees").get<float>(),
        camera.at("pitchDegrees").get<float>(),
        camera.at("filmBackPreset").get<std::string>(),
        camera.at("focalLengthMm").get<float>(),
        camera.at("nearClip").get<float>(),
        camera.at("farClip").get<float>(),
        camera.at("aperture").get<float>(),
        camera.at("shutterSeconds").get<float>(),
        camera.at("iso").get<float>(),
        {},
    };
    const std::optional<pathtracer::scene::Lens> lens = parseLens(camera.at("lens"), path);
    if (!lens.has_value()) {
        return std::nullopt;
    }
    config.lens = *lens;
    return config;
}

// Denominators in verticalFovRadians() and ev100(): a non-positive value gives inf or NaN, not a wrong-but-finite render.
bool validCamera(const CameraConfig& camera, const std::string& path) {
    if (camera.focalLengthMm <= 0.0F || camera.aperture <= 0.0F || camera.shutterSeconds <= 0.0F || camera.iso <= 0.0F) {
        std::cerr << "loadProfileConfig: " << path
                   << " has a non-positive focalLengthMm/aperture/shutterSeconds/iso\n";
        return false;
    }
    return true;
}

bool validRender(const RenderConfig& render, const std::string& path) {
    // Bounded at (0,1], not merely positive: above 1 renders past the framebuffer, at or below 0 the render target has no pixels.
    if (render.renderScale <= 0.0F || render.renderScale > 1.0F || render.interactiveRenderScale <= 0.0F ||
        render.interactiveRenderScale > 1.0F) {
        std::cerr << "loadProfileConfig: " << path
                   << " has a renderScale/interactiveRenderScale outside (0,1]\n";
        return false;
    }
    // A raw index into kAovNames, dereferenced unchecked by the spec block, so out of range is an out-of-bounds read.
    if (render.defaultAov < 0 || render.defaultAov >= static_cast<int>(pathtracer::debug::AovId::Count)) {
        std::cerr << "loadProfileConfig: " << path << " has a defaultAOV outside [0, "
                   << static_cast<int>(pathtracer::debug::AovId::Count) - 1 << "]\n";
        return false;
    }
    return true;
}

bool validPathTracer(const PathTracerConfig& pathTracer, const std::string& path) {
    // AO ray tfar. At or below zero every occlusion ray is degenerate and the AO lane reads a uniform 1.0: inert white, not an error.
    if (pathTracer.aoMaxDistance <= 0.0F) {
        std::cerr << "loadProfileConfig: " << path << " has a non-positive aoMaxDistance\n";
        return false;
    }
    // The Lookahead ramp divisor: at or below zero every covered pixel divides by it, giving inf or NaN, not the [0,1] gradient.
    if (pathTracer.lookaheadDistance <= 0.0F) {
        std::cerr << "loadProfileConfig: " << path << " has a non-positive lookaheadDistance\n";
        return false;
    }
    return true;
}

}  // namespace

// Reads every block before any range check, so a file with several faults reports the first in read order, then in check order.
std::optional<ProfileConfig> loadProfileConfig(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "loadProfileConfig: could not read " << path << '\n';
        return std::nullopt;
    }

    try {
        nlohmann::json j;
        file >> j;

        const nlohmann::json& camera = j.at("camera");
        const nlohmann::json& controls = j.at("controls");
        const nlohmann::json& render = j.at("render");
        const nlohmann::json& pathTracer = j.at("pathTracer");

        std::optional<RenderConfig> renderConfig = parseRenderOutput(render, path);
        if (!renderConfig.has_value()) {
            return std::nullopt;
        }
        const std::optional<CameraConfig> cameraConfig = parseCamera(camera, path);
        if (!cameraConfig.has_value()) {
            return std::nullopt;
        }
        const ControlsConfig controlsConfig{
            controls.at("flySpeedMetersPerSecond").get<float>(),
            controls.at("orbitSensitivityDegPerPixel").get<float>(),
        };
        renderConfig->renderScale = render.at("renderScale").get<float>();
        renderConfig->interactiveRenderScale = render.at("interactiveRenderScale").get<float>();
        renderConfig->defaultAov = render.at("defaultAOV").get<int>();
        renderConfig->vsync = render.at("vsync").get<bool>();
        const PathTracerConfig pathTracerConfig{
            pathTracer.at("samplesPerPixel").get<int>(),
            pathTracer.at("maxBounces").get<int>(),
            pathTracer.at("russianRouletteStartBounce").get<int>(),
            pathTracer.at("maxSamples").get<int>(),
            pathTracer.at("aoMaxDistance").get<float>(),
            pathTracer.at("lookaheadDistance").get<float>(),
        };

        if (!validCamera(*cameraConfig, path) || !validRender(*renderConfig, path) ||
            !validPathTracer(pathTracerConfig, path) || !validCounts(*renderConfig, pathTracerConfig, path)) {
            return std::nullopt;
        }
        return ProfileConfig{*cameraConfig, controlsConfig, *renderConfig, pathTracerConfig};
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "loadProfileConfig: " << path << ": " << e.what() << '\n';
        return std::nullopt;
    }
}

std::optional<std::vector<pathtracer::scene::Camera::FilmBackPreset>> loadFilmBackPresets(
    const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "loadFilmBackPresets: could not read " << path << '\n';
        return std::nullopt;
    }

    try {
        nlohmann::json j;
        file >> j;

        std::vector<pathtracer::scene::Camera::FilmBackPreset> presets;
        presets.reserve(j.size());
        for (const nlohmann::json& presetJson : j) {
            std::string name = presetJson.at("name").get<std::string>();
            const float widthMm = presetJson.at("widthMm").get<float>();
            const float heightMm = presetJson.at("heightMm").get<float>();
            // Physical sensor dimensions feeding verticalFovRadians() and the HUD aspect as denominators: non-positive is inf or NaN.
            if (widthMm <= 0.0F || heightMm <= 0.0F) {
                std::cerr << "loadFilmBackPresets: " << path << " has a non-positive filmBack for \""
                           << name << "\"\n";
                return std::nullopt;
            }
            presets.push_back({std::move(name), pathtracer::scene::Camera::FilmBack{widthMm, heightMm}});
        }
        return presets;
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "loadFilmBackPresets: " << path << ": " << e.what() << '\n';
        return std::nullopt;
    }
}

}  // namespace pathtracer::config
