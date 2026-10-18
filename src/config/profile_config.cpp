#include "pathtracer/config/profile_config.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iterator>
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

// profile.json's denoiser block; quality is one of kDenoiseQualityNames.
std::optional<DenoiserConfig> parseDenoiser(const nlohmann::json& denoiser, const std::string& path) {
    const std::string quality = denoiser.at("quality").get<std::string>();
    const auto* const names = std::begin(pathtracer::debug::kDenoiseQualityNames);
    const auto* const found = std::find(names, std::end(pathtracer::debug::kDenoiseQualityNames), quality);
    if (found == std::end(pathtracer::debug::kDenoiseQualityNames)) {
        std::cerr << "loadProfileConfig: " << path << " has an unrecognised denoiser quality \"" << quality << "\", expected fast, balanced or high\n";
        return std::nullopt;
    }
    return DenoiserConfig{denoiser.at("enabled").get<bool>(),
                          {static_cast<pathtracer::debug::DenoiseQuality>(found - names), denoiser.at("prefilterAuxiliary").get<bool>()}};
}

// Integer-typed first: get<int>() would silently truncate 16.5 to 16.
std::optional<pathtracer::gfx::ScalarType> parseBitDepth(const nlohmann::json& bitDepth) {
    return bitDepth.is_number_integer() ? pathtracer::gfx::scalarTypeFromBitDepth(bitDepth.get<int>()) : std::nullopt;
}

// Camera lens block: projection, Kannala-Brandt k1..k4 and field of view. Ranges are Camera::validate's, run once the camera is assembled.
std::optional<pathtracer::scene::Lens> parseLens(const nlohmann::json& lens, const std::string& path) {
    const std::string projection = lens.at("projection").get<std::string>();
    pathtracer::scene::Lens parsed;
    if (projection == "rectilinear") {
        parsed.projection = pathtracer::scene::LensProjection::Rectilinear;
    } else if (projection == "fisheyePolynomial") {
        parsed.projection = pathtracer::scene::LensProjection::FisheyePolynomial;
    } else if (projection == "omnidirectional") {
        parsed.projection = pathtracer::scene::LensProjection::Omnidirectional;
    } else {
        std::cerr << "loadProfileConfig: " << path << " has lens.projection " << projection
                  << ", expected rectilinear, fisheyePolynomial or omnidirectional\n";
        return std::nullopt;
    }
    const nlohmann::json& coefficients = lens.at("radialCoefficients");
    if (!coefficients.is_array() || coefficients.size() != parsed.radialCoefficients.size()) {
        std::cerr << "loadProfileConfig: " << path << " has lens.radialCoefficients " << coefficients.dump()
                  << ", expected four numbers k1..k4\n";
        return std::nullopt;
    }
    for (std::size_t i = 0; i < parsed.radialCoefficients.size(); ++i) {
        parsed.radialCoefficients[i] = toFloat(coefficients[i]);
    }
    parsed.maxFieldOfViewDegrees = toFloat(lens.at("maxFieldOfViewDegrees"));
    return parsed;
}

// Camera pixelLattice, spelled as the lens projections are; the HUD switches it at runtime as it does the projection.
std::optional<pathtracer::gfx::PixelLattice> parsePixelLattice(const nlohmann::json& lattice, const std::string& path) {
    const std::string name = lattice.get<std::string>();
    if (name == "square") {
        return pathtracer::gfx::PixelLattice::Square;
    }
    if (name == "hexagonal") {
        return pathtracer::gfx::PixelLattice::Hexagonal;
    }
    std::cerr << "loadProfileConfig: " << path << " has camera.pixelLattice " << name << ", expected square or hexagonal\n";
    return std::nullopt;
}

// The probe's readout by name; nullopt logs any other.
std::optional<pathtracer::debug::PixelProbeReadout> parsePixelProbe(const nlohmann::json& probe, const std::string& path) {
    const std::string name = probe.get<std::string>();
    if (name == "colour") {
        return pathtracer::debug::PixelProbeReadout::Colour;
    }
    if (name == "photometric") {
        return pathtracer::debug::PixelProbeReadout::Photometric;
    }
    if (name == "none") {
        return pathtracer::debug::PixelProbeReadout::None;
    }
    std::cerr << "loadProfileConfig: " << path << " has render.pixelProbe " << name << ", expected colour, photometric or none\n";
    return std::nullopt;
}

// The render block's output fields: display LUT, bit depth, image size, bloom and probe. Scales, AOV and vsync are read after controls.
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
    const std::optional<pathtracer::gfx::ScalarType> displayFormat = parseBitDepth(displayBitDepth);
    if (!displayFormat.has_value()) {
        std::cerr << "loadProfileConfig: " << path << " has displayBitDepth " << displayBitDepth.dump() << ", expected 16 or 32\n";
        return std::nullopt;
    }
    // Textures keep their own format in the texture cache: a profile-wide texture depth is retired, not ignored.
    if (render.contains("textureBitDepth")) {
        std::cerr << "loadProfileConfig: " << path << " sets the retired textureBitDepth; textures keep their own format\n";
        return std::nullopt;
    }
    const std::optional<pathtracer::debug::PixelProbeReadout> pixelProbe = parsePixelProbe(render.at("pixelProbe"), path);
    if (!pixelProbe.has_value()) {
        return std::nullopt;
    }
    const int width = render.at("width").get<int>();
    const int height = render.at("height").get<int>();
    return RenderConfig{width, height, 0.0F, 0.0F, 0, *defaultLut, false, *displayFormat, toFloat(render.at("bloomStrength")), *pixelProbe};
}

// The camera block in field order, then lens, lattice and acceptance. Ranges need the film back, so Camera::validate runs on the whole.
std::optional<CameraConfig> parseCamera(const nlohmann::json& camera, const std::string& path) {
    CameraConfig config{
        camera.at("position").get<glm::vec3>(),
        camera.at("rotation").get<glm::vec3>(),
        camera.at("filmBackPreset").get<std::string>(),
        toFloat(camera.at("focalLengthMm")),
        toFloat(camera.at("nearClip")),
        toFloat(camera.at("farClip")),
        toFloat(camera.at("aperture")),
        toFloat(camera.at("shutterSeconds")),
        toFloat(camera.at("iso")),
        {},
        {},
        toFloat(camera.at("acceptanceAngleDegrees")),
    };
    const std::optional<pathtracer::scene::Lens> lens = parseLens(camera.at("lens"), path);
    const std::optional<pathtracer::gfx::PixelLattice> lattice = parsePixelLattice(camera.at("pixelLattice"), path);
    if (!lens.has_value() || !lattice.has_value()) {
        return std::nullopt;
    }
    config.lens = *lens;
    config.pixelLattice = *lattice;
    return config;
}

// The viewer would open on an AOV no producer serves: denoised with the denoiser off, or on a lattice its network was not trained on.
bool servedDefaultAov(int defaultAov, const DenoiserConfig& denoiser, const CameraConfig& camera, const std::string& path) {
    if (!denoiser.enabled && defaultAov == static_cast<int>(pathtracer::debug::AovId::Denoised)) {
        std::cerr << "loadProfileConfig: " << path << ": defaultAOV is denoised, but the denoiser is disabled\n";
        return false;
    }
    if (!pathtracer::debug::aovSupportsLattice(static_cast<pathtracer::debug::AovId>(defaultAov), camera.pixelLattice)) {
        std::cerr << "loadProfileConfig: " << path << ": defaultAOV's network is trained on square images, not the camera's pixelLattice\n";
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
    // A fraction of energy: below 0 the mix extrapolates away from the glare, above 1 it subtracts the direct image.
    if (!(render.bloomStrength >= 0.0F && render.bloomStrength <= 1.0F)) {
        std::cerr << "loadProfileConfig: " << path << " has a bloomStrength outside [0,1]\n";
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

// Every key required, as everywhere in this file: a missing one throws and is reported by loadProfileConfig's catch.
PathTracerConfig parsePathTracer(const nlohmann::json& pathTracer) {
    const nlohmann::json& adaptive = pathTracer.at("adaptive");
    return PathTracerConfig{
        pathTracer.at("samplesPerPixel").get<int>(),
        pathTracer.at("maxBounces").get<int>(),
        pathTracer.at("russianRouletteStartBounce").get<int>(),
        pathTracer.at("maxSamples").get<int>(),
        toFloat(pathTracer.at("aoMaxDistance")),
        toFloat(pathTracer.at("lookaheadDistance")),
        pathTracer.at("replicates").get<int>(),
        toFloat(adaptive.at("threshold")),
        toFloat(adaptive.at("confidence")),
        adaptive.at("minSamples").get<int>(),
    };
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
    // Negative or NaN would stop nothing or everything; infinity would stop every texel at its first checkpoint.
    if (!std::isfinite(pathTracer.adaptiveThreshold) || pathTracer.adaptiveThreshold < 0.0F) {
        std::cerr << "loadProfileConfig: " << path << " has an adaptive.threshold outside [0, inf)\n";
        return false;
    }
    // The false discovery rate is 1 - confidence: at confidence 1 nothing could stop, at 0 every texel with spread would.
    if (!(pathTracer.adaptiveConfidence > 0.0F && pathTracer.adaptiveConfidence < 1.0F)) {
        std::cerr << "loadProfileConfig: " << path << " has an adaptive.confidence outside (0, 1)\n";
        return false;
    }
    return true;
}

}  // namespace

// Each count has a concrete failure mode, and the viewer overrides some after loading, so this runs at both boundaries.
bool validCounts(const RenderConfig& render, const PathTracerConfig& pathTracer, const std::string& source) {
    bool ok = true;
    const auto atLeast = [&](const char* name, int v, int low) {
        if (v < low) {
            std::cerr << "loadProfileConfig: " << source << ": " << name << " is " << v << ", expected >= " << low << "\n";
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
    // One replicate is one sequence, a valid render whose SNR is undefined; zero would divide the pass index.
    atLeast("replicates", pathTracer.replicates, 1);
    // A capped accumulation must end on a complete round, or its replicates settle at unequal counts.
    if (pathTracer.replicates >= 1 && pathTracer.maxSamples % pathTracer.replicates != 0) {
        std::cerr << "loadProfileConfig: " << source << ": maxSamples " << pathTracer.maxSamples << " is not a multiple of replicates "
                  << pathTracer.replicates << "\n";
        ok = false;
    }
    // Checkpoints fall where every replicate holds a power-of-two prefix of its sequence, so the first must be one of them.
    const int perReplicate = pathTracer.replicates >= 1 ? pathTracer.adaptiveMinSamples / pathTracer.replicates : 0;
    if (pathTracer.adaptiveMinSamples < pathTracer.replicates || pathTracer.adaptiveMinSamples % pathTracer.replicates != 0 ||
        !std::has_single_bit(static_cast<unsigned int>(perReplicate))) {
        std::cerr << "loadProfileConfig: " << source << ": adaptive.minSamples " << pathTracer.adaptiveMinSamples
                  << " is not replicates " << pathTracer.replicates << " times a power of two\n";
        ok = false;
    }
    // One replicate leaves no spread to bound the error by, so no texel could ever stop.
    if (pathTracer.adaptiveThreshold > 0.0F && pathTracer.replicates < 2) {
        std::cerr << "loadProfileConfig: " << source << ": adaptive sampling needs replicates >= 2, has " << pathTracer.replicates << "\n";
        ok = false;
    }
    // The traced image the render scale multiplies, and the denominator of the primary ray's aspect ratio.
    atLeast("render.width", render.width, 1);
    atLeast("render.height", render.height, 1);
    return ok;
}

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
        const std::optional<DenoiserConfig> denoiserConfig = parseDenoiser(j.at("denoiser"), path);
        if (!denoiserConfig.has_value()) {
            return std::nullopt;
        }

        std::optional<RenderConfig> renderConfig = parseRenderOutput(render, path);
        if (!renderConfig.has_value()) {
            return std::nullopt;
        }
        const std::optional<CameraConfig> cameraConfig = parseCamera(camera, path);
        if (!cameraConfig.has_value()) {
            return std::nullopt;
        }
        const ControlsConfig controlsConfig{
            toFloat(controls.at("flySpeedMetersPerSecond")),
            toFloat(controls.at("orbitSensitivityDegPerPixel")),
        };
        renderConfig->renderScale = toFloat(render.at("renderScale"));
        renderConfig->interactiveRenderScale = toFloat(render.at("interactiveRenderScale"));
        renderConfig->defaultAov = render.at("defaultAOV").get<int>();
        renderConfig->vsync = render.at("vsync").get<bool>();
        const PathTracerConfig pathTracerConfig = parsePathTracer(pathTracer);

        if (!validRender(*renderConfig, path) ||
            !validPathTracer(pathTracerConfig, path) || !validCounts(*renderConfig, pathTracerConfig, path)) {
            return std::nullopt;
        }
        if (!servedDefaultAov(renderConfig->defaultAov, *denoiserConfig, *cameraConfig, path)) {
            return std::nullopt;
        }
        return ProfileConfig{*cameraConfig, controlsConfig, *renderConfig, pathTracerConfig, *denoiserConfig};
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
            const pathtracer::scene::Camera::FilmBack filmBack{toFloat(presetJson.at("widthMm")),
                                                               toFloat(presetJson.at("heightMm"))};
            // Each preset checked alone: the HUD can swap any of them into the camera, not just profile.json's.
            if (!pathtracer::scene::Camera::validFilmBack(filmBack)) {
                std::cerr << "loadFilmBackPresets: " << path << " has a non-finite or non-positive filmBack for \""
                           << name << "\"\n";
                return std::nullopt;
            }
            presets.push_back({std::move(name), filmBack});
        }
        return presets;
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "loadFilmBackPresets: " << path << ": " << e.what() << '\n';
        return std::nullopt;
    }
}

}  // namespace pathtracer::config
