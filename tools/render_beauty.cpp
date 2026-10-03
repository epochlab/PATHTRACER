// Headless beauty render for before/after comparison, writing one AOV as PNG.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <OpenColorIO/OpenColorIO.h>
#include <glm/gtc/matrix_transform.hpp>
#include <zlib.h>

#include "pathtracer/api/headless_renderer.h"
#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/bench_log.h"
#include "pathtracer/debug/power_spectrum.h"
#include "pathtracer/debug/render_stats.h"
#include "check.h"
#include "stats.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/ocio_cpu_transform.h"
#include "pathtracer/gfx/ocio_display_transform.h"
#include "pathtracer/scene/camera.h"

namespace OCIO = OCIO_NAMESPACE;

namespace {

struct Options {
    std::string scenePath = "scenes/cornell.json";
    std::string outPath;
    std::string comparePath;
    // Linear-light companions to --out/--compare: the PNG path is display-transformed 8-bit, so it clamps the bright high-variance regions.
    std::string outExrPath;
    std::string compareExrPath;
    // How --compare-exr's error distributes over frequency: a change that rearranges error without reducing it is invisible to RMSE.
    bool errorSpectrum = false;
    int width = 0;   // 0 = profile.json's window size
    int height = 0;
    int passes = 64;
    // The scramble seed is one realization, not a sampler property: two seeds give independent error images with the same expected RMSE.
    std::uint32_t scrambleSeed = 1;
    float exposureEv = 0.0F;
    // -1 uses the scene's authored environment.lightEnabled; 0/1 override it, so an env-off Cornell capture needs no second scene.json.
    int envLight = -1;
    // Gate modes set a non-zero exit code, which is what makes them ctest entries; the reporting paths above deliberately do not.
    bool assertDeterministic = false;
    // Two independent randomizations of one estimator must agree within their own measured error: no reference image, no tuned threshold.
    bool assertConverged = false;
    // Resolved by --aov. Defaulting to Beauty keeps every existing invocation -- and the bit-identity gate built on them -- unchanged.
    pathtracer::debug::AovId aov = pathtracer::debug::AovId::Beauty;
    // Appends the timing run to this JSON Lines benchmark log (bench_log.h); empty = no log.
    std::string benchLogPath;
    // Positive: profile.json's polynomial fisheye at this focal length, so a capture can place the image circle inside the gate.
    float fisheyeFocalLengthMm = 0.0F;
};

// All AOVs are reachable now HeadlessRenderer drives the G-buffer and filters; names come from pathtracer/debug/aov.h, not restated here.
bool resolveAov(const std::string& requested, Options& options) {
    const pathtracer::debug::AovId aov = pathtracer::debug::aovIdFromName(requested);
    if (aov == pathtracer::debug::AovId::Count) {
        std::cerr << "render_beauty: unknown AOV \"" << requested << "\"; known AOVs are:";
        for (int i = 0; i < static_cast<int>(pathtracer::debug::AovId::Count); ++i) {
            std::cerr << ' ' << pathtracer::debug::kAovNames[i];
        }
        std::cerr << '\n';
        return false;
    }
    options.aov = aov;
    return true;
}

// Big-endian u32 append -- PNG is network byte order throughout.
void appendBe32(std::vector<unsigned char>& out, std::uint32_t value) {
    out.push_back(static_cast<unsigned char>((value >> 24) & 0xFFU));
    out.push_back(static_cast<unsigned char>((value >> 16) & 0xFFU));
    out.push_back(static_cast<unsigned char>((value >> 8) & 0xFFU));
    out.push_back(static_cast<unsigned char>(value & 0xFFU));
}

void appendChunk(std::vector<unsigned char>& out, const char* type,
                  const std::vector<unsigned char>& data) {
    appendBe32(out, static_cast<std::uint32_t>(data.size()));
    const std::size_t crcStart = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    const uLong crc = crc32(crc32(0L, Z_NULL, 0), out.data() + crcStart,
                             static_cast<uInt>(out.size() - crcStart));
    appendBe32(out, static_cast<std::uint32_t>(crc));
}

// Minimal 8-bit RGB PNG writer: zlib arrives transitively with OpenEXR, so a debug artifact needs no vendored image library.
bool writePng(const std::string& path, int width, int height,
               const std::vector<unsigned char>& rgb) {
    // Each scanline is prefixed with its filter byte; 0 = None, which compresses adequately here and keeps the encoder trivial.
    std::vector<unsigned char> raw;
    raw.reserve((static_cast<std::size_t>(width) * 3 + 1) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        const std::size_t rowStart = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 3;
        raw.insert(raw.end(), rgb.begin() + static_cast<std::ptrdiff_t>(rowStart),
                    rgb.begin() + static_cast<std::ptrdiff_t>(rowStart) +
                        (static_cast<std::ptrdiff_t>(width) * 3));
    }

    uLongf compressedSize = compressBound(static_cast<uLong>(raw.size()));
    std::vector<unsigned char> compressed(compressedSize);
    if (compress2(compressed.data(), &compressedSize, raw.data(), static_cast<uLong>(raw.size()),
                   Z_BEST_COMPRESSION) != Z_OK) {
        std::cerr << "render_beauty: zlib compression failed\n";
        return false;
    }
    compressed.resize(compressedSize);

    std::vector<unsigned char> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    std::vector<unsigned char> ihdr;
    appendBe32(ihdr, static_cast<std::uint32_t>(width));
    appendBe32(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(2);  // colour type 2 = truecolour RGB
    ihdr.push_back(0);  // deflate
    ihdr.push_back(0);  // adaptive filtering
    ihdr.push_back(0);  // no interlace
    appendChunk(png, "IHDR", ihdr);
    appendChunk(png, "IDAT", compressed);
    appendChunk(png, "IEND", {});

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "render_beauty: could not open " << path << " for writing\n";
        return false;
    }
    file.write(reinterpret_cast<const char*>(png.data()),
                static_cast<std::streamsize>(png.size()));
    return file.good();
}

// Reads back a PNG this tool wrote: only writePng's exact IHDR (8-bit, colour type 2, no interlace) and filter 0, anything else rejected.
bool readPng(const std::string& path, int& width, int& height, std::vector<unsigned char>& rgb) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "render_beauty: could not read " << path << '\n';
        return false;
    }
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),
                                            std::istreambuf_iterator<char>());
    if (bytes.size() < 8 || bytes[0] != 0x89 || bytes[1] != 'P') {
        std::cerr << "render_beauty: " << path << " is not a PNG\n";
        return false;
    }
    const auto be32 = [&bytes](std::size_t at) {
        return (static_cast<std::uint32_t>(bytes[at]) << 24) |
               (static_cast<std::uint32_t>(bytes[at + 1]) << 16) |
               (static_cast<std::uint32_t>(bytes[at + 2]) << 8) |
               static_cast<std::uint32_t>(bytes[at + 3]);
    };

    std::vector<unsigned char> idat;
    std::size_t at = 8;
    while (at + 8 <= bytes.size()) {
        const std::uint32_t length = be32(at);
        const std::string type(reinterpret_cast<const char*>(&bytes[at + 4]), 4);
        const std::size_t dataAt = at + 8;
        if (dataAt + length > bytes.size()) {
            break;
        }
        if (type == "IHDR") {
            // The 13-byte payload is indexed directly below; a chunk declaring less would read past the buffer on a truncated file.
            if (length < 13) {
                std::cerr << "render_beauty: " << path << " has a malformed IHDR\n";
                return false;
            }
            width = static_cast<int>(be32(dataAt));
            height = static_cast<int>(be32(dataAt + 4));
            if (bytes[dataAt + 8] != 8 || bytes[dataAt + 9] != 2 || bytes[dataAt + 12] != 0) {
                std::cerr << "render_beauty: " << path
                          << " is not the 8-bit non-interlaced RGB this tool writes\n";
                return false;
            }
        } else if (type == "IDAT") {
            idat.insert(idat.end(), bytes.begin() + static_cast<std::ptrdiff_t>(dataAt),
                         bytes.begin() + static_cast<std::ptrdiff_t>(dataAt + length));
        }
        at = dataAt + length + 4;  // + CRC
    }
    if (width <= 0 || height <= 0 || idat.empty()) {
        std::cerr << "render_beauty: " << path << " has no usable image data\n";
        return false;
    }

    const std::size_t stride = (static_cast<std::size_t>(width) * 3) + 1;
    uLongf rawSize = static_cast<uLongf>(stride * static_cast<std::size_t>(height));
    std::vector<unsigned char> raw(rawSize);
    if (uncompress(raw.data(), &rawSize, idat.data(), static_cast<uLong>(idat.size())) != Z_OK ||
        rawSize != stride * static_cast<std::size_t>(height)) {
        std::cerr << "render_beauty: " << path << " failed to inflate\n";
        return false;
    }
    rgb.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    for (int y = 0; y < height; ++y) {
        if (raw[static_cast<std::size_t>(y) * stride] != 0) {
            std::cerr << "render_beauty: " << path << " uses a PNG filter this tool cannot read\n";
            return false;
        }
        std::memcpy(&rgb[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 3],
                     &raw[(static_cast<std::size_t>(y) * stride) + 1],
                     static_cast<std::size_t>(width) * 3);
    }
    return true;
}

// Each octave band's share of total error power.
void reportErrorSpectrum(const pathtracer::gfx::HdrImage& image, const pathtracer::gfx::HdrImage& reference) {
    const auto pixels = static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height);
    std::vector<double> luminanceError(pixels);
    double squaredSum = 0.0;
    for (std::size_t p = 0; p < pixels; ++p) {
        const glm::vec3 delta = image.rgb(p) - reference.rgb(p);
        const double e = (0.2126 * delta.r) + (0.7152 * delta.g) + (0.0722 * delta.b);
        luminanceError[p] = e;
        squaredSum += e * e;
    }

    const std::array<double, pathtracer::debug::kSpectrumBands> bands =
        pathtracer::debug::octaveBandPower(luminanceError, image.width, image.height);
    // Bands differ widely in width, so the share relative to white noise is what matters; printed alongside, below 1.0 is blue, above red.
    const std::array<double, pathtracer::debug::kSpectrumBands> white =
        pathtracer::debug::whiteNoiseBandShare(image.width, image.height);
    const double total = std::accumulate(bands.begin(), bands.end(), 0.0);
    std::cout << "render_beauty: error spectrum -- luminance RMS "
              << std::sqrt(squaredSum / static_cast<double>(pixels))
              << ", octave bands (low to high), share of total power and ratio to white noise\n";
    for (int band = pathtracer::debug::kSpectrumBands - 1; band >= 0; --band) {
        const double high = 0.5 / std::exp2(band);
        const double share = bands[static_cast<std::size_t>(band)] / total;
        // A band can hold no lattice points at all below 256 px on an axis, and dividing by its share would print nan.
        const double whiteShare = white[static_cast<std::size_t>(band)];
        std::cout << "  " << std::setw(8) << (high * 0.5) << " - " << std::setw(8) << high << " c/px   "
                  << std::setw(8) << (100.0 * share) << " %";
        if (whiteShare > 0.0) {
            std::cout << "   " << std::setw(7) << (share / whiteShare) << "x white";
        }
        std::cout << "\n";
    }
}

// Everything the timed loop's cost depends on goes in config; output paths and exposure do not, so they never split two comparable runs.
bool appendTimingRecord(const Options& options, int argc, char** argv, int width, int height,
                        const std::string& aovName, const pathtracer::scene::PathTraceSettings& settings,
                        bool envLightEnabled, double gbufferMs, double filterMs,
                        pathtracer::gfx::ScalarType textureType, const std::vector<double>& milliseconds, const pathtracer::debug::RayCounts& rays,
                        const pathtracer::gfx::HdrImage& accumulated) {
    pathtracer::debug::BenchRecord record{
        .tool = "render_beauty",
        .argv = std::vector<std::string>(argv, argv + argc),
        .config = {{"scene", options.scenePath},
                   {"width", width},
                   {"height", height},
                   {"passes", options.passes},
                   {"seed", options.scrambleSeed},
                   {"aov", aovName},
                   {"env_light", envLightEnabled},
                   {"spp_per_pass", settings.samplesPerPixel},
                   {"max_bounces", settings.maxBounces},
                   {"rr_start_bounce", settings.russianRouletteStartBounce},
                   {"ao_max_distance", settings.aoMaxDistance},
                   {"texture_type", pathtracer::gfx::scalarTypeName(textureType)},
                   {"fisheye_focal_length_mm", options.fisheyeFocalLengthMm}},
        .samples = {milliseconds.empty() ? std::pair<std::string, std::vector<double>>{"gbuffer_ms", {gbufferMs}}
                                          : std::pair<std::string, std::vector<double>>{"pass_ms", milliseconds}},
        .work = {{"rays", {{"primary", rays.primary}, {"bounce", rays.bounce}, {"ao", rays.ao}, {"shadow", rays.shadow}}},
                 {"crc32", pathtracer::debug::floatCrc32(accumulated.texels)}},
    };
    // A Beauty filter runs once after the passes, so its wall clock is its own column rather than a pass sample.
    if (filterMs > 0.0) {
        record.samples["filter_ms"] = std::vector<double>{filterMs};
    }
    return pathtracer::debug::appendBenchRecord(options.benchLogPath, record);
}

bool parseArgs(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const auto needsValue = [&](const char* flag) {
            if (i + 1 < argc) {
                return true;
            }
            std::cerr << "render_beauty: " << flag << " requires a value\n";
            return false;
        };
        if (std::strcmp(argv[i], "--scene") == 0) {
            if (!needsValue("--scene")) { return false; }
            options.scenePath = argv[++i];
        } else if (std::strcmp(argv[i], "--out") == 0) {
            if (!needsValue("--out")) { return false; }
            options.outPath = argv[++i];
        } else if (std::strcmp(argv[i], "--compare") == 0) {
            if (!needsValue("--compare")) { return false; }
            options.comparePath = argv[++i];
        } else if (std::strcmp(argv[i], "--out-exr") == 0) {
            if (!needsValue("--out-exr")) { return false; }
            options.outExrPath = argv[++i];
        } else if (std::strcmp(argv[i], "--compare-exr") == 0) {
            if (!needsValue("--compare-exr")) { return false; }
            options.compareExrPath = argv[++i];
        } else if (std::strcmp(argv[i], "--error-spectrum") == 0) {
            options.errorSpectrum = true;
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            if (!needsValue("--seed")) { return false; }
            options.scrambleSeed = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--passes") == 0) {
            if (!needsValue("--passes")) { return false; }
            options.passes = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--width") == 0) {
            if (!needsValue("--width")) { return false; }
            options.width = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--height") == 0) {
            if (!needsValue("--height")) { return false; }
            options.height = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--exposure") == 0) {
            if (!needsValue("--exposure")) { return false; }
            options.exposureEv = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--aov") == 0) {
            if (!needsValue("--aov")) { return false; }
            if (!resolveAov(argv[++i], options)) { return false; }
        } else if (std::strcmp(argv[i], "--assert-deterministic") == 0) {
            options.assertDeterministic = true;
        } else if (std::strcmp(argv[i], "--assert-converged") == 0) {
            options.assertConverged = true;
        } else if (std::strcmp(argv[i], "--bench-log") == 0) {
            if (!needsValue("--bench-log")) { return false; }
            options.benchLogPath = argv[++i];
        } else if (std::strcmp(argv[i], "--fisheye") == 0) {
            if (!needsValue("--fisheye")) { return false; }
            options.fisheyeFocalLengthMm = static_cast<float>(std::atof(argv[++i]));
            if (!(options.fisheyeFocalLengthMm > 0.0F)) {
                std::cerr << "render_beauty: --fisheye expects a positive focal length in mm\n";
                return false;
            }
        } else if (std::strcmp(argv[i], "--env-light") == 0) {
            if (!needsValue("--env-light")) { return false; }
            options.envLight = std::atoi(argv[++i]) != 0 ? 1 : 0;
        } else {
            std::cerr << "render_beauty: unknown argument '" << argv[i]
                      << "'\nusage: render_beauty [--scene scenes/x.json] --out out.png [--out-exr out.exr] [--compare-exr ref.exr] "
                         "[--compare ref.png] [--error-spectrum] [--seed N] [--passes N] [--width W] [--height H] "
                         "[--exposure EV] [--aov name] [--fisheye FOCAL_MM] [--env-light 0|1] [--bench-log log.jsonl] [--assert-deterministic] [--assert-converged]\n";
            return false;
        }
    }
    if (options.outPath.empty() && !options.assertDeterministic && !options.assertConverged) {
        std::cerr << "render_beauty: --out is required unless running a gate\n";
        return false;
    }
    // The gates return before the timed loop, so a log request alongside one would silently record nothing.
    if (!options.benchLogPath.empty() && (options.assertDeterministic || options.assertConverged)) {
        std::cerr << "render_beauty: --bench-log records the timing path, which the --assert-* gates do not run\n";
        return false;
    }
    if (options.passes < 1) {
        std::cerr << "render_beauty: --passes must be at least 1\n";
        return false;
    }
    if (options.errorSpectrum && options.compareExrPath.empty()) {
        std::cerr << "render_beauty: --error-spectrum needs --compare-exr to have an error to analyse\n";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseArgs(argc, argv, options)) {
        return EXIT_FAILURE;
    }

    const std::string assetRoot = ASSET_ROOT_DIR;
    std::string error;
    const std::unique_ptr<pathtracer::api::HeadlessRenderer> renderer =
        pathtracer::api::HeadlessRenderer::open(assetRoot, options.scenePath, error);
    if (!renderer) {
        std::cerr << "render_beauty: " << error << "\n";
        return EXIT_FAILURE;
    }

    // Fixed camera from profile.json, no controller: what makes two runs comparable is that neither can have been nudged.
    const pathtracer::scene::Camera camera = options.fisheyeFocalLengthMm > 0.0F ? [&] {
        // Rebuilt only here: the degrees accessors round-trip the pose, which is not bit-exact, so the default path keeps the original.
        const pathtracer::scene::Camera& authored = renderer->defaultCamera();
        pathtracer::scene::Lens lens = authored.lens();
        lens.projection = pathtracer::scene::LensProjection::FisheyePolynomial;
        return pathtracer::scene::Camera{authored.position(), authored.yawDegrees(), authored.pitchDegrees(),
                                         authored.filmBack(), options.fisheyeFocalLengthMm, authored.nearClip(),
                                         authored.farClip(), authored.aperture(), authored.shutterSeconds(),
                                         authored.iso(), lens};
    }() : renderer->defaultCamera();
    const int width = options.width > 0 ? options.width : renderer->defaultWidth();
    const int height = options.height > 0 ? options.height : renderer->defaultHeight();
    // --env-light overrides the scene's own authored default (-1 = no override).
    const std::optional<bool> envLightOverride =
        options.envLight >= 0 ? std::optional<bool>(options.envLight != 0) : std::nullopt;
    const bool envLightEnabled = envLightOverride.value_or(renderer->defaultEnvLightEnabled());
    const std::string aovName = pathtracer::debug::kAovNames[static_cast<int>(options.aov)];

    // One accumulation parameterised by its randomization: the mean of `passes` single-sample passes, as PathTraceDriver does.
    const auto accumulate = [&](std::uint32_t scrambleSeed, int passes) {
        const pathtracer::api::HeadlessRenderer::Request request{
            .camera = camera,
            .width = width,
            .height = height,
            .samples = passes,
            .scrambleSeed = scrambleSeed,
            .aovs = {options.aov},
            .envLightEnabled = envLightOverride,
        };
        if (!renderer->render(request, error)) {
            // Unreachable: parseArgs rejects a non-positive pass count and the resolution is positive above, render()'s only failure modes.
            std::cerr << "render_beauty: " << error << "\n";
            std::exit(EXIT_FAILURE);
        }
        return renderer->lastImage(options.aov);
    };

    // --- Determinism gate: same seed, same floats, no statistics.
    if (options.assertDeterministic) {
        const pathtracer::gfx::HdrImage first = accumulate(options.scrambleSeed, options.passes);
        const pathtracer::gfx::HdrImage second = accumulate(options.scrambleSeed, options.passes);
        std::size_t differing = 0;
        double worst = 0.0;
        for (std::size_t i = 0; i < first.texels.size(); ++i) {
            if (first.texels[i] != second.texels[i]) {
                ++differing;
                worst = std::max(worst, std::fabs(static_cast<double>(first.texels[i]) -
                                                   static_cast<double>(second.texels[i])));
            }
        }
        if (differing != 0) {
            std::cerr << "render_beauty: FAILED determinism -- " << differing << " of " << first.texels.size()
                      << " floats differ between two runs at the same seed (worst " << worst
                      << "); the render depends on something that is not its inputs\n";
            return EXIT_FAILURE;
        }
        std::cout << "render_beauty: determinism PASSED -- " << first.texels.size()
                  << " floats bit-identical across two runs at seed " << options.scrambleSeed << "\n";
    }

    // --- Convergence gate: R=8 independent sub-renders, buffer-variance estimator.
    if (options.assertConverged) {
        constexpr int kSubRenders = 8;
        const int perSubRender = std::max(1, options.passes / kSubRenders);
        const auto family = [&](std::uint32_t base) {
            std::vector<pathtracer::gfx::HdrImage> members;
            members.reserve(kSubRenders);
            for (int r = 0; r < kSubRenders; ++r) {
                members.push_back(accumulate(base + static_cast<std::uint32_t>(r), perSubRender));
            }
            return members;
        };
        // Disjoint seed ranges, so no sub-render is shared between the families: sharing one would correlate them and shrink the gap.
        const std::vector<pathtracer::gfx::HdrImage> a = family(options.scrambleSeed);
        const std::vector<pathtracer::gfx::HdrImage> b = family(options.scrambleSeed + 1000U);

        const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        // Sidak over the pixels examined, so the threshold follows the resolution; luminance, since error visibility is a luminance effect.
        const double perPixelAlpha = tools::stats::sidak(tools::check::kFamilyAlpha, static_cast<int>(pixels));
        double worstZ = 0.0;
        std::size_t worstPixel = 0;
        std::size_t examined = 0;
        std::size_t constantDisagreements = 0;
        for (std::size_t px = 0; px < pixels; ++px) {
            const auto luminance = [&](const pathtracer::gfx::HdrImage& image) {
                const glm::vec3 rgb = image.rgb(px);
                return (0.2126 * rgb.r) + (0.7152 * rgb.g) + (0.0722 * rgb.b);
            };
            tools::stats::Welford wa;
            tools::stats::Welford wb;
            for (int r = 0; r < kSubRenders; ++r) {
                wa.add(luminance(a[static_cast<std::size_t>(r)]));
                wb.add(luminance(b[static_cast<std::size_t>(r)]));
            }
            const double va = wa.sampleVariance() / kSubRenders;
            const double vb = wb.sampleVariance() / kSubRenders;
            const double combined = va + vb;
            // Zero variance in both families: the z-score is undefined but the verdict is not -- equal constants agree, unequal disagree.
            if (!(combined > 0.0)) {
                constantDisagreements += wa.mean() != wb.mean() ? 1 : 0;
                continue;
            }
            ++examined;
            const double z = std::fabs(wa.mean() - wb.mean()) / std::sqrt(combined);
            if (z > worstZ) {
                worstZ = z;
                worstPixel = px;
            }
        }
        // Welch dof is bounded below by R-1 for equal samples, so using it is conservative: it can widen the threshold, never narrow it.
        const double threshold = tools::stats::studentTTwoSided(perPixelAlpha, kSubRenders - 1);
        if (examined == 0 || constantDisagreements != 0) {
            std::cerr << "render_beauty: FAILED convergence -- " << examined << " pixels had measurable variance and "
                      << constantDisagreements << " zero-variance pixels disagreed between families; a gate that "
                         "examined nothing, or two constant estimates that differ, is not a pass\n";
            return EXIT_FAILURE;
        }
        if (!(worstZ <= threshold)) {
            std::cerr << "render_beauty: FAILED convergence -- worst |z| " << worstZ << " at pixel " << worstPixel
                      << " (" << (worstPixel % static_cast<std::size_t>(width)) << ","
                      << (worstPixel / static_cast<std::size_t>(width)) << ") exceeds " << threshold
                      << " at per-pixel alpha " << perPixelAlpha << " over " << pixels
                      << " pixels; two independent randomizations of the same estimator disagree by more than their "
                         "own measured error\n";
            return EXIT_FAILURE;
        }
        std::cout << "render_beauty: convergence PASSED -- worst |z| " << worstZ << " against " << threshold
                  << " over " << pixels << " pixels (" << kSubRenders << " sub-renders x " << perSubRender
                  << " passes per family)\n";
    }

    if (options.assertDeterministic || options.assertConverged) {
        return EXIT_SUCCESS;  // a gate renders for its verdict, not for an image
    }

    // Per-pass wall clock on the trace only; compare the mean, which holds to ~1% where a pass minimum varies ~12% run to run.
    const pathtracer::gfx::HdrImage accumulated = accumulate(options.scrambleSeed, options.passes);
    const std::vector<double>& milliseconds = renderer->lastStats().passMilliseconds;

    const pathtracer::api::HeadlessRenderer::RenderStats& stats = renderer->lastStats();
    const pathtracer::debug::RayCounts rays = stats.rays;
    // A G-buffer AOV runs no path-traced passes, so there is no per-pass distribution to report for it.
    if (!milliseconds.empty()) {
        std::cout << "render_beauty: rays over " << options.passes << " passes -- primary " << rays.primary
                  << ", bounce " << rays.bounce << ", ao " << rays.ao << ", shadow " << rays.shadow << ", total "
                  << rays.total() << "\n";
        const auto [best, worst] = std::minmax_element(milliseconds.begin(), milliseconds.end());
        const double totalMs = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0);
        std::cout << "render_beauty: best-of-" << options.passes << ": " << *best << " ms/pass  (mean "
                  << totalMs / static_cast<double>(options.passes) << ", worst " << *worst << ", total "
                  << totalMs << ")\n";
    }
    if (stats.gbufferMilliseconds > 0.0) {
        std::cout << "render_beauty: G-buffer in " << stats.gbufferMilliseconds << " ms\n";
    }
    if (stats.filterMilliseconds > 0.0) {
        std::cout << "render_beauty: Beauty filter in " << stats.filterMilliseconds << " ms\n";
    }
    // Before any output encode, so the record's rusage covers load, build and the timed passes but not PNG/EXR writing.
    if (!options.benchLogPath.empty() &&
        !appendTimingRecord(options, argc, argv, width, height, aovName, renderer->baseSettings(),
                            envLightEnabled, stats.gbufferMilliseconds, stats.filterMilliseconds, renderer->textureType(),
                            milliseconds,
                            rays, accumulated)) {
        return EXIT_FAILURE;
    }

    if (!options.outExrPath.empty()) {
        if (!pathtracer::gfx::writeExr(options.outExrPath, accumulated)) {
            return EXIT_FAILURE;
        }
        std::cout << "render_beauty: wrote " << options.outExrPath << " (linear " << aovName << ", "
                  << width << "x" << height << ", " << options.passes << " passes)\n";
    }

    if (!options.compareExrPath.empty()) {
        const std::optional<pathtracer::gfx::HdrImage> reference = pathtracer::gfx::loadExr(options.compareExrPath);
        if (!reference.has_value()) {
            return EXIT_FAILURE;
        }
        if (reference->width != width || reference->height != height || reference->channels != accumulated.channels) {
            std::cerr << "render_beauty: --compare-exr image is " << reference->width << "x" << reference->height << "x"
                      << reference->channels << ", this render is " << width << "x" << height << "x" << accumulated.channels
                      << "\n";
            return EXIT_FAILURE;
        }
        // Absolute RMSE tracks the brightest pixels, relative MSE (Rousselle et al. 2011) dim ones equally; epsilon guards a black pixel.
        constexpr double kRelativeEpsilon = 1e-2;
        double squaredSum = 0.0;
        double relativeSum = 0.0;
        std::size_t counted = 0;
        for (std::size_t i = 0; i < accumulated.texels.size(); ++i) {
            const double delta = static_cast<double>(accumulated.texels[i]) - static_cast<double>(reference->texels[i]);
            const double ref = reference->texels[i];
            squaredSum += delta * delta;
            relativeSum += (delta * delta) / ((ref * ref) + kRelativeEpsilon);
            ++counted;
        }
        const double rmse = std::sqrt(squaredSum / static_cast<double>(counted));
        const double relativeMse = relativeSum / static_cast<double>(counted);
        std::cout << "render_beauty: vs " << options.compareExrPath << " -- linear RMSE " << rmse
                  << ", relMSE " << relativeMse << "\n";
        if (options.errorSpectrum) {
            reportErrorSpectrum(accumulated, *reference);
        }
    }

    const bool isBeauty = options.aov == pathtracer::debug::AovId::Beauty;
    // The same display decision presentFrame makes, so a PNG and the viewer agree by construction rather than by two copies staying level.
    const pathtracer::debug::AovDisplay prepared = pathtracer::debug::aovDisplay(
        options.aov, accumulated, {options.passes, renderer->baseSettings().maxBounces});
    pathtracer::debug::BipolarDisplay display = prepared.affine;
    // The photographic exposure, which only the caller knows; aovDisplay leaves the gain at unity for every AOV that takes one.
    if (pathtracer::debug::aovTakesDisplayExposure(options.aov)) {
        display.gain = glm::vec3(std::pow(2.0F, options.exposureEv));
    }
    // The shared encode takes packed RGB, so expand each texel exactly as the viewer's texture swizzle does.
    const pathtracer::gfx::HdrImage& source = prepared.mapped.texels.empty() ? accumulated : prepared.mapped;
    std::vector<float> linearRgb(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    for (std::size_t texel = 0; texel < linearRgb.size() / 3; ++texel) {
        const glm::vec3 rgb = source.rgb(texel);
        linearRgb[(texel * 3) + 0] = rgb.r;
        linearRgb[(texel * 3) + 1] = rgb.g;
        linearRgb[(texel * 3) + 2] = rgb.b;
    }
    std::vector<unsigned char> encoded(linearRgb.size());
    pathtracer::gfx::encodeForDisplay(linearRgb, width, height, display.gain, isBeauty, display.offset, encoded);
    if (!writePng(options.outPath, width, height, encoded)) {
        return EXIT_FAILURE;
    }
    std::cout << "render_beauty: wrote " << options.outPath << " (" << aovName << ", " << width
              << "x" << height << ", " << options.passes << " passes)\n";

    if (!options.comparePath.empty()) {
        int refWidth = 0;
        int refHeight = 0;
        std::vector<unsigned char> reference;
        if (!readPng(options.comparePath, refWidth, refHeight, reference)) {
            return EXIT_FAILURE;
        }
        if (refWidth != width || refHeight != height) {
            std::cerr << "render_beauty: --compare image is " << refWidth << "x" << refHeight
                      << ", this render is " << width << "x" << height << "\n";
            return EXIT_FAILURE;
        }
        int maxDelta = 0;
        double squaredSum = 0.0;
        // Signed mean alongside RMS: near-zero mean against non-zero RMS means light moved rather than appeared or vanished.
        double signedSum = 0.0;
        std::size_t differing = 0;
        for (std::size_t i = 0; i < encoded.size(); ++i) {
            const int signedDelta = static_cast<int>(encoded[i]) - static_cast<int>(reference[i]);
            const int delta = std::abs(signedDelta);
            maxDelta = std::max(maxDelta, delta);
            squaredSum += static_cast<double>(delta) * delta;
            signedSum += signedDelta;
            differing += delta != 0 ? 1 : 0;
        }
        const double rms = std::sqrt(squaredSum / static_cast<double>(encoded.size()));
        const double meanSigned = signedSum / static_cast<double>(encoded.size());
        std::cout << "render_beauty: vs " << options.comparePath << " -- max channel delta "
                  << maxDelta << "/255, RMS " << rms << ", mean signed " << meanSigned << ", "
                  << differing << "/" << encoded.size() << " channels differ\n";
    }
    return EXIT_SUCCESS;
}
