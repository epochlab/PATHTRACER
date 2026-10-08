// Correctness gate for the engine's file boundaries: the EXR round trip, the glTF loader and the JSON scene, profile and bench-log parsers.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <OpenColorIO/OpenColorIO.h>
#include <OpenImageIO/color.h>
#include <OpenImageIO/imageio.h>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include "check.h"
#include "pathtracer/config/profile_config.h"
#include "pathtracer/config/scene_config.h"
#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/bench_log.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/ocio_display_transform.h"
#include "pathtracer/gfx/texture.h"
#include "pathtracer/scene/gbuffer_shading.h"
#include "pathtracer/scene/gltf_loader.h"
#include "pathtracer/scene/shading_scene.h"

namespace {

// Written under the system temp directory: a validator must not need a writable checkout, nor leave anything for the next run.
std::filesystem::path scratchPath(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

// Values hostile to a lossy or narrowing round trip: denormal-scale, exact halves, far outside display range, and a negative.
pathtracer::gfx::HdrImage makeProbeImage() {
    // 7x5: deliberately not a power of two or a multiple of any tile size.
    pathtracer::gfx::HdrImage image = pathtracer::gfx::makeImage(7, 5, pathtracer::gfx::kRgbChannels);
    for (std::size_t i = 0; i < image.texels.size(); ++i) {
        const auto t = static_cast<float>(i);
        switch (i % pathtracer::gfx::kRgbChannels) {
            case 0: image.texels[i] = t * 1.5F; break;              // exact halves
            case 1: image.texels[i] = 1.0e-20F * (t + 1.0F); break;  // far below display range
            default: image.texels[i] = 65504.0F - t; break;          // near the half-float maximum, if one were used
        }
    }
    return image;
}

// Writes interleaved full-float `planes` through OIIO with `attributes` set, for probes writeExr never produces: RGBA, R-only, retagged.
bool writeExrPlanes(const std::filesystem::path& path, int width, int height, const std::vector<std::string>& planes,
                    const std::vector<float>& texels, const std::function<void(OIIO::ImageSpec&)>& attributes = {}) {
    OIIO::ImageSpec spec(width, height, static_cast<int>(planes.size()), OIIO::TypeFloat);
    spec.channelnames = planes;
    spec.alpha_channel = -1;
    if (attributes) {
        attributes(spec);
    }
    const std::unique_ptr<OIIO::ImageOutput> output = OIIO::ImageOutput::create("openexr");
    if (!output || !output->open(path.string(), spec) || !output->write_image(OIIO::TypeFloat, texels.data()) || !output->close()) {
        std::cerr << "io_validate: could not write " << path << ": " << (output ? output->geterror() : OIIO::geterror()) << '\n';
        return false;
    }
    return true;
}

// The file's own header, read back independently of loadImage: what another tool reading it would see.
std::optional<OIIO::ImageSpec> headerOf(const std::filesystem::path& path) {
    const std::unique_ptr<OIIO::ImageInput> input = OIIO::ImageInput::open(path.string());
    return input ? std::optional(input->spec()) : std::nullopt;
}

// The losslessness hdr_image.h claims in prose, bit-exact under both roles: Colour is an identity conversion into the working space.
PT_CHECK(exr_round_trip_is_lossless, Fast, Exact) {
    ctx.plan(8);
    const pathtracer::gfx::HdrImage original = makeProbeImage();
    for (const pathtracer::gfx::ImageRole role : {pathtracer::gfx::ImageRole::Colour, pathtracer::gfx::ImageRole::Data}) {
        const char* name = role == pathtracer::gfx::ImageRole::Colour ? "Colour" : "Data";
        const std::filesystem::path path = scratchPath("engine_io_validate_roundtrip.exr");
        std::filesystem::remove(path);
        PT_EXPECT(ctx, pathtracer::gfx::writeExr(path.string(), original, role), std::string(name) + ": writeExr failed on a valid image");
        const std::optional<pathtracer::gfx::HdrImage> loaded = pathtracer::gfx::loadImage(path.string(), role);
        std::filesystem::remove(path);
        if (!loaded.has_value()) {
            PT_EXPECT(ctx, false, std::string(name) + ": loadImage returned nullopt for a file writeExr had just written");
            PT_EXPECT(ctx, false, "dimensions unavailable");
            PT_EXPECT(ctx, false, "contents unavailable");
            continue;
        }
        PT_EXPECT(ctx, true, "loaded");
        PT_EXPECT(ctx, loaded->width == original.width && loaded->height == original.height && loaded->channels == original.channels,
                  std::string(name) + ": round trip changed the image's shape");
        std::size_t differing = 0;
        for (std::size_t i = 0; i < original.texels.size() && loaded->texels.size() == original.texels.size(); ++i) {
            differing += loaded->texels[i] != original.texels[i] ? 1 : 0;
        }
        PT_EXPECT(ctx, loaded->texels.size() == original.texels.size() && differing == 0,
                  std::string(name) + ": " + std::to_string(differing) + " floats changed across the round trip");
    }
}

// An AOV is written at the channels it carries, exactly the first `channels` of R,G,B, and loadImage reads it back as itself.
PT_CHECK(exr_write_keeps_the_declared_channels, Fast, Exact) {
    constexpr int kWidth = 3;
    constexpr int kHeight = 2;
    const std::vector<std::string> kPlanes{"R", "G", "B"};
    ctx.plan(2 * static_cast<int>(kPlanes.size()));
    for (int channels = 1; channels <= static_cast<int>(kPlanes.size()); ++channels) {
        pathtracer::gfx::HdrImage image = pathtracer::gfx::makeImage(kWidth, kHeight, channels);
        for (std::size_t i = 0; i < image.texels.size(); ++i) {
            image.texels[i] = 0.5F + static_cast<float>(i);  // distinct per float, so a stride or plane slip reads a wrong value
        }
        const std::filesystem::path path = scratchPath("engine_io_validate_channels.exr");
        std::optional<OIIO::ImageSpec> header;
        std::optional<pathtracer::gfx::HdrImage> loaded;
        if (pathtracer::gfx::writeExr(path.string(), image, pathtracer::gfx::ImageRole::Data)) {
            header = headerOf(path);
            loaded = pathtracer::gfx::loadImage(path.string(), pathtracer::gfx::ImageRole::Data);
        }
        std::filesystem::remove(path);
        const std::vector<std::string> expected(kPlanes.begin(), kPlanes.begin() + channels);
        PT_EXPECT(ctx, header && header->channelnames == expected, std::to_string(channels) + "-channel image wrote the wrong plane set");
        PT_EXPECT(ctx, loaded && loaded->channels == channels && loaded->texels == image.texels,
                  std::to_string(channels) + "-channel image did not load back as itself");
    }
}

// Colour EXRs carry the working space twice, as OIIO's tag and OpenEXR's chromaticities; the oracle is BT.709's own published xy.
PT_CHECK(exr_write_declares_its_colour_space, Fast, Exact) {
    // ITU-R BT.709-6 Table 1, items 1.3 and 1.4: R, G, B primaries and the D65 white, the working space's definition.
    constexpr std::array<float, 8> kBt709{0.640F, 0.330F, 0.300F, 0.600F, 0.150F, 0.060F, 0.3127F, 0.3290F};
    // Half a unit in the fourth decimal, the precision BT.709 publishes its white point at.
    constexpr float kPublishedPrecision = 5e-5F;
    ctx.plan(4);
    const pathtracer::gfx::HdrImage image{1, 1, pathtracer::gfx::kRgbChannels, {0.25F, 0.5F, 0.75F}};
    const std::filesystem::path path = scratchPath("engine_io_validate_tags.exr");
    std::optional<OIIO::ImageSpec> colour;
    std::optional<OIIO::ImageSpec> data;
    if (pathtracer::gfx::writeExr(path.string(), image, pathtracer::gfx::ImageRole::Colour)) {
        colour = headerOf(path);
    }
    if (pathtracer::gfx::writeExr(path.string(), image, pathtracer::gfx::ImageRole::Data)) {
        data = headerOf(path);
    }
    std::filesystem::remove(path);
    const OIIO::ColorConfig config(std::string("ocio://") + pathtracer::gfx::kOcioConfigName);
    PT_EXPECT(ctx, colour && config.equivalent(colour->get_string_attribute("oiio:ColorSpace"), pathtracer::gfx::kOcioSceneColorSpace),
              "a Colour EXR is not tagged with the working space");
    const OIIO::ParamValue* chromaticities =
        colour ? colour->find_attribute("chromaticities", OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, 8)) : nullptr;
    float worst = std::numeric_limits<float>::infinity();
    if (chromaticities != nullptr) {
        worst = 0.0F;
        for (std::size_t i = 0; i < kBt709.size(); ++i) {
            worst = std::max(worst, std::abs(static_cast<const float*>(chromaticities->data())[i] - kBt709[i]));
        }
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "Colour EXR chromaticities differ from BT.709 by %.3g, over its published %.3g", worst,
                  kPublishedPrecision);
    PT_EXPECT(ctx, worst <= kPublishedPrecision, detail);
    PT_EXPECT(ctx, data && config.isData(config.resolve(data->get_string_attribute("oiio:ColorSpace"))), "a Data EXR is not tagged as data");
    PT_EXPECT(ctx, data && data->find_attribute("chromaticities") == nullptr, "a Data EXR declares chromaticities");
}

// A Colour read lands in the working space through OCIO, the oracle the config's own CPU processor; only linear EXR is read at all.
PT_CHECK(image_load_converts_colour_to_the_working_space, Fast, Exact) {
    const std::vector<float> stored{0.18F, 0.5F, 0.9F, 1.0F, 0.0F, 4.0F};
    const auto oracle = [&](const char* source) {
        const OCIO_NAMESPACE::ConstConfigRcPtr config = OCIO_NAMESPACE::Config::CreateFromBuiltinConfig(pathtracer::gfx::kOcioConfigName);
        std::vector<float> expected = stored;
        OCIO_NAMESPACE::PackedImageDesc desc(expected.data(), 2, 1, 3);
        config->getProcessor(source, pathtracer::gfx::kOcioSceneColorSpace)->getDefaultCPUProcessor()->apply(desc);
        return expected;
    };
    ctx.plan(5);
    const std::filesystem::path path = scratchPath("engine_io_validate_acescg.exr");
    const bool written = writeExrPlanes(path, 2, 1, {"R", "G", "B"}, stored, [](OIIO::ImageSpec& spec) {
        spec.attribute("oiio:ColorSpace", "ACEScg");
    });
    const std::optional<pathtracer::gfx::HdrImage> colour = pathtracer::gfx::loadImage(path.string(), pathtracer::gfx::ImageRole::Colour);
    const std::optional<pathtracer::gfx::HdrImage> data = pathtracer::gfx::loadImage(path.string(), pathtracer::gfx::ImageRole::Data);
    std::filesystem::remove(path);
    PT_EXPECT(ctx, written && colour && colour->texels == oracle("ACEScg"), "an ACEScg EXR did not read as OCIO's ACEScg to working space");
    PT_EXPECT(ctx, data && data->texels == stored, "a Data read converted values it must leave raw");

    // Linear EXR is the one input: a PNG, and an EXR tagged with the sRGB texture curve, are refused rather than decoded.
    const std::filesystem::path png = scratchPath("engine_io_validate_srgb.png");
    const std::vector<std::uint8_t> codes{0, 10, 46, 128, 188, 255};
    OIIO::ImageSpec spec(2, 1, 3, OIIO::TypeUInt8);
    const std::unique_ptr<OIIO::ImageOutput> output = OIIO::ImageOutput::create("png");
    const bool pngWritten = output && output->open(png.string(), spec) && output->write_image(OIIO::TypeUInt8, codes.data()) && output->close();
    const std::filesystem::path encoded = scratchPath("engine_io_validate_srgb.exr");
    const bool encodedWritten = writeExrPlanes(encoded, 2, 1, {"R", "G", "B"}, stored, [](OIIO::ImageSpec& exrSpec) {
        exrSpec.attribute("oiio:ColorSpace", "sRGB - Texture");
    });
    std::cout << "  the stderr diagnostics below are expected: a PNG, and an sRGB-encoded EXR, are not linear EXR\n";
    PT_EXPECT(ctx, pngWritten && !pathtracer::gfx::loadImage(png.string(), pathtracer::gfx::ImageRole::Colour), "a PNG was read as scene data");
    PT_EXPECT(ctx, encodedWritten && !pathtracer::gfx::loadImage(encoded.string(), pathtracer::gfx::ImageRole::Colour),
              "an EXR tagged with an encoded colour space was read as linear");
    PT_EXPECT(ctx, !pathtracer::gfx::loadImage(png.string(), pathtracer::gfx::ImageRole::Data), "a PNG was read as data");
    std::filesystem::remove(png);
    std::filesystem::remove(encoded);
}

// Untagged EXRs mean BT.709 by OpenEXR's own rule; untagged chromaticities naming other primaries cannot be honoured, so are refused.
PT_CHECK(image_load_honours_untagged_exr_chromaticities, Fast, Exact) {
    const std::vector<float> stored{0.25F, 0.5F, 0.75F};
    const auto withChromaticities = [&](const char* name, std::array<float, 8> xy) {
        const std::filesystem::path path = scratchPath(name);
        writeExrPlanes(path, 1, 1, {"R", "G", "B"}, stored, [&](OIIO::ImageSpec& spec) {
            spec.attribute("chromaticities", OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, 8), xy.data());
        });
        return path;
    };
    ctx.plan(3);
    // ITU-R BT.709-6's primaries and D65, and ACES AP1 with its D60 white (SMPTE ST 2065-1 / S-2014-004).
    const std::filesystem::path rec709 = withChromaticities("engine_io_validate_709.exr", {0.64F, 0.33F, 0.30F, 0.60F, 0.15F, 0.06F, 0.3127F, 0.3290F});
    const std::filesystem::path ap1 = withChromaticities("engine_io_validate_ap1.exr", {0.713F, 0.293F, 0.165F, 0.830F, 0.128F, 0.044F, 0.32168F, 0.33767F});
    const std::optional<pathtracer::gfx::HdrImage> asRec709 = pathtracer::gfx::loadImage(rec709.string(), pathtracer::gfx::ImageRole::Colour);
    PT_EXPECT(ctx, asRec709 && asRec709->texels == stored, "an untagged BT.709 EXR did not read unchanged into the BT.709 working space");
    std::cout << "  the stderr diagnostic below is expected: untagged AP1 chromaticities name no space the file is tagged with\n";
    PT_EXPECT(ctx, !pathtracer::gfx::loadImage(ap1.string(), pathtracer::gfx::ImageRole::Colour),
              "untagged AP1 chromaticities were read as BT.709");
    PT_EXPECT(ctx, pathtracer::gfx::loadImage(ap1.string(), pathtracer::gfx::ImageRole::Data), "a Data read refused a file over its chromaticities");
    std::filesystem::remove(rec709);
    std::filesystem::remove(ap1);
}

// A display capture keeps 16 bits, straight alpha and the sRGB chunk (PNG 1.2, 4.2.2.6): read back independently of the writer.
PT_CHECK(display_png_is_16_bit_rgba_tagged, Fast, Exact) {
    const pathtracer::gfx::HdrImage image{2, 1, 4, {0.0F, 0.5F, 1.0F, 0.25F, 1.5F, -0.5F, 1.0F / 65535.0F, 1.0F}};
    ctx.plan(3);
    const std::filesystem::path path = scratchPath("engine_io_validate_display.png");
    const bool written = pathtracer::gfx::writeDisplayPng(path.string(), image);
    // OIIO associates alpha on read unless told the caller wants the file's straight values.
    OIIO::ImageSpec straight;
    straight.attribute("oiio:UnassociatedAlpha", 1);
    const std::unique_ptr<OIIO::ImageInput> input = OIIO::ImageInput::open(path.string(), &straight);
    std::vector<std::uint16_t> codes(image.texels.size());
    const bool read = input && input->read_image(0, 0, 0, 4, OIIO::TypeUInt16, codes.data());
    PT_EXPECT(ctx, written && read && input->spec().nchannels == 4 && input->spec().format == OIIO::TypeUInt16,
              "the capture is not a 16-bit, four-channel PNG");
    // Round to nearest after the clamp: 1.5 and -0.5 saturate, alpha 0.25 is stored unpremultiplied, one code survives as one code.
    const std::vector<std::uint16_t> expected{0, 32768, 65535, 16384, 65535, 0, 1, 65535};
    PT_EXPECT(ctx, codes == expected, "16-bit codes are not the clamped, rounded display values with straight alpha");
    std::ifstream file(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    PT_EXPECT(ctx, bytes.find("sRGB") != std::string::npos, "the capture carries no sRGB chunk for the display it encodes");
    std::filesystem::remove(path);
}

// HdrImage::rgb is the CPU half of the display contract Texture's swizzle states: a scalar broadcasts, a second channel leaves B 0.
PT_CHECK(hdr_image_rgb_expands_as_the_display_swizzle, Fast, Exact) {
    ctx.plan(3);
    const pathtracer::gfx::HdrImage scalar{1, 1, 1, {0.25F}};
    const pathtracer::gfx::HdrImage pair{1, 1, 2, {0.25F, 0.5F}};
    const pathtracer::gfx::HdrImage triple{1, 1, 3, {0.25F, 0.5F, 0.75F}};
    PT_EXPECT(ctx, scalar.rgb(0) == glm::vec3(0.25F), "a scalar did not broadcast to RGB");
    PT_EXPECT(ctx, pair.rgb(0) == glm::vec3(0.25F, 0.5F, 0.0F), "a two-channel texel did not read blue 0");
    PT_EXPECT(ctx, triple.rgb(0) == glm::vec3(0.25F, 0.5F, 0.75F), "an RGB texel did not read through unchanged");
}

// A missing file must be reported, not treated as an empty image: a caller given a zero-sized image renders black and never knows.
PT_CHECK(image_load_rejects_bad_input, Fast, Exact) {
    ctx.plan(2);
    const std::filesystem::path missing = scratchPath("engine_io_validate_does_not_exist.exr");
    std::filesystem::remove(missing);
    PT_EXPECT(ctx, !pathtracer::gfx::loadImage(missing.string(), pathtracer::gfx::ImageRole::Data).has_value(),
              "loadImage accepted a path that does not exist");

    // A file that exists but is not an image: the realistic corruption, which a magic-number check catches and a header check does not.
    const std::filesystem::path garbage = scratchPath("engine_io_validate_garbage.exr");
    {
        std::ofstream out(garbage, std::ios::binary);
        out << "this is not an OpenEXR file, but it is definitely a file";
    }
    PT_EXPECT(ctx, !pathtracer::gfx::loadImage(garbage.string(), pathtracer::gfx::ImageRole::Data).has_value(),
              "loadImage accepted a file whose contents are not an image");
    std::filesystem::remove(garbage);
}

// Writes `text` to a scratch .json and returns the path, so each rejection row states its own malformation inline.
std::filesystem::path writeJson(const char* name, const std::string& text) {
    const std::filesystem::path path = scratchPath(name);
    std::ofstream out(path);
    out << text;
    return path;
}

// Every shipped scene must load under the closed key sets: without this, each rejection row below could pass by rejecting everything.
PT_CHECK(scene_config_accepts_the_shipped_scenes, Fast, Exact) {
    const std::vector<const char*> shipped = {"cornell.json", "macbeth.json", "stump.json"};
    ctx.plan(static_cast<int>(shipped.size()) + 1);
    for (const char* name : shipped) {
        const std::filesystem::path scene = std::filesystem::path(ASSET_ROOT_DIR) / "scenes" / name;
        char detail[256];
        std::snprintf(detail, sizeof(detail), "loadSceneConfig rejected the shipped scene at %s", scene.string().c_str());
        PT_EXPECT(ctx, pathtracer::config::loadSceneConfig(scene.string()).has_value(), detail);
    }
    // Anti-vacuity for the textures key itself: macbeth binds its chart only through it, so an unparsed key would leave it empty.
    const std::optional<pathtracer::config::SceneConfig> macbeth =
        pathtracer::config::loadSceneConfig((std::filesystem::path(ASSET_ROOT_DIR) / "scenes" / "macbeth.json").string());
    PT_EXPECT(ctx, macbeth && macbeth->textures.size() == 2, "macbeth.json's textures did not parse to its two grids");
}

// The rejection half of the contract: each row is a malformation a real authoring mistake produces, and each must be reported.
PT_CHECK(scene_config_rejects_malformed_input, Fast, Exact) {
    struct Case {
        const char* name;
        const char* file;
        std::string text;
    };
    // Built by mutating a base that loads, so each row fails for the reason it names rather than passing vacuously on an earlier error.
    const auto scene = [](const std::string& lights) {
        return std::string(
                   "{\"model\":{\"gltfPath\":\"geometry/cornell/cornell_v001.gltf\","
                   "\"position\":[0,0,0],\"rotation\":[0,0,0]},"
                   "\"environment\":{\"hdriPath\":\"textures/republiqueHDR_2k.exr\"},"
                   "\"materialPath\":\"materials/clay.json\"") +
               lights + "}";
    };
    const std::string validLights =
        ",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
        "\"color\":[1,1,1],\"intensity\":5.0,\"twoSided\":false}]";

    const std::vector<Case> cases = {
        {"not JSON at all", "engine_io_scene_notjson.json", "{ this is not json"},
        {"empty file", "engine_io_scene_empty.json", ""},
        {"JSON array where an object is required", "engine_io_scene_array.json", "[1, 2, 3]"},
        {"missing the model section entirely", "engine_io_scene_nomodel.json",
         "{\"environment\":{\"hdriPath\":\"x.exr\"},\"materialPath\":\"materials/clay.json\"}"},
        // Negative radiance is not a scene, and would propagate as negative energy through every estimator.
        {"negative light intensity", "engine_io_scene_negintensity.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"color\":[1,1,1],\"intensity\":-5.0,\"twoSided\":false}]")},
        // A zero extent is a quad with no area; a negative one would mirror it and silently flip the emitting face.
        {"quad light with a zero size", "engine_io_scene_zerosizelight.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,0],"
               "\"color\":[1,1,1],\"intensity\":5.0,\"twoSided\":false}]")},
        {"quad light with a negative size", "engine_io_scene_negsizelight.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[-1,1],"
               "\"color\":[1,1,1],\"intensity\":5.0,\"twoSided\":false}]")},
        // The retired corner + edges form has no position, so it fails rather than placing a light nobody authored.
        {"the retired quad light origin/edge0/edge1 keys", "engine_io_scene_cornerlight.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"origin\":[0,0,0],\"edge0\":[1,0,0],\"edge1\":[0,0,1],"
               "\"color\":[1,1,1],\"intensity\":5.0,\"twoSided\":false}]")},
        // Closed key set: a leftover edge0 beside a complete placement would otherwise be ignored silently.
        {"a stale quad light edge0 key", "engine_io_scene_staleedge.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"edge0\":[1,0,0],\"color\":[1,1,1],\"intensity\":5.0,\"twoSided\":false}]")},
        // Each node maps slot names to paths; a bare path string would leave the slot unstated.
        {"textures entry that is not a slot object", "engine_io_scene_textureflat.json",
         scene(",\"textures\":{\"sphere01\":\"textures/macbeth.exr\"}")},
        // Retired keys, well-formed otherwise: an ignored textureOverrides would render the scene untextured with no diagnostic.
        {"the retired textureOverrides key", "engine_io_scene_textureoverrides.json",
         scene(",\"textureOverrides\":{\"sphere01\":{\"baseColorTexture\":\"textures/macbeth.exr\"}}")},
        {"the retired model.texturePath key", "engine_io_scene_texturepath.json",
         "{\"model\":{\"gltfPath\":\"geometry/cornell/cornell_v001.gltf\",\"texturePath\":\"\","
         "\"position\":[0,0,0],\"rotation\":[0,0,0]},\"environment\":{\"hdriPath\":\"textures/republiqueHDR_2k.exr\"},"
         "\"materialPath\":\"materials/clay.json\"}"},
        {"a misspelt environment key", "engine_io_scene_envtypo.json",
         "{\"model\":{\"gltfPath\":\"geometry/cornell/cornell_v001.gltf\",\"position\":[0,0,0],\"rotation\":[0,0,0]},"
         "\"environment\":{\"hdriPath\":\"textures/republiqueHDR_2k.exr\",\"lightEnable\":false},"
         "\"materialPath\":\"materials/clay.json\"}"},
        {"negative light colour", "engine_io_scene_negcolor.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"color\":[1,-1,1],\"intensity\":5.0,\"twoSided\":false}]")},
    };

    // Anti-vacuity: the base the scene() rows are built from must itself LOAD, or they would prove nothing.
    const std::filesystem::path basePath = writeJson("engine_io_scene_base.json", scene(validLights));
    const bool baseLoads = pathtracer::config::loadSceneConfig(basePath.string()).has_value();
    std::filesystem::remove(basePath);

    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, baseLoads,
                  "the unmutated base scene must load, or every mutated row below passes vacuously");
    for (const Case& testCase : cases) {
        const std::filesystem::path path = writeJson(testCase.file, testCase.text);
        const bool accepted = pathtracer::config::loadSceneConfig(path.string()).has_value();
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadSceneConfig accepted a scene with %s", testCase.name);
        PT_EXPECT(ctx, !accepted, detail);
        std::filesystem::remove(path);
    }
}

// Anti-vacuity for the rejection rows below: every material the repo ships must still load once loadMaterialConfig validates.
PT_CHECK(material_config_accepts_the_shipped_materials, Fast, Exact) {
    const std::vector<const char*> shipped = {"chrome.json", "clay.json", "constant.json", "glass.json", "principled.json"};
    ctx.plan(static_cast<int>(shipped.size()));
    for (const char* name : shipped) {
        const std::filesystem::path path = std::filesystem::path(ASSET_ROOT_DIR) / "materials" / name;
        char detail[256];
        std::snprintf(detail, sizeof(detail), "loadMaterialConfig rejected the shipped material %s", name);
        PT_EXPECT(ctx, pathtracer::config::loadMaterialConfig(path.string()).has_value(), detail);
    }
}

// Each row names the BSDF failure it prevents; without the loader's bounds these reach the integrator as NaN or negative energy.
PT_CHECK(material_config_rejects_malformed_input, Fast, Exact) {
    struct Case {
        const char* name;
        const char* file;
        std::string text;
    };
    // Mutations of one base that loads, so a row fails for the reason it names rather than vacuously on an earlier parse error.
    const auto material = [](const std::string& overrides) {
        return std::string("{\"diffuseColour\":[1,1,1],\"roughnessFactor\":0.5,\"roughnessMin\":0.045,"
                           "\"roughnessMax\":1.0,\"bumpStrength\":0.0") +
               overrides + "}";
    };

    const std::vector<Case> cases = {
        // std::clamp(v, lo, hi) has undefined behaviour when lo > hi, and resolveRoughness clamps with exactly these two.
        {"roughnessMin above roughnessMax", "engine_io_material_roughrange.json",
         "{\"diffuseColour\":[1,1,1],\"roughnessFactor\":0.5,\"roughnessMin\":0.9,\"roughnessMax\":0.2,\"bumpStrength\":0.0}"},
        // eonUniformMixWeight raises r to the 0.1 power, which is NaN for r < 0, and the NaN reaches the pixel through sampleEon.
        {"negative diffuseRoughness", "engine_io_material_negdiffrough.json", material(",\"diffuseRoughness\":-0.2")},
        // Outside [0,1] the EON quartic albedo fit is extrapolated, where 1 - E can go negative and the BRDF with it.
        {"diffuseRoughness above 1", "engine_io_material_diffroughhigh.json", material(",\"diffuseRoughness\":1.5")},
        // transmissionColor is a transmittance; above 1 sigma_a = -ln(colour)/depth is negative and Beer-Lambert amplifies without bound.
        {"transmissionColor above 1", "engine_io_material_transcolour.json",
         material(",\"transmissionColor\":[1.4,1.0,1.0],\"transmissionDepth\":0.4")},
        // Lobe selection probabilities are mixed by these; outside [0,1] glm::mix extrapolates and the prefix-sum partition goes negative.
        {"metallicFactor above 1", "engine_io_material_metallic.json", material(",\"metallicFactor\":1.5")},
        {"negative transmissionFactor", "engine_io_material_negtrans.json", material(",\"transmissionFactor\":-0.5")},
        // dielectricF0 divides by ior + 1, and every dielectric lobe by the etaI/etaT ratio.
        {"non-positive ior", "engine_io_material_zeroior.json", material(",\"ior\":0.0")},
        // The EON albedo inversion leaves rho unbounded above an albedo of 1, where its multiple-scatter denominator can reach zero.
        {"diffuseColour above 1", "engine_io_material_colour.json",
         "{\"diffuseColour\":[1,2,1],\"roughnessFactor\":0.5,\"roughnessMin\":0.045,\"roughnessMax\":1.0,\"bumpStrength\":0.0}"},
        {"negative transmissionDepth", "engine_io_material_negdepth.json", material(",\"transmissionDepth\":-1.0")},
        // A misspelt model must not load as the standard BSDF, silently shading what the author meant to be unlit.
        {"unknown shadingModel", "engine_io_material_unknownmodel.json", material(",\"shadingModel\":\"unlit\"")},
        // A constant surface scatters nothing, so a BSDF key in its file is dead input the author believes is live.
        {"BSDF key on a constant material", "engine_io_material_constantbsdf.json",
         "{\"shadingModel\":\"constant\",\"roughnessFactor\":0.5}"},
        // The constant path bypasses the standard parse, so its one live field must still pass the same reflectance bound.
        {"constant diffuseColour above 1", "engine_io_material_constantcolour.json",
         "{\"shadingModel\":\"constant\",\"diffuseColour\":[1,2,1]}"},
    };

    const std::filesystem::path basePath = writeJson("engine_io_material_base.json", material(""));
    const bool baseLoads = pathtracer::config::loadMaterialConfig(basePath.string()).has_value();
    std::filesystem::remove(basePath);

    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, baseLoads, "the unmutated base material must load, or every mutated row below passes vacuously");
    for (const Case& testCase : cases) {
        const std::filesystem::path path = writeJson(testCase.file, testCase.text);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadMaterialConfig accepted a material with %s", testCase.name);
        PT_EXPECT(ctx, !pathtracer::config::loadMaterialConfig(path.string()).has_value(), detail);
        std::filesystem::remove(path);
    }
}

PT_CHECK(profile_config_accepts_the_shipped_profile, Fast, Exact) {
    ctx.plan(1);
    const std::filesystem::path profile = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    char detail[256];
    std::snprintf(detail, sizeof(detail), "loadProfileConfig rejected the shipped profile at %s",
                  profile.string().c_str());
    PT_EXPECT(ctx, pathtracer::config::loadProfileConfig(profile.string()).has_value(), detail);
}

// The boundary values profile_config.cpp is responsible for: a resolution divides an aspect ratio, a film-back dimension the FOV.
PT_CHECK(profile_config_rejects_malformed_input, Fast, Exact) {
    struct Case {
        const char* name;
        const char* file;
        std::string text;
    };
    const std::vector<Case> cases = {
        {"not JSON at all", "engine_io_profile_notjson.json", "{ nope"},
        {"empty file", "engine_io_profile_empty.json", ""},
        {"JSON array where an object is required", "engine_io_profile_array.json", "[]"},
    };

    ctx.plan(static_cast<int>(cases.size()) + 1);
    for (const Case& testCase : cases) {
        const std::filesystem::path path = writeJson(testCase.file, testCase.text);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadProfileConfig accepted a profile with %s", testCase.name);
        PT_EXPECT(ctx, !pathtracer::config::loadProfileConfig(path.string()).has_value(), detail);
        std::filesystem::remove(path);
    }

    const std::filesystem::path missing = scratchPath("engine_io_profile_absent.json");
    std::filesystem::remove(missing);
    PT_EXPECT(ctx, !pathtracer::config::loadProfileConfig(missing.string()).has_value(),
                  "loadProfileConfig accepted a path that does not exist");
}

// render.vsync and the two bit depths, each varied alone: every accepted value maps to its own setting, every other is refused.
PT_CHECK(profile_config_render_display_settings, Fast, Exact) {
    using pathtracer::gfx::ScalarType;
    struct Case {
        std::string name;
        const char* key;
        nlohmann::json value;  // null = key removed
        bool accepted;
    };
    std::vector<Case> cases = {
        {"displayBitDepth 16", "displayBitDepth", 16, true},
        {"displayBitDepth 32", "displayBitDepth", 32, true},
        {"textureBitDepth 16", "textureBitDepth", 16, true},
        {"textureBitDepth 32", "textureBitDepth", 32, true},
        {"vsync false", "vsync", false, true},
        {"vsync true", "vsync", true, true},
        {"vsync 1", "vsync", 1, false},
        {"vsync \"true\"", "vsync", "true", false},
        {"vsync missing", "vsync", nullptr, false},
    };
    for (const char* key : {"displayBitDepth", "textureBitDepth"}) {
        for (const nlohmann::json& bad : {nlohmann::json(8), nlohmann::json(24), nlohmann::json(16.5), nlohmann::json("16"),
                                          nlohmann::json(nullptr)}) {
            cases.push_back({std::string(key) + " " + (bad.is_null() ? "missing" : bad.dump()), key, bad, false});
        }
    }

    // Each accepted row varies one key against the shipped profile, so the expectation assumes nothing about what the shipped depths are.
    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    const std::optional<pathtracer::config::ProfileConfig> shippedConfig = pathtracer::config::loadProfileConfig(shippedPath.string());
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, shippedConfig.has_value(), "the shipped profile does not load, so no row below means anything");
    if (!shippedConfig) {
        return;
    }
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        if (testCase.value.is_null()) {
            edited["render"].erase(testCase.key);
        } else {
            edited["render"][testCase.key] = testCase.value;
        }
        const std::filesystem::path path = writeJson("engine_io_profile_render.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        if (testCase.accepted) {
            const bool isDisplay = std::string(testCase.key) == "displayBitDepth";
            const bool isTexture = std::string(testCase.key) == "textureBitDepth";
            const std::optional<ScalarType> varied =
                testCase.value.is_number_integer() ? pathtracer::gfx::scalarTypeFromBitDepth(testCase.value.get<int>())
                                                   : std::nullopt;
            const ScalarType display = isDisplay ? *varied : shippedConfig->render.displayFormat;
            const ScalarType texture = isTexture ? *varied : shippedConfig->render.textureType;
            const bool vsync = isDisplay || isTexture ? shippedConfig->render.vsync : testCase.value.get<bool>();
            std::snprintf(detail, sizeof(detail), "loadProfileConfig rejected or mis-mapped %s", testCase.name.c_str());
            PT_EXPECT(ctx,
                          loaded.has_value() && loaded->render.displayFormat == display &&
                              loaded->render.textureType == texture && loaded->render.vsync == vsync,
                          detail);
        } else {
            std::snprintf(detail, sizeof(detail), "loadProfileConfig accepted %s", testCase.name.c_str());
            PT_EXPECT(ctx, !loaded.has_value(), detail);
        }
    }
}

// camera.lens parses its projection name and four coefficients here; value ranges are camera_validate's, on the assembled Camera.
PT_CHECK(profile_config_camera_lens, Fast, Exact) {
    struct Case {
        std::string name;
        nlohmann::json lens;
        std::optional<pathtracer::scene::LensProjection> projection;  // nullopt where the load must be refused
    };
    const nlohmann::json rectilinear = {{"projection", "rectilinear"},
                                      {"maxFieldOfViewDegrees", 180.0},
                                      {"radialCoefficients", {0.0, 0.0, 0.0, 0.0}}};
    nlohmann::json fisheye = rectilinear;
    fisheye["projection"] = "fisheyePolynomial";
    nlohmann::json omnidirectional = rectilinear;
    omnidirectional["projection"] = "omnidirectional";
    nlohmann::json threeCoefficients = rectilinear;
    threeCoefficients["radialCoefficients"] = {0.0, 0.0, 0.0};
    nlohmann::json unknownProjection = rectilinear;
    unknownProjection["projection"] = "pinhole";
    nlohmann::json missingProjection = rectilinear;
    missingProjection.erase("projection");

    const std::vector<Case> cases = {
        {"rectilinear with a zero polynomial", rectilinear, pathtracer::scene::LensProjection::Rectilinear},
        {"fisheyePolynomial with an equidistant polynomial", fisheye, pathtracer::scene::LensProjection::FisheyePolynomial},
        {"omnidirectional", omnidirectional, pathtracer::scene::LensProjection::Omnidirectional},
        {"an unknown projection name", unknownProjection, std::nullopt},
        {"no projection at all", missingProjection, std::nullopt},
        {"three coefficients instead of four", threeCoefficients, std::nullopt},
    };

    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()));
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        edited["camera"]["lens"] = testCase.lens;
        const std::filesystem::path path = writeJson("engine_io_profile_lens.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded =
            pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadProfileConfig %s %s", testCase.projection ? "mis-read or rejected" : "accepted",
                      testCase.name.c_str());
        const std::optional<pathtracer::scene::LensProjection> projection =
            loaded ? std::optional(loaded->camera.lens.projection) : std::nullopt;
        PT_EXPECT(ctx, projection == testCase.projection, detail);
    }
}

// render.defaultAOV is a raw index main.cpp dereferences unchecked, so the bound holds at load: both ends plus the first past the top.
PT_CHECK(profile_config_default_aov_is_in_range, Fast, Exact) {
    const int aovCount = static_cast<int>(pathtracer::debug::AovId::Count);
    const std::vector<std::pair<nlohmann::json, bool>> cases = {
        {0, true}, {aovCount - 1, true}, {aovCount, false}, {-1, false}, {nlohmann::json(nullptr), false},
    };

    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, pathtracer::config::loadProfileConfig(shippedPath.string()).has_value(),
                  "the shipped profile does not load, so no row below means anything");
    for (const auto& [value, accepted] : cases) {
        nlohmann::json edited = shipped;
        if (value.is_null()) {
            edited["render"].erase("defaultAOV");
        } else {
            edited["render"]["defaultAOV"] = value;
        }
        const std::filesystem::path path = writeJson("engine_io_profile_defaultaov.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadProfileConfig %s defaultAOV %s",
                      accepted ? "rejected" : "accepted", value.dump().c_str());
        PT_EXPECT(ctx, loaded.has_value() == accepted && (!accepted || loaded->render.defaultAov == value.get<int>()),
                      detail);
    }
}

// The two scene-scale distances, each a divisor at its point of use: at or below zero the lane is inf/NaN, not a bounded gradient.
PT_CHECK(profile_config_scene_scale_distances, Fast, Exact) {
    struct Case {
        std::string name;
        const char* key;
        nlohmann::json value;  // null = key removed
        bool accepted;
    };
    std::vector<Case> cases;
    for (const char* key : {"aoMaxDistance", "lookaheadDistance"}) {
        cases.push_back({std::string(key) + " 0.5", key, 0.5, true});
        for (const nlohmann::json& bad : {nlohmann::json(0.0), nlohmann::json(-1.0), nlohmann::json("1.0"),
                                          nlohmann::json(nullptr)}) {
            cases.push_back({std::string(key) + " " + (bad.is_null() ? "missing" : bad.dump()), key, bad, false});
        }
    }

    const auto distanceOf = [](const pathtracer::config::PathTracerConfig& config, const char* key) {
        return std::string(key) == "aoMaxDistance" ? config.aoMaxDistance : config.lookaheadDistance;
    };
    const auto otherKey = [](const char* key) {
        return std::string(key) == "aoMaxDistance" ? "lookaheadDistance" : "aoMaxDistance";
    };

    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    const std::optional<pathtracer::config::ProfileConfig> shippedConfig =
        pathtracer::config::loadProfileConfig(shippedPath.string());
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, shippedConfig.has_value(), "the shipped profile does not load, so no row below means anything");
    if (!shippedConfig) {
        return;
    }
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        if (testCase.value.is_null()) {
            edited["pathTracer"].erase(testCase.key);
        } else {
            edited["pathTracer"][testCase.key] = testCase.value;
        }
        const std::filesystem::path path = writeJson("engine_io_profile_distances.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        if (testCase.accepted) {
            std::snprintf(detail, sizeof(detail), "loadProfileConfig rejected or mis-mapped %s",
                          testCase.name.c_str());
            PT_EXPECT(ctx,
                          loaded.has_value() &&
                              distanceOf(loaded->pathTracer, testCase.key) == testCase.value.get<float>() &&
                              distanceOf(loaded->pathTracer, otherKey(testCase.key)) ==
                                  distanceOf(shippedConfig->pathTracer, otherKey(testCase.key)),
                          detail);
        } else {
            std::snprintf(detail, sizeof(detail), "loadProfileConfig accepted %s", testCase.name.c_str());
            PT_EXPECT(ctx, !loaded.has_value(), detail);
        }
    }
}

// The integer counts nothing downstream re-checks. samplesPerPixel 0 is the sharp one: it divides by a zero filter weight, writing NaN.
PT_CHECK(profile_config_integer_counts, Fast, Exact) {
    struct Case {
        std::string name;
        const char* section;
        const char* key;
        nlohmann::json value;  // null = key removed
        bool accepted;
    };
    std::vector<Case> cases;
    // Lowest legal value accepted, then each way of going under it. maxBounces/maxSamples 0 are meaningful: direct-only and unbounded.
    const std::vector<std::pair<const char*, int>> floors = {
        {"samplesPerPixel", 1}, {"maxBounces", 0}, {"russianRouletteStartBounce", 0}, {"maxSamples", 0}};
    for (const auto& [key, floor] : floors) {
        cases.push_back({std::string(key) + " " + std::to_string(floor), "pathTracer", key, floor, true});
        cases.push_back({std::string(key) + " " + std::to_string(floor - 1), "pathTracer", key, floor - 1, false});
        cases.push_back({std::string(key) + " missing", "pathTracer", key, nlohmann::json(nullptr), false});
    }
    // The traced image renderScale multiplies and the denominator of the primary ray's aspect ratio.
    for (const char* key : {"width", "height"}) {
        cases.push_back({std::string("render.") + key + " 1", "render", key, 1, true});
        cases.push_back({std::string("render.") + key + " 0", "render", key, 0, false});
        cases.push_back({std::string("render.") + key + " -1", "render", key, -1, false});
        cases.push_back({std::string("render.") + key + " missing", "render", key, nlohmann::json(nullptr), false});
    }

    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    const std::optional<pathtracer::config::ProfileConfig> shippedConfig =
        pathtracer::config::loadProfileConfig(shippedPath.string());
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()) + 1);
    PT_EXPECT(ctx, shippedConfig.has_value(), "the shipped profile does not load, so no row below means anything");
    if (!shippedConfig) {
        return;
    }
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        if (testCase.value.is_null()) {
            edited[testCase.section].erase(testCase.key);
        } else {
            edited[testCase.section][testCase.key] = testCase.value;
        }
        const std::filesystem::path path = writeJson("engine_io_profile_counts.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadProfileConfig %s %s", testCase.accepted ? "rejected" : "accepted",
                      testCase.name.c_str());
        PT_EXPECT(ctx, loaded.has_value() == testCase.accepted, detail);
    }
}

// loadImageTexture's typed read: Float16 must equal the IEEE round-to-nearest-even cast, and a source above kHalfMax is rejected there.
PT_CHECK(image_texture_half_load, Fast, Exact) {
    using pathtracer::gfx::ScalarType;
    constexpr int kRgb = pathtracer::gfx::kRgbChannels;
    const auto writeProbe = [](const char* name, const std::vector<float>& values) {
        pathtracer::gfx::HdrImage image{static_cast<int>(values.size()), 1, kRgb, {}};
        for (const float v : values) {
            image.texels.insert(image.texels.end(), {v, v, v});
        }
        const std::filesystem::path path = scratchPath(name);
        return pathtracer::gfx::writeExr(path.string(), image, pathtracer::gfx::ImageRole::Colour) ? std::optional(path) : std::nullopt;
    };
    const auto sameTexels = [](const pathtracer::gfx::ImageTexture& a, const std::vector<float>& expected) {
        for (int x = 0; x < a.width; ++x) {
            const glm::vec3 texel = a.texel(x, 0);
            if (texel.r != expected[static_cast<std::size_t>(x)] || texel.b != expected[static_cast<std::size_t>(x)]) {
                return false;
            }
        }
        return true;
    };
    ctx.plan(5);

    // Exact in binary16: small integers, dyadic fractions, the largest finite value, the smallest normal.
    const std::vector<float> exact = {0.0F, 1.0F, 1.5F, 0.25F, 2048.0F, -3.0F, pathtracer::gfx::kHalfMax, 1.0F / 16384.0F};
    const std::optional<std::filesystem::path> exactPath = writeProbe("engine_io_half_exact.exr", exact);
    const std::optional<pathtracer::gfx::ImageTexture> exact16 =
        exactPath ? pathtracer::gfx::loadImageTexture(exactPath->string(), ScalarType::Float16, kRgb) : std::nullopt;
    const std::optional<pathtracer::gfx::ImageTexture> exact32 =
        exactPath ? pathtracer::gfx::loadImageTexture(exactPath->string(), ScalarType::Float32, kRgb) : std::nullopt;
    PT_EXPECT(ctx, exact16 && exact32 && std::holds_alternative<std::vector<pathtracer::gfx::Half>>(exact16->texels) && sameTexels(*exact16, exact) &&
                           sameTexels(*exact32, exact),
                  "binary16-exact values did not load bit-identically at Float16 and Float32");

    // Arbitrary: needs rounding, including the exact midpoint above 1.0 (ties to even), a subnormal, and one below the smallest subnormal.
    const std::vector<float> arbitrary = {0.1F, 1.0F + pathtracer::gfx::kHalfUnitRoundoff, 3.14159265F, 1.0e-6F, 1.0e-20F, 60000.5F};
    std::vector<float> rounded;
    for (const float v : arbitrary) {
        rounded.push_back(static_cast<float>(static_cast<pathtracer::gfx::Half>(v)));
    }
    const std::optional<std::filesystem::path> arbitraryPath = writeProbe("engine_io_half_arbitrary.exr", arbitrary);
    const std::optional<pathtracer::gfx::ImageTexture> arbitrary16 =
        arbitraryPath ? pathtracer::gfx::loadImageTexture(arbitraryPath->string(), ScalarType::Float16, kRgb) : std::nullopt;
    PT_EXPECT(ctx, arbitrary16 && sameTexels(*arbitrary16, rounded),
                  "Float16 load differs from static_cast<Half> (round to nearest even)");
    const std::optional<pathtracer::gfx::ImageTexture> arbitrary32 =
        arbitraryPath ? pathtracer::gfx::loadImageTexture(arbitraryPath->string(), ScalarType::Float32, kRgb) : std::nullopt;
    PT_EXPECT(ctx, arbitrary32 && sameTexels(*arbitrary32, arbitrary), "Float32 load is not an exact copy");

    // Overflow: 70000 is finite in float and beyond kHalfMax, so Float16 must reject it and Float32 must not.
    const std::optional<std::filesystem::path> overPath = writeProbe("engine_io_half_overflow.exr", {1.0F, 70000.0F});
    PT_EXPECT(ctx, overPath && !pathtracer::gfx::loadImageTexture(overPath->string(), ScalarType::Float16, kRgb),
                  "Float16 accepted a texel above binary16's finite max");
    PT_EXPECT(ctx, overPath && pathtracer::gfx::loadImageTexture(overPath->string(), ScalarType::Float32, kRgb),
                  "Float32 rejected a finite texel");
    for (const std::optional<std::filesystem::path>& path : {exactPath, arbitraryPath, overPath}) {
        if (path) {
            std::filesystem::remove(*path);
        }
    }
}

// loadImageTexture keeps exactly the channels its slot reads: R alone for a scalar map, RGB otherwise, and a missing one is an error.
PT_CHECK(image_texture_channel_selection, Fast, Exact) {
    using pathtracer::gfx::ScalarType;
    constexpr int kWidth = 3;
    constexpr int kHeight = 2;
    constexpr std::size_t kTexels = static_cast<std::size_t>(kWidth) * kHeight;
    const auto probe = [](int x, int y) {
        const auto t = static_cast<float>(x + (kWidth * y));
        return glm::vec3(t, 10.0F + t, 20.0F + t);  // distinct per lane, so a stride or lane slip reads a wrong value
    };
    std::vector<float> rgba;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const glm::vec3 v = probe(x, y);
            rgba.insert(rgba.end(), {v.r, v.g, v.b, 1.0F});
        }
    }
    const std::filesystem::path rgbaPath = scratchPath("engine_io_channels_rgba.exr");
    const bool rgbaWritten = writeExrPlanes(rgbaPath, kWidth, kHeight, {"R", "G", "B", "A"}, rgba);

    // An R-only file, as a DCC writes a scalar map: OpenEXR zero-fills absent slices, so only an explicit check rejects it at RGB.
    const std::filesystem::path redPath = scratchPath("engine_io_channels_r.exr");
    std::vector<float> red(kTexels);
    for (std::size_t i = 0; i < kTexels; ++i) {
        red[i] = static_cast<float>(i);
    }
    const bool redWritten = writeExrPlanes(redPath, kWidth, kHeight, {"R"}, red);

    const auto matches = [&](const pathtracer::gfx::ImageTexture& image, int channels) {
        const auto* texels = std::get_if<std::vector<float>>(&image.texels);
        if (image.channels != channels || texels == nullptr || texels->size() != kTexels * static_cast<std::size_t>(channels)) {
            return false;
        }
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const glm::vec3 expected = channels == pathtracer::gfx::kScalarChannels
                                               ? glm::vec3(probe(x, y).r, 0.0F, 0.0F)
                                               : probe(x, y);
                if (image.texel(x, y) != expected) {
                    return false;
                }
            }
        }
        return true;
    };
    ctx.plan(7);
    PT_EXPECT(ctx, rgbaWritten && redWritten, "could not write the channel probes");
    const std::optional<pathtracer::gfx::ImageTexture> scalar =
        pathtracer::gfx::loadImageTexture(rgbaPath.string(), ScalarType::Float32, pathtracer::gfx::kScalarChannels);
    PT_EXPECT(ctx, scalar && matches(*scalar, pathtracer::gfx::kScalarChannels),
              "a scalar load of an RGBA file did not keep R alone at one float per texel");
    const std::optional<pathtracer::gfx::ImageTexture> rgb =
        pathtracer::gfx::loadImageTexture(rgbaPath.string(), ScalarType::Float32, pathtracer::gfx::kRgbChannels);
    PT_EXPECT(ctx, rgb && matches(*rgb, pathtracer::gfx::kRgbChannels),
              "an RGB load of an RGBA file did not keep exact RGB at three floats per texel");
    PT_EXPECT(ctx, pathtracer::gfx::loadImageTexture(redPath.string(), ScalarType::Float32, pathtracer::gfx::kScalarChannels),
              "an R-only file was rejected by a scalar slot, which reads R alone");
    std::cout << "  the stderr diagnostics below are expected: an R-only file has no G or B channel to read\n";
    PT_EXPECT(ctx, !pathtracer::gfx::loadImageTexture(redPath.string(), ScalarType::Float32, pathtracer::gfx::kRgbChannels),
              "an R-only file was accepted by an RGB slot, which would have read zero-filled G and B");
    const std::optional<pathtracer::gfx::HdrImage> redExr = pathtracer::gfx::loadImage(redPath.string(), pathtracer::gfx::ImageRole::Data);
    PT_EXPECT(ctx, redExr && redExr->channels == pathtracer::gfx::kScalarChannels && redExr->texels == red,
              "loadImage did not read an R-only file as its one channel");
    const std::optional<pathtracer::gfx::HdrImage> rgbExr = pathtracer::gfx::loadImage(rgbaPath.string(), pathtracer::gfx::ImageRole::Data);
    const auto rgbExrMatches = [&] {
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                if (rgbExr->rgb((static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x)) != probe(x, y)) {
                    return false;
                }
            }
        }
        return true;
    };
    PT_EXPECT(ctx, rgbExr && rgbExr->channels == pathtracer::gfx::kRgbChannels && rgbExrMatches(),
              "loadImage of an RGBA file did not return its exact RGB at three floats per texel");
    std::filesystem::remove(rgbaPath);
    std::filesystem::remove(redPath);
}

// sampleBilinear at Float16 against Float32: the bound is derived from u = 2^-11 and gamma_6 (Higham 2002, 3.1), never fitted.
PT_CHECK(image_texture_bilinear_half_bound, Fast, Exact) {
    using pathtracer::gfx::ScalarType;
    constexpr int kWidth = 13;
    constexpr int kHeight = 7;
    constexpr int kSamples = 20000;
    const double unitRoundoff = std::numeric_limits<float>::epsilon() / 2.0;
    const double gamma6 = 6.0 * unitRoundoff / (1.0 - (6.0 * unitRoundoff));
    const double bound = pathtracer::gfx::kHalfUnitRoundoff + (2.0 * gamma6);

    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    // Normal range of binary16: [2^-14, kHalfMax], log-uniform so every binade is exercised.
    std::uniform_real_distribution<float> logValue(-14.0F, std::log2(pathtracer::gfx::kHalfMax));
    std::uniform_real_distribution<float> unit(-1.0F, 2.0F);  // beyond [0,1] so both wrap directions are covered
    // Both strides: a scalar map filters one lane through its own code path, so it needs the bound checked separately.
    ctx.plan(4);
    for (const int channels : {pathtracer::gfx::kScalarChannels, pathtracer::gfx::kRgbChannels}) {
        std::vector<float> texels(static_cast<std::size_t>(kWidth) * kHeight * static_cast<std::size_t>(channels));
        for (float& v : texels) {
            v = std::exp2(logValue(rng));
        }
        const pathtracer::gfx::ImageTexture full{kWidth, kHeight, channels, texels};
        const pathtracer::gfx::ImageTexture half{kWidth, kHeight, channels,
                                                 std::vector<pathtracer::gfx::Half>(texels.begin(), texels.end())};
        double worst = 0.0;
        double largest = 0.0;
        for (int i = 0; i < kSamples; ++i) {
            const glm::vec2 uv(unit(rng), unit(rng));
            const glm::vec3 s32 = pathtracer::gfx::sampleBilinear(full, uv);
            const glm::vec3 s16 = pathtracer::gfx::sampleBilinear(half, uv);
            for (int c = 0; c < channels; ++c) {
                const double relative = std::fabs(static_cast<double>(s16[c]) - s32[c]) / s32[c];
                worst = std::max(worst, relative / bound);
                largest = std::max(largest, relative);
            }
        }
        char detail[192];
        std::snprintf(detail, sizeof(detail), "%d-channel: worst |s16 - s32| / s32 is %.4g of the derived bound %.4g",
                      channels, worst, bound);
        PT_EXPECT(ctx, worst <= 1.0, detail);
        std::snprintf(detail, sizeof(detail), "%d-channel: Float16 and Float32 never differed, the half path was not exercised",
                      channels);
        PT_EXPECT(ctx, largest > 0.0, detail);
    }
}

// The film-back catalogue's contract: every preset's dimensions feed verticalFovRadians() as a denominator and an aspect ratio.
PT_CHECK(film_back_presets_are_physically_valid, Fast, Exact) {
    ctx.plan(2);
    const std::filesystem::path sensor = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "sensor.json";
    const std::optional<std::vector<pathtracer::scene::Camera::FilmBackPreset>> presets =
        pathtracer::config::loadFilmBackPresets(sensor.string());
    if (!presets.has_value()) {
        PT_EXPECT(ctx, false, "loadFilmBackPresets rejected the shipped sensor.json");
        PT_EXPECT(ctx, false, "presets unavailable");
        return;
    }
    PT_EXPECT(ctx, !presets->empty(), "the shipped film-back catalogue is empty");
    bool allPositive = true;
    for (const pathtracer::scene::Camera::FilmBackPreset& preset : *presets) {
        allPositive = allPositive && preset.filmBack.widthMm > 0.0F && preset.filmBack.heightMm > 0.0F;
    }
    PT_EXPECT(ctx, allPositive, "a film-back preset has a non-positive dimension");
}

// Each preset passes Camera::validFilmBack alone, as the HUD swaps any in; 1e39, a finite double past FLT_MAX, is refused at the read.
PT_CHECK(film_back_presets_reject_non_finite_or_non_positive, Fast, Exact) {
    const std::vector<std::pair<std::string, bool>> cases = {
        {"36.0, \"heightMm\": 24.0", true},
        {"36.0, \"heightMm\": 0.0", false},
        {"-36.0, \"heightMm\": 24.0", false},
        {"36.0, \"heightMm\": 1e39", false},
    };
    ctx.plan(static_cast<int>(cases.size()));
    for (const auto& [dimensions, accepted] : cases) {
        const std::filesystem::path path =
            writeJson("engine_io_sensor.json", "[{\"name\": \"gate\", \"widthMm\": " + dimensions + "}]");
        const bool loaded = pathtracer::config::loadFilmBackPresets(path.string()).has_value();
        std::filesystem::remove(path);
        PT_EXPECT(ctx, loaded == accepted,
                  "loadFilmBackPresets " + std::string(accepted ? "rejected" : "accepted") + " widthMm " + dimensions);
    }
}

// Every float field of every loader refuses +-1e39, a finite double that get<float>() would narrow to inf before any range check.
PT_CHECK(json_float_reads_refuse_float_overflow, Fast, Exact) {
    struct Loader {
        const char* name;
        nlohmann::json base;
        bool (*loads)(const std::string&);
        std::vector<const char*> floatFields;  // JSON pointers into base
    };
    const nlohmann::json light = {{"type", "quad"},      {"position", {0, 0, 0}}, {"rotation", {0, 0, 0}}, {"size", {1, 1}},
                                  {"color", {1, 1, 1}}, {"intensity", 5.0}};
    const nlohmann::json scene = {
        {"model", {{"gltfPath", "geometry/cornell/cornell_v001.gltf"}, {"position", {0, 0, 0}}, {"rotation", {0, 0, 0}}}},
        {"environment", {{"hdriPath", "textures/republiqueHDR_2k.exr"}}},
        {"materialPath", "materials/clay.json"},
        {"lights", {light}}};
    // Every optional key present, so a pointer into it overwrites a number rather than creating a malformed sibling.
    const nlohmann::json material = {{"diffuseColour", {1, 1, 1}}, {"roughnessFactor", 0.5},  {"roughnessMin", 0.045},
                                     {"roughnessMax", 1.0},        {"bumpStrength", 0.0},     {"ior", 1.5},
                                     {"abbe", 0.0},                {"transmissionFactor", 0.0}, {"metallicFactor", 0.0},
                                     {"diffuseRoughness", 0.0},    {"transmissionColor", {1, 1, 1}},
                                     {"transmissionDepth", 0.0},   {"edgeTint", {1, 1, 1}}};
    std::ifstream shippedProfile(std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json");
    const std::vector<Loader> loaders = {
        {"loadSceneConfig", scene,
         [](const std::string& path) { return pathtracer::config::loadSceneConfig(path).has_value(); },
         {"/model/position/0", "/model/rotation/1", "/lights/0/position/2", "/lights/0/rotation/0", "/lights/0/size/1",
          "/lights/0/color/0", "/lights/0/intensity"}},
        {"loadMaterialConfig", material,
         [](const std::string& path) { return pathtracer::config::loadMaterialConfig(path).has_value(); },
         {"/diffuseColour/0", "/roughnessFactor", "/roughnessMin", "/roughnessMax", "/bumpStrength", "/ior", "/abbe",
          "/transmissionFactor", "/metallicFactor", "/diffuseRoughness", "/transmissionColor/1", "/transmissionDepth",
          "/edgeTint/2"}},
        {"loadMaterialConfig constant", {{"shadingModel", "constant"}, {"diffuseColour", {1, 1, 1}}},
         [](const std::string& path) { return pathtracer::config::loadMaterialConfig(path).has_value(); },
         {"/diffuseColour/2"}},
        {"loadProfileConfig", nlohmann::json::parse(shippedProfile),
         [](const std::string& path) { return pathtracer::config::loadProfileConfig(path).has_value(); },
         {"/camera/position/0", "/camera/rotation/0", "/camera/rotation/2", "/camera/focalLengthMm", "/camera/nearClip",
          "/camera/farClip", "/camera/aperture", "/camera/shutterSeconds", "/camera/iso", "/camera/lens/radialCoefficients/3",
          "/camera/lens/maxFieldOfViewDegrees", "/controls/flySpeedMetersPerSecond", "/controls/orbitSensitivityDegPerPixel",
          "/render/renderScale", "/render/interactiveRenderScale", "/pathTracer/aoMaxDistance", "/pathTracer/lookaheadDistance"}},
        {"loadFilmBackPresets", nlohmann::json::array({{{"name", "gate"}, {"widthMm", 36.0}, {"heightMm", 24.0}}}),
         [](const std::string& path) { return pathtracer::config::loadFilmBackPresets(path).has_value(); },
         {"/0/widthMm", "/0/heightMm"}},
    };

    int planned = 0;
    for (const Loader& loader : loaders) {
        planned += 1 + (2 * static_cast<int>(loader.floatFields.size()));
    }
    ctx.plan(planned + 3);
    for (const Loader& loader : loaders) {
        // Anti-vacuity: each row below differs from this base in one number only.
        const std::filesystem::path basePath = writeJson("engine_io_float_base.json", loader.base.dump());
        PT_EXPECT(ctx, loader.loads(basePath.string()), std::string(loader.name) + " rejected its unmutated base");
        std::filesystem::remove(basePath);
        for (const char* field : loader.floatFields) {
            for (const double overflow : {1e39, -1e39}) {
                nlohmann::json edited = loader.base;
                edited[nlohmann::json::json_pointer(field)] = overflow;
                const std::filesystem::path path = writeJson("engine_io_float_overflow.json", edited.dump());
                PT_EXPECT(ctx, !loader.loads(path.string()),
                          std::string(loader.name) + " accepted " + field + " = " + nlohmann::json(overflow).dump());
                std::filesystem::remove(path);
            }
        }
    }

    // The exact boundary, on intensity, which has no upper bound of its own: below FLT_MAX + ulp/2 rounds to FLT_MAX, the tie to inf.
    const double floatMax = std::numeric_limits<float>::max();
    const double halfUlp = 0.5 * (floatMax - std::nextafter(std::numeric_limits<float>::max(), 0.0F));
    const auto intensityLoaded = [&](double intensity) {
        nlohmann::json edited = scene;
        edited["lights"][0]["intensity"] = intensity;
        const std::filesystem::path path = writeJson("engine_io_float_boundary.json", edited.dump());
        const std::optional<pathtracer::config::SceneConfig> loaded = pathtracer::config::loadSceneConfig(path.string());
        std::filesystem::remove(path);
        return loaded ? std::optional(loaded->lights[0].intensity) : std::nullopt;
    };
    PT_EXPECT(ctx, intensityLoaded(floatMax) == std::numeric_limits<float>::max(), "FLT_MAX itself did not load exactly");
    PT_EXPECT(ctx, intensityLoaded(std::nextafter(floatMax + halfUlp, 0.0)) == std::numeric_limits<float>::max(),
              "the largest double rounding to FLT_MAX did not load as FLT_MAX");
    PT_EXPECT(ctx, !intensityLoaded(floatMax + halfUlp).has_value(), "FLT_MAX + ulp/2, which rounds to inf, loaded");
}

// Every appended record is exactly one line that parses back with every schema field, and appending never rewrites earlier lines.
PT_CHECK(bench_log_appends_one_parseable_line_per_record, Fast, Exact) {
    ctx.plan(5);
    const std::filesystem::path path = scratchPath("engine_io_validate_bench.jsonl");
    std::filesystem::remove(path);
    const pathtracer::debug::BenchRecord first{"io_validate", {"io_validate", "--flag"}, {{"width", 7}}, {{"ms", {1.5, 2.5}}}, {{"crc32", 1}}};
    const pathtracer::debug::BenchRecord second{"io_validate", {"io_validate"}, {{"width", 9}}, {{"ms", {3.0}}}, {{"crc32", 2}}};
    PT_EXPECT(ctx, pathtracer::debug::appendBenchRecord(path.string(), first) && pathtracer::debug::appendBenchRecord(path.string(), second), "appendBenchRecord failed on a writable scratch path");

    std::vector<nlohmann::json> records;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        records.push_back(nlohmann::json::parse(line, nullptr, /*allow_exceptions=*/false));
    }
    PT_EXPECT(ctx, records.size() == 2, "expected exactly two lines after two appends");

    const auto complete = [](const nlohmann::json& r) {
        return !r.is_discarded() && r.value("schema", 0) == pathtracer::debug::kBenchLogSchema && r.contains("time_utc") && r.contains("pid") &&
               !r["build"].value("git", std::string()).empty() && r["build"].value("uuid", std::string()).size() == 32 &&
               r["host"].value("logical_cpus", 0) > 0 && r["rusage"].contains("user_s") && r["rusage"].contains("nivcsw");
    };
    PT_EXPECT(ctx, records.size() == 2 && complete(records[0]) && complete(records[1]), "a record is missing a provenance or rusage field");
    PT_EXPECT(ctx, records.size() == 2 && records[0]["config"] == first.config && records[0]["samples"] == first.samples && records[0]["argv"] == first.argv,
                  "first record's caller-supplied content did not round-trip");
    PT_EXPECT(ctx, records.size() == 2 && records[1]["config"] == second.config && records[1]["work"] == second.work,
                  "second record's caller-supplied content did not round-trip");
    std::filesystem::remove(path);
}

// A path that cannot be opened is reported as a failure, never as a silently skipped record.
PT_CHECK(bench_log_rejects_unwritable_path, Fast, Exact) {
    ctx.plan(1);
    const std::filesystem::path path = scratchPath("engine_io_validate_no_such_dir") / "bench.jsonl";
    std::filesystem::remove_all(path.parent_path());
    const pathtracer::debug::BenchRecord record{"io_validate", {}, nlohmann::json::object(), nlohmann::json::object(), nlohmann::json::object()};
    PT_EXPECT(ctx, !pathtracer::debug::appendBenchRecord(path.string(), record), "appendBenchRecord reported success writing into a missing directory");
}

// Known answers from an independent implementation (Python's zlib.crc32 over the same little-endian bytes).
PT_CHECK(float_crc32_matches_reference, Fast, Exact) {
    ctx.plan(2);
    const std::vector<float> zero{0.0F};
    const std::vector<float> mixed{1.0F, -2.5F, 0.1F};
    PT_EXPECT(ctx, pathtracer::debug::floatCrc32(zero) == 0x2144DF1CU, "CRC-32 of four zero bytes is not 0x2144DF1C");
    PT_EXPECT(ctx, pathtracer::debug::floatCrc32(mixed) == 2706677804U, "CRC-32 of {1, -2.5, 0.1} disagrees with zlib.crc32");
}

// One glTF primitive's fixture data, written verbatim so each check states the exact attributes it exercises; empty means absent.
struct GltfPrimitive {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> tangents;
    std::vector<unsigned int> indices;
    int mode = 4;  // glTF primitive.mode: 0 POINTS, 1 LINES, 4 TRIANGLES, 5 TRIANGLE_STRIP, 6 TRIANGLE_FAN
};

// Writes a one-mesh .gltf and its .bin to scratch, the mesh instanced by one node per entry of nodeScales; returns the .gltf path.
std::filesystem::path writeGltf(const std::string& stem, const std::vector<GltfPrimitive>& prims,
                                const std::vector<glm::vec3>& nodeScales) {
    std::vector<unsigned char> bin;
    nlohmann::json bufferViews = nlohmann::json::array();
    nlohmann::json accessors = nlohmann::json::array();
    const auto addAccessor = [&](const auto& values, int componentType, const char* type) {
        const auto* first = reinterpret_cast<const unsigned char*>(values.data());
        const std::size_t bytes = values.size() * sizeof(values[0]);
        bufferViews.push_back({{"buffer", 0}, {"byteOffset", bin.size()}, {"byteLength", bytes}});
        bin.insert(bin.end(), first, first + bytes);
        accessors.push_back({{"bufferView", bufferViews.size() - 1}, {"componentType", componentType}, {"count", values.size()}, {"type", type}});
        return accessors.size() - 1;
    };
    constexpr int kFloat = 5126;
    constexpr int kUnsignedInt = 5125;
    nlohmann::json primitives = nlohmann::json::array();
    for (const GltfPrimitive& prim : prims) {
        nlohmann::json primitive = {{"mode", prim.mode}, {"attributes", {{"POSITION", addAccessor(prim.positions, kFloat, "VEC3")}}}};
        if (!prim.normals.empty()) {
            primitive["attributes"]["NORMAL"] = addAccessor(prim.normals, kFloat, "VEC3");
        }
        if (!prim.uvs.empty()) {
            primitive["attributes"]["TEXCOORD_0"] = addAccessor(prim.uvs, kFloat, "VEC2");
        }
        if (!prim.tangents.empty()) {
            primitive["attributes"]["TANGENT"] = addAccessor(prim.tangents, kFloat, "VEC4");
        }
        if (!prim.indices.empty()) {
            primitive["indices"] = addAccessor(prim.indices, kUnsignedInt, "SCALAR");
        }
        primitives.push_back(primitive);
    }
    nlohmann::json nodes = nlohmann::json::array();
    nlohmann::json sceneNodes = nlohmann::json::array();
    for (const glm::vec3& scale : nodeScales) {
        sceneNodes.push_back(nodes.size());
        nodes.push_back({{"mesh", 0}, {"scale", {scale.x, scale.y, scale.z}}});
    }
    const std::string binName = stem + ".bin";
    const nlohmann::json gltf = {{"asset", {{"version", "2.0"}}},
                                 {"scene", 0},
                                 {"scenes", {{{"nodes", sceneNodes}}}},
                                 {"nodes", nodes},
                                 {"meshes", {{{"primitives", primitives}}}},
                                 {"accessors", accessors},
                                 {"bufferViews", bufferViews},
                                 {"buffers", {{{"uri", binName}, {"byteLength", bin.size()}}}}};
    std::ofstream(scratchPath(binName.c_str()), std::ios::binary)
        .write(reinterpret_cast<const char*>(bin.data()), static_cast<std::streamsize>(bin.size()));
    const std::filesystem::path path = scratchPath((stem + ".gltf").c_str());
    std::ofstream(path) << gltf.dump();
    return path;
}

// Unit quad in z=0 facing +z, wound counter-clockwise from +z, with p = (u, 1-v): +x is +u and +y is image-up.
GltfPrimitive makeQuad() {
    const glm::vec3 up(0.0F, 0.0F, 1.0F);
    return GltfPrimitive{{{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 0.0F}, {0.0F, 1.0F, 0.0F}},
                         {up, up, up, up},
                         {{0.0F, 1.0F}, {1.0F, 1.0F}, {1.0F, 0.0F}, {0.0F, 0.0F}},
                         std::vector<glm::vec4>(4, glm::vec4(1.0F, 0.0F, 0.0F, 1.0F)),
                         {0, 1, 2, 0, 2, 3}};
}

// glTF 2.0 3.7.2.1: winding follows the determinant, so a mirrored node must keep its face normal on the shading normal's side.
PT_CHECK(gltf_mirrored_transform_keeps_geometric_and_shading_normals_agreed, Fast, Exact) {
    // Identity, a single-axis mirror and a non-uniform mirror: the handedness rule is sign(det M), not orthogonality.
    const std::vector<glm::vec3> scales = {{1.0F, 1.0F, 1.0F}, {-1.0F, 1.0F, 1.0F}, {2.0F, -0.5F, 3.0F}};
    const std::optional<pathtracer::scene::LoadedModel> model =
        pathtracer::scene::loadGltf(writeGltf("engine_io_validate_mirror", {makeQuad()}, scales).string());
    constexpr int kTrianglesPerQuad = 2;
    ctx.plan(1 + (static_cast<int>(scales.size()) * kTrianglesPerQuad * 2));
    PT_EXPECT(ctx, model && model->shadingTriangles.size() == scales.size() * kTrianglesPerQuad, "the fixture did not load as one quad per node");
    if (!model) {
        return;
    }
    for (const pathtracer::scene::ShadingTriangle& tri : model->shadingTriangles) {
        const glm::mat3 linear(model->instances[static_cast<std::size_t>(tri.instanceIndex)].transform);
        const pathtracer::scene::ShadingVertex shading = pathtracer::scene::interpolateShading(tri, 1.0F / 3.0F, 1.0F / 3.0F);
        PT_EXPECT(ctx, glm::dot(pathtracer::scene::geometricNormalOf(tri), shading.normal) > 0.0F,
                  "the geometric normal opposes the shading normal");
        // The authored bitangent is cross(+z, +x) * 1 = +y; a vector field maps by M, so the world bitangent must point along M * y.
        const glm::vec3 bitangent = glm::cross(shading.normal, glm::vec3(shading.tangent)) * shading.tangent.w;
        PT_EXPECT(ctx, glm::dot(bitangent, linear * glm::vec3(0.0F, 1.0F, 0.0F)) > 0.0F, "the world bitangent opposes M * B");
    }
}

// A planar primitive in z=0 with every attribute present and normals +z, so only topology and winding vary between rows.
GltfPrimitive planar(const std::vector<glm::vec2>& xy, std::vector<unsigned int> indices, int mode) {
    GltfPrimitive prim;
    for (const glm::vec2& p : xy) {
        prim.positions.emplace_back(p, 0.0F);
        prim.normals.emplace_back(0.0F, 0.0F, 1.0F);
        prim.uvs.push_back(p);
        prim.tangents.emplace_back(1.0F, 0.0F, 0.0F, 1.0F);
    }
    prim.indices = std::move(indices);
    prim.mode = mode;
    return prim;
}

// glTF 2.0 3.7.2.1: strip faces alternate vertex order and fans pivot on v0, so every face of a consistently wound ribbon faces +z.
PT_CHECK(gltf_strip_and_fan_triangulate_with_consistent_winding, Fast, Exact) {
    constexpr int kTriangles = 4;
    constexpr int kFan = 6;
    // Zigzag ribbon, first face counter-clockwise from +z: a strip that does not alternate vertex order flips every odd face.
    const std::vector<glm::vec2> ribbon = {{0.0F, 1.0F}, {0.0F, 0.0F}, {1.0F, 1.0F}, {1.0F, 0.0F}, {2.0F, 1.0F}, {2.0F, 0.0F}};
    // Fan centre then five rim points counter-clockwise from +z.
    const std::vector<glm::vec2> fan = {{0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F}, {-1.0F, 1.0F}, {-1.0F, 0.0F}};
    // The quad's indexed list against its non-indexed expansion: identical faces in identical order, bit for bit.
    const std::vector<glm::vec2> quad = {{0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F}};
    const std::vector<glm::vec2> quadExpanded = {quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]};
    const std::optional<pathtracer::scene::LoadedModel> strip = pathtracer::scene::loadGltf(
        writeGltf("engine_io_validate_strip", {planar(ribbon, {0, 1, 2, 3, 4, 5}, 5)}, {glm::vec3(1.0F)}).string());
    const std::optional<pathtracer::scene::LoadedModel> fanned = pathtracer::scene::loadGltf(
        writeGltf("engine_io_validate_fan", {planar(fan, {}, 6)}, {glm::vec3(1.0F)}).string());
    const std::optional<pathtracer::scene::LoadedModel> indexed = pathtracer::scene::loadGltf(
        writeGltf("engine_io_validate_indexed", {planar(quad, {0, 1, 2, 0, 2, 3}, 4)}, {glm::vec3(1.0F)}).string());
    const std::optional<pathtracer::scene::LoadedModel> unindexed = pathtracer::scene::loadGltf(
        writeGltf("engine_io_validate_unindexed", {planar(quadExpanded, {}, 4)}, {glm::vec3(1.0F)}).string());
    ctx.plan(4 + kTriangles + kFan - 2);
    PT_EXPECT(ctx, strip && strip->worldTriangles.size() == kTriangles, "a 6-vertex strip did not give 4 triangles");
    PT_EXPECT(ctx, fanned && fanned->worldTriangles.size() == kFan - 2, "a non-indexed 6-vertex fan did not give 4 triangles");
    for (const std::optional<pathtracer::scene::LoadedModel>* model : {&strip, &fanned}) {
        for (std::size_t i = 0; *model && i < (*model)->shadingTriangles.size(); ++i) {
            PT_EXPECT(ctx, pathtracer::scene::geometricNormalOf((*model)->shadingTriangles[i]).z > 0.0F, "a face is wound clockwise from +z");
        }
    }
    const auto samePositions = [](const pathtracer::scene::Triangle& a, const pathtracer::scene::Triangle& b) {
        return a.v0 == b.v0 && a.v1 == b.v1 && a.v2 == b.v2;
    };
    PT_EXPECT(ctx, indexed && unindexed && indexed->worldTriangles.size() == 2 && unindexed->worldTriangles.size() == 2,
              "the indexed or non-indexed quad did not give 2 triangles");
    PT_EXPECT(ctx, indexed && unindexed && indexed->worldTriangles.size() == unindexed->worldTriangles.size() &&
                       std::equal(indexed->worldTriangles.begin(), indexed->worldTriangles.end(), unindexed->worldTriangles.begin(), samePositions),
              "the non-indexed quad's faces differ from the indexed quad's");
}

// Points and lines have no area: they are skipped with the rest of the file kept, and a file holding nothing else still fails.
PT_CHECK(gltf_points_and_lines_are_skipped_not_fatal, Fast, Exact) {
    ctx.plan(3);
    GltfPrimitive points;
    points.positions = {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}};
    points.mode = 0;
    GltfPrimitive lines = points;
    lines.mode = 1;
    const std::optional<pathtracer::scene::LoadedModel> mixed = pathtracer::scene::loadGltf(
        writeGltf("engine_io_validate_mixed_topology", {points, makeQuad(), lines}, {glm::vec3(1.0F)}).string());
    PT_EXPECT(ctx, mixed.has_value(), "a point and a line primitive beside a triangle primitive failed the load");
    PT_EXPECT(ctx, mixed && mixed->instances.size() == 1 && mixed->worldTriangles.size() == 2,
              "the triangle primitive alone did not survive as one instance of 2 triangles");
    PT_EXPECT(ctx, !pathtracer::scene::loadGltf(writeGltf("engine_io_validate_points_only", {points}, {glm::vec3(1.0F)}).string()),
              "a file holding only points loaded as if it had a surface");
}

// glTF 2.0 3.9.3: tangent-space +x is +u and +y is image-up, i.e. -v; makeQuad maps +u to +x and image-up to +y, so T = +x, w = +1.
PT_CHECK(gltf_generated_tangent_follows_the_gltf_convention, Fast, Exact) {
    GltfPrimitive quad = makeQuad();
    quad.tangents.clear();
    GltfPrimitive flipped = quad;
    for (glm::vec2& uv : flipped.uvs) {
        uv.x = 1.0F - uv.x;  // u now runs along -x, a mirrored chart: T = -x and the frame turns left-handed, w = -1
    }
    // Identity, a mirrored node (T = M * +x = -x and w = sign(det M) = -1), and the mirrored chart.
    const std::optional<pathtracer::scene::LoadedModel> plain = pathtracer::scene::loadGltf(
        writeGltf("engine_io_validate_generated", {quad}, {glm::vec3(1.0F), glm::vec3(-1.0F, 1.0F, 1.0F)}).string());
    const std::optional<pathtracer::scene::LoadedModel> chart = pathtracer::scene::loadGltf(
        writeGltf("engine_io_validate_generated_flipped", {flipped}, {glm::vec3(1.0F)}).string());
    constexpr int kCorners = 6;
    ctx.plan(2 + (3 * kCorners));
    PT_EXPECT(ctx, plain && plain->shadingTriangles.size() == 4, "the untangented quad did not load under both nodes");
    PT_EXPECT(ctx, chart && chart->shadingTriangles.size() == 2, "the mirrored-chart quad did not load");
    // Exact up to two normalizations (MikkTSpace's, then the loader's after transforming), each within 2 ulp of unit length.
    const auto expect = [&](const pathtracer::scene::ShadingTriangle& tri, const glm::vec4& expected) {
        for (const pathtracer::scene::ShadingVertex* vertex : {&tri.v0, &tri.v1, &tri.v2}) {
            const glm::vec4 delta = glm::abs(vertex->tangent - expected);
            PT_EXPECT(ctx, std::max({delta.x, delta.y, delta.z}) <= 4.0F * std::numeric_limits<float>::epsilon() && delta.w == 0.0F,
                      "a generated tangent is not the analytic +u direction with the glTF handedness");
        }
    };
    for (std::size_t i = 0; plain && i < plain->shadingTriangles.size(); ++i) {
        const pathtracer::scene::ShadingTriangle& tri = plain->shadingTriangles[i];
        expect(tri, tri.instanceIndex == 0 ? glm::vec4(1.0F, 0.0F, 0.0F, 1.0F) : glm::vec4(-1.0F, 0.0F, 0.0F, -1.0F));
    }
    for (std::size_t i = 0; chart && i < chart->shadingTriangles.size(); ++i) {
        expect(chart->shadingTriangles[i], glm::vec4(-1.0F, 0.0F, 0.0F, -1.0F));
    }
}

// The stump's TANGENT came from its exporter's MikkTSpace; regenerating it from the same mesh must agree in handedness everywhere.
PT_CHECK(gltf_generated_tangents_agree_with_authored_mikktspace, Fast, Exact) {
    ctx.plan(4);
    const std::filesystem::path source = std::filesystem::path(ASSET_ROOT_DIR) / "geometry/broken_stump_rkswd_raw/rkswd_tier_2.gltf";
    nlohmann::json gltf = nlohmann::json::parse(std::ifstream(source));
    for (nlohmann::json& mesh : gltf.at("meshes")) {
        for (nlohmann::json& primitive : mesh.at("primitives")) {
            primitive.at("attributes").erase("TANGENT");
        }
    }
    // cgltf resolves a buffer uri against the .gltf's own directory, so the stripped copy sits beside a copy of the .bin.
    const std::string binName = gltf.at("buffers").at(0).at("uri").get<std::string>();
    std::filesystem::copy_file(source.parent_path() / binName, scratchPath(binName.c_str()), std::filesystem::copy_options::overwrite_existing);
    const std::filesystem::path stripped = scratchPath("engine_io_validate_stump_untangented.gltf");
    std::ofstream(stripped) << gltf.dump();

    const std::optional<pathtracer::scene::LoadedModel> authored = pathtracer::scene::loadGltf(source.string());
    const std::optional<pathtracer::scene::LoadedModel> generated = pathtracer::scene::loadGltf(stripped.string());
    PT_EXPECT(ctx, authored && generated && authored->shadingTriangles.size() == generated->shadingTriangles.size(),
              "the authored and stripped stump did not load to the same triangle count");
    if (!authored || !generated || authored->shadingTriangles.size() != generated->shadingTriangles.size()) {
        return;
    }
    // A zero-UV-area face has no tangent of its own; MikkTSpace hands it a neighbour's, so its corners are not compared.
    int compared = 0;
    int handednessMismatches = 0;
    int opposedTangents = 0;
    float maxAngle = 0.0F;
    for (std::size_t i = 0; i < authored->shadingTriangles.size(); ++i) {
        const pathtracer::scene::ShadingTriangle& a = authored->shadingTriangles[i];
        const pathtracer::scene::ShadingTriangle& g = generated->shadingTriangles[i];
        const glm::vec2 e1 = a.v1.uv - a.v0.uv;
        const glm::vec2 e2 = a.v2.uv - a.v0.uv;
        if ((e1.x * e2.y) - (e1.y * e2.x) == 0.0F) {
            continue;
        }
        for (const auto& [va, vg] : {std::pair{&a.v0, &g.v0}, std::pair{&a.v1, &g.v1}, std::pair{&a.v2, &g.v2}}) {
            const float cosine = glm::dot(glm::vec3(va->tangent), glm::vec3(vg->tangent));
            ++compared;
            handednessMismatches += va->tangent.w != vg->tangent.w ? 1 : 0;
            opposedTangents += cosine <= 0.0F ? 1 : 0;
            maxAngle = std::max(maxAngle, std::acos(std::clamp(cosine, -1.0F, 1.0F)));
        }
    }
    std::cout << "  stump: " << compared << " corners compared, max tangent angle " << glm::degrees(maxAngle) << " deg\n";
    PT_EXPECT(ctx, compared > 0, "no stump corner had a non-degenerate UV face to compare");
    PT_EXPECT(ctx, handednessMismatches == 0, "a generated tangent's handedness disagrees with the exporter's MikkTSpace");
    PT_EXPECT(ctx, opposedTangents == 0, "a generated tangent points away from the exporter's MikkTSpace tangent");
}

}  // namespace

PT_CHECK_MAIN("io")
