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
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <OpenColorIO/OpenColorIO.h>
#include <OpenImageIO/color.h>
#include <OpenImageIO/imageio.h>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include "check.h"
#include "fixtures.h"
#include "pathtracer/config/profile_config.h"
#include "pathtracer/config/scene_config.h"
#include "pathtracer/debug/aov.h"
#include "pathtracer/debug/bench_log.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/scalar_type.h"
#include "pathtracer/gfx/ocio_cpu_transform.h"
#include "pathtracer/gfx/ocio_display_transform.h"
#include "pathtracer/gfx/texture.h"
#include "pathtracer/scene/cie.h"
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

// A hexagonal EXR names its lattice and carries its 1 x sqrt(3)/2 cell as PixelAspectRatio; a square one adds neither, so it is unchanged.
PT_CHECK(exr_write_declares_its_pixel_lattice, Fast, Exact) {
    ctx.plan(3);
    const pathtracer::gfx::HdrImage image{2, 2, pathtracer::gfx::kRgbChannels, std::vector<float>(12, 0.5F)};
    const std::filesystem::path path = scratchPath("engine_io_validate_lattice.exr");
    std::optional<OIIO::ImageSpec> square;
    std::optional<OIIO::ImageSpec> hexagonal;
    if (pathtracer::gfx::writeExr(path.string(), image, pathtracer::gfx::ImageRole::Colour)) {
        square = headerOf(path);
    }
    if (pathtracer::gfx::writeExr(path.string(), image, pathtracer::gfx::ImageRole::Colour, pathtracer::gfx::PixelLattice::Hexagonal)) {
        hexagonal = headerOf(path);
    }
    std::filesystem::remove(path);
    PT_EXPECT(ctx,
              square && square->find_attribute("pathtracer:pixelLattice") == nullptr &&
                  square->get_float_attribute("PixelAspectRatio", 1.0F) == 1.0F,
              "a square EXR gained a lattice tag or a non-unit pixel aspect");
    PT_EXPECT(ctx, hexagonal && hexagonal->get_string_attribute("pathtracer:pixelLattice") == "Hexagonal",
              "a hexagonal EXR does not name its lattice");
    // EXR's pixel aspect is a displayed cell's width over its height: one column pitch over the sqrt(3)/2 row pitch.
    const float aspect = hexagonal ? hexagonal->get_float_attribute("PixelAspectRatio", 0.0F) : 0.0F;
    PT_EXPECT(ctx, std::abs(static_cast<double>(aspect) - (2.0 / std::sqrt(3.0))) <= std::numeric_limits<float>::epsilon(),
              "a hexagonal EXR's PixelAspectRatio " + std::to_string(aspect) + " is not 2/sqrt(3)");
}

// Colour EXRs carry the working space twice, as OIIO's tag and OpenEXR's chromaticities; the oracle is AP1's own published xy.
PT_CHECK(exr_write_declares_its_colour_space, Fast, Exact) {
    // SMPTE ST 2065-1 / ACES S-2014-004: AP1's R, G, B primaries and the ACES white, the working space's definition.
    constexpr std::array<float, 8> kAp1{0.713F, 0.293F, 0.165F, 0.830F, 0.128F, 0.044F, 0.32168F, 0.33767F};
    // Half a unit in the fifth decimal, the precision the ACES white is published at.
    constexpr float kPublishedPrecision = 5e-6F;
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
        for (std::size_t i = 0; i < kAp1.size(); ++i) {
            worst = std::max(worst, std::abs(static_cast<const float*>(chromaticities->data())[i] - kAp1[i]));
        }
    }
    char detail[160];
    std::snprintf(detail, sizeof(detail), "Colour EXR chromaticities differ from AP1 by %.3g, over its published %.3g", worst,
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
    const std::filesystem::path path = scratchPath("engine_io_validate_rec709.exr");
    const bool written = writeExrPlanes(path, 2, 1, {"R", "G", "B"}, stored, [](OIIO::ImageSpec& spec) {
        spec.attribute("oiio:ColorSpace", "Linear Rec.709 (sRGB)");
    });
    const std::optional<pathtracer::gfx::HdrImage> colour = pathtracer::gfx::loadImage(path.string(), pathtracer::gfx::ImageRole::Colour);
    const std::optional<pathtracer::gfx::HdrImage> data = pathtracer::gfx::loadImage(path.string(), pathtracer::gfx::ImageRole::Data);
    std::filesystem::remove(path);
    PT_EXPECT(ctx, written && colour && colour->texels == oracle("Linear Rec.709 (sRGB)"),
              "a linear Rec.709 EXR did not read as OCIO's conversion to the working space");
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

// Untagged EXRs mean BT.709 by OpenEXR's rule, converted to the scene space; other untagged primaries cannot be honoured, so are refused.
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
    std::vector<float> expected = stored;
    pathtracer::gfx::convertToSceneSpace(expected, pathtracer::gfx::kOcioLinearRec709);
    PT_EXPECT(ctx, asRec709 && asRec709->texels == expected, "an untagged BT.709 EXR did not read as OCIO's linear Rec.709 to the scene space");
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

// A binding is a path string or {path, colorSpace, channel, bump}, and the environment may name its colour space: all reach SceneConfig.
PT_CHECK(scene_config_reads_texture_bindings, Fast, Exact) {
    const std::filesystem::path path = writeJson(
        "engine_io_scene_bindings.json",
        "{\"model\":{\"gltfPath\":\"geometry/cornell/cornell_v001.gltf\",\"position\":[0,0,0],\"rotation\":[0,0,0]},"
        "\"environment\":{\"hdriPath\":\"textures/sky.exr\",\"colorSpace\":\"ACEScg\"},\"materialPath\":\"materials/clay.json\","
        "\"textures\":{\"node\":{\"base_color\":{\"path\":\"albedo.exr\",\"colorSpace\":\"ACEScg\"},"
        "\"specular_roughness\":{\"path\":\"orm.exr\",\"channel\":\"g\"},"
        "\"geometry_normal\":{\"path\":\"normal.exr\",\"bump\":{\"path\":\"height.exr\",\"height\":0.005}},"
        "\"base_metalness\":\"metal.exr\"}}}");
    const std::optional<pathtracer::config::SceneConfig> scene = pathtracer::config::loadSceneConfig(path.string());
    std::filesystem::remove(path);
    ctx.plan(5);
    PT_EXPECT(ctx, scene && scene->environment.colorSpace == "ACEScg", "the environment's colorSpace did not reach SceneConfig");
    const auto slot = [&](const char* name) { return scene ? scene->textures.at("node").at(name) : pathtracer::config::TextureConfig{}; };
    PT_EXPECT(ctx, slot("base_color").path == "albedo.exr" && slot("base_color").colorSpace == "ACEScg",
              "a {path, colorSpace} binding did not read both fields");
    PT_EXPECT(ctx, slot("specular_roughness").path == "orm.exr" && slot("specular_roughness").channel == 1,
              "a {path, channel} binding did not read channel g as 1");
    const std::optional<pathtracer::config::BumpConfig> bump = slot("geometry_normal").bump;
    PT_EXPECT(ctx, slot("geometry_normal").path == "normal.exr" && bump && bump->path == "height.exr" && bump->heightMetres == 0.005F,
              "a geometry_normal binding lost its normal map or its bump");
    PT_EXPECT(ctx, slot("base_metalness").path == "metal.exr" && !slot("base_metalness").colorSpace && slot("base_metalness").channel == 0,
              "a path-string binding gained a colour space or a channel");
}

// Lights and volumes alone make a scene, and its camera block replaces only profile.json's pose and far clip.
PT_CHECK(scene_config_reads_volumes_and_camera_without_a_model, Fast, Exact) {
    const std::filesystem::path path = writeJson(
        "engine_io_scene_volumes.json",
        "{\"environment\":{\"hdriPath\":\"textures/sky.exr\"},"
        "\"volumes\":[{\"vdbPath\":\"cloud.vdb\",\"grid\":\"density\",\"position\":[1,2,3],\"rotation\":[0,90,0],\"scale\":0.5,"
        "\"materialPath\":\"materials/volume.json\"}],"
        "\"camera\":{\"position\":[0,0,2200],\"rotation\":[0,10,0],\"farClip\":4000}}");
    const std::optional<pathtracer::config::SceneConfig> scene = pathtracer::config::loadSceneConfig(path.string());
    std::filesystem::remove(path);
    std::optional<pathtracer::config::ProfileConfig> profile =
        pathtracer::config::loadProfileConfig((std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json").string());
    const pathtracer::config::CameraConfig authored = profile ? profile->camera : pathtracer::config::CameraConfig{};
    if (scene && profile) {
        pathtracer::config::applySceneCamera(*scene, *profile);
    }
    ctx.plan(4);
    PT_EXPECT(ctx, scene && !scene->model.gltfPath && !scene->materialPath && !scene->environment.illuminanceLux,
              "a scene without a model gained a glTF or a material, or an uncalibrated sky an illuminance");
    PT_EXPECT(ctx, scene && scene->volumes.size() == 1 && scene->volumes[0].scale == 0.5F && scene->volumes[0].grid == "density" &&
                       scene->volumes[0].position == glm::vec3(1.0F, 2.0F, 3.0F),
              "the volume entry did not parse");
    PT_EXPECT(ctx, profile && profile->camera.position == glm::vec3(0.0F, 0.0F, 2200.0F) &&
                       profile->camera.rotation == glm::vec3(0.0F, 10.0F, 0.0F) && profile->camera.farClip == 4000.0F,
              "the scene camera's pose or far clip did not reach the profile");
    PT_EXPECT(ctx, profile && profile->camera.focalLengthMm == authored.focalLengthMm && profile->camera.nearClip == authored.nearClip,
              "the scene camera changed the profile's lens");
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
        "\"color\":[1,1,1],\"luminance\":5.0,\"twoSided\":false}]";

    const std::vector<Case> cases = {
        {"not JSON at all", "engine_io_scene_notjson.json", "{ this is not json"},
        {"empty file", "engine_io_scene_empty.json", ""},
        {"JSON array where an object is required", "engine_io_scene_array.json", "[1, 2, 3]"},
        // Every surface key names the model's nodes, so a materialPath with no model has nothing to shade.
        {"a materialPath without a model", "engine_io_scene_nomodel.json",
         "{\"environment\":{\"hdriPath\":\"x.exr\"},\"materialPath\":\"materials/clay.json\"}"},
        // A zero scale has no inverse placement; a misspelt key would load as its default, silently.
        {"a volume with a zero scale", "engine_io_scene_volumescale.json",
         scene(",\"volumes\":[{\"vdbPath\":\"a.vdb\",\"grid\":\"density\",\"position\":[0,0,0],\"rotation\":[0,0,0],"
               "\"scale\":0,\"materialPath\":\"materials/volume.json\"}]")},
        {"a volume with an unknown key", "engine_io_scene_volumekey.json",
         scene(",\"volumes\":[{\"vdbPath\":\"a.vdb\",\"gird\":\"density\",\"position\":[0,0,0],\"rotation\":[0,0,0],"
               "\"materialPath\":\"materials/volume.json\"}]")},
        {"a camera without its far clip", "engine_io_scene_camera.json", scene(",\"camera\":{\"position\":[0,0,9],\"rotation\":[0,0,0]}")},
        // Negative radiance is not a scene, and would propagate as negative energy through every estimator.
        {"negative light luminance", "engine_io_scene_negluminance.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"color\":[1,1,1],\"luminance\":-5.0,\"twoSided\":false}]")},
        // A zero extent is a quad with no area; a negative one would mirror it and silently flip the emitting face.
        {"quad light with a zero size", "engine_io_scene_zerosizelight.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,0],"
               "\"color\":[1,1,1],\"luminance\":5.0,\"twoSided\":false}]")},
        {"quad light with a negative size", "engine_io_scene_negsizelight.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[-1,1],"
               "\"color\":[1,1,1],\"luminance\":5.0,\"twoSided\":false}]")},
        // The retired corner + edges form has no position, so it fails rather than placing a light nobody authored.
        {"the retired quad light origin/edge0/edge1 keys", "engine_io_scene_cornerlight.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"origin\":[0,0,0],\"edge0\":[1,0,0],\"edge1\":[0,0,1],"
               "\"color\":[1,1,1],\"luminance\":5.0,\"twoSided\":false}]")},
        // Closed key set: a leftover edge0 beside a complete placement would otherwise be ignored silently.
        {"a stale quad light edge0 key", "engine_io_scene_staleedge.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"edge0\":[1,0,0],\"color\":[1,1,1],\"luminance\":5.0,\"twoSided\":false}]")},
        // Each node maps slot names to paths; a bare path string would leave the slot unstated.
        {"textures entry that is not a slot object", "engine_io_scene_textureflat.json",
         scene(",\"textures\":{\"sphere01\":\"textures/macbeth.exr\"}")},
        // A misspelt override would load the file under its own tag, the very error the override exists to correct.
        {"a texture object with an unknown key", "engine_io_scene_texturekey.json",
         scene(",\"textures\":{\"sphere01\":{\"base_color\":{\"path\":\"textures/macbeth.exr\",\"colourSpace\":\"ACEScg\"}}}")},
        // A channel names one of R, G, B: alpha is never read, and a typo must not fall back to R.
        {"a texture channel outside r, g, b", "engine_io_scene_texturechannel.json",
         scene(",\"textures\":{\"sphere01\":{\"specular_roughness\":{\"path\":\"orm.exr\",\"channel\":\"a\"}}}")},
        {"a bump without its height", "engine_io_scene_bumpheight.json",
         scene(",\"textures\":{\"sphere01\":{\"geometry_normal\":{\"bump\":{\"path\":\"height.exr\"}}}}")},
        // Retired keys, well-formed otherwise: an ignored textureOverrides would render the scene untextured with no diagnostic.
        {"the retired textureOverrides key", "engine_io_scene_textureoverrides.json",
         scene(",\"textureOverrides\":{\"sphere01\":{\"base_color\":\"textures/macbeth.exr\"}}")},
        // A uniform bool, as lightEnabled: a number is a type error, not a truthy switch.
        {"model.vertexColor as a number", "engine_io_scene_vertexcolor.json",
         "{\"model\":{\"gltfPath\":\"geometry/cornell/cornell_v001.gltf\",\"vertexColor\":0,"
         "\"position\":[0,0,0],\"rotation\":[0,0,0]},\"environment\":{\"hdriPath\":\"textures/republiqueHDR_2k.exr\"},"
         "\"materialPath\":\"materials/clay.json\"}"},
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
               "\"color\":[1,-1,1],\"luminance\":5.0,\"twoSided\":false}]")},
        // A light's colour is authored once: both keys leave it ambiguous, neither leaves it unset.
        {"a light with both color and temperature", "engine_io_scene_colortemp.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"color\":[1,1,1],\"temperature\":2856,\"luminance\":5.0}]")},
        {"a light with neither color nor temperature", "engine_io_scene_nocolor.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],\"luminance\":5.0}]")},
        {"a zero light temperature", "engine_io_scene_zerotemp.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"temperature\":0,\"luminance\":5.0}]")},
        // Each shape's own extent is a measure, so zero or negative is no light; each key set is closed per type.
        {"a disk light with a zero radius", "engine_io_scene_diskradius.json",
         scene(",\"lights\":[{\"type\":\"disk\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"radius\":0,\"color\":[1,1,1],\"luminance\":5.0}]")},
        {"a sphere light with a negative radius", "engine_io_scene_sphereradius.json",
         scene(",\"lights\":[{\"type\":\"sphere\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"radius\":-1,\"color\":[1,1,1],\"luminance\":5.0}]")},
        // A sphere is closed: its inside is never lit, so a twoSided switch would be a key with no meaning.
        {"a sphere light with twoSided", "engine_io_scene_spheresided.json",
         scene(",\"lights\":[{\"type\":\"sphere\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"radius\":1,\"color\":[1,1,1],\"luminance\":5.0,"
               "\"twoSided\":true}]")},
        {"a mesh light with a zero scale", "engine_io_scene_meshscale.json",
         scene(",\"lights\":[{\"type\":\"mesh\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"gltfPath\":\"m.gltf\",\"scale\":0,"
               "\"color\":[1,1,1],\"luminance\":5.0}]")},
        {"a mesh light without its gltfPath", "engine_io_scene_meshpath.json",
         scene(",\"lights\":[{\"type\":\"mesh\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"color\":[1,1,1],\"luminance\":5.0}]")},
        {"a disk light with a quad's size key", "engine_io_scene_disksize.json",
         scene(",\"lights\":[{\"type\":\"disk\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"radius\":1,\"size\":[1,1],"
               "\"color\":[1,1,1],\"luminance\":5.0}]")},
        // Point, spot and distant lights have no surface to hit: an unknown type fails rather than loading as another.
        {"a point light", "engine_io_scene_pointlight.json",
         scene(",\"lights\":[{\"type\":\"point\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"color\":[1,1,1],\"luminance\":5.0}]")},
        // A black colour has no luminance to normalise by: it is no light, not a dark one.
        {"a black light colour", "engine_io_scene_blackcolor.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"color\":[0,0,0],\"luminance\":5.0}]")},
        // A light's output is authored once, as a luminance or a flux: both leave it ambiguous, neither leaves it unset.
        {"a light with both luminance and lumens", "engine_io_scene_lumlm.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"color\":[1,1,1],\"luminance\":5.0,\"lumens\":800}]")},
        {"a light with neither luminance nor lumens", "engine_io_scene_nolum.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],\"color\":[1,1,1]}]")},
        {"a zero light luminance", "engine_io_scene_zerolum.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"color\":[1,1,1],\"luminance\":0}]")},
        {"negative light lumens", "engine_io_scene_neglumens.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"color\":[1,1,1],\"lumens\":-800}]")},
        // The unitless multiplier photometric units replaced: refused by name rather than read as cd/m^2.
        {"the legacy intensity key", "engine_io_scene_intensity.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"color\":[1,1,1],\"luminance\":5.0,\"intensity\":5.0}]")},
        // A calibration target in lux: zero would black the sky, which lightEnabled false already says.
        {"a zero environment lux", "engine_io_scene_zerolux.json",
         "{\"environment\":{\"hdriPath\":\"textures/republiqueHDR_2k.exr\",\"lux\":0}}"},
        {"a negative environment lux", "engine_io_scene_neglux.json",
         "{\"environment\":{\"hdriPath\":\"textures/republiqueHDR_2k.exr\",\"lux\":-1}}"},
        {"a string environment lux", "engine_io_scene_strlux.json",
         "{\"environment\":{\"hdriPath\":\"textures/republiqueHDR_2k.exr\",\"lux\":\"x\"}}"},
        // The locus leaves AP1 below about 600 K: 500 K has negative green, which a clamp would hide as a hue shift.
        {"a light temperature outside the scene gamut", "engine_io_scene_gamuttemp.json",
         scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
               "\"temperature\":500,\"luminance\":5.0}]")},
    };

    // Anti-vacuity: the base the scene() rows are built from must itself LOAD, or they would prove nothing.
    const std::filesystem::path basePath = writeJson("engine_io_scene_base.json", scene(validLights));
    const bool baseLoads = pathtracer::config::loadSceneConfig(basePath.string()).has_value();
    std::filesystem::remove(basePath);

    // Anti-vacuity for the temperature rows: an in-gamut temperature loads, resolved to the blackbody colour.
    const std::filesystem::path tempPath = writeJson("engine_io_scene_temp.json",
        scene(",\"lights\":[{\"type\":\"quad\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"size\":[1,1],"
              "\"temperature\":2856,\"luminance\":5.0}]"));
    const std::optional<pathtracer::config::SceneConfig> tempScene = pathtracer::config::loadSceneConfig(tempPath.string());
    std::filesystem::remove(tempPath);

    ctx.plan(static_cast<int>(cases.size()) + 2);
    PT_EXPECT(ctx, baseLoads,
                  "the unmutated base scene must load, or every mutated row below passes vacuously");
    PT_EXPECT(ctx, tempScene && tempScene->lights.at(0).color == glm::vec3(*pathtracer::scene::cie::blackbodyToScene(2856.0)),
              "a 2856 K light did not load as the blackbody colour");
    for (const Case& testCase : cases) {
        const std::filesystem::path path = writeJson(testCase.file, testCase.text);
        const bool accepted = pathtracer::config::loadSceneConfig(path.string()).has_value();
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadSceneConfig accepted a scene with %s", testCase.name);
        PT_EXPECT(ctx, !accepted, detail);
        std::filesystem::remove(path);
    }
}

// Each light type reads into its own shape with the shared keys beside it; a mesh's scale is optional at 1, a twoSided at false.
PT_CHECK(scene_config_reads_every_light_shape, Fast, Exact) {
    const std::filesystem::path path = writeJson(
        "engine_io_scene_shapes.json",
        "{\"environment\":{\"hdriPath\":\"textures/republiqueHDR_2k.exr\",\"lux\":405.5},\"lights\":["
        "{\"type\":\"disk\",\"position\":[1,2,3],\"rotation\":[10,20,30],\"radius\":0.25,\"color\":[1,0.5,0.25],\"luminance\":4.0,\"twoSided\":true},"
        "{\"type\":\"sphere\",\"position\":[0,1,0],\"rotation\":[0,0,0],\"radius\":0.5,\"temperature\":2856,\"luminance\":2.0},"
        "{\"type\":\"mesh\",\"position\":[0,0,1],\"rotation\":[0,90,0],\"gltfPath\":\"mesh.gltf\","
        "\"scale\":0.3,\"color\":[1,1,1],\"lumens\":800.0},"
        "{\"type\":\"mesh\",\"position\":[0,0,0],\"rotation\":[0,0,0],\"gltfPath\":\"m.gltf\",\"color\":[1,1,1],\"luminance\":1.0}]}");
    const std::optional<pathtracer::config::SceneConfig> scene = pathtracer::config::loadSceneConfig(path.string());
    std::filesystem::remove(path);
    ctx.plan(5);
    PT_EXPECT(ctx, scene && scene->lights.size() == 4, "a scene of one disk, one sphere and two mesh lights did not load as four lights");
    if (!scene || scene->lights.size() != 4) {
        return;
    }
    const auto* disk = std::get_if<pathtracer::config::DiskShape>(&scene->lights[0].shape);
    // The authored colour at unit luminance, by the loader's own product, so luminance alone carries the cd/m^2.
    const glm::vec3 authored(1.0F, 0.5F, 0.25F);
    const glm::vec3 unitColor = authored / glm::dot(pathtracer::scene::cie::sceneLuminanceWeights(), authored);
    const auto* luminance = std::get_if<pathtracer::config::Luminance>(&scene->lights[0].emission);
    PT_EXPECT(ctx, disk && disk->radius == 0.25F && scene->lights[0].position == glm::vec3(1.0F, 2.0F, 3.0F) &&
                       scene->lights[0].rotation == glm::vec3(10.0F, 20.0F, 30.0F) && scene->lights[0].twoSided &&
                       scene->lights[0].color == unitColor && luminance && luminance->candelasPerSquareMetre == 4.0F,
              "the disk's radius, placement, side, unit-luminance colour or luminance did not read as authored");
    const auto* sphere = std::get_if<pathtracer::config::SphereShape>(&scene->lights[1].shape);
    PT_EXPECT(ctx, sphere && sphere->radius == 0.5F && !scene->lights[1].twoSided &&
                       scene->lights[1].color == glm::vec3(*pathtracer::scene::cie::blackbodyToScene(2856.0)),
              "the sphere's radius or temperature did not read as authored");
    const auto* scaled = std::get_if<pathtracer::config::MeshShape>(&scene->lights[2].shape);
    const auto* unscaled = std::get_if<pathtracer::config::MeshShape>(&scene->lights[3].shape);
    PT_EXPECT(ctx, scaled && scaled->gltfPath == "mesh.gltf" && scaled->scale == 0.3F && unscaled &&
                       unscaled->scale == 1.0F && !scene->lights[3].twoSided,
              "a mesh light's path or scale did not read as authored, or an absent scale is not 1");
    const auto* flux = std::get_if<pathtracer::config::LuminousFlux>(&scene->lights[2].emission);
    PT_EXPECT(ctx, flux && flux->lumens == 800.0F && scene->environment.illuminanceLux == 405.5F,
              "a light's lumens or the environment's illuminance did not read as authored");
}

// Anti-vacuity for the rejection rows below: every material the repo ships must still load once loadMaterialConfig validates.
PT_CHECK(material_config_accepts_the_shipped_materials, Fast, Exact) {
    const std::vector<const char*> shipped = {"checker.json", "chrome.json", "clay.json", "emission.json", "glass.json", "principled.json"};
    ctx.plan(static_cast<int>(shipped.size()));
    for (const char* name : shipped) {
        const std::filesystem::path path = std::filesystem::path(ASSET_ROOT_DIR) / "materials" / name;
        char detail[256];
        std::snprintf(detail, sizeof(detail), "loadMaterialConfig rejected the shipped material %s", name);
        PT_EXPECT(ctx, pathtracer::config::loadMaterialConfig(path.string()).has_value(), detail);
    }
}

// geometry_thin_walled reaches the material beside a thin wall's subsurface and opacity.
PT_CHECK(material_config_thin_walled, Fast, Exact) {
    const std::filesystem::path path = writeJson("engine_io_material_thin.json",
                                                 "{\"geometry_thin_walled\":true,\"subsurface_weight\":0.5,\"geometry_opacity\":0.25}");
    const std::optional<pathtracer::config::MaterialConfig> material = pathtracer::config::loadMaterialConfig(path.string());
    std::filesystem::remove(path);
    ctx.plan(1);
    PT_EXPECT(ctx, material && material->geometryThinWalled && material->subsurfaceWeight == 0.5F && material->geometryOpacity == 0.25F,
              "a thin-walled material with subsurface and opacity did not load as written");
}

// A bulk material's interior inputs load as written: subsurface and transmission_scatter are media, a checker on a weight among them.
PT_CHECK(material_config_bulk_interior, Fast, Exact) {
    const std::filesystem::path path =
        writeJson("engine_io_material_bulk.json",
                  "{\"subsurface_weight\":{\"randomChecker\":{\"cellSize\":0.1}},\"subsurface_radius\":0.05,"
                  "\"subsurface_radius_scale\":[1,0.5,0],\"transmission_weight\":0.5,\"transmission_depth\":0.2,"
                  "\"transmission_scatter\":[0.1,0.2,0.3],\"transmission_scatter_anisotropy\":0.4}");
    const std::optional<pathtracer::config::MaterialConfig> material = pathtracer::config::loadMaterialConfig(path.string());
    std::filesystem::remove(path);
    ctx.plan(2);
    PT_EXPECT(ctx, material && !material->geometryThinWalled && material->subsurfaceRadius == 0.05F &&
                       material->subsurfaceRadiusScale == glm::vec3(1.0F, 0.5F, 0.0F) && material->transmissionScatter == glm::vec3(0.1F, 0.2F, 0.3F) &&
                       material->transmissionScatterAnisotropy == 0.4F,
              "a bulk material's interior inputs did not load as written");
    PT_EXPECT(ctx, material && material->randomCheckers.contains("subsurface_weight"), "a checker on a bulk subsurface_weight did not load");
}

// A {"randomChecker": {...}} object replaces one input's constant: it reaches materialOf as that input's pattern, seed 0 when omitted.
PT_CHECK(material_config_reads_random_checkers, Fast, Exact) {
    const std::filesystem::path path = writeJson("engine_io_material_checker.json",
                                                 "{\"base_color\":{\"randomChecker\":{\"cellSize\":0.01,\"seed\":7}},"
                                                 "\"specular_roughness\":{\"randomChecker\":{\"cellSize\":0.3}},\"base_weight\":0.5}");
    const std::optional<pathtracer::config::MaterialConfig> material = pathtracer::config::loadMaterialConfig(path.string());
    std::filesystem::remove(path);
    const pathtracer::scene::Material bound =
        material ? pathtracer::scene::materialOf(*material, material->randomCheckers) : pathtracer::scene::Material{};
    const auto* colour = std::get_if<pathtracer::scene::RandomChecker>(&bound.baseColor);
    const auto* roughness = std::get_if<pathtracer::scene::RandomChecker>(&bound.specularRoughness);
    const auto* weight = std::get_if<float>(&bound.baseWeight);
    ctx.plan(3);
    PT_EXPECT(ctx, colour && colour->cellSize == 0.01F && colour->seed == 7U, "a colour input's random checker did not bind as written");
    PT_EXPECT(ctx, roughness && roughness->cellSize == 0.3F && roughness->seed == 0U, "a scalar input's checker did not default seed 0");
    PT_EXPECT(ctx, weight && *weight == 0.5F, "a constant beside the checkers did not load as written");
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
        return std::string("{\"base_color\":[1,1,1],\"specular_roughness\":0.5") + overrides + "}";
    };

    const std::vector<Case> cases = {
        // Closed key set: a pre-OpenPBR file would otherwise load every input at its default, silently.
        {"the retired diffuseColour key", "engine_io_material_legacy.json", material(",\"diffuseColour\":[1,1,1]")},
        {"the retired shadingModel key", "engine_io_material_shadingmodel.json", material(",\"shadingModel\":\"constant\"")},
        // eonUniformMixWeight raises r to the 0.1 power, which is NaN for r < 0, and the NaN reaches the pixel through sampleEon.
        {"negative base_diffuse_roughness", "engine_io_material_negdiffrough.json", material(",\"base_diffuse_roughness\":-0.2")},
        // Outside [0,1] the EON quartic albedo fit is extrapolated, where 1 - E can go negative and the BRDF with it.
        {"base_diffuse_roughness above 1", "engine_io_material_diffroughhigh.json", material(",\"base_diffuse_roughness\":1.5")},
        // A transmittance above 1 amplifies along every path through the medium without bound.
        {"transmission_color above 1", "engine_io_material_transcolour.json",
         material(",\"transmission_color\":[1.4,1.0,1.0],\"transmission_depth\":0.4")},
        // Mix weights: outside [0,1] the bases mix with a negative weight and a lobe's selection mass goes negative.
        {"base_metalness above 1", "engine_io_material_metallic.json", material(",\"base_metalness\":1.5")},
        {"negative transmission_weight", "engine_io_material_negtrans.json", material(",\"transmission_weight\":-0.5")},
        {"specular_roughness above 1", "engine_io_material_rough.json", material(",\"specular_roughness\":1.5")},
        // The Fresnel reflectance divides by ior + 1, and every dielectric lobe by the index ratio.
        {"non-positive specular_ior", "engine_io_material_zeroior.json", material(",\"specular_ior\":0.0")},
        {"negative specular_weight", "engine_io_material_negweight.json", material(",\"specular_weight\":-0.5")},
        // base_color is a reflectance: above 1 the diffuse, and EON's multiple-scatter term with it, returns more than it receives.
        {"base_color above 1", "engine_io_material_colour.json", material(",\"base_color\":[1,2,1]")},
        {"negative transmission_depth", "engine_io_material_negdepth.json", material(",\"transmission_depth\":-1.0")},
        {"negative transmission_dispersion_abbe_number", "engine_io_material_negabbe.json",
         material(",\"transmission_dispersion_abbe_number\":-1.0")},
        {"transmission_scatter_anisotropy below -1", "engine_io_material_anisotropy.json",
         material(",\"transmission_scatter_anisotropy\":-1.5")},
        {"negative emission_luminance", "engine_io_material_negemission.json", material(",\"emission_luminance\":-1.0")},
        // A mean free path and a scattering coefficient are lengths' reciprocals: negative, the medium amplifies.
        {"negative subsurface_radius", "engine_io_material_negradius.json", material(",\"subsurface_weight\":1,\"subsurface_radius\":-0.1")},
        {"negative transmission_scatter", "engine_io_material_negscatter.json", material(",\"transmission_scatter\":[-0.1,0,0]")},
        // A presence probability: past [0, 1] the mix with the ambient takes a negative weight.
        {"geometry_opacity above 1", "engine_io_material_opacity.json", material(",\"geometry_opacity\":1.5")},
        // A uniform bool: a number is a type error, not a truthy switch.
        {"geometry_thin_walled as a number", "engine_io_material_thinnumber.json", material(",\"geometry_thin_walled\":1")},
        // A random checker's 0 is a constant 0 at half the cells: refused where the constant would be, as for an ior.
        {"a random checker on specular_ior", "engine_io_material_checkerior.json",
         material(",\"specular_ior\":{\"randomChecker\":{\"cellSize\":0.1}}")},
        // A zero cell size divides by nothing; a negative one has no meaning as a width.
        {"zero cellSize", "engine_io_material_checkerzero.json", material(",\"base_weight\":{\"randomChecker\":{\"cellSize\":0}}")},
        {"negative cellSize", "engine_io_material_checkerneg.json",
         material(",\"base_weight\":{\"randomChecker\":{\"cellSize\":-0.01}}")},
        {"a missing cellSize", "engine_io_material_checkermissing.json", material(",\"base_weight\":{\"randomChecker\":{\"seed\":1}}")},
        {"a negative seed", "engine_io_material_checkerseed.json",
         material(",\"base_weight\":{\"randomChecker\":{\"cellSize\":0.1,\"seed\":-1}}")},
        // Closed key sets: a misspelt parameter would load at its default, a stray sibling would be ignored.
        {"a misspelt randomChecker key", "engine_io_material_checkerkey.json",
         material(",\"base_weight\":{\"randomChecker\":{\"cellsize\":0.1}}")},
        {"a key beside randomChecker", "engine_io_material_checkersibling.json",
         material(",\"base_weight\":{\"randomChecker\":{\"cellSize\":0.1},\"seed\":1}")},
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
        // Retired: textures keep their own format in the cache, so any value is refused rather than silently ignored.
        {"the retired textureBitDepth", "textureBitDepth", 16, false},
        {"vsync false", "vsync", false, true},
        {"vsync true", "vsync", true, true},
        {"vsync 1", "vsync", 1, false},
        {"vsync \"true\"", "vsync", "true", false},
        {"vsync missing", "vsync", nullptr, false},
    };
    for (const nlohmann::json& bad : {nlohmann::json(8), nlohmann::json(24), nlohmann::json(16.5), nlohmann::json("16"),
                                      nlohmann::json(nullptr)}) {
        cases.push_back({std::string("displayBitDepth ") + (bad.is_null() ? "missing" : bad.dump()), "displayBitDepth", bad, false});
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
            const std::optional<ScalarType> varied =
                testCase.value.is_number_integer() ? pathtracer::gfx::scalarTypeFromBitDepth(testCase.value.get<int>())
                                                   : std::nullopt;
            const ScalarType display = isDisplay ? *varied : shippedConfig->render.displayFormat;
            const bool vsync = isDisplay ? shippedConfig->render.vsync : testCase.value.get<bool>();
            std::snprintf(detail, sizeof(detail), "loadProfileConfig rejected or mis-mapped %s", testCase.name.c_str());
            PT_EXPECT(ctx, loaded.has_value() && loaded->render.displayFormat == display && loaded->render.vsync == vsync, detail);
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

// camera.pixelLattice names a lattice, and a hexagonal one refuses a defaultAOV with no hexagonal version: denoised.
PT_CHECK(profile_config_camera_pixel_lattice, Fast, Exact) {
    using pathtracer::gfx::PixelLattice;
    struct Case {
        std::string name;
        nlohmann::json lattice;
        pathtracer::debug::AovId defaultAov;
        std::optional<PixelLattice> expected;  // nullopt where the load must be refused
    };
    const std::vector<Case> cases = {
        {"square", "square", pathtracer::debug::AovId::Sobel, PixelLattice::Square},
        {"hexagonal", "hexagonal", pathtracer::debug::AovId::Beauty, PixelLattice::Hexagonal},
        {"hexagonal opening on a per-texel filter", "hexagonal", pathtracer::debug::AovId::Luminance, PixelLattice::Hexagonal},
        {"hexagonal opening on sobel", "hexagonal", pathtracer::debug::AovId::Sobel, PixelLattice::Hexagonal},
        {"hexagonal opening on gabor", "hexagonal", pathtracer::debug::AovId::Gabor, PixelLattice::Hexagonal},
        {"hexagonal opening on denoised", "hexagonal", pathtracer::debug::AovId::Denoised, std::nullopt},
        {"an unknown lattice name", "triangular", pathtracer::debug::AovId::Beauty, std::nullopt},
        {"no lattice at all", nlohmann::json(nullptr), pathtracer::debug::AovId::Beauty, std::nullopt},
    };
    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()));
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        if (testCase.lattice.is_null()) {
            edited["camera"].erase("pixelLattice");
        } else {
            edited["camera"]["pixelLattice"] = testCase.lattice;
        }
        edited["render"]["defaultAOV"] = static_cast<int>(testCase.defaultAov);
        const std::filesystem::path path = writeJson("engine_io_profile_lattice.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        const std::optional<PixelLattice> lattice = loaded ? std::optional(loaded->camera.pixelLattice) : std::nullopt;
        PT_EXPECT(ctx, lattice == testCase.expected,
                  "loadProfileConfig " + std::string(testCase.expected ? "mis-read or rejected " : "accepted ") + testCase.name);
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

// render.pixelProbe names one of three readouts; any other spelling, type or an absent key is refused rather than defaulted.
PT_CHECK(profile_config_reads_the_pixel_probe_readout, Fast, Exact) {
    using pathtracer::debug::PixelProbeReadout;
    const std::vector<std::pair<nlohmann::json, std::optional<PixelProbeReadout>>> cases = {
        {"colour", PixelProbeReadout::Colour}, {"photometric", PixelProbeReadout::Photometric}, {"none", PixelProbeReadout::None},
        {"color", std::nullopt},               {1, std::nullopt},                                 {nullptr, std::nullopt},
    };
    std::ifstream shippedFile(std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json");
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    ctx.plan(static_cast<int>(cases.size()));
    for (const auto& [value, expected] : cases) {
        nlohmann::json edited = shipped;
        if (value.is_null()) {
            edited["render"].erase("pixelProbe");
        } else {
            edited["render"]["pixelProbe"] = value;
        }
        const std::filesystem::path path = writeJson("engine_io_profile_probe.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        PT_EXPECT(ctx, expected ? loaded && loaded->render.pixelProbe == *expected : !loaded,
                  "render.pixelProbe " + (value.is_null() ? std::string("missing") : value.dump()) + " did not read as expected");
    }
}

// render.bloomStrength is an energy fraction, so [0,1] closed: both ends and an interior value load, one step past either end does not.
PT_CHECK(profile_config_bloom_strength_is_in_range, Fast, Exact) {
    const std::vector<std::pair<nlohmann::json, bool>> cases = {
        {0.0, true}, {0.05, true}, {1.0, true}, {-0.01, false}, {1.01, false}, {nlohmann::json(nullptr), false},
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
            edited["render"].erase("bloomStrength");
        } else {
            edited["render"]["bloomStrength"] = value;
        }
        const std::filesystem::path path = writeJson("engine_io_profile_bloom.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadProfileConfig %s bloomStrength %s", accepted ? "rejected" : "accepted",
                      value.dump().c_str());
        PT_EXPECT(ctx, loaded.has_value() == accepted && (!accepted || loaded->render.bloomStrength == value.get<float>()),
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
        nlohmann::json patch;  // RFC 7396 merge patch on the shipped profile: null removes a key
        bool accepted;
    };
    const auto pathTracer = [](nlohmann::json fields) { return nlohmann::json{{"pathTracer", std::move(fields)}}; };
    // Off, so the floors below test the counts alone: one replicate is legal only without the stopping rule that needs two.
    const nlohmann::json uniform = {{"threshold", 0.0}};
    std::vector<Case> cases;
    // Lowest legal value accepted, then each way of going under it. maxBounces/maxSamples 0 are meaningful: direct-only and unbounded.
    const std::vector<std::pair<const char*, int>> floors = {
        {"samplesPerPixel", 1}, {"maxBounces", 0}, {"russianRouletteStartBounce", 0}, {"maxSamples", 0}, {"replicates", 1}};
    for (const auto& [key, floor] : floors) {
        cases.push_back({std::string(key) + " " + std::to_string(floor), pathTracer({{key, floor}, {"adaptive", uniform}}), true});
        cases.push_back({std::string(key) + " " + std::to_string(floor - 1), pathTracer({{key, floor - 1}, {"adaptive", uniform}}), false});
        cases.push_back({std::string(key) + " missing", pathTracer({{key, nullptr}}), false});
    }
    // The traced image renderScale multiplies and the primary ray's aspect, then the window's framebuffer.
    for (const char* key : {"width", "height", "displayWidth", "displayHeight"}) {
        cases.push_back({std::string("render.") + key + " 1", {{"render", {{key, 1}}}}, true});
        cases.push_back({std::string("render.") + key + " 0", {{"render", {{key, 0}}}}, false});
        cases.push_back({std::string("render.") + key + " -1", {{"render", {{key, -1}}}}, false});
        cases.push_back({std::string("render.") + key + " missing", {{"render", {{key, nullptr}}}}, false});
    }
    // Adaptive sampling: every key required; minSamples a power-of-two multiple of replicates; a stopping rule needs two replicates.
    for (const char* key : {"threshold", "confidence", "minSamples"}) {
        cases.push_back({std::string("adaptive.") + key + " missing", pathTracer({{"adaptive", {{key, nullptr}}}}), false});
    }
    cases.push_back({"adaptive missing", pathTracer({{"adaptive", nullptr}}), false});
    cases.push_back({"adaptive.threshold 0", pathTracer({{"adaptive", uniform}}), true});
    cases.push_back({"adaptive.threshold -0.01", pathTracer({{"adaptive", {{"threshold", -0.01}}}}), false});
    cases.push_back({"adaptive.confidence 0.5", pathTracer({{"adaptive", {{"confidence", 0.5}}}}), true});
    for (const double confidence : {0.0, 1.0}) {
        cases.push_back({"adaptive.confidence " + std::to_string(confidence), pathTracer({{"adaptive", {{"confidence", confidence}}}}), false});
    }
    const std::vector<std::pair<int, bool>> minSamples = {{4, true}, {16, true}, {12, false}, {2, false}, {0, false}};
    for (const auto& [count, accepted] : minSamples) {
        cases.push_back({"adaptive.minSamples " + std::to_string(count) + " at replicates 4",
                         pathTracer({{"replicates", 4}, {"adaptive", {{"minSamples", count}}}}), accepted});
    }
    cases.push_back({"adaptive on at replicates 1", pathTracer({{"replicates", 1}, {"adaptive", {{"threshold", 0.02}, {"minSamples", 1}}}}),
                     false});

    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    const std::optional<pathtracer::config::ProfileConfig> shippedConfig =
        pathtracer::config::loadProfileConfig(shippedPath.string());
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    // A capped accumulation must end on a complete round: replicates of maxSamples + 1 cannot divide a positive maxSamples.
    const int shippedMaxSamples = shipped["pathTracer"]["maxSamples"].get<int>();
    cases.push_back({"replicates not dividing maxSamples",
                     pathTracer({{"replicates", shippedMaxSamples + 1}, {"adaptive", {{"minSamples", shippedMaxSamples + 1}}}}),
                     shippedMaxSamples == 0});
    ctx.plan(static_cast<int>(cases.size()) + 3);
    PT_EXPECT(ctx, shippedConfig.has_value(), "the shipped profile does not load, so no row below means anything");
    if (!shippedConfig) {
        return;
    }
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        edited.merge_patch(testCase.patch);
        const std::filesystem::path path = writeJson("engine_io_profile_counts.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        char detail[224];
        std::snprintf(detail, sizeof(detail), "loadProfileConfig %s %s", testCase.accepted ? "rejected" : "accepted",
                      testCase.name.c_str());
        PT_EXPECT(ctx, loaded.has_value() == testCase.accepted, detail);
    }
    // The viewer's -max-samples lands after loading; the exported check is what refuses a cap that splits a round.
    pathtracer::config::PathTracerConfig overridden = shippedConfig->pathTracer;
    PT_EXPECT(ctx, pathtracer::config::validCounts(shippedConfig->render, overridden, "test"), "the shipped counts failed validCounts");
    overridden.maxSamples = overridden.replicates + 1;
    PT_EXPECT(ctx, overridden.replicates < 2 || !pathtracer::config::validCounts(shippedConfig->render, overridden, "test"),
              "validCounts accepted a maxSamples override that is not a multiple of replicates");
}

// The cache keeps each file's own format: a float32 texel past binary16's 65504, an unclipped sun, reads back exactly, not as Inf.
PT_CHECK(texture_keeps_float32_sources, Fast, Exact) {
    const std::vector<float> texels{1.0F, 70000.0F, 1.0e6F, 0.1F, 3.14159265F, 1.0e-20F};
    const std::filesystem::path path = scratchPath("engine_io_texture_float32.exr");
    const bool written = writeExrPlanes(path, 2, 1, {"R", "G", "B"}, texels);
    const std::shared_ptr<const pathtracer::gfx::ImageTexture> texture =
        written ? pathtracer::gfx::openTexture(path.string(), pathtracer::gfx::kRgbChannels, pathtracer::gfx::ImageRole::Data,
                                               pathtracer::gfx::TextureWrap::LatLong)
                : nullptr;
    ctx.plan(2);
    PT_EXPECT(ctx, texture && pathtracer::gfx::readTexels(*texture).value().texels == texels, "a float32 texture did not cache as an exact copy");
    // A point lookup at a texel centre is that texel: bilinear interpolates, so the 1e6 sun survives filtering too.
    PT_EXPECT(ctx, texture && pathtracer::gfx::sampleTexture(*texture, glm::vec2(0.75F, 0.5F)) == glm::vec3(texels[3], texels[4], texels[5]),
              "a point lookup at a texel centre is not that texel");
    std::filesystem::remove(path);
}

// openTexture keeps exactly the channels its slot reads: R alone for a scalar map, RGB otherwise, and a missing one is an error.
PT_CHECK(texture_channel_selection, Fast, Exact) {
    constexpr int kWidth = 3;
    constexpr int kHeight = 2;
    constexpr std::size_t kTexels = static_cast<std::size_t>(kWidth) * kHeight;
    const auto probe = [](std::size_t i) {
        const auto t = static_cast<float>(i);
        return glm::vec3(t, 10.0F + t, 20.0F + t);  // distinct per lane, so a stride or lane slip reads a wrong value
    };
    std::vector<float> rgba;
    std::vector<float> rgb;
    std::vector<float> red(kTexels);
    for (std::size_t i = 0; i < kTexels; ++i) {
        const glm::vec3 v = probe(i);
        rgba.insert(rgba.end(), {v.r, v.g, v.b, 1.0F});
        rgb.insert(rgb.end(), {v.r, v.g, v.b});
        red[i] = v.r;
    }
    const std::filesystem::path rgbaPath = scratchPath("engine_io_channels_rgba.exr");
    const bool rgbaWritten = writeExrPlanes(rgbaPath, kWidth, kHeight, {"R", "G", "B", "A"}, rgba);
    // An R-only file, as a DCC writes a scalar map: an RGB slot must refuse it rather than read zero-filled G and B.
    const std::filesystem::path redPath = scratchPath("engine_io_channels_r.exr");
    const bool redWritten = writeExrPlanes(redPath, kWidth, kHeight, {"R"}, red);
    const auto open = [](const std::filesystem::path& path, int channels) {
        return pathtracer::gfx::openTexture(path.string(), channels, pathtracer::gfx::ImageRole::Data, pathtracer::gfx::TextureWrap::Repeat);
    };
    ctx.plan(7);
    PT_EXPECT(ctx, rgbaWritten && redWritten, "could not write the channel probes");
    const std::shared_ptr<const pathtracer::gfx::ImageTexture> scalar = open(rgbaPath, pathtracer::gfx::kScalarChannels);
    PT_EXPECT(ctx, scalar && pathtracer::gfx::readTexels(*scalar).value().texels == red, "a scalar slot on an RGBA file did not read R alone");
    const std::shared_ptr<const pathtracer::gfx::ImageTexture> colour = open(rgbaPath, pathtracer::gfx::kRgbChannels);
    PT_EXPECT(ctx, colour && pathtracer::gfx::readTexels(*colour).value().texels == rgb, "an RGB slot on an RGBA file did not read exact RGB");
    PT_EXPECT(ctx, open(redPath, pathtracer::gfx::kScalarChannels) != nullptr, "an R-only file was refused by a scalar slot");
    std::cout << "  the stderr diagnostics below are expected: an R-only file has no G or B channel to read\n";
    PT_EXPECT(ctx, open(redPath, pathtracer::gfx::kRgbChannels) == nullptr, "an R-only file was accepted by an RGB slot");
    const std::optional<pathtracer::gfx::HdrImage> redImage = pathtracer::gfx::loadImage(redPath.string(), pathtracer::gfx::ImageRole::Data);
    PT_EXPECT(ctx, redImage && redImage->channels == pathtracer::gfx::kScalarChannels && redImage->texels == red,
              "loadImage did not read an R-only file as its one channel");
    const std::optional<pathtracer::gfx::HdrImage> rgbImage = pathtracer::gfx::loadImage(rgbaPath.string(), pathtracer::gfx::ImageRole::Data);
    PT_EXPECT(ctx, rgbImage && rgbImage->channels == pathtracer::gfx::kRgbChannels && rgbImage->texels == rgb,
              "loadImage of an RGBA file did not return its exact RGB at three floats per texel");
    std::filesystem::remove(rgbaPath);
    std::filesystem::remove(redPath);
}

// A colour texture converts once, before caching, exactly as loadImage does; a half source stays half: each texel that decode rounded.
PT_CHECK(texture_converts_colour_to_the_working_space, Fast, Exact) {
    const std::vector<float> stored{0.18F, 0.5F, 0.9F, 1.0F, 0.0F, 4.0F};
    const auto write = [&](const char* name, const char* colorSpace) {
        const std::filesystem::path path = scratchPath(name);
        const bool written = writeExrPlanes(path, 2, 1, {"R", "G", "B"}, stored, [&](OIIO::ImageSpec& spec) {
            spec.set_format(OIIO::TypeHalf);
            if (colorSpace != nullptr) {
                spec.attribute("oiio:ColorSpace", colorSpace);
            }
        });
        return written ? path : std::filesystem::path();
    };
    const std::filesystem::path tagged = write("engine_io_texture_rec709.exr", "Linear Rec.709 (sRGB)");
    const std::filesystem::path untagged = write("engine_io_texture_untagged.exr", nullptr);
    const auto open = [](const std::filesystem::path& path, pathtracer::gfx::ImageRole role, const std::optional<std::string>& colorSpace) {
        return pathtracer::gfx::openTexture(path.string(), pathtracer::gfx::kRgbChannels, role, pathtracer::gfx::TextureWrap::Repeat, colorSpace);
    };
    const auto texels = [](const std::shared_ptr<const pathtracer::gfx::ImageTexture>& texture) {
        return texture ? pathtracer::gfx::readTexels(*texture).value().texels : std::vector<float>();
    };
    const std::optional<pathtracer::gfx::HdrImage> reference = pathtracer::gfx::loadImage(tagged.string(), pathtracer::gfx::ImageRole::Colour);
    const std::optional<pathtracer::gfx::HdrImage> raw = pathtracer::gfx::loadImage(tagged.string(), pathtracer::gfx::ImageRole::Data);
    std::vector<float> expected;
    for (const float texel : reference ? reference->texels : std::vector<float>()) {
        expected.push_back(static_cast<float>(static_cast<pathtracer::gfx::Half>(texel)));
    }
    const std::filesystem::path png = scratchPath("engine_io_texture.png");
    const OIIO::ImageSpec pngSpec(2, 1, 3, OIIO::TypeUInt8);
    const std::vector<std::uint8_t> codes{0, 46, 128, 188, 255, 255};
    const std::unique_ptr<OIIO::ImageOutput> output = OIIO::ImageOutput::create("png");
    const bool pngWritten = output && output->open(png.string(), pngSpec) && output->write_image(OIIO::TypeUInt8, codes.data()) && output->close();
    ctx.plan(7);
    PT_EXPECT(ctx, !tagged.empty() && !untagged.empty() && reference && raw && reference->texels != raw->texels,
              "could not write the Rec.709 probes, or Rec.709 read as the working space");
    PT_EXPECT(ctx, texels(open(tagged, pathtracer::gfx::ImageRole::Colour, std::nullopt)) == expected,
              "a Rec.709 half texture is not loadImage's conversion rounded once to half");
    PT_EXPECT(ctx, texels(open(untagged, pathtracer::gfx::ImageRole::Colour, std::nullopt)) == expected,
              "an untagged half texture did not read as OpenEXR's BT.709 default");
    std::vector<float> asScene;
    for (const float texel : stored) {
        asScene.push_back(static_cast<float>(static_cast<pathtracer::gfx::Half>(texel)));
    }
    PT_EXPECT(ctx, texels(open(untagged, pathtracer::gfx::ImageRole::Colour, "ACEScg")) == asScene,
              "a scene colorSpace did not override an untagged file's BT.709 default");
    PT_EXPECT(ctx, raw && texels(open(tagged, pathtracer::gfx::ImageRole::Data, std::nullopt)) == raw->texels,
              "a data texture was converted rather than read raw");
    std::cout << "  the stderr diagnostics below are expected: a data texture refuses a colour space, and a PNG is not linear EXR\n";
    PT_EXPECT(ctx, !open(tagged, pathtracer::gfx::ImageRole::Data, "Linear Rec.709 (sRGB)"), "a data texture accepted a colour space");
    PT_EXPECT(ctx, pngWritten && !open(png, pathtracer::gfx::ImageRole::Colour, std::nullopt), "a PNG was opened as a texture");
    std::filesystem::remove(tagged);
    std::filesystem::remove(untagged);
    std::filesystem::remove(png);
}

// Minification through the MIP chain: a 16-texel footprint over a period-2 checker is its mean, where a point lookup aliases to 0 or 1.
PT_CHECK(texture_footprint_filters_through_mip_levels, Fast, Exact) {
    constexpr int kSize = 64;
    std::vector<float> checker(static_cast<std::size_t>(kSize) * kSize);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            checker[(static_cast<std::size_t>(y) * kSize) + static_cast<std::size_t>(x)] = static_cast<float>((x + y) % 2);
        }
    }
    const std::shared_ptr<const pathtracer::gfx::ImageTexture> texture =
        tools::fixtures::makeTexture(kSize, kSize, pathtracer::gfx::kScalarChannels, checker);
    // Every level from the first is exactly 0.5, so any blend of them is too, up to float rounding of its weights.
    const pathtracer::gfx::TextureFootprint footprint{glm::vec2(16.0F / kSize, 0.0F), glm::vec2(0.0F, 16.0F / kSize)};
    std::mt19937 rng(static_cast<std::mt19937::result_type>(ctx.seed()));
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    float worst = 0.0F;
    bool pointAliases = true;
    for (int i = 0; i < 256 && texture; ++i) {
        const glm::vec2 st(unit(rng), unit(rng));
        worst = std::max(worst, std::abs(pathtracer::gfx::sampleTexture(*texture, st, footprint).r - 0.5F));
        const glm::vec2 centre = (glm::floor(st * static_cast<float>(kSize)) + 0.5F) / static_cast<float>(kSize);
        const float point = pathtracer::gfx::sampleTexture(*texture, centre).r;
        pointAliases = pointAliases && (point == 0.0F || point == 1.0F);
    }
    ctx.plan(2);
    char detail[160];
    std::snprintf(detail, sizeof(detail), "a 16-texel footprint over a checker is %.3g from its mean 0.5", static_cast<double>(worst));
    PT_EXPECT(ctx, texture && worst <= 8.0F * std::numeric_limits<float>::epsilon(), detail);
    PT_EXPECT(ctx, texture && pointAliases, "a zero footprint at a texel centre did not read the texel itself");
}

// A NaN or Inf texel is refused at open, as loadImage refuses it: a shading input or CDF built on one is garbage.
PT_CHECK(texture_rejects_non_finite_texels, Fast, Exact) {
    const std::filesystem::path path = scratchPath("engine_io_texture_nan.exr");
    const bool written = writeExrPlanes(path, 1, 1, {"R", "G", "B"}, {0.5F, std::numeric_limits<float>::quiet_NaN(), 0.5F});
    std::cout << "  the stderr diagnostics below are expected: the texel is NaN\n";
    ctx.plan(2);
    PT_EXPECT(ctx, written && !pathtracer::gfx::openTexture(path.string(), pathtracer::gfx::kRgbChannels, pathtracer::gfx::ImageRole::Colour,
                                                            pathtracer::gfx::TextureWrap::Repeat),
              "a NaN texel was accepted by openTexture");
    PT_EXPECT(ctx, !pathtracer::gfx::loadImage(path.string(), pathtracer::gfx::ImageRole::Colour), "a NaN texel was accepted by loadImage");
    std::filesystem::remove(path);
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
                                  {"color", {1, 1, 1}}, {"luminance", 5.0}};
    const nlohmann::json scene = {
        {"model", {{"gltfPath", "geometry/cornell/cornell_v001.gltf"}, {"position", {0, 0, 0}}, {"rotation", {0, 0, 0}}}},
        {"environment", {{"hdriPath", "textures/republiqueHDR_2k.exr"}, {"lux", 405.6}}},
        {"materialPath", "materials/clay.json"},
        {"lights", {light}}};
    // Every OpenPBR input present at its default, so a pointer into it overwrites a number rather than creating a malformed sibling.
    const pathtracer::config::MaterialConfig kDefaultMaterial;
    nlohmann::json material = nlohmann::json::object();
    std::vector<std::string> materialFields;
    int colourInputs = 0;  // each colour input overflows a different channel in turn, so all three are reached
    pathtracer::scene::forEachInput(
        [&](const pathtracer::scene::InputSpec& spec, const auto& value) {
            const std::string key(spec.name);
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>, float>) {
                material[key] = value;
                materialFields.push_back("/" + key);
            } else {
                material[key] = {value.x, value.y, value.z};
                materialFields.push_back("/" + key + "/" + std::to_string(colourInputs++ % 3));
            }
        },
        kDefaultMaterial);
    const nlohmann::json texturedScene = {
        {"model", {{"gltfPath", "geometry/cornell/cornell_v001.gltf"}, {"position", {0, 0, 0}}, {"rotation", {0, 0, 0}}}},
        {"environment", {{"hdriPath", "textures/republiqueHDR_2k.exr"}}},
        {"materialPath", "materials/clay.json"},
        {"textures", {{"node", {{"geometry_normal", {{"bump", {{"path", "height.exr"}, {"height", 0.005}}}}}}}}}};
    std::ifstream shippedProfile(std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json");
    const std::vector<Loader> loaders = {
        {"loadSceneConfig", scene,
         [](const std::string& path) { return pathtracer::config::loadSceneConfig(path).has_value(); },
         {"/model/position/0", "/model/rotation/1", "/lights/0/position/2", "/lights/0/rotation/0", "/lights/0/size/1",
          "/lights/0/color/0", "/lights/0/luminance", "/environment/lux"}},
        {"loadSceneConfig bump", texturedScene,
         [](const std::string& path) { return pathtracer::config::loadSceneConfig(path).has_value(); },
         {"/textures/node/geometry_normal/bump/height"}},
        {"loadMaterialConfig", material,
         [](const std::string& path) { return pathtracer::config::loadMaterialConfig(path).has_value(); }, {}},
        {"loadProfileConfig", nlohmann::json::parse(shippedProfile),
         [](const std::string& path) { return pathtracer::config::loadProfileConfig(path).has_value(); },
         {"/camera/position/0", "/camera/rotation/0", "/camera/rotation/2", "/camera/focalLengthMm", "/camera/nearClip",
          "/camera/farClip", "/camera/aperture", "/camera/shutterSeconds", "/camera/iso", "/camera/lens/radialCoefficients/3",
          "/camera/lens/maxFieldOfViewDegrees", "/controls/flySpeedMetersPerSecond", "/controls/orbitSensitivityDegPerPixel",
          "/render/renderScale", "/render/interactiveRenderScale", "/render/bloomStrength", "/pathTracer/aoMaxDistance",
          "/pathTracer/lookaheadDistance"}},
        {"loadFilmBackPresets", nlohmann::json::array({{{"name", "gate"}, {"widthMm", 36.0}, {"heightMm", 24.0}}}),
         [](const std::string& path) { return pathtracer::config::loadFilmBackPresets(path).has_value(); },
         {"/0/widthMm", "/0/heightMm"}},
    };

    std::vector<Loader> allLoaders = loaders;
    for (Loader& loader : allLoaders) {
        if (loader.floatFields.empty()) {
            for (const std::string& field : materialFields) {
                loader.floatFields.push_back(field.c_str());
            }
        }
    }
    int planned = 0;
    for (const Loader& loader : allLoaders) {
        planned += 1 + (2 * static_cast<int>(loader.floatFields.size()));
    }
    ctx.plan(planned + 3);
    for (const Loader& loader : allLoaders) {
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

    // The exact boundary, on luminance, which has no upper bound of its own: below FLT_MAX + ulp/2 rounds to FLT_MAX, the tie to inf.
    const double floatMax = std::numeric_limits<float>::max();
    const double halfUlp = 0.5 * (floatMax - std::nextafter(std::numeric_limits<float>::max(), 0.0F));
    const auto luminanceLoaded = [&](double luminance) {
        nlohmann::json edited = scene;
        edited["lights"][0]["luminance"] = luminance;
        const std::filesystem::path path = writeJson("engine_io_float_boundary.json", edited.dump());
        const std::optional<pathtracer::config::SceneConfig> loaded = pathtracer::config::loadSceneConfig(path.string());
        std::filesystem::remove(path);
        return loaded ? std::optional(std::get<pathtracer::config::Luminance>(loaded->lights[0].emission).candelasPerSquareMetre)
                      : std::nullopt;
    };
    PT_EXPECT(ctx, luminanceLoaded(floatMax) == std::numeric_limits<float>::max(), "FLT_MAX itself did not load exactly");
    PT_EXPECT(ctx, luminanceLoaded(std::nextafter(floatMax + halfUlp, 0.0)) == std::numeric_limits<float>::max(),
              "the largest double rounding to FLT_MAX did not load as FLT_MAX");
    PT_EXPECT(ctx, !luminanceLoaded(floatMax + halfUlp).has_value(), "FLT_MAX + ulp/2, which rounds to inf, loaded");
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
    std::vector<glm::vec3> colours = {};  // COLOR_0
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
        if (!prim.colours.empty()) {
            primitive["attributes"]["COLOR_0"] = addAccessor(prim.colours, kFloat, "VEC3");
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

// glTF COLOR_0 is linear BT.709 (glTF 2.0 3.7.2.1), converted by OCIO to the scene space; absent, or vertexColor false, it is white.
PT_CHECK(gltf_vertex_colours_convert_to_the_scene_space, Fast, Exact) {
    GltfPrimitive quad = makeQuad();
    quad.colours = {{0.8F, 0.1F, 0.05F}, {0.1F, 0.6F, 0.2F}, {0.2F, 0.3F, 0.9F}, {1.0F, 1.0F, 1.0F}};
    const std::string tintedPath = writeGltf("engine_io_validate_tint", {quad}, {glm::vec3(1.0F)}).string();
    const std::optional<pathtracer::scene::LoadedModel> tinted = pathtracer::scene::loadGltf(tintedPath);
    const std::optional<pathtracer::scene::LoadedModel> untinted = pathtracer::scene::loadGltf(tintedPath, glm::mat4(1.0F), false);
    const std::optional<pathtracer::scene::LoadedModel> plain =
        pathtracer::scene::loadGltf(writeGltf("engine_io_validate_plain", {makeQuad()}, {glm::vec3(1.0F)}).string());
    ctx.plan(3);
    // Triangle 0 of the quad is vertices 0, 1, 2 and its second is 0, 2, 3, so v0 of triangle 1 is vertex 0 again.
    std::vector<float> expected = {0.8F, 0.1F, 0.05F, 0.1F, 0.6F, 0.2F, 0.2F, 0.3F, 0.9F};
    pathtracer::gfx::convertToSceneSpace(expected, pathtracer::gfx::kOcioLinearRec709);
    const bool converted = tinted && glm::vec3(expected[0], expected[1], expected[2]) == tinted->shadingTriangles[0].v0.colour &&
                           glm::vec3(expected[3], expected[4], expected[5]) == tinted->shadingTriangles[0].v1.colour &&
                           glm::vec3(expected[6], expected[7], expected[8]) == tinted->shadingTriangles[0].v2.colour;
    PT_EXPECT(ctx, converted, "COLOR_0 did not arrive as OCIO's linear Rec.709 to the scene space");
    PT_EXPECT(ctx, plain && plain->shadingTriangles[0].v0.colour == glm::vec3(1.0F), "a primitive with no COLOR_0 is not white");
    const bool white = untinted && std::ranges::all_of(untinted->shadingTriangles, [](const pathtracer::scene::ShadingTriangle& t) {
        return t.v0.colour == glm::vec3(1.0F) && t.v1.colour == glm::vec3(1.0F) && t.v2.colour == glm::vec3(1.0F);
    });
    PT_EXPECT(ctx, white, "vertexColor false still read COLOR_0");
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

// glTF 2.0 3.7.2.1: no NORMAL is flat normals per face, even across a welded vertex; no TEXCOORD_0 is uv 0, tangent about n.
PT_CHECK(gltf_missing_normals_and_uvs_follow_the_spec, Fast, Exact) {
    // Two faces folded 90 degrees about the shared x axis, welded: one in z=0 facing +z, one in y=0 facing +y.
    GltfPrimitive folded;
    folded.positions = {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}};
    folded.indices = {0, 1, 2, 0, 3, 1};
    GltfPrimitive normalsOnly = folded;
    normalsOnly.normals = std::vector<glm::vec3>(4, glm::normalize(glm::vec3(0.0F, -1.0F, 1.0F)));
    // Identity and a mirrored node: flat normals are taken in object space and must survive the re-wind like authored ones.
    const std::vector<glm::vec3> scales = {{1.0F, 1.0F, 1.0F}, {-1.0F, 1.0F, 1.0F}};
    const std::optional<pathtracer::scene::LoadedModel> flat =
        pathtracer::scene::loadGltf(writeGltf("engine_io_validate_positions_only", {folded}, scales).string());
    const std::optional<pathtracer::scene::LoadedModel> authored =
        pathtracer::scene::loadGltf(writeGltf("engine_io_validate_normals_only", {normalsOnly}, {glm::vec3(1.0F)}).string());
    ctx.plan(4);
    PT_EXPECT(ctx, flat && flat->shadingTriangles.size() == 4 && authored && authored->shadingTriangles.size() == 2,
              "a primitive without NORMAL or TEXCOORD_0 did not load");
    if (!flat || !authored) {
        return;
    }
    const auto orthonormal = [](const pathtracer::scene::ShadingVertex& v) {
        return std::abs(glm::dot(v.normal, glm::vec3(v.tangent))) < 1e-6F && std::abs(glm::length(glm::vec3(v.tangent)) - 1.0F) < 1e-6F &&
               std::abs(v.tangent.w) == 1.0F;
    };
    bool faceNormals = true;
    bool basis = true;
    bool zeroUv = true;
    for (const pathtracer::scene::ShadingTriangle& tri : flat->shadingTriangles) {
        const glm::vec3 face = pathtracer::scene::geometricNormalOf(tri);
        for (const pathtracer::scene::ShadingVertex* v : {&tri.v0, &tri.v1, &tri.v2}) {
            faceNormals = faceNormals && glm::length(v->normal - face) < 1e-6F;
            basis = basis && orthonormal(*v);
            zeroUv = zeroUv && v->uv == glm::vec2(0.0F);
        }
    }
    PT_EXPECT(ctx, faceNormals, "a missing NORMAL did not give each corner its own face's geometric normal");
    PT_EXPECT(ctx, basis && zeroUv, "a missing TEXCOORD_0 did not give uv 0 and a unit tangent orthogonal to the normal");
    const bool keptNormals = std::ranges::all_of(authored->shadingTriangles, [&](const pathtracer::scene::ShadingTriangle& tri) {
        return glm::length(tri.v0.normal - normalsOnly.normals[0]) < 1e-6F && orthonormal(tri.v0) && orthonormal(tri.v1) && orthonormal(tri.v2);
    });
    PT_EXPECT(ctx, keptNormals, "an authored NORMAL without TEXCOORD_0 was replaced, or its tangent left off the normal's plane");
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

// The denoiser block: each quality name maps, anything else and any missing key is refused, and a disabled denoiser takes Denoised along.
PT_CHECK(profile_config_denoiser, Fast, Exact) {
    struct Case {
        std::string name;
        nlohmann::json denoiser;
        int defaultAov;
        bool accepted;
    };
    const std::filesystem::path shippedPath = std::filesystem::path(ASSET_ROOT_DIR) / "config" / "profile.json";
    std::ifstream shippedFile(shippedPath);
    const nlohmann::json shipped = nlohmann::json::parse(shippedFile);
    const nlohmann::json block = shipped["denoiser"];
    const auto with = [&](const char* key, const nlohmann::json& value) {
        nlohmann::json edited = block;
        if (value.is_null()) {
            edited.erase(key);
        } else {
            edited[key] = value;
        }
        return edited;
    };
    constexpr int kBeauty = static_cast<int>(pathtracer::debug::AovId::Beauty);
    constexpr int kDenoised = static_cast<int>(pathtracer::debug::AovId::Denoised);
    std::vector<Case> cases;
    for (const char* quality : pathtracer::debug::kDenoiseQualityNames) {
        cases.push_back({std::string("quality ") + quality, with("quality", quality), kBeauty, true});
    }
    cases.push_back({"quality ultra", with("quality", "ultra"), kBeauty, false});
    cases.push_back({"quality High, case differs", with("quality", "High"), kBeauty, false});
    cases.push_back({"disabled, Beauty first", with("enabled", false), kBeauty, true});
    cases.push_back({"disabled, Denoised first", with("enabled", false), kDenoised, false});
    cases.push_back({"enabled, Denoised first", with("enabled", true), kDenoised, true});
    cases.push_back({"prefilterAuxiliary false", with("prefilterAuxiliary", false), kBeauty, true});
    cases.push_back({"enabled as a string", with("enabled", "true"), kBeauty, false});
    for (const char* key : {"enabled", "quality", "prefilterAuxiliary"}) {
        cases.push_back({std::string(key) + " missing", with(key, nullptr), kBeauty, false});
    }
    cases.push_back({"block missing", nullptr, kBeauty, false});
    ctx.plan(static_cast<int>(cases.size()));
    for (const Case& testCase : cases) {
        nlohmann::json edited = shipped;
        if (testCase.denoiser.is_null()) {
            edited.erase("denoiser");
        } else {
            edited["denoiser"] = testCase.denoiser;
        }
        edited["render"]["defaultAOV"] = testCase.defaultAov;
        const std::filesystem::path path = writeJson("engine_io_profile_denoiser.json", edited.dump());
        const std::optional<pathtracer::config::ProfileConfig> loaded = pathtracer::config::loadProfileConfig(path.string());
        std::filesystem::remove(path);
        if (!testCase.accepted) {
            PT_EXPECT(ctx, !loaded.has_value(), "loadProfileConfig accepted " + testCase.name);
            continue;
        }
        const std::string quality = testCase.denoiser["quality"].get<std::string>();
        PT_EXPECT(ctx,
                  loaded.has_value() && loaded->denoiser.enabled == testCase.denoiser["enabled"].get<bool>() &&
                      pathtracer::debug::kDenoiseQualityNames[static_cast<int>(loaded->denoiser.options.quality)] == quality &&
                      loaded->denoiser.options.prefilterAuxiliary == testCase.denoiser["prefilterAuxiliary"].get<bool>(),
                  "loadProfileConfig rejected or mis-mapped " + testCase.name);
    }
}

PT_CHECK_MAIN("io")
