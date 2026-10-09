#include "pathtracer/gfx/hdr_image.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string_view>
#include <system_error>
#include <utility>

#include <OpenImageIO/color.h>
#include <OpenImageIO/hash.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/texture.h>
#include <unistd.h>

#include "pathtracer/gfx/ocio_cpu_transform.h"
#include "pathtracer/gfx/scalar_type.h"

namespace pathtracer::gfx {

struct ImageTexture {
    OIIO::TextureSystem::TextureHandle* handle;
    OIIO::TextureOpt options;  // first channel and wraps; each lookup copies it, since texture() rewrites its wrap fields
    OIIO::ROI level0;          // the finest level's data window and the lookup's channel range
    std::string path;          // the file the cache serves, which readTexels decodes whole
};

namespace {

// The planes an image's channels map to, in order: an HdrImage of n channels is the first n.
constexpr std::array<const char*, 3> kRgbPlanes{"R", "G", "B"};

// The pinned OCIO config as OIIO sees it, so file tags resolve and colorconvert runs against the config the display uses.
const OIIO::ColorConfig& colorConfig() {
    static const OIIO::ColorConfig config(std::string("ocio://") + kOcioConfigName);
    return config;
}

// OpenEXR's meaning for a file with no colour tag: absent chromaticities are BT.709 primaries with a D65 white.
constexpr const char* kOpenExrDefaultColorSpace = "lin_rec709_scene";

// Half a unit in the fourth decimal, the precision ITU-R BT.709 publishes its white point at.
constexpr float kChromaticityTolerance = 5e-5F;

// The first channel and the length of the leading R, G, B run: R onward, or a single-channel image's sole channel (OIIO's Y).
struct ChannelRun {
    int first;
    int count;
};

std::optional<ChannelRun> leadingChannels(const OIIO::ImageSpec& spec, const std::string& path) {
    // OIIO orders a file's channels R, G, B first, so the leading run is contiguous and one read takes it.
    const int first = spec.nchannels == 1 ? 0 : spec.channelindex("R");
    if (first < 0) {
        std::cerr << path << " has no R channel\n";
        return std::nullopt;
    }
    int count = 1;
    while (count < kRgbChannels && first + count < spec.nchannels && spec.channel_name(first + count) == kRgbPlanes[count]) {
        ++count;
    }
    return ChannelRun{first, count};
}

// The linear config colour space a Colour EXR is in: the override, its oiio:ColorSpace tag, else OpenEXR's default; nullopt, logged.
std::optional<std::string> sourceColorSpace(const OIIO::ImageInput& input, const std::string& path,
                                            const std::optional<std::string>& colorSpace) {
    const OIIO::ImageSpec& spec = input.spec();
    std::string tag = colorSpace.value_or(std::string(spec.get_string_attribute("oiio:ColorSpace")));
    if (tag.empty()) {
        tag = kOpenExrDefaultColorSpace;
        // Untagged chromaticities name primaries no config space is matched to: only the default's own are honoured.
        if (const OIIO::ParamValue* declared = spec.find_attribute("chromaticities", OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, 8))) {
            const std::array<float, 8> expected = chromaticitiesOf(std::string(colorConfig().resolve(tag)).c_str());
            const auto* values = static_cast<const float*>(declared->data());
            for (std::size_t i = 0; i < expected.size(); ++i) {
                if (std::abs(values[i] - expected[i]) > kChromaticityTolerance) {
                    std::cerr << path << " declares chromaticities other than its untagged default " << tag
                              << "; name its colour space in the scene\n";
                    return std::nullopt;
                }
            }
        }
    }
    std::string resolved(colorConfig().resolve(tag));
    // Linear only: the pipeline is scene-linear end to end, so an encoded (display or texture-curve) EXR is an authoring error.
    if (colorConfig().getColorSpaceIndex(resolved) < 0 || colorConfig().isData(resolved) || !colorConfig().isColorSpaceLinear(resolved)) {
        std::cerr << path << " is in " << tag << ", not a linear colour space of " << kOcioConfigName << '\n';
        return std::nullopt;
    }
    return resolved;
}

// Converts texels in place from source into the working space; false, logged, on failure.
bool toWorkingSpace(const std::string& source, const std::string& path, HdrImage& image) {
    if (colorConfig().equivalent(source, kOcioSceneColorSpace)) {
        return true;
    }
    if (image.channels != kRgbChannels) {
        std::cerr << path << " needs a colour conversion from " << source << " but has no RGB\n";
        return false;
    }
    OIIO::ImageBuf buffer(OIIO::ImageSpec(image.width, image.height, image.channels, OIIO::TypeFloat), OIIO::make_span(image.texels));
    if (!OIIO::ImageBufAlgo::colorconvert(buffer, buffer, source, kOcioSceneColorSpace, false, "", "", &colorConfig())) {
        std::cerr << path << ": " << buffer.geterror() << '\n';
        return false;
    }
    return true;
}

// The run's `count` channels from `first` as float, row-major; nullopt, logged, on a read failure.
std::optional<HdrImage> readChannels(OIIO::ImageInput& input, const std::string& path, int first, int count) {
    const OIIO::ImageSpec& spec = input.spec();
    HdrImage image = makeImage(spec.width, spec.height, count);
    if (!input.read_image(0, 0, first, first + count, OIIO::TypeFloat, image.texels.data())) {
        std::cerr << path << ": " << input.geterror() << '\n';
        return std::nullopt;
    }
    return image;
}

// Linear OpenEXR is the pipeline's one image format; PNG is a capture output only, never read back as scene data.
bool isOpenExr(const OIIO::ImageInput& input, const std::string& path) {
    if (std::string_view(input.format_name()) == "openexr") {
        return true;
    }
    std::cerr << path << " is " << input.format_name() << ", not OpenEXR\n";
    return false;
}

// Rejects any Inf/NaN texel, logged: an importance-sampling CDF or a shading input built on one is garbage.
bool allFinite(const std::vector<float>& texels, const std::string& path) {
    if (std::all_of(texels.begin(), texels.end(), [](float texel) { return std::isfinite(texel); })) {
        return true;
    }
    std::cerr << path << " holds a non-finite texel\n";
    return false;
}

// Writes image's texels as `format` under spec; false, logged, on any failure.
bool writeImage(const std::string& path, const char* format, const OIIO::ImageSpec& spec, const HdrImage& image) {
    const std::unique_ptr<OIIO::ImageOutput> output = OIIO::ImageOutput::create(format);
    if (!output || !output->open(path, spec) || !output->write_image(OIIO::TypeFloat, image.texels.data()) || !output->close()) {
        std::cerr << "failed to write " << path << ": " << (output ? output->geterror() : OIIO::geterror()) << '\n';
        return false;
    }
    return true;
}

// The process's one TextureSystem: one tile cache, OIIO's default 1 GB, shared by every scene and renderer opened.
OIIO::TextureSystem& textureSystem() {
    static const std::shared_ptr<OIIO::TextureSystem> system = OIIO::TextureSystem::create(/*shared=*/false);
    return *system;
}

// A handle on `name` in the TextureSystem with its channel range and wraps; nullptr, logged, if the cache cannot serve it.
std::shared_ptr<const ImageTexture> textureOf(const std::string& name, int firstChannel, int channels, TextureWrap wrap) {
    OIIO::TextureSystem& system = textureSystem();
    OIIO::TextureSystem::TextureHandle* handle = system.get_texture_handle(OIIO::ustring(name));
    OIIO::ImageSpec spec;
    if (handle == nullptr || !system.good(handle) || !system.get_imagespec(handle, nullptr, 0, spec)) {
        std::cerr << name << ": " << system.geterror() << '\n';
        return nullptr;
    }
    OIIO::TextureOpt options;
    options.firstchannel = firstChannel;
    options.swrap = OIIO::Tex::Wrap::Periodic;
    options.twrap = wrap == TextureWrap::LatLong ? OIIO::Tex::Wrap::Clamp : OIIO::Tex::Wrap::Periodic;
    options.interpmode = OIIO::Tex::InterpMode::Bilinear;
    const OIIO::ROI level0(spec.x, spec.x + spec.width, spec.y, spec.y + spec.height, 0, 1, firstChannel, firstChannel + channels);
    return std::make_shared<const ImageTexture>(ImageTexture{handle, options, level0, name});
}

// Derived textures as tiled MIP-mapped EXR under temp, maketx's auto-tx: keyed by the source file and everything that shapes its texels.
std::filesystem::path derivedTexturePath(const std::string& key) {
    return std::filesystem::temp_directory_path() / "pathtracer-textures" /
           (OIIO::Strutil::fmt::format("{:016x}", OIIO::farmhash::Fingerprint64(key)) + ".exr");
}

// A half source stays half unless conversion took a texel past half's 65504; float and uint32 sources keep float's 24 bits.
OIIO::TypeDesc storageFor(OIIO::TypeDesc source, const HdrImage& image) {
    const bool inHalfRange = std::all_of(image.texels.begin(), image.texels.end(), [](float v) { return std::abs(v) <= kHalfMax; });
    return source == OIIO::TypeHalf && inHalfRange ? OIIO::TypeHalf : OIIO::TypeFloat;
}

// Writes image as a tiled MIP-mapped texture at path, published by an atomic rename so a concurrent reader sees all of it or none.
bool writeDerivedTexture(const std::filesystem::path& path, const HdrImage& image, OIIO::TypeDesc storage) {
    static std::atomic<unsigned> sequence{0};
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    const std::filesystem::path staging =
        path.parent_path() / OIIO::Strutil::fmt::format("{}.{}.{}.exr", path.stem().string(), getpid(), sequence++);
    const OIIO::ImageBuf buffer(OIIO::ImageSpec(image.width, image.height, image.channels, OIIO::TypeFloat),
                                OIIO::cspan<float>(image.texels));
    OIIO::ImageSpec config;
    config.format = storage;
    // A local cache, so decode CPU is dearer than disk: uncompressed tiles cost a first-touch pread, not a deflate per tile.
    config.attribute("compression", "none");
    if (!OIIO::ImageBufAlgo::make_texture(OIIO::ImageBufAlgo::MakeTxTexture, buffer, staging.string(), config)) {
        std::cerr << "failed to write texture " << staging << ": " << OIIO::geterror() << '\n';
        return false;
    }
    std::filesystem::rename(staging, path, error);
    if (error) {
        std::cerr << "failed to publish texture " << path << ": " << error.message() << '\n';
        return false;
    }
    return true;
}

}  // namespace

HdrImage makeImage(int width, int height, int channels) {
    return {width, height, channels,
            std::vector<float>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                                   static_cast<std::size_t>(channels),
                               0.0F)};
}

glm::vec3 HdrImage::rgb(std::size_t pixel) const {
    const float* texel = texels.data() + (pixel * static_cast<std::size_t>(channels));
    if (channels == kScalarChannels) {
        return glm::vec3(texel[0]);
    }
    return {texel[0], texel[1], channels >= kRgbChannels ? texel[2] : 0.0F};
}

std::optional<HdrImage> loadImage(const std::string& path, ImageRole role, const std::optional<std::string>& colorSpace) {
    const std::unique_ptr<OIIO::ImageInput> input = OIIO::ImageInput::open(path);
    if (!input) {
        std::cerr << "loadImage: " << OIIO::geterror() << '\n';
        return std::nullopt;
    }
    if (!isOpenExr(*input, path)) {
        return std::nullopt;
    }
    const std::optional<ChannelRun> run = leadingChannels(input->spec(), path);
    std::optional<HdrImage> image = run ? readChannels(*input, path, run->first, run->count) : std::nullopt;
    if (!image) {
        return std::nullopt;
    }
    if (role == ImageRole::Colour) {
        const std::optional<std::string> source = sourceColorSpace(*input, path, colorSpace);
        if (!source || !toWorkingSpace(*source, path, *image)) {
            return std::nullopt;
        }
    }
    if (!allFinite(image->texels, path)) {
        return std::nullopt;
    }
    return image;
}

bool writeExr(const std::string& path, const HdrImage& image, ImageRole role) {
    OIIO::ImageSpec spec(image.width, image.height, image.channels, OIIO::TypeFloat);
    spec.channelnames.assign(kRgbPlanes.begin(), kRgbPlanes.begin() + image.channels);
    if (role == ImageRole::Colour) {
        // Both, so a reader that knows only one of OIIO's colour tag or OpenEXR's chromaticities reads the same primaries.
        spec.attribute("oiio:ColorSpace", kOcioSceneColorSpace);
        const std::array<float, 8> chromaticities = chromaticitiesOf(kOcioSceneColorSpace);
        spec.attribute("chromaticities", OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, chromaticities.size()), chromaticities.data());
    } else {
        spec.attribute("oiio:ColorSpace", colorConfig().getColorSpaceNameByRole("data"));
    }
    return writeImage(path, "openexr", spec, image);
}

bool writeDisplayPng(const std::string& path, const HdrImage& image) {
    // 16 bits put the quantization step at 2^-16, below any display's resolution, so no dither is needed to hide banding.
    OIIO::ImageSpec spec(image.width, image.height, image.channels, OIIO::TypeUInt16);
    spec.attribute("oiio:UnassociatedAlpha", 1);
    spec.attribute("oiio:ColorSpace", kOcioSrgbDisplay);
    return writeImage(path, "png", spec, image);
}

std::shared_ptr<const ImageTexture> openTexture(const std::string& path, int channels, ImageRole role, TextureWrap wrap,
                                                const std::optional<std::string>& colorSpace, int channelOffset) {
    const std::unique_ptr<OIIO::ImageInput> input = OIIO::ImageInput::open(path);
    if (!input) {
        std::cerr << "openTexture: " << OIIO::geterror() << '\n';
        return nullptr;
    }
    if (!isOpenExr(*input, path)) {
        return nullptr;
    }
    const OIIO::ImageSpec& spec = input->spec();
    const std::optional<ChannelRun> run = leadingChannels(spec, path);
    if (!run) {
        return nullptr;
    }
    if (run->count < channelOffset + channels) {
        std::cerr << "openTexture: " << path << " has " << run->count << " of the " << channelOffset + channels
                  << " R, G, B channels its input reads\n";
        return nullptr;
    }
    const int first = run->first + channelOffset;
    if (role == ImageRole::Data && colorSpace) {
        std::cerr << "openTexture: " << path << " is read as data, so it takes no colour space\n";
        return nullptr;
    }
    const std::optional<std::string> source = role == ImageRole::Colour ? sourceColorSpace(*input, path, colorSpace) : std::nullopt;
    if (role == ImageRole::Colour && !source) {
        return nullptr;
    }
    const bool inWorkingSpace = !source || colorConfig().equivalent(*source, kOcioSceneColorSpace);
    // A tiled, MIP-mapped EXR already in the working space is what the cache serves best: read in place at its own format.
    if (inWorkingSpace && spec.tile_width > 0 && input->seek_subimage(0, 1)) {
        textureSystem().invalidate(OIIO::ustring(path));
        return textureOf(path, first, channels, wrap);
    }
    // Anything else is made into one, maketx's auto-tx: a scanline EXR through the cache decodes whole on first touch, serialised per file.
    const std::filesystem::path derived = derivedTexturePath(OIIO::Strutil::fmt::format(
        "{}|{}|{}|{}+{}|{}|{}|{}", std::filesystem::absolute(path).string(), std::filesystem::last_write_time(path).time_since_epoch().count(),
        std::filesystem::file_size(path), first, channels, source.value_or("data"), kOcioSceneColorSpace, kOcioConfigName));
    if (!std::filesystem::exists(derived)) {
        std::optional<HdrImage> image = readChannels(*input, path, first, channels);
        // OIIO's in-cache transform reads only tiled files, at the tile's own format, under its default config: converted here instead.
        if (!image || (!inWorkingSpace && !toWorkingSpace(*source, path, *image)) || !allFinite(image->texels, path) ||
            !writeDerivedTexture(derived, *image, storageFor(spec.format, *image))) {
            return nullptr;
        }
    }
    return textureOf(derived.string(), 0, channels, wrap);
}

std::shared_ptr<const ImageTexture> makeTexture(const HdrImage& image, TextureWrap wrap) {
    const OIIO::ImageBuf buffer(OIIO::ImageSpec(image.width, image.height, image.channels, OIIO::TypeFloat),
                                OIIO::cspan<float>(image.texels));
    const std::filesystem::path derived = derivedTexturePath(OIIO::Strutil::fmt::format(
        "{}x{}x{}|{}", image.width, image.height, image.channels, OIIO::ImageBufAlgo::computePixelHashSHA1(buffer)));
    if (!std::filesystem::exists(derived) && (!allFinite(image.texels, derived.string()) ||
                                              !writeDerivedTexture(derived, image, OIIO::TypeFloat))) {
        return nullptr;
    }
    return textureOf(derived.string(), 0, image.channels, wrap);
}

std::optional<HdrImage> readTexels(const ImageTexture& texture) {
    // The file's level 0 is what lookups filter: ImageInput decodes it on OIIO's threads, where the cache would fetch tile by tile.
    const std::unique_ptr<OIIO::ImageInput> input = OIIO::ImageInput::open(texture.path);
    if (!input) {
        std::cerr << "readTexels: " << OIIO::geterror() << '\n';
        return std::nullopt;
    }
    return readChannels(*input, texture.path, texture.level0.chbegin, texture.level0.nchannels());
}

glm::vec3 sampleTexture(const ImageTexture& texture, glm::vec2 st, const TextureFootprint& footprint) {
    OIIO::TextureOpt options = texture.options;
    glm::vec3 result(0.0F);
    textureSystem().texture(texture.handle, nullptr, options, st.x, st.y, footprint.dx.x, footprint.dx.y, footprint.dy.x,
                            footprint.dy.y, texture.level0.nchannels(), &result[0]);
    return result;
}

TextureGradient sampleTextureGradient(const ImageTexture& texture, glm::vec2 st, const TextureFootprint& footprint) {
    OIIO::TextureOpt options = texture.options;
    options.interpmode = OIIO::Tex::InterpMode::SmartBicubic;
    TextureGradient gradient{};
    textureSystem().texture(texture.handle, nullptr, options, st.x, st.y, footprint.dx.x, footprint.dx.y, footprint.dy.x,
                            footprint.dy.y, 1, &gradient.value, &gradient.dst.x, &gradient.dst.y);
    return gradient;
}

}  // namespace pathtracer::gfx
