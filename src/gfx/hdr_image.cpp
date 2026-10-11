#include "pathtracer/gfx/hdr_image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfChromaticities.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfStandardAttributes.h>
#include <OpenEXR/ImfThreading.h>
#include <OpenImageIO/color.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>

#include "pathtracer/gfx/ocio_cpu_transform.h"

namespace pathtracer::gfx {

namespace {

// Wraps a float pixel coordinate into [0, size) the way GL_REPEAT wraps a texture coordinate.
int wrapPixel(int coord, int size) {
    const int wrapped = coord % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

// The engine assumes every linear EXR is Rec.709-primaried and never checked: an ACEScg or P3 asset reads back wrong, silently.
bool chromaticitiesMismatchRec709(const Imf::Chromaticities& c) {
    constexpr float kTolerance = 1e-3F;
    const Imf::Chromaticities rec709;
    const auto differs = [](const Imath::V2f& a, const Imath::V2f& b) {
        return std::abs(a.x - b.x) > kTolerance || std::abs(a.y - b.y) > kTolerance;
    };
    return differs(c.red, rec709.red) || differs(c.green, rec709.green) ||
           differs(c.blue, rec709.blue) || differs(c.white, rec709.white);
}

template <typename T>
inline constexpr Imf::PixelType kExrPixelType = Imf::FLOAT;
template <>
inline constexpr Imf::PixelType kExrPixelType<Half> = Imf::HALF;

template <typename T>
inline constexpr ScalarType kScalarType = ScalarType::Float32;
template <>
inline constexpr ScalarType kScalarType<Half> = ScalarType::Float16;

// The planes an image's channels map to, in order: an HdrImage or ImageTexture of n channels is the first n.
constexpr std::array<const char*, 3> kRgbPlanes{"R", "G", "B"};

template <typename T>
struct ExrPixels {
    int width;
    int height;
    std::vector<T> texels;
};

// OpenEXR's global thread count defaults to 0, leaving every compressed scanline block to decompress on the calling thread.
void ensureExrThreadPool() {
    static const int configured = [] {
        const int threads = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
        Imf::setGlobalThreadCount(threads);
        return threads;
    }();
    (void)configured;
}

template <int N>
std::size_t texelIndex(int x, int y, int width) {
    return ((static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x)) * N;
}

// Calls f(texels, integral_constant<N>) through branches on type and channel count, so f inlines with a compile-time stride.
template <typename F>
glm::vec3 withTexels(const ImageTexture& image, F&& f) {
    const auto byChannels = [&](const auto& texels) {
        return image.channels == kScalarChannels ? f(texels, std::integral_constant<int, kScalarChannels>{})
                                                 : f(texels, std::integral_constant<int, kRgbChannels>{});
    };
    if (const auto* half = std::get_if<std::vector<Half>>(&image.texels)) {
        return byChannels(*half);
    }
    return byChannels(*std::get_if<std::vector<float>>(&image.texels));
}

// A scalar map widens to one float, so its bilinear filter is one lerp chain rather than three.
template <int N, typename T>
auto widenTexel(const std::vector<T>& texels, std::size_t idx) {
    if constexpr (N == kScalarChannels) {
        return static_cast<float>(texels[idx]);
    } else {
        return glm::vec3(static_cast<float>(texels[idx + 0]), static_cast<float>(texels[idx + 1]),
                         static_cast<float>(texels[idx + 2]));
    }
}

glm::vec3 asVec3(float r) { return {r, 0.0F, 0.0F}; }
glm::vec3 asVec3(const glm::vec3& rgb) { return rgb; }

// Binds image's interleaved texels as file's frame buffer, R,G,B up to stride; false, logged, if one of them is absent.
template <typename T>
bool bindExrChannels(Imf::InputFile& file, const Imath::Box2i& dw, ExrPixels<T>& image, std::size_t stride, const std::string& path) {
    // Interleaved straight into T: a Float16 read turns an over-range source into Inf, caught by allFinite.
    char* base = reinterpret_cast<char*>(image.texels.data()) -
                 ((static_cast<std::size_t>(dw.min.x) + (static_cast<std::size_t>(dw.min.y) * image.width)) * stride *
                  sizeof(T));
    const std::size_t xStride = stride * sizeof(T);
    const std::size_t yStride = xStride * static_cast<std::size_t>(image.width);
    Imf::FrameBuffer frameBuffer;
    for (std::size_t c = 0; c < stride; ++c) {
        // OpenEXR zero-fills an absent slice, so without this an RGB slot bound to a Y-only file would render black, silently.
        if (file.header().channels().findChannel(kRgbPlanes[c]) == nullptr) {
            std::cerr << "readExrChannels: " << path << " has no " << kRgbPlanes[c] << " channel\n";
            return false;
        }
        frameBuffer.insert(kRgbPlanes[c], Imf::Slice(kExrPixelType<T>, base + (c * sizeof(T)), xStride, yStride));
    }
    file.setFrameBuffer(frameBuffer);
    return true;
}

// Rejects any Inf/NaN texel, logged: a Float16 read also overflows a finite source above binary16's max to Inf.
template <typename T>
bool allFinite(const std::vector<T>& texels, const std::string& path) {
    for (const T texel : texels) {
        if (!std::isfinite(static_cast<float>(texel))) {
            std::cerr << path << ": non-finite texel read as " << scalarTypeName(kScalarType<T>)
                      << " (source Inf/NaN";
            if constexpr (std::is_same_v<T, Half>) {
                std::cerr << ", or a finite value above binary16's max " << kHalfMax;
            }
            std::cerr << ") -- rejecting rather than propagating garbage into importance sampling\n";
            return false;
        }
    }
    return true;
}

// Reads path's first `channels` of R,G,B interleaved as T. nullopt on I/O failure, an absent R/G/B, or a non-finite texel.
template <typename T>
std::optional<ExrPixels<T>> readExrChannels(const std::string& path, int channels) {
    try {
        ensureExrThreadPool();
        Imf::InputFile file(path.c_str());
        const Imath::Box2i& dw = file.header().dataWindow();
        if (dw.isEmpty()) {
            std::cerr << "readExrChannels: empty data window in " << path << '\n';
            return std::nullopt;
        }

        // Primaries, not transfer: absence is the documented Rec.709 assumption, while a present-but-different attribute is a real defect.
        if (Imf::hasChromaticities(file.header()) &&
            chromaticitiesMismatchRec709(Imf::chromaticities(file.header()))) {
            std::cerr << "readExrChannels: " << path
                      << " declares non-Rec.709 chromaticities -- colours will be systematically wrong "
                         "under this engine's Rec.709 assumption\n";
        }

        const int width = dw.max.x - dw.min.x + 1;
        const int height = dw.max.y - dw.min.y + 1;

        const auto stride = static_cast<std::size_t>(channels);
        ExrPixels<T> image{width, height, {}};
        image.texels.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * stride, T(0));

        if (!bindExrChannels(file, dw, image, stride, path)) {
            return std::nullopt;
        }
        file.readPixels(dw.min.y, dw.max.y);
        if (!allFinite(image.texels, path)) {
            return std::nullopt;
        }
        return image;
    } catch (const std::exception& e) {
        std::cerr << "readExrChannels: failed to load " << path << ": " << e.what() << '\n';
        return std::nullopt;
    }
}

// The pinned OCIO config as OIIO sees it, so file tags resolve and colorconvert runs against the config the display uses.
const OIIO::ColorConfig& colorConfig() {
    static const OIIO::ColorConfig config(std::string("ocio://") + kOcioConfigName);
    return config;
}

// OpenEXR's meaning for a file with no colour tag: absent chromaticities are BT.709 primaries with a D65 white.
constexpr const char* kOpenExrDefaultColorSpace = "lin_rec709_scene";

// Half a unit in the fourth decimal, the precision ITU-R BT.709 publishes its white point at.
constexpr float kChromaticityTolerance = 5e-5F;

// The linear config colour space a Colour EXR is in: its oiio:ColorSpace tag, else OpenEXR's default; nullopt, logged, otherwise.
std::optional<std::string> sourceColorSpace(const OIIO::ImageInput& input, const std::string& path) {
    const OIIO::ImageSpec& spec = input.spec();
    std::string tag(spec.get_string_attribute("oiio:ColorSpace"));
    if (tag.empty()) {
        tag = kOpenExrDefaultColorSpace;
        // Untagged chromaticities name primaries no config space is matched to: only the default's own are honoured.
        if (const OIIO::ParamValue* declared = spec.find_attribute("chromaticities", OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, 8))) {
            const std::array<float, 8> expected = chromaticitiesOf(std::string(colorConfig().resolve(tag)).c_str());
            const auto* values = static_cast<const float*>(declared->data());
            for (std::size_t i = 0; i < expected.size(); ++i) {
                if (std::abs(values[i] - expected[i]) > kChromaticityTolerance) {
                    std::cerr << "loadImage: " << path << " declares chromaticities other than its untagged default " << tag << '\n';
                    return std::nullopt;
                }
            }
        }
    }
    std::string resolved(colorConfig().resolve(tag));
    // Linear only: the pipeline is scene-linear end to end, so an encoded (display or texture-curve) EXR is an authoring error.
    if (colorConfig().getColorSpaceIndex(resolved) < 0 || colorConfig().isData(resolved) || !colorConfig().isColorSpaceLinear(resolved)) {
        std::cerr << "loadImage: " << path << " is tagged " << tag << ", not a linear colour space of " << kOcioConfigName << '\n';
        return std::nullopt;
    }
    return resolved;
}

// Converts a Colour image's texels in place from the file's colour space to the working space; false, logged, on failure.
bool toWorkingSpace(const OIIO::ImageInput& input, const std::string& path, HdrImage& image) {
    const std::optional<std::string> source = sourceColorSpace(input, path);
    if (!source) {
        return false;
    }
    if (colorConfig().equivalent(*source, kOcioSceneColorSpace)) {
        return true;
    }
    if (image.channels != kRgbChannels) {
        std::cerr << "loadImage: " << path << " needs a colour conversion from " << *source << " but has no RGB\n";
        return false;
    }
    OIIO::ImageBuf buffer(OIIO::ImageSpec(image.width, image.height, image.channels, OIIO::TypeFloat), OIIO::make_span(image.texels));
    if (!OIIO::ImageBufAlgo::colorconvert(buffer, buffer, *source, kOcioSceneColorSpace, false, "", "", &colorConfig())) {
        std::cerr << "loadImage: " << path << ": " << buffer.geterror() << '\n';
        return false;
    }
    return true;
}

// Writes image's texels as `format` under spec; false, logged, on any failure.
bool writeImage(const std::string& path, const char* format, const OIIO::ImageSpec& spec, const HdrImage& image) {
    const std::unique_ptr<OIIO::ImageOutput> output = OIIO::ImageOutput::create(format);
    if (!output || !output->open(path, spec) || !output->write_image(OIIO::TypeFloat, image.texels.data()) || !output->close()) {
        std::cerr << "writeImage: failed to write " << path << ": " << (output ? output->geterror() : OIIO::geterror()) << '\n';
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

std::optional<ImageTexture> loadImageTexture(const std::string& path, ScalarType type, int channels) {
    const auto load = [&path, channels]<typename T>() -> std::optional<ImageTexture> {
        std::optional<ExrPixels<T>> pixels = readExrChannels<T>(path, channels);
        if (!pixels) {
            return std::nullopt;
        }
        return ImageTexture{pixels->width, pixels->height, channels, std::move(pixels->texels)};
    };
    switch (type) {
        case ScalarType::Float16:
            return load.template operator()<Half>();
        case ScalarType::Float32:
            return load.template operator()<float>();
    }
    // Not dead: a scoped enum holds any value of its underlying type, so falling off a covered switch is still undefined behaviour.
    return std::nullopt;
}

glm::vec3 ImageTexture::texel(int x, int y) const {
    return withTexels(*this, [&](const auto& texels, auto channels) {
        constexpr int kN = decltype(channels)::value;
        return asVec3(widenTexel<kN>(texels, texelIndex<kN>(x, y, width)));
    });
}

std::optional<HdrImage> loadImage(const std::string& path, ImageRole role) {
    const std::unique_ptr<OIIO::ImageInput> input = OIIO::ImageInput::open(path);
    if (!input) {
        std::cerr << "loadImage: " << OIIO::geterror() << '\n';
        return std::nullopt;
    }
    // Linear OpenEXR is the pipeline's one image format; PNG is a capture output only, never read back as scene data.
    if (std::string_view(input->format_name()) != "openexr") {
        std::cerr << "loadImage: " << path << " is " << input->format_name() << ", not OpenEXR\n";
        return std::nullopt;
    }
    const OIIO::ImageSpec& spec = input->spec();
    // OIIO orders a file's channels R, G, B first, so the leading run is contiguous and one read takes it.
    const int first = spec.channelindex("R");
    if (first < 0) {
        std::cerr << "loadImage: " << path << " has no R channel\n";
        return std::nullopt;
    }
    int channels = 1;
    while (channels < kRgbChannels && first + channels < spec.nchannels && spec.channel_name(first + channels) == kRgbPlanes[channels]) {
        ++channels;
    }
    HdrImage image = makeImage(spec.width, spec.height, channels);
    if (!input->read_image(0, 0, first, first + channels, OIIO::TypeFloat, image.texels.data())) {
        std::cerr << "loadImage: " << path << ": " << input->geterror() << '\n';
        return std::nullopt;
    }
    if (role == ImageRole::Colour && !toWorkingSpace(*input, path, image)) {
        return std::nullopt;
    }
    if (!allFinite(image.texels, path)) {
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

glm::vec3 sampleBilinear(const ImageTexture& image, glm::vec2 uv, WrapMode wrap) {
    // Texel-center convention, matching GL_LINEAR.
    const float fx = (uv.x * static_cast<float>(image.width)) - 0.5F;
    const float fy = (uv.y * static_cast<float>(image.height)) - 0.5F;
    const int x0 = static_cast<int>(std::floor(fx));
    const int y0 = static_cast<int>(std::floor(fy));
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);
    const int wx0 = wrapPixel(x0, image.width);
    const int wx1 = wrapPixel(x0 + 1, image.width);
    // Clamping v holds the pole row instead of fetching the opposite pole, which wrapping v does at both ends of an equirect map.
    const auto resolveV = [&](int y) {
        return wrap == WrapMode::ClampV ? std::clamp(y, 0, image.height - 1) : wrapPixel(y, image.height);
    };
    const int wy0 = resolveV(y0);
    const int wy1 = resolveV(y0 + 1);

    // Four taps resolving to one texel make bilinear exactly that texel: the mix would only round a constant, and 1x1 maps are the norm.
    if (wx0 == wx1 && wy0 == wy1) {
        return image.texel(wx0, wy0);
    }

    return withTexels(image, [&](const auto& texels, auto channels) {
        constexpr int kN = decltype(channels)::value;
        const auto texel = [&](int x, int y) { return widenTexel<kN>(texels, texelIndex<kN>(x, y, image.width)); };
        const auto top = glm::mix(texel(wx0, wy0), texel(wx1, wy0), tx);
        const auto bottom = glm::mix(texel(wx0, wy1), texel(wx1, wy1), tx);
        return asVec3(glm::mix(top, bottom, ty));
    });
}

}  // namespace pathtracer::gfx
