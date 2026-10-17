// The viewer's GL display path run for real in a hidden context, against the CPU encode: upload, transform, dither, optics, histogram.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <vector>

// GLEW must precede GLFW: GLFW detects GLEW's include guard and skips the platform GL headers. Reversed, the declarations clash.
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <OpenColorIO/OpenColorIO.h>
#include <glm/glm.hpp>

#include "check.h"
#include "pathtracer/debug/histogram.h"
#include "pathtracer/debug/histogram_curve.h"
#include "pathtracer/gfx/bloom_pass.h"
#include "pathtracer/gfx/display_optics.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/gfx/ocio_cpu_transform.h"
#include "pathtracer/gfx/ocio_display_transform.h"
#include "pathtracer/gfx/pixel_lattice.h"
#include "pathtracer/gfx/post_process_pass.h"
#include "pathtracer/gfx/scalar_type.h"
#include "pathtracer/gfx/texture.h"
#include "pathtracer/scene/thread_pool.h"

namespace {

namespace OCIO = OCIO_NAMESPACE;
using pathtracer::gfx::HdrImage;
using pathtracer::gfx::OcioDisplayTransform;
using Lut = OcioDisplayTransform::Lut;
using pathtracer::gfx::PixelLattice;
using pathtracer::gfx::ScalarType;
using pathtracer::gfx::ViewportRect;

constexpr double kUnitRoundoff = 0x1p-24;  // GLSL 4.10 4.7.1: +, -, * correctly rounded, so each rounds by at most u
constexpr double kUlp = 0x1p-23;           // a float's ulp is at most 2^-23 of its magnitude
// GL 4.1 2.1.1: fixed-function arithmetic (interpolation, filtering) is "accurate to about 1 part in 10^5", the only bound GL states.
constexpr double kFixedFunction = 1e-5;

[[nodiscard]] double gamma(double n) { return (n * kUnitRoundoff) / (1.0 - (n * kUnitRoundoff)); }

// One hidden 4.1 core context per process, the viewer's own hints; null where no display exists, which every check reports as a skip.
GLFWwindow* glContext() {
    static GLFWwindow* const window = []() -> GLFWwindow* {
        if (glfwInit() != GLFW_TRUE) {
            return nullptr;
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        GLFWwindow* created = glfwCreateWindow(1, 1, "display_gl_validate", nullptr, nullptr);
        if (created == nullptr) {
            return nullptr;
        }
        glfwMakeContextCurrent(created);
        glewExperimental = GL_TRUE;
        const GLenum status = glewInit();
        // GLEW's init leaves a spurious error even on success; drained so no later check reads it as its own.
        while (glGetError() != GL_NO_ERROR) {
        }
        return status == GLEW_OK ? created : nullptr;
    }();
    return window;
}

// One colour attachment. Every draw goes to one, never the default framebuffer, whose pixels a hidden window need not own.
class RenderTarget {
public:
    RenderTarget(int width, int height, GLint internalFormat) : width_(width), height_(height) {
        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glGenFramebuffers(1, &fbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_, 0);
        glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    ~RenderTarget() {
        glDeleteFramebuffers(1, &fbo_);
        glDeleteTextures(1, &texture_);
    }
    RenderTarget(const RenderTarget&) = delete;
    RenderTarget& operator=(const RenderTarget&) = delete;

    [[nodiscard]] unsigned int fbo() const { return fbo_; }

    // Bottom-up rows as GL stores them, `components` per texel.
    template <typename T>
    [[nodiscard]] std::vector<T> read(GLenum format, int components) const {
        std::vector<T> texels(static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * static_cast<std::size_t>(components));
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width_, height_, format, std::is_same_v<T, float> ? GL_FLOAT : GL_UNSIGNED_BYTE, texels.data());
        return texels;
    }

private:
    int width_;
    int height_;
    unsigned int texture_ = 0;
    unsigned int fbo_ = 0;
};

// The viewer's display objects, built once per check like the viewer builds them once per run.
struct DisplayPath {
    OcioDisplayTransform transform;
    pathtracer::gfx::BloomPass bloom;
    pathtracer::gfx::PostProcessPass pass;
};

std::optional<DisplayPath> makeDisplayPath() {
    std::optional<OcioDisplayTransform> transform = OcioDisplayTransform::create();
    std::optional<pathtracer::gfx::BloomPass> bloom = pathtracer::gfx::BloomPass::create();
    if (!transform || !bloom) {
        return std::nullopt;
    }
    return DisplayPath{std::move(*transform), std::move(*bloom), {}};
}

// The state presentFrame sets before its draw; bloom built from the display texture when its strength is nonzero.
struct DisplayState {
    Lut lut = Lut::SRGB;
    glm::vec3 gain{1.0F};
    glm::vec3 offset{0.0F};
    int channelView = 0;
    bool invert = false;
    float aberration = 0.0F;
    float bloomStrength = 0.0F;
    PixelLattice lattice = PixelLattice::Square;
    bool wrapsHorizontally = false;
};

void present(DisplayPath& path, const pathtracer::gfx::Texture& image, int width, int height, ScalarType format,
             const DisplayState& state, ViewportRect rect, const RenderTarget& target) {
    if (state.bloomStrength > 0.0F) {
        path.bloom.build(image.id(), width, height, format, state.wrapsHorizontally);
    }
    path.bloom.bind(pathtracer::gfx::kBloomTextureUnit);
    path.transform.setActiveLut(state.lut);
    path.transform.setDisplayAffine(state.gain, state.offset);
    path.transform.setChannelView(state.channelView);
    path.transform.setInvert(state.invert);
    path.transform.setAberration(state.aberration);
    path.transform.setBloomStrength(state.bloomStrength);
    path.transform.setPixelLattice(state.lattice, state.wrapsHorizontally);
    path.transform.setImageRect(rect);
    path.transform.bind();
    path.pass.draw(image.id(), path.transform.activeShader(), rect, target.fbo());
    glFinish();
}

// Image texel (x, y), row 0 at the top, as drawn into rect: the vertex shader's v flip puts it at GL row rect.y + rect.height - 1 - y.
std::size_t drawnTexel(ViewportRect rect, int targetWidth, int x, int y) {
    return (static_cast<std::size_t>(rect.y + rect.height - 1 - y) * static_cast<std::size_t>(targetWidth)) +
           static_cast<std::size_t>(rect.x + x);
}

float ditherAt(int x, int y) {
    constexpr int kTile = pathtracer::gfx::kDitherMaskSize;
    return pathtracer::gfx::displayDitherMask()[(static_cast<std::size_t>(y % kTile) * kTile) + static_cast<std::size_t>(x % kTile)];
}

// Log-spaced radiance over the toe, the curve and far over-range, a tenth negative so the linear toe's continuation and the clamp run.
HdrImage testImage(int width, int height, std::uint64_t seed) {
    HdrImage image = pathtracer::gfx::makeImage(width, height, pathtracer::gfx::kRgbChannels);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> logMagnitude(std::log(1e-4), std::log(16.0));
    std::bernoulli_distribution negative(0.1);
    for (float& value : image.texels) {
        value = static_cast<float>((negative(rng) ? -0.05 : 1.0) * std::exp(logMagnitude(rng)));
    }
    return image;
}

// The texel the display texture stores: binary16 rounds once at upload, so the GPU's input is the rounded value, not the CPU's.
float stored(float value, ScalarType format) {
    return format == ScalarType::Float16 ? static_cast<float>(static_cast<pathtracer::gfx::Half>(value)) : value;
}

// --- The display transform in double, from the parameters of the very OCIO processor the shaders are generated from.

struct DisplayCurve {
    glm::dmat3 matrix{1.0};  // scene to display primaries; column-major, as glm stores it
    double exponent = 1.0;   // encoded-to-linear gamma; the display direction raises to 1/exponent
    double offset = 0.0;     // ExponentWithLinear's offset, 0 for a pure power
    bool linearToe = false;  // ExponentWithLinear (sRGB), not Exponent (BT.1886)
};

// The optimized processor is one matrix then one curve; anything else means the shader's structure changed and this reference with it.
std::optional<DisplayCurve> displayCurve(Lut lut) {
    const OCIO::ConstConfigRcPtr config = OCIO::Config::CreateFromBuiltinConfig(pathtracer::gfx::kOcioConfigName);
    const char* display = lut == Lut::Rec709 ? pathtracer::gfx::kOcioRec709Display : pathtracer::gfx::kOcioSrgbDisplay;
    const OCIO::ConstProcessorRcPtr processor =
        config->getProcessor(pathtracer::gfx::kOcioSceneColorSpace, display, pathtracer::gfx::kOcioView, OCIO::TRANSFORM_DIR_FORWARD)
            ->getOptimizedProcessor(OCIO::OPTIMIZATION_DEFAULT);
    const OCIO::GroupTransformRcPtr group = processor->createGroupTransform();
    if (group->getNumTransforms() != 2) {
        return std::nullopt;
    }
    const auto matrixTransform = OCIO::DynamicPtrCast<const OCIO::MatrixTransform>(group->getTransform(0));
    if (!matrixTransform) {
        return std::nullopt;
    }
    std::array<double, 16> m{};
    std::array<double, 4> offset{};
    matrixTransform->getMatrix(m.data());
    matrixTransform->getOffset(offset.data());
    if (offset[0] != 0.0 || offset[1] != 0.0 || offset[2] != 0.0) {
        return std::nullopt;
    }
    DisplayCurve curve;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            curve.matrix[column][row] = m[static_cast<std::size_t>((row * 4) + column)];
        }
    }
    double exponent[4] = {};  // OCIO's signature takes a C array by reference
    if (const auto monCurve = OCIO::DynamicPtrCast<const OCIO::ExponentWithLinearTransform>(group->getTransform(1))) {
        double curveOffset[4] = {};
        monCurve->getGamma(exponent);
        monCurve->getOffset(curveOffset);
        curve.offset = curveOffset[0];
        curve.linearToe = true;
        if (monCurve->getDirection() != OCIO::TRANSFORM_DIR_INVERSE || monCurve->getNegativeStyle() != OCIO::NEGATIVE_LINEAR) {
            return std::nullopt;
        }
    } else if (const auto power = OCIO::DynamicPtrCast<const OCIO::ExponentTransform>(group->getTransform(1))) {
        power->getValue(exponent);
        if (power->getDirection() != OCIO::TRANSFORM_DIR_INVERSE || power->getNegativeStyle() != OCIO::NEGATIVE_CLAMP) {
            return std::nullopt;
        }
    } else {
        return std::nullopt;
    }
    curve.exponent = exponent[0];
    return curve;
}

// The linear-toe break in the display direction: value and slope of the power segment meet the line there (OCIO's monCurve).
double toeBreak(const DisplayCurve& c) {
    return std::pow((c.offset * c.exponent) / ((c.exponent - 1.0) * (1.0 + c.offset)), c.exponent);
}

double toeSlope(const DisplayCurve& c) {
    return (c.offset / (c.exponent - 1.0)) / toeBreak(c);
}

double powerSegment(const DisplayCurve& c, double v) {
    return ((1.0 + c.offset) * std::pow(std::max(v, 0.0), 1.0 / c.exponent)) - c.offset;
}

double encode(const DisplayCurve& c, double v) {
    if (c.linearToe) {
        return v > toeBreak(c) ? powerSegment(c, v) : v * toeSlope(c);
    }
    return std::pow(std::max(v, 0.0), 1.0 / c.exponent);
}

// GLSL 4.10 4.7.1 on pow(v, p), v > 0: exp2(p log2 v), log2 at 3 ulp (2^-21 absolute on [0.5, 2]), exp2 at 3 + 2|z| ulp.
double powRelativeError(double v, double p) {
    if (!(v > 0.0)) {
        return 0.0;
    }
    const double log2v = std::log2(v);
    const double logError = (v >= 0.5 && v <= 2.0) ? 0x1p-21 : 3.0 * kUlp * std::abs(log2v);
    const double z = p * log2v;
    return (std::numbers::ln2 * ((p * logError) + (kUnitRoundoff * std::abs(z)))) + (kUlp * (3.0 + (2.0 * std::abs(z))));
}

// |GPU - exact| on one encoded lane at exact input m with upstream error dm: propagation, pow, and the float-rounded constants.
double curveError(const DisplayCurve& c, double m, double dm) {
    const double propagated = std::max(std::abs(encode(c, m + dm) - encode(c, m)), std::abs(encode(c, m) - encode(c, m - dm)));
    const double p = 1.0 / c.exponent;
    // Evaluation error is taken at the worse end of [m - dm, m + dm], so it bounds wherever the GPU's own input landed.
    double evaluation = 0.0;
    for (const double v : {m - dm, m, m + dm}) {
        const double power = std::pow(std::max(v, 0.0), p);
        const double scaled = (1.0 + c.offset) * power;
        // pow, its exponent constant rounded (relative u times |ln v| p), then scale (u) and offset (u) and their own constants.
        const double powError = (powRelativeError(v, p) + (kUnitRoundoff * p * std::abs(std::log(std::max(v, 1e-300))))) * scaled;
        if (!c.linearToe) {
            evaluation = std::max(evaluation, powError);
            continue;
        }
        const double linear = std::abs(v * toeSlope(c));
        const double powSegment = powError + (gamma(4.0) * (scaled + c.offset));
        // OCIO blends both segments by a 0/1 mask; without `precise` GLSL may refactor that to mix(), rounding on both segments' size.
        const double blend = gamma(3.0) * (std::abs(scaled - c.offset) + linear);
        evaluation = std::max(evaluation, powSegment + (gamma(2.0) * linear) + blend);
    }
    // The GPU tests its own rounded input against a float break point: within reach of it, either branch may be taken.
    double branch = 0.0;
    if (c.linearToe && std::abs(m - toeBreak(c)) <= dm + (kUnitRoundoff * toeBreak(c))) {
        branch = std::abs(powerSegment(c, m) - (m * toeSlope(c)));
    }
    return propagated + evaluation + branch;
}

// The display shader in double and its GLSL error bound per lane, from the texel the GPU sampled.
struct LaneBound {
    std::array<double, 3> value{};
    std::array<double, 3> error{};
};

LaneBound referenceTexel(const glm::vec3& texel, const DisplayState& s, const std::optional<DisplayCurve>& curve, float dither) {
    LaneBound out;
    glm::dvec3 a;
    glm::dvec3 da;
    for (int lane = 0; lane < 3; ++lane) {
        const double scaled = static_cast<double>(texel[lane]) * s.gain[lane];
        a[lane] = scaled + s.offset[lane];
        da[lane] = gamma(2.0) * (std::abs(scaled) + std::abs(static_cast<double>(s.offset[lane])));
    }
    if (s.channelView > 0) {
        a = glm::dvec3(a[s.channelView - 1]);
        da = glm::dvec3(da[s.channelView - 1]);
    }
    glm::dvec3 c = a;
    glm::dvec3 dc = da;
    if (curve) {
        const glm::dvec3 m = curve->matrix * a;
        for (int lane = 0; lane < 3; ++lane) {
            // gamma_3 for a 3-term dot product in any order or fusion, plus u for each matrix constant's rounding to float.
            double mass = 0.0;
            double inherited = 0.0;
            for (int j = 0; j < 3; ++j) {
                mass += std::abs(curve->matrix[j][lane] * a[j]);
                inherited += std::abs(curve->matrix[j][lane]) * da[j];
            }
            const double dm = (gamma(4.0) * mass) + inherited;
            c[lane] = encode(*curve, m[lane]);
            dc[lane] = curveError(*curve, m[lane], dm);
        }
    }
    for (std::size_t lane = 0; lane < 3; ++lane) {
        double v = c[static_cast<glm::length_t>(lane)];
        double dv = dc[static_cast<glm::length_t>(lane)];
        if (s.invert) {
            v = 1.0 - v;
            dv += kUnitRoundoff * std::abs(v);
        }
        v += static_cast<double>(dither);
        out.value[lane] = v;
        out.error[lane] = dv + (kUnitRoundoff * std::abs(v));
    }
    return out;
}

// The GL code of a value in [lo, hi] under GL 4.6 2.3.5.2's round-to-nearest; one code when no half-code boundary lies inside.
struct CodeRange {
    long lo;
    long hi;
};

CodeRange codesOf(double lo, double hi) {
    const auto code = [](double v) { return std::lround(255.0 * std::clamp(v, 0.0, 1.0)); };
    return {code(lo), code(hi)};
}

// The CPU encode's dithered value before quantisation; encodeForDisplay is sRGB-or-identity, so other LUTs compose its own stages.
std::vector<float> cpuDithered(const HdrImage& image, const DisplayState& s) {
    const std::size_t floats = image.texels.size();
    std::vector<float> mapped(floats);
    pathtracer::gfx::encodeForDisplay(image.texels, image.width, image.height, s.gain, false, s.offset, std::span<float>(mapped));
    if (s.channelView > 0) {
        for (std::size_t texel = 0; texel < floats; texel += 3) {
            std::fill_n(mapped.begin() + static_cast<std::ptrdiff_t>(texel), 3, mapped[texel + static_cast<std::size_t>(s.channelView - 1)]);
        }
    }
    pathtracer::gfx::applyOcioDisplayTransform(mapped, image.width, image.height, s.lut);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const std::size_t texel = ((static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width)) + static_cast<std::size_t>(x)) * 3;
            for (std::size_t lane = 0; lane < 3; ++lane) {
                const float displayed = s.invert ? 1.0F - mapped[texel + lane] : mapped[texel + lane];
                mapped[texel + lane] = std::clamp(displayed + ditherAt(x, y), 0.0F, 1.0F);
            }
        }
    }
    return mapped;
}

// Each upload path round-trips bit for bit, and each channel count samples as HdrImage::rgb expands it.
PT_CHECK(texture_upload_round_trips, Fast, Exact) {
    if (glContext() == nullptr) {
        ctx.skip("no GL context on this host");
        return;
    }
    std::optional<DisplayPath> path = makeDisplayPath();
    struct Shape {
        int width;
        int height;
    };
    // First upload, the same shape again (glTexSubImage2D), then a new shape (glTexImage2D); odd widths break 4-byte row alignment.
    constexpr std::array<Shape, 3> kShapes{{{13, 7}, {13, 7}, {6, 5}}};
    constexpr std::array<ScalarType, 2> kFormats{ScalarType::Float16, ScalarType::Float32};
    ctx.plan(1 + (static_cast<int>(kFormats.size()) * 3 * static_cast<int>(kShapes.size() + 1)));
    PT_EXPECT(ctx, path.has_value(), "the display shaders did not build");
    if (!path) {
        return;
    }
    std::mt19937_64 rng(ctx.seed());
    std::uniform_real_distribution<float> radiance(-8.0F, 8.0F);
    for (const ScalarType format : kFormats) {
        for (int channels = 1; channels <= 3; ++channels) {
            std::optional<pathtracer::gfx::Texture> texture;
            std::vector<float> texels;
            for (std::size_t upload = 0; upload < kShapes.size(); ++upload) {
                const Shape shape = kShapes[upload];
                texels.resize(static_cast<std::size_t>(shape.width * shape.height * channels));
                std::generate(texels.begin(), texels.end(), [&] { return radiance(rng); });
                // Half-range edges and a binary16 subnormal, so rounding, the largest finite value and gradual underflow all run.
                texels[0] = pathtracer::gfx::kHalfMax;
                texels[1] = 3e-6F;
                if (texture) {
                    texture->upload(shape.width, shape.height, channels, texels.data());
                } else {
                    texture = pathtracer::gfx::Texture::createFromFloatPixels(shape.width, shape.height, channels, texels.data(), format);
                }
                const int lanes = channels == 3 ? 4 : channels;
                const GLenum layout = channels == 1 ? GL_RED : (channels == 2 ? GL_RG : GL_RGBA);
                const std::size_t count = static_cast<std::size_t>(shape.width * shape.height * lanes);
                glBindTexture(GL_TEXTURE_2D, texture->id());
                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                bool identical = true;
                // RGB pads alpha with 1; Float16 is the compiler's binary16 conversion, Float32 the value itself.
                const auto expected = [&](std::size_t i) {
                    const std::size_t pixel = i / static_cast<std::size_t>(lanes);
                    const auto lane = static_cast<int>(i % static_cast<std::size_t>(lanes));
                    return lane < channels ? texels[(pixel * static_cast<std::size_t>(channels)) + static_cast<std::size_t>(lane)] : 1.0F;
                };
                if (format == ScalarType::Float16) {
                    std::vector<std::uint16_t> back(count);
                    glGetTexImage(GL_TEXTURE_2D, 0, layout, GL_HALF_FLOAT, back.data());
                    for (std::size_t i = 0; i < count; ++i) {
                        identical = identical && back[i] == std::bit_cast<std::uint16_t>(static_cast<pathtracer::gfx::Half>(expected(i)));
                    }
                } else {
                    std::vector<float> back(count);
                    glGetTexImage(GL_TEXTURE_2D, 0, layout, GL_FLOAT, back.data());
                    for (std::size_t i = 0; i < count; ++i) {
                        identical = identical && std::bit_cast<std::uint32_t>(back[i]) == std::bit_cast<std::uint32_t>(expected(i));
                    }
                }
                PT_EXPECT(ctx, identical, std::string(pathtracer::gfx::scalarTypeName(format)) + " " + std::to_string(channels) +
                                              "-channel upload " + std::to_string(upload) + " is not bit-identical");
            }
            // Raw at unit gain is the sampled texel plus the dither, one correctly rounded add: the swizzle's expansion is exact.
            const Shape last = kShapes.back();
            const RenderTarget target(last.width, last.height, GL_RGBA32F);
            const ViewportRect rect{0, 0, last.width, last.height};
            DisplayState raw;
            raw.lut = Lut::Raw;
            present(*path, *texture, last.width, last.height, format, raw, rect, target);
            const std::vector<float> drawn = target.read<float>(GL_RGBA, 4);
            bool expanded = true;
            for (int y = 0; y < last.height; ++y) {
                for (int x = 0; x < last.width; ++x) {
                    const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(last.width)) + static_cast<std::size_t>(x);
                    const auto lane = [&](int c) { return stored(texels[(pixel * static_cast<std::size_t>(channels)) + static_cast<std::size_t>(c)], format); };
                    const glm::vec3 rgb = channels == 1 ? glm::vec3(lane(0)) : glm::vec3(lane(0), lane(1), channels == 3 ? lane(2) : 0.0F);
                    const std::size_t at = drawnTexel(rect, last.width, x, y) * 4;
                    for (int c = 0; c < 3; ++c) {
                        expanded = expanded && drawn[at + static_cast<std::size_t>(c)] == rgb[c] + ditherAt(x, y);
                    }
                }
            }
            PT_EXPECT(ctx, expanded, std::to_string(channels) + "-channel texture does not sample as HdrImage::rgb expands it");
        }
    }
}

// The real display shaders against the double transform within GLSL's stated precision, and against the CPU encode code for code.
PT_CHECK(display_matches_the_cpu_encode, Fast, Exact) {
    if (glContext() == nullptr) {
        ctx.skip("no GL context on this host");
        return;
    }
    std::optional<DisplayPath> path = makeDisplayPath();
    // Wider than one dither tile, drawn off the target's origin, so the tile's anchoring and the row flip are both exercised.
    constexpr int kWidth = 160;
    constexpr int kHeight = 72;
    constexpr ViewportRect kRect{23, 17, kWidth, kHeight};
    constexpr int kTargetWidth = kRect.x + kWidth + 11;
    constexpr int kTargetHeight = kRect.y + kHeight + 9;
    std::vector<DisplayState> states(9);
    states[1].gain = glm::vec3(1.7F, 0.9F, 1.2F);
    states[1].offset = glm::vec3(0.01F, -0.02F, 0.0F);
    states[2].lut = Lut::Rec709;
    states[3].lut = Lut::Rec709;
    states[3].invert = true;
    for (int view = 1; view <= 3; ++view) {
        states[static_cast<std::size_t>(3 + view)].channelView = view;
        states[static_cast<std::size_t>(3 + view)].gain = states[1].gain;
    }
    states[7].lut = Lut::Raw;
    states[7].gain = glm::vec3(1.3F);
    states[7].offset = glm::vec3(0.05F);
    states[7].invert = true;
    states[8].lut = Lut::SRGB;  // the viewer's default 16-bit display texture, below
    constexpr std::array<ScalarType, 2> kFormats{ScalarType::Float32, ScalarType::Float16};
    // Two per state, plus encodeForDisplay's own bytes for the sRGB states without a view or invert: 0, 1 and the 16-bit one.
    ctx.plan(1 + (2 * static_cast<int>(states.size())) + 3);
    const std::optional<DisplayCurve> srgb = displayCurve(Lut::SRGB);
    const std::optional<DisplayCurve> rec709 = displayCurve(Lut::Rec709);
    PT_EXPECT(ctx, path.has_value() && srgb.has_value() && rec709.has_value(),
              "the display shaders did not build, or OCIO's processor is no longer one matrix and one curve");
    if (!path || !srgb || !rec709) {
        return;
    }
    const HdrImage image = testImage(kWidth, kHeight, ctx.seed());
    for (std::size_t i = 0; i < states.size(); ++i) {
        const DisplayState& state = states[i];
        const ScalarType format = i + 1 == states.size() ? kFormats[1] : kFormats[0];
        const pathtracer::gfx::Texture texture =
            pathtracer::gfx::Texture::createFromFloatPixels(kWidth, kHeight, 3, image.texels.data(), format);
        const std::optional<DisplayCurve> curve = state.lut == Lut::Raw ? std::nullopt : (state.lut == Lut::Rec709 ? rec709 : srgb);
        const RenderTarget floatTarget(kTargetWidth, kTargetHeight, GL_RGBA32F);
        const RenderTarget byteTarget(kTargetWidth, kTargetHeight, GL_RGBA8);
        present(*path, texture, kWidth, kHeight, format, state, kRect, floatTarget);
        present(*path, texture, kWidth, kHeight, format, state, kRect, byteTarget);
        const std::vector<float> drawn = floatTarget.read<float>(GL_RGBA, 4);
        const std::vector<unsigned char> codes = byteTarget.read<unsigned char>(GL_RGBA, 4);
        const std::vector<float> cpu = cpuDithered(image, state);
        // Where encodeForDisplay is the whole CPU path (sRGB, no view or invert), its own bytes are the codes compared.
        if (state.lut == Lut::SRGB && state.channelView == 0 && !state.invert) {
            std::vector<unsigned char> api(image.texels.size());
            pathtracer::gfx::encodeForDisplay(image.texels, kWidth, kHeight, state.gain, true, state.offset, std::span<unsigned char>(api));
            bool same = true;
            for (std::size_t j = 0; j < api.size(); ++j) {
                same = same && api[j] == static_cast<unsigned char>(std::lround(cpu[j] * 255.0F));
            }
            PT_EXPECT(ctx, same, "state " + std::to_string(i) + ": the composed CPU stages differ from encodeForDisplay");
        }
        double worstRatio = 0.0;
        std::size_t mismatched = 0;
        std::size_t straddling = 0;
        std::size_t differing = 0;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const std::size_t pixel = (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x);
                const float* source = image.texels.data() + (pixel * 3);
                const glm::vec3 sampled(stored(source[0], format), stored(source[1], format), stored(source[2], format));
                const LaneBound reference = referenceTexel(sampled, state, curve, ditherAt(x, y));
                const std::size_t at = drawnTexel(kRect, kTargetWidth, x, y) * 4;
                for (std::size_t lane = 0; lane < 3; ++lane) {
                    const double gpu = static_cast<double>(drawn[at + lane]);
                    const double error = std::abs(gpu - reference.value[lane]);
                    worstRatio = std::max(worstRatio, error / reference.error[lane]);
                    // The GPU's value lies within its bound of the reference; the CPU code came from its own value, so their hull decides.
                    const double cpuValue = static_cast<double>(cpu[(pixel * 3) + lane]);
                    const long cpuCode = std::lround(cpu[(pixel * 3) + lane] * 255.0F);
                    const CodeRange range = codesOf(std::min(reference.value[lane] - reference.error[lane], cpuValue),
                                                    std::max(reference.value[lane] + reference.error[lane], cpuValue));
                    const long gpuCode = codes[at + lane];
                    straddling += range.lo != range.hi ? 1 : 0;
                    differing += gpuCode != cpuCode ? 1 : 0;
                    mismatched += (gpuCode >= range.lo && gpuCode <= range.hi && range.hi - range.lo <= 1) ? 0 : 1;
                }
            }
        }
        const std::string label = "state " + std::to_string(i);
        std::printf("    %s: worst |GPU - exact| / bound %.3g, %zu of %zu codes differ from the CPU, %zu straddle a code boundary\n",
                    label.c_str(), worstRatio, differing, image.texels.size(), straddling);
        PT_EXPECT(ctx, worstRatio <= 1.0, label + ": GPU exceeds its GLSL error bound by " + std::to_string(worstRatio) + "x");
        PT_EXPECT(ctx, mismatched == 0, label + ": " + std::to_string(mismatched) + " codes outside the CPU/GPU rounding hull");
    }
}

// The fringe's cell at a film point, or every cell a fixed-function rounding of eta could make it: Voronoi distances move by at most eta.
std::vector<glm::ivec2> fringeCells(PixelLattice lattice, glm::vec2 film, int width, int height, bool wraps, double eta) {
    const glm::ivec2 nearest = pathtracer::gfx::nearestPixel(lattice, film, width, height, wraps);
    // nearestPixel ranks its clamped candidates by their true distance to the film point, so the same ranking decides reach here.
    const auto distance = [&](glm::ivec2 cell) {
        return static_cast<double>(glm::length(pathtracer::gfx::pixelCentre(lattice, cell.x, cell.y) - film));
    };
    std::vector<glm::ivec2> cells{nearest};
    const double reach = distance(nearest) + (2.0 * eta);
    // Two rows either side: beyond a clamped edge the cells tied with the nearest can sit two rows from it, as in rows 7 and 9 off row 8.
    for (int row = std::max(0, nearest.y - 2); row <= std::min(height - 1, nearest.y + 2); ++row) {
        for (int column = nearest.x - 2; column <= nearest.x + 2; ++column) {
            const glm::ivec2 cell(wraps ? ((column % width) + width) % width : std::clamp(column, 0, width - 1), row);
            if (cell != nearest && distance(cell) <= reach && std::find(cells.begin(), cells.end(), cell) == cells.end()) {
                cells.push_back(cell);
            }
        }
    }
    return cells;
}

// Bloom through the real pyramid against applyDisplayOptics at GL's stated precision; the fringe's cells against nearestPixel.
PT_CHECK(display_optics_match_the_cpu, Fast, Exact) {
    if (glContext() == nullptr) {
        ctx.skip("no GL context on this host");
        return;
    }
    std::optional<DisplayPath> path = makeDisplayPath();
    struct Fringe {
        PixelLattice lattice;
        bool wraps;
    };
    constexpr std::array<Fringe, 4> kFringes{{{PixelLattice::Square, false},
                                              {PixelLattice::Square, true},
                                              {PixelLattice::Hexagonal, false},
                                              {PixelLattice::Hexagonal, true}}};
    ctx.plan(1 + 2 + static_cast<int>(kFringes.size()) + 1);
    PT_EXPECT(ctx, path.has_value(), "the display shaders did not build");
    if (!path) {
        return;
    }
    pathtracer::scene::ThreadPool pool(static_cast<unsigned int>(ctx.threads()));

    // Powers of two, so every GL_LINEAR fetch lands on a dyadic fraction (1/4, 1/2, 3/4) and the coordinates themselves are exact.
    constexpr int kWidth = 256;
    constexpr int kHeight = 128;
    constexpr float kStrength = 0.5F;
    HdrImage glow = testImage(kWidth, kHeight, ctx.subSeed("bloom"));
    for (float& value : glow.texels) {
        value = std::abs(value);
    }
    const int octaves = pathtracer::gfx::bloomOctaves(kWidth, kHeight);
    // Positive convex sums keep relative error: per pass N fetches of 3 lerps (4x: a dyadic weight >= 1/4) plus N+2 shader roundings.
    const auto pyramid = [&](double perOp) {
        const double down = (12.0 * perOp) + gamma(static_cast<double>(pathtracer::gfx::kBloomDownTaps.size()) + 2.0);
        const double up = (12.0 * perOp) + gamma(static_cast<double>(pathtracer::gfx::kBloomUpTaps.size()) + 2.0);
        return octaves * (down + up);
    };
    const double relative = pyramid(kFixedFunction) + (12.0 * kFixedFunction) + pyramid(kUnitRoundoff);
    for (const bool wraps : {false, true}) {
        HdrImage expected = glow;
        pathtracer::gfx::applyDisplayOptics(expected, PixelLattice::Square, wraps, {.aberration = 0.0F, .bloomStrength = kStrength}, pool);
        pathtracer::gfx::Texture texture = pathtracer::gfx::Texture::createFromFloatPixels(kWidth, kHeight, 3, glow.texels.data(), ScalarType::Float32);
        texture.setWrapsHorizontally(wraps);
        const RenderTarget target(kWidth, kHeight, GL_RGBA32F);
        const ViewportRect rect{0, 0, kWidth, kHeight};
        DisplayState state;
        state.lut = Lut::Raw;
        state.bloomStrength = kStrength;
        state.wrapsHorizontally = wraps;
        present(*path, texture, kWidth, kHeight, ScalarType::Float32, state, rect, target);
        const std::vector<float> drawn = target.read<float>(GL_RGBA, 4);
        double worstRatio = 0.0;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const std::size_t pixel = (static_cast<std::size_t>(y) * kWidth) + static_cast<std::size_t>(x);
                for (std::size_t lane = 0; lane < 3; ++lane) {
                    const double source = glow.texels[(pixel * 3) + lane];
                    const double cpu = expected.texels[(pixel * 3) + lane];
                    const double bloom = (cpu - ((1.0 - kStrength) * source)) / kStrength;
                    const double value = cpu + ditherAt(x, y);
                    // The pyramid's relative error on the glare term, then mix's three roundings on both terms and the dither add.
                    const double bound = (kStrength * bloom * relative) + (2.0 * gamma(3.0) * ((1.0 - kStrength) * source + kStrength * bloom)) +
                                         gamma(2.0) * std::abs(value);
                    worstRatio = std::max(worstRatio, std::abs(static_cast<double>(drawn[(drawnTexel(rect, kWidth, x, y) * 4) + lane]) - value) / bound);
                }
            }
        }
        std::printf("    bloom %s: worst |GPU - CPU| / bound %.3g\n", wraps ? "wrapped" : "clamped", worstRatio);
        PT_EXPECT(ctx, worstRatio <= 1.0, std::string("bloom ") + (wraps ? "wrapped" : "clamped") + " exceeds its bound by " + std::to_string(worstRatio) + "x");
    }

    // Not a short decimal: 0.3 lands whole columns exactly on a cell edge, where GL's own rounding is unspecified.
    constexpr float kAberration = 0.2718282F;
    constexpr int kFringeWidth = 29;
    constexpr int kFringeHeight = 17;
    const HdrImage fringe = testImage(kFringeWidth, kFringeHeight, ctx.subSeed("fringe"));
    const auto lane = [&](glm::ivec2 cell, int channel) {
        return fringe.texels[(((static_cast<std::size_t>(cell.y) * kFringeWidth) + static_cast<std::size_t>(cell.x)) * 3) + static_cast<std::size_t>(channel)];
    };
    for (const Fringe& testCase : kFringes) {
        pathtracer::gfx::Texture texture = pathtracer::gfx::Texture::createFromFloatPixels(kFringeWidth, kFringeHeight, 3, fringe.texels.data(), ScalarType::Float32);
        texture.setWrapsHorizontally(testCase.wraps);
        // Magnified and non-integral, so fragments sample between cells and latticeTexel's rows, offsets and wrap are all exercised.
        constexpr ViewportRect kRect{0, 0, 97, 61};
        const RenderTarget target(kRect.width, kRect.height, GL_RGBA32F);
        DisplayState state;
        state.lut = Lut::Raw;
        state.aberration = kAberration;
        state.lattice = testCase.lattice;
        state.wrapsHorizontally = testCase.wraps;
        present(*path, texture, kFringeWidth, kFringeHeight, ScalarType::Float32, state, kRect, target);
        const std::vector<float> drawn = target.read<float>(GL_RGBA, 4);
        const glm::vec2 frame = pathtracer::gfx::frameExtent(testCase.lattice, kFringeWidth, kFringeHeight);
        // Interpolation and the sampler's scale at GL's 1e-5 each, the shader's shift in three correctly rounded steps, then x frame.
        const double eta = ((2.0 * kFixedFunction) + (5.0 * kUnitRoundoff)) * (1.0 + kAberration) * std::max(frame.x, frame.y);
        std::size_t wrong = 0;
        for (int y = 0; y < kRect.height; ++y) {
            for (int x = 0; x < kRect.width; ++x) {
                const glm::vec2 uv((static_cast<float>(x) + 0.5F) / static_cast<float>(kRect.width),
                                   (static_cast<float>(y) + 0.5F) / static_cast<float>(kRect.height));
                const glm::vec2 shift = (uv - 0.5F) * kAberration;
                const std::size_t at = drawnTexel(kRect, kRect.width, x, y) * 4;
                const std::array<glm::vec2, 3> points{uv - shift, uv, uv + shift};
                for (int channel = 0; channel < 3; ++channel) {
                    const std::vector<glm::ivec2> cells = fringeCells(testCase.lattice, points[static_cast<std::size_t>(channel)] * frame,
                                                                      kFringeWidth, kFringeHeight, testCase.wraps, eta);
                    const float gpu = drawn[at + static_cast<std::size_t>(channel)];
                    const bool any = std::any_of(cells.begin(), cells.end(), [&](glm::ivec2 cell) { return gpu == lane(cell, channel) + ditherAt(x, y); });
                    wrong += any ? 0 : 1;
                }
            }
        }
        PT_EXPECT(ctx, wrong == 0,
                  std::string(pathtracer::gfx::kPixelLatticeNames[static_cast<int>(testCase.lattice)]) +
                      (testCase.wraps ? " wrapped: " : " clamped: ") + std::to_string(wrong) + " fringe lanes read a cell nearestPixel cannot reach");
    }

    // At 1:1 on the square grid each fragment is a pixel centre, so the GPU frame is applyDisplayOptics' output itself.
    HdrImage expected = fringe;
    pathtracer::gfx::applyDisplayOptics(expected, PixelLattice::Square, false, {.aberration = kAberration, .bloomStrength = 0.0F}, pool);
    pathtracer::gfx::Texture texture = pathtracer::gfx::Texture::createFromFloatPixels(kFringeWidth, kFringeHeight, 3, fringe.texels.data(), ScalarType::Float32);
    const ViewportRect rect{0, 0, kFringeWidth, kFringeHeight};
    const RenderTarget target(kFringeWidth, kFringeHeight, GL_RGBA32F);
    DisplayState state;
    state.lut = Lut::Raw;
    state.aberration = kAberration;
    present(*path, texture, kFringeWidth, kFringeHeight, ScalarType::Float32, state, rect, target);
    const std::vector<float> drawn = target.read<float>(GL_RGBA, 4);
    std::size_t differing = 0;
    for (int y = 0; y < kFringeHeight; ++y) {
        for (int x = 0; x < kFringeWidth; ++x) {
            for (std::size_t c = 0; c < 3; ++c) {
                const float cpu = expected.texels[(((static_cast<std::size_t>(y) * kFringeWidth) + static_cast<std::size_t>(x)) * 3) + c];
                differing += drawn[(drawnTexel(rect, kFringeWidth, x, y) * 4) + c] == cpu + ditherAt(x, y) ? 0 : 1;
            }
        }
    }
    PT_EXPECT(ctx, differing == 0, std::to_string(differing) + " fringe lanes differ from applyDisplayOptics at 1:1");
}

// The HUD's histogram of a drawn frame: the GPU's blit, readback and binning against binPixels of the same frame's pixels.
PT_CHECK(histogram_bins_the_drawn_frame, Fast, Exact) {
    if (glContext() == nullptr) {
        ctx.skip("no GL context on this host");
        return;
    }
    std::optional<DisplayPath> path = makeDisplayPath();
    ctx.plan(4);
    PT_EXPECT(ctx, path.has_value(), "the display shaders did not build");
    if (!path) {
        return;
    }
    using pathtracer::debug::Histogram;
    // Exactly the capture's size, so the GL_LINEAR blit samples every texel at its centre and copies it unfiltered.
    const HdrImage image = testImage(Histogram::kWidth, Histogram::kHeight, ctx.seed());
    const pathtracer::gfx::Texture texture =
        pathtracer::gfx::Texture::createFromFloatPixels(image.width, image.height, 3, image.texels.data(), ScalarType::Float32);
    const RenderTarget target(image.width, image.height, GL_RGBA8);
    const ViewportRect rect{0, 0, image.width, image.height};
    present(*path, texture, image.width, image.height, ScalarType::Float32, DisplayState{}, rect, target);
    Histogram histogram;
    // A capture every kCaptureIntervalFrames, binned one capture later from the other pixel buffer: two intervals fill it.
    for (int frame = 0; frame < 2 * Histogram::kCaptureIntervalFrames; ++frame) {
        histogram.update(rect, target.fbo());
    }
    glFinish();
    PT_EXPECT(ctx, histogram.hasData(), "two capture intervals produced no bins");
    pathtracer::debug::HistogramBins expected{};
    pathtracer::debug::binPixels(target.read<unsigned char>(GL_RGB, 3), expected);
    PT_EXPECT(ctx, histogram.bins() == expected, "the GPU path's bins differ from binPixels of the drawn frame");
    std::uint64_t counted = 0;
    for (const auto& channel : histogram.bins()) {
        for (const std::uint32_t count : channel) {
            counted += count;
        }
    }
    PT_EXPECT(ctx, counted == 3ULL * Histogram::kWidth * Histogram::kHeight, "the bins do not hold every pixel once per channel");
}

}  // namespace

PT_CHECK_MAIN("display_gl")
