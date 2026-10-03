// Correctness gate for the exported C ABI: links only libpathtracer_c and calls nothing but pathtracer_c.h, as a foreign caller does.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "check.h"
#include "pathtracer/api/pathtracer_c.h"

extern "C" int ptHeaderAbiVersion(void);

namespace {

constexpr const char* kScene = "scenes/cornell.json";
// Small enough that a full request of every AOV stays a fast check; odd width so no row is a multiple of a vector width.
constexpr int kWidth = 17;
constexpr int kHeight = 9;
constexpr int kErrorCapacity = 512;
// Trailing guard floats per output buffer: a quiet NaN with a payload no renderer computes, compared bit for bit.
constexpr std::size_t kGuardCount = 64;
constexpr std::uint32_t kGuardBits = 0x7FC0BEEFU;
constexpr unsigned char kGuardByte = 0x5A;

// One scene per check, closed on scope exit: pt_renderer_close is the ABI's ownership contract.
struct OpenRenderer {
    PtRenderer* handle = nullptr;
    std::string error;

    OpenRenderer() {
        std::vector<char> err(kErrorCapacity, '\0');
        handle = pt_renderer_open(nullptr, kScene, err.data(), kErrorCapacity);
        error = err.data();
    }
    ~OpenRenderer() { pt_renderer_close(handle); }
    OpenRenderer(const OpenRenderer&) = delete;
    OpenRenderer& operator=(const OpenRenderer&) = delete;
    OpenRenderer(OpenRenderer&&) = delete;
    OpenRenderer& operator=(OpenRenderer&&) = delete;
};

// One AOV's output with kGuardCount guard floats past the extent pt_render may write.
struct GuardedBuffer {
    std::size_t extent;
    std::vector<float> storage;

    explicit GuardedBuffer(std::size_t floats) : extent(floats), storage(floats + kGuardCount, std::bit_cast<float>(kGuardBits)) {}

    [[nodiscard]] bool untouched(std::size_t from) const {
        return std::all_of(storage.begin() + static_cast<std::ptrdiff_t>(from), storage.end(),
                           [](float value) { return std::bit_cast<std::uint32_t>(value) == kGuardBits; });
    }
    [[nodiscard]] bool guardsIntact() const { return untouched(extent); }
    [[nodiscard]] bool allUntouched() const { return untouched(0); }
};

[[nodiscard]] PtRenderRequest makeRequest(const PtRenderer* renderer, const std::vector<int>& aovs, unsigned int seed) {
    PtRenderRequest request{};
    pt_renderer_default_camera(renderer, &request.camera);
    request.previous_camera = nullptr;
    request.width = kWidth;
    request.height = kHeight;
    request.samples = 2;
    request.seed = seed;
    request.aovs = aovs.data();
    request.aov_count = static_cast<int>(aovs.size());
    request.show_sky = PT_DEFAULT;
    request.env_light_enabled = PT_DEFAULT;
    return request;
}

[[nodiscard]] std::vector<GuardedBuffer> makeOutputs(const std::vector<int>& aovs) {
    std::vector<GuardedBuffer> outputs;
    outputs.reserve(aovs.size());
    for (const int aov : aovs) {
        outputs.emplace_back(static_cast<std::size_t>(kWidth) * kHeight * static_cast<std::size_t>(pt_aov_channels(aov)));
    }
    return outputs;
}

[[nodiscard]] std::vector<float*> pointersTo(std::vector<GuardedBuffer>& outputs) {
    std::vector<float*> pointers;
    pointers.reserve(outputs.size());
    for (GuardedBuffer& output : outputs) {
        pointers.push_back(output.storage.data());
    }
    return pointers;
}

[[nodiscard]] std::vector<int> everyAov() {
    std::vector<int> aovs(static_cast<std::size_t>(pt_aov_count()));
    for (std::size_t i = 0; i < aovs.size(); ++i) {
        aovs[i] = static_cast<int>(i);
    }
    return aovs;
}

[[nodiscard]] int beautyId() {
    return pt_aov_id("beauty");
}

// The macro a C caller compiles against and the value the loaded library reports must agree, or by-value structs are misread.
PT_CHECK(abi_version_matches_the_c_header, Fast, Exact) {
    ctx.plan(1);
    PT_EXPECT(ctx, pt_abi_version() == ptHeaderAbiVersion(),
              "library reports " + std::to_string(pt_abi_version()) + ", C header carries " + std::to_string(ptHeaderAbiVersion()));
}

// Ids are dense in [0, pt_aov_count): every name resolves back to its id, and anything outside answers with the documented sentinel.
PT_CHECK(aov_table_round_trips, Fast, Exact) {
    ctx.plan(3);
    const int count = pt_aov_count();
    std::string failures;
    for (int aov = 0; aov < count; ++aov) {
        const char* name = pt_aov_name(aov);
        const int channels = pt_aov_channels(aov);
        const int needsSamples = pt_aov_needs_samples(aov);
        if (name == nullptr || pt_aov_id(name) != aov || channels < 1 || channels > 3 || (needsSamples != 0 && needsSamples != 1)) {
            failures += " " + std::to_string(aov);
        }
    }
    PT_EXPECT(ctx, count > 0 && failures.empty(), "ids failing name, round trip, channel or needs-samples:" + failures);
    PT_EXPECT(ctx, pt_aov_name(-1) == nullptr && pt_aov_name(count) == nullptr && pt_aov_channels(-1) == -1 &&
                       pt_aov_channels(count) == -1 && pt_aov_needs_samples(-1) == 0 && pt_aov_needs_samples(count) == 0,
              "an out-of-range id must answer NULL, -1 and 0");
    PT_EXPECT(ctx, pt_aov_id(nullptr) == -1 && pt_aov_id("") == -1 && pt_aov_id("not an aov") == -1,
              "an unknown or null name must answer -1");
}

PT_CHECK(open_reports_null_and_missing_scenes, Fast, Exact) {
    ctx.plan(3);
    std::vector<char> err(kErrorCapacity, '\0');
    PT_EXPECT(ctx, pt_renderer_open(nullptr, nullptr, err.data(), kErrorCapacity) == nullptr && err[0] != '\0',
              "a null scene path must fail with a reason");
    err.assign(kErrorCapacity, '\0');
    PT_EXPECT(ctx, pt_renderer_open(nullptr, "scenes/missing.json", err.data(), kErrorCapacity) == nullptr && err[0] != '\0',
              "a missing scene must fail with a reason");
    PT_EXPECT(ctx, pt_renderer_open(nullptr, "scenes/missing.json", nullptr, 0) == nullptr &&
                       pt_renderer_open(nullptr, "scenes/missing.json", err.data(), 0) == nullptr,
              "a null or zero-capacity error buffer must still fail cleanly");
}

// err_cap bounds every write: the reason is truncated to err_cap - 1 bytes plus its NUL, and nothing past err_cap is touched.
PT_CHECK(error_buffer_is_truncated_and_terminated, Fast, Exact) {
    constexpr std::array<int, 3> kCapacities{1, 2, 8};
    ctx.plan(static_cast<int>(kCapacities.size()));
    for (const int capacity : kCapacities) {
        std::vector<char> err(static_cast<std::size_t>(capacity) + kGuardCount, static_cast<char>(kGuardByte));
        const PtRenderer* renderer = pt_renderer_open(nullptr, nullptr, err.data(), capacity);
        const std::size_t length = std::strlen(err.data());
        const bool guarded = std::all_of(err.begin() + capacity, err.end(),
                                         [](char byte) { return static_cast<unsigned char>(byte) == kGuardByte; });
        PT_EXPECT(ctx, renderer == nullptr && length == static_cast<std::size_t>(capacity - 1) && guarded,
                  "err_cap " + std::to_string(capacity) + ": reason length " + std::to_string(length) +
                      (guarded ? "" : ", wrote past err_cap"));
    }
}

// Every AOV in one request: each buffer is written to exactly width * height * channels floats, all finite, and not one float past.
PT_CHECK(render_writes_exactly_the_declared_extent, Slow, Exact) {
    ctx.plan(3);
    const OpenRenderer renderer;
    if (renderer.handle == nullptr) {
        PT_EXPECT(ctx, false, "scene load failed: " + renderer.error);
        PT_EXPECT(ctx, false, "skipped");
        PT_EXPECT(ctx, false, "skipped");
        return;
    }
    const std::vector<int> aovs = everyAov();
    std::vector<GuardedBuffer> outputs = makeOutputs(aovs);
    const std::vector<float*> pointers = pointersTo(outputs);
    const PtRenderRequest request = makeRequest(renderer.handle, aovs, 1);
    std::vector<char> err(kErrorCapacity, '\0');
    PT_EXPECT(ctx, pt_render(renderer.handle, &request, pointers.data(), err.data(), kErrorCapacity) == PT_OK,
              std::string("render failed: ") + err.data());
    std::string overrun;
    std::string nonFinite;
    for (std::size_t i = 0; i < aovs.size(); ++i) {
        const GuardedBuffer& output = outputs[i];
        if (!output.guardsIntact()) {
            overrun += std::string(" ") + pt_aov_name(aovs[i]);
        }
        if (!std::all_of(output.storage.begin(), output.storage.begin() + static_cast<std::ptrdiff_t>(output.extent),
                         [](float value) { return std::isfinite(value); })) {
            nonFinite += std::string(" ") + pt_aov_name(aovs[i]);
        }
    }
    PT_EXPECT(ctx, overrun.empty(), "wrote past the declared extent:" + overrun);
    PT_EXPECT(ctx, nonFinite.empty(), "left a non-finite or unwritten texel:" + nonFinite);
}

// A rejected request returns PT_ERROR naming its field and writes no output float, so a caller's buffers stay what it put there.
PT_CHECK(render_rejects_invalid_requests_without_writing, Slow, Exact) {
    struct Case {
        const char* label;
        const char* reasonFragment;
        void (*mutate)(PtRenderRequest&, std::vector<float*>&);
    };
    const std::array<Case, 8> cases{{
        {"aov_count 0", "no AOVs", [](PtRenderRequest& r, std::vector<float*>&) { r.aov_count = 0; }},
        {"null aovs", "no AOVs", [](PtRenderRequest& r, std::vector<float*>&) { r.aovs = nullptr; }},
        {"show_sky 2", "show_sky", [](PtRenderRequest& r, std::vector<float*>&) { r.show_sky = 2; }},
        {"env_light_enabled 2", "env_light_enabled", [](PtRenderRequest& r, std::vector<float*>&) { r.env_light_enabled = 2; }},
        {"lens_projection 7", "lens_projection", [](PtRenderRequest& r, std::vector<float*>&) { r.camera.lens_projection = 7; }},
        {"null out[1]", "output buffer 1", [](PtRenderRequest&, std::vector<float*>& p) { p[1] = nullptr; }},
        {"width 0", "resolution", [](PtRenderRequest& r, std::vector<float*>&) { r.width = 0; }},
        {"samples 0", "samples", [](PtRenderRequest& r, std::vector<float*>&) { r.samples = 0; }},
    }};
    ctx.plan(static_cast<int>(cases.size()) + 2);
    const OpenRenderer renderer;
    if (renderer.handle == nullptr) {
        for (std::size_t i = 0; i < cases.size() + 2; ++i) {
            PT_EXPECT(ctx, false, "scene load failed: " + renderer.error);
        }
        return;
    }
    const std::vector<int> aovs{beautyId(), pt_aov_id("depth")};
    for (const Case& testCase : cases) {
        std::vector<GuardedBuffer> outputs = makeOutputs(aovs);
        std::vector<float*> pointers = pointersTo(outputs);
        PtRenderRequest request = makeRequest(renderer.handle, aovs, 1);
        testCase.mutate(request, pointers);
        std::vector<char> err(kErrorCapacity, '\0');
        const int status = pt_render(renderer.handle, &request, pointers.data(), err.data(), kErrorCapacity);
        const bool named = err[0] != '\0' && std::string_view(err.data()).find(testCase.reasonFragment) != std::string_view::npos;
        const bool untouched = std::all_of(outputs.begin(), outputs.end(), [](const GuardedBuffer& b) { return b.allUntouched(); });
        PT_EXPECT(ctx, status == PT_ERROR && named && untouched,
                  std::string(testCase.label) + ": status " + std::to_string(status) + ", reason '" + err.data() + "'" +
                      (untouched ? "" : ", wrote an output"));
    }
    const std::vector<int> beauty{beautyId()};
    std::vector<GuardedBuffer> outputs = makeOutputs(beauty);
    const std::vector<float*> pointers = pointersTo(outputs);
    const PtRenderRequest request = makeRequest(renderer.handle, beauty, 1);
    std::vector<char> err(kErrorCapacity, '\0');
    PT_EXPECT(ctx, pt_render(nullptr, &request, pointers.data(), err.data(), kErrorCapacity) == PT_ERROR && err[0] != '\0',
              "a null renderer must fail with a reason");
    err.assign(kErrorCapacity, '\0');
    PT_EXPECT(ctx, pt_render(renderer.handle, nullptr, pointers.data(), err.data(), kErrorCapacity) == PT_ERROR &&
                       pt_render(renderer.handle, &request, nullptr, err.data(), kErrorCapacity) == PT_ERROR && err[0] != '\0',
              "a null request or output array must fail with a reason");
}

// The seed is the sampler's scramble: the same request reproduces bit for bit, and another seed draws another estimate.
PT_CHECK(render_is_deterministic_and_seed_sensitive, Slow, Exact) {
    ctx.plan(2);
    const OpenRenderer renderer;
    if (renderer.handle == nullptr) {
        PT_EXPECT(ctx, false, "scene load failed: " + renderer.error);
        PT_EXPECT(ctx, false, "skipped");
        return;
    }
    const std::vector<int> beauty{beautyId()};
    int failedRenders = 0;
    const auto render = [&](unsigned int seed) {
        std::vector<GuardedBuffer> outputs = makeOutputs(beauty);
        const std::vector<float*> pointers = pointersTo(outputs);
        const PtRenderRequest request = makeRequest(renderer.handle, beauty, seed);
        failedRenders += pt_render(renderer.handle, &request, pointers.data(), nullptr, 0) == PT_OK ? 0 : 1;
        return outputs.front().storage;
    };
    const std::vector<float> first = render(1);
    const std::vector<float> repeat = render(1);
    PT_EXPECT(ctx, failedRenders == 0 && std::memcmp(first.data(), repeat.data(), first.size() * sizeof(float)) == 0,
              "the same seed must render and reproduce every float");
    PT_EXPECT(ctx, std::memcmp(first.data(), render(2).data(), first.size() * sizeof(float)) != 0,
              "a different seed must change the estimate");
}

// Every byte in [first, last) equals value.
[[nodiscard]] bool allBytes(std::vector<unsigned char>::const_iterator first, std::vector<unsigned char>::const_iterator last,
                            unsigned char value) {
    return std::all_of(first, last, [value](unsigned char byte) { return byte == value; });
}

// Writes exactly w*h*3 bytes, clamps both ends exactly, and rejects null buffers and empty extents without writing.
PT_CHECK(display_encode_boundary, Fast, Exact) {
    ctx.plan(4);
    constexpr int kW = 5;
    constexpr int kH = 3;
    constexpr auto kBytes = static_cast<std::ptrdiff_t>(kW * kH * 3);
    // Beyond the dither's one-code-value amplitude on either side, so the clamp alone decides the code.
    std::vector<float> rgb(static_cast<std::size_t>(kBytes), 4.0F);
    std::fill(rgb.begin(), rgb.begin() + (kBytes / 2), -1.0F);
    std::vector<unsigned char> out(static_cast<std::size_t>(kBytes) + kGuardCount, kGuardByte);
    std::vector<char> err(kErrorCapacity, '\0');
    PT_EXPECT(ctx, pt_display_encode(rgb.data(), kW, kH, 0.0F, 0, out.data(), err.data(), kErrorCapacity) == PT_OK,
              std::string("encode failed: ") + err.data());
    PT_EXPECT(ctx, allBytes(out.begin(), out.begin() + (kBytes / 2), 0) && allBytes(out.begin() + (kBytes / 2), out.begin() + kBytes, 255),
              "-1 must encode to 0 and 4 to 255 exactly");
    PT_EXPECT(ctx, allBytes(out.begin() + kBytes, out.end(), kGuardByte), "wrote past width * height * 3 bytes");
    std::vector<unsigned char> untouched(static_cast<std::size_t>(kBytes), kGuardByte);
    const auto encode = [&](const float* in, int width, int height, unsigned char* to) {
        return pt_display_encode(in, width, height, 0.0F, 1, to, err.data(), kErrorCapacity);
    };
    const bool rejected = encode(nullptr, kW, kH, untouched.data()) == PT_ERROR && encode(rgb.data(), kW, kH, nullptr) == PT_ERROR &&
                          encode(rgb.data(), 0, kH, untouched.data()) == PT_ERROR &&
                          encode(rgb.data(), kW, -1, untouched.data()) == PT_ERROR;
    PT_EXPECT(ctx, rejected && allBytes(untouched.begin(), untouched.end(), kGuardByte),
              "null buffers and non-positive extents must fail without writing");
}

// Every handle-taking query is defined on NULL, so a caller's failed open never turns a cleanup path into a crash.
PT_CHECK(close_and_queries_accept_null, Fast, Exact) {
    ctx.plan(2);
    pt_renderer_close(nullptr);
    PtCamera camera{};
    camera.focal_length_mm = 1.0F;
    pt_renderer_default_camera(nullptr, &camera);
    PT_EXPECT(ctx, camera.focal_length_mm == 1.0F, "a null renderer must leave the camera untouched");
    PT_EXPECT(ctx, pt_renderer_default_width(nullptr) == 0 && pt_renderer_default_height(nullptr) == 0,
              "a null renderer has no size");
}

}  // namespace

PT_CHECK_MAIN("c_abi")
