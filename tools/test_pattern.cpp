#include <OpenEXR/ImfRgbaFile.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kPatchWidth = 100;
constexpr int kPatchCount = 7;
constexpr int kWidth = kPatchWidth * kPatchCount;  // 700
constexpr int kHeight = 100;
constexpr float kGrey18 = 0.18F;

struct Rgb {
    float r;
    float g;
    float b;
};

// Column-uniform by design: identical across every row, so it stays immune to an undetected vertical flip in EXR/OpenGL row handling.
Rgb colorForColumn(int x) {
    const int patch = x / kPatchWidth;
    switch (patch) {
        case 0:
            return {0.0F, 0.0F, 0.0F};  // black
        case 1:
            return {kGrey18, kGrey18, kGrey18};  // 18% grey
        case 2:
            return {1.0F, 1.0F, 1.0F};  // white
        case 3:
            return {1.0F, 0.0F, 0.0F};  // red
        case 4:
            return {0.0F, 1.0F, 0.0F};  // green
        case 5:
            return {0.0F, 0.0F, 1.0F};  // blue
        default: {
            const int local = x - patch * kPatchWidth;
            const float t = static_cast<float>(local) / static_cast<float>(kPatchWidth - 1);
            return {t, t, t};  // grayscale ramp, exact 0..1 endpoints
        }
    }
}

}  // namespace

int main() {
    const std::string outputPath = ASSET_ROOT_DIR "/textures/test_pattern.exr";

    std::vector<Imf::Rgba> pixels(static_cast<std::size_t>(kWidth) *
                                   static_cast<std::size_t>(kHeight));
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const Rgb c = colorForColumn(x);
            pixels[static_cast<std::size_t>(y) * kWidth + x] = Imf::Rgba(c.r, c.g, c.b);
        }
    }

    try {
        // WRITE_RGB: every engine loader reads R, G and B alone, so an alpha plane would be written and never read.
        Imf::RgbaOutputFile file(outputPath.c_str(), kWidth, kHeight, Imf::WRITE_RGB);
        file.setFrameBuffer(pixels.data(), 1, kWidth);
        file.writePixels(kHeight);
    } catch (const std::exception& e) {
        std::cerr << "test_pattern: failed to write " << outputPath << ": " << e.what()
                  << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_pattern: wrote " << outputPath << " (" << kWidth << "x" << kHeight
               << ")\n";
    return EXIT_SUCCESS;
}
