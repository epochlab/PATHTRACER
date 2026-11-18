// Writes the ColorChecker Classic as a scene-space (ACEScg) albedo EXR from BabelColor's averaged spectral reflectances.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "pathtracer/config/spectral_data.h"
#include "pathtracer/gfx/hdr_image.h"
#include "pathtracer/scene/cie.h"

namespace {

// The Classic chart's 6x4 layout, row-major from dark skin (top-left) to black (bottom-right), as colorchecker.json grids it.
constexpr int kColumns = 6;
constexpr int kRows = 4;

// Uniform patchPx-square blocks, row 0 at the top as glTF UVs read it: bilinear sampling is exact everywhere but each seam texel pair.
pathtracer::gfx::HdrImage makeChart(const std::vector<glm::dvec3>& patches, int patchPx) {
    const int width = kColumns * patchPx;
    const int height = kRows * patchPx;
    pathtracer::gfx::HdrImage image = pathtracer::gfx::makeImage(width, height, pathtracer::gfx::kRgbChannels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const glm::dvec3& rgb = patches[static_cast<std::size_t>(((y / patchPx) * kColumns) + (x / patchPx))];
            writeTexel(image, x, y, glm::vec3(rgb));
        }
    }
    return image;
}

}  // namespace

int main(int argc, char** argv) {
    std::string outPath = ASSET_ROOT_DIR "/textures/macbeth.exr";
    int patchPx = 64;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            outPath = argv[++i];
        } else if (std::strcmp(argv[i], "--patch-px") == 0 && i + 1 < argc) {
            patchPx = std::atoi(argv[++i]);
        } else {
            std::cerr << "usage: colorchecker_texture [--out PATH] [--patch-px N]\n";
            return EXIT_FAILURE;
        }
    }
    if (patchPx < 1) {
        std::cerr << "colorchecker_texture: --patch-px must be a positive integer\n";
        return EXIT_FAILURE;
    }

    const auto patches = pathtracer::config::loadChartCsv(ASSET_ROOT_DIR "/spectra/colorchecker_babelcolor_average.csv", "colorchecker_texture");
    if (!patches || patches->size() != static_cast<std::size_t>(kColumns * kRows)) {
        std::cerr << "colorchecker_texture: the chart needs " << kColumns * kRows << " patches\n";
        return EXIT_FAILURE;
    }
    std::vector<glm::dvec3> scene;
    for (const pathtracer::config::ChartPatch& patch : *patches) {
        scene.push_back(pathtracer::scene::cie::reflectanceToScene(patch.reflectance));
    }
    if (!pathtracer::gfx::writeExr(outPath, makeChart(scene, patchPx), pathtracer::gfx::ImageRole::Colour)) {
        std::cerr << "colorchecker_texture: failed to write " << outPath << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "colorchecker_texture: wrote " << outPath << " (" << kColumns * patchPx << "x" << kRows * patchPx
              << ", scene space, D65, CIE 1931 2-degree)\n";
    for (std::size_t i = 0; i < scene.size(); ++i) {
        std::printf("  %-22s %.6f %.6f %.6f\n", (*patches)[i].name.c_str(), scene[i].r, scene[i].g, scene[i].b);
    }
    return EXIT_SUCCESS;
}
