// Writes the ColorChecker Classic as a linear Rec.709 albedo EXR from BabelColor's averaged spectral reflectances.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "colorchecker.h"
#include "pathtracer/gfx/hdr_image.h"

namespace {

using tools::colorchecker::kColumns;
using tools::colorchecker::kRows;
using tools::colorchecker::Patch;

// Uniform patchPx-square blocks, row 0 at the top as glTF UVs read it: bilinear sampling is exact everywhere but each seam texel pair.
pathtracer::gfx::HdrImage makeChart(const std::vector<Patch>& patches, int patchPx) {
    const int width = kColumns * patchPx;
    const int height = kRows * patchPx;
    pathtracer::gfx::HdrImage image = pathtracer::gfx::makeImage(width, height, pathtracer::gfx::kRgbChannels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const glm::dvec3& rgb = patches[static_cast<std::size_t>(((y / patchPx) * kColumns) + (x / patchPx))].rec709;
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

    const std::string dataPath = TOOLS_DATA_DIR "/colorchecker_babelcolor_average.csv";
    const auto patches = tools::colorchecker::loadColorChecker(dataPath);
    if (!patches) {
        return EXIT_FAILURE;
    }
    if (!pathtracer::gfx::writeExr(outPath, makeChart(*patches, patchPx))) {
        std::cerr << "colorchecker_texture: failed to write " << outPath << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "colorchecker_texture: wrote " << outPath << " (" << kColumns * patchPx << "x" << kRows * patchPx
              << ", linear Rec.709, D65, CIE 1931 2-degree)\n";
    for (const Patch& patch : *patches) {
        std::printf("  %-22s %.6f %.6f %.6f\n", patch.name.c_str(), patch.rec709.r, patch.rec709.g, patch.rec709.b);
    }
    return EXIT_SUCCESS;
}
