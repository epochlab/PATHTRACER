// Prints the base_color/specular_color a metal's material JSON carries for a measured conductor, at max_digits10 so pasting round-trips.

#include <cstdio>
#include <iostream>
#include <limits>
#include <string_view>

#include "metal_fit.h"

namespace {

using tools::metal_fit::Fit;
using tools::metal_fit::Interpolation;

void printFloatTriple(const char* key, const glm::dvec3& v) {
    constexpr int kDigits = std::numeric_limits<float>::max_digits10;
    std::printf("    \"%s\": [%.*g, %.*g, %.*g],\n", key, kDigits, static_cast<float>(v.x), kDigits,
                static_cast<float>(v.y), kDigits, static_cast<float>(v.z));
}

void printTriple(const char* label, const glm::dvec3& v) {
    std::printf("  %-44s %.3e %.3e %.3e\n", label, v.x, v.y, v.z);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string_view(argv[1]) != "--nk") {
        std::cerr << "usage: metal_fit --nk TABLE.csv   (rows lambda_um,n,k spanning 360-830 nm)\n";
        return 2;
    }
    const auto table = tools::metal_fit::loadNkTable(argv[2]);
    if (!table) {
        return 1;
    }
    const auto fit = tools::metal_fit::fitF82(*table, Interpolation::Wavelength);
    const auto energyFit = tools::metal_fit::fitF82(*table, Interpolation::PhotonEnergy);
    if (!fit || !energyFit) {
        return 1;
    }
    std::printf("metal_fit: %s (%zu rows) -> CIE 1931 2-degree observer, D65, linear Rec.709\n", argv[2],
                table->size());
    printFloatTriple("base_color", fit->baseColor);
    printFloatTriple("specular_color", fit->specularColor);
    printTriple("hemispherical average (measured)", fit->averageTarget);
    printTriple("|F82 average - measured|", fit->averageResidual);
    printTriple("max |R_fit(mu) - R_cie(mu)|", fit->angularResidual);
    printTriple("base_color delta, linear in photon energy", energyFit->baseColor - fit->baseColor);
    printTriple("specular_color delta, linear in photon energy", energyFit->specularColor - fit->specularColor);
    return 0;
}
