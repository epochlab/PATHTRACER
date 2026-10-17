#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "pathtracer/debug/histogram.h"

// The HUD histogram's binning and curve shapes, GL- and ImGui-free so the CPU can test exactly what the panel draws.
namespace pathtracer::debug {

using HistogramBins = std::array<std::array<std::uint32_t, Histogram::kBins>, 3>;
using HistogramCurve = std::array<float, Histogram::kBins>;
using HistogramActive = std::array<bool, 3>;

// Counts 8-bit RGB triples into per-channel bins, replacing bins' previous contents. rgb holds pixels*3 bytes.
void binPixels(std::span<const unsigned char> rgb, HistogramBins& bins);

// R, G and B bins equal: the panel collapses them to one curve.
[[nodiscard]] bool isGrayscale(const HistogramBins& bins);

// A structurally unused channel has every pixel at bin 0. Checked over bins 1-255 so a background-dominated channel is not mistaken for it.
[[nodiscard]] HistogramActive activeChannels(const HistogramBins& bins);

// The grey curve in [0, 1]: smoothed sqrt counts at unit peak, or under 1% interior pixels (a mask) the raw full-range sqrt, spikes kept.
[[nodiscard]] HistogramCurve grayscaleHeights(const HistogramBins& bins);

// Per active channel, smoothed sqrt counts over one shared peak, so the tallest reaches 1 and the others keep their relative height.
[[nodiscard]] std::array<HistogramCurve, 3> rgbHeights(const HistogramBins& bins, const HistogramActive& active);

// Per bin, the minimum over active channels' heights: where the curves overlap.
[[nodiscard]] HistogramCurve overlapHeights(const std::array<HistogramCurve, 3>& heights, const HistogramActive& active);

}  // namespace pathtracer::debug
