#include "pathtracer/debug/histogram_curve.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace pathtracer::debug {

namespace {

constexpr int kLastBin = Histogram::kBins - 1;

// 9-tap triangular smoother (radius 4) over bins in [1, 254]. Bins 0 and 255 are never read, so clipping spikes do not bleed in.
void smoothChannel(const std::array<std::uint32_t, Histogram::kBins>& channelBins, std::uint32_t peak, HistogramCurve& out) {
    for (int bin = 0; bin < Histogram::kBins; ++bin) {
        float sum = 0.0F;
        float weightSum = 0.0F;
        for (int offset = -4; offset <= 4; ++offset) {
            const int neighbor = bin + offset;
            if (neighbor < 1 || neighbor > kLastBin - 1) {
                continue;
            }
            const int distance = offset < 0 ? -offset : offset;
            const float weight = 5.0F - static_cast<float>(distance);
            const float value = static_cast<float>(std::min(channelBins[static_cast<std::size_t>(neighbor)], peak));
            sum += weight * std::sqrt(value / static_cast<float>(peak));
            weightSum += weight;
        }
        out[static_cast<std::size_t>(bin)] = weightSum > 0.0F ? sum / weightSum : 0.0F;
    }
}

// Rescales a smoothed curve so its own maximum reaches 1.0: smoothing can only pull values down from the raw peak, never up.
void rescaleToUnitPeak(HistogramCurve& curve) {
    const float maxValue = *std::max_element(curve.begin(), curve.end());
    if (maxValue == 0.0F) {
        return;
    }
    for (float& value : curve) {
        value /= maxValue;
    }
}

// One peak shared across active channels, not per channel, so a channel with more signal reads taller rather than all filling the same.
std::uint32_t sharedPeak(const HistogramBins& bins, const HistogramActive& active) {
    std::uint32_t peak = 1;
    for (std::size_t c = 0; c < 3; ++c) {
        if (active[c]) {
            peak = std::max(peak, *std::max_element(bins[c].begin() + 1, bins[c].end() - 1));
        }
    }
    return peak;
}

}  // namespace

void binPixels(std::span<const unsigned char> rgb, HistogramBins& bins) {
    for (auto& channel : bins) {
        channel.fill(0);
    }
    for (std::size_t i = 0; i < rgb.size(); i += 3) {
        ++bins[0][rgb[i]];
        ++bins[1][rgb[i + 1]];
        ++bins[2][rgb[i + 2]];
    }
}

bool isGrayscale(const HistogramBins& bins) {
    return bins[0] == bins[1] && bins[1] == bins[2];
}

HistogramActive activeChannels(const HistogramBins& bins) {
    HistogramActive active{};
    for (std::size_t c = 0; c < 3; ++c) {
        active[c] = std::any_of(bins[c].begin() + 1, bins[c].end(), [](std::uint32_t count) { return count > 0; });
    }
    return active;
}

HistogramCurve grayscaleHeights(const HistogramBins& bins) {
    std::uint64_t interiorTotal = 0;
    for (int bin = 1; bin < kLastBin; ++bin) {
        interiorTotal += bins[0][static_cast<std::size_t>(bin)];
    }
    // Under 1% of the capture's pixels in the interior means near-binary content, an alpha or depth mask.
    const bool nearBinary = interiorTotal < static_cast<std::uint64_t>(Histogram::kWidth * Histogram::kHeight / 100);
    HistogramCurve heights{};
    if (nearBinary) {
        const std::uint32_t peak = std::max(1U, *std::max_element(bins[0].begin(), bins[0].end()));
        for (std::size_t bin = 0; bin < heights.size(); ++bin) {
            heights[bin] = std::sqrt(static_cast<float>(bins[0][bin]) / static_cast<float>(peak));
        }
        return heights;
    }
    const std::uint32_t peak = std::max(1U, *std::max_element(bins[0].begin() + 1, bins[0].end() - 1));
    smoothChannel(bins[0], peak, heights);
    rescaleToUnitPeak(heights);
    return heights;
}

std::array<HistogramCurve, 3> rgbHeights(const HistogramBins& bins, const HistogramActive& active) {
    const std::uint32_t peak = sharedPeak(bins, active);
    std::array<HistogramCurve, 3> heights{};
    float maxValue = 0.0F;
    for (std::size_t c = 0; c < 3; ++c) {
        if (active[c]) {
            smoothChannel(bins[c], peak, heights[c]);
            maxValue = std::max(maxValue, *std::max_element(heights[c].begin(), heights[c].end()));
        }
    }
    // The same rescale across every active channel at once, so the tallest reaches full height and the others keep their ratio.
    if (maxValue == 0.0F) {
        return heights;
    }
    for (std::size_t c = 0; c < 3; ++c) {
        if (active[c]) {
            for (float& value : heights[c]) {
                value /= maxValue;
            }
        }
    }
    return heights;
}

HistogramCurve overlapHeights(const std::array<HistogramCurve, 3>& heights, const HistogramActive& active) {
    HistogramCurve overlap{};
    for (std::size_t bin = 0; bin < overlap.size(); ++bin) {
        float value = 1.0F;
        for (std::size_t c = 0; c < 3; ++c) {
            if (active[c]) {
                value = std::min(value, heights[c][bin]);
            }
        }
        overlap[bin] = value;
    }
    return overlap;
}

}  // namespace pathtracer::debug
