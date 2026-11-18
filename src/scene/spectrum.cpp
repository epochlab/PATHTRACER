#include "pathtracer/scene/spectrum.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <span>

#include "pathtracer/scene/rgb_spectrum_table.h"
#include "pathtracer/scene/sampler.h"
#include "rgb_spectrum_data.h"

namespace pathtracer::scene {

namespace {

constexpr int kBins = cie::kSampleCount - 1;  // 1 nm bins between the CIE samples

// The CIE 1931 observer through xyzToScene as the sensor, at the 1 nm samples, and its channel-sensitivity sampling density.
struct Sensor {
    std::array<glm::vec3, cie::kSampleCount> sensitivity;  // K_m xyzToScene cmf: scene RGB in cd/m^2 per W/(sr m^2 nm) over 1 nm
    std::array<float, cie::kSampleCount> white;            // D65 / (K_m sum D65 yBar), the scene white at unit luminance
    std::array<double, cie::kSampleCount> weight;  // sum_c |s_c|, the density's shape at each sample
    WavelengthPdf density;
};

// mass normalised to a density per 1 nm bin and its CDF.
WavelengthPdf densityOf(const std::array<double, kBins>& mass) {
    WavelengthPdf density{};
    double total = 0.0;
    for (const double m : mass) {
        total += m;
    }
    double running = 0.0;
    for (std::size_t b = 0; b < mass.size(); ++b) {
        density.cdf[b] = static_cast<float>(running / total);
        running += mass[b];
        density.pdf[b] = static_cast<float>(mass[b] / total);
    }
    density.cdf.back() = 1.0F;
    return density;
}

Sensor buildSensor() {
    Sensor sensor{};
    double d65Luminance = 0.0;
    for (int i = 0; i < cie::kSampleCount; ++i) {
        const cie::TableRow row = cie::tableRow(i);
        d65Luminance += row.d65 * row.colourMatching.y;
    }
    for (int i = 0; i < cie::kSampleCount; ++i) {
        const cie::TableRow row = cie::tableRow(i);
        const glm::dvec3 s = static_cast<double>(kLuminousEfficacy) * (cie::xyzToScene() * row.colourMatching);
        const auto at = static_cast<std::size_t>(i);
        sensor.sensitivity[at] = glm::vec3(s);
        sensor.white[at] = static_cast<float>(row.d65 / (static_cast<double>(kLuminousEfficacy) * d65Luminance));
        sensor.weight[at] = std::abs(s.x) + std::abs(s.y) + std::abs(s.z);
    }
    std::array<double, kBins> mass{};
    for (std::size_t b = 0; b < mass.size(); ++b) {
        mass[b] = 0.5 * (sensor.weight[b] + sensor.weight[b + 1]);
    }
    sensor.density = densityOf(mass);
    return sensor;
}

const Sensor& sensor() {
    static const Sensor kSensor = buildSensor();
    return kSensor;
}

// One float of a table, by memcpy: the embedded bytes are not float objects, so they are read, never reinterpreted.
float tableFloat(std::span<const unsigned char> bytes, std::size_t index) {
    float value = 0.0F;
    std::memcpy(&value, bytes.data() + (index * sizeof(float)), sizeof(float));
    return value;
}

// An unbounded colour's peak goes to scale 1/2, grey's flat zero-coefficient fit: smoothest there, a sub-nm-edged box at scale 1.
constexpr float kMidScale = 0.5F;

// sign(x) / sqrt(1 + 1/x^2) is x / sqrt(1 + x^2) without overflow: exactly +-1 at infinite x, 0 at 0, so s spans [0, 1].
template <typename T>
T sigmoid(const T& x) {
    return 0.5F + (0.5F * glm::sign(x) * glm::inversesqrt(1.0F + (1.0F / (x * x))));
}

}  // namespace

const WavelengthPdf& sensorWavelengthPdf() { return sensor().density; }

WavelengthPdf emitterWavelengthPdf(const cie::Spectrum& spectralFlux) {
    const Sensor& s = sensor();
    std::array<double, kBins> emitted{};
    double emittedTotal = 0.0;
    for (std::size_t b = 0; b < emitted.size(); ++b) {
        emitted[b] = 0.5 * ((s.weight[b] * spectralFlux[b]) + (s.weight[b + 1] * spectralFlux[b + 1]));
        emittedTotal += emitted[b];
    }
    // No area light the sensor sees: the sensor's own density, exactly.
    if (!(emittedTotal > 0.0)) {
        return s.density;
    }
    std::array<double, kBins> mixture{};
    for (std::size_t b = 0; b < mixture.size(); ++b) {
        mixture[b] = (0.5 * static_cast<double>(s.density.pdf[b])) + (0.5 * emitted[b] / emittedTotal);
    }
    return densityOf(mixture);
}

SampledWavelengths sampleWavelengths(float u, const WavelengthPdf& density) {
    const Sensor& s = sensor();
    SampledWavelengths lambda{};
    for (int j = 0; j < kSpectrumSamples; ++j) {
        float uj = u + (static_cast<float>(j) / kSpectrumSamples);
        uj = uj < 1.0F ? uj : uj - 1.0F;
        const CdfSample bin = invertCdf(density.cdf.data(), kBins, uj);
        const auto at = static_cast<std::size_t>(bin.index);
        const float pdf = density.pdf[at];
        const glm::vec3 sensitivity = glm::mix(s.sensitivity[at], s.sensitivity[at + 1], bin.fraction);
        lambda.nm[j] = static_cast<float>(cie::kLambdaMinNm + bin.index) + bin.fraction;
        lambda.pdf[j] = pdf;
        lambda.white[j] = glm::mix(s.white[at], s.white[at + 1], bin.fraction);
        lambda.toScene[j] = sensitivity / (kSpectrumSamples * pdf);
        lambda.reflectanceToScene[j] = lambda.toScene[j] * lambda.white[j];
    }
    return lambda;
}

SampledSpectrum RgbSigmoid::operator()(const SampledWavelengths& lambda) const {
    const glm::vec4 x = (((c.x * lambda.nm) + c.y) * lambda.nm) + c.z;
    return SampledSpectrum(sigmoid(x));
}

float RgbSigmoid::operator()(float nm) const { return sigmoid((((c.x * nm) + c.y) * nm) + c.z); }

RgbSigmoid rgbSigmoid(const Rgb& rgb) {
    namespace table = rgb_spectrum;
    // The brightest channel picks the table, ties to the later one (the paper's convention); the other two are its ratios.
    int l = 0;
    for (int i = 1; i < 3; ++i) {
        l = rgb[static_cast<std::size_t>(i)] >= rgb[static_cast<std::size_t>(l)] ? i : l;
    }
    const float z = rgb[static_cast<std::size_t>(l)];
    if (!(z > 0.0F)) {
        return {glm::vec3(0.0F, 0.0F, -std::numeric_limits<float>::infinity())};
    }
    const std::span<const unsigned char> bytes = rgbSpectrumTableBytes();
    const float toNode = static_cast<float>(table::kResolution - 1) / z;
    const float x = rgb[static_cast<std::size_t>((l + 1) % 3)] * toNode;
    const float y = rgb[static_cast<std::size_t>((l + 2) % 3)] * toNode;
    const int xi = std::min(static_cast<int>(x), table::kResolution - 2);
    const int yi = std::min(static_cast<int>(y), table::kResolution - 2);
    // The scale knots are the table's first floats, ascending: the last knot not above z brackets it.
    int zi = 0;
    while (zi < table::kResolution - 2 && tableFloat(bytes, static_cast<std::size_t>(zi) + 1) <= z) {
        ++zi;
    }
    const float k0 = tableFloat(bytes, static_cast<std::size_t>(zi));
    const float k1 = tableFloat(bytes, static_cast<std::size_t>(zi) + 1);
    const glm::vec3 f(x - static_cast<float>(xi), y - static_cast<float>(yi), (z - k0) / (k1 - k0));
    glm::vec3 c(0.0F);
    for (int corner = 0; corner < 8; ++corner) {
        const int dx = corner & 1;
        const int dy = (corner >> 1) & 1;
        const int dz = corner >> 2;
        const float w = (dx != 0 ? f.x : 1.0F - f.x) * (dy != 0 ? f.y : 1.0F - f.y) * (dz != 0 ? f.z : 1.0F - f.z);
        const std::size_t at = table::kResolution + table::nodeOffset(l, zi + dz, yi + dy, xi + dx);
        c += w * glm::vec3(tableFloat(bytes, at), tableFloat(bytes, at + 1), tableFloat(bytes, at + 2));
    }
    return {c};
}

SampledSpectrum reflectanceSpectrum(const Rgb& rgb, const SampledWavelengths& lambda) {
    const Rgb unit = clamp(rgb, 0.0F, 1.0F);
    // Equal channels are the flat spectrum, exactly: no lookup, and white and grey stay bit-exact.
    if (unit.r == unit.g && unit.g == unit.b) {
        return SampledSpectrum(unit.r);
    }
    return rgbSigmoid(unit)(lambda);
}

UnboundedSpectrum fitUnbounded(const Rgb& rgb) {
    const Rgb positive = max(rgb, Rgb(0.0F));
    const float peak = maxChannel(positive);
    if (positive.r == positive.g && positive.g == positive.b) {
        return {{glm::vec3(0.0F, 0.0F, std::numeric_limits<float>::infinity())}, peak};
    }
    return {rgbSigmoid(positive * (kMidScale / peak)), peak / kMidScale};
}

SampledSpectrum illuminantSpectrum(const Rgb& rgb, const SampledWavelengths& lambda) {
    const UnboundedSpectrum spectrum = fitUnbounded(rgb);
    return spectrum(lambda) * lambda.white;
}

EmissionSpectrum::EmissionSpectrum(const cie::Spectrum& power) {
    const double scale = 1.0 / (static_cast<double>(kLuminousEfficacy) * cie::luminanceSum(power));
    for (std::size_t i = 0; i < radiance_.size(); ++i) {
        radiance_[i] = static_cast<float>(power[i] * scale);
    }
}

float EmissionSpectrum::luminance() const {
    double sum = 0.0;
    for (int i = 0; i < cie::kSampleCount; ++i) {
        sum += static_cast<double>(radiance_[static_cast<std::size_t>(i)]) * cie::tableRow(i).colourMatching.y;
    }
    return static_cast<float>(static_cast<double>(kLuminousEfficacy) * sum);
}

EmissionSpectrum& EmissionSpectrum::operator*=(float luminance) {
    for (float& value : radiance_) {
        value *= luminance;
    }
    return *this;
}

SampledSpectrum EmissionSpectrum::operator()(const SampledWavelengths& lambda) const {
    SampledSpectrum out(0.0F);
    for (int j = 0; j < kSpectrumSamples; ++j) {
        const float offset = lambda.nm[j] - static_cast<float>(cie::kLambdaMinNm);
        const int at = std::min(static_cast<int>(offset), kBins - 1);
        const float f = offset - static_cast<float>(at);
        out[j] = glm::mix(radiance_[static_cast<std::size_t>(at)], radiance_[static_cast<std::size_t>(at) + 1], f);
    }
    return out;
}

cie::Spectrum rgbPower(const Rgb& rgb) {
    const UnboundedSpectrum spectrum = fitUnbounded(rgb);
    cie::Spectrum power{};
    for (int i = 0; i < cie::kSampleCount; ++i) {
        power[static_cast<std::size_t>(i)] = static_cast<double>(spectrum.shape(static_cast<float>(cie::wavelengthNm(i)))) * cie::tableRow(i).d65;
    }
    return power;
}

EmissionSpectrum rgbEmission(const Rgb& rgb) { return EmissionSpectrum(rgbPower(rgb)); }

std::optional<EmissionSpectrum> filteredEmission(cie::Spectrum power, const cie::Spectrum& filter) {
    std::ranges::transform(power, filter, power.begin(), std::multiplies<>());
    if (!(cie::luminanceSum(power) > 0.0)) {
        return std::nullopt;
    }
    return EmissionSpectrum(power);
}

}  // namespace pathtracer::scene
