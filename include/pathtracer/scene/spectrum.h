#pragma once

#include <array>
#include <optional>

#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>

#include "pathtracer/scene/cie.h"
#include "pathtracer/scene/rgb.h"

// Spectral transport: a path's sampled wavelengths, the radiance it carries at them, and their projection onto scene RGB.
namespace pathtracer::scene {

// Wavelengths per path: a hero and its rotations (Wilkie et al. 2014), one 128-bit SIMD vector.
inline constexpr int kSpectrumSamples = 4;

// A quantity at a path's sampled wavelengths, one lane each; ops are glm's per lane, as Rgb's are per channel.
struct SampledSpectrum {
    glm::vec4 lanes;

    SampledSpectrum() = default;
    constexpr explicit SampledSpectrum(float value) : lanes(value) {}
    constexpr explicit SampledSpectrum(const glm::vec4& v) : lanes(v) {}

    [[nodiscard]] constexpr float& operator[](int i) { return lanes[i]; }
    [[nodiscard]] constexpr float operator[](int i) const { return lanes[i]; }

    constexpr SampledSpectrum& operator+=(const SampledSpectrum& o) { lanes += o.lanes; return *this; }
    constexpr SampledSpectrum& operator*=(const SampledSpectrum& o) { lanes *= o.lanes; return *this; }
    constexpr SampledSpectrum& operator-=(float s) { lanes -= s; return *this; }
    constexpr SampledSpectrum& operator*=(float s) { lanes *= s; return *this; }
    constexpr SampledSpectrum& operator/=(float s) { lanes /= s; return *this; }

    friend constexpr bool operator==(const SampledSpectrum&, const SampledSpectrum&) = default;
};

[[nodiscard]] constexpr SampledSpectrum operator-(const SampledSpectrum& a) { return SampledSpectrum(-a.lanes); }
[[nodiscard]] constexpr SampledSpectrum operator+(const SampledSpectrum& a, const SampledSpectrum& o) { return SampledSpectrum(a.lanes + o.lanes); }
[[nodiscard]] constexpr SampledSpectrum operator-(const SampledSpectrum& a, const SampledSpectrum& o) { return SampledSpectrum(a.lanes - o.lanes); }
[[nodiscard]] constexpr SampledSpectrum operator*(const SampledSpectrum& a, const SampledSpectrum& o) { return SampledSpectrum(a.lanes * o.lanes); }
[[nodiscard]] constexpr SampledSpectrum operator/(const SampledSpectrum& a, const SampledSpectrum& o) { return SampledSpectrum(a.lanes / o.lanes); }
[[nodiscard]] constexpr SampledSpectrum operator+(const SampledSpectrum& a, float s) { return SampledSpectrum(a.lanes + s); }
[[nodiscard]] constexpr SampledSpectrum operator-(const SampledSpectrum& a, float s) { return SampledSpectrum(a.lanes - s); }
[[nodiscard]] constexpr SampledSpectrum operator*(const SampledSpectrum& a, float s) { return SampledSpectrum(a.lanes * s); }
[[nodiscard]] constexpr SampledSpectrum operator/(const SampledSpectrum& a, float s) { return SampledSpectrum(a.lanes / s); }
[[nodiscard]] constexpr SampledSpectrum operator+(float s, const SampledSpectrum& a) { return SampledSpectrum(s + a.lanes); }
[[nodiscard]] constexpr SampledSpectrum operator-(float s, const SampledSpectrum& a) { return SampledSpectrum(s - a.lanes); }
[[nodiscard]] constexpr SampledSpectrum operator*(float s, const SampledSpectrum& a) { return SampledSpectrum(s * a.lanes); }
[[nodiscard]] constexpr SampledSpectrum operator/(float s, const SampledSpectrum& a) { return SampledSpectrum(s / a.lanes); }

// The equal-weight lane mean: selection masses and energies compare lanes on one scale, as channelMean compares Rgb's channels.
[[nodiscard]] constexpr float channelMean(const SampledSpectrum& c) { return ((c.lanes.x + c.lanes.y) + (c.lanes.z + c.lanes.w)) / 4.0F; }
[[nodiscard]] constexpr float channelSum(const SampledSpectrum& c) { return (c.lanes.x + c.lanes.y) + (c.lanes.z + c.lanes.w); }
[[nodiscard]] inline float maxChannel(const SampledSpectrum& c) { return glm::compMax(c.lanes); }
[[nodiscard]] inline float minChannel(const SampledSpectrum& c) { return glm::compMin(c.lanes); }
[[nodiscard]] constexpr bool isBlack(const SampledSpectrum& c) { return c == SampledSpectrum(0.0F); }

[[nodiscard]] inline SampledSpectrum exp(const SampledSpectrum& c) { return SampledSpectrum(glm::exp(c.lanes)); }
[[nodiscard]] inline SampledSpectrum log(const SampledSpectrum& c) { return SampledSpectrum(glm::log(c.lanes)); }
[[nodiscard]] inline SampledSpectrum abs(const SampledSpectrum& c) { return SampledSpectrum(glm::abs(c.lanes)); }
[[nodiscard]] inline SampledSpectrum sqrt(const SampledSpectrum& c) { return SampledSpectrum(glm::sqrt(c.lanes)); }
[[nodiscard]] inline SampledSpectrum pow(const SampledSpectrum& c, const SampledSpectrum& e) { return SampledSpectrum(glm::pow(c.lanes, e.lanes)); }
[[nodiscard]] inline SampledSpectrum max(const SampledSpectrum& a, const SampledSpectrum& o) { return SampledSpectrum(glm::max(a.lanes, o.lanes)); }
[[nodiscard]] inline SampledSpectrum clamp(const SampledSpectrum& c, float lo, float hi) { return SampledSpectrum(glm::clamp(c.lanes, lo, hi)); }
[[nodiscard]] inline SampledSpectrum mix(const SampledSpectrum& a, const SampledSpectrum& o, float t) { return SampledSpectrum(glm::mix(a.lanes, o.lanes, t)); }

// The luminous efficacy of radiation at the photopic peak, lm/W (CIE 018:2019): radiance in W/(sr m^2 nm) times it reads cd/m^2.
inline constexpr float kLuminousEfficacy = 683.002F;

// One path's wavelengths and the per-lane weights that project its spectra onto scene RGB, built once per path.
struct SampledWavelengths {
    glm::vec4 nm;   // lane wavelengths, kLambdaMinNm to kLambdaMaxNm
    glm::vec4 pdf;  // each lane's marginal density per nm, the same density for every lane
    // Columns K_m xyzToScene cmf(lambda) / (n pdf): radiance in W/(sr m^2 nm) to scene RGB in cd/m^2, the sensor's estimator.
    glm::mat4x3 toScene;
    // toScene times D65(lambda) / (K_m sum D65 yBar): a reflectance under the scene white, so a flat 1 reads (1, 1, 1).
    glm::mat4x3 reflectanceToScene;
    // D65(lambda) / (K_m sum D65 yBar): the scene white's radiance per unit luminance, what an RGB (1, 1, 1) emitter radiates.
    SampledSpectrum white;
};

// A hero-wavelength density, piecewise constant over the CIE 1 nm bins, and its CDF at the bin edges.
struct WavelengthPdf {
    std::array<float, cie::kSampleCount> cdf;  // kSampleCount - 1 bins' edges from 0 to 1, for invertCdf
    std::array<float, cie::kSampleCount - 1> pdf;  // each bin's density per nm
};

// The sensor's: p ~ sum_c |s_c|, each bin at its knots' mean, so |s_c| / p is bounded and no channel's weight spikes.
[[nodiscard]] const WavelengthPdf& sensorWavelengthPdf();

// Half the sensor's density, half the sensor's times the emitters' spectral flux: defensive one-sample MIS (Hesterberg 1995).
[[nodiscard]] WavelengthPdf emitterWavelengthPdf(const cie::Spectrum& spectralFlux);

// The hero at u and rotations u + j/n mod 1 in density's CDF: every lane keeps its pdf p, so each lane's estimator is unbiased.
[[nodiscard]] SampledWavelengths sampleWavelengths(float u, const WavelengthPdf& density = sensorWavelengthPdf());

// A radiance's scene RGB and a reflectance's under the scene white, through the path's sensor weights.
[[nodiscard]] inline Rgb toScene(const SampledSpectrum& radiance, const SampledWavelengths& lambda) {
    return Rgb(lambda.toScene * radiance.lanes);
}
[[nodiscard]] inline Rgb reflectanceToScene(const SampledSpectrum& reflectance, const SampledWavelengths& lambda) {
    return Rgb(lambda.reflectanceToScene * reflectance.lanes);
}

// Jakob & Hanika 2019's sigmoid polynomial in nm, s = 1/2 + x / (2 sqrt(1 + x^2)), x = (c0 lambda + c1) lambda + c2: within [0, 1].
struct RgbSigmoid {
    glm::vec3 c;
    [[nodiscard]] SampledSpectrum operator()(const SampledWavelengths& lambda) const;
    [[nodiscard]] float operator()(float nm) const;
};

// The coefficients for rgb in [0, 1]^3, trilinear in the table fitted under D65, the scene white (the paper's lookup); black is 0.
[[nodiscard]] RgbSigmoid rgbSigmoid(const Rgb& rgb);

// A reflectance or transmittance colour, clamped to [0, 1] per channel: no physical reflectance exceeds 1 or falls below 0.
[[nodiscard]] SampledSpectrum reflectanceSpectrum(const Rgb& rgb, const SampledWavelengths& lambda);

// A non-negative unbounded colour (coefficients, radii): 2 max(rgb) times the fit of rgb / (2 max(rgb)), fitted once; negatives to 0.
struct UnboundedSpectrum {
    RgbSigmoid shape;  // grey is the flat 1, c2 = +inf
    float scale;
    [[nodiscard]] SampledSpectrum operator()(const SampledWavelengths& lambda) const { return scale * shape(lambda); }
};
[[nodiscard]] UnboundedSpectrum fitUnbounded(const Rgb& rgb);
[[nodiscard]] inline SampledSpectrum unboundedSpectrum(const Rgb& rgb, const SampledWavelengths& lambda) { return fitUnbounded(rgb)(lambda); }

// An emitted radiance colour in cd/m^2: its unbounded fit times the scene white, so RGB (1, 1, 1) is D65 exactly.
[[nodiscard]] SampledSpectrum illuminantSpectrum(const Rgb& rgb, const SampledWavelengths& lambda);

// An emitter's spectral radiance tabulated at the CIE 1 nm samples in W/(sr m^2 nm), evaluated by linear interpolation.
class EmissionSpectrum {
public:
    // power's shape at 1 cd/m^2: scaled so its luminance K_m sum S yBar is exactly 1; power must have positive luminance.
    explicit EmissionSpectrum(const cie::Spectrum& power);
    [[nodiscard]] SampledSpectrum operator()(const SampledWavelengths& lambda) const;
    // Scales the radiance, so a unit-luminance spectrum times L reads L cd/m^2.
    EmissionSpectrum& operator*=(float luminance);
    // K_m sum S yBar over the CIE samples: the cd/m^2 a light of this radiance shows a photometer.
    [[nodiscard]] float luminance() const;
    // The tabulated radiance at the CIE samples.
    [[nodiscard]] const std::array<float, cie::kSampleCount>& samples() const { return radiance_; }
    friend bool operator==(const EmissionSpectrum&, const EmissionSpectrum&) = default;

private:
    std::array<float, cie::kSampleCount> radiance_;
};

// An RGB light's relative spectral power: illuminantSpectrum's shape tabulated on the CIE samples, so RGB (1, 1, 1) is D65.
[[nodiscard]] cie::Spectrum rgbPower(const Rgb& rgb);

// rgbPower at 1 cd/m^2.
[[nodiscard]] EmissionSpectrum rgbEmission(const Rgb& rgb);

// A lamp's power times its filter's transmittance at 1 cd/m^2, the filter inside the photometric unit (PhysLight); nullopt if none passes.
[[nodiscard]] std::optional<EmissionSpectrum> filteredEmission(cie::Spectrum power, const cie::Spectrum& filter);

}  // namespace pathtracer::scene
