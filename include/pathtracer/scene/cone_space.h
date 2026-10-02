#pragma once

#include <glm/glm.hpp>

// Cone opponency on the CIE 1931 2-degree observer cie.h tabulates; Hunt-Pointer-Estevez is exact on it, Smith-Pokorny (Judd-Vos) is not.
namespace pathtracer::scene::cone {

// Hunt-Pointer-Estevez (Estevez 1979; Hunt 1998 App. 1), normalised so the equal-energy stimulus gives L = M = S.
[[nodiscard]] const glm::dmat3& xyzToLms();

// Cone chromaticity in MacLeod & Boynton 1979's form: (l, s) = (L, S)/(L+M), exactly invariant to any positive scale on the stimulus.
[[nodiscard]] glm::dvec2 coneChromaticity(const glm::dvec3& lms);

// Rec.709 white's cone chromaticity, the achromatic origin. It is xyzToRec709()'s own white, so RGB (v, v, v) lands here by construction.
[[nodiscard]] glm::dvec2 whiteConeChromaticity();

// The two opponent axes as coefficients on the chromatic differences (R-G, B-G), over their shared denominator: one divide per texel.
struct OpponentBasis {
    // A zero-sum row carries no achromatic component, so writing it on differences makes an achromatic texel exactly zero at any intensity.
    glm::vec2 redGreenNumerator;    // l - l_white, positive toward L over M
    glm::vec2 blueYellowNumerator;  // (s - s_white) * (L+M)_white: S per unit L+M, one unit per S excitation at white's cone sum
    glm::vec3 denominator;          // L + M, the luminance-like cone sum both axes divide by
};
[[nodiscard]] const OpponentBasis& opponentBasis();

}  // namespace pathtracer::scene::cone
