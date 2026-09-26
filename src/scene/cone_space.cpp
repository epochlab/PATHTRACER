#include "pathtracer/scene/cone_space.h"

#include "pathtracer/scene/cie.h"

namespace pathtracer::scene::cone {

namespace {

// A row summing to zero is r.x(R-G) + r.z(B-G) identically, so keeping those two coefficients is the same operator with green implied.
[[nodiscard]] glm::vec2 differenceCoefficients(const glm::dvec3& row) {
    return {static_cast<float>(row.x), static_cast<float>(row.z)};
}

OpponentBasis deriveOpponentBasis() {
    const glm::dmat3 rgbToLms = xyzToLms() * glm::inverse(cie::xyzToRec709());
    // glm stores columns, so row r of the transform is the r-th component of each column: exactly the coefficients dotted against RGB.
    const glm::dvec3 longRow(rgbToLms[0][0], rgbToLms[1][0], rgbToLms[2][0]);
    const glm::dvec3 mediumRow(rgbToLms[0][1], rgbToLms[1][1], rgbToLms[2][1]);
    const glm::dvec3 shortRow(rgbToLms[0][2], rgbToLms[1][2], rgbToLms[2][2]);
    const glm::dvec3 sumRow = longRow + mediumRow;

    // White is RGB (1,1,1) by xyzToRec709()'s construction, so every reference excitation is one row sum, not a second integration.
    const double whiteSum = sumRow.x + sumRow.y + sumRow.z;
    const double whiteLong = (longRow.x + longRow.y + longRow.z) / whiteSum;
    const double whiteShort = shortRow.x + shortRow.y + shortRow.z;
    // l - l_white and s - s_white over the common denominator L+M; whiteSum scales S so that s is S per unit luminance at white.
    return OpponentBasis{differenceCoefficients(longRow - (whiteLong * sumRow)),
                         differenceCoefficients((shortRow * whiteSum) - (whiteShort * sumRow)), glm::vec3(sumRow)};
}

}  // namespace

const glm::dmat3& xyzToLms() {
    // Columns, glm's storage order: the L row is (0.38971, 0.68898, -0.07868), the S row is Z alone, which is what makes it exact on XYZ.
    static const glm::dmat3 kMatrix{{0.38971, -0.22981, 0.0}, {0.68898, 1.18340, 0.0}, {-0.07868, 0.04641, 1.0}};
    return kMatrix;
}

glm::dvec2 coneChromaticity(const glm::dvec3& lms) {
    const double sum = lms.x + lms.y;
    return {lms.x / sum, lms.z / sum};
}

glm::dvec2 whiteConeChromaticity() {
    static const glm::dvec2 kWhite =
        coneChromaticity(xyzToLms() * (glm::inverse(cie::xyzToRec709()) * glm::dvec3(1.0)));
    return kWhite;
}

const OpponentBasis& opponentBasis() {
    static const OpponentBasis kBasis = deriveOpponentBasis();
    return kBasis;
}

}  // namespace pathtracer::scene::cone
