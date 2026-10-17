#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Shared statistics for the validators: quadrature, survival functions, quantiles and confidence bands.
namespace tools::stats {

// Composite Simpson's rule over `panels` (even) intervals, generic in the return type so it composes for nested quadrature.
template <typename Integrand>
auto simpson(double lower, double upper, int panels, Integrand f) -> decltype(f(0.0)) {
    const double h = (upper - lower) / panels;
    auto sum = f(lower) + f(upper);
    for (int i = 1; i < panels; ++i) {
        sum = sum + ((i % 2 == 1 ? 4.0 : 2.0) * f(lower + (i * h)));
    }
    return (h / 3.0) * sum;
}

// Regularized upper incomplete gamma Q(a, x): series below the crossover, continued fraction above (Numerical Recipes 3rd ed. 6.2).
inline double regularizedGammaQ(double a, double x) {
    constexpr int kMaxIterations = 300;
    constexpr double kEpsilon = 1e-14;
    const double logGammaA = std::lgamma(a);
    const double tiny = std::numeric_limits<double>::min();
    if (x < a + 1.0) {
        double term = 1.0 / a;
        double sum = term;
        for (int n = 1; n < kMaxIterations && std::fabs(term) > std::fabs(sum) * kEpsilon; ++n) {
            term *= x / (a + n);
            sum += term;
        }
        return 1.0 - (sum * std::exp(-x + (a * std::log(x)) - logGammaA));
    }
    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < kMaxIterations; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = (an * d) + b;
        if (std::fabs(d) < tiny) { d = tiny; }
        c = b + (an / c);
        if (std::fabs(c) < tiny) { c = tiny; }
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::fabs(delta - 1.0) <= kEpsilon) { break; }
    }
    return h * std::exp(-x + (a * std::log(x)) - logGammaA);
}

// Upper-tail probability of the chi-square distribution: P(X > chi2) at `dof` degrees of freedom.
inline double chiSquareUpperTail(double chi2, int dof) {
    return regularizedGammaQ(0.5 * dof, 0.5 * chi2);
}

// Per-test significance holding familyAlpha across `tests` independent tests (Sidak 1967); agrees with Bonferroni to 1e-4 here.
inline double sidak(double familyAlpha, int tests) {
    return 1.0 - std::pow(1.0 - familyAlpha, 1.0 / static_cast<double>(tests));
}


// --- Quantiles: these retire "6 sigma" as a written constant, p = 1.97e-9 being a number nobody chose deliberately.

// Inverse standard normal CDF (Wichura 1988, AS 241, PPND16): relative accuracy ~1e-16, a closed rational form with no iteration.
inline double normalQuantile(double p) {
    const double q = p - 0.5;
    if (std::fabs(q) <= 0.425) {
        const double r = 0.180625 - (q * q);
        return q *
               (((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r +
                     45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r +
                  133.14166789178437745) * r + 3.387132872796366608) /
               (((((((5226.495278852854561 * r + 28729.085735721942674) * r + 39307.89580009271061) * r +
                     21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r +
                  42.313330701600911252) * r + 1.0);
    }
    double r = q < 0.0 ? p : 1.0 - p;
    r = std::sqrt(-std::log(r));
    double value = 0.0;
    if (r <= 5.0) {
        r -= 1.6;
        value = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r +
                      1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r +
                   4.6303378461565452959) * r + 1.42343711074968357734) /
                (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r +
                      0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r +
                   2.05319162663775882187) * r + 1.0);
    } else {
        r -= 5.0;
        value = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r +
                      0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r +
                   5.4637849111641143699) * r + 6.6579046435011037772) /
                (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r +
                      7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r +
                   0.59983220655588793769) * r + 1.0);
    }
    return q < 0.0 ? -value : value;
}

// Continued fraction for the incomplete beta function (Numerical Recipes 3rd ed. 6.4); backs regularizedBetaI.
inline double betaContinuedFraction(double a, double b, double x) {
    constexpr int kMaxIterations = 300;
    constexpr double kEpsilon = 1e-14;
    const double tiny = std::numeric_limits<double>::min();
    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - (qab * x / qap);
    if (std::fabs(d) < tiny) { d = tiny; }
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= kMaxIterations; ++m) {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + (aa * d);
        if (std::fabs(d) < tiny) { d = tiny; }
        c = 1.0 + (aa / c);
        if (std::fabs(c) < tiny) { c = tiny; }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + (aa * d);
        if (std::fabs(d) < tiny) { d = tiny; }
        c = 1.0 + (aa / c);
        if (std::fabs(c) < tiny) { c = tiny; }
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::fabs(delta - 1.0) <= kEpsilon) { break; }
    }
    return h;
}

// Regularized incomplete beta I_x(a,b) (Numerical Recipes 3rd ed. 6.4).
inline double regularizedBetaI(double a, double b, double x) {
    if (x <= 0.0) { return 0.0; }
    if (x >= 1.0) { return 1.0; }
    const double front =
        std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + (a * std::log(x)) + (b * std::log1p(-x)));
    return x < (a + 1.0) / (a + b + 2.0) ? front * betaContinuedFraction(a, b, x) / a
                                          : 1.0 - (front * betaContinuedFraction(b, a, 1.0 - x) / b);
}

// Two-sided Student-t quantile, not normal, because the variance is ESTIMATED: at R = 16 (15 dof) the difference is a factor of ~1.6.
inline double studentTTwoSided(double alpha, double dof) {
    // Monotone in t, so bisection is both correct and immune to the convergence caveats an inverse-series would carry.
    const auto tailProbability = [dof](double t) {
        return regularizedBetaI(0.5 * dof, 0.5, dof / (dof + (t * t)));  // = P(|T| > t)
    };
    double low = 0.0;
    double high = 1.0;
    while (tailProbability(high) > alpha && high < 1e12) { high *= 2.0; }
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (low + high);
        (tailProbability(mid) > alpha ? low : high) = mid;
    }
    return 0.5 * (low + high);
}

// Replicate count for randomized-QMC: independent scrambles make replicate means iid (Owen 1997b; L'Ecuyer & Lemieux 2002).
inline constexpr int kReplicates = 16;

// --- Estimators and bands ---------------------------------------------------------------------------------------

// A half-open interval stated in PASS form, so a NaN value or a NaN bound fails rather than silently passing.
struct Band {
    double lo = 0.0;
    double hi = 0.0;
    [[nodiscard]] bool contains(double v) const { return v >= lo && v <= hi; }
    [[nodiscard]] double halfWidth() const { return 0.5 * (hi - lo); }
};

// Welford's online variance (Welford 1962; numerics per Chan, Golub & LeVeque 1983): one pass, no catastrophic difference of sums.
class Welford {
public:
    void add(double x) {
        ++count_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(count_);
        m2_ += delta * (x - mean_);
    }
    [[nodiscard]] long long count() const { return count_; }
    [[nodiscard]] double mean() const { return mean_; }
    // Unbiased (n-1) sample variance.
    [[nodiscard]] double sampleVariance() const {
        return count_ > 1 ? m2_ / static_cast<double>(count_ - 1) : 0.0;
    }
    [[nodiscard]] double standardError() const {
        return count_ > 1 ? std::sqrt(sampleVariance() / static_cast<double>(count_)) : 0.0;
    }

private:
    long long count_ = 0;
    double mean_ = 0.0;
    double m2_ = 0.0;
};

// Band for the DIFFERENCE of two independent estimators at the null they estimate the same thing, on Welch's standard error.
inline Band differenceBand(const Welford& a, const Welford& b, double alpha) {
    const double combined = (a.sampleVariance() / static_cast<double>(a.count())) + (b.sampleVariance() / static_cast<double>(b.count()));
    if (!(combined > 0.0)) {
        return Band{0.0, 0.0};  // both estimators are constant; any difference at all is a real one
    }
    // Hsu 1938's min(n)-1 dof: size <= alpha at every variance ratio (Mickey & Brown 1966); Welch's dof is liberal at small alpha.
    const double dof = static_cast<double>(std::min(a.count(), b.count()) - 1);
    const double half = studentTTwoSided(alpha, dof) * std::sqrt(combined);
    return Band{-half, half};
}

// P(X <= k) or P(X >= k), X ~ Bin(n, p), summed outward from k past the mode, where terms fall geometrically: stops at remainder < ulp.
inline double binomialTail(long long k, long long n, double p, bool lower) {
    const auto pmf = [n, p](long long j) {
        return std::exp(std::lgamma(static_cast<double>(n) + 1.0) - std::lgamma(static_cast<double>(j) + 1.0) -
                        std::lgamma(static_cast<double>(n - j) + 1.0) + (static_cast<double>(j) * std::log(p)) +
                        (static_cast<double>(n - j) * std::log1p(-p)));
    };
    if (k < 0 || k > n) {
        return (k < 0) == lower ? 0.0 : 1.0;
    }
    if (p <= 0.0 || p >= 1.0) {
        const long long all = p <= 0.0 ? 0 : n;  // the degenerate law puts every trial on one side
        return (lower ? all <= k : all >= k) ? 1.0 : 0.0;
    }
    const double mode = std::floor(static_cast<double>(n + 1) * p);
    // The tail that contains the mode is the complement of the other, which is summed instead so its terms decrease from the start.
    if (lower ? static_cast<double>(k) >= mode : static_cast<double>(k) <= mode) {
        return 1.0 - binomialTail(lower ? k + 1 : k - 1, n, p, !lower);
    }
    const double odds = p / (1.0 - p);
    double term = pmf(k);
    double sum = 0.0;
    for (long long j = k; lower ? j >= 0 : j <= n; j += lower ? -1 : 1) {
        sum += term;
        const double ratio = lower ? static_cast<double>(j) / (static_cast<double>(n - j + 1) * odds)
                                   : (static_cast<double>(n - j) * odds) / static_cast<double>(j + 1);
        if (!(ratio < 1.0) || term * ratio / (1.0 - ratio) > 0.5 * std::numeric_limits<double>::epsilon() * sum) {
            term *= ratio;
            continue;
        }
        break;
    }
    return sum;
}

// Clopper & Pearson 1934: tails inverted at alpha/2, coverage >= 1 - alpha at every p, which Wilson's is not (Brown et al. 2001).
inline Band clopperPearsonBand(long long successes, long long trials, double alpha) {
    // Each bisection keeps its outer end, so rounding can only widen the band, never shrink its coverage.
    const auto bisect = [&](bool lowerEnd) {
        double low = 0.0;
        double high = 1.0;
        for (int i = 0; i < 200; ++i) {
            const double mid = 0.5 * (low + high);
            // Lower end: P(X >= k) rises with p to alpha/2; upper end: P(X <= k) falls with p to alpha/2.
            const bool belowEnd = lowerEnd ? binomialTail(successes, trials, mid, false) < 0.5 * alpha
                                           : binomialTail(successes, trials, mid, true) > 0.5 * alpha;
            (belowEnd ? low : high) = mid;
        }
        return lowerEnd ? low : high;
    };
    return Band{successes == 0 ? 0.0 : bisect(true), successes == trials ? 1.0 : bisect(false)};
}

// --- Distribution-free location shift (Hollander, Wolfe & Chicken, 3rd ed.): timing noise is skewed, so normal bands understate it.

// Null pmf of the Wilcoxon signed-rank T+ over 0..n(n+1)/2, built in probability space, so it cannot overflow at any n.
inline std::vector<double> signedRankNull(int n) {
    std::vector<double> pmf{1.0};
    for (int rank = 1; rank <= n; ++rank) {
        std::vector<double> next(pmf.size() + static_cast<std::size_t>(rank), 0.0);
        for (std::size_t t = 0; t < pmf.size(); ++t) {
            next[t] += 0.5 * pmf[t];
            next[t + static_cast<std::size_t>(rank)] += 0.5 * pmf[t];
        }
        pmf = std::move(next);
    }
    return pmf;
}

// Null pmf of Mann-Whitney U over 0..mn: p(i,j,u) = i/(i+j) p(i-1,j,u) + j/(i+j) p(i,j-1,u-i), recursing on the largest observation.
inline std::vector<double> rankSumNull(int m, int n) {
    std::vector<std::vector<double>> previous(static_cast<std::size_t>(n) + 1, std::vector<double>{1.0});  // i = 0
    for (int i = 1; i <= m; ++i) {
        std::vector<std::vector<double>> current(static_cast<std::size_t>(n) + 1);
        current[0] = {1.0};
        for (int j = 1; j <= n; ++j) {
            const std::vector<double>& xLast = previous[static_cast<std::size_t>(j)];
            const std::vector<double>& yLast = current[static_cast<std::size_t>(j) - 1];
            std::vector<double> pmf(static_cast<std::size_t>(i) * static_cast<std::size_t>(j) + 1, 0.0);
            const double px = static_cast<double>(i) / static_cast<double>(i + j);
            for (std::size_t u = 0; u < xLast.size(); ++u) {
                pmf[u] += px * xLast[u];
            }
            for (std::size_t u = 0; u < yLast.size(); ++u) {
                pmf[u + static_cast<std::size_t>(i)] += (1.0 - px) * yLast[u];
            }
            current[static_cast<std::size_t>(j)] = std::move(pmf);
        }
        previous = std::move(current);
    }
    return previous[static_cast<std::size_t>(n)];
}

// Smallest t with P(X >= t) <= tail under `pmf`; pmf.size() when no value is that extreme (too few observations).
inline std::size_t upperCritical(const std::vector<double>& pmf, double tail) {
    double upper = 0.0;
    std::size_t t = pmf.size();
    while (t > 0 && upper + pmf[t - 1] <= tail) {
        upper += pmf[--t];
    }
    return t;
}

// Point estimate with a two-sided interval of coverage >= 1 - alpha; (-inf, +inf) when no interval at alpha exists.
struct ShiftEstimate {
    double estimate = 0.0;
    double lower = -std::numeric_limits<double>::infinity();
    double upper = std::numeric_limits<double>::infinity();
    double coverage = 1.0;  // achieved coverage of [lower, upper] under the null distribution
};

// Median of a sorted sample, averaging the middle pair for an even count.
inline double sortedMedian(const std::vector<double>& sorted) {
    const std::size_t mid = sorted.size() / 2;
    return sorted.size() % 2 == 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

// Order-statistic interval [W_(C), W_(N+1-C)] with C = N + 1 - t, t the upper critical value (Hollander et al. 3.2/4.3).
inline ShiftEstimate orderStatisticInterval(const std::vector<double>& sorted, const std::vector<double>& nullPmf,
                                            double alpha) {
    ShiftEstimate result;
    result.estimate = sortedMedian(sorted);
    const std::size_t t = upperCritical(nullPmf, 0.5 * alpha);
    const std::size_t count = sorted.size();
    if (t >= nullPmf.size() || t == 0) {
        return result;
    }
    const std::size_t c = count + 1 - t;  // 1-based rank of the lower bound
    result.lower = sorted[c - 1];
    result.upper = sorted[count - c];
    double tailMass = 0.0;
    for (std::size_t u = t; u < nullPmf.size(); ++u) {
        tailMass += nullPmf[u];
    }
    result.coverage = 1.0 - (2.0 * tailMass);
    return result;
}

// Hodges-Lehmann centre from paired differences: median of the n(n+1)/2 Walsh averages (Hodges & Lehmann 1963; Hollander et al. 3.2).
inline ShiftEstimate hodgesLehmannPaired(const std::vector<double>& d, double alpha) {
    std::vector<double> walsh;
    walsh.reserve(d.size() * (d.size() + 1) / 2);
    for (std::size_t i = 0; i < d.size(); ++i) {
        for (std::size_t j = i; j < d.size(); ++j) {
            walsh.push_back(0.5 * (d[i] + d[j]));
        }
    }
    std::sort(walsh.begin(), walsh.end());
    return orderStatisticInterval(walsh, signedRankNull(static_cast<int>(d.size())), alpha);
}

// Hodges-Lehmann two-sample shift of y over x: median of all y_j - x_i (Hollander et al. 4.3); assumes only that y is x shifted.
inline ShiftEstimate hodgesLehmannShift(const std::vector<double>& x, const std::vector<double>& y, double alpha) {
    std::vector<double> differences;
    differences.reserve(x.size() * y.size());
    for (const double xi : x) {
        for (const double yj : y) {
            differences.push_back(yj - xi);
        }
    }
    std::sort(differences.begin(), differences.end());
    return orderStatisticInterval(differences, rankSumNull(static_cast<int>(x.size()), static_cast<int>(y.size())),
                                  alpha);
}

}  // namespace tools::stats
