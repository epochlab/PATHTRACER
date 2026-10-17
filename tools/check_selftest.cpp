// Guards the harness's own invariants -- the negated-pass form and name-derived seeds -- which nothing else would notice regressing.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "stats.h"

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// A helper Context cannot be constructed outside the runner, so NaN rejection is asserted on the raw comparison the macro performs.
PT_CHECK(nan_fails_every_ordered_comparison, Fast, Exact) {
    ctx.plan(5);
    PT_EXPECT(ctx, !(kNaN <= 1.0), "NaN <= x must be false, so a negated-pass assert rejects it");
    PT_EXPECT(ctx, !(kNaN >= 1.0), "NaN >= x must be false");
    PT_EXPECT(ctx, !(kNaN < 1.0), "NaN < x must be false");
    PT_EXPECT(ctx, !(kNaN > 1.0), "NaN > x must be false");
    PT_EXPECT(ctx, !(kNaN == kNaN), "NaN == NaN must be false");
}

// A band stated as ordered comparisons must reject a NaN value AND a NaN bound: a broken estimator and a broken tolerance derivation.
PT_CHECK(nan_band_rejects_both_sides, Fast, Exact) {
    ctx.plan(2);
    const auto within = [](double v, double lo, double hi) { return v >= lo && v <= hi; };
    PT_EXPECT(ctx, !within(kNaN, 0.0, 1.0), "a NaN measurement must fall outside every band");
    PT_EXPECT(ctx, !within(0.5, kNaN, kNaN), "a NaN band must contain nothing");
}

// Seeds are name-derived, so they must differ between checks and be stable within one.
PT_CHECK(seed_is_name_derived_and_stable, Fast, Exact) {
    ctx.plan(3);
    // Deliberately not `seed() == seed()`, a tautology the compiler may fold that would pass even if the derivation returned a constant.
    PT_EXPECT(ctx, ctx.seed() != 0, "a derived seed of zero means the mixing collapsed");
    PT_EXPECT(ctx, ctx.subSeed("row-a") != ctx.seed(), "a sub-seed must derive away from its master seed");
    PT_EXPECT(ctx, ctx.subSeed("row-a") != ctx.subSeed("row-b"), "distinct labels must give distinct sub-seeds");
}

// The correction must be strictly tighter than the family rate, and tighter as a check asserts more, so more assertions buy no slack.
PT_CHECK(significance_is_family_wise_corrected, Fast, Exact) {
    ctx.plan(2);
    const double perAssertion = ctx.alpha();
    PT_EXPECT(ctx, perAssertion > 0.0 && perAssertion < ::tools::check::kFamilyAlpha,
                  "per-assertion alpha must be positive and tighter than the family rate");
    PT_EXPECT(ctx, std::isfinite(perAssertion), "per-assertion alpha must be finite");
}

// Exact nulls against brute-force enumeration and published values: Wilcoxon n=10 P(T+<=8)=0.0244, Mann-Whitney 8x8 P(U<=13)=0.0249.
PT_CHECK(rank_nulls_match_enumeration, Fast, Exact) {
    ctx.plan(4);
    double worstSigned = 0.0;
    for (int n = 1; n <= 12; ++n) {
        std::vector<double> counted(static_cast<std::size_t>(n * (n + 1) / 2) + 1, 0.0);
        for (std::uint32_t signs = 0; signs < (1U << static_cast<unsigned>(n)); ++signs) {
            int t = 0;
            for (int r = 0; r < n; ++r) {
                t += ((signs >> static_cast<unsigned>(r)) & 1U) != 0 ? r + 1 : 0;
            }
            counted[static_cast<std::size_t>(t)] += 1.0 / static_cast<double>(1U << static_cast<unsigned>(n));
        }
        const std::vector<double> pmf = ::tools::stats::signedRankNull(n);
        for (std::size_t t = 0; t < counted.size(); ++t) {
            worstSigned = std::max(worstSigned, pmf.size() == counted.size() ? std::fabs(pmf[t] - counted[t]) : 1.0);
        }
    }
    PT_EXPECT(ctx, worstSigned <= 1e-15, "signedRankNull disagrees with subset enumeration");

    // Every placement of n y's among m + n ranks, U counting x's below each y.
    const auto enumerateRankSum = [](int m, int n) {
        std::vector<double> counted(static_cast<std::size_t>(m * n) + 1, 0.0);
        double total = 0.0;
        for (std::uint32_t mask = 0; mask < (1U << static_cast<unsigned>(m + n)); ++mask) {
            if (__builtin_popcount(mask) != n) {
                continue;
            }
            int u = 0;
            int xsBelow = 0;
            for (int k = 0; k < m + n; ++k) {
                if (((mask >> static_cast<unsigned>(k)) & 1U) != 0) {
                    u += xsBelow;
                } else {
                    ++xsBelow;
                }
            }
            counted[static_cast<std::size_t>(u)] += 1.0;
            total += 1.0;
        }
        for (double& c : counted) {
            c /= total;
        }
        return counted;
    };
    double worstRankSum = 0.0;
    for (const auto& [m, n] : {std::pair{5, 6}, std::pair{3, 8}, std::pair{7, 2}}) {
        const std::vector<double> counted = enumerateRankSum(m, n);
        const std::vector<double> pmf = ::tools::stats::rankSumNull(m, n);
        for (std::size_t u = 0; u < counted.size(); ++u) {
            worstRankSum = std::max(worstRankSum, pmf.size() == counted.size() ? std::fabs(pmf[u] - counted[u]) : 1.0);
        }
    }
    PT_EXPECT(ctx, worstRankSum <= 1e-15, "rankSumNull disagrees with placement enumeration");

    const std::vector<double> signed10 = ::tools::stats::signedRankNull(10);
    double lowerTail = 0.0;
    for (std::size_t t = 0; t <= 8; ++t) {
        lowerTail += signed10[t];
    }
    PT_EXPECT(ctx, std::fabs(lowerTail - 0.0244) < 5e-5, "Wilcoxon n=10 P(T+ <= 8) is not the tabulated 0.0244");
    const std::vector<double> rankSum88 = ::tools::stats::rankSumNull(8, 8);
    double uTail = 0.0;
    for (std::size_t u = 0; u <= 13; ++u) {
        uTail += rankSum88[u];
    }
    PT_EXPECT(ctx, std::fabs(uTail - 0.0249) < 5e-5, "Mann-Whitney 8x8 P(U <= 13) is not the tabulated 0.0249");
}

// Hand-computable estimates, and the too-small case: at n = 5 the most extreme T+ has probability 1/32 > alpha/2 = 0.025, so none exists.
PT_CHECK(hodges_lehmann_point_estimates, Fast, Exact) {
    ctx.plan(3);
    PT_EXPECT(ctx, ::tools::stats::hodgesLehmannPaired({1.0, 2.0, 3.0}, 0.05).estimate == 2.0, "Walsh-average median of {1,2,3} must be 2");
    PT_EXPECT(ctx, ::tools::stats::hodgesLehmannShift({0.0, 1.0}, {10.0, 12.0}, 0.05).estimate == 10.5, "median of pairwise differences {10,12,9,11} must be 10.5");
    const ::tools::stats::ShiftEstimate tooFew = ::tools::stats::hodgesLehmannPaired({0.1, 0.2, 0.3, 0.4, 0.5}, 0.05);
    PT_EXPECT(ctx, std::isinf(tooFew.lower) && std::isinf(tooFew.upper), "n = 5 cannot support a 95% signed-rank interval");
}

// Empirical coverage under a known shift must match the exact discrete coverage; a one-rank index error moves it far outside the band.
PT_CHECK(hodges_lehmann_coverage_is_exact, Fast, Statistical) {
    ctx.plan(2);
    constexpr int kTrials = 20000;
    constexpr double kShift = 0.3;
    constexpr double kAlpha = 0.05;
    std::mt19937_64 rng(ctx.seed());
    std::normal_distribution<double> noise(0.0, 1.0);

    long long pairedHits = 0;
    double pairedCoverage = 0.0;
    long long shiftHits = 0;
    double shiftCoverage = 0.0;
    for (int trial = 0; trial < kTrials; ++trial) {
        std::vector<double> d(12);
        for (double& v : d) {
            v = kShift + noise(rng);
        }
        const ::tools::stats::ShiftEstimate paired = ::tools::stats::hodgesLehmannPaired(d, kAlpha);
        pairedHits += paired.lower <= kShift && kShift <= paired.upper ? 1 : 0;
        pairedCoverage = paired.coverage;

        std::vector<double> x(7);
        std::vector<double> y(9);
        for (double& v : x) {
            v = noise(rng);
        }
        for (double& v : y) {
            v = kShift + noise(rng);
        }
        const ::tools::stats::ShiftEstimate shift = ::tools::stats::hodgesLehmannShift(x, y, kAlpha);
        shiftHits += shift.lower <= kShift && kShift <= shift.upper ? 1 : 0;
        shiftCoverage = shift.coverage;
    }
    PT_EXPECT(ctx, ::tools::stats::clopperPearsonBand(pairedHits, kTrials, ctx.alpha()).contains(pairedCoverage),
                  "paired interval's empirical coverage disagrees with its exact coverage");
    PT_EXPECT(ctx, ::tools::stats::clopperPearsonBand(shiftHits, kTrials, ctx.alpha()).contains(shiftCoverage),
                  "two-sample interval's empirical coverage disagrees with its exact coverage");
}


// The band's exact false rejection, P(p outside band(X)) for X ~ Bin(n, p), at p: accepted counts are contiguous, as both ends rise with k.
double falseRejection(const std::vector<::tools::stats::Band>& bands, double p) {
    const auto n = static_cast<long long>(bands.size()) - 1;
    long long first = 0;
    while (first <= n && !(bands[static_cast<std::size_t>(first)].hi >= p)) {
        ++first;
    }
    long long last = n;
    while (last >= 0 && !(bands[static_cast<std::size_t>(last)].lo <= p)) {
        --last;
    }
    return ::tools::stats::binomialTail(first - 1, n, p, true) + ::tools::stats::binomialTail(last + 1, n, p, false);
}

// Coverage is a sum of adjacent pmfs, unimodal in p between band ends, so the false-rejection supremum sits beside an end: it is <= alpha.
PT_CHECK(binomial_band_false_rejection_is_at_most_alpha, Fast, Exact) {
    const std::vector<long long> trials{16, 100, 1000};
    const std::vector<double> alphas{1e-2, 1e-4, 1e-6, 1e-8};
    ctx.plan(static_cast<int>(trials.size() * alphas.size()));
    for (const long long n : trials) {
        for (const double alpha : alphas) {
            std::vector<::tools::stats::Band> bands;
            for (long long k = 0; k <= n; ++k) {
                bands.push_back(::tools::stats::clopperPearsonBand(k, n, alpha));
            }
            const double u = 0.5 * std::numeric_limits<double>::epsilon();
            const double size = static_cast<double>(n) + 1.0;
            double worst = 0.0;
            for (const ::tools::stats::Band& band : bands) {
                for (const double end : {band.lo, band.hi}) {
                    for (const double side : {0.0, 1.0}) {
                        const double p = std::nextafter(end, side);
                        if (!(p > 0.0 && p < 1.0)) {
                            continue;
                        }
                        // Exponent: 3 lgammas <= size ln size, 2 log terms <= n|ln|, each rounding u; n+1 ratio steps and adds 6u.
                        const double exponent = (3.0 * size * std::log(size)) + (static_cast<double>(n) * (std::abs(std::log(p)) + std::abs(std::log1p(-p))));
                        const double relative = u * (exponent + (6.0 * size));
                        worst = std::max(worst, falseRejection(bands, p) / (alpha * (1.0 + relative)));
                    }
                }
            }
            PT_EXPECT(ctx, worst <= 1.0, "n=" + std::to_string(n) + " alpha=" + std::to_string(alpha) + " rejects at " + std::to_string(worst) + " alpha");
        }
    }
}

// Trapezoid nodes in s = ln x of chi-square(dof), weights h x f(x), h = refinement * sqrt(2/dof), out from the mode to remainder <= floor.
struct ChiSquareNodes {
    std::vector<double> x;
    std::vector<double> weight;
    double remainder = 0.0;
};

ChiSquareNodes chiSquareNodes(int dof, double refinement, double floor) {
    const double halfDof = 0.5 * static_cast<double>(dof);
    const double h = refinement / std::sqrt(halfDof);
    const auto logWeight = [&](double s) {
        return (halfDof * s) - (0.5 * std::exp(s)) - (halfDof * std::log(2.0)) - std::lgamma(halfDof) + std::log(h);
    };
    ChiSquareNodes nodes;
    const double mode = std::log(static_cast<double>(dof));
    for (const int direction : {-1, 1}) {
        // Outward, log g(s+step) - log g(s) falls monotonically on both sides, so the current ratio bounds every later one.
        for (int k = direction < 0 ? 0 : 1;; k += direction) {
            const double s = mode + (k * h);
            const double weight = std::exp(logWeight(s));
            nodes.x.push_back(std::exp(s));
            nodes.weight.push_back(weight);
            const double ratio = std::exp(logWeight(s + (direction * h)) - logWeight(s));
            if (ratio < 1.0 && weight * ratio / (1.0 - ratio) <= floor) {
                nodes.remainder += weight * ratio / (1.0 - ratio);
                break;
            }
        }
    }
    return nodes;
}

// P(D outside differenceBand) for D ~ N(0, w + (1 - w)), both sample variances integrated over their chi-square laws on one grid.
double differenceBandSize(int countA, int countB, double w, double alpha, double refinement, double floor) {
    // n/2 values at each of +-sqrt(w x) have sample variance n w x / (n - 1), the law of S^2 when Var(mean) = w and x ~ chi-square(n-1).
    const auto replicates = [](int count, double variance, const ChiSquareNodes& nodes) {
        std::vector<::tools::stats::Welford> out(nodes.x.size());
        for (std::size_t i = 0; i < nodes.x.size(); ++i) {
            for (int r = 0; r < count; ++r) {
                out[i].add((r % 2 == 0 ? 1.0 : -1.0) * std::sqrt(variance * nodes.x[i]));
            }
        }
        return out;
    };
    const ChiSquareNodes nodesA = chiSquareNodes(countA - 1, refinement, floor);
    const ChiSquareNodes nodesB = chiSquareNodes(countB - 1, refinement, floor);
    const std::vector<::tools::stats::Welford> a = replicates(countA, w, nodesA);
    const std::vector<::tools::stats::Welford> b = replicates(countB, 1.0 - w, nodesB);
    double size = 0.0;
    double massA = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        massA += nodesA.weight[i];
        for (std::size_t j = 0; j < b.size(); ++j) {
            const ::tools::stats::Band band = ::tools::stats::differenceBand(a[i], b[j], alpha);
            const double reject = 0.5 * (std::erfc(-band.lo / std::sqrt(2.0)) + std::erfc(band.hi / std::sqrt(2.0)));
            size += nodesA.weight[i] * nodesB.weight[j] * reject;
        }
    }
    double massB = 0.0;
    for (const double weight : nodesB.weight) {
        massB += weight;
    }
    // The rejection probability is <= 1, so the truncated grid adds at most the mass it leaves out.
    return size + ((massA + nodesA.remainder) * (massB + nodesB.remainder)) - (massA * massB);
}

// Exact size on a Gaussian null, the law every differenceBand call assumes: exponentially convergent trapezoid (Trefethen & Weideman 2014).
PT_CHECK(difference_band_false_rejection_is_at_most_alpha, Slow, Exact) {
    const std::vector<std::pair<int, int>> counts{{::tools::stats::kReplicates, ::tools::stats::kReplicates},
                                                  {::tools::stats::kReplicates, 4 * ::tools::stats::kReplicates}};
    const std::vector<double> varianceRatios{1.0 / 256.0, 1.0 / 16.0, 1.0, 16.0, 256.0};
    const std::vector<double> alphas{1e-2, 1e-4, 1e-6, 1e-8};
    ctx.plan(static_cast<int>(counts.size() * varianceRatios.size() * alphas.size()));
    for (const auto& [countA, countB] : counts) {
        for (const double ratio : varianceRatios) {
            const double w = ratio / (1.0 + ratio);
            for (const double alpha : alphas) {
                const double floor = std::numeric_limits<double>::epsilon() * alpha;
                // Halving h squares the trapezoid's exponentially small error, so their difference bounds the coarse one's from above.
                const double fine = differenceBandSize(countA, countB, w, alpha, 0.5, floor);
                const double coarse = differenceBandSize(countA, countB, w, alpha, 1.0, floor);
                const double bound = fine + std::abs(coarse - fine);
                PT_EXPECT(ctx, bound <= alpha,
                          "n=" + std::to_string(countA) + "/" + std::to_string(countB) + " variance ratio " + std::to_string(ratio) +
                              " alpha=" + std::to_string(alpha) + " rejects at " + std::to_string(bound / alpha) + " alpha");
            }
        }
    }
}

}  // namespace

PT_CHECK_MAIN("selftest")
