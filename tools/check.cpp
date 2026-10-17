#include "check.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numbers>
#include <thread>

#include "stats.h"

namespace tools::check {
namespace {

// SplitMix64 (Steele, Lea & Flood, OOPSLA 2014): a full-avalanche finalizer, so a small structured input gives a well-separated seed.
std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// FNV-1a: only needs to map a name to a well-spread 64-bit value; SplitMix64 above supplies the avalanche.
std::uint64_t fnv1a64(std::string_view text) {
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    for (const char c : text) {
        hash = (hash ^ static_cast<unsigned char>(c)) * 0x100000001B3ULL;
    }
    return hash;
}

std::vector<Registration>& mutableRegistry() {
    static std::vector<Registration> checks;
    return checks;
}

const char* speedName(Speed speed) { return speed == Speed::Fast ? "fast" : "slow"; }
const char* kindName(Kind kind) { return kind == Kind::Exact ? "exact" : "statistical"; }

// An order of magnitude above the measured worst case in each class: a hang detector, never a performance assertion.
int timeoutSeconds(Speed speed) { return speed == Speed::Fast ? 60 : 900; }

// --calibrate's alternative: a band built from half its true variance, sqrt 2 too narrow, found with power 1 - kCalibrationMiss.
constexpr double kCalibrationVarianceShortfall = 2.0;
constexpr double kCalibrationMiss = 0.1;

// A two-sided normal band at false-rejection a, narrowed by sqrt of the shortfall, rejects at 2(1 - Phi(z_a / sqrt(shortfall))).
double narrowedRate(double rate) {
    const double z = stats::normalQuantile(1.0 - (0.5 * rate)) / std::sqrt(kCalibrationVarianceShortfall);
    return std::erfc(z / std::numbers::sqrt2);
}

// The rejection alpha one calibrated check carries: the family's, Sidak-split over the binary's Statistical checks.
double calibrationAlpha() {
    const auto statistical = std::count_if(registry().begin(), registry().end(),
                                           [](const Registration& check) { return check.kind == Kind::Statistical; });
    return stats::sidak(kFamilyAlpha, static_cast<int>(std::max<std::ptrdiff_t>(1, statistical)));
}

// Wald's SPRT (Wald 1945) of rate <= target against the narrowed band's rate, truncated at the fixed-sample test of the same errors.
struct CalibrationDesign {
    double target;
    double alternative;
    double rejectAlpha;
    int truncationRuns;
    int criticalFailures;
};

CalibrationDesign calibrationDesign() {
    const double rejectAlpha = calibrationAlpha();
    // Wald's expected sample size under H0: the target that minimises it, a ternary search on log rate where it is unimodal.
    const auto expectedRuns = [&](double logRate) {
        const double a = std::exp(logRate);
        const double b = narrowedRate(a);
        const double drift = (a * std::log(b / a)) + ((1.0 - a) * std::log((1.0 - b) / (1.0 - a)));
        return (((1.0 - rejectAlpha) * std::log(kCalibrationMiss / (1.0 - rejectAlpha))) +
                (rejectAlpha * std::log((1.0 - kCalibrationMiss) / rejectAlpha))) / drift;
    };
    double low = std::log(rejectAlpha);
    double high = std::log(0.5);
    for (int i = 0; i < 200; ++i) {
        const double left = low + ((high - low) / 3.0);
        const double right = high - ((high - low) / 3.0);
        if (expectedRuns(left) < expectedRuns(right)) {
            high = right;
        } else {
            low = left;
        }
    }
    CalibrationDesign design{std::exp(0.5 * (low + high)), 0.0, rejectAlpha, 0, 0};
    design.alternative = narrowedRate(design.target);
    for (int runs = 1;; ++runs) {
        int critical = 0;
        while (stats::binomialTail(critical, runs, design.target, false) > rejectAlpha) {
            ++critical;
        }
        if (critical <= runs && stats::binomialTail(critical, runs, design.alternative, false) >= 1.0 - kCalibrationMiss) {
            design.truncationRuns = runs;
            design.criticalFailures = critical;
            return design;
        }
    }
}

struct Calibration {
    int runs = 0;
    int failures = 0;
    bool miscalibrated = false;
    bool derived = false;  // every run asked ctx.alpha() for its band
};

// Runs one Statistical check under fresh seeds at the design's target until the SPRT decides, or its truncation does.
Calibration calibrate(const Registration& check, std::uint64_t suiteSeed, std::string_view suiteName, int threads,
                      const CalibrationDesign& design) {
    const double reject = std::log((1.0 - kCalibrationMiss) / design.rejectAlpha);
    const double accept = std::log(kCalibrationMiss / (1.0 - design.rejectAlpha));
    const double failStep = std::log(design.alternative / design.target);
    const double passStep = std::log((1.0 - design.alternative) / (1.0 - design.target));
    Calibration result;
    result.derived = true;
    double logLikelihood = 0.0;
    while (result.runs < design.truncationRuns) {
        Context ctx(check, splitmix64(suiteSeed + static_cast<std::uint64_t>(result.runs) + 1), suiteName, threads, design.target, true);
        check.fn(ctx);
        ++result.runs;
        const bool failed = ctx.failures() > 0 || (ctx.planned() >= 0 && ctx.asserted() != ctx.planned());
        result.failures += failed ? 1 : 0;
        result.derived = result.derived && ctx.alphaQueried();
        logLikelihood += failed ? failStep : passStep;
        if (logLikelihood >= reject || logLikelihood <= accept) {
            result.miscalibrated = logLikelihood >= reject;
            return result;
        }
    }
    result.miscalibrated = result.failures >= design.criticalFailures;
    return result;
}

}  // namespace

const std::vector<Registration>& registry() { return mutableRegistry(); }

void registerCheck(const Registration& registration) { mutableRegistry().push_back(registration); }

Context::Context(const Registration& registration, std::uint64_t suiteSeed, std::string_view suiteName, int threads, double perCheckAlpha,
                 bool quiet)
    : seed_(splitmix64(suiteSeed ^ fnv1a64(suiteName) ^ fnv1a64(registration.name))),
      threads_(threads),
      perCheckAlpha_(perCheckAlpha),
      quiet_(quiet) {}

std::uint64_t Context::subSeed(std::string_view label) const { return splitmix64(seed_ ^ fnv1a64(label)); }

void Context::plan(int assertions) { planned_ = assertions; }

double Context::alpha(std::source_location where) {
    alphaQueried_ = true;
    // Unsplit, each assertion would carry the whole check's alpha, so its false-rejection rate would grow with the plan.
    if (planned_ < 0) {
        recordFailure("ctx.plan() before ctx.alpha()", "", where.file_name(), static_cast<int>(where.line()));
    }
    return stats::sidak(perCheckAlpha_, planned_ > 0 ? planned_ : 1);
}

void Context::recordFailure(std::string_view expression, const std::string& detail, const char* file, int line) {
    ++failures_;
    ++asserted_;
    if (quiet_) {
        return;
    }
    std::cout << "    FAIL " << file << ":" << line << "  " << expression << "\n";
    if (!detail.empty()) {
        std::cout << "         " << detail << "\n";
    }
}

int run(int argc, char** argv, const char* suiteName) {
    std::uint64_t suiteSeed = 0x243F6A8885A308D3ULL;  // fixed: determinism is a requirement, so never clock-derived
    std::string exactName;
    std::string filter;
    // POST_BUILD discovery runs the binary by its full $<TARGET_FILE> path, so argv[0] is what ctest must invoke.
    const std::string exePath = argv[0] != nullptr ? argv[0] : suiteName;
    int threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    bool list = false;
    bool listCtest = false;
    bool calibration = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--list") {
            list = true;
        } else if (arg == "--list-ctest") {
            listCtest = true;
        } else if (arg == "--calibrate") {
            calibration = true;
        } else if (arg.starts_with("--check=")) {
            exactName = arg.substr(std::strlen("--check="));
        } else if (arg.starts_with("--filter=")) {
            filter = arg.substr(std::strlen("--filter="));
        } else if (arg.starts_with("--seed=")) {
            suiteSeed = std::strtoull(std::string(arg.substr(std::strlen("--seed="))).c_str(), nullptr, 10);
        } else if (arg.starts_with("--threads=")) {
            threads = std::max(1, std::atoi(std::string(arg.substr(std::strlen("--threads="))).c_str()));
        } else {
            std::cerr << suiteName << ": unrecognised argument " << arg
                      << "\nusage: " << suiteName
                      << " [--list] [--list-ctest] [--calibrate] [--check=NAME] [--filter=SUBSTRING] [--seed=N] [--threads=N]\n";
            return EXIT_FAILURE;
        }
    }

    if (list) {
        for (const Registration& check : registry()) {
            std::cout << check.name << "  " << speedName(check.speed) << "  " << kindName(check.kind) << "\n";
        }
        return EXIT_SUCCESS;
    }

    // Emits the ctest fragment TEST_INCLUDE_FILES consumes, the mechanism CMake's own gtest_discover_tests uses in POST_BUILD mode.
    if (listCtest) {
        const int calibrationRuns = calibrationDesign().truncationRuns;
        for (const Registration& check : registry()) {
            const int perTest = check.speed == Speed::Slow ? threads : 1;
            // Old-style positional add_test: the NAME/COMMAND signature exists only at configure time, not in an include script.
            std::cout << "add_test(" << suiteName << "." << check.name << " \"" << exePath << "\" --check="
                      << check.name << " --seed=" << suiteSeed << " --threads=" << perTest << ")\n"
                      << "set_tests_properties(" << suiteName << "." << check.name << " PROPERTIES LABELS \""
                      << suiteName << ";" << speedName(check.speed) << ";" << kindName(check.kind) << "\" TIMEOUT "
                      << timeoutSeconds(check.speed) << " PROCESSORS " << perTest << " SKIP_RETURN_CODE " << kSkipExitCode
                      << ")\n";
            if (check.kind != Kind::Statistical) {
                continue;
            }
            // CMake's CONFIGURATIONS guard, which an include script cannot spell: `ctest -C Calibration -L calibration`.
            const std::string name = std::string("calibration.") + suiteName + "." + check.name;
            std::cout << "if(CTEST_CONFIGURATION_TYPE MATCHES \"^[Cc][Aa][Ll][Ii][Bb][Rr][Aa][Tt][Ii][Oo][Nn]$\")\n"
                      << "  add_test(" << name << " \"" << exePath << "\" --calibrate --check=" << check.name << " --seed=" << suiteSeed
                      << " --threads=" << perTest << ")\n"
                      << "  set_tests_properties(" << name << " PROPERTIES LABELS \"calibration;" << suiteName << ";slow;statistical\" TIMEOUT "
                      << calibrationRuns * timeoutSeconds(check.speed) << " PROCESSORS " << perTest
                      << " SKIP_RETURN_CODE " << kSkipExitCode << ")\n"
                      << "endif()\n";
        }
        return EXIT_SUCCESS;
    }

    int failures = 0;
    int ran = 0;
    int skipped = 0;
    const CalibrationDesign design = calibration ? calibrationDesign() : CalibrationDesign{};
    const auto suiteStart = std::chrono::steady_clock::now();
    for (const Registration& check : registry()) {
        if (!exactName.empty() && exactName != check.name) {
            continue;
        }
        if (!filter.empty() && std::string_view(check.name).find(filter) == std::string_view::npos) {
            continue;
        }
        if (calibration) {
            if (check.kind != Kind::Statistical) {
                continue;
            }
            ++ran;
            const auto start = std::chrono::steady_clock::now();
            const Calibration result = calibrate(check, suiteSeed, suiteName, threads, design);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const stats::Band rate = stats::clopperPearsonBand(result.failures, result.runs, design.rejectAlpha);
            const char* verdict = !result.derived ? "UNDERIVED" : (result.miscalibrated ? "MISCALIBRATED" : "CALIBRATED");
            std::printf("  [%7.3fs] %-46s %s  (%d of %d runs failed, target %.3f against %.3f, rate in [%.3f, %.3f])\n", seconds,
                        check.name, verdict, result.failures, result.runs, design.target, design.alternative, rate.lo, rate.hi);
            failures += (result.miscalibrated || !result.derived) ? 1 : 0;
            continue;
        }
        ++ran;
        // Corrected across ALL registered checks, not the filtered subset, so `--check=x` runs at the suite's own significance.
        Context ctx(check, suiteSeed, suiteName, threads, stats::sidak(kFamilyAlpha, static_cast<int>(registry().size())), false);
        const auto start = std::chrono::steady_clock::now();
        check.fn(ctx);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        if (!ctx.skipReason().empty()) {
            ++skipped;
            std::printf("  [%7.3fs] %-46s SKIP  (%s)\n", seconds, check.name, ctx.skipReason().c_str());
            continue;
        }
        int checkFailures = ctx.failures();
        // A realized assertion count disagreeing with plan() was corrected for the wrong number of tests, so its bands are wrong.
        if (ctx.planned() >= 0 && ctx.asserted() != ctx.planned()) {
            std::cout << "    FAIL " << check.name << " planned " << ctx.planned() << " assertions but made "
                      << ctx.asserted() << "\n";
            ++checkFailures;
        }
        failures += checkFailures;
        std::printf("  [%7.3fs] %-46s %s  (%d assertions)\n", seconds, check.name,
                    checkFailures == 0 ? "PASS" : "FAIL", ctx.asserted());
    }

    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - suiteStart).count();
    if (ran == 0) {
        std::cerr << suiteName << ": no check matched\n";
        return EXIT_FAILURE;
    }
    std::printf("\n%s: %d check%s in %.3fs -- %s\n", suiteName, ran, ran == 1 ? "" : "s", total,
                failures != 0 ? "FAILURES PRESENT" : (skipped != 0 ? "passed, some skipped" : "all passed"));
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    return skipped != 0 ? kSkipExitCode : EXIT_SUCCESS;
}

}  // namespace tools::check
