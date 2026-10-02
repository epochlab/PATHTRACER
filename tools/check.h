#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Shared runner for the validators, not a test framework: a check is addressable by name, reproducible from the CLI, and has one alpha.
namespace tools::check {

// Fast checks are analytic and sub-second, Slow ones carry the Monte Carlo sweeps. Binary, not a millisecond budget, which asserts perf.
enum class Speed { Fast, Slow };

// Exact checks assert counts, bit-identity or a forward-error bound and cannot flake; Statistical ones carry a confidence band.
enum class Kind { Exact, Statistical };

class Context;
using Fn = void (*)(Context&);

struct Registration {
    const char* name;
    Fn fn;
    Speed speed;
    Kind kind;
};

// Function-local static storage, so registration order across translation units cannot matter.
const std::vector<Registration>& registry();
void registerCheck(const Registration& registration);

struct Registrar {
    explicit Registrar(const Registration& registration) { registerCheck(registration); }
};

class Context {
public:
    Context(const Registration& registration, std::uint64_t suiteSeed, std::string_view suiteName, int threads);

    // This check's master seed, derived from its NAME not its registry position, so adding a check reshuffles no other check's realization.
    [[nodiscard]] std::uint64_t seed() const { return seed_; }
    // Per-case seed by the same name-derived argument one level down, so adding a row to a sweep leaves existing rows bit-identical.
    [[nodiscard]] std::uint64_t subSeed(std::string_view label) const;

    [[nodiscard]] int threads() const { return threads_; }

    // Declares how many assertions this check makes, so significance can be corrected for them. Call once, before asserting.
    void plan(int assertions);
    // Per-assertion two-sided significance, already corrected across the whole registry and this check's assertions.
    [[nodiscard]] double alpha() const;

    void recordPass() { ++asserted_; }
    void recordFailure(std::string_view expression, const std::string& detail, const char* file, int line);

    [[nodiscard]] int failures() const { return failures_; }
    [[nodiscard]] int asserted() const { return asserted_; }
    [[nodiscard]] int planned() const { return planned_; }

private:
    std::uint64_t seed_;
    int threads_;
    int planned_ = -1;
    int asserted_ = 0;
    int failures_ = 0;
};

// Family-wise error rate per binary: across B binaries the suite rate is 1-(1-alpha)^B, about 1.4e-3 at B=14, one red per 700 runs.
inline constexpr double kFamilyAlpha = 1e-4;

int run(int argc, char** argv, const char* suiteName);

}  // namespace tools::check

// Defines and registers a check. The body receives `ctx`.
#define PT_CHECK(checkName, speed, kind)                                                    \
    static void checkName(::tools::check::Context&);                                            \
    static const ::tools::check::Registrar checkName##_registrar{                               \
        {#checkName, &checkName, ::tools::check::Speed::speed, ::tools::check::Kind::kind}};    \
    static void checkName(::tools::check::Context& ctx)

#define PT_CHECK_MAIN(suiteName) \
    int main(int argc, char** argv) { return ::tools::check::run(argc, argv, suiteName); }

// The caller writes the PASS condition and the macro inverts it once: that is what makes NaN fail. Never restate these as failures.
#define PT_EXPECT(ctx, passExpression, detail)                                     \
    do {                                                                               \
        if (!(passExpression)) {                                                       \
            (ctx).recordFailure(#passExpression, (detail), __FILE__, __LINE__);         \
        } else {                                                                       \
            (ctx).recordPass();                                                        \
        }                                                                              \
    } while (false)
