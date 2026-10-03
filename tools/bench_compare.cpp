// Benchmark-log analysis: interleaved paired trials (Abedi & Brecht 2017), replicating per process invocation (Kalibera & Jones 2013).

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pathtracer/debug/bench_log.h"
#include "stats.h"

extern char** environ;  // NOLINT(readability-redundant-declaration) -- POSIX leaves it undeclared in <unistd.h> on Darwin

namespace {

using nlohmann::json;

struct Options {
    std::string command;
    std::string logPath;
    std::string a;
    std::string b;
    std::string tool;
    std::string metric;
    int rounds = 0;
    double alpha = 0.05;
    std::optional<std::uint64_t> seed;
    std::vector<std::string> childArgs;
};

constexpr const char* kUsage =
    "usage: bench_compare run --a BIN --b BIN --rounds N --log PATH [--metric M] [--alpha A] [--seed S] -- ARGS...\n"
    "         ARGS must make the child append to PATH (render_beauty/gbuffer_bench --bench-log PATH, pathtracer -bench PATH)\n"
    "       bench_compare compare --log PATH --a ID --b ID [--metric M] [--alpha A]\n"
    "       bench_compare history --log PATH --tool T [--metric M] [--alpha A]\n"
    "  ID: a build uuid prefix or git SHA; M: a samples column or rusage field (default: the only samples column)\n";

std::optional<Options> parseArgs(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << kUsage;
        return std::nullopt;
    }
    Options options;
    options.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--") {
            options.childArgs.assign(argv + i + 1, argv + argc);
            break;
        }
        if (i + 1 >= argc) {
            std::cerr << "bench_compare: " << flag << " expects a value\n" << kUsage;
            return std::nullopt;
        }
        const std::string value = argv[++i];
        char* end = nullptr;
        if (flag == "--log") {
            options.logPath = value;
        } else if (flag == "--a") {
            options.a = value;
        } else if (flag == "--b") {
            options.b = value;
        } else if (flag == "--tool") {
            options.tool = value;
        } else if (flag == "--metric") {
            options.metric = value;
        } else if (flag == "--rounds") {
            options.rounds = static_cast<int>(std::strtol(value.c_str(), &end, 10));
        } else if (flag == "--alpha") {
            options.alpha = std::strtod(value.c_str(), &end);
        } else if (flag == "--seed") {
            options.seed = std::strtoull(value.c_str(), &end, 10);
        } else {
            std::cerr << "bench_compare: unknown flag " << flag << '\n' << kUsage;
            return std::nullopt;
        }
        if (end != nullptr && (end == value.c_str() || *end != '\0')) {
            std::cerr << "bench_compare: " << flag << " expects a number, got " << value << '\n';
            return std::nullopt;
        }
    }
    const bool valid = !options.logPath.empty() && options.alpha > 0.0 && options.alpha < 1.0 &&
                       ((options.command == "run" && !options.a.empty() && !options.b.empty() && options.rounds > 0 &&
                         !options.childArgs.empty()) ||
                        (options.command == "compare" && !options.a.empty() && !options.b.empty()) ||
                        (options.command == "history" && !options.tool.empty()));
    if (!valid) {
        std::cerr << "bench_compare: missing or invalid arguments for '" << options.command << "'\n" << kUsage;
        return std::nullopt;
    }
    return options;
}

// Every line must parse at the current schema with every indexed object: a malformed or foreign record is an error, never skipped.
constexpr const char* kRecordObjects[] = {"build", "config", "samples", "rusage", "work"};

std::optional<std::vector<json>> loadLog(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "bench_compare: cannot read " << path << '\n';
        return std::nullopt;
    }
    std::vector<json> records;
    int lineNumber = 0;
    for (std::string line; std::getline(in, line);) {
        ++lineNumber;
        json record = json::parse(line, nullptr, /*allow_exceptions=*/false);
        const auto hasObjects = [&record] {
            return std::all_of(std::begin(kRecordObjects), std::end(kRecordObjects),
                               [&record](const char* key) { return record.contains(key) && record[key].is_object(); });
        };
        // Columns are read as numbers below, so they are type-checked here: loadLog is the one validation boundary.
        const auto numericColumns = [&record] {
            for (const auto& [name, column] : record["samples"].items()) {
                if (!column.is_array() || !std::all_of(column.begin(), column.end(), [](const json& v) { return v.is_number(); })) {
                    return false;
                }
            }
            return std::all_of(record["rusage"].begin(), record["rusage"].end(), [](const json& v) { return v.is_number(); });
        };
        // is_object() before value(): nlohmann's value() throws on a non-object, and a bare `null` or `[]` line parses fine.
        if (record.is_discarded() || !record.is_object() || record.value("schema", 0) != pathtracer::debug::kBenchLogSchema || !hasObjects() ||
            !record.contains("tool") || !record.contains("pid") || !numericColumns()) {
            std::cerr << "bench_compare: " << path << ':' << lineNumber << " is not a schema-"
                      << pathtracer::debug::kBenchLogSchema << " record\n";
            return std::nullopt;
        }
        records.push_back(std::move(record));
    }
    return records;
}

// A samples column's mean per event (NaN if empty), or a scalar rusage field; nullopt if the record carries neither.
std::optional<double> metricValue(const json& record, const std::string& metric) {
    const json& samples = record["samples"];
    if (samples.contains(metric)) {
        double total = 0.0;
        for (const json& v : samples[metric]) {
            total += v.get<double>();
        }
        return total / static_cast<double>(samples[metric].size());
    }
    if (record["rusage"].contains(metric)) {
        return record["rusage"][metric].get<double>();
    }
    return std::nullopt;
}

std::optional<std::string> resolveMetric(const std::string& requested, const json& record) {
    if (!requested.empty()) {
        if (!metricValue(record, requested)) {
            std::cerr << "bench_compare: metric '" << requested << "' is neither a samples column nor a rusage field\n";
            return std::nullopt;
        }
        return requested;
    }
    if (record["samples"].size() != 1) {
        std::cerr << "bench_compare: record has " << record["samples"].size() << " samples columns; choose one with --metric:";
        for (const auto& [name, column] : record["samples"].items()) {
            std::cerr << ' ' << name;
        }
        std::cerr << '\n';
        return std::nullopt;
    }
    return record["samples"].begin().key();
}

bool matchesId(const json& record, const std::string& id) {
    const std::string uuid = record["build"].value("uuid", std::string());
    const std::string git = record["build"].value("git", std::string());
    return uuid.starts_with(id) || git == id || git.starts_with(id + "+");
}

std::string shortId(const json& record) {
    return record["build"].value("uuid", std::string()).substr(0, 8) + " " + record["build"].value("git", std::string());
}

// A log-ratio needs a strictly positive value from every record, and config equality does not constrain the column set.
bool positiveValues(const std::vector<const json*>& group, const std::string& metric) {
    for (const json* r : group) {
        const std::optional<double> value = metricValue(*r, metric);
        if (!value) {
            std::cerr << "bench_compare: " << shortId(*r) << " pid " << (*r).value("pid", -1) << " has no metric " << metric
                      << "; the two builds do not record the same columns\n";
            return false;
        }
        if (!(*value > 0.0)) {
            std::cerr << "bench_compare: " << shortId(*r) << " pid " << (*r).value("pid", -1) << " has a non-positive or empty " << metric
                      << "; no ratio exists\n";
            return false;
        }
    }
    return true;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return tools::stats::sortedMedian(values);
}

// Ratio B/A from a shift on the log scale, with its verdict; a CI containing 1 is reported as unresolved, not as a winner.
void printRatio(const char* label, const tools::stats::ShiftEstimate& logShift, double alpha, int n) {
    const double ratio = std::exp(logShift.estimate);
    if (std::isinf(logShift.lower)) {
        std::printf("%s B/A = %.4f, no %.0f%% interval exists at n = %d; add rounds\n", label, ratio,
                    100.0 * (1.0 - alpha), n);
        return;
    }
    const double lo = std::exp(logShift.lower);
    const double hi = std::exp(logShift.upper);
    std::printf("%s B/A = %.4f  [%.4f, %.4f] at %.1f%% exact coverage, n = %d\n", label, ratio, lo, hi,
                100.0 * logShift.coverage, n);
    if (lo > 1.0) {
        std::printf("  verdict: B slower by %.2f%% (resolved)\n", 100.0 * (ratio - 1.0));
    } else if (hi < 1.0) {
        std::printf("  verdict: B faster by %.2f%% (resolved)\n", 100.0 * (1.0 - ratio));
    } else {
        std::printf("  verdict: not resolved at n = %d -- the interval contains 1\n", n);
    }
}

void printGroup(const char* name, const std::vector<const json*>& group, const std::string& metric) {
    std::vector<double> values;
    std::vector<double> preemptions;
    for (const json* r : group) {
        values.push_back(*metricValue(*r, metric));
        preemptions.push_back((*r)["rusage"].value("nivcsw", 0.0));
    }
    std::printf("  %s %s  n = %zu  median %s %.6g  median nivcsw %.0f\n", name, shortId(*group.front()).c_str(),
                group.size(), metric.c_str(), median(values), median(preemptions));
}

// Identical work is what makes a timing ratio mean "faster at the same job"; a differing CRC is reported, not fatal.
void printWork(const std::vector<const json*>& a, const std::vector<const json*>& b) {
    const json& reference = (*a.front())["work"];
    const auto same = [&](const json* r) { return (*r)["work"] == reference; };
    const bool identical = std::all_of(a.begin(), a.end(), same) && std::all_of(b.begin(), b.end(), same);
    std::printf("  work: %s\n", identical ? "identical across every run" : "DIFFERS -- A and B did not compute the same thing");
}

std::optional<pid_t> spawnQuiet(const std::string& binary, const std::vector<std::string>& args) {
    std::vector<std::string> storage{binary};
    storage.insert(storage.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (std::string& s : storage) {
        argv.push_back(s.data());
    }
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);  // stderr stays live for failures
    pid_t pid = 0;
    const int status = posix_spawn(&pid, binary.c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (status != 0) {
        std::cerr << "bench_compare: cannot spawn " << binary << ": " << std::strerror(status) << '\n';
        return std::nullopt;
    }
    return pid;
}

// Runs one child to completion and returns the one record it appended, found by pid among lines written after it started.
std::optional<json> runOnce(const std::string& binary, const Options& options) {
    std::size_t recordsBefore = 0;
    if (std::filesystem::exists(options.logPath)) {
        const std::optional<std::vector<json>> before = loadLog(options.logPath);
        if (!before) {
            return std::nullopt;
        }
        recordsBefore = before->size();
    }
    const std::optional<pid_t> pid = spawnQuiet(binary, options.childArgs);
    if (!pid) {
        return std::nullopt;
    }
    int status = 0;
    if (waitpid(*pid, &status, 0) != *pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::cerr << "bench_compare: " << binary << " failed (status " << status << "); aborting the whole run\n";
        return std::nullopt;
    }
    const std::optional<std::vector<json>> after = loadLog(options.logPath);
    if (!after) {
        return std::nullopt;
    }
    for (std::size_t i = recordsBefore; i < after->size(); ++i) {
        if ((*after)[i].value("pid", -1) == *pid) {
            return (*after)[i];
        }
    }
    std::cerr << "bench_compare: " << binary << " exited cleanly but wrote no record to " << options.logPath
              << "; the child arguments must name that log\n";
    return std::nullopt;
}

int runCommand(const Options& options) {
    const std::uint64_t seed = options.seed.value_or(std::random_device{}());
    std::printf("bench_compare: %d rounds, order seed %llu (replay with --seed)\n", options.rounds,
                static_cast<unsigned long long>(seed));
    std::mt19937_64 rng(seed);
    std::vector<json> aRuns;
    std::vector<json> bRuns;
    for (int round = 0; round < options.rounds; ++round) {
        const bool aFirst = (rng() & 1U) == 0;
        for (const bool runA : {aFirst, !aFirst}) {
            std::optional<json> record = runOnce(runA ? options.a : options.b, options);
            if (!record) {
                return EXIT_FAILURE;
            }
            (runA ? aRuns : bRuns).push_back(std::move(*record));
        }
        if (aRuns.back()["config"] != bRuns.back()["config"]) {
            std::cerr << "bench_compare: A and B resolved different configs from the same arguments; not the same workload\n";
            return EXIT_FAILURE;
        }
        std::fprintf(stderr, "\r  round %d/%d", round + 1, options.rounds);
    }
    std::fprintf(stderr, "\n");

    const std::optional<std::string> metric = resolveMetric(options.metric, aRuns.front());
    if (!metric) {
        return EXIT_FAILURE;
    }
    std::vector<const json*> aGroup;
    std::vector<const json*> bGroup;
    for (std::size_t i = 0; i < aRuns.size(); ++i) {
        aGroup.push_back(&aRuns[i]);
        bGroup.push_back(&bRuns[i]);
    }
    if (!positiveValues(aGroup, *metric) || !positiveValues(bGroup, *metric)) {
        return EXIT_FAILURE;
    }
    std::vector<double> logRatios;
    for (std::size_t i = 0; i < aRuns.size(); ++i) {
        logRatios.push_back(std::log(*metricValue(bRuns[i], *metric) / *metricValue(aRuns[i], *metric)));
    }
    printRatio(("paired " + *metric).c_str(), tools::stats::hodgesLehmannPaired(logRatios, options.alpha),
               options.alpha, options.rounds);
    printGroup("A", aGroup, *metric);
    printGroup("B", bGroup, *metric);
    printWork(aGroup, bGroup);
    return EXIT_SUCCESS;
}

// Selects records of `id` sharing `config`; an id spanning several builds is ambiguous and refused.
std::optional<std::vector<const json*>> selectGroup(const std::vector<json>& records, const std::string& id,
                                                   const json& config) {
    std::vector<const json*> group;
    for (const json& r : records) {
        if (matchesId(r, id) && r["config"] == config) {
            if (!group.empty() && r["build"].value("uuid", std::string()) != (*group.front())["build"].value("uuid", std::string())) {
                std::cerr << "bench_compare: '" << id << "' matches builds " << shortId(*group.front()) << " and "
                          << shortId(r) << "; use a uuid prefix\n";
                return std::nullopt;
            }
            group.push_back(&r);
        }
    }
    if (group.empty()) {
        std::cerr << "bench_compare: no record of '" << id << "' at the selected config\n";
        return std::nullopt;
    }
    return group;
}

std::vector<double> logTotals(const std::vector<const json*>& group, const std::string& metric) {
    std::vector<double> values;
    for (const json* r : group) {
        values.push_back(std::log(*metricValue(*r, metric)));
    }
    return values;
}

int compareCommand(const Options& options, const std::vector<json>& records) {
    // The workload is the latest A record's config; records at any other config are not comparable and are left out.
    const auto latestA = std::find_if(records.rbegin(), records.rend(), [&](const json& r) { return matchesId(r, options.a); });
    if (latestA == records.rend()) {
        std::cerr << "bench_compare: no record of '" << options.a << "'\n";
        return EXIT_FAILURE;
    }
    const std::optional<std::vector<const json*>> a = selectGroup(records, options.a, (*latestA)["config"]);
    const std::optional<std::vector<const json*>> b = selectGroup(records, options.b, (*latestA)["config"]);
    if (!a || !b) {
        return EXIT_FAILURE;
    }
    const std::optional<std::string> metric = resolveMetric(options.metric, *a->front());
    if (!metric || !positiveValues(*a, *metric) || !positiveValues(*b, *metric)) {
        return EXIT_FAILURE;
    }
    std::printf("bench_compare: unpaired -- drift between the two groups is not controlled; prefer 'run' for a claim\n");
    printRatio(("unpaired " + *metric).c_str(),
               tools::stats::hodgesLehmannShift(logTotals(*a, *metric), logTotals(*b, *metric), options.alpha),
               options.alpha, static_cast<int>(std::min(a->size(), b->size())));
    printGroup("A", *a, *metric);
    printGroup("B", *b, *metric);
    printWork(*a, *b);
    return EXIT_SUCCESS;
}

int historyCommand(const Options& options, const std::vector<json>& records) {
    const auto latest = std::find_if(records.rbegin(), records.rend(), [&](const json& r) { return r["tool"] == options.tool; });
    if (latest == records.rend()) {
        std::cerr << "bench_compare: no " << options.tool << " records in " << options.logPath << '\n';
        return EXIT_FAILURE;
    }
    const json& config = (*latest)["config"];
    const std::optional<std::string> metric = resolveMetric(options.metric, *latest);
    if (!metric) {
        return EXIT_FAILURE;
    }
    // One row per build, in order of first appearance, at the latest record's workload.
    std::vector<std::vector<const json*>> builds;
    std::size_t excluded = 0;
    for (const json& r : records) {
        if (r["tool"] != options.tool) {
            continue;
        }
        if (r["config"] != config) {
            ++excluded;
            continue;
        }
        const auto sameBuild = [&](const std::vector<const json*>& g) {
            return (*g.front())["build"].value("uuid", std::string()) == r["build"].value("uuid", std::string());
        };
        const auto it = std::find_if(builds.begin(), builds.end(), sameBuild);
        if (it == builds.end()) {
            builds.emplace_back().push_back(&r);
        } else {
            it->push_back(&r);
        }
    }
    if (!std::all_of(builds.begin(), builds.end(), [&](const std::vector<const json*>& g) { return positiveValues(g, *metric); })) {
        return EXIT_FAILURE;
    }
    std::printf("bench_compare: %s history, metric %s, config %s (%zu records at other configs excluded)\n",
                options.tool.c_str(), metric->c_str(), config.dump().c_str(), excluded);
    for (std::size_t i = 0; i < builds.size(); ++i) {
        const std::vector<const json*>& g = builds[i];
        std::printf("%s  %-28s n = %-3zu median %.6g\n", (*g.front()).value("time_utc", std::string()).c_str(),
                    shortId(*g.front()).c_str(), g.size(), std::exp(median(logTotals(g, *metric))));
        if (i > 0) {
            printRatio("    vs previous:", tools::stats::hodgesLehmannShift(logTotals(builds[i - 1], *metric), logTotals(g, *metric), options.alpha),
                       options.alpha, static_cast<int>(std::min(builds[i - 1].size(), g.size())));
        }
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    const std::optional<Options> options = parseArgs(argc, argv);
    if (!options) {
        return EXIT_FAILURE;
    }
    if (options->command == "run") {
        return runCommand(*options);
    }
    const std::optional<std::vector<json>> records = loadLog(options->logPath);
    if (!records) {
        return EXIT_FAILURE;
    }
    return options->command == "compare" ? compareCommand(*options, *records) : historyCommand(*options, *records);
}
