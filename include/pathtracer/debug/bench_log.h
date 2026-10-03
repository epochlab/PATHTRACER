#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace pathtracer::debug {

// Schema version of the JSON Lines benchmark log; bump on any field rename or meaning change so a reader can refuse a record.
inline constexpr int kBenchLogSchema = 2;

// The benchmark workload every tool defaults to, so logs from different tools and days stay comparable: 512x256 at 16 samples per pixel.
inline constexpr int kBenchWidth = 512;
inline constexpr int kBenchHeight = 256;
inline constexpr int kBenchSamples = 16;

// One timing run's tool-specific content. appendBenchRecord adds provenance (build, host) and rusage itself, so no caller can omit them.
struct BenchRecord {
    std::string tool;
    std::vector<std::string> argv;
    nlohmann::json config;   // resolved inputs only, never output paths: two runs are comparable iff their configs are equal
    nlohmann::json samples;  // column name -> raw per-iteration values; summaries are the reader's job (Kalibera & Jones 2013)
    nlohmann::json work;     // what was computed (ray counts, output CRC), so an A/B can tell whether both sides did identical work
};

// Appends `record` as one line to `path`, created if absent, in a single O_APPEND write(2), so a line is never half-written.
[[nodiscard]] bool appendBenchRecord(const std::string& path, const BenchRecord& record);

// zlib CRC-32 of a float buffer's bytes: equal CRCs across an A/B mean both sides produced bit-identical output.
[[nodiscard]] std::uint32_t floatCrc32(std::span<const float> values);

}  // namespace pathtracer::debug
