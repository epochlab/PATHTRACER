// Correctness checks for volume.cpp: VDB loading and validation, the majorant grid, ratio and spectral tracking, Henyey-Greenstein.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>

#include "check.h"
#include "fixtures.h"
#include "pathtracer/scene/rotation.h"
#include "pathtracer/scene/sampler.h"
#include "pathtracer/scene/volume.h"
#include "stats.h"
#include "volume_majorant.h"

namespace {

using pathtracer::config::VolumeConfig;
using pathtracer::scene::FreeFlightEvent;
using pathtracer::scene::Ray;
using pathtracer::scene::Sampler;
using pathtracer::scene::VolumeSet;
using tools::fixtures::kPi;

// A per-check directory under the system temp: grids reach the renderer as files, through the one loading path a scene uses.
class Scratch {
public:
    explicit Scratch(const tools::check::Context& ctx)
        : root_(std::filesystem::temp_directory_path() / ("volume_validate_" + std::to_string(ctx.seed()))) {
        openvdb::initialize();  // registers the grid types file I/O needs
        std::filesystem::create_directories(root_);
    }
    ~Scratch() { std::filesystem::remove_all(root_); }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

    [[nodiscard]] std::string root() const { return root_.string(); }

    // Writes `grids` to <root>/<name>.vdb and returns the asset-relative path.
    std::string writeGrids(const std::string& name, const openvdb::GridPtrVec& grids) const {
        openvdb::io::File file((root_ / (name + ".vdb")).string());
        file.write(grids);
        file.close();
        return name + ".vdb";
    }

    std::string writeText(const std::string& name, const std::string& text) const {
        std::ofstream((root_ / name).string()) << text;
        return name;
    }

    // A volume material file with the given coefficients, per grid unit at density 1.
    std::string writeMaterial(const std::string& name, const glm::vec3& absorption, const glm::vec3& scattering, float anisotropy) const {
        const auto rgb = [](const glm::vec3& c) {
            return "[" + std::to_string(c.x) + ", " + std::to_string(c.y) + ", " + std::to_string(c.z) + "]";
        };
        return writeText(name + ".json", "{\"absorption\": " + rgb(absorption) + ", \"scattering\": " + rgb(scattering) +
                                             ", \"anisotropy\": " + std::to_string(anisotropy) + "}");
    }

private:
    std::filesystem::path root_;
};

openvdb::FloatGrid::Ptr makeGrid(double voxelSize) {
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0F);
    grid->setName("density");
    grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(voxelSize));
    return grid;
}

// Density 1 over index voxels [0, n)^3: the trilinear field is 1 inside and ramps to 0 over one voxel past each face.
openvdb::FloatGrid::Ptr makeBox(int n, double voxelSize) {
    openvdb::FloatGrid::Ptr grid = makeGrid(voxelSize);
    grid->tree().fill(openvdb::CoordBBox(openvdb::Coord(0), openvdb::Coord(n - 1)), 1.0F, /*active=*/true);
    return grid;
}

VolumeConfig volumeConfig(const std::string& vdbPath, const std::string& materialPath) {
    return VolumeConfig{vdbPath, "density", glm::vec3(0.0F), glm::vec3(0.0F), 1.0F, materialPath};
}

VolumeSet loadOrDie(const std::vector<VolumeConfig>& configs, const std::string& root) {
    std::string error;
    std::optional<VolumeSet> set = VolumeSet::load(configs, root, error);
    if (!set) {
        std::fprintf(stderr, "volume_validate: %s\n", error.c_str());
        std::abort();
    }
    return *set;
}

// The config's own placement, transcribed: translate * rotationXyz * scale, the map from grid space to world.
glm::mat4 worldFromGrid(const VolumeConfig& config) {
    return glm::translate(glm::mat4(1.0F), config.position) * glm::mat4(pathtracer::scene::rotationXyz(config.rotation)) *
           glm::scale(glm::mat4(1.0F), glm::vec3(config.scale));
}

// Every value the trilinear stencil at index point p reads must sit at or below the majorant of p's cell, floor(p/8).
PT_CHECK(majorant_bounds_every_trilinear_stencil, Slow, Exact) {
    openvdb::initialize();
    openvdb::FloatGrid::Ptr grid = makeGrid(1.0);
    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
    std::mt19937 rng(static_cast<std::uint32_t>(ctx.seed()));
    std::uniform_int_distribution<int> coordinate(-40, 40);
    std::uniform_real_distribution<float> value(0.0F, 1.0F);
    for (int i = 0; i < 400; ++i) {
        accessor.setValue(openvdb::Coord(coordinate(rng), coordinate(rng), coordinate(rng)), value(rng));
    }
    // Spikes on both sides of leaf faces, at negative coordinates too, where truncating division would misplace them.
    for (const int c : {-17, -16, -9, -8, -1, 0, 7, 8, 15, 16}) {
        accessor.setValue(openvdb::Coord(c, c, -c), 4.0F);
    }
    accessor.setValueOff(openvdb::Coord(-30, 5, 22), 8.0F);  // inactive, but the sampler still reads it
    // A 128^3 tile clear of the rest, so its interior cells come from the fill and its shell from the max-merge.
    grid->tree().fill(openvdb::CoordBBox(openvdb::Coord(64), openvdb::Coord(191)), 0.5F, /*active=*/true);

    std::string error;
    const openvdb::FloatGrid::Ptr majorant = pathtracer::scene::buildMajorantGrid(*grid, error);
    openvdb::FloatGrid::ConstAccessor cells = majorant->getConstAccessor();
    openvdb::FloatGrid::ConstAccessor density = grid->getConstAccessor();
    const int cell = pathtracer::scene::kMajorantCellVoxels;
    long long violations = 0;
    long long stencils = 0;
    const openvdb::CoordBBox span(openvdb::Coord(-7), openvdb::Coord(25));
    // A stencil depends only on floor(p) - 8c, which over the guaranteed [8c-1, 8c+9) takes exactly the ten bases -1..8: all of them.
    const openvdb::CoordBBox bases(openvdb::Coord(-1), openvdb::Coord(cell));
    for (auto it = span.begin(); it; ++it) {
        const openvdb::Coord c = *it;
        const float bound = cells.getValue(c);
        for (auto offset = bases.begin(); offset; ++offset) {
            const openvdb::Coord base = openvdb::Coord(c.x() * cell, c.y() * cell, c.z() * cell) + *offset;
            for (int corner = 0; corner < 8; ++corner) {
                const openvdb::Coord read = base + openvdb::Coord(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
                violations += density.getValue(read) > bound ? 1 : 0;
            }
            ++stencils;
        }
    }
    ctx.plan(3);
    PT_EXPECT(ctx, majorant != nullptr, error);
    PT_EXPECT(ctx, violations == 0, std::to_string(violations) + " stencil reads above their cell's majorant");
    PT_EXPECT(ctx, stencils > 0, "no stencils visited");
}

// One ray through a chromatic box and a linear ramp, placed, rotated, scaled and with a non-unit direction: ratio tracking's mean.
PT_CHECK(transmittance_matches_closed_form, Slow, Statistical) {
    const Scratch scratch(ctx);
    constexpr int kVoxels = 64;
    constexpr double kVoxelSize = 0.5;
    openvdb::FloatGrid::Ptr ramp = makeBox(kVoxels, kVoxelSize);
    ramp->setName("ramp");
    // Density i/63 at voxel column i: linear inside, so trilinear is exact there, and 0.5 * 1 * one voxel past the dense face.
    openvdb::FloatGrid::Accessor rampAccessor = ramp->getAccessor();
    for (auto it = openvdb::CoordBBox(openvdb::Coord(0), openvdb::Coord(kVoxels - 1)).begin(); it; ++it) {
        rampAccessor.setValue(*it, static_cast<float>((*it).x()) / static_cast<float>(kVoxels - 1));
    }
    const std::string vdb = scratch.writeGrids("box", {makeBox(kVoxels, kVoxelSize), ramp});
    const glm::vec3 absorption(0.02F, 0.04F, 0.07F);
    const glm::vec3 scattering(0.05F, 0.0F, 0.01F);
    const std::string material = scratch.writeMaterial("medium", absorption, scattering, 0.3F);
    // In voxels along x: box 63 + 2 * 0.5, ramp 31.5 + 0.5; times the voxel size for grid units, in which the coefficients are given.
    const std::array<std::pair<std::string, double>, 2> grids{{{"density", 64.0}, {"ramp", 32.0}}};

    constexpr int kTrials = 4096;
    ctx.plan(6);
    for (const auto& [name, voxels] : grids) {
        VolumeConfig config = volumeConfig(vdb, material);
        config.grid = name;
        config.position = glm::vec3(1.0F, -2.0F, 0.5F);
        config.rotation = glm::vec3(30.0F, 40.0F, 50.0F);
        config.scale = 2.0F;
        const VolumeSet set = loadOrDie({config}, scratch.root());
        // A row at the box's y, z centre, from well outside to well past it, mapped to world with a length-3 direction.
        const glm::mat4 toWorld = worldFromGrid(config);
        const float centre = 0.5F * static_cast<float>(kVoxels - 1) * static_cast<float>(kVoxelSize);
        const glm::vec3 start(toWorld * glm::vec4(-8.0F, centre, centre, 1.0F));
        const glm::vec3 end(toWorld * glm::vec4((static_cast<float>(kVoxels) * static_cast<float>(kVoxelSize)) + 8.0F, centre, centre, 1.0F));
        const glm::vec3 dir = 3.0F * glm::normalize(end - start);
        const Ray ray{start, dir, 0.0F, std::numeric_limits<float>::infinity()};
        const double depth = voxels * kVoxelSize;
        std::array<tools::stats::Welford, 3> estimate;
        for (int i = 0; i < kTrials; ++i) {
            Sampler sampler(0, 0, i, kTrials, static_cast<std::uint32_t>(ctx.subSeed(name)));
            const glm::vec3 t = set.transmittance(ray, 0.0F, std::numeric_limits<float>::infinity(), sampler);
            for (int c = 0; c < 3; ++c) {
                estimate[c].add(t[c]);
            }
        }
        for (int c = 0; c < 3; ++c) {
            const double expected = std::exp(-static_cast<double>(absorption[c] + scattering[c]) * depth);
            const double half = tools::stats::studentTTwoSided(ctx.alpha(), static_cast<double>(kTrials - 1)) * estimate[c].standardError();
            PT_EXPECT(ctx, std::abs(estimate[c].mean() - expected) <= half,
                      name + " channel " + std::to_string(c) + ": " + std::to_string(estimate[c].mean()) + " vs " + std::to_string(expected));
        }
    }
}

// Spectral tracking from inside a chromatic box: E[w 1{escape}] = T and E[w 1{scatter}] = (sigma_s/sigma_t)(1 - T) per channel.
PT_CHECK(spectral_tracking_is_unbiased, Slow, Statistical) {
    const Scratch scratch(ctx);
    const std::string vdb = scratch.writeGrids("box", {makeBox(64, 1.0)});
    const glm::vec3 absorption(0.01F, 0.05F, 0.02F);
    const glm::vec3 scattering(0.06F, 0.01F, 0.03F);
    const VolumeSet set = loadOrDie({volumeConfig(vdb, scratch.writeMaterial("medium", absorption, scattering, 0.0F))}, scratch.root());
    // Starts and ends inside, where the density is exactly 1, as a camera inside the medium and a surface hit within it.
    constexpr float kLength = 40.0F;
    const Ray ray{glm::vec3(10.0F, 31.0F, 31.0F), glm::vec3(1.0F, 0.0F, 0.0F), 0.0F, kLength};
    const glm::vec3 throughput(0.7F, 1.0F, 0.4F);  // a coloured history, which the probabilities follow and the weights undo

    constexpr int kTrials = 1 << 15;
    std::array<tools::stats::Welford, 3> escaped;
    std::array<tools::stats::Welford, 3> scattered;
    tools::stats::Welford inside;
    for (int i = 0; i < kTrials; ++i) {
        Sampler sampler(0, 0, i, kTrials, static_cast<std::uint32_t>(ctx.seed()));
        const pathtracer::scene::FreeFlight flight = set.sampleFreeFlight(ray, 0.0F, kLength, throughput, sampler);
        for (int c = 0; c < 3; ++c) {
            escaped[c].add(flight.event == FreeFlightEvent::Escaped ? flight.weight[c] : 0.0F);
            scattered[c].add(flight.event == FreeFlightEvent::Scattered ? flight.weight[c] : 0.0F);
        }
        inside.add(flight.event != FreeFlightEvent::Scattered || (flight.t >= 0.0F && flight.t < kLength) ? 1.0 : 0.0);
    }
    ctx.plan(7);
    const double t = tools::stats::studentTTwoSided(ctx.alpha(), static_cast<double>(kTrials - 1));
    for (int c = 0; c < 3; ++c) {
        const double extinction = static_cast<double>(absorption[c] + scattering[c]);
        const double transmittance = std::exp(-extinction * kLength);
        const double scatter = (static_cast<double>(scattering[c]) / extinction) * (1.0 - transmittance);
        PT_EXPECT(ctx, std::abs(escaped[c].mean() - transmittance) <= t * escaped[c].standardError(),
                  "escape channel " + std::to_string(c) + ": " + std::to_string(escaped[c].mean()) + " vs " + std::to_string(transmittance));
        PT_EXPECT(ctx, std::abs(scattered[c].mean() - scatter) <= t * scattered[c].standardError(),
                  "scatter channel " + std::to_string(c) + ": " + std::to_string(scattered[c].mean()) + " vs " + std::to_string(scatter));
    }
    PT_EXPECT(ctx, inside.mean() == 1.0, "a scattering fell outside [t0, t1)");
}

// A grey scattering-only medium's free-flight distances binned against 1 - exp(-sigma t): the exponential law itself, not its mean.
PT_CHECK(free_flight_distance_distribution, Slow, Statistical) {
    const Scratch scratch(ctx);
    const std::string vdb = scratch.writeGrids("box", {makeBox(64, 1.0)});
    constexpr float kSigma = 0.1F;
    const VolumeSet set =
        loadOrDie({volumeConfig(vdb, scratch.writeMaterial("medium", glm::vec3(0.0F), glm::vec3(kSigma), 0.0F))}, scratch.root());
    constexpr float kLength = 30.0F;
    const Ray ray{glm::vec3(31.0F, 31.0F, 15.0F), glm::vec3(0.0F, 0.0F, 1.0F), 0.0F, kLength};
    constexpr int kBins = 16;  // equal-width in t, plus one bin for the escapes
    constexpr int kTrials = 1 << 15;
    std::array<int, kBins + 1> counts{};
    bool unitWeights = true;
    for (int i = 0; i < kTrials; ++i) {
        Sampler sampler(0, 0, i, kTrials, static_cast<std::uint32_t>(ctx.seed()));
        const pathtracer::scene::FreeFlight flight = set.sampleFreeFlight(ray, 0.0F, kLength, glm::vec3(1.0F), sampler);
        unitWeights = unitWeights && flight.weight == glm::vec3(1.0F);
        const int bin = flight.event == FreeFlightEvent::Scattered
                            ? std::min(static_cast<int>(flight.t / kLength * static_cast<float>(kBins)), kBins - 1)
                            : kBins;
        ++counts[static_cast<std::size_t>(bin)];
    }
    double chi2 = 0.0;
    for (int b = 0; b <= kBins; ++b) {
        const double lo = static_cast<double>(b) * kLength / kBins;
        const double hi = static_cast<double>(b + 1) * kLength / kBins;
        const double p = b < kBins ? std::exp(-kSigma * lo) - std::exp(-kSigma * hi) : std::exp(-static_cast<double>(kSigma) * kLength);
        const double expected = p * kTrials;
        chi2 += (counts[static_cast<std::size_t>(b)] - expected) * (counts[static_cast<std::size_t>(b)] - expected) / expected;
    }
    ctx.plan(2);
    PT_EXPECT(ctx, tools::stats::chiSquareUpperTail(chi2, kBins) > ctx.alpha(), "chi2 " + std::to_string(chi2));
    PT_EXPECT(ctx, unitWeights, "a grey medium's spectral weight left 1");
}

// The pdf integrates to 1, E[cos] = g pins the sign convention, and drawn cosines bin to the pdf's own mass, for g across its range.
PT_CHECK(henyey_greenstein_sampling, Slow, Statistical) {
    constexpr int kBins = 32;
    constexpr int kTrials = 1 << 16;
    const std::array<float, 5> gs{-0.8F, -0.3F, 0.0F, 0.5F, 0.877F};
    ctx.plan(static_cast<int>(gs.size()) * 3);
    for (const float g : gs) {
        // Simpson over cos in [-1, 1] at 2 * 4096 panels: the pdf is smooth for |g| < 1, so the rule's error is far below the 1e-4 asked.
        constexpr int kPanels = 8192;
        double mass = 0.0;
        for (int i = 0; i <= kPanels; ++i) {
            const double weight = (i == 0 || i == kPanels) ? 1.0 : ((i % 2) != 0 ? 4.0 : 2.0);
            mass += weight * pathtracer::scene::henyeyGreenstein(-1.0F + (2.0F * static_cast<float>(i) / kPanels), g);
        }
        mass *= 2.0 * kPi * (2.0 / kPanels) / 3.0;
        const glm::vec3 dir = glm::normalize(glm::vec3(0.3F, -0.5F, 0.8F));
        std::array<int, kBins> counts{};
        tools::stats::Welford meanCos;
        for (int i = 0; i < kTrials; ++i) {
            Sampler sampler(0, 0, i, kTrials, static_cast<std::uint32_t>(ctx.subSeed(std::to_string(g))));
            const glm::vec3 wi = pathtracer::scene::sampleHenyeyGreenstein(dir, g, sampler.next2D());
            const float cosTheta = glm::dot(dir, wi);
            meanCos.add(cosTheta);
            ++counts[static_cast<std::size_t>(std::clamp(static_cast<int>((cosTheta + 1.0F) * 0.5F * kBins), 0, kBins - 1))];
        }
        // Each bin's mass is the closed-form CDF difference: P(cos < x) = (1 - g^2)/(2g) (1/sqrt(1 + g^2 - 2gx) - 1/(1 + g)), or (x+1)/2.
        const auto cdf = [g](double x) {
            if (g == 0.0F) {
                return 0.5 * (x + 1.0);
            }
            const double gd = g;
            return ((1.0 - (gd * gd)) / (2.0 * gd)) * ((1.0 / std::sqrt(1.0 + (gd * gd) - (2.0 * gd * x))) - (1.0 / (1.0 + gd)));
        };
        double chi2 = 0.0;
        int dof = -1;
        for (int b = 0; b < kBins; ++b) {
            const double expected = kTrials * (cdf(-1.0 + (2.0 * (b + 1) / kBins)) - cdf(-1.0 + (2.0 * b / kBins)));
            if (expected >= 5.0) {
                chi2 += (counts[static_cast<std::size_t>(b)] - expected) * (counts[static_cast<std::size_t>(b)] - expected) / expected;
                ++dof;
            }
        }
        const double t = tools::stats::studentTTwoSided(ctx.alpha(), static_cast<double>(kTrials - 1));
        const std::string label = "g " + std::to_string(g);
        PT_EXPECT(ctx, std::abs(mass - 1.0) < 1e-4, label + ": pdf mass " + std::to_string(mass));
        PT_EXPECT(ctx, std::abs(meanCos.mean() - g) <= t * meanCos.standardError(), label + ": E[cos] " + std::to_string(meanCos.mean()));
        PT_EXPECT(ctx, tools::stats::chiSquareUpperTail(chi2, dof) > ctx.alpha(), label + ": chi2 " + std::to_string(chi2));
    }
}

// A ray between two blobs crosses only empty majorant cells: transmittance and the flight weight are exactly 1, and nothing is drawn.
PT_CHECK(empty_space_draws_nothing, Fast, Exact) {
    const Scratch scratch(ctx);
    openvdb::FloatGrid::Ptr grid = makeGrid(1.0);
    grid->tree().fill(openvdb::CoordBBox(openvdb::Coord(0), openvdb::Coord(15)), 1.0F, true);
    grid->tree().fill(openvdb::CoordBBox(openvdb::Coord(0, 0, 200), openvdb::Coord(15, 15, 215)), 1.0F, true);
    const VolumeSet set =
        loadOrDie({volumeConfig(scratch.writeGrids("blobs", {grid}), scratch.writeMaterial("medium", glm::vec3(1.0F), glm::vec3(1.0F), 0.0F))},
                  scratch.root());
    // From the first blob's far side to the second's near side, one cell of margin clear of each majorant's dilated support.
    const Ray ray{glm::vec3(8.0F, 8.0F, 40.0F), glm::vec3(0.0F, 0.0F, 1.0F), 0.0F, 140.0F};
    Sampler used(0, 0, 0, 1, 7U);
    const glm::vec3 transmittance = set.transmittance(ray, 0.0F, 140.0F, used);
    const pathtracer::scene::FreeFlight flight = set.sampleFreeFlight(ray, 0.0F, 140.0F, glm::vec3(1.0F), used);
    Sampler fresh(0, 0, 0, 1, 7U);
    ctx.plan(3);
    PT_EXPECT(ctx, transmittance == glm::vec3(1.0F), "transmittance through empty cells left 1");
    PT_EXPECT(ctx, flight.event == FreeFlightEvent::Escaped && flight.weight == glm::vec3(1.0F), "the flight did not escape at weight 1");
    PT_EXPECT(ctx, used.next2D() == fresh.next2D(), "empty space consumed sampler dimensions");
}

// Every authoring fault is refused at load with a reason, never rendered: each row names the fault it plants.
PT_CHECK(volume_load_rejects_invalid_inputs, Fast, Exact) {
    const Scratch scratch(ctx);
    const std::string material = scratch.writeMaterial("medium", glm::vec3(0.1F), glm::vec3(0.1F), 0.0F);
    const auto fog = [] { return makeBox(8, 1.0); };
    openvdb::Vec3SGrid::Ptr vector = openvdb::Vec3SGrid::create();
    vector->setName("density");
    openvdb::FloatGrid::Ptr levelSet = fog();
    levelSet->setGridClass(openvdb::GRID_LEVEL_SET);
    openvdb::FloatGrid::Ptr negative = fog();
    negative->getAccessor().setValueOff(openvdb::Coord(20), -1.0F);
    openvdb::FloatGrid::Ptr notANumber = fog();
    notANumber->getAccessor().setValue(openvdb::Coord(3), std::numeric_limits<float>::quiet_NaN());
    openvdb::FloatGrid::Ptr background = openvdb::FloatGrid::create(0.5F);
    background->setName("density");
    background->tree().fill(openvdb::CoordBBox(openvdb::Coord(0), openvdb::Coord(7)), 1.0F, true);
    openvdb::FloatGrid::Ptr frustum = fog();
    frustum->setTransform(openvdb::math::Transform::createFrustumTransform(
        openvdb::BBoxd(openvdb::Vec3d(0.0), openvdb::Vec3d(7.0)), 0.5, 1.0, 1.0));
    openvdb::FloatGrid::Ptr empty = makeGrid(1.0);

    struct Row {
        std::string fault;
        VolumeConfig config;
    };
    const auto row = [&](const std::string& fault, const std::string& vdb, const std::string& materialPath) {
        return Row{fault, volumeConfig(vdb, materialPath)};
    };
    const std::string good = scratch.writeGrids("good", {fog()});
    Row missingGrid = row("missing grid", good, material);
    missingGrid.config.grid = "smoke";
    const std::vector<Row> rows{
        row("missing file", "absent.vdb", material),
        missingGrid,
        row("vector grid", scratch.writeGrids("vector", {vector}), material),
        row("level set", scratch.writeGrids("levelset", {levelSet}), material),
        row("negative inactive value", scratch.writeGrids("negative", {negative}), material),
        row("NaN value", scratch.writeGrids("nan", {notANumber}), material),
        row("non-zero background", scratch.writeGrids("background", {background}), material),
        row("frustum transform", scratch.writeGrids("frustum", {frustum}), material),
        row("no density", scratch.writeGrids("empty", {empty}), material),
        row("missing material", good, "absent.json"),
        row("anisotropy 1", good, scratch.writeMaterial("forward", glm::vec3(0.1F), glm::vec3(0.1F), 1.0F)),
        row("negative scattering", good, scratch.writeMaterial("negative", glm::vec3(0.1F), glm::vec3(-0.1F, 0.1F, 0.1F), 0.0F)),
        row("unknown key", good, scratch.writeText("typo.json", R"({"absorption": [0, 0, 0], "scatering": [1, 1, 1], "anisotropy": 0})")),
    };
    ctx.plan(static_cast<int>(rows.size()) + 1);
    for (const Row& r : rows) {
        std::string error;
        PT_EXPECT(ctx, !VolumeSet::load({r.config}, scratch.root(), error).has_value() && !error.empty(), r.fault + " loaded");
    }
    std::string error;
    PT_EXPECT(ctx, VolumeSet::load({volumeConfig(good, material)}, scratch.root(), error).has_value(), "the valid control failed: " + error);
}

}  // namespace

PT_CHECK_MAIN("volume_validate")
