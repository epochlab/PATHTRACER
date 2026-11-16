#include "pathtracer/scene/volume.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <openvdb/io/File.h>

#include "medium.h"
#include "pathtracer/scene/rotation.h"
#include "shading_math.h"

namespace pathtracer::scene {

using NanoRay = nanovdb::math::Ray<float>;
using NanoAccessor = nanovdb::DefaultReadAccessor<float>;

struct VolumeSet::Volume {
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;
    const nanovdb::FloatGrid* density;
    // World to the grid's index space shifted by 1/2, so voxel i holds its value over [i, i+1) per axis: box reconstruction.
    glm::mat3 indexFromWorld;
    glm::vec3 indexOffset;
    nanovdb::CoordBBox support;  // every voxel and tile of non-zero density, inactive ones included
    float gridUnitsPerWorldUnit;  // 1/scale: the coefficients are per grid unit, so a world ray parameter covers this many
    UnboundedSpectrum absorption;  // per grid unit at density 1, the material's density folded in
    UnboundedSpectrum scattering;
    float anisotropy;
};

namespace {

// One volume's density along a ray as spans of constant value: tiles and voxels by hierarchical DDA, 0 off its support.
class VoxelSpans {
public:
    explicit VoxelSpans(const VolumeSet::Volume& volume) : volume_(&volume), accessor_(volume.density->getAccessor()) {}

    // False when [t0, t1) misses the support; else the current span is the empty space from t0 to the support.
    bool start(const Ray& ray, float t0, float t1) {
        const VolumeSet::Volume& volume = *volume_;
        const glm::vec3 origin = (volume.indexFromWorld * ray.origin) + volume.indexOffset;
        const glm::vec3 dir = volume.indexFromWorld * ray.dir;
        // An affine map keeps the ray parameter, so the index ray spans the same [t0, t1); clip pads the support's max by one voxel.
        ray_ = NanoRay(nanovdb::Vec3f(origin.x, origin.y, origin.z), nanovdb::Vec3f(dir.x, dir.y, dir.z), t0, t1);
        if (!ray_.clip(volume.support)) {
            return false;
        }
        gridUnitsPerT_ = volume.gridUnitsPerWorldUnit * glm::length(ray.dir);
        t1_ = t1;
        end_ = ray_.t0();
        return true;
    }

    [[nodiscard]] float end() const { return end_; }
    [[nodiscard]] bool past() const { return phase_ == Phase::After; }
    // The span's density per ray parameter: times a coefficient per grid unit, its collision rate.
    [[nodiscard]] float rate() const { return density_ * gridUnitsPerT_; }

    // On to the span starting at end(): the support's next tile or voxel, else past it, empty to t1.
    void advance() {
        if (phase_ == Phase::Before) {
            hdda_.init(ray_, ray_.t0(), ray_.t1(), 1);
            phase_ = Phase::Inside;
            enterCell();
        } else if (phase_ == Phase::Inside && hdda_.step()) {
            enterCell();
        } else {
            phase_ = Phase::After;
            end_ = t1_;
            density_ = 0.0F;
        }
    }

private:
    enum class Phase { Before, Inside, After };

    // Resizes the DDA to the constant region holding its voxel; each resize clamps to the previous cell, so it never skips one.
    void enterCell() {
        while (hdda_.update(ray_, static_cast<int>(accessor_.getDim(hdda_.voxel(), ray_)))) {
        }
        density_ = accessor_.getValue(hdda_.voxel());
        end_ = hdda_.next();
    }

    const VolumeSet::Volume* volume_;
    NanoAccessor accessor_;
    NanoRay ray_;
    nanovdb::math::HDDA<NanoRay> hdda_;
    Phase phase_ = Phase::Before;
    float gridUnitsPerT_ = 0.0F;
    float t1_ = 0.0F;
    float end_ = 0.0F;
    float density_ = 0.0F;
};

// A volume's spans with its coefficients per grid unit at the path's wavelengths.
struct Walker {
    VoxelSpans spans;
    SampledSpectrum scattering;
    SampledSpectrum extinction;
    float anisotropy;
};

// Each volume meeting the segment, reused per thread so a walk allocates nothing per segment.
thread_local std::vector<Walker> walkers;

std::vector<Walker>& startWalkers(const std::vector<VolumeSet::Volume>& volumes, const Ray& ray, float t0, float t1,
                                  const SampledWavelengths& lambda) {
    std::vector<Walker>& active = walkers;
    active.clear();
    for (const VolumeSet::Volume& volume : volumes) {
        VoxelSpans spans(volume);
        if (spans.start(ray, t0, t1)) {
            const SampledSpectrum scattering = volume.scattering(lambda);
            active.push_back({spans, scattering, volume.absorption(lambda) + scattering, volume.anisotropy});
        }
    }
    return active;
}

// visit(t, end, rate) on each span of [t0, t1) with every density constant, rate the summed extinction per t; false stops the walk.
template <typename Visit>
void forEachSpan(std::vector<Walker>& active, float t0, float t1, Visit&& visit) {
    // Overlapping media sum their coefficients (Fascione et al. 2018 sec. 3.4): one span ends where any volume's does.
    for (float t = t0; t < t1;) {
        float end = t1;
        SampledSpectrum rate(0.0F);
        bool beyond = true;
        for (const Walker& walker : active) {
            beyond = beyond && walker.spans.past();
            end = std::min(end, walker.spans.end());
            rate += walker.spans.rate() * walker.extinction;
        }
        // Every support is behind: nothing remains to add, and an unbounded t1 never meets 0 * inf.
        if (beyond) {
            return;
        }
        end = std::max(end, t);  // a cell face rounded behind t never steps the walk back
        if (!visit(t, end, rate)) {
            return;
        }
        for (Walker& walker : active) {
            if (walker.spans.end() <= end) {
                walker.spans.advance();
            }
        }
        t = end;
    }
}

// The scattering volume by its share of throughput-weighted scattering, and that share; nullptr when none scatters a carried lane.
std::pair<const Walker*, float> pickScatterer(const std::vector<Walker>& active, const SampledSpectrum& throughput, Sampler& sampler) {
    if (active.size() == 1) {
        return {active.data(), 1.0F};
    }
    const auto mass = [&](const Walker& walker) { return channelMean(throughput * walker.spans.rate() * walker.scattering); };
    float total = 0.0F;
    for (const Walker& walker : active) {
        total += mass(walker);
    }
    if (!(total > 0.0F)) {
        return {nullptr, 0.0F};
    }
    // The walker whose cumulative mass first exceeds the draw; past the float sum, the last with positive mass.
    float pick = sampler.next1D() * total;
    std::pair<const Walker*, float> chosen{nullptr, 0.0F};
    for (const Walker& walker : active) {
        const float m = mass(walker);
        if (m > 0.0F) {
            chosen = {&walker, m / total};
            if (pick < m) {
                break;
            }
            pick -= m;
        }
    }
    return chosen;
}

// Past this optical depth exp(-depth) may round to 0; below it, it cannot, so the exact zero test is skipped.
const float kUnderflowDepth = -std::log(std::numeric_limits<float>::denorm_min());

// The named grid as a FloatGrid; nullptr and `error` when the file, the name or the value type is wrong.
openvdb::FloatGrid::Ptr readGrid(const std::string& path, const std::string& name, std::string& error) {
    openvdb::FloatGrid::Ptr grid;
    try {
        openvdb::io::File file(path);
        file.open();
        if (!file.hasGrid(name)) {
            error = path + ": no grid named '" + name + "'";
            return nullptr;
        }
        grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(name));
    } catch (const openvdb::Exception& e) {
        error = path + ": " + e.what();
        return nullptr;
    }
    if (grid == nullptr) {
        error = path + ": grid '" + name + "' is not a float grid";
    }
    return grid;
}

// World to the grid's index space, composed and inverted once in double, so the per-ray map is one float multiply-add.
glm::dmat4 indexFromWorldOf(const openvdb::FloatGrid& grid, const pathtracer::config::VolumeConfig& config) {
    // OpenVDB's row-vector Mat4d, index to the grid's world, is glm's column-major index to world element for element.
    const openvdb::math::Mat4d indexToGrid = grid.transform().baseMap()->getAffineMap()->getMat4();
    glm::dmat4 gridFromIndex;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            gridFromIndex[c][r] = indexToGrid(c, r);
        }
    }
    const glm::dmat4 worldFromGrid = glm::translate(glm::dmat4(1.0), glm::dvec3(config.position)) *
                                     glm::dmat4(glm::dmat3(rotationXyz(config.rotation))) *
                                     glm::scale(glm::dmat4(1.0), glm::dvec3(config.scale));
    return glm::inverse(worldFromGrid * gridFromIndex);
}

// Every value finite and non-negative, inactive ones included, as the walk reads them all; the bounds of those above 0.
std::optional<openvdb::CoordBBox> densitySupport(const openvdb::FloatGrid& density, std::string& error) {
    openvdb::CoordBBox support;
    for (auto it = density.tree().cbeginValueAll(); it; ++it) {
        const float value = *it;
        if (!(value >= 0.0F) || !std::isfinite(value)) {
            const openvdb::Coord at = it.getCoord();
            error = "density " + std::to_string(value) + " at (" + std::to_string(at.x()) + ", " + std::to_string(at.y()) + ", " +
                    std::to_string(at.z()) + ") is negative or not finite";
            return std::nullopt;
        }
        if (value > 0.0F) {
            support.expand(it.getBoundingBox());
        }
    }
    return support;
}

std::optional<VolumeSet::Volume> loadVolume(const pathtracer::config::VolumeConfig& config, const std::string& assetRoot,
                                            std::string& error) {
    const std::string path = assetRoot + "/" + config.vdbPath;
    const std::optional<pathtracer::config::VolumeMaterialConfig> material =
        pathtracer::config::loadVolumeMaterialConfig(assetRoot + "/" + config.materialPath);
    if (!material) {
        error = "failed to load volume material " + assetRoot + "/" + config.materialPath;
        return std::nullopt;
    }
    const openvdb::FloatGrid::Ptr grid = readGrid(path, config.grid, error);
    if (grid == nullptr) {
        return std::nullopt;
    }
    // A level set stores signed distance, not density; a non-zero background would fill all of space with medium.
    if (grid->getGridClass() == openvdb::GRID_LEVEL_SET || grid->background() != 0.0F) {
        error = path + ": grid '" + config.grid + "' is not a fog volume with background 0";
        return std::nullopt;
    }
    if (!grid->transform().isLinear()) {
        error = path + ": grid '" + config.grid + "' has a non-linear (frustum) transform";
        return std::nullopt;
    }
    const std::optional<openvdb::CoordBBox> support = densitySupport(*grid, error);
    if (!support) {
        error = path + ": " + error;
        return std::nullopt;
    }
    if (support->empty()) {
        error = path + ": grid '" + config.grid + "' holds no density";
        return std::nullopt;
    }

    const glm::dmat4 indexFromWorld = indexFromWorldOf(*grid, config);
    // No stats: a stats pass flags nodes without active voxels to be skipped, which would hide their inactive density from getDim.
    VolumeSet::Volume volume{
        nanovdb::tools::createNanoGrid(*grid, nanovdb::tools::StatsMode::Disable, nanovdb::CheckMode::Disable),
        nullptr,
        glm::mat3(glm::dmat3(indexFromWorld)),
        glm::vec3(indexFromWorld[3]) + 0.5F,
        nanovdb::CoordBBox(nanovdb::Coord(support->min().x(), support->min().y(), support->min().z()),
                           nanovdb::Coord(support->max().x(), support->max().y(), support->max().z())),
        1.0F / config.scale,
        fitUnbounded(material->density * material->absorption),
        fitUnbounded(material->density * material->scattering),
        material->anisotropy,
    };
    volume.density = volume.handle.grid<float>();
    return volume;
}

}  // namespace

float henyeyGreenstein(float cosTheta, float g) {
    const float denominator = 1.0F + (g * g) - (2.0F * g * cosTheta);
    return (1.0F - (g * g)) / (4.0F * kPi * denominator * std::sqrt(denominator));
}

glm::vec3 sampleHenyeyGreenstein(const glm::vec3& dir, float g, glm::vec2 u) {
    // cos = (1 + g^2 - ((1 - g^2)/(1 + g s))^2)/(2g), s = 2u - 1, with the numerator's factor of g divided out: exact at g = 0.
    const float s = (2.0F * u.x) - 1.0F;
    const float k = 1.0F + (g * s);
    const float cosTheta = ((2.0F * s) + (g * (3.0F + (s * s))) + (2.0F * g * g * s) + (g * g * g * ((s * s) - 1.0F))) / (2.0F * k * k);
    const float sinTheta = std::sqrt(std::max(0.0F, 1.0F - (cosTheta * cosTheta)));
    const float phi = 2.0F * kPi * u.y;
    return glm::transpose(toFrameAbout(dir)) * glm::vec3(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
}

std::optional<VolumeSet> VolumeSet::load(const std::vector<pathtracer::config::VolumeConfig>& configs, const std::string& assetRoot,
                                         std::string& error) {
    openvdb::initialize();
    std::vector<Volume> volumes;
    volumes.reserve(configs.size());
    for (const pathtracer::config::VolumeConfig& config : configs) {
        std::optional<Volume> volume = loadVolume(config, assetRoot, error);
        if (!volume) {
            return std::nullopt;
        }
        volumes.push_back(std::move(*volume));
    }
    VolumeSet set;
    set.volumes_ = std::make_shared<const std::vector<Volume>>(std::move(volumes));
    return set;
}

bool VolumeSet::empty() const { return volumes_ == nullptr || volumes_->empty(); }

FreeFlight VolumeSet::sampleFreeFlight(const Ray& ray, float t0, float t1, const SampledSpectrum& throughput, const SampledWavelengths& lambda,
                                        Sampler& sampler) const {
    FreeFlight flight{FreeFlightEvent::Escaped, t1, 0.0F, SampledSpectrum(1.0F)};
    std::vector<Walker>& active = startWalkers(*volumes_, ray, t0, t1, lambda);
    if (active.empty()) {
        return flight;
    }
    // The lane odds, fixed before the walk, from the coefficients summed at unit density: the homogeneous walk's rule for one volume.
    HomogeneousMedium unit;
    for (const Walker& walker : active) {
        unit.absorption += walker.extinction - walker.scattering;
        unit.scattering += walker.scattering;
    }
    const bool canScatter = scatters(unit);
    const SampledSpectrum odds = channelOdds(unit, throughput);
    SampledSpectrum depth(0.0F);
    SampledSpectrum extinction(0.0F);  // the summed rate at the collision
    int lane = -1;  // undrawn until a span that can scatter, so a ray through empty space consumes nothing
    float target = 0.0F;
    forEachSpan(active, t0, t1, [&](float t, float end, const SampledSpectrum& rate) {
        if (lane < 0 && canScatter && !isBlack(rate)) {
            const glm::vec2 u = sampler.next2D();
            lane = sampleLane(odds, u.x);
            target = -std::log1p(-u.y);
        }
        // Constant rates make each lane's optical depth linear in t, so the drawn lane's crossing of its target is exact.
        if (lane < 0 || !(rate[lane] > 0.0F) || depth[lane] + (rate[lane] * (end - t)) <= target) {
            depth += rate * (end - t);
            return true;
        }
        flight.t = std::min(t + ((target - depth[lane]) / rate[lane]), end);
        depth += rate * (flight.t - t);
        extinction = rate;
        flight.event = FreeFlightEvent::Scattered;
        return false;
    });
    if (flight.event == FreeFlightEvent::Escaped) {
        // Undrawn, escape was certain and the weight is T itself: 1 through empty space, the attenuation of a medium that cannot scatter.
        flight.weight = lane < 0 ? exp(-depth) : passWeight(odds, exp(-depth));
        return flight;
    }
    const auto [chosen, share] = pickScatterer(active, throughput, sampler);
    // No volume scatters a lane the path still carries, so every continuation is exactly 0.
    if (chosen == nullptr) {
        flight.weight = SampledSpectrum(0.0F);
        return flight;
    }
    flight.g = chosen->anisotropy;
    flight.weight = collisionWeight(odds, exp(-depth), chosen->spans.rate() * chosen->scattering, extinction) / share;
    return flight;
}

SampledSpectrum VolumeSet::transmittance(const Ray& ray, float t0, float t1, const SampledWavelengths& lambda) const {
    SampledSpectrum depth(0.0F);
    forEachSpan(startWalkers(*volumes_, ray, t0, t1, lambda), t0, t1, [&](float t, float end, const SampledSpectrum& rate) {
        depth += rate * (end - t);
        // Once every lane's transmittance is exactly 0 in float, the rest of the ray cannot change it.
        return !(minChannel(depth) > kUnderflowDepth && isBlack(exp(-depth)));
    });
    return exp(-depth);
}

}  // namespace pathtracer::scene
