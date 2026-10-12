#include "pathtracer/scene/volume.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/component_wise.hpp>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Prune.h>

#include "pathtracer/scene/rotation.h"
#include "shading_math.h"
#include "volume_majorant.h"

namespace pathtracer::scene {

using NanoRay = nanovdb::math::Ray<float>;
using NanoAccessor = nanovdb::DefaultReadAccessor<float>;

struct VolumeSet::Volume {
    nanovdb::GridHandle<nanovdb::HostBuffer> densityHandle;
    nanovdb::GridHandle<nanovdb::HostBuffer> majorantHandle;
    const nanovdb::FloatGrid* density;
    const nanovdb::FloatGrid* majorant;
    // World to the density grid's index space, voxel centres at integers; the majorant's index space is that divided by 8.
    glm::mat3 indexFromWorld;
    glm::vec3 indexOffset;
    float gridUnitsPerWorldUnit;  // 1/scale: the coefficients are per grid unit, so a world ray parameter covers this many
    glm::vec3 absorption;
    glm::vec3 scattering;
    float extinctionBound;  // max channel of absorption + scattering: the majorant density's scalar spectral bound
    float anisotropy;
};

namespace {

// floor(a / b) for b > 0: C++ division truncates toward zero, which misplaces every negative index coordinate.
int floorDiv(int a, int b) { return (a >= 0 ? a : a - b + 1) / b; }

openvdb::Coord floorDiv(const openvdb::Coord& a, int b) {
    return {floorDiv(a.x(), b), floorDiv(a.y(), b), floorDiv(a.z(), b)};
}

// One volume's tentative collisions in increasing t: exponential flights against its majorant, span by span along a DDA.
class Tracker {
public:
    explicit Tracker(const VolumeSet::Volume& volume)
        : volume_(&volume), majorantAccessor_(volume.majorant->getAccessor()), densityAccessor_(volume.density->getAccessor()) {}

    // False when [t0, t1) misses every majorant cell, so the volume is skipped without a single draw.
    bool start(const Ray& ray, float t0, float t1) {
        const VolumeSet::Volume& volume = *volume_;
        const glm::vec3 origin = (volume.indexFromWorld * ray.origin) + volume.indexOffset;
        const glm::vec3 dir = volume.indexFromWorld * ray.dir;
        indexOrigin_ = nanovdb::math::Vec3f(origin.x, origin.y, origin.z);
        indexDir_ = nanovdb::math::Vec3f(dir.x, dir.y, dir.z);
        // An affine map keeps the ray parameter, so the majorant's ray is the index ray scaled by 1/8 with the same [t0, t1).
        const float toCells = 1.0F / static_cast<float>(kMajorantCellVoxels);
        NanoRay cellRay(indexOrigin_ * toCells, indexDir_ * toCells, t0, t1);
        if (!cellRay.clip(volume.majorant->tree().bbox())) {
            return false;
        }
        sigmaPerT_ = volume.extinctionBound * volume.gridUnitsPerWorldUnit * glm::length(ray.dir);
        dda_.init(cellRay);
        t_ = dda_.time();
        enterSpan();
        return true;
    }

    // The next tentative collision's t and the event variate drawn with it; nullopt past the segment. Empty spans draw nothing.
    std::optional<glm::vec2> next(Sampler& sampler) {
        float opticalDepth = -1.0F;  // undrawn until a span with a non-zero majorant is reached
        float eventVariate = 0.0F;
        for (;;) {
            const float sigma = spanMajorant_ * sigmaPerT_;
            if (sigma > 0.0F) {
                if (opticalDepth < 0.0F) {
                    const glm::vec2 u = sampler.next2D();
                    opticalDepth = -std::log1p(-u.x);
                    eventVariate = u.y;
                }
                const float spanDepth = sigma * (spanEnd_ - t_);
                if (opticalDepth < spanDepth) {
                    t_ += opticalDepth / sigma;
                    return glm::vec2(t_, eventVariate);
                }
                opticalDepth -= spanDepth;
            }
            if (!dda_.step()) {
                return std::nullopt;
            }
            t_ = dda_.time();
            enterSpan();
        }
    }

    // Trilinear density at the collision just returned; at most the span's majorant, up to one rounding of the interpolation.
    [[nodiscard]] float density() const {
        return nanovdb::math::createSampler<1, NanoAccessor, false>(densityAccessor_)(indexOrigin_ + (indexDir_ * t_));
    }

    [[nodiscard]] float majorant() const { return spanMajorant_; }
    [[nodiscard]] const VolumeSet::Volume& volume() const { return *volume_; }

private:
    void enterSpan() {
        spanEnd_ = dda_.next();
        spanMajorant_ = majorantAccessor_.getValue(dda_.voxel());
    }

    const VolumeSet::Volume* volume_;
    NanoAccessor majorantAccessor_;
    NanoAccessor densityAccessor_;
    nanovdb::math::Vec3f indexOrigin_;
    nanovdb::math::Vec3f indexDir_;
    nanovdb::math::DDA<NanoRay> dda_;
    float sigmaPerT_ = 0.0F;  // extinctionBound per ray parameter: majorant density times this is the span's collision rate
    float t_ = 0.0F;
    float spanEnd_ = 0.0F;
    float spanMajorant_ = 0.0F;
};

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

// Each volume's tracker and its next tentative collision, reused per thread so a free flight allocates nothing per segment.
thread_local std::vector<std::pair<Tracker, glm::vec2>> pendingCollisions;

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
    const openvdb::FloatGrid::Ptr majorant = buildMajorantGrid(*grid, error);
    if (majorant == nullptr) {
        error = path + ": " + error;
        return std::nullopt;
    }
    if (majorant->empty()) {
        error = path + ": grid '" + config.grid + "' holds no density";
        return std::nullopt;
    }

    const glm::dmat4 indexFromWorld = indexFromWorldOf(*grid, config);
    // Bounding boxes are the only stats the tracker reads (the majorant root's, for clipping); no checksum, as nothing verifies one.
    VolumeSet::Volume volume{
        nanovdb::tools::createNanoGrid(*grid, nanovdb::tools::StatsMode::BBox, nanovdb::CheckMode::Disable),
        nanovdb::tools::createNanoGrid(*majorant, nanovdb::tools::StatsMode::BBox, nanovdb::CheckMode::Disable),
        nullptr,
        nullptr,
        glm::mat3(glm::dmat3(indexFromWorld)),
        glm::vec3(indexFromWorld[3]),
        1.0F / config.scale,
        material->absorption,
        material->scattering,
        glm::compMax(material->absorption + material->scattering),
        material->anisotropy,
    };
    volume.density = volume.densityHandle.grid<float>();
    volume.majorant = volume.majorantHandle.grid<float>();
    return volume;
}

}  // namespace

openvdb::FloatGrid::Ptr buildMajorantGrid(const openvdb::FloatGrid& density, std::string& error) {
    openvdb::FloatGrid::Ptr majorant = openvdb::FloatGrid::create(0.0F);
    openvdb::FloatGrid::Accessor cells = majorant->getAccessor();
    for (auto it = density.tree().cbeginValueAll(); it; ++it) {
        const float value = *it;
        if (!(value >= 0.0F) || !std::isfinite(value)) {
            const openvdb::Coord at = it.getCoord();
            error = "density " + std::to_string(value) + " at (" + std::to_string(at.x()) + ", " + std::to_string(at.y()) + ", " +
                    std::to_string(at.z()) + ") is negative or not finite";
            return nullptr;
        }
        if (value == 0.0F) {
            continue;
        }
        const openvdb::CoordBBox box = it.getBoundingBox();
        // Cells c with [8c-1, 8c+9] meeting [a, b]; those with it inside [a, b] read this value alone, so a tile fills them as one block.
        const openvdb::Coord lo = floorDiv(box.min() - openvdb::Coord(2), kMajorantCellVoxels);
        const openvdb::Coord hi = floorDiv(box.max() + openvdb::Coord(1), kMajorantCellVoxels);
        const openvdb::Coord innerLo = floorDiv(box.min() + openvdb::Coord(kMajorantCellVoxels), kMajorantCellVoxels);
        const openvdb::Coord innerHi = floorDiv(box.max() - openvdb::Coord(kMajorantCellVoxels + 1), kMajorantCellVoxels);
        const openvdb::CoordBBox inner(innerLo, innerHi);
        if (!inner.empty()) {
            majorant->tree().fill(inner, value, /*active=*/true);
            cells.clear();  // fill may replace nodes the accessor has cached
        }
        // The shell: every other cell max-merges, since neighbouring values reach it too. O(surface), even for a root-level tile.
        for (int x = lo.x(); x <= hi.x(); ++x) {
            for (int y = lo.y(); y <= hi.y(); ++y) {
                const bool innerColumn = !inner.empty() && x >= innerLo.x() && x <= innerHi.x() && y >= innerLo.y() && y <= innerHi.y();
                for (int z = lo.z(); z <= hi.z(); ++z) {
                    if (innerColumn && z == innerLo.z()) {
                        z = innerHi.z();
                        continue;
                    }
                    const openvdb::Coord cell(x, y, z);
                    if (cells.getValue(cell) < value) {
                        cells.setValue(cell, value);
                    }
                }
            }
        }
    }
    openvdb::tools::prune(majorant->tree());
    return majorant;
}

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

FreeFlight VolumeSet::sampleFreeFlight(const Ray& ray, float t0, float t1, const glm::vec3& throughput, Sampler& sampler) const {
    // Overlapping volumes are independent Poisson processes of tentative collisions; their union is met in t order, earliest first.
    std::vector<std::pair<Tracker, glm::vec2>>& pending = pendingCollisions;
    pending.clear();
    for (const Volume& volume : *volumes_) {
        Tracker tracker(volume);
        if (tracker.start(ray, t0, t1)) {
            if (const std::optional<glm::vec2> collision = tracker.next(sampler)) {
                pending.emplace_back(tracker, *collision);
            }
        }
    }
    FreeFlight flight{FreeFlightEvent::Escaped, t1, 0.0F, glm::vec3(1.0F)};
    while (!pending.empty()) {
        const auto earliest = std::min_element(pending.begin(), pending.end(), [](const auto& a, const auto& b) { return a.second.x < b.second.x; });
        Tracker& tracker = earliest->first;
        const Volume& volume = tracker.volume();
        const float density = tracker.density();
        const float bound = tracker.majorant() * volume.extinctionBound;
        const glm::vec3 absorption = density * volume.absorption;
        const glm::vec3 scattering = density * volume.scattering;
        const glm::vec3 null = glm::max(glm::vec3(bound) - absorption - scattering, glm::vec3(0.0F));
        // History-aware probabilities (Kutz et al. 2017 sec. 5.1): each event's share of the throughput-weighted collision rate.
        const glm::vec3 history = throughput * flight.weight;
        const float pa = channelMean(history * absorption);
        const float ps = channelMean(history * scattering);
        const float pn = channelMean(history * null);
        const float total = pa + ps + pn;
        // Zero history is a path carrying nothing, as a hero path rrMinProb keeps alive can be: ending it is exact.
        const float pick = earliest->second.y * total;
        if (!(total > 0.0F) || pick < pa) {
            flight.event = FreeFlightEvent::Absorbed;
            return flight;
        }
        // Each kept event weighs (sigma_x/bound)/P_x, so the expectation over events sums sigma_x/bound, 1 per channel; 1 when grey.
        if (pick < pa + ps) {
            flight.event = FreeFlightEvent::Scattered;
            flight.t = earliest->second.x;
            flight.g = volume.anisotropy;
            flight.weight *= (scattering / bound) / (ps / total);
            return flight;
        }
        flight.weight *= (null / bound) / (pn / total);
        if (const std::optional<glm::vec2> collision = tracker.next(sampler)) {
            earliest->second = *collision;
        } else {
            *earliest = pending.back();
            pending.pop_back();
        }
    }
    return flight;
}

glm::vec3 VolumeSet::transmittance(const Ray& ray, float t0, float t1, Sampler& sampler) const {
    glm::vec3 transmittance(1.0F);
    // Independent media multiply: each volume's ratio-tracking estimate is unbiased on its own, and so is their product.
    for (const Volume& volume : *volumes_) {
        Tracker tracker(volume);
        if (!tracker.start(ray, t0, t1)) {
            continue;
        }
        const glm::vec3 extinction = volume.absorption + volume.scattering;
        while (tracker.next(sampler)) {
            const float bound = tracker.majorant() * volume.extinctionBound;
            transmittance *= glm::max(glm::vec3(1.0F) - (tracker.density() * extinction / bound), glm::vec3(0.0F));
            if (transmittance == glm::vec3(0.0F)) {
                return transmittance;
            }
        }
    }
    return transmittance;
}

}  // namespace pathtracer::scene
