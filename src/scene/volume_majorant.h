#pragma once

#include <string>

#include <openvdb/openvdb.h>

namespace pathtracer::scene {

// Majorant cells are leaf-sized: one coarse cell spans kMajorantCellVoxels density voxels per axis, the tree's own blocking.
inline constexpr int kMajorantCellVoxels = static_cast<int>(openvdb::FloatTree::LeafNodeType::DIM);

// Max over voxels [8c-1, 8c+9], the trilinear support of [8c-1, 8c+9), inactive ones included; nullptr on a negative or NaN value.
[[nodiscard]] openvdb::FloatGrid::Ptr buildMajorantGrid(const openvdb::FloatGrid& density, std::string& error);

}  // namespace pathtracer::scene
