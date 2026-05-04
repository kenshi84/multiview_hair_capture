// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <string>
#include <vector>

#include "common/point_cloud.h"

// A strand is a sequence of 3D points with directions.
struct Strand {
  std::vector<float> positions;   // x,y,z interleaved
  std::vector<float> directions;  // dx,dy,dz interleaved
  std::vector<int> labels;

  size_t NumPoints() const {
    return positions.size() / 3;
  }

  float Length() const;

  void AddPoint(float x, float y, float z, float dx, float dy, float dz,
                int label = 0) {
    positions.push_back(x);
    positions.push_back(y);
    positions.push_back(z);
    directions.push_back(dx);
    directions.push_back(dy);
    directions.push_back(dz);
    labels.push_back(label);
  }
};

namespace strand_io {

// Write strands in binary format.
// Format: num_strands (int), then per strand: num_points (int),
// then x,y,z,dx,dy,dz per point.
bool WriteBinary(const std::string& path, const std::vector<Strand>& strands);

// Read strands from binary format.
bool ReadBinary(const std::string& path, std::vector<Strand>& strands);

// Write strands as a PLY point cloud (binary_little_endian)
// with x,y,z,nx,ny,nz,label(uint) properties.
bool WritePly(const std::string& path, const std::vector<Strand>& strands);

// Convert strands to a merged PointCloud (label = strand index).
PointCloud StrandsToPointCloud(const std::vector<Strand>& strands);

}  // namespace strand_io
