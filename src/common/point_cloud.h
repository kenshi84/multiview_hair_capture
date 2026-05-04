// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <cstddef>
#include <vector>

struct PointCloud {
  std::vector<float> positions;   // x,y,z interleaved (3 * N)
  std::vector<float> directions;  // dx,dy,dz interleaved (3 * N)
  std::vector<int> labels;        // per-point label (N)

  size_t NumPoints() const {
    return positions.size() / 3;
  }

  void Reserve(size_t n) {
    positions.reserve(n * 3);
    directions.reserve(n * 3);
    labels.reserve(n);
  }

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

  void Clear() {
    positions.clear();
    directions.clear();
    labels.clear();
  }

  float* PositionPtr(size_t i) {
    return &positions[i * 3];
  }
  const float* PositionPtr(size_t i) const {
    return &positions[i * 3];
  }
  float* DirectionPtr(size_t i) {
    return &directions[i * 3];
  }
  const float* DirectionPtr(size_t i) const {
    return &directions[i * 3];
  }
};
