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
#include <memory>
#include <vector>

#include "nanoflann.hpp"

struct PointCloudAdaptor {
  const float* pts;
  size_t num_points;

  PointCloudAdaptor(const float* data, size_t n) : pts(data), num_points(n) {}

  inline size_t kdtree_get_point_count() const {
    return num_points;
  }

  inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
    return pts[idx * 3 + dim];
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }
};

// Use uint32_t as index type to match nanoflann default
using KdTreeIndex = uint32_t;

class KdTree {
 public:
  using Tree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor, 3,
      KdTreeIndex>;

  KdTree() = default;

  void Build(const float* positions, size_t num_points) {
    adaptor_ = std::make_unique<PointCloudAdaptor>(positions, num_points);
    tree_ = std::make_unique<Tree>(3, *adaptor_,
                                   nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree_->buildIndex();
  }

  size_t RadiusSearch(
      const float query[3], float radius,
      std::vector<nanoflann::ResultItem<KdTreeIndex, float>>& results) const {
    results.clear();
    nanoflann::SearchParameters params;
    params.sorted = false;
    return tree_->radiusSearch(query, radius * radius, results, params);
  }

 private:
  std::unique_ptr<PointCloudAdaptor> adaptor_;
  std::unique_ptr<Tree> tree_;
};
