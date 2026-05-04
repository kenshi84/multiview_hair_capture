// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "strand/strand_cleaner.h"

#include "common/kdtree.h"
#include "common/logger.h"
#include "common/timer.h"

namespace strand_cleaner {

std::vector<Strand> RemoveShort(const std::vector<Strand>& strands, float min_length) {
  std::vector<Strand> result;
  for (const auto& s : strands) {
    if (s.Length() >= min_length) {
      result.push_back(s);
    }
  }
  LOG_INFO("RemoveShort: %zu -> %zu strands (min_length=%.1f)", strands.size(),
           result.size(), min_length);
  return result;
}

std::vector<Strand> RemoveOutliers(const std::vector<Strand>& strands, float radius,
                                   int min_neighbors) {
  if (strands.empty())
    return strands;

  // Build a single point cloud from all strands with strand labels
  std::vector<float> all_positions;
  std::vector<uint32_t> all_labels;
  for (size_t si = 0; si < strands.size(); ++si) {
    for (size_t pi = 0; pi < strands[si].NumPoints(); ++pi) {
      all_positions.push_back(strands[si].positions[pi * 3]);
      all_positions.push_back(strands[si].positions[pi * 3 + 1]);
      all_positions.push_back(strands[si].positions[pi * 3 + 2]);
      all_labels.push_back(static_cast<uint32_t>(si));
    }
  }

  size_t total_pts = all_labels.size();
  LOG_DEBUG("RemoveOutliers: building KD-tree on %zu total points", total_pts);

  // Build KD-tree on all strand points
  KdTree tree;
  tree.Build(all_positions.data(), total_pts);

  // For each strand, check if any point has enough distinct neighboring strands
  std::vector<Strand> result;
  std::vector<nanoflann::ResultItem<KdTreeIndex, float>> neighbors;

  for (size_t si = 0; si < strands.size(); ++si) {
    bool is_outlier = true;
    std::vector<uint32_t> nei_labels;

    for (size_t pi = 0; pi < strands[si].NumPoints() && is_outlier; ++pi) {
      const float* pt = &strands[si].positions[pi * 3];
      tree.RadiusSearch(pt, radius, neighbors);

      for (const auto& nb : neighbors) {
        uint32_t nb_label = all_labels[nb.first];
        if (nb_label != static_cast<uint32_t>(si)) {
          // Check if this label is already counted
          bool found = false;
          for (auto l : nei_labels) {
            if (l == nb_label) {
              found = true;
              break;
            }
          }
          if (!found)
            nei_labels.push_back(nb_label);
        }
        if (static_cast<int>(nei_labels.size()) >= min_neighbors) {
          is_outlier = false;
          break;
        }
      }
    }

    if (!is_outlier) {
      result.push_back(strands[si]);
    }
  }

  LOG_INFO("RemoveOutliers: %zu -> %zu strands (radius=%.1f, min_nei=%d)",
           strands.size(), result.size(), radius, min_neighbors);
  return result;
}

std::vector<Strand> Clean(const std::vector<Strand>& strands, const Config& config) {
  ScopedTimer timer("Strand Cleaning");
  auto result = RemoveShort(strands, config.clean_min_length);
  if (config.clean_outlier_radius > 0) {
    result = RemoveOutliers(result, config.clean_outlier_radius,
                            config.clean_outlier_min_neighbors);
  }
  return result;
}

}  // namespace strand_cleaner
