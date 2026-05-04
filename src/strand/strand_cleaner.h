// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <vector>

#include "common/config.h"
#include "strand/strand_io.h"

namespace strand_cleaner {

// Remove strands shorter than min_length.
std::vector<Strand> RemoveShort(const std::vector<Strand>& strands, float min_length);

// Remove spatially isolated strands (outliers) using KD-tree on centroids.
std::vector<Strand> RemoveOutliers(const std::vector<Strand>& strands, float radius,
                                   int min_neighbors = 3);

// Combined cleaning pipeline.
std::vector<Strand> Clean(const std::vector<Strand>& strands, const Config& config);

}  // namespace strand_cleaner
