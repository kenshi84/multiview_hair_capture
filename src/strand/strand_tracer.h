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
#include "common/point_cloud.h"
#include "strand/strand_io.h"

namespace strand_tracer {

// Generate strands from mean-shift filtered point cloud using forward Euler.
// Section 5 of the paper: trace strands by following orientation field.
std::vector<Strand> Trace(const PointCloud& cloud, const Config& config);

}  // namespace strand_tracer
