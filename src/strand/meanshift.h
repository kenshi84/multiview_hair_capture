// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include "common/config.h"
#include "common/point_cloud.h"

namespace meanshift {

// Run CUDA mean-shift filtering on an oriented point cloud.
// Implements the mean-shift fusion from Section 5 of the paper.
// With num_gpus > 1, splits points across GPUs (each gets full grid).
PointCloud RunCuda(const PointCloud& input, const Config& config, int gpu_id,
                   int num_gpus = 1);

}  // namespace meanshift
