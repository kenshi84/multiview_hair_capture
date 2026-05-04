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

#include "common/camera_array.h"
#include "common/config.h"
#include "common/point_cloud.h"
#include "mvs/patchmatch_mvs.h"

namespace depth_fusion {

// Image-space cross-validation fusion.
// For each reference view, projects its line3D points into neighbor views,
// checks position and orientation consistency, and keeps points with
// sufficient multi-view support.
PointCloud FusePointClouds(const std::vector<ViewMvsResult>& view_results,
                           const CameraArray& cameras, const Config& config);

}  // namespace depth_fusion
