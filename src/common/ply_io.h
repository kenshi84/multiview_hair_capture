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

namespace ply_io {

bool ReadPointCloud(const std::string& path, PointCloud& cloud);

bool WritePointCloud(const std::string& path, const PointCloud& cloud,
                     bool binary = true);

bool WriteMesh(const std::string& path, const std::vector<float>& vertices,
               const std::vector<float>& normals, const std::vector<int>& faces,
               bool binary = true);

}  // namespace ply_io
