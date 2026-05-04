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

#include "common/image_io.h"

namespace image_loader {

// Load image as grayscale float [0,1], optionally downsample
Image LoadGrayImage(const std::string& path, float downsample);

// Load mask image, optionally downsample
Image LoadMaskImage(const std::string& path, float downsample);

// Undistort an image using the OpenCV distortion model (iterative inverse).
// K: 3x3 intrinsic (row-major), dist: [k1,k2,p1,p2,k3]
void UndistortImage(Image& img, const float K[9], const float dist[5]);

// Format camera-specific path from pattern (e.g., "%06d.png")
std::string FormatPath(const std::string& pattern, unsigned int cam_id);

}  // namespace image_loader
