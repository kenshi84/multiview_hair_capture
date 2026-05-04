// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <cuda_runtime.h>

// 3D line segment: center point + unit direction (symmetric: v == -v)
struct Line3D {
  float3 p;  // center position
  float3 v;  // unit direction (symmetric: (p,v) = (p,-v))
};

// Gabor filter parameters
struct GaborParams {
  int ksize;
  float sigma;
  float gamma;
  float lambd;
};

// GPU camera parameters for CUDA kernels (flat float arrays)
struct GpuCamera {
  float K[9];       // 3x3 intrinsic, row-major
  float R[9];       // 3x3 rotation
  float t[3];       // translation vector
  float center[3];  // camera center = -R^T * t
  float dist[5];    // distortion: k1, k2, p1, p2, k3 (OpenCV order)
};
