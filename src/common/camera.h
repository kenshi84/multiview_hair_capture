// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <array>
#include <cmath>
#include <string>

#include "common/types.h"

class Camera {
 public:
  Camera();
  Camera(const float intrinsic[9], const float extrinsic[12], const float distortion[5],
         unsigned int id);

  void Downsize(float factor_x, float factor_y);

  GpuCamera BuildGpuCamera() const;

  unsigned int id() const {
    return id_;
  }
  const float* center() const {
    return center_;
  }
  const float* intrinsic() const {
    return A_;
  }
  const float* inv_intrinsic() const {
    return inv_A_;
  }
  const float* extrinsic() const {
    return Rt_;
  }

 private:
  void ComputeDerived();

  float A_[9];       // 3x3 intrinsic, row-major
  float inv_A_[9];   // 3x3 inverse intrinsic
  float Rt_[12];     // 3x4 extrinsic [R|t], row-major
  float center_[3];  // camera center = -R^T * t
  float k_[5];       // distortion: k1, k2, p1, p2, k3 (OpenCV order)
  unsigned int id_;
};
