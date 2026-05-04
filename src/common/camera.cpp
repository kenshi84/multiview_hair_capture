// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/camera.h"

#include <cassert>
#include <cstring>

namespace {

void Mat3Transpose(const float in[9], float out[9]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      out[i * 3 + j] = in[j * 3 + i];
}

void Mat3Inv(const float m[9], float out[9]) {
  float det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
              m[2] * (m[3] * m[7] - m[4] * m[6]);
  float inv_det = 1.0f / det;
  out[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
  out[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
  out[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
  out[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
  out[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
  out[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
  out[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
  out[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
  out[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;
}

void Mat3Vec(const float m[9], const float v[3], float out[3]) {
  for (int i = 0; i < 3; ++i) {
    out[i] = 0;
    for (int j = 0; j < 3; ++j)
      out[i] += m[i * 3 + j] * v[j];
  }
}

}  // namespace

Camera::Camera() : id_(0) {
  memset(A_, 0, sizeof(A_));
  A_[0] = A_[4] = A_[8] = 1.0f;
  memset(Rt_, 0, sizeof(Rt_));
  Rt_[0] = Rt_[4] = Rt_[8] = 1.0f;
  memset(k_, 0, sizeof(k_));
  ComputeDerived();
}

Camera::Camera(const float intrinsic[9], const float extrinsic[12],
               const float distortion[5], unsigned int id)
    : id_(id) {
  memcpy(A_, intrinsic, sizeof(A_));
  memcpy(Rt_, extrinsic, sizeof(Rt_));
  memcpy(k_, distortion, sizeof(k_));
  ComputeDerived();
}

void Camera::ComputeDerived() {
  Mat3Inv(A_, inv_A_);

  // Center = -R^T * t
  float R[9], RT[9], t[3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j)
      R[i * 3 + j] = Rt_[i * 4 + j];
    t[i] = Rt_[i * 4 + 3];
  }
  Mat3Transpose(R, RT);
  float neg_t[3];
  Mat3Vec(RT, t, neg_t);
  center_[0] = -neg_t[0];
  center_[1] = -neg_t[1];
  center_[2] = -neg_t[2];
}

void Camera::Downsize(float factor_x, float factor_y) {
  A_[0] *= factor_x;
  A_[1] *= factor_x;
  A_[3] *= factor_y;
  A_[4] *= factor_y;
  A_[2] = (A_[2] + 0.5f) * factor_x - 0.5f;
  A_[5] = (A_[5] + 0.5f) * factor_y - 0.5f;
  ComputeDerived();
}

GpuCamera Camera::BuildGpuCamera() const {
  GpuCamera gpu;
  memcpy(gpu.K, A_, sizeof(gpu.K));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j)
      gpu.R[i * 3 + j] = Rt_[i * 4 + j];
    gpu.t[i] = Rt_[i * 4 + 3];
  }
  memcpy(gpu.center, center_, sizeof(gpu.center));
  memcpy(gpu.dist, k_, sizeof(gpu.dist));
  return gpu;
}
