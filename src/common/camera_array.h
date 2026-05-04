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

#include "common/camera.h"

class CameraArray {
 public:
  CameraArray() = default;

  bool LoadFromJson(const std::string& json_path);

  void DownsizeCameras(float factor);

  std::vector<int> SelectNeighborViews(int ref_index, float min_angle_deg,
                                       float max_angle_deg, int max_views) const;

  const Camera& GetCamera(int index) const {
    return cameras_[index];
  }
  int NumCameras() const {
    return static_cast<int>(cameras_.size());
  }
  unsigned int GetCamId(int index) const {
    return cameras_[index].id();
  }

 private:
  std::vector<Camera> cameras_;
};
