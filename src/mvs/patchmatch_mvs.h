// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <mutex>
#include <vector>

#include "common/camera_array.h"
#include "common/config.h"
#include "common/math_util.h"
#include "common/point_cloud.h"
#include "common/types.h"

// Per-view MVS result stored in camera-local coordinates for fusion.
struct ViewMvsResult {
  int ref_index = -1;            // index in CameraArray
  unsigned int width = 0;        // image width (after downsample)
  unsigned int height = 0;       // image height (after downsample)
  std::vector<Line3D> line_map;  // camera-local line3D (width*height)
  std::vector<int> nei_indices;  // neighbor view indices in CameraArray
};

class PatchMatchMvs {
 public:
  PatchMatchMvs(const Config& config, const CameraArray& cameras);
  ~PatchMatchMvs();

  void Run();

  const std::vector<ViewMvsResult>& view_results() const {
    return view_results_;
  }

 private:
  void ProcessReferenceView(int ref_index, int gpu_id);

  const Config& config_;
  const CameraArray& cameras_;
  std::vector<ViewMvsResult> view_results_;
  std::mutex results_mutex_;
};
