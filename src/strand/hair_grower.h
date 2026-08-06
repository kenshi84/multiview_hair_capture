// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <cstddef>
#include <vector>

#include "common/camera_array.h"
#include "common/config.h"
#include "strand/strand_io.h"

namespace hair_grower {

// Summary of one grow invocation.  GrowCuda always logs the same information;
// this structure is primarily useful to tests and library callers.
struct GrowthStats {
  size_t input_strands = 0;
  size_t input_points = 0;
  size_t candidate_tips = 0;
  size_t grown_tips = 0;
  size_t points_added = 0;
  size_t usable_views = 0;
  size_t batches = 0;
  int gpus_used = 0;
};

// Extend both geometrically supported ends of every strand using the
// multi-view orientation constraints described in Section 6 of the paper.
//
// The returned vector has exactly the same strand order and count as `strands`.
// All original samples (including their direction and label values) are copied
// without modification.  A strand with fewer than two distinct points, or a
// tip which has insufficient image support, is returned unchanged.
//
// Throws std::invalid_argument for an invalid grow configuration and
// std::runtime_error if CUDA/view preparation cannot provide grow_min_views
// usable views.  In particular, this makes a failed grow stage distinguishable
// from a successful identity result.
//
// Cameras use the unchanged input calibration schema. GrowCuda applies
// config.downsample internally, matching the MVS image/intrinsic preparation.
std::vector<Strand> GrowCuda(const std::vector<Strand>& strands,
                             const CameraArray& cameras, const Config& config,
                             int gpu_id, int num_gpus = 1);

// As above, additionally returning execution statistics. `stats` may be null.
std::vector<Strand> GrowCuda(const std::vector<Strand>& strands,
                             const CameraArray& cameras, const Config& config,
                             int gpu_id, int num_gpus, GrowthStats* stats);

}  // namespace hair_grower
