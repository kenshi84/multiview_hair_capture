// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// PatchMatchEngine: manages GPU state and runs the Line-based PatchMatch
// solver for a single pyramid level.

#pragma once

#include <vector>

#include "common/types.h"
#include "mvs/patchmatch/params.h"
#include "mvs/patchmatch/types.h"

class PatchMatchEngine {
 public:
  PatchMatchEngine(unsigned int width, unsigned int height, unsigned int numnei);
  ~PatchMatchEngine();

  // Set camera parameters for the current level
  void InitParameters(const float intrinsic[9], const std::vector<GpuCamera>& nei_cams,
                      const PatchMatchParams& params, unsigned int width,
                      unsigned int height);

  // Run hair-specific PatchMatch
  void RunHair(PatchMatchInput& input, PatchMatchState& state,
               PatchMatchParams& params);

  PatchMatchInput& GetInput() {
    return input_;
  }
  PatchMatchState& GetState() {
    return state_;
  }

 private:
  PatchMatchInput input_;
  PatchMatchState state_;
  unsigned int width_;
  unsigned int height_;
  unsigned int numnei_;
};
