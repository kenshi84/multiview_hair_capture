// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// HierarchicalPatchMatch: manages multi-resolution coarse-to-fine processing
// for Line-based PatchMatch MVS. At each pyramid level, downsamples images,
// computes Gabor orientations, runs PatchMatch, and upsamples results.

#pragma once

#include <string>
#include <vector>

#include "common/math_util.h"
#include "common/types.h"
#include "mvs/patchmatch/engine.h"
#include "mvs/patchmatch/params.h"

class HierarchicalPatchMatch {
 public:
  HierarchicalPatchMatch(const Mat3x3f& K, const std::vector<GpuCamera>& nei_cams,
                         const AlgorithmParams& params,
                         std::vector<unsigned int>& widths,
                         std::vector<unsigned int>& heights, unsigned int numLevels);
  ~HierarchicalPatchMatch();

  void SetOutputFolder(const std::string& path) {
    out_folder_ = path;
  }
  bool use_mask = false;

  // Run the full hierarchical Line-based PatchMatch solver.
  // d_refGrayMap: reference grayscale image on GPU (level 0, non-pitched)
  // d_neiTexObjs: HOST array of texture objects for neighbor gray images at
  //               level 0 (already created by caller)
  // pitch: pitch in BYTES of the level-0 neighbor pitched allocations
  // d_estLineMap: output Line3D map (caller-allocated, level-0 resolution)
  // d_orient2D: output 2D orientation map (from reference Gabor at level 0)
  // d_variance: output orientation variance map (level 0)
  // d_cost: output total matching cost map (level 0)
  // d_refMask: optional reference mask (nullptr if unused)
  // d_neiMasks: optional neighbor masks (nullptr if unused)
  void Run(float* d_refGrayMap, cudaTextureObject_t* d_neiTexObjs, size_t pitch,
           Line3D* d_estLineMap, float* d_orient2D, float* d_variance, float* d_cost,
           unsigned char* d_refMask, unsigned char** d_neiMasks);

 private:
  void Allocate(std::vector<unsigned int>& widths, std::vector<unsigned int>& heights,
                unsigned int numLevels);
  void Deallocate();

  Mat3x3f intrinsic_;
  std::vector<GpuCamera> nei_cams_;
  AlgorithmParams params_;
  int numnei_;
  unsigned int num_levels_;
  std::string out_folder_;

  PatchMatchEngine* engine_ = nullptr;

  std::vector<unsigned int> widths_, heights_;
  std::vector<Mat3x3f> intrinsic_at_level_;
  std::vector<std::vector<GpuCamera>> nei_cams_at_level_;
  std::vector<AlgorithmParams> params_at_level_;

  // Per-level GPU buffers
  std::vector<float*> ref_gray_;         // level 0 = external pointer
  std::vector<float*> ref_orient_;       // all levels allocated
  std::vector<float*> ref_orient_var_;   // all levels allocated
  std::vector<float**> nei_gray_;        // pitched; level 0 = nullptr (use
                                         // texobj from caller)
  std::vector<float**> nei_orient_;      // pitched; all levels allocated
  std::vector<float**> nei_orient_var_;  // pitched; all levels allocated
  std::vector<Line3D*> est_line_;        // level 0 = external pointer
  std::vector<float*> cost_color_;       // all levels allocated
  std::vector<float*> cost_orient_;      // all levels allocated
  std::vector<float*> cost_total_;       // all levels allocated

  // Texture objects (HOST-side handle arrays, one per neighbor per level)
  std::vector<cudaTextureObject_t*> nei_gray_tex_;        // level 0 = external
  std::vector<cudaTextureObject_t*> nei_orient_tex_;      // all levels
  std::vector<cudaTextureObject_t*> nei_orient_var_tex_;  // all levels

  // Pitches in BYTES for pitched allocations
  std::vector<size_t> pitch_gray_;        // per level (levels > 0 only)
  std::vector<size_t> pitch_orient_;      // per level
  std::vector<size_t> pitch_orient_var_;  // per level
};
