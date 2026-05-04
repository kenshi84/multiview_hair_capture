// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// Algorithm parameters for PatchMatch solver

#pragma once

struct PatchMatchParams {
  int box_hsize = 10;
  int box_vsize = 10;
  float depthMin = 800.0f;
  float depthMax = 1400.0f;
  float deltaNormal = 0.2f;
  float deltaDisp = 50.0f;
  bool bMaskInput = false;
  int maxIt = 8;

  // Hair-specific defaults
  float hair_alpha = 0.1f;
  float hair_pt_sample_radius = 10.0f;
  int hair_pt_sample_kappa = 41;
  float hair_delta_orient = 0.8f;
  float hair_delta_depth = 30.0f;
  float hair_spatial_prop_radius = 5.0f;
  int hair_num_view_select = 4;
};

// Full algorithm parameters (superset used by orchestration)
struct AlgorithmParams {
  bool hair_gaussian_pyramid = true;
  int hair_orient_rotate_res = 180;
  int hair_orient_gabor_ksize = 21;
  float hair_orient_gabor_sigma = 1.12f;
  float hair_orient_gabor_gamma = 0.28f;
  float hair_orient_gabor_lambd = 3.00f;
  float hair_alpha = 0.1f;
  float hair_pt_sample_radius = 10.0f;
  int hair_pt_sample_kappa = 41;
  float hair_delta_orient = 0.8f;
  float hair_delta_depth = 30.0f;
  float hair_spatial_prop_radius = 5.0f;
  int hair_num_view_select = 4;

  int box_hsize = 10;
  int box_vsize = 10;
  int iterations = 8;
  float depthMin = -1.0f;
  float depthMax = -1.0f;
  float deltaNormal = 0.8f;
  float deltaDisp = 50.0f;
  bool bReconWithMask = false;
  int cols = 0;
  int rows = 0;
};
