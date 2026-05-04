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

struct Config {
  // [data]
  std::string cameras_json;
  std::string image_dir;
  std::string mask_dir;
  std::string output_dir;
  float downsample = 1.0f;
  bool distorted_images = false;

  // [mvs]
  float min_depth = 800.0f;
  float max_depth = 1400.0f;
  int iterations = 8;
  int patch_size = 21;
  int num_hierarchy_levels = 4;
  int num_neighbor_views = 25;
  float alpha = 0.1f;
  bool use_mask = false;
  int num_gpus = 1;
  float min_angle = 1.0f;
  float max_angle = 90.0f;
  float pt_sample_radius = 10.0f;
  int pt_sample_kappa = 41;
  float delta_orient = 0.8f;
  float delta_depth = 30.0f;
  float spatial_prop_radius = 5.0f;
  int num_view_select = 4;
  bool gaussian_pyramid = false;

  // [mvs.gabor]
  int gabor_num_orientations = 180;
  int gabor_kernel_size = 21;
  float gabor_sigma = 1.12f;
  float gabor_gamma = 0.28f;
  float gabor_lambda = 3.00f;

  // [mvs.fusion]
  float fusion_position_threshold = 1.0f;
  float fusion_orientation_threshold = 10.0f;
  int fusion_min_neighbors = 4;
  float fusion_cost_threshold = 0.2f;

  // [meanshift]
  float meanshift_neighbor_radius = 2.0f;
  int meanshift_min_neighbors = 10;
  float meanshift_sigma_position = 0.1f;
  float meanshift_sigma_orientation = 30.0f;
  float meanshift_convergence = 0.002f;
  int meanshift_max_iterations = 1000;

  // [trace]
  float trace_step_size = 0.1f;
  float trace_neighbor_radius = 0.1f;
  float trace_angle_threshold = 30.0f;
  float trace_min_strand_length = 2.0f;

  // [clean]
  float clean_min_length = 5.0f;
  float clean_outlier_radius = 10.0f;
  int clean_outlier_min_neighbors = 3;

  // [debug]
  bool save_intermediates = false;
  int gpu_id = 0;
  std::string log_level = "info";
  std::string log_file;
  bool profile = true;

  static Config LoadFromToml(const std::string& path);
};
