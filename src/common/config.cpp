// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/config.h"

#include <stdexcept>

#define TOML_HEADER_ONLY 1
#include "toml.hpp"

Config Config::LoadFromToml(const std::string& path) {
  Config cfg;

  toml::table tbl;
  try {
    tbl = toml::parse_file(path);
  } catch (const toml::parse_error& err) {
    throw std::runtime_error("Failed to parse config " + path + ": " +
                             std::string(err.description()));
  }

  auto str = [&](const char* section, const char* key, std::string& out) {
    if (auto v = tbl[section][key].value<std::string>())
      out = *v;
  };
  auto i = [&](const char* section, const char* key, int& out) {
    if (auto v = tbl[section][key].value<int64_t>())
      out = static_cast<int>(*v);
  };
  auto f = [&](const char* section, const char* key, float& out) {
    if (auto v = tbl[section][key].value<double>())
      out = static_cast<float>(*v);
  };
  auto b = [&](const char* section, const char* key, bool& out) {
    if (auto v = tbl[section][key].value<bool>())
      out = *v;
  };
  auto f2 = [&](const char* s1, const char* s2, const char* key, float& out) {
    if (auto v = tbl[s1][s2][key].value<double>())
      out = static_cast<float>(*v);
  };
  auto i2 = [&](const char* s1, const char* s2, const char* key, int& out) {
    if (auto v = tbl[s1][s2][key].value<int64_t>())
      out = static_cast<int>(*v);
  };

  // [data]
  str("data", "cameras_json", cfg.cameras_json);
  str("data", "image_dir", cfg.image_dir);
  str("data", "mask_dir", cfg.mask_dir);
  str("data", "output_dir", cfg.output_dir);
  f("data", "downsample", cfg.downsample);
  b("data", "distorted_images", cfg.distorted_images);

  // [mvs]
  f("mvs", "min_depth", cfg.min_depth);
  f("mvs", "max_depth", cfg.max_depth);
  i("mvs", "iterations", cfg.iterations);
  i("mvs", "patch_size", cfg.patch_size);
  i("mvs", "num_hierarchy_levels", cfg.num_hierarchy_levels);
  i("mvs", "num_neighbor_views", cfg.num_neighbor_views);
  f("mvs", "alpha", cfg.alpha);
  b("mvs", "use_mask", cfg.use_mask);
  i("mvs", "mask_min_neighbor_views", cfg.mask_min_neighbor_views);
  i("mvs", "num_gpus", cfg.num_gpus);
  f("mvs", "min_angle", cfg.min_angle);
  f("mvs", "max_angle", cfg.max_angle);
  f("mvs", "pt_sample_radius", cfg.pt_sample_radius);
  i("mvs", "pt_sample_kappa", cfg.pt_sample_kappa);
  f("mvs", "delta_orient", cfg.delta_orient);
  f("mvs", "delta_depth", cfg.delta_depth);
  f("mvs", "spatial_prop_radius", cfg.spatial_prop_radius);
  i("mvs", "num_view_select", cfg.num_view_select);
  b("mvs", "gaussian_pyramid", cfg.gaussian_pyramid);

  // [mvs.gabor]
  i2("mvs", "gabor", "num_orientations", cfg.gabor_num_orientations);
  i2("mvs", "gabor", "kernel_size", cfg.gabor_kernel_size);
  f2("mvs", "gabor", "sigma", cfg.gabor_sigma);
  f2("mvs", "gabor", "gamma", cfg.gabor_gamma);
  f2("mvs", "gabor", "lambda", cfg.gabor_lambda);
  f2("mvs", "gabor", "min_contrast", cfg.gabor_min_contrast);
  f2("mvs", "gabor", "min_response", cfg.gabor_min_response);

  // [mvs.fusion]
  f2("mvs", "fusion", "position_threshold", cfg.fusion_position_threshold);
  f2("mvs", "fusion", "orientation_threshold", cfg.fusion_orientation_threshold);
  i2("mvs", "fusion", "min_neighbors", cfg.fusion_min_neighbors);
  f2("mvs", "fusion", "cost_threshold", cfg.fusion_cost_threshold);

  // [meanshift]
  f("meanshift", "neighbor_radius", cfg.meanshift_neighbor_radius);
  i("meanshift", "min_neighbors", cfg.meanshift_min_neighbors);
  f("meanshift", "sigma_position", cfg.meanshift_sigma_position);
  f("meanshift", "sigma_orientation", cfg.meanshift_sigma_orientation);
  f("meanshift", "convergence", cfg.meanshift_convergence);
  i("meanshift", "max_iterations", cfg.meanshift_max_iterations);

  // [trace]
  f("trace", "step_size", cfg.trace_step_size);
  f("trace", "neighbor_radius", cfg.trace_neighbor_radius);
  f("trace", "angle_threshold", cfg.trace_angle_threshold);
  f("trace", "min_strand_length", cfg.trace_min_strand_length);

  // [clean]
  f("clean", "min_length", cfg.clean_min_length);
  f("clean", "outlier_radius", cfg.clean_outlier_radius);
  i("clean", "outlier_min_neighbors", cfg.clean_outlier_min_neighbors);

  // [debug]
  b("debug", "save_intermediates", cfg.save_intermediates);
  i("debug", "gpu_id", cfg.gpu_id);
  str("debug", "log_level", cfg.log_level);
  str("debug", "log_file", cfg.log_file);
  b("debug", "profile", cfg.profile);

  return cfg;
}
