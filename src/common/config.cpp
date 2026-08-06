// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/config.h"

#include <cmath>
#include <limits>
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

  // [grow]
  f("grow", "step_size", cfg.grow_step_size);
  f("grow", "cone_half_angle", cfg.grow_cone_half_angle);
  f("grow", "direction_sample_step", cfg.grow_direction_sample_step);
  i("grow", "window_width", cfg.grow_window_width);
  i("grow", "window_length", cfg.grow_window_length);
  f("grow", "max_pixel_angle", cfg.grow_max_pixel_angle);
  i("grow", "min_scored_pixels", cfg.grow_min_scored_pixels);
  i("grow", "min_views", cfg.grow_min_views);
  f("grow", "max_direction_change", cfg.grow_max_direction_change);
  i("grow", "irls_iterations", cfg.grow_irls_iterations);
  f("grow", "max_growth_length", cfg.grow_max_growth_length);
  b("grow", "use_mask", cfg.grow_use_mask);
  f("grow", "min_intensity", cfg.grow_min_intensity);

  // [debug]
  b("debug", "save_intermediates", cfg.save_intermediates);
  i("debug", "gpu_id", cfg.gpu_id);
  str("debug", "log_level", cfg.log_level);
  str("debug", "log_file", cfg.log_file);
  b("debug", "profile", cfg.profile);

  auto require_finite_positive = [](float value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0f)
      throw std::runtime_error(std::string(name) + " must be finite and > 0");
  };
  auto require_finite_range = [](float value, float lo, float hi,
                                 const char* name) {
    if (!std::isfinite(value) || value < lo || value > hi) {
      throw std::runtime_error(std::string(name) + " must be in [" +
                               std::to_string(lo) + ", " +
                               std::to_string(hi) + "]");
    }
  };

  require_finite_positive(cfg.grow_step_size, "grow.step_size");
  if (!std::isfinite(cfg.grow_cone_half_angle) ||
      cfg.grow_cone_half_angle < 0.0f || cfg.grow_cone_half_angle >= 90.0f) {
    throw std::runtime_error(
        "grow.cone_half_angle must be finite and in [0, 90)");
  }
  require_finite_positive(cfg.grow_direction_sample_step,
                          "grow.direction_sample_step");
  if (cfg.grow_cone_half_angle > 0.0f &&
      cfg.grow_direction_sample_step > cfg.grow_cone_half_angle) {
    throw std::runtime_error(
        "grow.direction_sample_step cannot exceed grow.cone_half_angle");
  }
  const double grow_half_direction_samples =
      std::floor(static_cast<double>(cfg.grow_cone_half_angle) /
                 cfg.grow_direction_sample_step);
  if (grow_half_direction_samples >
      (static_cast<double>(std::numeric_limits<int>::max()) - 1.0) / 2.0) {
    throw std::runtime_error(
        "grow.cone_half_angle / grow.direction_sample_step is too large");
  }
  if (cfg.grow_window_width <= 0)
    throw std::runtime_error("grow.window_width must be > 0");
  if (cfg.grow_window_width % 2 == 0)
    throw std::runtime_error("grow.window_width must be odd");
  if (cfg.grow_window_length <= 0)
    throw std::runtime_error("grow.window_length must be > 0");
  require_finite_range(cfg.grow_max_pixel_angle, 0.0f, 90.0f,
                       "grow.max_pixel_angle");
  if (cfg.grow_min_scored_pixels <= 0)
    throw std::runtime_error("grow.min_scored_pixels must be > 0");
  const long long max_window_pixels =
      static_cast<long long>(cfg.grow_window_width) * cfg.grow_window_length;
  if (cfg.grow_min_scored_pixels > max_window_pixels) {
    throw std::runtime_error(
        "grow.min_scored_pixels cannot exceed grow.window_width * "
        "grow.window_length");
  }
  if (cfg.grow_min_views < 2)
    throw std::runtime_error("grow.min_views must be >= 2");
  if (!std::isfinite(cfg.grow_max_direction_change) ||
      cfg.grow_max_direction_change <= 0.0f ||
      cfg.grow_max_direction_change > 180.0f) {
    throw std::runtime_error(
        "grow.max_direction_change must be finite and in (0, 180]");
  }
  if (cfg.grow_irls_iterations < 0)
    throw std::runtime_error("grow.irls_iterations must be >= 0");
  require_finite_positive(cfg.grow_max_growth_length,
                          "grow.max_growth_length");
  const double growth_step_ratio =
      static_cast<double>(cfg.grow_max_growth_length) / cfg.grow_step_size;
  const double nearest_growth_steps = std::round(growth_step_ratio);
  const double growth_step_tolerance =
      32.0 * std::numeric_limits<float>::epsilon() *
      std::fmax(1.0, std::fabs(growth_step_ratio));
  const double stable_growth_steps =
      std::fabs(growth_step_ratio - nearest_growth_steps) <=
              growth_step_tolerance
          ? nearest_growth_steps
          : std::floor(growth_step_ratio);
  if (stable_growth_steps > std::numeric_limits<int>::max())
    throw std::runtime_error("grow.max_growth_length / grow.step_size is too large");
  require_finite_range(cfg.grow_min_intensity, 0.0f, 1.0f,
                       "grow.min_intensity");

  return cfg;
}
