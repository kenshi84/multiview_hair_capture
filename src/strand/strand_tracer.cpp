// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "strand/strand_tracer.h"

#include <cmath>
#include <vector>

#include "common/kdtree.h"
#include "common/logger.h"
#include "common/timer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace strand_tracer {

namespace {

// Forward Euler step with multi-step lookahead.
// Probes at increasing distances along the current direction, searching for
// valid neighbors at each probe point. Uses uniform weighting on neighbor
// positions (not Gaussian). If found beyond step_size, reprojects back.
bool ForwardEulerStep(const float pos[3], const float dir[3], const PointCloud& cloud,
                      const KdTree& tree, float nei_radius, float step_size,
                      float max_lookahead, float angle_thresh_deg,
                      const std::vector<bool>& is_removed, float out_pos[3],
                      float out_dir[3]) {
  std::vector<nanoflann::ResultItem<KdTreeIndex, float>> neighbors;

  for (float move_step = step_size; move_step < max_lookahead; move_step += step_size) {
    // Probe position along current direction
    float probe[3] = {pos[0] + move_step * dir[0], pos[1] + move_step * dir[1],
                      pos[2] + move_step * dir[2]};

    tree.RadiusSearch(probe, nei_radius, neighbors);

    // Average valid neighbor positions and directions (uniform weight)
    float sum_px = 0, sum_py = 0, sum_pz = 0;
    float sum_dx = 0, sum_dy = 0, sum_dz = 0;
    int cnt_valid = 0;

    for (const auto& nb : neighbors) {
      size_t idx = nb.first;
      if (is_removed[idx])
        continue;

      const float* nb_pos = cloud.PositionPtr(idx);
      const float* nb_dir_raw = cloud.DirectionPtr(idx);

      // Normalize neighbor direction
      float nd[3] = {nb_dir_raw[0], nb_dir_raw[1], nb_dir_raw[2]};
      float nd_len = std::sqrt(nd[0] * nd[0] + nd[1] * nd[1] + nd[2] * nd[2]);
      if (nd_len < 1e-10f)
        continue;
      nd[0] /= nd_len;
      nd[1] /= nd_len;
      nd[2] /= nd_len;

      // Handle direction symmetry
      float dot = dir[0] * nd[0] + dir[1] * nd[1] + dir[2] * nd[2];
      if (dot < 0) {
        nd[0] = -nd[0];
        nd[1] = -nd[1];
        nd[2] = -nd[2];
        dot = -dot;
      }

      // Angle check in degrees
      float angle = std::acos(std::min(1.0f, std::max(0.0f, dot))) /
                    static_cast<float>(M_PI) * 180.0f;

      // Distance from current position to neighbor (skip if too close)
      float dx = pos[0] - nb_pos[0];
      float dy = pos[1] - nb_pos[1];
      float dz = pos[2] - nb_pos[2];
      float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

      if (angle < angle_thresh_deg && dist > 1e-3f) {
        cnt_valid++;
        sum_px += nb_pos[0];
        sum_py += nb_pos[1];
        sum_pz += nb_pos[2];
        sum_dx += nd[0];
        sum_dy += nd[1];
        sum_dz += nd[2];
      }
    }

    if (cnt_valid > 0) {
      float avg_px = sum_px / cnt_valid;
      float avg_py = sum_py / cnt_valid;
      float avg_pz = sum_pz / cnt_valid;

      // Normalize averaged direction
      float avg_len = std::sqrt(sum_dx * sum_dx + sum_dy * sum_dy + sum_dz * sum_dz);
      if (avg_len < 1e-10f)
        return false;

      if (move_step != step_size) {
        // Found at lookahead > step_size: re-project to step_size distance
        float re_dx = avg_px - pos[0];
        float re_dy = avg_py - pos[1];
        float re_dz = avg_pz - pos[2];
        float re_len = std::sqrt(re_dx * re_dx + re_dy * re_dy + re_dz * re_dz);
        if (re_len < 1e-10f)
          return false;
        out_dir[0] = re_dx / re_len;
        out_dir[1] = re_dy / re_len;
        out_dir[2] = re_dz / re_len;
        out_pos[0] = pos[0] + step_size * out_dir[0];
        out_pos[1] = pos[1] + step_size * out_dir[1];
        out_pos[2] = pos[2] + step_size * out_dir[2];
      } else {
        out_pos[0] = avg_px;
        out_pos[1] = avg_py;
        out_pos[2] = avg_pz;
        out_dir[0] = sum_dx / avg_len;
        out_dir[1] = sum_dy / avg_len;
        out_dir[2] = sum_dz / avg_len;
      }
      return true;
    }
  }

  return false;
}

// Mark input points within `radius` of the strand line as used.
// Implementation: 2*radius sphere search, then perpendicular-distance filter.
void RemovePointsNearStrand(const Strand& strand, const PointCloud& cloud,
                            std::vector<bool>& is_removed, const KdTree& tree,
                            float radius) {
  std::vector<nanoflann::ResultItem<KdTreeIndex, float>> neighbors;
  for (size_t i = 0; i < strand.NumPoints(); ++i) {
    const float* sp = &strand.positions[i * 3];
    const float* sd = &strand.directions[i * 3];

    // Sphere search at 2x radius
    tree.RadiusSearch(sp, 2.0f * radius, neighbors);

    // Cylinder filter: perpendicular distance from point-to-line
    for (const auto& nb : neighbors) {
      const float* np = cloud.PositionPtr(nb.first);
      // Vector from strand point to neighbor
      float dx = np[0] - sp[0];
      float dy = np[1] - sp[1];
      float dz = np[2] - sp[2];
      // Project onto strand direction
      float proj = dx * sd[0] + dy * sd[1] + dz * sd[2];
      // Perpendicular distance squared
      float perp_sq = (dx * dx + dy * dy + dz * dz) - proj * proj;
      if (perp_sq < radius * radius) {
        is_removed[nb.first] = true;
      }
    }
  }
}

}  // namespace

std::vector<Strand> Trace(const PointCloud& cloud, const Config& config) {
  ScopedTimer timer("Strand Tracing");

  size_t n = cloud.NumPoints();
  LOG_INFO("Tracing strands from %zu points", n);

  float step_size = config.trace_step_size;
  float nei_radius = config.trace_neighbor_radius;
  float angle_thresh_deg = config.trace_angle_threshold;  // in degrees
  float min_length = config.trace_min_strand_length;
  float max_lookahead = min_length;  // max lookahead distance = min strand length

  // Build KD-tree
  KdTree tree;
  tree.Build(cloud.positions.data(), n);

  std::vector<bool> is_removed(n, false);
  std::vector<Strand> strands;
  int label = 0;

  for (size_t seed = 0; seed < n; ++seed) {
    if (is_removed[seed])
      continue;

    // Trace forward
    Strand forward_strand;
    float pos[3] = {cloud.positions[seed * 3], cloud.positions[seed * 3 + 1],
                    cloud.positions[seed * 3 + 2]};
    float dir[3] = {cloud.directions[seed * 3], cloud.directions[seed * 3 + 1],
                    cloud.directions[seed * 3 + 2]};

    forward_strand.AddPoint(pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], label);

    int max_steps = 10000;
    for (int step = 0; step < max_steps; ++step) {
      float new_pos[3], new_dir[3];
      if (!ForwardEulerStep(pos, dir, cloud, tree, nei_radius, step_size, max_lookahead,
                            angle_thresh_deg, is_removed, new_pos, new_dir)) {
        break;
      }
      forward_strand.AddPoint(new_pos[0], new_pos[1], new_pos[2], new_dir[0],
                              new_dir[1], new_dir[2], label);
      pos[0] = new_pos[0];
      pos[1] = new_pos[1];
      pos[2] = new_pos[2];
      dir[0] = new_dir[0];
      dir[1] = new_dir[1];
      dir[2] = new_dir[2];
    }

    // Trace backward (negate direction)
    Strand backward_strand;
    pos[0] = cloud.positions[seed * 3];
    pos[1] = cloud.positions[seed * 3 + 1];
    pos[2] = cloud.positions[seed * 3 + 2];
    dir[0] = -cloud.directions[seed * 3];
    dir[1] = -cloud.directions[seed * 3 + 1];
    dir[2] = -cloud.directions[seed * 3 + 2];

    for (int step = 0; step < max_steps; ++step) {
      float new_pos[3], new_dir[3];
      if (!ForwardEulerStep(pos, dir, cloud, tree, nei_radius, step_size, max_lookahead,
                            angle_thresh_deg, is_removed, new_pos, new_dir)) {
        break;
      }
      backward_strand.AddPoint(new_pos[0], new_pos[1], new_pos[2], new_dir[0],
                               new_dir[1], new_dir[2], label);
      pos[0] = new_pos[0];
      pos[1] = new_pos[1];
      pos[2] = new_pos[2];
      dir[0] = new_dir[0];
      dir[1] = new_dir[1];
      dir[2] = new_dir[2];
    }

    // Merge: reverse backward + forward
    Strand full_strand;
    for (int i = static_cast<int>(backward_strand.NumPoints()) - 1; i >= 0; --i) {
      full_strand.AddPoint(
          backward_strand.positions[i * 3], backward_strand.positions[i * 3 + 1],
          backward_strand.positions[i * 3 + 2], -backward_strand.directions[i * 3],
          -backward_strand.directions[i * 3 + 1],
          -backward_strand.directions[i * 3 + 2], label);
    }
    for (size_t i = 0; i < forward_strand.NumPoints(); ++i) {
      full_strand.AddPoint(
          forward_strand.positions[i * 3], forward_strand.positions[i * 3 + 1],
          forward_strand.positions[i * 3 + 2], forward_strand.directions[i * 3],
          forward_strand.directions[i * 3 + 1], forward_strand.directions[i * 3 + 2],
          label);
    }

    if (full_strand.Length() > min_length) {
      strands.push_back(std::move(full_strand));
      label++;
    }

    // Mark nearby points as used
    RemovePointsNearStrand(forward_strand, cloud, is_removed, tree, nei_radius);
    RemovePointsNearStrand(backward_strand, cloud, is_removed, tree, nei_radius);
    is_removed[seed] = true;
  }

  LOG_INFO("Generated %zu strands", strands.size());
  return strands;
}

}  // namespace strand_tracer
