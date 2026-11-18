// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "mvs/depth_fusion.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "common/logger.h"
#include "common/math_util.h"
#include "common/timer.h"
#include "common/types.h"

namespace depth_fusion {

// Check if a Line3D is valid (non-zero position and direction).
// Invalid lines have both position and direction set to zero.
static inline bool IsValidLine(const Line3D& l) {
  float pp = l.p.x * l.p.x + l.p.y * l.p.y + l.p.z * l.p.z;
  float vv = l.v.x * l.v.x + l.v.y * l.v.y + l.v.z * l.v.z;
  return pp > 0 || vv > 0;
}

// Project a world-space 3D point into a neighbor view.
// Returns the image position (out_x, out_y) in the neighbor's image space.
// The neighbor camera is specified by its world-to-camera [R|t] and scaled K.
static inline bool ProjectToNeighbor(const Vec3f& p_ref, const Mat3x3f& nei_R,
                                     const Vec3f& nei_t, const Mat3x3f& nei_K,
                                     unsigned int nei_w, unsigned int nei_h,
                                     float& out_x, float& out_y) {
  // Transform from world coordinates to neighbor camera coordinates.
  Vec3f p_nei = nei_R * p_ref + nei_t;
  if (p_nei.z <= 0)
    return false;

  // Project to image plane: imgpos = K * p_nei / z (full matrix multiply)
  Vec3f imgpos = nei_K * Vec3f(p_nei.x, p_nei.y, p_nei.z);
  float fx = imgpos.x / imgpos.z;
  float fy = imgpos.y / imgpos.z;

  out_x = fx;
  out_y = fy;

  // Bounds check (with 1-pixel margin for 3x3 patch)
  if (fx < 1.0f || fx >= static_cast<float>(nei_w - 1) || fy < 1.0f ||
      fy >= static_cast<float>(nei_h - 1))
    return false;

  return true;
}

PointCloud FusePointClouds(const std::vector<ViewMvsResult>& view_results,
                           const CameraArray& cameras, const Config& config) {
  ScopedTimer timer("Depth Fusion (image-space cross-validation)");

  float pos_thresh = config.fusion_position_threshold;
  float orient_cos_thresh =
      std::cos(config.fusion_orientation_threshold / 180.0f * static_cast<float>(M_PI));
  int min_neighbors = config.fusion_min_neighbors;
  constexpr int kPatchSize = 1;  // half-size of patch (3x3 = 2*1+1)

  int num_views = static_cast<int>(view_results.size());
  LOG_INFO("  Fusing %d views with image-space cross-validation", num_views);
  LOG_INFO("  pos_thresh=%.1f mm, orient_thresh=%.1f deg, min_nei=%d", pos_thresh,
           config.fusion_orientation_threshold, min_neighbors);

  // view_results[i] corresponds to CameraArray index i.
  // ProcessReferenceView converts valid line positions and directions to world
  // coordinates before storing them. Project reference positions into each
  // neighbor's image to find candidates, then compare both lines directly in
  // world coordinates.

  PointCloud fused;
  size_t total_valid = 0;
  size_t total_fused = 0;

  for (int vi = 0; vi < num_views; vi++) {
    const ViewMvsResult& ref = view_results[vi];
    if (ref.ref_index < 0 || ref.line_map.empty())
      continue;

    unsigned int w = ref.width;
    unsigned int h = ref.height;

    int num_nei = static_cast<int>(ref.nei_indices.size());

    // Precompute each neighbor's world-to-camera transform, scaled intrinsics,
    // and world-space line3D map (if available).
    struct NeiInfo {
      Mat3x3f R_rel;      // world-to-neighbor camera rotation
      Vec3f t_rel;        // world-to-neighbor camera translation
      Mat3x3f R_rel_T;    // transpose (neighbor camera-to-world rotation)
      Mat3x3f K;          // neighbor's scaled intrinsic
      const Line3D* map;  // neighbor's line3D map (world coords)
      unsigned int w, h;  // neighbor image dimensions
    };

    std::vector<NeiInfo> nei_infos(num_nei);
    bool any_nei_valid = false;

    for (int k = 0; k < num_nei; k++) {
      NeiInfo& ni = nei_infos[k];

      int nei_idx = ref.nei_indices[k];

      // Use the neighbor's world-to-camera extrinsics and scaled intrinsics.
      if (nei_idx >= 0 && nei_idx < cameras.NumCameras()) {
        const Camera& nei_cam = cameras.GetCamera(nei_idx);
        GpuCamera nei_gpu_cam = nei_cam.BuildGpuCamera();
        memcpy(ni.K.Ptr(), nei_gpu_cam.K, 9 * sizeof(float));
        memcpy(ni.R_rel.Ptr(), nei_gpu_cam.R, 9 * sizeof(float));
        ni.t_rel = Vec3f(nei_gpu_cam.t[0], nei_gpu_cam.t[1], nei_gpu_cam.t[2]);
      }
      ni.R_rel_T = ni.R_rel.T();

      if (nei_idx >= 0 && nei_idx < num_views && view_results[nei_idx].ref_index >= 0 &&
          !view_results[nei_idx].line_map.empty()) {
        ni.map = view_results[nei_idx].line_map.data();
        ni.w = view_results[nei_idx].width;
        ni.h = view_results[nei_idx].height;
        any_nei_valid = true;
      } else {
        ni.map = nullptr;
        ni.w = 0;
        ni.h = 0;
      }
    }

    if (!any_nei_valid) {
      LOG_WARN("  View %d: no neighbor line3D maps available, skipping", vi);
      continue;
    }

    size_t view_valid = 0;
    size_t view_fused = 0;

    for (unsigned int y = 0; y < h; y++) {
      for (unsigned int x = 0; x < w; x++) {
        unsigned int idx = y * w + x;
        const Line3D& ref_line = ref.line_map[idx];
        if (!IsValidLine(ref_line))
          continue;
        view_valid++;

        Vec3f ref_p(ref_line.p.x, ref_line.p.y, ref_line.p.z);
        Vec3f ref_v(ref_line.v.x, ref_line.v.y, ref_line.v.z);

        int support = 0;

        for (int k = 0; k < num_nei; k++) {
          const NeiInfo& ni = nei_infos[k];
          if (ni.map == nullptr)
            continue;

          // Project the reference view's world-space point into the neighbor image.
          float img_x, img_y;
          if (!ProjectToNeighbor(ref_p, ni.R_rel, ni.t_rel, ni.K, ni.w, ni.h, img_x,
                                 img_y))
            continue;

          // Search 3x3 patch around projected position
          int cx = static_cast<int>(img_x + 0.5f);
          int cy = static_cast<int>(img_y + 0.5f);
          bool found = false;

          for (int dy = -kPatchSize; dy <= kPatchSize && !found; dy++) {
            for (int dx = -kPatchSize; dx <= kPatchSize && !found; dx++) {
              int nx = cx + dx;
              int ny = cy + dy;
              if (nx < 0 || nx >= static_cast<int>(ni.w) || ny < 0 ||
                  ny >= static_cast<int>(ni.h))
                continue;

              unsigned int nei_idx_px = ny * ni.w + nx;
              const Line3D& nei_line = ni.map[nei_idx_px];
              if (!IsValidLine(nei_line))
                continue;

              Vec3f nei_p(nei_line.p.x, nei_line.p.y, nei_line.p.z);
              Vec3f nei_v(nei_line.v.x, nei_line.v.y, nei_line.v.z);

              // Position consistency in world coordinates.
              float dist = (ref_p - nei_p).Magnitude();
              if (dist > pos_thresh)
                continue;

              // Orientation consistency in world coordinates (symmetric direction).
              float dot = ref_v * nei_v;
              if (std::abs(dot) > orient_cos_thresh) {
                found = true;
              }
            }
          }

          if (found) {
            support++;
            if (support >= min_neighbors)
              break;
          }
        }

        if (support >= min_neighbors) {
          view_fused++;

          // Line3D is already in world coordinates
          fused.AddPoint(ref_p.x, ref_p.y, ref_p.z, ref_v.x, ref_v.y, ref_v.z);
        }
      }
    }

    total_valid += view_valid;
    total_fused += view_fused;
    LOG_INFO("  View %d: %zu fused (of %zu valid)", vi, view_fused, view_valid);
  }

  LOG_INFO("  Total fused points: %zu (from %zu valid across %d views)", total_fused,
           total_valid, num_views);
  return fused;
}

}  // namespace depth_fusion
