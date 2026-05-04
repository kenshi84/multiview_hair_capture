// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/camera_array.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

#include "common/logger.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool CameraArray::LoadFromJson(const std::string& json_path) {
  std::ifstream ifs(json_path);
  if (!ifs.is_open()) {
    LOG_ERROR("Failed to open cameras.json: %s", json_path.c_str());
    return false;
  }

  nlohmann::json root;
  try {
    ifs >> root;
  } catch (const std::exception& e) {
    LOG_ERROR("JSON parse error: %s", e.what());
    return false;
  }

  auto& cams = root["cameras"];
  cameras_.clear();

  for (auto it = cams.begin(); it != cams.end(); ++it) {
    unsigned int cam_id = std::stoul(it.key());
    auto& cam_data = it.value();

    float intrinsic[9] = {0};
    auto& K = cam_data["intrinsic"];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        intrinsic[r * 3 + c] = K[r][c].get<float>();

    float extrinsic[12] = {0};
    auto& E = cam_data["extrinsic"];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        extrinsic[r * 4 + c] = E[r][c].get<float>();

    float distortion[5] = {0};
    if (cam_data.contains("distortion")) {
      auto& D = cam_data["distortion"];
      for (int i = 0; i < std::min(5, static_cast<int>(D.size())); ++i)
        distortion[i] = D[i].get<float>();
    }

    cameras_.emplace_back(intrinsic, extrinsic, distortion, cam_id);
  }

  LOG_INFO("Loaded %zu cameras from %s", cameras_.size(), json_path.c_str());
  return true;
}

void CameraArray::DownsizeCameras(float factor) {
  for (auto& cam : cameras_)
    cam.Downsize(factor, factor);
}

std::vector<int> CameraArray::SelectNeighborViews(int ref_index, float min_angle_deg,
                                                  float max_angle_deg,
                                                  int max_views) const {
  float min_rad = min_angle_deg / 180.0f * static_cast<float>(M_PI);
  float max_rad = max_angle_deg / 180.0f * static_cast<float>(M_PI);

  // Anchor for angle measurement: centroid of all camera centers.
  // This makes the angle a true baseline angle relative to the rig center,
  // independent of where the world origin is placed.
  float anchor[3] = {0.0f, 0.0f, 0.0f};
  for (const auto& cam : cameras_) {
    const float* c = cam.center();
    anchor[0] += c[0];
    anchor[1] += c[1];
    anchor[2] += c[2];
  }
  const float inv_n = 1.0f / static_cast<float>(cameras_.size());
  anchor[0] *= inv_n;
  anchor[1] *= inv_n;
  anchor[2] *= inv_n;

  // Vector from anchor to reference camera.
  const float* ref_c = cameras_[ref_index].center();
  float ref_v[3] = {ref_c[0] - anchor[0], ref_c[1] - anchor[1], ref_c[2] - anchor[2]};
  float ref_mag =
      std::sqrt(ref_v[0] * ref_v[0] + ref_v[1] * ref_v[1] + ref_v[2] * ref_v[2]);

  std::vector<std::pair<float, int>> candidates;
  for (int i = 0; i < static_cast<int>(cameras_.size()); ++i) {
    if (i == ref_index)
      continue;

    const float* nei_c = cameras_[i].center();
    float nei_v[3] = {nei_c[0] - anchor[0], nei_c[1] - anchor[1], nei_c[2] - anchor[2]};
    float nei_mag =
        std::sqrt(nei_v[0] * nei_v[0] + nei_v[1] * nei_v[1] + nei_v[2] * nei_v[2]);

    float dot = ref_v[0] * nei_v[0] + ref_v[1] * nei_v[1] + ref_v[2] * nei_v[2];
    float cos_angle = dot / (ref_mag * nei_mag + 1e-10f);
    cos_angle = std::min(1.0f, std::max(-1.0f, cos_angle));
    float angle = std::acos(cos_angle);

    if (angle > min_rad && angle < max_rad) {
      candidates.emplace_back(angle, i);
    }
  }

  std::sort(candidates.begin(), candidates.end());

  int n = std::min(max_views, static_cast<int>(candidates.size()));
  std::vector<int> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i)
    result.push_back(candidates[i].second);

  return result;
}
