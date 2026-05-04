// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "mvs/image_loader.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "common/logger.h"

namespace image_loader {

Image LoadGrayImage(const std::string& path, float downsample) {
  Image img = image_io::LoadGrayscaleF32(path);
  if (img.Empty())
    return {};

  if (downsample != 1.0f) {
    int new_w = static_cast<int>(img.width * downsample);
    int new_h = static_cast<int>(img.height * downsample);
    img = image_io::Resize(img, new_w, new_h);
  }

  return img;
}

Image LoadMaskImage(const std::string& path, float downsample) {
  Image img = image_io::LoadMask(path);
  if (img.Empty())
    return {};

  if (downsample != 1.0f) {
    int new_w = static_cast<int>(img.width * downsample);
    int new_h = static_cast<int>(img.height * downsample);
    img = image_io::Resize(img, new_w, new_h);
  }

  return img;
}

void UndistortImage(Image& img, const float K[9], const float dist[5]) {
  if (img.Empty())
    return;

  int w = img.width, h = img.height;
  float fx = K[0], fy = K[4], cx = K[2], cy = K[5];

  // For each undistorted output pixel, apply the forward distortion model to find
  // the corresponding source pixel in the distorted input image, then bilinear sample.
  // No iteration needed — forward distortion is a direct polynomial evaluation.
  std::vector<float> out(w * h, 0.0f);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      // Normalize undistorted pixel
      float xn = (static_cast<float>(x) - cx) / fx;
      float yn = (static_cast<float>(y) - cy) / fy;

      // Apply distortion to get distorted normalized coords
      float r2 = xn * xn + yn * yn;
      float r4 = r2 * r2;
      float r6 = r4 * r2;
      float rd = 1.0f + dist[0] * r2 + dist[1] * r4 + dist[4] * r6;
      float xd = xn * rd + 2.0f * dist[2] * xn * yn + dist[3] * (r2 + 2.0f * xn * xn);
      float yd = yn * rd + dist[2] * (r2 + 2.0f * yn * yn) + 2.0f * dist[3] * xn * yn;

      // Convert back to pixel coords in distorted image
      float sx = xd * fx + cx;
      float sy = yd * fy + cy;

      // Bilinear sample from distorted image
      int x0 = static_cast<int>(std::floor(sx));
      int y0 = static_cast<int>(std::floor(sy));
      if (x0 < 0 || x0 >= w - 1 || y0 < 0 || y0 >= h - 1)
        continue;

      float fx_ = sx - x0, fy_ = sy - y0;
      float v = (1 - fx_) * (1 - fy_) * img.data[y0 * w + x0] +
                fx_ * (1 - fy_) * img.data[y0 * w + x0 + 1] +
                (1 - fx_) * fy_ * img.data[(y0 + 1) * w + x0] +
                fx_ * fy_ * img.data[(y0 + 1) * w + x0 + 1];
      out[y * w + x] = v;
    }
  }

  memcpy(img.data, out.data(), w * h * sizeof(float));
}

std::string FormatPath(const std::string& pattern, unsigned int cam_id) {
  char buf[1024];
  snprintf(buf, sizeof(buf), pattern.c_str(), cam_id);
  return std::string(buf);
}

}  // namespace image_loader
