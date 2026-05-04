// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <cstdlib>
#include <string>

// Simple RAII image container
struct Image {
  float* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 0;

  Image() = default;
  Image(int w, int h, int c);
  ~Image();

  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;
  Image(Image&& other) noexcept;
  Image& operator=(Image&& other) noexcept;

  size_t NumPixels() const {
    return static_cast<size_t>(width) * height;
  }
  size_t NumElements() const {
    return NumPixels() * channels;
  }
  bool Empty() const {
    return data == nullptr;
  }
};

namespace image_io {

// Load image as grayscale float [0,1]. Supports PNG, JPG, and EXR.
Image LoadGrayscaleF32(const std::string& path);

// Load PNG mask as uint8 then convert to float (0 or 1)
Image LoadMask(const std::string& path);

// Resize image using stb_image_resize2 (area filter for downsampling)
Image Resize(const Image& src, int new_w, int new_h);

// Write float image (single channel). EXR if path ends in .exr, otherwise raw binary.
bool WriteExrFloat(const std::string& path, const float* data, int width, int height);

// Write color-mapped visualization PNG from single-channel float data.
// Maps values to HSV colormap (red=high, blue=low).
// If normalize=true, maps [min,max] to [0,1] first.
// ignore_above: pixels with value >= this are painted black (excluded from range).
//               Set to a large value (default) to include everything.
bool WriteVisPng(const std::string& path, const float* data, int width, int height,
                 bool normalize = true, float ignore_above = 1e30f);

}  // namespace image_io
