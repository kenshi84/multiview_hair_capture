// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/image_io.h"

#include <cstring>

#include "common/logger.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

// --- Image RAII ---

Image::Image(int w, int h, int c) : width(w), height(h), channels(c) {
  data = static_cast<float*>(calloc(static_cast<size_t>(w) * h * c, sizeof(float)));
}

Image::~Image() {
  free(data);
}

Image::Image(Image&& other) noexcept
    : data(other.data),
      width(other.width),
      height(other.height),
      channels(other.channels) {
  other.data = nullptr;
  other.width = other.height = other.channels = 0;
}

Image& Image::operator=(Image&& other) noexcept {
  if (this != &other) {
    free(data);
    data = other.data;
    width = other.width;
    height = other.height;
    channels = other.channels;
    other.data = nullptr;
    other.width = other.height = other.channels = 0;
  }
  return *this;
}

namespace image_io {

static bool HasExtension(const std::string& path, const std::string& ext) {
  if (path.size() < ext.size())
    return false;
  std::string tail = path.substr(path.size() - ext.size());
  for (auto& c : tail)
    c = static_cast<char>(tolower(c));
  return tail == ext;
}

Image LoadGrayscaleF32(const std::string& path) {
  // EXR path: load float data directly via tinyexr
  if (HasExtension(path, ".exr")) {
    float* out = nullptr;
    int w, h;
    const char* err = nullptr;
    int ret = LoadEXR(&out, &w, &h, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
      LOG_ERROR("Failed to load EXR: %s (%s)", path.c_str(),
                err ? err : "unknown error");
      FreeEXRErrorMessage(err);
      return {};
    }
    // LoadEXR returns RGBA float. Convert to grayscale using Rec. 709 luminance.
    Image img(w, h, 1);
    for (int i = 0; i < w * h; ++i) {
      float r = out[i * 4], g = out[i * 4 + 1], b = out[i * 4 + 2];
      img.data[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }
    free(out);
    return img;
  }

  // PNG/JPG path: load uint8 via stb_image, convert to float
  int w, h, c;
  unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, 1);
  if (!raw) {
    LOG_ERROR("Failed to load image: %s", path.c_str());
    return {};
  }

  Image img(w, h, 1);
  for (int i = 0; i < w * h; ++i) {
    img.data[i] = raw[i] / 255.0f;
  }
  stbi_image_free(raw);
  return img;
}

Image LoadMask(const std::string& path) {
  int w, h, c;
  unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, 1);
  if (!raw) {
    LOG_ERROR("Failed to load mask: %s", path.c_str());
    return {};
  }

  Image img(w, h, 1);
  for (int i = 0; i < w * h; ++i) {
    img.data[i] = raw[i] > 0 ? 1.0f : 0.0f;
  }
  stbi_image_free(raw);
  return img;
}

Image Resize(const Image& src, int new_w, int new_h) {
  if (src.Empty())
    return {};

  Image dst(new_w, new_h, src.channels);

  // stb_image_resize2 works on uint8 or float; use float path
  stbir_resize_float_linear(src.data, src.width, src.height,
                            src.width * src.channels * sizeof(float), dst.data, new_w,
                            new_h, new_w * src.channels * sizeof(float),
                            static_cast<stbir_pixel_layout>(src.channels));

  return dst;
}

bool WriteExrFloat(const std::string& path, const float* data, int width, int height) {
  if (HasExtension(path, ".exr")) {
    // Write native EXR via tinyexr
    const char* err = nullptr;
    int ret = SaveEXR(data, width, height, 1, 0, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
      LOG_ERROR("Failed to write EXR: %s (%s)", path.c_str(),
                err ? err : "unknown error");
      FreeEXRErrorMessage(err);
      return false;
    }
    return true;
  }

  // Write as raw float binary: header (w,h as uint32) + float data
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    LOG_ERROR("Failed to write float image: %s", path.c_str());
    return false;
  }
  uint32_t dims[2] = {(uint32_t) width, (uint32_t) height};
  fwrite(dims, sizeof(uint32_t), 2, f);
  fwrite(data, sizeof(float), (size_t) width * height, f);
  fclose(f);
  return true;
}

// HSV to RGB conversion (h in [0,360), s,v in [0,1])
static void HsvToRgb(float h, float s, float v, unsigned char& r, unsigned char& g,
                     unsigned char& b) {
  float c = v * s;
  float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
  float m = v - c;
  float r1, g1, b1;
  if (h < 60) {
    r1 = c;
    g1 = x;
    b1 = 0;
  } else if (h < 120) {
    r1 = x;
    g1 = c;
    b1 = 0;
  } else if (h < 180) {
    r1 = 0;
    g1 = c;
    b1 = x;
  } else if (h < 240) {
    r1 = 0;
    g1 = x;
    b1 = c;
  } else if (h < 300) {
    r1 = x;
    g1 = 0;
    b1 = c;
  } else {
    r1 = c;
    g1 = 0;
    b1 = x;
  }
  r = static_cast<unsigned char>((r1 + m) * 255.0f);
  g = static_cast<unsigned char>((g1 + m) * 255.0f);
  b = static_cast<unsigned char>((b1 + m) * 255.0f);
}

bool WriteVisPng(const std::string& path, const float* data, int width, int height,
                 bool normalize, float ignore_above) {
  int npix = width * height;

  // Find range for normalization, ignoring pixels above threshold
  float vmin = 1e30f, vmax = -1e30f;
  if (normalize) {
    for (int i = 0; i < npix; i++) {
      if (data[i] >= ignore_above || data[i] == 0.0f)
        continue;
      if (data[i] < vmin)
        vmin = data[i];
      if (data[i] > vmax)
        vmax = data[i];
    }
    if (vmin > vmax) {
      vmin = 0.0f;
      vmax = 1.0f;
    }
  } else {
    vmin = 0.0f;
    vmax = 1.0f;
  }
  float range = vmax - vmin;
  if (range < 1e-10f)
    range = 1.0f;

  // Map to HSV colormap and write RGB PNG
  std::vector<unsigned char> rgb(npix * 3);
  for (int i = 0; i < npix; i++) {
    // Black for zero/invalid or ignored pixels
    if (data[i] == 0.0f || data[i] >= ignore_above) {
      rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = 0;
    } else {
      float t = (data[i] - vmin) / range;
      if (t < 0)
        t = 0;
      if (t > 1)
        t = 1;
      float hue = (1.0f - t) * 240.0f;  // red(high) → blue(low)
      HsvToRgb(hue, 1.0f, 1.0f, rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
    }
  }

  int ret = stbi_write_png(path.c_str(), width, height, 3, rgb.data(), width * 3);
  if (!ret) {
    LOG_ERROR("Failed to write vis PNG: %s", path.c_str());
    return false;
  }
  return true;
}

}  // namespace image_io
