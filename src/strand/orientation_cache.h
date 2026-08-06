// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/camera.h"
#include "common/camera_array.h"
#include "common/config.h"
#include "common/types.h"

// Persistent level-zero Gabor maps used by multi-view strand growing.
//
// The on-disk representation is deliberately private and versioned.  The
// public representation below is one uint32_t per pixel: the low 16 bits are
// the IEEE binary16 hair-line normal angle (radians, modulo pi), and the high
// 16 bits are its binary16 circular variance.  A non-positive or non-finite
// variance is represented by an all-zero word and is therefore invalid.
namespace orientation_cache {

struct PreparedView {
  int camera_index = -1;
  unsigned int camera_id = 0;
  int width = 0;
  int height = 0;
  GpuCamera camera{};
  std::vector<uint32_t> orientation_variance;

  // Optional combined foreground gate.  When present it has width*height
  // bytes and a nonzero byte means that both configured foreground tests
  // (mask and/or minimum intensity) passed.  Empty means ungated.
  std::vector<uint8_t> boundary;
};

// Path of the cache file for a physical camera.
std::string CachePath(const Config& config, unsigned int camera_id);

// Compute the cache fingerprint from source-image contents and metadata, all
// relevant image/Gabor/grow configuration, and the supplied (already
// downsampled) camera calibration. Returns false when a required source is not
// readable.
bool ComputeFingerprint(const Config& config, const Camera& camera,
                        int camera_index, uint64_t* fingerprint);

// CPU-testable, validated cache I/O.  StoreView uses the current fingerprint
// and atomically replaces a complete cache file.  LoadView rejects stale,
// truncated, oversized, corrupt, or incompatible files.
bool StoreView(const Config& config, const Camera& camera,
               const PreparedView& view);
bool LoadView(const Config& config, const Camera& camera, int camera_index,
              PreparedView* view);

// Persist Gabor maps that are already resident on a GPU (the MVS fast path).
// A zero pitch means a tightly packed width*sizeof(float) row.  When gpu_id is
// negative, the current CUDA device is retained.  Foreground boundary bytes,
// when configured, are prepared from the same source image and mask.
bool StoreDeviceMaps(const Config& config, const Camera& camera,
                     int camera_index, int width, int height,
                     const float* d_theta, const float* d_variance,
                     size_t theta_pitch_bytes = 0,
                     size_t variance_pitch_bytes = 0, int gpu_id = -1);

// Load a current cache entry or regenerate it by loading/downsampling and,
// when requested, undistorting the source image before invoking the existing
// CUDA Gabor implementation.
bool EnsureCache(const Config& config, const Camera& camera, int camera_index,
                 int gpu_id, PreparedView* view);

// Prepare every usable camera.  Unreadable/corrupt/incompatible views are
// skipped with warnings.  All returned records share one image resolution;
// callers enforce their algorithm-specific minimum-view count.
std::vector<PreparedView> PrepareViews(const CameraArray& cameras,
                                       const Config& config, int gpu_id);

}  // namespace orientation_cache
