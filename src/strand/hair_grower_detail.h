// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// CPU-testable primitives shared by the CUDA implementation.  This is a
// private header: production users should include strand/hair_grower.h.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/config.h"
#include "common/types.h"
#include "strand/strand_io.h"

namespace hair_grower {
namespace detail {

constexpr float kGrowPi = 3.14159265358979323846f;

struct GrowParams {
  float step_size = 0.1f;
  float cone_half_angle = 5.0f * kGrowPi / 180.0f;
  float direction_sample_step = kGrowPi / 180.0f;
  int direction_samples = 11;
  int window_width = 3;
  int window_length = 10;
  float max_pixel_angle = 5.0f * kGrowPi / 180.0f;
  int min_scored_pixels = 10;
  int min_views = 8;
  float max_direction_change = 45.0f * kGrowPi / 180.0f;
  int irls_iterations = 2;
  float max_growth_length = 100.0f;
  int max_steps = 1000;
};

struct ProjectedDirection {
  float x = 0.0f;
  float y = 0.0f;
  float dx = 0.0f;
  float dy = 0.0f;
  float normal_x = 0.0f;
  float normal_y = 0.0f;
};

// A non-owning prepared image view. orientation_variance contains width*height
// packed half2 words: low 16 bits = line-normal angle in radians, high 16 bits
// = circular variance. A variance <= 0 or non-finite value is invalid.
// foreground is either null (ungated) or width*height bytes; nonzero passes.
struct ViewData {
  GpuCamera camera{};
  int camera_index = -1;
  unsigned int camera_id = 0;
  int width = 0;
  int height = 0;
  const uint32_t* orientation_variance = nullptr;
  const uint8_t* foreground = nullptr;
};

struct ViewScore {
  bool valid = false;
  int scored_pixels = 0;
  float mean_error = 0.0f;
  float direction_x = 0.0f;
  float direction_y = 0.0f;
  std::array<float, 3> plane_normal{{0.0f, 0.0f, 0.0f}};
};

enum class StopReason : int {
  kNotRun = 0,
  kInsufficientViewDirections = 1,
  kDirectionChange = 2,
  kInsufficientForeground = 3,
  kLengthLimit = 4,
  kInvalidTangent = 5,
};

struct TipSeed {
  size_t strand_index = 0;
  bool prepend = false;
  std::array<float, 3> point{{0.0f, 0.0f, 0.0f}};
  std::array<float, 3> tangent{{0.0f, 0.0f, 0.0f}};
};

struct GrowthSample {
  std::array<float, 3> position{{0.0f, 0.0f, 0.0f}};
  // Direction points outwards from the original strand tip. MergeTipGrowth
  // flips it for samples prepended to the original point order.
  std::array<float, 3> outward_direction{{0.0f, 0.0f, 0.0f}};
};

struct TipGrowth {
  std::vector<GrowthSample> samples;
  StopReason stop_reason = StopReason::kNotRun;
};

struct DeviceRunStats {
  size_t batches = 0;
};

bool ValidateConfig(const Config& config, std::string* error = nullptr);
GrowParams MakeGrowParams(const Config& config);

// Angular distance for unoriented lines. The result is in [0, pi/2].
float LineAngleDifference(float a, float b);

// Analytically project a world point and tangent. The image direction is the
// exact perspective derivative, not a finite difference.
bool ProjectDirection(const GpuCamera& camera,
                      const std::array<float, 3>& point,
                      const std::array<float, 3>& tangent,
                      ProjectedDirection* projected);

// Construct the world-space normal of the plane through the camera center and
// the 2D image line passing through (x,y) with direction (dx,dy).
bool ImageDirectionToPlaneNormal(const GpuCamera& camera, float x, float y,
                                 float dx, float dy,
                                 std::array<float, 3>* plane_normal);

// Smallest eigenvector of sum(n*n^T), followed by the requested fixed number
// of IRLS updates using w=1/(dot(n,d)^2+1e-8). The result is sign-aligned with
// reference_direction. Fixed-order Jacobi sweeps make the result deterministic.
bool SolveDirectionIRLS(
    const std::vector<std::array<float, 3>>& plane_normals,
    const std::array<float, 3>& reference_direction, int irls_iterations,
    std::array<float, 3>* direction);

float HalfBitsToFloat(uint16_t bits);
ViewScore ScoreViewCpu(const ViewData& view,
                       const std::array<float, 3>& point,
                       const std::array<float, 3>& tangent,
                       const GrowParams& params);
int CountPointSupportCpu(const std::vector<ViewData>& views,
                         const std::array<float, 3>& point);
TipGrowth GrowTipCpu(const TipSeed& tip, const std::vector<ViewData>& views,
                     const GrowParams& params);

std::vector<TipSeed> BuildInitialTips(const std::vector<Strand>& strands);
Strand MergeTipGrowth(const Strand& original, const TipGrowth* prepend,
                      const TipGrowth* append);

// Implemented in hair_grower.cu. The output vector is indexed like `tips`;
// only [tip_begin,tip_end) is written by this invocation.
// max_batch_tips_for_test is private test instrumentation; production callers
// leave it at zero so batching remains entirely memory-sized and has no public
// configuration knob.
DeviceRunStats GrowTipsOnDevice(const std::vector<ViewData>& views,
                                const std::vector<TipSeed>& tips,
                                size_t tip_begin, size_t tip_end,
                                const GrowParams& params, int device_id,
                                std::vector<TipGrowth>* output,
                                size_t max_batch_tips_for_test = 0);

}  // namespace detail
}  // namespace hair_grower
