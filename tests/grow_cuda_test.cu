// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/camera_array.h"
#include "common/config.h"
#include "strand/hair_grower.h"
#include "strand/hair_grower_detail.h"

namespace {

class TestSuite {
 public:
  void Run(const char* name, const std::function<void()>& test) {
    current_ = name;
    const int before = failures_;
    test();
    if (failures_ == before)
      std::cout << "[PASS] " << name << '\n';
  }

  void Check(bool condition, const char* expression, const char* file,
             int line) {
    if (condition)
      return;
    ++failures_;
    std::cerr << "[FAIL] " << current_ << " (" << file << ':' << line
              << "): check failed: " << expression << '\n';
  }

  void Near(float actual, float expected, float tolerance,
            const char* expression, const char* file, int line) {
    if (std::isfinite(actual) && std::fabs(actual - expected) <= tolerance)
      return;
    ++failures_;
    std::cerr << "[FAIL] " << current_ << " (" << file << ':' << line
              << "): " << expression << " was " << actual << ", expected "
              << expected << " +/- " << tolerance << '\n';
  }

  int failures() const { return failures_; }

 private:
  const char* current_ = "setup";
  int failures_ = 0;
};

TestSuite suite;

#define CHECK(condition) suite.Check((condition), #condition, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tolerance) \
  suite.Near((actual), (expected), (tolerance), #actual, __FILE__, __LINE__)

using Vec3 = std::array<float, 3>;
using hair_grower::detail::GrowParams;
using hair_grower::detail::TipGrowth;
using hair_grower::detail::TipSeed;
using hair_grower::detail::ViewData;

uint32_t PackOrientationVariance(float orientation, float variance = 1.0f) {
  const uint16_t angle_bits =
      __half_as_ushort(__float2half_rn(orientation));
  const uint16_t variance_bits =
      __half_as_ushort(__float2half_rn(variance));
  return static_cast<uint32_t>(angle_bits) |
         (static_cast<uint32_t>(variance_bits) << 16);
}

GpuCamera MakeGpuCamera(float center_x, float center_y,
                        float principal = 32.0f) {
  GpuCamera camera{};
  camera.K[0] = 50.0f;
  camera.K[2] = principal;
  camera.K[4] = 50.0f;
  camera.K[5] = principal;
  camera.K[8] = 1.0f;
  camera.R[0] = camera.R[4] = camera.R[8] = 1.0f;
  camera.center[0] = center_x;
  camera.center[1] = center_y;
  camera.t[0] = -center_x;
  camera.t[1] = -center_y;
  return camera;
}

struct SyntheticViews {
  std::vector<std::vector<uint32_t>> maps;
  std::vector<std::vector<uint8_t>> gates;
  std::vector<ViewData> views;
};

SyntheticViews MakeRingViews(int count, bool gated,
                             float principal = 32.0f) {
  constexpr int width = 64;
  constexpr int height = 64;
  const Vec3 point{{0.0f, 0.0f, 20.0f}};
  const Vec3 tangent{{0.0f, 0.0f, 1.0f}};
  SyntheticViews result;
  result.maps.resize(count);
  result.gates.resize(count);
  result.views.resize(count);
  for (int i = 0; i < count; ++i) {
    const float angle = 2.0f * hair_grower::detail::kGrowPi * i / count;
    const GpuCamera camera =
        MakeGpuCamera(2.0f * std::cos(angle), 2.0f * std::sin(angle),
                      principal);
    hair_grower::detail::ProjectedDirection projected;
    if (!hair_grower::detail::ProjectDirection(camera, point, tangent,
                                                &projected)) {
      std::cerr << "synthetic camera projection failed\n";
      std::abort();
    }
    const float normal_angle =
        std::atan2(projected.normal_y, projected.normal_x);
    result.maps[i].assign(width * height,
                          PackOrientationVariance(normal_angle));
    if (gated)
      result.gates[i].assign(width * height, 255u);
    ViewData& view = result.views[i];
    view.camera = camera;
    view.camera_index = i;
    view.camera_id = static_cast<unsigned int>(1000 + i);
    view.width = width;
    view.height = height;
    view.orientation_variance = result.maps[i].data();
    view.foreground = gated ? result.gates[i].data() : nullptr;
  }
  return result;
}

GrowParams MakeParams(int max_steps = 4) {
  GrowParams params;
  params.step_size = 1.0f;
  params.cone_half_angle = 5.0f * hair_grower::detail::kGrowPi / 180.0f;
  params.direction_sample_step = hair_grower::detail::kGrowPi / 180.0f;
  params.direction_samples = 11;
  params.window_width = 3;
  params.window_length = 10;
  params.max_pixel_angle = 5.0f * hair_grower::detail::kGrowPi / 180.0f;
  params.min_scored_pixels = 10;
  params.min_views = 8;
  params.max_direction_change =
      45.0f * hair_grower::detail::kGrowPi / 180.0f;
  params.irls_iterations = 2;
  params.max_growth_length = static_cast<float>(max_steps);
  params.max_steps = max_steps;
  return params;
}

std::vector<TipSeed> MakeTips(size_t count) {
  std::vector<TipSeed> tips(count);
  for (size_t i = 0; i < count; ++i) {
    tips[i].strand_index = i;
    tips[i].point = {{0.0f, 0.0f, 20.0f}};
    tips[i].tangent = {{0.0f, 0.0f, 1.0f}};
  }
  return tips;
}

void CheckGrowthNear(const TipGrowth& actual, const TipGrowth& expected,
                     float tolerance) {
  CHECK(actual.stop_reason == expected.stop_reason);
  CHECK(actual.samples.size() == expected.samples.size());
  if (actual.samples.size() != expected.samples.size())
    return;
  for (size_t i = 0; i < actual.samples.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      CHECK_NEAR(actual.samples[i].position[axis],
                 expected.samples[i].position[axis], tolerance);
      CHECK_NEAR(actual.samples[i].outward_direction[axis],
                 expected.samples[i].outward_direction[axis], tolerance);
    }
  }
}

bool GrowthBitwiseEqual(const TipGrowth& a, const TipGrowth& b) {
  if (a.stop_reason != b.stop_reason || a.samples.size() != b.samples.size())
    return false;
  for (size_t i = 0; i < a.samples.size(); ++i) {
    if (std::memcmp(a.samples[i].position.data(), b.samples[i].position.data(),
                    3 * sizeof(float)) != 0 ||
        std::memcmp(a.samples[i].outward_direction.data(),
                    b.samples[i].outward_direction.data(),
                    3 * sizeof(float)) != 0) {
      return false;
    }
  }
  return true;
}

void TestCudaMatchesCpuAndIsRepeatable() {
  SyntheticViews synthetic = MakeRingViews(8, false);
  const GrowParams params = MakeParams();
  const std::vector<TipSeed> tips = MakeTips(129);
  const TipGrowth cpu =
      hair_grower::detail::GrowTipCpu(tips.front(), synthetic.views, params);
  CHECK(cpu.samples.size() == static_cast<size_t>(params.max_steps));

  std::vector<TipGrowth> whole(tips.size());
  constexpr size_t forced_batch_tips = 17;
  const auto whole_stats = hair_grower::detail::GrowTipsOnDevice(
      synthetic.views, tips, 0, tips.size(), params, 0, &whole,
      forced_batch_tips);
  CHECK(whole_stats.batches ==
        (tips.size() + forced_batch_tips - 1) / forced_batch_tips);
  for (const TipGrowth& growth : whole)
    CheckGrowthNear(growth, cpu, 2e-4f);

  std::vector<TipGrowth> repeated(tips.size());
  (void)hair_grower::detail::GrowTipsOnDevice(
      synthetic.views, tips, 0, tips.size(), params, 0, &repeated);
  for (size_t i = 0; i < tips.size(); ++i)
    CHECK(GrowthBitwiseEqual(whole[i], repeated[i]));

  // Exercise batch boundaries explicitly. This gives deterministic coverage
  // independent of how much free memory the CTest worker happens to have.
  std::vector<TipGrowth> split(tips.size());
  const size_t split_at = 53;
  const auto first_stats = hair_grower::detail::GrowTipsOnDevice(
      synthetic.views, tips, 0, split_at, params, 0, &split);
  const auto second_stats = hair_grower::detail::GrowTipsOnDevice(
      synthetic.views, tips, split_at, tips.size(), params, 0, &split);
  CHECK(first_stats.batches >= 1);
  CHECK(second_stats.batches >= 1);
  for (size_t i = 0; i < tips.size(); ++i)
    CHECK(GrowthBitwiseEqual(whole[i], split[i]));
}

void TestCudaSupportBasedGating() {
  const GrowParams params = MakeParams(2);
  const std::vector<TipSeed> tips = MakeTips(1);
  SyntheticViews synthetic = MakeRingViews(9, true);

  // One all-background view is ignored; the remaining eight still support
  // growth. No single background observation acts as a veto.
  std::fill(synthetic.gates[8].begin(), synthetic.gates[8].end(), 0u);
  std::vector<TipGrowth> output(1);
  (void)hair_grower::detail::GrowTipsOnDevice(
      synthetic.views, tips, 0, 1, params, 0, &output);
  CHECK(output[0].samples.size() == 2);
  const TipGrowth cpu_supported =
      hair_grower::detail::GrowTipCpu(tips[0], synthetic.views, params);
  CheckGrowthNear(output[0], cpu_supported, 2e-4f);

  // Dropping one more view crosses the exact 8-view threshold.
  std::fill(synthetic.gates[7].begin(), synthetic.gates[7].end(), 0u);
  output[0] = TipGrowth{};
  (void)hair_grower::detail::GrowTipsOnDevice(
      synthetic.views, tips, 0, 1, params, 0, &output);
  CHECK(output[0].samples.empty());
  CHECK(output[0].stop_reason ==
        hair_grower::detail::StopReason::kInsufficientViewDirections);
}

void TestCudaCrossingAndImageEdges() {
  const GrowParams params = MakeParams(1);
  const std::vector<TipSeed> tips = MakeTips(1);

  // In every view, only the ten center-line pixels carry the continuing hair
  // normal. The rest of the forward window carries the perpendicular crossing
  // normal. This exercises the exact min_scored_pixels boundary at a crossing.
  SyntheticViews crossing = MakeRingViews(8, false);
  const Vec3 point{{0.0f, 0.0f, 20.0f}};
  const Vec3 tangent{{0.0f, 0.0f, 1.0f}};
  for (size_t view_index = 0; view_index < crossing.views.size(); ++view_index) {
    hair_grower::detail::ProjectedDirection projected;
    CHECK(hair_grower::detail::ProjectDirection(
        crossing.views[view_index].camera, point, tangent, &projected));
    const float target = std::atan2(projected.normal_y, projected.normal_x);
    std::fill(crossing.maps[view_index].begin(),
              crossing.maps[view_index].end(),
              PackOrientationVariance(
                  target + 0.5f * hair_grower::detail::kGrowPi));
    for (int along = 1; along <= params.window_length; ++along) {
      const int x = static_cast<int>(
          std::floor(projected.x + along * projected.dx + 0.5f));
      const int y = static_cast<int>(
          std::floor(projected.y + along * projected.dy + 0.5f));
      if (x >= 0 && y >= 0 && x < crossing.views[view_index].width &&
          y < crossing.views[view_index].height) {
        crossing.maps[view_index][static_cast<size_t>(y) *
                                      crossing.views[view_index].width +
                                  x] = PackOrientationVariance(target);
      }
    }
  }
  const TipGrowth crossing_cpu =
      hair_grower::detail::GrowTipCpu(tips[0], crossing.views, params);
  CHECK(crossing_cpu.samples.size() == 1);
  std::vector<TipGrowth> crossing_cuda(1);
  (void)hair_grower::detail::GrowTipsOnDevice(
      crossing.views, tips, 0, 1, params, 0, &crossing_cuda);
  CheckGrowthNear(crossing_cuda[0], crossing_cpu, 2e-4f);

  // Moving the principal point close to the top-left corner puts several
  // forward windows against an image edge while retaining eight valid views.
  SyntheticViews edge = MakeRingViews(8, false, 6.0f);
  const TipGrowth edge_cpu =
      hair_grower::detail::GrowTipCpu(tips[0], edge.views, params);
  CHECK(edge_cpu.samples.size() == 1);
  std::vector<TipGrowth> edge_cuda(1);
  (void)hair_grower::detail::GrowTipsOnDevice(edge.views, tips, 0, 1, params, 0,
                                               &edge_cuda);
  CheckGrowthNear(edge_cuda[0], edge_cpu, 2e-4f);
}

void TestSecondGpuParityWhenAvailable(int device_count) {
  if (device_count < 2) {
    std::cout << "[SKIP] second-GPU parity (only one CUDA device)\n";
    return;
  }
  SyntheticViews synthetic = MakeRingViews(8, false);
  const GrowParams params = MakeParams(2);
  const std::vector<TipSeed> tips = MakeTips(17);
  std::vector<TipGrowth> first(tips.size()), second(tips.size());
  (void)hair_grower::detail::GrowTipsOnDevice(
      synthetic.views, tips, 0, tips.size(), params, 0, &first);
  (void)hair_grower::detail::GrowTipsOnDevice(
      synthetic.views, tips, 0, tips.size(), params, 1, &second);
  for (size_t i = 0; i < tips.size(); ++i)
    CheckGrowthNear(second[i], first[i], 2e-4f);
}

void TestEmptyAndUnsupportedPublicIdentity() {
  CameraArray no_cameras;
  Config config;
  hair_grower::GrowthStats stats;
  const std::vector<Strand> empty;
  const auto empty_result =
      hair_grower::GrowCuda(empty, no_cameras, config, 0, 1, &stats);
  CHECK(empty_result.empty());
  CHECK(stats.input_strands == 0);
  CHECK(stats.candidate_tips == 0);
  CHECK(stats.points_added == 0);

  Strand point;
  point.AddPoint(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7);
  const std::vector<Strand> unsupported{point};
  const auto identity =
      hair_grower::GrowCuda(unsupported, no_cameras, config, 0, 1, &stats);
  CHECK(identity.size() == 1);
  CHECK(identity[0].positions == point.positions);
  CHECK(identity[0].directions == point.directions);
  CHECK(identity[0].labels == point.labels);
  CHECK(stats.candidate_tips == 0);
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess || device_count == 0) {
    std::cout << "[SKIP] grow CUDA tests: no CUDA device is available\n";
    return 0;
  }

  suite.Run("CUDA matches CPU, repeats, and respects split batches",
            TestCudaMatchesCpuAndIsRepeatable);
  suite.Run("CUDA support-based foreground gating", TestCudaSupportBasedGating);
  suite.Run("CUDA crossings and image edges", TestCudaCrossingAndImageEdges);
  suite.Run("second-GPU parity when available",
            [device_count]() { TestSecondGpuParityWhenAvailable(device_count); });
  suite.Run("empty and unsupported public identity",
            TestEmptyAndUnsupportedPublicIdentity);

  if (suite.failures() != 0)
    std::cerr << suite.failures() << " grow CUDA test assertion(s) failed\n";
  return suite.failures() == 0 ? 0 : 1;
}
