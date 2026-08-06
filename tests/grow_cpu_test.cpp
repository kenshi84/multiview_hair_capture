// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "common/camera.h"
#include "common/config.h"
#include "strand/hair_grower_detail.h"
#include "strand/orientation_cache.h"

namespace {

class TestSuite {
 public:
  void Run(const char* name, const std::function<void()>& test) {
    current_ = name;
    try {
      test();
      if (failed_in_current_ == 0)
        std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception& e) {
      Fail(__FILE__, __LINE__, std::string("unexpected exception: ") + e.what());
    } catch (...) {
      Fail(__FILE__, __LINE__, "unexpected non-standard exception");
    }
    failed_in_current_ = 0;
  }

  void Check(bool condition, const char* expression, const char* file,
             int line) {
    if (!condition)
      Fail(file, line, std::string("check failed: ") + expression);
  }

  void Near(float actual, float expected, float tolerance,
            const char* expression, const char* file, int line) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
      Fail(file, line, std::string(expression) + " was " +
                           std::to_string(actual) + ", expected " +
                           std::to_string(expected) + " +/- " +
                           std::to_string(tolerance));
    }
  }

  int failures() const { return failures_; }

 private:
  void Fail(const char* file, int line, const std::string& message) {
    ++failures_;
    ++failed_in_current_;
    std::cerr << "[FAIL] " << current_ << " (" << file << ':' << line
              << "): " << message << '\n';
  }

  const char* current_ = "setup";
  int failures_ = 0;
  int failed_in_current_ = 0;
};

TestSuite suite;

#define CHECK(condition) suite.Check((condition), #condition, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tolerance) \
  suite.Near((actual), (expected), (tolerance), #actual, __FILE__, __LINE__)

class TempDirectory {
 public:
  TempDirectory() {
    const auto seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    path_ = std::filesystem::temp_directory_path() /
            ("hair-grow-test-" + std::to_string(seed));
    std::filesystem::create_directories(path_);
  }

  ~TempDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void WriteText(const std::filesystem::path& path, const std::string& text,
               bool append = false) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, append ? (std::ios::out | std::ios::app)
                                 : std::ios::out);
  if (!out)
    throw std::runtime_error("cannot write " + path.string());
  out << text;
}

std::string GrowToml(const std::string& fields) {
  return "[grow]\n" + fields + "\n";
}

void ExpectInvalidGrowConfig(const std::string& fields,
                             const std::string& expected_message) {
  TempDirectory temp;
  const auto path = temp.path() / "invalid.toml";
  WriteText(path, GrowToml(fields));
  try {
    (void)Config::LoadFromToml(path.string());
    CHECK(false);
  } catch (const std::runtime_error& e) {
    CHECK(std::string(e.what()).find(expected_message) != std::string::npos);
  }
}

Camera MakeIdentityCamera(unsigned int id) {
  const float intrinsic[9] = {100.0f, 0.0f, 32.0f, 0.0f, 100.0f,
                              24.0f,  0.0f, 0.0f, 1.0f};
  const float extrinsic[12] = {1.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 1.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 1.0f, 0.0f};
  const float distortion[5] = {};
  return Camera(intrinsic, extrinsic, distortion, id);
}

GpuCamera MakeGpuCamera(float center_x = 0.0f, float center_y = 0.0f,
                        float center_z = 0.0f) {
  GpuCamera camera{};
  camera.K[0] = 50.0f;
  camera.K[2] = 32.0f;
  camera.K[4] = 50.0f;
  camera.K[5] = 32.0f;
  camera.K[8] = 1.0f;
  camera.R[0] = camera.R[4] = camera.R[8] = 1.0f;
  camera.center[0] = center_x;
  camera.center[1] = center_y;
  camera.center[2] = center_z;
  camera.t[0] = -center_x;
  camera.t[1] = -center_y;
  camera.t[2] = -center_z;
  return camera;
}

// The tests write host-side synthetic half2 maps without depending on CUDA's
// host half implementation. This converter is round-to-nearest-even and is
// intentionally limited to the finite binary32 inputs used by the tests.
uint16_t FloatToHalfBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t exponent_bits = (bits >> 23) & 0xffu;
  uint32_t mantissa = bits & 0x7fffffu;
  int exponent = static_cast<int>(exponent_bits) - 127 + 15;
  if (exponent <= 0) {
    if (exponent < -10)
      return static_cast<uint16_t>(sign);
    mantissa |= 0x800000u;
    const int shift = 14 - exponent;
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u)))
      ++rounded;
    return static_cast<uint16_t>(sign | rounded);
  }
  if (exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7c00u);
  uint32_t rounded = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
    ++rounded;
    if (rounded == 0x400u) {
      rounded = 0;
      ++exponent;
      if (exponent >= 31)
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<uint16_t>(sign |
                               (static_cast<uint32_t>(exponent) << 10) |
                               rounded);
}

uint32_t PackOrientationVariance(float orientation, float variance = 1.0f) {
  return static_cast<uint32_t>(FloatToHalfBits(orientation)) |
         (static_cast<uint32_t>(FloatToHalfBits(variance)) << 16);
}

using Vec3 = std::array<float, 3>;
using hair_grower::detail::GrowParams;
using hair_grower::detail::ProjectedDirection;
using hair_grower::detail::ViewData;

struct SyntheticViews {
  std::vector<std::vector<uint32_t>> maps;
  std::vector<std::vector<uint8_t>> foreground;
  std::vector<ViewData> views;
};

SyntheticViews MakeRingViews(int count = 8, bool gated = false) {
  constexpr int width = 64;
  constexpr int height = 64;
  const Vec3 point{{0.0f, 0.0f, 20.0f}};
  const Vec3 tangent{{0.0f, 0.0f, 1.0f}};
  SyntheticViews result;
  result.maps.resize(count);
  result.foreground.resize(count);
  result.views.resize(count);
  for (int i = 0; i < count; ++i) {
    const float angle = 2.0f * hair_grower::detail::kGrowPi * i / count;
    GpuCamera camera = MakeGpuCamera(2.0f * std::cos(angle),
                                     2.0f * std::sin(angle), 0.0f);
    ProjectedDirection projected;
    if (!hair_grower::detail::ProjectDirection(camera, point, tangent,
                                                &projected)) {
      throw std::runtime_error("synthetic ring camera does not project");
    }
    const float normal_angle =
        std::atan2(projected.normal_y, projected.normal_x);
    result.maps[i].assign(width * height,
                          PackOrientationVariance(normal_angle));
    if (gated)
      result.foreground[i].assign(width * height, 255u);
    result.views[i].camera = camera;
    result.views[i].camera_index = i;
    result.views[i].camera_id = static_cast<unsigned int>(100 + i);
    result.views[i].width = width;
    result.views[i].height = height;
    result.views[i].orientation_variance = result.maps[i].data();
    result.views[i].foreground =
        gated ? result.foreground[i].data() : nullptr;
  }
  return result;
}

GrowParams MakeSyntheticParams(int max_steps = 3) {
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

void TestGrowConfigDefaults() {
  const Config cfg;
  CHECK_NEAR(cfg.grow_step_size, 0.1f, 1e-7f);
  CHECK_NEAR(cfg.grow_cone_half_angle, 5.0f, 1e-7f);
  CHECK_NEAR(cfg.grow_direction_sample_step, 1.0f, 1e-7f);
  CHECK(cfg.grow_window_width == 3);
  CHECK(cfg.grow_window_length == 10);
  CHECK_NEAR(cfg.grow_max_pixel_angle, 5.0f, 1e-7f);
  CHECK(cfg.grow_min_scored_pixels == 10);
  CHECK(cfg.grow_min_views == 8);
  CHECK_NEAR(cfg.grow_max_direction_change, 45.0f, 1e-7f);
  CHECK(cfg.grow_irls_iterations == 2);
  CHECK_NEAR(cfg.grow_max_growth_length, 100.0f, 1e-7f);
  CHECK(hair_grower::detail::MakeGrowParams(cfg).max_steps == 1000);
  CHECK(!cfg.grow_use_mask);
  CHECK_NEAR(cfg.grow_min_intensity, 0.0f, 1e-7f);
}

void TestGrowConfigOverridesAndInclusiveBoundaries() {
  TempDirectory temp;
  const auto path = temp.path() / "overrides.toml";
  WriteText(path, GrowToml(
                      "step_size = 0.25\n"
                      "cone_half_angle = 0.0\n"
                      "direction_sample_step = 0.5\n"
                      "window_width = 5\n"
                      "window_length = 7\n"
                      "max_pixel_angle = 90.0\n"
                      "min_scored_pixels = 35\n"
                      "min_views = 2\n"
                      "max_direction_change = 180.0\n"
                      "irls_iterations = 0\n"
                      "max_growth_length = 0.25\n"
                      "use_mask = true\n"
                      "min_intensity = 1.0"));
  const Config cfg = Config::LoadFromToml(path.string());
  CHECK_NEAR(cfg.grow_step_size, 0.25f, 1e-7f);
  CHECK_NEAR(cfg.grow_cone_half_angle, 0.0f, 1e-7f);
  CHECK_NEAR(cfg.grow_direction_sample_step, 0.5f, 1e-7f);
  CHECK(cfg.grow_window_width == 5);
  CHECK(cfg.grow_window_length == 7);
  CHECK_NEAR(cfg.grow_max_pixel_angle, 90.0f, 1e-7f);
  CHECK(cfg.grow_min_scored_pixels == 35);
  CHECK(cfg.grow_min_views == 2);
  CHECK_NEAR(cfg.grow_max_direction_change, 180.0f, 1e-7f);
  CHECK(cfg.grow_irls_iterations == 0);
  CHECK_NEAR(cfg.grow_max_growth_length, 0.25f, 1e-7f);
  CHECK(cfg.grow_use_mask);
  CHECK_NEAR(cfg.grow_min_intensity, 1.0f, 1e-7f);
}

void TestGrowConfigRejectsInvalidValues() {
  ExpectInvalidGrowConfig("step_size = 0.0", "grow.step_size");
  ExpectInvalidGrowConfig("cone_half_angle = -0.01", "grow.cone_half_angle");
  ExpectInvalidGrowConfig("cone_half_angle = 90.0", "grow.cone_half_angle");
  ExpectInvalidGrowConfig("direction_sample_step = 0.0",
                          "grow.direction_sample_step");
  ExpectInvalidGrowConfig(
      "cone_half_angle = 4.0\ndirection_sample_step = 5.0",
      "grow.direction_sample_step");
  ExpectInvalidGrowConfig(
      "cone_half_angle = 5.0\ndirection_sample_step = 1e-20",
      "too large");
  ExpectInvalidGrowConfig("window_width = 0", "grow.window_width");
  ExpectInvalidGrowConfig("window_width = 4", "grow.window_width");
  ExpectInvalidGrowConfig("window_length = 0", "grow.window_length");
  ExpectInvalidGrowConfig("max_pixel_angle = -0.01", "grow.max_pixel_angle");
  ExpectInvalidGrowConfig("max_pixel_angle = 90.01", "grow.max_pixel_angle");
  ExpectInvalidGrowConfig("min_scored_pixels = 0",
                          "grow.min_scored_pixels");
  ExpectInvalidGrowConfig("min_scored_pixels = 31",
                          "grow.min_scored_pixels");
  ExpectInvalidGrowConfig("min_views = 1", "grow.min_views");
  ExpectInvalidGrowConfig("max_direction_change = 0.0",
                          "grow.max_direction_change");
  ExpectInvalidGrowConfig("max_direction_change = 180.01",
                          "grow.max_direction_change");
  ExpectInvalidGrowConfig("irls_iterations = -1", "grow.irls_iterations");
  ExpectInvalidGrowConfig("max_growth_length = 0.0",
                          "grow.max_growth_length");
  ExpectInvalidGrowConfig("step_size = 1e-20", "too large");
  ExpectInvalidGrowConfig("min_intensity = -0.01", "grow.min_intensity");
  ExpectInvalidGrowConfig("min_intensity = 1.01", "grow.min_intensity");
}

void TestLineAnglesAndHalfDecoding() {
  using hair_grower::detail::HalfBitsToFloat;
  using hair_grower::detail::LineAngleDifference;
  constexpr float pi = hair_grower::detail::kGrowPi;
  CHECK_NEAR(LineAngleDifference(0.0f, pi), 0.0f, 1e-6f);
  CHECK_NEAR(LineAngleDifference(-0.1f, pi - 0.1f), 0.0f, 1e-6f);
  CHECK_NEAR(LineAngleDifference(0.0f, 0.5f * pi), 0.5f * pi, 1e-6f);
  CHECK_NEAR(LineAngleDifference(0.0f, pi - 0.2f), 0.2f, 1e-6f);
  CHECK(std::isinf(LineAngleDifference(
      std::numeric_limits<float>::quiet_NaN(), 0.0f)));

  CHECK_NEAR(HalfBitsToFloat(0x3c00u), 1.0f, 0.0f);
  CHECK_NEAR(HalfBitsToFloat(0xbc00u), -1.0f, 0.0f);
  // Smallest positive binary16 subnormal. This catches the easy-to-miss
  // off-by-one when rebiasing a normalized subnormal mantissa.
  CHECK_NEAR(HalfBitsToFloat(0x0001u), std::ldexp(1.0f, -24), 0.0f);

  Config non_divisor;
  non_divisor.grow_cone_half_angle = 5.0f;
  non_divisor.grow_direction_sample_step = 3.0f;
  const GrowParams symmetric =
      hair_grower::detail::MakeGrowParams(non_divisor);
  CHECK(symmetric.direction_samples == 3);  // -3, 0, +3 degrees.
}

void TestProjectionAndPlaneConvention() {
  const GpuCamera camera = MakeGpuCamera();
  const Vec3 point{{1.0f, 2.0f, 10.0f}};
  const Vec3 tangent{{1.0f, 0.0f, 1.0f}};
  ProjectedDirection projected;
  CHECK(hair_grower::detail::ProjectDirection(camera, point, tangent,
                                               &projected));
  CHECK_NEAR(projected.x, 37.0f, 1e-5f);
  CHECK_NEAR(projected.y, 42.0f, 1e-5f);
  const float length = std::sqrt(4.5f * 4.5f + 1.0f);
  CHECK_NEAR(projected.dx, 4.5f / length, 1e-5f);
  CHECK_NEAR(projected.dy, -1.0f / length, 1e-5f);
  CHECK_NEAR(projected.normal_x, -projected.dy, 1e-7f);
  CHECK_NEAR(projected.normal_y, projected.dx, 1e-7f);

  ProjectedDirection invalid;
  CHECK(!hair_grower::detail::ProjectDirection(
      camera, {{0.0f, 0.0f, -1.0f}}, {{1.0f, 0.0f, 0.0f}}, &invalid));
  CHECK(!hair_grower::detail::ProjectDirection(
      camera, {{0.0f, 0.0f, 10.0f}}, {{0.0f, 0.0f, 1.0f}}, &invalid));

  Vec3 plane_normal;
  CHECK(hair_grower::detail::ImageDirectionToPlaneNormal(
      camera, 32.0f, 32.0f, 0.0f, 1.0f, &plane_normal));
  CHECK_NEAR(std::fabs(plane_normal[0]), 1.0f, 1e-6f);
  CHECK_NEAR(plane_normal[1], 0.0f, 1e-6f);
  CHECK_NEAR(plane_normal[2], 0.0f, 1e-6f);
  CHECK(hair_grower::detail::ImageDirectionToPlaneNormal(
      camera, 32.0f, 32.0f, 1.0f, 0.0f, &plane_normal));
  CHECK_NEAR(plane_normal[0], 0.0f, 1e-6f);
  CHECK_NEAR(std::fabs(plane_normal[1]), 1.0f, 1e-6f);
  CHECK_NEAR(plane_normal[2], 0.0f, 1e-6f);
}

void TestPlaneSolveIrlsAndDegeneracy() {
  std::vector<Vec3> rank_one(8, {{1.0f, 0.0f, 0.0f}});
  Vec3 direction;
  CHECK(!hair_grower::detail::SolveDirectionIRLS(
      rank_one, {{0.0f, 0.0f, 1.0f}}, 2, &direction));

  std::vector<Vec3> normals;
  for (int i = 0; i < 8; ++i) {
    const float angle = hair_grower::detail::kGrowPi * i / 8.0f;
    normals.push_back({{std::cos(angle), std::sin(angle), 0.0f}});
  }
  CHECK(hair_grower::detail::SolveDirectionIRLS(
      normals, {{0.0f, 0.0f, 1.0f}}, 2, &direction));
  CHECK_NEAR(direction[0], 0.0f, 1e-5f);
  CHECK_NEAR(direction[1], 0.0f, 1e-5f);
  CHECK(direction[2] > 0.9999f);

  // A bad plane should be strongly down-weighted by the stabilized IRLS
  // refinements while the consistent eight-plane solution remains +Z.
  normals.push_back({{0.0f, 0.6f, 0.8f}});
  CHECK(hair_grower::detail::SolveDirectionIRLS(
      normals, {{0.0f, 0.0f, 1.0f}}, 2, &direction));
  CHECK(direction[2] > 0.995f);
  CHECK(hair_grower::detail::SolveDirectionIRLS(
      normals, {{0.0f, 0.0f, -1.0f}}, 2, &direction));
  CHECK(direction[2] < -0.995f);
}

void SetDirectionalSamples(std::vector<uint32_t>* map, int width, int height,
                           float x, float y, float dx, float dy, int length,
                           uint32_t packed) {
  for (int along = 1; along <= length; ++along) {
    const int ix = static_cast<int>(std::floor(x + along * dx + 0.5f));
    const int iy = static_cast<int>(std::floor(y + along * dy + 0.5f));
    if (ix >= 0 && iy >= 0 && ix < width && iy < height)
      (*map)[static_cast<size_t>(iy) * width + ix] = packed;
  }
}

void TestGaborNormalScoringAndCandidateChoice() {
  constexpr int width = 64;
  constexpr int height = 64;
  const GpuCamera camera = MakeGpuCamera();
  const Vec3 point{{0.0f, 0.0f, 10.0f}};
  const Vec3 tangent{{0.0f, 1.0f, 0.0f}};
  std::vector<uint32_t> map(width * height,
                            PackOrientationVariance(0.0f));
  ViewData view;
  view.camera = camera;
  view.width = width;
  view.height = height;
  view.orientation_variance = map.data();

  GrowParams params;
  params.cone_half_angle = 0.0f;
  params.direction_sample_step = 1.0f;
  params.direction_samples = 1;
  params.window_width = 3;
  params.window_length = 10;
  params.max_pixel_angle = 0.5f * hair_grower::detail::kGrowPi / 180.0f;
  params.min_scored_pixels = 10;
  const auto score =
      hair_grower::detail::ScoreViewCpu(view, point, tangent, params);
  CHECK(score.valid);
  CHECK(score.scored_pixels == 30);
  CHECK_NEAR(score.direction_x, 0.0f, 1e-6f);
  CHECK_NEAR(score.direction_y, 1.0f, 1e-6f);

  std::fill(map.begin(), map.end(),
            PackOrientationVariance(0.5f * hair_grower::detail::kGrowPi));
  CHECK(!hair_grower::detail::ScoreViewCpu(view, point, tangent, params).valid);

  // Both candidates clear min_scored_pixels. The +45-degree candidate has
  // fewer samples, but they agree much better with the projected segment
  // normal; score selection must minimize mean error rather than maximize
  // the count once the support threshold is satisfied.
  std::fill(map.begin(), map.end(), 0u);
  params.cone_half_angle = 45.0f * hair_grower::detail::kGrowPi / 180.0f;
  params.direction_sample_step =
      45.0f * hair_grower::detail::kGrowPi / 180.0f;
  params.direction_samples = 3;
  params.window_width = 1;
  params.window_length = 10;
  params.max_pixel_angle = 5.0f * hair_grower::detail::kGrowPi / 180.0f;
  params.min_scored_pixels = 5;
  const float diagonal = std::sqrt(0.5f);
  const float target_normal = hair_grower::detail::kGrowPi;
  SetDirectionalSamples(
      &map, width, height, 32.0f, 32.0f, diagonal, diagonal, 10,
      PackOrientationVariance(
          target_normal + 4.0f * hair_grower::detail::kGrowPi / 180.0f));
  SetDirectionalSamples(
      &map, width, height, 32.0f, 32.0f, -diagonal, diagonal, 6,
      PackOrientationVariance(
          target_normal + 0.25f * hair_grower::detail::kGrowPi / 180.0f));
  const auto preferred =
      hair_grower::detail::ScoreViewCpu(view, point, tangent, params);
  CHECK(preferred.valid);
  CHECK(preferred.direction_x < -0.5f);
  CHECK(preferred.mean_error <
        1.0f * hair_grower::detail::kGrowPi / 180.0f);
}

void TestSupportThresholdAndCpuTipGrowth() {
  SyntheticViews synthetic = MakeRingViews(8, true);
  const Vec3 point{{0.0f, 0.0f, 20.0f}};
  CHECK(hair_grower::detail::CountPointSupportCpu(synthetic.views, point) == 8);

  ProjectedDirection projected;
  CHECK(hair_grower::detail::ProjectDirection(
      synthetic.views.back().camera, point, {{0.0f, 0.0f, 1.0f}},
      &projected));
  const int ix = static_cast<int>(std::floor(projected.x + 0.5f));
  const int iy = static_cast<int>(std::floor(projected.y + 0.5f));
  synthetic.foreground.back()[static_cast<size_t>(iy) * 64 + ix] = 0;
  CHECK(hair_grower::detail::CountPointSupportCpu(synthetic.views, point) == 7);
  synthetic.foreground.back()[static_cast<size_t>(iy) * 64 + ix] = 255;

  hair_grower::detail::TipSeed tip;
  tip.point = point;
  tip.tangent = {{0.0f, 0.0f, 1.0f}};
  const GrowParams params = MakeSyntheticParams(3);
  const auto growth =
      hair_grower::detail::GrowTipCpu(tip, synthetic.views, params);
  CHECK(growth.stop_reason == hair_grower::detail::StopReason::kLengthLimit);
  CHECK(growth.samples.size() == 3);
  for (size_t i = 0; i < growth.samples.size(); ++i) {
    CHECK_NEAR(growth.samples[i].position[0], 0.0f, 1e-4f);
    CHECK_NEAR(growth.samples[i].position[1], 0.0f, 1e-4f);
    CHECK_NEAR(growth.samples[i].position[2], 21.0f + i, 1e-4f);
    CHECK(growth.samples[i].outward_direction[2] > 0.999f);
  }

  synthetic.views.pop_back();
  const auto unsupported =
      hair_grower::detail::GrowTipCpu(tip, synthetic.views, params);
  CHECK(unsupported.samples.empty());
  CHECK(unsupported.stop_reason ==
        hair_grower::detail::StopReason::kInsufficientViewDirections);
}

void TestTipSeedsMergeSignsAndOrdering() {
  Strand duplicated;
  duplicated.AddPoint(0.0f, 0.0f, 0.0f, 9.0f, 8.0f, 7.0f, 10);
  duplicated.AddPoint(0.0f, 0.0f, 0.0f, 6.0f, 5.0f, 4.0f, 11);
  duplicated.AddPoint(2.0f, 0.0f, 0.0f, 3.0f, 2.0f, 1.0f, 12);
  Strand unsupported;
  unsupported.AddPoint(4.0f, 5.0f, 6.0f, 0.0f, 1.0f, 0.0f, 20);
  const std::vector<Strand> strands{duplicated, unsupported};
  const auto tips = hair_grower::detail::BuildInitialTips(strands);
  CHECK(tips.size() == 2);
  CHECK(tips[0].strand_index == 0 && tips[0].prepend);
  CHECK_NEAR(tips[0].point[0], 0.0f, 0.0f);
  CHECK(tips[0].tangent[0] < -0.999f);
  CHECK(tips[1].strand_index == 0 && !tips[1].prepend);
  CHECK_NEAR(tips[1].point[0], 2.0f, 0.0f);
  CHECK(tips[1].tangent[0] > 0.999f);

  Strand original;
  original.AddPoint(0.0f, 0.0f, 0.0f, 2.0f, 3.0f, 4.0f, 30);
  original.AddPoint(1.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 31);
  original.AddPoint(2.0f, 0.0f, 0.0f, 8.0f, 9.0f, 10.0f, 32);
  hair_grower::detail::TipGrowth prepend;
  prepend.samples.push_back(
      {{{-1.0f, 0.0f, 0.0f}}, {{-1.0f, 0.0f, 0.0f}}});
  prepend.samples.push_back(
      {{{-2.0f, 0.0f, 0.0f}}, {{-1.0f, 0.0f, 0.0f}}});
  hair_grower::detail::TipGrowth append;
  append.samples.push_back(
      {{{3.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}}});
  const Strand merged =
      hair_grower::detail::MergeTipGrowth(original, &prepend, &append);
  CHECK(merged.NumPoints() == 6);
  const float expected_x[6] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
  for (size_t i = 0; i < 6; ++i)
    CHECK_NEAR(merged.positions[i * 3], expected_x[i], 0.0f);
  CHECK(merged.directions[0] > 0.999f);
  CHECK(merged.directions[3] > 0.999f);
  CHECK(merged.directions[15] > 0.999f);
  CHECK(merged.labels[0] == 30 && merged.labels[1] == 30);
  CHECK(merged.labels[5] == 32);
  for (size_t i = 0; i < original.NumPoints(); ++i) {
    for (size_t component = 0; component < 3; ++component) {
      CHECK_NEAR(merged.positions[(i + 2) * 3 + component],
                 original.positions[i * 3 + component], 0.0f);
      CHECK_NEAR(merged.directions[(i + 2) * 3 + component],
                 original.directions[i * 3 + component], 0.0f);
    }
    CHECK(merged.labels[i + 2] == original.labels[i]);
  }
}

void TestOrientationCacheRoundTripAndInvalidation() {
  TempDirectory temp;
  const unsigned int camera_id = 42;
  const auto image_path = temp.path() / "image_42.dat";
  WriteText(image_path, "source-image-v1");
  const auto original_write_time = std::filesystem::last_write_time(image_path);

  Config cfg;
  cfg.image_dir = (temp.path() / "image_%u.dat").string();
  cfg.output_dir = (temp.path() / "output").string();
  const Camera camera = MakeIdentityCamera(camera_id);

  uint64_t fingerprint = 0;
  CHECK(orientation_cache::ComputeFingerprint(cfg, camera, 0, &fingerprint));
  CHECK(fingerprint != 0);

  orientation_cache::PreparedView input;
  input.camera_index = 0;
  input.camera_id = camera_id;
  input.width = 3;
  input.height = 2;
  input.camera = camera.BuildGpuCamera();
  input.orientation_variance = {0x3c003800u, 0x40003a00u, 0x42003b00u,
                                0x44003400u, 0x46003600u, 0x48003700u};
  CHECK(orientation_cache::StoreView(cfg, camera, input));
  CHECK(std::filesystem::is_regular_file(
      orientation_cache::CachePath(cfg, camera_id)));

  orientation_cache::PreparedView loaded;
  CHECK(orientation_cache::LoadView(cfg, camera, 0, &loaded));
  CHECK(loaded.camera_index == input.camera_index);
  CHECK(loaded.camera_id == input.camera_id);
  CHECK(loaded.width == input.width);
  CHECK(loaded.height == input.height);
  CHECK(loaded.orientation_variance == input.orientation_variance);
  for (int i = 0; i < 9; ++i)
    CHECK_NEAR(loaded.camera.K[i], input.camera.K[i], 0.0f);

  Config changed_gabor = cfg;
  changed_gabor.gabor_sigma += 0.25f;
  orientation_cache::PreparedView rejected;
  CHECK(!orientation_cache::LoadView(changed_gabor, camera, 0, &rejected));

  // Same-size replacement with the timestamp restored must still invalidate
  // the cache: the source fingerprint includes bytes, not metadata alone.
  WriteText(image_path, "source-image-v2");
  std::filesystem::last_write_time(image_path, original_write_time);
  uint64_t same_metadata_fingerprint = 0;
  CHECK(orientation_cache::ComputeFingerprint(
      cfg, camera, 0, &same_metadata_fingerprint));
  CHECK(same_metadata_fingerprint != fingerprint);
  CHECK(!orientation_cache::LoadView(cfg, camera, 0, &rejected));

  CHECK(orientation_cache::StoreView(cfg, camera, input));
  WriteText(image_path, "-changed", true);
  uint64_t changed_fingerprint = 0;
  CHECK(orientation_cache::ComputeFingerprint(cfg, camera, 0,
                                              &changed_fingerprint));
  CHECK(changed_fingerprint != fingerprint);
  CHECK(!orientation_cache::LoadView(cfg, camera, 0, &rejected));

  CHECK(orientation_cache::StoreView(cfg, camera, input));
  const auto cache_path = orientation_cache::CachePath(cfg, camera_id);
  const auto original_size = std::filesystem::file_size(cache_path);
  CHECK(original_size > 1);
  std::filesystem::resize_file(cache_path, original_size - 1);
  CHECK(!orientation_cache::LoadView(cfg, camera, 0, &rejected));
}

}  // namespace

int main() {
  suite.Run("grow config defaults", TestGrowConfigDefaults);
  suite.Run("grow config overrides and inclusive boundaries",
            TestGrowConfigOverridesAndInclusiveBoundaries);
  suite.Run("grow config rejects invalid values",
            TestGrowConfigRejectsInvalidValues);
  suite.Run("line angles and half decoding", TestLineAnglesAndHalfDecoding);
  suite.Run("projection and plane convention", TestProjectionAndPlaneConvention);
  suite.Run("plane solve IRLS and degeneracy",
            TestPlaneSolveIrlsAndDegeneracy);
  suite.Run("Gabor normal scoring and candidate choice",
            TestGaborNormalScoringAndCandidateChoice);
  suite.Run("support threshold and CPU tip growth",
            TestSupportThresholdAndCpuTipGrowth);
  suite.Run("tip seeds, merge signs, and ordering",
            TestTipSeedsMergeSignsAndOrdering);
  suite.Run("orientation cache round trip and invalidation",
            TestOrientationCacheRoundTripAndInvalidation);

  if (suite.failures() != 0)
    std::cerr << suite.failures() << " grow CPU test assertion(s) failed\n";
  return suite.failures() == 0 ? 0 : 1;
}
