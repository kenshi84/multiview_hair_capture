// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "strand/hair_grower.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>
#include <omp.h>

#include "common/logger.h"
#include "common/timer.h"
#include "strand/hair_grower_detail.h"
#include "strand/orientation_cache.h"

namespace hair_grower {
namespace detail {
namespace {

using Vec3 = std::array<float, 3>;

float Dot(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

float Norm(const Vec3& v) {
  return std::sqrt(Dot(v, v));
}

bool Normalize(Vec3* v) {
  const float length = Norm(*v);
  if (!std::isfinite(length) || length <= 1e-10f)
    return false;
  for (float& x : *v)
    x /= length;
  return true;
}

bool IsFinite(const Vec3& v) {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

// Apply one fixed-order Jacobi rotation to a symmetric 3x3 matrix.  Matrix V
// stores eigenvectors in columns.
void JacobiRotate(float a[3][3], float v[3][3], int p, int q) {
  const float apq = a[p][q];
  if (std::fabs(apq) <= 1e-20f)
    return;

  const float tau = (a[q][q] - a[p][p]) / (2.0f * apq);
  const float t = std::copysign(1.0f, tau) /
                  (std::fabs(tau) + std::sqrt(1.0f + tau * tau));
  const float c = 1.0f / std::sqrt(1.0f + t * t);
  const float s = t * c;
  const float app = a[p][p];
  const float aqq = a[q][q];

  a[p][p] = app - t * apq;
  a[q][q] = aqq + t * apq;
  a[p][q] = a[q][p] = 0.0f;

  for (int r = 0; r < 3; ++r) {
    if (r == p || r == q)
      continue;
    const float arp = a[r][p];
    const float arq = a[r][q];
    a[r][p] = a[p][r] = c * arp - s * arq;
    a[r][q] = a[q][r] = s * arp + c * arq;
  }
  for (int r = 0; r < 3; ++r) {
    const float vrp = v[r][p];
    const float vrq = v[r][q];
    v[r][p] = c * vrp - s * vrq;
    v[r][q] = s * vrp + c * vrq;
  }
}

bool SmallestEigenvector(const float matrix[6], Vec3* result) {
  const float trace = matrix[0] + matrix[3] + matrix[5];
  if (!std::isfinite(trace) || trace <= 0.0f)
    return false;
  float a[3][3] = {{matrix[0], matrix[1], matrix[2]},
                   {matrix[1], matrix[3], matrix[4]},
                   {matrix[2], matrix[4], matrix[5]}};
  float v[3][3] = {{1.0f, 0.0f, 0.0f},
                   {0.0f, 1.0f, 0.0f},
                   {0.0f, 0.0f, 1.0f}};
  // A fixed sweep count and pair order are intentional: CPU/CUDA results are
  // repeatable and do not depend on a convergence race near equal eigenvalues.
  for (int sweep = 0; sweep < 12; ++sweep) {
    JacobiRotate(a, v, 0, 1);
    JacobiRotate(a, v, 0, 2);
    JacobiRotate(a, v, 1, 2);
  }
  int minimum = 0;
  if (a[1][1] < a[minimum][minimum])
    minimum = 1;
  if (a[2][2] < a[minimum][minimum])
    minimum = 2;
  int second = minimum == 0 ? 1 : 0;
  for (int index = 0; index < 3; ++index) {
    if (index != minimum && a[index][index] < a[second][second])
      second = index;
  }
  if (!std::isfinite(a[minimum][minimum]) || !std::isfinite(a[second][second]) ||
      a[second][second] - a[minimum][minimum] <= 1e-6f * trace)
    return false;
  *result = {{v[0][minimum], v[1][minimum], v[2][minimum]}};
  return IsFinite(*result) && Normalize(result);
}

void AccumulateNormalMatrix(const std::vector<Vec3>& normals, const Vec3* estimate,
                            float matrix[6]) {
  std::fill(matrix, matrix + 6, 0.0f);
  for (const Vec3& n : normals) {
    float weight = 1.0f;
    if (estimate) {
      const float residual = Dot(n, *estimate);
      // Multiplying all specified IRLS weights by 1e-8 leaves the eigenvector
      // unchanged and keeps accumulation well conditioned in single precision.
      weight = 1e-8f / (residual * residual + 1e-8f);
    }
    matrix[0] += weight * n[0] * n[0];
    matrix[1] += weight * n[0] * n[1];
    matrix[2] += weight * n[0] * n[2];
    matrix[3] += weight * n[1] * n[1];
    matrix[4] += weight * n[1] * n[2];
    matrix[5] += weight * n[2] * n[2];
  }
}

bool ProjectPoint(const GpuCamera& camera, const Vec3& point, float* x, float* y) {
  float q[3];
  for (int r = 0; r < 3; ++r) {
    q[r] = camera.R[r * 3] * point[0] + camera.R[r * 3 + 1] * point[1] +
           camera.R[r * 3 + 2] * point[2] + camera.t[r];
  }
  if (!std::isfinite(q[2]) || q[2] <= 1e-6f)
    return false;
  const float nu = camera.K[0] * q[0] + camera.K[1] * q[1] + camera.K[2] * q[2];
  const float nv = camera.K[3] * q[0] + camera.K[4] * q[1] + camera.K[5] * q[2];
  const float den = camera.K[6] * q[0] + camera.K[7] * q[1] + camera.K[8] * q[2];
  if (!std::isfinite(den) || std::fabs(den) <= 1e-10f)
    return false;
  *x = nu / den;
  *y = nv / den;
  return std::isfinite(*x) && std::isfinite(*y);
}

bool PixelPasses(const ViewData& view, float x, float y) {
  const int ix = static_cast<int>(std::floor(x + 0.5f));
  const int iy = static_cast<int>(std::floor(y + 0.5f));
  if (ix < 0 || iy < 0 || ix >= view.width || iy >= view.height)
    return false;
  return view.foreground == nullptr || view.foreground[iy * view.width + ix] != 0;
}

double StableGrowthStepCount(float length, float step_size) {
  const double ratio = static_cast<double>(length) / step_size;
  const double nearest = std::round(ratio);
  const double tolerance = 32.0 * std::numeric_limits<float>::epsilon() *
                           std::fmax(1.0, std::fabs(ratio));
  return std::fabs(ratio - nearest) <= tolerance ? nearest
                                                 : std::floor(ratio);
}

}  // namespace

bool ValidateConfig(const Config& c, std::string* error) {
  auto fail = [&](const char* message) {
    if (error)
      *error = message;
    return false;
  };
  auto positive = [](float x) { return std::isfinite(x) && x > 0.0f; };
  if (!positive(c.grow_step_size))
    return fail("grow.step_size must be finite and positive");
  if (!std::isfinite(c.grow_cone_half_angle) || c.grow_cone_half_angle < 0.0f ||
      c.grow_cone_half_angle >= 90.0f)
    return fail("grow.cone_half_angle must be in [0, 90)");
  if (!positive(c.grow_direction_sample_step))
    return fail("grow.direction_sample_step must be finite and positive");
  if (c.grow_cone_half_angle > 0.0f &&
      c.grow_direction_sample_step > c.grow_cone_half_angle)
    return fail("grow.direction_sample_step must not exceed cone_half_angle");
  const double half_direction_samples =
      std::floor(static_cast<double>(c.grow_cone_half_angle) /
                 c.grow_direction_sample_step);
  if (half_direction_samples >
      (static_cast<double>(std::numeric_limits<int>::max()) - 1.0) / 2.0)
    return fail("grow cone/sample ratio is too large");
  if (c.grow_window_width <= 0 || (c.grow_window_width & 1) == 0)
    return fail("grow.window_width must be a positive odd integer");
  if (c.grow_window_length <= 0)
    return fail("grow.window_length must be positive");
  if (!std::isfinite(c.grow_max_pixel_angle) || c.grow_max_pixel_angle < 0.0f ||
      c.grow_max_pixel_angle > 90.0f)
    return fail("grow.max_pixel_angle must be in [0, 90]");
  if (c.grow_min_scored_pixels <= 0 ||
      static_cast<int64_t>(c.grow_min_scored_pixels) >
          static_cast<int64_t>(c.grow_window_width) * c.grow_window_length)
    return fail("grow.min_scored_pixels must fit in the directional window");
  if (c.grow_min_views < 2)
    return fail("grow.min_views must be at least two");
  if (!std::isfinite(c.grow_max_direction_change) ||
      c.grow_max_direction_change <= 0.0f || c.grow_max_direction_change > 180.0f)
    return fail("grow.max_direction_change must be in (0, 180]");
  if (c.grow_irls_iterations < 0)
    return fail("grow.irls_iterations must be nonnegative");
  if (!positive(c.grow_max_growth_length))
    return fail("grow.max_growth_length must be finite and positive");
  if (!std::isfinite(c.grow_min_intensity) || c.grow_min_intensity < 0.0f ||
      c.grow_min_intensity > 1.0f)
    return fail("grow.min_intensity must be in [0, 1]");
  const double steps =
      StableGrowthStepCount(c.grow_max_growth_length, c.grow_step_size);
  if (steps > static_cast<double>(std::numeric_limits<int>::max()))
    return fail("grow length/step ratio is too large");
  return true;
}

GrowParams MakeGrowParams(const Config& c) {
  GrowParams p;
  constexpr float radians = kGrowPi / 180.0f;
  p.step_size = c.grow_step_size;
  p.cone_half_angle = c.grow_cone_half_angle * radians;
  p.direction_sample_step = c.grow_direction_sample_step * radians;
  const int half_direction_samples = static_cast<int>(std::floor(
      static_cast<double>(c.grow_cone_half_angle) /
      c.grow_direction_sample_step));
  p.direction_samples = 2 * half_direction_samples + 1;
  p.window_width = c.grow_window_width;
  p.window_length = c.grow_window_length;
  p.max_pixel_angle = c.grow_max_pixel_angle * radians;
  p.min_scored_pixels = c.grow_min_scored_pixels;
  p.min_views = c.grow_min_views;
  p.max_direction_change = c.grow_max_direction_change * radians;
  p.irls_iterations = c.grow_irls_iterations;
  p.max_growth_length = c.grow_max_growth_length;
  p.max_steps = static_cast<int>(
      StableGrowthStepCount(p.max_growth_length, p.step_size));
  return p;
}

float LineAngleDifference(float a, float b) {
  if (!std::isfinite(a) || !std::isfinite(b))
    return std::numeric_limits<float>::infinity();
  float difference = std::fmod(std::fabs(a - b), kGrowPi);
  if (difference > 0.5f * kGrowPi)
    difference = kGrowPi - difference;
  return difference;
}

bool ProjectDirection(const GpuCamera& camera, const Vec3& point,
                      const Vec3& tangent, ProjectedDirection* projected) {
  if (!projected || !IsFinite(point) || !IsFinite(tangent))
    return false;
  float q[3], dq[3];
  for (int r = 0; r < 3; ++r) {
    q[r] = camera.R[r * 3] * point[0] + camera.R[r * 3 + 1] * point[1] +
           camera.R[r * 3 + 2] * point[2] + camera.t[r];
    dq[r] = camera.R[r * 3] * tangent[0] + camera.R[r * 3 + 1] * tangent[1] +
            camera.R[r * 3 + 2] * tangent[2];
  }
  if (!std::isfinite(q[2]) || q[2] <= 1e-6f)
    return false;

  float numerator[2], d_numerator[2];
  for (int r = 0; r < 2; ++r) {
    numerator[r] = camera.K[r * 3] * q[0] + camera.K[r * 3 + 1] * q[1] +
                   camera.K[r * 3 + 2] * q[2];
    d_numerator[r] = camera.K[r * 3] * dq[0] + camera.K[r * 3 + 1] * dq[1] +
                     camera.K[r * 3 + 2] * dq[2];
  }
  const float denominator =
      camera.K[6] * q[0] + camera.K[7] * q[1] + camera.K[8] * q[2];
  const float d_denominator =
      camera.K[6] * dq[0] + camera.K[7] * dq[1] + camera.K[8] * dq[2];
  if (!std::isfinite(denominator) || std::fabs(denominator) <= 1e-10f)
    return false;

  const float inverse = 1.0f / denominator;
  projected->x = numerator[0] * inverse;
  projected->y = numerator[1] * inverse;
  projected->dx =
      (d_numerator[0] * denominator - numerator[0] * d_denominator) * inverse * inverse;
  projected->dy =
      (d_numerator[1] * denominator - numerator[1] * d_denominator) * inverse * inverse;
  const float direction_length = std::hypot(projected->dx, projected->dy);
  if (!std::isfinite(direction_length) || direction_length <= 1e-8f)
    return false;
  projected->dx /= direction_length;
  projected->dy /= direction_length;
  projected->normal_x = -projected->dy;
  projected->normal_y = projected->dx;
  return std::isfinite(projected->x) && std::isfinite(projected->y);
}

bool ImageDirectionToPlaneNormal(const GpuCamera& camera, float x, float y,
                                 float dx, float dy, Vec3* plane_normal) {
  if (!plane_normal || !std::isfinite(x) || !std::isfinite(y) ||
      !std::isfinite(dx) || !std::isfinite(dy))
    return false;
  const float length = std::hypot(dx, dy);
  if (length <= 1e-10f)
    return false;
  dx /= length;
  dy /= length;
  // Homogeneous image line l=(normal_x,normal_y,c), with l^T [x y 1]=0.
  const float line[3] = {-dy, dx, dy * x - dx * y};
  float camera_normal[3];  // K^T l
  for (int r = 0; r < 3; ++r) {
    camera_normal[r] = camera.K[r] * line[0] + camera.K[3 + r] * line[1] +
                       camera.K[6 + r] * line[2];
  }
  // Plane normal in world coordinates is R^T K^T l.
  for (int r = 0; r < 3; ++r) {
    (*plane_normal)[r] = camera.R[r] * camera_normal[0] +
                         camera.R[3 + r] * camera_normal[1] +
                         camera.R[6 + r] * camera_normal[2];
  }
  return IsFinite(*plane_normal) && Normalize(plane_normal);
}

bool SolveDirectionIRLS(const std::vector<Vec3>& input_normals,
                        const Vec3& reference, int iterations, Vec3* direction) {
  if (!direction || input_normals.size() < 2 || iterations < 0)
    return false;
  std::vector<Vec3> normals;
  normals.reserve(input_normals.size());
  for (Vec3 normal : input_normals) {
    if (IsFinite(normal) && Normalize(&normal))
      normals.push_back(normal);
  }
  if (normals.size() < 2)
    return false;

  float matrix[6];
  AccumulateNormalMatrix(normals, nullptr, matrix);
  Vec3 estimate;
  if (!SmallestEigenvector(matrix, &estimate))
    return false;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    AccumulateNormalMatrix(normals, &estimate, matrix);
    Vec3 updated;
    if (!SmallestEigenvector(matrix, &updated))
      return false;
    if (Dot(updated, estimate) < 0.0f)
      for (float& x : updated)
        x = -x;
    estimate = updated;
  }
  Vec3 sign_reference = reference;
  if (!Normalize(&sign_reference))
    return false;
  if (Dot(estimate, sign_reference) < 0.0f)
    for (float& x : estimate)
      x = -x;
  *direction = estimate;
  return true;
}

float HalfBitsToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  uint32_t exponent = (bits >> 10) & 0x1fu;
  uint32_t mantissa = bits & 0x03ffu;
  uint32_t word;
  if (exponent == 0) {
    if (mantissa == 0) {
      word = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x03ffu;
      const uint32_t float_exponent = static_cast<uint32_t>(127 - 14 - shift);
      word = sign | (float_exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    word = sign | 0x7f800000u | (mantissa << 13);
  } else {
    word = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &word, sizeof(result));
  return result;
}

ViewScore ScoreViewCpu(const ViewData& view, const Vec3& point,
                       const Vec3& tangent, const GrowParams& params) {
  ViewScore result;
  if (view.width <= 0 || view.height <= 0 || !view.orientation_variance)
    return result;
  ProjectedDirection projected;
  if (!ProjectDirection(view.camera, point, tangent, &projected))
    return result;

  const float target_normal_angle =
      std::atan2(projected.normal_y, projected.normal_x);
  int best_count = -1;
  float best_error = std::numeric_limits<float>::infinity();
  float best_offset = 0.0f;
  float best_dx = 0.0f, best_dy = 0.0f;
  for (int candidate = 0; candidate < params.direction_samples; ++candidate) {
    const float offset =
        (candidate - params.direction_samples / 2) *
        params.direction_sample_step;
    const float cosine = std::cos(offset);
    const float sine = std::sin(offset);
    const float dx = cosine * projected.dx - sine * projected.dy;
    const float dy = sine * projected.dx + cosine * projected.dy;
    const float nx = -dy, ny = dx;
    int count = 0;
    float error_sum = 0.0f;
    const int half_width = params.window_width / 2;
    for (int along = 1; along <= params.window_length; ++along) {
      for (int across = -half_width; across <= half_width; ++across) {
        const float sample_x = projected.x + along * dx + across * nx;
        const float sample_y = projected.y + along * dy + across * ny;
        const int ix = static_cast<int>(std::floor(sample_x + 0.5f));
        const int iy = static_cast<int>(std::floor(sample_y + 0.5f));
        if (ix < 0 || iy < 0 || ix >= view.width || iy >= view.height)
          continue;
        const size_t index = static_cast<size_t>(iy) * view.width + ix;
        if (view.foreground && view.foreground[index] == 0)
          continue;
        const uint32_t packed = view.orientation_variance[index];
        const float theta = HalfBitsToFloat(static_cast<uint16_t>(packed));
        const float variance = HalfBitsToFloat(static_cast<uint16_t>(packed >> 16));
        if (!std::isfinite(theta) || !std::isfinite(variance) || variance <= 0.0f)
          continue;
        const float error = LineAngleDifference(theta, target_normal_angle);
        if (error > params.max_pixel_angle)
          continue;
        ++count;
        error_sum += error;
      }
    }
    const float mean_error = count ? error_sum / count
                                   : std::numeric_limits<float>::infinity();
    if (count < params.min_scored_pixels)
      continue;
    const bool better = mean_error < best_error - 1e-7f ||
                        (std::fabs(mean_error - best_error) <= 1e-7f &&
                         count > best_count) ||
                        (std::fabs(mean_error - best_error) <= 1e-7f &&
                         count == best_count &&
                         std::fabs(offset) < std::fabs(best_offset) - 1e-7f);
    if (better) {
      best_count = count;
      best_error = mean_error;
      best_offset = offset;
      best_dx = dx;
      best_dy = dy;
    }
  }
  if (best_count < params.min_scored_pixels)
    return result;
  Vec3 normal;
  if (!ImageDirectionToPlaneNormal(view.camera, projected.x, projected.y, best_dx,
                                   best_dy, &normal))
    return result;
  result.valid = true;
  result.scored_pixels = best_count;
  result.mean_error = best_error;
  result.direction_x = best_dx;
  result.direction_y = best_dy;
  result.plane_normal = normal;
  return result;
}

int CountPointSupportCpu(const std::vector<ViewData>& views, const Vec3& point) {
  int support = 0;
  for (const ViewData& view : views) {
    float x, y;
    if (ProjectPoint(view.camera, point, &x, &y) && PixelPasses(view, x, y))
      ++support;
  }
  return support;
}

TipGrowth GrowTipCpu(const TipSeed& tip, const std::vector<ViewData>& views,
                     const GrowParams& params) {
  TipGrowth growth;
  Vec3 point = tip.point;
  Vec3 tangent = tip.tangent;
  if (!Normalize(&tangent)) {
    growth.stop_reason = StopReason::kInvalidTangent;
    return growth;
  }
  for (int step = 0; step < params.max_steps; ++step) {
    std::vector<Vec3> normals;
    normals.reserve(views.size());
    for (const ViewData& view : views) {
      const ViewScore score = ScoreViewCpu(view, point, tangent, params);
      if (score.valid)
        normals.push_back(score.plane_normal);
    }
    if (static_cast<int>(normals.size()) < params.min_views) {
      growth.stop_reason = StopReason::kInsufficientViewDirections;
      return growth;
    }
    Vec3 direction;
    if (!SolveDirectionIRLS(normals, tangent, params.irls_iterations, &direction)) {
      growth.stop_reason = StopReason::kInsufficientViewDirections;
      return growth;
    }
    const float cosine = std::max(-1.0f, std::min(1.0f, Dot(direction, tangent)));
    if (std::acos(cosine) > params.max_direction_change) {
      growth.stop_reason = StopReason::kDirectionChange;
      return growth;
    }
    Vec3 next = {{point[0] + params.step_size * direction[0],
                  point[1] + params.step_size * direction[1],
                  point[2] + params.step_size * direction[2]}};
    if (CountPointSupportCpu(views, next) < params.min_views) {
      growth.stop_reason = StopReason::kInsufficientForeground;
      return growth;
    }
    growth.samples.push_back({next, direction});
    point = next;
    tangent = direction;
  }
  growth.stop_reason = StopReason::kLengthLimit;
  return growth;
}

std::vector<TipSeed> BuildInitialTips(const std::vector<Strand>& strands) {
  std::vector<TipSeed> tips;
  tips.reserve(strands.size() * 2);
  constexpr float distinct_squared = 1e-12f;
  for (size_t strand_index = 0; strand_index < strands.size(); ++strand_index) {
    const Strand& strand = strands[strand_index];
    const size_t count = strand.NumPoints();
    if (count < 2 || strand.positions.size() < count * 3)
      continue;
    for (int end = 0; end < 2; ++end) {
      const size_t endpoint = end == 0 ? 0 : count - 1;
      const Vec3 point = {{strand.positions[endpoint * 3],
                           strand.positions[endpoint * 3 + 1],
                           strand.positions[endpoint * 3 + 2]}};
      bool found = false;
      Vec3 tangent{};
      for (size_t distance = 1; distance < count; ++distance) {
        const size_t adjacent = end == 0 ? distance : count - 1 - distance;
        tangent = {{point[0] - strand.positions[adjacent * 3],
                    point[1] - strand.positions[adjacent * 3 + 1],
                    point[2] - strand.positions[adjacent * 3 + 2]}};
        if (Dot(tangent, tangent) > distinct_squared && Normalize(&tangent)) {
          found = true;
          break;
        }
      }
      if (found)
        tips.push_back({strand_index, end == 0, point, tangent});
    }
  }
  return tips;
}

Strand MergeTipGrowth(const Strand& original, const TipGrowth* prepend,
                      const TipGrowth* append) {
  Strand result;
  const size_t front_count = prepend ? prepend->samples.size() : 0;
  const size_t back_count = append ? append->samples.size() : 0;
  const size_t total = front_count + original.NumPoints() + back_count;
  result.positions.reserve(total * 3);
  result.directions.reserve(total * 3);
  result.labels.reserve(total);
  const int front_label = original.labels.empty() ? 0 : original.labels.front();
  const int back_label = original.labels.empty() ? 0 : original.labels.back();

  if (prepend) {
    for (auto it = prepend->samples.rbegin(); it != prepend->samples.rend(); ++it) {
      result.AddPoint(it->position[0], it->position[1], it->position[2],
                      -it->outward_direction[0], -it->outward_direction[1],
                      -it->outward_direction[2], front_label);
    }
  }
  // Copy original values verbatim (including potentially non-unit directions).
  for (size_t i = 0; i < original.NumPoints(); ++i) {
    const size_t p = i * 3;
    const float dx = p + 2 < original.directions.size() ? original.directions[p] : 0;
    const float dy = p + 2 < original.directions.size() ? original.directions[p + 1] : 0;
    const float dz = p + 2 < original.directions.size() ? original.directions[p + 2] : 0;
    const int label = i < original.labels.size() ? original.labels[i] : 0;
    result.AddPoint(original.positions[p], original.positions[p + 1],
                    original.positions[p + 2], dx, dy, dz, label);
  }
  if (append) {
    for (const GrowthSample& sample : append->samples) {
      result.AddPoint(sample.position[0], sample.position[1], sample.position[2],
                      sample.outward_direction[0], sample.outward_direction[1],
                      sample.outward_direction[2], back_label);
    }
  }
  return result;
}

}  // namespace detail

std::vector<Strand> GrowCuda(const std::vector<Strand>& strands,
                             const CameraArray& cameras, const Config& config,
                             int gpu_id, int num_gpus) {
  return GrowCuda(strands, cameras, config, gpu_id, num_gpus, nullptr);
}

std::vector<Strand> GrowCuda(const std::vector<Strand>& strands,
                             const CameraArray& cameras, const Config& config,
                             int gpu_id, int num_gpus, GrowthStats* stats) {
  ScopedTimer timer("Multi-view hair growing");
  GrowthStats local_stats;
  local_stats.input_strands = strands.size();
  for (const Strand& strand : strands)
    local_stats.input_points += strand.NumPoints();

  std::string config_error;
  if (!detail::ValidateConfig(config, &config_error))
    throw std::invalid_argument(config_error);
  if (strands.empty()) {
    if (stats)
      *stats = local_stats;
    return strands;
  }

  const std::vector<detail::TipSeed> tips = detail::BuildInitialTips(strands);
  local_stats.candidate_tips = tips.size();
  // Unsupported strands are a successful identity operation and deliberately
  // do not require image preparation or a CUDA device.
  if (tips.empty()) {
    LOG_INFO("Hair growing: no strand has a geometrically supported tip");
    if (stats)
      *stats = local_stats;
    return strands;
  }

  int device_count = 0;
  const cudaError_t device_error = cudaGetDeviceCount(&device_count);
  if (device_error != cudaSuccess || device_count <= 0)
    throw std::runtime_error("hair growing requires a CUDA device");
  if (gpu_id < 0 || gpu_id >= device_count) {
    std::ostringstream message;
    message << "grow gpu_id " << gpu_id << " is outside [0," << device_count << ')';
    throw std::runtime_error(message.str());
  }
  num_gpus = std::max(1, num_gpus);
  num_gpus = std::min(num_gpus, device_count - gpu_id);

  // The public API accepts the original calibration schema. Own the same
  // intrinsic scaling performed by MVS so direct API callers cannot pair
  // downsampled images with full-resolution intrinsics accidentally.
  CameraArray prepared_cameras = cameras;
  if (config.downsample != 1.0f)
    prepared_cameras.DownsizeCameras(config.downsample);
  const std::vector<orientation_cache::PreparedView> prepared =
      orientation_cache::PrepareViews(prepared_cameras, config, gpu_id);
  std::vector<detail::ViewData> views;
  views.reserve(prepared.size());
  int common_width = -1, common_height = -1;
  const auto compatible_camera = [](const GpuCamera& camera) {
    for (float value : camera.K)
      if (!std::isfinite(value))
        return false;
    for (float value : camera.R)
      if (!std::isfinite(value))
        return false;
    for (float value : camera.t)
      if (!std::isfinite(value))
        return false;
    const float intrinsic_determinant =
        camera.K[0] * (camera.K[4] * camera.K[8] -
                       camera.K[5] * camera.K[7]) -
        camera.K[1] * (camera.K[3] * camera.K[8] -
                       camera.K[5] * camera.K[6]) +
        camera.K[2] * (camera.K[3] * camera.K[7] -
                       camera.K[4] * camera.K[6]);
    const float rotation_determinant =
        camera.R[0] * (camera.R[4] * camera.R[8] -
                       camera.R[5] * camera.R[7]) -
        camera.R[1] * (camera.R[3] * camera.R[8] -
                       camera.R[5] * camera.R[6]) +
        camera.R[2] * (camera.R[3] * camera.R[7] -
                       camera.R[4] * camera.R[6]);
    return std::isfinite(intrinsic_determinant) &&
           std::fabs(intrinsic_determinant) > 1e-10f &&
           std::isfinite(rotation_determinant) &&
           std::fabs(rotation_determinant) > 1e-6f;
  };
  for (const auto& view : prepared) {
    const size_t pixels = view.width > 0 && view.height > 0
                              ? static_cast<size_t>(view.width) * view.height
                              : 0;
    if (pixels == 0 || view.orientation_variance.size() != pixels ||
        (!view.boundary.empty() && view.boundary.size() != pixels)) {
      LOG_WARN("Grow: skipping camera %u with incompatible cached map", view.camera_id);
      continue;
    }
    if (!compatible_camera(view.camera)) {
      LOG_WARN("Grow: skipping camera %u with incompatible calibration",
               view.camera_id);
      continue;
    }
    if (common_width < 0) {
      common_width = view.width;
      common_height = view.height;
    }
    if (view.width != common_width || view.height != common_height) {
      LOG_WARN("Grow: skipping camera %u (%dx%d; expected %dx%d)", view.camera_id,
               view.width, view.height, common_width, common_height);
      continue;
    }
    detail::ViewData data;
    data.camera = view.camera;
    data.camera_index = view.camera_index;
    data.camera_id = view.camera_id;
    data.width = view.width;
    data.height = view.height;
    data.orientation_variance = view.orientation_variance.data();
    data.foreground = view.boundary.empty() ? nullptr : view.boundary.data();
    views.push_back(data);
  }
  local_stats.usable_views = views.size();
  if (static_cast<int>(views.size()) < config.grow_min_views) {
    std::ostringstream message;
    message << "hair growing needs at least " << config.grow_min_views
            << " usable views, but only " << views.size() << " were prepared";
    throw std::runtime_error(message.str());
  }

  const detail::GrowParams params = detail::MakeGrowParams(config);
  std::vector<detail::TipGrowth> growth(tips.size());
  const int workers = std::min<int>(num_gpus, tips.size());
  local_stats.gpus_used = workers;
  std::exception_ptr worker_error;
  std::mutex error_mutex;
  std::vector<size_t> batch_counts(workers, 0);

#pragma omp parallel for num_threads(workers) schedule(static)
  for (int worker = 0; worker < workers; ++worker) {
    const size_t begin = tips.size() * worker / workers;
    const size_t end = tips.size() * (worker + 1) / workers;
    try {
      const detail::DeviceRunStats device_stats = detail::GrowTipsOnDevice(
          views, tips, begin, end, params, gpu_id + worker, &growth);
      batch_counts[worker] = device_stats.batches;
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!worker_error)
        worker_error = std::current_exception();
    }
  }
  if (worker_error)
    std::rethrow_exception(worker_error);
  for (size_t count : batch_counts)
    local_stats.batches += count;

  std::vector<const detail::TipGrowth*> front(strands.size(), nullptr);
  std::vector<const detail::TipGrowth*> back(strands.size(), nullptr);
  for (size_t i = 0; i < tips.size(); ++i) {
    if (tips[i].prepend)
      front[tips[i].strand_index] = &growth[i];
    else
      back[tips[i].strand_index] = &growth[i];
    if (!growth[i].samples.empty())
      ++local_stats.grown_tips;
    local_stats.points_added += growth[i].samples.size();
  }

  std::vector<Strand> result;
  result.reserve(strands.size());
  for (size_t i = 0; i < strands.size(); ++i)
    result.push_back(detail::MergeTipGrowth(strands[i], front[i], back[i]));

  LOG_INFO("Hair growing: %zu views, %zu/%zu tips extended, %zu points added "
           "(%d GPU(s), %zu batch(es))",
           local_stats.usable_views, local_stats.grown_tips,
           local_stats.candidate_tips, local_stats.points_added,
           local_stats.gpus_used, local_stats.batches);
  if (stats)
    *stats = local_stats;
  return result;
}

}  // namespace hair_grower
