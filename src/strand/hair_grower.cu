// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// One CUDA block grows one independent strand tip. Threads score the prepared
// views in parallel, while thread zero performs the tiny deterministic 3x3
// robust solve. A block follows its tip to termination, avoiding a host/kernel
// round trip for every 0.1 mm step.

#include "strand/hair_grower_detail.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "common/cuda_check.h"
#include "common/logger.h"

namespace hair_grower {
namespace detail {
namespace {

constexpr int kThreadsPerTip = 128;
constexpr float kPi = 3.14159265358979323846f;

struct DeviceTip {
  float3 point;
  float3 tangent;
};

struct DeviceSample {
  float3 position;
  float3 direction;
};

__device__ __forceinline__ float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 scale3(float3 v, float scale) {
  return make_float3(v.x * scale, v.y * scale, v.z * scale);
}

__device__ __forceinline__ bool normalize3(float3* v) {
  const float length_squared = dot3(*v, *v);
  if (!isfinite(length_squared) || length_squared <= 1e-20f)
    return false;
  *v = scale3(*v, rsqrtf(length_squared));
  return isfinite(v->x) && isfinite(v->y) && isfinite(v->z);
}

__device__ __forceinline__ float lineAngleDifferenceDevice(float a, float b) {
  float difference = fmodf(fabsf(a - b), kPi);
  if (difference > 0.5f * kPi)
    difference = kPi - difference;
  return difference;
}

__device__ bool projectPointDevice(const GpuCamera& camera, float3 point, float* x,
                                   float* y) {
  float q[3];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    q[row] = camera.R[row * 3] * point.x + camera.R[row * 3 + 1] * point.y +
             camera.R[row * 3 + 2] * point.z + camera.t[row];
  }
  if (!isfinite(q[2]) || q[2] <= 1e-6f)
    return false;
  const float numerator_x =
      camera.K[0] * q[0] + camera.K[1] * q[1] + camera.K[2] * q[2];
  const float numerator_y =
      camera.K[3] * q[0] + camera.K[4] * q[1] + camera.K[5] * q[2];
  const float denominator =
      camera.K[6] * q[0] + camera.K[7] * q[1] + camera.K[8] * q[2];
  if (!isfinite(denominator) || fabsf(denominator) <= 1e-10f)
    return false;
  *x = numerator_x / denominator;
  *y = numerator_y / denominator;
  return isfinite(*x) && isfinite(*y);
}

__device__ bool projectDirectionDevice(const GpuCamera& camera, float3 point,
                                       float3 tangent, float* x, float* y,
                                       float* dx, float* dy, float* normal_x,
                                       float* normal_y) {
  float q[3], dq[3];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    q[row] = camera.R[row * 3] * point.x + camera.R[row * 3 + 1] * point.y +
             camera.R[row * 3 + 2] * point.z + camera.t[row];
    dq[row] = camera.R[row * 3] * tangent.x + camera.R[row * 3 + 1] * tangent.y +
              camera.R[row * 3 + 2] * tangent.z;
  }
  if (!isfinite(q[2]) || q[2] <= 1e-6f)
    return false;

  float numerator[2], d_numerator[2];
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    numerator[row] = camera.K[row * 3] * q[0] +
                     camera.K[row * 3 + 1] * q[1] +
                     camera.K[row * 3 + 2] * q[2];
    d_numerator[row] = camera.K[row * 3] * dq[0] +
                       camera.K[row * 3 + 1] * dq[1] +
                       camera.K[row * 3 + 2] * dq[2];
  }
  const float denominator =
      camera.K[6] * q[0] + camera.K[7] * q[1] + camera.K[8] * q[2];
  const float d_denominator =
      camera.K[6] * dq[0] + camera.K[7] * dq[1] + camera.K[8] * dq[2];
  if (!isfinite(denominator) || fabsf(denominator) <= 1e-10f)
    return false;
  const float inverse = 1.0f / denominator;
  *x = numerator[0] * inverse;
  *y = numerator[1] * inverse;
  *dx = (d_numerator[0] * denominator - numerator[0] * d_denominator) *
        inverse * inverse;
  *dy = (d_numerator[1] * denominator - numerator[1] * d_denominator) *
        inverse * inverse;
  const float length = hypotf(*dx, *dy);
  if (!isfinite(length) || length <= 1e-8f)
    return false;
  *dx /= length;
  *dy /= length;
  *normal_x = -*dy;
  *normal_y = *dx;
  return isfinite(*x) && isfinite(*y);
}

__device__ bool imageDirectionToPlaneNormalDevice(const GpuCamera& camera, float x,
                                                   float y, float dx, float dy,
                                                   float3* normal) {
  const float length = hypotf(dx, dy);
  if (!isfinite(length) || length <= 1e-10f)
    return false;
  dx /= length;
  dy /= length;
  const float line[3] = {-dy, dx, dy * x - dx * y};
  float camera_normal[3];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    camera_normal[row] = camera.K[row] * line[0] + camera.K[3 + row] * line[1] +
                         camera.K[6 + row] * line[2];
  }
  normal->x = camera.R[0] * camera_normal[0] + camera.R[3] * camera_normal[1] +
              camera.R[6] * camera_normal[2];
  normal->y = camera.R[1] * camera_normal[0] + camera.R[4] * camera_normal[1] +
              camera.R[7] * camera_normal[2];
  normal->z = camera.R[2] * camera_normal[0] + camera.R[5] * camera_normal[1] +
              camera.R[8] * camera_normal[2];
  return normalize3(normal);
}

__device__ bool pixelPassesGate(cudaTextureObject_t foreground_texture,
                                bool use_foreground, int width, int height,
                                float x, float y, int layer) {
  const int ix = static_cast<int>(floorf(x + 0.5f));
  const int iy = static_cast<int>(floorf(y + 0.5f));
  if (ix < 0 || iy < 0 || ix >= width || iy >= height)
    return false;
  if (!use_foreground)
    return true;
  return tex2DLayered<unsigned char>(foreground_texture, ix + 0.5f, iy + 0.5f,
                                     layer) != 0;
}

__device__ bool scoreViewDevice(cudaTextureObject_t orientation_texture,
                                cudaTextureObject_t foreground_texture,
                                bool use_foreground, const GpuCamera& camera,
                                int width, int height, int layer, float3 point,
                                float3 tangent, const GrowParams& params,
                                float3* plane_normal) {
  float x, y, projected_dx, projected_dy, normal_x, normal_y;
  if (!projectDirectionDevice(camera, point, tangent, &x, &y, &projected_dx,
                              &projected_dy, &normal_x, &normal_y))
    return false;
  const float target_normal_angle = atan2f(normal_y, normal_x);
  int best_count = -1;
  float best_error = CUDART_INF_F;
  float best_offset = 0.0f;
  float best_dx = 0.0f, best_dy = 0.0f;

  for (int candidate = 0; candidate < params.direction_samples; ++candidate) {
    const float offset =
        (candidate - params.direction_samples / 2) *
        params.direction_sample_step;
    float sine, cosine;
    sincosf(offset, &sine, &cosine);
    const float direction_x = cosine * projected_dx - sine * projected_dy;
    const float direction_y = sine * projected_dx + cosine * projected_dy;
    const float perpendicular_x = -direction_y;
    const float perpendicular_y = direction_x;
    int count = 0;
    float error_sum = 0.0f;
    const int half_width = params.window_width / 2;
    for (int along = 1; along <= params.window_length; ++along) {
      for (int across = -half_width; across <= half_width; ++across) {
        const float sample_x =
            x + along * direction_x + across * perpendicular_x;
        const float sample_y =
            y + along * direction_y + across * perpendicular_y;
        const int ix = static_cast<int>(floorf(sample_x + 0.5f));
        const int iy = static_cast<int>(floorf(sample_y + 0.5f));
        if (ix < 0 || iy < 0 || ix >= width || iy >= height)
          continue;
        if (use_foreground &&
            tex2DLayered<unsigned char>(foreground_texture, ix + 0.5f,
                                        iy + 0.5f, layer) == 0)
          continue;
        // The layered texture stores the two IEEE half bit patterns as an
        // unsigned short pair. This avoids implicit texture normalization.
        const ushort2 packed = tex2DLayered<ushort2>(
            orientation_texture, ix + 0.5f, iy + 0.5f, layer);
        const float theta = __half2float(__ushort_as_half(packed.x));
        const float variance = __half2float(__ushort_as_half(packed.y));
        if (!isfinite(theta) || !isfinite(variance) || variance <= 0.0f)
          continue;
        // Gabor theta is a line normal. Compare it to the existing projected
        // segment normal; the candidate changes where the forward window goes.
        const float error = lineAngleDifferenceDevice(theta, target_normal_angle);
        if (error > params.max_pixel_angle)
          continue;
        ++count;
        error_sum += error;
      }
    }
    const float mean_error = count > 0 ? error_sum / count : CUDART_INF_F;
    if (count < params.min_scored_pixels)
      continue;
    const bool better = mean_error < best_error - 1e-7f ||
                        (fabsf(mean_error - best_error) <= 1e-7f &&
                         count > best_count) ||
                        (fabsf(mean_error - best_error) <= 1e-7f &&
                         count == best_count &&
                         fabsf(offset) < fabsf(best_offset) - 1e-7f);
    if (better) {
      best_count = count;
      best_error = mean_error;
      best_offset = offset;
      best_dx = direction_x;
      best_dy = direction_y;
    }
  }
  if (best_count < params.min_scored_pixels)
    return false;
  return imageDirectionToPlaneNormalDevice(camera, x, y, best_dx, best_dy,
                                           plane_normal);
}

__device__ void jacobiRotateDevice(float a[3][3], float v[3][3], int p, int q) {
  const float apq = a[p][q];
  if (fabsf(apq) <= 1e-20f)
    return;
  const float tau = (a[q][q] - a[p][p]) / (2.0f * apq);
  const float t = copysignf(1.0f, tau) /
                  (fabsf(tau) + sqrtf(1.0f + tau * tau));
  const float c = rsqrtf(1.0f + t * t);
  const float s = t * c;
  const float app = a[p][p];
  const float aqq = a[q][q];
  a[p][p] = app - t * apq;
  a[q][q] = aqq + t * apq;
  a[p][q] = a[q][p] = 0.0f;
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    if (row == p || row == q)
      continue;
    const float arp = a[row][p];
    const float arq = a[row][q];
    a[row][p] = a[p][row] = c * arp - s * arq;
    a[row][q] = a[q][row] = s * arp + c * arq;
  }
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    const float vrp = v[row][p];
    const float vrq = v[row][q];
    v[row][p] = c * vrp - s * vrq;
    v[row][q] = s * vrp + c * vrq;
  }
}

__device__ bool smallestEigenvectorDevice(const float matrix[6], float3* result) {
  const float trace = matrix[0] + matrix[3] + matrix[5];
  if (!isfinite(trace) || trace <= 0.0f)
    return false;
  float a[3][3] = {{matrix[0], matrix[1], matrix[2]},
                   {matrix[1], matrix[3], matrix[4]},
                   {matrix[2], matrix[4], matrix[5]}};
  float v[3][3] = {{1.0f, 0.0f, 0.0f},
                   {0.0f, 1.0f, 0.0f},
                   {0.0f, 0.0f, 1.0f}};
#pragma unroll
  for (int sweep = 0; sweep < 12; ++sweep) {
    jacobiRotateDevice(a, v, 0, 1);
    jacobiRotateDevice(a, v, 0, 2);
    jacobiRotateDevice(a, v, 1, 2);
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
  if (!isfinite(a[minimum][minimum]) || !isfinite(a[second][second]) ||
      a[second][second] - a[minimum][minimum] <= 1e-6f * trace)
    return false;
  *result = make_float3(v[0][minimum], v[1][minimum], v[2][minimum]);
  return normalize3(result);
}

__device__ void accumulateNormalMatrixDevice(const float* normals, int count,
                                              const float3* estimate,
                                              float matrix[6]) {
#pragma unroll
  for (int element = 0; element < 6; ++element)
    matrix[element] = 0.0f;
  for (int view = 0; view < count; ++view) {
    const float3 normal = make_float3(normals[view * 3], normals[view * 3 + 1],
                                      normals[view * 3 + 2]);
    if (dot3(normal, normal) <= 0.5f)
      continue;
    float weight = 1.0f;
    if (estimate) {
      const float residual = dot3(normal, *estimate);
      weight = 1e-8f / (residual * residual + 1e-8f);
    }
    matrix[0] += weight * normal.x * normal.x;
    matrix[1] += weight * normal.x * normal.y;
    matrix[2] += weight * normal.x * normal.z;
    matrix[3] += weight * normal.y * normal.y;
    matrix[4] += weight * normal.y * normal.z;
    matrix[5] += weight * normal.z * normal.z;
  }
}

__device__ bool solveDirectionDevice(const float* normals, int num_views,
                                     float3 reference, int iterations,
                                     float3* direction) {
  float matrix[6];
  accumulateNormalMatrixDevice(normals, num_views, nullptr, matrix);
  float3 estimate;
  if (!smallestEigenvectorDevice(matrix, &estimate))
    return false;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    accumulateNormalMatrixDevice(normals, num_views, &estimate, matrix);
    float3 updated;
    if (!smallestEigenvectorDevice(matrix, &updated))
      return false;
    if (dot3(updated, estimate) < 0.0f)
      updated = scale3(updated, -1.0f);
    estimate = updated;
  }
  if (!normalize3(&reference))
    return false;
  if (dot3(estimate, reference) < 0.0f)
    estimate = scale3(estimate, -1.0f);
  *direction = estimate;
  return true;
}

__global__ void growTipsKernel(
    const DeviceTip* __restrict__ tips, int num_tips,
    const GpuCamera* __restrict__ cameras, int num_views,
    cudaTextureObject_t orientation_texture,
    cudaTextureObject_t foreground_texture, bool use_foreground, int width,
    int height, GrowParams params, DeviceSample* __restrict__ output_samples,
    int* __restrict__ output_counts, int* __restrict__ output_reasons) {
  const int tip_index = blockIdx.x;
  if (tip_index >= num_tips)
    return;
  const int thread = threadIdx.x;
  extern __shared__ float shared_normals[];  // 3*num_views floats
  __shared__ float3 point;
  __shared__ float3 tangent;
  __shared__ float3 candidate_direction;
  __shared__ float3 candidate_point;
  __shared__ int active;
  __shared__ int support;
  __shared__ int output_count;
  __shared__ int stop_reason;

  if (thread == 0) {
    point = tips[tip_index].point;
    tangent = tips[tip_index].tangent;
    active = normalize3(&tangent) ? 1 : 0;
    output_count = 0;
    stop_reason = active ? static_cast<int>(StopReason::kLengthLimit)
                         : static_cast<int>(StopReason::kInvalidTangent);
  }
  __syncthreads();

  for (int step = 0; step < params.max_steps; ++step) {
    if (!active)
      break;
    for (int view = thread; view < num_views; view += blockDim.x) {
      float3 normal = make_float3(0.0f, 0.0f, 0.0f);
      if (!scoreViewDevice(orientation_texture, foreground_texture,
                           use_foreground, cameras[view], width, height, view,
                           point, tangent, params, &normal))
        normal = make_float3(0.0f, 0.0f, 0.0f);
      shared_normals[view * 3] = normal.x;
      shared_normals[view * 3 + 1] = normal.y;
      shared_normals[view * 3 + 2] = normal.z;
    }
    __syncthreads();

    if (thread == 0) {
      int valid_views = 0;
      for (int view = 0; view < num_views; ++view) {
        const float length_squared =
            shared_normals[view * 3] * shared_normals[view * 3] +
            shared_normals[view * 3 + 1] * shared_normals[view * 3 + 1] +
            shared_normals[view * 3 + 2] * shared_normals[view * 3 + 2];
        if (length_squared > 0.5f)
          ++valid_views;
      }
      if (valid_views < params.min_views ||
          !solveDirectionDevice(shared_normals, num_views, tangent,
                                params.irls_iterations, &candidate_direction)) {
        active = 0;
        stop_reason =
            static_cast<int>(StopReason::kInsufficientViewDirections);
      } else {
        const float cosine = fmaxf(-1.0f, fminf(1.0f, dot3(candidate_direction,
                                                           tangent)));
        if (acosf(cosine) > params.max_direction_change) {
          active = 0;
          stop_reason = static_cast<int>(StopReason::kDirectionChange);
        } else {
          candidate_point =
              make_float3(point.x + params.step_size * candidate_direction.x,
                          point.y + params.step_size * candidate_direction.y,
                          point.z + params.step_size * candidate_direction.z);
        }
      }
      support = 0;
    }
    __syncthreads();
    if (!active)
      break;

    int local_support = 0;
    for (int view = thread; view < num_views; view += blockDim.x) {
      float x, y;
      if (projectPointDevice(cameras[view], candidate_point, &x, &y) &&
          pixelPassesGate(foreground_texture, use_foreground, width, height, x, y,
                          view))
        ++local_support;
    }
    if (local_support)
      atomicAdd(&support, local_support);
    __syncthreads();

    if (thread == 0) {
      if (support < params.min_views) {
        active = 0;
        stop_reason = static_cast<int>(StopReason::kInsufficientForeground);
      } else {
        DeviceSample& sample =
            output_samples[static_cast<size_t>(tip_index) * params.max_steps + step];
        sample.position = candidate_point;
        sample.direction = candidate_direction;
        ++output_count;
        point = candidate_point;
        tangent = candidate_direction;
      }
    }
    __syncthreads();
  }
  if (thread == 0) {
    output_counts[tip_index] = output_count;
    output_reasons[tip_index] = stop_reason;
  }
}

struct DeviceResources {
  cudaArray_t orientation_array = nullptr;
  cudaArray_t foreground_array = nullptr;
  cudaTextureObject_t orientation_texture = 0;
  cudaTextureObject_t foreground_texture = 0;
  GpuCamera* cameras = nullptr;

  DeviceResources() = default;
  DeviceResources(const DeviceResources&) = delete;
  DeviceResources& operator=(const DeviceResources&) = delete;
  DeviceResources(DeviceResources&& other) noexcept
      : orientation_array(other.orientation_array),
        foreground_array(other.foreground_array),
        orientation_texture(other.orientation_texture),
        foreground_texture(other.foreground_texture),
        cameras(other.cameras) {
    other.orientation_array = nullptr;
    other.foreground_array = nullptr;
    other.orientation_texture = 0;
    other.foreground_texture = 0;
    other.cameras = nullptr;
  }
  DeviceResources& operator=(DeviceResources&&) = delete;

  ~DeviceResources() {
    if (orientation_texture)
      cudaDestroyTextureObject(orientation_texture);
    if (foreground_texture)
      cudaDestroyTextureObject(foreground_texture);
    if (orientation_array)
      cudaFreeArray(orientation_array);
    if (foreground_array)
      cudaFreeArray(foreground_array);
    if (cameras)
      cudaFree(cameras);
  }
};

void CreateTexture(cudaArray_t array, cudaTextureObject_t* texture) {
  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = array;
  cudaTextureDesc description{};
  description.addressMode[0] = cudaAddressModeClamp;
  description.addressMode[1] = cudaAddressModeClamp;
  description.filterMode = cudaFilterModePoint;
  description.readMode = cudaReadModeElementType;
  description.normalizedCoords = 0;
  CUDA_CHECK(cudaCreateTextureObject(texture, &resource, &description, nullptr));
}

DeviceResources UploadViews(const std::vector<ViewData>& views) {
  DeviceResources resources;
  const int width = views.front().width;
  const int height = views.front().height;
  const int layers = static_cast<int>(views.size());
  const cudaExtent extent = make_cudaExtent(width, height, layers);

  // Unsigned 16x2 stores exactly the two binary16 bit patterns. Kernels decode
  // with __ushort_as_half, so no texture conversion or normalization occurs.
  const cudaChannelFormatDesc half2_bits =
      cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned);
  CUDA_CHECK(cudaMalloc3DArray(&resources.orientation_array, &half2_bits, extent,
                               cudaArrayLayered));

  bool any_foreground = false;
  for (const ViewData& view : views)
    any_foreground = any_foreground || view.foreground != nullptr;
  if (any_foreground) {
    const cudaChannelFormatDesc byte = cudaCreateChannelDesc<unsigned char>();
    CUDA_CHECK(cudaMalloc3DArray(&resources.foreground_array, &byte, extent,
                                 cudaArrayLayered));
  }

  std::vector<uint8_t> all_foreground;
  if (any_foreground)
    all_foreground.assign(static_cast<size_t>(width) * height, 255);
  for (int layer = 0; layer < layers; ++layer) {
    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(
        const_cast<uint32_t*>(views[layer].orientation_variance),
        static_cast<size_t>(width) * sizeof(uint32_t), width, height);
    copy.dstArray = resources.orientation_array;
    copy.dstPos = make_cudaPos(0, 0, layer);
    copy.extent = make_cudaExtent(width, height, 1);
    copy.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copy));

    if (any_foreground) {
      const uint8_t* gate = views[layer].foreground
                                ? views[layer].foreground
                                : all_foreground.data();
      cudaMemcpy3DParms gate_copy{};
      gate_copy.srcPtr = make_cudaPitchedPtr(const_cast<uint8_t*>(gate), width,
                                             width, height);
      gate_copy.dstArray = resources.foreground_array;
      gate_copy.dstPos = make_cudaPos(0, 0, layer);
      gate_copy.extent = make_cudaExtent(width, height, 1);
      gate_copy.kind = cudaMemcpyHostToDevice;
      CUDA_CHECK(cudaMemcpy3D(&gate_copy));
    }
  }
  CreateTexture(resources.orientation_array, &resources.orientation_texture);
  if (any_foreground)
    CreateTexture(resources.foreground_array, &resources.foreground_texture);

  std::vector<GpuCamera> host_cameras;
  host_cameras.reserve(views.size());
  for (const ViewData& view : views)
    host_cameras.push_back(view.camera);
  CUDA_CHECK(cudaMalloc(&resources.cameras,
                        host_cameras.size() * sizeof(GpuCamera)));
  CUDA_CHECK(cudaMemcpy(resources.cameras, host_cameras.data(),
                        host_cameras.size() * sizeof(GpuCamera),
                        cudaMemcpyHostToDevice));
  return resources;
}

size_t SafeBytesPerTip(const GrowParams& params) {
  const size_t samples = static_cast<size_t>(params.max_steps);
  if (samples >
      (std::numeric_limits<size_t>::max() - sizeof(DeviceTip) - 2 * sizeof(int)) /
          sizeof(DeviceSample))
    throw std::runtime_error("grow output allocation size overflow");
  return sizeof(DeviceTip) + 2 * sizeof(int) + samples * sizeof(DeviceSample);
}

}  // namespace

DeviceRunStats GrowTipsOnDevice(const std::vector<ViewData>& views,
                                const std::vector<TipSeed>& tips,
                                size_t tip_begin, size_t tip_end,
                                const GrowParams& params, int device_id,
                                std::vector<TipGrowth>* output,
                                size_t max_batch_tips_for_test) {
  DeviceRunStats stats;
  if (tip_begin >= tip_end)
    return stats;
  if (!output || output->size() < tips.size())
    throw std::invalid_argument("GrowTipsOnDevice received an undersized output");
  if (views.empty())
    throw std::invalid_argument("GrowTipsOnDevice received no views");
  CUDA_CHECK(cudaSetDevice(device_id));
  cudaGetLastError();

  if (params.max_steps <= 0) {
    for (size_t index = tip_begin; index < tip_end; ++index)
      (*output)[index].stop_reason = StopReason::kLengthLimit;
    return stats;
  }

  DeviceResources resources = UploadViews(views);
  size_t free_bytes = 0, total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const size_t per_tip = SafeBytesPerTip(params);
  // Leave 20% of currently free memory for the CUDA runtime and allocations
  // from concurrent pipeline components. Failed allocations are retried with
  // successively smaller batches below.
  size_t batch_capacity = std::max<size_t>(1, (free_bytes / 5 * 4) / per_tip);
  batch_capacity = std::min(batch_capacity, tip_end - tip_begin);
  if (max_batch_tips_for_test > 0)
    batch_capacity = std::min(batch_capacity, max_batch_tips_for_test);
  LOG_DEBUG("Grow GPU %d: %.1f MiB free, auto batch=%zu tips", device_id,
            free_bytes / (1024.0 * 1024.0), batch_capacity);

  for (size_t begin = tip_begin; begin < tip_end;) {
    size_t count = std::min(batch_capacity, tip_end - begin);
    DeviceTip* device_tips = nullptr;
    DeviceSample* device_samples = nullptr;
    int* device_counts = nullptr;
    int* device_reasons = nullptr;

    while (true) {
      cudaError_t allocation = cudaMalloc(&device_tips, count * sizeof(DeviceTip));
      if (allocation == cudaSuccess) {
        allocation = cudaMalloc(
            &device_samples,
            count * static_cast<size_t>(params.max_steps) * sizeof(DeviceSample));
      }
      if (allocation == cudaSuccess)
        allocation = cudaMalloc(&device_counts, count * sizeof(int));
      if (allocation == cudaSuccess)
        allocation = cudaMalloc(&device_reasons, count * sizeof(int));
      if (allocation == cudaSuccess)
        break;
      if (device_tips)
        cudaFree(device_tips);
      if (device_samples)
        cudaFree(device_samples);
      if (device_counts)
        cudaFree(device_counts);
      if (device_reasons)
        cudaFree(device_reasons);
      device_tips = nullptr;
      device_samples = nullptr;
      device_counts = nullptr;
      device_reasons = nullptr;
      cudaGetLastError();
      if (count == 1)
        CUDA_CHECK(allocation);
      count = std::max<size_t>(1, count / 2);
      batch_capacity = count;
    }

    std::vector<DeviceTip> host_tips(count);
    for (size_t local = 0; local < count; ++local) {
      const TipSeed& tip = tips[begin + local];
      host_tips[local].point =
          make_float3(tip.point[0], tip.point[1], tip.point[2]);
      host_tips[local].tangent =
          make_float3(tip.tangent[0], tip.tangent[1], tip.tangent[2]);
    }
    CUDA_CHECK(cudaMemcpy(device_tips, host_tips.data(), count * sizeof(DeviceTip),
                          cudaMemcpyHostToDevice));

    const size_t shared_bytes = views.size() * 3 * sizeof(float);
    int shared_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit,
                                      cudaDevAttrMaxSharedMemoryPerBlock,
                                      device_id));
    if (shared_bytes > static_cast<size_t>(shared_limit)) {
      cudaFree(device_tips);
      cudaFree(device_samples);
      cudaFree(device_counts);
      cudaFree(device_reasons);
      throw std::runtime_error("too many grow views for per-tip shared memory");
    }

    const bool use_foreground = resources.foreground_texture != 0;
    growTipsKernel<<<static_cast<unsigned int>(count), kThreadsPerTip,
                     shared_bytes>>>(
        device_tips, static_cast<int>(count), resources.cameras,
        static_cast<int>(views.size()), resources.orientation_texture,
        resources.foreground_texture, use_foreground, views.front().width,
        views.front().height, params, device_samples, device_counts,
        device_reasons);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> host_counts(count), host_reasons(count);
    CUDA_CHECK(cudaMemcpy(host_counts.data(), device_counts, count * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_reasons.data(), device_reasons, count * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Only accepted prefixes are initialized by the kernel. Compact those
    // prefixes on the host instead of copying fixed-stride tails, which avoids
    // exposing stale device bytes and keeps host memory proportional to actual
    // growth rather than count*max_steps.
    std::vector<int> sample_counts(count);
    std::vector<size_t> sample_offsets(count + 1, 0);
    for (size_t local = 0; local < count; ++local) {
      sample_counts[local] =
          std::max(0, std::min(params.max_steps, host_counts[local]));
      sample_offsets[local + 1] =
          sample_offsets[local] + static_cast<size_t>(sample_counts[local]);
    }
    std::vector<DeviceSample> host_samples(sample_offsets.back());
    for (size_t local = 0; local < count;) {
      const int sample_count = sample_counts[local];
      if (sample_count == 0) {
        ++local;
        continue;
      }

      // Consecutive full-length tips occupy one contiguous initialized device
      // range and can be downloaded in a single transfer.
      size_t run_end = local + 1;
      if (sample_count == params.max_steps) {
        while (run_end < count &&
               sample_counts[run_end] == params.max_steps)
          ++run_end;
      }
      const size_t samples_to_copy =
          sample_offsets[run_end] - sample_offsets[local];
      CUDA_CHECK(cudaMemcpy(
          host_samples.data() + sample_offsets[local],
          device_samples + local * static_cast<size_t>(params.max_steps),
          samples_to_copy * sizeof(DeviceSample), cudaMemcpyDeviceToHost));
      local = run_end;
    }

    for (size_t local = 0; local < count; ++local) {
      TipGrowth& growth = (*output)[begin + local];
      const int sample_count = sample_counts[local];
      growth.samples.reserve(sample_count);
      for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
        const DeviceSample& sample =
            host_samples[sample_offsets[local] +
                         sample_index];
        growth.samples.push_back(
            {{{sample.position.x, sample.position.y, sample.position.z}},
             {{sample.direction.x, sample.direction.y, sample.direction.z}}});
      }
      growth.stop_reason = static_cast<StopReason>(host_reasons[local]);
    }

    cudaFree(device_tips);
    cudaFree(device_samples);
    cudaFree(device_counts);
    cudaFree(device_reasons);
    begin += count;
    ++stats.batches;
  }
  return stats;
}

}  // namespace detail
}  // namespace hair_grower
