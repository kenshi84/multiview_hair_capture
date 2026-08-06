// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace orientation_cache {
namespace detail {
namespace {

__global__ void PackOrientationVarianceKernel(
    const float* theta, size_t theta_pitch_bytes, const float* variance,
    size_t variance_pitch_bytes, int width, int height, uint32_t* packed) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height)
    return;

  const auto* theta_row = reinterpret_cast<const float*>(
      reinterpret_cast<const unsigned char*>(theta) +
      static_cast<size_t>(y) * theta_pitch_bytes);
  const auto* variance_row = reinterpret_cast<const float*>(
      reinterpret_cast<const unsigned char*>(variance) +
      static_cast<size_t>(y) * variance_pitch_bytes);
  float angle = theta_row[x];
  float confidence_variance = variance_row[x];
  const size_t index = static_cast<size_t>(y) * width + x;

  if (!isfinite(angle) || !isfinite(confidence_variance) ||
      confidence_variance <= 0.0f) {
    packed[index] = 0;
    return;
  }

  // Gabor angles describe unoriented line normals.  Canonicalizing here
  // protects the cache contract if a future producer emits an equivalent
  // angle outside [0, pi).
  constexpr float kPi = 3.14159265358979323846f;
  angle = fmodf(angle, kPi);
  if (angle < 0.0f)
    angle += kPi;

  // Preserve the zero half-variance code point exclusively for invalid
  // pixels.  IEEE binary16's smallest positive value is 2^-24; values above
  // the finite half range saturate instead of becoming infinity.
  constexpr float kSmallestPositiveHalf = 0x1.0p-24f;
  constexpr float kLargestFiniteHalf = 65504.0f;
  confidence_variance =
      fminf(kLargestFiniteHalf,
            fmaxf(kSmallestPositiveHalf, confidence_variance));

  const __half angle_half = __float2half_rn(angle);
  const __half variance_half = __float2half_rn(confidence_variance);
  const uint32_t angle_bits = static_cast<uint32_t>(__half_as_ushort(angle_half));
  uint32_t variance_bits =
      static_cast<uint32_t>(__half_as_ushort(variance_half));
  if ((variance_bits & 0x7fffu) == 0)
    variance_bits = 1u;
  packed[index] = angle_bits | (variance_bits << 16);
}

void SetError(std::string* error, const char* operation, cudaError_t status) {
  if (!error)
    return;
  *error = std::string(operation) + ": " + cudaGetErrorString(status);
}

}  // namespace

bool PackDeviceMapsToHost(const float* d_theta, size_t theta_pitch_bytes,
                          const float* d_variance,
                          size_t variance_pitch_bytes, int width, int height,
                          uint32_t* packed, std::string* error) {
  if (error)
    error->clear();
  if (!d_theta || !d_variance || !packed || width <= 0 || height <= 0 ||
      theta_pitch_bytes < static_cast<size_t>(width) * sizeof(float) ||
      variance_pitch_bytes < static_cast<size_t>(width) * sizeof(float)) {
    if (error)
      *error = "invalid map pointer, dimensions, or pitch";
    return false;
  }

  const uint64_t pixel_count = static_cast<uint64_t>(width) *
                               static_cast<uint64_t>(height);
  if (pixel_count > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
    if (error)
      *error = "map dimensions overflow host address space";
    return false;
  }

  uint32_t* d_packed = nullptr;
  cudaError_t status =
      cudaMalloc(&d_packed, static_cast<size_t>(pixel_count) * sizeof(uint32_t));
  if (status != cudaSuccess) {
    SetError(error, "cudaMalloc(packed half2 map)", status);
    return false;
  }

  const dim3 block(16, 16);
  const dim3 grid((static_cast<unsigned int>(width) + block.x - 1) / block.x,
                  (static_cast<unsigned int>(height) + block.y - 1) / block.y);
  PackOrientationVarianceKernel<<<grid, block>>>(
      d_theta, theta_pitch_bytes, d_variance, variance_pitch_bytes, width,
      height, d_packed);
  status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaMemcpy(packed, d_packed,
                        static_cast<size_t>(pixel_count) * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost);
  }

  const cudaError_t free_status = cudaFree(d_packed);
  if (status != cudaSuccess) {
    SetError(error, "pack half2 map", status);
    return false;
  }
  if (free_status != cudaSuccess) {
    SetError(error, "cudaFree(packed half2 map)", free_status);
    return false;
  }
  return true;
}

}  // namespace detail
}  // namespace orientation_cache
