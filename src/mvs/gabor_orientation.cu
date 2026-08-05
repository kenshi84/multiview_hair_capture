// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/cuda_check.h"
#include "common/cuda_vec_ops.h"
#include "common/types.h"
#include "mvs/patchmatch/constants.h"

#include <cmath>
#include <cstdio>

#ifndef PI
#define PI CUDART_PI_F
#endif

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Even/odd Gabor pair at integer offset (x,y). Combining the pair magnitude
// makes the orientation estimate independent of strand polarity and phase.
__device__ __forceinline__ float2 getGaborPairAt(int x, int y, float theta,
                                                 float sigma, float gamma,
                                                 float lambd) {
  float sigma_x = sigma;
  float sigma_y = sigma / gamma;
  float c = __cosf(theta);
  float s = __sinf(theta);
  float xr = x * c + y * s;
  float yr = -x * s + y * c;
  float envelope = __expf(
      -0.5f *
      (xr * xr / (sigma_x * sigma_x) + yr * yr / (sigma_y * sigma_y)));
  float sin_phase;
  float cos_phase;
  __sincosf(2.0f * PI / lambd * xr, &sin_phase, &cos_phase);
  return make_float2(envelope * cos_phase, envelope * sin_phase);
}

// Angular distance in [0, PI/2] respecting orientation symmetry.
__device__ __forceinline__ float angleDistance(float a, float b) {
  float d = fabsf(a - b);
  if (d > PI / 2.0f)
    d = PI - d;
  return d;
}

// Welford accumulation avoids cancellation when measuring nearly constant
// image patches. The returned energy is sum((pixel - mean)^2).
__device__ __forceinline__ bool getLinearPatchStats(
    const float* gray, unsigned int width, unsigned int height, unsigned int px,
    unsigned int py, int half, float* mean, float* energy, int* count) {
  *mean = 0.0f;
  *energy = 0.0f;
  *count = 0;

  for (int ky = -half; ky <= half; ++ky) {
    for (int kx = -half; kx <= half; ++kx) {
      int sx = static_cast<int>(px) + kx;
      int sy = static_cast<int>(py) + ky;
      if (sx < 0 || sx >= static_cast<int>(width) || sy < 0 ||
          sy >= static_cast<int>(height))
        continue;

      float pixel = gray[sy * width + sx];
      ++(*count);
      float delta = pixel - *mean;
      *mean += delta / static_cast<float>(*count);
      *energy += delta * (pixel - *mean);
    }
  }

  return *count > 0;
}

__device__ __forceinline__ bool getTexturePatchStats(
    cudaTextureObject_t gray, unsigned int width, unsigned int height,
    unsigned int px, unsigned int py, int half, float* mean, float* energy,
    int* count) {
  *mean = 0.0f;
  *energy = 0.0f;
  *count = 0;

  for (int ky = -half; ky <= half; ++ky) {
    for (int kx = -half; kx <= half; ++kx) {
      int sx = static_cast<int>(px) + kx;
      int sy = static_cast<int>(py) + ky;
      if (sx < 0 || sx >= static_cast<int>(width) || sy < 0 ||
          sy >= static_cast<int>(height))
        continue;

      float pixel = tex2D<float>(gray, sx + 0.5f, sy + 0.5f);
      ++(*count);
      float delta = pixel - *mean;
      *mean += delta / static_cast<float>(*count);
      *energy += delta * (pixel - *mean);
    }
  }

  return *count > 0;
}

// ---------------------------------------------------------------------------
// Kernel: Gabor orientation from linear float buffer
// ---------------------------------------------------------------------------

__global__ void GaborOrientationKernel(float* d_OrientMap, float* d_VarianceMap,
                                       float* d_GrayMap, unsigned int width,
                                       unsigned int height, int rotate_res,
                                       GaborParams params, float* d_maskMap) {
  unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height)
    return;

  unsigned int idx = py * width + px;

  if (d_maskMap != nullptr && d_maskMap[idx] == 0.0f) {
    d_OrientMap[idx] = 0.0f;
    d_VarianceMap[idx] = 0.0f;
    return;
  }

  int half = params.ksize / 2;
  float patch_mean;
  float patch_energy;
  int sample_count;
  if (!getLinearPatchStats(d_GrayMap, width, height, px, py, half, &patch_mean,
                           &patch_energy, &sample_count) ||
      patch_energy <= 1e-12f ||
      sqrtf(patch_energy / static_cast<float>(sample_count)) <
          params.min_contrast) {
    d_OrientMap[idx] = 0.0f;
    d_VarianceMap[idx] = 0.0f;
    return;
  }

  float vec_val[ORIENT2D_ROTATE_RES];
  float val_max = -1e30f;
  float val_min = 1e30f;
  int idx_max = 0;

  for (int ri = 0; ri < rotate_res; ++ri) {
    float theta = PI * static_cast<float>(ri) / static_cast<float>(rotate_res);
    float response_even = 0.0f;
    float response_odd = 0.0f;
    float kernel_sum_even = 0.0f;
    float kernel_sum_odd = 0.0f;
    float kernel_square_sum = 0.0f;

    for (int ky = -half; ky <= half; ++ky) {
      for (int kx = -half; kx <= half; ++kx) {
        int sx = static_cast<int>(px) + kx;
        int sy = static_cast<int>(py) + ky;
        if (sx < 0 || sx >= static_cast<int>(width) || sy < 0 ||
            sy >= static_cast<int>(height))
          continue;

        float2 g =
            getGaborPairAt(kx, ky, theta, params.sigma, params.gamma, params.lambd);
        float centered_pixel = d_GrayMap[sy * width + sx] - patch_mean;
        response_even += g.x * centered_pixel;
        response_odd += g.y * centered_pixel;
        kernel_sum_even += g.x;
        kernel_sum_odd += g.y;
        kernel_square_sum += g.x * g.x + g.y * g.y;
      }
    }

    // Centering the patch is algebraically equivalent to removing the
    // discrete kernel DC component. Normalize by patch and kernel energy so
    // the response threshold has a stable meaning across image contrast.
    float kernel_energy =
        kernel_square_sum -
        (kernel_sum_even * kernel_sum_even + kernel_sum_odd * kernel_sum_odd) /
            static_cast<float>(sample_count);
    kernel_energy = fmaxf(kernel_energy, 0.0f);
    float normalization = sqrtf(patch_energy * kernel_energy);
    float response = normalization > 1e-12f
                         ? hypotf(response_even, response_odd) / normalization
                         : 0.0f;

    vec_val[ri] = response;

    if (response > val_max) {
      val_max = response;
      idx_max = ri;
    }
    if (response < val_min) {
      val_min = response;
    }
  }

  float theta_max = PI * static_cast<float>(idx_max) / static_cast<float>(rotate_res);

  float variance = 0.0f;
  float range = val_max - val_min;
  if (val_max < params.min_response || range <= 1e-12f) {
    d_OrientMap[idx] = 0.0f;
    d_VarianceMap[idx] = 0.0f;
    return;
  }
  for (int ri = 0; ri < rotate_res; ++ri) {
    float theta_i = PI * static_cast<float>(ri) / static_cast<float>(rotate_res);
    float theta_diff = angleDistance(theta_i, theta_max) / (PI / 2.0f);
    float w = (vec_val[ri] - val_min) / range;
    variance += theta_diff * theta_diff * w;
  }
  variance /= rotate_res;

  d_OrientMap[idx] = theta_max;
  d_VarianceMap[idx] = variance;
}

// ---------------------------------------------------------------------------
// Kernel: Gabor orientation from cudaTextureObject_t (single pass)
// ---------------------------------------------------------------------------

__global__ void GaborOrientationTextureKernel(
    float* d_OrientMap, unsigned int pitchOrient, float* d_VarianceMap,
    unsigned int pitchVariance, cudaTextureObject_t d_GrayMap, unsigned int width,
    unsigned int height, int rotate_res, GaborParams params, float* d_maskMap) {
  unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height)
    return;

  // pitchOrient/pitchVariance are in elements (floats), not bytes
  // (caller converts from byte pitch before calling)

  // Skip masked-out pixels (mask is non-pitched)
  if (d_maskMap != nullptr && d_maskMap[py * width + px] == 0.0f) {
    d_OrientMap[py * pitchOrient + px] = 0.0f;
    d_VarianceMap[py * pitchVariance + px] = 0.0f;
    return;
  }

  int half = params.ksize / 2;
  float patch_mean;
  float patch_energy;
  int sample_count;
  if (!getTexturePatchStats(d_GrayMap, width, height, px, py, half, &patch_mean,
                            &patch_energy, &sample_count) ||
      patch_energy <= 1e-12f ||
      sqrtf(patch_energy / static_cast<float>(sample_count)) <
          params.min_contrast) {
    d_OrientMap[py * pitchOrient + px] = 0.0f;
    d_VarianceMap[py * pitchVariance + px] = 0.0f;
    return;
  }

  float vec_val[ORIENT2D_ROTATE_RES];
  float val_max = -1e30f;
  float val_min = 1e30f;
  int idx_max = 0;

  for (int ri = 0; ri < rotate_res; ++ri) {
    float theta = PI * static_cast<float>(ri) / static_cast<float>(rotate_res);
    float response_even = 0.0f;
    float response_odd = 0.0f;
    float kernel_sum_even = 0.0f;
    float kernel_sum_odd = 0.0f;
    float kernel_square_sum = 0.0f;

    for (int ky = -half; ky <= half; ++ky) {
      for (int kx = -half; kx <= half; ++kx) {
        int sx = static_cast<int>(px) + kx;
        int sy = static_cast<int>(py) + ky;
        if (sx < 0 || sx >= static_cast<int>(width) || sy < 0 ||
            sy >= static_cast<int>(height))
          continue;

        float2 g =
            getGaborPairAt(kx, ky, theta, params.sigma, params.gamma, params.lambd);
        float pixel = tex2D<float>(d_GrayMap, sx + 0.5f, sy + 0.5f);
        float centered_pixel = pixel - patch_mean;
        response_even += g.x * centered_pixel;
        response_odd += g.y * centered_pixel;
        kernel_sum_even += g.x;
        kernel_sum_odd += g.y;
        kernel_square_sum += g.x * g.x + g.y * g.y;
      }
    }

    float kernel_energy =
        kernel_square_sum -
        (kernel_sum_even * kernel_sum_even + kernel_sum_odd * kernel_sum_odd) /
            static_cast<float>(sample_count);
    kernel_energy = fmaxf(kernel_energy, 0.0f);
    float normalization = sqrtf(patch_energy * kernel_energy);
    float response = normalization > 1e-12f
                         ? hypotf(response_even, response_odd) / normalization
                         : 0.0f;

    vec_val[ri] = response;

    if (response > val_max) {
      val_max = response;
      idx_max = ri;
    }
    if (response < val_min) {
      val_min = response;
    }
  }

  float theta_max = PI * static_cast<float>(idx_max) / static_cast<float>(rotate_res);

  // Circular weighted variance
  float variance = 0.0f;
  float range = val_max - val_min;
  if (val_max < params.min_response || range <= 1e-12f) {
    d_OrientMap[py * pitchOrient + px] = 0.0f;
    d_VarianceMap[py * pitchVariance + px] = 0.0f;
    return;
  }
  for (int ri = 0; ri < rotate_res; ++ri) {
    float theta_i = PI * static_cast<float>(ri) / static_cast<float>(rotate_res);
    float theta_diff = angleDistance(theta_i, theta_max) / (PI / 2.0f);
    float w = (vec_val[ri] - val_min) / range;
    variance += theta_diff * theta_diff * w;
  }
  variance /= rotate_res;

  d_OrientMap[py * pitchOrient + px] = theta_max;
  d_VarianceMap[py * pitchVariance + px] = variance;
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

void ComputeGaborOrientation(float* d_OrientMap, float* d_VarianceMap, float* d_GrayMap,
                             unsigned int width, unsigned int height, int rotate_res,
                             GaborParams params, float* d_maskMap) {
  dim3 block(T_PER_BLOCK, T_PER_BLOCK);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  GaborOrientationKernel<<<grid, block>>>(d_OrientMap, d_VarianceMap, d_GrayMap, width,
                                          height, rotate_res, params, d_maskMap);
  {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      printf("CUDA kernel error in gabor: %s\n", cudaGetErrorString(e));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

void ComputeGaborOrientationTexture(float* d_OrientMap, unsigned int pitchOrient,
                                    float* d_VarianceMap, unsigned int pitchVariance,
                                    cudaTextureObject_t d_GrayMap, unsigned int width,
                                    unsigned int height, int rotate_res,
                                    GaborParams params, float* d_maskMap) {
  dim3 block(T_PER_BLOCK, T_PER_BLOCK);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  GaborOrientationTextureKernel<<<grid, block>>>(
      d_OrientMap, pitchOrient, d_VarianceMap, pitchVariance, d_GrayMap, width, height,
      rotate_res, params, d_maskMap);
  {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
      printf("CUDA kernel error in gabor: %s\n", cudaGetErrorString(e));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}
