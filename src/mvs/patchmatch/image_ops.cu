// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/cuda_check.h"
#include "common/cuda_math.h"
#include "common/cuda_vec_ops.h"
#include "common/types.h"
#include "mvs/patchmatch/constants.h"

// ============================================================================
// 5x5 Gaussian kernel weights (sum = 256)
// ============================================================================
__constant__ unsigned char g_gauss5x5[5][5] = {{1, 4, 6, 4, 1},
                                               {4, 16, 24, 16, 4},
                                               {6, 24, 36, 24, 6},
                                               {4, 16, 24, 16, 4},
                                               {1, 4, 6, 4, 1}};

// ============================================================================
// ResampleImage: bilinear interpolation for downsampling float maps
// ============================================================================
__global__ void ResampleImageKernel(float* dst, unsigned int dst_w, unsigned int dst_h,
                                    float* src, unsigned int src_w, unsigned int src_h,
                                    float* mask) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h)
    return;

  // Coordinate mapping: edge-to-edge (N-1)/(M-1)
  const float scaleW = (float) (src_w - 1) / (float) (dst_w - 1);
  const float scaleH = (float) (src_h - 1) / (float) (dst_h - 1);
  float fx = x * scaleW;
  float fy = y * scaleH;

  int x0 = (int) fx;
  int y0 = (int) fy;
  int x1 = min(x0 + 1, (int) src_w - 1);
  int y1 = min(y0 + 1, (int) src_h - 1);

  float ax = fx - (float) x0;
  float ay = fy - (float) y0;

  if (mask) {
    float m00 = mask[y0 * src_w + x0];
    float m10 = mask[y0 * src_w + x1];
    float m01 = mask[y1 * src_w + x0];
    float m11 = mask[y1 * src_w + x1];
    float mval = (1.0f - ax) * (1.0f - ay) * m00 + ax * (1.0f - ay) * m10 +
                 (1.0f - ax) * ay * m01 + ax * ay * m11;
    if (mval < 0.5f) {
      dst[y * dst_w + x] = MINF;
      return;
    }
  }

  float v00 = src[y0 * src_w + x0];
  float v10 = src[y0 * src_w + x1];
  float v01 = src[y1 * src_w + x0];
  float v11 = src[y1 * src_w + x1];

  float val = (1.0f - ax) * (1.0f - ay) * v00 + ax * (1.0f - ay) * v10 +
              (1.0f - ax) * ay * v01 + ax * ay * v11;
  dst[y * dst_w + x] = val;
}

void ResampleImage(float* dst, unsigned int dst_w, unsigned int dst_h, float* src,
                   unsigned int src_w, unsigned int src_h, float* mask) {
  dim3 block(T_PER_BLOCK, T_PER_BLOCK);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  ResampleImageKernel<<<grid, block>>>(dst, dst_w, dst_h, src, src_w, src_h, mask);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// ResampleImageTexture: bilinear from texture object
// ============================================================================
__global__ void ResampleImageTextureKernel(float* dst, unsigned int dst_w,
                                           unsigned int dst_h, unsigned int pitch,
                                           cudaTextureObject_t src, unsigned int src_w,
                                           unsigned int src_h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h)
    return;

  // Coordinate mapping: edge-to-edge, then use texture bilinear
  const float scaleW = (float) (src_w - 1) / (float) (dst_w - 1);
  const float scaleH = (float) (src_h - 1) / (float) (dst_h - 1);
  float fx = x * scaleW;
  float fy = y * scaleH;

  if ((unsigned int) fx < src_w && (unsigned int) fy < src_h) {
    // tex2D with +0.5 for half-pixel offset (texture coords are center-based)
    dst[y * pitch + x] = tex2D<float>(src, fx + 0.5f, fy + 0.5f);
  }
}

void ResampleImageTexture(float* dst, unsigned int dst_w, unsigned int dst_h,
                          unsigned int pitch, cudaTextureObject_t src,
                          unsigned int src_w, unsigned int src_h) {
  dim3 block(T_PER_BLOCK, T_PER_BLOCK);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  ResampleImageTextureKernel<<<grid, block>>>(dst, dst_w, dst_h, pitch, src, src_w,
                                              src_h);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// ResampleLine3DMap: bilinear for Line3D (interpolate p and v, renormalize v)
// ============================================================================
__global__ void ResampleLine3DMapKernel(Line3D* dst, unsigned int dst_w,
                                        unsigned int dst_h, Line3D* src,
                                        unsigned int src_w, unsigned int src_h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h)
    return;

  // Nearest neighbor interpolation for Line3D (not bilinear)
  const float scaleW = (float) (src_w - 1) / (float) (dst_w - 1);
  const float scaleH = (float) (src_h - 1) / (float) (dst_h - 1);
  unsigned int xInput = (unsigned int) (x * scaleW + 0.5f);
  unsigned int yInput = (unsigned int) (y * scaleH + 0.5f);

  if (xInput < src_w && yInput < src_h) {
    dst[y * dst_w + x] = src[yInput * src_w + xInput];
  }
}

void ResampleLine3DMap(Line3D* dst, unsigned int dst_w, unsigned int dst_h, Line3D* src,
                       unsigned int src_w, unsigned int src_h) {
  dim3 block(T_PER_BLOCK, T_PER_BLOCK);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
  ResampleLine3DMapKernel<<<grid, block>>>(dst, dst_w, dst_h, src, src_w, src_h);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// GaussianSmooth: 5x5 Gaussian blur from device array
// ============================================================================
__global__ void GaussianSmoothKernel(float* dst, float* src, unsigned int width,
                                     unsigned int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  float sum = 0.0f;
  float wsum = 0.0f;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      int sx = min(max(x + dx, 0), (int) width - 1);
      int sy = min(max(y + dy, 0), (int) height - 1);
      float w = (float) g_gauss5x5[dy + 2][dx + 2];
      sum += w * src[sy * width + sx];
      wsum += w;
    }
  }
  dst[y * width + x] = sum / wsum;
}

void GaussianSmooth(float* dst, float* src, unsigned int width, unsigned int height) {
  dim3 block(T_PER_BLOCK, T_PER_BLOCK);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  GaussianSmoothKernel<<<grid, block>>>(dst, src, width, height);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// GaussianSmoothTexture: 5x5 Gaussian blur from texture object
// ============================================================================
__global__ void GaussianSmoothTextureKernel(float* dst, cudaTextureObject_t src,
                                            unsigned int width, unsigned int height,
                                            unsigned int pitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  float sum = 0.0f;
  float wsum = 0.0f;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      float sx = (float) (x + dx) + 0.5f;
      float sy = (float) (y + dy) + 0.5f;
      float w = (float) g_gauss5x5[dy + 2][dx + 2];
      sum += w * tex2D<float>(src, sx, sy);
      wsum += w;
    }
  }
  dst[y * pitch + x] = sum / wsum;
}

void GaussianSmoothTexture(float* dst, cudaTextureObject_t src, unsigned int width,
                           unsigned int height, unsigned int pitch) {
  dim3 block(T_PER_BLOCK, T_PER_BLOCK);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  GaussianSmoothTextureKernel<<<grid, block>>>(dst, src, width, height, pitch);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// InitializeLine3D: fill array with a constant Line3D value
// ============================================================================
__global__ void InitializeLine3DKernel(Line3D* d_output, Line3D val, unsigned int width,
                                       unsigned int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  d_output[y * width + x] = val;
}

void InitializeLine3D(Line3D* d_output, Line3D val, unsigned int width,
                      unsigned int height) {
  dim3 block(T_PER_BLOCK, T_PER_BLOCK);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  InitializeLine3DKernel<<<grid, block>>>(d_output, val, width, height);
  CUDA_CHECK(cudaDeviceSynchronize());
}
