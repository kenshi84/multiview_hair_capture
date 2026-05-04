// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// PatchMatch solver input/state/parameter structs for GPU kernels

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "common/cuda_math.h"
#include "common/types.h"

#define MAX_NEIGHBOR_VIEWS 4

// Per-neighbor camera parameters (GPU-side)
struct CameraParams {
  float3x3 Kmat;
  float3x3 Rmat;
  float3 tvec;
};

// GPU input buffers
struct PatchMatchInput {
  unsigned int width;
  unsigned int height;

  float* d_refGrayMapFloat;
  cudaTextureObject_t* d_neiTexMapsObj;

  unsigned char* d_refMaskMapUchar;
  unsigned char** d_neiMaskMapsUchar;

  float* d_refOrientMapFloat;
  cudaTextureObject_t* d_neiOrientTexMapsObj;

  float* d_refOrientVarianceMapFloat;
  cudaTextureObject_t* d_neiOrientVarianceTexMapsObj;

  CameraParams* d_calibparams;

  int numnei;

  float3x3 invIntrinsic;
  float3x3 matIntrinsic;
};

// GPU state buffers (mutable during solve)
struct PatchMatchState {
  curandState* d_cs;
  Line3D* d_line3D;
  float* d_cost_orient;
  float* d_cost_color;
  float* d_cost_total;
};
