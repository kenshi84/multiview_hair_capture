// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <cuda_runtime.h>

#include "common/types.h"

// Compute per-pixel 2D hair orientation and circular variance from grayscale
// image using Gabor filter bank. d_maskMap may be nullptr (no masking).
void ComputeGaborOrientation(float* d_OrientMap, float* d_VarianceMap, float* d_GrayMap,
                             unsigned int width, unsigned int height, int rotate_res,
                             GaborParams params, float* d_maskMap);

// Same as above but reads grayscale from a cudaTextureObject_t.
// Orientation and variance maps use pitched allocations (pitches in bytes).
void ComputeGaborOrientationTexture(float* d_OrientMap, unsigned int pitchOrient,
                                    float* d_VarianceMap, unsigned int pitchVariance,
                                    cudaTextureObject_t d_GrayMap, unsigned int width,
                                    unsigned int height, int rotate_res,
                                    GaborParams params, float* d_maskMap);
