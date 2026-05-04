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

#include "common/cuda_math.h"
#include "common/types.h"

void ResampleImage(float* dst, unsigned int dst_w, unsigned int dst_h, float* src,
                   unsigned int src_w, unsigned int src_h, float* mask);

void ResampleImageTexture(float* dst, unsigned int dst_w, unsigned int dst_h,
                          unsigned int pitch, cudaTextureObject_t src,
                          unsigned int src_w, unsigned int src_h);

void ResampleLine3DMap(Line3D* dst, unsigned int dst_w, unsigned int dst_h, Line3D* src,
                       unsigned int src_w, unsigned int src_h);

void GaussianSmooth(float* dst, float* src, unsigned int width, unsigned int height);

void GaussianSmoothTexture(float* dst, cudaTextureObject_t src, unsigned int width,
                           unsigned int height, unsigned int pitch);

void InitializeLine3D(Line3D* d_output, Line3D val, unsigned int width,
                      unsigned int height);
