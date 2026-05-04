// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// Throws std::runtime_error on CUDA failure (safe in multi-threaded context)
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      char msg[512];                                                            \
      snprintf(msg, sizeof(msg), "CUDA error at %s:%d: %s", __FILE__, __LINE__, \
               cudaGetErrorString(err));                                        \
      fprintf(stderr, "%s\n", msg);                                             \
      throw std::runtime_error(msg);                                            \
    }                                                                           \
  } while (0)

#define CUDA_CHECK_LAST()                                                              \
  do {                                                                                 \
    cudaError_t err = cudaGetLastError();                                              \
    if (err != cudaSuccess) {                                                          \
      char msg[512];                                                                   \
      snprintf(msg, sizeof(msg), "CUDA kernel error at %s:%d: %s", __FILE__, __LINE__, \
               cudaGetErrorString(err));                                               \
      fprintf(stderr, "%s\n", msg);                                                    \
      throw std::runtime_error(msg);                                                   \
    }                                                                                  \
  } while (0)
