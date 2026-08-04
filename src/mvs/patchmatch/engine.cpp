// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "mvs/patchmatch/engine.h"

#include <cstring>

#include "common/cuda_check.h"
#include "common/cuda_math.h"
#include "common/logger.h"

// Declared in propagation.cu
void RunPatchMatch(PatchMatchInput& input, PatchMatchState& state,
                   PatchMatchParams& params);

PatchMatchEngine::PatchMatchEngine(unsigned int width, unsigned int height,
                                   unsigned int numnei)
    : input_{}, state_{}, width_(width), height_(height), numnei_(numnei) {
  input_.width = width;
  input_.height = height;
  input_.numnei = numnei;

  unsigned int npixels = width * height;

  // Allocate input buffers
  CUDA_CHECK(cudaMalloc(&input_.d_calibparams, numnei * sizeof(CameraParams)));
  CUDA_CHECK(cudaMalloc(&input_.d_neiTexMapsObj, numnei * sizeof(cudaTextureObject_t)));
  CUDA_CHECK(
      cudaMalloc(&input_.d_neiOrientTexMapsObj, numnei * sizeof(cudaTextureObject_t)));
  CUDA_CHECK(cudaMalloc(&input_.d_neiOrientVarianceTexMapsObj,
                        numnei * sizeof(cudaTextureObject_t)));

  // Allocate state buffers
  CUDA_CHECK(cudaMalloc(&state_.d_cs, npixels * sizeof(curandState)));
  CUDA_CHECK(cudaMalloc(&state_.d_cost_orient, npixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state_.d_cost_color, npixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state_.d_cost_total, npixels * sizeof(float)));

  CUDA_CHECK(cudaMemset(state_.d_cost_orient, 0, npixels * sizeof(float)));
  CUDA_CHECK(cudaMemset(state_.d_cost_color, 0, npixels * sizeof(float)));
  CUDA_CHECK(cudaMemset(state_.d_cost_total, 0, npixels * sizeof(float)));
}

PatchMatchEngine::~PatchMatchEngine() {
  // Free input buffers
  if (input_.d_calibparams)
    cudaFree(input_.d_calibparams);
  if (input_.d_neiTexMapsObj)
    cudaFree(input_.d_neiTexMapsObj);
  if (input_.d_neiOrientTexMapsObj)
    cudaFree(input_.d_neiOrientTexMapsObj);
  if (input_.d_neiOrientVarianceTexMapsObj)
    cudaFree(input_.d_neiOrientVarianceTexMapsObj);

  // Free state buffers
  if (state_.d_cs)
    cudaFree(state_.d_cs);
  // d_line3D is supplied and owned by HierarchicalPatchMatch for each level.
  if (state_.d_cost_orient)
    cudaFree(state_.d_cost_orient);
  if (state_.d_cost_color)
    cudaFree(state_.d_cost_color);
  if (state_.d_cost_total)
    cudaFree(state_.d_cost_total);
}

void PatchMatchEngine::InitParameters(const float intrinsic[9],
                                      const std::vector<GpuCamera>& nei_cams,
                                      const PatchMatchParams& params,
                                      unsigned int width, unsigned int height) {
  width_ = width;
  height_ = height;
  numnei_ = static_cast<unsigned int>(nei_cams.size());

  input_.width = width;
  input_.height = height;
  input_.numnei = numnei_;

  // Set intrinsic and inverse intrinsic
  float3x3 K(intrinsic);
  float3x3 invK = K.getInverse();
  input_.matIntrinsic = K;
  input_.invIntrinsic = invK;

  // Copy camera params to GPU
  std::vector<CameraParams> cam_params(numnei_);
  for (unsigned int i = 0; i < numnei_; i++) {
    const GpuCamera& gpu_cam = nei_cams[i];
    cam_params[i].Kmat = float3x3(gpu_cam.K);
    cam_params[i].Rmat = float3x3(gpu_cam.R);
    cam_params[i].tvec = make_float3(gpu_cam.t[0], gpu_cam.t[1], gpu_cam.t[2]);
  }

  CUDA_CHECK(cudaMemcpy(input_.d_calibparams, cam_params.data(),
                        numnei_ * sizeof(CameraParams), cudaMemcpyHostToDevice));
}

void PatchMatchEngine::RunHair(PatchMatchInput& input, PatchMatchState& state,
                               PatchMatchParams& params) {
  LOG_DEBUG("PatchMatchEngine::RunHair w=%d h=%d numnei=%d maxIt=%d", input.width,
            input.height, input.numnei, params.maxIt);
  RunPatchMatch(input, state, params);
  LOG_DEBUG("PatchMatchEngine::RunHair completed");
}
