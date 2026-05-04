// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include <curand_kernel.h>

#include "common/cuda_check.h"
#include "common/cuda_math.h"
#include "common/cuda_vec_ops.h"
#include "common/logger.h"
#include "common/types.h"
#include "mvs/patchmatch/constants.h"
#include "mvs/patchmatch/cost.cuh"
#include "mvs/patchmatch/geometry.cuh"
#include "mvs/patchmatch/types.h"
#include "mvs/patchmatch/params.h"

#define PATCH_SIZE 16

__device__ inline void SpatialPropCandidateHair(const PatchMatchInput& input,
                                                PatchMatchState& state,
                                                const int2& pos_ref,
                                                const int2& pos_new,
                                                const PatchMatchParams& parameters) {
  const int width = input.width;
  const int height = input.height;
  if (pos_new.x < 0 || pos_new.x >= width || pos_new.y < 0 || pos_new.y >= height)
    return;

  const int ctrind_ref = pos_ref.y * width + pos_ref.x;
  float cost_ref = state.d_cost_total[ctrind_ref];

  Line3D line_new = state.d_line3D[pos_new.y * width + pos_new.x];
  Line3D ray = GetRayAtPos(make_float2((float) pos_ref.x, (float) pos_ref.y),
                           input.invIntrinsic);
  line_new.p = LineLineIntersectPA(ray, line_new);

  float depth_new = line_new.p.z;
  if (depth_new < parameters.depthMin || depth_new > parameters.depthMax)
    return;
  line_new.p.z = fmaxf(fminf(line_new.p.z, parameters.depthMax), parameters.depthMin);

  float2 cost = MultiViewLineCost(input, state, pos_ref, line_new, parameters);
  float cost_new =
      parameters.hair_alpha * cost.x + (1.0f - parameters.hair_alpha) * cost.y;

  if (cost_new < cost_ref) {
    state.d_line3D[ctrind_ref] = line_new;
    state.d_cost_total[ctrind_ref] = cost_new;
    state.d_cost_color[ctrind_ref] = cost.x;
    state.d_cost_orient[ctrind_ref] = cost.y;
  }
}

__device__ inline void LineRefinementHair(const PatchMatchInput& input,
                                          PatchMatchState& state, const int2& pos,
                                          const PatchMatchParams& parameters) {
  const int ctrind = pos.y * input.width + pos.x;
  if (pos.x >= input.width || pos.y >= input.height)
    return;

  curandState& localState = state.d_cs[ctrind];
  float cost_now = state.d_cost_total[ctrind];
  Line3D line_now = state.d_line3D[ctrind];
  float delta_orient = parameters.hair_delta_orient;
  float delta_depth = parameters.hair_delta_depth;
  Line3D line_new;

  const int maxiter = 9;
  for (int i = 0; i < maxiter; i++) {
    GetRndLine3DCu(line_now, line_new, pos, input.invIntrinsic, delta_depth,
                   delta_orient, parameters.depthMin, parameters.depthMax, &localState);

    float2 cost = MultiViewLineCost(input, state, pos, line_new, parameters);
    float cost_new =
        parameters.hair_alpha * cost.x + (1.0f - parameters.hair_alpha) * cost.y;

    if (cost_new < cost_now) {
      state.d_line3D[ctrind] = line_new;
      state.d_cost_total[ctrind] = cost_new;
      state.d_cost_color[ctrind] = cost.x;
      state.d_cost_orient[ctrind] = cost.y;
      cost_now = cost_new;
      line_now = line_new;
    }

    if (i % 3 == 2) {
      delta_orient /= 2.0f;
      delta_depth /= 2.0f;
    }
  }
}

__global__ void CostComputeKernel(PatchMatchInput input, PatchMatchState state,
                                  PatchMatchParams parameters) {
  const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                           blockIdx.y * blockDim.y + threadIdx.y);
  if (p.x >= input.width || p.y >= input.height)
    return;

  const int ctrind = p.y * input.width + p.x;
  curand_init(clock64(), p.y, p.x, &(state.d_cs[ctrind]));
  curandState& localState = state.d_cs[ctrind];
  Line3D& line_now = state.d_line3D[ctrind];

  if (length(line_now.p) == 0.0f && length(line_now.v) == 0.0f) {
    RndUnitVectorOnHemisphere(line_now.v, make_float3(0, 0, 1), &localState);
    float depth_now =
        CurandBetween(&localState, parameters.depthMin, parameters.depthMax);
    line_now.p = GetPtCu(p, depth_now, input.invIntrinsic);
  }

  float2 cost = MultiViewLineCost(input, state, p, line_now, parameters);
  state.d_cost_color[ctrind] = cost.x;
  state.d_cost_orient[ctrind] = cost.y;
  state.d_cost_total[ctrind] =
      parameters.hair_alpha * cost.x + (1.0f - parameters.hair_alpha) * cost.y;
}

__global__ void PropagationKernelBlack(PatchMatchInput input, PatchMatchState state,
                                       PatchMatchParams parameters) {
  const int width = input.width;
  const int height = input.height;
  const float prop_radius = parameters.hair_spatial_prop_radius;
  int prop_radius_int = (int) ceilf(prop_radius);
  if (prop_radius_int % 2 == 0)
    prop_radius_int++;

  int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                     blockIdx.y * blockDim.y + threadIdx.y);
  if (threadIdx.x % 2 == 0)
    p.y = p.y * 2;
  else
    p.y = p.y * 2 + 1;

  if (p.x < width && p.y < height) {
    for (int iy = -prop_radius_int; iy <= prop_radius_int; iy++)
      for (int ix = -prop_radius_int; ix <= prop_radius_int; ix++) {
        if ((abs(ix) + abs(iy)) % 2 == 0)
          continue;
        if ((float) (ix * ix + iy * iy) > prop_radius * prop_radius + 1e-9f)
          continue;
        SpatialPropCandidateHair(input, state, p, make_int2(p.x + ix, p.y + iy),
                                 parameters);
      }
    LineRefinementHair(input, state, p, parameters);
  }
}

__global__ void PropagationKernelRed(PatchMatchInput input, PatchMatchState state,
                                     PatchMatchParams parameters) {
  const int width = input.width;
  const int height = input.height;
  const float prop_radius = parameters.hair_spatial_prop_radius;
  int prop_radius_int = (int) ceilf(prop_radius);
  if (prop_radius_int % 2 == 0)
    prop_radius_int++;

  int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                     blockIdx.y * blockDim.y + threadIdx.y);
  if (threadIdx.x % 2 == 0)
    p.y = p.y * 2 + 1;
  else
    p.y = p.y * 2;

  if (p.x < width && p.y < height) {
    for (int iy = -prop_radius_int; iy <= prop_radius_int; iy++)
      for (int ix = -prop_radius_int; ix <= prop_radius_int; ix++) {
        if ((abs(ix) + abs(iy)) % 2 == 0)
          continue;
        if ((float) (ix * ix + iy * iy) > prop_radius * prop_radius + 1e-9f)
          continue;
        SpatialPropCandidateHair(input, state, p, make_int2(p.x + ix, p.y + iy),
                                 parameters);
      }
    LineRefinementHair(input, state, p, parameters);
  }
}

void RunPatchMatch(PatchMatchInput& input, PatchMatchState& state,
                   PatchMatchParams& params) {
  dim3 blockSize(PATCH_SIZE, PATCH_SIZE);
  dim3 gridSize((input.width + PATCH_SIZE - 1) / PATCH_SIZE,
                (input.height + PATCH_SIZE - 1) / PATCH_SIZE);

  CostComputeKernel<<<gridSize, blockSize>>>(input, state, params);
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 gridSizeHalf((input.width + PATCH_SIZE - 1) / PATCH_SIZE,
                    ((input.height / 2) + PATCH_SIZE - 1) / PATCH_SIZE);

  for (int it = 0; it < params.maxIt; it++) {
    PropagationKernelBlack<<<gridSizeHalf, blockSize>>>(input, state, params);
    CUDA_CHECK(cudaDeviceSynchronize());
    PropagationKernelRed<<<gridSizeHalf, blockSize>>>(input, state, params);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}
