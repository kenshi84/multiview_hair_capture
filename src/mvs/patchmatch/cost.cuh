// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// Device-only hair cost function for Line-based PatchMatch MVS.

#pragma once

#include "common/cuda_math.h"
#include "common/cuda_vec_ops.h"
#include "common/types.h"
#include "mvs/patchmatch/constants.h"
#include "mvs/patchmatch/geometry.cuh"
#include "mvs/patchmatch/types.h"
#include "mvs/patchmatch/params.h"

#define MAXCOST_HAIR_COLOR 1.0f
#define MAXCOST_HAIR_ORIENT 1.0f
#define HAIR_LPMVS_KAPPA 41

__device__ inline bool IsInsideMask(const unsigned char* mask, const float2& pos,
                                    int width, int height) {
  if (mask == nullptr || !IsPositionValid(pos, width, height))
    return false;
  int x = static_cast<int>(roundf(pos.x));
  int y = static_cast<int>(roundf(pos.y));
  return mask[y * width + x] != 0;
}

__device__ inline float2 MultiViewLineCost(const PatchMatchInput& input,
                                           const PatchMatchState& state,
                                           const int2& pos, const Line3D& line_now,
                                           const PatchMatchParams& parameters) {
  const int ctrind = pos.y * input.width + pos.x;
  const float* d_ref_gray = input.d_refGrayMapFloat;
  const float* d_ref_orient = input.d_refOrientMapFloat;
  const float* d_ref_orient_variance = input.d_refOrientVarianceMapFloat;
  const cudaTextureObject_t* d_neiGrayTex = input.d_neiTexMapsObj;
  const cudaTextureObject_t* d_neiOrientTex = input.d_neiOrientTexMapsObj;
  const cudaTextureObject_t* d_neiOrientVarianceTex =
      input.d_neiOrientVarianceTexMapsObj;
  const CameraParams* d_cameras = input.d_calibparams;
  const int numnei = input.numnei;
  const float3x3 invK = input.invIntrinsic;
  const int width = input.width;
  const int height = input.height;
  const float sample_radius = parameters.hair_pt_sample_radius;
  int sample_kappa = parameters.hair_pt_sample_kappa;
  const float alpha = parameters.hair_alpha;
  const int num_view_select = parameters.hair_num_view_select;
  const bool use_mask = parameters.bMaskInput;
  const int max_neighbor_views = MIN(numnei, MAX_NEIGHBOR_VIEWS);

  // Build a compact set of neighbor views eligible for cost evaluation. A
  // masked-out projection excludes only that neighbor; it must not invalidate
  // an otherwise supported hypothesis in every view.
  int eligible_views[MAX_NEIGHBOR_VIEWS];
  int eligible_view_count = 0;
  if (use_mask) {
    if (input.d_refMaskMapUchar == nullptr ||
        input.d_neiMaskMapsUchar == nullptr ||
        input.d_refMaskMapUchar[ctrind] == 0) {
      return make_float2(MAXCOST_HAIR_COLOR, MAXCOST_HAIR_ORIENT);
    }

    // Search the full configured neighbor list so a masked-out close view can
    // be replaced by a later valid view, up to the solver's fixed capacity.
    for (int view = 0;
         view < numnei && eligible_view_count < MAX_NEIGHBOR_VIEWS; view++) {
      const unsigned char* neighbor_mask = input.d_neiMaskMapsUchar[view];
      if (neighbor_mask == nullptr)
        continue;
      float2 pos_nei = GetCorrespondingPointInNeighbor(
          make_float2((float) pos.x, (float) pos.y), line_now, invK, d_cameras[view]);
      if (IsInsideMask(neighbor_mask, pos_nei, width, height))
        eligible_views[eligible_view_count++] = view;
    }

    int min_masked_views = parameters.hair_mask_min_neighbor_views;
    min_masked_views = MAX(1, MIN(min_masked_views, max_neighbor_views));
    if (eligible_view_count < min_masked_views) {
      return make_float2(MAXCOST_HAIR_COLOR, MAXCOST_HAIR_ORIENT);
    }
  } else {
    for (int view = 0; view < max_neighbor_views; view++)
      eligible_views[eligible_view_count++] = view;
  }

  if (eligible_view_count == 0) {
    return make_float2(MAXCOST_HAIR_COLOR, MAXCOST_HAIR_ORIENT);
  }

  if (sample_kappa % 2 == 0)
    sample_kappa++;

  float vec_cost_total[MAX_NEIGHBOR_VIEWS];
  float vec_cost_color[MAX_NEIGHBOR_VIEWS];
  float vec_cost_orient[MAX_NEIGHBOR_VIEWS + 1];
  vec_cost_orient[0] = 0.0f;
  for (int i = 0; i < MAX_NEIGHBOR_VIEWS; i++) {
    vec_cost_total[i] = 0.0f;
    vec_cost_color[i] = 0.0f;
    vec_cost_orient[i + 1] = 0.0f;
  }

  matNxM<4, 4> line_now_mat = Line3DToLineMat(line_now);
  Line2D l_ref = Project3DLineTo2D(line_now_mat, input.matIntrinsic,
                                   float3x3::identity(), make_float3(0.0f));

  float2 vec_pos_ref[HAIR_LPMVS_KAPPA];
  Sample2DPointsOnLine(vec_pos_ref, pos, l_ref, sample_radius, HAIR_LPMVS_KAPPA);

  // Masks deliberately gate only the hypothesis center and neighbor-view
  // eligibility. Applying them to every support sample clips the NCC and
  // orientation windows at approximate mask boundaries, suppressing valid
  // strands whose centers are still inside the foreground region.

  // Color cost (NCC along line samples)
  int cost_color_count = 0;
  float cost_color = 0.0f;

  for (int eligible_index = 0; eligible_index < eligible_view_count;
       eligible_index++) {
    int view = eligible_views[eligible_index];
    float mean_left = 0.0f, std_left = 0.0f;
    float mean_right = 0.0f, std_right = 0.0f;
    float tmpcost = 0.0f;
    int sizesample = 0;

    for (int i = 0; i < sample_kappa; i++) {
      float2 pos_ref = vec_pos_ref[i];
      float2 pos_nei =
          GetCorrespondingPointInNeighbor(pos_ref, line_now, invK, d_cameras[view]);
      if (!IsPositionValid(pos_ref, width, height) ||
          !IsPositionValid(pos_nei, width, height))
        continue;

      float tmpintenref = GetIntensityBilinear(pos_ref, d_ref_gray, width, height);
      float tmpintennei =
          GetIntensityBilinearTex(pos_nei, d_neiGrayTex[view], width, height);

      mean_left += tmpintenref;
      std_left += tmpintenref * tmpintenref;
      mean_right += tmpintennei;
      std_right += tmpintennei * tmpintennei;
      tmpcost += tmpintenref * tmpintennei;
      sizesample++;
    }

    if (sizesample == 0) {
      tmpcost = MAXCOST_HAIR_COLOR;
    } else {
      mean_left /= (float) sizesample;
      mean_right /= (float) sizesample;
      tmpcost -= sizesample * mean_left * mean_right;
      float std_total_sqrt = sqrtf((std_left - sizesample * mean_left * mean_left) *
                                   (std_right - sizesample * mean_right * mean_right));
      if (std_total_sqrt == 0.0f || isnan(std_total_sqrt)) {
        tmpcost = MAXCOST_HAIR_COLOR;
      } else {
        tmpcost /= std_total_sqrt;
        tmpcost = (1.0f - tmpcost) * 0.5f;
        tmpcost = MIN(MAXCOST_HAIR_COLOR, tmpcost);
      }
    }
    vec_cost_color[cost_color_count++] = tmpcost;
  }

  // Orientation cost
  int cost_orient_count = 0;
  float cost_orient = 0.0f;

  // Reference view
  {
    float tmpcost_ref = 0.0f;
    float tmpweight_ref = 0.0f;
    float lineorient_ref = Get2DLineOrient(l_ref);

    for (int i = 0; i < sample_kappa; i++) {
      float2 pos_ref = vec_pos_ref[i];
      if (!IsPositionValid(pos_ref, width, height))
        continue;

      float tmporient = GetOrientationNearest(pos_ref, d_ref_orient, width, height);
      float tmpvariance =
          GetIntensityNearest(pos_ref, d_ref_orient_variance, width, height);
      float tmpconf = ConvertOrientVarianceToConfidence2(tmpvariance);

      float thetadiff = AngleDifference(lineorient_ref, tmporient);
      thetadiff /= (CUDART_PI_F * 0.5f);
      tmpcost_ref += tmpconf * thetadiff;
      tmpweight_ref += tmpconf;
    }

    if (tmpweight_ref == 0.0f) {
      tmpcost_ref = MAXCOST_HAIR_ORIENT;
    } else {
      tmpcost_ref /= tmpweight_ref;
    }
    vec_cost_orient[cost_orient_count++] = tmpcost_ref;
  }

  // Neighbor views
  for (int eligible_index = 0; eligible_index < eligible_view_count;
       eligible_index++) {
    int view = eligible_views[eligible_index];
    float tmpcost_nei = 0.0f;
    float tmpweight_nei = 0.0f;

    Line2D l_nei = Project3DLineTo2D(line_now_mat, d_cameras[view].Kmat,
                                     d_cameras[view].Rmat, d_cameras[view].tvec);
    float lineorient_nei = Get2DLineOrient(l_nei);

    for (int i = 0; i < sample_kappa; i++) {
      float2 pos_ref = vec_pos_ref[i];
      float2 pos_nei =
          GetCorrespondingPointInNeighbor(pos_ref, line_now, invK, d_cameras[view]);
      if (!IsPositionValid(pos_ref, width, height) ||
          !IsPositionValid(pos_nei, width, height))
        continue;

      float tmporient =
          GetOrientationNearestTex(pos_nei, d_neiOrientTex[view], width, height);
      float tmpvariance =
          GetIntensityNearestTex(pos_nei, d_neiOrientVarianceTex[view], width, height);
      float tmpconf = ConvertOrientVarianceToConfidence2(tmpvariance);

      float thetadiff = AngleDifference(lineorient_nei, tmporient);
      thetadiff /= (CUDART_PI_F * 0.5f);
      tmpcost_nei += tmpconf * thetadiff;
      tmpweight_nei += tmpconf;
    }

    if (tmpweight_nei == 0.0f) {
      tmpcost_nei = MAXCOST_HAIR_ORIENT;
    } else {
      tmpcost_nei /= tmpweight_nei;
    }
    vec_cost_orient[cost_orient_count++] = tmpcost_nei;
  }

  // View selection
  if (cost_color_count == 0 || cost_orient_count == 0) {
    cost_color = MAXCOST_HAIR_COLOR;
    cost_orient = MAXCOST_HAIR_ORIENT;
  } else {
    for (int i = 0; i < eligible_view_count; i++) {
      vec_cost_total[i] =
          alpha * vec_cost_color[i] + (1.0f - alpha) * vec_cost_orient[i + 1];
    }

    int selected_view_count = MIN(num_view_select, eligible_view_count);
    int view_count = 0;
    for (int i = 0; i < eligible_view_count; i++) {
      int order = GetOrder(vec_cost_total, eligible_view_count, i);
      if (order < selected_view_count) {
        view_count++;
        cost_color += vec_cost_color[i];
        cost_orient += vec_cost_orient[i + 1];
      }
    }
    if (view_count == 0) {
      cost_color = MAXCOST_HAIR_COLOR;
      cost_orient = MAXCOST_HAIR_ORIENT;
    } else {
      cost_color /= (float) view_count;
      cost_orient += (float) view_count * vec_cost_orient[0];
      cost_orient /= (float) (2 * view_count);
    }
  }

  return make_float2(cost_color, cost_orient);
}
