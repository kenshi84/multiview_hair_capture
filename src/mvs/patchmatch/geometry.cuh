// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// Device-only helper functions for Plucker line representation and ray-line
// geometry used by the Line-based PatchMatch MVS solver.

#pragma once

#include <curand_kernel.h>

#include "common/cuda_math.h"
#include "common/cuda_vec_ops.h"
#include "common/types.h"
#include "mvs/patchmatch/constants.h"
#include "mvs/patchmatch/types.h"

// ============================================================================
// 2D line representation: ax + by + c = 0
// ============================================================================
struct Line2D {
  float a, b, c;
};

// ============================================================================
// Random number helpers
// ============================================================================
__device__ inline float CurandBetween(curandState* cs, float lo, float hi) {
  return curand_uniform(cs) * (hi - lo) + lo;
}

// ============================================================================
// Marsaglia sphere point picking for random unit vector on hemisphere
// ============================================================================
__device__ inline void RndUnitVectorSphereMarsaglia(float4* v, curandState* cs) {
  float x = 1.0f, y = 1.0f, sum = 2.0f;
  while (sum >= 1.0f) {
    x = CurandBetween(cs, -1.0f, 1.0f);
    y = CurandBetween(cs, -1.0f, 1.0f);
    sum = x * x + y * y;
  }
  float sq = sqrtf(1.0f - sum);
  v->x = 2.0f * x * sq;
  v->y = 2.0f * y * sq;
  v->z = 1.0f - 2.0f * sum;
}

// Flip vector to the hemisphere OPPOSITE to viewVector (dot > 0 => negate)
__device__ inline float VecOnHemisphere(float3& v, const float3& viewVector) {
  float dp = v.x * viewVector.x + v.y * viewVector.y + v.z * viewVector.z;
  if (dp > 0.0f) {
    v.x = -v.x;
    v.y = -v.y;
    v.z = -v.z;
  }
  return dp;
}

__device__ inline float VecOnHemisphere(float4* v, const float3& viewVector) {
  float dp = v->x * viewVector.x + v->y * viewVector.y + v->z * viewVector.z;
  if (dp > 0.0f) {
    v->x = -v->x;
    v->y = -v->y;
    v->z = -v->z;
  }
  return dp;
}

__device__ inline void RndUnitVectorOnHemisphere(float3& v, const float3& viewVector,
                                                 curandState* cs) {
  float4 tmp = make_float4(v.x, v.y, v.z, 0.0f);
  RndUnitVectorSphereMarsaglia(&tmp, cs);
  VecOnHemisphere(&tmp, viewVector);
  v.x = tmp.x;
  v.y = tmp.y;
  v.z = tmp.z;
}

// ============================================================================
// Normalize float3 in-place
// ============================================================================
__device__ inline void NormalizeCu(float3& v) {
  float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  if (len > 1e-8f) {
    v.x /= len;
    v.y /= len;
    v.z /= len;
  }
}

// ============================================================================
// Point from pixel + depth using inverse intrinsic
// ============================================================================
__device__ inline float3 GetPtCu(const int2& p, float depth, const float3x3& invK) {
  return invK * make_float3(p.x * depth, p.y * depth, depth);
}

// ============================================================================
// Plucker line matrix (Hartley & Zisserman, p.70)
// ============================================================================
__device__ inline matNxM<4, 4> Line3DToLineMat(const Line3D& line) {
  float3 p = line.p;
  float3 v = line.v;
  float scale = 10.0f;

  matNxM<4, 1> A(make_float4(p.x, p.y, p.z, 1.0f));
  matNxM<4, 1> B(
      make_float4(p.x + scale * v.x, p.y + scale * v.y, p.z + scale * v.z, 1.0f));

  matNxM<4, 4> ABt = A * B.getTranspose();
  matNxM<4, 4> BAt = B * A.getTranspose();
  return ABt - BAt;
}

// ============================================================================
// Build K[R|t] projection matrix
// ============================================================================
__device__ inline matNxM<3, 4> KRtMat(const matNxM<3, 3>& K, const matNxM<3, 3>& R,
                                      const matNxM<3, 1>& t) {
  matNxM<3, 4> Rt;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      Rt(i, j) = R(i, j);
  for (int i = 0; i < 3; i++)
    Rt(i, 3) = t(i);
  return K * Rt;
}

// ============================================================================
// Extract 2D line vector from antisymmetric 3x3 matrix (H&Z Eq A4.5)
// ============================================================================
__device__ inline Line2D Line2DMatToVec(const matNxM<3, 3>& mat) {
  Line2D vec;
  vec.a = (mat(2, 1) - mat(1, 2)) * 0.5f;
  vec.b = (mat(0, 2) - mat(2, 0)) * 0.5f;
  vec.c = (mat(1, 0) - mat(0, 1)) * 0.5f;
  return vec;
}

// ============================================================================
// Project 3D line to 2D image line (H&Z Eq 8.2): l = [P*L*P^T]_vec
// ============================================================================
__device__ inline Line2D Project3DLineTo2D(const matNxM<4, 4>& L, const float3x3& intr,
                                           const float3x3& rmat, const float3& tvec) {
  matNxM<3, 3> K(intr);
  matNxM<3, 3> R(rmat);
  matNxM<3, 1> t(tvec);
  matNxM<3, 4> P = KRtMat(K, R, t);
  matNxM<3, 3> lx = (P * L) * P.getTranspose();
  return Line2DMatToVec(lx);
}

// ============================================================================
// 2D line orientation angle in [0, pi]
// ============================================================================
__device__ inline float Get2DLineOrient(const Line2D& line) {
  // Y-axis towards bottom of image
  float theta = atan2f(line.b, line.a);  // [-pi, pi]
  if (theta < 0)
    theta += CUDART_PI_F;    // [0, pi]
  theta = MAX(theta, 0.0f);  // [0, pi]
  return theta;
}

// ============================================================================
// Angle difference in [0, pi/2] (symmetric directions)
// ============================================================================
__device__ inline float AngleDifference(float theta1, float theta2) {
  float theta_diff1 = fabsf(theta1 - theta2);
  float theta_diff2 = theta1 < theta2 ? fabsf(theta1 + CUDART_PI_F - theta2)
                                      : fabsf(theta1 - CUDART_PI_F - theta2);
  return fminf(theta_diff1, theta_diff2);
}

// ============================================================================
// Get view ray at pixel position
// ============================================================================
__device__ inline Line3D GetRayAtPos(const float2& pos, const float3x3& invK) {
  float depth = 1000.0f;
  Line3D ray;
  ray.p = make_float3(0.0f, 0.0f, 0.0f);
  ray.v = invK * make_float3(pos.x * depth, pos.y * depth, depth);
  return ray;
}

// ============================================================================
// Line-line intersection: midpoint of closest approach
// ============================================================================
__device__ inline float3 LineLineIntersect(Line3D l1, Line3D l2) {
  const float eps = 1e-16f;
  float3 p1 = l1.p, p2 = l1.p + l1.v;
  float3 p3 = l2.p, p4 = l2.p + l2.v;
  float3 p13 = p1 - p3, p43 = p4 - p3, p21 = p2 - p1;

  float d1343 = p13.x * p43.x + p13.y * p43.y + p13.z * p43.z;
  float d4321 = p43.x * p21.x + p43.y * p21.y + p43.z * p21.z;
  float d1321 = p13.x * p21.x + p13.y * p21.y + p13.z * p21.z;
  float d4343 = p43.x * p43.x + p43.y * p43.y + p43.z * p43.z;
  float d2121 = p21.x * p21.x + p21.y * p21.y + p21.z * p21.z;

  float denom = d2121 * d4343 - d4321 * d4321;
  if (fabsf(denom) < eps)
    return make_float3(0.0f);
  float numer = d1343 * d4321 - d1321 * d4343;
  float mua = numer / denom;
  float mub = (d1343 + d4321 * mua) / d4343;

  float3 pa = p1 + mua * p21;
  float3 pb = p3 + mub * p43;
  return (pa + pb) * 0.5f;
}

// Line-line: return PA only (closest point on first line)
__device__ inline float3 LineLineIntersectPA(Line3D l1, Line3D l2) {
  const float eps = 1e-16f;
  float3 p1 = l1.p, p2 = l1.p + l1.v;
  float3 p3 = l2.p, p4 = l2.p + l2.v;
  float3 p13 = p1 - p3, p43 = p4 - p3, p21 = p2 - p1;

  float d1343 = p13.x * p43.x + p13.y * p43.y + p13.z * p43.z;
  float d4321 = p43.x * p21.x + p43.y * p21.y + p43.z * p21.z;
  float d1321 = p13.x * p21.x + p13.y * p21.y + p13.z * p21.z;
  float d4343 = p43.x * p43.x + p43.y * p43.y + p43.z * p43.z;
  float d2121 = p21.x * p21.x + p21.y * p21.y + p21.z * p21.z;

  float denom = d2121 * d4343 - d4321 * d4321;
  if (fabsf(denom) < eps)
    return make_float3(0.0f);
  float numer = d1343 * d4321 - d1321 * d4343;
  float mua = numer / denom;
  return p1 + mua * p21;
}

// ============================================================================
// 2D projection: P = K*(R*pt + t), return (px/pz, py/pz)
// ============================================================================
__device__ inline float2 Get2dProjectionPerspectiveCam(const float3x3& Kmat,
                                                       const float3x3& Rmat,
                                                       const float3& tvec,
                                                       const float3& vet) {
  float3 tmppos = Kmat * (Rmat * vet + tvec);
  tmppos.x /= tmppos.z;
  tmppos.y /= tmppos.z;
  return make_float2(tmppos.x, tmppos.y);
}

// ============================================================================
// Get corresponding point in neighbor view via line-ray intersection
// ============================================================================
__device__ inline float2 GetCorrespondingPointInNeighbor(const float2& pos_ref,
                                                         const Line3D& L,
                                                         const float3x3& invK,
                                                         const CameraParams& cam_nei) {
  Line3D ray = GetRayAtPos(pos_ref, invK);
  float3 pt_3D = LineLineIntersect(ray, L);
  return Get2dProjectionPerspectiveCam(cam_nei.Kmat, cam_nei.Rmat, cam_nei.tvec, pt_3D);
}

// ============================================================================
// Sample 2D points along a line centered at pos
// ============================================================================
__device__ inline void Sample2DPointsOnLine(float2* pts, const int2& pos,
                                            const Line2D& l_ref, float radius,
                                            int kappa) {
  int cnt = 0;
  float2 pt_ctr = make_float2((float) pos.x, (float) pos.y);
  pts[cnt++] = pt_ctr;

  int kappa_half = (kappa - 1) / 2;
  float unit_itv = radius / (float) kappa_half;

  for (int i = 0; i < kappa_half; i++) {
    float d = unit_itv * (i + 1);
    float dx, dy;
    if (l_ref.b != 0.0f) {
      float aob = l_ref.a / l_ref.b;
      dx = __fsqrt_rd(d * d / (1 + aob * aob));
      dy = -aob * dx;
    } else {
      dx = 0.0f;
      dy = d;
    }
    pts[cnt++] = pt_ctr + make_float2(dx, dy);
    pts[cnt++] = pt_ctr - make_float2(dx, dy);
  }
}

// ============================================================================
// Position validity check
// ============================================================================
__device__ inline bool IsPositionValid(float2 pos, int w, int h) {
  return pos.x > 0 && pos.x < (w - 1) && pos.y > 0 && pos.y < (h - 1);
}

// ============================================================================
// Bilinear intensity sampling from device array
// ============================================================================
__device__ inline float GetIntensityBilinear(float2 pos, const float* d_input,
                                             int width, int height) {
  int2 p00 = make_int2((int) floorf(pos.x), (int) floorf(pos.y));
  int2 p01 = make_int2(p00.x, p00.y + 1);
  int2 p10 = make_int2(p00.x + 1, p00.y);
  int2 p11 = make_int2(p00.x + 1, p00.y + 1);

  float alpha = pos.x - p00.x;
  float beta = pos.y - p00.y;

  float v00 = d_input[p00.y * width + p00.x];
  float v01 = d_input[p01.y * width + p01.x];
  float v10 = d_input[p10.y * width + p10.x];
  float v11 = d_input[p11.y * width + p11.x];

  return v00 * (1.0f - alpha) * (1.0f - beta) + v01 * (1.0f - alpha) * beta +
         v10 * alpha * (1.0f - beta) + v11 * alpha * beta;
}

// ============================================================================
// Bilinear intensity from texture
// ============================================================================
__device__ inline float GetIntensityBilinearTex(float2 pos, cudaTextureObject_t tex,
                                                int width, int height) {
  return tex2D<float>(tex, pos.x + 0.5f, pos.y + 0.5f);
}

// ============================================================================
// Nearest-neighbor intensity from device array / texture
// ============================================================================
__device__ inline float GetIntensityNearest(float2 pos, const float* d_input, int width,
                                            int height) {
  int2 p = make_int2((int) roundf(pos.x), (int) roundf(pos.y));
  return d_input[p.y * width + p.x];
}

__device__ inline float GetIntensityNearestTex(float2 pos, cudaTextureObject_t tex,
                                               int width, int height) {
  int2 p = make_int2((int) roundf(pos.x), (int) roundf(pos.y));
  return tex2D<float>(tex, p.x + 0.5f, p.y + 0.5f);
}

// ============================================================================
// Orientation sampling (nearest neighbor)
// ============================================================================
__device__ inline float GetOrientationNearest(float2 pos, const float* d_orient,
                                              int width, int height) {
  int2 p = make_int2((int) roundf(pos.x), (int) roundf(pos.y));
  return d_orient[p.y * width + p.x];
}

__device__ inline float GetOrientationNearestTex(float2 pos, cudaTextureObject_t tex,
                                                 int width, int height) {
  int2 p = make_int2((int) roundf(pos.x), (int) roundf(pos.y));
  return tex2D<float>(tex, p.x + 0.5f, p.y + 0.5f);
}

// ============================================================================
// Variance to confidence: 1 / (variance^2), returns 0 if variance==0
// ============================================================================
__device__ inline float ConvertOrientVarianceToConfidence2(float variance) {
  return variance == 0.0f ? 0.0f : 1.0f / (variance * variance);
}

// ============================================================================
// Random line perturbation for refinement
// ============================================================================
__device__ inline void GetRndLine3DCu(const Line3D& line_now, Line3D& line_new,
                                      const int2& pos, const float3x3& invK,
                                      float max_delta_z, float max_delta_n,
                                      float min_depth, float max_depth,
                                      curandState* cs) {
  // New depth
  float depth = line_now.p.z;
  float delta_z = CurandBetween(cs, -max_delta_z, max_delta_z);
  float depth_new = fminf(fmaxf(depth + delta_z, min_depth), max_depth);
  line_new.p = invK * make_float3(pos.x * depth_new, pos.y * depth_new, depth_new);

  // New orientation (additive perturbation + normalize + hemisphere)
  line_new.v.x = line_now.v.x + CurandBetween(cs, -max_delta_n, max_delta_n);
  line_new.v.y = line_now.v.y + CurandBetween(cs, -max_delta_n, max_delta_n);
  line_new.v.z = line_now.v.z + CurandBetween(cs, -max_delta_n, max_delta_n);
  NormalizeCu(line_new.v);
  VecOnHemisphere(line_new.v, make_float3(0.0f, 0.0f, 1.0f));
}

// ============================================================================
// View ordering: count how many elements in vec are less than vec[idx]
// ============================================================================
__device__ inline int GetOrder(const float* vec, int len, int idx) {
  int order = 0;
  for (int i = 0; i < len; i++) {
    if (vec[i] < vec[idx])
      order++;
  }
  return order;
}
