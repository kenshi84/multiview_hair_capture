// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// CUDA float2/float3/float4 operator overloads

#pragma once

#include <cuda_runtime.h>
#include <cmath>

// ==================== float2 ====================

inline __host__ __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ float2 operator*(float2 a, float s) {
  return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ float2 operator*(float s, float2 a) {
  return make_float2(s * a.x, s * a.y);
}
inline __host__ __device__ float2 operator/(float2 a, float s) {
  float inv = 1.0f / s;
  return make_float2(a.x * inv, a.y * inv);
}
inline __host__ __device__ float2 operator-(float2 a) {
  return make_float2(-a.x, -a.y);
}
inline __host__ __device__ void operator+=(float2& a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}
inline __host__ __device__ void operator-=(float2& a, float2 b) {
  a.x -= b.x;
  a.y -= b.y;
}
inline __host__ __device__ void operator*=(float2& a, float s) {
  a.x *= s;
  a.y *= s;
}
inline __host__ __device__ void operator/=(float2& a, float s) {
  float inv = 1.0f / s;
  a.x *= inv;
  a.y *= inv;
}
inline __host__ __device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float length(float2 a) {
  return sqrtf(dot(a, a));
}
inline __host__ __device__ float2 normalize(float2 a) {
  return a / length(a);
}

// ==================== float3 ====================

inline __host__ __device__ float3 make_float3(float s) {
  return make_float3(s, s, s);
}
inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(float s, float3 a) {
  return make_float3(s * a.x, s * a.y, s * a.z);
}
inline __host__ __device__ float3 operator/(float3 a, float s) {
  float inv = 1.0f / s;
  return make_float3(a.x * inv, a.y * inv, a.z * inv);
}
inline __host__ __device__ float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ void operator+=(float3& a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline __host__ __device__ void operator-=(float3& a, float3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}
inline __host__ __device__ void operator*=(float3& a, float s) {
  a.x *= s;
  a.y *= s;
  a.z *= s;
}
inline __host__ __device__ void operator/=(float3& a, float s) {
  float inv = 1.0f / s;
  a.x *= inv;
  a.y *= inv;
  a.z *= inv;
}
inline __host__ __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}
inline __host__ __device__ float length(float3 a) {
  return sqrtf(dot(a, a));
}
inline __host__ __device__ float3 normalize(float3 a) {
  return a / length(a);
}
inline __host__ __device__ float3 fminf(float3 a, float3 b) {
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
inline __host__ __device__ float3 fabsf(float3 a) {
  return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

// ==================== float4 ====================

inline __host__ __device__ float4 make_float4(float s) {
  return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a, float w) {
  return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ float4 operator-(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ float4 operator*(float4 a, float s) {
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ float4 operator*(float s, float4 a) {
  return make_float4(s * a.x, s * a.y, s * a.z, s * a.w);
}
inline __host__ __device__ float4 operator/(float4 a, float s) {
  float inv = 1.0f / s;
  return make_float4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
}
inline __host__ __device__ float4 operator-(float4 a) {
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ void operator+=(float4& a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
inline __host__ __device__ void operator-=(float4& a, float4 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
inline __host__ __device__ void operator*=(float4& a, float s) {
  a.x *= s;
  a.y *= s;
  a.z *= s;
  a.w *= s;
}
inline __host__ __device__ float dot(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __host__ __device__ float length(float4 a) {
  return sqrtf(dot(a, a));
}
inline __host__ __device__ float4 normalize(float4 a) {
  return a / length(a);
}

// ==================== Clamp ====================

inline __host__ __device__ float clamp(float x, float lo, float hi) {
  return fminf(hi, fmaxf(lo, x));
}
inline __host__ __device__ int clamp(int x, int lo, int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// ==================== MIN/MAX macros ====================

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
