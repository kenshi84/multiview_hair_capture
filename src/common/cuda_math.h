// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// CUDA matrix types for GPU kernels (written from scratch)
// Provides float2x2, float2x3, float3x2, float3x3, float3x4, float4x4, matNxM

#pragma once

#include "common/cuda_check.h"
#include "common/cuda_vec_ops.h"

// ==================== float2x2 ====================

class float2x2 {
 public:
  float m[4];

  inline __host__ __device__ float2x2() {
    m[0] = m[1] = m[2] = m[3] = 0;
  }
  inline __host__ __device__ float2x2(float m00, float m01, float m10, float m11) {
    m[0] = m00;
    m[1] = m01;
    m[2] = m10;
    m[3] = m11;
  }
  inline __host__ __device__ float2x2(const float v[4]) {
    m[0] = v[0];
    m[1] = v[1];
    m[2] = v[2];
    m[3] = v[3];
  }

  inline __host__ __device__ float& operator()(int r, int c) {
    return m[r * 2 + c];
  }
  inline __host__ __device__ float operator()(int r, int c) const {
    return m[r * 2 + c];
  }

  inline __host__ __device__ float det() const {
    return m[0] * m[3] - m[1] * m[2];
  }

  inline __host__ __device__ float2x2 getInverse() const {
    float d = 1.0f / det();
    return float2x2(m[3] * d, -m[1] * d, -m[2] * d, m[0] * d);
  }

  inline __host__ __device__ float2 operator*(float2 v) const {
    return make_float2(m[0] * v.x + m[1] * v.y, m[2] * v.x + m[3] * v.y);
  }

  inline __host__ __device__ float2x2 operator*(float s) const {
    return float2x2(m[0] * s, m[1] * s, m[2] * s, m[3] * s);
  }

  inline __host__ __device__ float2x2 operator*(const float2x2& o) const {
    return float2x2(m[0] * o.m[0] + m[1] * o.m[2], m[0] * o.m[1] + m[1] * o.m[3],
                    m[2] * o.m[0] + m[3] * o.m[2], m[2] * o.m[1] + m[3] * o.m[3]);
  }

  inline __host__ __device__ float2x2 operator+(const float2x2& o) const {
    return float2x2(m[0] + o.m[0], m[1] + o.m[1], m[2] + o.m[2], m[3] + o.m[3]);
  }

  static inline __host__ __device__ float2x2 identity() {
    return float2x2(1, 0, 0, 1);
  }
};

// ==================== float3x3 ====================

class float3x3 {
 public:
  float m[9];

  inline __host__ __device__ float3x3() {
    for (int i = 0; i < 9; ++i)
      m[i] = 0;
  }
  inline __host__ __device__ float3x3(const float v[9]) {
    for (int i = 0; i < 9; ++i)
      m[i] = v[i];
  }
  inline __host__ __device__ float3x3(float m00, float m01, float m02, float m10,
                                      float m11, float m12, float m20, float m21,
                                      float m22) {
    m[0] = m00;
    m[1] = m01;
    m[2] = m02;
    m[3] = m10;
    m[4] = m11;
    m[5] = m12;
    m[6] = m20;
    m[7] = m21;
    m[8] = m22;
  }

  inline __host__ __device__ float& operator()(int r, int c) {
    return m[r * 3 + c];
  }
  inline __host__ __device__ float operator()(int r, int c) const {
    return m[r * 3 + c];
  }

  inline __host__ __device__ float3 operator*(float3 v) const {
    return make_float3(m[0] * v.x + m[1] * v.y + m[2] * v.z,
                       m[3] * v.x + m[4] * v.y + m[5] * v.z,
                       m[6] * v.x + m[7] * v.y + m[8] * v.z);
  }

  inline __host__ __device__ float3x3 operator*(const float3x3& o) const {
    float3x3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        r.m[i * 3 + j] = 0;
        for (int k = 0; k < 3; ++k)
          r.m[i * 3 + j] += m[i * 3 + k] * o.m[k * 3 + j];
      }
    return r;
  }

  inline __host__ __device__ float3x3 operator*(float s) const {
    float3x3 r;
    for (int i = 0; i < 9; ++i)
      r.m[i] = m[i] * s;
    return r;
  }

  inline __host__ __device__ float3x3 operator+(const float3x3& o) const {
    float3x3 r;
    for (int i = 0; i < 9; ++i)
      r.m[i] = m[i] + o.m[i];
    return r;
  }

  inline __host__ __device__ float3x3 operator-(const float3x3& o) const {
    float3x3 r;
    for (int i = 0; i < 9; ++i)
      r.m[i] = m[i] - o.m[i];
    return r;
  }

  inline __host__ __device__ float det() const {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
  }

  inline __host__ __device__ float3x3 getInverse() const {
    float3x3 r;
    float d = 1.0f / det();
    r.m[0] = (m[4] * m[8] - m[5] * m[7]) * d;
    r.m[1] = (m[2] * m[7] - m[1] * m[8]) * d;
    r.m[2] = (m[1] * m[5] - m[2] * m[4]) * d;
    r.m[3] = (m[5] * m[6] - m[3] * m[8]) * d;
    r.m[4] = (m[0] * m[8] - m[2] * m[6]) * d;
    r.m[5] = (m[2] * m[3] - m[0] * m[5]) * d;
    r.m[6] = (m[3] * m[7] - m[4] * m[6]) * d;
    r.m[7] = (m[1] * m[6] - m[0] * m[7]) * d;
    r.m[8] = (m[0] * m[4] - m[1] * m[3]) * d;
    return r;
  }

  inline __host__ __device__ float3x3 getTranspose() const {
    float3x3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        r.m[i * 3 + j] = m[j * 3 + i];
    return r;
  }

  inline __host__ __device__ void setZero() {
    for (int i = 0; i < 9; ++i)
      m[i] = 0;
  }

  inline __host__ __device__ float3 getRow(int i) const {
    return make_float3(m[3 * i], m[3 * i + 1], m[3 * i + 2]);
  }

  inline __host__ __device__ void setRow(int i, float3 v) {
    m[3 * i] = v.x;
    m[3 * i + 1] = v.y;
    m[3 * i + 2] = v.z;
  }

  inline __host__ __device__ void normalizeRows() {
    for (int i = 0; i < 3; ++i) {
      float3 r = getRow(i);
      r = r / length(r);
      setRow(i, r);
    }
  }

  static inline __host__ __device__ float3x3 identity() {
    return float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
  }

  static inline __host__ __device__ float3x3 tensorProduct(float3 a, float3 b) {
    return float3x3(a.x * b.x, a.x * b.y, a.x * b.z, a.y * b.x, a.y * b.y, a.y * b.z,
                    a.z * b.x, a.z * b.y, a.z * b.z);
  }
};

// ==================== float2x3 ====================

class float2x3 {
 public:
  float m[6];

  inline __host__ __device__ float2x3() {
    for (int i = 0; i < 6; ++i)
      m[i] = 0;
  }
  inline __host__ __device__ float2x3(const float v[6]) {
    for (int i = 0; i < 6; ++i)
      m[i] = v[i];
  }

  inline __host__ __device__ float& operator()(int r, int c) {
    return m[r * 3 + c];
  }
  inline __host__ __device__ float operator()(int r, int c) const {
    return m[r * 3 + c];
  }

  inline __host__ __device__ float2 operator*(float3 v) const {
    return make_float2(m[0] * v.x + m[1] * v.y + m[2] * v.z,
                       m[3] * v.x + m[4] * v.y + m[5] * v.z);
  }

  inline __host__ __device__ float2x3 operator*(float s) const {
    float2x3 r;
    for (int i = 0; i < 6; ++i)
      r.m[i] = m[i] * s;
    return r;
  }

  inline __host__ __device__ float2x3 operator/(float s) const {
    float2x3 r;
    for (int i = 0; i < 6; ++i)
      r.m[i] = m[i] / s;
    return r;
  }
};

// ==================== float3x2 ====================

class float3x2 {
 public:
  float m[6];

  inline __host__ __device__ float3x2() {
    for (int i = 0; i < 6; ++i)
      m[i] = 0;
  }
  inline __host__ __device__ float3x2(const float v[6]) {
    for (int i = 0; i < 6; ++i)
      m[i] = v[i];
  }

  inline __host__ __device__ float& operator()(int r, int c) {
    return m[r * 2 + c];
  }
  inline __host__ __device__ float operator()(int r, int c) const {
    return m[r * 2 + c];
  }

  inline __host__ __device__ float3 operator*(float2 v) const {
    return make_float3(m[0] * v.x + m[1] * v.y, m[2] * v.x + m[3] * v.y,
                       m[4] * v.x + m[5] * v.y);
  }

  inline __host__ __device__ float3x2 operator*(float s) const {
    float3x2 r;
    for (int i = 0; i < 6; ++i)
      r.m[i] = m[i] * s;
    return r;
  }

  inline __host__ __device__ float2x3 getTranspose() const {
    float2x3 r;
    r.m[0] = m[0];
    r.m[1] = m[2];
    r.m[2] = m[4];
    r.m[3] = m[1];
    r.m[4] = m[3];
    r.m[5] = m[5];
    return r;
  }
};

// Free functions for mixed-size multiplies
inline __host__ __device__ float2x2 matMul(const float2x3& a, const float3x2& b) {
  return float2x2(a.m[0] * b.m[0] + a.m[1] * b.m[2] + a.m[2] * b.m[4],
                  a.m[0] * b.m[1] + a.m[1] * b.m[3] + a.m[2] * b.m[5],
                  a.m[3] * b.m[0] + a.m[4] * b.m[2] + a.m[5] * b.m[4],
                  a.m[3] * b.m[1] + a.m[4] * b.m[3] + a.m[5] * b.m[5]);
}

inline __host__ __device__ float2x3 matMul(const float2x3& a, const float3x3& b) {
  float2x3 r;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) {
      r.m[i * 3 + j] = 0;
      for (int k = 0; k < 3; ++k)
        r.m[i * 3 + j] += a.m[i * 3 + k] * b.m[k * 3 + j];
    }
  return r;
}

inline __host__ __device__ float3 matMul(float2 a, const float2x3& b) {
  return make_float3(a.x * b.m[0] + a.y * b.m[3], a.x * b.m[1] + a.y * b.m[4],
                     a.x * b.m[2] + a.y * b.m[5]);
}

// ==================== float3x4 ====================

class float3x4 {
 public:
  float m[12];

  inline __host__ __device__ float3x4() {
    for (int i = 0; i < 12; ++i)
      m[i] = 0;
  }
  inline __host__ __device__ float3x4(const float v[12]) {
    for (int i = 0; i < 12; ++i)
      m[i] = v[i];
  }
  inline __host__ __device__ float3x4(const float3x3& r) {
    m[0] = r.m[0];
    m[1] = r.m[1];
    m[2] = r.m[2];
    m[3] = 0;
    m[4] = r.m[3];
    m[5] = r.m[4];
    m[6] = r.m[5];
    m[7] = 0;
    m[8] = r.m[6];
    m[9] = r.m[7];
    m[10] = r.m[8];
    m[11] = 0;
  }

  inline __host__ __device__ float& operator()(int r, int c) {
    return m[r * 4 + c];
  }
  inline __host__ __device__ float operator()(int r, int c) const {
    return m[r * 4 + c];
  }

  // Transform point (implicit w=1)
  inline __host__ __device__ float3 operator*(float3 v) const {
    return make_float3(m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3],
                       m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7],
                       m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11]);
  }

  // Transform float4
  inline __host__ __device__ float4 operator*(float4 v) const {
    return make_float4(m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w,
                       m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7] * v.w,
                       m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11] * v.w, v.w);
  }

  inline __host__ __device__ float3x4 operator*(float s) const {
    float3x4 r;
    for (int i = 0; i < 12; ++i)
      r.m[i] = m[i] * s;
    return r;
  }

  inline __host__ __device__ float3x3 getFloat3x3() const {
    return float3x3(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
  }

  inline __host__ __device__ float3 getTranslation() const {
    return make_float3(m[3], m[7], m[11]);
  }

  inline __host__ __device__ void setFloat3x3(const float3x3& r) {
    m[0] = r.m[0];
    m[1] = r.m[1];
    m[2] = r.m[2];
    m[4] = r.m[3];
    m[5] = r.m[4];
    m[6] = r.m[5];
    m[8] = r.m[6];
    m[9] = r.m[7];
    m[10] = r.m[8];
  }

  inline __host__ __device__ void setTranslation(float3 t) {
    m[3] = t.x;
    m[7] = t.y;
    m[11] = t.z;
  }

  inline __host__ __device__ float3x4 getInverse() const {
    float3x3 R = getFloat3x3();
    float3x3 Ri = R.getInverse();
    float3 t = getTranslation();
    float3 neg_Ri_t = Ri * t;
    float3x4 r(Ri);
    r.setTranslation(make_float3(-neg_Ri_t.x, -neg_Ri_t.y, -neg_Ri_t.z));
    return r;
  }

  // Multiply two 3x4 matrices (implicit last row [0,0,0,1])
  inline __host__ __device__ float3x4 operator*(const float3x4& o) const {
    float3x4 r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        r.m[i * 4 + j] = 0;
        for (int k = 0; k < 3; ++k)
          r.m[i * 4 + j] += m[i * 4 + k] * o.m[k * 4 + j];
      }
      r.m[i * 4 + 3] = m[i * 4 + 3];
      for (int k = 0; k < 3; ++k)
        r.m[i * 4 + 3] += m[i * 4 + k] * o.m[k * 4 + 3];
    }
    return r;
  }
};

// ==================== float4x4 ====================

class float4x4 {
 public:
  float m[16];

  inline __host__ __device__ float4x4() {
    for (int i = 0; i < 16; ++i)
      m[i] = 0;
  }
  inline __host__ __device__ float4x4(const float v[16]) {
    for (int i = 0; i < 16; ++i)
      m[i] = v[i];
  }
  inline __host__ __device__ float4x4(const float3x4& o) {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
        m[i * 4 + j] = o.m[i * 4 + j];
    m[12] = 0;
    m[13] = 0;
    m[14] = 0;
    m[15] = 1;
  }
  inline __host__ __device__ float4x4(const float3x3& o) {
    m[0] = o.m[0];
    m[1] = o.m[1];
    m[2] = o.m[2];
    m[3] = 0;
    m[4] = o.m[3];
    m[5] = o.m[4];
    m[6] = o.m[5];
    m[7] = 0;
    m[8] = o.m[6];
    m[9] = o.m[7];
    m[10] = o.m[8];
    m[11] = 0;
    m[12] = 0;
    m[13] = 0;
    m[14] = 0;
    m[15] = 1;
  }

  inline __host__ __device__ float& operator()(int r, int c) {
    return m[r * 4 + c];
  }
  inline __host__ __device__ float operator()(int r, int c) const {
    return m[r * 4 + c];
  }

  inline __host__ __device__ float4 operator*(float4 v) const {
    return make_float4(m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w,
                       m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7] * v.w,
                       m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11] * v.w,
                       m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15] * v.w);
  }

  inline __host__ __device__ float3 operator*(float3 v) const {
    return make_float3(m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3],
                       m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7],
                       m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11]);
  }

  inline __host__ __device__ float3x3 getFloat3x3() const {
    return float3x3(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
  }

  inline __host__ __device__ float3x4 getFloat3x4() const {
    float3x4 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
        r.m[i * 4 + j] = m[i * 4 + j];
    return r;
  }

  inline __host__ __device__ void setIdentity() {
    for (int i = 0; i < 16; ++i)
      m[i] = 0;
    m[0] = m[5] = m[10] = m[15] = 1;
  }

  inline __host__ __device__ float4x4 getTranspose() const {
    float4x4 r;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        r.m[i * 4 + j] = m[j * 4 + i];
    return r;
  }

  static inline __host__ __device__ float4x4 identity() {
    float4x4 r;
    r.setIdentity();
    return r;
  }
};

// ==================== matNxM ====================

template <unsigned int N, unsigned int M>
class matNxM {
 public:
  float entries[N * M];

  inline __host__ __device__ matNxM() {
    for (unsigned int i = 0; i < N * M; ++i)
      entries[i] = 0;
  }

  inline __host__ __device__ matNxM(const float values[N * M]) {
    for (unsigned int i = 0; i < N * M; ++i)
      entries[i] = values[i];
  }

  inline __host__ __device__ void setZero() {
    for (unsigned int i = 0; i < N * M; ++i)
      entries[i] = 0;
  }

  inline __host__ __device__ void setIdentity() {
    setZero();
    for (unsigned int i = 0; i < (N < M ? N : M); ++i)
      entries[i * M + i] = 1.0f;
  }

  static inline __host__ __device__ matNxM<N, M> getIdentity() {
    matNxM<N, M> r;
    r.setIdentity();
    return r;
  }

  inline __host__ __device__ float& operator()(unsigned int i, unsigned int j) {
    return entries[i * M + j];
  }
  inline __host__ __device__ float operator()(unsigned int i, unsigned int j) const {
    return entries[i * M + j];
  }
  inline __host__ __device__ float& operator()(unsigned int i) {
    return entries[i];
  }
  inline __host__ __device__ float operator()(unsigned int i) const {
    return entries[i];
  }

  template <unsigned int NO, unsigned int MO>
  inline __host__ __device__ matNxM<N, MO> operator*(const matNxM<NO, MO>& o) const {
    matNxM<N, MO> r;
    for (unsigned int i = 0; i < N; ++i)
      for (unsigned int j = 0; j < MO; ++j) {
        float sum = 0;
        for (unsigned int k = 0; k < M; ++k)
          sum += (*this)(i, k) * o(k, j);
        r(i, j) = sum;
      }
    return r;
  }

  inline __host__ __device__ matNxM<M, N> getTranspose() const {
    matNxM<M, N> r;
    for (unsigned int i = 0; i < M; ++i)
      for (unsigned int j = 0; j < N; ++j)
        r(i, j) = (*this)(j, i);
    return r;
  }

  inline __host__ __device__ matNxM<N, M> operator+(const matNxM<N, M>& o) const {
    matNxM<N, M> r;
    for (unsigned int i = 0; i < N * M; ++i)
      r.entries[i] = entries[i] + o.entries[i];
    return r;
  }

  inline __host__ __device__ matNxM<N, M>& operator+=(const matNxM<N, M>& o) {
    for (unsigned int i = 0; i < N * M; ++i)
      entries[i] += o.entries[i];
    return *this;
  }

  inline __host__ __device__ matNxM<N, M> operator-(const matNxM<N, M>& o) const {
    matNxM<N, M> r;
    for (unsigned int i = 0; i < N * M; ++i)
      r.entries[i] = entries[i] - o.entries[i];
    return r;
  }

  inline __host__ __device__ matNxM<N, M>& operator-=(const matNxM<N, M>& o) {
    for (unsigned int i = 0; i < N * M; ++i)
      entries[i] -= o.entries[i];
    return *this;
  }

  inline __host__ __device__ matNxM<N, M> operator*(float t) const {
    matNxM<N, M> r;
    for (unsigned int i = 0; i < N * M; ++i)
      r.entries[i] = entries[i] * t;
    return r;
  }

  inline __host__ __device__ matNxM<N, M>& operator*=(float t) {
    for (unsigned int i = 0; i < N * M; ++i)
      entries[i] *= t;
    return *this;
  }

  inline __host__ __device__ matNxM<N, M> operator/(float t) const {
    matNxM<N, M> r;
    for (unsigned int i = 0; i < N * M; ++i)
      r.entries[i] = entries[i] / t;
    return r;
  }

  inline __host__ __device__ matNxM<N, M>& operator/=(float t) {
    for (unsigned int i = 0; i < N * M; ++i)
      entries[i] /= t;
    return *this;
  }

  inline __host__ __device__ float det() const;
  inline __host__ __device__ matNxM<N, M> getInverse() const;

  // Generic constructors for conversion from CUDA types (specialized below)
  template <class B>
  explicit inline __host__ __device__ matNxM(const B& other);

  template <class B>
  explicit inline __host__ __device__ matNxM(const B& other0, const B& other1);

  // Generic cast operators (specialized below)
  inline __host__ __device__ operator float();
  inline __host__ __device__ operator float2();
  inline __host__ __device__ operator float3();
  inline __host__ __device__ operator float4();
  inline __host__ __device__ operator float2x2();
  inline __host__ __device__ operator float3x3();
  inline __host__ __device__ operator float4x4();

  template <unsigned int NO, unsigned int MO>
  inline __host__ __device__ void getBlock(unsigned int rs, unsigned int cs,
                                           matNxM<NO, MO>& out) const {
    for (unsigned int i = 0; i < NO; ++i)
      for (unsigned int j = 0; j < MO; ++j)
        out(i, j) = (*this)(rs + i, cs + j);
  }
};

// matNxM specializations for det/inverse

template <>
inline __host__ __device__ float matNxM<2, 2>::det() const {
  return entries[0] * entries[3] - entries[1] * entries[2];
}

template <>
inline __host__ __device__ matNxM<2, 2> matNxM<2, 2>::getInverse() const {
  matNxM<2, 2> r;
  r.entries[0] = entries[3];
  r.entries[1] = -entries[1];
  r.entries[2] = -entries[2];
  r.entries[3] = entries[0];
  return r * (1.0f / det());
}

template <>
inline __host__ __device__ float matNxM<3, 3>::det() const {
  return entries[0] * (entries[4] * entries[8] - entries[5] * entries[7]) -
         entries[1] * (entries[3] * entries[8] - entries[5] * entries[6]) +
         entries[2] * (entries[3] * entries[7] - entries[4] * entries[6]);
}

template <>
inline __host__ __device__ matNxM<3, 3> matNxM<3, 3>::getInverse() const {
  matNxM<3, 3> r;
  r.entries[0] = entries[4] * entries[8] - entries[5] * entries[7];
  r.entries[1] = -entries[1] * entries[8] + entries[2] * entries[7];
  r.entries[2] = entries[1] * entries[5] - entries[2] * entries[4];
  r.entries[3] = -entries[3] * entries[8] + entries[5] * entries[6];
  r.entries[4] = entries[0] * entries[8] - entries[2] * entries[6];
  r.entries[5] = -entries[0] * entries[5] + entries[2] * entries[3];
  r.entries[6] = entries[3] * entries[7] - entries[4] * entries[6];
  r.entries[7] = -entries[0] * entries[7] + entries[1] * entries[6];
  r.entries[8] = entries[0] * entries[4] - entries[1] * entries[3];
  return r * (1.0f / det());
}

// matNxM conversions to/from CUDA vector types

template <>
template <>
inline __host__ __device__ matNxM<2, 1>::matNxM(const float2& v) {
  entries[0] = v.x;
  entries[1] = v.y;
}

template <>
template <>
inline __host__ __device__ matNxM<3, 1>::matNxM(const float3& v) {
  entries[0] = v.x;
  entries[1] = v.y;
  entries[2] = v.z;
}

template <>
template <>
inline __host__ __device__ matNxM<4, 1>::matNxM(const float4& v) {
  entries[0] = v.x;
  entries[1] = v.y;
  entries[2] = v.z;
  entries[3] = v.w;
}

template <>
template <>
inline __host__ __device__ matNxM<3, 2>::matNxM(const float3& c0, const float3& c1) {
  entries[0] = c0.x;
  entries[1] = c1.x;
  entries[2] = c0.y;
  entries[3] = c1.y;
  entries[4] = c0.z;
  entries[5] = c1.z;
}

template <>
inline __host__ __device__ matNxM<1, 1>::operator float() {
  return entries[0];
}

template <>
inline __host__ __device__ matNxM<2, 1>::operator float2() {
  return make_float2(entries[0], entries[1]);
}

template <>
inline __host__ __device__ matNxM<3, 1>::operator float3() {
  return make_float3(entries[0], entries[1], entries[2]);
}

template <>
inline __host__ __device__ matNxM<4, 1>::operator float4() {
  return make_float4(entries[0], entries[1], entries[2], entries[3]);
}

template <>
template <>
inline __host__ __device__ matNxM<2, 2>::matNxM(const float2x2& o) {
  for (int i = 0; i < 4; ++i)
    entries[i] = o.m[i];
}

template <>
template <>
inline __host__ __device__ matNxM<3, 3>::matNxM(const float3x3& o) {
  for (int i = 0; i < 9; ++i)
    entries[i] = o.m[i];
}

template <>
template <>
inline __host__ __device__ matNxM<4, 4>::matNxM(const float4x4& o) {
  for (int i = 0; i < 16; ++i)
    entries[i] = o.m[i];
}

template <>
inline __host__ __device__ matNxM<2, 2>::operator float2x2() {
  return float2x2(entries[0], entries[1], entries[2], entries[3]);
}

template <>
inline __host__ __device__ matNxM<3, 3>::operator float3x3() {
  float3x3 r;
  for (int i = 0; i < 9; ++i)
    r.m[i] = entries[i];
  return r;
}

template <>
inline __host__ __device__ matNxM<4, 4>::operator float4x4() {
  float4x4 r;
  for (int i = 0; i < 16; ++i)
    r.m[i] = entries[i];
  return r;
}

// Type aliases
using mat4x4 = matNxM<4, 4>;
using mat3x3 = matNxM<3, 3>;
using mat2x3 = matNxM<2, 3>;
using mat3x2 = matNxM<3, 2>;
using mat2x2 = matNxM<2, 2>;
using mat1x2 = matNxM<1, 2>;
using mat2x1 = matNxM<2, 1>;
using mat1x3 = matNxM<1, 3>;
using mat3x1 = matNxM<3, 1>;
using mat1x1 = matNxM<1, 1>;
