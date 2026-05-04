// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// Host-side vector/matrix types for MVS orchestration and depth fusion

#pragma once

#include <cassert>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <class ET>
class Vec3;
template <class ET>
class Mat3x3;
template <class ET>
class Mat4x4;

template <class ET>
class Vec2 {
 public:
  ET x, y;
  Vec2() : x(0), y(0) {}
  Vec2(ET x, ET y) : x(x), y(y) {}
  Vec2 operator+(const Vec2& v) const {
    return {x + v.x, y + v.y};
  }
  Vec2 operator-(const Vec2& v) const {
    return {x - v.x, y - v.y};
  }
  Vec2 operator*(ET f) const {
    return {x * f, y * f};
  }
  Vec2 operator/(ET f) const {
    return {x / f, y / f};
  }
  ET operator*(const Vec2& v) const {
    return x * v.x + y * v.y;
  }
  const ET& operator[](int i) const {
    assert(i >= 0 && i < 2);
    return i == 0 ? x : y;
  }
  ET& operator[](int i) {
    assert(i >= 0 && i < 2);
    return i == 0 ? x : y;
  }
};

template <class ET>
class Vec3 {
 public:
  ET x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(ET x, ET y, ET z) : x(x), y(y), z(z) {}
  Vec3(ET f) : x(f), y(f), z(f) {}
  Vec3 operator+(const Vec3& v) const {
    return {x + v.x, y + v.y, z + v.z};
  }
  Vec3 operator-(const Vec3& v) const {
    return {x - v.x, y - v.y, z - v.z};
  }
  Vec3 operator-() const {
    return {-x, -y, -z};
  }
  Vec3 operator*(ET f) const {
    return {x * f, y * f, z * f};
  }
  Vec3 operator/(ET f) const {
    return {x / f, y / f, z / f};
  }
  Vec3& operator+=(const Vec3& v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  Vec3& operator*=(ET f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
  }
  ET operator*(const Vec3& v) const {
    return x * v.x + y * v.y + z * v.z;
  }
  Vec3 Cross(const Vec3& v) const {
    return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
  }
  ET MagnitudeSq() const {
    return x * x + y * y + z * z;
  }
  ET Magnitude() const {
    return std::sqrt(MagnitudeSq());
  }
  Vec3 Unit() const {
    return *this / Magnitude();
  }
  const ET& operator[](int i) const {
    assert(i >= 0 && i < 3);
    return i == 0 ? x : (i == 1 ? y : z);
  }
  ET& operator[](int i) {
    assert(i >= 0 && i < 3);
    return i == 0 ? x : (i == 1 ? y : z);
  }
};

template <class ET>
class Vec4 {
 public:
  ET x, y, z, w;
  Vec4() : x(0), y(0), z(0), w(0) {}
  Vec4(ET x, ET y, ET z, ET w) : x(x), y(y), z(z), w(w) {}
  Vec4(const Vec3<ET>& v, ET w = 1) : x(v.x), y(v.y), z(v.z), w(w) {}
  Vec4 operator*(ET f) const {
    return {x * f, y * f, z * f, w * f};
  }
  ET operator*(const Vec4& v) const {
    return x * v.x + y * v.y + z * v.z + w * v.w;
  }
  const ET& operator[](int i) const {
    assert(i >= 0 && i < 4);
    return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
  }
  ET& operator[](int i) {
    assert(i >= 0 && i < 4);
    return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
  }
};

template <class ET>
class Mat3x3 {
 public:
  ET m[9];
  Mat3x3() {
    memset(m, 0, sizeof(m));
  }
  Mat3x3(ET _00, ET _01, ET _02, ET _10, ET _11, ET _12, ET _20, ET _21, ET _22) {
    m[0] = _00;
    m[1] = _01;
    m[2] = _02;
    m[3] = _10;
    m[4] = _11;
    m[5] = _12;
    m[6] = _20;
    m[7] = _21;
    m[8] = _22;
  }
  Mat3x3& MakeI() {
    memset(m, 0, sizeof(m));
    m[0] = m[4] = m[8] = ET(1);
    return *this;
  }
  const ET& operator()(int r, int c) const {
    return m[r * 3 + c];
  }
  ET& operator()(int r, int c) {
    return m[r * 3 + c];
  }
  const ET* Ptr() const {
    return m;
  }
  ET* Ptr() {
    return m;
  }

  Vec3<ET> operator*(const Vec3<ET>& v) const {
    return {m[0] * v.x + m[1] * v.y + m[2] * v.z, m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z};
  }
  Mat3x3 operator*(const Mat3x3& o) const {
    Mat3x3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
          r(i, j) += (*this)(i, k) * o(k, j);
    return r;
  }
  Mat3x3 T() const {
    Mat3x3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        r(i, j) = (*this)(j, i);
    return r;
  }
  ET Det() const {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
           m[2] * (m[3] * m[7] - m[4] * m[6]);
  }
  Mat3x3 Inv() const {
    Mat3x3 r;
    ET d = ET(1) / Det();
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
  Vec3<ET> GetCol(int c) const {
    return {m[c], m[3 + c], m[6 + c]};
  }
};

template <class ET>
class Mat4x4 {
 public:
  ET m[16];
  Mat4x4() {
    memset(m, 0, sizeof(m));
  }
  Mat4x4& MakeI() {
    memset(m, 0, sizeof(m));
    m[0] = m[5] = m[10] = m[15] = ET(1);
    return *this;
  }
  const ET& operator()(int r, int c) const {
    return m[r * 4 + c];
  }
  ET& operator()(int r, int c) {
    return m[r * 4 + c];
  }

  Vec4<ET> operator*(const Vec4<ET>& v) const {
    Vec4<ET> r;
    for (int i = 0; i < 4; ++i)
      r[i] =
          m[i * 4] * v.x + m[i * 4 + 1] * v.y + m[i * 4 + 2] * v.z + m[i * 4 + 3] * v.w;
    return r;
  }
  Mat4x4 operator*(const Mat4x4& o) const {
    Mat4x4 r;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
          r(i, j) += (*this)(i, k) * o(k, j);
    return r;
  }
  Mat3x3<ET> Submat3x3() const {
    Mat3x3<ET> r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        r(i, j) = (*this)(i, j);
    return r;
  }
  Vec3<ET> transform_point(const Vec3<ET>& v) const {
    return {
        (*this)(0, 0) * v.x + (*this)(0, 1) * v.y + (*this)(0, 2) * v.z + (*this)(0, 3),
        (*this)(1, 0) * v.x + (*this)(1, 1) * v.y + (*this)(1, 2) * v.z + (*this)(1, 3),
        (*this)(2, 0) * v.x + (*this)(2, 1) * v.y + (*this)(2, 2) * v.z +
            (*this)(2, 3)};
  }
};

using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;
using Mat3x3f = Mat3x3<float>;
using Mat4x4f = Mat4x4<float>;
