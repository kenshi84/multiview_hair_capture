// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <chrono>
#include <string>

#include "common/logger.h"

class Timer {
 public:
  Timer() {
    Reset();
  }
  void Reset() {
    start_ = std::chrono::steady_clock::now();
  }

  double ElapsedSeconds() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_).count();
  }

  double ElapsedMs() const {
    return ElapsedSeconds() * 1000.0;
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

class ScopedTimer {
 public:
  explicit ScopedTimer(const std::string& label) : label_(label) {}
  ~ScopedTimer() {
    LOG_INFO("[%s] %.3f s", label_.c_str(), timer_.ElapsedSeconds());
  }

 private:
  std::string label_;
  Timer timer_;
};
