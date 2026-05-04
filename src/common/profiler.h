// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// Scoped CPU profiling for pipeline stage timing.

#pragma once

#include <chrono>
#include <map>
#include <mutex>
#include <string>

class Profiler {
 public:
  static Profiler& Get();

  static void Enable(bool on) {
    Get().enabled_ = on;
  }
  static bool IsEnabled() {
    return Get().enabled_;
  }

  void BeginScope(const std::string& name);
  void EndScope(const std::string& name);

  static void PrintReport();

 private:
  struct Entry {
    double total_ms = 0;
    int count = 0;
  };

  bool enabled_ = false;
  std::mutex mutex_;
  std::map<std::string, Entry> entries_;
  std::map<std::string, std::chrono::steady_clock::time_point> cpu_starts_;
};

// Scoped CPU profiling
class ProfileScope {
 public:
  explicit ProfileScope(const char* name) : name_(name) {
    if (Profiler::IsEnabled())
      Profiler::Get().BeginScope(name_);
  }
  ~ProfileScope() {
    if (Profiler::IsEnabled())
      Profiler::Get().EndScope(name_);
  }

 private:
  std::string name_;
};

#define PROFILE_SCOPE(name) ProfileScope _prof_scope_(name)
