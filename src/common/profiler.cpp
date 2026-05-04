// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/profiler.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "common/logger.h"

Profiler& Profiler::Get() {
  static Profiler instance;
  return instance;
}

void Profiler::BeginScope(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  cpu_starts_[name] = std::chrono::steady_clock::now();
}

void Profiler::EndScope(const std::string& name) {
  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cpu_starts_.find(name);
  if (it == cpu_starts_.end())
    return;
  double ms = std::chrono::duration<double, std::milli>(now - it->second).count();
  auto& e = entries_[name];
  e.total_ms += ms;
  e.count++;
  cpu_starts_.erase(it);
}

static void FormatDuration(double ms, char* buf, size_t bufsize) {
  double total_sec = ms / 1000.0;
  if (total_sec < 1.0) {
    snprintf(buf, bufsize, "%.0fms", ms);
  } else if (total_sec < 60.0) {
    snprintf(buf, bufsize, "%.1fs", total_sec);
  } else if (total_sec < 3600.0) {
    int m = static_cast<int>(total_sec) / 60;
    double s = total_sec - m * 60;
    snprintf(buf, bufsize, "%dm %.0fs", m, s);
  } else {
    int h = static_cast<int>(total_sec) / 3600;
    int m = (static_cast<int>(total_sec) % 3600) / 60;
    double s = total_sec - h * 3600 - m * 60;
    snprintf(buf, bufsize, "%dh %dm %.0fs", h, m, s);
  }
}

void Profiler::PrintReport() {
  auto& prof = Get();
  if (!prof.enabled_ || prof.entries_.empty())
    return;

  std::vector<std::pair<std::string, Entry>> sorted(prof.entries_.begin(),
                                                    prof.entries_.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.second.total_ms > b.second.total_ms;
  });

  LOG_INFO("[PROFILE] Pipeline Summary:");
  for (const auto& [name, e] : sorted) {
    char dur[64];
    FormatDuration(e.total_ms, dur, sizeof(dur));
    LOG_INFO("  %-40s %10s", name.c_str(), dur);
  }
}
