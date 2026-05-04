// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <string>

enum class LogLevel { kError = 0, kWarning = 1, kInfo = 2, kDebug = 3 };

class Logger {
 public:
  static void Init(LogLevel level, const std::string& log_file = "");
  static void Init(const std::string& level_str, const std::string& log_file = "");
  static void Log(LogLevel level, const char* fmt, ...);

 private:
  static Logger& Get();
  void LogImpl(LogLevel level, const char* fmt, va_list args);

  LogLevel level_ = LogLevel::kInfo;
  FILE* file_ = nullptr;
};

#define LOG_ERROR(...) Logger::Log(LogLevel::kError, __VA_ARGS__)
#define LOG_WARN(...) Logger::Log(LogLevel::kWarning, __VA_ARGS__)
#define LOG_INFO(...) Logger::Log(LogLevel::kInfo, __VA_ARGS__)
#define LOG_DEBUG(...) Logger::Log(LogLevel::kDebug, __VA_ARGS__)
