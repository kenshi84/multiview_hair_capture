// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/logger.h"

#include <cstring>

static const char* LevelTag(LogLevel level) {
  switch (level) {
    case LogLevel::kError:
      return "ERROR";
    case LogLevel::kWarning:
      return "WARN";
    case LogLevel::kInfo:
      return "INFO";
    case LogLevel::kDebug:
      return "DEBUG";
  }
  return "???";
}

static LogLevel ParseLevel(const std::string& s) {
  if (s == "error")
    return LogLevel::kError;
  if (s == "warn" || s == "warning")
    return LogLevel::kWarning;
  if (s == "debug")
    return LogLevel::kDebug;
  return LogLevel::kInfo;
}

Logger& Logger::Get() {
  static Logger instance;
  return instance;
}

void Logger::Init(LogLevel level, const std::string& log_file) {
  auto& g = Get();
  g.level_ = level;
  if (g.file_) {
    fclose(g.file_);
    g.file_ = nullptr;
  }
  if (!log_file.empty()) {
    g.file_ = fopen(log_file.c_str(), "a");
  }
}

void Logger::Init(const std::string& level_str, const std::string& log_file) {
  Init(ParseLevel(level_str), log_file);
}

void Logger::Log(LogLevel level, const char* fmt, ...) {
  auto& g = Get();
  if (level > g.level_)
    return;
  va_list args;
  va_start(args, fmt);
  g.LogImpl(level, fmt, args);
  va_end(args);
}

void Logger::LogImpl(LogLevel level, const char* fmt, va_list args) {
  const char* tag = LevelTag(level);

  FILE* out = (level <= LogLevel::kWarning) ? stderr : stdout;
  fprintf(out, "[%s] ", tag);
  va_list args_copy;
  va_copy(args_copy, args);
  vfprintf(out, fmt, args_copy);
  va_end(args_copy);
  fprintf(out, "\n");
  fflush(out);

  if (file_) {
    time_t now = time(nullptr);
    struct tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    fprintf(file_, "[%s][%02d:%02d:%02d] ", tag, tm_buf.tm_hour, tm_buf.tm_min,
            tm_buf.tm_sec);
    vfprintf(file_, fmt, args);
    fprintf(file_, "\n");
    fflush(file_);
  }
}
