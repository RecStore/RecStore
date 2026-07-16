#pragma once

#include <fmt/core.h>
#include <fmt/format.h>

#include <cstdarg>
#include <cstdio>
#include <string>

namespace base {

template <typename... Args>
inline std::string
SFormat(fmt::format_string<Args...> fmt_str, Args&&... args) {
  return fmt::format(fmt_str, std::forward<Args>(args)...);
}

inline std::string StringVPrintf(const char* fmt_str, std::va_list args) {
  va_list args_copy;
  va_copy(args_copy, args);
  const int size = std::vsnprintf(nullptr, 0, fmt_str, args_copy);
  va_end(args_copy);
  if (size <= 0) {
    return {};
  }

  std::string result(static_cast<size_t>(size), '\0');
  std::vsnprintf(result.data(), result.size() + 1, fmt_str, args);
  return result;
}

inline std::string StringPrintf(const char* fmt_str, ...) {
  va_list args;
  va_start(args, fmt_str);
  std::string result = StringVPrintf(fmt_str, args);
  va_end(args);
  return result;
}

inline std::string PrettyPrintBytesString(double bytes) {
  static constexpr const char* kUnits[] = {"B", "KB", "MB", "GB", "TB", "PB"};
  size_t unit_index                     = 0;
  double value                          = bytes;
  while (value >= 1024.0 &&
         unit_index + 1 < (sizeof(kUnits) / sizeof(kUnits[0]))) {
    value /= 1024.0;
    ++unit_index;
  }
  if (unit_index == 0) {
    return fmt::format("{:.0f} {}", value, kUnits[unit_index]);
  }
  return fmt::format("{:.2f} {}", value, kUnits[unit_index]);
}

} // namespace base
