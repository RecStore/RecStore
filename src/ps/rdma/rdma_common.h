#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <folly/portability/GFlags.h>

DECLARE_string(rdma_rc_namespace);

namespace petps {

inline std::uint64_t NowNs() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline std::uint64_t Exchange(std::atomic<std::uint64_t>* value) {
  return value->exchange(0, std::memory_order_relaxed);
}

inline std::string NamespaceToken() {
  if (!FLAGS_rdma_rc_namespace.empty()) {
    return FLAGS_rdma_rc_namespace;
  }
  return "default";
}

inline const std::int32_t* FixedSlotStatusWord(
    const void* buffer, std::size_t key_count, std::size_t value_size) {
  return reinterpret_cast<const std::int32_t*>(
      reinterpret_cast<const char*>(buffer) +
      key_count * static_cast<std::size_t>(value_size));
}

inline std::int32_t* FixedSlotStatusWord(
    void* buffer, std::size_t key_count, std::size_t value_size) {
  return reinterpret_cast<std::int32_t*>(
      reinterpret_cast<char*>(buffer) +
      key_count * static_cast<std::size_t>(value_size));
}

} // namespace petps
