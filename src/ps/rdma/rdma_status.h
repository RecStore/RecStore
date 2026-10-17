#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace petps {

// Client-visible failure raised when a shard is judged unreachable by the
// slow-path health verdict, or when its lane was retired after losing a
// completion. Kept distinct from the server-returned RpcStatus codes and from
// a plain wait timeout so callers can fail fast without retrying blindly.
class ShardUnreachableError : public std::runtime_error {
public:
  explicit ShardUnreachableError(const std::string& message)
      : std::runtime_error(message) {}
};

enum class RpcStatus : std::int32_t {
  kPending           = 123456789,
  kOk                = 0,
  kInvalidPayload    = -1,
  kWrongShard        = -2,
  kBatchTooLarge     = -3,
  kValueSizeMismatch = -4,
  kServerDraining    = -5,
};

inline const char* RpcStatusToString(RpcStatus status) {
  switch (status) {
  case RpcStatus::kPending:
    return "pending";
  case RpcStatus::kOk:
    return "ok";
  case RpcStatus::kInvalidPayload:
    return "invalid_payload";
  case RpcStatus::kWrongShard:
    return "wrong_shard";
  case RpcStatus::kBatchTooLarge:
    return "batch_too_large";
  case RpcStatus::kValueSizeMismatch:
    return "value_size_mismatch";
  case RpcStatus::kServerDraining:
    return "server_draining";
  }
  return "unknown";
}

} // namespace petps
