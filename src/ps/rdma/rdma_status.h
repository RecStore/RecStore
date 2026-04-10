#pragma once

#include <cstdint>

namespace petps {

enum class RpcStatus : std::int32_t {
  kPending           = 123456789,
  kOk                = 0,
  kInvalidPayload    = -1,
  kWrongShard        = -2,
  kBatchTooLarge     = -3,
  kValueSizeMismatch = -4,
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
  }
  return "unknown";
}

} // namespace petps
