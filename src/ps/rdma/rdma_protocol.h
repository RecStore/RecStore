#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

#include "base/array.h"
#include "base/flatc.h"
#include "base/log.h"
#include "base/tensor.h"
#include "ps/base/parameters.h"
#include "ps/rdma/rdma_status.h"

namespace petps {

inline constexpr std::uint32_t kRcProtocolMagic            = 0x52435053;
inline constexpr std::uint16_t kRcProtocolVersion          = 1;
inline constexpr std::size_t kTableNameBytes               = 64;
inline constexpr std::uint32_t kRcSlotReady                = 1;
inline constexpr std::uint32_t kRcSlotDone                 = 2;
inline constexpr std::uint32_t kRcFlagGetDirectSg          = 1U << 0;
inline constexpr std::uint32_t kRcFlagGetAllowFallbackCopy = 1U << 1;

enum class RcOp : std::uint16_t {
  kGet       = 1,
  kPut       = 2,
  kUpdate    = 3,
  kInitTable = 4,
  kUpdateFlat = 5,
};

enum class RcHashMethod : std::uint8_t {
  kCityHash  = 1,
  kSimpleMod = 2,
};

struct alignas(64) RequestDescriptor {
  std::uint32_t magic          = kRcProtocolMagic;
  std::uint16_t version        = kRcProtocolVersion;
  std::uint16_t op             = static_cast<std::uint16_t>(RcOp::kGet);
  std::uint64_t seq            = 0; // Monotonic lane-local request sequence.
  std::uint32_t shard_id       = 0; // Logical shard targeted by this RPC.
  std::uint32_t client_id      = 0; // Logical client owner of this lane.
  std::uint32_t qp_index       = 0; // Lane index within the client.
  std::uint32_t key_count      = 0; // Number of keys in the payload.
  std::uint32_t value_size     = 0; // Row size in bytes for GET responses.
  std::uint32_t embedding_dim  = 0; // Row size expressed as float count.
  std::uint32_t payload_offset = 0; // Offset from slot base to payload.
  std::uint32_t payload_bytes  = 0; // Bytes occupied by the payload.
  std::uint32_t response_bytes = 0; // Bytes expected in the response payload.
  std::uint32_t reserved0      = 0;
  std::uint64_t client_response_addr =
      0; // Optional client response address for verbs RC.
  std::uint32_t client_response_rkey =
      0;                                // Optional client response remote key.
  std::uint32_t client_status_rkey = 0; // Optional client status remote key.
  std::uint64_t client_status_addr =
      0;                       // Optional client status address for verbs RC.
  std::uint32_t flags     = 0; // Op-specific protocol flags.
  std::uint32_t reserved1 = 0;
  std::array<char, kTableNameBytes>
      table_name{}; // Optional logical table name.
};

struct alignas(64) CommitWord {
  std::atomic<std::uint64_t> seq{0}; // Mirrors RequestDescriptor::seq.
  std::atomic<std::uint32_t> state{
      0}; // READY/DONE state published by client/server.
  std::uint32_t checksum_or_reserved =
      0; // Reserved for future integrity checks.
};

struct alignas(64) StatusWord {
  std::atomic<std::uint64_t> seq{0}; // Mirrors the request seq completed here.
  std::atomic<std::uint32_t> state{
      0}; // DONE when the server has finished writing.
  std::int32_t status          = 0; // RpcStatus value returned by the server.
  std::uint32_t response_bytes = 0; // Payload bytes valid in the response slot.
  std::uint32_t reserved       = 0;
};

static_assert(sizeof(RequestDescriptor) == 192, "RequestDescriptor size");
static_assert(alignof(RequestDescriptor) == 64, "RequestDescriptor align");
static_assert(alignof(CommitWord) == 64, "CommitWord align");
static_assert(alignof(StatusWord) == 64, "StatusWord align");

inline std::size_t Align64(std::size_t value) {
  return (value + 63U) & ~std::size_t{63U};
}

inline std::size_t GetKeysPerRpcByResponseBudget(
    std::size_t value_size, std::size_t mtu_bytes, std::size_t response_mtu) {
  if (value_size == 0 || mtu_bytes == 0 || response_mtu == 0) {
    return 0;
  }
  return (mtu_bytes * response_mtu) / value_size;
}

inline std::size_t GetRequestBytes(std::size_t key_count) {
  return key_count * sizeof(std::uint64_t);
}

inline std::size_t
GetResponseBytes(std::size_t key_count, std::size_t value_size) {
  return key_count * value_size;
}

inline std::size_t
FixedSlotResponseBytes(std::size_t key_count, std::size_t value_size) {
  return GetResponseBytes(key_count, value_size) + sizeof(std::int32_t);
}

inline std::size_t InitTablePayloadBytes() { return sizeof(std::uint64_t) * 3; }

inline std::size_t PutPayloadBudget(std::size_t request_slot_bytes) {
  if (request_slot_bytes <=
      Align64(sizeof(RequestDescriptor)) + Align64(sizeof(CommitWord))) {
    return 0;
  }
  return request_slot_bytes - Align64(sizeof(RequestDescriptor)) -
         Align64(sizeof(CommitWord));
}

inline std::size_t ParameterReaderBytes(const ParameterCompressReader& reader) {
  return static_cast<std::size_t>(reader.byte_size());
}

inline std::size_t PutPayloadBytes(
    base::ConstArray<std::uint64_t> keys,
    const float* values,
    int64_t embedding_dim,
    std::string* payload,
    std::string* error = nullptr) {
  if (payload == nullptr) {
    if (error != nullptr) {
      *error = "payload buffer is null";
    }
    return 0;
  }
  if (keys.Size() != 0 && (values == nullptr || embedding_dim <= 0)) {
    if (error != nullptr) {
      *error = "values pointer or embedding dim is invalid";
    }
    return 0;
  }

  ParameterCompressor compressor;
  CompressEmbeddingRows(
      &compressor, keys.Data(), values, keys.Size(), embedding_dim);

  payload->clear();
  compressor.ToBlock(payload);
  return payload->size();
}

inline std::size_t PutPayloadBytes(
    base::ConstArray<std::uint64_t> keys,
    const base::RecTensor& values,
    std::string* payload,
    std::string* error = nullptr) {
  if (keys.Size() == 0) {
    return PutPayloadBytes(keys, nullptr, 0, payload, error);
  }
  if (values.dtype() != base::DataType::FLOAT32 || values.dim() != 2 ||
      values.shape(0) != static_cast<int64_t>(keys.Size()) ||
      values.data() == nullptr) {
    if (error != nullptr) {
      *error = "keys and values size mismatch";
    }
    return 0;
  }
  return PutPayloadBytes(
      keys, values.data_as<float>(), values.shape(1), payload, error);
}

inline std::size_t UpdatePayloadBytes(
    base::ConstArray<std::uint64_t> keys,
    const float* values,
    int64_t embedding_dim,
    std::string* payload,
    std::string* error = nullptr) {
  return PutPayloadBytes(keys, values, embedding_dim, payload, error);
}

inline std::size_t UpdatePayloadBytes(
    base::ConstArray<std::uint64_t> keys,
    const base::RecTensor& values,
    std::string* payload,
    std::string* error = nullptr) {
  return PutPayloadBytes(keys, values, payload, error);
}

inline std::size_t FlatUpdatePayloadBytes(
    std::size_t key_count, std::size_t embedding_dim) {
  constexpr std::size_t kKeyBytes = sizeof(std::uint64_t);
  constexpr std::size_t kFloatBytes = sizeof(float);
  const std::size_t max_size = std::numeric_limits<std::size_t>::max();
  if (embedding_dim == 0 || embedding_dim > (max_size - kKeyBytes) / kFloatBytes) {
    return 0;
  }
  const std::size_t row_bytes = kKeyBytes + embedding_dim * kFloatBytes;
  return key_count > max_size / row_bytes ? 0 : key_count * row_bytes;
}

inline std::size_t UpdatePayloadBytesFlat(
    base::ConstArray<std::uint64_t> keys,
    const float* values,
    std::size_t embedding_dim,
    std::string* payload,
    std::string* error = nullptr) {
  if (payload == nullptr || (keys.Size() > 0 && values == nullptr)) {
    if (error != nullptr) {
      *error = "payload buffer or values is null";
    }
    return 0;
  }

  const std::size_t payload_bytes =
      FlatUpdatePayloadBytes(keys.Size(), embedding_dim);
  if (payload_bytes == 0) {
    if (error != nullptr) {
      *error = "invalid flat update shape";
    }
    return 0;
  }
  const std::size_t key_bytes = keys.Size() * sizeof(std::uint64_t);
  const std::size_t value_bytes = keys.Size() * embedding_dim * sizeof(float);
  payload->resize(payload_bytes);
  if (key_bytes > 0) {
    std::memcpy(payload->data(), keys.Data(), key_bytes);
    std::memcpy(payload->data() + key_bytes, values, value_bytes);
  }
  return payload->size();
}

inline std::size_t PackFlatUpdatePayloadGather(
    const std::uint64_t* keys,
    const float* values,
    std::size_t num_rows,
    std::size_t embedding_dim,
    const std::size_t* row_indices,
    std::size_t row_count,
    void* payload,
    std::size_t payload_capacity,
    std::string* error = nullptr) {
  if ((row_count > 0 &&
       (keys == nullptr || values == nullptr || row_indices == nullptr)) ||
      payload == nullptr) {
    if (error != nullptr) {
      *error = "flat update gather input is null";
    }
    return 0;
  }
  const std::size_t payload_bytes =
      FlatUpdatePayloadBytes(row_count, embedding_dim);
  if (payload_bytes == 0 || payload_bytes > payload_capacity) {
    if (error != nullptr) {
      *error = "flat update gather payload does not fit";
    }
    return 0;
  }

  auto* payload_keys = static_cast<std::uint64_t*>(payload);
  auto* payload_values = reinterpret_cast<float*>(
      static_cast<char*>(payload) + row_count * sizeof(std::uint64_t));
  const std::size_t row_bytes = embedding_dim * sizeof(float);
  for (std::size_t row = 0; row < row_count; ++row) {
    const std::size_t source_row = row_indices[row];
    if (source_row >= num_rows) {
      if (error != nullptr) {
        *error = "flat update gather row index is out of range";
      }
      return 0;
    }
    payload_keys[row] = keys[source_row];
    std::memcpy(
        payload_values + row * embedding_dim,
        values + source_row * embedding_dim,
        row_bytes);
  }
  return payload_bytes;
}

inline bool CopyTableName(std::string_view table_name,
                          std::array<char, kTableNameBytes>* storage) {
  if (storage == nullptr || table_name.size() >= kTableNameBytes) {
    return false;
  }
  storage->fill('\0');
  std::memcpy(storage->data(), table_name.data(), table_name.size());
  return true;
}

inline std::string_view
DescriptorTableName(const RequestDescriptor& descriptor) {
  return std::string_view(
      descriptor.table_name.data(),
      std::find(
          descriptor.table_name.begin(), descriptor.table_name.end(), '\0') -
          descriptor.table_name.begin());
}

inline bool ValidateRequestDescriptor(
    const RequestDescriptor& descriptor,
    std::size_t request_slot_bytes,
    std::size_t response_slot_bytes,
    std::string* error = nullptr) {
  if (descriptor.magic != kRcProtocolMagic) {
    if (error != nullptr) {
      *error = "bad request magic";
    }
    return false;
  }
  if (descriptor.version != kRcProtocolVersion) {
    if (error != nullptr) {
      *error = "bad request version";
    }
    return false;
  }
  if (descriptor.payload_offset < sizeof(RequestDescriptor) ||
      static_cast<std::size_t>(descriptor.payload_offset) +
              descriptor.payload_bytes >
          request_slot_bytes) {
    if (error != nullptr) {
      *error = "request payload exceeds slot capacity";
    }
    return false;
  }
  if (descriptor.response_bytes > response_slot_bytes) {
    if (error != nullptr) {
      *error = "response exceeds slot capacity";
    }
    return false;
  }
  return true;
}

inline void ResetStatusWord(StatusWord* status, std::uint64_t seq) {
  status->status         = static_cast<std::int32_t>(RpcStatus::kPending);
  status->response_bytes = 0;
  status->seq.store(seq, std::memory_order_release);
  status->state.store(0, std::memory_order_release);
}

inline bool StatusWordDone(const StatusWord& status, std::uint64_t seq) {
  return status.state.load(std::memory_order_acquire) == kRcSlotDone &&
         status.seq.load(std::memory_order_acquire) == seq;
}

} // namespace petps
