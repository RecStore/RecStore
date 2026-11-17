#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace recstore {

constexpr uint64_t kLocalShmMagic         = 0x52454353544F5245ULL; // "RECSTORE"
constexpr uint32_t kLocalShmVersion       = 2;
constexpr uint32_t kLocalShmReservedBytes = 64;

enum class LocalQueueKind : uint32_t {
  kFree = 0,
};

enum class LocalOpcode : uint32_t {
  kInvalid              = 0,
  kInitTable            = 1,
  kGet                  = 2,
  kPut                  = 3,
  kUpdateFlat           = 4,
  kSparseGradientCommit = 5,
};

enum class LocalSlotState : uint32_t {
  kFree    = 0,
  kWriting = 1,
  kReady   = 2,
  kRunning = 3,
  kDone    = 4,
  kError   = 5,
};

enum class LocalStatusCode : uint32_t {
  kOk                = 0,
  kUnknownError      = 1,
  kInvalidRequest    = 2,
  kBufferTooSmall    = 3,
  kEpochMismatch     = 4,
  kTimeout           = 5,
  kUnsupportedOpcode = 6,
};

struct alignas(64) LocalShmControlBlock {
  uint64_t magic;
  uint32_t version;
  uint32_t reserved0;

  uint64_t server_epoch;
  int64_t server_pid;

  uint32_t slot_count;
  uint32_t slot_buffer_bytes;
  uint32_t ready_queue_count;
  uint32_t reserved1;

  std::atomic<uint64_t> next_request_id;
  std::atomic<uint32_t> fatal_status;
  std::atomic<uint32_t> request_doorbell;
  std::atomic<uint32_t> free_doorbell;
  std::atomic<uint32_t> active_clients;

  uint8_t reserved[kLocalShmReservedBytes];
};

struct alignas(64) LocalShmQueueHeader {
  std::atomic<uint32_t> enqueue_pos;
  std::atomic<uint32_t> dequeue_pos;
  uint32_t capacity;
  uint32_t reserved0;

  uint8_t reserved[kLocalShmReservedBytes];
};

struct alignas(16) LocalShmQueueCell {
  std::atomic<uint32_t> sequence;
  uint32_t value;
  uint64_t reserved0;
};

struct alignas(64) LocalShmSlotHeader {
  std::atomic<uint32_t> state;
  uint32_t opcode;

  uint32_t status_code;
  uint32_t client_id;

  uint64_t request_id;
  int64_t client_pid;

  uint32_t table_name_len;
  uint32_t key_count;

  uint32_t embedding_dim;
  // Client optimizer learning rate; 0 means "not advertised".
  float learning_rate;

  uint64_t input_bytes;
  uint64_t output_bytes;

  uint64_t server_seen_epoch;
  uint64_t user_tag;
  uint64_t client_enqueue_timestamp_ns;
  uint64_t server_dequeue_timestamp_ns;
  uint64_t server_backend_duration_us;

  std::atomic<uint32_t> completion_doorbell;
  uint32_t error_message_len;

  uint8_t reserved[kLocalShmReservedBytes - sizeof(uint64_t) * 3];
};

static_assert(std::is_standard_layout<LocalShmControlBlock>::value,
              "LocalShmControlBlock must be standard layout");
static_assert(std::is_standard_layout<LocalShmQueueHeader>::value,
              "LocalShmQueueHeader must be standard layout");
static_assert(std::is_standard_layout<LocalShmQueueCell>::value,
              "LocalShmQueueCell must be standard layout");
static_assert(std::is_standard_layout<LocalShmSlotHeader>::value,
              "LocalShmSlotHeader must be standard layout");

} // namespace recstore
