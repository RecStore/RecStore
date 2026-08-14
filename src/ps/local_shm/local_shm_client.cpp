#include "ps/local_shm/local_shm_client.h"
#include "ps/local_shm/local_shm_futex.h"
#include "ps/local_shm/local_shm_queue.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

namespace recstore {

namespace {

std::size_t InitTablePayloadBytes(const std::string& table_name) {
  return table_name.size() + sizeof(uint64_t) * 3;
}

std::size_t GetPayloadBytes(std::size_t key_count) {
  return sizeof(uint64_t) * key_count;
}

std::size_t
GetFlatPayloadBytes(std::size_t key_count, std::size_t embedding_dim) {
  return sizeof(uint64_t) * key_count +
         sizeof(float) * key_count * embedding_dim;
}

std::size_t PutPayloadBytes(std::size_t key_count, std::size_t embedding_dim) {
  return sizeof(uint64_t) * key_count +
         sizeof(float) * key_count * embedding_dim;
}

std::size_t UpdateFlatPayloadBytes(const std::string& table_name,
                                   std::size_t key_count,
                                   std::size_t embedding_dim) {
  return table_name.size() + sizeof(uint64_t) * key_count +
         sizeof(float) * key_count * embedding_dim;
}

uint64_t SteadyNowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

double NsToUs(uint64_t ns) { return static_cast<double>(ns) / 1000.0; }

thread_local LocalShmRequestProfile g_last_request_profile;
thread_local LocalShmActiveRequestProfile g_active_request_profile;

void ResetFlatGetHandle(LocalShmFlatGetHandle* handle) {
  if (handle == nullptr) {
    return;
  }
  handle->slot_id       = LocalShmFlatGetHandle::kInvalidSlotId;
  handle->request_id    = 0;
  handle->values        = nullptr;
  handle->num_rows      = 0;
  handle->embedding_dim = 0;
  handle->output_bytes  = 0;
}

bool IsFlatGetHandleClear(const LocalShmFlatGetHandle& handle) {
  return handle.slot_id == LocalShmFlatGetHandle::kInvalidSlotId &&
         handle.request_id == 0 && handle.values == nullptr &&
         handle.num_rows == 0 && handle.embedding_dim == 0 &&
         handle.output_bytes == 0;
}

void MarkError(LocalShmSlotHeader* header, LocalStatusCode code) {
  header->status_code = static_cast<uint32_t>(code);
  header->state.store(static_cast<uint32_t>(LocalSlotState::kError));
}

void MarkActiveRequestOpcode(uint32_t opcode) {
  if (!g_active_request_profile.active) {
    return;
  }
  g_active_request_profile.opcode = opcode;
}

uint32_t ReadQueueSelector(const json& config, uint32_t fallback) {
  if (config.contains("ready_queue_index")) {
    return config["ready_queue_index"].get<uint32_t>();
  }
  if (config.contains("client_rank")) {
    return config["client_rank"].get<uint32_t>();
  }
  if (config.contains("rank")) {
    return config["rank"].get<uint32_t>();
  }
  if (config.contains("local_rank")) {
    return config["local_rank"].get<uint32_t>();
  }
  if (const char* env = std::getenv("RECSTORE_LOCAL_SHM_READY_QUEUE_INDEX")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  if (const char* env = std::getenv("LOCAL_RANK")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  if (const char* env = std::getenv("RANK")) {
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
  }
  return fallback;
}

} // namespace

LocalShmPSClient::LocalShmPSClient(json config) : BasePSClient(config) {
  region_name_ = config.value("region_name", "recstore_local_ps");
  timeout_ms_  = config.value("client_timeout_ms", 30000);
  client_id_   = static_cast<uint32_t>(::getpid());
  if (!region_.Attach(region_name_)) {
    LOG(WARNING) << "LocalShmPSClient failed to attach region " << region_name_;
    return;
  }
  ready_queue_id_ = ResolveReadyQueueId(config);
  thread_ready_queue_sharding_ =
      config.value("thread_ready_queue_sharding", false);
  LOG(INFO) << "LocalShmPSClient attached region=" << region_name_
            << " pid=" << ::getpid() << " client_id=" << client_id_
            << " timeout_ms=" << timeout_ms_
            << " ready_queue_count=" << region_.ready_queue_count()
            << " ready_queue_id=" << ready_queue_id_
            << " thread_ready_queue_sharding=" << thread_ready_queue_sharding_;
}

int LocalShmPSClient::GetParameter(const base::ConstArray<uint64_t>& keys,
                                   float* values) {
  return this->GetParameterFlat(
      keys, values, static_cast<int64_t>(keys.Size()), /*embedding_dim=*/0);
}

bool LocalShmPSClient::GetSlotPayloadRegion(const void** base,
                                            std::size_t* bytes) const {
  if (base == nullptr || bytes == nullptr) {
    return false;
  }
  *base  = nullptr;
  *bytes = 0;
  if (!region_.IsOpen() || region_.slot_count() == 0 ||
      region_.slot_buffer_bytes() == 0) {
    return false;
  }
  *base  = region_.slot_payload(0);
  *bytes = static_cast<std::size_t>(region_.slot_count()) *
           AlignUp(region_.slot_buffer_bytes());
  return *base != nullptr && *bytes > 0;
}

int LocalShmPSClient::GetParameterFlat(
    const base::ConstArray<uint64_t>& keys,
    float* values,
    int64_t num_rows,
    int64_t embedding_dim) {
  if (values == nullptr) {
    LOG(ERROR) << "LocalShmPSClient::GetParameter invalid_state"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_ << " ready_queue_id="
               << ready_queue_id_ << " key_count=" << num_rows
               << " embedding_dim=" << embedding_dim;
    return -1;
  }
  LocalShmFlatGetHandle handle;
  ResetActiveRequestProfile();
  g_active_request_profile.active        = true;
  g_active_request_profile.request_start = std::chrono::steady_clock::now();
  if (SubmitGetParameterFlat(keys, num_rows, embedding_dim, &handle) != 0) {
    return -1;
  }
  const int wait_rc = WaitGetParameterFlat(&handle);
  if (wait_rc == 0 && handle.output_bytes > 0) {
    std::memcpy(
        values, handle.values, static_cast<std::size_t>(handle.output_bytes));
  }
  ReleaseGetParameterFlat(&handle);
  return wait_rc;
}

int LocalShmPSClient::SubmitGetParameterFlat(
    const base::ConstArray<uint64_t>& keys,
    int64_t num_rows,
    int64_t embedding_dim,
    LocalShmFlatGetHandle* handle) {
  if (handle == nullptr) {
    LOG(ERROR) << "LocalShmPSClient::SubmitGetParameterFlat null_handle"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_
               << " ready_queue_id=" << ready_queue_id_;
    return -1;
  }
  if (!IsFlatGetHandleClear(*handle)) {
    LOG(ERROR) << "LocalShmPSClient::SubmitGetParameterFlat handle_not_clear"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_ << " ready_queue_id="
               << ready_queue_id_ << " slot_id=" << handle->slot_id
               << " request_id=" << handle->request_id
               << " output_bytes=" << handle->output_bytes;
    return -1;
  }
  if (!region_.IsOpen()) {
    LOG(ERROR) << "LocalShmPSClient::SubmitGetParameterFlat invalid_state"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_ << " ready_queue_id="
               << ready_queue_id_ << " key_count=" << num_rows
               << " embedding_dim=" << embedding_dim;
    return -1;
  }
  if (num_rows < 0 || keys.Size() != static_cast<size_t>(num_rows)) {
    LOG(ERROR) << "LocalShmPSClient::SubmitGetParameterFlat invalid_row_count"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_
               << " ready_queue_id=" << ready_queue_id_
               << " key_count=" << keys.Size() << " num_rows=" << num_rows
               << " embedding_dim=" << embedding_dim;
    return -1;
  }
  if (embedding_dim < 0) {
    LOG(ERROR)
        << "LocalShmPSClient::SubmitGetParameterFlat invalid_embedding_dim"
        << " pid=" << static_cast<int>(::getpid()) << " backend=local_shm"
        << " region_name=" << region_name_
        << " ready_queue_id=" << ready_queue_id_ << " key_count=" << num_rows
        << " embedding_dim=" << embedding_dim;
    return -1;
  }

  const int slot = AcquireSlot();
  if (slot < 0) {
    LOG(ERROR) << "LocalShmPSClient::SubmitGetParameterFlat acquire_slot_failed"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_ << " ready_queue_id="
               << ready_queue_id_ << " key_count=" << num_rows
               << " embedding_dim=" << embedding_dim;
    return -1;
  }

  auto* header  = region_.slot_header(static_cast<uint32_t>(slot));
  auto* payload = region_.slot_payload(static_cast<uint32_t>(slot));
  const std::size_t input_bytes = GetPayloadBytes(keys.Size());
  const std::size_t payload_bytes =
      GetFlatPayloadBytes(keys.Size(), static_cast<std::size_t>(embedding_dim));
  if (payload_bytes > region_.slot_buffer_bytes()) {
    LOG(ERROR)
        << "LocalShmPSClient::SubmitGetParameterFlat input_too_large"
        << " pid=" << static_cast<int>(::getpid()) << " backend=local_shm"
        << " region_name=" << region_name_
        << " ready_queue_id=" << ready_queue_id_ << " key_count=" << num_rows
        << " embedding_dim=" << embedding_dim << " slot_id=" << slot
        << " input_bytes=" << input_bytes << " payload_bytes=" << payload_bytes
        << " slot_buffer_bytes=" << region_.slot_buffer_bytes();
    MarkError(header, LocalStatusCode::kBufferTooSmall);
    ReleaseSlot(static_cast<uint32_t>(slot));
    return -1;
  }

  const uint64_t request_id = NextRequestId();
  header->opcode            = static_cast<uint32_t>(LocalOpcode::kGet);
  MarkActiveRequestOpcode(header->opcode);
  header->status_code       = static_cast<uint32_t>(LocalStatusCode::kOk);
  header->client_id         = client_id_;
  header->request_id        = request_id;
  header->client_pid        = static_cast<int64_t>(::getpid());
  header->table_name_len    = 0;
  header->key_count         = static_cast<uint32_t>(num_rows);
  header->embedding_dim     = static_cast<uint32_t>(embedding_dim);
  header->input_bytes       = input_bytes;
  header->output_bytes      = 0;
  header->server_seen_epoch = region_.control()->server_epoch;
  header->error_message_len = 0;
  header->client_enqueue_timestamp_ns = 0;
  header->server_dequeue_timestamp_ns = 0;
  header->server_backend_duration_us  = 0;

  if (keys.Size() > 0) {
    std::memcpy(payload, keys.Data(), input_bytes);
  }
  header->state.store(static_cast<uint32_t>(LocalSlotState::kReady));
  const uint32_t ready_queue_id       = CurrentReadyQueueId();
  const auto enqueue_start            = std::chrono::steady_clock::now();
  header->client_enqueue_timestamp_ns = SteadyNowNs();
  auto* ready_header = region_.ready_queue_header(ready_queue_id);
  CHECK(LocalShmQueueEnqueue(ready_header,
                             region_.ready_queue_cells(ready_queue_id),
                             static_cast<uint32_t>(slot)));
  FutexWakeOne(&ready_header->enqueue_pos);
  if (g_active_request_profile.active) {
    g_active_request_profile.enqueue_us =
        std::chrono::duration_cast< std::chrono::duration<double, std::micro>>(
            std::chrono::steady_clock::now() - enqueue_start)
            .count();
  }

  handle->slot_id       = static_cast<uint32_t>(slot);
  handle->request_id    = request_id;
  handle->values        = nullptr;
  handle->num_rows      = num_rows;
  handle->embedding_dim = embedding_dim;
  handle->output_bytes  = 0;
  return 0;
}

int LocalShmPSClient::WaitGetParameterFlat(LocalShmFlatGetHandle* handle) {
  if (handle == nullptr) {
    LOG(ERROR) << "LocalShmPSClient::WaitGetParameterFlat null_handle"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_
               << " ready_queue_id=" << ready_queue_id_;
    return -1;
  }
  if (handle->slot_id == LocalShmFlatGetHandle::kInvalidSlotId ||
      handle->request_id == 0) {
    LOG(ERROR) << "LocalShmPSClient::WaitGetParameterFlat invalid_handle"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_ << " ready_queue_id="
               << ready_queue_id_ << " slot_id=" << handle->slot_id
               << " request_id=" << handle->request_id;
    return -1;
  }
  if (handle->values != nullptr) {
    LOG(ERROR) << "LocalShmPSClient::WaitGetParameterFlat already_waited"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_ << " ready_queue_id="
               << ready_queue_id_ << " slot_id=" << handle->slot_id
               << " request_id=" << handle->request_id;
    return -1;
  }

  auto* header          = region_.slot_header(handle->slot_id);
  const auto wait_start = std::chrono::steady_clock::now();
  const bool ok         = WaitForSlot(handle->slot_id, handle->request_id);
  if (g_active_request_profile.active) {
    g_active_request_profile.wait_us =
        std::chrono::duration_cast< std::chrono::duration<double, std::micro>>(
            std::chrono::steady_clock::now() - wait_start)
            .count();
  }
  if (!ok) {
    LOG(ERROR) << "LocalShmPSClient::WaitGetParameterFlat wait_failed"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_ << " ready_queue_id="
               << ready_queue_id_ << " slot_id=" << handle->slot_id
               << " request_id=" << handle->request_id;
    return -1;
  }
  if (header->status_code != static_cast<uint32_t>(LocalStatusCode::kOk)) {
    LOG(ERROR) << "LocalShmPSClient::WaitGetParameterFlat request_failed"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_ << " ready_queue_id="
               << ready_queue_id_ << " slot_id=" << handle->slot_id
               << " request_id=" << handle->request_id
               << " status_code=" << header->status_code;
    return -1;
  }

  if (handle->embedding_dim > 0) {
    const std::size_t expected_output_bytes =
        sizeof(float) * static_cast<std::size_t>(handle->num_rows) *
        static_cast<std::size_t>(handle->embedding_dim);
    if (header->output_bytes != expected_output_bytes) {
      LOG(ERROR)
          << "LocalShmPSClient::WaitGetParameterFlat output_size_mismatch"
          << " pid=" << static_cast<int>(::getpid()) << " backend=local_shm"
          << " region_name=" << region_name_ << " ready_queue_id="
          << ready_queue_id_ << " slot_id=" << handle->slot_id
          << " request_id=" << handle->request_id
          << " expected_output_bytes=" << expected_output_bytes
          << " actual_output_bytes=" << header->output_bytes
          << " server_embedding_dim=" << header->embedding_dim;
      return -1;
    }
  }

  auto* payload = region_.slot_payload(handle->slot_id);
  handle->values =
      reinterpret_cast<float*>(payload + GetPayloadBytes(handle->num_rows));
  handle->output_bytes = header->output_bytes;
  if (g_active_request_profile.active) {
    FinalizeActiveRequestProfile(
        header, g_active_request_profile.request_start);
  }
  return 0;
}

void LocalShmPSClient::ReleaseGetParameterFlat(LocalShmFlatGetHandle* handle) {
  if (handle == nullptr) {
    LOG(ERROR) << "LocalShmPSClient::ReleaseGetParameterFlat null_handle"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_
               << " ready_queue_id=" << ready_queue_id_;
    return;
  }
  if (handle->slot_id == LocalShmFlatGetHandle::kInvalidSlotId) {
    LOG(ERROR) << "LocalShmPSClient::ReleaseGetParameterFlat invalid_handle"
               << " pid=" << static_cast<int>(::getpid())
               << " backend=local_shm"
               << " region_name=" << region_name_ << " ready_queue_id="
               << ready_queue_id_ << " request_id=" << handle->request_id;
    ResetFlatGetHandle(handle);
    return;
  }
  ReleaseSlot(handle->slot_id);
  ResetFlatGetHandle(handle);
}

int LocalShmPSClient::PutParameter(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  if (!region_.IsOpen()) {
    return -1;
  }
  if (keys.Size() != values.size()) {
    return -1;
  }
  const std::size_t embedding_dim = values.empty() ? 0 : values.front().size();
  for (const auto& row : values) {
    if (row.size() != embedding_dim) {
      return -1;
    }
  }

  const int slot = AcquireSlot();
  if (slot < 0) {
    return -1;
  }
  ResetActiveRequestProfile();
  g_active_request_profile.active        = true;
  g_active_request_profile.request_start = std::chrono::steady_clock::now();
  auto* header  = region_.slot_header(static_cast<uint32_t>(slot));
  auto* payload = region_.slot_payload(static_cast<uint32_t>(slot));
  const std::size_t input_bytes = PutPayloadBytes(keys.Size(), embedding_dim);
  if (input_bytes > region_.slot_buffer_bytes()) {
    MarkError(header, LocalStatusCode::kBufferTooSmall);
    ReleaseSlot(static_cast<uint32_t>(slot));
    return -1;
  }

  const uint64_t request_id = NextRequestId();
  header->opcode            = static_cast<uint32_t>(LocalOpcode::kPut);
  MarkActiveRequestOpcode(header->opcode);
  header->status_code       = static_cast<uint32_t>(LocalStatusCode::kOk);
  header->client_id         = client_id_;
  header->request_id        = request_id;
  header->client_pid        = static_cast<int64_t>(::getpid());
  header->table_name_len    = 0;
  header->key_count         = static_cast<uint32_t>(keys.Size());
  header->embedding_dim     = static_cast<uint32_t>(embedding_dim);
  header->input_bytes       = input_bytes;
  header->output_bytes      = 0;
  header->server_seen_epoch = region_.control()->server_epoch;
  header->error_message_len = 0;
  header->client_enqueue_timestamp_ns = 0;
  header->server_dequeue_timestamp_ns = 0;
  header->server_backend_duration_us  = 0;

  uint8_t* cursor = payload;
  if (keys.Size() > 0) {
    std::memcpy(cursor, keys.Data(), sizeof(uint64_t) * keys.Size());
    cursor += sizeof(uint64_t) * keys.Size();
  }
  for (const auto& row : values) {
    if (!row.empty()) {
      std::memcpy(cursor, row.data(), sizeof(float) * row.size());
      cursor += sizeof(float) * row.size();
    }
  }
  header->state.store(static_cast<uint32_t>(LocalSlotState::kReady));
  const uint32_t ready_queue_id       = CurrentReadyQueueId();
  const auto enqueue_start            = std::chrono::steady_clock::now();
  header->client_enqueue_timestamp_ns = SteadyNowNs();
  auto* ready_header = region_.ready_queue_header(ready_queue_id);
  CHECK(LocalShmQueueEnqueue(ready_header,
                             region_.ready_queue_cells(ready_queue_id),
                             static_cast<uint32_t>(slot)));
  FutexWakeOne(&ready_header->enqueue_pos);
  g_active_request_profile.enqueue_us =
      std::chrono::duration_cast< std::chrono::duration<double, std::micro>>(
          std::chrono::steady_clock::now() - enqueue_start)
          .count();

  const auto wait_start = std::chrono::steady_clock::now();
  const bool ok =
      WaitForSlot(static_cast<uint32_t>(slot), request_id) &&
      header->status_code == static_cast<uint32_t>(LocalStatusCode::kOk);
  g_active_request_profile.wait_us =
      std::chrono::duration_cast< std::chrono::duration<double, std::micro>>(
          std::chrono::steady_clock::now() - wait_start)
          .count();
  FinalizeActiveRequestProfile(header, g_active_request_profile.request_start);
  ReleaseSlot(static_cast<uint32_t>(slot));
  return ok ? 0 : -1;
}

int LocalShmPSClient::UpdateParameter(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>* grads) {
  if (grads == nullptr) {
    return -1;
  }
  if (keys.Size() != grads->size()) {
    return -1;
  }
  const std::size_t embedding_dim = grads->empty() ? 0 : grads->front().size();
  std::vector<float> flat;
  flat.reserve(grads->size() * embedding_dim);
  for (const auto& row : *grads) {
    if (row.size() != embedding_dim) {
      return -1;
    }
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return UpdateParameterFlat(
      table_name,
      keys,
      flat.empty() ? nullptr : flat.data(),
      static_cast<int64_t>(grads->size()),
      static_cast<int64_t>(embedding_dim));
}

int LocalShmPSClient::UpdateParameterFlat(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim) {
  if (!region_.IsOpen() || grads == nullptr || num_rows < 0 ||
      embedding_dim <= 0 || keys.Size() != static_cast<size_t>(num_rows)) {
    return -1;
  }
  const auto known_dim_it = table_embedding_dims_.find(table_name);
  if (known_dim_it != table_embedding_dims_.end() &&
      known_dim_it->second != embedding_dim) {
    LOG(ERROR) << "LocalShmPSClient::UpdateParameterFlat embedding_dim mismatch"
               << " table_name=" << table_name << " expected="
               << known_dim_it->second << " actual=" << embedding_dim;
    return -1;
  }
  const int slot = AcquireSlot();
  if (slot < 0) {
    return -1;
  }
  ResetActiveRequestProfile();
  g_active_request_profile.active        = true;
  g_active_request_profile.request_start = std::chrono::steady_clock::now();

  auto* header  = region_.slot_header(static_cast<uint32_t>(slot));
  auto* payload = region_.slot_payload(static_cast<uint32_t>(slot));
  const std::size_t input_bytes = UpdateFlatPayloadBytes(
      table_name, keys.Size(), static_cast<std::size_t>(embedding_dim));
  if (input_bytes > region_.slot_buffer_bytes()) {
    MarkError(header, LocalStatusCode::kBufferTooSmall);
    ReleaseSlot(static_cast<uint32_t>(slot));
    return -1;
  }

  const uint64_t request_id = NextRequestId();
  header->opcode            = static_cast<uint32_t>(LocalOpcode::kUpdateFlat);
  MarkActiveRequestOpcode(header->opcode);
  header->status_code       = static_cast<uint32_t>(LocalStatusCode::kOk);
  header->client_id         = client_id_;
  header->request_id        = request_id;
  header->client_pid        = static_cast<int64_t>(::getpid());
  header->table_name_len    = static_cast<uint32_t>(table_name.size());
  header->key_count         = static_cast<uint32_t>(keys.Size());
  header->embedding_dim     = static_cast<uint32_t>(embedding_dim);
  header->input_bytes       = input_bytes;
  header->output_bytes      = 0;
  header->server_seen_epoch = region_.control()->server_epoch;
  header->error_message_len = 0;
  header->client_enqueue_timestamp_ns = 0;
  header->server_dequeue_timestamp_ns = 0;
  header->server_backend_duration_us  = 0;

  uint8_t* cursor = payload;
  if (!table_name.empty()) {
    std::memcpy(cursor, table_name.data(), table_name.size());
    cursor += table_name.size();
  }
  if (keys.Size() > 0) {
    std::memcpy(cursor, keys.Data(), sizeof(uint64_t) * keys.Size());
    cursor += sizeof(uint64_t) * keys.Size();
  }
  std::memcpy(cursor,
              grads,
              sizeof(float) * keys.Size() * static_cast<size_t>(embedding_dim));
  header->state.store(static_cast<uint32_t>(LocalSlotState::kReady));
  const uint32_t ready_queue_id       = CurrentReadyQueueId();
  const auto enqueue_start            = std::chrono::steady_clock::now();
  header->client_enqueue_timestamp_ns = SteadyNowNs();
  auto* ready_header = region_.ready_queue_header(ready_queue_id);
  CHECK(LocalShmQueueEnqueue(ready_header,
                             region_.ready_queue_cells(ready_queue_id),
                             static_cast<uint32_t>(slot)));
  FutexWakeOne(&ready_header->enqueue_pos);
  g_active_request_profile.enqueue_us =
      std::chrono::duration_cast< std::chrono::duration<double, std::micro>>(
          std::chrono::steady_clock::now() - enqueue_start)
          .count();

  const auto wait_start = std::chrono::steady_clock::now();
  const bool ok =
      WaitForSlot(static_cast<uint32_t>(slot), request_id) &&
      header->status_code == static_cast<uint32_t>(LocalStatusCode::kOk);
  g_active_request_profile.wait_us =
      std::chrono::duration_cast< std::chrono::duration<double, std::micro>>(
          std::chrono::steady_clock::now() - wait_start)
          .count();
  FinalizeActiveRequestProfile(header, g_active_request_profile.request_start);
  ReleaseSlot(static_cast<uint32_t>(slot));
  return ok ? 0 : -1;
}

int LocalShmPSClient::InitEmbeddingTable(const std::string& table_name,
                                         const EmbeddingTableConfig& config) {
  if (!region_.IsOpen()) {
    return -1;
  }
  const int slot = AcquireSlot();
  if (slot < 0) {
    return -1;
  }
  ResetActiveRequestProfile();
  g_active_request_profile.active        = true;
  g_active_request_profile.request_start = std::chrono::steady_clock::now();

  auto* header  = region_.slot_header(static_cast<uint32_t>(slot));
  auto* payload = region_.slot_payload(static_cast<uint32_t>(slot));
  const std::size_t input_bytes = InitTablePayloadBytes(table_name);
  if (input_bytes > region_.slot_buffer_bytes()) {
    MarkError(header, LocalStatusCode::kBufferTooSmall);
    ReleaseSlot(static_cast<uint32_t>(slot));
    return -1;
  }

  const uint64_t request_id = NextRequestId();
  header->opcode            = static_cast<uint32_t>(LocalOpcode::kInitTable);
  MarkActiveRequestOpcode(header->opcode);
  header->status_code       = static_cast<uint32_t>(LocalStatusCode::kOk);
  header->client_id         = client_id_;
  header->request_id        = request_id;
  header->client_pid        = static_cast<int64_t>(::getpid());
  header->table_name_len    = static_cast<uint32_t>(table_name.size());
  header->key_count         = 0;
  header->embedding_dim     = static_cast<uint32_t>(config.embedding_dim);
  header->input_bytes       = input_bytes;
  header->output_bytes      = 0;
  header->server_seen_epoch = region_.control()->server_epoch;
  header->error_message_len = 0;
  header->client_enqueue_timestamp_ns = 0;
  header->server_dequeue_timestamp_ns = 0;
  header->server_backend_duration_us  = 0;

  uint8_t* cursor = payload;
  if (!table_name.empty()) {
    std::memcpy(cursor, table_name.data(), table_name.size());
    cursor += table_name.size();
  }
  std::memcpy(cursor, &config.num_embeddings, sizeof(config.num_embeddings));
  cursor += sizeof(config.num_embeddings);
  std::memcpy(cursor, &config.embedding_dim, sizeof(config.embedding_dim));
  cursor += sizeof(config.embedding_dim);
  std::memcpy(cursor, &config.table_id, sizeof(config.table_id));
  header->state.store(static_cast<uint32_t>(LocalSlotState::kReady));
  const uint32_t ready_queue_id       = CurrentReadyQueueId();
  const auto enqueue_start            = std::chrono::steady_clock::now();
  header->client_enqueue_timestamp_ns = SteadyNowNs();
  auto* ready_header = region_.ready_queue_header(ready_queue_id);
  CHECK(LocalShmQueueEnqueue(ready_header,
                             region_.ready_queue_cells(ready_queue_id),
                             static_cast<uint32_t>(slot)));
  FutexWakeOne(&ready_header->enqueue_pos);
  g_active_request_profile.enqueue_us =
      std::chrono::duration_cast< std::chrono::duration<double, std::micro>>(
          std::chrono::steady_clock::now() - enqueue_start)
          .count();

  const auto wait_start = std::chrono::steady_clock::now();
  const bool ok =
      WaitForSlot(static_cast<uint32_t>(slot), request_id) &&
      header->status_code == static_cast<uint32_t>(LocalStatusCode::kOk);
  g_active_request_profile.wait_us =
      std::chrono::duration_cast< std::chrono::duration<double, std::micro>>(
          std::chrono::steady_clock::now() - wait_start)
          .count();
  FinalizeActiveRequestProfile(header, g_active_request_profile.request_start);
  const int tag = ok ? static_cast<int>(header->user_tag) : -1;
  ReleaseSlot(static_cast<uint32_t>(slot));
  if (ok) {
    table_embedding_dims_[table_name] = config.embedding_dim;
  }
  return tag;
}

int LocalShmPSClient::AsyncGetParameter(const base::ConstArray<uint64_t>& keys,
                                        float* values) {
  return GetParameter(keys, values);
}

void LocalShmPSClient::Command(PSCommand) {}

uint64_t
LocalShmPSClient::PrefetchParameter(const base::ConstArray<uint64_t>&) {
  return 0;
}

bool LocalShmPSClient::IsPrefetchDone(uint64_t) { return false; }

void LocalShmPSClient::WaitForPrefetch(uint64_t) {}

bool LocalShmPSClient::GetPrefetchResult(uint64_t,
                                         std::vector<std::vector<float>>*) {
  return false;
}

bool LocalShmPSClient::GetPrefetchResultFlat(
    uint64_t, std::vector<float>*, int64_t*, int64_t) {
  return false;
}

int LocalShmPSClient::AcquireSlot() {
  if (!region_.IsOpen()) {
    return -1;
  }
  const uint32_t free_queue_id = CurrentReadyQueueId();
  auto* queue_header           = region_.free_queue_header(free_queue_id);
  auto* queue_cells            = region_.free_queue_cells(free_queue_id);
  const auto acquire_start     = std::chrono::steady_clock::now();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);

  while (std::chrono::steady_clock::now() < deadline) {
    uint32_t slot_id = 0;
    if (LocalShmQueueDequeue(queue_header, queue_cells, &slot_id)) {
      auto* header = region_.slot_header(slot_id);
      header->state.store(static_cast<uint32_t>(LocalSlotState::kWriting),
                          std::memory_order_release);
      if (g_active_request_profile.active) {
        g_active_request_profile.acquire_slot_us =
            std::chrono::duration_cast<
                std::chrono::duration<double, std::micro>>(
                std::chrono::steady_clock::now() - acquire_start)
                .count();
      }
      return static_cast<int>(slot_id);
    }
    const uint32_t observed =
        queue_header->enqueue_pos.load(std::memory_order_acquire);
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      break;
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    FutexWaitUntilValueChange(&queue_header->enqueue_pos, observed, remaining);
  }
  return -1;
}

void LocalShmPSClient::ReleaseSlot(uint32_t slot_id) {
  const auto release_start = std::chrono::steady_clock::now();
  auto* header             = region_.slot_header(slot_id);
  header->state.store(
      static_cast<uint32_t>(LocalSlotState::kFree), std::memory_order_release);
  const uint32_t free_queue_id = slot_id % region_.ready_queue_count();
  auto* free_header            = region_.free_queue_header(free_queue_id);
  CHECK(LocalShmQueueEnqueue(
      free_header, region_.free_queue_cells(free_queue_id), slot_id));
  FutexWakeOne(&free_header->enqueue_pos);
  g_last_request_profile.release_us =
      std::chrono::duration_cast< std::chrono::duration<double, std::micro>>(
          std::chrono::steady_clock::now() - release_start)
          .count();
}

bool LocalShmPSClient::WaitForSlot(uint32_t slot_id, uint64_t request_id) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);
  auto* header = region_.slot_header(slot_id);
  while (std::chrono::steady_clock::now() < deadline) {
    const uint32_t state = header->state.load();
    if (header->request_id == request_id &&
        (state == static_cast<uint32_t>(LocalSlotState::kDone) ||
         state == static_cast<uint32_t>(LocalSlotState::kError))) {
      return true;
    }

    const uint32_t observed =
        header->completion_doorbell.load(std::memory_order_acquire);
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      break;
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    FutexWaitUntilValueChange(
        &header->completion_doorbell, observed, remaining);
  }
  LOG(ERROR)
      << "LocalShmPSClient::WaitForSlot timeout request_id=" << request_id
      << " slot_id=" << slot_id << " state=" << header->state.load()
      << " status_code=" << header->status_code
      << " output_bytes=" << header->output_bytes << " completion_doorbell="
      << header->completion_doorbell.load(std::memory_order_acquire)
      << " server_epoch=" << region_.control()->server_epoch;
  header->status_code = static_cast<uint32_t>(LocalStatusCode::kTimeout);
  return false;
}

uint64_t LocalShmPSClient::NextRequestId() {
  return region_.control()->next_request_id.fetch_add(1);
}

uint32_t LocalShmPSClient::ResolveReadyQueueId(const json& config) const {
  const uint32_t queue_count = region_.ready_queue_count();
  CHECK_GT(queue_count, 0U);
  return ReadQueueSelector(config, client_id_) % queue_count;
}

uint32_t LocalShmPSClient::CurrentReadyQueueId() const {
  const uint32_t queue_count = region_.ready_queue_count();
  CHECK_GT(queue_count, 0U);
  if (!thread_ready_queue_sharding_ || queue_count == 1) {
    return ready_queue_id_;
  }
  static std::atomic<uint32_t> next_thread_queue{0};
  thread_local const uint32_t thread_queue =
      next_thread_queue.fetch_add(1, std::memory_order_relaxed);
  return (ready_queue_id_ + thread_queue) % queue_count;
}

LocalShmRequestProfile LocalShmPSClient::GetLastRequestProfile() const {
  return g_last_request_profile;
}

void LocalShmPSClient::ResetActiveRequestProfile() const {
  g_last_request_profile   = LocalShmRequestProfile{};
  g_active_request_profile = LocalShmActiveRequestProfile{};
}

void LocalShmPSClient::FinalizeActiveRequestProfile(
    const LocalShmSlotHeader* header,
    const std::chrono::steady_clock::time_point& request_start) const {
  g_last_request_profile.opcode = g_active_request_profile.opcode;
  g_last_request_profile.acquire_slot_us =
      g_active_request_profile.acquire_slot_us;
  g_last_request_profile.enqueue_us = g_active_request_profile.enqueue_us;
  g_last_request_profile.wait_us    = g_active_request_profile.wait_us;
  g_last_request_profile.request_total_us =
      std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
          std::chrono::steady_clock::now() - request_start)
          .count();
  g_last_request_profile.server_backend_us =
      static_cast<double>(header->server_backend_duration_us);
  if (header->client_enqueue_timestamp_ns != 0 &&
      header->server_dequeue_timestamp_ns >=
          header->client_enqueue_timestamp_ns) {
    g_last_request_profile.server_queue_wait_us =
        NsToUs(header->server_dequeue_timestamp_ns -
               header->client_enqueue_timestamp_ns);
  }
  g_active_request_profile.active = false;
}

FACTORY_REGISTER(BasePSClient, local_shm, LocalShmPSClient, json);

} // namespace recstore
