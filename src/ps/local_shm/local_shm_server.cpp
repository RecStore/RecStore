#include "ps/local_shm/local_shm_server.h"
#include "ps/local_shm/local_shm_futex.h"
#include "ps/local_shm/local_shm_queue.h"
#include "ps/local_shm/local_shm_stage_report.h"

#include <cstring>
#include <thread>
#include <vector>

#include "base/factory.h"
#include "base/log.h"
#include "ps/base/cache_ps_impl.h"

namespace recstore {

namespace {

uint64_t SteadyNowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void FinishWithStatus(LocalShmSlotHeader* header, LocalStatusCode code) {
  header->status_code = static_cast<uint32_t>(code);
  header->state.store(
      code == LocalStatusCode::kOk
          ? static_cast<uint32_t>(LocalSlotState::kDone)
          : static_cast<uint32_t>(LocalSlotState::kError),
      std::memory_order_release);
  header->completion_doorbell.fetch_add(1, std::memory_order_release);
  FutexWakeOne(&header->completion_doorbell);
}

} // namespace

LocalShmStoreRuntime::LocalShmStoreRuntime(
    LocalShmRegion* region,
    ::CachePS* cache_ps,
    uint32_t ready_queue_id,
    uint32_t worker_tid,
    uint32_t ready_queue_burst_limit)
    : region_(region),
      cache_ps_(cache_ps),
      ready_queue_id_(ready_queue_id),
      worker_tid_(worker_tid),
      ready_queue_burst_limit_(std::max<uint32_t>(1, ready_queue_burst_limit)) {
}

void LocalShmStoreRuntime::Run() {
  auto* ready_header = region_->ready_queue_header(ready_queue_id_);
  while (!stop_.load()) {
    const uint32_t observed_before_wait =
        ready_header->enqueue_pos.load(std::memory_order_acquire);
    uint32_t processed = 0;
    DrainReadyQueue(ready_queue_id_, &processed);
    if (processed == 0) {
      if (!stop_.load(std::memory_order_acquire)) {
        FutexWaitUntilValueChange(
            &ready_header->enqueue_pos,
            observed_before_wait,
            std::chrono::milliseconds(100));
      }
    }
  }
}

bool LocalShmStoreRuntime::DrainReadyQueue(uint32_t ready_queue_id,
                                           uint32_t* processed) {
  auto* ready_header   = region_->ready_queue_header(ready_queue_id);
  auto* ready_cells    = region_->ready_queue_cells(ready_queue_id);
  uint32_t slot_id     = 0;
  bool found_work      = false;
  uint32_t burst_count = 0;
  while (burst_count < ready_queue_burst_limit_ &&
         LocalShmQueueDequeue(ready_header, ready_cells, &slot_id)) {
    auto* header      = region_->slot_header(slot_id);
    uint32_t expected = static_cast<uint32_t>(LocalSlotState::kReady);
    if (header->state.compare_exchange_strong(
            expected, static_cast<uint32_t>(LocalSlotState::kRunning))) {
      header->server_dequeue_timestamp_ns = SteadyNowNs();
      ProcessSlot(slot_id);
      ++(*processed);
      ++burst_count;
      found_work = true;
    }
  }
  return found_work;
}

void LocalShmStoreRuntime::Stop() {
  stop_.store(true, std::memory_order_release);
  auto* ready_header = region_->ready_queue_header(ready_queue_id_);
  ready_header->enqueue_pos.fetch_add(1, std::memory_order_release);
  FutexWakeOne(&ready_header->enqueue_pos);
}

void LocalShmStoreRuntime::ProcessSlot(uint32_t slot_id) {
  auto* header             = region_->slot_header(slot_id);
  auto* payload            = region_->slot_payload(slot_id);
  const auto process_start = std::chrono::steady_clock::now();
  LocalShmStageReportScope stage_scope(
      header->request_id,
      header->opcode,
      ready_queue_id_,
      header->key_count,
      header->embedding_dim);
  try {
    switch (static_cast<LocalOpcode>(header->opcode)) {
    case LocalOpcode::kInitTable: {
      const std::string table_name(
          reinterpret_cast<const char*>(payload), header->table_name_len);
      const uint8_t* cursor   = payload + header->table_name_len;
      uint64_t num_embeddings = 0;
      uint64_t embedding_dim  = 0;
      uint64_t table_id       = 0;
      std::memcpy(&num_embeddings, cursor, sizeof(num_embeddings));
      cursor += sizeof(num_embeddings);
      std::memcpy(&embedding_dim, cursor, sizeof(embedding_dim));
      cursor += sizeof(embedding_dim);
      if (header->input_bytes >=
          header->table_name_len + sizeof(uint64_t) * 3) {
        std::memcpy(&table_id, cursor, sizeof(table_id));
      }
      const auto backend_start = std::chrono::steady_clock::now();
      const int tag =
          cache_ps_->InitTable(table_name, num_embeddings, embedding_dim, table_id);
      header->user_tag = tag >= 0 ? static_cast<uint64_t>(tag) : 0;
      header->server_backend_duration_us =
          static_cast<uint64_t>(LocalShmElapsedUs(backend_start));
      ReportLocalShmStageMetric(
          "init_table_backend_us", LocalShmElapsedUs(backend_start));
      ReportLocalShmStageMetric(
          "server_process_total_us", LocalShmElapsedUs(process_start));
      FinishWithStatus(
          header, tag >= 0 ? LocalStatusCode::kOk : LocalStatusCode::kUnknownError);
      return;
    }
    case LocalOpcode::kGet: {
      const auto* keys = reinterpret_cast<const uint64_t*>(payload);
      const base::ConstArray<uint64_t> key_array(keys, header->key_count);
      const uint32_t embedding_dim = header->embedding_dim;
      if (embedding_dim == 0) {
        std::vector<ParameterPack> packs;
        packs.reserve(header->key_count);
        const auto backend_start = std::chrono::steady_clock::now();
        if (!cache_ps_->GetParameterRun2Completion(
                key_array, packs, static_cast<int>(worker_tid_))) {
          LOG(ERROR) << "LocalShmStoreRuntime::ProcessSlot get_failed"
                     << " slot_id=" << slot_id << " request_id="
                     << header->request_id << " key_count=" << header->key_count
                     << " computed_embedding_dim=0"
                     << " output_bytes=0"
                     << " reason=GetParameterRun2Completion";
          ReportLocalShmStageMetric(
              "get_backend_us", LocalShmElapsedUs(backend_start));
          ReportLocalShmStageMetric(
              "server_process_total_us", LocalShmElapsedUs(process_start));
          FinishWithStatus(header, LocalStatusCode::kUnknownError);
          return;
        }
        header->server_backend_duration_us =
            static_cast<uint64_t>(LocalShmElapsedUs(backend_start));
        ReportLocalShmStageMetric(
            "get_backend_us", LocalShmElapsedUs(backend_start));
        int64_t max_embedding_dim = 0;
        for (const auto& pack : packs) {
          max_embedding_dim = std::max<int64_t>(max_embedding_dim, pack.dim);
        }
        const std::size_t output_bytes =
            sizeof(float) * packs.size() *
            static_cast<std::size_t>(max_embedding_dim);
        const std::size_t key_bytes =
            sizeof(uint64_t) * static_cast<std::size_t>(header->key_count);
        const std::size_t payload_bytes = key_bytes + output_bytes;
        if (payload_bytes > region_->slot_buffer_bytes()) {
          LOG(ERROR) << "LocalShmStoreRuntime::ProcessSlot get_buffer_too_small"
                     << " slot_id=" << slot_id << " request_id="
                     << header->request_id << " key_count=" << header->key_count
                     << " computed_embedding_dim=" << max_embedding_dim
                     << " output_bytes=" << output_bytes
                     << " payload_bytes=" << payload_bytes
                     << " slot_buffer_bytes=" << region_->slot_buffer_bytes();
          ReportLocalShmStageMetric(
              "server_process_total_us", LocalShmElapsedUs(process_start));
          FinishWithStatus(header, LocalStatusCode::kBufferTooSmall);
          return;
        }
        const auto payload_pack_start = std::chrono::steady_clock::now();
        uint8_t* output               = payload + key_bytes;
        std::memset(output, 0, output_bytes);
        float* out = reinterpret_cast<float*>(output);
        for (std::size_t row = 0; row < packs.size(); ++row) {
          if (packs[row].dim > 0 && packs[row].emb_data != nullptr) {
            std::copy_n(
                packs[row].emb_data,
                packs[row].dim,
                out + row * static_cast<std::size_t>(max_embedding_dim));
          }
        }
        ReportLocalShmStageMetric(
            "get_payload_pack_us", LocalShmElapsedUs(payload_pack_start));
        header->embedding_dim = static_cast<uint32_t>(max_embedding_dim);
        header->output_bytes  = output_bytes;
        ReportLocalShmStageMetric(
            "server_process_total_us", LocalShmElapsedUs(process_start));
        FinishWithStatus(header, LocalStatusCode::kOk);
        return;
      }

      const std::size_t output_bytes =
          sizeof(float) * static_cast<std::size_t>(header->key_count) *
          static_cast<std::size_t>(embedding_dim);
      const std::size_t key_bytes =
          sizeof(uint64_t) * static_cast<std::size_t>(header->key_count);
      const std::size_t payload_bytes = key_bytes + output_bytes;
      if (payload_bytes > region_->slot_buffer_bytes()) {
        LOG(ERROR) << "LocalShmStoreRuntime::ProcessSlot get_buffer_too_small"
                   << " slot_id=" << slot_id << " request_id="
                   << header->request_id << " key_count=" << header->key_count
                   << " computed_embedding_dim=" << embedding_dim
                   << " output_bytes=" << output_bytes
                   << " payload_bytes=" << payload_bytes
                   << " slot_buffer_bytes=" << region_->slot_buffer_bytes();
        ReportLocalShmStageMetric(
            "server_process_total_us", LocalShmElapsedUs(process_start));
        FinishWithStatus(header, LocalStatusCode::kBufferTooSmall);
        return;
      }
      const auto backend_start = std::chrono::steady_clock::now();
      if (!cache_ps_->GetParameterFlat(
              key_array,
              reinterpret_cast<float*>(payload + key_bytes),
              static_cast<int64_t>(header->key_count),
              static_cast<int64_t>(embedding_dim),
              static_cast<int>(worker_tid_))) {
        LOG(ERROR) << "LocalShmStoreRuntime::ProcessSlot get_failed"
                   << " slot_id=" << slot_id << " request_id="
                   << header->request_id << " key_count=" << header->key_count
                   << " computed_embedding_dim=" << embedding_dim
                   << " output_bytes=" << output_bytes
                   << " reason=GetParameterFlat";
        ReportLocalShmStageMetric(
            "get_backend_us", LocalShmElapsedUs(backend_start));
        ReportLocalShmStageMetric(
            "server_process_total_us", LocalShmElapsedUs(process_start));
        FinishWithStatus(header, LocalStatusCode::kUnknownError);
        return;
      }
      header->server_backend_duration_us =
          static_cast<uint64_t>(LocalShmElapsedUs(backend_start));
      ReportLocalShmStageMetric(
          "get_backend_us", LocalShmElapsedUs(backend_start));
      header->output_bytes = output_bytes;
      ReportLocalShmStageMetric(
          "server_process_total_us", LocalShmElapsedUs(process_start));
      FinishWithStatus(header, LocalStatusCode::kOk);
      return;
    }
    case LocalOpcode::kPut: {
      const auto* keys   = reinterpret_cast<const uint64_t*>(payload);
      const auto* values = reinterpret_cast<const float*>(
          payload + sizeof(uint64_t) * header->key_count);
      const auto backend_start = std::chrono::steady_clock::now();
      cache_ps_->PutDenseParameterBatch(
          keys,
          values,
          static_cast<int>(header->key_count),
          static_cast<int>(header->embedding_dim),
          static_cast<int>(worker_tid_));
      header->server_backend_duration_us =
          static_cast<uint64_t>(LocalShmElapsedUs(backend_start));
      ReportLocalShmStageMetric(
          "put_backend_us", LocalShmElapsedUs(backend_start));
      ReportLocalShmStageMetric(
          "server_process_total_us", LocalShmElapsedUs(process_start));
      FinishWithStatus(header, LocalStatusCode::kOk);
      return;
    }
    case LocalOpcode::kUpdateFlat: {
      const std::string table_name(
          reinterpret_cast<const char*>(payload), header->table_name_len);
      const uint8_t* cursor = payload + header->table_name_len;
      const auto* keys      = reinterpret_cast<const uint64_t*>(cursor);
      cursor += sizeof(uint64_t) * header->key_count;
      const auto* grads = reinterpret_cast<const float*>(cursor);
      const base::ConstArray<uint64_t> key_array(keys, header->key_count);
      const auto backend_start = std::chrono::steady_clock::now();
      const bool ok            = cache_ps_->UpdateParameterFlat(
          table_name,
          key_array,
          grads,
          static_cast<int64_t>(header->key_count),
          static_cast<int64_t>(header->embedding_dim),
          static_cast<int>(worker_tid_));
      header->server_backend_duration_us =
          static_cast<uint64_t>(LocalShmElapsedUs(backend_start));
      ReportLocalShmStageMetric(
          "update_backend_us", LocalShmElapsedUs(backend_start));
      ReportLocalShmStageMetric(
          "server_process_total_us", LocalShmElapsedUs(process_start));
      FinishWithStatus(
          header, ok ? LocalStatusCode::kOk : LocalStatusCode::kUnknownError);
      return;
    }
    default:
      ReportLocalShmStageMetric(
          "server_process_total_us", LocalShmElapsedUs(process_start));
      FinishWithStatus(header, LocalStatusCode::kUnsupportedOpcode);
      return;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "LocalShmStoreRuntime::ProcessSlot exception slot_id="
               << slot_id << " request_id=" << header->request_id
               << " opcode=" << header->opcode
               << " key_count=" << header->key_count << " what=" << e.what();
    ReportLocalShmStageMetric(
        "server_process_total_us", LocalShmElapsedUs(process_start));
    FinishWithStatus(header, LocalStatusCode::kUnknownError);
    return;
  }
}

void LocalShmParameterServer::Init(const json& config) {
  config_ = config;
  local_config_ =
      config.contains("local_shm") ? config["local_shm"] : json::object();
  const uint32_t slot_count = local_config_.value("slot_count", 64);
  const uint32_t slot_buffer_bytes =
      local_config_.value("slot_buffer_bytes", 8 * 1024 * 1024);
  const uint32_t ready_queue_count =
      local_config_.value("ready_queue_count", 1);
  const uint32_t ready_queue_burst_limit =
      std::max<uint32_t>(1, local_config_.value("ready_queue_burst_limit", 8));
  const std::string region_name =
      local_config_.value("region_name", "recstore_local_ps");

  region_ = std::make_unique<LocalShmRegion>();
  CHECK(region_->Create(
      region_name, slot_count, slot_buffer_bytes, ready_queue_count));
  json cache_ps_config =
      config_.contains("cache_ps") ? config_["cache_ps"] : json::object();
  const uint32_t configured_num_threads =
      cache_ps_config.value("num_threads", 1U);
  cache_ps_config["num_threads"] =
      std::max<uint32_t>(configured_num_threads, ready_queue_count);
  cache_ps_ = std::make_shared<CachePS>(cache_ps_config);
  runtimes_.clear();
  runtimes_.reserve(ready_queue_count);
  for (uint32_t ready_queue_id = 0; ready_queue_id < ready_queue_count;
       ++ready_queue_id) {
    runtimes_.push_back(std::make_unique<LocalShmStoreRuntime>(
        region_.get(),
        cache_ps_.get(),
        ready_queue_id,
        ready_queue_id,
        ready_queue_burst_limit));
  }
}

LocalShmParameterServer::~LocalShmParameterServer() = default;

void LocalShmParameterServer::Run() {
  CHECK(!runtimes_.empty());
  worker_threads_.clear();
  worker_threads_.reserve(runtimes_.size() > 0 ? runtimes_.size() - 1 : 0);
  for (std::size_t idx = 1; idx < runtimes_.size(); ++idx) {
    worker_threads_.emplace_back([this, idx]() { runtimes_[idx]->Run(); });
  }
  runtimes_[0]->Run();
  for (auto& worker : worker_threads_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  worker_threads_.clear();
}

void LocalShmParameterServer::Stop() {
  for (const auto& runtime : runtimes_) {
    if (runtime != nullptr) {
      runtime->Stop();
    }
  }
}

FACTORY_REGISTER(
    BaseParameterServer, LocalShmParameterServer, LocalShmParameterServer);

} // namespace recstore
