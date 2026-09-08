#include "ps/rdma/rdma_distributed_executor.h"

#include <exception>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

namespace recstore {

RdmaDistributedExecutor::RdmaDistributedExecutor(
    std::vector<petps::PetPSClient*> clients,
    int value_size,
    int first_batch_id,
    int batch_id_step)
    : clients_(std::move(clients)),
      value_size_(value_size),
      next_id_(first_batch_id),
      id_step_(batch_id_step) {
  if (clients_.empty() || value_size_ <= 0 ||
      (id_step_ != 1 && id_step_ != -1)) {
    throw std::invalid_argument("invalid RDMA distributed executor config");
  }
}

RdmaDistributedExecutor::~RdmaDistributedExecutor() {
  for (auto& [batch_id, batch] : batches_) {
    (void)batch_id;
    try {
      ReleaseShardRpcs(&batch.shard_rpcs);
    } catch (const std::exception& error) {
      LOG(ERROR) << "failed to release RDMA batch during shutdown: "
                 << error.what();
    }
  }
  for (auto& [update_id, update] : pending_updates_) {
    (void)update_id;
    for (const auto& [client_index, rpc_id] : update.shard_rpcs) {
      try {
        ClientAt(client_index)->RevokeRPCResource(rpc_id);
      } catch (const std::exception& error) {
        LOG(ERROR) << "failed to release RDMA update during shutdown: "
                   << error.what();
      }
    }
  }
}

petps::PetPSClient* RdmaDistributedExecutor::ClientAt(int client_index) const {
  if (client_index < 0 || client_index >= static_cast<int>(clients_.size()) ||
      clients_[static_cast<std::size_t>(client_index)] == nullptr) {
    throw std::runtime_error("RDMA batch references an invalid shard client");
  }
  return clients_[static_cast<std::size_t>(client_index)];
}

int RdmaDistributedExecutor::RegisterBatch(shard_routing::BatchRequest batch) {
  std::unique_lock<std::mutex> guard(batches_mu_);
  if (next_id_ < std::numeric_limits<int>::min() ||
      next_id_ > std::numeric_limits<int>::max()) {
    guard.unlock();
    ReleaseShardRpcs(&batch.shard_rpcs);
    throw std::runtime_error("RDMA distributed batch handle space exhausted");
  }
  const int batch_id = static_cast<int>(next_id_);
  next_id_ += id_step_;
  if (batches_.find(batch_id) != batches_.end()) {
    guard.unlock();
    ReleaseShardRpcs(&batch.shard_rpcs);
    throw std::runtime_error("duplicate RDMA distributed batch handle");
  }
  if (!batches_.emplace(batch_id, std::move(batch)).second) {
    throw std::runtime_error("duplicate RDMA distributed batch handle");
  }
  return batch_id;
}

int RdmaDistributedExecutor::SubmitGetParameter(
    const std::vector<shard_routing::ShardChunk>& chunks,
    std::size_t total_key_count,
    int value_size,
    float* values,
    bool is_async,
    int async_req_id,
    std::size_t max_in_flight) {
  if (values == nullptr || value_size <= 0 || max_in_flight == 0) {
    throw std::invalid_argument("invalid RDMA GET request");
  }

  shard_routing::BatchRequest batch;
  batch.user_buffer     = values;
  batch.total_key_count = total_key_count;
  batch.value_size      = value_size;
  shard_routing::WriteBatchStatus(&batch, value_size);
  batch.receive_buffers.reserve(chunks.size());
  batch.shard_rpcs.reserve(chunks.size());
  std::vector<shard_routing::PendingShardRpc> window;
  window.reserve(std::min(max_in_flight, chunks.size()));

  auto drain_window = [this, &batch, &window]() {
    WaitShardRpcs(window);
    ReleaseShardRpcs(&window);
    for (auto& pending : window) {
      batch.shard_rpcs.push_back(std::move(pending));
    }
    window.clear();
  };

  try {
    for (std::size_t chunk_index = 0; chunk_index < chunks.size();
         ++chunk_index) {
      const auto& chunk = chunks[chunk_index];
      if (chunk.keys.empty() || chunk.keys.size() != chunk.positions.size()) {
        throw std::runtime_error("RDMA GET chunk shape mismatch");
      }
      for (const std::size_t position : chunk.positions) {
        if (position >= total_key_count) {
          throw std::invalid_argument(
              "RDMA GET chunk position is out of range");
        }
      }
      auto* client = ClientAt(chunk.client_index);
      batch.receive_buffers.emplace_back(
          chunk.keys.size() * static_cast<std::size_t>(value_size) +
              sizeof(std::int32_t),
          0);
      void* recv = batch.receive_buffers.back().data();
      shard_routing::PendingShardRpc pending{
          chunk.shard_id,
          chunk.client_index,
          -1,
          chunk.positions,
          recv,
          chunk.keys.size(),
      };
      pending.rpc_id = client->GetParameter(
          base::ConstArray<std::uint64_t>(chunk.keys),
          static_cast<float*>(recv),
          is_async,
          async_req_id,
          value_size / static_cast<int>(sizeof(float)));
      if (pending.rpc_id < 0) {
        throw std::runtime_error("failed to submit RDMA GET");
      }
      window.push_back(std::move(pending));
      if (window.size() >= max_in_flight && chunk_index + 1 < chunks.size()) {
        drain_window();
      }
    }
  } catch (...) {
    ReleaseShardRpcs(&window);
    ReleaseShardRpcs(&batch.shard_rpcs);
    throw;
  }
  for (auto& pending : window) {
    batch.shard_rpcs.push_back(std::move(pending));
  }

  const int batch_id = RegisterBatch(std::move(batch));
  if (!is_async) {
    WaitBatch(batch_id);
  }
  return batch_id;
}

bool RdmaDistributedExecutor::FinalizeBatch(
    shard_routing::BatchRequest* batch) const {
  if (batch->assembled) {
    return batch->status_code ==
           static_cast<std::int32_t>(petps::RpcStatus::kOk);
  }
  batch->status_code =
      shard_routing::DecodeBatchStatus(*batch, batch->value_size);
  if (batch->status_code == static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
    shard_routing::MergeBatchRows(batch, batch->value_size);
  }
  shard_routing::WriteBatchStatus(batch, batch->value_size);
  batch->assembled = true;
  return batch->status_code == static_cast<std::int32_t>(petps::RpcStatus::kOk);
}

bool RdmaDistributedExecutor::QueryBatchFinished(int batch_id) {
  std::lock_guard<std::mutex> guard(batches_mu_);
  auto it = batches_.find(batch_id);
  if (it == batches_.end()) {
    throw std::runtime_error("unknown RDMA distributed batch handle");
  }
  for (const auto& pending : it->second.shard_rpcs) {
    if (pending.released) {
      continue;
    }
    if (!ClientAt(pending.client_index)->QueryRPCFinished(pending.rpc_id)) {
      return false;
    }
  }
  for (const auto& pending : it->second.shard_rpcs) {
    if (!pending.released) {
      ClientAt(pending.client_index)->WaitRPCFinish(pending.rpc_id);
    }
  }
  return FinalizeBatch(&it->second);
}

const float* RdmaDistributedExecutor::BorrowBatchResult(
    int batch_id, std::int32_t* status_code, std::size_t* response_bytes) {
  std::lock_guard<std::mutex> guard(batches_mu_);
  const auto it = batches_.find(batch_id);
  if (it == batches_.end()) {
    throw std::runtime_error("unknown RDMA distributed batch handle");
  }
  const auto& batch = it->second;
  if (batch.shard_rpcs.size() != 1 || batch.shard_rpcs.front().released ||
      batch.shard_rpcs.front().key_count != batch.total_key_count) {
    return nullptr;
  }
  const auto& pending = batch.shard_rpcs.front();
  for (std::size_t i = 0; i < pending.original_positions.size(); ++i) {
    if (pending.original_positions[i] != i) {
      return nullptr;
    }
  }
  auto* client = ClientAt(pending.client_index);
  std::size_t key_count = 0;
  const float* payload  = client->BorrowGetResultPayload(
      pending.rpc_id, &key_count, response_bytes, status_code);
  return key_count == batch.total_key_count ? payload : nullptr;
}

std::uint64_t RdmaDistributedExecutor::SubmitPrefetch(
    const std::vector<shard_routing::ShardChunk>& chunks,
    std::size_t key_count,
    std::int64_t embedding_dim,
    int value_size,
    std::size_t max_in_flight) {
  if (key_count == 0 || embedding_dim <= 0) {
    throw std::invalid_argument("invalid RDMA prefetch request");
  }
  const std::size_t value_count =
      key_count * static_cast<std::size_t>(embedding_dim);
  auto buffer = std::make_shared<std::vector<float>>(value_count + 1, 0.0f);
  const int batch_id = SubmitGetParameter(
      chunks, key_count, value_size, buffer->data(), true, 0, max_in_flight);
  std::lock_guard<std::mutex> guard(prefetches_mu_);
  const std::uint64_t prefetch_id = next_prefetch_id_++;
  prefetches_.emplace(
      prefetch_id,
      PendingPrefetch{buffer,
                      batch_id,
                      static_cast<std::int64_t>(key_count),
                      embedding_dim,
                      value_size});
  return prefetch_id;
}

RdmaDistributedExecutor::PendingPrefetch
RdmaDistributedExecutor::GetPrefetch(std::uint64_t prefetch_id) const {
  std::lock_guard<std::mutex> guard(prefetches_mu_);
  const auto it = prefetches_.find(prefetch_id);
  if (it == prefetches_.end()) {
    throw std::runtime_error("unknown or consumed RDMA prefetch handle");
  }
  return it->second;
}

void RdmaDistributedExecutor::ErasePrefetch(std::uint64_t prefetch_id) {
  std::lock_guard<std::mutex> guard(prefetches_mu_);
  if (prefetches_.erase(prefetch_id) == 0) {
    throw std::runtime_error("unknown or consumed RDMA prefetch handle");
  }
}

bool RdmaDistributedExecutor::IsPrefetchDone(std::uint64_t prefetch_id) {
  return QueryBatchFinished(GetPrefetch(prefetch_id).batch_id);
}

void RdmaDistributedExecutor::WaitForPrefetch(std::uint64_t prefetch_id) {
  WaitBatch(GetPrefetch(prefetch_id).batch_id);
}

std::int64_t
RdmaDistributedExecutor::PrefetchEmbeddingDim(std::uint64_t prefetch_id) const {
  return GetPrefetch(prefetch_id).embedding_dim;
}

RdmaDistributedExecutor::PrefetchResult
RdmaDistributedExecutor::ReadPrefetch(std::uint64_t prefetch_id) {
  const PendingPrefetch state = GetPrefetch(prefetch_id);
  PrefetchResult result;
  result.buffer         = state.buffer;
  result.key_count      = state.key_count;
  result.embedding_dim  = state.embedding_dim;
  result.response_bytes = 0;
  result.status_code    = static_cast<std::int32_t>(petps::RpcStatus::kOk);
  result.payload        = BorrowBatchResult(
      state.batch_id, &result.status_code, &result.response_bytes);
  if (result.payload == nullptr) {
    WaitBatch(state.batch_id);
    const auto* status_word = petps::FixedSlotStatusWord(
        state.buffer->data(),
        static_cast<std::size_t>(state.key_count),
        state.value_size);
    result.status_code    = *status_word;
    result.response_bytes = static_cast<std::size_t>(state.key_count) *
                            static_cast<std::size_t>(state.value_size);
    result.payload = state.buffer->data();
  }
  return result;
}

void RdmaDistributedExecutor::ReleasePrefetch(std::uint64_t prefetch_id) {
  const PendingPrefetch state = GetPrefetch(prefetch_id);
  ErasePrefetch(prefetch_id);
  ReleaseBatch(state.batch_id);
}

void RdmaDistributedExecutor::WaitShardRpcs(
    const std::vector<shard_routing::PendingShardRpc>& shard_rpcs) const {
  std::vector<bool> finished(shard_rpcs.size(), false);
  std::size_t remaining = 0;
  for (std::size_t i = 0; i < shard_rpcs.size(); ++i) {
    finished[i] = shard_rpcs[i].released;
    if (!finished[i]) {
      ++remaining;
    }
  }
  while (remaining > 0) {
    bool made_progress = false;
    for (std::size_t i = 0; i < shard_rpcs.size(); ++i) {
      if (finished[i]) {
        continue;
      }
      const auto& pending = shard_rpcs[i];
      auto* client        = ClientAt(pending.client_index);
      if (!client->QueryRPCFinished(pending.rpc_id)) {
        continue;
      }
      client->WaitRPCFinish(pending.rpc_id);
      finished[i]   = true;
      made_progress = true;
      --remaining;
    }
    if (!made_progress) {
      std::this_thread::yield();
    }
  }
}

void RdmaDistributedExecutor::ReleaseShardRpcs(
    std::vector<shard_routing::PendingShardRpc>* shard_rpcs) const {
  if (shard_rpcs == nullptr) {
    throw std::invalid_argument("RDMA shard RPC list must not be null");
  }
  std::exception_ptr first_error;
  for (auto& pending : *shard_rpcs) {
    if (pending.released) {
      continue;
    }
    pending.released = true;
    try {
      ClientAt(pending.client_index)->RevokeRPCResource(pending.rpc_id);
    } catch (...) {
      if (first_error == nullptr) {
        first_error = std::current_exception();
      }
    }
  }
  if (first_error != nullptr) {
    std::rethrow_exception(first_error);
  }
}

int RdmaDistributedExecutor::PutParameter(
    const std::vector<shard_routing::ShardChunk>& chunks,
    const base::RecTensor& values) const {
  for (const auto& chunk : chunks) {
    const int64_t dim = values.shape(1);
    base::RecTensor chunk_values(
        {static_cast<int64_t>(chunk.positions.size()), dim},
        base::DataType::FLOAT32);
    std::size_t chunk_row = 0;
    for (const std::size_t position : chunk.positions) {
      if (position >= static_cast<std::size_t>(values.shape(0))) {
        throw std::invalid_argument("RDMA PUT chunk position is out of range");
      }
      std::memcpy(
          chunk_values.data_as<float>() +
              chunk_row++ * static_cast<std::size_t>(dim),
          values.data_as<float>() + position * static_cast<std::size_t>(dim),
          static_cast<std::size_t>(dim) * sizeof(float));
    }
    const int rc =
        ClientAt(chunk.client_index)->PutParameter(
            base::ConstArray<std::uint64_t>(chunk.keys), chunk_values);
    if (rc != 0) {
      return rc;
    }
  }
  return 0;
}

int RdmaDistributedExecutor::UpdateParameter(
    const std::string& table_name,
    const std::vector<shard_routing::ShardChunk>& chunks,
    const base::RecTensor& grads) const {
  for (const auto& chunk : chunks) {
    const int64_t dim = grads.shape(1);
    base::RecTensor chunk_grads(
        {static_cast<int64_t>(chunk.positions.size()), dim},
        base::DataType::FLOAT32);
    std::size_t chunk_row = 0;
    for (const std::size_t position : chunk.positions) {
      if (position >= static_cast<std::size_t>(grads.shape(0))) {
        throw std::invalid_argument(
            "RDMA UPDATE chunk position is out of range");
      }
      std::memcpy(
          chunk_grads.data_as<float>() + chunk_row++ * static_cast<std::size_t>(dim),
          grads.data_as<float>() + position * static_cast<std::size_t>(dim),
          static_cast<std::size_t>(dim) * sizeof(float));
    }
    const int rc =
        ClientAt(chunk.client_index)
            ->UpdateParameter(table_name,
                              base::ConstArray<std::uint64_t>(chunk.keys),
                              chunk_grads);
    if (rc != 0) {
      return rc;
    }
  }
  return 0;
}

int RdmaDistributedExecutor::WaitUpdateRpcs(
    std::vector<std::pair<int, int>>* shard_rpcs) const {
  int result        = 0;
  std::size_t index = 0;
  try {
    for (; index < shard_rpcs->size(); ++index) {
      const auto [client_index, rpc_id] = (*shard_rpcs)[index];
      auto* client = ClientAt(client_index);
      if (client->WaitUpdateParameter(rpc_id) != 0) {
        result = -1;
      }
    }
  } catch (...) {
    std::exception_ptr first_error = std::current_exception();
    for (++index; index < shard_rpcs->size(); ++index) {
      try {
        const auto [client_index, rpc_id] = (*shard_rpcs)[index];
        ClientAt(client_index)->RevokeRPCResource(rpc_id);
      } catch (...) {
        // Preserve the operation error while still releasing later shards.
      }
    }
    shard_rpcs->clear();
    std::rethrow_exception(first_error);
  }
  shard_rpcs->clear();
  return result;
}

std::uint64_t RdmaDistributedExecutor::SubmitUpdateParameterFlat(
    const std::string& table_name,
    base::ConstArray<std::uint64_t> keys,
    const float* grads,
    std::size_t embedding_dim,
    const std::vector<shard_routing::ShardChunk>& chunks,
    std::size_t max_in_flight) {
  if ((keys.Size() > 0 && grads == nullptr) || embedding_dim == 0 ||
      max_in_flight == 0) {
    throw std::invalid_argument("invalid RDMA flat UPDATE request");
  }

  std::vector<std::pair<int, int>> pending;
  auto drain = [this, &pending]() {
    if (WaitUpdateRpcs(&pending) != 0) {
      throw std::runtime_error("RDMA embedding update failed");
    }
  };

  try {
    for (const auto& chunk : chunks) {
      auto* client = ClientAt(chunk.client_index);
      if (chunk.keys.size() != chunk.positions.size()) {
        throw std::runtime_error("RDMA UPDATE chunk shape mismatch");
      }
      bool contiguous = !chunk.positions.empty();
      for (std::size_t i = 1; i < chunk.positions.size(); ++i) {
        contiguous =
            contiguous && chunk.positions[i] == chunk.positions.front() + i;
      }
      const int rpc_id =
          contiguous
              ? client->SubmitUpdateParameterFlat(
                    table_name,
                    base::ConstArray<std::uint64_t>(chunk.keys),
                    grads + chunk.positions.front() * embedding_dim,
                    embedding_dim)
              : client->SubmitUpdateParameterFlatGather(
                    table_name,
                    keys.Data(),
                    grads,
                    keys.Size(),
                    embedding_dim,
                    chunk.positions.data(),
                    chunk.positions.size());
      if (rpc_id < 0) {
        throw std::runtime_error("failed to submit RDMA embedding update");
      }
      pending.emplace_back(chunk.client_index, rpc_id);
      if (pending.size() >= max_in_flight) {
        drain();
      }
    }
  } catch (...) {
    try {
      drain();
    } catch (...) {
      // Preserve the operation error that initiated cleanup.
    }
    throw;
  }

  std::lock_guard<std::mutex> guard(updates_mu_);
  const std::uint64_t update_id = next_update_id_++;
  pending_updates_.emplace(
      update_id, PendingUpdate{std::move(pending), std::this_thread::get_id()});
  return update_id;
}

int RdmaDistributedExecutor::WaitUpdateParameterFlat(std::uint64_t update_id) {
  PendingUpdate update;
  {
    std::lock_guard<std::mutex> guard(updates_mu_);
    const auto it = pending_updates_.find(update_id);
    if (it == pending_updates_.end()) {
      throw std::runtime_error(
          "unknown or already consumed RDMA update handle");
    }
    if (it->second.owner != std::this_thread::get_id()) {
      throw std::runtime_error(
          "RDMA update handle must be waited by its submitting thread");
    }
    update = std::move(it->second);
    pending_updates_.erase(it);
  }
  return WaitUpdateRpcs(&update.shard_rpcs);
}

void RdmaDistributedExecutor::WaitBatch(int batch_id) {
  std::vector<shard_routing::PendingShardRpc> shard_rpcs;
  {
    std::lock_guard<std::mutex> guard(batches_mu_);
    auto it = batches_.find(batch_id);
    if (it == batches_.end()) {
      throw std::runtime_error("unknown RDMA distributed batch handle");
    }
    if (it->second.assembled) {
      return;
    }
    shard_rpcs = it->second.shard_rpcs;
  }
  WaitShardRpcs(shard_rpcs);
  std::lock_guard<std::mutex> guard(batches_mu_);
  auto it = batches_.find(batch_id);
  if (it == batches_.end()) {
    throw std::runtime_error("RDMA distributed batch released while waiting");
  }
  FinalizeBatch(&it->second);
}

void RdmaDistributedExecutor::ReleaseBatch(int batch_id) {
  shard_routing::BatchRequest batch;
  {
    std::lock_guard<std::mutex> guard(batches_mu_);
    auto it = batches_.find(batch_id);
    if (it == batches_.end()) {
      throw std::runtime_error(
          "unknown or released RDMA distributed batch handle");
    }
    batch = std::move(it->second);
    batches_.erase(it);
  }
  ReleaseShardRpcs(&batch.shard_rpcs);
}

} // namespace recstore
