#include "ps/rdma/petps_client.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <folly/portability/GFlags.h>

#include "ps/rdma/control_plane.h"
#include "ps/rdma/rdma_common.h"
#include "ps/rdma/rc_options.h"

DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(num_client_processes);
DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);
DEFINE_string(rdma_get_response_mode,
              "direct_sg",
              "RDMA GET response mode: direct_sg or staging_copy");

namespace petps {
namespace {

using petps::Exchange;
using petps::NamespaceToken;
using petps::NowNs;

std::size_t ComputeMaxGetKeysPerRpc() {
  return GetKeysPerRpcByResponseBudget(
      static_cast<std::size_t>(FLAGS_value_size),
      static_cast<std::size_t>(FLAGS_rdma_rc_mtu_bytes),
      static_cast<std::size_t>(FLAGS_rdma_rc_target_response_mtu));
}

std::int32_t WaitStatus(const StatusWord* status, std::uint64_t seq) {
  const auto start    = std::chrono::steady_clock::now();
  int spin_iterations = 0;
  while (!StatusWordDone(*status, seq)) {
    if (spin_iterations < FLAGS_rdma_rc_wait_spin_iterations) {
      ++spin_iterations;
    } else {
      spin_iterations = 0;
      std::this_thread::yield();
    }
    if (FLAGS_rdma_wait_timeout_ms > 0) {
      const auto elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      if (elapsed_ms > FLAGS_rdma_wait_timeout_ms) {
        throw std::runtime_error("RC write RPC wait timeout");
      }
    }
  }
  return status->status;
}

void FillBaseDescriptor(
    RequestDescriptor* descriptor,
    std::uint64_t seq,
    std::size_t key_count,
    const RcClientQpView& view,
    std::uint32_t shard_id,
    std::uint32_t client_id) {
  *descriptor            = RequestDescriptor{};
  descriptor->seq        = seq;
  descriptor->shard_id   = shard_id;
  descriptor->client_id  = client_id;
  descriptor->qp_index   = static_cast<std::uint32_t>(view.qp_index);
  descriptor->key_count  = static_cast<std::uint32_t>(key_count);
  descriptor->value_size = static_cast<std::uint32_t>(FLAGS_value_size);
  descriptor->embedding_dim =
      static_cast<std::uint32_t>(FLAGS_value_size / sizeof(float));
  descriptor->payload_offset =
      static_cast<std::uint32_t>(Align64(sizeof(RequestDescriptor)));
  descriptor->client_response_addr =
      reinterpret_cast<std::uint64_t>(view.response_payload);
  descriptor->client_status_addr = reinterpret_cast<std::uint64_t>(view.status);
}

} // namespace

PetPSClient::PetPSClient(const std::string& host, int port, int shard)
    : PetPSClient(host, port, shard, -1) {}

PetPSClient::PetPSClient(
    const std::string& host, int port, int shard, int logical_client_id)
    : BaseParameterClient(host, port, shard),
      namespace_token_(NamespaceToken()),
      explicit_client_id_(logical_client_id) {}

PetPSClient::~PetPSClient() = default;

void PetPSClient::Barrier(const std::string&, int) {}

void PetPSClient::InitializeTransport() {
  if (transport_ != nullptr) {
    return;
  }
  client_id_ =
      explicit_client_id_ >= 0
          ? explicit_client_id_
          : (FLAGS_rdma_rc_client_id_base >= 0
                 ? FLAGS_rdma_rc_client_id_base
                 : FLAGS_global_id - FLAGS_num_server_processes);
  if (client_id_ < 0) {
    throw std::runtime_error("invalid RC write logical client_id");
  }
  const int logical_num_clients =
      FLAGS_rdma_rc_num_logical_clients >= 0
          ? FLAGS_rdma_rc_num_logical_clients
          : FLAGS_num_client_processes;
  if (client_id_ >= logical_num_clients) {
    throw std::runtime_error("RC write logical client_id out of range");
  }
  config_.shard_id                 = shard_;
  config_.client_id                = client_id_;
  config_.num_clients              = logical_num_clients;
  config_.qps_per_client_per_shard = FLAGS_rdma_rc_qps_per_client_per_shard;
  config_.slots_per_qp             = FLAGS_rdma_rc_slots_per_qp;
  config_.request_slot_bytes =
      static_cast<std::size_t>(FLAGS_rdma_rc_request_slot_bytes);
  config_.response_slot_bytes =
      static_cast<std::size_t>(FLAGS_rdma_rc_response_slot_bytes);
  config_.control_plane_host       = FLAGS_rdma_control_plane_host;
  config_.control_plane_port       = FLAGS_rdma_control_plane_port;
  config_.control_plane_timeout_ms = FLAGS_rdma_control_plane_timeout_ms;
  config_.namespace_token          = namespace_token_;

  transport_ = std::make_unique<RcShardClientTransport>(config_);
  RdmaControlPlaneClient control_plane({
      config_.control_plane_host,
      config_.control_plane_port,
      config_.control_plane_timeout_ms,
  });
  control_plane.WaitServer(shard_, config_.control_plane_timeout_ms);
  qps_.clear();
  qps_.reserve(static_cast<std::size_t>(config_.qps_per_client_per_shard));
  for (int qp = 0; qp < config_.qps_per_client_per_shard; ++qp) {
    QpContext context;
    context.qp_index = qp;
    context.slots.reserve(static_cast<std::size_t>(config_.slots_per_qp));
    for (int slot_in_qp = 0; slot_in_qp < config_.slots_per_qp; ++slot_in_qp) {
      context.slots.push_back(
          SlotContext{transport_->OpenSlot(qp, slot_in_qp), 1, false});
    }
    qps_.push_back(std::move(context));
  }
}

void PetPSClient::InitThread() {
  std::lock_guard<std::mutex> guard(mu_);
  InitializeTransport();
  thread_initialized_ = true;
}

std::size_t PetPSClient::ResponseBufferBytes(std::size_t key_count) const {
  return GetResponseBytes(
             key_count, static_cast<std::size_t>(FLAGS_value_size)) +
         sizeof(std::int32_t);
}

void* PetPSClient::GetReceiveBuffer(size_t size) {
  std::lock_guard<std::mutex> guard(mu_);
  receive_buffers_.emplace_back(size, 0);
  return receive_buffers_.back().data();
}

const float* PetPSClient::BorrowGetResultPayload(
    int rpc_id,
    std::size_t* key_count,
    std::size_t* response_bytes,
    std::int32_t* status_code) {
  PendingRpc pending;
  {
    std::lock_guard<std::mutex> guard(mu_);
    if (!PendingRpcLocked(rpc_id, &pending)) {
      return nullptr;
    }
  }

  auto& slot                 = SlotAt(pending.qp_index, pending.slot_in_qp);
  const bool profile_enabled = FLAGS_rdma_rc_profile_interval_ms > 0;
  const std::uint64_t wait_start_ns = profile_enabled ? NowNs() : 0;
  const std::int32_t rc_status      = WaitStatus(slot.view.status, pending.seq);
  if (profile_enabled) {
    profile_.wait_rpc_count.fetch_add(1, std::memory_order_relaxed);
    profile_.wait_status_ns.fetch_add(
        NowNs() - wait_start_ns, std::memory_order_relaxed);
  }

  const std::size_t actual_response_bytes = std::min<std::size_t>(
      slot.view.status->response_bytes, pending.response_bytes);
  if (key_count != nullptr) {
    *key_count = pending.key_count;
  }
  if (response_bytes != nullptr) {
    *response_bytes = actual_response_bytes;
  }
  if (status_code != nullptr) {
    *status_code = rc_status;
  }
  if (pending.recv_buffer != nullptr) {
    auto* user_status = FixedSlotStatusWord(
        pending.recv_buffer, pending.key_count, FLAGS_value_size);
    *user_status = rc_status;
  }
  MaybeReportProfile();
  return reinterpret_cast<const float*>(slot.view.response_payload);
}

PetPSClient::SlotHandle PetPSClient::AcquireIdleSlot() {
  if (FLAGS_rdma_rc_profile_interval_ms > 0) {
    profile_.acquire_qp_count.fetch_add(1, std::memory_order_relaxed);
  }
  for (std::size_t qp_index = 0; qp_index < qps_.size(); ++qp_index) {
    auto& qp = qps_[qp_index];
    for (std::size_t slot_in_qp = 0; slot_in_qp < qp.slots.size();
         ++slot_in_qp) {
      if (!qp.slots[slot_in_qp].busy) {
        qp.slots[slot_in_qp].busy = true;
        return SlotHandle{
            static_cast<int>(qp_index),
            static_cast<int>(slot_in_qp),
        };
      }
    }
  }
  if (FLAGS_rdma_rc_profile_interval_ms > 0) {
    profile_.acquire_qp_failures.fetch_add(1, std::memory_order_relaxed);
  }
  throw std::runtime_error("no idle RC write slot available");
}

PetPSClient::SlotContext& PetPSClient::SlotAt(int qp_index, int slot_in_qp) {
  auto& qp = qps_.at(static_cast<std::size_t>(qp_index));
  return qp.slots.at(static_cast<std::size_t>(slot_in_qp));
}

const PetPSClient::SlotContext&
PetPSClient::SlotAt(int qp_index, int slot_in_qp) const {
  const auto& qp = qps_.at(static_cast<std::size_t>(qp_index));
  return qp.slots.at(static_cast<std::size_t>(slot_in_qp));
}

void PetPSClient::EnsureThreadInitializedLocked() const {
  if (!thread_initialized_) {
    throw std::runtime_error("PetPSClient::InitThread must be called first");
  }
}

bool PetPSClient::PendingRpcLocked(int rpc_id, PendingRpc* pending) const {
  const auto it = pending_rpcs_.find(rpc_id);
  if (it == pending_rpcs_.end()) {
    return false;
  }
  if (pending != nullptr) {
    *pending = it->second;
  }
  return true;
}

bool PetPSClient::RequestPayloadFitsSlot(std::size_t payload_bytes) const {
  return Align64(sizeof(RequestDescriptor)) + payload_bytes +
             Align64(sizeof(CommitWord)) <=
         config_.request_slot_bytes;
}

float* PetPSClient::AllocateStatusReceiveBufferLocked() {
  receive_buffers_.emplace_back(sizeof(std::int32_t), 0);
  return reinterpret_cast<float*>(receive_buffers_.back().data());
}

void PetPSClient::MaybeReportProfile() {
  if (FLAGS_rdma_rc_profile_interval_ms <= 0) {
    return;
  }
  const std::uint64_t now = NowNs();
  const std::uint64_t interval =
      static_cast<std::uint64_t>(FLAGS_rdma_rc_profile_interval_ms) * 1000000;
  std::uint64_t expected =
      profile_.next_report_ns.load(std::memory_order_relaxed);
  if (expected == 0) {
    profile_.next_report_ns.compare_exchange_strong(
        expected, now + interval, std::memory_order_relaxed);
    return;
  }
  if (now < expected ||
      !profile_.next_report_ns.compare_exchange_strong(
          expected, now + interval, std::memory_order_relaxed)) {
    return;
  }

  const std::uint64_t submit_count    = Exchange(&profile_.submit_rpc_count);
  const std::uint64_t wait_count      = Exchange(&profile_.wait_rpc_count);
  const std::uint64_t revoke_count    = Exchange(&profile_.revoke_rpc_count);
  const std::uint64_t submit_ns       = Exchange(&profile_.submit_request_ns);
  const std::uint64_t wait_ns         = Exchange(&profile_.wait_status_ns);
  const std::uint64_t copy_ns         = Exchange(&profile_.copy_response_ns);
  const std::uint64_t revoke_ns       = Exchange(&profile_.revoke_resource_ns);
  const std::uint64_t pending_samples = Exchange(&profile_.pending_rpc_samples);
  const std::uint64_t pending_sum     = Exchange(&profile_.pending_rpc_sum);
  std::cout
      << "component=rdma_rc_client_profile"
      << " shard=" << shard_ << " client_id=" << client_id_
      << " submit_count=" << submit_count << " wait_count=" << wait_count
      << " revoke_count=" << revoke_count
      << " acquire_qp_count=" << Exchange(&profile_.acquire_qp_count)
      << " acquire_qp_failures=" << Exchange(&profile_.acquire_qp_failures)
      << " submit_avg_ns=" << (submit_count == 0 ? 0 : submit_ns / submit_count)
      << " wait_status_avg_ns=" << (wait_count == 0 ? 0 : wait_ns / wait_count)
      << " copy_response_avg_ns="
      << (wait_count == 0 ? 0 : copy_ns / wait_count)
      << " copied_bytes=" << Exchange(&profile_.response_bytes_copied)
      << " revoke_avg_ns=" << (revoke_count == 0 ? 0 : revoke_ns / revoke_count)
      << " pending_rpc_peak=" << Exchange(&profile_.pending_rpc_peak)
      << " pending_rpc_avg="
      << (pending_samples == 0 ? 0 : pending_sum / pending_samples)
      << " pending_rpc_last="
      << profile_.pending_rpc_last.load(std::memory_order_relaxed) << std::endl;
}

void PetPSClient::FillGetDescriptor(
    RequestDescriptor* descriptor,
    std::uint64_t seq,
    std::size_t key_count,
    std::size_t response_bytes,
    const RcClientQpView& view) const {
  FillBaseDescriptor(
      descriptor,
      seq,
      key_count,
      view,
      static_cast<std::uint32_t>(shard_),
      static_cast<std::uint32_t>(client_id_));
  descriptor->op = static_cast<std::uint16_t>(RcOp::kGet);
  descriptor->payload_bytes =
      static_cast<std::uint32_t>(GetRequestBytes(key_count));
  descriptor->response_bytes = static_cast<std::uint32_t>(response_bytes);
  if (FLAGS_rdma_get_response_mode == "direct_sg") {
    descriptor->flags |= kRcFlagGetDirectSg | kRcFlagGetAllowFallbackCopy;
  } else if (FLAGS_rdma_get_response_mode != "staging_copy") {
    LOG(FATAL) << "unsupported --rdma_get_response_mode="
               << FLAGS_rdma_get_response_mode;
  }
}

void PetPSClient::FillPutDescriptor(
    RequestDescriptor* descriptor,
    std::uint64_t seq,
    std::size_t key_count,
    std::size_t payload_bytes,
    const RcClientQpView& view) const {
  FillBaseDescriptor(
      descriptor,
      seq,
      key_count,
      view,
      static_cast<std::uint32_t>(shard_),
      static_cast<std::uint32_t>(client_id_));
  descriptor->op             = static_cast<std::uint16_t>(RcOp::kPut);
  descriptor->payload_bytes  = static_cast<std::uint32_t>(payload_bytes);
  descriptor->response_bytes = 0;
}

void PetPSClient::FillUpdateDescriptor(
    RequestDescriptor* descriptor,
    std::uint64_t seq,
    std::size_t key_count,
    std::size_t payload_bytes,
    const std::string& table_name,
    const RcClientQpView& view) const {
  FillPutDescriptor(descriptor, seq, key_count, payload_bytes, view);
  descriptor->op = static_cast<std::uint16_t>(RcOp::kUpdate);
  if (!CopyTableName(table_name, &descriptor->table_name)) {
    throw std::runtime_error("UPDATE table name too long");
  }
}

void PetPSClient::FillInitTableDescriptor(
    RequestDescriptor* descriptor,
    std::uint64_t seq,
    const std::string& table_name,
    const RcClientQpView& view) const {
  FillPutDescriptor(
      descriptor, seq, /*key_count=*/0, InitTablePayloadBytes(), view);
  descriptor->op = static_cast<std::uint16_t>(RcOp::kInitTable);
  if (!CopyTableName(table_name, &descriptor->table_name)) {
    throw std::runtime_error("INIT table name too long");
  }
}

int PetPSClient::SubmitRpcLocked(
    SlotContext* slot,
    const RequestDescriptor& descriptor,
    const void* payload,
    std::size_t payload_bytes,
    float* recv_buffer,
    std::size_t key_count,
    std::size_t response_bytes,
    bool is_async) {
  if (slot == nullptr) {
    throw std::runtime_error("slot context is null");
  }
  ResetStatusWord(slot->view.status, descriptor.seq);
  const bool profile_enabled          = FLAGS_rdma_rc_profile_interval_ms > 0;
  const std::uint64_t submit_start_ns = profile_enabled ? NowNs() : 0;
  transport_->SubmitRequest(slot->view, descriptor, payload, payload_bytes);
  if (profile_enabled) {
    profile_.submit_rpc_count.fetch_add(1, std::memory_order_relaxed);
    profile_.submit_request_ns.fetch_add(
        NowNs() - submit_start_ns, std::memory_order_relaxed);
  }
  VLOG(1) << "component=rdma_rc_client event=submit shard=" << shard_
          << " client_id=" << client_id_ << " qp=" << slot->view.qp_index
          << " slot=" << slot->view.slot_index << " seq=" << descriptor.seq
          << " op=" << descriptor.op << " key_count=" << key_count
          << " payload_bytes=" << payload_bytes
          << " response_bytes=" << response_bytes;

  const int rpc_id = next_rpc_id_.fetch_add(1);
  pending_rpcs_.emplace(
      rpc_id,
      PendingRpc{
          slot->view.qp_index,
          slot->view.slot_in_qp,
          slot->view.slot_index,
          descriptor.seq,
          recv_buffer,
          key_count,
          response_bytes,
      });
  if (profile_enabled) {
    const std::uint64_t pending_size = pending_rpcs_.size();
    profile_.pending_rpc_samples.fetch_add(1, std::memory_order_relaxed);
    profile_.pending_rpc_sum.fetch_add(pending_size, std::memory_order_relaxed);
    profile_.pending_rpc_last.store(pending_size, std::memory_order_relaxed);
    std::uint64_t peak =
        profile_.pending_rpc_peak.load(std::memory_order_relaxed);
    while (pending_size > peak &&
           !profile_.pending_rpc_peak.compare_exchange_weak(
               peak, pending_size, std::memory_order_relaxed)) {
    }
    MaybeReportProfile();
  }
  if (!is_async) {
    WaitRPCFinish(rpc_id);
  }
  return rpc_id;
}

int PetPSClient::GetParameter(base::ConstArray<uint64_t> keys,
                              std::vector<std::vector<float>>* values) {
  values->clear();
  if (keys.Size() == 0) {
    return 0;
  }
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  std::vector<float> flat(keys.Size() * embedding_dim + 1, 0.0f);
  const int rpc_id = GetParameter(keys, flat.data(), false, 0);
  const auto* status =
      FixedSlotStatusWord(flat.data(), keys.Size(), FLAGS_value_size);
  if (*status != static_cast<std::int32_t>(RpcStatus::kOk)) {
    RevokeRPCResource(rpc_id);
    return -1;
  }
  CopyFlatRowsToVectors(
      flat.data(),
      keys.Size(),
      static_cast<std::size_t>(embedding_dim),
      values);
  RevokeRPCResource(rpc_id);
  return 0;
}

int PetPSClient::GetParameter(
    base::ConstArray<uint64_t> keys, float* values, bool isAsync, int) {
  if (keys.Size() == 0) {
    auto* status =
        reinterpret_cast<std::int32_t*>(reinterpret_cast<char*>(values));
    *status = static_cast<std::int32_t>(RpcStatus::kOk);
    return 0;
  }
  int rpc_id = 0;
  {
    std::lock_guard<std::mutex> guard(mu_);
    EnsureThreadInitializedLocked();
    if (keys.Size() > ComputeMaxGetKeysPerRpc()) {
      throw std::runtime_error(
          "single-shard GET batch exceeds RC response budget");
    }

    const SlotHandle slot_handle = AcquireIdleSlot();
    auto& slot = SlotAt(slot_handle.qp_index, slot_handle.slot_in_qp);
    RequestDescriptor descriptor;
    const std::size_t response_bytes = GetResponseBytes(
        keys.Size(), static_cast<std::size_t>(FLAGS_value_size));
    FillGetDescriptor(
        &descriptor, slot.next_seq++, keys.Size(), response_bytes, slot.view);
    if (descriptor.payload_bytes >
        PutPayloadBudget(config_.request_slot_bytes)) {
      slot.busy = false;
      throw std::runtime_error("GET request exceeds RC request slot");
    }
    rpc_id = SubmitRpcLocked(
        &slot,
        descriptor,
        keys.Data(),
        descriptor.payload_bytes,
        values,
        keys.Size(),
        response_bytes,
        true);
  }
  if (!isAsync) {
    WaitRPCFinish(rpc_id);
  }
  return rpc_id;
}

bool PetPSClient::QueryRPCFinished(int rpc_id) {
  std::lock_guard<std::mutex> guard(mu_);
  PendingRpc pending;
  if (!PendingRpcLocked(rpc_id, &pending)) {
    return true;
  }
  const auto& slot = SlotAt(pending.qp_index, pending.slot_in_qp);
  return StatusWordDone(*slot.view.status, pending.seq);
}

void PetPSClient::WaitRPCFinish(int rpc_id) {
  PendingRpc pending;
  {
    std::lock_guard<std::mutex> guard(mu_);
    if (!PendingRpcLocked(rpc_id, &pending)) {
      return;
    }
  }

  auto& slot                 = SlotAt(pending.qp_index, pending.slot_in_qp);
  const bool profile_enabled = FLAGS_rdma_rc_profile_interval_ms > 0;
  const std::uint64_t wait_start_ns = profile_enabled ? NowNs() : 0;
  const std::int32_t status_code    = WaitStatus(slot.view.status, pending.seq);
  if (profile_enabled) {
    profile_.wait_rpc_count.fetch_add(1, std::memory_order_relaxed);
    profile_.wait_status_ns.fetch_add(
        NowNs() - wait_start_ns, std::memory_order_relaxed);
  }
  VLOG(1) << "component=rdma_rc_client event=done shard=" << shard_
          << " client_id=" << client_id_ << " qp=" << pending.qp_index
          << " slot=" << pending.slot_index << " seq=" << pending.seq
          << " status=" << status_code
          << " response_bytes=" << pending.response_bytes;
  const std::size_t actual_response_bytes = std::min<std::size_t>(
      slot.view.status->response_bytes, pending.response_bytes);
  if (actual_response_bytes > 0 && !FLAGS_rdma_rc_skip_client_copy) {
    const std::uint64_t copy_start_ns = profile_enabled ? NowNs() : 0;
    std::memcpy(
        pending.recv_buffer, slot.view.response_payload, actual_response_bytes);
    if (profile_enabled) {
      profile_.copy_response_ns.fetch_add(
          NowNs() - copy_start_ns, std::memory_order_relaxed);
      profile_.response_bytes_copied.fetch_add(
          actual_response_bytes, std::memory_order_relaxed);
    }
  }
  auto* user_status = FixedSlotStatusWord(
      pending.recv_buffer, pending.key_count, FLAGS_value_size);
  *user_status = status_code;
  MaybeReportProfile();
}

void PetPSClient::RevokeRPCResource(int rpc_id) {
  std::lock_guard<std::mutex> guard(mu_);
  const auto it = pending_rpcs_.find(rpc_id);
  if (it == pending_rpcs_.end()) {
    return;
  }
  const bool profile_enabled          = FLAGS_rdma_rc_profile_interval_ms > 0;
  const std::uint64_t revoke_start_ns = profile_enabled ? NowNs() : 0;
  auto& slot = SlotAt(it->second.qp_index, it->second.slot_in_qp);
  transport_->ClearRequestSlot(slot.view);
  slot.busy = false;
  pending_rpcs_.erase(it);
  if (profile_enabled) {
    const std::uint64_t pending_size = pending_rpcs_.size();
    profile_.pending_rpc_samples.fetch_add(1, std::memory_order_relaxed);
    profile_.pending_rpc_sum.fetch_add(pending_size, std::memory_order_relaxed);
    profile_.pending_rpc_last.store(pending_size, std::memory_order_relaxed);
    profile_.revoke_rpc_count.fetch_add(1, std::memory_order_relaxed);
    profile_.revoke_resource_ns.fetch_add(
        NowNs() - revoke_start_ns, std::memory_order_relaxed);
    MaybeReportProfile();
  }
}

int PetPSClient::PutParameter(const std::vector<uint64_t>& keys,
                              const std::vector<std::vector<float>>& values) {
  if (keys.size() != values.size()) {
    return -1;
  }
  if (keys.empty()) {
    return 0;
  }

  std::size_t begin = 0;
  while (begin < keys.size()) {
    std::size_t end =
        std::min(begin + static_cast<std::size_t>(FLAGS_max_kv_num_per_request),
                 keys.size());
    std::vector<std::uint64_t> key_slice(
        keys.begin() + begin, keys.begin() + end);
    std::vector<std::vector<float>> value_slice(
        values.begin() + begin, values.begin() + end);

    std::string payload;
    std::string error;
    const std::size_t payload_bytes =
        PutPayloadBytes(key_slice, value_slice, &payload, &error);
    if (payload_bytes == 0 && !key_slice.empty()) {
      throw std::runtime_error("RC PUT payload build failed: " + error);
    }

    float* recv = nullptr;
    int rpc_id  = 0;
    {
      std::lock_guard<std::mutex> guard(mu_);
      EnsureThreadInitializedLocked();
      const SlotHandle slot_handle = AcquireIdleSlot();
      auto& slot = SlotAt(slot_handle.qp_index, slot_handle.slot_in_qp);
      RequestDescriptor descriptor;
      FillPutDescriptor(
          &descriptor,
          slot.next_seq++,
          key_slice.size(),
          payload_bytes,
          slot.view);
      if (!RequestPayloadFitsSlot(payload_bytes)) {
        slot.busy = false;
        throw std::runtime_error("PUT request exceeds RC request slot");
      }
      recv   = AllocateStatusReceiveBufferLocked();
      rpc_id = SubmitRpcLocked(
          &slot, descriptor, payload.data(), payload_bytes, recv, 0, 0, true);
    }
    WaitRPCFinish(rpc_id);
    const auto* status = reinterpret_cast<const std::int32_t*>(recv);
    RevokeRPCResource(rpc_id);
    if (*status != static_cast<std::int32_t>(RpcStatus::kOk)) {
      return -1;
    }
    begin = end;
  }

  return 0;
}

int PetPSClient::InitEmbeddingTable(const std::string& table_name,
                                    std::uint64_t num_embeddings,
                                    std::uint64_t embedding_dim) {
  const std::array<std::uint64_t, 2> payload_words = {
      num_embeddings,
      embedding_dim,
  };

  float* recv = nullptr;
  int rpc_id  = 0;
  {
    std::lock_guard<std::mutex> guard(mu_);
    EnsureThreadInitializedLocked();
    const SlotHandle slot_handle = AcquireIdleSlot();
    auto& slot = SlotAt(slot_handle.qp_index, slot_handle.slot_in_qp);
    RequestDescriptor descriptor;
    FillInitTableDescriptor(
        &descriptor, slot.next_seq++, table_name, slot.view);
    if (!RequestPayloadFitsSlot(descriptor.payload_bytes)) {
      slot.busy = false;
      throw std::runtime_error("INIT request exceeds RC request slot");
    }
    recv   = AllocateStatusReceiveBufferLocked();
    rpc_id = SubmitRpcLocked(
        &slot,
        descriptor,
        payload_words.data(),
        descriptor.payload_bytes,
        recv,
        0,
        0,
        true);
  }

  WaitRPCFinish(rpc_id);
  const auto* status = reinterpret_cast<const std::int32_t*>(recv);
  RevokeRPCResource(rpc_id);
  return (*status == static_cast<std::int32_t>(RpcStatus::kOk)) ? 0 : -1;
}

int PetPSClient::UpdateParameter(const std::string& table_name,
                                 base::ConstArray<uint64_t> keys,
                                 const std::vector<std::vector<float>>* grads) {
  if (keys.Size() == 0) {
    return 0;
  }
  if (grads == nullptr) {
    return -1;
  }
  if (keys.Size() != grads->size()) {
    return -1;
  }

  std::size_t begin            = 0;
  const std::size_t total_keys = static_cast<std::size_t>(keys.Size());
  while (begin < total_keys) {
    const std::size_t end =
        std::min(begin + static_cast<std::size_t>(FLAGS_max_kv_num_per_request),
                 total_keys);
    std::vector<std::uint64_t> key_slice(
        keys.Data() + begin, keys.Data() + end);
    std::vector<std::vector<float>> grad_slice(
        grads->begin() + begin, grads->begin() + end);

    std::string payload;
    std::string error;
    const std::size_t payload_bytes =
        UpdatePayloadBytes(key_slice, grad_slice, &payload, &error);
    if (payload_bytes == 0 && !key_slice.empty()) {
      throw std::runtime_error("RC UPDATE payload build failed: " + error);
    }

    float* recv = nullptr;
    int rpc_id  = 0;
    {
      std::lock_guard<std::mutex> guard(mu_);
      EnsureThreadInitializedLocked();

      const SlotHandle slot_handle = AcquireIdleSlot();
      auto& slot = SlotAt(slot_handle.qp_index, slot_handle.slot_in_qp);
      RequestDescriptor descriptor;
      FillUpdateDescriptor(
          &descriptor,
          slot.next_seq++,
          key_slice.size(),
          payload_bytes,
          table_name,
          slot.view);
      if (!RequestPayloadFitsSlot(payload_bytes)) {
        slot.busy = false;
        throw std::runtime_error("UPDATE request exceeds RC request slot");
      }
      recv   = AllocateStatusReceiveBufferLocked();
      rpc_id = SubmitRpcLocked(
          &slot, descriptor, payload.data(), payload_bytes, recv, 0, 0, true);
    }

    WaitRPCFinish(rpc_id);
    const auto* status = reinterpret_cast<const std::int32_t*>(recv);
    RevokeRPCResource(rpc_id);
    if (*status != static_cast<std::int32_t>(RpcStatus::kOk)) {
      return -1;
    }
    begin = end;
  }

  return 0;
}

int PetPSClient::SubmitUpdateParameterFlat(
    const std::string& table_name,
    base::ConstArray<uint64_t> keys,
    const float* grads,
    std::size_t embedding_dim) {
  if (keys.Size() == 0) {
    return 0;
  }
  if (grads == nullptr || embedding_dim == 0 ||
      keys.Size() > static_cast<std::size_t>(FLAGS_max_kv_num_per_request)) {
    return -1;
  }

  std::string payload;
  std::string error;
  const std::size_t payload_bytes = UpdatePayloadBytesFlat(
      keys, grads, embedding_dim, &payload, &error);
  if (payload_bytes == 0) {
    throw std::runtime_error("RC UPDATE payload build failed: " + error);
  }

  std::lock_guard<std::mutex> guard(mu_);
  EnsureThreadInitializedLocked();
  const SlotHandle slot_handle = AcquireIdleSlot();
  auto& slot = SlotAt(slot_handle.qp_index, slot_handle.slot_in_qp);
  RequestDescriptor descriptor;
  FillUpdateDescriptor(
      &descriptor,
      slot.next_seq++,
      keys.Size(),
      payload_bytes,
      table_name,
      slot.view);
  if (!RequestPayloadFitsSlot(payload_bytes)) {
    slot.busy = false;
    throw std::runtime_error("UPDATE request exceeds RC request slot");
  }
  float* recv = AllocateStatusReceiveBufferLocked();
  return SubmitRpcLocked(
      &slot, descriptor, payload.data(), payload_bytes, recv, 0, 0, true);
}

int PetPSClient::WaitUpdateParameter(int rpc_id) {
  if (rpc_id == 0) {
    return 0;
  }
  WaitRPCFinish(rpc_id);
  PendingRpc pending;
  {
    std::lock_guard<std::mutex> guard(mu_);
    if (!PendingRpcLocked(rpc_id, &pending)) {
      return -1;
    }
  }
  const auto status = *reinterpret_cast<const std::int32_t*>(pending.recv_buffer);
  RevokeRPCResource(rpc_id);
  return status == static_cast<std::int32_t>(RpcStatus::kOk) ? 0 : -1;
}

int PetPSClient::FakePutParameter(base::ConstArray<uint64_t> keys,
                                  float* values) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  std::vector<std::vector<float>> rows;
  rows.reserve(keys.Size());
  for (int i = 0; i < keys.Size(); ++i) {
    rows.emplace_back(
        values + i * embedding_dim, values + (i + 1) * embedding_dim);
  }
  return PutParameter(keys.ToVector(), rows);
}

} // namespace petps
