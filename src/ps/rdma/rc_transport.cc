#include "ps/rdma/rc_transport.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <folly/portability/GFlags.h>

#include "ps/rdma/control_plane.h"
#include "ps/rdma/rdma_common.h"
#include "ps/rdma/rc_options.h"

namespace petps {
namespace {

using petps::Exchange;
using petps::NowNs;

enum class TrackedWrKind : std::uint64_t {
  kSubmitCommit   = 1,
  kResponseStatus = 2,
};

struct RcTransportProfileCounters {
  std::atomic<std::uint64_t> submit_request_count{0};
  std::atomic<std::uint64_t> submit_descriptor_write_count{0};
  std::atomic<std::uint64_t> submit_commit_write_count{0};
  std::atomic<std::uint64_t> submit_request_ns{0};
  std::atomic<std::uint64_t> drain_pending_submit_count{0};
  std::atomic<std::uint64_t> drain_pending_submit_ns{0};

  std::atomic<std::uint64_t> complete_response_count{0};
  std::atomic<std::uint64_t> response_payload_write_count{0};
  std::atomic<std::uint64_t> response_payload_sg_write_count{0};
  std::atomic<std::uint64_t> response_payload_sg_wr_count{0};
  std::atomic<std::uint64_t> response_status_write_count{0};
  std::atomic<std::uint64_t> response_payload_bytes{0};
  std::atomic<std::uint64_t> complete_response_ns{0};
  std::atomic<std::uint64_t> drain_pending_response_count{0};
  std::atomic<std::uint64_t> drain_pending_response_ns{0};
  std::atomic<std::uint64_t> next_report_ns{0};
};

RcTransportProfileCounters& TransportProfile() {
  static RcTransportProfileCounters counters;
  return counters;
}

void MaybeReportTransportProfile(const RcTransportConfig& config,
                                 const char* role) {
  if (FLAGS_rdma_rc_profile_interval_ms <= 0) {
    return;
  }
  auto& counters          = TransportProfile();
  const std::uint64_t now = NowNs();
  const std::uint64_t interval =
      static_cast<std::uint64_t>(FLAGS_rdma_rc_profile_interval_ms) * 1000000;
  std::uint64_t expected =
      counters.next_report_ns.load(std::memory_order_relaxed);
  if (expected == 0) {
    counters.next_report_ns.compare_exchange_strong(
        expected, now + interval, std::memory_order_relaxed);
    return;
  }
  if (now < expected ||
      !counters.next_report_ns.compare_exchange_strong(
          expected, now + interval, std::memory_order_relaxed)) {
    return;
  }

  const std::uint64_t submit_count = Exchange(&counters.submit_request_count);
  const std::uint64_t submit_ns    = Exchange(&counters.submit_request_ns);
  const std::uint64_t complete_count =
      Exchange(&counters.complete_response_count);
  const std::uint64_t complete_ns = Exchange(&counters.complete_response_ns);
  const std::uint64_t drain_submit_count =
      Exchange(&counters.drain_pending_submit_count);
  const std::uint64_t drain_submit_ns =
      Exchange(&counters.drain_pending_submit_ns);
  const std::uint64_t drain_response_count =
      Exchange(&counters.drain_pending_response_count);
  const std::uint64_t drain_response_ns =
      Exchange(&counters.drain_pending_response_ns);
  std::cout
      << "component=rdma_rc_transport_profile role=" << role
      << " shard=" << config.shard_id << " client_id=" << config.client_id
      << " submit_count=" << submit_count << " submit_descriptor_writes="
      << Exchange(&counters.submit_descriptor_write_count)
      << " submit_commit_writes="
      << Exchange(&counters.submit_commit_write_count)
      << " submit_avg_ns=" << (submit_count == 0 ? 0 : submit_ns / submit_count)
      << " drain_submit_count=" << drain_submit_count << " drain_submit_avg_ns="
      << (drain_submit_count == 0 ? 0 : drain_submit_ns / drain_submit_count)
      << " complete_count=" << complete_count << " response_payload_writes="
      << Exchange(&counters.response_payload_write_count)
      << " response_payload_sg_writes="
      << Exchange(&counters.response_payload_sg_write_count)
      << " response_payload_sg_wrs="
      << Exchange(&counters.response_payload_sg_wr_count)
      << " response_status_writes="
      << Exchange(&counters.response_status_write_count)
      << " response_payload_bytes="
      << Exchange(&counters.response_payload_bytes) << " complete_avg_ns="
      << (complete_count == 0 ? 0 : complete_ns / complete_count)
      << " drain_response_count=" << drain_response_count
      << " drain_response_avg_ns="
      << (drain_response_count == 0 ? 0
                                    : drain_response_ns / drain_response_count)
      << std::endl;
}

std::size_t TotalClientSlotsPerShard(const RcTransportConfig& config) {
  return static_cast<std::size_t>(config.qps_per_client_per_shard) *
         static_cast<std::size_t>(config.slots_per_qp);
}

std::size_t ClientSlotBytes(const RcTransportConfig& config) {
  return config.response_slot_bytes + config.request_slot_bytes;
}

std::size_t ClientLaneBytes(const RcTransportConfig& config) {
  return static_cast<std::size_t>(config.slots_per_qp) *
         ClientSlotBytes(config);
}

int LogicalClientsPerProcess(const RcTransportConfig& config) {
  if (config.num_os_clients <= 0) {
    throw std::runtime_error("num_os_clients must be positive");
  }
  if (config.num_clients < config.num_os_clients) {
    throw std::runtime_error(
        "logical client count smaller than OS client process count");
  }
  if (config.num_clients % config.num_os_clients != 0) {
    throw std::runtime_error(
        "logical client count must be divisible by OS client process count");
  }
  return config.num_clients / config.num_os_clients;
}

int OsClientIndexForLogicalClient(const RcTransportConfig& config,
                                  int client_id) {
  return client_id / LogicalClientsPerProcess(config);
}

int LocalLogicalClientIndex(const RcTransportConfig& config, int client_id) {
  return client_id % LogicalClientsPerProcess(config);
}

int RawLaneForLogicalClient(
    const RcTransportConfig& config, int client_id, int qp_index) {
  return LocalLogicalClientIndex(config, client_id) *
             config.qps_per_client_per_shard +
         qp_index;
}

int RawLanesPerOsClient(const RcTransportConfig& config) {
  return LogicalClientsPerProcess(config) * config.qps_per_client_per_shard;
}

std::size_t ServerLaneBytes(const RcTransportConfig& config) {
  return static_cast<std::size_t>(config.num_clients) *
         static_cast<std::size_t>(config.slots_per_qp) *
         config.request_slot_bytes;
}

std::size_t
ClientShardLaneOffset(const RcTransportConfig& config, int raw_lane) {
  return (static_cast<std::size_t>(config.shard_id) *
              static_cast<std::size_t>(RawLanesPerOsClient(config)) +
          static_cast<std::size_t>(raw_lane)) *
         ClientLaneBytes(config);
}

std::size_t ClientSlotOffset(const RcTransportConfig& config, int slot_in_qp) {
  return static_cast<std::size_t>(slot_in_qp) * ClientSlotBytes(config);
}

std::size_t ServerRequestOffset(
    const RcTransportConfig& config, int client_id, int slot_in_qp) {
  return (static_cast<std::size_t>(client_id) *
              static_cast<std::size_t>(config.slots_per_qp) +
          static_cast<std::size_t>(slot_in_qp)) *
         config.request_slot_bytes;
}

std::size_t ClientResponseOffsetForRawLane(
    const RcTransportConfig& config, int raw_lane, int slot_in_qp) {
  return ClientShardLaneOffset(config, raw_lane) +
         ClientSlotOffset(config, slot_in_qp);
}

std::size_t
ClientResponseOffset(const RcTransportConfig& config, int slot_in_qp) {
  return ClientResponseOffsetForRawLane(
      config, RawLaneForLogicalClient(config, config.client_id, 0), slot_in_qp);
}

std::size_t
ClientRequestStagingOffset(const RcTransportConfig& config, int slot_in_qp) {
  return ClientResponseOffset(config, slot_in_qp) + config.response_slot_bytes;
}

std::uint64_t RequestCommitOffset(const RcTransportConfig& config) {
  return config.request_slot_bytes - Align64(sizeof(CommitWord));
}

std::uint64_t ResponseStatusOffset(const RcTransportConfig& config) {
  return config.response_slot_bytes - Align64(sizeof(StatusWord));
}

int GlobalSlotIndex(const RcTransportConfig& config,
                    int client_id,
                    int qp_index,
                    int slot_in_qp) {
  return static_cast<int>(
      (static_cast<std::size_t>(client_id) * TotalClientSlotsPerShard(config)) +
      (static_cast<std::size_t>(qp_index) *
       static_cast<std::size_t>(config.slots_per_qp)) +
      static_cast<std::size_t>(slot_in_qp));
}

void DecodeGlobalSlotIndex(
    const RcTransportConfig& config,
    int slot_index,
    int* client_id,
    int* qp_index,
    int* slot_in_qp) {
  if (slot_index < 0) {
    throw std::runtime_error("slot_index out of range");
  }
  const std::size_t slots_per_client = TotalClientSlotsPerShard(config);
  if (slots_per_client == 0) {
    throw std::runtime_error("slots_per_client is zero");
  }
  const std::size_t slot = static_cast<std::size_t>(slot_index);
  if (client_id != nullptr) {
    *client_id = static_cast<int>(slot / slots_per_client);
  }
  const std::size_t slot_in_client = slot % slots_per_client;
  if (qp_index != nullptr) {
    *qp_index = static_cast<int>(
        slot_in_client / static_cast<std::size_t>(config.slots_per_qp));
  }
  if (slot_in_qp != nullptr) {
    *slot_in_qp = static_cast<int>(
        slot_in_client % static_cast<std::size_t>(config.slots_per_qp));
  }
}

int ResponseSlotOrdinal(
    const RcTransportConfig& config, int client_id, int slot_in_qp) {
  return client_id * config.slots_per_qp + slot_in_qp;
}

std::uint64_t MakeTrackedWrId(TrackedWrKind kind, int slot_ordinal) {
  return (static_cast<std::uint64_t>(kind) << 32) |
         static_cast<std::uint32_t>(slot_ordinal);
}

int DecodeTrackedWrId(
    TrackedWrKind kind, std::uint64_t wr_id, std::size_t expected_slots) {
  const std::uint64_t kind_bits = wr_id >> 32;
  if (kind_bits != static_cast<std::uint64_t>(kind)) {
    throw std::runtime_error("unexpected tracked RC WR kind");
  }
  const std::uint32_t slot_ordinal = static_cast<std::uint32_t>(wr_id);
  if (slot_ordinal >= expected_slots) {
    throw std::runtime_error("tracked RC WR slot ordinal out of range");
  }
  return static_cast<int>(slot_ordinal);
}

void WaitForTrackedCompletion(
    RawVerbsTransport* verbs,
    std::vector<std::uint8_t>* ready,
    TrackedWrKind kind,
    int slot_ordinal,
    const std::string& context) {
  if (verbs == nullptr || ready == nullptr) {
    throw std::runtime_error("tracked completion state is null");
  }
  auto& ready_flags = *ready;
  if (slot_ordinal < 0 ||
      static_cast<std::size_t>(slot_ordinal) >= ready_flags.size()) {
    throw std::runtime_error("tracked completion slot ordinal out of range");
  }
  if (ready_flags[static_cast<std::size_t>(slot_ordinal)] != 0) {
    ready_flags[static_cast<std::size_t>(slot_ordinal)] = 0;
    return;
  }

  while (true) {
    RawVerbsCompletion completion;
    if (!verbs->Poll(&completion, FLAGS_rdma_wait_timeout_ms)) {
      throw std::runtime_error(
          "RC verbs write completion timeout " + context + " expected_wr_id=" +
          std::to_string(MakeTrackedWrId(kind, slot_ordinal)));
    }
    const int completed_slot =
        DecodeTrackedWrId(kind, completion.wr_id, ready_flags.size());
    if (completed_slot == slot_ordinal) {
      return;
    }
    ready_flags[static_cast<std::size_t>(completed_slot)] = 1;
  }
}

void DrainTrackedPendingWrite(
    RawVerbsTransport* verbs,
    std::vector<std::uint8_t>* pending,
    std::vector<std::uint8_t>* ready,
    TrackedWrKind kind,
    int slot_ordinal,
    const std::string& context) {
  if (pending == nullptr || ready == nullptr) {
    return;
  }
  auto& pending_flags = *pending;
  if (slot_ordinal < 0 ||
      static_cast<std::size_t>(slot_ordinal) >= pending_flags.size()) {
    throw std::runtime_error("tracked pending slot ordinal out of range");
  }
  if (pending_flags[static_cast<std::size_t>(slot_ordinal)] == 0) {
    return;
  }
  WaitForTrackedCompletion(verbs, ready, kind, slot_ordinal, context);
  pending_flags[static_cast<std::size_t>(slot_ordinal)] = 0;
}

void DrainTrackedPendingWrite(
    RawVerbsTransport* verbs,
    std::vector<std::uint8_t>* pending,
    std::vector<std::uint8_t>* ready,
    TrackedWrKind kind,
    int slot_ordinal,
    const std::string& context,
    bool profile_enabled,
    std::atomic<std::uint64_t>* drain_count,
    std::atomic<std::uint64_t>* drain_ns) {
  if (pending == nullptr || ready == nullptr || drain_count == nullptr ||
      drain_ns == nullptr) {
    return;
  }
  if (slot_ordinal < 0 ||
      static_cast<std::size_t>(slot_ordinal) >= pending->size() ||
      pending->at(static_cast<std::size_t>(slot_ordinal)) == 0) {
    return;
  }
  if (!profile_enabled) {
    DrainTrackedPendingWrite(
        verbs, pending, ready, kind, slot_ordinal, context);
    return;
  }

  const std::uint64_t drain_start_ns = NowNs();
  DrainTrackedPendingWrite(verbs, pending, ready, kind, slot_ordinal, context);
  drain_count->fetch_add(1, std::memory_order_relaxed);
  drain_ns->fetch_add(NowNs() - drain_start_ns, std::memory_order_relaxed);
}

std::string WriteContext(
    const RcTransportConfig& config,
    int client_id,
    int qp_index,
    int slot_in_qp,
    std::uint64_t seq,
    std::uint64_t remote_offset,
    int remote_node,
    const char* phase) {
  return "phase=" + std::string(phase) +
         " shard=" + std::to_string(config.shard_id) + " client_id=" +
         std::to_string(client_id) + " qp=" + std::to_string(qp_index) +
         " slot_in_qp=" + std::to_string(slot_in_qp) + " seq=" +
         std::to_string(seq) + " remote_node=" + std::to_string(remote_node) +
         " remote_offset=" + std::to_string(remote_offset);
}

void ValidateClientId(const RcTransportConfig& config, int client_id) {
  if (client_id < 0 || client_id >= config.num_clients) {
    throw std::runtime_error("client_id out of range");
  }
}

void ValidateSlotInQp(const RcTransportConfig& config, int slot_in_qp) {
  if (slot_in_qp < 0 || slot_in_qp >= config.slots_per_qp) {
    throw std::runtime_error("slot_in_qp out of range");
  }
}

void ValidateTransportConfig(const RcTransportConfig& config) {
  if (config.node_id < 0 || config.num_servers <= 0 ||
      config.num_os_clients <= 0 || config.num_clients <= 0) {
    throw std::runtime_error(
        "RDMA transport deployment identity is incomplete");
  }
  if (config.request_slot_bytes <=
          Align64(sizeof(RequestDescriptor)) + Align64(sizeof(CommitWord)) ||
      config.response_slot_bytes <= Align64(sizeof(StatusWord))) {
    throw std::runtime_error("RDMA request/response slot bytes are too small");
  }
}

RawVerbsConfig MakeRawConfig(
    const RcTransportConfig& config,
    int local_lane,
    std::size_t local_region_bytes,
    bool is_client,
    int only_node_id) {
  RawVerbsConfig raw;
  raw.global_id        = config.node_id;
  raw.local_lane       = local_lane;
  raw.remote_lane      = local_lane;
  raw.only_node_id     = only_node_id;
  raw.num_servers      = config.num_servers;
  raw.num_clients      = config.num_os_clients;
  raw.device_name      = config.fabric.device;
  raw.port_num         = static_cast<std::uint8_t>(config.fabric.port);
  raw.fabric_mode      = config.fabric.mode;
  raw.gid_index        = config.fabric.gid_index;
  raw.hop_limit        = static_cast<std::uint8_t>(config.fabric.hop_limit);
  raw.traffic_class    = static_cast<std::uint8_t>(config.fabric.traffic_class);
  raw.flow_label       = config.fabric.flow_label;
  raw.deployment_id    = config.deployment_id;
  raw.deployment_epoch = config.deployment_epoch;
  raw.protocol_version = config.protocol_version;
  raw.configuration_digest = config.configuration_digest;
  raw.fabric_digest        = config.fabric_digest;
  raw.max_inline_data =
      static_cast<std::uint32_t>(std::max(0, FLAGS_rdma_rc_inline_bytes));
  raw.connect_to_servers       = is_client;
  raw.connect_to_clients       = !is_client;
  raw.local_region_bytes       = local_region_bytes;
  raw.control_plane_host       = config.control_plane_host;
  raw.control_plane_port       = config.control_plane_port;
  raw.control_plane_timeout_ms = config.control_plane_timeout_ms;
  return raw;
}

} // namespace

RcShardClientTransport::RcShardClientTransport(const RcTransportConfig& config)
    : config_(config), server_node_id_(config.shard_id) {
  ValidateTransportConfig(config_);
  ValidateClientId(config_, config_.client_id);
  if (config_.slots_per_qp <= 0) {
    throw std::runtime_error("slots_per_qp must be positive");
  }
  if (server_node_id_ < 0 || server_node_id_ >= config_.num_servers) {
    throw std::runtime_error("server shard id out of global node range");
  }
  lanes_.reserve(static_cast<std::size_t>(config_.qps_per_client_per_shard));
  for (int qp = 0; qp < config_.qps_per_client_per_shard; ++qp) {
    Lane lane;
    const int raw_lane =
        RawLaneForLogicalClient(config_, config_.client_id, qp);
    const std::size_t local_bytes =
        static_cast<std::size_t>(config_.num_servers) *
        static_cast<std::size_t>(RawLanesPerOsClient(config_)) *
        ClientLaneBytes(config_);
    RawVerbsConfig raw =
        MakeRawConfig(config_, raw_lane, local_bytes, true, server_node_id_);
    raw.reserved_region_offset = ClientShardLaneOffset(config_, raw_lane);
    raw.reserved_region_bytes  = ClientLaneBytes(config_);
    lane.verbs                 = std::make_unique<RawVerbsTransport>(raw);
    lane.lane_base             = lane.verbs->LocalPointer(GlobalAddress{
        static_cast<std::uint16_t>(config_.node_id),
        static_cast<std::uint64_t>(ClientShardLaneOffset(config_, raw_lane)),
    });
    std::memset(lane.lane_base, 0, ClientLaneBytes(config_));
    lane.submit_completion_pending.assign(
        static_cast<std::size_t>(config_.slots_per_qp), 0);
    lane.submit_completion_ready.assign(
        static_cast<std::size_t>(config_.slots_per_qp), 0);
    lane.verbs->PublishAndConnect();
    lanes_.push_back(std::move(lane));
  }
}

RcShardClientTransport::~RcShardClientTransport() {
  try {
    for (std::size_t qp = 0; qp < lanes_.size(); ++qp) {
      Lane& lane = lanes_[qp];
      if (!lane.verbs) {
        continue;
      }
      for (int slot_in_qp = 0; slot_in_qp < config_.slots_per_qp;
           ++slot_in_qp) {
        DrainTrackedPendingWrite(
            lane.verbs.get(),
            &lane.submit_completion_pending,
            &lane.submit_completion_ready,
            TrackedWrKind::kSubmitCommit,
            slot_in_qp,
            WriteContext(
                config_,
                config_.client_id,
                static_cast<int>(qp),
                slot_in_qp,
                0,
                ServerRequestOffset(config_, config_.client_id, slot_in_qp) +
                    RequestCommitOffset(config_),
                server_node_id_,
                "shutdown_submit_commit"));
      }
    }
  } catch (...) {
  }
}

RcShardClientTransport::Lane& RcShardClientTransport::LaneAt(int qp_index) {
  if (qp_index < 0 || qp_index >= config_.qps_per_client_per_shard) {
    throw std::runtime_error("qp_index out of range");
  }
  return lanes_.at(static_cast<std::size_t>(qp_index));
}

const RcShardClientTransport::Lane&
RcShardClientTransport::LaneAt(int qp_index) const {
  if (qp_index < 0 || qp_index >= config_.qps_per_client_per_shard) {
    throw std::runtime_error("qp_index out of range");
  }
  return lanes_.at(static_cast<std::size_t>(qp_index));
}

RcClientQpView RcShardClientTransport::OpenQp(int qp_index) {
  return OpenSlot(qp_index, 0);
}

RcClientQpView RcShardClientTransport::OpenSlot(int qp_index, int slot_in_qp) {
  ValidateSlotInQp(config_, slot_in_qp);
  const Lane& lane = LaneAt(qp_index);
  auto* slot_base  = static_cast<char*>(lane.lane_base) +
                    ClientSlotOffset(config_, slot_in_qp);
  auto* response_slot    = static_cast<void*>(slot_base);
  auto* response_payload = slot_base;
  auto* status           = reinterpret_cast<StatusWord*>(
      response_payload + config_.response_slot_bytes -
      Align64(sizeof(StatusWord)));

  auto* request_slot = slot_base + config_.response_slot_bytes;
  auto* descriptor   = reinterpret_cast<RequestDescriptor*>(request_slot);
  auto* payload      = request_slot + Align64(sizeof(RequestDescriptor));
  auto* commit       = reinterpret_cast<CommitWord*>(
      request_slot + config_.request_slot_bytes - Align64(sizeof(CommitWord)));
  const int slot_index =
      GlobalSlotIndex(config_, config_.client_id, qp_index, slot_in_qp);

  return RcClientQpView{
      qp_index,
      slot_in_qp,
      slot_index,
      request_slot,
      descriptor,
      payload,
      commit,
      response_slot,
      response_payload,
      status,
  };
}

void RcShardClientTransport::SubmitRequest(
    const RcClientQpView& view,
    const RequestDescriptor& descriptor,
    const void* payload,
    std::size_t payload_bytes) {
  const bool profile_enabled   = FLAGS_rdma_rc_profile_interval_ms > 0;
  const std::uint64_t start_ns = profile_enabled ? NowNs() : 0;
  ValidateSlotInQp(config_, view.slot_in_qp);
  Lane& lane = LaneAt(view.qp_index);
  const std::uint64_t remote_request_offset =
      ServerRequestOffset(config_, config_.client_id, view.slot_in_qp);
  auto& counters = TransportProfile();
  DrainTrackedPendingWrite(
      lane.verbs.get(),
      &lane.submit_completion_pending,
      &lane.submit_completion_ready,
      TrackedWrKind::kSubmitCommit,
      view.slot_in_qp,
      WriteContext(
          config_,
          config_.client_id,
          view.qp_index,
          view.slot_in_qp,
          descriptor.seq - 1,
          remote_request_offset + RequestCommitOffset(config_),
          server_node_id_,
          "previous_submit_commit"),
      profile_enabled,
      &counters.drain_pending_submit_count,
      &counters.drain_pending_submit_ns);

  auto* request_slot     = static_cast<char*>(view.request_slot);
  auto* local_descriptor = reinterpret_cast<RequestDescriptor*>(request_slot);
  auto* local_payload    = request_slot + Align64(sizeof(RequestDescriptor));
  auto* local_commit     = reinterpret_cast<CommitWord*>(
      request_slot + RequestCommitOffset(config_));
  *local_descriptor = descriptor;
  if (payload_bytes > 0 && payload != local_payload) {
    std::memcpy(local_payload, payload, payload_bytes);
  }
  local_commit->seq.store(descriptor.seq, std::memory_order_release);
  local_commit->state.store(kRcSlotReady, std::memory_order_release);

  lane.verbs->Write(
      request_slot,
      GlobalAddress{
          static_cast<std::uint16_t>(server_node_id_),
          remote_request_offset,
      },
      Align64(sizeof(RequestDescriptor)) + payload_bytes,
      /*wr_id=*/0,
      false);
  if (profile_enabled) {
    TransportProfile().submit_descriptor_write_count.fetch_add(
        1, std::memory_order_relaxed);
  }

  lane.verbs->Write(
      local_commit,
      GlobalAddress{
          static_cast<std::uint16_t>(server_node_id_),
          remote_request_offset + RequestCommitOffset(config_),
      },
      sizeof(CommitWord),
      MakeTrackedWrId(TrackedWrKind::kSubmitCommit, view.slot_in_qp),
      true);
  lane.submit_completion_pending[static_cast<std::size_t>(view.slot_in_qp)] = 1;
  if (profile_enabled) {
    auto& profile = TransportProfile();
    profile.submit_request_count.fetch_add(1, std::memory_order_relaxed);
    profile.submit_commit_write_count.fetch_add(1, std::memory_order_relaxed);
    profile.submit_request_ns.fetch_add(
        NowNs() - start_ns, std::memory_order_relaxed);
    MaybeReportTransportProfile(config_, "client");
  }
}

void RcShardClientTransport::ClearRequestSlot(const RcClientQpView& view) {
  auto* commit = view.commit;
  commit->state.store(0, std::memory_order_release);
}

RcShardServerTransport::RcShardServerTransport(const RcTransportConfig& config)
    : config_(config) {
  ValidateTransportConfig(config_);
  if (config_.node_id < 0 || config_.node_id >= config_.num_servers) {
    throw std::runtime_error("server global_id out of range");
  }
  if (config_.slots_per_qp <= 0) {
    throw std::runtime_error("slots_per_qp must be positive");
  }
  lanes_.reserve(static_cast<std::size_t>(config_.qps_per_client_per_shard));
  const int raw_lane_count = RawLanesPerOsClient(config_);
  for (int raw_lane = 0; raw_lane < raw_lane_count; ++raw_lane) {
    Lane lane;
    const int response_slots = config_.num_clients * config_.slots_per_qp;
    const std::size_t local_bytes =
        ServerLaneBytes(config_) +
        static_cast<std::size_t>(response_slots) * config_.response_slot_bytes;
    lane.verbs = std::make_unique<RawVerbsTransport>(
        MakeRawConfig(config_, raw_lane, local_bytes, false, -1));
    lane.request_slots =
        lane.verbs->AllocateRegistered(ServerLaneBytes(config_));
    std::memset(lane.request_slots, 0, ServerLaneBytes(config_));
    lane.response_staging.reserve(static_cast<std::size_t>(response_slots));
    lane.response_completion_pending.assign(
        static_cast<std::size_t>(response_slots), 0);
    lane.response_completion_ready.assign(
        static_cast<std::size_t>(response_slots), 0);
    for (int slot = 0; slot < response_slots; ++slot) {
      void* response_slot =
          lane.verbs->AllocateRegistered(config_.response_slot_bytes);
      std::memset(response_slot, 0, config_.response_slot_bytes);
      lane.response_staging.push_back(response_slot);
    }
    lane.verbs->PublishAndConnect();
    lanes_.push_back(std::move(lane));
  }
}

RcShardServerTransport::~RcShardServerTransport() {
  try {
    const int logical_clients_per_process = LogicalClientsPerProcess(config_);
    for (std::size_t raw_lane_index = 0; raw_lane_index < lanes_.size();
         ++raw_lane_index) {
      Lane& lane = lanes_[raw_lane_index];
      if (!lane.verbs) {
        continue;
      }
      const int local_logical_client = static_cast<int>(
          raw_lane_index /
          static_cast<std::size_t>(config_.qps_per_client_per_shard));
      const int qp_index = static_cast<int>(
          raw_lane_index %
          static_cast<std::size_t>(config_.qps_per_client_per_shard));
      for (int os_client = 0; os_client < config_.num_os_clients; ++os_client) {
        const int client =
            os_client * logical_clients_per_process + local_logical_client;
        for (int slot_in_qp = 0; slot_in_qp < config_.slots_per_qp;
             ++slot_in_qp) {
          const int raw_lane =
              RawLaneForLogicalClient(config_, client, qp_index);
          const int client_node_id =
              config_.num_servers +
              OsClientIndexForLogicalClient(config_, client);
          const int response_slot =
              ResponseSlotOrdinal(config_, client, slot_in_qp);
          DrainTrackedPendingWrite(
              lane.verbs.get(),
              &lane.response_completion_pending,
              &lane.response_completion_ready,
              TrackedWrKind::kResponseStatus,
              response_slot,
              WriteContext(
                  config_,
                  client,
                  qp_index,
                  slot_in_qp,
                  0,
                  ClientResponseOffsetForRawLane(
                      config_, raw_lane, slot_in_qp) +
                      ResponseStatusOffset(config_),
                  client_node_id,
                  "shutdown_response_status"));
        }
      }
    }
  } catch (...) {
  }
}

RcShardServerTransport::Lane&
RcShardServerTransport::LaneAt(int client_id, int qp_index) {
  ValidateClientId(config_, client_id);
  if (qp_index < 0 || qp_index >= config_.qps_per_client_per_shard) {
    throw std::runtime_error("qp_index out of range");
  }
  return lanes_.at(static_cast<std::size_t>(
      RawLaneForLogicalClient(config_, client_id, qp_index)));
}

const RcShardServerTransport::Lane&
RcShardServerTransport::LaneAt(int client_id, int qp_index) const {
  ValidateClientId(config_, client_id);
  if (qp_index < 0 || qp_index >= config_.qps_per_client_per_shard) {
    throw std::runtime_error("qp_index out of range");
  }
  return lanes_.at(static_cast<std::size_t>(
      RawLaneForLogicalClient(config_, client_id, qp_index)));
}

int RcShardServerTransport::TotalSlots() const {
  return static_cast<int>(static_cast<std::size_t>(config_.num_clients) *
                          TotalClientSlotsPerShard(config_));
}

void RcShardServerTransport::RegisterLocalMemoryRegion(
    void* base, std::size_t bytes) {
  for (auto& lane : lanes_) {
    if (lane.verbs) {
      lane.verbs->RegisterMemoryRegion(base, bytes);
    }
  }
}

int RcShardServerTransport::SlotIndex(
    int client_id, int qp_index, int slot_in_qp) const {
  ValidateClientId(config_, client_id);
  ValidateSlotInQp(config_, slot_in_qp);
  if (qp_index < 0 || qp_index >= config_.qps_per_client_per_shard) {
    throw std::runtime_error("qp_index out of range");
  }
  return GlobalSlotIndex(config_, client_id, qp_index, slot_in_qp);
}

void RcShardServerTransport::DecodeSlotIndex(
    int slot_index, int* client_id, int* qp_index, int* slot_in_qp) const {
  if (slot_index < 0 || slot_index >= TotalSlots()) {
    throw std::runtime_error("slot_index out of range");
  }
  DecodeGlobalSlotIndex(config_, slot_index, client_id, qp_index, slot_in_qp);
}

void* RcShardServerTransport::RequestSlot(int slot_index) const {
  int client_id  = -1;
  int qp_index   = -1;
  int slot_in_qp = -1;
  DecodeSlotIndex(slot_index, &client_id, &qp_index, &slot_in_qp);
  const Lane& lane = LaneAt(client_id, qp_index);
  return static_cast<char*>(lane.request_slots) +
         ServerRequestOffset(config_, client_id, slot_in_qp);
}

RequestDescriptor*
RcShardServerTransport::RequestDescriptorAt(int slot_index) const {
  return reinterpret_cast<RequestDescriptor*>(RequestSlot(slot_index));
}

char* RcShardServerTransport::RequestPayloadAt(int slot_index) const {
  return static_cast<char*>(RequestSlot(slot_index)) +
         Align64(sizeof(RequestDescriptor));
}

CommitWord* RcShardServerTransport::RequestCommitAt(int slot_index) const {
  return reinterpret_cast<CommitWord*>(
      static_cast<char*>(RequestSlot(slot_index)) +
      RequestCommitOffset(config_));
}

RcShardServerTransport::ResponseView RcShardServerTransport::OpenClientResponse(
    int client_id, int qp_index, int slot_in_qp) {
  ValidateClientId(config_, client_id);
  ValidateSlotInQp(config_, slot_in_qp);
  Lane& lane = LaneAt(client_id, qp_index);
  auto* slot =
      static_cast<char*>(lane.response_staging.at(static_cast<std::size_t>(
          ResponseSlotOrdinal(config_, client_id, slot_in_qp))));
  auto* payload = static_cast<char*>(slot);
  auto* status =
      reinterpret_cast<StatusWord*>(payload + ResponseStatusOffset(config_));
  return ResponseView{slot, payload, status};
}

void RcShardServerTransport::CompleteResponse(
    int client_id,
    int qp_index,
    int slot_in_qp,
    const ResponseView& response,
    std::uint64_t seq) {
  const bool profile_enabled   = FLAGS_rdma_rc_profile_interval_ms > 0;
  const std::uint64_t start_ns = profile_enabled ? NowNs() : 0;
  ValidateClientId(config_, client_id);
  ValidateSlotInQp(config_, slot_in_qp);
  Lane& lane              = LaneAt(client_id, qp_index);
  const int response_slot = ResponseSlotOrdinal(config_, client_id, slot_in_qp);
  const int raw_lane = RawLaneForLogicalClient(config_, client_id, qp_index);
  const int client_node_id =
      config_.num_servers + OsClientIndexForLogicalClient(config_, client_id);
  auto& counters = TransportProfile();
  DrainTrackedPendingWrite(
      lane.verbs.get(),
      &lane.response_completion_pending,
      &lane.response_completion_ready,
      TrackedWrKind::kResponseStatus,
      response_slot,
      WriteContext(
          config_,
          client_id,
          qp_index,
          slot_in_qp,
          seq - 1,
          ClientResponseOffsetForRawLane(config_, raw_lane, slot_in_qp) +
              ResponseStatusOffset(config_),
          client_node_id,
          "previous_response_status"),
      profile_enabled,
      &counters.drain_pending_response_count,
      &counters.drain_pending_response_ns);
  response.status->seq.store(seq, std::memory_order_release);
  response.status->state.store(kRcSlotDone, std::memory_order_release);

  if (response.status->response_bytes > 0) {
    const std::uint64_t response_payload_offset =
        ClientResponseOffsetForRawLane(config_, raw_lane, slot_in_qp);
    lane.verbs->Write(
        response.payload,
        GlobalAddress{
            static_cast<std::uint16_t>(client_node_id),
            response_payload_offset,
        },
        response.status->response_bytes,
        /*wr_id=*/0,
        false);
    if (profile_enabled) {
      auto& profile = TransportProfile();
      profile.response_payload_write_count.fetch_add(
          1, std::memory_order_relaxed);
      profile.response_payload_bytes.fetch_add(
          response.status->response_bytes, std::memory_order_relaxed);
    }
  }

  const std::uint64_t response_status_offset =
      ClientResponseOffsetForRawLane(config_, raw_lane, slot_in_qp) +
      ResponseStatusOffset(config_);
  lane.verbs->Write(
      response.status,
      GlobalAddress{
          static_cast<std::uint16_t>(client_node_id),
          response_status_offset,
      },
      sizeof(StatusWord),
      MakeTrackedWrId(TrackedWrKind::kResponseStatus, response_slot),
      true);
  lane.response_completion_pending[static_cast<std::size_t>(response_slot)] = 1;
  if (profile_enabled) {
    auto& profile = TransportProfile();
    profile.complete_response_count.fetch_add(1, std::memory_order_relaxed);
    profile.response_status_write_count.fetch_add(1, std::memory_order_relaxed);
    profile.complete_response_ns.fetch_add(
        NowNs() - start_ns, std::memory_order_relaxed);
    MaybeReportTransportProfile(config_, "server");
  }
}

void RcShardServerTransport::WriteResponsePayloadSg(
    int client_id,
    int qp_index,
    int slot_in_qp,
    base::ConstArray<RawVerbsSge> sges,
    std::uint64_t response_offset,
    std::uint64_t bytes) {
  ValidateClientId(config_, client_id);
  ValidateSlotInQp(config_, slot_in_qp);
  if (sges.Size() == 0 || bytes == 0) {
    return;
  }
  Lane& lane         = LaneAt(client_id, qp_index);
  const int raw_lane = RawLaneForLogicalClient(config_, client_id, qp_index);
  const int client_node_id =
      config_.num_servers + OsClientIndexForLogicalClient(config_, client_id);
  lane.verbs->WriteSg(
      sges,
      GlobalAddress{
          static_cast<std::uint16_t>(client_node_id),
          ClientResponseOffsetForRawLane(config_, raw_lane, slot_in_qp) +
              response_offset,
      },
      /*wr_id=*/0,
      false);
  if (FLAGS_rdma_rc_profile_interval_ms > 0) {
    auto& profile = TransportProfile();
    profile.response_payload_sg_write_count.fetch_add(
        1, std::memory_order_relaxed);
    profile.response_payload_sg_wr_count.fetch_add(
        1, std::memory_order_relaxed);
    profile.response_payload_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
}

void RcShardServerTransport::CompleteResponseStatusOnly(
    int client_id,
    int qp_index,
    int slot_in_qp,
    const ResponseView& response,
    std::uint64_t seq) {
  const bool profile_enabled   = FLAGS_rdma_rc_profile_interval_ms > 0;
  const std::uint64_t start_ns = profile_enabled ? NowNs() : 0;
  ValidateClientId(config_, client_id);
  ValidateSlotInQp(config_, slot_in_qp);
  Lane& lane              = LaneAt(client_id, qp_index);
  const int response_slot = ResponseSlotOrdinal(config_, client_id, slot_in_qp);
  const int raw_lane = RawLaneForLogicalClient(config_, client_id, qp_index);
  const int client_node_id =
      config_.num_servers + OsClientIndexForLogicalClient(config_, client_id);
  auto& counters = TransportProfile();
  DrainTrackedPendingWrite(
      lane.verbs.get(),
      &lane.response_completion_pending,
      &lane.response_completion_ready,
      TrackedWrKind::kResponseStatus,
      response_slot,
      WriteContext(
          config_,
          client_id,
          qp_index,
          slot_in_qp,
          seq - 1,
          ClientResponseOffsetForRawLane(config_, raw_lane, slot_in_qp) +
              ResponseStatusOffset(config_),
          client_node_id,
          "previous_response_status"),
      profile_enabled,
      &counters.drain_pending_response_count,
      &counters.drain_pending_response_ns);
  response.status->seq.store(seq, std::memory_order_release);
  response.status->state.store(kRcSlotDone, std::memory_order_release);

  const std::uint64_t response_status_offset =
      ClientResponseOffsetForRawLane(config_, raw_lane, slot_in_qp) +
      ResponseStatusOffset(config_);
  lane.verbs->Write(
      response.status,
      GlobalAddress{
          static_cast<std::uint16_t>(client_node_id),
          response_status_offset,
      },
      sizeof(StatusWord),
      MakeTrackedWrId(TrackedWrKind::kResponseStatus, response_slot),
      true);
  lane.response_completion_pending[static_cast<std::size_t>(response_slot)] = 1;
  if (profile_enabled) {
    auto& profile = TransportProfile();
    profile.complete_response_count.fetch_add(1, std::memory_order_relaxed);
    profile.response_status_write_count.fetch_add(1, std::memory_order_relaxed);
    profile.complete_response_ns.fetch_add(
        NowNs() - start_ns, std::memory_order_relaxed);
    MaybeReportTransportProfile(config_, "server");
  }
}

} // namespace petps
