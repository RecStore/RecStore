#include "ps/rdma/raw_verbs_transport.h"

#include <arpa/inet.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#include "ps/rdma/control_plane.h"

namespace petps {
namespace {

constexpr int kRawVerbsPort          = 1;
constexpr std::uint32_t kRawVerbsPsn = 3185;
constexpr int kRawVerbsCqDepth       = 4096;
constexpr int kRawVerbsRecvDepth     = 1024;

std::string IbvError(const char* op) { return std::string(op) + " failed"; }

std::string QpCreateError(const RawVerbsConfig& config, int node) {
  return "ibv_create_qp failed: likely insufficient RDMA QP resources "
         "(global_id=" +
         std::to_string(config.global_id) +
         ", local_lane=" + std::to_string(config.local_lane) +
         ", remote_lane=" + std::to_string(config.remote_lane) +
         ", node=" + std::to_string(node) +
         ", num_servers=" + std::to_string(config.num_servers) +
         ", num_clients=" + std::to_string(config.num_clients) +
         "). Reduce --client-count or --qps-per-client-per-shard.";
}

struct OpenedRawVerbsDevice {
  ibv_context* context = nullptr;
  int gid_index        = -1;
};

int ReadDeviceNumaNode(ibv_device* device) {
  std::ifstream input(std::string("/sys/class/infiniband/") +
                      ibv_get_device_name(device) + "/device/numa_node");
  int numa_node = -1;
  input >> numa_node;
  return numa_node;
}

int FindUsableGidIndex(ibv_context* context) {
  ibv_port_attr port_attr{};
  if (ibv_query_port(context, kRawVerbsPort, &port_attr) != 0 ||
      port_attr.state != IBV_PORT_ACTIVE || port_attr.gid_tbl_len <= 0) {
    return -1;
  }
  std::vector<ibv_gid> gids(static_cast<std::size_t>(port_attr.gid_tbl_len));
  std::vector<bool> roce_v2_gids(
      static_cast<std::size_t>(port_attr.gid_tbl_len));
  const std::string gid_type_dir =
      std::string("/sys/class/infiniband/") +
      ibv_get_device_name(context->device) + "/ports/" +
      std::to_string(kRawVerbsPort) + "/gid_attrs/types/";
  for (int i = 0; i < port_attr.gid_tbl_len; ++i) {
    if (ibv_query_gid(context, kRawVerbsPort, i, &gids[i]) != 0) {
      std::memset(&gids[i], 0, sizeof(gids[i]));
    }
    std::ifstream type_input(gid_type_dir + std::to_string(i));
    std::string gid_type;
    std::getline(type_input, gid_type);
    roce_v2_gids[static_cast<std::size_t>(i)] = gid_type == "RoCE v2";
  }
  return SelectRawVerbsGidIndex(gids, roce_v2_gids);
}

OpenedRawVerbsDevice OpenDeviceForNuma(int numa_id) {
  // ibv_fork_init must be called before any other libibverbs API.
  // It sets MADV_DONTFORK on all mmap'd regions (CQ/QP memory) so that
  // forked child processes (e.g. DataLoader workers) don't inherit them.
  // Without this, child exit corrupts the parent's RDMA state.
  static const bool fork_init_done = []() {
    ibv_fork_init();
    return true;
  }();
  (void)fork_init_done;

  int device_count     = 0;
  ibv_device** devices = ibv_get_device_list(&device_count);
  if (devices == nullptr || device_count == 0) {
    throw std::runtime_error("no RDMA devices found");
  }
  std::vector<int> device_numa_nodes(static_cast<std::size_t>(device_count));
  std::vector<bool> usable_devices(static_cast<std::size_t>(device_count));
  std::vector<int> gid_indices(static_cast<std::size_t>(device_count), -1);
  for (int i = 0; i < device_count; ++i) {
    device_numa_nodes[static_cast<std::size_t>(i)] =
        ReadDeviceNumaNode(devices[i]);
    ibv_context* candidate = ibv_open_device(devices[i]);
    if (candidate == nullptr) {
      continue;
    }
    gid_indices[static_cast<std::size_t>(i)] = FindUsableGidIndex(candidate);
    usable_devices[static_cast<std::size_t>(i)] =
        gid_indices[static_cast<std::size_t>(i)] >= 0;
    ibv_close_device(candidate);
  }
  const int device_index = SelectRawVerbsDeviceIndex(
      numa_id, device_numa_nodes, usable_devices);
  if (device_index < 0) {
    ibv_free_device_list(devices);
    throw std::runtime_error("no active RDMA device with a non-zero GID found");
  }
  ibv_context* context = ibv_open_device(devices[device_index]);
  const int gid_index  = gid_indices[static_cast<std::size_t>(device_index)];
  ibv_free_device_list(devices);
  if (context == nullptr) {
    throw std::runtime_error("ibv_open_device failed");
  }
  return {context, gid_index};
}

void ModifyQpToInit(ibv_qp* qp) {
  ibv_qp_attr attr{};
  attr.qp_state        = IBV_QPS_INIT;
  attr.port_num        = kRawVerbsPort;
  attr.pkey_index      = 0;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
                         IBV_ACCESS_REMOTE_ATOMIC;
  const int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    throw std::runtime_error(IbvError("ibv_modify_qp INIT"));
  }
}

void FillAhAttr(ibv_ah_attr* ah_attr,
                std::uint16_t remote_lid,
                const std::uint8_t* remote_gid,
                int local_gid_index) {
  std::memset(ah_attr, 0, sizeof(*ah_attr));
  ah_attr->dlid          = remote_lid;
  ah_attr->sl            = 0;
  ah_attr->src_path_bits = 0;
  ah_attr->port_num      = kRawVerbsPort;
  if (remote_gid != nullptr) {
    ah_attr->is_global = 1;
    std::memcpy(&ah_attr->grh.dgid, remote_gid, 16);
    ah_attr->grh.sgid_index = local_gid_index;
    ah_attr->grh.hop_limit  = 1;
  }
}

void ModifyQpToRtr(ibv_qp* qp,
                   const RawVerbsNodeMeta& remote,
                   int local_gid_index) {
  ibv_port_attr port_attr{};
  if (ibv_query_port(qp->context, kRawVerbsPort, &port_attr) != 0) {
    throw std::runtime_error(IbvError("ibv_query_port for active MTU"));
  }
  ibv_qp_attr attr{};
  attr.qp_state           = IBV_QPS_RTR;
  attr.path_mtu           = port_attr.active_mtu;
  attr.dest_qp_num        = remote.qpn;
  attr.rq_psn             = remote.psn;
  attr.max_dest_rd_atomic = 16;
  attr.min_rnr_timer      = 12;
  FillAhAttr(&attr.ah_attr, remote.lid, remote.gid, local_gid_index);
  const int flags =
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
      IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    throw std::runtime_error(IbvError("ibv_modify_qp RTR"));
  }
}

void ModifyQpToRts(ibv_qp* qp) {
  ibv_qp_attr attr{};
  attr.qp_state      = IBV_QPS_RTS;
  attr.sq_psn        = kRawVerbsPsn;
  attr.timeout       = 14;
  attr.retry_cnt     = 7;
  attr.rnr_retry     = 7;
  attr.max_rd_atomic = 16;
  const int flags =
      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
      IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    throw std::runtime_error(IbvError("ibv_modify_qp RTS"));
  }
}

} // namespace

struct RawVerbsTransport::Impl {
  explicit Impl(const RawVerbsConfig& c)
      : config(c), allocator(c.local_region_bytes, c.allocation_start_offset) {}

  RawVerbsConfig config;
  ibv_context* context = nullptr;
  int gid_index        = -1;
  ibv_pd* pd           = nullptr;
  ibv_cq* cq           = nullptr;
  ibv_mr* local_mr     = nullptr;
  std::vector<ibv_mr*> extra_mrs;
  void* local_base        = nullptr;
  bool owns_local_base    = false;
  std::size_t local_bytes = 0;
  RawVerbsRegionAllocator allocator;
  std::vector<RawVerbsNodeMeta> metas;
  std::vector<RawVerbsRemoteMemory> remotes;
  std::vector<ibv_qp*> qps;
  std::vector<std::uint32_t> max_inline_data_per_node;
  ibv_wc wc_batch[kRawVerbsPollBatchSize] = {};
  RawVerbsCompletionBatchCursor batch_cursor;
};

RawVerbsTransport::RawVerbsTransport(const RawVerbsConfig& config)
    : impl_(std::make_unique<Impl>(config)) {
  const auto opened = OpenDeviceForNuma(config.numa_id);
  impl_->context     = opened.context;
  impl_->gid_index   = opened.gid_index;
  impl_->pd      = ibv_alloc_pd(impl_->context);
  if (impl_->pd == nullptr) {
    throw std::runtime_error("ibv_alloc_pd failed");
  }
  impl_->cq =
      ibv_create_cq(impl_->context, kRawVerbsCqDepth, nullptr, nullptr, 0);
  if (impl_->cq == nullptr) {
    throw std::runtime_error("ibv_create_cq failed");
  }

  impl_->local_bytes = config.local_region_bytes;
  impl_->local_base  = reinterpret_cast<void*>(config.local_base_addr);
  if (impl_->local_base == nullptr) {
    const int rc = posix_memalign(&impl_->local_base, 4096, impl_->local_bytes);
    if (rc != 0) {
      throw std::runtime_error("posix_memalign failed for raw verbs region");
    }
    impl_->owns_local_base = true;
  }
  impl_->local_mr = ibv_reg_mr(
      impl_->pd,
      impl_->local_base,
      impl_->local_bytes,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  if (impl_->local_mr == nullptr) {
    throw std::runtime_error("ibv_reg_mr failed");
  }
  if (config.reserved_region_bytes != 0) {
    impl_->allocator.SetReservedRegion(
        {config.reserved_region_offset, config.reserved_region_bytes});
  }

  const int node_count = config.num_servers + config.num_clients;
  impl_->qps.resize(static_cast<std::size_t>(node_count), nullptr);
  impl_->max_inline_data_per_node.assign(
      static_cast<std::size_t>(node_count), 0);
  for (int node = 0; node < node_count; ++node) {
    if (!ShouldRawVerbsConnectToNode(config, node)) {
      continue;
    }
    ibv_qp_init_attr init_attr{};
    init_attr.send_cq             = impl_->cq;
    init_attr.recv_cq             = impl_->cq;
    init_attr.qp_type             = IBV_QPT_RC;
    init_attr.cap.max_send_wr     = 1024;
    init_attr.cap.max_recv_wr     = kRawVerbsRecvDepth;
    init_attr.cap.max_send_sge    = 32;
    init_attr.cap.max_recv_sge    = 1;
    init_attr.cap.max_inline_data = config.max_inline_data;
    ibv_qp* qp                    = ibv_create_qp(impl_->pd, &init_attr);
    if (qp == nullptr) {
      throw std::runtime_error(QpCreateError(config, node));
    }
    impl_->qps[static_cast<std::size_t>(node)] = qp;
    impl_->max_inline_data_per_node[static_cast<std::size_t>(node)] =
        init_attr.cap.max_inline_data;
  }
}

RawVerbsTransport::~RawVerbsTransport() {
  if (!impl_) {
    return;
  }
  for (ibv_qp* qp : impl_->qps) {
    if (qp != nullptr) {
      ibv_destroy_qp(qp);
    }
  }
  if (impl_->local_mr != nullptr) {
    ibv_dereg_mr(impl_->local_mr);
  }
  for (ibv_mr* mr : impl_->extra_mrs) {
    if (mr != nullptr) {
      ibv_dereg_mr(mr);
    }
  }
  if (impl_->cq != nullptr) {
    ibv_destroy_cq(impl_->cq);
  }
  if (impl_->pd != nullptr) {
    ibv_dealloc_pd(impl_->pd);
  }
  if (impl_->context != nullptr) {
    ibv_close_device(impl_->context);
  }
  if (impl_->owns_local_base && impl_->local_base != nullptr) {
    free(impl_->local_base);
  }
}

void RawVerbsTransport::RegisterThread() {}

namespace {
bool MrContains(ibv_mr* mr, const void* ptr, std::size_t bytes) {
  if (mr == nullptr) {
    return false;
  }
  const auto begin    = reinterpret_cast<std::uintptr_t>(ptr);
  const auto end      = begin + bytes;
  const auto mr_begin = reinterpret_cast<std::uintptr_t>(mr->addr);
  const auto mr_end   = mr_begin + static_cast<std::uintptr_t>(mr->length);
  return begin >= mr_begin && end >= begin && end <= mr_end;
}
} // namespace

ibv_mr*
RawVerbsTransport::FindLocalMr(const void* ptr, std::size_t bytes) const {
  if (impl_->local_mr != nullptr && MrContains(impl_->local_mr, ptr, bytes)) {
    return impl_->local_mr;
  }
  for (ibv_mr* mr : impl_->extra_mrs) {
    if (MrContains(mr, ptr, bytes)) {
      return mr;
    }
  }
  return nullptr;
}

void RawVerbsTransport::RegisterMemoryRegion(void* base, std::size_t bytes) {
  if (base == nullptr || bytes == 0) {
    return;
  }
  if (FindLocalMr(base, bytes) != nullptr) {
    return;
  }
  ibv_mr* mr = ibv_reg_mr(
      impl_->pd,
      base,
      bytes,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  if (mr == nullptr) {
    throw std::runtime_error("ibv_reg_mr failed for extra raw verbs region");
  }
  impl_->extra_mrs.push_back(mr);
}

void* RawVerbsTransport::AllocateRegistered(std::size_t bytes) {
  const std::uint64_t offset = impl_->allocator.Allocate(bytes);
  return static_cast<char*>(impl_->local_base) + offset;
}

std::uint64_t RawVerbsTransport::SaveAllocationState() const {
  return impl_->allocator.Checkpoint();
}

void RawVerbsTransport::RestoreAllocationState(std::uint64_t checkpoint) {
  impl_->allocator.Restore(checkpoint);
}

GlobalAddress RawVerbsTransport::LocalAddress(void* ptr) const {
  const auto base = reinterpret_cast<std::uintptr_t>(impl_->local_base);
  const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
  if (addr < base || addr >= base + impl_->local_bytes) {
    throw std::runtime_error("pointer outside raw verbs local region");
  }
  return GlobalAddress{
      static_cast<std::uint16_t>(impl_->config.global_id),
      static_cast<std::uint64_t>(addr - base),
  };
}

void* RawVerbsTransport::LocalPointer(GlobalAddress address) const {
  if (address.nodeID != impl_->config.global_id) {
    throw std::runtime_error(
        "raw verbs local pointer requested for remote node");
  }
  if (address.offset >= impl_->local_bytes) {
    throw std::runtime_error("raw verbs local offset outside region");
  }
  return static_cast<char*>(impl_->local_base) + address.offset;
}

RawVerbsNodeMeta RawVerbsTransport::LocalMeta() const {
  ibv_port_attr port_attr{};
  if (ibv_query_port(impl_->context, kRawVerbsPort, &port_attr) != 0) {
    throw std::runtime_error("ibv_query_port failed");
  }
  ibv_gid gid{};
  if (ibv_query_gid(impl_->context, kRawVerbsPort, impl_->gid_index, &gid) !=
      0) {
    throw std::runtime_error("ibv_query_gid failed");
  }
  RawVerbsNodeMeta meta{};
  meta.node_id   = static_cast<std::uint16_t>(impl_->config.global_id);
  meta.lid       = port_attr.lid;
  meta.psn       = kRawVerbsPsn;
  meta.rkey      = impl_->local_mr->rkey;
  meta.base_addr = reinterpret_cast<std::uint64_t>(impl_->local_base);
  std::memcpy(meta.gid, &gid, sizeof(meta.gid));
  return meta;
}

void RawVerbsTransport::Publish() {
  const int node_count = impl_->config.num_servers + impl_->config.num_clients;
  const RawVerbsNodeMeta local = LocalMeta();
  RdmaControlPlaneClient control_plane({
      impl_->config.control_plane_host,
      impl_->config.control_plane_port,
      impl_->config.control_plane_timeout_ms,
  });
  for (int node = 0; node < node_count; ++node) {
    if (!ShouldRawVerbsConnectToNode(impl_->config, node)) {
      continue;
    }
    RawVerbsNodeMeta peer_local = local;
    peer_local.qpn = impl_->qps[static_cast<std::size_t>(node)]->qp_num;
    control_plane.PublishMeta(
        impl_->config.global_id,
        impl_->config.local_lane,
        node,
        impl_->config.remote_lane,
        peer_local);
  }
}

void RawVerbsTransport::Connect() {
  const int node_count = impl_->config.num_servers + impl_->config.num_clients;
  const RawVerbsNodeMeta local = LocalMeta();
  RdmaControlPlaneClient control_plane({
      impl_->config.control_plane_host,
      impl_->config.control_plane_port,
      impl_->config.control_plane_timeout_ms,
  });
  impl_->metas.assign(static_cast<std::size_t>(node_count), RawVerbsNodeMeta{});
  impl_->remotes.assign(
      static_cast<std::size_t>(node_count), RawVerbsRemoteMemory{});

  for (int node = 0; node < node_count; ++node) {
    if (node == impl_->config.global_id) {
      impl_->metas[static_cast<std::size_t>(node)]   = local;
      impl_->remotes[static_cast<std::size_t>(node)] = RawVerbsRemoteMemory{
          local.node_id,
          local.base_addr,
          local.rkey,
      };
      continue;
    }
    if (!ShouldRawVerbsConnectToNode(impl_->config, node)) {
      continue;
    }
    const RawVerbsNodeMeta meta = control_plane.GetMeta(
        node,
        impl_->config.remote_lane,
        impl_->config.global_id,
        impl_->config.local_lane,
        impl_->config.control_plane_timeout_ms);
    impl_->metas[static_cast<std::size_t>(node)]   = meta;
    impl_->remotes[static_cast<std::size_t>(node)] = RawVerbsRemoteMemory{
        meta.node_id,
        meta.base_addr,
        meta.rkey,
    };
  }

  for (int node = 0; node < node_count; ++node) {
    if (!ShouldRawVerbsConnectToNode(impl_->config, node)) {
      continue;
    }
    ibv_qp* qp = impl_->qps[static_cast<std::size_t>(node)];
    ModifyQpToInit(qp);
    ModifyQpToRtr(
        qp, impl_->metas[static_cast<std::size_t>(node)], impl_->gid_index);
    ModifyQpToRts(qp);
    for (int i = 0; i < kRawVerbsRecvDepth; ++i) {
      ibv_recv_wr wr{};
      ibv_recv_wr* bad_wr = nullptr;
      wr.wr_id            = static_cast<std::uint64_t>(node);
      wr.sg_list          = nullptr;
      wr.num_sge          = 0;
      if (ibv_post_recv(qp, &wr, &bad_wr) != 0) {
        throw std::runtime_error("ibv_post_recv failed");
      }
    }
  }
}

void RawVerbsTransport::PublishAndConnect() {
  Publish();
  Connect();
}

void RawVerbsTransport::Write(
    const void* local,
    GlobalAddress remote,
    std::size_t bytes,
    std::uint64_t wr_id,
    bool signaled) {
  if (remote.nodeID >= impl_->remotes.size()) {
    throw std::runtime_error("raw verbs write remote node out of range");
  }
  ibv_sge sge{};
  sge.addr         = reinterpret_cast<std::uint64_t>(local);
  sge.length       = static_cast<std::uint32_t>(bytes);
  ibv_mr* local_mr = FindLocalMr(local, bytes);
  if (local_mr == nullptr) {
    throw std::runtime_error("raw verbs write local buffer is not registered");
  }
  sge.lkey = local_mr->lkey;
  ibv_send_wr wr{};
  wr.wr_id      = wr_id;
  wr.opcode     = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  if (bytes > 0 &&
      bytes <= impl_->max_inline_data_per_node[static_cast<std::size_t>(
                   remote.nodeID)]) {
    wr.send_flags |= IBV_SEND_INLINE;
  }
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.rdma.remote_addr =
      impl_->remotes[remote.nodeID].base_addr + remote.offset;
  wr.wr.rdma.rkey     = impl_->remotes[remote.nodeID].rkey;
  ibv_send_wr* bad_wr = nullptr;
  if (ibv_post_send(impl_->qps[remote.nodeID], &wr, &bad_wr) != 0) {
    throw std::runtime_error("ibv_post_send write failed");
  }
}

void RawVerbsTransport::WriteSg(
    base::ConstArray<RawVerbsSge> sges,
    GlobalAddress remote,
    std::uint64_t wr_id,
    bool signaled) {
  if (remote.nodeID >= impl_->remotes.size()) {
    throw std::runtime_error("raw verbs write-sg remote node out of range");
  }
  if (sges.Size() == 0) {
    return;
  }
  std::vector<ibv_sge> verbs_sges;
  verbs_sges.reserve(static_cast<std::size_t>(sges.Size()));
  for (const auto& entry : sges) {
    if (entry.bytes == 0) {
      continue;
    }
    if (entry.bytes > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("raw verbs write-sg entry too large");
    }
    ibv_mr* local_mr = FindLocalMr(entry.data, entry.bytes);
    if (local_mr == nullptr) {
      throw std::runtime_error(
          "raw verbs write-sg local buffer is not registered");
    }
    ibv_sge sge{};
    sge.addr   = reinterpret_cast<std::uint64_t>(entry.data);
    sge.length = static_cast<std::uint32_t>(entry.bytes);
    sge.lkey   = local_mr->lkey;
    verbs_sges.push_back(sge);
  }
  if (verbs_sges.empty()) {
    return;
  }
  ibv_send_wr wr{};
  wr.wr_id      = wr_id;
  wr.opcode     = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.sg_list    = verbs_sges.data();
  wr.num_sge    = static_cast<int>(verbs_sges.size());
  wr.wr.rdma.remote_addr =
      impl_->remotes[remote.nodeID].base_addr + remote.offset;
  wr.wr.rdma.rkey     = impl_->remotes[remote.nodeID].rkey;
  ibv_send_wr* bad_wr = nullptr;
  if (ibv_post_send(impl_->qps[remote.nodeID], &wr, &bad_wr) != 0) {
    throw std::runtime_error("ibv_post_send write-sg failed");
  }
}

void RawVerbsTransport::WriteWithImm(
    const void* local,
    GlobalAddress remote,
    std::size_t bytes,
    std::uint32_t imm_data,
    std::uint64_t wr_id,
    bool signaled) {
  if (remote.nodeID >= impl_->remotes.size()) {
    throw std::runtime_error(
        "raw verbs write-with-imm remote node out of range");
  }
  ibv_sge sge{};
  sge.addr   = reinterpret_cast<std::uint64_t>(local);
  sge.length = static_cast<std::uint32_t>(bytes);
  sge.lkey   = impl_->local_mr->lkey;
  ibv_send_wr wr{};
  wr.wr_id      = wr_id;
  wr.opcode     = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.imm_data   = htonl(imm_data);
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  if (bytes > 0 &&
      bytes <= impl_->max_inline_data_per_node[static_cast<std::size_t>(
                   remote.nodeID)]) {
    wr.send_flags |= IBV_SEND_INLINE;
  }
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.rdma.remote_addr =
      impl_->remotes[remote.nodeID].base_addr + remote.offset;
  wr.wr.rdma.rkey     = impl_->remotes[remote.nodeID].rkey;
  ibv_send_wr* bad_wr = nullptr;
  if (ibv_post_send(impl_->qps[remote.nodeID], &wr, &bad_wr) != 0) {
    throw std::runtime_error("ibv_post_send write-with-imm failed");
  }
}

void RawVerbsTransport::Read(
    void* local,
    GlobalAddress remote,
    std::size_t bytes,
    std::uint64_t wr_id,
    bool signaled) {
  if (remote.nodeID >= impl_->remotes.size()) {
    throw std::runtime_error("raw verbs read remote node out of range");
  }
  ibv_sge sge{};
  sge.addr   = reinterpret_cast<std::uint64_t>(local);
  sge.length = static_cast<std::uint32_t>(bytes);
  sge.lkey   = impl_->local_mr->lkey;
  ibv_send_wr wr{};
  wr.wr_id      = wr_id;
  wr.opcode     = IBV_WR_RDMA_READ;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.sg_list    = &sge;
  wr.num_sge    = 1;
  wr.wr.rdma.remote_addr =
      impl_->remotes[remote.nodeID].base_addr + remote.offset;
  wr.wr.rdma.rkey     = impl_->remotes[remote.nodeID].rkey;
  ibv_send_wr* bad_wr = nullptr;
  if (ibv_post_send(impl_->qps[remote.nodeID], &wr, &bad_wr) != 0) {
    throw std::runtime_error("ibv_post_send read failed");
  }
}

void RawVerbsTransport::SendDoorbell(
    std::uint16_t node_id, std::uint32_t imm_data, std::uint64_t wr_id) {
  if (node_id >= impl_->qps.size()) {
    throw std::runtime_error("raw verbs doorbell remote node out of range");
  }
  ibv_send_wr wr{};
  wr.opcode           = IBV_WR_SEND_WITH_IMM;
  wr.imm_data         = htonl(imm_data);
  wr.send_flags       = IBV_SEND_SIGNALED;
  wr.wr_id            = wr_id;
  wr.sg_list          = nullptr;
  wr.num_sge          = 0;
  ibv_send_wr* bad_wr = nullptr;
  if (ibv_post_send(impl_->qps[node_id], &wr, &bad_wr) != 0) {
    throw std::runtime_error("ibv_post_send doorbell failed");
  }
}

bool RawVerbsTransport::Poll(RawVerbsCompletion* completion, int timeout_ms) {
  const auto deadline =
      timeout_ms > 0
          ? std::chrono::steady_clock::now() +
                std::chrono::milliseconds(timeout_ms)
          : std::chrono::steady_clock::time_point::max();
  while (true) {
    if (!impl_->batch_cursor.HasCachedCompletion()) {
      const int n =
          ibv_poll_cq(impl_->cq, kRawVerbsPollBatchSize, impl_->wc_batch);
      if (n < 0) {
        throw std::runtime_error("ibv_poll_cq failed");
      }
      if (n == 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
          return false;
        }
        std::this_thread::yield();
        continue;
      }
      impl_->batch_cursor.Reset(impl_->wc_batch, n);
    }
    ibv_wc& wc = *impl_->batch_cursor.TakeCachedCompletion();
    if (wc.status != IBV_WC_SUCCESS) {
      throw std::runtime_error(
          std::string("raw verbs CQ error: ") + ibv_wc_status_str(wc.status));
    }
    if (completion != nullptr) {
      completion->wr_id    = wc.wr_id;
      completion->opcode   = wc.opcode;
      completion->has_imm  = (wc.wc_flags & IBV_WC_WITH_IMM) != 0;
      completion->imm_data = completion->has_imm ? ntohl(wc.imm_data) : 0;
    }
    if (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      const std::uint16_t node_id = static_cast<std::uint16_t>(wc.wr_id);
      ibv_recv_wr wr{};
      ibv_recv_wr* bad_wr = nullptr;
      wr.wr_id            = node_id;
      wr.sg_list          = nullptr;
      wr.num_sge          = 0;
      if (node_id < impl_->qps.size() &&
          ibv_post_recv(impl_->qps[node_id], &wr, &bad_wr) != 0) {
        throw std::runtime_error("ibv_post_recv repost failed");
      }
    }
    return true;
  }
}

std::uint32_t RawVerbsTransport::max_inline_data(std::uint16_t node_id) const {
  if (node_id >= impl_->max_inline_data_per_node.size()) {
    throw std::runtime_error("raw verbs inline query remote node out of range");
  }
  return impl_->max_inline_data_per_node[static_cast<std::size_t>(node_id)];
}

} // namespace petps
