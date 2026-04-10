#include "petps_client.h"

#include <atomic>
#include <cstring>
#include <vector>

#include "petps_magic.h"
#include "ps/base/Postoffice.h"
#include "ps/base/shard_manager.h"

DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);
DEFINE_int32(numa_id, 0, "");

namespace petps {

void PetPSClient::Init() {
  constexpr uint64_t dsm_size   = 1 * 1000 * define::MB;
  static uint64_t dsm_base_addr = (uint64_t)hugePageAlloc(dsm_size);

  ClusterInfo cluster;
  cluster.serverNR = XPostoffice::GetInstance()->NumServers();
  cluster.clientNR = XPostoffice::GetInstance()->NumClients();
  DSMConfig config(CacheConfig(), cluster, 0, true);
  config.NIC_name = '0' + FLAGS_numa_id;
  config.dsmSize  = dsm_size;
  config.baseAddr = (uint64_t)dsm_base_addr;
  dsm_ = DSM::getInstance(config, XPostoffice::GetInstance()->GlobalID());

  CHECK_EQ(dsm_->getMyNodeID(), XPostoffice::GetInstance()->GlobalID())
      << "inconsistent postoffice and wq dsm";

  extern int global_socket_id;
  ::global_socket_id = FLAGS_numa_id;
  LOG(INFO) << "set NUMA ID = " << FLAGS_numa_id;
}

int PetPSClient::GetParameter(base::ConstArray<uint64_t> keys,
                              std::vector<std::vector<float>>* values) {
  values->clear();
  if (keys.Size() == 0) {
    return 0;
  }

  const int embedding_dim = FLAGS_value_size / sizeof(float);
  std::vector<float> flat(keys.Size() * embedding_dim + 1, 0.0f);
  int rpc_id = GetParameter(keys, flat.data(), false, 0);
  WaitRPCFinish(rpc_id);

  values->reserve(keys.Size());
  for (int i = 0; i < keys.Size(); ++i) {
    std::vector<float> row(embedding_dim);
    std::memcpy(row.data(), flat.data() + i * embedding_dim, FLAGS_value_size);
    values->push_back(std::move(row));
  }
  RevokeRPCResource(rpc_id);
  return 0;
}

std::vector<int> PetPSClient::GetServerThreadIDs() {
  auto m     = RawMessage::get_new_msg();
  m->node_id = dsm_->getMyNodeID();
  m->t_id    = dsm_->getMyThreadID();
  m->type    = GET_SERVER_THREADIDS;
  // RPC to the server ID <shard_>
  dsm_->rpc_call(m, shard_, 0);
  uint64_t wr_id;
  while (1) {
    auto recv = dsm_->rpc_fast_wait(&wr_id);
    // FB_LOG_EVERY_MS(WARNING, 5000)
    //     << "client wait the result of GetServerThreadIDs";
    if (recv) {
      CHECK_EQ(recv->type, RESP_GET_SERVER_THREADIDS);
      Cursor cursor;
      Slice extra_data = recv->get_string(cursor);
      base::ConstArray<int> cores(
          (int*)extra_data.s, extra_data.len / sizeof(int));
      LOG(INFO) << folly::sformat(
          "client{} {}th thread are routed to PS{} ({})th thread",
          m->node_id,
          (int)m->t_id,
          shard_,
          cores.Debug());
      return cores.ToVector();
    }
  }
  LOG(FATAL) << "not reach here";
  return std::vector<int>();
}

int PetPSClient::SelectServerThreadID() const {
  static int round_robin = 0;
  int ret                = serverThreadIdsRoutedTo_[round_robin];
  round_robin            = (round_robin + 1) % serverThreadIdsRoutedTo_.size();
  return ret;
}

std::size_t PetPSClient::ResponseBufferBytes(std::size_t key_count) const {
  return FixedSlotResponseBytes(key_count, FLAGS_value_size);
}

void* PetPSClient::GetReceiveBuffer(size_t size) {
  static std::atomic<uint64_t> client_memory_offset_acc{0};
  GlobalAddress gaddr;
  gaddr.nodeID = dsm_->getMyNodeID();
  gaddr.offset = client_memory_offset_acc.fetch_add(size);
  return dsm_->addr(gaddr);
}

// this interface assume all keys with the same embedding dimension
int PetPSClient::GetParameter(base::ConstArray<uint64_t> keys,
                              float* values,
                              bool isAsync,
                              int async_req_id) {
  thread_local auto m = RawMessage::get_new_msg();
  GlobalAddress gaddr = dsm_->gaddr(values);
  m->type             = RpcType::GET;
  m->receive_gaddr    = gaddr;

  std::int32_t* poll = reinterpret_cast<std::int32_t*>(
      reinterpret_cast<char*>(values) + keys.Size() * FLAGS_value_size);
  rpcId2PollMap_[rpcIDAcc_] = poll;
  *poll                     = static_cast<std::int32_t>(RpcStatus::kPending);

#ifdef RPC_DEBUG
  for (auto each : keys) {
    CHECK_EQ(shard_, ShardManager::KeyPartition(each))
        << each << " not belong to this PS";
  }
#endif

  dsm_->rpc_call(m,
                 shard_,
                 SelectServerThreadID(),
                 Slice(keys.binary_data(), keys.binary_size()));

  if (!isAsync) {
    WaitRPCFinish(rpcIDAcc_);
  }
  rpcIDAcc_++;
  return rpcIDAcc_ - 1;
}

bool PetPSClient::QueryRPCFinished(int rpc_id) {
  auto* poll = rpcId2PollMap_[rpc_id];
  return *poll != static_cast<std::int32_t>(RpcStatus::kPending);
}

void PetPSClient::WaitRPCFinish(int rpc_id) {
  auto* poll = rpcId2PollMap_[rpc_id];
  while (*poll == static_cast<std::int32_t>(RpcStatus::kPending)) {
#ifdef RPC_DEBUG
    FB_LOG_EVERY_MS(INFO, 1000) << "poll = " << *poll << "\n";
    // << "values[0]=" << values[0] << "\n"
    // << "values[1]=" << values[1] << "\n"
    // << "values[2]=" << values[2] << "\n"
    // << "values[3]=" << values[3] << "\n"
    // << "values[4]=" << values[4] << "\n"
    // << "values[5]=" << values[5] << "\n";
#endif
  }
  return;
}

void PetPSClient::RevokeRPCResource(int rpc_id) {
  rpcId2PollMap_.erase(rpc_id);
};

int PetPSClient::PutParameter(const std::vector<uint64_t>& keys,
                              const std::vector<std::vector<float>>& values) {
  CHECK_EQ(keys.size(), values.size());
  if (keys.empty()) {
    return 0;
  }

  const std::size_t embedding_dim = FLAGS_value_size / sizeof(float);
  for (const auto& row : values) {
    if (row.size() != embedding_dim) {
      LOG(ERROR) << "PutParameter value size mismatch, expected dim="
                 << embedding_dim << " got=" << row.size();
      return -1;
    }
  }

  std::string payload = EncodePutPayload(keys, values);
  auto* ack =
      reinterpret_cast<std::int32_t*>(GetReceiveBuffer(sizeof(std::int32_t)));
  *ack = static_cast<std::int32_t>(RpcStatus::kPending);

  thread_local auto m = RawMessage::get_new_msg();
  m->type             = RpcType::PUT;
  m->receive_gaddr    = dsm_->gaddr(ack);

  rpcId2PollMap_[rpcIDAcc_] = ack;
  dsm_->rpc_call(
      m, shard_, SelectServerThreadID(), Slice(payload.data(), payload.size()));

  WaitRPCFinish(rpcIDAcc_);
  const std::int32_t status = *ack;
  rpcId2PollMap_.erase(rpcIDAcc_);
  rpcIDAcc_++;
  return status == static_cast<std::int32_t>(RpcStatus::kOk) ? 0 : -1;
}

int PetPSClient::FakePutParameter(base::ConstArray<uint64_t> keys,
                                  float* values) {
  thread_local auto m = RawMessage::get_new_msg();
  GlobalAddress gaddr = dsm_->gaddr(values);
  m->type             = RpcType::PUT;
  m->receive_gaddr    = gaddr;

  // LOG(INFO) << "send PS Put";
  std::int32_t* poll        = reinterpret_cast<std::int32_t*>(values);
  rpcId2PollMap_[rpcIDAcc_] = poll;
  *poll                     = static_cast<std::int32_t>(RpcStatus::kPending);

#ifdef RPC_DEBUG
  for (auto each : keys) {
    CHECK_EQ(shard_, ShardManager::KeyPartition(each))
        << each << " not belong to this PS";
  }
#endif

  dsm_->rpc_call(m,
                 shard_,
                 SelectServerThreadID(),
                 Slice(keys.binary_data(), keys.binary_size()));

  // if (!isAsync) {
  //   WaitRPCFinish(rpcIDAcc_);
  // }
  rpcIDAcc_++;
  return rpcIDAcc_ - 1;
}

} // namespace petps
