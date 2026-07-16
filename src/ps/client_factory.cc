#include "ps/client_factory.h"

#include <stdexcept>

#include "base/factory.h"
#include "ps/brpc/brpc_ps_client.h"
#include "ps/brpc/dist_brpc_ps_client.h"
#include "ps/grpc/grpc_ps_client.h"
#include "ps/grpc/dist_grpc_ps_client.h"
#include "ps/local_shm/local_shm_client.h"
#include "ps/rdma/rdma_ps_client_adapter.h"
#include "framework/common/ps_client_config_adapter.h"

namespace recstore {

namespace {

const char* TypeKeyForFactory(PSClientType type) {
  switch (type) {
  case PSClientType::kGrpc:
    return "grpc";
  case PSClientType::kBrpc:
    return "brpc";
  case PSClientType::kLocalShm:
    return "local_shm";
  case PSClientType::kRdma:
    return "rdma";
  }

  throw std::invalid_argument("Unknown PSClientType");
}

} // namespace

std::unique_ptr<BasePSClient>
CreatePSClient(const PSClientCreateOptions& options) {
  if (options.type == PSClientType::kRdma) {
    return std::make_unique<RDMAPSClientAdapter>(options.raw_config);
  }

  if (HasFrameworkDistributedClientConfig(options.raw_config)) {
    if (options.type == PSClientType::kGrpc) {
      return std::make_unique<DistributedGRPCParameterClient>(
          options.raw_config);
    }
    if (options.type == PSClientType::kBrpc) {
      return std::make_unique<DistributedBRPCParameterClient>(
          options.raw_config);
    }
  }

  BasePSClient* client = base::Factory<BasePSClient, json>::NewInstance(
      TypeKeyForFactory(options.type), options.transport_config);
  if (client != nullptr) {
    return std::unique_ptr<BasePSClient>(client);
  }

  if (options.type == PSClientType::kGrpc) {
    return std::make_unique<GRPCParameterClient>(options.transport_config);
  }

  if (options.type == PSClientType::kBrpc) {
    return std::make_unique<BRPCParameterClient>(options.transport_config);
  }

  if (options.type == PSClientType::kLocalShm) {
    return std::make_unique<LocalShmPSClient>(options.transport_config);
  }

  throw std::runtime_error("Failed to create PS client");
}

} // namespace recstore
