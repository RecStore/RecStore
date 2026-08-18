#include "ps/client_factory.h"

#include <stdexcept>

#include "ps/brpc/dist_brpc_ps_client.h"
#include "ps/grpc/dist_grpc_ps_client.h"
#include "ps/local_shm/local_shm_client.h"
#include "ps/rdma/rdma_ps_client_adapter.h"

namespace recstore {

std::unique_ptr<BasePSClient>
CreatePSClient(const PSClientCreateOptions& options) {
  switch (options.type) {
  case PSClientType::kRdma:
    return std::make_unique<RDMAPSClientAdapter>(options.raw_config);
  case PSClientType::kGrpc:
    return std::make_unique<DistributedGRPCParameterClient>(options.raw_config);
  case PSClientType::kBrpc:
    return std::make_unique<DistributedBRPCParameterClient>(options.raw_config);
  case PSClientType::kLocalShm:
    return std::make_unique<LocalShmPSClient>(options.transport_config);
  }

  throw std::runtime_error("Failed to create PS client");
}

} // namespace recstore
