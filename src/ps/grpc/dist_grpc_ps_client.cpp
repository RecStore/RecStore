#include "dist_grpc_ps_client.h"

#include "base/factory.h"

namespace recstore {

FACTORY_REGISTER(
    BasePSClient, distributed_grpc, DistributedGRPCParameterClient, json);

} // namespace recstore
