#include "dist_brpc_ps_client.h"

#include "base/factory.h"

namespace recstore {

FACTORY_REGISTER(
    BasePSClient, distributed_brpc, DistributedBRPCParameterClient, json);

} // namespace recstore
