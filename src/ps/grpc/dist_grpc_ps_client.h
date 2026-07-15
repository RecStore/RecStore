#pragma once

#include "base/json.h"
#include "ps/base/dist_sharded_client.h"
#include "grpc_ps_client.h"

using json = nlohmann::json;

namespace recstore {

/**
 * @brief Distributed gRPC parameter-server client
 *
 * Many-to-many connections; routes keys to shards via a hash function.
 * Server list and hash method come from JSON config. All logic lives in
 * DistributedShardedClient<GRPCParameterClient>.
 */
class DistributedGRPCParameterClient
    : public DistributedShardedClient<GRPCParameterClient> {
public:
  explicit DistributedGRPCParameterClient(json config)
      : DistributedShardedClient<GRPCParameterClient>(std::move(config),
                                                      "GRPC") {}
};

} // namespace recstore
