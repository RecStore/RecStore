#pragma once

#include "base/json.h"
#include "ps/base/dist_sharded_client.h"
#include "brpc_ps_client.h"

using json = nlohmann::json;

namespace recstore {

/**
 * @brief Distributed bRPC parameter-server client
 *
 * Many-to-many connections; routes keys to shards via a hash function.
 * Server list and hash method come from JSON config. All logic lives in
 * DistributedShardedClient<BRPCParameterClient>.
 */
class DistributedBRPCParameterClient
    : public DistributedShardedClient<BRPCParameterClient> {
public:
  explicit DistributedBRPCParameterClient(json config)
      : DistributedShardedClient<BRPCParameterClient>(std::move(config),
                                                      "BRPC") {}
};

} // namespace recstore
