#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "framework/common/ps_client_config_adapter.h"
#include "ps/client_factory.h"
#include "ps/rdma/rdma_deployment.h"
#include "ps/rdma/rdma_ps_client_adapter.h"

DECLARE_string(rdma_get_response_mode);

namespace recstore {
namespace {

class ScopedEnvVar {
public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    const char* existing = std::getenv(name_);
    if (existing != nullptr) {
      previous_ = existing;
    }
    if (::setenv(name_, value, 1) != 0) {
      throw std::runtime_error(std::string("setenv failed for ") + name_);
    }
  }

  ~ScopedEnvVar() {
    if (previous_.has_value()) {
      ::setenv(name_, previous_->c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }

private:
  const char* name_;
  std::optional<std::string> previous_;
};

} // namespace

TEST(RDMAPSClientAdapterTest, ResolveEmbeddedIdentityFromTorchEnv) {
  ScopedEnvVar rank("RANK", "1");
  ScopedEnvVar world_size("WORLD_SIZE", "2");

  const auto identity = ResolveEmbeddedRdmaClientIdentity(1, 2);
  EXPECT_EQ(identity.client_index, 1);
  EXPECT_EQ(identity.num_client_processes, 2);
  EXPECT_EQ(identity.global_id, 2);
}

TEST(RDMAPSClientAdapterTest, ResolveEmbeddedIdentityPrefersExplicitOverride) {
  ScopedEnvVar client_index("RECSTORE_RDMA_OS_CLIENT_INDEX", "0");
  ScopedEnvVar num_clients("RECSTORE_RDMA_NUM_CLIENT_PROCESSES", "3");
  ScopedEnvVar rank("RANK", "2");
  ScopedEnvVar world_size("WORLD_SIZE", "2");

  const auto identity = ResolveEmbeddedRdmaClientIdentity(1, 3);
  EXPECT_EQ(identity.client_index, 0);
  EXPECT_EQ(identity.num_client_processes, 3);
  EXPECT_EQ(identity.global_id, 1);
}

TEST(RDMAPSClientAdapterTest, ResolveEmbeddedIdentityRejectsOutOfRangeIndex) {
  ScopedEnvVar rank("RANK", "2");
  ScopedEnvVar world_size("WORLD_SIZE", "2");

  EXPECT_THROW(ResolveEmbeddedRdmaClientIdentity(1, 2), std::runtime_error);
}

TEST(RDMAPSClientAdapterTest,
     ResolveEmbeddedIdentityRejectsDeploymentClientCountMismatch) {
  ScopedEnvVar rank("RANK", "0");
  ScopedEnvVar world_size("WORLD_SIZE", "2");

  EXPECT_THROW(ResolveEmbeddedRdmaClientIdentity(1, 1), std::runtime_error);
}

TEST(RDMAPSClientAdapterTest, RuntimeReadsGetResponseModeFromEnv) {
  ScopedEnvVar response_mode("RECSTORE_RDMA_GET_RESPONSE_MODE", "staging_copy");

  InitializeRdmaProcessRuntime();

  EXPECT_EQ(FLAGS_rdma_get_response_mode, "staging_copy");
}

TEST(RDMAPSClientAdapterTest, SingleShardClientHandlesUnknownRpcHandlesSafely) {
  petps::PetPSClient client("127.0.0.1", 25000, 0, 0);

  EXPECT_TRUE(client.QueryRPCFinished(7));
  EXPECT_NO_THROW(client.WaitRPCFinish(7));
  EXPECT_NO_THROW(client.RevokeRPCResource(7));
  EXPECT_EQ(client.WaitUpdateParameter(7), -1);
}

TEST(RDMAPSClientAdapterTest, FactoryCreatesRdmaClientAndSupportsTableInit) {
  json config = {
      {"cache_ps",
       {{"ps_type", "RDMA"},
        {
            "base_kv_config",
            {{"value", {{"default_value_size_hint", 16}}}},
        },
        {"num_threads", 1}}},
      {"client", {{"host", "127.0.0.1"}, {"port", 25000}, {"shard", 0}}},
      {"distributed_client",
       {{"num_shards", 1},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 64},
        {"servers",
         json::array(
             {{{"host", "127.0.0.1"}, {"port", 25000}, {"shard", 0}}})}}},
      {"rdma_deployment",
       {{"deployment_id", "adapter-factory-test"},
        {"epoch", 1},
        {"protocol_version", 1},
        {"num_clients", 1},
        {"nodes",
         json::array({json{{"node_id", 0},
                           {"role", "server"},
                           {"device", "mlx5_0"},
                           {"port", 1},
                           {"gid_index", 0},
                           {"mode", "ib"}},
                      json{{"node_id", 1},
                           {"role", "client"},
                           {"device", "mlx5_0"},
                           {"port", 1},
                           {"gid_index", 0},
                           {"mode", "ib"}}})}}},
  };

  auto client =
      CreatePSClient(ResolvePSClientOptionsFromFrameworkConfig(config));
  ASSERT_NE(client, nullptr);
  auto* adapter = dynamic_cast<RDMAPSClientAdapter*>(client.get());
  ASSERT_NE(adapter, nullptr);

  base::ConstArray<uint64_t> empty_keys;
  base::RecTensor empty_grads({0, 4}, base::DataType::FLOAT32);
  EXPECT_THROW(
      adapter->SubmitUpdateParameterAsync("table", empty_keys, empty_grads),
      std::invalid_argument);
}

json DeploymentConfig(std::string hash_method = "city_hash") {
  return json{
      {"num_shards", 2},
      {"hash_method", std::move(hash_method)},
      {"max_keys_per_request", 4},
      {"servers",
       json::array({json{{"host", "host-1"}, {"port", 25001}, {"shard", 1}},
                    json{{"host", "host-0"}, {"port", 25000}, {"shard", 0}}})}};
}

json FabricDeploymentConfig() {
  auto config                                   = json::object();
  config["distributed_client"]                  = DeploymentConfig();
  config["rdma_deployment"]["deployment_id"]    = "unit-test";
  config["rdma_deployment"]["epoch"]            = 1;
  config["rdma_deployment"]["protocol_version"] = 1;
  config["rdma_deployment"]["num_clients"]      = 1;
  config["rdma_deployment"]["nodes"]            = json::array(
      {json{{"node_id", 0},
            {"role", "server"},
            {"device", "mlx5_0"},
            {"port", 1},
            {"gid_index", 0},
            {"mode", "ib"}},
                  json{{"node_id", 1},
            {"role", "server"},
            {"device", "mlx5_0"},
            {"port", 1},
            {"gid_index", 0},
            {"mode", "ib"}},
                  json{{"node_id", 2},
            {"role", "client"},
            {"device", "mlx5_0"},
            {"port", 1},
            {"gid_index", 0},
            {"mode", "ib"}}});
  return config;
}

TEST(RDMAPSClientAdapterTest, EmptyWritesDoNotInitializeRdmaTransport) {
  RDMAPSClientAdapter adapter(FabricDeploymentConfig());
  const base::ConstArray<uint64_t> empty_keys(nullptr, 0);
  base::RecTensor empty_values({0, 4}, base::DataType::FLOAT32);

  EXPECT_EQ(adapter.PutParameter(empty_keys, empty_values), 0);
  EXPECT_EQ(adapter.UpdateParameter("table", empty_keys, empty_values), 0);
}

TEST(ResolvedRdmaDeploymentTest, RejectsNonPositiveShardAndRequestLimits) {
  for (const char* field : {"num_shards", "max_keys_per_request"}) {
    for (const int value : {0, -1}) {
      auto config   = DeploymentConfig();
      config[field] = value;
      SCOPED_TRACE(field);
      SCOPED_TRACE(value);
      EXPECT_THROW(ParseResolvedRdmaDeployment(config), std::invalid_argument);
    }
  }
}

TEST(ResolvedRdmaDeploymentTest, RejectsMalformedServerEntries) {
  auto config          = DeploymentConfig();
  config["servers"][0] = 1;
  EXPECT_THROW(ParseResolvedRdmaDeployment(config), std::invalid_argument);

  config                       = DeploymentConfig();
  config["servers"][0]["host"] = "";
  EXPECT_THROW(ParseResolvedRdmaDeployment(config), std::invalid_argument);

  for (const int port : {0, 65536}) {
    config                       = DeploymentConfig();
    config["servers"][0]["port"] = port;
    SCOPED_TRACE(port);
    EXPECT_THROW(ParseResolvedRdmaDeployment(config), std::invalid_argument);
  }
}

TEST(ResolvedRdmaDeploymentTest, RejectsServerCountMismatch) {
  auto config = DeploymentConfig();
  config["servers"].erase(1);
  EXPECT_THROW(ParseResolvedRdmaDeployment(config), std::invalid_argument);

  config = DeploymentConfig();
  config["servers"].push_back(config["servers"][0]);
  EXPECT_THROW(ParseResolvedRdmaDeployment(config), std::invalid_argument);
}

TEST(ResolvedRdmaDeploymentTest, PreservesInputOrderAndExplicitRouting) {
  const auto deployment = ParseResolvedRdmaDeployment(DeploymentConfig());
  ASSERT_EQ(deployment.endpoints.size(), 2u);
  EXPECT_EQ(deployment.endpoints[0].shard, 1);
  EXPECT_EQ(deployment.endpoints[1].shard, 0);
  EXPECT_EQ(deployment.shard_to_endpoint_index.at(0), 1);
  EXPECT_EQ(deployment.shard_to_endpoint_index.at(1), 0);
}

TEST(ResolvedRdmaDeploymentTest, RejectsDuplicateShard) {
  auto config                   = DeploymentConfig();
  config["servers"][1]["shard"] = 1;
  EXPECT_THROW(ParseResolvedRdmaDeployment(config), std::invalid_argument);
}

TEST(ResolvedRdmaDeploymentTest, RejectsMissingOrOutOfRangeShard) {
  auto config                   = DeploymentConfig();
  config["servers"][1]["shard"] = 2;
  EXPECT_THROW(ParseResolvedRdmaDeployment(config), std::invalid_argument);
}

TEST(ResolvedRdmaDeploymentTest, RejectsMissingShard) {
  auto config = DeploymentConfig();
  config["servers"][1].erase("shard");
  EXPECT_THROW(ParseResolvedRdmaDeployment(config), std::invalid_argument);
}

TEST(ResolvedRdmaDeploymentTest, AllowsSharedEndpointForDifferentShards) {
  auto config                  = DeploymentConfig();
  config["servers"][1]["host"] = "host-1";
  config["servers"][1]["port"] = 25001;
  const auto deployment        = ParseResolvedRdmaDeployment(config);
  EXPECT_EQ(deployment.shard_to_endpoint_index.at(0), 1);
  EXPECT_EQ(deployment.shard_to_endpoint_index.at(1), 0);
}

TEST(ResolvedRdmaDeploymentTest, RejectsUnknownHashMethod) {
  EXPECT_THROW(ParseResolvedRdmaDeployment(DeploymentConfig("murmur")),
               std::invalid_argument);
}

TEST(ResolvedRdmaDeploymentTest, RejectsMissingRequiredFields) {
  for (const char* field :
       {"num_shards", "hash_method", "max_keys_per_request", "servers"}) {
    auto config = DeploymentConfig();
    config.erase(field);
    SCOPED_TRACE(field);
    EXPECT_THROW(ParseResolvedRdmaDeployment(config), std::invalid_argument);
  }
}

TEST(ResolvedRdmaDeploymentTest, SingleShardUsesDistributedEndpoint) {
  const json config = {
      {"num_shards", 1},
      {"hash_method", "simple_mod"},
      {"max_keys_per_request", 4},
      {"servers",
       json::array(
           {json{{"host", "single-host"}, {"port", 54321}, {"shard", 0}}})}};
  const auto deployment = ParseResolvedRdmaDeployment(config);
  ASSERT_EQ(deployment.endpoints.size(), 1u);
  EXPECT_EQ(deployment.endpoints[0].host, "single-host");
  EXPECT_EQ(deployment.endpoints[0].port, 54321);
  EXPECT_EQ(deployment.shard_to_endpoint_index.at(0), 0);
}

TEST(ResolvedRdmaDeploymentTest, ParsesExplicitPerNodeFabric) {
  auto config                                   = json::object();
  config["distributed_client"]                  = DeploymentConfig();
  config["rdma_deployment"]["deployment_id"]    = "unit-test";
  config["rdma_deployment"]["epoch"]            = 1;
  config["rdma_deployment"]["protocol_version"] = 1;
  config["rdma_deployment"]["num_clients"]      = 1;
  config["rdma_deployment"]["nodes"]            = json::array();
  config["rdma_deployment"]["nodes"].push_back(json{
      {"node_id", 0},
      {"role", "server"},
      {"device", "mlx5_0"},
      {"port", 1},
      {"gid_index", 0},
      {"mode", "rocev2"},
      {"hop_limit", 64},
      {"traffic_class", 0},
      {"flow_label", 0}});
  config["rdma_deployment"]["nodes"].push_back(json{
      {"node_id", 1},
      {"role", "server"},
      {"device", "mlx5_0"},
      {"port", 1},
      {"gid_index", 0},
      {"mode", "rocev2"},
      {"hop_limit", 64},
      {"traffic_class", 0},
      {"flow_label", 0}});
  config["rdma_deployment"]["nodes"].push_back(json{
      {"node_id", 2},
      {"role", "client"},
      {"device", "mlx5_1"},
      {"port", 2},
      {"gid_index", 3},
      {"mode", "rocev2"},
      {"hop_limit", 64},
      {"traffic_class", 8},
      {"flow_label", 12}});
  const auto deployment = ParseResolvedRdmaDeploymentConfig(config);
  EXPECT_EQ(deployment.fabrics.at(2).device, "mlx5_1");
  EXPECT_EQ(deployment.fabrics.at(2).port, 2);
  EXPECT_EQ(deployment.fabrics.at(2).gid_index, 3);
  EXPECT_EQ(deployment.fabrics.at(2).hop_limit, 64);
}

TEST(ResolvedRdmaDeploymentTest, RejectsMissingRocEv2NetworkFields) {
  auto config                                   = json::object();
  config["distributed_client"]                  = DeploymentConfig();
  config["rdma_deployment"]["deployment_id"]    = "unit-test";
  config["rdma_deployment"]["epoch"]            = 1;
  config["rdma_deployment"]["protocol_version"] = 1;
  config["rdma_deployment"]["num_clients"]      = 1;
  config["rdma_deployment"]["nodes"]            = json::array({json{
      {"node_id", 0},
      {"role", "server"},
      {"device", "mlx5_0"},
      {"port", 1},
      {"gid_index", 0},
      {"mode", "rocev2"}}});
  EXPECT_THROW(ParseResolvedRdmaDeploymentConfig(config),
               std::invalid_argument);
}

TEST(ResolvedRdmaDeploymentTest, RejectsMissingOrUnsupportedProtocolIdentity) {
  for (const char* field : {"deployment_id", "epoch", "protocol_version"}) {
    auto config = FabricDeploymentConfig();
    config["rdma_deployment"].erase(field);
    SCOPED_TRACE(field);
    EXPECT_THROW(ParseResolvedRdmaDeploymentConfig(config),
                 std::invalid_argument);
  }

  auto config                                   = FabricDeploymentConfig();
  config["rdma_deployment"]["protocol_version"] = 2;
  EXPECT_THROW(ParseResolvedRdmaDeploymentConfig(config),
               std::invalid_argument);
}

TEST(ResolvedRdmaDeploymentTest, RejectsDuplicateFabricNodeId) {
  auto config                                      = FabricDeploymentConfig();
  config["rdma_deployment"]["nodes"][2]["node_id"] = 1;
  EXPECT_THROW(ParseResolvedRdmaDeploymentConfig(config),
               std::invalid_argument);
}

TEST(ResolvedRdmaDeploymentTest, DigestIgnoresExplicitIdArrayOrder) {
  auto first  = FabricDeploymentConfig();
  auto second = first;
  std::reverse(second["distributed_client"]["servers"].begin(),
               second["distributed_client"]["servers"].end());
  std::reverse(second["rdma_deployment"]["nodes"].begin(),
               second["rdma_deployment"]["nodes"].end());

  const auto first_deployment  = ParseResolvedRdmaDeploymentConfig(first);
  const auto second_deployment = ParseResolvedRdmaDeploymentConfig(second);
  EXPECT_EQ(first_deployment.configuration_digest,
            second_deployment.configuration_digest);
  EXPECT_EQ(first_deployment.fabric_digest, second_deployment.fabric_digest);
}

TEST(ResolvedRdmaDeploymentTest, RejectsFabricRoleAndNodeIdMismatch) {
  auto config                                   = FabricDeploymentConfig();
  config["rdma_deployment"]["nodes"][0]["role"] = "client";
  EXPECT_THROW(ParseResolvedRdmaDeploymentConfig(config),
               std::invalid_argument);

  config                                        = FabricDeploymentConfig();
  config["rdma_deployment"]["nodes"][2]["role"] = "server";
  EXPECT_THROW(ParseResolvedRdmaDeploymentConfig(config),
               std::invalid_argument);
}

TEST(ResolvedRdmaDeploymentTest, RejectsMissingFabricServerOrClientNode) {
  auto config = FabricDeploymentConfig();
  config["rdma_deployment"]["nodes"].erase(1);
  EXPECT_THROW(ParseResolvedRdmaDeploymentConfig(config),
               std::invalid_argument);

  config = FabricDeploymentConfig();
  config["rdma_deployment"]["nodes"].erase(2);
  EXPECT_THROW(ParseResolvedRdmaDeploymentConfig(config),
               std::invalid_argument);
}

TEST(ResolvedRdmaDeploymentTest, RejectsMalformedFabricNodeList) {
  auto config                        = FabricDeploymentConfig();
  config["rdma_deployment"]["nodes"] = json::object();
  EXPECT_THROW(ParseResolvedRdmaDeploymentConfig(config),
               std::invalid_argument);

  config                             = FabricDeploymentConfig();
  config["rdma_deployment"]["nodes"] = json::array();
  EXPECT_THROW(ParseResolvedRdmaDeploymentConfig(config),
               std::invalid_argument);
}

} // namespace recstore
