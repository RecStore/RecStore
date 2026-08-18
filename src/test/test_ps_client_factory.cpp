#include <gtest/gtest.h>

#include "framework/common/ps_client_config_adapter.h"
#include "ps/client_factory.h"
#include "ps/brpc/dist_brpc_ps_client.h"
#include "ps/grpc/dist_grpc_ps_client.h"
#include "ps/local_shm/local_shm_client.h"

namespace recstore {

TEST(PSClientFactoryTest, AllowsRdmaForFrameworkUsage) {
  json config = {
      {"cache_ps", {{"ps_type", "RDMA"}}},
      {"client", {{"host", "127.0.0.1"}, {"port", 25000}, {"shard", 0}}},
      {"distributed_client",
       {{"num_shards", 1},
        {"hash_method", "city_hash"},
        {"servers",
         json::array(
             {{{"host", "127.0.0.1"}, {"port", 25000}, {"shard", 0}}})}}},
  };

  EXPECT_EQ(ResolveFrameworkPSClientType(config), PSClientType::kRdma);

  std::unique_ptr<BasePSClient> client =
      CreatePSClient(ResolvePSClientOptionsFromFrameworkConfig(config));
  ASSERT_NE(client, nullptr);
}

TEST(PSClientFactoryTest, RejectsUnknownType) {
  json config = {{"cache_ps", {{"ps_type", "banana"}}}};

  EXPECT_THROW(
      {
        const auto type = ResolveFrameworkPSClientType(config);
        (void)type;
      },
      std::invalid_argument);
}

TEST(PSClientFactoryTest, UsesDefaultGrpcClientConfig) {
  json config = {{"cache_ps", {{"ps_type", "grpc"}}}};

  EXPECT_EQ(ResolveFrameworkPSClientType(config), PSClientType::kGrpc);

  json client_config = ResolveFrameworkPSClientTransportConfig(config);
  EXPECT_EQ(client_config["host"], "127.0.0.1");
  EXPECT_EQ(client_config["port"], 15000);
  EXPECT_EQ(client_config["shard"], 0);
}

TEST(PSClientFactoryTest, PreservesExplicitBrpcClientConfig) {
  json config = {
      {"cache_ps", {{"ps_type", "BRPC"}}},
      {"client", {{"host", "10.0.0.5"}, {"port", 25123}, {"shard", 1}}},
  };

  EXPECT_EQ(ResolveFrameworkPSClientType(config), PSClientType::kBrpc);

  json client_config = ResolveFrameworkPSClientTransportConfig(config);
  EXPECT_EQ(client_config["host"], "10.0.0.5");
  EXPECT_EQ(client_config["port"], 25123);
  EXPECT_EQ(client_config["shard"], 1);
}

TEST(PSClientFactoryTest, ResolvesDistributedClientConfigWithFieldFallback) {
  json config = {
      {"cache_ps",
       {{"ps_type", "BRPC"}, {"num_shards", 2}, {"hash_method", "city_hash"}}},
      {"distributed_client",
       {{"servers",
         json::array(
             {{{"host", "10.0.0.1"}, {"port", 25123}, {"shard", 1}},
              {{"host", "10.0.0.2"}, {"port", 25124}, {"shard", 0}}})}}},
  };

  json client_config = ResolveFrameworkDistributedClientConfig(config);
  EXPECT_EQ(client_config["num_shards"], 2);
  EXPECT_EQ(client_config["hash_method"], "city_hash");
  ASSERT_TRUE(client_config["servers"].is_array());
  ASSERT_EQ(client_config["servers"].size(), 2);
  EXPECT_EQ(client_config["servers"][0]["shard"], 1);

  PSClientCreateOptions options =
      ResolvePSClientOptionsFromFrameworkConfig(config);
  EXPECT_EQ(options.raw_config["distributed_client"]["num_shards"], 2);
  EXPECT_EQ(options.raw_config["distributed_client"]["hash_method"],
            "city_hash");
}

TEST(PSClientFactoryTest, DefaultsDistributedNumShardsToServerCount) {
  json config = {
      {"cache_ps", {{"ps_type", "GRPC"}}},
      {"distributed_client",
       {{"hash_method", "simple_mod"},
        {"servers",
         json::array(
             {{{"host", "10.0.0.1"}, {"port", 25123}, {"shard", 0}},
              {{"host", "10.0.0.2"}, {"port", 25124}, {"shard", 1}}})}}},
  };

  json client_config = ResolveFrameworkDistributedClientConfig(config);
  EXPECT_EQ(client_config["num_shards"], 2);
  EXPECT_TRUE(HasFrameworkDistributedClientConfig(config));
}

TEST(PSClientFactoryTest, CreatesDistributedGrpcClientWhenConfigured) {
  json config = {
      {"cache_ps", {{"ps_type", "GRPC"}}},
      {"distributed_client",
       {{"num_shards", 2},
        {"hash_method", "simple_mod"},
        {"servers",
         json::array(
             {{{"host", "127.0.0.1"}, {"port", 25000}, {"shard", 0}},
              {{"host", "127.0.0.1"}, {"port", 25001}, {"shard", 1}}})}}},
  };

  std::unique_ptr<BasePSClient> client =
      CreatePSClient(ResolvePSClientOptionsFromFrameworkConfig(config));
  ASSERT_NE(client, nullptr);
  EXPECT_NE(dynamic_cast<DistributedGRPCParameterClient*>(client.get()),
            nullptr);
}

TEST(PSClientFactoryTest, CreatesDistributedBrpcClientWhenConfigured) {
  json config = {
      {"cache_ps", {{"ps_type", "BRPC"}}},
      {"distributed_client",
       {{"num_shards", 2},
        {"hash_method", "simple_mod"},
        {"servers",
         json::array(
             {{{"host", "127.0.0.1"}, {"port", 25000}, {"shard", 0}},
              {{"host", "127.0.0.1"}, {"port", 25001}, {"shard", 1}}})}}},
  };

  std::unique_ptr<BasePSClient> client =
      CreatePSClient(ResolvePSClientOptionsFromFrameworkConfig(config));
  ASSERT_NE(client, nullptr);
  EXPECT_NE(dynamic_cast<DistributedBRPCParameterClient*>(client.get()),
            nullptr);
}

TEST(PSClientFactoryTest, CreatesDistributedBrpcClientForSingleServer) {
  json config = {
      {"cache_ps", {{"ps_type", "BRPC"}}},
      {"client", {{"host", "127.0.0.1"}, {"port", 25000}, {"shard", 0}}},
  };

  std::unique_ptr<BasePSClient> client =
      CreatePSClient(ResolvePSClientOptionsFromFrameworkConfig(config));
  ASSERT_NE(client, nullptr);
  EXPECT_NE(dynamic_cast<DistributedBRPCParameterClient*>(client.get()),
            nullptr);
}

TEST(PSClientFactoryTest, CreatesLocalShmClientFromConfig) {
  json config = {
      {"cache_ps", {{"ps_type", "LOCAL_SHM"}}},
      {"local_shm",
       {{"region_name", "recstore_local_ps_factory_test"},
        {"slot_count", 4},
        {"slot_buffer_bytes", 4096},
        {"client_timeout_ms", 1000}}},
  };

  EXPECT_EQ(ResolveFrameworkPSClientType(config), PSClientType::kLocalShm);

  json client_config = ResolveFrameworkPSClientTransportConfig(config);
  EXPECT_EQ(client_config["region_name"], "recstore_local_ps_factory_test");
  EXPECT_EQ(client_config["slot_count"], 4);
  EXPECT_EQ(client_config["slot_buffer_bytes"], 4096);

  std::unique_ptr<BasePSClient> client =
      CreatePSClient(ResolvePSClientOptionsFromFrameworkConfig(config));
  ASSERT_NE(client, nullptr);
  EXPECT_NE(dynamic_cast<LocalShmPSClient*>(client.get()), nullptr);
}

} // namespace recstore
