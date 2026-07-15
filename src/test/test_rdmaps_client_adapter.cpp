#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/common/ps_client_config_adapter.h"
#include "ps/client_factory.h"
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

  const auto identity = ResolveEmbeddedRdmaClientIdentity(1);
  EXPECT_EQ(identity.client_index, 1);
  EXPECT_EQ(identity.num_client_processes, 2);
  EXPECT_EQ(identity.global_id, 2);
}

TEST(RDMAPSClientAdapterTest, ResolveEmbeddedIdentityPrefersExplicitOverride) {
  ScopedEnvVar client_index("RECSTORE_RDMA_OS_CLIENT_INDEX", "0");
  ScopedEnvVar num_clients("RECSTORE_RDMA_NUM_CLIENT_PROCESSES", "3");
  ScopedEnvVar rank("RANK", "2");
  ScopedEnvVar world_size("WORLD_SIZE", "2");

  const auto identity = ResolveEmbeddedRdmaClientIdentity(1);
  EXPECT_EQ(identity.client_index, 0);
  EXPECT_EQ(identity.num_client_processes, 3);
  EXPECT_EQ(identity.global_id, 1);
}

TEST(RDMAPSClientAdapterTest, ResolveEmbeddedIdentityRejectsOutOfRangeIndex) {
  ScopedEnvVar rank("RANK", "2");
  ScopedEnvVar world_size("WORLD_SIZE", "2");

  EXPECT_THROW(ResolveEmbeddedRdmaClientIdentity(1), std::runtime_error);
}

TEST(RDMAPSClientAdapterTest, RuntimeReadsGetResponseModeFromEnv) {
  ScopedEnvVar response_mode(
      "RECSTORE_RDMA_GET_RESPONSE_MODE", "staging_copy");

  InitializeRdmaProcessRuntime();

  EXPECT_EQ(FLAGS_rdma_get_response_mode, "staging_copy");
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
  };

  auto client =
      CreatePSClient(ResolvePSClientOptionsFromFrameworkConfig(config));
  ASSERT_NE(client, nullptr);
  EXPECT_NE(dynamic_cast<RDMAPSClientAdapter*>(client.get()), nullptr);
}

} // namespace recstore
