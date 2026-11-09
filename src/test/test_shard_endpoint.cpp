#include "ps/base/shard_endpoint.h"

#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>

namespace recstore {
namespace {

nlohmann::json CachePsWithServers() {
  return {{"num_shards", 1},
          {"servers",
           {{{"host", "10.0.0.1"}, {"port", 15123}, {"shard", 0}},
            {{"host", "10.0.0.2"}, {"port", 15124}, {"shard", 1}}}}};
}

TEST(ShardEndpoint, FallsBackWhenServersMissing) {
  const nlohmann::json cache_ps = {{"num_shards", 1}};
  EXPECT_FALSE(SelectSingleShardEndpoint(cache_ps, std::nullopt).has_value());
}

TEST(ShardEndpoint, SelectsShardZeroByDefault) {
  const auto endpoint =
      SelectSingleShardEndpoint(CachePsWithServers(), std::nullopt);
  ASSERT_TRUE(endpoint.has_value());
  EXPECT_EQ(endpoint->host, "10.0.0.1");
  EXPECT_EQ(endpoint->port, 15123);
  EXPECT_EQ(endpoint->shard, 0);
  EXPECT_EQ(FormatShardEndpoint(*endpoint), "10.0.0.1:15123");
}

TEST(ShardEndpoint, SelectsRequestedShard) {
  const auto endpoint = SelectSingleShardEndpoint(CachePsWithServers(), 1);
  ASSERT_TRUE(endpoint.has_value());
  EXPECT_EQ(endpoint->host, "10.0.0.2");
  EXPECT_EQ(endpoint->port, 15124);
  EXPECT_EQ(endpoint->shard, 1);
  EXPECT_EQ(FormatShardEndpoint(*endpoint), "10.0.0.2:15124");
}

TEST(ShardEndpoint, MissingRequestedShardIsAnError) {
  EXPECT_THROW(SelectSingleShardEndpoint(CachePsWithServers(), 2),
               std::runtime_error);
}

TEST(ShardEndpoint, MalformedEntryIsAnError) {
  const nlohmann::json cache_ps = {
      {"servers", {{{"host", "10.0.0.1"}, {"shard", 0}}}}};
  EXPECT_THROW(SelectSingleShardEndpoint(cache_ps, std::nullopt),
               std::runtime_error);
}

} // namespace
} // namespace recstore
