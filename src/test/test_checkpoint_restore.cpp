#include "ps/base/checkpoint_restore.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace recstore {
namespace {

TEST(CheckpointRestoreConfigTest, NoBlockDisablesRestore) {
  const nlohmann::json cache_ps = {{"num_shards", 1}};

  const auto config = ParseCheckpointRestoreConfig(cache_ps);

  EXPECT_TRUE(config.restore_dir.empty());
  EXPECT_TRUE(config.fail_if_missing);
}

TEST(CheckpointRestoreConfigTest, ReadsDirectoryAndOnMissing) {
  const nlohmann::json cache_ps = {
      {"checkpoint",
       {{"restore_dir", "/shared/ckpt/run-7"}, {"on_missing", "start_empty"}}}};

  const auto config = ParseCheckpointRestoreConfig(cache_ps);

  EXPECT_EQ(config.restore_dir, "/shared/ckpt/run-7");
  EXPECT_FALSE(config.fail_if_missing);
}

TEST(CheckpointRestoreConfigTest, DefaultsToFailWhenManifestIsMissing) {
  const nlohmann::json cache_ps = {
      {"checkpoint", {{"restore_dir", "/shared/ckpt/run-7"}}}};

  const auto config = ParseCheckpointRestoreConfig(cache_ps);

  EXPECT_EQ(config.restore_dir, "/shared/ckpt/run-7");
  EXPECT_TRUE(config.fail_if_missing);
}

TEST(CheckpointRestoreConfigTest, RejectsUnknownOnMissing) {
  const nlohmann::json cache_ps = {
      {"checkpoint", {{"restore_dir", "/tmp/x"}, {"on_missing", "maybe"}}}};

  EXPECT_THROW(ParseCheckpointRestoreConfig(cache_ps), std::invalid_argument);
}

TEST(CheckpointRestoreConfigTest, RejectsNonStringRestoreDir) {
  const nlohmann::json cache_ps = {{"checkpoint", {{"restore_dir", 42}}}};

  EXPECT_THROW(ParseCheckpointRestoreConfig(cache_ps), std::invalid_argument);
}

TEST(CheckpointRestoreConfigTest, RejectsNonObjectBlock) {
  const nlohmann::json cache_ps = {{"checkpoint", "enabled"}};

  EXPECT_THROW(ParseCheckpointRestoreConfig(cache_ps), std::invalid_argument);
}

TEST(CheckpointMetadataTest, BindsShardIdAndOptimizerLayout) {
  const std::string metadata =
      R"({"checkpoint_id":"step-00000007","identity":{"table":{"name":"t_a"}}})";

  const auto bound = nlohmann::json::parse(BindCheckpointMetadata(metadata, 3));

  EXPECT_EQ(bound["shard_id"], 3);
  EXPECT_EQ(bound["optimizer_layout"], kCheckpointOptimizerLayout);
  EXPECT_EQ(bound["checkpoint_id"], "step-00000007");
}

TEST(CheckpointMetadataTest, OverridesClientClaimedOptimizerLayout) {
  const std::string metadata =
      R"({"checkpoint_id":"step-00000007","identity":{"table":"t_a"},)"
      R"("optimizer_layout":99})";

  const auto bound = nlohmann::json::parse(BindCheckpointMetadata(metadata, 0));

  EXPECT_EQ(bound["optimizer_layout"], kCheckpointOptimizerLayout);
}

TEST(CheckpointMetadataTest, RejectsMissingIdentityOrCheckpointId) {
  EXPECT_THROW(BindCheckpointMetadata(R"({"identity":{"table":"t_a"}})", 0),
               std::invalid_argument);
  EXPECT_THROW(
      BindCheckpointMetadata(R"({"checkpoint_id":"x","identity":1})", 0),
      std::invalid_argument);
  EXPECT_THROW(
      BindCheckpointMetadata(R"({"checkpoint_id":"","identity":{}})", 0),
      std::invalid_argument);
  EXPECT_THROW(
      BindCheckpointMetadata(R"({"checkpoint_id":"x","identity":{}})", -1),
      std::invalid_argument);
}

} // namespace
} // namespace recstore
