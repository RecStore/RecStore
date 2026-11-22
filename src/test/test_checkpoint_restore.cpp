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

} // namespace
} // namespace recstore
