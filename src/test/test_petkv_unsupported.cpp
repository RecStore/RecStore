#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include <unistd.h>

#include "storage/kv_engine/engine_petkv.h"

namespace {

class PetKVUnsupportedTest : public ::testing::Test {
protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("recstore-petkv-" + std::to_string(getpid()));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_ / "shm");
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  BaseKVConfig MakeConfig() const {
    BaseKVConfig config;
    config.json_config_ = {
        {"path", (root_ / "shm").string()},
        {"shard_num", 1},
        {"value_size", 4},
        {"capacity", 16},
        {"value_capacity", 1 << 20},
    };
    return config;
  }

  std::filesystem::path root_;
};

TEST_F(PetKVUnsupportedTest, StoresAndReadsValues) {
  KVEnginePetKV engine(MakeConfig());

  const std::string value("\x01\x02\x03\x04", 4);
  engine.Put(7, value, 0);

  std::string readback;
  engine.Get(7, readback, 0);
  EXPECT_EQ(readback, value);
}

TEST_F(PetKVUnsupportedTest, ReportsCheckpointsAsUnsupported) {
  KVEnginePetKV engine(MakeConfig());
  const std::string path = (root_ / "checkpoint").string();

  EXPECT_EQ(engine.CheckpointRecordCount(), 0u);
  EXPECT_FALSE(engine.SaveCheckpoint(path, "{}"));
  EXPECT_FALSE(engine.LoadCheckpoint(path, "{}"));
}

TEST_F(PetKVUnsupportedTest, ClearDoesNotPretendToResetTheStore) {
  KVEnginePetKV engine(MakeConfig());
  const std::string value("\x09\x08\x07\x06", 4);
  engine.Put(3, value, 0);

  engine.clear();

  // clear() is intentionally unsupported: the override logs an explicit error
  // instead of inheriting the base warning, and does not corrupt the store.
  std::string readback;
  engine.Get(3, readback, 0);
  EXPECT_EQ(readback, value);
}

} // namespace
