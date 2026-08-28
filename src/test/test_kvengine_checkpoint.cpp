#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "memory/shm_file.h"
#include "storage/kv_engine/engine_composite.h"

namespace {

class KVEngineCheckpointTest : public ::testing::Test {
protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("recstore-kv-checkpoint-" + std::to_string(getpid()));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);

    base::PMMmapRegisterCenter::GetConfig().backend =
        base::PMMmapRegisterCenter::Backend::kAnonymousDram;
    base::PMMmapRegisterCenter::GetConfig().numa_id = 0;
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  std::unique_ptr<KVEngineComposite> MakeEngine(const std::string& name) {
    const std::filesystem::path value_path = root_ / name / "value";
    std::filesystem::create_directories(value_path.parent_path());

    BaseKVConfig config;
    config.num_threads_ = 2;
    config.json_config_ = {
        {"engine_type", "KVEngineComposite"},
        {"capacity", 1024},
        {"index", {{"type", "DRAM_UNORDERED_MAP"}}},
        {"value",
         {{"type", "DRAM_VALUE_STORE"},
          {"path", value_path.string()},
          {"default_value_size_hint", 4 * sizeof(float)},
          {"dram_allocator",
           {{"type", "PERSIST_LOOP_SLAB"}, {"capacity_bytes", 1024 * 1024}}}}}};
    return std::make_unique<KVEngineComposite>(config);
  }

  template <size_t N>
  static std::string FloatBytes(const std::array<float, N>& values) {
    return std::string(reinterpret_cast<const char*>(values.data()),
                       values.size() * sizeof(float));
  }

  std::filesystem::path root_;
};

TEST_F(KVEngineCheckpointTest, RoundTripsParameterAndRowWiseAccumulator) {
  constexpr uint64_t parameter_key   = 42;
  constexpr uint64_t batch_key       = 43;
  constexpr uint64_t accumulator_key = (1ULL << 56) | parameter_key;
  const std::string metadata =
      R"({"run_id":"hstu-test","step":10,"shard_id":0})";
  const std::string parameter =
      FloatBytes(std::array<float, 4>{1.0f, -2.0f, 3.5f, 4.0f});
  const std::array<float, 4> batch_parameter = {5.0f, 6.0f, 7.0f, 8.0f};
  const std::string accumulator = FloatBytes(std::array<float, 1>{12.5f});
  const std::filesystem::path checkpoint = root_ / "sparse.ckpt";

  auto source                           = MakeEngine("source");
  const std::vector<uint64_t> bulk_keys = {parameter_key};
  source->BulkLoad(base::ConstArray<uint64_t>(bulk_keys), parameter.data());
  const std::vector<uint64_t> batch_keys            = {batch_key};
  std::vector<base::ConstArray<float>> batch_values = {
      base::ConstArray<float>(batch_parameter.data(), batch_parameter.size())};
  source->BatchPut(base::ConstArray<uint64_t>(batch_keys), &batch_values, 0);
  source->Put(accumulator_key, accumulator, 0);
  ASSERT_EQ(source->CheckpointRecordCount(), 3);
  ASSERT_TRUE(source->SaveCheckpoint(checkpoint.string(), metadata));
  EXPECT_FALSE(std::filesystem::exists(checkpoint.string() + ".tmp"));

  // Loading cannot silently merge a checkpoint into live sparse state.
  EXPECT_FALSE(source->LoadCheckpoint(checkpoint.string(), metadata));
  EXPECT_EQ(source->CheckpointRecordCount(), 3);
  source.reset();

  auto restored = MakeEngine("restored");
  ASSERT_TRUE(restored->LoadCheckpoint(checkpoint.string(), metadata));
  EXPECT_EQ(restored->CheckpointRecordCount(), 3);

  std::string actual;
  restored->Get(parameter_key, actual, 0);
  EXPECT_EQ(actual, parameter);
  restored->Get(batch_key, actual, 0);
  EXPECT_EQ(actual, FloatBytes(batch_parameter));
  restored->Get(accumulator_key, actual, 0);
  EXPECT_EQ(actual, accumulator);
}

TEST_F(KVEngineCheckpointTest, RoundTripsConcurrentWriterKeys) {
  constexpr uint64_t keys_per_thread = 100;
  const std::string metadata = R"({"run_id":"concurrent-writers"})";
  const std::string value = FloatBytes(std::array<float, 1>{3.5f});
  const std::filesystem::path checkpoint = root_ / "concurrent.ckpt";
  auto source = MakeEngine("concurrent-source");

  std::vector<std::thread> writers;
  for (unsigned tid = 0; tid < 2; ++tid) {
    writers.emplace_back([&, tid]() {
      for (uint64_t i = 0; i < keys_per_thread; ++i) {
        source->Put(tid * keys_per_thread + i, value, tid);
      }
    });
  }
  for (auto& writer : writers) {
    writer.join();
  }

  ASSERT_EQ(source->CheckpointRecordCount(), 2 * keys_per_thread);
  ASSERT_TRUE(source->SaveCheckpoint(checkpoint.string(), metadata));
  source.reset();

  auto restored = MakeEngine("concurrent-restored");
  ASSERT_TRUE(restored->LoadCheckpoint(checkpoint.string(), metadata));
  EXPECT_EQ(restored->CheckpointRecordCount(), 2 * keys_per_thread);
  std::string actual;
  restored->Get(0, actual, 0);
  EXPECT_EQ(actual, value);
  restored->Get(2 * keys_per_thread - 1, actual, 1);
  EXPECT_EQ(actual, value);
}

TEST_F(KVEngineCheckpointTest, RejectsMetadataMismatchAndMalformedFiles) {
  const std::string metadata =
      R"({"run_id":"hstu-test","step":10,"shard_id":0})";
  const std::filesystem::path checkpoint = root_ / "valid.ckpt";

  auto source = MakeEngine("source");
  source->Put(7, FloatBytes(std::array<float, 2>{1.0f, 2.0f}), 0);
  ASSERT_TRUE(source->SaveCheckpoint(checkpoint.string(), metadata));
  source.reset();

  auto mismatch = MakeEngine("mismatch");
  EXPECT_FALSE(mismatch->LoadCheckpoint(
      checkpoint.string(), R"({"run_id":"different","step":10,"shard_id":0})"));
  EXPECT_EQ(mismatch->CheckpointRecordCount(), 0);
  mismatch.reset();

  std::ifstream input(checkpoint, std::ios::binary);
  ASSERT_TRUE(input.good());
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  ASSERT_GT(bytes.size(), 1);

  const std::filesystem::path truncated = root_ / "truncated.ckpt";
  {
    std::ofstream output(truncated, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size() - 1));
    ASSERT_TRUE(output.good());
  }
  auto truncated_target = MakeEngine("truncated-target");
  EXPECT_FALSE(truncated_target->LoadCheckpoint(truncated.string(), metadata));
  EXPECT_EQ(truncated_target->CheckpointRecordCount(), 0);
  truncated_target.reset();

  const std::filesystem::path bad_magic = root_ / "bad-magic.ckpt";
  bytes.front() ^= 0x1;
  {
    std::ofstream output(bad_magic, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(output.good());
  }
  auto bad_magic_target = MakeEngine("bad-magic-target");
  EXPECT_FALSE(bad_magic_target->LoadCheckpoint(bad_magic.string(), metadata));
  EXPECT_EQ(bad_magic_target->CheckpointRecordCount(), 0);

  const std::filesystem::path corrupt_value = root_ / "corrupt-value.ckpt";
  bytes.front() ^= 0x1;
  ASSERT_GT(bytes.size(), sizeof(uint64_t));
  bytes[bytes.size() - sizeof(uint64_t) - 1] ^= 0x1;
  {
    std::ofstream output(corrupt_value, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(output.good());
  }
  auto corrupt_value_target = MakeEngine("corrupt-value-target");
  EXPECT_FALSE(
      corrupt_value_target->LoadCheckpoint(corrupt_value.string(), metadata));
  EXPECT_EQ(corrupt_value_target->CheckpointRecordCount(), 0);
}

} // namespace
