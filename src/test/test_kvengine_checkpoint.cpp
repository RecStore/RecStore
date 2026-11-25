#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "memory/shm_file.h"
#include "storage/kv_engine/engine_composite.h"

namespace {

// Holds the first value read of a checkpoint until a concurrent Put lands, so
// the test can tell whether commits wait for the whole dump.
class BlockingDramValueStore : public DramValueStore {
public:
  using DramValueStore::DramValueStore;

  void ReadValue(uint64_t handle, std::string& value) override {
    if (block_next_.exchange(false)) {
      read_entered_.store(true);
      std::unique_lock<std::mutex> lock(wait_mu_);
      cv_.wait_for(lock, std::chrono::seconds(2), [this] {
        return put_done_.load();
      });
      put_landed_during_read_.store(put_done_.load());
    }
    DramValueStore::ReadValue(handle, value);
  }

  void BlockNextRead() { block_next_.store(true); }

  void WaitForRead() {
    while (!read_entered_.load())
      std::this_thread::yield();
  }

  void NotifyPutDone() {
    {
      std::lock_guard<std::mutex> lock(wait_mu_);
      put_done_.store(true);
    }
    cv_.notify_all();
  }

  bool PutLandedDuringRead() const { return put_landed_during_read_.load(); }

private:
  std::atomic<bool> block_next_{false};
  std::atomic<bool> read_entered_{false};
  std::atomic<bool> put_done_{false};
  std::atomic<bool> put_landed_during_read_{false};
  std::mutex wait_mu_;
  std::condition_variable cv_;
};

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

  std::unique_ptr<KVEngineComposite>
  MakeEngine(const std::string& name,
             const std::string& index_type = "DRAM_UNORDERED_MAP") {
    const std::filesystem::path value_path = root_ / name / "value";
    std::filesystem::create_directories(value_path.parent_path());

    BaseKVConfig config;
    config.num_threads_ = 2;
    config.json_config_ = {
        {"engine_type", "KVEngineComposite"},
        {"capacity", 1024},
        {"index", {{"type", index_type}}},
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
  const std::string metadata         = R"({"run_id":"concurrent-writers"})";
  const std::string value            = FloatBytes(std::array<float, 1>{3.5f});
  const std::filesystem::path checkpoint = root_ / "concurrent.ckpt";
  auto source                            = MakeEngine("concurrent-source");

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

TEST_F(KVEngineCheckpointTest, RoundTripsOverwriteDeleteAndReinsert) {
  for (const std::string index : {"DRAM_UNORDERED_MAP", "DRAM_PET_HASH"}) {
    SCOPED_TRACE(index);
    auto source                       = MakeEngine(index + "-overwrite", index);
    const std::array<float, 4> first  = {1, 1, 1, 1};
    const std::array<float, 4> second = {2, 2, 2, 2};
    const std::array<float, 4> last   = {3, 3, 3, 3};
    for (uint64_t key : {10, 20, 30}) {
      source->Put(key, FloatBytes(first), 0);
      source->Put(key, FloatBytes(second), 1);
    }
    const std::vector<uint64_t> batch_keys = {20, 30, 40};
    std::vector<base::ConstArray<float>> batch_values(
        batch_keys.size(), base::ConstArray<float>(last.data(), last.size()));
    source->BatchPut(base::ConstArray<uint64_t>(batch_keys), &batch_values, 1);
    const std::vector<uint64_t> deleted = {10, 20, 999};
    ASSERT_EQ(source->BatchDelete(base::ConstArray<uint64_t>(deleted), 0), 2);
    ASSERT_EQ(source->CheckpointRecordCount(), 2);
    source->Put(10, FloatBytes(first), 1);
    const std::vector<uint64_t> duplicate_keys            = {20, 20};
    std::vector<base::ConstArray<float>> duplicate_values = {
        base::ConstArray<float>(second.data(), second.size()),
        base::ConstArray<float>(last.data(), last.size())};
    source->BatchPut(
        base::ConstArray<uint64_t>(duplicate_keys), &duplicate_values, 0);
    ASSERT_EQ(source->CheckpointRecordCount(), 4);
    const auto checkpoint = root_ / (index + "-overwrite.ckpt");
    ASSERT_TRUE(source->SaveCheckpoint(checkpoint.string(), index));
    source.reset();
    auto restored = MakeEngine(index + "-restored", index);
    ASSERT_TRUE(restored->LoadCheckpoint(checkpoint.string(), index));
    ASSERT_EQ(restored->CheckpointRecordCount(), 4);
    std::string value;
    restored->Get(10, value, 0);
    EXPECT_EQ(value, FloatBytes(first));
    for (uint64_t key : {20, 30, 40}) {
      restored->Get(key, value, 0);
      EXPECT_EQ(value, FloatBytes(last));
    }
    restored->Get(999, value, 0);
    EXPECT_TRUE(value.empty());
  }
}

TEST_F(KVEngineCheckpointTest, SnapshotsConcurrentPetHashWriters) {
  auto source               = MakeEngine("pet-concurrent", "DRAM_PET_HASH");
  const std::string initial = FloatBytes(std::array<float, 4>{1, 1, 1, 1});
  const std::array<std::string, 2> updates = {
      FloatBytes(std::array<float, 4>{2, 2, 2, 2}),
      FloatBytes(std::array<float, 4>{3, 3, 3, 3})};
  for (uint64_t key = 0; key < 64; ++key)
    source->Put(key, initial, 0);
  std::atomic<unsigned> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> writers;
  for (unsigned tid = 0; tid < 2; ++tid) {
    writers.emplace_back([&, tid] {
      ready.fetch_add(1);
      while (!start.load())
        std::this_thread::yield();
      for (unsigned round = 0; round < 100; ++round) {
        for (uint64_t key = 0; key < 128; ++key) {
          source->Put(key, updates[tid], tid);
        }
      }
    });
  }
  while (ready.load() != 2)
    std::this_thread::yield();
  start.store(true);
  std::vector<std::filesystem::path> checkpoints;
  bool saved = true;
  for (int i = 0; i < 3; ++i) {
    checkpoints.push_back(
        root_ / ("concurrent-" + std::to_string(i) + ".ckpt"));
    saved &= source->SaveCheckpoint(checkpoints.back().string(), "concurrent");
  }
  for (auto& writer : writers)
    writer.join();
  ASSERT_TRUE(saved);
  ASSERT_EQ(source->CheckpointRecordCount(), 128);
  checkpoints.push_back(root_ / "concurrent-final.ckpt");
  ASSERT_TRUE(
      source->SaveCheckpoint(checkpoints.back().string(), "concurrent"));
  source.reset();
  for (size_t i = 0; i < checkpoints.size(); ++i) {
    auto restored =
        MakeEngine("snapshot-" + std::to_string(i), "DRAM_PET_HASH");
    ASSERT_TRUE(
        restored->LoadCheckpoint(checkpoints[i].string(), "concurrent"));
    EXPECT_GE(restored->CheckpointRecordCount(), 64);
    EXPECT_LE(restored->CheckpointRecordCount(), 128);
    for (uint64_t key = 0; key < 128; ++key) {
      std::string value;
      restored->Get(key, value, 0);
      if (key < 64 || i + 1 == checkpoints.size())
        EXPECT_FALSE(value.empty());
      EXPECT_TRUE(value.empty() || value == initial || value == updates[0] ||
                  value == updates[1]);
    }
  }
}

TEST_F(KVEngineCheckpointTest, DumpKeepsCommitsOffTheLockAndStaysPointInTime) {
  const std::filesystem::path value_path = root_ / "deferred" / "value";
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

  auto store = std::make_unique<BlockingDramValueStore>(config);
  BlockingDramValueStore* store_ptr = store.get();
  auto source                       = std::make_unique<KVEngineComposite>(
      std::make_unique<DramUnorderedMapIndex>(config), std::move(store), 2);

  const std::string initial = FloatBytes(std::array<float, 4>{1, 1, 1, 1});
  const std::string updated = FloatBytes(std::array<float, 4>{2, 2, 2, 2});
  for (uint64_t key = 0; key < 8; ++key)
    source->Put(key, initial, 0);

  const std::filesystem::path checkpoint = root_ / "deferred.ckpt";
  store_ptr->BlockNextRead();
  std::atomic<bool> saved{false};
  std::thread dump([&] {
    saved.store(source->SaveCheckpoint(checkpoint.string(), "deferred"));
  });
  store_ptr->WaitForRead();
  // Both land after the key snapshot: the dump must not wait for them, and
  // key 0 must still be written with the value from the snapshot instant.
  source->Put(0, updated, 0);
  source->Put(100, updated, 0);
  store_ptr->NotifyPutDone();
  dump.join();

  ASSERT_TRUE(saved.load());
  EXPECT_TRUE(store_ptr->PutLandedDuringRead());
  source.reset();

  auto restored = MakeEngine("deferred-restored");
  ASSERT_TRUE(restored->LoadCheckpoint(checkpoint.string(), "deferred"));
  EXPECT_EQ(restored->CheckpointRecordCount(), 8);
  std::string actual;
  restored->Get(0, actual, 0);
  EXPECT_EQ(actual, initial);
  restored->Get(100, actual, 0);
  EXPECT_TRUE(actual.empty());
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
