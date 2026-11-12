#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "storage/io_backend/force_link.h"
#include "storage/io_backend/io_backend.h"
#include "storage/value_store/hybrid_value_store.h"
#include "storage/value_store/ssd_log_value_store.h"

namespace {

class InMemoryLogBackend : public IOBackend {
public:
  explicit InMemoryLogBackend(const BaseKVConfig& config) : IOBackend(config) {}
  void init() override {}
  void*
  GetPage(coroutine<void>::push_type&, uint64_t, PageID_t page_id) override {
    return GetPage(page_id);
  }
  void* GetPage(PageID_t) override {
    throw std::runtime_error("GetPage is unsupported");
  }
  void
  Unpin(coroutine<void>::push_type&, uint64_t, PageID_t, void*, bool) override {
    throw std::runtime_error("Unpin is unsupported");
  }
  void Unpin(PageID_t, void*, bool) override {
    throw std::runtime_error("Unpin is unsupported");
  }
  char* AllocateBuffer() override { return AllocateBuffer(1); }
  char* AllocateBuffer(uint64_t page_count) override {
    return new char[page_count * PAGE_SIZE]();
  }
  void FreeBuffer(char* buf) override { delete[] buf; }
  void submit() override {}
  void PollCompletion() override {}

  uint64_t Writes() const { return writes_.load(); }
  uint64_t Reads() const { return reads_.load(); }

protected:
  void ReadPageAsync(coroutine<void>::push_type&,
                     uint64_t,
                     PageID_t page_id,
                     char* buffer) override {
    ReadPageSync(page_id, buffer);
  }
  void ReadPageSync(PageID_t page_id, char* buffer) override {
    reads_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = pages_.find(page_id);
    if (found == pages_.end()) {
      std::memset(buffer, 0, PAGE_SIZE);
      return;
    }
    std::memcpy(buffer, found->second.data(), PAGE_SIZE);
  }
  void WritePageAsync(coroutine<void>::push_type&,
                      uint64_t,
                      PageID_t page_id,
                      char* buffer) override {
    WritePageSync(page_id, buffer);
  }
  void WritePageSync(PageID_t page_id, char* buffer) override {
    std::lock_guard<std::mutex> lock(mutex_);
    pages_[page_id] = std::string(buffer, PAGE_SIZE);
    writes_.fetch_add(1, std::memory_order_relaxed);
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<PageID_t, std::string> pages_;
  std::atomic<uint64_t> reads_{0};
  std::atomic<uint64_t> writes_{0};
};

FACTORY_REGISTER(
    IOBackend, INMEMORY_LOG, InMemoryLogBackend, const BaseKVConfig&);

BaseKVConfig Config(uint64_t log_capacity_bytes,
                    std::vector<int> size_classes = {128, 8192}) {
  BaseKVConfig config;
  config.json_config_ = {
      {"value",
       {{"type", "SSD_LOG_VALUE_STORE"},
        {"path", "/tmp/recstore-test-value-log.db"},
        {"default_value_size_hint", 128},
        {"ssd_allocator",
         {{"type", "SSD_SLAB"},
          {"capacity_bytes", 128ULL << 20},
          {"size_classes", size_classes},
          {"io",
           {{"type", "INMEMORY_LOG"},
            {"queue_depth", 8},
            {"base_offset_bytes", 0}}}}},
        {"write_log",
         {{"enabled", true},
          {"path", "/tmp/recstore-test-write-log.log"},
          {"capacity_bytes", log_capacity_bytes},
          {"segment_bytes", 64ULL << 10},
          {"io", {{"type", "INMEMORY_LOG"}, {"queue_depth", 8}}}}}}}};
  return config;
}

uint64_t FieldOf(const std::string& info, const std::string& field) {
  const auto pos = info.find(field);
  if (pos == std::string::npos) {
    return 0;
  }
  return std::stoull(info.substr(pos + field.size()));
}

std::string Value(char fill, size_t size = 128) {
  return std::string(size, fill);
}

TEST(SsdLogValueStoreTest, LatestWriteIsVisibleImmediately) {
  SsdLogValueStore store(Config(1ULL << 20));
  const auto first      = Value('a');
  const uint64_t handle = store.AllocAndWrite(first.data(), first.size());
  ASSERT_NE(handle, kValueHandleNone);

  std::string out(first.size(), '\0');
  ASSERT_EQ(store.Read(handle, out.data(), out.size()), first.size());
  EXPECT_EQ(out, first);

  // The entry still sits in the append buffer; a read must flush it first.
  const auto second = Value('b');
  store.Write(handle, second.data(), second.size());
  out.assign(second.size(), '\0');
  ASSERT_EQ(store.Read(handle, out.data(), out.size()), second.size());
  EXPECT_EQ(out, second);

  const std::string info = store.ExtraResultFields();
  EXPECT_EQ(FieldOf(info, " write_log_appends="), 2U);
  EXPECT_EQ(FieldOf(info, " write_log_fallbacks="), 0U);
  EXPECT_EQ(FieldOf(info, " write_log_index_entries="), 1U);
}

TEST(SsdLogValueStoreTest, BatchReadMixesLogAndInPlaceRows) {
  SsdLogValueStore store(Config(1ULL << 20));
  std::vector<uint64_t> handles;
  std::vector<std::string> expected;
  for (size_t i = 0; i < 200; ++i) {
    expected.push_back(Value(static_cast<char>('A' + (i % 26))));
    handles.push_back(
        store.AllocAndWrite(expected.back().data(), expected.back().size()));
  }
  for (size_t i = 0; i < handles.size(); i += 2) {
    expected[i] = Value(static_cast<char>('a' + (i % 26)));
    store.Write(handles[i], expected[i].data(), expected[i].size());
  }
  std::vector<ValueStore::ReadResult> results;
  store.BatchRead(handles, results);
  ASSERT_EQ(results.size(), handles.size());
  for (size_t i = 0; i < handles.size(); ++i) {
    EXPECT_EQ(std::string(results[i].data.begin(), results[i].data.end()),
              expected[i])
        << "row=" << i;
  }
}

TEST(SsdLogValueStoreTest, FallsBackWhenLogIsFull) {
  // 64 KiB = 16 pages, about 448 rows of 128 B.
  SsdLogValueStore store(Config(64ULL << 10));
  std::vector<uint64_t> handles;
  std::vector<std::string> expected;
  for (size_t i = 0; i < 1000; ++i) {
    expected.push_back(Value(static_cast<char>('A' + (i % 26))));
    handles.push_back(
        store.AllocAndWrite(expected.back().data(), expected.back().size()));
    ASSERT_NE(handles.back(), kValueHandleNone);
  }
  const std::string info = store.ExtraResultFields();
  EXPECT_GT(FieldOf(info, " write_log_fallbacks="), 0U);
  EXPECT_EQ(FieldOf(info, " write_log_pages_written="), 16U);

  for (size_t i = 0; i < handles.size(); ++i) {
    std::string out(expected[i].size(), '\0');
    ASSERT_EQ(store.Read(handles[i], out.data(), out.size()),
              expected[i].size())
        << "row=" << i;
    EXPECT_EQ(out, expected[i]) << "row=" << i;
  }
}

TEST(SsdLogValueStoreTest, FreeDropsIndexEntry) {
  SsdLogValueStore store(Config(1ULL << 20));
  const auto value      = Value('x');
  const uint64_t handle = store.AllocAndWrite(value.data(), value.size());
  ASSERT_NE(handle, kValueHandleNone);
  EXPECT_EQ(FieldOf(store.ExtraResultFields(), " write_log_index_entries="),
            1U);
  store.Free(handle);
  EXPECT_EQ(FieldOf(store.ExtraResultFields(), " write_log_index_entries="),
            0U);
}

TEST(SsdLogValueStoreTest, OversizedValueFallsBackToInPlaceWrite) {
  SsdLogValueStore store(Config(1ULL << 20, {128, 8192}));
  const auto value      = Value('z', 5000);
  const uint64_t handle = store.AllocAndWrite(value.data(), value.size());
  ASSERT_NE(handle, kValueHandleNone);
  const std::string info = store.ExtraResultFields();
  EXPECT_EQ(FieldOf(info, " write_log_appends="), 0U);
  EXPECT_EQ(FieldOf(info, " write_log_fallbacks="), 1U);

  std::string out(value.size(), '\0');
  ASSERT_EQ(store.Read(handle, out.data(), out.size()), value.size());
  EXPECT_EQ(out, value);
}

// The tiered e2e path wraps the log store in CLOCK, which moves rows between
// DRAM and SSD and writes demoted rows back through the same handle. A logged
// update must survive promotion, demotion, and a later overwrite.
TEST(SsdLogValueStoreTest, ClockTieringKeepsLatestValueAcrossMoves) {
  namespace fs = std::filesystem;
  const fs::path root =
      fs::temp_directory_path() /
      ("recstore-log-clock-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::remove_all(root);
  fs::create_directories(root);

  BaseKVConfig config;
  config.num_threads_ = 4;
  config.json_config_ = {
      {"value",
       {{"type", "TIERED_VALUE_STORE"},
        {"dram_allocator",
         {{"type", "CONCURRENT_SLAB_MEMORY_POOL"},
          {"capacity_bytes", 1ULL << 20},
          {"path", (root / "dram").string()}}},
        {"tiering",
         {{"cache_policy", "clock"},
          {"high_watermark_ratio", 0.05},
          {"promote_min_misses", 1},
          {"clock_async_exchange", false}}},
        {"ssd_allocator",
         {{"type", "SSD_SLAB"},
          {"capacity_bytes", 64ULL << 20},
          {"size_classes", {128}},
          {"path", (root / "ssd.db").string()},
          {"io", {{"type", "INMEMORY_LOG"}, {"queue_depth", 8}}}}},
        {"write_log",
         {{"enabled", true},
          {"path", (root / "value.log").string()},
          {"capacity_bytes", 8ULL << 20},
          {"segment_bytes", 64ULL << 10},
          {"io", {{"type", "INMEMORY_LOG"}, {"queue_depth", 8}}}}}}}};
  HybridValueStore store(config);

  constexpr size_t kRows = 4000;
  std::vector<std::string> values(kRows);
  std::vector<uint64_t> handles(kRows, kValueHandleNone);
  for (size_t i = 0; i < kRows; ++i) {
    values[i]  = std::string(128, static_cast<char>('a' + i % 26));
    handles[i] = store.AllocAndWrite(values[i].data(), values[i].size());
    ASSERT_NE(handles[i], kValueHandleNone) << "row=" << i;
  }

  auto expect_all = [&](size_t round) {
    std::vector<ValueStore::ReadResult> out;
    store.BatchRead(handles, out);
    ASSERT_EQ(out.size(), handles.size());
    for (size_t i = 0; i < kRows; ++i) {
      EXPECT_EQ(std::string(out[i].data.begin(), out[i].data.end()), values[i])
          << "round=" << round << " row=" << i;
    }
  };

  // Two passes give SSD-resident rows the misses needed for promotion.
  expect_all(1);
  expect_all(2);
  for (size_t i = 0; i < kRows; i += 3) {
    values[i] = std::string(128, static_cast<char>('A' + i % 26));
    store.Write(handles[i], values[i].data(), values[i].size());
  }
  // Further passes move rows both ways, so demotion must write back the
  // updated value instead of leaving the stale in-place copy behind.
  expect_all(3);
  expect_all(4);
  expect_all(5);

  const std::string info = store.ExtraResultFields();
  EXPECT_GT(FieldOf(info, " write_log_appends="), 0U);
  EXPECT_EQ(FieldOf(info, " write_log_fallbacks="), 0U);

  for (uint64_t handle : handles) {
    store.Free(handle);
  }
  fs::remove_all(root);
}

} // namespace
