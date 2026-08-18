#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <unistd.h>
#include <vector>
#include <sys/mman.h>

#include "base/json.h"
#include "base/tensor.h"
#include "framework/common/local_shm_op_component.h"
#include "ps/base/base_client.h"
#include "ps/local_shm/local_shm_client.h"
#define private public
#include "ps/local_shm/local_shm_server.h"
#undef private

namespace {

std::string NormalizeRegionNameForShm(const std::string& region_name) {
  if (!region_name.empty() && region_name.front() == '/') {
    return region_name;
  }
  return "/" + region_name;
}

void RemoveRegionIfExists(const std::string& region_name) {
  ::shm_unlink(NormalizeRegionNameForShm(region_name).c_str());
}

std::string MakeUniqueRegionName(const std::string& prefix) {
  static std::atomic<uint64_t> counter{0};
  return prefix + "_" + std::to_string(::getpid()) + "_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()) +
         "_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

class LocalShmOpComponentTest : public ::testing::Test {
protected:
  json MakeLocalShmConfig(const std::string& region_name) {
    static std::atomic<uint64_t> storage_counter{0};
    const std::string storage_path =
        "/dev/shm/recstore_lshm_comp_" + std::to_string(::getpid()) + "_" +
        std::to_string(storage_counter.fetch_add(1, std::memory_order_relaxed));
    cleanup_paths_.push_back(storage_path);
    cleanup_region_names_.push_back(region_name);
    std::error_code ec;
    std::filesystem::remove_all(storage_path, ec);
    RemoveRegionIfExists(region_name);
    return {
        {"cache_ps",
         {{"num_threads", 1},
          {"ps_type", "LOCAL_SHM"},
          {"base_kv_config",
           {{"capacity", 1024},
            {"index", {{"type", "DRAM_EXTENDIBLE_HASH"}}},
            {"value",
             {{"type", "DRAM_VALUE_STORE"},
              {"path", storage_path + "/value"},
              {"default_value_size_hint", 16},
              {"dram_allocator",
               {{"type", "PERSIST_LOOP_SLAB"},
                {"capacity_bytes", 1024 * 16}}}}}}}}},
        {"local_shm",
         {{"region_name", region_name},
          {"local_rank", 0},
          {"slot_count", 8},
          {"ready_queue_count", 1},
          {"ready_queue_burst_limit", 8},
          {"slot_buffer_bytes", 1 << 20},
          {"client_timeout_ms", 1000}}},
    };
  }

  void TearDown() override {
    std::error_code ec;
    for (const auto& path : cleanup_paths_) {
      std::filesystem::remove_all(path, ec);
    }
    for (const auto& region_name : cleanup_region_names_) {
      RemoveRegionIfExists(region_name);
    }
  }

private:
  std::vector<std::string> cleanup_paths_;
  std::vector<std::string> cleanup_region_names_;
};

TEST_F(LocalShmOpComponentTest, StoragePathIsScopedToRegionName) {
  const auto config_a = MakeLocalShmConfig("recstore_local_shm_component_a");
  const auto config_b = MakeLocalShmConfig("recstore_local_shm_component_b");

  EXPECT_NE(config_a["cache_ps"]["base_kv_config"]["value"]["path"],
            config_b["cache_ps"]["base_kv_config"]["value"]["path"]);
}

TEST_F(LocalShmOpComponentTest, LookupFlatReadsValuesFromLocalShmClient) {
  const auto config =
      MakeLocalShmConfig(MakeUniqueRegionName("recstore_local_shm_component"));
  recstore::LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  recstore::LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_component", {128, 4}), 0);

  std::vector<uint64_t> key_storage = {3, 8};
  std::vector<float> value_flat     = {
      3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  base::RecTensor value_storage(value_flat.data(), {2, 4});
  base::ConstArray<uint64_t> key_array(key_storage);
  ASSERT_EQ(client.PutParameter(key_array, value_storage), 0);

  std::vector<float> output_storage(8, 0.0f);
  base::RecTensor key_tensor(key_storage.data(), {2});
  base::RecTensor value_tensor(output_storage.data(), {2, 4});

  recstore::LocalShmLookupFlat(&client, "local_shm", key_tensor, value_tensor);

  EXPECT_FLOAT_EQ(output_storage[0], 3.0f);
  EXPECT_FLOAT_EQ(output_storage[1], 4.0f);
  EXPECT_FLOAT_EQ(output_storage[4], 8.0f);
  EXPECT_FLOAT_EQ(output_storage[7], 11.0f);

  server.Stop();
  server_thread.join();
}

} // namespace
