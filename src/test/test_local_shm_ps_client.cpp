#include <gtest/gtest.h>

#include <cstdlib>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/mman.h>

#include "base/json.h"
#include "base/tensor.h"
#define private public
#include "ps/local_shm/local_shm_client.h"
#undef private
#include "ps/local_shm/local_shm_queue.h"
#define private public
#include "ps/local_shm/local_shm_server.h"
#undef private

namespace recstore {
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

class LocalShmPSClientTest : public ::testing::Test {
protected:
  json MakeLocalShmConfig(const std::string& region_name,
                          uint32_t ready_queue_count       = 1,
                          uint32_t ready_queue_burst_limit = 8) {
    static std::atomic<uint64_t> storage_counter{0};
    const std::string storage_path =
        "/dev/shm/recstore_lshm_" + std::to_string(::getpid()) + "_" +
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
          {"ready_queue_count", ready_queue_count},
          {"ready_queue_burst_limit", ready_queue_burst_limit},
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

TEST_F(LocalShmPSClientTest, StoragePathIsScopedToRegionName) {
  const auto config_a = MakeLocalShmConfig("recstore_local_shm_ps_client_a");
  const auto config_b = MakeLocalShmConfig("recstore_local_shm_ps_client_b");

  EXPECT_NE(config_a["cache_ps"]["base_kv_config"]["value"]["path"],
            config_b["cache_ps"]["base_kv_config"]["value"]["path"]);
}

TEST_F(LocalShmPSClientTest, FactoryClientTypeCanBeConstructed) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_factory"));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  EXPECT_EQ(client.InitEmbeddingTable("table_a", {128, 4}), 0);

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest, InitCreatesOneRuntimePerReadyQueue) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_multi_queue"),
      /*ready_queue_count=*/2);
  LocalShmParameterServer server;

  server.Init(config);

  ASSERT_EQ(server.runtimes_.size(), 2u);
  ASSERT_NE(server.runtimes_[0], nullptr);
  ASSERT_NE(server.runtimes_[1], nullptr);
  EXPECT_EQ(server.runtimes_[0]->ready_queue_id_, 0u);
  EXPECT_EQ(server.runtimes_[1]->ready_queue_id_, 1u);
}

TEST_F(LocalShmPSClientTest, ThreadReadyQueueShardingIsOptIn) {
  auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_thread_queues"),
      /*ready_queue_count=*/4);
  LocalShmParameterServer server;
  server.Init(config);

  LocalShmPSClient fixed_client(config["local_shm"]);
  const uint32_t fixed_queue = fixed_client.CurrentReadyQueueId();
  ASSERT_LT(fixed_queue, 4u);

  std::vector<uint32_t> fixed_observed(8, 0);
  std::vector<std::thread> fixed_threads;
  for (std::size_t i = 0; i < fixed_observed.size(); ++i) {
    fixed_threads.emplace_back([&fixed_client, &fixed_observed, i]() {
      fixed_observed[i] = fixed_client.CurrentReadyQueueId();
    });
  }
  for (auto& thread : fixed_threads) {
    thread.join();
  }
  for (const auto queue_id : fixed_observed) {
    EXPECT_EQ(queue_id, fixed_queue);
  }

  config["local_shm"]["thread_ready_queue_sharding"] = true;
  LocalShmPSClient sharded_client(config["local_shm"]);
  std::vector<uint32_t> sharded_observed(8, 0);
  std::vector<std::thread> sharded_threads;
  for (std::size_t i = 0; i < sharded_observed.size(); ++i) {
    sharded_threads.emplace_back([&sharded_client, &sharded_observed, i]() {
      sharded_observed[i] = sharded_client.CurrentReadyQueueId();
    });
  }
  for (auto& thread : sharded_threads) {
    thread.join();
  }
  for (const auto queue_id : sharded_observed) {
    EXPECT_LT(queue_id, 4u);
  }
  std::unordered_set<uint32_t> sharded_unique(
      sharded_observed.begin(), sharded_observed.end());
  EXPECT_GT(sharded_unique.size(), 1u);
}

TEST_F(LocalShmPSClientTest, LastRequestProfileCapturesTransportStages) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_profile"));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_profile", {128, 4}), 0);

  std::vector<uint64_t> keys = {1, 3};
  std::vector<float> value_flat = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  base::RecTensor values(value_flat.data(), {2, 4});
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  std::vector<float> readback(8, 0.0f);
  base::RecTensor read_t(readback.data(), {2, 4});
  ASSERT_EQ(client.GetParameter(key_array, read_t), 0);

  const auto profile = client.GetLastRequestProfile();
  EXPECT_EQ(profile.opcode, static_cast<uint32_t>(LocalOpcode::kGet));
  EXPECT_GT(profile.request_total_us, 0.0);
  EXPECT_GE(profile.acquire_slot_us, 0.0);
  EXPECT_GE(profile.enqueue_us, 0.0);
  EXPECT_GE(profile.wait_us, 0.0);
  EXPECT_GE(profile.release_us, 0.0);
  EXPECT_GE(profile.server_queue_wait_us, 0.0);
  EXPECT_GE(profile.server_backend_us, 0.0);

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest, LastRequestProfileCapturesOpcodePerRequestType) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_profile_opcode"));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_profile_opcode", {128, 4}), 0);
  EXPECT_EQ(client.GetLastRequestProfile().opcode,
            static_cast<uint32_t>(LocalOpcode::kInitTable));

  std::vector<uint64_t> keys = {7, 9};
  std::vector<float> value_flat = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  base::RecTensor values(value_flat.data(), {2, 4});
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);
  EXPECT_EQ(client.GetLastRequestProfile().opcode,
            static_cast<uint32_t>(LocalOpcode::kPut));

  std::vector<float> readback(8, 0.0f);
  ASSERT_EQ(client.GetParameterFlat(key_array, readback.data(), 2, 4), 0);
  EXPECT_EQ(client.GetLastRequestProfile().opcode,
            static_cast<uint32_t>(LocalOpcode::kGet));

  std::vector<float> grads = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  ASSERT_EQ(client.UpdateParameterFlat(
                "table_profile_opcode", key_array, grads.data(), 2, 4),
            0);
  EXPECT_EQ(client.GetLastRequestProfile().opcode,
            static_cast<uint32_t>(LocalOpcode::kUpdateFlat));

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest, ReleaseSlotDoesNotUseGlobalFreeDoorbell) {
  auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_free_queue_wait"),
      /*ready_queue_count=*/2);
  config["local_shm"]["thread_ready_queue_sharding"] = true;
  LocalShmParameterServer server;
  server.Init(config);
  LocalShmPSClient client(config["local_shm"]);

  ASSERT_TRUE(client.region_.IsOpen());
  auto* control = client.region_.control();
  ASSERT_NE(control, nullptr);
  EXPECT_EQ(control->free_doorbell.load(std::memory_order_acquire), 0u);

  const int slot = client.AcquireSlot();
  ASSERT_GE(slot, 0);
  client.ReleaseSlot(static_cast<uint32_t>(slot));

  EXPECT_EQ(control->free_doorbell.load(std::memory_order_acquire), 0u);
}

TEST_F(LocalShmPSClientTest, PutGetAndUpdateFlatRoundTrip) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_rw"));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_b", {128, 4}), 0);

  std::vector<uint64_t> keys = {1, 5};
  std::vector<float> value_flat = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  base::RecTensor values(value_flat.data(), {2, 4});
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  std::vector<float> readback(8, 0.0f);
  base::RecTensor read_t(readback.data(), {2, 4});
  ASSERT_EQ(client.GetParameter(key_array, read_t), 0);
  EXPECT_EQ(readback[0], 1.0f);
  EXPECT_EQ(readback[1], 2.0f);
  EXPECT_EQ(readback[4], 5.0f);
  EXPECT_EQ(readback[7], 8.0f);

  std::vector<float> grads = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  ASSERT_EQ(
      client.UpdateParameterFlat("table_b", key_array, grads.data(), 2, 4), 0);
  ASSERT_EQ(client.GetParameter(key_array, read_t), 0);
  EXPECT_FLOAT_EQ(readback[0], 0.99f);
  EXPECT_FLOAT_EQ(readback[1], 1.99f);
  EXPECT_FLOAT_EQ(readback[4], 4.98f);
  EXPECT_FLOAT_EQ(readback[7], 7.98f);

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest, UpdateParameterFlatInitializesMissingRows) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_update_missing"));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_update_missing", {128, 4}), 0);

  std::vector<uint64_t> keys = {17};
  base::ConstArray<uint64_t> key_array(keys);
  std::vector<float> grads = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_EQ(client.UpdateParameterFlat(
                "table_update_missing", key_array, grads.data(), 1, 4),
            0);

  std::vector<float> readback(4, 0.0f);
  ASSERT_EQ(client.GetParameterFlat(key_array, readback.data(), 1, 4), 0);
  EXPECT_FLOAT_EQ(readback[0], -0.01f);
  EXPECT_FLOAT_EQ(readback[1], -0.02f);
  EXPECT_FLOAT_EQ(readback[2], -0.03f);
  EXPECT_FLOAT_EQ(readback[3], -0.04f);

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest, GetParameterFlatRoundTripUsesFixedEmbeddingDim) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_flat_get"));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_flat", {128, 4}), 0);

  std::vector<uint64_t> keys = {2, 6};
  std::vector<float> value_flat = {
      2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  base::RecTensor values(value_flat.data(), {2, 4});
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  std::vector<float> readback(8, 0.0f);
  ASSERT_EQ(client.GetParameterFlat(key_array, readback.data(), 2, 4), 0);
  EXPECT_EQ(readback[0], 2.0f);
  EXPECT_EQ(readback[1], 3.0f);
  EXPECT_EQ(readback[4], 6.0f);
  EXPECT_EQ(readback[7], 9.0f);

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest, GetParameterFlatZerosMissingRowsOnly) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_flat_partial_hit"));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_partial_hit", {128, 4}), 0);

  std::vector<uint64_t> put_keys = {2};
  std::vector<float> put_flat    = {2.0f, 3.0f, 4.0f, 5.0f};
  base::RecTensor put_values(put_flat.data(), {1, 4});
  base::ConstArray<uint64_t> put_key_array(put_keys);
  ASSERT_EQ(client.PutParameter(put_key_array, put_values), 0);

  std::vector<uint64_t> get_keys = {2, 999};
  base::ConstArray<uint64_t> get_key_array(get_keys);
  std::vector<float> readback(8, -1.0f);
  ASSERT_EQ(client.GetParameterFlat(get_key_array, readback.data(), 2, 4), 0);

  EXPECT_FLOAT_EQ(readback[0], 2.0f);
  EXPECT_FLOAT_EQ(readback[1], 3.0f);
  EXPECT_FLOAT_EQ(readback[2], 4.0f);
  EXPECT_FLOAT_EQ(readback[3], 5.0f);
  EXPECT_FLOAT_EQ(readback[4], 0.0f);
  EXPECT_FLOAT_EQ(readback[5], 0.0f);
  EXPECT_FLOAT_EQ(readback[6], 0.0f);
  EXPECT_FLOAT_EQ(readback[7], 0.0f);

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest,
       SubmitWaitAndReleaseGetParameterFlatExposesSlotPayload) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_flat_handle"));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_slot", {128, 4}), 0);

  std::vector<uint64_t> keys = {9, 13};
  std::vector<float> value_flat = {
      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  base::RecTensor values(value_flat.data(), {2, 4});
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  LocalShmFlatGetHandle handle;
  ASSERT_EQ(client.SubmitGetParameterFlat(key_array, 2, 4, &handle), 0);
  ASSERT_NE(handle.slot_id, std::numeric_limits<uint32_t>::max());
  ASSERT_EQ(client.WaitGetParameterFlat(&handle), 0);
  ASSERT_NE(handle.values, nullptr);
  const void* client_payload_base  = nullptr;
  std::size_t client_payload_bytes = 0;
  ASSERT_TRUE(
      client.GetSlotPayloadRegion(&client_payload_base, &client_payload_bytes));
  auto* client_slot_payload =
      static_cast<const uint8_t*>(client_payload_base) +
      AlignUp(config["local_shm"]["slot_buffer_bytes"].get<uint32_t>()) *
          handle.slot_id;
  EXPECT_EQ(handle.values,
            reinterpret_cast<const float*>(
                client_slot_payload + sizeof(uint64_t) * keys.size()));
  ASSERT_EQ(handle.num_rows, 2);
  ASSERT_EQ(handle.embedding_dim, 4);
  ASSERT_EQ(handle.output_bytes, sizeof(float) * 8);

  EXPECT_FLOAT_EQ(handle.values[0], 9.0f);
  EXPECT_FLOAT_EQ(handle.values[1], 10.0f);
  EXPECT_FLOAT_EQ(handle.values[4], 13.0f);
  EXPECT_FLOAT_EQ(handle.values[7], 16.0f);

  client.ReleaseGetParameterFlat(&handle);
  EXPECT_EQ(handle.values, nullptr);
  EXPECT_EQ(handle.num_rows, 0);
  EXPECT_EQ(handle.embedding_dim, 0);
  EXPECT_EQ(handle.output_bytes, 0);

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest, GetParameterFlatRejectsMismatchedEmbeddingDim) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_flat_dim_mismatch"));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_mismatch", {128, 4}), 0);

  std::vector<uint64_t> keys = {7};
  std::vector<float> value_flat = {7.0f, 8.0f, 9.0f, 10.0f};
  base::RecTensor values(value_flat.data(), {1, 4});
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  std::vector<float> readback(3, 0.0f);
  EXPECT_EQ(client.GetParameterFlat(key_array, readback.data(), 1, 3), -1);

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest, UpdateParameterFlatRejectsMismatchedEmbeddingDim) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_update_dim_mismatch"));
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  LocalShmPSClient client(config["local_shm"]);
  ASSERT_EQ(client.InitEmbeddingTable("table_update_mismatch", {128, 4}), 0);

  std::vector<uint64_t> keys = {11};
  std::vector<float> value_flat = {1.0f, 2.0f, 3.0f, 4.0f};
  base::RecTensor values(value_flat.data(), {1, 4});
  base::ConstArray<uint64_t> key_array(keys);
  ASSERT_EQ(client.PutParameter(key_array, values), 0);

  std::vector<float> grads = {0.1f, 0.2f, 0.3f};
  EXPECT_EQ(client.UpdateParameterFlat(
                "table_update_mismatch", key_array, grads.data(), 1, 3),
            -1);

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest, MultiClientUsesIndependentReadyQueues) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_multi"), 2);
  LocalShmParameterServer server;
  server.Init(config);

  std::thread server_thread([&]() { server.Run(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  json client0_config                 = config["local_shm"];
  json client1_config                 = config["local_shm"];
  client0_config["ready_queue_index"] = 0;
  client1_config["ready_queue_index"] = 1;

  LocalShmPSClient client0(client0_config);
  LocalShmPSClient client1(client1_config);
  ASSERT_EQ(client0.InitEmbeddingTable("table_c", {128, 4}), 0);

  std::thread worker0([&]() {
    std::vector<uint64_t> keys = {1, 3};
    std::vector<float> value_flat = {
        1.0f, 1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    base::RecTensor values(value_flat.data(), {2, 4});
    base::ConstArray<uint64_t> key_array(keys);
    EXPECT_EQ(client0.PutParameter(key_array, values), 0);
    std::vector<float> readback(8, 0.0f);
    base::RecTensor read_t(readback.data(), {2, 4});
    EXPECT_EQ(client0.GetParameter(key_array, read_t), 0);
    EXPECT_FLOAT_EQ(readback[0], 1.0f);
    EXPECT_FLOAT_EQ(readback[4], 3.0f);
  });

  std::thread worker1([&]() {
    std::vector<uint64_t> keys = {2, 4};
    std::vector<float> value_flat = {
        2.0f, 2.0f, 2.0f, 2.0f, 4.0f, 4.0f, 4.0f, 4.0f};
    base::RecTensor values(value_flat.data(), {2, 4});
    base::ConstArray<uint64_t> key_array(keys);
    EXPECT_EQ(client1.PutParameter(key_array, values), 0);
    std::vector<float> readback(8, 0.0f);
    base::RecTensor read_t(readback.data(), {2, 4});
    EXPECT_EQ(client1.GetParameter(key_array, read_t), 0);
    EXPECT_FLOAT_EQ(readback[0], 2.0f);
    EXPECT_FLOAT_EQ(readback[4], 4.0f);
  });

  worker0.join();
  worker1.join();

  server.Stop();
  server_thread.join();
}

TEST_F(LocalShmPSClientTest, LocalRankEnvironmentSelectsReadyQueue) {
  auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_ps_client_env"), 2);
  config["local_shm"].erase("local_rank");
  LocalShmRegion region;
  ASSERT_TRUE(region.Create(
      config["local_shm"]["region_name"].get<std::string>(), 8, 1 << 20, 2));

  ASSERT_EQ(::setenv("LOCAL_RANK", "1", 1), 0);
  LocalShmPSClient client(config["local_shm"]);

  std::vector<uint64_t> keys;
  base::ConstArray<uint64_t> key_array(keys);
  LocalShmFlatGetHandle handle;
  ASSERT_EQ(client.SubmitGetParameterFlat(key_array, 0, 4, &handle), 0);
  ASSERT_NE(handle.slot_id, LocalShmFlatGetHandle::kInvalidSlotId);

  uint32_t slot_id = 0;
  EXPECT_FALSE(LocalShmQueueDequeue(
      region.ready_queue_header(0), region.ready_queue_cells(0), &slot_id));
  ASSERT_TRUE(LocalShmQueueDequeue(
      region.ready_queue_header(1), region.ready_queue_cells(1), &slot_id));
  EXPECT_EQ(slot_id, handle.slot_id);

  client.ReleaseGetParameterFlat(&handle);
  ::unsetenv("LOCAL_RANK");
}

TEST_F(LocalShmPSClientTest, ServerDrainReadyQueueHonorsBurstLimit) {
  const auto config = MakeLocalShmConfig(
      MakeUniqueRegionName("recstore_local_shm_server_burst"), 2, 2);
  LocalShmRegion region;
  ASSERT_TRUE(region.Create(
      config["local_shm"]["region_name"].get<std::string>(), 8, 1 << 20, 2));

  for (uint32_t slot_id = 0; slot_id < 3; ++slot_id) {
    auto* header   = region.slot_header(slot_id);
    header->opcode = static_cast<uint32_t>(LocalOpcode::kInvalid);
    header->state.store(static_cast<uint32_t>(LocalSlotState::kReady),
                        std::memory_order_release);
    ASSERT_TRUE(LocalShmQueueEnqueue(
        region.ready_queue_header(0), region.ready_queue_cells(0), slot_id));
  }

  LocalShmStoreRuntime runtime(
      &region,
      nullptr,
      /*ready_queue_id=*/0,
      /*worker_tid=*/0,
      /*ready_queue_burst_limit=*/2);
  uint32_t processed = 0;
  EXPECT_TRUE(runtime.DrainReadyQueue(0, &processed));
  EXPECT_EQ(processed, 2U);

  EXPECT_EQ(region.slot_header(0)->state.load(std::memory_order_acquire),
            static_cast<uint32_t>(LocalSlotState::kError));
  EXPECT_EQ(region.slot_header(1)->state.load(std::memory_order_acquire),
            static_cast<uint32_t>(LocalSlotState::kError));
  EXPECT_EQ(region.slot_header(2)->state.load(std::memory_order_acquire),
            static_cast<uint32_t>(LocalSlotState::kReady));

  uint32_t remaining_slot_id = 0;
  EXPECT_TRUE(LocalShmQueueDequeue(
      region.ready_queue_header(0),
      region.ready_queue_cells(0),
      &remaining_slot_id));
  EXPECT_EQ(remaining_slot_id, 2U);
}

} // namespace
} // namespace recstore
