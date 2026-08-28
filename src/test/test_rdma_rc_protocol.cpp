#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "base/tensor.h"
#include "ps/rdma/rdma_protocol.h"

namespace {

TEST(RdmaRcProtocolTest, ComputesGetKeysPerRpcFor512BValue) {
  EXPECT_EQ(petps::GetKeysPerRpcByResponseBudget(512, 4096, 200),
            static_cast<std::size_t>(1600));
}

TEST(RdmaRcProtocolTest, DescriptorAndStatusAreCachelineAligned) {
  EXPECT_EQ(sizeof(petps::RequestDescriptor), static_cast<std::size_t>(192));
  EXPECT_EQ(alignof(petps::RequestDescriptor), static_cast<std::size_t>(64));
  EXPECT_EQ(alignof(petps::CommitWord), static_cast<std::size_t>(64));
  EXPECT_EQ(alignof(petps::StatusWord), static_cast<std::size_t>(64));
}

TEST(RdmaRcProtocolTest, PutPayloadRoundTripBuildsValidReader) {
  std::vector<std::uint64_t> keys = {10, 20};
  std::vector<float> value_flat   = {1.0f, 2.0f, 3.0f, 4.0f};
  base::RecTensor values(value_flat.data(), {2, 2});
  std::string payload;
  std::string error;
  const std::size_t bytes =
      petps::PutPayloadBytes(keys, values, &payload, &error);
  ASSERT_GT(bytes, 0u) << error;
  const auto* reader =
      reinterpret_cast<const ParameterCompressReader*>(payload.data());
  ASSERT_TRUE(reader->Valid(static_cast<int>(payload.size())));
  ASSERT_EQ(reader->item_size(), 2);
  EXPECT_EQ(reader->item(0)->key, 10u);
  EXPECT_EQ(reader->item(1)->key, 20u);
  EXPECT_EQ(reader->item(0)->dim, 2);
  EXPECT_FLOAT_EQ(reader->item(1)->data()[1], 4.0f);
}

TEST(RdmaRcProtocolTest, PutPayloadAcceptsConstArraySlice) {
  std::vector<std::uint64_t> keys = {1, 10, 20, 99};
  std::vector<float> value_flat   = {
      0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f};
  const base::ConstArray<std::uint64_t> key_slice =
      base::ConstArray<std::uint64_t>(keys).SubArray(1, 3);
  base::RecTensor values(value_flat.data() + 2, {2, 2});
  std::string payload;
  std::string error;
  const std::size_t bytes =
      petps::PutPayloadBytes(key_slice, values, &payload, &error);
  ASSERT_GT(bytes, 0u) << error;
  EXPECT_EQ(key_slice.Data(), keys.data() + 1);
  const auto* reader =
      reinterpret_cast<const ParameterCompressReader*>(payload.data());
  ASSERT_TRUE(reader->Valid(static_cast<int>(payload.size())));
  ASSERT_EQ(reader->item_size(), 2);
  EXPECT_EQ(reader->item(0)->key, 10u);
  EXPECT_EQ(reader->item(1)->key, 20u);
}

TEST(RdmaRcProtocolTest, FlatUpdatePayloadMatchesRowPayload) {
  std::vector<std::uint64_t> keys = {10, 20};
  std::vector<float> value_flat = {1.0f, 2.0f, 3.0f, 4.0f};
  base::RecTensor values(value_flat.data(), {2, 2});
  const std::vector<float> flat_values   = {1.0f, 2.0f, 3.0f, 4.0f};
  std::string row_payload;
  std::string flat_payload;
  std::string error;

  ASSERT_GT(petps::UpdatePayloadBytes(
                base::ConstArray<std::uint64_t>(keys), values,
                &row_payload, &error),
            0u)
      << error;
  ASSERT_GT(petps::UpdatePayloadBytesFlat(
                base::ConstArray<std::uint64_t>(keys),
                flat_values.data(),
                2,
                &flat_payload,
                &error),
            0u)
      << error;

  // Flat and row payloads use different serialization formats (flat stores
  // keys + values contiguously; row uses ParameterCompressor with per-row
  // dim fields). Verify each independently instead of requiring binary equality.
  const auto* flat_keys =
      reinterpret_cast<const std::uint64_t*>(flat_payload.data());
  const auto* flat_values_ptr = reinterpret_cast<const float*>(
      flat_payload.data() + keys.size() * sizeof(std::uint64_t));
  EXPECT_EQ(flat_keys[0], 10u);
  EXPECT_EQ(flat_keys[1], 20u);
  EXPECT_FLOAT_EQ(flat_values_ptr[0], 1.0f);
  EXPECT_FLOAT_EQ(flat_values_ptr[1], 2.0f);
  EXPECT_FLOAT_EQ(flat_values_ptr[2], 3.0f);
  EXPECT_FLOAT_EQ(flat_values_ptr[3], 4.0f);

  const auto* reader =
      reinterpret_cast<const ParameterCompressReader*>(row_payload.data());
  ASSERT_TRUE(reader->Valid(static_cast<int>(row_payload.size())));
  ASSERT_EQ(reader->item_size(), 2);
  EXPECT_EQ(reader->item(0)->key, 10u);
  EXPECT_EQ(reader->item(1)->key, 20u);
  EXPECT_FLOAT_EQ(reader->item(0)->data()[0], 1.0f);
  EXPECT_FLOAT_EQ(reader->item(0)->data()[1], 2.0f);
  EXPECT_FLOAT_EQ(reader->item(1)->data()[0], 3.0f);
  EXPECT_FLOAT_EQ(reader->item(1)->data()[1], 4.0f);
}

TEST(RdmaRcProtocolTest, FlatUpdatePayloadStoresContiguousKeysAndValues) {
  std::vector<std::uint64_t> keys      = {10, 20};
  const std::vector<float> flat_values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::string flat_payload;
  std::string error;

  ASSERT_GT(petps::UpdatePayloadBytesFlat(
                base::ConstArray<std::uint64_t>(keys),
                flat_values.data(),
                2,
                &flat_payload,
                &error),
            0u)
      << error;
  ASSERT_EQ(flat_payload.size(), petps::FlatUpdatePayloadBytes(2, 2));
  const auto* payload_keys =
      reinterpret_cast<const std::uint64_t*>(flat_payload.data());
  const auto* payload_values = reinterpret_cast<const float*>(
      flat_payload.data() + keys.size() * sizeof(std::uint64_t));
  EXPECT_EQ(std::vector<std::uint64_t>(payload_keys, payload_keys + keys.size()), keys);
  EXPECT_EQ(
      std::vector<float>(payload_values, payload_values + flat_values.size()),
      flat_values);
}

TEST(RdmaRcProtocolTest, FlatUpdatePayloadRejectsOverflow) {
  const std::size_t max_size = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(petps::FlatUpdatePayloadBytes(1, max_size), 0u);
  EXPECT_EQ(petps::FlatUpdatePayloadBytes(max_size, 1), 0u);
}

TEST(RdmaRcProtocolTest, FlatUpdateGatherPacksSelectedRowsInOrder) {
  const std::vector<std::uint64_t> keys = {10, 20, 30};
  const std::vector<float> values = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<std::size_t> rows = {2, 0};
  std::vector<char> payload(petps::FlatUpdatePayloadBytes(rows.size(), 2));
  std::string error;

  ASSERT_EQ(
      petps::PackFlatUpdatePayloadGather(
          keys.data(),
          values.data(),
          keys.size(),
          2,
          rows.data(),
          rows.size(),
          payload.data(),
          payload.size(),
          &error),
      payload.size())
      << error;
  const auto* payload_keys =
      reinterpret_cast<const std::uint64_t*>(payload.data());
  const auto* payload_values = reinterpret_cast<const float*>(
      payload.data() + rows.size() * sizeof(std::uint64_t));
  EXPECT_EQ(std::vector<std::uint64_t>(payload_keys, payload_keys + 2),
            (std::vector<std::uint64_t>{30, 10}));
  EXPECT_EQ(std::vector<float>(payload_values, payload_values + 4),
            (std::vector<float>{5.0f, 6.0f, 1.0f, 2.0f}));
}

TEST(RdmaRcProtocolTest, FlatUpdateGatherRejectsOutOfRangeRows) {
  const std::vector<std::uint64_t> keys = {10};
  const std::vector<float> values = {1.0f, 2.0f};
  const std::size_t row = 1;
  std::vector<char> payload(petps::FlatUpdatePayloadBytes(1, 2));
  std::string error;

  EXPECT_EQ(
      petps::PackFlatUpdatePayloadGather(
          keys.data(),
          values.data(),
          keys.size(),
          2,
          &row,
          1,
          payload.data(),
          payload.size(),
          &error),
      0u);
  EXPECT_EQ(error, "flat update gather row index is out of range");
}

TEST(RdmaRcProtocolTest, StatusWordDoneRequiresMatchingSeq) {
  petps::StatusWord status;
  petps::ResetStatusWord(&status, 7);
  EXPECT_FALSE(petps::StatusWordDone(status, 7));
  status.seq.store(6, std::memory_order_release);
  status.state.store(petps::kRcSlotDone, std::memory_order_release);
  EXPECT_FALSE(petps::StatusWordDone(status, 7));
  status.seq.store(7, std::memory_order_release);
  EXPECT_TRUE(petps::StatusWordDone(status, 7));
}

} // namespace
