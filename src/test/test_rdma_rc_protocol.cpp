#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

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
  std::vector<std::uint64_t> keys        = {10, 20};
  std::vector<std::vector<float>> values = {{1.0f, 2.0f}, {3.0f, 4.0f}};
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

TEST(RdmaRcProtocolTest, FlatUpdatePayloadMatchesRowPayload) {
  std::vector<std::uint64_t> keys        = {10, 20};
  std::vector<std::vector<float>> values = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  const std::vector<float> flat_values   = {1.0f, 2.0f, 3.0f, 4.0f};
  std::string row_payload;
  std::string flat_payload;
  std::string error;

  ASSERT_GT(petps::UpdatePayloadBytes(
                keys, values, &row_payload, &error),
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
  EXPECT_EQ(flat_payload, row_payload);
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
