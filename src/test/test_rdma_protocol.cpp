#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "ps/rdma/rdma_protocol.h"

namespace petps {

TEST(RdmaProtocolTest, EncodesAndDecodesFixedWidthPutPayload) {
  std::vector<uint64_t> keys             = {11, 12};
  std::vector<std::vector<float>> values = {
      {1.0f, 2.0f, 3.0f, 4.0f},
      {5.0f, 6.0f, 7.0f, 8.0f},
  };

  std::string payload = EncodePutPayload(keys, values);

  DecodedPutPayload decoded;
  std::string error;
  ASSERT_TRUE(DecodePutPayload(payload, &decoded, &error)) << error;
  EXPECT_EQ(decoded.embedding_dim, 4u);
  EXPECT_EQ(decoded.keys, keys);
  EXPECT_EQ(decoded.values.size(), 8u);
  EXPECT_FLOAT_EQ(decoded.values[0], 1.0f);
  EXPECT_FLOAT_EQ(decoded.values[7], 8.0f);
}

TEST(RdmaProtocolTest, RejectsTruncatedPayload) {
  std::string payload(sizeof(PutPayloadHeader) - 1, '\0');
  DecodedPutPayload decoded;
  std::string error;
  EXPECT_FALSE(DecodePutPayload(payload, &decoded, &error));
  EXPECT_NE(error.find("header"), std::string::npos);
}

TEST(RdmaProtocolTest, ComputesFixedSlotResponseBytes) {
  EXPECT_EQ(FixedSlotResponseBytes(3, 16), 3u * 16u + sizeof(std::int32_t));
}

} // namespace petps
