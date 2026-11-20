#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ps/base/parameters.h"

namespace {

std::string
CompressRows(const std::vector<std::pair<uint64_t, std::vector<float>>>& rows) {
  ParameterCompressor compressor;
  float dummy = 0.0f;
  for (const auto& row : rows) {
    const int dim = static_cast<int>(row.second.size());
    ParameterPack pack(row.first, dim, dim == 0 ? &dummy : row.second.data());
    compressor.AddItem(pack, nullptr);
  }
  std::string block;
  compressor.ToBlock(&block);
  return block;
}

const ParameterCompressReader* ReaderOf(const std::string& block) {
  return reinterpret_cast<const ParameterCompressReader*>(block.data());
}

} // namespace

TEST(ParameterCompressTest, FlatCopyMarksMissingRows) {
  const std::string block =
      CompressRows({{7, {1.0f, 2.0f}}, {8, {}}, {9, {3.0f, 4.0f}}});
  const auto* reader = ReaderOf(block);
  ASSERT_TRUE(reader->Valid(static_cast<int>(block.size())));
  ASSERT_EQ(reader->item_size(), 3);

  std::vector<float> values(6, -1.0f);
  std::vector<uint8_t> found(3, 0xff);

  CopyCompressItemsToFlat(reader, values.data(), 2, 0, found.data());

  EXPECT_EQ(found, (std::vector<uint8_t>{1, 0, 1}));
  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_FLOAT_EQ(values[1], 2.0f);
  EXPECT_FLOAT_EQ(values[2], 0.0f);
  EXPECT_FLOAT_EQ(values[3], 0.0f);
  EXPECT_FLOAT_EQ(values[4], 3.0f);
  EXPECT_FLOAT_EQ(values[5], 4.0f);
}

TEST(ParameterCompressTest, FlatCopyWithoutMaskKeepsLegacyValues) {
  const std::string block =
      CompressRows({{7, {1.0f, 2.0f}}, {8, {}}, {9, {3.0f, 4.0f}}});
  const auto* reader = ReaderOf(block);

  std::vector<float> values(6, -1.0f);
  CopyCompressItemsToFlat(reader, values.data(), 2, 0, nullptr);

  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_FLOAT_EQ(values[2], 0.0f);
  EXPECT_FLOAT_EQ(values[4], 3.0f);
}

TEST(ParameterCompressTest, FlatCopyWritesMaskAtRowOffset) {
  const std::string block = CompressRows({{7, {1.0f, 2.0f}}, {8, {}}});
  const auto* reader      = ReaderOf(block);

  std::vector<float> values(8, -1.0f);
  std::vector<uint8_t> found(4, 0xff);

  CopyCompressItemsToFlat(reader, values.data(), 2, 2, found.data());

  EXPECT_EQ(found, (std::vector<uint8_t>{0xff, 0xff, 1, 0}));
  EXPECT_FLOAT_EQ(values[0], -1.0f);
  EXPECT_FLOAT_EQ(values[4], 1.0f);
  EXPECT_FLOAT_EQ(values[5], 2.0f);
  EXPECT_FLOAT_EQ(values[6], 0.0f);
  EXPECT_FLOAT_EQ(values[7], 0.0f);
}
