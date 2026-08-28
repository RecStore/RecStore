#include <gtest/gtest.h>

#include <vector>

#include "base/tensor.h"

TEST(RecTensor, BorrowedDoesNotFreeExternalBuffer) {
  std::vector<float> buf = {1.0f, 2.0f, 3.0f, 4.0f};
  {
    base::RecTensor view(buf.data(), {2, 2});
    EXPECT_FALSE(view.owns());
    EXPECT_EQ(view.data(), buf.data());
    view.data_as<float>()[0] = 9.0f;
  }
  EXPECT_EQ(buf[0], 9.0f);
  EXPECT_EQ(buf[3], 4.0f);
}

TEST(RecTensor, OwnedAllocatesAndOwns) {
  base::RecTensor owned({2, 3}, base::DataType::FLOAT32);
  EXPECT_TRUE(owned.owns());
  ASSERT_NE(owned.data(), nullptr);
  EXPECT_EQ(owned.num_elements(), 6u);
  EXPECT_EQ(owned.dtype(), base::DataType::FLOAT32);
  owned.data_as<float>()[5] = 1.5f;
  EXPECT_EQ(owned.data_as<float>()[5], 1.5f);
}

TEST(RecTensor, CopyIsShallowView) {
  base::RecTensor owned({2, 2}, base::DataType::FLOAT32);
  owned.data_as<float>()[0] = 7.0f;
  base::RecTensor view      = owned;
  EXPECT_TRUE(owned.owns());
  EXPECT_FALSE(view.owns());
  EXPECT_EQ(view.data(), owned.data());
  EXPECT_EQ(view.data_as<float>()[0], 7.0f);
}

TEST(RecTensor, MoveTransfersOwnership) {
  base::RecTensor owned({2, 2}, base::DataType::FLOAT32);
  void* ptr = owned.data();
  owned.data_as<float>()[1] = 4.0f;

  base::RecTensor moved(std::move(owned));
  EXPECT_TRUE(moved.owns());
  EXPECT_EQ(moved.data(), ptr);
  EXPECT_EQ(moved.data_as<float>()[1], 4.0f);
  EXPECT_FALSE(owned.owns());
  EXPECT_EQ(owned.data(), nullptr);
}
