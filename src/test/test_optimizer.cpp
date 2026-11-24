#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "optimizer/optimizer.h"

namespace {

class InMemoryKV final : public BaseKV {
public:
  InMemoryKV() : BaseKV(BaseKVConfig{}) {}

  void Get(uint64_t key, std::string& value, unsigned) override {
    const auto it = values_.find(key);
    value         = it == values_.end() ? std::string() : it->second;
  }

  void Put(uint64_t key, const std::string_view& value, unsigned) override {
    values_[key] = std::string(value);
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned) override {
    values->clear();
    values->reserve(keys.Size());
    for (const auto key : keys) {
      const auto it = values_.find(key);
      if (it == values_.end()) {
        values->emplace_back();
      } else {
        values->emplace_back(it->second);
      }
    }
  }

  std::vector<float> ReadFloats(uint64_t key) {
    std::string value;
    Get(key, value, 0);
    return base::ConstArray<float>(value).ToVector();
  }

private:
  std::unordered_map<uint64_t, std::string> values_;
};

uint64_t TaggedKey(uint64_t key, TAG_TYPE tag) {
  constexpr int tag_bits = sizeof(TAG_TYPE) * 8;
  constexpr int shift    = sizeof(uint64_t) * 8 - tag_bits;
  return (static_cast<uint64_t>(tag) << shift) | key;
}

TEST(OptimizerFactoryTest, SelectsConfiguredOptimizerAndRejectsUnknownType) {
  auto optimizer = CreateOptimizer(json{
      {"type", "RowWiseAdagrad"},
      {"learning_rate", 0.001},
      {"epsilon", 1e-8},
  });
  EXPECT_NE(dynamic_cast<RowWiseAdaGrad*>(optimizer.get()), nullptr);
  auto adamw = CreateOptimizer(json{
      {"type", "AdamW"},
      {"learning_rate", 0.001},
      {"beta1", 0.9},
      {"beta2", 0.98},
      {"epsilon", 1e-8},
      {"weight_decay", 0.1},
  });
  EXPECT_NE(dynamic_cast<AdamW*>(adamw.get()), nullptr);
  EXPECT_THROW(
      CreateOptimizer(json{{"type", "Adam"}, {"learning_rate", 0.001}}),
      std::invalid_argument);
}

TEST(AdamWTest, MatchesTwoStepBiasCorrectedFormulaAndPersistsState) {
  constexpr float learning_rate   = 0.001f;
  constexpr float beta1           = 0.9f;
  constexpr float beta2           = 0.98f;
  constexpr float epsilon         = 1e-8f;
  constexpr float weight_decay    = 0.1f;
  constexpr int64_t embedding_dim = 2;
  constexpr uint64_t key          = 7;
  constexpr uint64_t step_key     = (std::numeric_limits<uint64_t>::max() >> 8);

  InMemoryKV kv;
  const std::vector<float> initial{1.0f, -2.0f};
  kv.Put(key,
         std::string(reinterpret_cast<const char*>(initial.data()),
                     initial.size() * sizeof(float)),
         0);

  const std::vector<uint64_t> keys{key};
  const std::vector<float> first_grad{3.0f, 4.0f};
  {
    AdamW optimizer(learning_rate, beta1, beta2, epsilon, weight_decay);
    optimizer.Init({"table"}, EmbeddingTableConfig{16, embedding_dim}, &kv);
    optimizer.UpdateFlat("table", keys, first_grad.data(), 1, embedding_dim, 0);
  }

  std::vector<float> m(embedding_dim), v(embedding_dim), expected = initial;
  for (int i = 0; i < embedding_dim; ++i) {
    m[i]        = (1.0f - beta1) * first_grad[i];
    v[i]        = (1.0f - beta2) * first_grad[i] * first_grad[i];
    expected[i] = (1.0f - learning_rate * weight_decay) * expected[i] -
                  learning_rate * (m[i] / (1.0f - beta1)) /
                      (std::sqrt(v[i] / (1.0f - beta2)) + epsilon);
  }
  auto actual = kv.ReadFloats(key);
  ASSERT_EQ(actual.size(), expected.size());
  for (int i = 0; i < embedding_dim; ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-6);
  }
  EXPECT_FLOAT_EQ(
      kv.ReadFloats(TaggedKey(
          step_key,
          MakeTensorTag(0, AdamW::kTensorStep, AdamW::kTensorsPerTable)))[0],
      1.0f);

  // Recreating the optimizer models a PS restart after its KV checkpoint is
  // restored. The next update must resume at step 2.
  AdamW restarted_optimizer(learning_rate, beta1, beta2, epsilon, weight_decay);
  restarted_optimizer.Init(
      {"table"}, EmbeddingTableConfig{16, embedding_dim}, &kv);
  const std::vector<float> second_grad{1.0f, -2.0f};
  restarted_optimizer.UpdateFlat(
      "table", keys, second_grad.data(), 1, embedding_dim, 0);
  for (int i = 0; i < embedding_dim; ++i) {
    m[i] = beta1 * m[i] + (1.0f - beta1) * second_grad[i];
    v[i] = beta2 * v[i] + (1.0f - beta2) * second_grad[i] * second_grad[i];
    expected[i] =
        (1.0f - learning_rate * weight_decay) * expected[i] -
        learning_rate * (m[i] / (1.0f - std::pow(beta1, 2.0f))) /
            (std::sqrt(v[i] / (1.0f - std::pow(beta2, 2.0f))) + epsilon);
  }
  actual = kv.ReadFloats(key);
  for (int i = 0; i < embedding_dim; ++i) {
    EXPECT_NEAR(actual[i], expected[i], 1e-6);
  }
  EXPECT_FLOAT_EQ(
      kv.ReadFloats(TaggedKey(
          step_key,
          MakeTensorTag(0, AdamW::kTensorStep, AdamW::kTensorsPerTable)))[0],
      2.0f);
}

TEST(AdamWTest, KeepsTablesWithDistinctIdsInSeparateKeySpaces) {
  constexpr int64_t embedding_dim = 2;
  constexpr uint64_t key          = 7;
  constexpr uint64_t step_key     = (std::numeric_limits<uint64_t>::max() >> 8);

  InMemoryKV kv;
  AdamW optimizer(0.001f, 0.9f, 0.98f, 1e-8f, 0.1f);
  EmbeddingTableConfig first{16, embedding_dim};
  EmbeddingTableConfig second{16, embedding_dim};
  second.table_id      = 1;
  const auto first_tag = MakeTensorTag(
      first.table_id, AdamW::kTensorParameter, AdamW::kTensorsPerTable);
  const auto second_tag = MakeTensorTag(
      second.table_id, AdamW::kTensorParameter, AdamW::kTensorsPerTable);
  ASSERT_EQ(optimizer.Init({"first"}, first, &kv), static_cast<int>(first_tag));
  ASSERT_EQ(optimizer.Init({"second"}, second, &kv),
            static_cast<int>(second_tag));

  const std::vector<uint64_t> first_keys{TaggedKey(key, first_tag)};
  const std::vector<float> first_grad{1.0f, 1.0f};
  optimizer.UpdateFlat(
      "first", first_keys, first_grad.data(), 1, embedding_dim, 0);
  const std::vector<uint64_t> second_keys{TaggedKey(key, second_tag)};
  const std::vector<float> second_grad{-2.0f, -2.0f};
  optimizer.UpdateFlat(
      "second", second_keys, second_grad.data(), 1, embedding_dim, 0);

  // Same row id and same step: each table owns its own parameter, moments and
  // step counter.
  const auto first_param  = kv.ReadFloats(TaggedKey(key, first_tag));
  const auto second_param = kv.ReadFloats(TaggedKey(key, second_tag));
  ASSERT_EQ(first_param.size(), embedding_dim);
  ASSERT_EQ(second_param.size(), embedding_dim);
  EXPECT_NE(first_param[0], second_param[0]);
  EXPECT_NE(first_param[1], second_param[1]);
  // Moments live in their own per-table tags as well.
  const auto first_moment  = kv.ReadFloats(TaggedKey(
      key,
      MakeTensorTag(0, AdamW::kTensorFirstMoment, AdamW::kTensorsPerTable)));
  const auto second_moment = kv.ReadFloats(TaggedKey(
      key,
      MakeTensorTag(1, AdamW::kTensorFirstMoment, AdamW::kTensorsPerTable)));
  ASSERT_EQ(first_moment.size(), embedding_dim);
  ASSERT_EQ(second_moment.size(), embedding_dim);
  EXPECT_NEAR(first_moment[0], 0.1f, 1e-6);
  EXPECT_NEAR(second_moment[0], -0.2f, 1e-6);
  EXPECT_FLOAT_EQ(
      kv.ReadFloats(TaggedKey(
          step_key,
          MakeTensorTag(0, AdamW::kTensorStep, AdamW::kTensorsPerTable)))[0],
      1.0f);
  EXPECT_FLOAT_EQ(
      kv.ReadFloats(TaggedKey(
          step_key,
          MakeTensorTag(1, AdamW::kTensorStep, AdamW::kTensorsPerTable)))[0],
      1.0f);
}

TEST(AdamWTest, RejectsSecondNameSharingOneTableId) {
  InMemoryKV kv;
  AdamW optimizer(0.001f, 0.9f, 0.98f, 1e-8f, 0.1f);
  optimizer.Init({"table_a"}, EmbeddingTableConfig{16, 2}, &kv);
  // Two names under one table_id would still map onto the same tags.
  EXPECT_THROW(optimizer.Init({"table_b"}, EmbeddingTableConfig{16, 2}, &kv),
               std::runtime_error);
}

TEST(AdamWTest, RejectsTableIdThatOverflowsTheEightBitTag) {
  InMemoryKV kv;
  AdamW optimizer(0.001f, 0.9f, 0.98f, 1e-8f, 0.1f);
  EmbeddingTableConfig last{16, 2};
  last.table_id = 63;
  // 63 * 4 + 3 == 255, the last tag the 8-bit key layout can address.
  EXPECT_EQ(optimizer.Init({"last"}, last, &kv),
            static_cast<int>(MakeTensorTag(63, 0, 4)));

  EmbeddingTableConfig overflow{16, 2};
  overflow.table_id = 64;
  // The tag must fail loudly instead of wrapping around to table 0.
  EXPECT_THROW(optimizer.Init({"overflow"}, overflow, &kv), std::runtime_error);
}

TEST(AdamWTest, RejectsClientKeysTaggedForAnotherTable) {
  InMemoryKV kv;
  AdamW optimizer(0.001f, 0.9f, 0.98f, 1e-8f, 0.1f);
  EmbeddingTableConfig config{16, 2};
  config.table_id = 1;
  const auto table_tag =
      MakeTensorTag(1, AdamW::kTensorParameter, AdamW::kTensorsPerTable);
  const auto foreign_tag =
      MakeTensorTag(2, AdamW::kTensorParameter, AdamW::kTensorsPerTable);
  const std::vector<float> grad{1.0f, 1.0f};
  optimizer.Init({"table"}, config, &kv);
  const std::vector<uint64_t> keys{TaggedKey(7, foreign_tag)};
  EXPECT_THROW(optimizer.UpdateFlat("table", keys, grad.data(), 1, 2, 0),
               std::runtime_error);
  const std::vector<uint64_t> own_keys{TaggedKey(7, table_tag)};
  EXPECT_NO_THROW(
      optimizer.UpdateFlat("table", own_keys, grad.data(), 1, 2, 0));
}

TEST(AdamWTest, AllowsSingleTableWithNonZeroTableId) {
  InMemoryKV kv;
  AdamW optimizer(0.001f, 0.9f, 0.98f, 1e-8f, 0.1f);
  EmbeddingTableConfig config{16, 2};
  config.table_id = 5;
  // A fused table may be registered under any backend table id; the returned
  // parameter tag carries it.
  EXPECT_EQ(optimizer.Init({"fused"}, config, &kv),
            static_cast<int>(MakeTensorTag(
                5, AdamW::kTensorParameter, AdamW::kTensorsPerTable)));
}

TEST(AdamWTest, AllowsReRegisteringTheSameTable) {
  InMemoryKV kv;
  AdamW optimizer(0.001f, 0.9f, 0.98f, 1e-8f, 0.1f);
  optimizer.Init({"table"}, EmbeddingTableConfig{16, 2}, &kv);
  // Ranks and restarts re-register the same table; only a different name is
  // rejected.
  EXPECT_NO_THROW(optimizer.Init({"table"}, EmbeddingTableConfig{16, 2}, &kv));
}

TEST(RowWiseAdaGradTest, MatchesTwoStepFormulaAndInitializesMissingRows) {
  constexpr float learning_rate   = 0.001f;
  constexpr float epsilon         = 1e-8f;
  constexpr int64_t embedding_dim = 2;
  constexpr uint64_t key          = 7;

  InMemoryKV kv;
  RowWiseAdaGrad optimizer(learning_rate, epsilon);
  optimizer.Init({"table"}, EmbeddingTableConfig{16, embedding_dim}, &kv);

  const std::vector<uint64_t> keys{key};
  const std::vector<float> first_grad{3.0f, 4.0f};
  optimizer.UpdateFlat("table", keys, first_grad.data(), 1, embedding_dim, 0);

  const float first_accumulator = 12.5f;
  const float first_scale =
      learning_rate / (std::sqrt(first_accumulator) + epsilon);
  const std::vector<float> expected_first{
      -first_scale * first_grad[0], -first_scale * first_grad[1]};
  const auto first_value = kv.ReadFloats(key);
  ASSERT_EQ(first_value.size(), expected_first.size());
  EXPECT_NEAR(first_value[0], expected_first[0], 1e-7);
  EXPECT_NEAR(first_value[1], expected_first[1], 1e-7);

  const auto first_state =
      kv.ReadFloats(TaggedKey(key, static_cast<TAG_TYPE>(MOMENT_1)));
  ASSERT_EQ(first_state.size(), 1);
  EXPECT_FLOAT_EQ(first_state[0], first_accumulator);

  const std::vector<float> second_grad{1.0f, -2.0f};
  optimizer.UpdateFlat("table", keys, second_grad.data(), 1, embedding_dim, 0);

  const float second_accumulator = first_accumulator + 2.5f;
  const float second_scale =
      learning_rate / (std::sqrt(second_accumulator) + epsilon);
  const auto second_value = kv.ReadFloats(key);
  ASSERT_EQ(second_value.size(), expected_first.size());
  EXPECT_NEAR(
      second_value[0], expected_first[0] - second_scale * second_grad[0], 1e-7);
  EXPECT_NEAR(
      second_value[1], expected_first[1] - second_scale * second_grad[1], 1e-7);

  const auto second_state =
      kv.ReadFloats(TaggedKey(key, static_cast<TAG_TYPE>(MOMENT_1)));
  ASSERT_EQ(second_state.size(), 1);
  EXPECT_FLOAT_EQ(second_state[0], second_accumulator);
}

} // namespace
