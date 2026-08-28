#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define private public
#include "framework/op.h"
#undef private

namespace {

class RecordingPSClient : public recstore::BasePSClient {
public:
  RecordingPSClient() : recstore::BasePSClient(json::object()) {}

  int GetParameter(const base::ConstArray<uint64_t>& keys,
                   base::RecTensor& values) override {
    get_calls++;
    last_get_rows = static_cast<int64_t>(keys.size);
    if (!recstore::IsFloatEmbeddingValues(values, keys.size)) {
      return -1;
    }
    const int64_t D = values.shape(1);
    embedding_dim   = D;
    float* dst      = values.data_as<float>();
    for (int row = 0; row < keys.size; ++row) {
      auto it    = rows.find(keys[row]);
      float* out = dst + row * D;
      if (it == rows.end()) {
        std::fill(out, out + D, 0.0f);
      } else {
        std::copy_n(it->second.begin(),
                    std::min<int64_t>(D, it->second.size()),
                    out);
      }
    }
    return read_return;
  }

  int PutParameter(const base::ConstArray<uint64_t>& keys,
                   const base::RecTensor& values) override {
    put_calls++;
    last_put_rows = static_cast<int64_t>(keys.size);
    if (!recstore::IsFloatEmbeddingValues(values, keys.size)) {
      return -1;
    }
    const int64_t D = values.shape(1);
    embedding_dim   = D;
    last_put_dim    = D;
    const float* src = values.data_as<float>();
    last_put_values.assign(src, src + values.num_elements());
    for (int i = 0; i < keys.size; ++i) {
      rows[keys[i]].assign(src + i * D, src + (i + 1) * D);
    }
    return write_return;
  }

  int UpdateParameter(const std::string& table_name,
                      const base::ConstArray<uint64_t>& keys,
                      const base::RecTensor& grads) override {
    update_calls++;
    last_update_table = table_name;
    last_update_rows  = static_cast<int64_t>(keys.size);
    last_update_dim   = grads.dim() == 2 ? grads.shape(1) : 0;
    last_update_keys.assign(keys.Data(), keys.Data() + keys.size);
    if (grads.data() != nullptr) {
      last_update_grads.assign(
          grads.data_as<float>(),
          grads.data_as<float>() + grads.num_elements());
    }
    return update_return;
  }

  int InitEmbeddingTable(
      const std::string& table_name,
      const recstore::EmbeddingTableConfig& config) override {
    init_table_calls++;
    last_init_table  = table_name;
    last_init_config = config;
    return init_table_return;
  }

  void Command(recstore::PSCommand command) override { last_command = command; }

  uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys) override {
    prefetch_calls++;
    last_prefetch_keys.assign(keys.Data(), keys.Data() + keys.size);
    return next_prefetch_id++;
  }

  bool IsPrefetchDone(uint64_t prefetch_id) override {
    last_prefetch_done_id = prefetch_id;
    return prefetch_done_return;
  }

  void WaitForPrefetch(uint64_t prefetch_id) override {
    last_wait_prefetch_id = prefetch_id;
  }

  bool GetPrefetchResult(uint64_t prefetch_id,
                         base::RecTensor& values) override {
    last_result_prefetch_id = prefetch_id;
    if (!recstore::EnsureEmbeddingOutput(
            values, static_cast<int64_t>(prefetch_result.size()))) {
      return false;
    }
    const int64_t D = values.shape(1);
    float* dst      = values.data_as<float>();
    for (size_t row = 0; row < prefetch_result.size(); ++row) {
      const auto& src = prefetch_result[row];
      std::copy_n(src.begin(),
                  std::min<int64_t>(D, src.size()),
                  dst + row * static_cast<size_t>(D));
    }
    return true;
  }

  int64_t embedding_dim                 = 3;
  int read_return                       = 0;
  int write_return                      = 0;
  int update_return                     = 0;
  int init_table_return                 = 0;
  bool prefetch_done_return             = true;
  uint64_t next_prefetch_id             = 700;
  int get_calls                         = 0;
  int put_calls                         = 0;
  int update_calls                      = 0;
  int init_table_calls                  = 0;
  int prefetch_calls                    = 0;
  int64_t last_get_rows                 = 0;
  int64_t last_put_rows                 = 0;
  int64_t last_update_rows              = 0;
  int64_t last_update_dim               = 0;
  uint64_t last_prefetch_done_id        = 0;
  uint64_t last_wait_prefetch_id        = 0;
  uint64_t last_result_prefetch_id      = 0;
  std::string last_update_table;
  std::string last_init_table;
  recstore::EmbeddingTableConfig last_init_config{0, 0};
  recstore::PSCommand last_command = recstore::PSCommand::CLEAR_PS;
  std::vector<uint64_t> last_update_keys;
  std::vector<uint64_t> last_prefetch_keys;
  std::vector<float> last_update_grads;
  std::vector<float> last_put_values;
  int64_t last_put_dim = 0;
  std::vector<std::vector<float>> prefetch_result{{1.0f, 2.0f, 3.0f}};
  std::unordered_map<uint64_t, std::vector<float>> rows;
};

class InjectedClientOpTest : public ::testing::Test {
protected:
  void SetUp() override {
    previous_ps_client_ = recstore::KVClientOp::ps_client_;
    previous_ps_client_holder_ =
        std::move(recstore::KVClientOp::ps_client_holder_);

    auto client = std::make_unique<RecordingPSClient>();
    client_     = client.get();
    recstore::KVClientOp::ps_client_holder_ = std::move(client);
    recstore::KVClientOp::ps_client_        = client_;
    op_                   = std::make_unique<recstore::KVClientOp>();
    op_->ps_backend_name_ = "grpc";
  }

  void TearDown() override {
    op_.reset();
    recstore::KVClientOp::ps_client_holder_.reset();
    recstore::KVClientOp::ps_client_holder_ =
        std::move(previous_ps_client_holder_);
    recstore::KVClientOp::ps_client_ = previous_ps_client_;
  }

  std::unique_ptr<recstore::KVClientOp> op_;
  RecordingPSClient* client_                  = nullptr;
  recstore::BasePSClient* previous_ps_client_ = nullptr;
  std::unique_ptr<recstore::BasePSClient> previous_ps_client_holder_;
};

base::RecTensor UInt64Tensor(std::vector<uint64_t>* values) {
  return base::RecTensor(
      values->data(), {static_cast<int64_t>(values->size())});
}

base::RecTensor
FloatTensor(std::vector<float>* values, int64_t rows, int64_t cols) {
  return base::RecTensor(values->data(), {rows, cols});
}

std::string WriteHierKVConfig() {
  const std::string path =
      "/tmp/recstore_test_op_hierkv_" + std::to_string(::getpid()) + ".json";
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to create config file: " + path);
  }
  out << R"({"cache_ps":{"ps_type":"hierkv"}})";
  return path;
}

class OpTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    config_path_ = WriteHierKVConfig();
    setenv("RECSTORE_CONFIG", config_path_.c_str(), 1);
  }

  static void TearDownTestSuite() {
    unsetenv("RECSTORE_CONFIG");
    std::remove(config_path_.c_str());
  }

  recstore::KVClientOp op_;

private:
  static std::string config_path_;
};

std::string OpTest::config_path_;

TEST(InitStrategyTest, FactoryMethodsPopulateTypeAndParameters) {
  const auto normal = recstore::InitStrategy::Normal(0.5f, 2.0f);
  EXPECT_EQ(normal.type, recstore::InitStrategyType::Normal);
  EXPECT_FLOAT_EQ(normal.mean, 0.5f);
  EXPECT_FLOAT_EQ(normal.std, 2.0f);

  const auto uniform = recstore::InitStrategy::Uniform(-3.0f, 4.0f);
  EXPECT_EQ(uniform.type, recstore::InitStrategyType::Uniform);
  EXPECT_FLOAT_EQ(uniform.lower, -3.0f);
  EXPECT_FLOAT_EQ(uniform.upper, 4.0f);

  EXPECT_EQ(recstore::InitStrategy::Xavier().type,
            recstore::InitStrategyType::Xavier);
  EXPECT_EQ(recstore::InitStrategy::Zero().type,
            recstore::InitStrategyType::Zero);
}

TEST_F(OpTest, ConstructorUsesHierKVBackendFromConfig) {
  EXPECT_EQ(op_.CurrentPSBackend(), "hierkv");
}

TEST_F(OpTest, GetKVClientOpReturnsStableSingleton) {
  auto first  = recstore::GetKVClientOp();
  auto second = recstore::GetKVClientOp();

  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first.get(), second.get());
}

TEST_F(OpTest, UnsupportedOperationsThrowNotImplemented) {
  std::vector<uint64_t> key_values{901};
  std::vector<float> value_values{1.0f, 2.0f, 3.0f};
  auto keys   = UInt64Tensor(&key_values);
  auto values = FloatTensor(&value_values, 1, 3);

  EXPECT_THROW(op_.EmbDelete(keys), std::runtime_error);
  EXPECT_THROW(op_.EmbExists(keys), std::runtime_error);
  EXPECT_THROW(op_.WaitForWrite(1), std::runtime_error);
  EXPECT_THROW(op_.SaveToFile("/tmp/unused"), std::runtime_error);
  EXPECT_THROW(op_.LoadFromFile("/tmp/unused"), std::runtime_error);
  EXPECT_THROW(op_.EmbWriteAsync(keys, values), std::runtime_error);
  EXPECT_THROW(op_.IsWriteDone(1), std::runtime_error);
}

TEST_F(OpTest, EmbInitWithTensorValuesDelegatesToWrite) {
  std::vector<uint64_t> key_values{910};
  std::vector<float> init_values{10.0f, 20.0f, 30.0f};
  std::vector<float> read_values(3, -1.0f);
  auto keys        = UInt64Tensor(&key_values);
  auto read_tensor = FloatTensor(&read_values, 1, 3);

  op_.EmbInit(keys, FloatTensor(&init_values, 1, 3));
  op_.EmbRead(keys, read_tensor);

  EXPECT_EQ(read_values, init_values);
}

TEST_F(OpTest, EmbInitWithStrategyValidatesKeys) {
  std::vector<uint64_t> key_values{920, 921};
  auto keys = UInt64Tensor(&key_values);
  EXPECT_NO_THROW(op_.EmbInit(keys, recstore::InitStrategy::Zero()));

  std::vector<float> bad_key_values{1.0f, 2.0f};
  base::RecTensor bad_keys(bad_key_values.data(), {2});
  EXPECT_THROW(op_.EmbInit(bad_keys, recstore::InitStrategy::Zero()),
               std::invalid_argument);
}

TEST_F(OpTest, HierKVWriteReadUpdateAndMissingRowsRoundTrip) {
  std::vector<uint64_t> write_key_values{1001, 1002};
  std::vector<float> write_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto write_keys = UInt64Tensor(&write_key_values);
  op_.EmbWrite(write_keys, FloatTensor(&write_values, 2, 3));

  std::vector<uint64_t> read_key_values{1001, 1003};
  std::vector<float> read_values(6, -1.0f);
  auto read_keys   = UInt64Tensor(&read_key_values);
  auto read_tensor = FloatTensor(&read_values, 2, 3);
  op_.EmbRead(read_keys, read_tensor);
  EXPECT_EQ(read_values,
            (std::vector<float>{1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f}));

  std::vector<float> grads{10.0f, 20.0f, 30.0f, 1.0f, 2.0f, 3.0f};
  op_.EmbUpdate("default", read_keys, FloatTensor(&grads, 2, 3));

  std::fill(read_values.begin(), read_values.end(), -1.0f);
  op_.EmbRead(read_keys, read_tensor);
  EXPECT_EQ(read_values,
            (std::vector<float>{0.9f, 1.8f, 2.7f, -0.01f, -0.02f, -0.03f}));
}

TEST_F(OpTest, InitEmbeddingTableSucceedsForMatchingHierKVTable) {
  const recstore::EmbeddingTableConfig config{
      .num_embeddings = 1024,
      .embedding_dim  = 3,
  };

  EXPECT_EQ(op_.InitEmbeddingTable("table_for_op_test", config), 0);
  EXPECT_EQ(op_.InitEmbeddingTable("table_for_op_test", config), 0);
}

TEST_F(OpTest, PrefetchResultsConsumeCachedRows) {
  std::vector<uint64_t> key_values{1101, 1102};
  std::vector<float> values{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  auto keys = UInt64Tensor(&key_values);
  auto rows = FloatTensor(&values, 2, 3);
  op_.EmbWrite(keys, rows);

  const uint64_t prefetch_id = op_.EmbPrefetch(keys, rows);
  EXPECT_TRUE(op_.IsPrefetchDone(prefetch_id));
  EXPECT_NO_THROW(op_.WaitForPrefetch(prefetch_id));

  std::vector<float> fetched_storage(6, -1.0f);
  auto fetched = FloatTensor(&fetched_storage, 2, 3);
  op_.GetPretchResult(prefetch_id, fetched);
  EXPECT_EQ(fetched_storage, values);
  EXPECT_FALSE(op_.IsPrefetchDone(prefetch_id));

  const uint64_t owned_id = op_.EmbPrefetch(keys, rows);
  base::RecTensor owned_fetched({0, 3}, base::DataType::FLOAT32);
  op_.GetPretchResult(owned_id, owned_fetched);
  ASSERT_EQ(owned_fetched.shape(0), 2);
  ASSERT_EQ(owned_fetched.shape(1), 3);
  EXPECT_EQ(std::vector<float>(owned_fetched.data_as<float>(),
                               owned_fetched.data_as<float>() + 6),
            values);

  std::vector<uint64_t> missing_keys{88001, 88002};
  base::RecTensor dummy({0, 3}, base::DataType::FLOAT32);
  const uint64_t missing_id =
      op_.EmbPrefetch(UInt64Tensor(&missing_keys), dummy);
  base::RecTensor missing({0, 3}, base::DataType::FLOAT32);
  op_.GetPretchResult(missing_id, missing);
  EXPECT_EQ(missing.shape(0), 2);
  EXPECT_EQ(std::vector<float>(missing.data_as<float>(),
                               missing.data_as<float>() + 6),
            (std::vector<float>(6, 0.0f)));
}

TEST_F(OpTest, TensorValidationRejectsInvalidInputsBeforeBackendMutation) {
  std::vector<uint64_t> key_values{1201, 1202};
  std::vector<float> good_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto keys        = UInt64Tensor(&key_values);
  auto good_tensor = FloatTensor(&good_values, 2, 3);

  std::vector<int32_t> int_keys{1, 2};
  base::RecTensor bad_key_dtype(int_keys.data(), {2});
  EXPECT_THROW(op_.EmbWrite(bad_key_dtype, good_tensor), std::invalid_argument);

  std::vector<uint64_t> two_dim_key_values{1, 2};
  base::RecTensor bad_key_shape(two_dim_key_values.data(), {1, 2});
  EXPECT_THROW(op_.EmbRead(bad_key_shape, good_tensor), std::invalid_argument);

  std::vector<int32_t> int_values{1, 2, 3, 4, 5, 6};
  base::RecTensor bad_value_dtype(int_values.data(), {2, 3});
  EXPECT_THROW(op_.EmbWrite(keys, bad_value_dtype), std::invalid_argument);

  base::RecTensor bad_value_shape(good_values.data(), {6});
  EXPECT_THROW(op_.EmbWrite(keys, bad_value_shape), std::invalid_argument);

  std::vector<float> one_row_values{1.0f, 2.0f, 3.0f};
  EXPECT_THROW(op_.EmbWrite(keys, FloatTensor(&one_row_values, 1, 3)),
               std::invalid_argument);
  auto one_row_tensor = FloatTensor(&one_row_values, 1, 3);
  EXPECT_THROW(op_.EmbRead(keys, one_row_tensor), std::invalid_argument);
  EXPECT_THROW(op_.EmbUpdate(keys, FloatTensor(&one_row_values, 1, 3)),
               std::invalid_argument);
}

TEST_F(InjectedClientOpTest, NonHierKVWriteCopiesRowsAndRejectsClientFailure) {
  std::vector<uint64_t> key_values{2001, 2002};
  std::vector<float> value_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto keys   = UInt64Tensor(&key_values);
  auto values = FloatTensor(&value_values, 2, 3);

  op_->EmbWrite(keys, values);

  EXPECT_EQ(client_->put_calls, 1);
  EXPECT_EQ(client_->last_put_dim, 3);
  EXPECT_EQ(client_->last_put_values, value_values);

  client_->write_return = -1;
  EXPECT_THROW(op_->EmbWrite(keys, values), std::runtime_error);
}

TEST_F(InjectedClientOpTest, NonHierKVReadClearsOutputAndChecksRows) {
  client_->rows.emplace(2101, std::vector<float>{9.0f, 8.0f, 7.0f});
  std::vector<uint64_t> key_values{2101, 2102};
  std::vector<float> value_values(6, -5.0f);
  auto keys   = UInt64Tensor(&key_values);
  auto values = FloatTensor(&value_values, 2, 3);

  op_->EmbRead(keys, values);

  EXPECT_EQ(client_->get_calls, 1);
  EXPECT_EQ(client_->last_get_rows, 2);
  EXPECT_EQ(value_values,
            (std::vector<float>{9.0f, 8.0f, 7.0f, 0.0f, 0.0f, 0.0f}));

  client_->read_return = -1;
  EXPECT_THROW(op_->EmbRead(keys, values), std::runtime_error);

  std::vector<float> one_row_values(3, 0.0f);
  auto one_row_tensor = FloatTensor(&one_row_values, 1, 3);
  EXPECT_THROW(op_->EmbRead(keys, one_row_tensor), std::invalid_argument);
}

TEST_F(InjectedClientOpTest, EmbUpdateAsyncRejectsUnsupportedBackend) {
  std::vector<uint64_t> key_values{2401};
  std::vector<float> grad_values{0.1f, 0.2f, 0.3f};
  auto keys  = UInt64Tensor(&key_values);
  auto grads = FloatTensor(&grad_values, 1, 3);

  EXPECT_THROW(op_->EmbUpdateAsync("table_x", keys, grads), std::runtime_error);
  EXPECT_THROW(op_->WaitForEmbUpdate(0), std::runtime_error);
}

TEST_F(InjectedClientOpTest, NonHierKVUpdatePassesTableKeysAndFlatGrads) {
  std::vector<uint64_t> key_values{2201, 2202};
  std::vector<float> grad_values{0.1f, 0.2f, 0.3f, 1.1f, 1.2f, 1.3f};
  auto keys  = UInt64Tensor(&key_values);
  auto grads = FloatTensor(&grad_values, 2, 3);

  op_->EmbUpdate("table_x", keys, grads);

  EXPECT_EQ(client_->update_calls, 1);
  EXPECT_EQ(client_->last_update_table, "table_x");
  EXPECT_EQ(client_->last_update_rows, 2);
  EXPECT_EQ(client_->last_update_dim, 3);
  EXPECT_EQ(client_->last_update_keys, key_values);
  EXPECT_EQ(client_->last_update_grads, grad_values);

  client_->update_return = -1;
  EXPECT_THROW(op_->EmbUpdate(keys, grads), std::runtime_error);
}

TEST_F(InjectedClientOpTest, NonHierKVInitEmbeddingTableReturnsClientStatus) {
  const recstore::EmbeddingTableConfig config{
      .num_embeddings = 64,
      .embedding_dim  = 3,
  };

  EXPECT_EQ(op_->InitEmbeddingTable("table_init", config), 0);
  EXPECT_EQ(client_->init_table_calls, 1);
  EXPECT_EQ(client_->last_init_table, "table_init");
  EXPECT_EQ(client_->last_init_config.num_embeddings, 64);
  EXPECT_EQ(client_->last_init_config.embedding_dim, 3);

  client_->init_table_return = -1;
  EXPECT_EQ(op_->InitEmbeddingTable("table_init", config), -1);
}

TEST_F(InjectedClientOpTest, NonHierKVPrefetchDelegatesStatusAndResults) {
  std::vector<uint64_t> key_values{2301, 2302};
  std::vector<float> dummy_values(6, 0.0f);
  auto keys  = UInt64Tensor(&key_values);
  auto dummy = FloatTensor(&dummy_values, 2, 3);

  const uint64_t prefetch_id = op_->EmbPrefetch(keys, dummy);
  EXPECT_EQ(prefetch_id, 700);
  EXPECT_EQ(client_->last_prefetch_keys, key_values);

  EXPECT_TRUE(op_->IsPrefetchDone(prefetch_id));
  EXPECT_EQ(client_->last_prefetch_done_id, prefetch_id);
  op_->WaitForPrefetch(prefetch_id);
  EXPECT_EQ(client_->last_wait_prefetch_id, prefetch_id);

  base::RecTensor rows({0, 3}, base::DataType::FLOAT32);
  op_->GetPretchResult(prefetch_id, rows);
  EXPECT_EQ(client_->last_result_prefetch_id, prefetch_id);
  ASSERT_EQ(rows.shape(0), 1);
  ASSERT_EQ(rows.shape(1), 3);
  EXPECT_EQ(std::vector<float>(rows.data_as<float>(), rows.data_as<float>() + 3),
            (std::vector<float>{1.0f, 2.0f, 3.0f}));

  std::vector<float> invalid_storage(6);
  auto invalid_rows = FloatTensor(&invalid_storage, 2, 3);
  EXPECT_THROW(op_->GetPretchResult(prefetch_id, invalid_rows),
               std::runtime_error);
}

} // namespace
