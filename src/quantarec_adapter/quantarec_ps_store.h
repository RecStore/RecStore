#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace recstore {
namespace quantarec_adapter {

/// Mirrors ``quantarec::train_ps_service::EmbeddingGeneratorParams`` subset.
struct EmbeddingGeneratorConfig {
  std::string type;
  uint64_t random_seed = 0;
  double constant_init_val = 0.0;
  double uniform_a = 0.0;
  double uniform_b = 1.0;
  double normal_mean = 0.0;
  double normal_std = 1.0;
  double trunc_mean = 0.0;
  double trunc_std = 1.0;
  double trunc_a = -2.0;
  double trunc_b = 2.0;
};

struct QuantarecPsTableConfig {
  std::string shared_name;
  int64_t block_size = 0;
  std::vector<int64_t> embedding_shape;
  int dtype = 0;
  bool trainable = true;
  std::vector<std::string> children;
  EmbeddingGeneratorConfig generator;
};

struct QuantarecPsLookupStats {
  std::uint64_t index_lookup_ns = 0;
  std::uint64_t row_copy_ns = 0;
  std::uint64_t admission_ns = 0;
  std::uint64_t zero_fill_ns = 0;
  std::uint64_t missing_rows = 0;
};

struct QuantarecPsLookupResult {
  std::vector<int64_t> index;
  std::vector<float> embs;
  std::vector<uint8_t> is_new;
  QuantarecPsLookupStats stats;
};

struct QuantarecPsStoreStats {
  std::string shared_name;
  int64_t total_ids = 0;
  int64_t embedding_dim = 0;
  int64_t block_size = 0;
  std::vector<std::string> children;
  std::vector<int64_t> child_id_sizes;
  QuantarecPsLookupStats last_lookup_stats;
};

/// DRAM-only Index + flat ValueStore adapter for Quantarec Training PS.
///
/// Index: ``id -> dense row index`` (compatible with Quantarec grad_index).
/// Value: contiguous ``float32[capacity_rows][embedding_dim]`` rows.
class QuantarecPsStore {
 public:
  static constexpr int64_t kNullIndex = -1;

  void Reset(const QuantarecPsTableConfig& config);
  QuantarecPsLookupResult Lookup(const int64_t* ids, int n_ids, bool readonly);
  void Insert(const int64_t* ids, int n_ids, const float* embs);
  QuantarecPsStoreStats GetStats() const;

  int64_t EmbeddingDim() const { return embedding_dim_; }
  const std::string& SharedName() const { return shared_name_; }

 private:
  int64_t AdmitNewRow(uint64_t id);
  void InitNewRow(float* row);
  float* RowPtr(int64_t row_index);
  const float* RowPtr(int64_t row_index) const;

  std::string shared_name_;
  int64_t block_size_ = 0;
  int64_t embedding_dim_ = 0;
  bool trainable_ = true;
  std::vector<std::string> children_;

  std::unordered_map<uint64_t, int64_t> id_to_row_;
  std::vector<uint64_t> row_to_id_;
  std::vector<float> embeddings_;
  std::vector<int64_t> modify_ms_;
  int64_t next_row_index_ = 0;

  EmbeddingGeneratorConfig generator_;
  std::mt19937_64 rng_{0};

  QuantarecPsLookupStats last_lookup_stats_;
};

}  // namespace quantarec_adapter
}  // namespace recstore
