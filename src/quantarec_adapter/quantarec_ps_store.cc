#include "quantarec_adapter/quantarec_ps_store.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "quantarec_adapter/quantarec_row_codec.h"

namespace recstore {
namespace quantarec_adapter {
namespace {

using Clock = std::chrono::steady_clock;

int64_t ElapsedNs(const Clock::time_point& t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0)
      .count();
}

double SampleTruncNormal(std::mt19937_64& rng,
                         double mean,
                         double stddev,
                         double a,
                         double b) {
  std::normal_distribution<double> dist(mean, stddev);
  for (int attempt = 0; attempt < 64; ++attempt) {
    const double sample = dist(rng);
    if (sample >= a && sample <= b) {
      return sample;
    }
  }
  return std::clamp(mean, a, b);
}

}  // namespace

void QuantarecPsStore::Reset(const QuantarecPsTableConfig& config) {
  if (config.shared_name.empty()) {
    throw std::invalid_argument("QuantarecPsStore::Reset: shared_name is empty");
  }
  if (config.block_size <= 0) {
    throw std::invalid_argument("QuantarecPsStore::Reset: block_size must be positive");
  }
  const int64_t embedding_dim = FlatEmbeddingDim(config.embedding_shape);
  if (embedding_dim <= 0) {
    throw std::invalid_argument("QuantarecPsStore::Reset: embedding_dim must be positive");
  }

  shared_name_ = config.shared_name;
  block_size_ = config.block_size;
  embedding_dim_ = embedding_dim;
  trainable_ = config.trainable;
  children_ = config.children;
  generator_ = config.generator;
  rng_.seed(generator_.random_seed != 0 ? generator_.random_seed : 1);

  id_to_row_.clear();
  row_to_id_.clear();
  embeddings_.clear();
  modify_ms_.clear();
  next_row_index_ = 0;
  last_lookup_stats_ = {};

  const int64_t reserve_rows = std::max<int64_t>(block_size_, 1024);
  embeddings_.reserve(static_cast<size_t>(reserve_rows * embedding_dim_));
  modify_ms_.reserve(static_cast<size_t>(reserve_rows));
  row_to_id_.reserve(static_cast<size_t>(reserve_rows));
}

float* QuantarecPsStore::RowPtr(int64_t row_index) {
  if (row_index < 0) {
    return nullptr;
  }
  const size_t offset =
      static_cast<size_t>(row_index) * static_cast<size_t>(embedding_dim_);
  if (offset + static_cast<size_t>(embedding_dim_) > embeddings_.size()) {
    return nullptr;
  }
  return embeddings_.data() + offset;
}

const float* QuantarecPsStore::RowPtr(int64_t row_index) const {
  return const_cast<QuantarecPsStore*>(this)->RowPtr(row_index);
}

void QuantarecPsStore::InitNewRow(float* row) {
  const std::string& t = generator_.type;
  if (t.empty() || t == "EmptyGenerator") {
    ZeroFillRow(row, embedding_dim_);
    return;
  }
  if (t == "ConstantGenerator") {
    for (int64_t i = 0; i < embedding_dim_; ++i) {
      row[i] = static_cast<float>(generator_.constant_init_val);
    }
    return;
  }
  if (t == "UniformGenerator") {
    std::uniform_real_distribution<double> dist(generator_.uniform_a,
                                                generator_.uniform_b);
    for (int64_t i = 0; i < embedding_dim_; ++i) {
      row[i] = static_cast<float>(dist(rng_));
    }
    return;
  }
  if (t == "NormalGenerator") {
    std::normal_distribution<double> dist(generator_.normal_mean,
                                          generator_.normal_std);
    for (int64_t i = 0; i < embedding_dim_; ++i) {
      row[i] = static_cast<float>(dist(rng_));
    }
    return;
  }
  if (t == "TruncNormalGenerator") {
    for (int64_t i = 0; i < embedding_dim_; ++i) {
      row[i] = static_cast<float>(SampleTruncNormal(
          rng_, generator_.trunc_mean, generator_.trunc_std,
          generator_.trunc_a, generator_.trunc_b));
    }
    return;
  }
  ZeroFillRow(row, embedding_dim_);
}

int64_t QuantarecPsStore::AdmitNewRow(uint64_t id) {
  const int64_t row_index = next_row_index_++;
  const size_t new_size =
      static_cast<size_t>(next_row_index_) * static_cast<size_t>(embedding_dim_);
  embeddings_.resize(new_size, 0.0f);
  modify_ms_.resize(static_cast<size_t>(next_row_index_), 0);
  row_to_id_.resize(static_cast<size_t>(next_row_index_), 0);
  row_to_id_[static_cast<size_t>(row_index)] = id;
  id_to_row_[id] = row_index;

  float* row = RowPtr(row_index);
  InitNewRow(row);
  return row_index;
}

QuantarecPsLookupResult QuantarecPsStore::Lookup(const int64_t* ids,
                                                 int n_ids,
                                                 bool readonly) {
  QuantarecPsLookupResult out;
  out.index.resize(static_cast<size_t>(n_ids));
  out.embs.resize(static_cast<size_t>(n_ids) *
                   static_cast<size_t>(embedding_dim_));
  if (!readonly) {
    out.is_new.resize(static_cast<size_t>(n_ids), 0);
  }

  QuantarecPsLookupStats stats;
  const auto lookup_t0 = Clock::now();

  for (int i = 0; i < n_ids; ++i) {
    const uint64_t id = static_cast<uint64_t>(ids[i]);
    const auto index_t0 = Clock::now();
    auto it = id_to_row_.find(id);
    stats.index_lookup_ns += static_cast<std::uint64_t>(ElapsedNs(index_t0));

    int64_t row_index = kNullIndex;
    bool is_new = false;
    if (it == id_to_row_.end()) {
      if (readonly) {
        stats.missing_rows += 1;
        const auto zero_t0 = Clock::now();
        ZeroFillRow(out.embs.data() +
                        static_cast<size_t>(i) * static_cast<size_t>(embedding_dim_),
                    embedding_dim_);
        stats.zero_fill_ns += static_cast<std::uint64_t>(ElapsedNs(zero_t0));
        out.index[static_cast<size_t>(i)] = kNullIndex;
        continue;
      }
      const auto admit_t0 = Clock::now();
      row_index = AdmitNewRow(id);
      is_new = true;
      stats.admission_ns += static_cast<std::uint64_t>(ElapsedNs(admit_t0));
    } else {
      row_index = it->second;
    }

    out.index[static_cast<size_t>(i)] = row_index;
    if (!readonly) {
      out.is_new[static_cast<size_t>(i)] = is_new ? 1 : 0;
    }

    const auto copy_t0 = Clock::now();
    const float* src = RowPtr(row_index);
    float* dst = out.embs.data() +
                 static_cast<size_t>(i) * static_cast<size_t>(embedding_dim_);
    if (src == nullptr) {
      ZeroFillRow(dst, embedding_dim_);
      stats.missing_rows += 1;
      stats.zero_fill_ns += static_cast<std::uint64_t>(ElapsedNs(copy_t0));
    } else {
      std::memcpy(dst, src,
                  static_cast<size_t>(embedding_dim_) * sizeof(float));
      stats.row_copy_ns += static_cast<std::uint64_t>(ElapsedNs(copy_t0));
    }
  }

  stats.index_lookup_ns += static_cast<std::uint64_t>(ElapsedNs(lookup_t0));
  out.stats = stats;
  last_lookup_stats_ = stats;
  return out;
}

void QuantarecPsStore::Insert(const int64_t* ids,
                              int n_ids,
                              const float* embs) {
  if (n_ids == 0) {
    return;
  }
  for (int i = 0; i < n_ids; ++i) {
    const uint64_t id = static_cast<uint64_t>(ids[i]);
    auto it = id_to_row_.find(id);
    int64_t row_index = kNullIndex;
    if (it == id_to_row_.end()) {
      row_index = AdmitNewRow(id);
    } else {
      row_index = it->second;
    }
    float* dst = RowPtr(row_index);
    const float* src = embs + static_cast<size_t>(i) *
                                  static_cast<size_t>(embedding_dim_);
    std::memcpy(dst, src,
                static_cast<size_t>(embedding_dim_) * sizeof(float));
  }
}

QuantarecPsStoreStats QuantarecPsStore::GetStats() const {
  QuantarecPsStoreStats stats;
  stats.shared_name = shared_name_;
  stats.total_ids = static_cast<int64_t>(id_to_row_.size());
  stats.embedding_dim = embedding_dim_;
  stats.block_size = block_size_;
  stats.children = children_;
  stats.child_id_sizes.assign(children_.size(), 0);
  stats.last_lookup_stats = last_lookup_stats_;
  return stats;
}

}  // namespace quantarec_adapter
}  // namespace recstore
