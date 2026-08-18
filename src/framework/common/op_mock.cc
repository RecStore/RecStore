#ifdef USE_FAKE_KVCLIENT
#  include "framework/op.h"
#  include <mutex>
#  include <unordered_map>
#  include <vector>

#  include "base/tensor.h"
#  include "ps/base/base_client.h"
#  include <algorithm>
#  include <cstring>
#  include <cstdlib>
#  include <iostream>
#  include <stdexcept>

#  ifndef RECSTORE_LOG
// Fallback logging when included standalone.
static int get_log_level() {
  static int level = []() {
    const char* env = std::getenv("RECSTORE_LOG_LEVEL");
    if (!env)
      return 1;
    return std::atoi(env);
  }();
  return level;
}
#    define RECSTORE_LOG(level, msg)                                           \
      do {                                                                     \
        if (get_log_level() >= level) {                                        \
          std::cout << msg << std::endl;                                       \
        }                                                                      \
      } while (0)
#  endif

namespace recstore {

KVClientOp::KVClientOp() : embedding_dim_(-1), learning_rate_(0.01f) {
  std::cout << "KVClientOp: Initialized MOCK (in-memory) backend." << std::endl;
}

void KVClientOp::EmbInit(const base::RecTensor& keys,
                         const InitStrategy& strategy) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (embedding_dim_ == -1) {
    throw std::runtime_error("KVClientOp Error: Must call EmbWrite or set "
                             "dimension before using InitStrategy.");
  }
  const uint64_t* key_data = keys.data_as<uint64_t>();
  const int64_t num_keys   = keys.shape(0);
  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key = key_data[i];
    store_[key]  = std::vector<float>(embedding_dim_, 0.0f);
  }
}

void KVClientOp::EmbRead(const base::RecTensor& keys, base::RecTensor& values) {
  std::lock_guard<std::mutex> lock(mtx_);
  const int64_t emb_dim  = values.shape(1);
  const int64_t num_keys = keys.shape(0);
  RECSTORE_LOG(2,
               "[INFO] [MOCK] EmbRead called: num_keys="
                   << num_keys << ", emb_dim=" << emb_dim
                   << ", keys ptr=" << keys.data_as<uint64_t>()
                   << ", values ptr=" << values.data_as<float>());
  if (num_keys == 0) {
    RECSTORE_LOG(3, "[DEBUG] [MOCK] EmbRead: num_keys==0, early return");
    return;
  }
  if (embedding_dim_ != -1 && embedding_dim_ != emb_dim) {
    RECSTORE_LOG(0,
                 "[ERROR] [MOCK] EmbRead: embedding_dim mismatch: "
                     << embedding_dim_ << " vs " << emb_dim);
    throw std::runtime_error(
        "KVClientOp Error: Inconsistent embedding dimension for read.");
  }
  const uint64_t* key_data = keys.data_as<uint64_t>();
  float* value_data        = values.data_as<float>();
  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key = key_data[i];
    auto it      = store_.find(key);
    if (it == store_.end()) {
      std::fill_n(value_data + i * emb_dim, emb_dim, 0.0f);
      RECSTORE_LOG(3,
                   "[DEBUG] [MOCK] EmbRead: key="
                       << key << " not found, filled with zeros");
    } else {
      if (it->second.size() != emb_dim) {
        RECSTORE_LOG(0,
                     "[ERROR] [MOCK] EmbRead: stored dim mismatch for key="
                         << key << ", stored=" << it->second.size()
                         << ", requested=" << emb_dim);
        throw std::runtime_error(
            "KVClientOp FATAL: Dimension mismatch for key " +
            std::to_string(key) +
            ". Stored dim: " + std::to_string(it->second.size()) +
            ", Requested dim: " + std::to_string(emb_dim));
      }
      std::copy(it->second.begin(), it->second.end(), value_data + i * emb_dim);
      RECSTORE_LOG(
          3, "[DEBUG] [MOCK] EmbRead: key=" << key << " read OK, values=[...]");
    }
  }
}

void KVClientOp::EmbWrite(const base::RecTensor& keys,
                          const base::RecTensor& values) {
  std::lock_guard<std::mutex> lock(mtx_);
  const int64_t emb_dim  = values.shape(1);
  const int64_t num_keys = keys.shape(0);
  RECSTORE_LOG(2,
               "[INFO] [MOCK] EmbWrite called: num_keys="
                   << num_keys << ", emb_dim=" << emb_dim
                   << ", keys ptr=" << keys.data_as<uint64_t>()
                   << ", values ptr=" << values.data_as<float>());
  if (embedding_dim_ == -1) {
    embedding_dim_ = emb_dim;
    RECSTORE_LOG(1,
                 "[WARNING] [MOCK] EmbWrite: embedding_dim_ inferred as "
                     << embedding_dim_);
    std::cout << "KVClientOp: Inferred and set embedding dimension to "
              << embedding_dim_ << std::endl;
  } else if (embedding_dim_ != emb_dim) {
    RECSTORE_LOG(0,
                 "[ERROR] [MOCK] EmbWrite: embedding_dim mismatch: "
                     << embedding_dim_ << " vs " << emb_dim);
    throw std::runtime_error(
        "KVClientOp Error: Inconsistent embedding dimension for write. "
        "Expected " +
        std::to_string(embedding_dim_) + ", but got " +
        std::to_string(emb_dim));
  }
  const uint64_t* key_data = keys.data_as<uint64_t>();
  const float* value_data  = values.data_as<float>();
  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key       = key_data[i];
    const float* start = value_data + i * emb_dim;
    const float* end   = start + emb_dim;
    store_[key].assign(start, end);
    RECSTORE_LOG(
        3, "[DEBUG] [MOCK] EmbWrite: key=" << key << " written, values=[...]");
  }
}

void KVClientOp::EmbUpdate(const base::RecTensor& keys,
                           const base::RecTensor& grads) {
  // std::lock_guard<std::mutex> lock(mtx_);
  // const int64_t emb_dim  = grads.shape(1);
  // const int64_t num_keys = keys.shape(0);
  // RECSTORE_LOG(2,
  //              "[INFO] [MOCK] EmbUpdate called: num_keys="
  //                  << num_keys << ", emb_dim=" << emb_dim
  //                  << ", keys ptr=" << keys.data_as<uint64_t>()
  //                  << ", grads ptr=" << grads.data_as<float>());
  // if (embedding_dim_ == -1) {
  //   embedding_dim_ = emb_dim;
  //   RECSTORE_LOG(1,
  //                "[WARNING] [MOCK] EmbUpdate: embedding_dim_ inferred as "
  //                    << embedding_dim_);
  // } else if (embedding_dim_ != emb_dim) {
  //   RECSTORE_LOG(0,
  //                "[ERROR] [MOCK] EmbUpdate: embedding_dim mismatch: "
  //                    << embedding_dim_ << " vs " << emb_dim);
  //   throw std::runtime_error(
  //       "KVClientOp Error: Inconsistent embedding dimension for update.");
  // }
  // const uint64_t* key_data = keys.data_as<uint64_t>();
  // const float* grad_data   = grads.data_as<float>();
  // for (int64_t i = 0; i < num_keys; ++i) {
  //   uint64_t key = key_data[i];
  //   auto it      = store_.find(key);
  //   if (it != store_.end()) {
  //     for (int64_t j = 0; j < emb_dim; ++j) {
  //       it->second[j] -= learning_rate_ * grad_data[i * emb_dim + j];
  //     }
  //     RECSTORE_LOG(
  //         3,
  //         "[DEBUG] [MOCK] EmbUpdate: key=" << key << " updated,
  //         grads=[...]");
  //   }
  // }
}

void KVClientOp::GetPretchResult(uint64_t prefetch_id, RecTensor& values) {
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = prefetch_results_.find(prefetch_id);
  if (it == prefetch_results_.end()) {
    throw std::runtime_error("Invalid prefetch_id or result already consumed");
  }
  const auto& rows = it->second;
  if (!EnsureEmbeddingOutput(values, static_cast<int64_t>(rows.size()))) {
    throw std::runtime_error("prefetch output tensor has invalid shape");
  }
  const int64_t D = values.shape(1);
  float* dst      = values.data_as<float>();
  if (dst != nullptr && !rows.empty()) {
    std::memset(dst,
                0,
                static_cast<size_t>(rows.size()) * static_cast<size_t>(D) *
                    sizeof(float));
  }
  for (size_t i = 0; i < rows.size(); ++i) {
    if (rows[i].empty()) {
      continue;
    }
    const int64_t copy_d =
        std::min<int64_t>(D, static_cast<int64_t>(rows[i].size()));
    std::memcpy(dst + i * static_cast<size_t>(D),
                rows[i].data(),
                static_cast<size_t>(copy_d) * sizeof(float));
  }
  prefetch_results_.erase(it);
}
uint64_t KVClientOp::EmbPrefetch(const base::RecTensor& keys,
                                 const base::RecTensor& values) {
  std::lock_guard<std::mutex> lock(mtx_);

  int64_t emb_dim = values.shape(1);
  // If caller passes empty dummy tensor (common in tests), fall back to known
  // dim.
  if (emb_dim == 0) {
    if (embedding_dim_ != -1) {
      emb_dim = embedding_dim_;
    } else if (!store_.empty()) {
      emb_dim = static_cast<int64_t>(store_.begin()->second.size());
    }
  }
  if (emb_dim <= 0) {
    throw std::runtime_error(
        "KVClientOp Error: Prefetch embedding dimension unknown. Call EmbWrite "
        "first or pass a non-empty values tensor.");
  }
  if (embedding_dim_ == -1) {
    embedding_dim_ = emb_dim;
  } else if (embedding_dim_ != emb_dim) {
    throw std::runtime_error(
        "KVClientOp Error: Inconsistent embedding dimension for prefetch.");
  }
  const int64_t num_keys   = keys.shape(0);
  const uint64_t* key_data = keys.data_as<uint64_t>();

  uint64_t id = next_prefetch_id_++;
  auto& out   = prefetch_results_[id];
  out.resize(num_keys);

  for (int64_t i = 0; i < num_keys; ++i) {
    uint64_t key = key_data[i];
    auto it      = store_.find(key);
    if (it == store_.end()) {
      out[i] = std::vector<float>(emb_dim, 0.0f);
    } else {
      out[i] = it->second;
      if ((int64_t)out[i].size() != emb_dim) {
        throw std::runtime_error("KVClientOp Error: Stored embedding dimension "
                                 "mismatch in prefetch.");
      }
    }
  }

  return id;
}
void KVClientOp::WaitForPrefetch(uint64_t prefetch_id) {
  // Prefetch is synchronous in mock; ensure id exists.
  std::lock_guard<std::mutex> lock(mtx_);
  if (prefetch_results_.find(prefetch_id) == prefetch_results_.end()) {
    throw std::runtime_error("Invalid prefetch_id");
  }
}
bool KVClientOp::IsWriteDone(uint64_t write_id) {
  throw std::runtime_error("Not impl");
}
bool KVClientOp::IsPrefetchDone(uint64_t prefetch_id) {
  std::lock_guard<std::mutex> lock(mtx_);
  return prefetch_results_.find(prefetch_id) != prefetch_results_.end();
}

bool KVClientOp::GetLocalLookupFlatPayloadRegion(const void** base,
                                                 std::size_t* bytes) {
  if (base != nullptr) {
    *base = nullptr;
  }
  if (bytes != nullptr) {
    *bytes = 0;
  }
  return false;
}

} // namespace recstore
#endif
