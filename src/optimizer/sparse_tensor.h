#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "../storage/kv_engine/base_kv.h"

#define TAG_TYPE uint8_t

enum TensorType { PARAMETER = 0, MOMENT_1 = 1, MOMENT_2 = 2 };

// ponytail: 8-bit tag, widen TAG_TYPE if table_id * tensors_per_table > 255
inline constexpr int KeyTagShift() {
  return static_cast<int>(sizeof(uint64_t) * 8) -
         static_cast<int>(sizeof(TAG_TYPE) * 8);
}

inline TAG_TYPE ExtractKeyTag(uint64_t key) {
  return static_cast<TAG_TYPE>(key >> KeyTagShift());
}

inline TAG_TYPE MakeTensorTag(uint64_t table_id,
                              int role,
                              int tensors_per_table) {
  if (tensors_per_table <= 0 || role < 0 || role >= tensors_per_table) {
    throw std::runtime_error("invalid embedding tensor tag role");
  }
  const uint64_t tag = table_id * static_cast<uint64_t>(tensors_per_table) +
                       static_cast<uint64_t>(role);
  if (tag > std::numeric_limits<TAG_TYPE>::max()) {
    throw std::runtime_error("embedding table tag overflow");
  }
  return static_cast<TAG_TYPE>(tag);
}

class SparseTensor {
private:
  std::string name;
  TensorType type;
  TAG_TYPE tag;
  std::vector<uint64_t> shape;
  BaseKV* kv;
  uint64_t concatKeyAndTag(uint64_t key, TAG_TYPE tag);

public:
  SparseTensor() = default;
  void init(std::string& name,
            TensorType type,
            TAG_TYPE tag,
            std::vector<uint64_t>& shape,
            BaseKV* kv);
  void Get(const uint64_t key, std::string& value, unsigned tid);
  void Put(const uint64_t key, const std::string_view& value, unsigned tid);
  void BatchGet(const std::vector<uint64_t>& keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid);
  int64_t EmbeddingDim() const;
  TAG_TYPE Tag() const { return tag; }
  bool ApplySgdUpdateFlat(
      const base::ConstArray<uint64_t>& keys,
      const float* grads,
      int64_t num_rows,
      int64_t embedding_dim,
      float learning_rate,
      unsigned tid);
};
