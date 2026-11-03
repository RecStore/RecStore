#include "sparse_tensor.h"

uint64_t SparseTensor::concatKeyAndTag(uint64_t key, TAG_TYPE tag) {
  constexpr int tag_bits = sizeof(TAG_TYPE) * 8;
  constexpr int shift    = (sizeof(uint64_t) * 8) - tag_bits;
  key &= (~0ULL >> tag_bits);
  return (static_cast<uint64_t>(tag) << shift) | key;
}

void SparseTensor::init(std::string& name,
                        TensorType type,
                        TAG_TYPE tag,
                        std::vector<uint64_t>& shape,
                        BaseKV* kv) {
  this->name  = name;
  this->type  = type;
  this->tag   = tag;
  this->shape = shape;
  this->kv    = kv;
}

void SparseTensor::Get(const uint64_t key, std::string& value, unsigned tid) {
  auto _key = concatKeyAndTag(key, tag);
  kv->Get(_key, value, tid);
}

void SparseTensor::Put(
    const uint64_t key, const std::string_view& value, unsigned tid) {
  auto _key = concatKeyAndTag(key, tag);
  kv->Put(_key, value, tid);
}

void SparseTensor::BatchGet(const std::vector<uint64_t>& keys,
                            std::vector<base::ConstArray<float>>* values,
                            unsigned tid) {
  std::vector<uint64_t> hashed_keys;
  hashed_keys.reserve(keys.size());
  for (auto k : keys) {
    hashed_keys.push_back(concatKeyAndTag(k, tag));
  }
  kv->BatchGet(base::ConstArray<uint64_t>(hashed_keys), values, tid);
}

int64_t SparseTensor::EmbeddingDim() const {
  if (shape.size() < 2) {
    return 0;
  }
  return static_cast<int64_t>(shape[1]);
}

bool SparseTensor::ApplySgdUpdateFlat(
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim,
    float learning_rate,
    unsigned tid,
    BaseKV::UpdateFlatStats* stats) {
  return kv != nullptr &&
         kv->ApplySgdUpdateFlat(
             keys,
             grads,
             num_rows,
             embedding_dim,
             learning_rate,
             tag,
             tid,
             stats);
}
