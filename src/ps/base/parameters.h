#pragma once
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <vector>

#include "base/flatc.h"

#pragma pack(push, 1)
struct ParameterCompressItem {
  uint64_t key;
  int dim;

  static int GetSize(int dim) {
    return sizeof(ParameterCompressItem) + dim * sizeof(float);
  }
  const float* data() const { return embedding; }

  int byte_size() const {
    return sizeof(ParameterCompressItem) + dim * sizeof(float);
  }
  float embedding[0]; // this must be the tail
};
#pragma pack(pop)

template <>
struct Pack<ParameterCompressItem> {
  static constexpr const bool implemented = true;
  uint64_t key                            = 0;
  int dim                                 = 0;
  const float* emb_data                   = nullptr;
  Pack<ParameterCompressItem>()           = default;
  Pack<ParameterCompressItem>(uint64_t key, int dim, const float* emb_data)
      : key(key), dim(dim), emb_data(emb_data) {}
  void CompressAppend(std::string* output) const;
};

using ParameterPack           = Pack<ParameterCompressItem>;
using ParameterCompressor     = FlatItemCompressor<ParameterCompressItem>;
using ParameterCompressReader = FlatItemCompressReader<ParameterCompressItem>;

inline void CopyCompressItemsToFlat(const ParameterCompressReader* reader,
                                    float* dst,
                                    int64_t embedding_dim,
                                    size_t dst_row_offset = 0) {
  if (reader == nullptr || dst == nullptr || embedding_dim <= 0) {
    return;
  }
  for (int index = 0; index < reader->item_size(); ++index) {
    auto item = reader->item(index);
    float* row =
        dst + (dst_row_offset + static_cast<size_t>(index)) *
                  static_cast<size_t>(embedding_dim);
    if (item->dim == 0) {
      std::memset(row, 0, static_cast<size_t>(embedding_dim) * sizeof(float));
      continue;
    }
    const int64_t copy_d =
        std::min<int64_t>(embedding_dim, static_cast<int64_t>(item->dim));
    std::memcpy(row, item->embedding, static_cast<size_t>(copy_d) * sizeof(float));
    if (copy_d < embedding_dim) {
      std::memset(row + copy_d,
                  0,
                  static_cast<size_t>(embedding_dim - copy_d) * sizeof(float));
    }
  }
}

inline void CompressEmbeddingRows(ParameterCompressor* compressor,
                                  const uint64_t* keys,
                                  const float* values,
                                  size_t n,
                                  int64_t embedding_dim,
                                  std::vector<std::string>* blocks = nullptr) {
  for (size_t i = 0; i < n; ++i) {
    ParameterPack pack;
    pack.key      = keys[i];
    pack.dim      = static_cast<int>(embedding_dim);
    pack.emb_data = values + i * embedding_dim;
    compressor->AddItem(pack, blocks);
  }
}