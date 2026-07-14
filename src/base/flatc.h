#pragma once
#include <string>
#include <type_traits>
#include <cstdint>
#include <vector>

#pragma pack(push, 1)
template <typename ItemT>
struct FlatItemCompressReader {
  int size;
  int item_size() const { return size; }
  const ItemT* item(int i) const {
    return reinterpret_cast<const ItemT*>(values() + offset(i));
  }
  const char* values() const {
    return reinterpret_cast<const char*>(offsets() + size);
  }
  const int* offsets() const { return reinterpret_cast<const int*>(this + 1); }
  int offset(int i) const { return i > 0 ? offsets()[i - 1] : 0; }
  int byte_size() const {
    return sizeof(FlatItemCompressReader) + size * sizeof(int) + offset(size);
  }
  bool Valid(int expect_bytes) const {
    return expect_bytes >= sizeof(FlatItemCompressReader) &&
           byte_size() == expect_bytes;
  }
};
static_assert(sizeof(FlatItemCompressReader<int>) == sizeof(int), "size check");

template <typename ValueT>
struct FlatKVCompressReader {
  typedef FlatItemCompressReader<ValueT> ValueReader;

  int size;

  uint64_t key(int i) const { return keys()[i]; }
  const ValueT* value(int i) const { return value_reader()->item(i); }
  int item_size() const { return size; }
  int byte_size() const {
    return sizeof(FlatKVCompressReader) + size * sizeof(uint64_t) +
           value_reader()->byte_size();
  }
  bool Valid(int expect_bytes) const {
    return expect_bytes >= sizeof(FlatKVCompressReader) &&
           byte_size() == expect_bytes &&
           this->item_size() == value_reader()->item_size();
  }

  const uint64_t* keys() const {
    return reinterpret_cast<const uint64_t*>(this + 1);
  }
  const ValueReader* value_reader() const {
    return reinterpret_cast<const ValueReader*>(keys() + size);
  }
};
static_assert(sizeof(FlatKVCompressReader<int>) == sizeof(int), "size check");

#pragma pack(pop)

template <typename ItemT>
struct Pack {
  static constexpr const bool implemented = false;
};

// Item 的一些操作可能是不确定的，需要用户自己定制
template <typename ItemT>
struct FlatItemDetail {
  // 返回 item 大小，有些 item 可能是变长的，这里通过这个接口拿到 size 信息，
  // 默认 item 的大小是通过 item.byte_size() 来拿，如果不是需要模版特化一下
  static int ByteSize(const ItemT& item) { return item.byte_size(); }
  // Compress ItemT 自身，默认直接 memory copy
  static void CompressAppend(const ItemT& item, std::string* output) {
    auto item_head = reinterpret_cast<const char*>(&item);
    output->append(item_head, ByteSize(item));
  }
  // 如果 ItemT 是从其他的结构得到，可以模版特化 Pack<ItemT>，实现
  // CompressAppend
  static void CompressAppend(const Pack<ItemT>& pack, std::string* output) {
    static_assert(Pack<ItemT>::implemented,
                  "compress from item pack, but not implemented");
    pack.CompressAppend(output);
  }
};

template <typename ItemT>
class FlatItemCompressor {
public:
  explicit FlatItemCompressor(int block_size = kDefaultBlockSize)
      : block_size_(block_size) {
    Clear();
  }

  void Clear() {
    offsets_.resize(1);
    item_data_.clear();
  }

  // Pre-allocate internal buffers to avoid repeated reallocation when the
  // number of items and their total byte size are known in advance.
  void Reserve(int item_count, int estimated_data_bytes) {
    offsets_.reserve(static_cast<size_t>(item_count) + 1);
    item_data_.reserve(static_cast<size_t>(estimated_data_bytes));
  }

  template <typename RawItemT>
  int AddItem(const RawItemT& raw_item, std::vector<std::string>* blocks) {
    FlatItemDetail<ItemT>::CompressAppend(raw_item, &item_data_);
    offsets_.push_back(item_data_.size());
    if (blocks && byte_size() >= block_size_) {
      return ToBlock(blocks);
    } else {
      return 0;
    }
  }

  template <typename RawItemT>
  void BatchAddItems(const std::vector<RawItemT>& items) {
    for (const auto& item : items) {
      FlatItemDetail<ItemT>::CompressAppend(item, &item_data_);
      offsets_.push_back(item_data_.size());
    }
  }
  int ToBlock(std::vector<std::string>* blocks) {
    std::string* pblock = nullptr;
    int size            = 0;
    for (; size < blocks->size(); ++size) {
      if (blocks->at(size).empty()) {
        pblock = &blocks->at(size);
        break;
      }
    }
    if (offsets_.size() < 2) {
      return size;
    }
    if (!pblock) {
      blocks->emplace_back();
      pblock = &blocks->back();
    }
    ToBlock(pblock);
    return size + 1;
  }
  void ToBlock(std::string* block) {
    if (offsets_.size() < 2)
      return;
    block->clear();
    AppendToBlock(block);
  }
  void AppendToBlock(std::string* block) {
    if (offsets_.size() < 2)
      return;
    offsets_[0] = offsets_.size() - 1;
    block->append(reinterpret_cast<const char*>(&offsets_[0]),
                  offsets_.size() * sizeof(int));
    block->append(item_data_.data(), item_data_.size());
    Clear();
  }

  template <typename BufferT>
  void AppendToIOBuf(BufferT* buf) {
    if (offsets_.size() < 2)
      return;
    offsets_[0] = offsets_.size() - 1;
    buf->append(reinterpret_cast<const char*>(&offsets_[0]),
                offsets_.size() * sizeof(int));
    buf->append(item_data_.data(), item_data_.size());
    Clear();
  }

  int byte_size() const {
    return offsets_.size() * sizeof(int) + item_data_.size();
  }
  int ByteSize(const ItemT& item) {
    return FlatItemDetail<ItemT>::ByteSize(item);
  }

  static constexpr const int kDefaultBlockSize = (1 << 20) * 16;

protected:
  int block_size_ = 0;
  std::vector<int> offsets_;
  std::string item_data_;
};

template <typename ValueT>
class FlatKVCompressor {
public:
  explicit FlatKVCompressor(int block_size = kDefaultBlockSize)
      : block_size_(block_size) {}

  void
  AddItem(uint64_t key, const ValueT& value, std::vector<std::string>* blocks) {
    keys_.push_back(key);
    value_compressor_.AddItem(value, nullptr);
    if (byte_size() >= block_size_)
      ToBlock(blocks);
  }

  int byte_size() const {
    return sizeof(int) + sizeof(uint64_t) * keys_.size() +
           value_compressor_.byte_size();
  }

  void ToBlock(std::vector<std::string>* blocks) {
    if (keys_.size() < 1)
      return;
    blocks->emplace_back();
    auto new_block = &blocks->back();
    int key_num    = keys_.size();
    new_block->append(reinterpret_cast<const char*>(&key_num), sizeof(int));
    new_block->append(reinterpret_cast<const char*>(&keys_[0]),
                      keys_.size() * sizeof(uint64_t));
    value_compressor_.AppendToBlock(new_block);
    Clear();
  }

  template <typename BufferT>
  void AppendToIOBuf(BufferT* buf) {
    if (keys_.size() < 1)
      return;
    int key_num = keys_.size();
    buf->append(reinterpret_cast<const char*>(&key_num), sizeof(int));
    buf->append(reinterpret_cast<const char*>(&keys_[0]),
                keys_.size() * sizeof(uint64_t));
    value_compressor_.AppendToIOBuf(buf);
    Clear();
  }

  void Clear() { keys_.clear(); }

  static constexpr const int kDefaultBlockSize = (1 << 20) * 16;

private:
  int block_size_ = 0;
  std::vector<uint64_t> keys_;
  FlatItemCompressor<ValueT> value_compressor_;
};
