#pragma once

#include "../../io_backend/io_backend.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include "base/factory.h"
#include "../index.h"

class CCEH;
struct Directory;
struct Segment;

// Constants to make Segment fit in a 4KB page
// Original kNumSlot was 1024.
// sizeof(Pair) = 16. sizeof(sema) = 8. sizeof(local_depth) = 8.
// 4096 - 16 = 4080 for bucket. 4080 / 16 = 255.
// Let's use 252 to make it divisible by kNumPairPerCacheLine.
constexpr size_t kNumPairPerCacheLine = 4;
constexpr size_t kNumCacheLine =
    8; // For probing distance, not directly for size
constexpr size_t kCuckooThreshold = 16;
// constexpr size_t kCuckooThreshold = 32;

struct Segment {
  static const size_t kNumSlot   = (PAGE_SIZE - sizeof(size_t)) / sizeof(Pair);
  static const size_t kNumGroups = kNumSlot / kNumPairPerCacheLine; // 63

  Segment(void) {}
  ~Segment(void) {}

  void initSegment(void) {
    for (int i = 0; i < kNumSlot; ++i)
      bucket[i].key = INVALID;
    local_depth = 0;
  }

  void initSegment(size_t depth) {
    for (int i = 0; i < kNumSlot; ++i)
      bucket[i].key = INVALID;
    local_depth = depth;
  }

  bool Insert4split(Key_t&, Value_t, size_t);
  PageID_t* Split(coroutine<void>::push_type& sink, int index, IOBackend*);
  PageID_t* Split(IOBackend*);
  std::vector<std::pair<size_t, size_t>> find_path(size_t, size_t);
  void execute_path(std::vector<std::pair<size_t, size_t>>&, Key_t&, Value_t);
  void execute_path(std::vector<std::pair<size_t, size_t>>&, Pair);
  size_t numElement(void);

  Pair bucket[kNumSlot];
  size_t local_depth;
  char padding[PAGE_SIZE - sizeof(Pair) * kNumSlot - sizeof(size_t)];
};

// To fit Directory in a 4KB page, it is replaced by a two-level structure:
// DirectoryHeader -> DirectoryPage -> Segment
// A DirectoryPage is a page containing pointers to Segments.
struct DirectoryPage {
  static const size_t kNumPointers = PAGE_SIZE / sizeof(PageID_t);
  PageID_t segments[kNumPointers];
};

// DirectoryHeader is the entry point of the directory.
// It contains pointers to all DirectoryPages.
struct DirectoryHeader {
  // To fit DirectoryHeader in a 4KB page.
  // metadata = 16 bytes. 4096-16 = 4080 bytes for segments.
  // 4080 / sizeof(PageID_t) = 4080 / 8 = 510
  static const size_t kMaxDirectoryPages =
      (PAGE_SIZE - sizeof(size_t) * 2) / sizeof(PageID_t);

  size_t depth;
  size_t capacity;
  PageID_t dir_pages[kMaxDirectoryPages];

  DirectoryHeader(void) {}
  ~DirectoryHeader(void) {}

  void initDirectory(size_t _depth) {
    depth    = _depth;
    capacity = pow(2, _depth);
    if ((capacity / DirectoryPage::kNumPointers) > kMaxDirectoryPages)
      capacity = kMaxDirectoryPages * DirectoryPage::kNumPointers;
  }

  void initDirectory(void) {
    // some default depth
    size_t _depth = 10;
    depth         = _depth;
    capacity      = pow(2, _depth);
    if ((capacity / DirectoryPage::kNumPointers) > kMaxDirectoryPages)
      capacity = kMaxDirectoryPages * DirectoryPage::kNumPointers;
  }
};

class CCEH : public Index {
public:
  CCEH(const BaseKVConfig& config);

  void Put(coroutine<void>::push_type& sink,
           int index,
           Key_t,
           Value_t,
           unsigned tid) override;
  void Put(Key_t, Value_t, unsigned tid) override;
  bool Delete(Key_t&) override;
  void Get(coroutine<void>::push_type& sink,
           int index,
           Key_t,
           Value_t&,
           unsigned tid) override;
  void Get(Key_t, Value_t&, unsigned tid) override;
  void BatchPut(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override;
  void BatchGet(base::ConstArray<Key_t> keys,
                Value_t* pointers,
                unsigned tid) override;

  bool crashed = true;
  std::unique_ptr<IOBackend> io_backend;

private:
  void initCCEH(size_t);
  std::shared_mutex& get_segment_lock(PageID_t page_id) const;

  PageID_t dir_header_page_id;
  mutable std::shared_mutex dir_mutex;
  mutable std::mutex segment_locks_mutex;
  mutable std::unordered_map<PageID_t, std::unique_ptr<std::shared_mutex>>
      segment_locks;

  Value_t FindAnyway(const Key_t&);
};

FACTORY_REGISTER(Index, SSD, CCEH, const BaseKVConfig&);