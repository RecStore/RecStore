#pragma once

#include <atomic>
#include <boost/coroutine2/all.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <folly/Format.h>
#include <folly/GLog.h>
#include <folly/Likely.h>
#include <glog/logging.h>
#include <sys/user.h>
#include <vector>
#include "../index/index.h"

using boost::coroutines2::coroutine;
extern thread_local int pending;
extern thread_local std::vector<std::unique_ptr<coroutine<void>::pull_type>>
    coros;

typedef uint64_t PageID_t;
const PageID_t INVALID_PAGE = -1;

class IOBackend {
public:
  IOBackend(const BaseKVConfig& config) {
    if (!config.json_config_.contains("page_id_offset"))
      LOG(WARNING) << "IOBackend config missing 'page_id_offset'";
    if (!config.json_config_.contains("queue_cnt"))
      LOG(WARNING) << "IOBackend config missing 'queue_cnt'";
    next_page_id.store(config.json_config_.value("page_id_offset", (PageID_t)1),
                       std::memory_order_relaxed);
    queue_cnt = config.json_config_.value("queue_cnt", 512);
  }
  virtual ~IOBackend() {}
  virtual void init() = 0;

  PageID_t AllocatePage(coroutine<void>::push_type& sink, uint64_t index) {
    PageID_t new_page_id = next_page_id.fetch_add(1, std::memory_order_relaxed);
    WritePageAsync(sink, index, new_page_id, empty_page);
    return new_page_id;
  }
  PageID_t AllocatePage() {
    PageID_t new_page_id = next_page_id.fetch_add(1, std::memory_order_relaxed);
    WritePageSync(new_page_id, empty_page);
    return new_page_id;
  }
  PageID_t GetNextPageID() const {
    return next_page_id.load(std::memory_order_relaxed);
  }
  void SetNextPageID(PageID_t page_id) {
    next_page_id.store(page_id, std::memory_order_relaxed);
  }

  void ReadPage(coroutine<void>::push_type& sink,
                uint64_t index,
                PageID_t page_id,
                char* buffer) {
    ReadPageAsync(sink, index, page_id, buffer);
  }
  void ReadPage(PageID_t page_id, char* buffer) {
    ReadPageSync(page_id, buffer);
  }

  void WritePage(coroutine<void>::push_type& sink,
                 uint64_t index,
                 PageID_t page_id,
                 char* buffer) {
    WritePageAsync(sink, index, page_id, buffer);
  }
  void WritePage(PageID_t page_id, char* buffer) {
    WritePageSync(page_id, buffer);
  }

  virtual void* GetPage(
      coroutine<void>::push_type& sink, uint64_t index, PageID_t page_id) = 0;
  virtual void* GetPage(PageID_t page_id)                                 = 0;

  // Unpin a page, if dirty, write it back
  virtual void
  Unpin(coroutine<void>::push_type& sink,
        uint64_t index,
        PageID_t page_id,
        void* page_data,
        bool is_dirty)                                                 = 0;
  virtual void Unpin(PageID_t page_id, void* page_data, bool is_dirty) = 0;

  virtual char* AllocateBuffer() = 0;
  // Allocate a contiguous buffer of page_count pages (zero-filled).
  virtual char* AllocateBuffer(uint64_t page_count) = 0;
  virtual void FreeBuffer(char* buf)                = 0;

  virtual void submit()         = 0;
  virtual void PollCompletion() = 0;

  struct IOEntry {
    PageID_t page_id;    // starting page
    char* buffer;        // contiguous buffer
    uint64_t page_count; // number of pages in this buffer
  };

  // Batch write using sliding-window async IO (no coroutines).
  // Buffers are NOT freed by this method; caller is responsible.
  virtual void BatchWritePages(const std::vector<IOEntry>& entries) {
    for (auto& e : entries) {
      for (uint64_t i = 0; i < e.page_count; i++) {
        WritePageSync(e.page_id + i, e.buffer + i * PAGE_SIZE);
      }
    }
  }

  // Batch read using sliding-window async IO (no coroutines).
  // Buffers are NOT freed by this method; caller is responsible.
  virtual void BatchReadPages(const std::vector<IOEntry>& entries) {
    for (auto& e : entries) {
      for (uint64_t i = 0; i < e.page_count; i++) {
        ReadPageSync(e.page_id + i, e.buffer + i * PAGE_SIZE);
      }
    }
  }

protected:
  std::atomic<PageID_t> next_page_id{1};
  char* empty_page = nullptr;
  int queue_cnt    = 512;

  virtual void ReadPageAsync(coroutine<void>::push_type& sink,
                             uint64_t index,
                             PageID_t page_id,
                             char* buffer)                  = 0;
  virtual void ReadPageSync(PageID_t page_id, char* buffer) = 0;

  virtual void WritePageAsync(coroutine<void>::push_type& sink,
                              uint64_t index,
                              PageID_t page_id,
                              char* buffer)                  = 0;
  virtual void WritePageSync(PageID_t page_id, char* buffer) = 0;
};
