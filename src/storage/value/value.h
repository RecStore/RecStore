#pragma once

#include "../io_backend/io_backend.h"
#include "../../memory/allocators/allocator_factory.h"
#include "../../memory/memory_factory.h"
#include "pointer.h"

#include "storage/kv_engine/base_kv.h"
#include <string>
#include <string_view>
#include <vector>
#include <limits>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <list>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <tbb/concurrent_hash_map.h>
#include <array>
#include <functional>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/uio.h>

template <typename Key>
class SimpleLRUCache {
public:
  explicit SimpleLRUCache(size_t capacity) : capacity_(capacity) {}

  void insert(const Key& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      order_.splice(order_.begin(), order_, it->second);
    } else {
      if (order_.size() >= capacity_) {
        Key lru_key = order_.back();
        order_.pop_back();
        map_.erase(lru_key);
      }
      order_.push_front(key);
      map_[key] = order_.begin();
    }
  }

  std::vector<Key> evictBatch(size_t batch_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Key> evicted_keys;
    if (batch_size > 0) {
      evicted_keys.reserve(batch_size);
      for (size_t i = 0; i < batch_size && !order_.empty(); ++i) {
        Key lru_key = order_.back();
        order_.pop_back();
        map_.erase(lru_key);
        evicted_keys.push_back(lru_key);
      }
    }
    return evicted_keys;
  }

private:
  std::mutex mutex_;
  size_t capacity_;
  std::list<Key> order_;
  std::unordered_map<Key, typename std::list<Key>::iterator> map_;
};

class ValueManager {
public:

  ValueManager(const std::string& shm_file_path,
               size_t shm_capacity,
               const std::string& ssd_file_path,
               size_t ssd_capacity,
               const BaseKVConfig& config)
      : lru_(std::max<size_t>(4096, shm_capacity / 256)),
        shm_capacity_bytes_(shm_capacity) {
    json dram_cfg = config.json_config_;
    json ssd_cfg  = config.json_config_;
    if (config.json_config_.contains("dram_allocator_type"))
      dram_cfg["allocator_type"] = config.json_config_.at("dram_allocator_type");
    if (config.json_config_.contains("ssd_allocator_type"))
      ssd_cfg["allocator_type"] = config.json_config_.at("ssd_allocator_type");
    if (config.json_config_.contains("dram_value_memory_management"))
      dram_cfg["value_memory_management"] = config.json_config_.at("dram_value_memory_management");
    if (config.json_config_.contains("ssd_value_memory_management"))
      ssd_cfg["value_memory_management"] = config.json_config_.at("ssd_value_memory_management");

    shm_manage = base::allocators::CreateAllocator(dram_cfg, shm_file_path, shm_capacity, "DRAM");


      uint64_t ssd_value_offset = 1; // io_uring: start at page 1 (page 0 unused)
        if (config.json_config_.at("io_backend_type").get<std::string>() == "SPDK") {
          ssd_value_offset = config.json_config_.value("spdk_value_offset", (uint64_t)1000000);
        }
        BaseKVConfig ssd_config                   = config;
        ssd_config.json_config_["file_path"]      = ssd_file_path;
        ssd_config.json_config_["page_id_offset"] = ssd_value_offset;

    using IOF = base::Factory<IOBackend, const BaseKVConfig&>;
    value_io_backend.reset(IOF::NewInstance(ssd_config.json_config_.at("io_backend_type").get<std::string>(), ssd_config));
    value_io_backend->init();
    
    if (!shm_manage || !value_io_backend)
      throw std::runtime_error("failed to initialize hybrid value allocators");
    
   
    evictor_ = std::thread([this] { this->EvictAndPromoteLoop(); });
  }

  ValueManager(const ValueManager&)            = delete;
  ValueManager& operator=(const ValueManager&) = delete;

  ~ValueManager() {
    stop_.store(true, std::memory_order_release);
    evict_cv_.notify_all();
    if (evictor_.joinable())
      evictor_.join();
  }

  UnifiedPointer WriteValue(const std::string_view& value) {

    UnifiedPointer p_write = WriteMem(value);
    if (!p_write) {
      {
        std::lock_guard<std::mutex> lk(evict_mu_);
        need_evict_ = true;
      }
      evict_cv_.notify_one();
      {
        std::unique_lock<std::mutex> lk(evict_mu_);
        evict_cv_.wait_for(lk, std::chrono::milliseconds(5), [&] {
          return stop_.load() || shm_used_bytes_.load() <= HighWatermark();
        });
      }
      if (stop_.load())
        return UnifiedPointer();
      p_write = WriteMem(value);
    }
    if (!p_write)
      p_write = WriteDisk(value);
    if (!p_write) {
      LOG(ERROR) << "Failed to allocate memory ";
      return UnifiedPointer();
    }
    else{
      return p_write;
    }
  }
  std::string RetrieveValue(uint64_t old_raw, unsigned tid) {
    UnifiedPointer p = UnifiedPointer::FromRaw(old_raw);
    switch (p.type()) {
    case UnifiedPointer::Type::Memory: {
      auto v = ReadFromMemRaw(p); // 不触 LRU
      //TouchLRU(key);              // 业务访问才触碰
      return v;
    }
    case UnifiedPointer::Type::Disk: {
      auto v = ReadFromDiskNoHeat(p); // 不升温
      // 1/8 抽样计数，降低热点写入压力
      // if ((std::hash<Key_t>{}(key) & 0x7) == 0) {
      //   tbb::concurrent_hash_map<Key_t, uint8_t>::accessor acc;
      //   if (hot_.insert(acc, key))
      //     acc->second = 0;
      //   if (acc->second < kPromoteThreshold)
      //     acc->second++;
      // }
      return v; // 不触 LRU，避免污染
    }
    case UnifiedPointer::Type::PMem: {
      uint64_t raw_val = p.asPMemOffset();
      uint8_t len      = static_cast<uint8_t>(raw_val >> 54);
      char buf[8];
      uint64_t data_val = raw_val & ((1ULL << 54) - 1);
      std::memcpy(buf, &data_val, sizeof(uint64_t));
      return std::string(buf, len);
    }
    default:
      return {};
    }
  }

private:
  // 可选：提高条带锁数量，减轻高并发读写时的条带争用
  static constexpr size_t kNumLocks = 1024; // 原为 256
  mutable std::array<std::shared_mutex, kNumLocks> locks_;

  std::shared_mutex& get_lock_for_key(Key_t key) const {
    return locks_[std::hash<Key_t>{}(key) & (kNumLocks - 1)];
  }

  std::unique_ptr<base::MallocApi> shm_manage;
  std::unique_ptr<IOBackend> value_io_backend;

  SimpleLRUCache<Key_t> lru_;
  tbb::concurrent_hash_map<Key_t, uint8_t> hot_;

  // 更保守的热点阈值，减少短期偶发访问的提升
  static constexpr uint8_t kPromoteThreshold = 8; // 原为 2

  std::atomic<bool> stop_{false};

  const size_t shm_capacity_bytes_;
  std::atomic<size_t> shm_used_bytes_{0};
  std::thread evictor_;
  std::condition_variable evict_cv_;
  std::mutex evict_mu_;
  bool need_evict_{false};

  // 水位控制
  static constexpr double kHighWatermarkRatio = 0.85;
  static constexpr double kLowWatermarkRatio  = 0.70;
  size_t HighWatermark() const {
    return (size_t)(shm_capacity_bytes_ * kHighWatermarkRatio);
  }
  size_t LowWatermark() const {
    return (size_t)(shm_capacity_bytes_ * kLowWatermarkRatio);
  }

  inline void TouchLRU(Key_t key) { lru_.insert(key); }

  inline uint16_t PeekLenMem(const UnifiedPointer& p) const {
    const uint8_t* bytes = static_cast<const uint8_t*>(p.asMemoryPointer());
    return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
  }

  inline std::string ReadFromMemRaw(const UnifiedPointer& p) const {
    const uint8_t* bytes = static_cast<const uint8_t*>(p.asMemoryPointer());
    uint16_t len         = static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
    return std::string(reinterpret_cast<const char*>(bytes + 2), len);
  }

  inline std::string ReadFromDiskNoHeat(const UnifiedPointer& p) const {
    PageID_t start_page_id = p.asDiskPageId();
    uint16_t page_count = p.asDiskPageCount();
    std::string value;
    uint32_t actual_size = 0;
    uint32_t written     = 0;
    for (int i = 0; i < page_count; i++) {
      PageID_t pid = start_page_id + i;
      char* buffer = (char*)value_io_backend->GetPage(pid);
      if (i == 0) {
        memcpy(&actual_size, buffer, 4);
        value.resize(actual_size);
        uint32_t data_in_page =
            std::min((uint32_t)(PAGE_SIZE - 4), actual_size);
        memcpy(value.data(), buffer + 4, data_in_page);
        written += data_in_page;
      } else {
        uint32_t data_in_page =
            std::min((uint32_t)PAGE_SIZE, actual_size - written);
        memcpy(value.data() + written, buffer, data_in_page);
        written += data_in_page;
      }
      value_io_backend->Unpin(pid, buffer, false);
    }
    return value;
  }

  void FreePointer(const UnifiedPointer& p) {
    if (!p)
      return;
    if (p.type() == UnifiedPointer::Type::PMem)
      return; // inline 小值无需释放

    if (p.type() == UnifiedPointer::Type::Memory) {
      void* mem_ptr     = p.asMemoryPointer();
      uint16_t data_len = 0;
      std::memcpy(&data_len, mem_ptr, sizeof(data_len));
      shm_manage->Free(mem_ptr);
      shm_used_bytes_.fetch_sub(
          (size_t)data_len + 2, std::memory_order_relaxed);
    } else if (p.type() == UnifiedPointer::Type::Disk) {
      LOG(INFO) << "Not implemented";
    }
  }

  UnifiedPointer WriteMem(const std::string_view& value) {
    if (value.size() > 0 && value.size() <= 7) { //足够小直接写进指针里
      uint64_t inline_val = 0;
      std::memcpy(&inline_val, value.data(), value.size());
      uint64_t final_val = (static_cast<uint64_t>(value.size()) << 54) |
                           (inline_val & ((1ULL << 54) - 1));
      return UnifiedPointer::FromPMem(final_val);
    }

    if (value.size() > std::numeric_limits<uint16_t>::max())
      return UnifiedPointer();
    const uint16_t data_len = static_cast<uint16_t>(value.size());
    const int total_size    = static_cast<int>(sizeof(data_len) + data_len);
    char* ptr               = shm_manage->New(total_size);
    if (!ptr)
      return UnifiedPointer();
    uint8_t lenle[2] = {
        (uint8_t)(data_len & 0xFF), (uint8_t)((data_len >> 8) & 0xFF)};
    std::memcpy(ptr, lenle, 2);
    if (data_len)
      std::memcpy(ptr + 2, value.data(), data_len);
    shm_used_bytes_.fetch_add((size_t)total_size, std::memory_order_relaxed);
    // 这里有个lru_(key)先删除了 index应该在上层
    return UnifiedPointer::FromMemory(ptr);
  }

  UnifiedPointer WriteDisk(const std::string_view& value) {
    uint32_t actual_size   = (uint32_t)value.size();
    uint16_t pages_needed  = (actual_size + 4 + PAGE_SIZE - 1) / PAGE_SIZE;
    PageID_t start_page_id = 0;
    uint32_t written       = 0;
    for (int i = 0; i < pages_needed; i++) {
      PageID_t page_id = value_io_backend->AllocatePage();
      if (i == 0)
        start_page_id = page_id;
      char* buffer = (char*)value_io_backend->GetPage(page_id);
      if (i == 0) {
        memcpy(buffer, &actual_size, 4);
        uint32_t data_in_page =
            std::min((uint32_t)(PAGE_SIZE - 4), actual_size);
        memcpy(buffer + 4, value.data(), data_in_page);
        written += data_in_page;
      } else {
        uint32_t data_in_page =
            std::min((uint32_t)PAGE_SIZE, actual_size - written);
        memcpy(buffer, value.data() + written, data_in_page);
        written += data_in_page;
      }
      value_io_backend->Unpin(page_id, buffer, true);
    }
    return UnifiedPointer::FromDisk(start_page_id, pages_needed);
  }

  void EvictAndPromoteLoop() {
    const size_t kEvictionBatchSize = 512; // 原 64
    const size_t kPromoteScanBudget = 512; // 每轮最多扫描热点条目数
    const size_t kDecayPerRound     = 256; // 每轮轻度衰减项数

    while (!stop_.load(std::memory_order_acquire)) {
      // {
      //   std::unique_lock<std::mutex> lk(evict_mu_);
      //   evict_cv_.wait_for(lk, std::chrono::milliseconds(20), [&] {
      //     return stop_.load() || need_evict_ ||
      //            shm_used_bytes_.load() > HighWatermark();
      //   });
      // }
      // if (stop_.load())
      //   break;

      // // 1) 先逐出到低水位以下，避免在高水位阶段还去做提升
      // if (shm_used_bytes_.load(std::memory_order_relaxed) > LowWatermark()) {
      //   std::vector<Key_t> victim_keys = lru_.evictBatch(kEvictionBatchSize);
      //   for (const auto& victim_key : victim_keys) {
      //     if (shm_used_bytes_.load(std::memory_order_relaxed) <= LowWatermark())
      //       break;

      //     UnifiedPointer mem_ptr;
      //     {
      //       auto& lk = get_lock_for_key(victim_key);
      //       std::shared_lock<std::shared_mutex> g(lk);
      //       uint64_t ptr_raw = 0;
      //       index->Get(victim_key, ptr_raw, 0);
      //       mem_ptr = UnifiedPointer::FromRaw(ptr_raw);
      //       if (mem_ptr.type() != UnifiedPointer::Type::Memory)
      //         continue;
      //     }

      //     // No-Touch 读取，避免再插回 LRU
      //     std::string value      = ReadFromMemRaw(mem_ptr);
      //     UnifiedPointer ssd_ptr = WriteDisk(value);
      //     if (!ssd_ptr) {
      //       TouchLRU(victim_key);
      //       continue;
      //     }

      //     bool swapped = false;
      //     {
      //       auto& lk = get_lock_for_key(victim_key);
      //       std::unique_lock<std::shared_mutex> g(lk);
      //       uint64_t current_ptr_raw = 0;
      //       index->Get(victim_key, current_ptr_raw, 0);
      //       if (current_ptr_raw == mem_ptr.RawValue()) {
      //         index->Put(victim_key, ssd_ptr.RawValue(), 0);
      //         swapped = true;
      //       }
      //     }
      //     if (swapped)
      //       FreePointer(mem_ptr);
      //     else
      //       FreePointer(ssd_ptr);
      //   }
      // }

      // // 2) 有“预算”才做提升；提升中途若达到低水位，立刻停止
      // size_t promote_headroom = 0;
      // {
      //   size_t used = shm_used_bytes_.load(std::memory_order_relaxed);
      //   size_t budget_floor =
      //       LowWatermark() > shm_capacity_bytes_ / 20
      //           ? LowWatermark() - shm_capacity_bytes_ / 20 // 留 5% 余量
      //           : LowWatermark();
      //   promote_headroom = (used < budget_floor) ? (budget_floor - used) : 0;
      // }

      // if (promote_headroom > 0) {
      //   size_t scanned = 0;
      //   for (auto it = hot_.begin(); it != hot_.end(); ++it) {
      //     if (scanned++ >= kPromoteScanBudget)
      //       break; // 限额扫描
      //     if (it->second < kPromoteThreshold)
      //       continue;

      //     Key_t key = it->first;
      //     UnifiedPointer disk_ptr;
      //     {
      //       auto& lk = get_lock_for_key(key);
      //       std::shared_lock<std::shared_mutex> g(lk);
      //       uint64_t ptr_raw = 0;
      //       index->Get(key, ptr_raw, 0);
      //       disk_ptr = UnifiedPointer::FromRaw(ptr_raw);
      //       if (disk_ptr.type() != UnifiedPointer::Type::Disk) {
      //         hot_.erase(key);
      //         continue;
      //       }
      //     }

      //     std::string value = ReadFromDiskNoHeat(disk_ptr);
      //     auto mem_ptr      = WriteMem(key, value);
      //     if (!mem_ptr)
      //       continue;

      //     bool swapped = false;
      //     {
      //       auto& lk = get_lock_for_key(key);
      //       std::unique_lock<std::shared_mutex> g(lk);
      //       uint64_t current_ptr_raw = 0;
      //       index->Get(key, current_ptr_raw, 0);
      //       if (current_ptr_raw == disk_ptr.RawValue()) {
      //         index->Put(key, mem_ptr.RawValue(), 0);
      //         swapped = true;
      //       }
      //     }
      //     if (swapped) {
      //       FreePointer(disk_ptr);
      //       hot_.erase(key);
      //       if (shm_used_bytes_.load(std::memory_order_relaxed) >=
      //           LowWatermark())
      //         break; // 用完预算
      //     } else {
      //       FreePointer(mem_ptr);
      //     }
      //   }

      //   // 轻度衰减少量热点分（避免全表扫描）
      //   size_t dec = 0;
      //   for (auto it = hot_.begin(); it != hot_.end() && dec < kDecayPerRound;
      //        ++it, ++dec) {
      //     if (it->second > 0)
      //       it->second--;
      //   }
      // }

      // {
      //   std::lock_guard<std::mutex> lk(evict_mu_);
      //   need_evict_ = false;
      //   evict_cv_.notify_all();
      // }
    }
  }
};

class SSDValueManager {
public:
  SSDValueManager(const BaseKVConfig& config,
                  const std::string& file_path,
                  PageID_t page_offset) {
    BaseKVConfig ssd_config                   = config;
    ssd_config.json_config_["file_path"]      = file_path;
    ssd_config.json_config_["page_id_offset"] = page_offset;

    std::string io_backend_type =
        config.json_config_.at("io_backend_type").get<std::string>();
    using IOF = base::Factory<IOBackend, const BaseKVConfig&>;
    io_backend_.reset(IOF::NewInstance(io_backend_type, ssd_config));
    io_backend_->init();
  }

  SSDValueManager(const SSDValueManager&)            = delete;
  SSDValueManager& operator=(const SSDValueManager&) = delete;

  UnifiedPointer Write(const std::string_view& value) {
    uint32_t actual_size   = (uint32_t)value.size();
    uint16_t pages_needed  = (uint16_t)((actual_size + 4 + PAGE_SIZE - 1) / PAGE_SIZE);
    PageID_t start_page_id = io_backend_->AllocatePages(pages_needed);

    char* buffer = io_backend_->AllocateBuffer(pages_needed);
    memcpy(buffer, &actual_size, 4);
    memcpy(buffer + 4, value.data(), actual_size);

    IOBackend::IOEntry entry{start_page_id, buffer, pages_needed};
    io_backend_->BatchWritePages({entry});
    io_backend_->FreeBuffer(buffer);

    return UnifiedPointer::FromDisk(start_page_id, pages_needed);
  }

  std::string Read(const UnifiedPointer& ptr) {
    PageID_t start_page_id = ptr.asDiskPageId();
    uint16_t page_count    = ptr.asDiskPageCount();
    std::string value;
    uint32_t actual_size = 0;
    uint32_t written     = 0;
    for (int i = 0; i < page_count; i++) {
      PageID_t pid = start_page_id + i;
      char* buffer = (char*)io_backend_->GetPage(pid);
      if (i == 0) {
        memcpy(&actual_size, buffer, 4);
        value.resize(actual_size);
        uint32_t data_in_page =
            std::min((uint32_t)(PAGE_SIZE - 4), actual_size);
        memcpy(value.data(), buffer + 4, data_in_page);
        written += data_in_page;
      } else {
        uint32_t data_in_page =
            std::min((uint32_t)PAGE_SIZE, actual_size - written);
        memcpy(value.data() + written, buffer, data_in_page);
        written += data_in_page;
      }
      io_backend_->Unpin(pid, buffer, false);
    }
    return value;
  }

  void Free(const UnifiedPointer& ptr) {
    // TODO: 等 free-list 实现后再处理
    (void)ptr;
  }

  std::vector<UnifiedPointer> BatchWrite(
      const std::vector<std::string_view>& values) {
    int n = values.size();
    std::vector<UnifiedPointer> ptrs(n);
    std::vector<uint16_t> page_counts(n);
    std::vector<PageID_t> start_page_ids(n);

    for (int i = 0; i < n; i++) {
      uint32_t actual_size = (uint32_t)values[i].size();
      page_counts[i]       = (uint16_t)((actual_size + 4 + PAGE_SIZE - 1) / PAGE_SIZE);
      start_page_ids[i]    = io_backend_->AllocatePages(page_counts[i]);
    }

    std::vector<IOBackend::IOEntry> entries;
    entries.reserve(n);
    for (int i = 0; i < n; i++) {
      uint32_t actual_size = (uint32_t)values[i].size();
      char* buffer         = io_backend_->AllocateBuffer(page_counts[i]);
      memcpy(buffer, &actual_size, 4);
      memcpy(buffer + 4, values[i].data(), actual_size);
      entries.push_back({start_page_ids[i], buffer, page_counts[i]});
    }
    io_backend_->BatchWritePages(entries);
    for (auto& e : entries)
      io_backend_->FreeBuffer(e.buffer);

    for (int i = 0; i < n; i++)
      ptrs[i] = UnifiedPointer::FromDisk(start_page_ids[i], page_counts[i]);
    return ptrs;
  }

  std::vector<std::string> BatchRead(
      const std::vector<UnifiedPointer>& ptrs) {
    int n = ptrs.size();
    std::vector<IOBackend::IOEntry> entries;
    entries.reserve(n);
    for (auto& p : ptrs) {
      char* buffer = io_backend_->AllocateBuffer(p.asDiskPageCount());
      entries.push_back({p.asDiskPageId(), buffer, p.asDiskPageCount()});
    }
    io_backend_->BatchReadPages(entries);

    std::vector<std::string> results(n);
    for (int i = 0; i < n; i++) {
      char* buf            = entries[i].buffer;
      uint32_t actual_size = 0;
      memcpy(&actual_size, buf, 4);
      results[i].assign(buf + 4, actual_size);
      io_backend_->FreeBuffer(buf);
    }
    return results;
  }

  IOBackend* GetIOBackend() { return io_backend_.get(); }

private:
  std::unique_ptr<IOBackend> io_backend_;
};
