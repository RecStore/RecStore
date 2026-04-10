#pragma once

#include "../../memory/allocators/allocator_factory.h"
#include "../../memory/memory_factory.h"
#include "base/factory.h"
#include "pointer.h"
#include "../index/index.h"
#include "storage/cache/cache_factory.h"
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
#include <condition_variable>
#include <cstring>
#include <tbb/concurrent_hash_map.h>
#include <array>
#include <functional>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/uio.h>

class ValueManager {
public:
  std::unique_ptr<Index> index;

  ValueManager(const std::string& shm_file_path,
               size_t shm_capacity,
               const std::string& ssd_file_path,
               size_t ssd_capacity,
               const BaseKVConfig& index_config)
      : cache_policy_(storage::cache::CreateCachePolicy<Key_t>(
            index_config.json_config_.value("cache_policy", "LRU"),
            std::max<size_t>(4096, shm_capacity / 256))),
        shm_capacity_bytes_(shm_capacity) {
    json dram_cfg = index_config.json_config_;
    json ssd_cfg  = index_config.json_config_;
    if (index_config.json_config_.contains("dram_allocator_type"))
      dram_cfg["allocator_type"] =
          index_config.json_config_.at("dram_allocator_type");
    if (index_config.json_config_.contains("ssd_allocator_type"))
      ssd_cfg["allocator_type"] =
          index_config.json_config_.at("ssd_allocator_type");
    if (index_config.json_config_.contains("dram_value_memory_management"))
      dram_cfg["value_memory_management"] =
          index_config.json_config_.at("dram_value_memory_management");
    if (index_config.json_config_.contains("ssd_value_memory_management"))
      ssd_cfg["value_memory_management"] =
          index_config.json_config_.at("ssd_value_memory_management");

    shm_manage = base::allocators::CreateAllocator(
        dram_cfg, shm_file_path, shm_capacity, "DRAM");
    ssd_manage = base::allocators::CreateAllocator(
        ssd_cfg, ssd_file_path, ssd_capacity, "SSD");
    if (!shm_manage || !ssd_manage)
      throw std::runtime_error("failed to initialize hybrid value allocators");

    using IndexF = base::Factory<Index, const BaseKVConfig&>;
    std::string index_type =
        index_config.json_config_.value("index_type", "DRAM");
    index.reset(IndexF::NewInstance(index_type, index_config));

    fd_ssd = ::open(ssd_file_path.c_str(), O_RDWR);
    if (fd_ssd < 0) {
      LOG(ERROR) << "ssd open error";
    }
    evictor_ = std::thread([this] { this->EvictAndPromoteLoop(); });
  }

  ValueManager(const ValueManager&)            = delete;
  ValueManager& operator=(const ValueManager&) = delete;

  ~ValueManager() {
    stop_.store(true, std::memory_order_release);
    evict_cv_.notify_all();
    if (evictor_.joinable())
      evictor_.join();
    if (fd_ssd >= 0) {
      ::fdatasync(fd_ssd);
      ::close(fd_ssd);
    }
  }

  void WriteValue(Key_t key, const std::string_view& value, unsigned tid) {
    // 先尝试在持锁下拿旧指针并做等长原地写
    uint64_t old_raw = 0;
    {
      auto& lk = get_lock_for_key(key);
      std::unique_lock<std::shared_mutex> g(lk);
      index->Get(key, old_raw, tid);

      if (old_raw) {
        UnifiedPointer oldp = UnifiedPointer::FromRaw(old_raw);
        if (oldp.type() == UnifiedPointer::Type::Memory) {
          uint16_t old_len = PeekLenMem(oldp);
          if (old_len == value.size()) {
            // 等长原地写内存
            uint8_t* bytes = static_cast<uint8_t*>(oldp.asMemoryPointer());
            std::memcpy(bytes + 2, value.data(), old_len);
            TouchCache(key);
            return;
          }
        } else if (oldp.type() == UnifiedPointer::Type::Disk) {
          // 等长原地写“盘上映射区”（避免一次 New）
          const off_t pool_off = static_cast<off_t>(oldp.asDiskPageId());
          char* disk_ptr       = ssd_manage->GetMallocData(pool_off);
          uint16_t old_len     = 0;
          std::memcpy(&old_len, disk_ptr, sizeof(old_len));
          if (old_len == value.size()) {
            std::memcpy(disk_ptr + 2, value.data(), old_len);
            return;
          }
        }
      }
    }

    UnifiedPointer p_write = WriteMem(key, value);
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
        return;
      p_write = WriteMem(key, value);
    }
    if (!p_write)
      p_write = WriteDisk(value);
    if (!p_write) {
      LOG(ERROR) << "Failed to allocate memory for key " << key;
      return;
    }

    // 更新索引并释放旧块（若存在且非 PMem）
    {
      auto& lk = get_lock_for_key(key);
      std::unique_lock<std::shared_mutex> g(lk);
      index->Get(key, old_raw, tid);
      index->Put(key, p_write.RawValue(), tid);
    }
    if (old_raw)
      FreePointer(UnifiedPointer::FromRaw(old_raw));
  }
  std::string RetrieveValue(Key_t key, unsigned tid) {
    auto& lk = get_lock_for_key(key);
    std::shared_lock<std::shared_mutex> g(lk);
    uint64_t pointer = 0;
    index->Get(key, pointer, tid);
    UnifiedPointer p = UnifiedPointer::FromRaw(pointer);
    switch (p.type()) {
    case UnifiedPointer::Type::Memory: {
      auto v = ReadFromMemRaw(p); // 不触 LRU
      TouchCache(key);            // 业务访问才触碰
      return v;
    }
    case UnifiedPointer::Type::Disk: {
      auto v = ReadFromDiskNoHeat(p); // 不升温
      // 1/8 抽样计数，降低热点写入压力
      if ((std::hash<Key_t>{}(key) & 0x7) == 0) {
        tbb::concurrent_hash_map<Key_t, uint8_t>::accessor acc;
        if (hot_.insert(acc, key))
          acc->second = 0;
        if (acc->second < kPromoteThreshold)
          acc->second++;
      }
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
  std::unique_ptr<base::MallocApi> ssd_manage;
  int fd_ssd = -1;

  std::unique_ptr<CachePolicy<Key_t>> cache_policy_;
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

  inline void TouchCache(Key_t key) { cache_policy_->OnAccess(key); }

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
    const off_t pool_off = static_cast<off_t>(p.asDiskPageId());
    const char* disk_ptr = ssd_manage->GetMallocData(pool_off);
    uint16_t value_len   = 0;
    std::memcpy(&value_len, disk_ptr, sizeof(value_len));
    return std::string(disk_ptr + sizeof(value_len), value_len);
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
      ssd_manage->Free(ssd_manage->GetMallocData((int64)p.asDiskPageId()));
    }
  }

  UnifiedPointer WriteMem(Key_t key, const std::string_view& value) {
    if (value.size() > 0 && value.size() <= 7) {
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
    // 只有真正写入内存的数据，才进入内存 LRU
    cache_policy_->OnInsert(key);
    return UnifiedPointer::FromMemory(ptr);
  }

  UnifiedPointer WriteDisk(const std::string_view& value) {
    if (value.size() > std::numeric_limits<uint16_t>::max())
      return UnifiedPointer();
    const uint16_t data_len = static_cast<uint16_t>(value.size());
    const int total_size    = static_cast<int>(sizeof(data_len) + data_len);
    char* disk_ptr          = ssd_manage->New(total_size);
    if (!disk_ptr) {
      LOG(ERROR) << "NEW SSD WRONG";
      return UnifiedPointer();
    }
    off_t pool_off = ssd_manage->GetMallocOffset(disk_ptr);
    if (pool_off == -1) {
      ssd_manage->Free(disk_ptr);
      return UnifiedPointer();
    }
    const off_t file_off =
        static_cast<off_t>(ssd_manage->DataBaseOffset()) + pool_off;
    uint8_t lenle[2] = {static_cast<uint8_t>(data_len & 0xFF),
                        static_cast<uint8_t>((data_len >> 8) & 0xFF)};

    if (pwrite(fd_ssd, lenle, 2, file_off) != 2 ||
        (data_len && pwrite(fd_ssd, value.data(), data_len, file_off + 2) !=
                         static_cast<ssize_t>(data_len))) {
      ssd_manage->Free(disk_ptr);
      return UnifiedPointer();
    }
    if (fdatasync(fd_ssd) < 0) {
      ssd_manage->Free(disk_ptr);
      return UnifiedPointer();
    }
    return UnifiedPointer::FromDiskPageId(static_cast<uint64_t>(pool_off));
  }

  void EvictAndPromoteLoop() {
    const size_t kEvictionBatchSize = 512; // 原 64
    const size_t kPromoteScanBudget = 512; // 每轮最多扫描热点条目数
    const size_t kDecayPerRound     = 256; // 每轮轻度衰减项数

    while (!stop_.load(std::memory_order_acquire)) {
      {
        std::unique_lock<std::mutex> lk(evict_mu_);
        evict_cv_.wait_for(lk, std::chrono::milliseconds(20), [&] {
          return stop_.load() || need_evict_ ||
                 shm_used_bytes_.load() > HighWatermark();
        });
      }
      if (stop_.load())
        break;

      // 1) 先逐出到低水位以下，避免在高水位阶段还去做提升
      if (shm_used_bytes_.load(std::memory_order_relaxed) > LowWatermark()) {
        std::vector<Key_t> victim_keys =
            cache_policy_->Evict(kEvictionBatchSize);
        for (const auto& victim_key : victim_keys) {
          if (shm_used_bytes_.load(std::memory_order_relaxed) <= LowWatermark())
            break;

          UnifiedPointer mem_ptr;
          {
            auto& lk = get_lock_for_key(victim_key);
            std::shared_lock<std::shared_mutex> g(lk);
            uint64_t ptr_raw = 0;
            index->Get(victim_key, ptr_raw, 0);
            mem_ptr = UnifiedPointer::FromRaw(ptr_raw);
            if (mem_ptr.type() != UnifiedPointer::Type::Memory)
              continue;
          }

          // No-Touch 读取，避免再插回 LRU
          std::string value      = ReadFromMemRaw(mem_ptr);
          UnifiedPointer ssd_ptr = WriteDisk(value);
          if (!ssd_ptr) {
            TouchCache(victim_key);
            continue;
          }

          bool swapped = false;
          {
            auto& lk = get_lock_for_key(victim_key);
            std::unique_lock<std::shared_mutex> g(lk);
            uint64_t current_ptr_raw = 0;
            index->Get(victim_key, current_ptr_raw, 0);
            if (current_ptr_raw == mem_ptr.RawValue()) {
              index->Put(victim_key, ssd_ptr.RawValue(), 0);
              swapped = true;
            }
          }
          if (swapped)
            FreePointer(mem_ptr);
          else
            FreePointer(ssd_ptr);
        }
      }

      // 2) 有“预算”才做提升；提升中途若达到低水位，立刻停止
      size_t promote_headroom = 0;
      {
        size_t used = shm_used_bytes_.load(std::memory_order_relaxed);
        size_t budget_floor =
            LowWatermark() > shm_capacity_bytes_ / 20
                ? LowWatermark() - shm_capacity_bytes_ / 20 // 留 5% 余量
                : LowWatermark();
        promote_headroom = (used < budget_floor) ? (budget_floor - used) : 0;
      }

      if (promote_headroom > 0) {
        size_t scanned = 0;
        for (auto it = hot_.begin(); it != hot_.end(); ++it) {
          if (scanned++ >= kPromoteScanBudget)
            break; // 限额扫描
          if (it->second < kPromoteThreshold)
            continue;

          Key_t key = it->first;
          UnifiedPointer disk_ptr;
          {
            auto& lk = get_lock_for_key(key);
            std::shared_lock<std::shared_mutex> g(lk);
            uint64_t ptr_raw = 0;
            index->Get(key, ptr_raw, 0);
            disk_ptr = UnifiedPointer::FromRaw(ptr_raw);
            if (disk_ptr.type() != UnifiedPointer::Type::Disk) {
              hot_.erase(key);
              continue;
            }
          }

          std::string value = ReadFromDiskNoHeat(disk_ptr);
          auto mem_ptr      = WriteMem(key, value);
          if (!mem_ptr)
            continue;

          bool swapped = false;
          {
            auto& lk = get_lock_for_key(key);
            std::unique_lock<std::shared_mutex> g(lk);
            uint64_t current_ptr_raw = 0;
            index->Get(key, current_ptr_raw, 0);
            if (current_ptr_raw == disk_ptr.RawValue()) {
              index->Put(key, mem_ptr.RawValue(), 0);
              swapped = true;
            }
          }
          if (swapped) {
            FreePointer(disk_ptr);
            hot_.erase(key);
            if (shm_used_bytes_.load(std::memory_order_relaxed) >=
                LowWatermark())
              break; // 用完预算
          } else {
            FreePointer(mem_ptr);
          }
        }

        // 轻度衰减少量热点分（避免全表扫描）
        size_t dec = 0;
        for (auto it = hot_.begin(); it != hot_.end() && dec < kDecayPerRound;
             ++it, ++dec) {
          if (it->second > 0)
            it->second--;
        }
      }

      {
        std::lock_guard<std::mutex> lk(evict_mu_);
        need_evict_ = false;
        evict_cv_.notify_all();
      }
    }
  }
};
