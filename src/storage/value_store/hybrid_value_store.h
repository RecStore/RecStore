#pragma once

#include <atomic>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "base/factory.h"
#include "storage/cache/tiered_read_cache.h"
#include "storage/value_store/tiered_clock_value_store.h"
#include "storage/value_store/dram_value_store.h"
#include "storage/value_store/ssd_static_layout_value_store.h"
#include "storage/value_store/ssd_value_store.h"

class HybridValueStore : public ValueStore {
public:
  explicit HybridValueStore(const BaseKVConfig& config)
      : dram_store_(BuildDramConfig(config)) {
    const auto& value = config.json_config_.at("value");
    ssd_store_.reset(
        base::Factory<ValueStore, const BaseKVConfig&>::NewInstance(
            SsdStoreType(value), BuildSsdConfig(config)));
    ssd_static_           = ssd_store_ && !ssd_store_->GetLayoutId().plain();
    const auto& v         = config.json_config_.at("value");
    const auto& dram      = v.at("dram_allocator");
    dram_capacity_bytes_  = dram.at("capacity_bytes").get<uint64_t>();
    dram_placement_bytes_ = dram_capacity_bytes_;
    const auto& tiering =
        v.contains("tiering") ? v.at("tiering") : json::object();
    high_watermark_ratio_ = tiering.value("high_watermark_ratio", 0.85);
    std::string cache_policy = tiering.value("cache_policy", "none");
    const bool adaptive =
        cache_policy == "adaptive" || cache_policy == "ADAPTIVE";
    if (cache_policy.empty() || cache_policy == "none") {
      return;
    }
    if (cache_policy != "lru" && cache_policy != "LRU" &&
        cache_policy != "clock" && cache_policy != "CLOCK" && !adaptive) {
      throw std::invalid_argument(
          "unsupported tiering.cache_policy: " + cache_policy);
    }
    double budget_ratio = tiering.value("cache_budget_ratio", 0.05);
    if (!adaptive && cache_policy != "clock" && cache_policy != "CLOCK" &&
        (budget_ratio <= 0.0 || budget_ratio > 1.0)) {
      throw std::invalid_argument(
          "tiering.cache_budget_ratio must be in (0, 1]");
    }
    const int64_t shard_count =
        tiering.value("cache_shards", static_cast<int64_t>(16));
    if (shard_count < 1 || shard_count > 4096) {
      throw std::invalid_argument("tiering.cache_shards must be in [1, 4096]");
    }
    const int64_t promote_min_misses =
        tiering.value("promote_min_misses", static_cast<int64_t>(2));
    if (promote_min_misses < 1 || promote_min_misses > UINT32_MAX) {
      throw std::invalid_argument("tiering.promote_min_misses must be >= 1");
    }
    const int64_t expected_dram_bytes =
        tiering.value("expected_dram_bytes", static_cast<int64_t>(0));
    if (expected_dram_bytes < 0) {
      throw std::invalid_argument("tiering.expected_dram_bytes must be >= 0");
    }
    if (adaptive) {
      if (!std::isfinite(high_watermark_ratio_) ||
          high_watermark_ratio_ < 0.0 || high_watermark_ratio_ > 1.0)
        throw std::invalid_argument(
            "adaptive high_watermark_ratio must be in [0, 1]");
      const double limit = high_watermark_ratio_ * dram_capacity_bytes_;
      cache_policy       = "clock";
      if (expected_dram_bytes > 0) {
        if (expected_dram_bytes <= limit)
          return;
        const double overflow = expected_dram_bytes - limit;
        // For <= 0.1% overflow, avoid per-resident migration metadata. A
        // small LRU keeps SSD hot rows while original DRAM reads stay direct.
        // Unknown or larger working sets use full CLOCK replacement.
        if (limit > 0 && overflow / expected_dram_bytes <= 0.001) {
          cache_policy = "lru";
          budget_ratio = overflow / limit;
        }
      }
    }
    if (cache_policy == "clock" || cache_policy == "CLOCK") {
      if (!std::isfinite(high_watermark_ratio_) || high_watermark_ratio_ < 0.0 ||
          high_watermark_ratio_ > 1.0) {
        throw std::invalid_argument("CLOCK high_watermark_ratio must be in [0, 1]");
      }
      clock_ = std::make_unique<TieredClockValueStore>(
          &dram_store_, ssd_store_.get(),
          static_cast<uint64_t>(high_watermark_ratio_ * dram_capacity_bytes_),
          static_cast<size_t>(shard_count), static_cast<uint32_t>(promote_min_misses),
          tiering.value("retain_backing_copies", true),
          tiering.value("clock_frequency_admission", false));
      return;
    }
    // This optional load-time estimate includes allocator size-class rounding.
    // Omit the estimate for unknown or growing working sets.
    if (expected_dram_bytes > 0 &&
        static_cast<double>(expected_dram_bytes) <=
            high_watermark_ratio_ * static_cast<double>(dram_capacity_bytes_)) {
      return;
    }
    const uint64_t budget_bytes = static_cast<uint64_t>(
        budget_ratio * static_cast<double>(dram_capacity_bytes_));
    // The cache budget is reserved up front rather than shared dynamically:
    // primary placement then stops at high_watermark * (capacity - budget), so
    // primary + cache stays inside the same DRAM watermark the cache-less path
    // already uses.
    dram_placement_bytes_ = dram_capacity_bytes_ - budget_bytes;
    cache_                = std::make_unique<TieredReadCache>(
        &dram_store_,
        static_cast<size_t>(shard_count),
        budget_bytes,
        static_cast<uint64_t>(
            high_watermark_ratio_ * static_cast<double>(dram_capacity_bytes_)),
        &dram_bytes_reserved_,
        static_cast<uint32_t>(promote_min_misses));
  }

  uint64_t Alloc(size_t size) override {
    if (clock_) return clock_->Alloc(size);
    if (ShouldUseDram(size)) {
      const uint64_t raw = dram_store_.Alloc(size);
      if (raw != kValueHandleNone) {
        dram_bytes_reserved_.fetch_add(
            dram_store_.SlotCapacity(raw), std::memory_order_relaxed);
        dram_live_allocs_.fetch_add(1, std::memory_order_relaxed);
        dram_total_allocs_.fetch_add(1, std::memory_order_relaxed);
        return MakeDramHandle(raw);
      }
    }
    const uint64_t raw = ssd_store_->Alloc(size);
    if (raw == kValueHandleNone) {
      return kValueHandleNone;
    }
    ssd_live_allocs_.fetch_add(1, std::memory_order_relaxed);
    ssd_total_allocs_.fetch_add(1, std::memory_order_relaxed);
    return MakeSsdHandle(raw);
  }

  void Write(uint64_t handle, const void* data, size_t size) override {
    if (clock_) { clock_->Write(handle, data, size); return; }
    if (handle == kValueHandleNone) {
      return;
    }
    if (IsOnSSD(handle)) {
      const uint64_t raw = SsdRawHandle(handle);
      ssd_store_->Write(raw, data, size);
      if (cache_) {
        cache_->WriteThrough(raw, data, size);
      }
      return;
    }
    dram_store_.Write(DramRawHandle(handle), data, size);
  }

  uint64_t AllocAndWrite(const void* data, size_t size) override {
    if (clock_) return clock_->AllocAndWrite(data, size);
    const uint64_t handle = Alloc(size);
    if (handle != kValueHandleNone) {
      Write(handle, data, size);
    }
    return handle;
  }

  uint64_t AllocAndWrite(uint64_t key, const void* data, size_t size) override {
    if (clock_) return clock_->AllocAndWrite(key, data, size);
    if (!ssd_static_) {
      return AllocAndWrite(data, size);
    }
    const uint64_t handle = PlaceStatic(key, size);
    if (handle != kValueHandleNone) {
      Write(handle, data, size);
    }
    return handle;
  }

  std::vector<uint64_t> BatchAllocAndWrite(
      const std::vector<WriteSpec>& specs) override {
    return ValueStore::BatchAllocAndWrite(specs);
  }

  std::vector<uint64_t> BatchAllocAndWrite(
      const std::vector<uint64_t>& keys,
      const std::vector<WriteSpec>& specs) override {
    return clock_ ? clock_->BatchAllocAndWrite(keys, specs)
                  : ValueStore::BatchAllocAndWrite(keys, specs);
  }

  void ReadValue(uint64_t handle, std::string& value) override {
    if (clock_) clock_->ReadValue(handle, value);
    else if (IsOnSSD(handle)) ValueStore::ReadValue(handle, value);
    else dram_store_.ReadValue(DramRawHandle(handle), value);
  }

  bool ReadFlatFixedRows(const uint64_t* handles,
                         size_t num_rows,
                         void* out_buf,
                         size_t row_bytes,
                         uint64_t* missing_rows) const override {
    if (clock_)
      return clock_->ReadFlatFixedRows(
          handles, num_rows, out_buf, row_bytes, missing_rows);
    if (ssd_static_ || !handles || !out_buf || row_bytes == 0)
      return false;
    auto* out = static_cast<char*>(out_buf);
    uint64_t missing = 0;
    std::vector<uint64_t> pending;
    std::vector<size_t> rows;
    for (size_t i = 0; i < num_rows; ++i) {
      if (handles[i] == kValueHandleNone) {
        std::memset(out + i * row_bytes, 0, row_bytes);
        ++missing;
      } else if (!IsOnSSD(handles[i])) {
        // Legacy placement never migrates DRAM rows. As in DramValueStore,
        // the caller's fixed-width hint permits copying directly to output.
        std::memcpy(out + i * row_bytes,
                    dram_store_.DirectPtr(DramRawHandle(handles[i])), row_bytes);
      } else {
        pending.push_back(handles[i]);
        rows.push_back(i);
      }
    }
    if (!pending.empty()) {
      std::vector<ReadResult> fetched;
      const_cast<HybridValueStore*>(this)->BatchRead(pending, fetched);
      for (size_t i = 0; i < rows.size(); ++i) {
        if (fetched[i].data.size() != row_bytes)
          return false;
        std::memcpy(out + rows[i] * row_bytes, fetched[i].data.data(), row_bytes);
      }
    }
    if (missing_rows)
      *missing_rows = missing;
    return true;
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    if (clock_) return clock_->Read(handle, out_buf, buf_size);
    if (handle == kValueHandleNone) {
      return 0;
    }
    if (IsOnSSD(handle)) {
      const uint64_t raw = SsdRawHandle(handle);
      uint64_t barrier   = 0;
      if (cache_) {
        size_t copied = 0;
        if (cache_->CopyOut(raw, out_buf, buf_size, &copied, &barrier)) {
          return copied;
        }
      }
      const size_t actual = ssd_store_->Read(raw, out_buf, buf_size);
      // Only cache a value that was read in full; a truncated read would put a
      // short value in the cache and every later hit would return it. The
      // stored length is not exposed, so a complete read is inferred from a
      // short read or from a buffer that could hold the whole slot: comparing
      // `actual` with `SlotCapacity` would be a length-vs-capacity comparison
      // and would reject every value smaller than its slab or buddy slot.
      if (cache_ &&
          ReadCompleted(actual, buf_size, ssd_store_->SlotCapacity(raw))) {
        cache_->Promote(raw, out_buf, actual, barrier);
      }
      return actual;
    }
    return dram_store_.Read(DramRawHandle(handle), out_buf, buf_size);
  }

  void Free(uint64_t handle) override {
    if (clock_) { clock_->Free(handle); return; }
    if (handle == kValueHandleNone) {
      return;
    }
    if (IsOnSSD(handle)) {
      const uint64_t raw = SsdRawHandle(handle);
      if (cache_) {
        cache_->Invalidate(raw);
      }
      ssd_store_->Free(raw);
      ssd_live_allocs_.fetch_sub(1, std::memory_order_relaxed);
      return;
    }
    const uint64_t raw = DramRawHandle(handle);
    const size_t cap   = dram_store_.SlotCapacity(raw);
    dram_store_.Free(raw);
    dram_live_allocs_.fetch_sub(1, std::memory_order_relaxed);
    uint64_t cur = dram_bytes_reserved_.load(std::memory_order_relaxed);
    while (cur != 0 &&
           !dram_bytes_reserved_.compare_exchange_weak(
               cur, cur > cap ? cur - cap : 0, std::memory_order_relaxed)) {
    }
  }

  const char* DirectPtr(uint64_t handle) const override {
    if (clock_ || handle == kValueHandleNone || IsOnSSD(handle)) {
      return nullptr;
    }
    return dram_store_.DirectPtr(DramRawHandle(handle));
  }

  char* RDMABackingData() const override {
    return clock_ ? nullptr : dram_store_.RDMABackingData();
  }

  size_t RDMABackingSize() const override {
    return clock_ ? 0 : dram_store_.RDMABackingSize();
  }

  size_t SlotCapacity(uint64_t handle) const override {
    if (clock_) return clock_->SlotCapacity(handle);
    if (handle == kValueHandleNone) {
      return 0;
    }
    if (IsOnSSD(handle)) {
      return ssd_store_->SlotCapacity(SsdRawHandle(handle));
    }
    return dram_store_.SlotCapacity(DramRawHandle(handle));
  }

  void BatchWrite(const std::vector<uint64_t>& handles,
                  const std::vector<WriteSpec>& specs) override {
    if (clock_) { clock_->BatchWrite(handles, specs); return; }
    if (handles.size() != specs.size()) {
      throw std::invalid_argument("HybridValueStore::BatchWrite size mismatch");
    }
    std::vector<uint64_t> dram_handles;
    std::vector<WriteSpec> dram_specs;
    std::vector<uint64_t> ssd_handles;
    std::vector<WriteSpec> ssd_specs;
    dram_handles.reserve(handles.size());
    dram_specs.reserve(handles.size());
    ssd_handles.reserve(handles.size());
    ssd_specs.reserve(handles.size());
    for (size_t i = 0; i < handles.size(); ++i) {
      if (handles[i] == kValueHandleNone) {
        continue;
      }
      if (IsOnSSD(handles[i])) {
        ssd_handles.push_back(SsdRawHandle(handles[i]));
        ssd_specs.push_back(specs[i]);
      } else {
        dram_handles.push_back(DramRawHandle(handles[i]));
        dram_specs.push_back(specs[i]);
      }
    }
    if (!dram_handles.empty()) {
      dram_store_.BatchWrite(dram_handles, dram_specs);
    }
    if (!ssd_handles.empty()) {
      ssd_store_->BatchWrite(ssd_handles, ssd_specs);
      if (cache_) {
        for (size_t i = 0; i < ssd_handles.size(); ++i) {
          cache_->WriteThrough(
              ssd_handles[i], ssd_specs[i].data, ssd_specs[i].size);
        }
      }
    }
  }

  void BatchRead(const std::vector<uint64_t>& handles,
                 std::vector<ReadResult>& out_results) override {
    if (clock_) { clock_->BatchRead(handles, out_results); return; }
    out_results.clear();
    out_results.resize(handles.size());

    std::vector<uint64_t> ssd_handles;
    std::vector<size_t> ssd_indices;
    std::vector<uint64_t> ssd_barriers;
    ssd_handles.reserve(handles.size());
    ssd_indices.reserve(handles.size());
    ssd_barriers.reserve(handles.size());

    for (size_t i = 0; i < handles.size(); ++i) {
      const uint64_t handle = handles[i];
      if (handle == kValueHandleNone) {
        continue;
      }
      if (IsOnSSD(handle)) {
        const uint64_t raw    = SsdRawHandle(handle);
        const size_t capacity = ssd_store_->SlotCapacity(raw);
        out_results[i].data.resize(capacity);
        size_t copied    = 0;
        uint64_t barrier = 0;
        if (cache_ &&
            cache_->CopyOut(
                raw, out_results[i].data.data(), capacity, &copied, &barrier)) {
          out_results[i].data.resize(copied);
          continue;
        }
        // Captured before the batch read reaches SSD, checked again by
        // Promote(): a write racing the read must not leave the old value
        // cached.
        ssd_barriers.push_back(barrier);
        ssd_handles.push_back(raw);
        ssd_indices.push_back(i);
        continue;
      }

      const uint64_t raw = DramRawHandle(handle);
      out_results[i].data.resize(dram_store_.SlotCapacity(raw));
      const size_t actual = dram_store_.Read(
          raw, out_results[i].data.data(), out_results[i].data.size());
      out_results[i].data.resize(actual);
    }

    if (ssd_handles.empty()) {
      return;
    }

    std::vector<ReadResult> ssd_results;
    ssd_store_->BatchRead(ssd_handles, ssd_results);
    if (ssd_results.size() != ssd_indices.size()) {
      throw std::runtime_error(
          "HybridValueStore::BatchRead result size mismatch");
    }
    for (size_t i = 0; i < ssd_indices.size(); ++i) {
      if (cache_ && !ssd_results[i].data.empty()) {
        cache_->Promote(ssd_handles[i],
                        ssd_results[i].data.data(),
                        ssd_results[i].data.size(),
                        ssd_barriers[i]);
      }
      out_results[ssd_indices[i]] = std::move(ssd_results[i]);
    }
  }

  std::string GetInfo() const override {
    if (clock_) return "HybridValueStore(" + ExtraResultFields() + ")";
    std::ostringstream os;
    os << "HybridValueStore(dram_reserved="
       << dram_bytes_reserved_.load(std::memory_order_relaxed)
       << ", dram_capacity=" << dram_capacity_bytes_ << ", cache_bytes="
       << (cache_ ? cache_->Bytes() : 0) << ", dram=" << dram_store_.GetInfo()
       << ", ssd=" << ssd_store_->GetInfo() << ")";
    return os.str();
  }

  void ResetStats() override {
    if (ssd_store_) {
      ssd_store_->ResetStats();
    }
  }

  std::string ExtraResultFields() const override {
    if (clock_) return clock_->ExtraResultFields() + ssd_store_->ExtraResultFields();
    const uint64_t dram_live =
        dram_live_allocs_.load(std::memory_order_relaxed);
    const uint64_t ssd_live = ssd_live_allocs_.load(std::memory_order_relaxed);
    const uint64_t live     = dram_live + ssd_live;
    const uint64_t dram_total =
        dram_total_allocs_.load(std::memory_order_relaxed);
    const uint64_t ssd_total =
        ssd_total_allocs_.load(std::memory_order_relaxed);
    const uint64_t total = dram_total + ssd_total;
    const double live_ratio =
        live == 0 ? 0.0 : static_cast<double>(ssd_live) / live;
    const double total_ratio =
        total == 0 ? 0.0 : static_cast<double>(ssd_total) / total;
    std::ostringstream os;
    os << std::fixed << std::setprecision(6)
       << " tiered_dram_capacity_bytes=" << dram_capacity_bytes_
       << " tiered_dram_placement_bytes=" << dram_placement_bytes_
       << " tiered_high_watermark_ratio=" << high_watermark_ratio_
       << " tiered_dram_live_allocs=" << dram_live << " tiered_ssd_live_allocs="
       << ssd_live << " tiered_live_allocs=" << live
       << " tiered_ssd_live_ratio=" << live_ratio
       << " tiered_dram_total_allocs=" << dram_total
       << " tiered_ssd_total_allocs=" << ssd_total
       << " tiered_total_allocs=" << total
       << " tiered_ssd_total_ratio=" << total_ratio << " tiered_dram_bytes="
       << dram_bytes_reserved_.load(std::memory_order_relaxed)
       << " tiered_dram_limit_bytes="
       << static_cast<uint64_t>(high_watermark_ratio_ * dram_capacity_bytes_);
    if (cache_) {
      const uint64_t hits   = cache_->Hits();
      const uint64_t misses = cache_->Misses();
      const double hit_rate =
          hits + misses == 0
              ? 0.0
              : static_cast<double>(hits) / static_cast<double>(hits + misses);
      os << " tiered_cache_policy=lru tiered_cache_shards="
         << cache_->ShardCount()
         << " tiered_cache_promote_min_misses=" << cache_->PromoteMinMisses()
         << " tiered_cache_budget_bytes=" << cache_->BudgetBytes()
         << " tiered_cache_effective_budget_bytes="
         << cache_->EffectiveBudgetBytes()
         << " tiered_cache_dram_limit_bytes=" << cache_->DramLimitBytes()
         << " tiered_cache_bytes=" << cache_->Bytes()
         << " tiered_cache_entries=" << cache_->Entries()
         << " tiered_cache_hits=" << hits << " tiered_cache_misses=" << misses
         << " tiered_cache_hit_rate=" << hit_rate
         << " tiered_cache_promotions=" << cache_->Promotions()
         << " tiered_cache_evictions=" << cache_->Evictions()
         << " tiered_cache_alloc_failures=" << cache_->AllocFailures()
         << " tiered_cache_admission_rejects=" << cache_->AdmissionRejects()
         << " tiered_cache_stale_promotions=" << cache_->StalePromotions();
    } else {
      os << " tiered_cache_policy=none";
    }
    return os.str() + ssd_store_->ExtraResultFields();
  }

  ValueStore::LayoutId GetLayoutId() const override {
    return ssd_store_ ? ssd_store_->GetLayoutId() : ValueStore::LayoutId{};
  }

private:
  static constexpr uint64_t kSsdFlag = 1ULL << 63;

  static bool IsOnSSD(uint64_t handle) { return (handle & kSsdFlag) != 0; }

  static uint64_t MakeDramHandle(uint64_t raw) { return raw; }

  static uint64_t MakeSsdHandle(uint64_t raw) { return raw | kSsdFlag; }

  static uint64_t DramRawHandle(uint64_t handle) { return handle; }

  static uint64_t SsdRawHandle(uint64_t handle) { return handle & ~kSsdFlag; }

  // True when a read of `actual` bytes into a `buf_size` buffer returned the
  // whole stored value. A short read is conclusive; a full buffer is only
  // conclusive when it could hold the slot's capacity.
  static bool
  ReadCompleted(size_t actual, size_t buf_size, size_t slot_capacity) {
    if (actual == 0) {
      return false;
    }
    return actual < buf_size || buf_size >= slot_capacity;
  }

  static BaseKVConfig BuildDramConfig(const BaseKVConfig& config) {
    BaseKVConfig out = config;
    auto& j          = out.json_config_;
    if (!j.contains("value") || !j.at("value").contains("dram_allocator")) {
      return out;
    }
    json value    = j.at("value");
    value["type"] = "DRAM_VALUE_STORE";
    if (value.at("dram_allocator").contains("path")) {
      value["path"] = value.at("dram_allocator").at("path");
    }
    value.erase("ssd_allocator");
    value.erase("ssd_layout");
    value.erase("tiering");
    j["value"] = value;
    return out;
  }

  static std::string SsdStoreType(const json& value) {
    if (!value.contains("ssd_layout")) {
      return "SSD_VALUE_STORE";
    }
    const std::string strategy =
        value.at("ssd_layout").value("strategy", "plain");
    if (strategy.empty() || strategy == "plain") {
      return "SSD_VALUE_STORE";
    }
    if (strategy == "bandana" || strategy == "maxembed") {
      return "SSD_STATIC_LAYOUT_VALUE_STORE";
    }
    throw std::invalid_argument("unknown SSD layout strategy: " + strategy);
  }

  static BaseKVConfig BuildSsdConfig(const BaseKVConfig& config) {
    BaseKVConfig out = config;
    auto& j          = out.json_config_;
    if (!j.contains("value") || !j.at("value").contains("ssd_allocator")) {
      return out;
    }
    json value    = j.at("value");
    value["type"] = SsdStoreType(value);
    if (value.at("ssd_allocator").contains("path")) {
      value["path"] = value.at("ssd_allocator").at("path");
    }
    value.erase("dram_allocator");
    value.erase("tiering");
    j["value"] = value;
    return out;
  }

  bool ShouldUseDram(size_t size) const {
    if (dram_placement_bytes_ == 0) {
      return false;
    }
    const double next =
        static_cast<double>(
            dram_bytes_reserved_.load(std::memory_order_relaxed) + size) /
        static_cast<double>(dram_placement_bytes_);
    return next <= high_watermark_ratio_;
  }

  uint64_t PlaceStatic(uint64_t key, size_t size) {
    if (key > ssd_layout::kMaxKey) {
      throw std::invalid_argument("static SSD key is out of range");
    }
    const size_t row_size = ssd_store_->SlotCapacity(key + 1);
    if (size != row_size) {
      throw std::invalid_argument("static SSD layout value size mismatch");
    }
    if (ShouldUseDram(size)) {
      const uint64_t raw = dram_store_.Alloc(size);
      if (raw != kValueHandleNone) {
        dram_bytes_reserved_.fetch_add(
            dram_store_.SlotCapacity(raw), std::memory_order_relaxed);
        dram_live_allocs_.fetch_add(1, std::memory_order_relaxed);
        dram_total_allocs_.fetch_add(1, std::memory_order_relaxed);
        return MakeDramHandle(raw);
      }
    }
    ssd_live_allocs_.fetch_add(1, std::memory_order_relaxed);
    ssd_total_allocs_.fetch_add(1, std::memory_order_relaxed);
    return MakeSsdHandle(key + 1);
  }

  DramValueStore dram_store_;
  std::unique_ptr<ValueStore> ssd_store_;
  // Optional DRAM read cache for SSD-resident values. There is no destructor:
  // the cached slots belong to dram_store_, which is destroyed with this
  // object, so they are released with the allocator rather than one by one.
  std::unique_ptr<TieredReadCache> cache_;
  std::unique_ptr<TieredClockValueStore> clock_;
  bool ssd_static_              = false;
  uint64_t dram_capacity_bytes_ = 0;
  // Capacity available to primary placement; equals dram_capacity_bytes_ minus
  // the reserved read-cache budget.
  uint64_t dram_placement_bytes_ = 0;
  double high_watermark_ratio_   = 0.85;
  std::atomic<uint64_t> dram_bytes_reserved_{0};
  std::atomic<uint64_t> dram_live_allocs_{0};
  std::atomic<uint64_t> ssd_live_allocs_{0};
  std::atomic<uint64_t> dram_total_allocs_{0};
  std::atomic<uint64_t> ssd_total_allocs_{0};
};

FACTORY_REGISTER(
    ValueStore, TIERED_VALUE_STORE, HybridValueStore, const BaseKVConfig&);
