#pragma once

#include <cstddef>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "storage/cache/cache.h"

template <typename Key>
class LRUCachePolicy final : public CachePolicy<Key> {
public:
  explicit LRUCachePolicy(size_t capacity) : capacity_(capacity) {}

  void OnInsert(const Key& key) override {
    std::lock_guard<std::mutex> lock(mutex_);
    TouchLocked(key);
    TrimToCapacityLocked();
  }

  void OnAccess(const Key& key) override {
    std::lock_guard<std::mutex> lock(mutex_);
    TouchLocked(key);
  }

  std::vector<Key> Evict(size_t count) override {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Key> evicted;
    if (count == 0 || order_.empty()) {
      return evicted;
    }
    evicted.reserve(count);
    for (size_t i = 0; i < count && !order_.empty(); ++i) {
      const Key key = order_.back();
      order_.pop_back();
      map_.erase(key);
      evicted.push_back(key);
    }
    return evicted;
  }

  size_t Size() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return order_.size();
  }

  bool Contains(const Key& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return map_.find(key) != map_.end();
  }

private:
  void TouchLocked(const Key& key) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      order_.splice(order_.begin(), order_, it->second);
      return;
    }
    order_.push_front(key);
    map_[key] = order_.begin();
  }

  void TrimToCapacityLocked() {
    if (capacity_ == 0) {
      map_.clear();
      order_.clear();
      return;
    }
    while (order_.size() > capacity_) {
      const Key key = order_.back();
      order_.pop_back();
      map_.erase(key);
    }
  }

  const size_t capacity_;
  mutable std::mutex mutex_;
  std::list<Key> order_;
  std::unordered_map<Key, typename std::list<Key>::iterator> map_;
};
