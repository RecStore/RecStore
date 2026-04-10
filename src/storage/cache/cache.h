#pragma once

#include <cstddef>
#include <vector>

template <typename Key>
class CachePolicy {
public:
  virtual ~CachePolicy() = default;

  // Record key insertion into cache policy state.
  virtual void OnInsert(const Key& key) = 0;

  // Record key access (hit) into cache policy state.
  virtual void OnAccess(const Key& key) = 0;

  // Evict up to `count` keys and return them in eviction order.
  virtual std::vector<Key> Evict(size_t count) = 0;

  // Current tracked key count.
  virtual size_t Size() const = 0;
};
