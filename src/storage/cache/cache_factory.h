#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>

#include "storage/cache/lru/lru_cache.h"

namespace storage {
namespace cache {

inline std::string ToUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return s;
}

template <typename Key>
inline std::unique_ptr<CachePolicy<Key>>
CreateCachePolicy(const std::string& cache_policy, size_t capacity) {
  const std::string policy = ToUpper(cache_policy);
  if (policy.empty() || policy == "LRU") {
    return std::make_unique<LRUCachePolicy<Key>>(capacity);
  }
  throw std::invalid_argument("unsupported cache_policy: " + cache_policy);
}

} // namespace cache
} // namespace storage
