#pragma once

#include <assert.h>
#include <iostream>
#include <thread>
#include <unordered_map>

namespace base {
template <typename T, typename... Args>
struct FactoryCreator {
  FactoryCreator() {}
  virtual ~FactoryCreator() {}
  virtual T* create(Args... args) = 0;
};

template <typename T, typename... Args>
struct Factory {
  static std::unordered_map<std::string, FactoryCreator<T, Args...>*>&
  creators() {
    static std::unordered_map<std::string, FactoryCreator<T, Args...>*>
        creators;
    return creators;
  }
  static T* NewInstance(const std::string& key, Args... args) {
    auto creator = creators()[key];
    if (creator)
      return creator->create(args...);
    std::cerr << "no Factory instance for key: " << key << std::endl;
    std::exit(-1);
    return NULL;
  }
};

template <typename T, typename IMPL, typename... Args>
struct FactoryCreatorImpl : public FactoryCreator<T, Args...> {
  T* create(Args... args) { return new IMPL(args...); }
  FactoryCreatorImpl() {}
  explicit FactoryCreatorImpl(const std::string& key) {
    if (Factory<T, Args...>::creators().find(key) ==
        Factory<T, Args...>::creators().end()) {
      Factory<T, Args...>::creators().insert({key, new FactoryCreatorImpl()});
    }
  }
};

#define FACTORY_REGISTER(T, key, IMPL, ...)                                    \
  static base::FactoryCreatorImpl<T, IMPL, ##__VA_ARGS__>                      \
      FACTORY_REGISTER_##T##_##key(#key);

} // namespace base