#pragma once

#include <set>
#include <stdexcept>
#include <string>

#include "base_kv.h"

namespace base {

struct EngineResolved {
  std::string engine;
  BaseKVConfig cfg;
};

inline EngineResolved ResolveEngine(BaseKVConfig cfg) {
  auto& j = cfg.json_config_;

  if (j.contains("external_engine_type")) {
    throw std::invalid_argument(
        "external_engine_type is removed; use engine_type");
  }
  const std::string engine =
      j.contains("engine_type")
          ? j.at("engine_type").get<std::string>()
          : "KVEngineComposite";
  static const std::set<std::string> kKnownEngines = {
      "KVEngineComposite",
      "KVEnginePetKV",
      "KVEngineDETable",
      "KVEngineFasterKV",
      "KVEngineHPSHashMap",
      "KVEngineHPSRocksDB",
  };
  if (!kKnownEngines.count(engine)) {
    throw std::invalid_argument("unknown engine_type: " + engine);
  }

  return EngineResolved{engine, std::move(cfg)};
}

} // namespace base
