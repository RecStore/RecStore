// engine_selector.h
#pragma once
#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "memory/allocators/allocator_factory.h"
#include "base_kv.h"

namespace base {

inline std::string Upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return s;
}

inline void CanonicalizeAlias(json& j, const char* lhs, const char* rhs) {
  const bool has_lhs = j.contains(lhs);
  const bool has_rhs = j.contains(rhs);
  if (has_lhs && has_rhs && j.at(lhs) != j.at(rhs)) {
    throw std::invalid_argument(
        std::string(lhs) + " conflicts with alias " + rhs);
  }
  if (!has_lhs && has_rhs)
    j[lhs] = j.at(rhs);
  if (has_lhs && !has_rhs)
    j[rhs] = j.at(lhs);
}

struct EngineResolved {
  std::string
      engine; // "KVEngineExtendibleHash" / "KVEngineCCEH" / "KVEngineHybrid"
  BaseKVConfig cfg; // 归一化后的配置（写回 engine_type、补默认/剔无关）
};

inline EngineResolved ResolveEngine(BaseKVConfig cfg) {
  auto& j = cfg.json_config_;

  // 1) 字段别名统一（新老字段双向兼容）
  CanonicalizeAlias(j, "path", "DATA_DIR");
  CanonicalizeAlias(j, "index_type", "index_medium");
  CanonicalizeAlias(j, "value_type", "value_medium");
  CanonicalizeAlias(j, "capacity", "ENTRY_CAPACITY");
  CanonicalizeAlias(j, "value_size", "VALUE_SIZE_BYTES");
  CanonicalizeAlias(j, "shmcapacity", "DRAM_SIZE");
  CanonicalizeAlias(j, "ssdcapacity", "SSD_SIZE");
  CanonicalizeAlias(j, "queue_cnt", "IO_QUEUE_COUNT");
  base::allocators::NormalizeAllocatorConfig(j);

  std::string idx = Upper(j.value("index_type", "DRAM")); // 缺省 DRAM 索引
  std::string val = Upper(j.value("value_type", ""));     // DRAM / SSD / HYBRID
  if (val.empty())
    throw std::invalid_argument("value_type is required (DRAM/SSD/HYBRID)");

  if (idx == "DRAM/SSD")
    idx = "DRAM";
  j["index_type"]   = idx;
  j["index_medium"] = idx;
  j["value_type"]   = val;
  j["value_medium"] = val;

  const bool mode_explicit = j.contains("mode");
  std::string mode =
      Upper(j.value("mode", (val == "HYBRID") ? "DYNAMIC" : "STATIC"));
  if (mode != "STATIC" && mode != "DYNAMIC")
    throw std::invalid_argument("mode must be STATIC or DYNAMIC");
  j["mode"] = mode;

  if (mode_explicit) {
    const bool static_ok =
        ((idx == "DRAM" && (val == "DRAM" || val == "SSD")) ||
         (idx == "SSD" && val == "SSD"));
    const bool dynamic_ok =
        ((idx == "DRAM" || idx == "SSD") && val == "HYBRID");
    if (mode == "STATIC" && !static_ok) {
      throw std::invalid_argument(
          "STATIC mode supports only: (DRAM,DRAM), (DRAM,SSD), (SSD,SSD)");
    }
    if (mode == "DYNAMIC" && !dynamic_ok) {
      throw std::invalid_argument(
          "DYNAMIC mode supports only HYBRID value with DRAM/SSD index");
    }
  }

  std::string engine;
  if (val == "HYBRID") {
    engine = "KVEngineHybrid";
    if (idx != "DRAM" && idx != "SSD")
      throw std::invalid_argument(
          "HYBRID value currently supports DRAM/SSD index");
    // Hybrid 需要 DRAM/SSD 双层容量
    if (!j.contains("shmcapacity") || !j.contains("ssdcapacity")) {
      throw std::invalid_argument("HYBRID requires DRAM_SIZE/SSD_SIZE (or "
                                  "legacy shmcapacity/ssdcapacity)");
    }
    j["DRAM_SIZE"] = j.at("shmcapacity");
    j["SSD_SIZE"]  = j.at("ssdcapacity");
    if (!j.contains("cache_policy"))
      j["cache_policy"] = "LRU";
  } else if (val == "SSD") {
    if (idx == "DRAM")
      engine = "KVEngineExtendibleHash";
    else if (idx == "SSD")
      engine = "KVEngineCCEH";
    else
      throw std::invalid_argument("index_type must be DRAM or SSD");

    // 非 Hybrid 固定长度路径至少需要 value_size。
    // capacity 可选：若缺失则优先由 DRAM_SIZE/SSD_SIZE 反推。
    if (!j.contains("value_size")) {
      throw std::invalid_argument(
          "Non-HYBRID requires VALUE_SIZE_BYTES (or legacy value_size)");
    }
    const uint64_t value_size = j.at("value_size").get<uint64_t>();
    if (value_size == 0) {
      throw std::invalid_argument("value_size must be > 0");
    }
    if (!j.contains("capacity")) {
      uint64_t bytes_budget = 0;
      if (val == "DRAM" && j.contains("DRAM_SIZE"))
        bytes_budget = j.at("DRAM_SIZE").get<uint64_t>();
      else if (val == "SSD" && j.contains("SSD_SIZE"))
        bytes_budget = j.at("SSD_SIZE").get<uint64_t>();

      if (bytes_budget == 0) {
        throw std::invalid_argument(
            "Non-HYBRID requires ENTRY_CAPACITY (legacy capacity) or "
            "DRAM_SIZE/SSD_SIZE to infer it");
      }
      j["capacity"] = std::max<uint64_t>(1, bytes_budget / value_size);
    }

    j.erase("shmcapacity");
    j.erase("ssdcapacity");
    if (val == "DRAM" && !j.contains("DRAM_SIZE")) {
      j["DRAM_SIZE"] =
          j.at("capacity").get<uint64_t>() * j.at("value_size").get<uint64_t>();
    }
    if (val == "SSD" && !j.contains("SSD_SIZE")) {
      j["SSD_SIZE"] =
          j.at("capacity").get<uint64_t>() * j.at("value_size").get<uint64_t>();
    }
  } else {
    throw std::invalid_argument("value_type must be DRAM/SSD/HYBRID");
  }

  // SSD index 默认 IO 配置，避免遗漏导致运行时报错
  if (idx == "SSD") {
    if (!j.contains("io_backend_type"))
      j["io_backend_type"] = "IOURING";
    if (!j.contains("queue_cnt"))
      j["queue_cnt"] = 512;
    if (!j.contains("page_id_offset"))
      j["page_id_offset"] = 0;
    if (!j.contains("file_path") && j.contains("path")) {
      j["file_path"] = j.at("path").get<std::string>() + "/index_cceh.db";
    }
  }

  // 变长 value 支持边界：当前仅 HYBRID(ValueManager 路径) 支持
  std::string value_layout =
      Upper(j.value("value_layout", (val == "HYBRID") ? "VARIABLE" : "FIXED"));
  if (value_layout != "FIXED" && value_layout != "VARIABLE") {
    throw std::invalid_argument("value_layout must be FIXED or VARIABLE");
  }
  if (value_layout == "VARIABLE" && val != "HYBRID") {
    throw std::invalid_argument("VARIABLE value_layout is only supported by "
                                "ValueManager (HYBRID mode)");
  }
  j["value_layout"]          = value_layout;
  j["variable_value_enable"] = (value_layout == "VARIABLE");

  // 若用户显式写了 engine_type，做一致性校验
  if (j.contains("engine_type")) {
    if (Upper(j["engine_type"].get<std::string>()) != Upper(engine)) {
      throw std::invalid_argument(
          "engine_type conflicts with (index_type,value_type) deduction");
    }
  }
  j["engine_type"] = engine; // 回写，保持向后兼容

  return EngineResolved{engine, std::move(cfg)};
}

} // namespace base
