// cceh_db.cc
#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>
#if defined(_MSC_VER)
#  include "direct.h"
#  define mkdir(x, y) _mkdir(x)
#endif

#include "core/db.h"
#include "core/core_workload.h"
#include "core/db_factory.h"
#include "utils/properties.h"
#include "utils/utils.h"

// RecStore headers
#include "../../../src/storage/kv_engine/base_kv.h"
#include "../../../src/base/factory.h"
#include "../../../src/memory/shm_file.h"
#include "../../../src/storage/kv_engine/ssd_index_ssd_value.h"

using ycsbc::DB;

namespace ycsbc {

static std::atomic<unsigned> g_next_tid{0};
static inline unsigned ThreadTid() {
  thread_local unsigned t = g_next_tid.fetch_add(1, std::memory_order_relaxed);
  return t;
}

// FNV-1a 64
static inline uint64_t fnv1a64(const std::string& s) {
  const uint64_t O = 1469598103934665603ull, P = 1099511628211ull;
  uint64_t h = O;
  for (unsigned char c : s) {
    h ^= c;
    h *= P;
  }
  return h;
}

// 兼容 "user12345"；若非纯数字落到 FNV
static inline uint64_t ToKey(const std::string& k) {
  try {
    size_t pos = 0;
    uint64_t v = std::stoull(k, &pos, 10);
    if (pos == k.size())
      return v;
  } catch (...) {
  }
  return fnv1a64(k);
}

class CCEHDB : public DB {
public:
  CCEHDB() = default;
  ~CCEHDB() override { Cleanup(); }

  void Init() override {
    const utils::Properties& p = *props_;
    std::lock_guard<std::mutex> lg(mu_);

    if (ref_cnt_++ > 0)
      return;

    field_cnt_ =
        std::stoi(p.GetProperty(ycsbc::CoreWorkload::FIELD_COUNT_PROPERTY,
                                ycsbc::CoreWorkload::FIELD_COUNT_DEFAULT));

    base::PMMmapRegisterCenter::GetConfig().use_dram = true; // 简化部署

    std::string path = p.GetProperty(
        "cceh.path",
        "/home/xieminhui/tgj/RecStore/third_party/YCSB-cpp/data-store");
    size_t capacity = std::stoull(p.GetProperty("cceh.capacity", "16777216"));
    value_size_     = std::stoull(p.GetProperty("cceh.value_size", "1000"));

    if (mkdir(path.c_str(), 0775) && errno != EEXIST) {
      throw utils::Exception(std::string("mkdir failed: ") + strerror(errno));
    }

    BaseKVConfig cfg;
    cfg.json_config_["path"]       = path;
    cfg.json_config_["capacity"]   = capacity;
    cfg.json_config_["value_size"] = value_size_;
    cfg.json_config_["queue_size"] =
        std::stoi(p.GetProperty("cceh.queue_size", "1024"));

    engine_ = new KVEngineCCEH(cfg);
  }

  void Cleanup() override {
    std::lock_guard<std::mutex> lg(mu_);
    if (ref_cnt_ == 0)
      return;
    if (--ref_cnt_ == 0) {
      delete engine_;
      engine_ = nullptr;
    }
  }

  Status Read(const std::string&,
              const std::string& key,
              const std::vector<std::string>* fields,
              std::vector<Field>& result) override {
    if (!engine_)
      return kError;
    std::string blob;
    engine_->Get(ToKey(key), blob, ThreadTid());
    if (blob.empty())
      return kNotFound;
    if (fields) {
      DeserializeRowFilter(&result, blob.data(), blob.size(), *fields);
    } else {
      DeserializeRow(&result, blob.data(), blob.size());
    }
    return kOK;
  }

  Status Scan(const std::string&,
              const std::string&,
              int,
              const std::vector<std::string>*,
              std::vector<std::vector<Field>>&) override {
    return kNotFound;
  }

  Status Update(const std::string&,
                const std::string& key,
                std::vector<Field>& values) override {
    if (!engine_)
      return kError;
    std::string blob;
    engine_->Get(ToKey(key), blob, ThreadTid());
    if (blob.empty())
      return kNotFound;

    std::vector<Field> cur;
    DeserializeRow(&cur, blob.data(), blob.size());
    for (auto& nf : values) {
      bool found = false;
      for (auto& cf : cur)
        if (cf.name == nf.name) {
          cf.value = nf.value;
          found    = true;
          break;
        }
      assert(found);
    }
    std::string out;
    SerializeRow(cur, &out);
    engine_->Put(
        ToKey(key), std::string_view(out.data(), out.size()), ThreadTid());
    return kOK;
  }

  Status Insert(const std::string&,
                const std::string& key,
                std::vector<Field>& values) override {
    if (!engine_)
      return kError;
    std::string out;
    SerializeRow(values, &out);
    engine_->Put(
        ToKey(key), std::string_view(out.data(), out.size()), ThreadTid());
    return kOK;
  }

  Status Delete(const std::string&, const std::string& key) override {
    if (!engine_)
      return kError;
    engine_->Put(ToKey(key), std::string_view(), ThreadTid()); // 简易“删除”
    return kOK;
  }

private:
  static void SerializeRow(const std::vector<Field>& values, std::string* out) {
    // Build body: [len name][len value]...
    std::string body;
    body.reserve(256);
    for (const auto& f : values) {
      uint32_t nlen = static_cast<uint32_t>(f.name.size());
      uint32_t vlen = static_cast<uint32_t>(f.value.size());
      body.append(reinterpret_cast<const char*>(&nlen), sizeof(uint32_t));
      body.append(f.name.data(), f.name.size());
      body.append(reinterpret_cast<const char*>(&vlen), sizeof(uint32_t));
      body.append(f.value.data(), f.value.size());
    }
    // Prefix payload_len (without padding)
    out->clear();
    uint32_t payload_len = static_cast<uint32_t>(body.size());
    out->append(reinterpret_cast<const char*>(&payload_len), sizeof(uint32_t));
    out->append(body);
    // Pad to fixed value_size_ bytes if configured (>0)
    if (value_size_ > 0 && out->size() < value_size_) {
      out->resize(value_size_, '\0');
    }
  }

  static void DeserializeRowFilter(
      std::vector<Field>* values,
      const char* p,
      size_t n,
      const std::vector<std::string>& fields) {
    values->clear();
    if (n < sizeof(uint32_t))
      return;
    uint32_t payload_len = *reinterpret_cast<const uint32_t*>(p);
    p += sizeof(uint32_t);
    if (payload_len > n - sizeof(uint32_t))
      payload_len = static_cast<uint32_t>(n - sizeof(uint32_t));
    const char* lim = p + payload_len;

    auto it = fields.begin();
    while (p < lim && it != fields.end()) {
      if (p + sizeof(uint32_t) > lim)
        break;
      uint32_t nlen = *reinterpret_cast<const uint32_t*>(p);
      p += 4;
      if (p + nlen > lim)
        break;
      std::string name(p, nlen);
      p += nlen;
      if (p + sizeof(uint32_t) > lim)
        break;
      uint32_t vlen = *reinterpret_cast<const uint32_t*>(p);
      p += 4;
      if (p + vlen > lim)
        break;
      std::string val(p, vlen);
      p += vlen;
      if (*it == name) {
        values->push_back({name, val});
        ++it;
      }
    }
  }

  static void
  DeserializeRow(std::vector<Field>* values, const char* p, size_t n) {
    values->clear();
    if (n < sizeof(uint32_t))
      return;
    uint32_t payload_len = *reinterpret_cast<const uint32_t*>(p);
    p += sizeof(uint32_t);
    if (payload_len > n - sizeof(uint32_t))
      payload_len = static_cast<uint32_t>(n - sizeof(uint32_t));
    const char* lim = p + payload_len;
    while (p < lim) {
      if (p + sizeof(uint32_t) > lim)
        break;
      uint32_t nlen = *reinterpret_cast<const uint32_t*>(p);
      p += 4;
      if (p + nlen > lim)
        break;
      std::string name(p, nlen);
      p += nlen;
      if (p + sizeof(uint32_t) > lim)
        break;
      uint32_t vlen = *reinterpret_cast<const uint32_t*>(p);
      p += 4;
      if (p + vlen > lim)
        break;
      std::string val(p, vlen);
      p += vlen;
      values->push_back({name, val});
    }
  }

private:
  static inline KVEngineCCEH* engine_ = nullptr;
  static inline size_t field_cnt_     = 0;
  static inline size_t value_size_    = 0;
  static inline int ref_cnt_          = 0;
  static inline std::mutex mu_;
};

static DB* NewCCEH() { return new CCEHDB(); }
// 注册到 YCSB
const bool registered_cceh = ycsbc::DBFactory::RegisterDB("cceh", NewCCEH);

} // namespace ycsbc
