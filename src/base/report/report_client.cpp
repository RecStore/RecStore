#include "report_client.h"

#include "base/config.h"
#include "base/timer.h"

#include <curl/curl.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace recstore {
thread_local uint64_t g_trace_id = 0;
}

size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp);

namespace {

constexpr const char* kDefaultApiUrl    = "http://127.0.0.1:8081/report";
constexpr const char* kDefaultJsonlPath = "recstore_report_events.jsonl";
// The placeholders keep per-process files apart: a PS server and every training
// rank call report() in the same run.
constexpr const char* kDefaultCsvPath = "recstore_metrics_{host}_{pid}.csv";
constexpr const char* kCsvHeader =
    "timestamp_us,host,pid,run_id,rank,shard,table_name,unique_id,"
    "metric_name,metric_value,source";

// Everything one report() call needs to decide where a sample goes. Resolved
// from recstore_config.json plus environment overrides.
struct ObservabilityConfig {
  bool remote = false;
  bool csv    = false;
  bool jsonl  = false;
  bool glog   = true; // Legacy default: every event is also logged.

  std::string remote_endpoint = kDefaultApiUrl;
  long remote_timeout_ms      = 3000;
  size_t remote_queue_max     = 10000;

  std::string csv_path     = kDefaultCsvPath;
  size_t csv_flush_every_n = 1;

  std::string jsonl_path     = kDefaultJsonlPath;
  size_t jsonl_flush_every_n = 1;
};

std::string ToLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value;
}

bool ParseLocalReportMode(const std::string& value) {
  const std::string normalized = ToLower(value);
  return normalized == "local" || normalized == "off" ||
         normalized == "disable" || normalized == "disabled" ||
         normalized == "false" || normalized == "0";
}

bool ParseRemoteReportMode(const std::string& value) {
  const std::string normalized = ToLower(value);
  return normalized == "grafana" || normalized == "remote" ||
         normalized == "on" || normalized == "true" || normalized == "1";
}

size_t ParsePositiveSize(const char* value, size_t fallback) {
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end                  = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end != value && *end == '\0' && parsed > 0) {
    return static_cast<size_t>(parsed);
  }
  LOG(WARNING) << "Invalid positive integer '" << value << "', fallback to "
               << fallback << ".";
  return fallback;
}

int64_t GetTimestampUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

const char* HostName() {
  static const std::string name = []() {
    char buffer[256] = {0};
    if (::gethostname(buffer, sizeof(buffer) - 1) != 0) {
      return std::string("unknown");
    }
    return std::string(buffer);
  }();
  return name.c_str();
}

int32_t ProcessId() {
  static const int32_t pid = static_cast<int32_t>(::getpid());
  return pid;
}

const char* RunId() {
  // Re-read per call: tests and launchers set RECSTORE_RUN_ID around a run,
  // and a process-wide cache would freeze whatever was present at first use.
  thread_local std::string run_id;
  const char* value = std::getenv("RECSTORE_RUN_ID");
  run_id.assign(value == nullptr ? "" : value);
  return run_id.c_str();
}

int32_t EnvInt(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return -1;
  }
  char* end    = nullptr;
  const long v = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    return -1;
  }
  return static_cast<int32_t>(v);
}

int32_t EnvRank() {
  const int32_t rank = EnvInt("RANK");
  return rank >= 0 ? rank : EnvInt("LOCAL_RANK");
}

int32_t EnvShard() {
  const int32_t shard = EnvInt("RECSTORE_SHARD_ID");
  return shard >= 0 ? shard : EnvInt("SHARD_ID");
}

void AppendJsonString(std::string& output, const char* value) {
  output.push_back('"');
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value);
       *p != '\0';
       ++p) {
    switch (*p) {
    case '"':
      output.append("\\\"");
      break;
    case '\\':
      output.append("\\\\");
      break;
    case '\b':
      output.append("\\b");
      break;
    case '\f':
      output.append("\\f");
      break;
    case '\n':
      output.append("\\n");
      break;
    case '\r':
      output.append("\\r");
      break;
    case '\t':
      output.append("\\t");
      break;
    default:
      if (*p < 0x20) {
        static constexpr char kHex[] = "0123456789abcdef";
        output.append("\\u00");
        output.push_back(kHex[*p >> 4]);
        output.push_back(kHex[*p & 0x0F]);
      } else {
        output.push_back(static_cast<char>(*p));
      }
      break;
    }
  }
  output.push_back('"');
}

template <typename T>
void AppendNumber(std::string& output, T value) {
  char buffer[64];
  auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (result.ec == std::errc()) {
    output.append(buffer, result.ptr);
  } else {
    output.append(std::to_string(value));
  }
}

// Wire format for the remote report service. Kept byte-compatible with the
// historical payload so existing ClickHouse ingestion does not need to change;
// the extra dimensions live in the CSV sink and in MetricSink implementations.
std::string ToJsonLine(const recstore::MetricRecord& record) {
  std::string output;
  output.reserve(
      160 + std::strlen(record.table_name) + std::strlen(record.unique_id) +
      std::strlen(record.metric_name) + std::strlen(record.source));
  output.append("{\"table_name\":");
  AppendJsonString(output, record.table_name);
  output.append(",\"unique_id\":");
  AppendJsonString(output, record.unique_id);
  output.append(",\"metric_name\":");
  AppendJsonString(output, record.metric_name);
  output.append(",\"metric_value\":");
  AppendNumber(output, record.metric_value);
  output.append(",\"timestamp_us\":");
  AppendNumber(output, record.timestamp_us);
  output.append(",\"source\":");
  AppendJsonString(output, record.source);
  output.push_back('}');
  return output;
}

void AppendCsvField(std::string& output, const char* value) {
  bool needs_quotes = false;
  for (const char* p = value; *p != '\0'; ++p) {
    if (*p == ',' || *p == '"' || *p == '\n' || *p == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) {
    output.append(value);
    return;
  }
  output.push_back('"');
  for (const char* p = value; *p != '\0'; ++p) {
    if (*p == '"') {
      output.append("\"\"");
    } else {
      output.push_back(*p);
    }
  }
  output.push_back('"');
}

std::string ExpandCsvPath(const std::string& pattern) {
  std::string path      = pattern;
  const std::string pid = std::to_string(ProcessId());
  const std::pair<const char*, const std::string> replacements[] = {
      {"{host}", HostName()},
      {"{pid}", pid},
  };
  for (const auto& [token, value] : replacements) {
    size_t pos             = 0;
    const size_t token_len = std::strlen(token);
    while ((pos = path.find(token, pos)) != std::string::npos) {
      path.replace(pos, token_len, value);
      pos += value.size();
    }
  }
  return path;
}

// ── Config resolution ──────────────────────────────────────────────────────

struct ConfigState {
  bool initialized = false;
  std::filesystem::path path;
  std::filesystem::file_time_type mtime{};
  ObservabilityConfig base;
  bool path_resolved = false;
  std::filesystem::path resolved_path;
};

ConfigState& State() {
  static ConfigState state;
  return state;
}

std::mutex& StateMutex() {
  static std::mutex mutex;
  return mutex;
}

void ApplySinkList(const std::string& list, ObservabilityConfig& config) {
  size_t start = 0;
  while (start <= list.size()) {
    const size_t comma      = list.find(',', start);
    const std::string token = ToLower(list.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start));
    if (token == "remote") {
      config.remote = true;
    } else if (token == "csv") {
      config.csv = true;
    } else if (token == "jsonl") {
      config.jsonl = true;
    } else if (token == "glog") {
      config.glog = true;
    } else if (!token.empty() && token != "none" && token != "null") {
      LOG(WARNING) << "Unknown observability sink '" << token << "'.";
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
}

void ApplyObservabilityBlock(const json& block, ObservabilityConfig& config) {
  if (block.contains("sinks") && block["sinks"].is_array()) {
    config.remote = false;
    config.csv    = false;
    config.jsonl  = false;
    config.glog   = false;
    for (const auto& entry : block["sinks"]) {
      if (!entry.is_string()) {
        continue;
      }
      const std::string name = ToLower(entry.get<std::string>());
      if (name == "remote") {
        config.remote = true;
      } else if (name == "csv") {
        config.csv = true;
      } else if (name == "jsonl") {
        config.jsonl = true;
      } else if (name == "glog") {
        config.glog = true;
      } else if (name != "none" && !name.empty()) {
        LOG(WARNING) << "Unknown observability sink '" << name << "'.";
      }
    }
  }

  if (block.contains("remote") && block["remote"].is_object()) {
    const auto& remote = block["remote"];
    if (remote.contains("endpoint") && remote["endpoint"].is_string()) {
      const std::string endpoint = remote["endpoint"].get<std::string>();
      if (!endpoint.empty()) {
        config.remote_endpoint = endpoint;
      }
    }
    if (remote.contains("timeout_ms") && remote["timeout_ms"].is_number()) {
      const long timeout = remote["timeout_ms"].get<long>();
      if (timeout > 0) {
        config.remote_timeout_ms = timeout;
      }
    }
    if (remote.contains("queue_max") && remote["queue_max"].is_number()) {
      const long limit = remote["queue_max"].get<long>();
      if (limit > 0) {
        config.remote_queue_max = static_cast<size_t>(limit);
      }
    }
  }

  if (block.contains("csv") && block["csv"].is_object()) {
    const auto& csv = block["csv"];
    if (csv.contains("path") && csv["path"].is_string()) {
      const std::string path = csv["path"].get<std::string>();
      if (!path.empty()) {
        config.csv_path = path;
      }
    }
    if (csv.contains("flush_every_n") && csv["flush_every_n"].is_number()) {
      const long n = csv["flush_every_n"].get<long>();
      if (n > 0) {
        config.csv_flush_every_n = static_cast<size_t>(n);
      }
    }
  }

  if (block.contains("jsonl") && block["jsonl"].is_object()) {
    const auto& jsonl = block["jsonl"];
    if (jsonl.contains("path") && jsonl["path"].is_string()) {
      const std::string path = jsonl["path"].get<std::string>();
      if (!path.empty()) {
        config.jsonl_path = path;
      }
    }
    if (jsonl.contains("flush_every_n") && jsonl["flush_every_n"].is_number()) {
      const long n = jsonl["flush_every_n"].get<long>();
      if (n > 0) {
        config.jsonl_flush_every_n = static_cast<size_t>(n);
      }
    }
  }
}

void ApplyConfigFile(const std::filesystem::path& path,
                     ObservabilityConfig& config) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return;
  }
  json root;
  try {
    ifs >> root;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Error reading " << path << ": " << e.what();
    return;
  }

  if (root.contains("observability") && root["observability"].is_object()) {
    ApplyObservabilityBlock(root["observability"], config);
    return;
  }

  // Legacy layout: report_API alone decides whether the remote sink is on.
  const std::string api = root.value("report_API", std::string());
  config.remote         = !api.empty();
  if (!api.empty()) {
    config.remote_endpoint = api;
  }
}

void ApplyEnvOverrides(ObservabilityConfig& config) {
  if (const char* list = std::getenv("RECSTORE_REPORT_SINKS");
      list != nullptr && list[0] != '\0') {
    config.remote = false;
    config.csv    = false;
    config.jsonl  = false;
    config.glog   = false;
    ApplySinkList(list, config);
  } else {
    if (const char* mode = std::getenv("RECSTORE_REPORT_MODE");
        mode != nullptr && mode[0] != '\0') {
      if (ParseLocalReportMode(mode)) {
        config.remote = false;
      } else if (ParseRemoteReportMode(mode)) {
        config.remote = true;
      }
    }
    // Legacy local sink selector. `csv` is new; the other three keep their
    // original meaning.
    if (const char* sink = std::getenv("RECSTORE_REPORT_LOCAL_SINK");
        sink != nullptr && sink[0] != '\0') {
      const std::string mode = ToLower(sink);
      if (mode == "jsonl" || mode == "both" || mode == "csv") {
        config.glog  = mode == "both";
        config.jsonl = mode == "jsonl" || mode == "both";
        config.csv   = mode == "csv";
      }
    }
  }

  if (const char* path = std::getenv("RECSTORE_REPORT_JSONL_PATH");
      path != nullptr && path[0] != '\0') {
    config.jsonl_path = path;
  }
  config.jsonl_flush_every_n = ParsePositiveSize(
      std::getenv("RECSTORE_REPORT_FLUSH_EVERY_N"), config.jsonl_flush_every_n);
  if (const char* path = std::getenv("RECSTORE_REPORT_CSV_PATH");
      path != nullptr && path[0] != '\0') {
    config.csv_path = path;
  }
  config.csv_flush_every_n =
      ParsePositiveSize(std::getenv("RECSTORE_REPORT_CSV_FLUSH_EVERY_N"),
                        config.csv_flush_every_n);
}

const std::filesystem::path& ConfigPath() {
  auto& state = State();
  if (!state.path_resolved) {
    state.path_resolved = true;
    const auto found    = base::FindRecStoreConfigPath();
    if (found.has_value()) {
      state.resolved_path = *found;
    }
  }
  return state.resolved_path;
}

ObservabilityConfig ResolveConfig() {
  std::lock_guard<std::mutex> lock(StateMutex());
  auto& state = State();

  const std::filesystem::path& path = ConfigPath();
  std::filesystem::file_time_type mtime{};
  if (!path.empty()) {
    std::error_code ec;
    mtime = std::filesystem::last_write_time(path, ec);
    if (ec) {
      mtime = {};
    }
  }

  if (!state.initialized || state.path != path || state.mtime != mtime) {
    state.base = ObservabilityConfig{};
    if (!path.empty()) {
      ApplyConfigFile(path, state.base);
    }
    state.path        = path;
    state.mtime       = mtime;
    state.initialized = true;
  }

  ObservabilityConfig effective = state.base;
  ApplyEnvOverrides(effective);
  return effective;
}

// ── Sinks ─────────────────────────────────────────────────────────────────

class AsyncReportQueue {
public:
  static AsyncReportQueue& GetInstance() {
    static AsyncReportQueue instance;
    return instance;
  }

  void Enqueue(const std::string& payload) {
    const size_t limit = ResolveConfig().remote_queue_max;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (queue_.size() >= limit) {
        queue_.pop();
      }
      queue_.push(payload);
    }
    cv_.notify_one();
  }

private:
  AsyncReportQueue() : stop_(false) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    worker_ = std::thread(&AsyncReportQueue::WorkerLoop, this);
  }

  ~AsyncReportQueue() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    curl_global_cleanup();
  }

  void WorkerLoop() {
    CURL* curl = curl_easy_init();
    if (!curl)
      return;

    struct curl_slist* headers =
        curl_slist_append(nullptr, "Content-Type: application/json");

    std::string response_buffer;
    std::string current_url;

    while (true) {
      std::string payload;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });

        if (stop_ && queue_.empty()) {
          break;
        }

        payload = std::move(queue_.front());
        queue_.pop();
      }

      const ObservabilityConfig config = ResolveConfig();
      if (current_url != config.remote_endpoint) {
        current_url = config.remote_endpoint;
        curl_easy_setopt(curl, CURLOPT_URL, current_url.c_str());
      }
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
      curl_easy_setopt(
          curl,
          CURLOPT_TIMEOUT,
          config.remote_timeout_ms / 1000 > 0
              ? config.remote_timeout_ms / 1000
              : 1L);

      response_buffer.clear();
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_buffer);
      curl_easy_setopt(curl, CURLOPT_POST, 1L);
      curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
      curl_easy_setopt(
          curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.length()));

      CURLcode res = curl_easy_perform(curl);
      if (res != CURLE_OK) {
        LOG(ERROR) << "CURL perform failed: " << curl_easy_strerror(res);
      } else {
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code < 200 || http_code >= 300) {
          LOG(ERROR) << "API failed (HTTP Code: " << http_code << ").";
        }
      }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
  }

  std::queue<std::string> queue_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_;
  std::thread worker_;
};

// Append-only file shared by the CSV and JSONL sinks. Samples arrive from
// several processes (the PS server and every training rank call report()) and
// may target the same path, so each batch goes out as one O_APPEND write under
// an advisory lock: lines stay whole and the header is written exactly once.
class AppendFile {
public:
  AppendFile()                             = default;
  AppendFile(const AppendFile&)            = delete;
  AppendFile& operator=(const AppendFile&) = delete;
  ~AppendFile() { Close(); }

  bool Open(const std::filesystem::path& path) {
    Close();
    const auto parent = path.parent_path();
    std::error_code ec;
    if (!parent.empty()) {
      std::filesystem::create_directories(parent, ec);
    }
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    return fd_ >= 0;
  }

  bool IsOpen() const { return fd_ >= 0; }

  // Writes `header` plus a newline when the file is still empty, under the
  // same lock that protects row batches.
  bool WriteHeaderIfEmpty(const std::string& header) {
    if (fd_ < 0) {
      return false;
    }
    const bool locked = ::flock(fd_, LOCK_EX) == 0;
    struct stat info {};
    const bool empty = ::fstat(fd_, &info) == 0 && info.st_size == 0;
    const bool wrote = empty && WriteAll(header + '\n');
    if (locked) {
      ::flock(fd_, LOCK_UN);
    }
    return wrote;
  }

  void Append(const std::string& data) { pending_.append(data); }

  void Flush() {
    if (fd_ < 0 || pending_.empty()) {
      return;
    }
    const bool locked = ::flock(fd_, LOCK_EX) == 0;
    if (!WriteAll(pending_)) {
      LOG(ERROR) << "Failed to append " << pending_.size()
                 << " bytes to report sink file";
    }
    if (locked) {
      ::flock(fd_, LOCK_UN);
    }
    pending_.clear();
  }

  void Close() {
    Flush();
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

private:
  bool WriteAll(const std::string& data) {
    const char* cursor = data.data();
    size_t remaining   = data.size();
    while (remaining > 0) {
      const ssize_t written = ::write(fd_, cursor, remaining);
      if (written > 0) {
        cursor += written;
        remaining -= static_cast<size_t>(written);
        continue;
      }
      if (written < 0 && errno == EINTR) {
        continue;
      }
      return false;
    }
    return true;
  }

  int fd_ = -1;
  std::string pending_;
};

class CsvSink {
public:
  static CsvSink& GetInstance() {
    static CsvSink instance;
    return instance;
  }

  void Emit(const recstore::MetricRecord& record,
            const ObservabilityConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    ConfigureIfNeeded(config);
    if (!file_.IsOpen()) {
      return;
    }

    std::string line;
    line.reserve(160);
    AppendNumber(line, record.timestamp_us);
    line.push_back(',');
    AppendCsvField(line, record.host);
    line.push_back(',');
    AppendNumber(line, record.pid);
    line.push_back(',');
    AppendCsvField(line, record.run_id);
    line.push_back(',');
    AppendNumber(line, record.rank);
    line.push_back(',');
    AppendNumber(line, record.shard);
    line.push_back(',');
    AppendCsvField(line, record.table_name);
    line.push_back(',');
    AppendCsvField(line, record.unique_id);
    line.push_back(',');
    AppendCsvField(line, record.metric_name);
    line.push_back(',');
    AppendNumber(line, record.metric_value);
    line.push_back(',');
    AppendCsvField(line, record.source);
    line.push_back('\n');

    file_.Append(line);
    ++writes_since_flush_;
    if (writes_since_flush_ >= config.csv_flush_every_n) {
      file_.Flush();
      writes_since_flush_ = 0;
    }
  }

  void Flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.Flush();
  }

private:
  CsvSink() = default;

  ~CsvSink() {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.Close();
  }

  void ConfigureIfNeeded(const ObservabilityConfig& config) {
    if (configured_ && path_ == config.csv_path &&
        flush_every_n_ == config.csv_flush_every_n) {
      return;
    }

    file_.Close();
    writes_since_flush_ = 0;
    path_               = config.csv_path;
    flush_every_n_      = config.csv_flush_every_n;

    const std::filesystem::path output_path(ExpandCsvPath(config.csv_path));
    if (!file_.Open(output_path)) {
      LOG(ERROR) << "Failed to open report CSV sink: " << output_path;
    } else {
      file_.WriteHeaderIfEmpty(kCsvHeader);
    }
    configured_ = true;
  }

  std::mutex mutex_;
  AppendFile file_;
  bool configured_ = false;
  std::string path_;
  size_t flush_every_n_      = 1;
  size_t writes_since_flush_ = 0;
};

class JsonlSink {
public:
  static JsonlSink& GetInstance() {
    static JsonlSink instance;
    return instance;
  }

  void Emit(const std::string& line, const ObservabilityConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    ConfigureIfNeeded(config);
    if (!file_.IsOpen()) {
      return;
    }
    file_.Append(line);
    file_.Append("\n");
    ++writes_since_flush_;
    if (writes_since_flush_ >= config.jsonl_flush_every_n) {
      file_.Flush();
      writes_since_flush_ = 0;
    }
  }

  void Flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.Flush();
  }

private:
  JsonlSink() = default;

  ~JsonlSink() {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.Close();
  }

  void ConfigureIfNeeded(const ObservabilityConfig& config) {
    if (configured_ && path_ == config.jsonl_path &&
        flush_every_n_ == config.jsonl_flush_every_n) {
      return;
    }

    file_.Close();
    writes_since_flush_ = 0;
    path_               = config.jsonl_path;
    flush_every_n_      = config.jsonl_flush_every_n;

    if (!file_.Open(config.jsonl_path)) {
      LOG(ERROR) << "Failed to open local report JSONL sink: "
                 << config.jsonl_path;
    }
    configured_ = true;
  }

  std::mutex mutex_;
  AppendFile file_;
  bool configured_ = false;
  std::string path_;
  size_t flush_every_n_      = 1;
  size_t writes_since_flush_ = 0;
};

class Dispatcher {
public:
  static Dispatcher& GetInstance() {
    static Dispatcher instance;
    return instance;
  }

  void AddSink(std::unique_ptr<recstore::MetricSink> sink) {
    if (!sink) {
      return;
    }
    std::unique_lock<std::shared_mutex> lock(user_mutex_);
    user_sinks_.push_back(std::move(sink));
  }

  void Emit(const recstore::MetricRecord& record) {
    const ObservabilityConfig config = ResolveConfig();

    std::string payload;
    auto json_payload = [&payload, &record]() -> const std::string& {
      if (payload.empty()) {
        payload = ToJsonLine(record);
      }
      return payload;
    };

    if (config.remote) {
      AsyncReportQueue::GetInstance().Enqueue(json_payload());
    }
    if (config.jsonl) {
      JsonlSink::GetInstance().Emit(json_payload(), config);
    }
    if (config.glog) {
      LOG(INFO) << "REPORT_LOCAL_EVENT " << json_payload();
    }
    if (config.csv) {
      CsvSink::GetInstance().Emit(record, config);
    }

    std::shared_lock<std::shared_mutex> lock(user_mutex_);
    for (auto& sink : user_sinks_) {
      sink->Emit(record);
    }
  }

  void Flush() {
    CsvSink::GetInstance().Flush();
    JsonlSink::GetInstance().Flush();
    std::shared_lock<std::shared_mutex> lock(user_mutex_);
    for (auto& sink : user_sinks_) {
      sink->Flush();
    }
  }

private:
  Dispatcher() = default;
  std::shared_mutex user_mutex_;
  std::vector<std::unique_ptr<recstore::MetricSink>> user_sinks_;
};

bool TryRecordLatencyMetricToTimer(const recstore::MetricRecord& record) {
  double value_ns = 0.0;
  if (std::strcmp(record.metric_name, "duration_ns") == 0 ||
      std::strcmp(record.metric_name, "latency_ns") == 0) {
    value_ns = record.metric_value;
  } else if (std::strcmp(record.metric_name, "duration_us") == 0 ||
             std::strcmp(record.metric_name, "latency_us") == 0) {
    value_ns = record.metric_value * 1000.0;
  } else {
    return false;
  }

  thread_local std::string cached_table;
  thread_local std::string cached_metric;
  thread_local std::string cached_timer_name;
  if (cached_table != record.table_name ||
      cached_metric != record.metric_name) {
    cached_table      = record.table_name;
    cached_metric     = record.metric_name;
    cached_timer_name = cached_table + "." + cached_metric;
  }
  xmh::Timer::ManualRecordNs(cached_timer_name, value_ns);
  return true;
}

recstore::MetricRecord
MakeRecord(const char* table_name,
           const char* unique_id,
           const char* metric_name,
           double metric_value,
           const char* source) {
  recstore::MetricRecord record;
  record.timestamp_us = GetTimestampUs();
  record.host         = HostName();
  record.pid          = ProcessId();
  record.run_id       = RunId();
  record.rank         = EnvRank();
  record.shard        = EnvShard();
  record.table_name   = table_name;
  record.unique_id    = unique_id;
  record.metric_name  = metric_name;
  record.metric_value = metric_value;
  record.source       = source;
  return record;
}

} // namespace

size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
  ((std::string*)userp)->append((char*)contents, size * nmemb);
  return size * nmemb;
}

bool send_json_request(const std::string& json_payload) {
  if (!ResolveConfig().remote) {
    return true;
  }
  AsyncReportQueue::GetInstance().Enqueue(json_payload);
  return true;
}

bool is_report_remote_enabled_for_test() { return ResolveConfig().remote; }

bool is_report_local_jsonl_enabled_for_test() { return ResolveConfig().jsonl; }

bool is_report_csv_enabled_for_test() { return ResolveConfig().csv; }

void ResetReportConfigForTest() {
  std::lock_guard<std::mutex> lock(StateMutex());
  State()               = ConfigState{};
  State().path_resolved = false;
  State().resolved_path.clear();
}

namespace recstore {

void RegisterMetricSink(std::unique_ptr<MetricSink> sink) {
  Dispatcher::GetInstance().AddSink(std::move(sink));
}

} // namespace recstore

extern "C" bool
report(const char* table_name,
       const char* unique_id,
       const char* metric_name,
       double metric_value) {
  const recstore::MetricRecord record =
      MakeRecord(table_name, unique_id, metric_name, metric_value, "report");

  TryRecordLatencyMetricToTimer(record);
  Dispatcher::GetInstance().Emit(record);
  return true;
}

extern "C" bool report_flame_graph(
    const char* table_name, const char* unique_id, const FlameGraphData& data) {
  std::string combined_uid = std::string(unique_id) + "|" + data.label;

  auto now        = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);
  struct tm gmt_tm;
  gmtime_r(&now_time_t, &gmt_tm);
  gmt_tm.tm_hour           = 0;
  gmt_tm.tm_min            = 0;
  gmt_tm.tm_sec            = 0;
  auto start_of_day_s      = timegm(&gmt_tm);
  uint64_t start_of_day_us = static_cast<uint64_t>(start_of_day_s) * 1000000;

  double adjusted_start = data.start - static_cast<double>(start_of_day_us);

  report(table_name, combined_uid.c_str(), "level", data.level);
  report(table_name, combined_uid.c_str(), "value", data.value);
  report(table_name, combined_uid.c_str(), "self", data.self);
  report(table_name, combined_uid.c_str(), "start", adjusted_start);

  return true;
}

ReportTimeline::ReportTimeline(const std::string& table_name,
                               const std::string& unique_id,
                               const std::string& metric_name)
    : table_name_(table_name),
      unique_id_(unique_id),
      metric_name_(metric_name),
      is_done_(false) {
  report(table_name_.c_str(),
         unique_id_.c_str(),
         metric_name_.c_str(),
         static_cast<double>(STATE_RUNNING));
}

ReportTimeline::~ReportTimeline() {
  if (!is_done_) {
    done(STATE_SUCCESS);
  }
}

void ReportTimeline::done(TimelineState state) {
  if (!is_done_) {
    report(table_name_.c_str(),
           unique_id_.c_str(),
           metric_name_.c_str(),
           static_cast<double>(state));
    is_done_ = true;
  }
}
