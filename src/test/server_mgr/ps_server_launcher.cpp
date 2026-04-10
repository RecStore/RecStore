#include "ps_server_launcher.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "base/json.h"

namespace recstore::test {
namespace {

constexpr size_t kMaxRecentLogs = 300;

std::string Trim(const std::string& s) {
  const auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

std::optional<int> ParseIntEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }
  char* end   = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    return std::nullopt;
  }
  return static_cast<int>(parsed);
}

std::vector<int> ParsePortsEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return {};
  }
  std::vector<int> ports;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::string trimmed = Trim(item);
    if (trimmed.empty()) {
      continue;
    }
    char* end   = nullptr;
    long parsed = std::strtol(trimmed.c_str(), &end, 10);
    if (end == trimmed.c_str() || *end != '\0') {
      return {};
    }
    ports.push_back(static_cast<int>(parsed));
  }
  return ports;
}

bool IsTruthyEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  std::string lowered(value);
  std::transform(
      lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return lowered == "1" || lowered == "true" || lowered == "yes";
}

std::filesystem::path MakeAbsolute(const std::filesystem::path& path) {
  if (path.empty()) {
    return {};
  }
  if (path.is_absolute()) {
    return path;
  }
  return std::filesystem::absolute(path);
}

std::string TimestampNow() {
  auto now      = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm_buf;
  localtime_r(&t, &tm_buf);
  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
  return oss.str();
}

} // namespace

PSServerLauncher::PSServerLauncher(LauncherOptions options)
    : options_(std::move(options)) {
  if (options_.server_path.empty()) {
    options_.server_path = FindPsServerBinary();
  } else {
    options_.server_path = MakeAbsolute(options_.server_path);
  }

  if (options_.config_path.empty()) {
    auto config = FindConfigFile();
    if (config.has_value()) {
      options_.config_path = *config;
    }
  } else {
    options_.config_path = MakeAbsolute(options_.config_path);
  }

  if (options_.log_dir.empty()) {
    options_.log_dir = "./logs";
  }
  options_.log_dir = MakeAbsolute(options_.log_dir);
}

PSServerLauncher::~PSServerLauncher() { Stop(); }

LauncherOptions PSServerLauncher::LoadOptionsFromEnvironment() {
  LauncherOptions options;

  const char* server_path = std::getenv("PS_SERVER_PATH");
  if (server_path != nullptr && *server_path != '\0') {
    options.server_path = MakeAbsolute(server_path);
  } else {
    options.server_path = FindPsServerBinary();
  }

  const char* config_path = std::getenv("RECSTORE_CONFIG");
  if (config_path != nullptr && *config_path != '\0') {
    options.config_path = MakeAbsolute(config_path);
  } else {
    auto discovered_config = FindConfigFile();
    if (discovered_config.has_value()) {
      options.config_path = *discovered_config;
    }
  }

  const char* log_dir = std::getenv("PS_LOG_DIR");
  if (log_dir != nullptr && *log_dir != '\0') {
    options.log_dir = MakeAbsolute(log_dir);
  } else {
    options.log_dir = MakeAbsolute("./logs");
  }

  if (auto timeout = ParseIntEnv("PS_TIMEOUT");
      timeout.has_value() && *timeout > 0) {
    options.startup_timeout_sec = *timeout;
  }
  if (auto shard_num = ParseIntEnv("PS_NUM_SHARDS");
      shard_num.has_value() && *shard_num > 0) {
    options.num_shards = *shard_num;
  }

  const char* override_ps_type = std::getenv("PS_SERVER_PS_TYPE");
  if (override_ps_type != nullptr && *override_ps_type != '\0') {
    options.override_ps_type = std::string(override_ps_type);
  }
  options.override_ports = ParsePortsEnv("PS_SERVER_PORTS");

  return options;
}

std::vector<int> PSServerLauncher::DefaultPorts() {
  return {15000, 15001, 15002, 15003};
}

std::vector<int> PSServerLauncher::ExtractPortsFromConfig(
    const std::filesystem::path& config_path) {
  if (config_path.empty()) {
    return DefaultPorts();
  }
  std::ifstream in(config_path);
  if (!in.good()) {
    return DefaultPorts();
  }

  try {
    json config;
    in >> config;
    std::vector<int> ports;

    if (config.contains("cache_ps") && config["cache_ps"].contains("servers")) {
      for (const auto& server : config["cache_ps"]["servers"]) {
        if (server.contains("port")) {
          ports.push_back(server["port"].get<int>());
        }
      }
    }

    if (ports.empty() && config.contains("distributed_client") &&
        config["distributed_client"].contains("servers")) {
      for (const auto& server : config["distributed_client"]["servers"]) {
        if (server.contains("port")) {
          ports.push_back(server["port"].get<int>());
        }
      }
    }

    if (ports.empty()) {
      return DefaultPorts();
    }

    std::sort(ports.begin(), ports.end());
    ports.erase(std::unique(ports.begin(), ports.end()), ports.end());
    return ports;
  } catch (...) {
    return DefaultPorts();
  }
}

bool PSServerLauncher::IsPortOpen(
    const std::string& host, int port, int timeout_sec) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    return false;
  }

  struct timeval timeout;
  timeout.tv_sec  = timeout_sec;
  timeout.tv_usec = 0;
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port   = htons(static_cast<uint16_t>(port));
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    close(sock);
    return false;
  }

  int rc = connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  close(sock);
  return rc == 0;
}

std::vector<int>
PSServerLauncher::CheckOpenPorts(const std::vector<int>& ports) {
  std::vector<int> open_ports;
  for (int port : ports) {
    if (IsPortOpen("127.0.0.1", port)) {
      open_ports.push_back(port);
    }
  }
  return open_ports;
}

LaunchDecision
PSServerLauncher::EvaluateLaunchDecision(const LauncherOptions& options) {
  LaunchDecision decision;
  decision.configured_ports = ExtractPortsFromConfig(options.config_path);
  if (!options.override_ports.empty()) {
    decision.configured_ports = options.override_ports;
  }
  decision.open_ports = CheckOpenPorts(decision.configured_ports);

  const bool all_ports_ready =
      !decision.configured_ports.empty() &&
      decision.open_ports.size() == decision.configured_ports.size();
  const bool partial_ports_open =
      !decision.open_ports.empty() && !all_ports_ready;

  if (IsTruthyEnv("NO_PS_SERVER")) {
    decision.should_start = false;
    decision.reason       = "NO_PS_SERVER";
    return decision;
  }

  if (partial_ports_open) {
    std::ostringstream oss;
    oss << "ps_server ports are partially available: expected=[";
    for (size_t i = 0; i < decision.configured_ports.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << decision.configured_ports[i];
    }
    oss << "], open=[";
    for (size_t i = 0; i < decision.open_ports.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << decision.open_ports[i];
    }
    oss << "]";

    decision.should_start = false;
    decision.should_fail  = true;
    decision.reason       = oss.str();
    return decision;
  }

  const bool is_ci = IsTruthyEnv("CI") || IsTruthyEnv("GITHUB_ACTIONS");
  if (is_ci && all_ports_ready) {
    decision.should_start = false;
    decision.reason       = "ci_reuse_running";
    return decision;
  }

  if (all_ports_ready) {
    decision.should_start = false;
    decision.reason       = "already_running";
    return decision;
  }

  decision.should_start = true;
  decision.reason       = is_ci ? "ci_server_not_ready" : "server_not_ready";
  return decision;
}

std::optional<std::filesystem::path> PSServerLauncher::FindConfigFile() {
  const char* from_env = std::getenv("RECSTORE_CONFIG");
  if (from_env != nullptr && *from_env != '\0') {
    std::filesystem::path env_path = MakeAbsolute(from_env);
    if (std::filesystem::exists(env_path)) {
      return env_path;
    }
  }

  std::filesystem::path current = std::filesystem::current_path();
  while (!current.empty()) {
    std::filesystem::path candidate = current / "recstore_config.json";
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }

    std::filesystem::path parent = current.parent_path();
    if (parent == current) {
      break;
    }
    current = parent;
  }

  return std::nullopt;
}

std::filesystem::path PSServerLauncher::FindPsServerBinary() {
  const char* from_env = std::getenv("PS_SERVER_PATH");
  if (from_env != nullptr && *from_env != '\0') {
    return MakeAbsolute(from_env);
  }

  const std::vector<std::filesystem::path> candidates = {
      "./bin/ps_server",
      "./build/bin/ps_server",
      "../bin/ps_server",
      "../../build/bin/ps_server",
      "../../../build/bin/ps_server",
      "../../../../build/bin/ps_server"};

  for (const auto& candidate : candidates) {
    auto abs = MakeAbsolute(candidate);
    if (std::filesystem::exists(abs)) {
      return abs;
    }
  }
  return MakeAbsolute("./build/bin/ps_server");
}

void PSServerLauncher::SetError(const std::string& error) {
  std::lock_guard<std::mutex> lock(mu_);
  last_error_ = error;
}

void PSServerLauncher::AppendLogLine(const std::string& line) {
  std::lock_guard<std::mutex> lock(mu_);
  recent_logs_.push_back(line);
  if (recent_logs_.size() > kMaxRecentLogs) {
    recent_logs_.erase(recent_logs_.begin());
  }
  if (log_file_.is_open()) {
    log_file_ << line << std::endl;
  }
  if (options_.verbose) {
    std::cout << line << std::endl;
  }
}

std::optional<int> PSServerLauncher::ParseReadyShard(const std::string& line) {
  if (line.find("listening on") == std::string::npos) {
    return std::nullopt;
  }

  const std::string shard_marker = "Server shard";
  auto shard_pos                 = line.find(shard_marker);
  if (shard_pos != std::string::npos) {
    std::string remain = line.substr(shard_pos + shard_marker.size());
    std::istringstream iss(remain);
    int shard_id = -1;
    if (iss >> shard_id) {
      return shard_id;
    }
  }

  if (line.find("Server listening on") != std::string::npos) {
    return 0;
  }

  return std::nullopt;
}

void PSServerLauncher::OutputThreadLoop(int fd,
                                        const std::string& stream_name) {
  FILE* stream = fdopen(fd, "r");
  if (stream == nullptr) {
    AppendLogLine("[" + stream_name + "] failed to open pipe");
    return;
  }

  char* line = nullptr;
  size_t len = 0;
  while (true) {
    if (stop_output_threads_) {
      break;
    }

    ssize_t read = getline(&line, &len, stream);
    if (read < 0) {
      break;
    }

    std::string content = Trim(line);
    std::string tagged  = "[" + stream_name + "] " + content;
    AppendLogLine(tagged);

    auto ready_shard = ParseReadyShard(content);
    if (ready_shard.has_value()) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        ready_shards_.insert(*ready_shard);
      }
      cv_.notify_all();
    }
  }

  if (line != nullptr) {
    free(line);
  }
  fclose(stream);
}

void PSServerLauncher::StartOutputThreads() {
  stop_output_threads_ = false;
  if (stdout_fd_ >= 0) {
    stdout_thread_ =
        std::thread([this]() { OutputThreadLoop(stdout_fd_, "STDOUT"); });
  }
  if (stderr_fd_ >= 0) {
    stderr_thread_ =
        std::thread([this]() { OutputThreadLoop(stderr_fd_, "STDERR"); });
  }
}

void PSServerLauncher::StopOutputThreads() {
  stop_output_threads_ = true;
  if (stdout_thread_.joinable()) {
    stdout_thread_.join();
  }
  if (stderr_thread_.joinable()) {
    stderr_thread_.join();
  }
  stdout_fd_ = -1;
  stderr_fd_ = -1;
}

bool PSServerLauncher::SpawnProcess() {
  if (!std::filesystem::exists(options_.server_path)) {
    SetError("Server binary not found: " + options_.server_path.string());
    return false;
  }

  std::error_code ec;
  std::filesystem::create_directories(options_.log_dir, ec);
  if (ec) {
    SetError("Failed to create log dir: " + options_.log_dir.string());
    return false;
  }

  std::filesystem::path launch_config_path = PrepareConfigForLaunch();
  if (launch_config_path.empty()) {
    return false;
  }

  int stdout_pipe[2] = {-1, -1};
  int stderr_pipe[2] = {-1, -1};
  if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
    SetError("Failed to create process pipes");
    return false;
  }

  std::filesystem::path log_path =
      options_.log_dir / ("ps_server_cpp_" + TimestampNow() + ".log");
  result_.log_path = log_path;
  log_file_.open(log_path, std::ios::out | std::ios::app);
  if (!log_file_.is_open()) {
    SetError("Failed to open log file: " + log_path.string());
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);
    return false;
  }

  pid_ = fork();
  if (pid_ < 0) {
    SetError("fork() failed");
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);
    return false;
  }

  if (pid_ == 0) {
    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);

    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);

    std::filesystem::path cwd =
        options_.server_path.parent_path().parent_path().parent_path();
    if (!cwd.empty()) {
      chdir(cwd.c_str());
    }

    std::vector<std::string> args;
    args.push_back(options_.server_path.string());
    if (!launch_config_path.empty()) {
      args.push_back("--config_path");
      args.push_back(launch_config_path.string());
    }

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& arg : args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    execv(options_.server_path.c_str(), argv.data());
    _exit(127);
  }

  close(stdout_pipe[1]);
  close(stderr_pipe[1]);
  stdout_fd_ = stdout_pipe[0];
  stderr_fd_ = stderr_pipe[0];

  started_                    = true;
  result_.started_new_process = true;
  result_.pid                 = pid_;
  AppendLogLine("Starting PS Server: " + options_.server_path.string());
  AppendLogLine("PS Server process pid: " + std::to_string(pid_));
  if (!launch_config_path.empty()) {
    AppendLogLine("Config path: " + launch_config_path.string());
  }

  StartOutputThreads();
  return true;
}

std::filesystem::path PSServerLauncher::PrepareConfigForLaunch() {
  if (!options_.override_ps_type.has_value() &&
      options_.override_ports.empty()) {
    return options_.config_path;
  }

  if (options_.config_path.empty()) {
    SetError("override_ps_type is set but config_path is empty");
    return {};
  }

  std::ifstream in(options_.config_path);
  if (!in.good()) {
    SetError(
        "Failed to open config for override: " + options_.config_path.string());
    return {};
  }

  json config;
  try {
    in >> config;
  } catch (const std::exception& e) {
    SetError(std::string("Failed to parse config json: ") + e.what());
    return {};
  }

  if (!config.contains("cache_ps") || !config["cache_ps"].is_object()) {
    SetError("Config missing object field cache_ps");
    return {};
  }
  if (options_.override_ps_type.has_value()) {
    config["cache_ps"]["ps_type"] = *options_.override_ps_type;
  }

  if (!options_.override_ports.empty()) {
    if (!config["cache_ps"].contains("servers") ||
        !config["cache_ps"]["servers"].is_array()) {
      SetError("Config cache_ps.servers is missing or invalid");
      return {};
    }

    auto& cache_servers = config["cache_ps"]["servers"];
    if (cache_servers.size() != options_.override_ports.size()) {
      SetError("override_ports size does not match cache_ps.servers size");
      return {};
    }
    for (size_t i = 0; i < options_.override_ports.size(); ++i) {
      cache_servers[i]["host"]  = "127.0.0.1";
      cache_servers[i]["port"]  = options_.override_ports[i];
      cache_servers[i]["shard"] = static_cast<int>(i);
    }

    if (config.contains("distributed_client") &&
        config["distributed_client"].is_object() &&
        config["distributed_client"].contains("servers") &&
        config["distributed_client"]["servers"].is_array()) {
      auto& client_servers = config["distributed_client"]["servers"];
      if (client_servers.size() == options_.override_ports.size()) {
        for (size_t i = 0; i < options_.override_ports.size(); ++i) {
          client_servers[i]["host"]  = "127.0.0.1";
          client_servers[i]["port"]  = options_.override_ports[i];
          client_servers[i]["shard"] = static_cast<int>(i);
        }
      }
    }

    if (config.contains("client") && config["client"].is_object()) {
      config["client"]["host"]  = "127.0.0.1";
      config["client"]["port"]  = options_.override_ports.front();
      config["client"]["shard"] = 0;
    }
  }

  generated_config_path_ =
      options_.log_dir / ("ps_server_config_" + TimestampNow() + ".json");
  std::ofstream out(generated_config_path_, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    SetError(
        "Failed to write generated config: " + generated_config_path_.string());
    generated_config_path_.clear();
    return {};
  }
  out << std::setw(2) << config << std::endl;
  return generated_config_path_;
}

void PSServerLauncher::CleanupLaunchConfig() {
  if (generated_config_path_.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::remove(generated_config_path_, ec);
  generated_config_path_.clear();
}

bool PSServerLauncher::IsProcessAlive() const {
  if (pid_ <= 0) {
    return false;
  }
  int rc = kill(pid_, 0);
  return rc == 0 || errno == EPERM;
}

bool PSServerLauncher::WaitUntilReady() {
  if (options_.startup_delay_ms > 0) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(options_.startup_delay_ms));
  }

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::seconds(options_.startup_timeout_sec);

  std::unique_lock<std::mutex> lock(mu_);
  while (ready_shards_.size() < static_cast<size_t>(options_.num_shards)) {
    if (!IsProcessAlive()) {
      last_error_ = "ps_server exited before reaching ready state";
      return false;
    }

    if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
      std::ostringstream oss;
      oss << "Timeout waiting for ps_server ready shards: got="
          << ready_shards_.size() << ", expected=" << options_.num_shards;
      last_error_ = oss.str();
      return false;
    }
  }

  result_.ready_shards.assign(ready_shards_.begin(), ready_shards_.end());
  return true;
}

bool PSServerLauncher::Start() {
  if (started_) {
    return true;
  }

  LaunchDecision decision = EvaluateLaunchDecision(options_);
  if (decision.should_fail) {
    result_.reason = decision.reason;
    SetError(decision.reason);
    return false;
  }

  if (!decision.should_start) {
    result_.reason              = decision.reason;
    result_.started_new_process = false;
    return true;
  }

  if (!SpawnProcess()) {
    return false;
  }

  if (!WaitUntilReady()) {
    Stop();
    return false;
  }

  result_.reason = "started";
  return true;
}

void PSServerLauncher::Stop() {
  if (!started_) {
    return;
  }

  if (pid_ > 0 && result_.started_new_process) {
    kill(pid_, SIGTERM);
    auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(options_.graceful_shutdown_timeout_sec);

    while (std::chrono::steady_clock::now() < deadline) {
      int status = 0;
      pid_t rc   = waitpid(pid_, &status, WNOHANG);
      if (rc == pid_) {
        pid_ = -1;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (pid_ > 0) {
      kill(pid_, SIGKILL);
      int status = 0;
      waitpid(pid_, &status, 0);
      pid_ = -1;
    }
  }

  StopOutputThreads();
  if (log_file_.is_open()) {
    log_file_.flush();
    log_file_.close();
  }
  CleanupLaunchConfig();
  started_ = false;
}

bool PSServerLauncher::IsRunning() const {
  return started_ && IsProcessAlive();
}

ScopedPSServer::ScopedPSServer(LauncherOptions options, bool auto_start)
    : launcher_(std::move(options)) {
  if (auto_start) {
    Start();
  }
}

ScopedPSServer::~ScopedPSServer() { Stop(); }

bool ScopedPSServer::Start() {
  if (!launcher_.Start()) {
    throw std::runtime_error(
        "Failed to start ps_server: " + launcher_.GetLastError());
  }
  return true;
}

void ScopedPSServer::Stop() { launcher_.Stop(); }

} // namespace recstore::test
