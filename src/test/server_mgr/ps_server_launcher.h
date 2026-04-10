#pragma once

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>

namespace recstore::test {

struct LauncherOptions {
  std::filesystem::path server_path;
  std::filesystem::path config_path;
  std::optional<std::string> override_ps_type;
  std::vector<int> override_ports;
  std::filesystem::path log_dir     = "./logs";
  int startup_timeout_sec           = 60;
  int graceful_shutdown_timeout_sec = 5;
  int startup_delay_ms              = 2000;
  int num_shards                    = 2;
  bool verbose                      = false;
};

struct LaunchDecision {
  bool should_start = true;
  bool should_fail  = false;
  std::string reason;
  std::vector<int> configured_ports;
  std::vector<int> open_ports;
};

struct LaunchResult {
  bool started_new_process = false;
  std::optional<pid_t> pid;
  std::vector<int> ready_shards;
  std::filesystem::path log_path;
  std::string reason;
};

class PSServerLauncher {
public:
  explicit PSServerLauncher(LauncherOptions options = {});
  ~PSServerLauncher();

  PSServerLauncher(const PSServerLauncher&)            = delete;
  PSServerLauncher& operator=(const PSServerLauncher&) = delete;

  static LauncherOptions LoadOptionsFromEnvironment();
  static LaunchDecision EvaluateLaunchDecision(const LauncherOptions& options);
  static std::vector<int>
  ExtractPortsFromConfig(const std::filesystem::path& config_path);
  static std::optional<int> ParseReadyShard(const std::string& line);

  bool Start();
  void Stop();
  bool IsRunning() const;

  const LaunchResult& GetResult() const { return result_; }
  const std::string& GetLastError() const { return last_error_; }

private:
  static std::filesystem::path FindPsServerBinary();
  static std::optional<std::filesystem::path> FindConfigFile();
  static std::vector<int> CheckOpenPorts(const std::vector<int>& ports);
  static bool
  IsPortOpen(const std::string& host, int port, int timeout_sec = 1);
  static std::vector<int> DefaultPorts();

  bool SpawnProcess();
  std::filesystem::path PrepareConfigForLaunch();
  void CleanupLaunchConfig();
  bool WaitUntilReady();
  bool IsProcessAlive() const;
  void StartOutputThreads();
  void StopOutputThreads();
  void OutputThreadLoop(int fd, const std::string& stream_name);
  void AppendLogLine(const std::string& line);
  void SetError(const std::string& error);

private:
  LauncherOptions options_;
  LaunchResult result_;
  std::string last_error_;

  pid_t pid_     = -1;
  int stdout_fd_ = -1;
  int stderr_fd_ = -1;
  bool started_  = false;

  std::mutex mu_;
  std::condition_variable cv_;
  std::set<int> ready_shards_;
  std::vector<std::string> recent_logs_;
  std::ofstream log_file_;
  std::atomic<bool> stop_output_threads_{false};
  std::thread stdout_thread_;
  std::thread stderr_thread_;
  std::filesystem::path generated_config_path_;
};

class ScopedPSServer {
public:
  explicit ScopedPSServer(LauncherOptions options = {}, bool auto_start = true);
  ~ScopedPSServer();

  ScopedPSServer(const ScopedPSServer&)            = delete;
  ScopedPSServer& operator=(const ScopedPSServer&) = delete;

  bool Start();
  void Stop();
  PSServerLauncher& Launcher() { return launcher_; }

private:
  PSServerLauncher launcher_;
};

} // namespace recstore::test
