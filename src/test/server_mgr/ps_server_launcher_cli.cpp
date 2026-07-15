#include "ps_server_launcher.h"

#include "base/json.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace recstore::test {
namespace {

std::atomic<bool> g_stop_requested{false};

void HandleStopSignal(int) { g_stop_requested = true; }

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " decision [--config PATH]\n"
      << "  " << argv0
      << " serve [--server-path PATH] [--config PATH] [--log-dir PATH]\n"
      << "        [--timeout SEC] [--num-shards N] [--startup-delay-ms MS]"
      << " [--verbose]\n";
}

LauncherOptions OptionsFromArgs(int argc, char** argv, int start_index) {
  LauncherOptions options = PSServerLauncher::LoadOptionsFromEnvironment();

  for (int i = start_index; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + flag);
      }
      return argv[++i];
    };

    if (arg == "--server-path") {
      options.server_path = require_value("--server-path");
    } else if (arg == "--config") {
      options.config_path = require_value("--config");
    } else if (arg == "--log-dir") {
      options.log_dir = require_value("--log-dir");
    } else if (arg == "--timeout") {
      options.startup_timeout_sec = std::stoi(require_value("--timeout"));
    } else if (arg == "--num-shards") {
      options.num_shards = std::stoi(require_value("--num-shards"));
    } else if (arg == "--startup-delay-ms") {
      options.startup_delay_ms = std::stoi(require_value("--startup-delay-ms"));
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  return options;
}

int RunDecision(const LauncherOptions& options) {
  const LaunchDecision decision =
      PSServerLauncher::EvaluateLaunchDecision(options);

  json out = {
      {"should_start", decision.should_start},
      {"should_fail", decision.should_fail},
      {"reason", decision.reason},
      {"configured_ports", decision.configured_ports},
      {"open_ports", decision.open_ports},
  };
  std::cout << out.dump() << std::endl;
  return decision.should_fail ? 2 : 0;
}

int RunServe(const LauncherOptions& options) {
  LaunchDecision decision = PSServerLauncher::EvaluateLaunchDecision(options);
  if (decision.should_fail) {
    std::cerr << decision.reason << std::endl;
    return 2;
  }
  if (!decision.should_start) {
    std::cout << "SKIP\t" << decision.reason << std::endl;
    return 0;
  }

  std::signal(SIGTERM, HandleStopSignal);
  std::signal(SIGINT, HandleStopSignal);

  PSServerLauncher launcher(options);
  if (!launcher.Start()) {
    std::cerr << launcher.GetLastError() << std::endl;
    return 1;
  }

  std::cout << "READY" << std::endl;
  std::cout.flush();

  while (!g_stop_requested && launcher.IsRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  launcher.Stop();
  return 0;
}

} // namespace
} // namespace recstore::test

int main(int argc, char** argv) {
  using recstore::test::LauncherOptions;
  using recstore::test::OptionsFromArgs;
  using recstore::test::PrintUsage;
  using recstore::test::RunDecision;
  using recstore::test::RunServe;

  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string command = argv[1];
  try {
    const LauncherOptions options = OptionsFromArgs(argc, argv, 2);
    if (command == "decision") {
      return RunDecision(options);
    }
    if (command == "serve") {
      return RunServe(options);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  PrintUsage(argv[0]);
  return 1;
}
