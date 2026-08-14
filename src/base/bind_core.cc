#include "base/bind_core.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace base {
int global_socket_id = 0;

std::vector<std::vector<int>> parse_numa_nodes() {
  FILE* pipe = popen("lscpu", "r");
  if (!pipe) {
    LOG(FATAL) << "Failed to run lscpu command" << std::endl;
  }

  char buffer[2000];
  std::string result = "";
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }
  pclose(pipe);

  std::istringstream ss(result);
  std::string line;
  std::vector<std::string> numa_lines;
  while (std::getline(ss, line)) {
    if (line.find("NUMA node") != std::string::npos &&
        line.find("CPU(s):") != std::string::npos) {
      numa_lines.push_back(line);
    }
  }

  std::vector<std::vector<int>> core_table;

  for (size_t i = 0; i < numa_lines.size(); ++i) {
    core_table.push_back(std::vector<int>());
    std::string& numa_line = numa_lines[i];
    size_t pos             = numa_line.find("CPU(s):");
    if (pos != std::string::npos) {
      std::string cpus = numa_line.substr(pos + 7);
      std::istringstream cpu_stream(cpus);
      std::string cpu_range;
      while (std::getline(cpu_stream, cpu_range, ',')) {
        size_t dash_pos = cpu_range.find('-');
        if (dash_pos != std::string::npos) {
          int start = std::stoi(cpu_range.substr(0, dash_pos));
          int end   = std::stoi(cpu_range.substr(dash_pos + 1));
          for (int cpu = start; cpu <= end; ++cpu) {
            core_table[i].push_back(cpu);
          }
        } else {
          core_table[i].push_back(std::stoi(cpu_range));
        }
      }
    }
  }
  return core_table;
}

std::vector<std::vector<int>> core_table = parse_numa_nodes();

int ReadEnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char* end         = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' ||
      parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    LOG(WARNING) << "invalid " << name << "=" << value
                 << ", fallback=" << fallback;
    return fallback;
  }
  return static_cast<int>(parsed);
}

void bind_core_by_index(int raw_core_idx) {
  if (core_table.empty()) {
    LOG(WARNING) << "skip core binding: no NUMA core table";
    return;
  }
  if (global_socket_id < 0 ||
      global_socket_id >= static_cast<int>(core_table.size())) {
    LOG(WARNING) << "skip core binding: invalid global_socket_id="
                 << global_socket_id << " numa_nodes=" << core_table.size();
    return;
  }
  const auto& numa_cores = core_table[global_socket_id];
  if (numa_cores.empty()) {
    LOG(WARNING) << "skip core binding: empty core list for socket "
                 << global_socket_id;
    return;
  }
  cpu_set_t allowed_mask;
  CPU_ZERO(&allowed_mask);
  if (sched_getaffinity(0, sizeof(allowed_mask), &allowed_mask) != 0) {
    LOG(WARNING) << "skip core binding: sched_getaffinity failed";
    return;
  }
  std::vector<int> cores;
  cores.reserve(numa_cores.size());
  for (int cpu : numa_cores) {
    if (CPU_ISSET(cpu, &allowed_mask)) {
      cores.push_back(cpu);
    }
  }
  if (cores.empty()) {
    LOG(WARNING) << "skip core binding: no allowed CPUs for socket "
                 << global_socket_id;
    return;
  }
  const int core_idx =
      ((raw_core_idx % static_cast<int>(cores.size())) +
       static_cast<int>(cores.size())) %
      static_cast<int>(cores.size());
  LOG(WARNING) << "bind to core " << cores[core_idx] << " socket="
               << global_socket_id << " requested_core_index=" << raw_core_idx
               << " core_index=" << core_idx;
  std::cerr << "component=bind_core event=bind"
            << " socket=" << global_socket_id << " requested_core_index="
            << raw_core_idx << " core_index=" << core_idx
            << " cpu=" << cores[core_idx] << std::endl;
  bind_core(cores[core_idx]);
}

void auto_bind_core() {
  static std::atomic<int> cur_id{0};
  const int offset = ReadEnvInt("RECSTORE_BIND_CORE_OFFSET", 0);
  const int next   = cur_id.fetch_add(1);
  bind_core_by_index(offset + next);
}

void bind_core_with_env_offset(int core_idx) {
  bind_core_by_index(ReadEnvInt("RECSTORE_BIND_CORE_OFFSET", 0) + core_idx);
}
} // namespace base
