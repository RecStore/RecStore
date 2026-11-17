#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "base/json.h"
#include "ps/base/base_ps_server.h"
#include "ps/local_shm/local_shm_region.h"

class CachePS;

namespace recstore {

class LocalShmStoreRuntime {
public:
  LocalShmStoreRuntime(LocalShmRegion* region,
                       ::CachePS* cache_ps,
                       uint32_t ready_queue_id,
                       uint32_t worker_tid,
                       uint32_t ready_queue_burst_limit = 8);

  void Run();
  void Stop();

private:
  bool DrainReadyQueue(uint32_t ready_queue_id, uint32_t* processed);
  void ProcessSlot(uint32_t slot_id);

private:
  LocalShmRegion* region_;
  ::CachePS* cache_ps_;
  std::atomic<bool> stop_{false};
  uint32_t ready_queue_id_          = 0;
  uint32_t worker_tid_              = 0;
  uint32_t ready_queue_burst_limit_ = 8;
};

class LocalShmParameterServer : public BaseParameterServer {
public:
  LocalShmParameterServer() = default;
  ~LocalShmParameterServer();
  void Init(const json& config) override;
  void Run() override;
  void Stop();
  // Call after requests finish for a consistent live-allocation snapshot.
  std::string ExtraResultFields() const;

private:
  json local_config_;
  std::unique_ptr<LocalShmRegion> region_;
  std::shared_ptr<::CachePS> cache_ps_;
  std::vector<std::unique_ptr<LocalShmStoreRuntime>> runtimes_;
  std::vector<std::thread> worker_threads_;
};

} // namespace recstore
