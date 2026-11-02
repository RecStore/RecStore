#pragma once

#include "ps/base/sparse_gradient_commit.h"
#include <chrono>
#include <future>
#include <thread>

namespace recstore {

inline SparseGradientCommitHeader MakeSparseGradientCommitHeader(
    int64_t session,
    int64_t step,
    int64_t rank,
    int64_t world,
    int64_t dim,
    double lr,
    int64_t divisor,
    int64_t kind) {
  if (session <= 0 || step < 0 || rank < 0 || world <= rank || world > 65536 ||
      dim <= 0 || dim > 65536 || kind < 0 || kind > 3 ||
      (divisor != 1 && divisor != world)) {
    throw std::invalid_argument("Invalid sparse gradient commit metadata");
  }
  SparseGradientCommitHeader h;
  h.session       = session;
  h.step          = step;
  h.rank          = rank;
  h.world         = world;
  h.dim           = dim;
  h.learning_rate = lr;
  h.divisor       = divisor;
  h.kind          = kind;
  const char* sorted_stream = std::getenv("RECSTORE_SPARSE_COMMIT_SORTED_STREAM");
  if (world == 2 && sorted_stream && std::strcmp(sorted_stream, "1") == 0)
    h.version = 2;
  DecodeSparseGradientCommit(PackSparseGradientCommit(h));
  return h;
}

inline SparseGradientCommitHeader MakeCanonicalSparseGradientCommitHeader(
    int64_t session, int64_t step, int64_t rank, int64_t world,
    int64_t dim, double lr, int64_t kind) {
  auto h = MakeSparseGradientCommitHeader(session, step, rank, world, dim, lr, 1, kind);
  h.version = 3;
  DecodeSparseGradientCommit(PackSparseGradientCommit(h));
  return h;
}

inline SparseGradientCommitHeader MakeOwnerExactReplaceHeader(
    int64_t session, int64_t step, int64_t rank, int64_t world,
    int64_t dim, double lr, int64_t kind) {
  auto h = MakeSparseGradientCommitHeader(session, step, rank, world, dim, lr, 1, kind);
  h.version = 4;
  DecodeSparseGradientCommit(PackSparseGradientCommit(h));
  return h;
}

// The adapter's readiness callback owns staged buffers and source storage
// until the worker finishes. No framework tensor or device API is needed here.
class SparseGradientCommitRuntime {
public:
  using Exchange =
      std::function<int64_t(SparseGradientCommitHeader,
                            const base::RecTensor&,
                            const base::RecTensor&)>;

  int64_t Submit(Exchange exchange,
                 SparseGradientCommitHeader h,
                 base::RecTensor keys,
                 base::RecTensor grads,
                 std::function<void()> ready,
                 std::chrono::milliseconds timeout = std::chrono::seconds(60)) {
    const bool profile = SparseCommitProfileEnabled();
    const auto submitted = profile ? SparseCommitProfileNowNs() : 0;
    auto work = std::async(
        std::launch::async,
        [exchange = std::move(exchange),
         h,
         keys  = std::move(keys),
         grads = std::move(grads),
         ready = std::move(ready),
         timeout,
         profile,
         submitted]() mutable {
          const auto started = profile ? SparseCommitProfileNowNs() : 0;
          base::RecTensor empty_keys({0}, base::DataType::UINT64);
          base::RecTensor empty_grads(
              {0, int64_t(h.dim)}, base::DataType::FLOAT32);
          try {
            if (ready)
              ready();
            const auto ready_done = profile ? SparseCommitProfileNowNs() : 0;
            auto committed      = exchange(h, keys, grads);
            const auto exchanged = profile ? SparseCommitProfileNowNs() : 0;
            uint64_t polls = 0;
            h.kind              = 1;
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (committed < h.step) {
              if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("Sparse gradient commit timed out");
              std::this_thread::sleep_for(std::chrono::microseconds(100));
              committed = exchange(h, empty_keys, empty_grads);
              if (profile)
                ++polls;
            }
            if (profile) {
              const auto completed = SparseCommitProfileNowNs();
              std::fprintf(stderr,
                  "component=sparse_commit event=runtime_phase session=%llu "
                  "step=%lld rank=%u world=%u worker_start_ms=%.6f "
                  "ready_ms=%.6f first_exchange_ms=%.6f poll_count=%llu "
                  "poll_ms=%.6f total_ms=%.6f\n",
                  static_cast<unsigned long long>(h.session),
                  static_cast<long long>(h.step), h.rank, h.world,
                  (started - submitted) / 1e6, (ready_done - started) / 1e6,
                  (exchanged - ready_done) / 1e6,
                  static_cast<unsigned long long>(polls),
                  (completed - exchanged) / 1e6, (completed - submitted) / 1e6);
            }
          } catch (...) {
            // A failed optimizer may have applied part of an update. Abort
            // peers and preserve the original error; never resubmit gradients.
            try {
              h.kind = 3;
              exchange(h, empty_keys, empty_grads);
            } catch (...) {
            }
            throw;
          }
        });
    std::lock_guard<std::mutex> guard(mu_);
    const auto id = next_++;
    pending_.emplace(id, std::move(work));
    return id;
  }

  void Wait(int64_t id) {
    std::future<void> work;
    {
      std::lock_guard<std::mutex> guard(mu_);
      auto it = pending_.find(id);
      if (it == pending_.end())
        throw std::runtime_error("Unknown sparse gradient commit handle");
      work = std::move(it->second);
      pending_.erase(it);
    }
    work.get();
  }

private:
  std::mutex mu_;
  int64_t next_ = 1;
  std::unordered_map<int64_t, std::future<void>> pending_;
};

} // namespace recstore
