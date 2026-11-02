#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

namespace recstore::framework {

enum LookupProfileIndex : std::size_t {
  kLookupTotalMs = 0,
  kLookupKeysStageMs,
  kLookupSubmitMs,
  kLookupWaitMs,
  kLookupPayloadPinMs,
  kLookupFallbackCopyMs,
  kLookupValuesH2DEnqueueMs,
  kLookupProfileSize,
};

enum UpdateProfileIndex : std::size_t {
  kUpdateTotalMs = 0,
  kUpdateKeysStageMs,
  kUpdateGradsStageMs,
  kUpdateShmCallMs,
  kUpdateStageWaitMs,
  kUpdateSlotAcquireMs,
  kUpdateCopyEnqueueMs,
  kUpdateWaitMs,
  kUpdateServerQueueWaitMs,
  kUpdateServerBackendMs,
  kUpdateProfileSize,
};

inline thread_local std::vector<double>
    g_last_local_lookup_flat_profile(kLookupProfileSize, 0.0);
inline thread_local std::vector<double>
    g_last_local_update_flat_profile(kUpdateProfileSize, 0.0);

inline std::chrono::steady_clock::time_point SteadyNow() {
  return std::chrono::steady_clock::now();
}

inline double ElapsedMs(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
             SteadyNow() - start)
      .count();
}

inline void ResetLocalLookupFlatProfile() {
  std::fill(g_last_local_lookup_flat_profile.begin(),
            g_last_local_lookup_flat_profile.end(),
            0.0);
}

inline void ResetLocalUpdateFlatProfile() {
  std::fill(g_last_local_update_flat_profile.begin(),
            g_last_local_update_flat_profile.end(),
            0.0);
}

} // namespace recstore::framework
