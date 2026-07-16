#pragma once
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cpptrace/cpptrace.hpp>
#include <iostream>
#include <mutex>
#include <thread>

#include "base/log.h"

namespace base {

// constexpr bool kDetectDeadLock = true;
constexpr bool kDetectDeadLock = false;

class Atomic {
public:
  static bool CAS(int* ptr, int old_val, int new_val) {
    return __sync_bool_compare_and_swap(ptr, old_val, new_val);
  }

  static bool CAS(void** ptr, void* old_val, void* new_val) {
    return __sync_bool_compare_and_swap(ptr, old_val, new_val);
  }

  template <typename T>
  static T load(const volatile T* obj) {
    return __atomic_load_n(obj, __ATOMIC_SEQ_CST);
  }

  template <typename T>
  static void store(volatile T* obj, T desired) {
    return __atomic_store_n(obj, desired, __ATOMIC_SEQ_CST);
  }
};

class SpinLock {
  std::atomic_flag locked = ATOMIC_FLAG_INIT;

public:
  void Lock() {
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    if (kDetectDeadLock)
      start_time = std::chrono::steady_clock::now();

    while (locked.test_and_set(std::memory_order_acquire)) {
      ;
      if (kDetectDeadLock) {
        auto end_time = std::chrono::steady_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::seconds>(
                end_time - start_time)
                .count();
        // if (int(duration) % 5 == 1) {
        //   cpptrace::generate_trace().print();
        // }
        if (duration > 2)
          LOG(FATAL) << "may be deadlocked";
      }
    }
  }
  void Unlock() { locked.clear(std::memory_order_release); }

  void AssertLockHold() {
    assert(locked.test_and_set(std::memory_order_acquire));
    CHECK(locked.test_and_set(std::memory_order_acquire));
  }
};

class NamedSpinLock {
  std::atomic_flag locked = ATOMIC_FLAG_INIT;

  std::string locked_success_info_;

public:
  void Lock(std::string locked_success_info) {
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    if (kDetectDeadLock)
      start_time = std::chrono::steady_clock::now();

    while (locked.test_and_set(std::memory_order_acquire)) {
      ;
      if (kDetectDeadLock) {
        auto end_time = std::chrono::steady_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::seconds>(
                end_time - start_time)
                .count();
        // if (int(duration) % 5 == 1) {
        //   cpptrace::generate_trace().print();
        // }
        if (duration > 5)
          LOG(FATAL) << "may be deadlocked, " << locked_success_info_;
      }
    }
    locked_success_info_ = locked_success_info;
  }
  void Unlock() {
    locked.clear(std::memory_order_release);
    locked_success_info_ = "";
  }

  void AssertLockHold() {
    assert(locked.test_and_set(std::memory_order_acquire));
  }
};

class PlaceboLock {
public:
  void Lock() { ; }
  void Unlock() { ; }
};

template <class T>
class LockGuard {
  T& lock_;

public:
  LockGuard(T& lock) : lock_(lock) { lock_.Lock(); }
  ~LockGuard() { lock_.Unlock(); }
};

template <class T>
class NamedLockGuard {
  T& lock_;

public:
  NamedLockGuard(T& lock, const std::string& info) : lock_(lock) {
    lock_.Lock(info);
  }
  ~NamedLockGuard() { lock_.Unlock(); }
};

class Barrier {
public:
  explicit Barrier(int count) : count_(count), bar_(0) {}

  void Wait() {
    int passed_old = passed_.load(std::memory_order_relaxed);

    if (bar_.fetch_add(1) == (count_ - 1)) {
      // The last thread, faced barrier.
      bar_ = 0;
      // Synchronize and store in one operation.
      passed_.store(passed_old + 1, std::memory_order_release);
    } else {
      // Not the last thread. Wait others.
      while (passed_.load(std::memory_order_relaxed) == passed_old) {
      };
      // Need to synchronize cache with other threads, passed barrier.
      std::atomic_thread_fence(std::memory_order_acquire);
    }
  }

private:
  int count_;
  std::atomic_int bar_;
  std::atomic_int passed_ = 0;
};

// class BitLockTable {
//  public:
//   BitLockTable(int64_t num_locks) : lockTable((num_locks + 63) / 64) {
//     for (int i = 0; i < lockTable.size(); i++) {
//       lockTable[i].store(0, std::memory_order_relaxed);
//     }
//   }

//   void lock(int index) {
//     uint64_t bitIndex = index % 64;
//     uint64_t arrayIndex = index / 64;
//     uint64_t bit = 1 << bitIndex;
//     uint64_t expected =
//     lockTable[arrayIndex].load(std::memory_order_relaxed);

//     while (true) {
//       if ((expected & bit) == 0) {  // Check if the bit is not set
//         // Attempt to set the bit
//         if (lockTable[arrayIndex].compare_exchange_weak(
//                 expected, expected | bit, std::memory_order_acquire,
//                 std::memory_order_relaxed)) {
//           break;  // Successfully acquired the lock
//         }
//       } else {
//         // The bit is already set, wait and then read the current value again
//         expected = lockTable[arrayIndex].load(std::memory_order_relaxed);
//       }
//     }
//   }

//   void unlock(int index) {
//     uint64_t bitIndex = index % 64;
//     uint64_t arrayIndex = index / 64;
//     uint64_t bit = 1 << bitIndex;
//     // Clear the bit to unlock
//     lockTable[arrayIndex].fetch_and(~bit, std::memory_order_release);
//   }

//  private:
//   std::vector<std::atomic<uint64_t>> lockTable;
// };

class BitLockTable {
public:
  BitLockTable(int64_t num_locks) : lockTable(num_locks) {}

  void lock(int index) { lockTable[index].Lock(); }

  void unlock(int index) { lockTable[index].Unlock(); }

private:
  std::vector<base::SpinLock> lockTable;
};

// class ReaderFriendlyLock {
//   std::vector<uint64_t[8]> lock_vec_;

//  public:
//   DELETE_COPY_CONSTRUCTOR_AND_ASSIGNMENT(ReaderFriendlyLock);

//   ReaderFriendlyLock(ReaderFriendlyLock&& rhs) noexcept {
//     *this = std::move(rhs);
//   }
//   ReaderFriendlyLock& operator=(ReaderFriendlyLock&& rhs) {
//     std::swap(this->lock_vec_, rhs.lock_vec_);
//     return *this;
//   }

//   ReaderFriendlyLock() : lock_vec_(util::Schedule::max_nr_threads()) {
//     for (int i = 0; i < util::Schedule::max_nr_threads(); ++i) {
//       lock_vec_[i][0] = 0;
//       lock_vec_[i][1] = 0;
//     }
//   }

//   bool lock() {
//     for (int i = 0; i < util::Schedule::max_nr_threads(); ++i) {
//       while (!CAS(&lock_vec_[i][0], 0, 1)) {
//       }
//     }
//     return true;
//   }

//   bool try_lock() {
//     for (int i = 0; i < util::Schedule::max_nr_threads(); ++i) {
//       if (!CAS(&lock_vec_[i][0], 0, 1)) {
//         for (i--; i >= 0; i--) {
//           compiler_barrier();
//           lock_vec_[i][0] = 0;
//         }
//         return false;
//       }
//     }
//     return true;
//   }

//   bool try_lock_shared() {
//     if (lock_vec_[util::Schedule::thread_id()][1]) {
//       pr_once(info, "recursive lock!");
//       return true;
//     }
//     return CAS(&lock_vec_[util::Schedule::thread_id()][0], 0, 1);
//   }

//   bool lock_shared() {
//     if (lock_vec_[util::Schedule::thread_id()][1]) {
//       pr_once(info, "recursive lock!");
//       return true;
//     }
//     while (!CAS(&lock_vec_[util::Schedule::thread_id()][0], 0, 1)) {
//     }
//     lock_vec_[util::Schedule::thread_id()][1] = 1;
//     return true;
//   }

//   void unlock() {
//     compiler_barrier();
//     for (int i = 0; i < util::Schedule::max_nr_threads(); ++i) {
//       lock_vec_[i][0] = 0;
//     }
//   }

//   void unlock_shared() {
//     compiler_barrier();
//     lock_vec_[util::Schedule::thread_id()][0] = 0;
//     lock_vec_[util::Schedule::thread_id()][1] = 0;
//   }
// };

} // namespace base