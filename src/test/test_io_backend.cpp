#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <mutex>
#include <boost/coroutine2/all.hpp>
#include <gtest/gtest.h>
#include "base/factory.h"
#include "storage/io_backend/force_link.h"
#include "storage/io_backend/io_backend.h"
#include "test_io_uring_helper.h"

using boost::coroutines2::coroutine;

namespace {
constexpr const char* kSpdkPcieAddress = "0000:c2:00.0";

bool HasFreeHugepages() {
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open())
    return false;

  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.rfind("HugePages_Free:", 0) != 0)
      continue;
    std::istringstream iss(line.substr(std::strlen("HugePages_Free:")));
    int free_pages = 0;
    if (iss >> free_pages)
      return free_pages > 0;
    return false;
  }
  return false;
}

bool HasConfiguredPcieDevice() {
  const std::filesystem::path pci_path =
      std::filesystem::path("/sys/bus/pci/devices") / kSpdkPcieAddress;
  return std::filesystem::exists(pci_path);
}

bool CanUseSpdkBackend(std::string* reason) {
  if (!HasConfiguredPcieDevice()) {
    if (reason != nullptr)
      *reason = "configured SPDK PCIe device is missing";
    return false;
  }
  if (!HasFreeHugepages()) {
    if (reason != nullptr)
      *reason = "no free hugepages detected";
    return false;
  }
  return true;
}

bool HasRegisteredBackend(const std::string& backend, std::string* reason) {
  using IOF            = base::Factory<IOBackend, const BaseKVConfig&>;
  const auto& creators = IOF::creators();
  if (creators.find(backend) != creators.end())
    return true;
  if (reason != nullptr)
    *reason = "backend factory is not registered for " + backend;
  return false;
}

std::string ResolveBackendFromEnv() {
  const char* raw_backend = std::getenv("RECSTORE_IO_BACKEND");
  if (raw_backend == nullptr || std::string(raw_backend).empty())
    return "IOURING";

  std::string backend(raw_backend);
  std::transform(
      backend.begin(), backend.end(), backend.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
      });
  if (backend != "IOURING" && backend != "SPDK")
    return "IOURING";
  return backend;
}

BaseKVConfig
MakeConfig(const std::string& backend, const std::string& file_path) {
  BaseKVConfig config;
  config.num_threads_ = 1;
  config.json_config_ = {
      {"io_backend_type", backend},
      {"queue_cnt", 64},
      {"page_id_offset", 1},
      {"file_path", file_path},
  };
  return config;
}

class IOBackendTest : public ::testing::Test {
protected:
  void SetUp() override {
    static std::once_flag force_link_once;
    std::call_once(force_link_once, []() { ForceLinkIOBackends(); });

    const char* raw_backend = std::getenv("RECSTORE_IO_BACKEND");
    const bool backend_forced =
        (raw_backend != nullptr && std::string(raw_backend).size() > 0);

    backend_ = ResolveBackendFromEnv();
    if (backend_ == "SPDK") {
      std::string spdk_unavailable_reason;
      if (!CanUseSpdkBackend(&spdk_unavailable_reason)) {
        if (backend_forced) {
          GTEST_SKIP() << "SPDK requested by RECSTORE_IO_BACKEND, but "
                       << spdk_unavailable_reason;
        }
        backend_ = "IOURING";
      }
    }

    if (backend_ == "IOURING") {
      std::string io_uring_unavailable_reason;
      if (!test_utils::CanUseIoUring(&io_uring_unavailable_reason))
        GTEST_SKIP() << io_uring_unavailable_reason;
    }

    std::string backend_registration_reason;
    if (!HasRegisteredBackend(backend_, &backend_registration_reason))
      GTEST_SKIP() << backend_registration_reason;

    const auto ts =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    file_path_ = "/tmp/test_io_backend_" + std::to_string(ts) + ".db";

    BaseKVConfig config = MakeConfig(backend_, file_path_);
    using IOF           = base::Factory<IOBackend, const BaseKVConfig&>;
    backend_impl_.reset(IOF::NewInstance(backend_, config));
    ASSERT_NE(backend_impl_, nullptr);
    backend_impl_->init();
  }

  void TearDown() override {
    backend_impl_.reset();
    if (!file_path_.empty())
      std::filesystem::remove(file_path_);
  }

  std::string backend_;
  std::string file_path_;
  std::unique_ptr<IOBackend> backend_impl_;
};

TEST_F(IOBackendTest, SyncReadWriteRoundTrip) {
  PageID_t page_id = backend_impl_->AllocatePage();

  auto* write_page = static_cast<char*>(backend_impl_->GetPage(page_id));
  ASSERT_NE(write_page, nullptr);
  for (size_t i = 0; i < PAGE_SIZE; ++i) {
    write_page[i] = static_cast<char>((i * 7) % 251);
  }
  backend_impl_->Unpin(page_id, write_page, true);

  auto* read_page = static_cast<char*>(backend_impl_->GetPage(page_id));
  ASSERT_NE(read_page, nullptr);
  for (size_t i = 0; i < PAGE_SIZE; ++i) {
    ASSERT_EQ(read_page[i], static_cast<char>((i * 7) % 251))
        << "mismatch at offset " << i;
  }
  backend_impl_->Unpin(page_id, read_page, false);
}

TEST_F(IOBackendTest, BatchReadWriteRoundTrip) {
  for (int spins : {0, 8192}) {
    backend_impl_.reset();
    auto config = MakeConfig(backend_, file_path_);
    config.json_config_["completion_poll_spins"] = spins;
    using IOF = base::Factory<IOBackend, const BaseKVConfig&>;
    backend_impl_.reset(IOF::NewInstance(backend_, config));
    backend_impl_->init();
    // queue_cnt=64 permits 32 in flight: exercise three full windows and a
    // tail.
    constexpr int kNumPages = 97;

    std::vector<IOBackend::IOEntry> write_entries;
    std::vector<char*> write_buffers;
    write_entries.reserve(kNumPages);
    write_buffers.reserve(kNumPages);

    for (int i = 0; i < kNumPages; ++i) {
      PageID_t page_id = backend_impl_->AllocatePage();
      char* buf        = backend_impl_->AllocateBuffer(1);
      ASSERT_NE(buf, nullptr);
      std::memset(buf, i + 11, PAGE_SIZE);
      write_entries.push_back({page_id, buf, 1});
      write_buffers.push_back(buf);
    }
    backend_impl_->BatchWritePages(write_entries);

    std::vector<IOBackend::IOEntry> read_entries;
    std::vector<char*> read_buffers;
    read_entries.reserve(kNumPages);
    read_buffers.reserve(kNumPages);
    for (int i = 0; i < kNumPages; ++i) {
      char* buf = backend_impl_->AllocateBuffer(1);
      ASSERT_NE(buf, nullptr);
      read_entries.push_back({write_entries[i].page_id, buf, 1});
      read_buffers.push_back(buf);
    }
    backend_impl_->BatchReadPages(read_entries);

    for (int i = 0; i < kNumPages; ++i) {
      for (size_t off = 0; off < PAGE_SIZE; ++off) {
        ASSERT_EQ(read_buffers[i][off], static_cast<char>(i + 11))
            << "page=" << i << ", offset=" << off;
      }
    }

    for (char* buf : write_buffers)
      backend_impl_->FreeBuffer(buf);
    for (char* buf : read_buffers)
      backend_impl_->FreeBuffer(buf);
  }
}

TEST_F(IOBackendTest, BatchReadWorkDrainsExceptionsAndReusesRing) {
  for (const std::string type : {"IOURING"}) {
    backend_impl_.reset();
    auto config                      = MakeConfig(type, file_path_);
    config.json_config_["queue_cnt"] = 4;
    backend_impl_.reset(
        base::Factory<IOBackend, const BaseKVConfig&>::NewInstance(
            type, config));
    backend_impl_->init();
    std::vector<IOBackend::IOEntry> writes, reads;
    for (size_t i = 0; i < 7; ++i) {
      char* source = backend_impl_->AllocateBuffer(3);
      char* target = backend_impl_->AllocateBuffer(3);
      std::memset(source, 20 + i, 3 * PAGE_SIZE);
      // Cross a 2MiB shard extent, with more entries than the two-entry window.
      writes.push_back({510 + i * 1024, source, 3});
      reads.push_back({510 + i * 1024, target, 3});
    }
    backend_impl_->BatchWritePages(writes);
    int calls = 0;
    backend_impl_->BatchReadPagesWithWork(reads, [&] { ++calls; });
    EXPECT_EQ(calls, 1);
    for (size_t i = 0; i < reads.size(); ++i)
      for (size_t j = 0; j < 3 * PAGE_SIZE; ++j)
        ASSERT_EQ(reads[i].buffer[j], static_cast<char>(20 + i));
    EXPECT_THROW(backend_impl_->BatchReadPagesWithWork(
                     reads,
                     [&] {
                       ++calls;
                       throw std::runtime_error("CPU copy failure");
                     }),
                 std::runtime_error);
    EXPECT_EQ(calls, 2);
    for (auto& entry : reads)
      std::memset(entry.buffer, 0, 3 * PAGE_SIZE);
    backend_impl_->BatchReadPages(reads);
    for (size_t i = 0; i < reads.size(); ++i)
      for (size_t j = 0; j < 3 * PAGE_SIZE; ++j)
        ASSERT_EQ(reads[i].buffer[j], static_cast<char>(20 + i));
    backend_impl_->BatchReadPagesWithWork({}, [&] { ++calls; });
    EXPECT_EQ(calls, 3);
    EXPECT_THROW(
        backend_impl_->BatchReadPagesWithWork(
            {}, [] { throw std::runtime_error("empty copy failure"); }),
        std::runtime_error);
    for (auto& entry : writes)
      backend_impl_->FreeBuffer(entry.buffer);
    for (auto& entry : reads)
      backend_impl_->FreeBuffer(entry.buffer);
  }
}

} // namespace
