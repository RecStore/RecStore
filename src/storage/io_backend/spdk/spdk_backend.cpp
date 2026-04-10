#include "../io_backend.h"
#include <algorithm>
#include <atomic>
#include <fcntl.h>
#include "spdk/env.h"
#include "spdk/nvme.h"
#include "spdk/env.h"
#include "spdk/nvme.h"
#include "base/factory.h"
#include "storage/kv_engine/base_kv.h"

static const char* pcie_address = "0000:c2:00.0";

class SpdkBackend : public IOBackend {
private:
  inline static std::atomic<bool> controller_active_{false};
  inline static std::mutex env_mutex_;
  inline static bool env_initialized_ = false;
  struct ThreadQpair;
  inline static std::mutex thread_qpair_mutex_;
  inline static std::vector<ThreadQpair*> active_thread_qpairs_;
  spdk_env_opts opts;
  struct spdk_nvme_transport_id trid = {};
  struct ns_entry {
    struct spdk_nvme_ctrlr* ctrlr;
    struct spdk_nvme_ns* ns;
  } ns_entry;
  inline static struct ns_entry shared_ns_entry_ = {};
  inline static char* shared_empty_page_         = nullptr;
  inline static std::atomic<int> instance_count_{0};

  struct ThreadQpair {
    struct spdk_nvme_qpair* qpair = NULL;
    bool initialized              = false;
    bool registered               = false;
    ThreadQpair()                 = default;
    ~ThreadQpair() {
      if (registered)
        SpdkBackend::unregister_thread_qpair(this);
      if (SpdkBackend::controller_active_.load(std::memory_order_acquire))
        release();
    }

    static ThreadQpair& instance() {
      static thread_local ThreadQpair thread_qpair;
      return thread_qpair;
    }

    struct spdk_nvme_qpair* get(struct spdk_nvme_ctrlr* ctrlr, int queue_size) {
      if (!initialized) {
        CHECK_NE(ctrlr, nullptr) << "No NVMe controller";
        struct spdk_nvme_io_qpair_opts opts;
        spdk_nvme_ctrlr_get_default_io_qpair_opts(ctrlr, &opts, sizeof(opts));
        opts.io_queue_size     = queue_size + 1; // 你想要的队列深度
        opts.io_queue_requests = queue_size + 1; // 一般和队列深度一致
        qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, &opts, sizeof(opts));
        CHECK_NE(qpair, nullptr) << "Failed to allocate IO qpair";
        initialized = true;
        SpdkBackend::register_thread_qpair(this);
      }
      return qpair;
    }

    void release() {
      if (initialized && qpair != nullptr) {
        spdk_nvme_ctrlr_free_io_qpair(qpair);
        qpair       = nullptr;
        initialized = false;
      }
    }
  };

  struct spdk_nvme_qpair* get_thread_qpair() {
    ThreadQpair& thread_qpair = ThreadQpair::instance();
    return thread_qpair.get(ns_entry.ctrlr, queue_cnt);
  }

  static void register_thread_qpair(ThreadQpair* thread_qpair) {
    std::lock_guard<std::mutex> lock(thread_qpair_mutex_);
    if (!thread_qpair->registered) {
      active_thread_qpairs_.push_back(thread_qpair);
      thread_qpair->registered = true;
    }
  }

  static void unregister_thread_qpair(ThreadQpair* thread_qpair) {
    std::lock_guard<std::mutex> lock(thread_qpair_mutex_);
    auto it = std::find(active_thread_qpairs_.begin(),
                        active_thread_qpairs_.end(),
                        thread_qpair);
    if (it != active_thread_qpairs_.end())
      active_thread_qpairs_.erase(it);
    thread_qpair->registered = false;
  }

  static void release_all_thread_qpairs() {
    std::vector<ThreadQpair*> to_release;
    {
      std::lock_guard<std::mutex> lock(thread_qpair_mutex_);
      to_release.swap(active_thread_qpairs_);
      for (ThreadQpair* thread_qpair : to_release)
        thread_qpair->registered = false;
    }
    for (ThreadQpair* thread_qpair : to_release)
      thread_qpair->release();
  }

  static bool probe_cb(void* cb_ctx,
                       const struct spdk_nvme_transport_id* trid,
                       struct spdk_nvme_ctrlr_opts* opts) {
    if (strcmp(pcie_address, trid->traddr) != 0)
      return false;
    LOG(INFO) << "Attaching to " << trid->traddr;
    return true;
  }

  static void attach_cb(void* cb_ctx,
                        const struct spdk_nvme_transport_id* trid,
                        struct spdk_nvme_ctrlr* ctrlr,
                        const struct spdk_nvme_ctrlr_opts* opts) {
    SpdkBackend* ptr = (SpdkBackend*)(cb_ctx);
    LOG(INFO) << fmt::format("Attached to {}", trid->traddr);
    for (uint32_t i = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); i != 0;
         i          = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, i)) {
      struct spdk_nvme_ns* ns = spdk_nvme_ctrlr_get_ns(ctrlr, i);
      if (ns && spdk_nvme_ns_is_active(ns)) {
        ptr->ns_entry.ctrlr = ctrlr;
        ptr->ns_entry.ns    = ns;
        LOG(INFO) << fmt::format(
            "Using NS {} size {}\n",
            i,
            (unsigned long)spdk_nvme_ns_get_size(ptr->ns_entry.ns));
        break;
      }
    }
  }

  static void io_complete(void* arg, const struct spdk_nvme_cpl* cpl) {
    int64_t t = (int64_t)(intptr_t)arg;
    if (!spdk_nvme_cpl_is_success(cpl))
      fprintf(stderr, "I/O error!\n");
    pending--;
    if (t >= 0)
      (*coros[t])();
  }

  static void batch_write_complete(void* arg, const struct spdk_nvme_cpl* cpl) {
    int* inflight = reinterpret_cast<int*>(arg);
    if (!spdk_nvme_cpl_is_success(cpl))
      LOG(ERROR) << "BatchWritePages: I/O error!";
    (*inflight)--;
  }

public:
  SpdkBackend(const BaseKVConfig& config) : IOBackend(config){};
  ~SpdkBackend() {
    if (controller_active_.load(std::memory_order_acquire)) {
      int remaining =
          instance_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
      if (remaining == 0) {
        release_all_thread_qpairs();
        controller_active_.store(false, std::memory_order_release);
      }
    }
  }
  void init() override {
    LOG(INFO) << "init spdk backend";
    std::lock_guard<std::mutex> lock(env_mutex_);
    if (!env_initialized_) {
      LOG(INFO) << "Initializing NVMe Controllers";
      opts           = {};
      opts.opts_size = sizeof(spdk_env_opts);
      spdk_env_opts_init(&opts);
      trid = {};
      spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_PCIE);
      snprintf(trid.subnqn, sizeof(trid.subnqn), "%s", SPDK_NVMF_DISCOVERY_NQN);
      LOG(INFO) << "pcie_address: " << pcie_address;
      LOG(INFO) << "sizeof(trid.traddr): " << sizeof(trid.traddr);
      snprintf(trid.traddr, sizeof(trid.traddr), "%s", pcie_address);
      LOG(INFO) << "opts.iova_mode: " << opts.iova_mode;
      LOG(INFO) << "trid.traddr: " << trid.traddr;
      LOG(INFO) << "trid.subnqn: " << trid.subnqn;
      LOG(INFO) << "sizeof(spdk_env_opts): " << sizeof(spdk_env_opts);

      CHECK(spdk_env_init(&opts) >= 0) << "Unable to initialize SPDK env\n";
      CHECK_EQ(spdk_nvme_probe(&trid, this, probe_cb, attach_cb, NULL), 0)
          << "Failed to probe NVMe device";
      CHECK_NE(ns_entry.ns, nullptr) << "Namespace is not initialized";
      CHECK_NE(ns_entry.ctrlr, nullptr) << "Controller is not initialized";
      shared_ns_entry_   = ns_entry;
      shared_empty_page_ = (char*)spdk_zmalloc(
          PAGE_SIZE, 64, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
      CHECK_NE(shared_empty_page_, nullptr) << "Failed to allocate empty page";
      empty_page = shared_empty_page_;
      controller_active_.store(true, std::memory_order_release);
      env_initialized_ = true;
      LOG(INFO) << "env initialized successfully" << std::endl;
    } else {
      ns_entry   = shared_ns_entry_;
      empty_page = shared_empty_page_;
      CHECK_NE(empty_page, nullptr) << "Shared empty page is not initialized";
      controller_active_.store(true, std::memory_order_release);
      LOG(INFO) << "env initialized successfully" << std::endl;
    }
    instance_count_.fetch_add(1, std::memory_order_acq_rel);
  }

  void* GetPage(coroutine<void>::push_type& sink,
                uint64_t index,
                PageID_t page_id) override {
    char* buffer = (char*)spdk_zmalloc(
        PAGE_SIZE, 64, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    CHECK_NE(buffer, nullptr) << "Failed to allocate memory for page";
    ReadPage(sink, index, page_id, buffer);
    return reinterpret_cast<void*>(buffer);
  }
  void* GetPage(PageID_t page_id) override {
    char* buffer = (char*)spdk_zmalloc(
        PAGE_SIZE, 64, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    CHECK_NE(buffer, nullptr) << "Failed to allocate memory for page";
    ReadPage(page_id, buffer);
    return reinterpret_cast<void*>(buffer);
  }

  char* AllocateBuffer() override {
    char* buf = (char*)spdk_zmalloc(
        PAGE_SIZE, 64, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    CHECK_NE(buf, nullptr) << "Failed to allocate DMA buffer";
    return buf;
  }
  char* AllocateBuffer(uint64_t page_count) override {
    char* buf = (char*)spdk_zmalloc(
        PAGE_SIZE * page_count,
        64,
        NULL,
        SPDK_ENV_SOCKET_ID_ANY,
        SPDK_MALLOC_DMA);
    CHECK_NE(buf, nullptr) << "Failed to allocate DMA buffer (" << page_count
                           << " pages)";
    return buf;
  }
  void FreeBuffer(char* buf) override { spdk_free(buf); }

  // Unpin a page, if dirty, write it back
  void Unpin(coroutine<void>::push_type& sink,
             uint64_t index,
             PageID_t page_id,
             void* page_data,
             bool is_dirty) override {
    if (is_dirty)
      WritePage(sink, index, page_id, reinterpret_cast<char*>(page_data));
    spdk_free(page_data);
  }
  void Unpin(PageID_t page_id, void* page_data, bool is_dirty) override {
    if (is_dirty)
      WritePage(page_id, reinterpret_cast<char*>(page_data));
    spdk_free(page_data);
  }

  void ReadPageAsync(coroutine<void>::push_type& sink,
                     uint64_t index,
                     PageID_t page_id,
                     char* buffer) override {
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    pending++;
    int ret = spdk_nvme_ns_cmd_read(
        ns_entry.ns,
        qpair,
        buffer,
        page_id * (PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns)),
        PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns),
        io_complete,
        (void*)index,
        0);
    if (ret != 0) {
      pending--;
      throw std::runtime_error("Failed to read page:" + std::to_string(ret));
    }
    sink();
  }
  void ReadPageSync(PageID_t page_id, char* buffer) override {
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    pending++;
    int ret = spdk_nvme_ns_cmd_read(
        ns_entry.ns,
        qpair,
        buffer,
        page_id * (PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns)),
        PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns),
        io_complete,
        (void*)(-1),
        0);
    if (ret != 0) {
      pending--;
      throw std::runtime_error("Failed to read page:" + std::to_string(ret));
    }
    while (pending > 0) {
      spdk_nvme_qpair_process_completions(qpair, 0);
    }
  }

  void WritePageAsync(coroutine<void>::push_type& sink,
                      uint64_t index,
                      PageID_t page_id,
                      char* buffer) override {
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    pending++;
    int ret = spdk_nvme_ns_cmd_write(
        ns_entry.ns,
        qpair,
        buffer,
        page_id * (PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns)),
        PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns),
        io_complete,
        (void*)index,
        0);
    if (ret != 0) {
      pending--;
      throw std::runtime_error("Failed to write page:" + std::to_string(ret));
    }
    sink();
  }
  void WritePageSync(PageID_t page_id, char* buffer) override {
    // LOG(INFO) << "WritePageSync: page_id=" << page_id
    //           << " qpair_initialized=" << ThreadQpair::instance().initialized
    //           << " pending=" << pending
    //           << " thread=" << std::this_thread::get_id();
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    // LOG(INFO) << "WritePageSync: qpair=" << (void*)qpair;
    pending++;
    int ret = spdk_nvme_ns_cmd_write(
        ns_entry.ns,
        qpair,
        buffer,
        page_id * (PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns)),
        PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns),
        io_complete,
        (void*)(-1),
        0);
    // LOG(INFO) << "WritePageSync: cmd_write ret=" << ret;
    if (ret != 0) {
      pending--;
      throw std::runtime_error("Failed to write page:" + std::to_string(ret));
    }
    int poll_count = 0;
    while (pending > 0) {
      int n = spdk_nvme_qpair_process_completions(qpair, 0);
      poll_count++;
      if (poll_count % 1000000 == 0)
        LOG(WARNING) << "WritePageSync: stuck polling, pending=" << pending
                     << " poll_count=" << poll_count << " completions=" << n;
    }
  }

  void submit() override {
    // SPDK submits requests during spdk_nvme_ns_cmd_{read,write} calls.
  }

  void PollCompletion() override {
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    spdk_nvme_qpair_process_completions(qpair, 0);
  }

  void BatchWritePages(const std::vector<IOEntry>& entries) override {
    if (entries.empty())
      return;
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    int max_inflight              = std::max(queue_cnt / 2, 1);
    size_t submitted              = 0;
    int inflight                  = 0;
    uint32_t sectors_per_page =
        PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns);

    while (submitted < entries.size() || inflight > 0) {
      while (submitted < entries.size() && inflight < max_inflight) {
        auto& e = entries[submitted];
        int ret = spdk_nvme_ns_cmd_write(
            ns_entry.ns,
            qpair,
            e.buffer,
            e.page_id * sectors_per_page,
            e.page_count * sectors_per_page,
            batch_write_complete,
            &inflight,
            0);
        if (ret != 0)
          throw std::runtime_error(
              "BatchWritePages: write failed: " + std::to_string(ret));
        inflight++;
        submitted++;
      }
      spdk_nvme_qpair_process_completions(qpair, 0);
    }
  }

  void BatchReadPages(const std::vector<IOEntry>& entries) override {
    if (entries.empty())
      return;
    struct spdk_nvme_qpair* qpair = get_thread_qpair();
    int max_inflight              = std::max(queue_cnt / 2, 1);
    size_t submitted              = 0;
    int inflight                  = 0;
    uint32_t sectors_per_page =
        PAGE_SIZE / spdk_nvme_ns_get_sector_size(ns_entry.ns);

    while (submitted < entries.size() || inflight > 0) {
      while (submitted < entries.size() && inflight < max_inflight) {
        auto& e = entries[submitted];
        int ret = spdk_nvme_ns_cmd_read(
            ns_entry.ns,
            qpair,
            e.buffer,
            e.page_id * sectors_per_page,
            e.page_count * sectors_per_page,
            batch_write_complete,
            &inflight,
            0);
        if (ret != 0)
          throw std::runtime_error(
              "BatchReadPages: read failed: " + std::to_string(ret));
        inflight++;
        submitted++;
      }
      spdk_nvme_qpair_process_completions(qpair, 0);
    }
  }
};

extern "C" void RecStoreForceLinkSpdkBackend() {}

FACTORY_REGISTER(IOBackend, SPDK, SpdkBackend, const BaseKVConfig&);
