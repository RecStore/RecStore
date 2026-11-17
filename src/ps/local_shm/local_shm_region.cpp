#include "local_shm_region.h"

#include "ps/local_shm/local_shm_queue.h"

#include <cerrno>
#include <cstring>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace recstore {

namespace {

std::string NormalizeRegionName(const std::string& region_name) {
  if (!region_name.empty() && region_name.front() == '/') {
    return region_name;
  }
  return "/" + region_name;
}

} // namespace

LocalShmRegion::~LocalShmRegion() { Close(); }

bool LocalShmRegion::Create(const std::string& region_name,
                            uint32_t slot_count,
                            uint32_t slot_buffer_bytes,
                            uint32_t ready_queue_count) {
  Close();
  if (slot_count == 0 || slot_buffer_bytes == 0 || ready_queue_count == 0) {
    LOG(ERROR) << "LocalShmRegion::Create invalid geometry";
    return false;
  }

  region_name_       = NormalizeRegionName(region_name);
  slot_count_        = slot_count;
  slot_buffer_bytes_ = slot_buffer_bytes;
  ready_queue_count_ = ready_queue_count;
  mapped_size_ =
      TotalRegionBytes(slot_count_, slot_buffer_bytes_, ready_queue_count_);

  fd_ = shm_open(region_name_.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd_ < 0) {
    LOG(ERROR) << "shm_open create failed for " << region_name_ << ": "
               << std::strerror(errno);
    return false;
  }
  if (ftruncate(fd_, static_cast<off_t>(mapped_size_)) != 0) {
    LOG(ERROR) << "ftruncate failed for " << region_name_ << ": "
               << std::strerror(errno);
    Close();
    return false;
  }
  if (!MapRegion(fd_, mapped_size_)) {
    Close();
    return false;
  }

  auto* ctrl              = control();
  ctrl->magic             = kLocalShmMagic;
  ctrl->version           = kLocalShmVersion;
  ctrl->reserved0         = 0;
  ctrl->server_epoch      = 1;
  ctrl->server_pid        = static_cast<int64_t>(getpid());
  ctrl->slot_count        = slot_count_;
  ctrl->slot_buffer_bytes = slot_buffer_bytes_;
  ctrl->ready_queue_count = ready_queue_count_;
  ctrl->reserved1         = 0;
  ctrl->next_request_id   = 1;
  ctrl->fatal_status      = 0;
  ctrl->request_doorbell  = 0;
  ctrl->free_doorbell     = 0;
  ctrl->active_clients    = 0;
  std::memset(ctrl->reserved, 0, sizeof(ctrl->reserved));

  for (uint32_t free_queue_id = 0; free_queue_id < ready_queue_count_;
       ++free_queue_id) {
    LocalShmQueueInitialize(free_queue_header(free_queue_id),
                            free_queue_cells(free_queue_id),
                            slot_count_);
  }
  for (uint32_t ready_queue_id = 0; ready_queue_id < ready_queue_count_;
       ++ready_queue_id) {
    LocalShmQueueInitialize(ready_queue_header(ready_queue_id),
                            ready_queue_cells(ready_queue_id),
                            slot_count_);
  }

  for (uint32_t slot_id = 0; slot_id < slot_count_; ++slot_id) {
    auto* header              = slot_header(slot_id);
    header->state             = static_cast<uint32_t>(LocalSlotState::kFree);
    header->opcode            = static_cast<uint32_t>(LocalOpcode::kInvalid);
    header->status_code       = static_cast<uint32_t>(LocalStatusCode::kOk);
    header->client_id         = 0;
    header->request_id        = 0;
    header->client_pid        = 0;
    header->table_name_len    = 0;
    header->key_count         = 0;
    header->embedding_dim     = 0;
    header->learning_rate     = 0.0f;
    header->input_bytes       = 0;
    header->output_bytes      = 0;
    header->server_seen_epoch = ctrl->server_epoch;
    header->user_tag          = 0;
    header->client_enqueue_timestamp_ns = 0;
    header->server_dequeue_timestamp_ns = 0;
    header->server_backend_duration_us  = 0;
    header->completion_doorbell         = 0;
    header->error_message_len           = 0;
    std::memset(header->reserved, 0, sizeof(header->reserved));
    std::memset(slot_payload(slot_id), 0, slot_buffer_bytes_);
    const uint32_t free_queue_id = slot_id % ready_queue_count_;
    CHECK(LocalShmQueueEnqueue(free_queue_header(free_queue_id),
                               free_queue_cells(free_queue_id),
                               slot_id));
  }

  return true;
}

bool LocalShmRegion::Attach(const std::string& region_name,
                            uint32_t expected_slot_count,
                            uint32_t expected_slot_buffer_bytes,
                            uint32_t expected_ready_queue_count) {
  Close();
  region_name_ = NormalizeRegionName(region_name);
  fd_          = shm_open(region_name_.c_str(), O_RDWR, 0666);
  if (fd_ < 0) {
    LOG(ERROR) << "shm_open attach failed for " << region_name_ << ": "
               << std::strerror(errno);
    return false;
  }

  struct stat st {};
  if (fstat(fd_, &st) != 0) {
    LOG(ERROR) << "fstat failed for " << region_name_ << ": "
               << std::strerror(errno);
    Close();
    return false;
  }
  if (!MapRegion(fd_, static_cast<std::size_t>(st.st_size))) {
    Close();
    return false;
  }
  const auto* ctrl   = control();
  slot_count_        = ctrl->slot_count;
  slot_buffer_bytes_ = ctrl->slot_buffer_bytes;
  ready_queue_count_ = ctrl->ready_queue_count;
  if (ctrl->magic != kLocalShmMagic || ctrl->version != kLocalShmVersion) {
    LOG(ERROR) << "LocalShmRegion attach validation failed for "
               << region_name_;
    Close();
    return false;
  }
  if (!ValidateGeometry(expected_slot_count,
                        expected_slot_buffer_bytes,
                        expected_ready_queue_count)) {
    Close();
    return false;
  }
  return true;
}

void LocalShmRegion::Close() {
  if (base_ != nullptr && mapped_size_ > 0) {
    munmap(base_, mapped_size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
  base_              = nullptr;
  fd_                = -1;
  mapped_size_       = 0;
  slot_count_        = 0;
  slot_buffer_bytes_ = 0;
  ready_queue_count_ = 0;
  region_name_.clear();
}

LocalShmControlBlock* LocalShmRegion::control() {
  return reinterpret_cast<LocalShmControlBlock*>(base_);
}

const LocalShmControlBlock* LocalShmRegion::control() const {
  return reinterpret_cast<const LocalShmControlBlock*>(base_);
}

LocalShmQueueHeader* LocalShmRegion::queue_header(LocalQueueKind kind) {
  auto* bytes = reinterpret_cast<uint8_t*>(base_) +
                QueueHeaderOffset(kind, ready_queue_count_);
  return reinterpret_cast<LocalShmQueueHeader*>(bytes);
}

const LocalShmQueueHeader*
LocalShmRegion::queue_header(LocalQueueKind kind) const {
  const auto* bytes = reinterpret_cast<const uint8_t*>(base_) +
                      QueueHeaderOffset(kind, ready_queue_count_);
  return reinterpret_cast<const LocalShmQueueHeader*>(bytes);
}

LocalShmQueueCell* LocalShmRegion::queue_cells(LocalQueueKind kind) {
  auto* bytes = reinterpret_cast<uint8_t*>(base_) +
                QueueCellArrayOffset(slot_count_, kind, ready_queue_count_);
  return reinterpret_cast<LocalShmQueueCell*>(bytes);
}

const LocalShmQueueCell*
LocalShmRegion::queue_cells(LocalQueueKind kind) const {
  const auto* bytes =
      reinterpret_cast<const uint8_t*>(base_) +
      QueueCellArrayOffset(slot_count_, kind, ready_queue_count_);
  return reinterpret_cast<const LocalShmQueueCell*>(bytes);
}

LocalShmQueueHeader* LocalShmRegion::free_queue_header(uint32_t free_queue_id) {
  CHECK_LT(free_queue_id, ready_queue_count_);
  auto* bytes = reinterpret_cast<uint8_t*>(base_) +
                FreeQueueHeaderOffset(free_queue_id, ready_queue_count_);
  return reinterpret_cast<LocalShmQueueHeader*>(bytes);
}

const LocalShmQueueHeader*
LocalShmRegion::free_queue_header(uint32_t free_queue_id) const {
  CHECK_LT(free_queue_id, ready_queue_count_);
  const auto* bytes = reinterpret_cast<const uint8_t*>(base_) +
                      FreeQueueHeaderOffset(free_queue_id, ready_queue_count_);
  return reinterpret_cast<const LocalShmQueueHeader*>(bytes);
}

LocalShmQueueCell* LocalShmRegion::free_queue_cells(uint32_t free_queue_id) {
  CHECK_LT(free_queue_id, ready_queue_count_);
  auto* bytes =
      reinterpret_cast<uint8_t*>(base_) +
      FreeQueueCellsOffset(slot_count_, free_queue_id, ready_queue_count_);
  return reinterpret_cast<LocalShmQueueCell*>(bytes);
}

const LocalShmQueueCell*
LocalShmRegion::free_queue_cells(uint32_t free_queue_id) const {
  CHECK_LT(free_queue_id, ready_queue_count_);
  const auto* bytes =
      reinterpret_cast<const uint8_t*>(base_) +
      FreeQueueCellsOffset(slot_count_, free_queue_id, ready_queue_count_);
  return reinterpret_cast<const LocalShmQueueCell*>(bytes);
}

LocalShmQueueHeader*
LocalShmRegion::ready_queue_header(uint32_t ready_queue_id) {
  CHECK_LT(ready_queue_id, ready_queue_count_);
  auto* bytes = reinterpret_cast<uint8_t*>(base_) +
                ReadyQueueHeaderOffset(ready_queue_id, ready_queue_count_);
  return reinterpret_cast<LocalShmQueueHeader*>(bytes);
}

const LocalShmQueueHeader*
LocalShmRegion::ready_queue_header(uint32_t ready_queue_id) const {
  CHECK_LT(ready_queue_id, ready_queue_count_);
  const auto* bytes =
      reinterpret_cast<const uint8_t*>(base_) +
      ReadyQueueHeaderOffset(ready_queue_id, ready_queue_count_);
  return reinterpret_cast<const LocalShmQueueHeader*>(bytes);
}

LocalShmQueueCell* LocalShmRegion::ready_queue_cells(uint32_t ready_queue_id) {
  CHECK_LT(ready_queue_id, ready_queue_count_);
  auto* bytes =
      reinterpret_cast<uint8_t*>(base_) +
      ReadyQueueCellsOffset(slot_count_, ready_queue_id, ready_queue_count_);
  return reinterpret_cast<LocalShmQueueCell*>(bytes);
}

const LocalShmQueueCell*
LocalShmRegion::ready_queue_cells(uint32_t ready_queue_id) const {
  CHECK_LT(ready_queue_id, ready_queue_count_);
  const auto* bytes =
      reinterpret_cast<const uint8_t*>(base_) +
      ReadyQueueCellsOffset(slot_count_, ready_queue_id, ready_queue_count_);
  return reinterpret_cast<const LocalShmQueueCell*>(bytes);
}

LocalShmSlotHeader* LocalShmRegion::slot_header(uint32_t slot_id) {
  CHECK_LT(slot_id, slot_count_);
  auto* bytes = reinterpret_cast<uint8_t*>(base_) +
                SlotHeadersOffset(slot_count_, ready_queue_count_) +
                sizeof(LocalShmSlotHeader) * static_cast<std::size_t>(slot_id);
  return reinterpret_cast<LocalShmSlotHeader*>(bytes);
}

const LocalShmSlotHeader* LocalShmRegion::slot_header(uint32_t slot_id) const {
  CHECK_LT(slot_id, slot_count_);
  const auto* bytes =
      reinterpret_cast<const uint8_t*>(base_) +
      SlotHeadersOffset(slot_count_, ready_queue_count_) +
      sizeof(LocalShmSlotHeader) * static_cast<std::size_t>(slot_id);
  return reinterpret_cast<const LocalShmSlotHeader*>(bytes);
}

uint8_t* LocalShmRegion::slot_payload(uint32_t slot_id) {
  CHECK_LT(slot_id, slot_count_);
  return reinterpret_cast<uint8_t*>(base_) +
         SlotPayloadOffset(
             slot_count_, slot_buffer_bytes_, slot_id, ready_queue_count_);
}

const uint8_t* LocalShmRegion::slot_payload(uint32_t slot_id) const {
  CHECK_LT(slot_id, slot_count_);
  return reinterpret_cast<const uint8_t*>(base_) +
         SlotPayloadOffset(
             slot_count_, slot_buffer_bytes_, slot_id, ready_queue_count_);
}

bool LocalShmRegion::MapRegion(int fd, std::size_t mapped_size) {
  void* mapped =
      mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    LOG(ERROR) << "mmap failed: " << std::strerror(errno);
    base_ = nullptr;
    return false;
  }
  base_        = mapped;
  mapped_size_ = mapped_size;
  return true;
}

bool LocalShmRegion::ValidateGeometry(
    uint32_t expected_slot_count,
    uint32_t expected_slot_buffer_bytes,
    uint32_t expected_ready_queue_count) const {
  if (expected_slot_count != 0 && expected_slot_count != slot_count_) {
    LOG(ERROR) << "slot_count mismatch: expected " << expected_slot_count
               << " actual " << slot_count_;
    return false;
  }
  if (expected_slot_buffer_bytes != 0 &&
      expected_slot_buffer_bytes != slot_buffer_bytes_) {
    LOG(ERROR) << "slot_buffer_bytes mismatch: expected "
               << expected_slot_buffer_bytes << " actual "
               << slot_buffer_bytes_;
    return false;
  }
  if (expected_ready_queue_count != 0 &&
      expected_ready_queue_count != ready_queue_count_) {
    LOG(ERROR) << "ready_queue_count mismatch: expected "
               << expected_ready_queue_count << " actual "
               << ready_queue_count_;
    return false;
  }
  return true;
}

} // namespace recstore
