#pragma once

#include <cstdint>
#include <cassert>
#include <iostream>
#include <string>
#include <sstream>

class UnifiedPointer {
public:
  enum class Type : uint8_t {
    Invalid = 0b00,
    Memory  = 0b01,
    Disk    = 0b10,
    PMem    = 0b11
  };

  UnifiedPointer() : raw_((static_cast<uint64_t>(Type::Invalid) << 62)){};

  explicit operator bool() const { return type() != Type::Invalid; }

  static UnifiedPointer FromInvalid() {
    return UnifiedPointer(); // 直接使用默认构造
  }

  static UnifiedPointer FromRaw(uint64_t raw) {
    UnifiedPointer p;
    p.raw_ = raw;
    return p;
  }

  // 构造内存指针
  static UnifiedPointer FromMemory(void* ptr) {
    UnifiedPointer p;
    p.set(Type::Memory, reinterpret_cast<uint64_t>(ptr));
    return p;
  }

  // 构造磁盘页ID
  static UnifiedPointer FromDisk(uint64_t page_id , uint16_t page_count) {
    UnifiedPointer p;
    assert((page_id & ~((1ULL << 48) - 1)) == 0 && "page_id exceeds 48 bits");
    assert(page_count < (1ULL << 14) && "page_count exceeds 14 bits");
    p.set(Type::Disk, page_id | ((uint64_t)page_count << 48));
    return p;
  }

  // 构造 PMem 指针
  static UnifiedPointer FromPMem(uint64_t pmem_offset) {
    UnifiedPointer p;
    p.set(Type::PMem, pmem_offset);
    return p;
  }

  Type type() const { return static_cast<Type>((raw_ >> 62) & 0b11); }

  uint64_t value() const { return raw_ & VALUE_MASK; }

  uint64_t RawValue() const { return raw_; }

  void* asMemoryPointer() const {
    assert(type() == Type::Memory);
    return reinterpret_cast<void*>(value());
  }

  uint64_t asDiskPageId() const {
    assert(type() == Type::Disk);
    return value() & ((1ULL << 48) - 1);
  }

  uint16_t asDiskPageCount() const {
    assert(type() == Type::Disk);
    return (value() >> 48) & ((1ULL << 14) - 1);
  }

  uint64_t asPMemOffset() const {
    assert(type() == Type::PMem);
    return value();
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "Type: ";
    switch (type()) {
    case Type::Memory:
      oss << "Memory, Addr: 0x" << std::hex << value();
      break;
    case Type::Disk:
      oss << "Disk, PageID: " << std::dec << asDiskPageId() << ", PageCount: " << asDiskPageCount();
      break;
    case Type::PMem:
      oss << "PMem, Offset: " << std::dec << value();
      break;
    case Type::Invalid:
    default:
      oss << "Invalid";
      break;
    }
    return oss.str();
  }

private:
  static constexpr uint64_t VALUE_MASK = (1ULL << 62) - 1;

  void set(Type t, uint64_t value) {
    assert((value & ~VALUE_MASK) == 0 && "Value overflow in 62 bits");
    raw_ = (static_cast<uint64_t>(t) << 62) | value;
  }

  uint64_t raw_;
};