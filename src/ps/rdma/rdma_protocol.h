#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace petps {

inline constexpr std::uint32_t kPutPayloadMagic = 0x50545053;
inline constexpr std::uint16_t kProtocolVersion = 1;

struct PutPayloadHeader {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t reserved;
  std::uint32_t key_count;
  std::uint32_t embedding_dim;
};

struct DecodedPutPayload {
  std::uint32_t embedding_dim = 0;
  std::vector<std::uint64_t> keys;
  std::vector<float> values;
};

inline std::size_t
FixedSlotResponseBytes(std::size_t key_count, std::size_t value_size_bytes) {
  return key_count * value_size_bytes + sizeof(std::int32_t);
}

inline std::string
EncodePutPayload(const std::vector<std::uint64_t>& keys,
                 const std::vector<std::vector<float>>& values) {
  if (keys.empty()) {
    return {};
  }

  const std::size_t embedding_dim = values.front().size();
  PutPayloadHeader header{
      kPutPayloadMagic,
      kProtocolVersion,
      0,
      static_cast<std::uint32_t>(keys.size()),
      static_cast<std::uint32_t>(embedding_dim),
  };

  std::string payload;
  payload.resize(
      sizeof(PutPayloadHeader) + keys.size() * sizeof(std::uint64_t) +
      keys.size() * embedding_dim * sizeof(float));

  char* cursor = payload.data();
  std::memcpy(cursor, &header, sizeof(header));
  cursor += sizeof(header);

  std::memcpy(cursor, keys.data(), keys.size() * sizeof(std::uint64_t));
  cursor += keys.size() * sizeof(std::uint64_t);

  for (const auto& row : values) {
    std::memcpy(cursor, row.data(), embedding_dim * sizeof(float));
    cursor += embedding_dim * sizeof(float);
  }

  return payload;
}

inline bool DecodePutPayload(
    std::string_view payload, DecodedPutPayload* decoded, std::string* error) {
  if (payload.size() < sizeof(PutPayloadHeader)) {
    if (error != nullptr) {
      *error = "payload smaller than header";
    }
    return false;
  }

  PutPayloadHeader header{};
  std::memcpy(&header, payload.data(), sizeof(header));

  if (header.magic != kPutPayloadMagic) {
    if (error != nullptr) {
      *error = "bad payload magic";
    }
    return false;
  }
  if (header.version != kProtocolVersion) {
    if (error != nullptr) {
      *error = "bad protocol version";
    }
    return false;
  }

  const std::size_t expected_bytes =
      sizeof(PutPayloadHeader) + header.key_count * sizeof(std::uint64_t) +
      static_cast<std::size_t>(header.key_count) * header.embedding_dim *
          sizeof(float);

  if (payload.size() != expected_bytes) {
    if (error != nullptr) {
      *error = "payload byte size mismatch";
    }
    return false;
  }

  decoded->embedding_dim = header.embedding_dim;
  decoded->keys.resize(header.key_count);
  decoded->values.resize(
      static_cast<std::size_t>(header.key_count) * header.embedding_dim);

  const char* cursor = payload.data() + sizeof(PutPayloadHeader);
  std::memcpy(
      decoded->keys.data(), cursor, header.key_count * sizeof(std::uint64_t));
  cursor += header.key_count * sizeof(std::uint64_t);
  std::memcpy(
      decoded->values.data(), cursor, decoded->values.size() * sizeof(float));

  return true;
}

} // namespace petps
