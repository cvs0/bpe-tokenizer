#pragma once

#include "bpe/encoding.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bpe::detail {

inline constexpr std::uint32_t kNoRank = 0xFFFFFFFFu;

inline std::uint64_t hash_bytes(const char* data, std::size_t n) noexcept {
  constexpr std::uint64_t kMul = 0x9DDFEA08EB382D69ULL;
  std::uint64_t h = 0xCBF29CE484222325ULL ^ (static_cast<std::uint64_t>(n) * kMul);
  const auto* p = reinterpret_cast<const std::uint8_t*>(data);
  while (n >= 8) {
    std::uint64_t word = 0;
    std::memcpy(&word, p, 8);
    h ^= word;
    h *= kMul;
    h ^= h >> 32;
    p += 8;
    n -= 8;
  }
  if (n >= 4) {
    std::uint32_t word = 0;
    std::memcpy(&word, p, 4);
    h ^= word;
    h *= kMul;
    p += 4;
    n -= 4;
  }
  while (n > 0) {
    h ^= *p++;
    h *= kMul;
    --n;
  }
  h ^= h >> 33;
  h *= kMul;
  h ^= h >> 33;
  if (h == 0) {
    h = 0x9E3779B97F4A7C15ULL;
  }
  return h;
}

// Open-addressed map from token bytes to id. Keys live in an external arena.
class ByteMap {
public:
  void reserve(std::size_t count) { rehash(cap_for(count)); }

  void insert(const char* arena, std::uint32_t offset, std::uint32_t length, std::uint32_t id) {
    if (slots_.empty() || (size_ + 1) * 2 > slots_.size()) {
      rehash(slots_.empty() ? 16 : slots_.size() * 2);
    }
    const std::uint64_t h = hash_bytes(arena + offset, length);
    place(slots_, h, offset, length, id, arena);
    ++size_;
  }

  [[nodiscard]] bool
  find(const char* arena, const char* data, std::size_t length, std::uint32_t& id) const noexcept {
    if (slots_.empty() || length > 0xFFFFFFFFu) {
      return false;
    }
    const std::uint64_t h = hash_bytes(data, length);
    const std::size_t mask = slots_.size() - 1;
    std::size_t i = static_cast<std::size_t>(h) & mask;
    for (std::size_t n = 0; n < slots_.size(); ++n) {
      const Slot& slot = slots_[i];
      if (slot.hash == 0) {
        return false;
      }
      if (slot.hash == h && slot.length == length &&
          std::memcmp(arena + slot.offset, data, length) == 0) {
        id = slot.id;
        return true;
      }
      i = (i + 1) & mask;
    }
    return false;
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
  struct Slot {
    std::uint64_t hash = 0; // 0: empty. Real hashes are forced non-zero.
    std::uint32_t offset = 0;
    std::uint32_t length = 0;
    std::uint32_t id = 0;
  };

  static std::size_t cap_for(std::size_t count) {
    std::size_t cap = 16;
    while (cap < count * 2) {
      cap *= 2;
    }
    return cap;
  }

  void rehash(std::size_t cap) {
    if (cap < 16) {
      cap = 16;
    }
    std::vector<Slot> next(cap);
    for (const Slot& slot : slots_) {
      if (slot.hash != 0) {
        place_new(next, slot);
      }
    }
    slots_.swap(next);
  }

  static void place(std::vector<Slot>& slots,
                    std::uint64_t hash,
                    std::uint32_t offset,
                    std::uint32_t length,
                    std::uint32_t id,
                    const char* arena) {
    const std::size_t mask = slots.size() - 1;
    std::size_t i = static_cast<std::size_t>(hash) & mask;
    for (;;) {
      Slot& slot = slots[i];
      if (slot.hash == 0) {
        slot.hash = hash;
        slot.offset = offset;
        slot.length = length;
        slot.id = id;
        return;
      }
      if (slot.hash == hash && slot.length == length &&
          std::memcmp(arena + slot.offset, arena + offset, length) == 0) {
        throw std::runtime_error("duplicate token bytes");
      }
      i = (i + 1) & mask;
    }
  }

  static void place_new(std::vector<Slot>& slots, const Slot& src) {
    const std::size_t mask = slots.size() - 1;
    std::size_t i = static_cast<std::size_t>(src.hash) & mask;
    while (slots[i].hash != 0) {
      i = (i + 1) & mask;
    }
    slots[i] = src;
  }

  std::vector<Slot> slots_;
  std::size_t size_ = 0;
};

// Open-addressed map (left_id, right_id) -> merged token id.
class PairMap {
public:
  void reserve(std::size_t count) {
    std::size_t cap = 16;
    while (cap < count * 2) {
      cap *= 2;
    }
    rehash(cap);
  }

  void insert(std::uint64_t key, std::uint32_t value) {
    if (slots_.empty() || (size_ + 1) * 2 > slots_.size()) {
      rehash(slots_.empty() ? 16 : slots_.size() * 2);
    }
    const std::size_t mask = slots_.size() - 1;
    std::size_t i = static_cast<std::size_t>(key * 0x9E3779B185EBCA87ULL) & mask;
    for (;;) {
      Slot& slot = slots_[i];
      if (!slot.occupied) {
        slot.occupied = 1;
        slot.key = key;
        slot.value = value;
        ++size_;
        return;
      }
      if (slot.key == key) {
        if (slot.value != value) {
          throw std::runtime_error("conflicting merge rank");
        }
        return;
      }
      i = (i + 1) & mask;
    }
  }

  [[nodiscard]] std::uint32_t find(std::uint32_t left, std::uint32_t right) const noexcept {
    if (slots_.empty()) {
      return kNoRank;
    }
    const std::uint64_t key = (static_cast<std::uint64_t>(left) << 32) | right;
    const std::size_t mask = slots_.size() - 1;
    std::size_t i = static_cast<std::size_t>(key * 0x9E3779B185EBCA87ULL) & mask;
    for (std::size_t n = 0; n < slots_.size(); ++n) {
      const Slot& slot = slots_[i];
      if (!slot.occupied) {
        return kNoRank;
      }
      if (slot.key == key) {
        return slot.value;
      }
      i = (i + 1) & mask;
    }
    return kNoRank;
  }

private:
  struct Slot {
    std::uint64_t key = 0;
    std::uint32_t value = 0;
    std::uint32_t occupied = 0;
  };

  void rehash(std::size_t cap) {
    std::vector<Slot> next(cap);
    const std::size_t mask = cap - 1;
    for (const Slot& slot : slots_) {
      if (!slot.occupied) {
        continue;
      }
      std::size_t i = static_cast<std::size_t>(slot.key * 0x9E3779B185EBCA87ULL) & mask;
      while (next[i].occupied) {
        i = (i + 1) & mask;
      }
      next[i] = slot;
    }
    slots_.swap(next);
  }

  std::vector<Slot> slots_;
  std::size_t size_ = 0;
};

struct TokenRec {
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
  std::uint32_t id = 0;
};

struct Model {
  Pattern pattern = Pattern::Gpt2;
  std::string name;
  std::vector<char> arena;
  ByteMap dict;
  PairMap pairs;
  std::array<std::uint32_t, 256> byte_id{};
  std::vector<std::uint32_t> dec_off;
  std::vector<std::uint32_t> dec_len;
  std::vector<TokenRec> tokens;
  std::vector<SpecialToken> specials;
  std::size_t n_mergeable = 0;
};

void append_merge(const Model& model, std::string_view piece, std::vector<std::uint32_t>& out);

} // namespace bpe::detail
