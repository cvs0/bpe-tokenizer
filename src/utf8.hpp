#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace bpe::utf8 {

inline bool is_ascii(std::string_view text) noexcept {
  const auto* p = reinterpret_cast<const std::uint8_t*>(text.data());
  std::size_t i = 0;
  for (; i + 8 <= text.size(); i += 8) {
    std::uint64_t word = 0;
    std::memcpy(&word, p + i, 8);
    if (word & 0x8080808080808080ULL) {
      return false;
    }
  }
  for (; i < text.size(); ++i) {
    if (p[i] & 0x80U) {
      return false;
    }
  }
  return true;
}

// Advances `index` over one scalar. False on invalid UTF-8 or end of input.
inline bool next(std::string_view text, std::size_t& index, std::uint32_t& cp) noexcept {
  if (index >= text.size()) {
    return false;
  }
  const auto* p = reinterpret_cast<const std::uint8_t*>(text.data());
  const std::uint8_t c0 = p[index];
  if (c0 < 0x80U) {
    cp = c0;
    ++index;
    return true;
  }
  if ((c0 & 0xE0U) == 0xC0U) {
    if (index + 1 >= text.size()) {
      return false;
    }
    const std::uint8_t c1 = p[index + 1];
    if ((c1 & 0xC0U) != 0x80U) {
      return false;
    }
    cp = (static_cast<std::uint32_t>(c0 & 0x1FU) << 6) | (c1 & 0x3FU);
    if (cp < 0x80U) {
      return false;
    }
    index += 2;
    return true;
  }
  if ((c0 & 0xF0U) == 0xE0U) {
    if (index + 2 >= text.size()) {
      return false;
    }
    const std::uint8_t c1 = p[index + 1];
    const std::uint8_t c2 = p[index + 2];
    if ((c1 & 0xC0U) != 0x80U || (c2 & 0xC0U) != 0x80U) {
      return false;
    }
    cp = (static_cast<std::uint32_t>(c0 & 0x0FU) << 12) |
         (static_cast<std::uint32_t>(c1 & 0x3FU) << 6) | (c2 & 0x3FU);
    if (cp < 0x800U || (cp >= 0xD800U && cp <= 0xDFFFU)) {
      return false;
    }
    index += 3;
    return true;
  }
  if ((c0 & 0xF8U) == 0xF0U) {
    if (index + 3 >= text.size()) {
      return false;
    }
    const std::uint8_t c1 = p[index + 1];
    const std::uint8_t c2 = p[index + 2];
    const std::uint8_t c3 = p[index + 3];
    if ((c1 & 0xC0U) != 0x80U || (c2 & 0xC0U) != 0x80U || (c3 & 0xC0U) != 0x80U) {
      return false;
    }
    cp = (static_cast<std::uint32_t>(c0 & 0x07U) << 18) |
         (static_cast<std::uint32_t>(c1 & 0x3FU) << 12) |
         (static_cast<std::uint32_t>(c2 & 0x3FU) << 6) | (c3 & 0x3FU);
    if (cp < 0x10000U || cp > 0x10FFFFU) {
      return false;
    }
    index += 4;
    return true;
  }
  return false;
}

inline bool valid(std::string_view text) noexcept {
  if (is_ascii(text)) {
    return true;
  }
  std::size_t index = 0;
  std::uint32_t cp = 0;
  while (index < text.size()) {
    if (!next(text, index, cp)) {
      return false;
    }
  }
  return true;
}

inline bool decode(std::string_view text,
                   std::vector<std::uint32_t>& cps,
                   std::vector<std::uint32_t>& byte_of) {
  cps.clear();
  byte_of.clear();
  if (text.size() > 0xFFFFFFFFu) {
    return false;
  }
  cps.reserve(text.size());
  byte_of.reserve(text.size() + 1);
  std::size_t index = 0;
  while (index < text.size()) {
    byte_of.push_back(static_cast<std::uint32_t>(index));
    std::uint32_t cp = 0;
    if (!next(text, index, cp)) {
      return false;
    }
    cps.push_back(cp);
  }
  byte_of.push_back(static_cast<std::uint32_t>(text.size()));
  return true;
}

} // namespace bpe::utf8
