#pragma once

#include "bpe/encoding.hpp"
#include "unicode.hpp"
#include "utf8.hpp"

#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace bpe::detail {
namespace scan {

inline bool is_letter(std::uint32_t cp) noexcept {
  return (unicode::flags(cp) & unicode::kLetter) != 0;
}
inline bool is_number(std::uint32_t cp) noexcept {
  return (unicode::flags(cp) & unicode::kNumber) != 0;
}
inline bool is_space(std::uint32_t cp) noexcept {
  return (unicode::flags(cp) & unicode::kSpace) != 0;
}
inline bool is_other(std::uint32_t cp) noexcept {
  return (unicode::flags(cp) & (unicode::kLetter | unicode::kNumber | unicode::kSpace)) == 0;
}

// Leftmost-first scan of
// '(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
// ` ?` is U+0020. `\s++$` is a whitespace run that reaches the end (possessive, no give-back).
// `\s+(?!\S)` on an internal run of length r>=2 matches r-1 and leaves the last whitespace.
template <typename Get, typename Emit> void gpt2(std::size_t n, Get get, Emit emit) {
  std::size_t i = 0;
  while (i < n) {
    const std::uint32_t c0 = get(i);
    if (c0 == static_cast<std::uint32_t>('\'')) {
      if (i + 1 < n) {
        const std::uint32_t c1 = get(i + 1);
        if (c1 == 's' || c1 == 'd' || c1 == 'm' || c1 == 't') {
          emit(i, i + 2);
          i += 2;
          continue;
        }
      }
      if (i + 2 < n) {
        const std::uint32_t c1 = get(i + 1);
        const std::uint32_t c2 = get(i + 2);
        if ((c1 == 'l' && c2 == 'l') || (c1 == 'v' && c2 == 'e') || (c1 == 'r' && c2 == 'e')) {
          emit(i, i + 3);
          i += 3;
          continue;
        }
      }
    }

    const std::size_t spaced = (c0 == static_cast<std::uint32_t>(' ')) ? i + 1 : i;
    auto take = [&](auto pred) -> bool {
      std::size_t k = spaced;
      while (k < n && pred(get(k))) {
        ++k;
      }
      if (k > spaced) {
        emit(i, k);
        i = k;
        return true;
      }
      return false;
    };
    if (take(is_letter) || take(is_number) || take(is_other)) {
      continue;
    }

    if (!is_space(c0)) {
      throw std::logic_error("gpt2 pretokenizer did not advance");
    }
    std::size_t k = i + 1;
    while (k < n && is_space(get(k))) {
      ++k;
    }
    // `\s++$` consumes the rest of the string when it is all whitespace.
    if (k == n) {
      emit(i, k);
      break;
    }
    // `\s+(?!\S)`: greedy whitespace backtracks by one and the lookahead succeeds.
    if (k - i >= 2) {
      emit(i, k - 1);
      i = k - 1;
      continue;
    }
    emit(i, i + 1);
    ++i;
  }
}

// Leftmost-first scan of the cl100k_base pattern in tiktoken 0.14.0:
// '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|
// ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s Case-fold for the contraction is ASCII
// plus U+017F (latin small letter long s), which is the only non-ASCII scalar the engine treats as
// a single-character contraction before the letter run.
template <typename Get, typename Emit> void cl100k(std::size_t n, Get get, Emit emit) {
  std::size_t i = 0;
  while (i < n) {
    const std::uint32_t c0 = get(i);
    if (c0 == static_cast<std::uint32_t>('\'')) {
      if (i + 1 < n) {
        const std::uint32_t c1 = get(i + 1);
        if (c1 == 's' || c1 == 'S' || c1 == 'd' || c1 == 'D' || c1 == 'm' || c1 == 'M' ||
            c1 == 't' || c1 == 'T' || c1 == 0x017Fu) {
          emit(i, i + 2);
          i += 2;
          continue;
        }
      }
      if (i + 2 < n) {
        auto lower = [](std::uint32_t c) { return (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c; };
        const std::uint32_t a = lower(get(i + 1));
        const std::uint32_t b = lower(get(i + 2));
        if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') || (a == 'r' && b == 'e')) {
          emit(i, i + 3);
          i += 3;
          continue;
        }
      }
    }

    const bool prefix_ok = c0 != '\r' && c0 != '\n' && !is_letter(c0) && !is_number(c0);
    if (prefix_ok && i + 1 < n && is_letter(get(i + 1))) {
      std::size_t k = i + 2;
      while (k < n && is_letter(get(k))) {
        ++k;
      }
      emit(i, k);
      i = k;
      continue;
    }
    if (is_letter(c0)) {
      std::size_t k = i + 1;
      while (k < n && is_letter(get(k))) {
        ++k;
      }
      emit(i, k);
      i = k;
      continue;
    }
    if (is_number(c0)) {
      std::size_t k = i + 1;
      while (k < n && (k - i) < 3 && is_number(get(k))) {
        ++k;
      }
      emit(i, k);
      i = k;
      continue;
    }

    {
      const std::size_t j = (c0 == static_cast<std::uint32_t>(' ')) ? i + 1 : i;
      if (j < n && is_other(get(j))) {
        std::size_t k = j + 1;
        while (k < n && is_other(get(k))) {
          ++k;
        }
        while (k < n) {
          const std::uint32_t ck = get(k);
          if (ck != '\r' && ck != '\n') {
            break;
          }
          ++k;
        }
        emit(i, k);
        i = k;
        continue;
      }
    }

    if (!is_space(c0)) {
      throw std::logic_error("cl100k pretokenizer did not advance");
    }
    std::size_t k = i + 1;
    while (k < n && is_space(get(k))) {
      ++k;
    }
    if (k == n) {
      emit(i, k);
      break;
    }
    // `\s*[\r\n]`: greedy `\s*` then one CR/LF, so the match ends at the last CR/LF in the run.
    std::size_t last_break = static_cast<std::size_t>(-1);
    for (std::size_t t = i; t < k; ++t) {
      const std::uint32_t ct = get(t);
      if (ct == '\r' || ct == '\n') {
        last_break = t;
      }
    }
    if (last_break != static_cast<std::size_t>(-1)) {
      emit(i, last_break + 1);
      i = last_break + 1;
      continue;
    }
    if (k - i >= 2) {
      emit(i, k - 1);
      i = k - 1;
      continue;
    }
    emit(i, i + 1);
    ++i;
  }
}

template <typename Fn> bool for_each_piece(Pattern pattern, std::string_view text, Fn&& fn) {
  auto run = [&](auto&& get, auto&& emit, std::size_t n) {
    if (pattern == Pattern::Gpt2) {
      gpt2(n, get, emit);
    } else {
      cl100k(n, get, emit);
    }
  };
  if (utf8::is_ascii(text)) {
    run([&](std::size_t i) -> std::uint32_t { return static_cast<std::uint8_t>(text[i]); },
        [&](std::size_t a, std::size_t b) { fn(text.substr(a, b - a)); },
        text.size());
    return true;
  }
  std::vector<std::uint32_t> cps;
  std::vector<std::uint32_t> byte_of;
  if (!utf8::decode(text, cps, byte_of)) {
    return false;
  }
  run([&](std::size_t i) -> std::uint32_t { return cps[i]; },
      [&](std::size_t a, std::size_t b) { fn(text.substr(byte_of[a], byte_of[b] - byte_of[a])); },
      cps.size());
  return true;
}

} // namespace scan
} // namespace bpe::detail
