#include "model.hpp"
#include "scan.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace bpe {
namespace {

struct Word {
  std::vector<std::uint32_t> syms;
  std::uint64_t freq = 0;
};

struct Item {
  std::int64_t count = 0;
  std::uint64_t key = 0;
  // priority_queue pops the greatest element. Highest count, then smallest key.
  bool operator<(const Item& other) const noexcept {
    if (count != other.count) {
      return count < other.count;
    }
    return key > other.key;
  }
};

std::uint64_t pair_key(std::uint32_t left, std::uint32_t right) {
  return (static_cast<std::uint64_t>(left) << 32) | right;
}

} // namespace

Encoding train(std::string_view corpus, const TrainConfig& config) {
  if (config.vocab_size < 256) {
    throw std::invalid_argument("vocab_size must be at least 256");
  }
  std::unordered_map<std::string, std::uint64_t> freqs;
  const bool ok = detail::scan::for_each_piece(
      config.pattern, corpus, [&](std::string_view piece) { ++freqs[std::string(piece)]; });
  if (!ok) {
    throw std::invalid_argument("invalid UTF-8");
  }

  std::vector<Word> words;
  words.reserve(freqs.size());
  for (const auto& entry : freqs) {
    Word word;
    word.freq = entry.second;
    word.syms.reserve(entry.first.size());
    for (unsigned char byte : entry.first) {
      word.syms.push_back(byte);
    }
    words.push_back(std::move(word));
  }

  std::unordered_map<std::uint64_t, std::int64_t> counts;
  std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> where;
  for (std::uint32_t wi = 0; wi < words.size(); ++wi) {
    const Word& word = words[wi];
    const auto weight = static_cast<std::int64_t>(word.freq);
    for (std::size_t i = 0; i + 1 < word.syms.size(); ++i) {
      const std::uint64_t key = pair_key(word.syms[i], word.syms[i + 1]);
      counts[key] += weight;
      where[key].push_back(wi);
    }
  }

  std::priority_queue<Item> heap;
  for (const auto& entry : counts) {
    if (entry.second > 0) {
      heap.push(Item{entry.second, entry.first});
    }
  }

  std::vector<std::string> sym_bytes(256);
  for (int byte = 0; byte < 256; ++byte) {
    sym_bytes[static_cast<std::size_t>(byte)] = std::string(1, static_cast<char>(byte));
  }

  auto touch = [&](std::uint64_t key, std::int64_t delta, std::uint32_t wi) {
    std::int64_t& count = counts[key];
    count += delta;
    if (count < 0) {
      throw std::logic_error("BPE pair count went negative");
    }
    if (delta > 0) {
      where[key].push_back(wi);
    }
    heap.push(Item{count, key});
  };

  while (sym_bytes.size() < config.vocab_size && !heap.empty()) {
    const Item item = heap.top();
    heap.pop();
    const auto it = counts.find(item.key);
    if (it == counts.end() || it->second != item.count || item.count <= 0) {
      continue;
    }
    if (static_cast<std::uint64_t>(item.count) < config.min_frequency) {
      break;
    }
    const auto left = static_cast<std::uint32_t>(item.key >> 32);
    const auto right = static_cast<std::uint32_t>(item.key);
    const auto created = static_cast<std::uint32_t>(sym_bytes.size());
    sym_bytes.push_back(sym_bytes[left] + sym_bytes[right]);

    std::vector<std::uint32_t> owners = where[item.key];
    std::sort(owners.begin(), owners.end());
    owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
    for (std::uint32_t wi : owners) {
      std::vector<std::uint32_t>& syms = words[wi].syms;
      const auto weight = static_cast<std::int64_t>(words[wi].freq);
      std::size_t i = 0;
      while (i + 1 < syms.size()) {
        if (syms[i] != left || syms[i + 1] != right) {
          ++i;
          continue;
        }
        if (i > 0) {
          touch(pair_key(syms[i - 1], left), -weight, wi);
          touch(pair_key(syms[i - 1], created), weight, wi);
        }
        touch(pair_key(left, right), -weight, wi);
        if (i + 2 < syms.size()) {
          const std::uint32_t after = syms[i + 2];
          touch(pair_key(right, after), -weight, wi);
          touch(pair_key(created, after), weight, wi);
        }
        syms[i] = created;
        syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(i + 1));
        ++i;
      }
    }
  }

  std::vector<std::pair<std::string, std::uint32_t>> tokens;
  tokens.reserve(sym_bytes.size());
  for (std::uint32_t id = 0; id < sym_bytes.size(); ++id) {
    tokens.emplace_back(std::move(sym_bytes[id]), id);
  }
  const char* name = config.pattern == Pattern::Gpt2 ? "train-gpt2" : "train-cl100k";
  return Encoding::from_tokens(config.pattern, name, std::move(tokens));
}

} // namespace bpe
