#include "model.hpp"

#include <cstring>
#include <queue>
#include <stdexcept>
#include <vector>

namespace bpe::detail {
namespace {

constexpr std::size_t kSmallPiece = 128;

struct HeapNode {
  std::uint32_t id = 0;
  std::uint32_t prev = kNoRank;
  std::uint32_t next = kNoRank;
  std::uint32_t version = 0;
};

struct Cand {
  std::uint32_t rank = 0;
  std::uint32_t index = 0;
  std::uint32_t version = 0;
};

// Min-heap by rank, then by start index. priority_queue is a max-heap, so invert.
struct CandOrder {
  bool operator()(const Cand& a, const Cand& b) const noexcept {
    if (a.rank != b.rank) {
      return a.rank > b.rank;
    }
    return a.index > b.index;
  }
};

void merge_small(const Model& model, std::string_view piece, std::vector<std::uint32_t>& out) {
  const auto n = static_cast<std::uint32_t>(piece.size());
  std::uint32_t ids[kSmallPiece];
  std::uint32_t rank[kSmallPiece];
  for (std::uint32_t i = 0; i < n; ++i) {
    ids[i] = model.byte_id[static_cast<std::uint8_t>(piece[i])];
  }
  for (std::uint32_t i = 0; i + 1 < n; ++i) {
    rank[i] = model.pairs.find(ids[i], ids[i + 1]);
  }
  std::uint32_t len = n;
  while (len > 1) {
    std::uint32_t best = kNoRank;
    std::uint32_t at = 0;
    for (std::uint32_t i = 0; i + 1 < len; ++i) {
      if (rank[i] < best) {
        best = rank[i];
        at = i;
      }
    }
    if (best == kNoRank) {
      break;
    }
    // best is the id of the concatenated token. That is the tiktoken rank invariant.
    ids[at] = best;
    if (at > 0) {
      rank[at - 1] = model.pairs.find(ids[at - 1], ids[at]);
    }
    if (at + 2 < len) {
      rank[at] = model.pairs.find(ids[at], ids[at + 2]);
    } else {
      rank[at] = kNoRank;
    }
    const std::uint32_t tail = len - at - 2;
    std::memmove(
        ids + at + 1, ids + at + 2, static_cast<std::size_t>(tail) * sizeof(std::uint32_t));
    std::memmove(
        rank + at + 1, rank + at + 2, static_cast<std::size_t>(tail) * sizeof(std::uint32_t));
    --len;
  }
  out.insert(out.end(), ids, ids + len);
}

// Doubly linked symbols plus a binary heap of candidate merges.
// Stale heap entries are dropped by a per-node version, so each real merge is O(log n).
void merge_heap(const Model& model, std::string_view piece, std::vector<std::uint32_t>& out) {
  const auto n = static_cast<std::uint32_t>(piece.size());
  std::vector<HeapNode> nodes(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    nodes[i].id = model.byte_id[static_cast<std::uint8_t>(piece[i])];
    nodes[i].prev = (i == 0) ? kNoRank : i - 1;
    nodes[i].next = (i + 1 == n) ? kNoRank : i + 1;
  }
  std::priority_queue<Cand, std::vector<Cand>, CandOrder> heap;
  for (std::uint32_t i = 0; i + 1 < n; ++i) {
    const std::uint32_t rank = model.pairs.find(nodes[i].id, nodes[i + 1].id);
    if (rank != kNoRank) {
      heap.push(Cand{rank, i, 0});
    }
  }
  while (!heap.empty()) {
    const Cand cand = heap.top();
    heap.pop();
    if (cand.version != nodes[cand.index].version) {
      continue;
    }
    const std::uint32_t i = cand.index;
    const std::uint32_t j = nodes[i].next;
    if (j == kNoRank) {
      continue;
    }
    const std::uint32_t k = nodes[j].next;
    nodes[i].id = cand.rank;
    nodes[i].next = k;
    if (k != kNoRank) {
      nodes[k].prev = i;
    }
    ++nodes[j].version;
    ++nodes[i].version;
    if (k != kNoRank) {
      const std::uint32_t rank = model.pairs.find(nodes[i].id, nodes[k].id);
      if (rank != kNoRank) {
        heap.push(Cand{rank, i, nodes[i].version});
      }
    }
    const std::uint32_t prev = nodes[i].prev;
    if (prev != kNoRank) {
      ++nodes[prev].version;
      const std::uint32_t rank = model.pairs.find(nodes[prev].id, nodes[i].id);
      if (rank != kNoRank) {
        heap.push(Cand{rank, prev, nodes[prev].version});
      }
    }
  }
  for (std::uint32_t i = 0; i != kNoRank; i = nodes[i].next) {
    out.push_back(nodes[i].id);
  }
}

} // namespace

void append_merge(const Model& model, std::string_view piece, std::vector<std::uint32_t>& out) {
  if (piece.empty()) {
    return;
  }
  if (piece.size() > 0xFFFFFFFFu) {
    throw std::runtime_error("piece exceeds 32-bit length");
  }
  if (piece.size() == 1) {
    out.push_back(model.byte_id[static_cast<std::uint8_t>(piece[0])]);
    return;
  }
  if (piece.size() <= kSmallPiece) {
    merge_small(model, piece, out);
  } else {
    merge_heap(model, piece, out);
  }
}

} // namespace bpe::detail
