#include "model.hpp"
#include "scan.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <utility>

namespace bpe {
namespace {

const detail::Model& checked(const std::shared_ptr<const detail::Model>& model) {
  if (!model) {
    throw std::logic_error("empty encoding");
  }
  return *model;
}

void append_ordinary(const detail::Model& model,
                     std::string_view text,
                     std::vector<std::uint32_t>& ids) {
  const bool ok = detail::scan::for_each_piece(model.pattern, text, [&](std::string_view piece) {
    std::uint32_t id = 0;
    if (model.dict.find(model.arena.data(), piece.data(), piece.size(), id)) {
      ids.push_back(id);
    } else {
      detail::append_merge(model, piece, ids);
    }
  });
  if (!ok) {
    throw std::invalid_argument("invalid UTF-8");
  }
}

// Leftmost match, then the longest special. Built-in gpt2 and cl100k specials are not prefixes
// of each other, so this agrees with tiktoken's alternation on those sets.
bool find_special(std::string_view text,
                  std::size_t from,
                  const std::vector<SpecialToken>& specs,
                  std::size_t& which,
                  std::size_t& pos) {
  bool found = false;
  which = 0;
  pos = 0;
  for (std::size_t i = 0; i < specs.size(); ++i) {
    if (specs[i].text.empty()) {
      continue;
    }
    const std::size_t at = text.find(specs[i].text, from);
    if (at == std::string_view::npos) {
      continue;
    }
    if (!found || at < pos || (at == pos && specs[i].text.size() > specs[which].text.size())) {
      found = true;
      which = i;
      pos = at;
    }
  }
  return found;
}

std::vector<std::uint32_t> encode_specials(const detail::Model& model,
                                           std::string_view text,
                                           const std::vector<SpecialToken>& allowed,
                                           bool error_on_disallowed) {
  if (error_on_disallowed) {
    for (const SpecialToken& special : model.specials) {
      bool ok = false;
      for (const SpecialToken& item : allowed) {
        if (item.text == special.text) {
          ok = true;
          break;
        }
      }
      if (!ok && text.find(special.text) != std::string_view::npos) {
        throw std::runtime_error("disallowed special token: " + special.text);
      }
    }
  }
  std::vector<std::uint32_t> ids;
  ids.reserve(text.size() / 4 + 1);
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    std::size_t which = 0;
    std::size_t pos = 0;
    if (!find_special(text, cursor, allowed, which, pos)) {
      append_ordinary(model, text.substr(cursor), ids);
      break;
    }
    if (pos > cursor) {
      append_ordinary(model, text.substr(cursor, pos - cursor), ids);
    }
    ids.push_back(allowed[which].id);
    cursor = pos + allowed[which].text.size();
  }
  return ids;
}

} // namespace

const std::string& Encoding::name() const {
  return checked(model_).name;
}

Pattern Encoding::pattern() const {
  return checked(model_).pattern;
}

std::size_t Encoding::vocab_size() const {
  const detail::Model& model = checked(model_);
  return model.n_mergeable + model.specials.size();
}

std::uint32_t Encoding::max_token_id() const {
  const detail::Model& model = checked(model_);
  if (model.dec_off.empty()) {
    throw std::logic_error("encoding has no tokens");
  }
  return static_cast<std::uint32_t>(model.dec_off.size() - 1);
}

const std::vector<SpecialToken>& Encoding::special_tokens() const {
  return checked(model_).specials;
}

std::string Encoding::token_bytes(std::uint32_t id) const {
  const detail::Model& model = checked(model_);
  if (id >= model.dec_off.size() || model.dec_off[id] == detail::kNoRank) {
    throw std::out_of_range("unknown token id");
  }
  return std::string(model.arena.data() + model.dec_off[id], model.dec_len[id]);
}

std::vector<std::uint32_t> Encoding::encode_ordinary(std::string_view text) const {
  const detail::Model& model = checked(model_);
  std::vector<std::uint32_t> ids;
  ids.reserve(text.size() / 4 + 1);
  append_ordinary(model, text, ids);
  return ids;
}

std::vector<std::uint32_t> Encoding::encode(std::string_view text,
                                            const std::unordered_set<std::string>& allowed) const {
  const detail::Model& model = checked(model_);
  std::vector<SpecialToken> picked;
  picked.reserve(allowed.size());
  for (const std::string& name : allowed) {
    bool found = false;
    for (const SpecialToken& special : model.specials) {
      if (special.text == name) {
        picked.push_back(special);
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::invalid_argument("unknown special token: " + name);
    }
  }
  return encode_specials(model, text, picked, true);
}

std::vector<std::uint32_t> Encoding::encode_with_specials(std::string_view text) const {
  const detail::Model& model = checked(model_);
  return encode_specials(model, text, model.specials, false);
}

std::vector<std::vector<std::uint32_t>>
Encoding::encode_ordinary_batch(std::span<const std::string_view> texts,
                                std::size_t num_threads) const {
  checked(model_);
  std::vector<std::vector<std::uint32_t>> out(texts.size());
  if (texts.empty()) {
    return out;
  }
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }
  if (num_threads == 0) {
    num_threads = 1;
  }
  if (num_threads > texts.size()) {
    num_threads = texts.size();
  }
  if (num_threads <= 1) {
    for (std::size_t i = 0; i < texts.size(); ++i) {
      out[i] = encode_ordinary(texts[i]);
    }
    return out;
  }

  std::vector<std::uint64_t> prefix(texts.size() + 1);
  for (std::size_t i = 0; i < texts.size(); ++i) {
    prefix[i + 1] = prefix[i] + texts[i].size();
  }
  const std::uint64_t total = prefix.back();
  std::vector<std::size_t> bounds(num_threads + 1);
  bounds[0] = 0;
  std::size_t cursor = 0;
  for (std::size_t t = 1; t < num_threads; ++t) {
    const std::uint64_t target = total * static_cast<std::uint64_t>(t) / num_threads;
    while (cursor < texts.size() && prefix[cursor] < target) {
      ++cursor;
    }
    bounds[t] = cursor;
  }
  bounds[num_threads] = texts.size();

  std::vector<std::exception_ptr> errors(num_threads);
  std::vector<std::thread> pool;
  pool.reserve(num_threads);
  for (std::size_t t = 0; t < num_threads; ++t) {
    pool.emplace_back([&, t] {
      try {
        for (std::size_t i = bounds[t]; i < bounds[t + 1]; ++i) {
          out[i] = encode_ordinary(texts[i]);
        }
      } catch (...) {
        errors[t] = std::current_exception();
      }
    });
  }
  for (std::thread& thread : pool) {
    thread.join();
  }
  for (const std::exception_ptr& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  return out;
}

std::string Encoding::decode(std::span<const std::uint32_t> ids) const {
  const detail::Model& model = checked(model_);
  std::size_t bytes = 0;
  for (std::uint32_t id : ids) {
    if (id >= model.dec_off.size() || model.dec_off[id] == detail::kNoRank) {
      throw std::out_of_range("unknown token id");
    }
    bytes += model.dec_len[id];
  }
  std::string text(bytes, '\0');
  std::size_t offset = 0;
  for (std::uint32_t id : ids) {
    const std::uint32_t n = model.dec_len[id];
    if (n != 0) {
      std::memcpy(text.data() + offset, model.arena.data() + model.dec_off[id], n);
    }
    offset += n;
  }
  return text;
}

} // namespace bpe
