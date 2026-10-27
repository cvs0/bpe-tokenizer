#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bpe {

enum class Pattern {
  Gpt2,   // tiktoken gpt2 / r50k_base / p50k_base split
  Cl100k, // tiktoken cl100k_base split
};

struct SpecialToken {
  std::string text;
  std::uint32_t id = 0;
};

namespace detail {
struct Model;
}

// Immutable byte-level BPE model. Copies share the loaded tables.
// encode_* requires valid UTF-8 and throws std::invalid_argument otherwise.
// decode returns the raw token bytes and does not check UTF-8.
class Encoding {
public:
  Encoding() = default;

  // "gpt2" reads encoder.json and vocab.bpe. "cl100k_base" reads cl100k_base.tiktoken.
  [[nodiscard]] static Encoding preset(std::string_view name, const std::filesystem::path& dir);

  [[nodiscard]] static Encoding from_gpt2_files(const std::filesystem::path& encoder_json,
                                                const std::filesystem::path& vocab_bpe);

  [[nodiscard]] static Encoding from_tiktoken(const std::filesystem::path& path,
                                              Pattern pattern,
                                              std::vector<SpecialToken> specials = {});

  // First line must be "# pattern=gpt2" or "# pattern=cl100k".
  [[nodiscard]] static Encoding load(const std::filesystem::path& path);

  // One-byte tokens for every byte 0..255 are required.
  // A multi-byte token id is the merge priority of that exact byte string.
  // Smaller id wins. tiktoken ranks satisfy this.
  [[nodiscard]] static Encoding
  from_tokens(Pattern pattern,
              std::string name,
              std::vector<std::pair<std::string, std::uint32_t>> tokens,
              std::vector<SpecialToken> specials = {});

  void save_tiktoken(const std::filesystem::path& path) const;

  [[nodiscard]] bool empty() const noexcept { return model_ == nullptr; }
  [[nodiscard]] const std::string& name() const;
  [[nodiscard]] Pattern pattern() const;
  [[nodiscard]] std::size_t vocab_size() const;
  [[nodiscard]] std::uint32_t max_token_id() const;
  [[nodiscard]] const std::vector<SpecialToken>& special_tokens() const;
  [[nodiscard]] std::string token_bytes(std::uint32_t id) const;

  [[nodiscard]] std::vector<std::uint32_t> encode_ordinary(std::string_view text) const;

  // `allowed` entries must be registered specials. Any other special text throws.
  [[nodiscard]] std::vector<std::uint32_t>
  encode(std::string_view text, const std::unordered_set<std::string>& allowed) const;

  [[nodiscard]] std::vector<std::uint32_t> encode_with_specials(std::string_view text) const;

  // num_threads == 0 uses hardware_concurrency. Tables are read-only; no locks on the hot path.
  [[nodiscard]] std::vector<std::vector<std::uint32_t>>
  encode_ordinary_batch(std::span<const std::string_view> texts, std::size_t num_threads = 0) const;

  [[nodiscard]] std::string decode(std::span<const std::uint32_t> ids) const;

private:
  explicit Encoding(std::shared_ptr<const detail::Model> model) : model_(std::move(model)) {}

  std::shared_ptr<const detail::Model> model_;
};

struct TrainConfig {
  std::size_t vocab_size = 512; // includes the 256 byte tokens
  Pattern pattern = Pattern::Gpt2;
  std::uint64_t min_frequency = 2;
};

// Frequency BPE on pieces from `config.pattern`.
// Byte b has id b. Merge ids start at 256.
// Tie break: higher pair frequency, then smaller left id, then smaller right id.
[[nodiscard]] Encoding train(std::string_view corpus, const TrainConfig& config);

} // namespace bpe
