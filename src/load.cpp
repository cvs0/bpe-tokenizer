#include "model.hpp"
#include "utf8.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace bpe {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int base64_digit(char c) {
  if (c >= 'A' && c <= 'Z') {
    return c - 'A';
  }
  if (c >= 'a' && c <= 'z') {
    return c - 'a' + 26;
  }
  if (c >= '0' && c <= '9') {
    return c - '0' + 52;
  }
  if (c == '+') {
    return 62;
  }
  if (c == '/') {
    return 63;
  }
  return -1;
}

std::string base64_decode(std::string_view in) {
  std::string out;
  out.reserve(in.size() * 3 / 4);
  int value = 0;
  int bits = -8;
  for (char c : in) {
    if (c == '=') {
      break;
    }
    const int digit = base64_digit(c);
    if (digit < 0) {
      throw std::runtime_error("invalid base64 in vocab");
    }
    value = (value << 6) + digit;
    bits += 6;
    if (bits >= 0) {
      out.push_back(static_cast<char>((value >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return out;
}

std::string base64_encode(std::string_view in) {
  static constexpr char kTable[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((in.size() + 2) / 3 * 4);
  std::uint32_t buffer = 0;
  int bits = 0;
  for (unsigned char c : in) {
    buffer = (buffer << 8) | c;
    bits += 8;
    while (bits >= 6) {
      bits -= 6;
      out.push_back(kTable[(buffer >> bits) & 0x3F]);
    }
  }
  if (bits > 0) {
    out.push_back(kTable[(buffer << (6 - bits)) & 0x3F]);
  }
  while (out.size() % 4 != 0) {
    out.push_back('=');
  }
  return out;
}

bool printable_byte(int byte) {
  if (byte == 32) {
    return false;
  }
  if (byte >= 33 && byte <= 126) {
    return true;
  }
  if (byte >= 161 && byte <= 172) {
    return true;
  }
  if (byte >= 174 && byte <= 255) {
    return true;
  }
  return false;
}

// GPT-2 data-gym map: Unicode code point -> original byte.
// Printable bytes (except space) map to themselves. The other 68 bytes map from U+0100 upward
// in ascending byte order. This is tiktoken's bytes_to_unicode for U+0000..U+00FF.
std::array<int16_t, 384> data_gym_map() {
  std::array<int16_t, 384> map{};
  map.fill(-1);
  int extra = 0;
  for (int byte = 0; byte < 256; ++byte) {
    if (printable_byte(byte)) {
      map[static_cast<std::size_t>(byte)] = static_cast<int16_t>(byte);
    } else {
      map[static_cast<std::size_t>(256 + extra)] = static_cast<int16_t>(byte);
      ++extra;
    }
  }
  return map;
}

std::string map_codepoints(std::string_view utf8, const std::array<int16_t, 384>& map) {
  std::string out;
  out.reserve(utf8.size());
  std::size_t index = 0;
  while (index < utf8.size()) {
    std::uint32_t cp = 0;
    if (!utf8::next(utf8, index, cp)) {
      throw std::runtime_error("invalid UTF-8 in GPT-2 vocab");
    }
    if (cp >= map.size() || map[cp] < 0) {
      throw std::runtime_error("code point outside the GPT-2 byte map");
    }
    out.push_back(static_cast<char>(map[cp]));
  }
  return out;
}

struct Parser {
  const char* p;
  const char* end;

  explicit Parser(std::string_view text) : p(text.data()), end(text.data() + text.size()) {}

  void skip_ws() {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) {
      ++p;
    }
  }

  [[noreturn]] void fail(const char* message) const { throw std::runtime_error(message); }

  char get() {
    if (p >= end) {
      fail("truncated JSON");
    }
    return *p++;
  }

  std::uint32_t hex4() {
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = get();
      value <<= 4;
      if (c >= '0' && c <= '9') {
        value += static_cast<std::uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        value += static_cast<std::uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        value += static_cast<std::uint32_t>(c - 'A' + 10);
      } else {
        fail("bad hex escape in JSON");
      }
    }
    return value;
  }

  void append_cp(std::string& out, std::uint32_t cp) const {
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
      fail("invalid code point in JSON string");
    }
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  std::string parse_string() {
    if (get() != '"') {
      fail("expected JSON string");
    }
    std::string out;
    while (p < end) {
      const char c = get();
      if (c == '"') {
        return out;
      }
      if (c != '\\') {
        if (static_cast<unsigned char>(c) < 0x80) {
          out.push_back(c);
          continue;
        }
        --p;
        const char* start = p;
        std::string_view rest(start, static_cast<std::size_t>(end - start));
        std::size_t local = 0;
        std::uint32_t cp = 0;
        if (!utf8::next(rest, local, cp)) {
          fail("invalid UTF-8 in JSON string");
        }
        append_cp(out, cp);
        p = start + local;
        continue;
      }
      const char esc = get();
      switch (esc) {
      case '"':
      case '\\':
      case '/':
        out.push_back(esc);
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'u': {
        std::uint32_t cp = hex4();
        if (cp >= 0xD800 && cp <= 0xDBFF) {
          if (get() != '\\' || get() != 'u') {
            fail("lone surrogate in JSON string");
          }
          const std::uint32_t low = hex4();
          if (low < 0xDC00 || low > 0xDFFF) {
            fail("bad surrogate pair in JSON string");
          }
          cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
          fail("lone low surrogate in JSON string");
        }
        append_cp(out, cp);
        break;
      }
      default:
        fail("unknown JSON escape");
      }
    }
    fail("unterminated JSON string");
  }

  std::uint32_t parse_uint() {
    skip_ws();
    if (p >= end || !std::isdigit(static_cast<unsigned char>(*p))) {
      fail("expected JSON integer");
    }
    std::uint64_t value = 0;
    while (p < end && std::isdigit(static_cast<unsigned char>(*p))) {
      value = value * 10 + static_cast<std::uint64_t>(*p - '0');
      if (value > 0xFFFFFFFFULL) {
        fail("token id does not fit in uint32");
      }
      ++p;
    }
    return static_cast<std::uint32_t>(value);
  }
};

std::shared_ptr<const detail::Model>
build_model(Pattern pattern,
            std::string name,
            std::vector<std::pair<std::string, std::uint32_t>> tokens,
            std::vector<SpecialToken> specials) {
  auto model = std::make_shared<detail::Model>();
  model->pattern = pattern;
  model->name = std::move(name);
  model->byte_id.fill(detail::kNoRank);
  std::uint32_t max_id = 0;
  std::size_t arena_bytes = 0;
  for (const auto& token : tokens) {
    if (token.first.empty()) {
      throw std::runtime_error("empty token bytes");
    }
    max_id = std::max(max_id, token.second);
    arena_bytes += token.first.size();
  }
  for (const SpecialToken& special : specials) {
    if (special.text.empty() || !utf8::valid(special.text)) {
      throw std::runtime_error("special token must be non-empty valid UTF-8");
    }
    max_id = std::max(max_id, special.id);
    arena_bytes += special.text.size();
  }
  model->dec_off.assign(static_cast<std::size_t>(max_id) + 1, detail::kNoRank);
  model->dec_len.assign(static_cast<std::size_t>(max_id) + 1, 0);
  model->arena.reserve(arena_bytes);
  model->dict.reserve(tokens.size());
  model->tokens.reserve(tokens.size());

  std::unordered_map<std::string, std::uint32_t> seen;
  seen.reserve(tokens.size() * 2);
  for (auto& token : tokens) {
    if (!seen.emplace(token.first, token.second).second) {
      throw std::runtime_error("duplicate token bytes");
    }
    if (model->dec_off[token.second] != detail::kNoRank) {
      throw std::runtime_error("duplicate token id");
    }
    const auto offset = static_cast<std::uint32_t>(model->arena.size());
    model->arena.insert(model->arena.end(), token.first.begin(), token.first.end());
    const auto length = static_cast<std::uint32_t>(token.first.size());
    model->dict.insert(model->arena.data(), offset, length, token.second);
    model->dec_off[token.second] = offset;
    model->dec_len[token.second] = length;
    if (length == 1) {
      model->byte_id[static_cast<std::uint8_t>(token.first[0])] = token.second;
    }
    model->tokens.push_back(detail::TokenRec{offset, length, token.second});
  }
  for (int byte = 0; byte < 256; ++byte) {
    if (model->byte_id[static_cast<std::size_t>(byte)] == detail::kNoRank) {
      throw std::runtime_error("vocab is missing a single-byte token");
    }
  }
  model->n_mergeable = tokens.size();
  model->specials.reserve(specials.size());
  for (SpecialToken& special : specials) {
    if (model->dec_off[special.id] != detail::kNoRank) {
      throw std::runtime_error("special token id collides with another token");
    }
    const auto offset = static_cast<std::uint32_t>(model->arena.size());
    model->arena.insert(model->arena.end(), special.text.begin(), special.text.end());
    model->dec_off[special.id] = offset;
    model->dec_len[special.id] = static_cast<std::uint32_t>(special.text.size());
    model->specials.push_back(std::move(special));
  }

  std::size_t splits = 0;
  for (const detail::TokenRec& token : model->tokens) {
    if (token.length >= 2) {
      splits += static_cast<std::size_t>(token.length - 1);
    }
  }
  if (splits > 0) {
    model->pairs.reserve(splits);
  }
  const char* base = model->arena.data();
  for (const detail::TokenRec& token : model->tokens) {
    if (token.length < 2) {
      continue;
    }
    const char* bytes = base + token.offset;
    for (std::uint32_t cut = 1; cut < token.length; ++cut) {
      std::uint32_t left = 0;
      std::uint32_t right = 0;
      if (!model->dict.find(base, bytes, cut, left)) {
        continue;
      }
      if (!model->dict.find(base, bytes + cut, token.length - cut, right)) {
        continue;
      }
      model->pairs.insert((static_cast<std::uint64_t>(left) << 32) | right, token.id);
    }
  }
  return model;
}

std::vector<SpecialToken> cl100k_specials() {
  return {
      {"<|endoftext|>", 100257},
      {"<|fim_prefix|>", 100258},
      {"<|fim_middle|>", 100259},
      {"<|fim_suffix|>", 100260},
      {"<|endofprompt|>", 100276},
  };
}

std::vector<SpecialToken> gpt2_specials() {
  return {{"<|endoftext|>", 50256}};
}

Pattern parse_pattern(std::string_view name) {
  if (name == "gpt2") {
    return Pattern::Gpt2;
  }
  if (name == "cl100k") {
    return Pattern::Cl100k;
  }
  throw std::runtime_error("unknown pattern name");
}

const char* pattern_header(Pattern pattern) {
  return pattern == Pattern::Gpt2 ? "gpt2" : "cl100k";
}

} // namespace

Encoding Encoding::from_tokens(Pattern pattern,
                               std::string name,
                               std::vector<std::pair<std::string, std::uint32_t>> tokens,
                               std::vector<SpecialToken> specials) {
  return Encoding(build_model(pattern, std::move(name), std::move(tokens), std::move(specials)));
}

Encoding Encoding::from_tiktoken(const std::filesystem::path& path,
                                 Pattern pattern,
                                 std::vector<SpecialToken> specials) {
  const std::string text = read_file(path);
  std::vector<std::pair<std::string, std::uint32_t>> tokens;
  tokens.reserve(100000);
  std::size_t i = 0;
  while (i < text.size()) {
    std::size_t end = text.find('\n', i);
    if (end == std::string::npos) {
      end = text.size();
    }
    std::string_view line(text.data() + i, end - i);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    i = end + (end < text.size() ? 1 : 0);
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const std::size_t sp = line.find(' ');
    if (sp == std::string_view::npos) {
      throw std::runtime_error("malformed .tiktoken line");
    }
    std::string bytes = base64_decode(line.substr(0, sp));
    std::uint32_t id = 0;
    try {
      std::size_t used = 0;
      unsigned long value = std::stoul(std::string(line.substr(sp + 1)), &used, 10);
      if (used != line.size() - sp - 1 || value > 0xFFFFFFFFul) {
        throw std::runtime_error("bad rank");
      }
      id = static_cast<std::uint32_t>(value);
    } catch (const std::exception&) {
      throw std::runtime_error("malformed rank in .tiktoken file");
    }
    tokens.emplace_back(std::move(bytes), id);
  }
  std::string name = path.stem().string();
  return from_tokens(pattern, std::move(name), std::move(tokens), std::move(specials));
}

Encoding Encoding::load(const std::filesystem::path& path) {
  const std::string text = read_file(path);
  Pattern pattern = Pattern::Gpt2;
  bool saw_pattern = false;
  std::vector<SpecialToken> specials;
  std::size_t i = 0;
  while (i < text.size()) {
    std::size_t end = text.find('\n', i);
    if (end == std::string::npos) {
      end = text.size();
    }
    std::string_view line(text.data() + i, end - i);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    if (line.empty() || line.front() != '#') {
      break;
    }
    constexpr std::string_view kPattern = "# pattern=";
    constexpr std::string_view kSpecial = "# special ";
    if (line.rfind(kPattern, 0) == 0) {
      pattern = parse_pattern(line.substr(kPattern.size()));
      saw_pattern = true;
    } else if (line.rfind(kSpecial, 0) == 0) {
      const std::string_view rest = line.substr(kSpecial.size());
      const std::size_t sp = rest.find(' ');
      if (sp == std::string_view::npos) {
        throw std::runtime_error("malformed special header");
      }
      SpecialToken special;
      special.text = base64_decode(rest.substr(0, sp));
      special.id = static_cast<std::uint32_t>(std::stoul(std::string(rest.substr(sp + 1))));
      specials.push_back(std::move(special));
    }
    i = end + (end < text.size() ? 1 : 0);
    if (end == text.size()) {
      break;
    }
  }
  if (!saw_pattern) {
    throw std::runtime_error("tiktoken file is missing a # pattern= header");
  }
  return from_tiktoken(path, pattern, std::move(specials));
}

Encoding Encoding::from_gpt2_files(const std::filesystem::path& encoder_json,
                                   const std::filesystem::path& vocab_bpe) {
  const auto map = data_gym_map();
  std::vector<std::uint8_t> byte_order;
  byte_order.reserve(256);
  for (int byte = 0; byte < 256; ++byte) {
    if (printable_byte(byte)) {
      byte_order.push_back(static_cast<std::uint8_t>(byte));
    }
  }
  for (int byte = 0; byte < 256; ++byte) {
    if (!printable_byte(byte)) {
      byte_order.push_back(static_cast<std::uint8_t>(byte));
    }
  }
  if (byte_order.size() != 256) {
    throw std::logic_error("byte order is not 256 entries");
  }

  std::vector<std::pair<std::string, std::uint32_t>> tokens;
  tokens.reserve(50256);
  for (std::uint32_t id = 0; id < 256; ++id) {
    tokens.emplace_back(std::string(1, static_cast<char>(byte_order[id])), id);
  }

  const std::string merges = read_file(vocab_bpe);
  std::size_t line_start = 0;
  bool first = true;
  std::uint32_t next_id = 256;
  while (line_start < merges.size()) {
    std::size_t end = merges.find('\n', line_start);
    if (end == std::string::npos) {
      end = merges.size();
    }
    std::string_view line(merges.data() + line_start, end - line_start);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    line_start = end + (end < merges.size() ? 1 : 0);
    if (line.empty()) {
      continue;
    }
    if (first) {
      first = false;
      if (line.rfind("#version", 0) == 0) {
        continue;
      }
    }
    const std::size_t sp = line.find(' ');
    if (sp == std::string_view::npos) {
      throw std::runtime_error("malformed vocab.bpe merge");
    }
    const std::string left = map_codepoints(line.substr(0, sp), map);
    const std::string right = map_codepoints(line.substr(sp + 1), map);
    tokens.emplace_back(left + right, next_id);
    ++next_id;
  }

  const std::string json = read_file(encoder_json);
  Parser parser(json);
  parser.skip_ws();
  if (parser.get() != '{') {
    throw std::runtime_error("encoder.json must be an object");
  }
  std::unordered_map<std::string, std::uint32_t> encoder;
  encoder.reserve(50300);
  parser.skip_ws();
  if (parser.p < parser.end && *parser.p == '}') {
    ++parser.p;
  } else {
    while (true) {
      parser.skip_ws();
      std::string key = parser.parse_string();
      parser.skip_ws();
      if (parser.get() != ':') {
        throw std::runtime_error("expected ':' in encoder.json");
      }
      const std::uint32_t id = parser.parse_uint();
      if (key != "<|endoftext|>" && key != "<|startoftext|>") {
        encoder.emplace(map_codepoints(key, map), id);
      } else if (key == "<|endoftext|>" && id != 50256) {
        throw std::runtime_error("encoder.json <|endoftext|> id is not 50256");
      }
      parser.skip_ws();
      if (parser.p >= parser.end) {
        throw std::runtime_error("truncated encoder.json");
      }
      if (*parser.p == ',') {
        ++parser.p;
        continue;
      }
      if (*parser.p == '}') {
        ++parser.p;
        break;
      }
      throw std::runtime_error("expected ',' or '}' in encoder.json");
    }
  }

  if (encoder.size() != tokens.size()) {
    throw std::runtime_error("vocab.bpe and encoder.json have different sizes");
  }
  for (const auto& token : tokens) {
    const auto it = encoder.find(token.first);
    if (it == encoder.end() || it->second != token.second) {
      throw std::runtime_error("vocab.bpe does not match encoder.json");
    }
  }
  return from_tokens(Pattern::Gpt2, "gpt2", std::move(tokens), gpt2_specials());
}

Encoding Encoding::preset(std::string_view name, const std::filesystem::path& dir) {
  if (name == "gpt2") {
    return from_gpt2_files(dir / "encoder.json", dir / "vocab.bpe");
  }
  if (name == "cl100k_base") {
    return from_tiktoken(dir / "cl100k_base.tiktoken", Pattern::Cl100k, cl100k_specials());
  }
  throw std::invalid_argument("unknown preset (expected gpt2 or cl100k_base)");
}

void Encoding::save_tiktoken(const std::filesystem::path& path) const {
  if (!model_) {
    throw std::logic_error("empty encoding");
  }
  const detail::Model& model = *model_;
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot write " + path.string());
  }
  out << "# pattern=" << pattern_header(model.pattern) << '\n';
  for (const SpecialToken& special : model.specials) {
    out << "# special " << base64_encode(special.text) << ' ' << special.id << '\n';
  }
  std::vector<detail::TokenRec> tokens = model.tokens;
  std::sort(tokens.begin(), tokens.end(), [](const detail::TokenRec& a, const detail::TokenRec& b) {
    return a.id < b.id;
  });
  for (const detail::TokenRec& token : tokens) {
    out << base64_encode(std::string_view(model.arena.data() + token.offset, token.length)) << ' '
        << token.id << '\n';
  }
}

} // namespace bpe
