#include "bpe/encoding.hpp"
#include "sha256.hpp"
#include "utf8.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string utf8_of(std::uint32_t cp) {
  std::string out;
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
  return out;
}

int b64_digit(char c) {
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

std::string b64_decode(std::string_view in) {
  std::string out;
  int value = 0;
  int bits = -8;
  for (char c : in) {
    if (c == '=') {
      break;
    }
    const int digit = b64_digit(c);
    if (digit < 0) {
      return {};
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

std::vector<std::uint32_t> parse_ids(std::string_view text) {
  std::vector<std::uint32_t> ids;
  std::size_t i = 0;
  while (i < text.size()) {
    if (text[i] == ',') {
      ++i;
      continue;
    }
    std::uint64_t value = 0;
    if (text[i] < '0' || text[i] > '9') {
      break;
    }
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
      value = value * 10 + static_cast<std::uint64_t>(text[i] - '0');
      ++i;
    }
    ids.push_back(static_cast<std::uint32_t>(value));
  }
  return ids;
}

void feed_text(bpe::test::Sha256& hash, const bpe::Encoding& enc, std::string text) {
  const std::vector<std::uint32_t> ids = enc.encode_ordinary(text);
  hash.u32(static_cast<std::uint32_t>(ids.size()));
  for (std::uint32_t id : ids) {
    hash.u32(id);
  }
}

std::string sweep(const bpe::Encoding& enc, std::uint32_t limit) {
  bpe::test::Sha256 hash;
  for (std::uint32_t cp = 0; cp < limit; ++cp) {
    if (cp >= 0xD800 && cp <= 0xDFFF) {
      continue;
    }
    const std::string ch = utf8_of(cp);
    feed_text(hash, enc, ch);
    feed_text(hash, enc, " " + ch);
    feed_text(hash, enc, ch + " ");
    feed_text(hash, enc, ch + "\n");
    feed_text(hash, enc, "\n" + ch);
    feed_text(hash, enc, "'" + ch);
    feed_text(hash, enc, "'" + ch + "a");
    feed_text(hash, enc, ch + ch);
    feed_text(hash, enc, "a" + ch + "1");
  }
  return hash.hex();
}

bool vocabs_present() {
  const std::filesystem::path dir(BPE_DATA_DIR);
  return std::filesystem::exists(dir / "encoder.json") &&
         std::filesystem::exists(dir / "vocab.bpe") &&
         std::filesystem::exists(dir / "cl100k_base.tiktoken");
}

bpe::Encoding load_preset(const char* name) {
  return bpe::Encoding::preset(name, std::filesystem::path(BPE_DATA_DIR));
}

} // namespace

TEST(Tiktoken, Fixtures) {
  if (!vocabs_present()) {
    GTEST_SKIP() << "run scripts/download_vocabs.py --dest data";
  }
  const std::filesystem::path path = std::filesystem::path(BPE_FIXTURE_DIR) / "tiktoken.tsv";
  ASSERT_TRUE(std::filesystem::exists(path));
  const bpe::Encoding gpt2 = load_preset("gpt2");
  const bpe::Encoding cl100k = load_preset("cl100k_base");
  std::ifstream in(path);
  std::string line;
  int checked = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const std::size_t tab1 = line.find('\t');
    const std::size_t tab2 = line.find('\t', tab1 + 1);
    ASSERT_NE(tab1, std::string::npos);
    ASSERT_NE(tab2, std::string::npos);
    const std::string model = line.substr(0, tab1);
    const std::string text = b64_decode(line.substr(tab1 + 1, tab2 - tab1 - 1));
    const std::vector<std::uint32_t> expect = parse_ids(line.substr(tab2 + 1));
    const bpe::Encoding& enc = model == "gpt2" ? gpt2 : cl100k;
    const auto got = enc.encode_ordinary(text);
    EXPECT_EQ(got, expect) << model << " text bytes " << text.size();
    EXPECT_EQ(enc.decode(got), text);
    ++checked;
  }
  EXPECT_GT(checked, 100);
}

TEST(Tiktoken, Specials) {
  if (!vocabs_present()) {
    GTEST_SKIP() << "run scripts/download_vocabs.py --dest data";
  }
  const bpe::Encoding gpt2 = load_preset("gpt2");
  const std::string eot = "<|endoftext|>";
  EXPECT_THROW((void)gpt2.encode(eot, {}), std::runtime_error);
  EXPECT_EQ(gpt2.encode(eot, {eot}), (std::vector<std::uint32_t>{50256}));
  const auto with = gpt2.encode_with_specials("A" + eot + "Z");
  ASSERT_GE(with.size(), 3u);
  EXPECT_EQ(with[1], 50256u);
  EXPECT_EQ(gpt2.decode(with), "A" + eot + "Z");
  const auto ordinary = gpt2.encode_ordinary(eot);
  EXPECT_GT(ordinary.size(), 1u);

  const bpe::Encoding cl = load_preset("cl100k_base");
  EXPECT_EQ(cl.encode("<|fim_prefix|>", {"<|fim_prefix|>"}), (std::vector<std::uint32_t>{100258}));
  EXPECT_EQ(cl.encode_with_specials("<|endofprompt|>"), (std::vector<std::uint32_t>{100276}));
}

TEST(Tiktoken, LowPlaneSweep) {
  if (!vocabs_present()) {
    GTEST_SKIP() << "run scripts/download_vocabs.py --dest data";
  }
  const bpe::Encoding gpt2 = load_preset("gpt2");
  const bpe::Encoding cl = load_preset("cl100k_base");
  // cp in [0, 2048). Same contexts and little-endian length/id framing as scripts that
  // hashed tiktoken 0.14.0.
  EXPECT_EQ(sweep(gpt2, 2048), "d3f22a56e4742e7df71b7212d39a12aafc7c3f5df166535081fa0a281189576c");
  EXPECT_EQ(sweep(cl, 2048), "6e5713ca68c56bd850bcdbebe24325b422495e8274ca144984440b21eb3994fb");
}

TEST(Tiktoken, FullSweep) {
#ifndef NDEBUG
  GTEST_SKIP() << "full scalar sweep runs in Release";
#else
  if (!vocabs_present()) {
    GTEST_SKIP() << "run scripts/download_vocabs.py --dest data";
  }
  const bpe::Encoding gpt2 = load_preset("gpt2");
  const bpe::Encoding cl = load_preset("cl100k_base");
  EXPECT_EQ(sweep(gpt2, 0x110000),
            "50cd292994a1c43cdaaeb0308dbc157fc52a5de6f7a4776dc374fd32c4441b6a");
  EXPECT_EQ(sweep(cl, 0x110000),
            "f3ca2e1af21688c61b7844e66a82dba8f7f91726ccc63256d9a550a6f55228ff");
#endif
}

// Silence an unused-function warning if a build defines nothing that calls utf8::valid.
TEST(Utf8, ValidHelperAcceptsAscii) {
  EXPECT_TRUE(bpe::utf8::valid("ascii"));
}
