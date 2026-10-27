#include "bpe/encoding.hpp"
#include "sha256.hpp"
#include "unicode.hpp"

#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

bpe::Encoding toy_gpt2() {
  std::vector<std::pair<std::string, std::uint32_t>> tokens;
  tokens.reserve(260);
  for (int byte = 0; byte < 256; ++byte) {
    tokens.emplace_back(std::string(1, static_cast<char>(byte)), static_cast<std::uint32_t>(byte));
  }
  tokens.emplace_back("aa", 256);
  tokens.emplace_back("ab", 257);
  tokens.emplace_back("abc", 258);
  tokens.emplace_back("bc", 300);
  return bpe::Encoding::from_tokens(bpe::Pattern::Gpt2, "toy", std::move(tokens));
}

} // namespace

TEST(Sha256, Abc) {
  bpe::test::Sha256 hash;
  hash.update("abc", 3);
  EXPECT_EQ(hash.hex(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(Unicode, Classes) {
  EXPECT_TRUE(bpe::unicode::flags('A') & bpe::unicode::kLetter);
  EXPECT_TRUE(bpe::unicode::flags('0') & bpe::unicode::kNumber);
  EXPECT_TRUE(bpe::unicode::flags(' ') & bpe::unicode::kSpace);
  EXPECT_TRUE(bpe::unicode::flags('\n') & bpe::unicode::kSpace);
  EXPECT_TRUE(bpe::unicode::flags(0x4E2D) & bpe::unicode::kLetter);
  EXPECT_TRUE(bpe::unicode::flags(0x208F) & bpe::unicode::kLetter);
  EXPECT_FALSE(bpe::unicode::flags('!') &
               (bpe::unicode::kLetter | bpe::unicode::kNumber | bpe::unicode::kSpace));
  const std::uint32_t spaces[] = {0x9,    0xA,    0xB,    0xC,    0xD,    0x20,   0x85,
                                  0xA0,   0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004,
                                  0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028,
                                  0x2029, 0x202F, 0x205F, 0x3000};
  for (std::uint32_t cp : spaces) {
    EXPECT_EQ(bpe::unicode::flags(cp), bpe::unicode::kSpace) << std::hex << cp;
  }
}

TEST(Merge, ToyRanksMatchTiktokenOrder) {
  const bpe::Encoding enc = toy_gpt2();
  EXPECT_EQ(enc.encode_ordinary("aaa"), (std::vector<std::uint32_t>{256, 97}));
  EXPECT_EQ(enc.encode_ordinary("aaaa"), (std::vector<std::uint32_t>{256, 256}));
  EXPECT_EQ(enc.encode_ordinary("abc"), (std::vector<std::uint32_t>{258}));
  EXPECT_EQ(enc.encode_ordinary("abx"), (std::vector<std::uint32_t>{257, 120}));
  EXPECT_EQ(enc.encode_ordinary("bc"), (std::vector<std::uint32_t>{300}));
  EXPECT_EQ(enc.encode_ordinary("abcd"), (std::vector<std::uint32_t>{258, 100}));
  EXPECT_EQ(enc.encode_ordinary("aa"), (std::vector<std::uint32_t>{256}));
  std::string long_run(200, 'a');
  const auto ids = enc.encode_ordinary(long_run);
  ASSERT_EQ(ids.size(), 100u);
  for (std::uint32_t id : ids) {
    EXPECT_EQ(id, 256u);
  }
  EXPECT_EQ(enc.decode(ids), long_run);
}

TEST(Merge, Cl100kDigitChunks) {
  std::vector<std::pair<std::string, std::uint32_t>> tokens;
  for (int byte = 0; byte < 256; ++byte) {
    tokens.emplace_back(std::string(1, static_cast<char>(byte)), static_cast<std::uint32_t>(byte));
  }
  tokens.emplace_back("123", 400);
  const bpe::Encoding enc =
      bpe::Encoding::from_tokens(bpe::Pattern::Cl100k, "digits", std::move(tokens));
  EXPECT_EQ(enc.encode_ordinary("1234"), (std::vector<std::uint32_t>{400, '4'}));
  EXPECT_EQ(enc.encode_ordinary("1234567"), (std::vector<std::uint32_t>{400, '4', '5', '6', '7'}));
}

TEST(Encode, RejectsInvalidUtf8) {
  const bpe::Encoding enc = toy_gpt2();
  EXPECT_THROW((void)enc.encode_ordinary("\xff"), std::invalid_argument);
  EXPECT_THROW((void)enc.encode_ordinary("\xc0\x80"), std::invalid_argument);
  EXPECT_THROW((void)enc.encode_ordinary("\xed\xa0\x80"), std::invalid_argument);
  const std::string truncated = "\xe4\xb8";
  EXPECT_THROW((void)enc.encode_ordinary(truncated), std::invalid_argument);
}

TEST(Encode, RoundTripAndBatch) {
  const bpe::Encoding enc = toy_gpt2();
  const std::vector<std::string> texts = {"", "Hello", "a  b", "it's", "\n\n", "e\u0301", "中文"};
  std::vector<std::string_view> views;
  for (const std::string& text : texts) {
    views.emplace_back(text);
    EXPECT_EQ(enc.decode(enc.encode_ordinary(text)), text);
  }
  const auto batch = enc.encode_ordinary_batch(views, 3);
  ASSERT_EQ(batch.size(), texts.size());
  for (std::size_t i = 0; i < texts.size(); ++i) {
    EXPECT_EQ(batch[i], enc.encode_ordinary(texts[i]));
  }
}

TEST(Train, FrequencyAndTieBreak) {
  bpe::TrainConfig config;
  config.pattern = bpe::Pattern::Gpt2;
  config.vocab_size = 257;
  config.min_frequency = 2;
  const bpe::Encoding enc = bpe::train("abc abc ab", config);
  EXPECT_EQ(enc.token_bytes(256), "ab");
  EXPECT_EQ(enc.encode_ordinary("abc"), (std::vector<std::uint32_t>{256, 'c'}));
  EXPECT_EQ(enc.decode(enc.encode_ordinary("abc abc ab")), "abc abc ab");

  config.vocab_size = 400;
  config.min_frequency = 2;
  const bpe::Encoding rare = bpe::train("ab cd", config);
  EXPECT_EQ(rare.vocab_size(), 256u);

  // Newlines are their own pieces, so (a,b) and (c,d) both have count 2.
  // Smaller left id wins: 'a' before 'c'.
  config.vocab_size = 257;
  config.min_frequency = 1;
  const bpe::Encoding tie = bpe::train("ab\ncd\nab\ncd", config);
  EXPECT_EQ(tie.token_bytes(256), "ab");

  config.vocab_size = 258;
  config.min_frequency = 2;
  const bpe::Encoding overlap = bpe::train("aaa aaa", config);
  EXPECT_EQ(overlap.token_bytes(256), "aa");
  EXPECT_EQ(overlap.token_bytes(257), "aaa");
  EXPECT_EQ(overlap.encode_ordinary("aaa"), (std::vector<std::uint32_t>{257}));

  const auto path = std::filesystem::temp_directory_path() / "bpe-train-roundtrip.tiktoken";
  overlap.save_tiktoken(path);
  const bpe::Encoding loaded = bpe::Encoding::load(path);
  EXPECT_EQ(loaded.encode_ordinary("aaa aaa"), overlap.encode_ordinary("aaa aaa"));
  EXPECT_EQ(loaded.decode(loaded.encode_ordinary("aaa")), "aaa");
  std::filesystem::remove(path);
}
