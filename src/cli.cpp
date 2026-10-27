#include "bpe/encoding.hpp"
#include "utf8.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

void usage() {
  std::cerr << "usage:\n"
            << "  bpe encode --preset gpt2|cl100k_base --data DIR (--text STR | --file PATH)\n"
            << "             [--specials | --allow TOKEN]\n"
            << "  bpe decode --preset gpt2|cl100k_base --data DIR (--ids 1,2,3 | --file PATH)\n"
            << "  bpe train --input PATH --output PATH [--pattern gpt2|cl100k]\n"
            << "            [--vocab-size N] [--min-frequency N]\n"
            << "  bpe bench --preset gpt2|cl100k_base --data DIR --file PATH\n"
            << "            [--threads N] [--chunk N] [--repeats N]\n"
            << "\n"
            << "encode default is ordinary BPE (specials are plain text).\n"
            << "--specials emits every registered special as one id.\n"
            << "--allow TOKEN emits that special and rejects any other special text.\n"
            << "bench --chunk 0 encodes the file as one string. Otherwise the file is split into\n"
            << "UTF-8 chunks of at most N bytes, each ending on a code point boundary.\n";
}

struct Options {
  std::string command;
  std::string preset = "gpt2";
  std::string data = "data";
  std::string file;
  std::string text;
  std::string output;
  std::string pattern = "gpt2";
  std::string ids;
  std::vector<std::string> allow;
  bool specials = false;
  bool has_text = false;
  std::size_t vocab_size = 512;
  std::uint64_t min_frequency = 2;
  std::size_t threads = 1;
  std::size_t chunk = 0;
  int repeats = 5;
};

Options parse(int argc, char** argv) {
  if (argc < 2) {
    usage();
    throw std::runtime_error("missing command");
  }
  Options opt;
  opt.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--preset") {
      opt.preset = need("--preset");
    } else if (arg == "--data") {
      opt.data = need("--data");
    } else if (arg == "--file" || arg == "--input") {
      opt.file = need(arg == "--input" ? "--input" : "--file");
    } else if (arg == "--text") {
      opt.text = need("--text");
      opt.has_text = true;
    } else if (arg == "--output") {
      opt.output = need("--output");
    } else if (arg == "--pattern") {
      opt.pattern = need("--pattern");
    } else if (arg == "--ids") {
      opt.ids = need("--ids");
    } else if (arg == "--allow") {
      opt.allow.push_back(need("--allow"));
    } else if (arg == "--specials") {
      opt.specials = true;
    } else if (arg == "--vocab-size") {
      opt.vocab_size = static_cast<std::size_t>(std::stoull(need("--vocab-size")));
    } else if (arg == "--min-frequency") {
      opt.min_frequency = std::stoull(need("--min-frequency"));
    } else if (arg == "--threads") {
      opt.threads = static_cast<std::size_t>(std::stoull(need("--threads")));
    } else if (arg == "--chunk") {
      opt.chunk = static_cast<std::size_t>(std::stoull(need("--chunk")));
    } else if (arg == "--repeats") {
      opt.repeats = std::stoi(need("--repeats"));
    } else if (arg == "--help" || arg == "-h") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + std::string(arg));
    }
  }
  return opt;
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

bpe::Pattern pattern_from(const std::string& name) {
  if (name == "gpt2") {
    return bpe::Pattern::Gpt2;
  }
  if (name == "cl100k" || name == "cl100k_base") {
    return bpe::Pattern::Cl100k;
  }
  throw std::runtime_error("pattern must be gpt2 or cl100k");
}

std::vector<std::uint32_t> parse_ids(std::string_view text) {
  std::vector<std::uint32_t> ids;
  std::size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' ||
                               text[i] == '\t' || text[i] == ',')) {
      ++i;
    }
    if (i >= text.size()) {
      break;
    }
    if (text[i] < '0' || text[i] > '9') {
      throw std::runtime_error("bad token id");
    }
    std::uint64_t value = 0;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
      value = value * 10 + static_cast<std::uint64_t>(text[i] - '0');
      if (value > 0xFFFFFFFFULL) {
        throw std::runtime_error("token id overflow");
      }
      ++i;
    }
    ids.push_back(static_cast<std::uint32_t>(value));
  }
  return ids;
}

void print_ids(const std::vector<std::uint32_t>& ids) {
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) {
      std::cout << ' ';
    }
    std::cout << ids[i];
  }
  std::cout << '\n';
}

// Successive slices of at most `limit` bytes ending on a code point boundary.
// limit == 0 keeps the whole buffer as one slice. A code point longer than `limit`
// is emitted alone.
std::vector<std::string_view> chunk_text(std::string_view data, std::size_t limit) {
  if (limit == 0) {
    return {data};
  }
  std::vector<std::string_view> out;
  std::size_t i = 0;
  while (i < data.size()) {
    std::size_t j = i;
    std::size_t last = i;
    while (j < data.size()) {
      std::size_t next = j;
      std::uint32_t cp = 0;
      if (!bpe::utf8::next(data, next, cp)) {
        throw std::runtime_error("invalid UTF-8 in input");
      }
      if (next - i > limit && last > i) {
        break;
      }
      last = next;
      j = next;
      if (last - i >= limit) {
        break;
      }
    }
    out.emplace_back(data.data() + i, last - i);
    i = last;
  }
  return out;
}

void cmd_encode(const Options& opt) {
  const bpe::Encoding enc = bpe::Encoding::preset(opt.preset, opt.data);
  std::string owned;
  std::string_view text;
  if (opt.has_text) {
    text = opt.text;
  } else if (!opt.file.empty()) {
    owned = read_file(opt.file);
    text = owned;
  } else {
    owned.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
    text = owned;
  }
  std::vector<std::uint32_t> ids;
  if (opt.specials) {
    ids = enc.encode_with_specials(text);
  } else if (!opt.allow.empty()) {
    ids = enc.encode(text, std::unordered_set<std::string>(opt.allow.begin(), opt.allow.end()));
  } else {
    ids = enc.encode_ordinary(text);
  }
  print_ids(ids);
}

void cmd_decode(const Options& opt) {
  const bpe::Encoding enc = bpe::Encoding::preset(opt.preset, opt.data);
  std::string raw;
  if (!opt.ids.empty()) {
    raw = opt.ids;
  } else if (!opt.file.empty()) {
    raw = read_file(opt.file);
  } else {
    raw.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
  }
  const std::string bytes = enc.decode(parse_ids(raw));
  std::cout.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void cmd_train(const Options& opt) {
  if (opt.file.empty() || opt.output.empty()) {
    throw std::runtime_error("train requires --input/--file and --output");
  }
  bpe::TrainConfig config;
  config.vocab_size = opt.vocab_size;
  config.min_frequency = opt.min_frequency;
  config.pattern = pattern_from(opt.pattern);
  const std::string corpus = read_file(opt.file);
  const bpe::Encoding enc = bpe::train(corpus, config);
  enc.save_tiktoken(opt.output);
  std::cerr << "vocab_size " << enc.vocab_size() << "\n";
}

void cmd_bench(const Options& opt) {
  if (opt.file.empty()) {
    throw std::runtime_error("bench requires --file");
  }
  if (opt.repeats < 1) {
    throw std::runtime_error("--repeats must be >= 1");
  }
  const bpe::Encoding enc = bpe::Encoding::preset(opt.preset, opt.data);
  const std::string file = read_file(opt.file);
  const std::vector<std::string_view> chunks = chunk_text(file, opt.chunk);
  std::uint64_t bytes = 0;
  for (std::string_view chunk : chunks) {
    bytes += chunk.size();
  }
  auto run = [&]() {
    std::uint64_t tokens = 0;
    if (opt.threads <= 1) {
      for (std::string_view chunk : chunks) {
        tokens += enc.encode_ordinary(chunk).size();
      }
    } else {
      const auto batch = enc.encode_ordinary_batch(chunks, opt.threads);
      for (const auto& ids : batch) {
        tokens += ids.size();
      }
    }
    return tokens;
  };
  const std::uint64_t tokens = run();
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(opt.repeats));
  for (int i = 0; i < opt.repeats; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t check = run();
    const auto t1 = std::chrono::steady_clock::now();
    if (check != tokens) {
      throw std::runtime_error("bench token count changed between repeats");
    }
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  double median = samples[samples.size() / 2];
  if (samples.size() % 2 == 0) {
    median = 0.5 * (samples[samples.size() / 2 - 1] + samples[samples.size() / 2]);
  }
  const double mb_per_s = median > 0.0 ? (static_cast<double>(bytes) / median) / 1.0e6 : 0.0;
  std::cout << "bytes " << bytes << "\n";
  std::cout << "chunks " << chunks.size() << "\n";
  std::cout << "tokens " << tokens << "\n";
  std::cout << "threads " << opt.threads << "\n";
  std::cout << "chunk " << opt.chunk << "\n";
  std::cout << "repeats " << opt.repeats << "\n";
  std::cout.setf(std::ios::fixed);
  std::cout.precision(6);
  std::cout << "median_seconds " << median << "\n";
  std::cout.precision(3);
  std::cout << "mb_per_s " << mb_per_s << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse(argc, argv);
    if (opt.command == "encode") {
      cmd_encode(opt);
    } else if (opt.command == "decode") {
      cmd_decode(opt);
    } else if (opt.command == "train") {
      if (opt.file.empty() && !opt.text.empty()) {
        throw std::runtime_error("train reads --file");
      }
      cmd_train(opt);
    } else if (opt.command == "bench") {
      cmd_bench(opt);
    } else {
      usage();
      std::cerr << "unknown command\n";
      return 2;
    }
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
