#include "bpe/encoding.hpp"

#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

bpe::Pattern pattern_from(const std::string& name) {
  if (name == "gpt2") {
    return bpe::Pattern::Gpt2;
  }
  if (name == "cl100k" || name == "cl100k_base") {
    return bpe::Pattern::Cl100k;
  }
  throw std::invalid_argument("pattern must be gpt2 or cl100k");
}

} // namespace

PYBIND11_MODULE(bpe, m) {
  m.doc() = "Byte-level BPE compatible with tiktoken gpt2 and cl100k_base.";

  py::enum_<bpe::Pattern>(m, "Pattern")
      .value("gpt2", bpe::Pattern::Gpt2)
      .value("cl100k", bpe::Pattern::Cl100k);

  py::class_<bpe::SpecialToken>(m, "SpecialToken")
      .def(py::init<>())
      .def_readwrite("text", &bpe::SpecialToken::text)
      .def_readwrite("id", &bpe::SpecialToken::id);

  py::class_<bpe::Encoding>(m, "Encoding")
      .def_static("preset", &bpe::Encoding::preset, py::arg("name"), py::arg("directory"))
      .def_static("from_gpt2_files",
                  &bpe::Encoding::from_gpt2_files,
                  py::arg("encoder_json"),
                  py::arg("vocab_bpe"))
      .def_static(
          "from_tiktoken",
          [](const std::string& path, const std::string& pattern) {
            return bpe::Encoding::from_tiktoken(path, pattern_from(pattern));
          },
          py::arg("path"),
          py::arg("pattern"))
      .def_static("load", &bpe::Encoding::load, py::arg("path"))
      .def("save_tiktoken", &bpe::Encoding::save_tiktoken, py::arg("path"))
      .def("name", &bpe::Encoding::name)
      .def("vocab_size", &bpe::Encoding::vocab_size)
      .def("max_token_id", &bpe::Encoding::max_token_id)
      .def(
          "encode_ordinary",
          [](const bpe::Encoding& enc, const std::string& text) {
            py::gil_scoped_release release;
            return enc.encode_ordinary(text);
          },
          py::arg("text"))
      .def(
          "encode",
          [](const bpe::Encoding& enc,
             const std::string& text,
             const std::unordered_set<std::string>& allowed) {
            py::gil_scoped_release release;
            return enc.encode(text, allowed);
          },
          py::arg("text"),
          py::arg("allowed"))
      .def(
          "encode_with_specials",
          [](const bpe::Encoding& enc, const std::string& text) {
            py::gil_scoped_release release;
            return enc.encode_with_specials(text);
          },
          py::arg("text"))
      .def(
          "encode_ordinary_batch",
          [](const bpe::Encoding& enc,
             const std::vector<std::string>& texts,
             std::size_t num_threads) {
            std::vector<std::string_view> views;
            views.reserve(texts.size());
            for (const std::string& text : texts) {
              views.emplace_back(text);
            }
            py::gil_scoped_release release;
            return enc.encode_ordinary_batch(views, num_threads);
          },
          py::arg("texts"),
          py::arg("num_threads") = 0)
      .def(
          "decode",
          [](const bpe::Encoding& enc, const std::vector<std::uint32_t>& ids) {
            return py::bytes(enc.decode(ids));
          },
          py::arg("ids"))
      .def(
          "token_bytes",
          [](const bpe::Encoding& enc, std::uint32_t id) { return py::bytes(enc.token_bytes(id)); },
          py::arg("id"));

  m.def(
      "train",
      [](const std::string& corpus,
         std::size_t vocab_size,
         const std::string& pattern,
         std::uint64_t min_frequency) {
        bpe::TrainConfig config;
        config.vocab_size = vocab_size;
        config.pattern = pattern_from(pattern);
        config.min_frequency = min_frequency;
        py::gil_scoped_release release;
        return bpe::train(corpus, config);
      },
      py::arg("corpus"),
      py::arg("vocab_size") = 512,
      py::arg("pattern") = "gpt2",
      py::arg("min_frequency") = 2);
}
