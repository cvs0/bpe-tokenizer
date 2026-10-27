#!/usr/bin/env python3
"""Write tests/fixtures/tiktoken.tsv from tiktoken 0.14.0 encode_ordinary."""

from __future__ import annotations

import argparse
import base64
import pathlib

import tiktoken


def cases() -> list[str]:
    samples = [
        "",
        " ",
        "  ",
        "\n",
        "\r\n",
        "\t",
        " \t \n",
        "hello",
        "hello world",
        "Hello, world!",
        "  leading",
        "trailing  ",
        "a  b",
        "a   b",
        "the   quick\n\nbrown",
        "it's a test, isn't it?",
        "I'm well; you're not.",
        "IT'S",
        "You'll",
        "DON'T",
        "foo's",
        "FOO'S",
        "'sa",
        "'LL",
        "'ſa",
        "123",
        "1234",
        "12345",
        "1234567890",
        "  123",
        "\t123",
        "price is $1,234.50!",
        "path/to/file_name-2.cpp",
        "def f(x):\n    return x + 1\n",
        "{\n  \"a\": [1, 2, 3]\n}\n",
        "https://example.com/a?b=1&c=2",
        "user@example.com",
        "emoji 😀 👨‍👩‍👧 🇺🇸",
        "🚀" * 8,
        "中文测试",
        "日本語のかな",
        "العربية",
        "עברית",
        "e\u0301",
        "A\u0301",
        "ﬁ ligature",
        "\u00a0nbsp",
        "\u2028",
        "\u2029",
        "\u3000ideographic",
        "\u208f\n",
        "\u209d\n",
        "\u200b",
        "\u200d",
        "\ufeff",
        "<|endoftext|>",
        "<|fim_prefix|>code<|fim_suffix|>",
        "end\n",
        "foo\n ",
        " \n ",
        "\n \nX",
        "  \n  X",
        "!\n\n",
        "a" * 300,
        " " * 200,
        "!" * 180,
        "\n" * 40,
        "q" * 180,
        "mix\r\nwindows\r\n",
        "line1\nline2\n\nline4",
        "()[]{}<>",
        "---",
        "...",
        "a\u00adsoft",
        "\u0000null",
        "byte \u007f del",
    ]
    # A short multilingual paragraph and a code block.
    samples.append(
        "The quick brown fox jumps over the lazy dog.\n"
        "Portez ce vieux whisky au juge blond qui fume.\n"
        "Съешь же ещё этих мягких французских булок.\n"
    )
    samples.append(
        "int main() {\n"
        "  const char* s = \"hi\\n\";\n"
        "  return s[0] == 'h' ? 0 : 1;\n"
        "}\n"
    )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tiktoken.tsv",
    )
    args = parser.parse_args()
    lines = ["# tiktoken 0.14.0 encode_ordinary"]
    for name in ("gpt2", "cl100k_base"):
        enc = tiktoken.get_encoding(name)
        for text in cases():
            ids = enc.encode_ordinary(text)
            blob = base64.b64encode(text.encode("utf-8")).decode("ascii")
            body = ",".join(str(i) for i in ids)
            lines.append(f"{name}\t{blob}\t{body}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output} ({len(lines) - 1} cases)")


if __name__ == "__main__":
    main()
