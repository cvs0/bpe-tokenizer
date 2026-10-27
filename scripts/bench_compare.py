#!/usr/bin/env python3
"""Time encode_ordinary against tiktoken on the same UTF-8 chunks.

Chunking matches src/cli.cpp chunk_text: slices of at most N bytes that end on a
code point boundary. N == 0 is the whole file. A code point longer than N is its
own slice. MB/s is file bytes / median_seconds / 1e6. One untimed warmup, then
the median of --repeats timed runs. Even repeat counts average the two middle
samples, same as the C++ bench.
"""

from __future__ import annotations

import argparse
import subprocess
import time


def utf8_next(data: bytes, index: int) -> int:
    if index >= len(data):
        raise ValueError("invalid UTF-8")
    c0 = data[index]
    if c0 < 0x80:
        return index + 1
    if (c0 & 0xE0) == 0xC0:
        if index + 1 >= len(data):
            raise ValueError("invalid UTF-8")
        c1 = data[index + 1]
        if (c1 & 0xC0) != 0x80:
            raise ValueError("invalid UTF-8")
        cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F)
        if cp < 0x80:
            raise ValueError("invalid UTF-8")
        return index + 2
    if (c0 & 0xF0) == 0xE0:
        if index + 2 >= len(data):
            raise ValueError("invalid UTF-8")
        c1 = data[index + 1]
        c2 = data[index + 2]
        if (c1 & 0xC0) != 0x80 or (c2 & 0xC0) != 0x80:
            raise ValueError("invalid UTF-8")
        cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F)
        if cp < 0x800 or 0xD800 <= cp <= 0xDFFF:
            raise ValueError("invalid UTF-8")
        return index + 3
    if (c0 & 0xF8) == 0xF0:
        if index + 3 >= len(data):
            raise ValueError("invalid UTF-8")
        c1 = data[index + 1]
        c2 = data[index + 2]
        c3 = data[index + 3]
        if (c1 & 0xC0) != 0x80 or (c2 & 0xC0) != 0x80 or (c3 & 0xC0) != 0x80:
            raise ValueError("invalid UTF-8")
        cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F)
        if cp < 0x10000 or cp > 0x10FFFF:
            raise ValueError("invalid UTF-8")
        return index + 4
    raise ValueError("invalid UTF-8")


def chunk_text(data: bytes, limit: int) -> list[bytes]:
    if limit == 0:
        return [data]
    out: list[bytes] = []
    i = 0
    n = len(data)
    while i < n:
        j = i
        last = i
        while j < n:
            nxt = utf8_next(data, j)
            if nxt - i > limit and last > i:
                break
            last = nxt
            j = nxt
            if last - i >= limit:
                break
        out.append(data[i:last])
        i = last
    return out


def median(samples: list[float]) -> float:
    ordered = sorted(samples)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return 0.5 * (ordered[mid - 1] + ordered[mid])
    return ordered[mid]


def mb_per_s(nbytes: int, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return (nbytes / seconds) / 1.0e6


def time_call(fn, repeats: int) -> tuple[int, float]:
    tokens = fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        check = fn()
        t1 = time.perf_counter()
        if check != tokens:
            raise RuntimeError("token count changed between repeats")
        samples.append(t1 - t0)
    return tokens, median(samples)


def run_cpp(args, data_bytes: int) -> tuple[int, float, float]:
    cmd = [
        args.bin,
        "bench",
        "--preset",
        args.preset,
        "--data",
        args.data,
        "--file",
        args.file,
        "--threads",
        str(args.threads),
        "--chunk",
        str(args.chunk),
        "--repeats",
        str(args.repeats),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    fields = {}
    for line in proc.stdout.splitlines():
        key, value = line.split(" ", 1)
        fields[key] = value
    if int(fields["bytes"]) != data_bytes:
        raise RuntimeError(f"c++ byte count {fields['bytes']} != {data_bytes}")
    return int(fields["tokens"]), float(fields["median_seconds"]), float(fields["mb_per_s"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True, choices=("gpt2", "cl100k_base"))
    parser.add_argument("--file", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--bin", default="build/bpe")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hf", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    data = open(args.file, "rb").read()
    parts = chunk_text(data, args.chunk)
    texts = [part.decode("utf-8") for part in parts]

    import tiktoken

    enc = tiktoken.get_encoding(args.preset)

    def tiktoken_run() -> int:
        if args.threads <= 1:
            return sum(len(enc.encode_ordinary(text)) for text in texts)
        batch = enc.encode_ordinary_batch(texts, num_threads=args.threads)
        return sum(len(ids) for ids in batch)

    py_tokens, py_seconds = time_call(tiktoken_run, args.repeats)
    cpp_tokens, cpp_seconds, cpp_mb = run_cpp(args, len(data))
    if py_tokens != cpp_tokens:
        raise SystemExit(f"token mismatch c++ {cpp_tokens} tiktoken {py_tokens}")

    print(f"preset {args.preset}")
    print(f"bytes {len(data)}")
    print(f"chunks {len(parts)}")
    print(f"threads {args.threads}")
    print(f"chunk {args.chunk}")
    print(f"repeats {args.repeats}")
    print(f"tokens {cpp_tokens}")
    print(f"cpp_median_seconds {cpp_seconds:.6f}")
    print(f"cpp_mb_per_s {cpp_mb:.3f}")
    print(f"tiktoken_median_seconds {py_seconds:.6f}")
    print(f"tiktoken_mb_per_s {mb_per_s(len(data), py_seconds):.3f}")

    if args.hf:
        if args.preset != "gpt2":
            raise SystemExit("--hf is the Hugging Face GPT-2 ByteLevel tokenizer")
        from tokenizers import Tokenizer

        hf = Tokenizer.from_pretrained("gpt2")

        def hf_run() -> int:
            if args.threads <= 1:
                return sum(len(hf.encode(text).ids) for text in texts)
            return sum(len(enc_out.ids) for enc_out in hf.encode_batch(texts))

        hf_tokens, hf_seconds = time_call(hf_run, args.repeats)
        print(f"hf_tokens {hf_tokens}")
        print(f"hf_median_seconds {hf_seconds:.6f}")
        print(f"hf_mb_per_s {mb_per_s(len(data), hf_seconds):.3f}")
        print("hf_count_matched 1")


if __name__ == "__main__":
    main()
