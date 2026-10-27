#!/usr/bin/env python3
"""Download GPT-2 and cl100k_base vocabularies and check sha256."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import urllib.request

FILES = {
    "vocab.bpe": (
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    ),
    "encoder.json": (
        "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    ),
    "cl100k_base.tiktoken": (
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    ),
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=pathlib.Path, default=pathlib.Path("data"))
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        path = args.dest / name
        if path.exists() and sha256(path) == expected:
            print(f"ok {path}")
            continue
        print(f"get {url}")
        urllib.request.urlretrieve(url, path)
        actual = sha256(path)
        if actual != expected:
            path.unlink(missing_ok=True)
            raise SystemExit(f"hash mismatch for {name}: {actual} != {expected}")
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
