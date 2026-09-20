#!/usr/bin/env python3
"""Verify and extract UAV-Flow depth tar.zst shards."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import zstandard as zstd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(
        (args.dataset_root / "metadata" / "shards.json").read_text()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for number, record in enumerate(index["shards"], 1):
        path = args.dataset_root / "data" / record["name"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {path}")
        print(f"[{number}/{len(index['shards'])}] extracting {path.name}")
        with path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    archive.extractall(args.output, filter="data")
    metadata_out = args.output / "metadata"
    metadata_out.mkdir(exist_ok=True)
    for name in (
        "episodes.csv", "source_summary.json", "instruction_overrides.json",
        "shards.json",
    ):
        source = args.dataset_root / "metadata" / name
        (metadata_out / name).write_bytes(source.read_bytes())
    (args.output / "_SUCCESS").write_text(
        f"{index['episodes']} episodes extracted\n"
    )


if __name__ == "__main__":
    main()
