"""Download TinyStories and encode it once per vocabulary.

    uv run python data.py

Writes data/<vocab>/{train,val}.bin (uint16 token ids) and vocab.json.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from vocab import EOT_TEXT, Vocab, char_vocab, clean, train_bpe

REPO = "roneneldan/TinyStories"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}
DATA_DIR = Path(__file__).parent / "data"
BPE_TRAINING_STORIES = 200_000
BATCH = 20_000


def read_stories(path: Path) -> Iterator[str]:
    """Yield cleaned, non-empty stories from a raw TinyStories text file."""
    pending = ""
    with path.open(encoding="utf-8", errors="ignore") as f:
        while chunk := f.read(1 << 24):
            *done, pending = (pending + chunk).split(EOT_TEXT)
            yield from filter(None, map(clean, done))
    if last := clean(pending):
        yield last


def batches(stories: Iterator[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for story in stories:
        batch.append(story)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_file(vocab: Vocab, raw: Path, out: Path) -> tuple[int, int]:
    """Encode one raw file to `out`. Returns (tokens, characters)."""
    tokens = chars = 0
    with out.open("wb") as f:
        for batch in batches(read_stories(raw), BATCH):
            ids = np.fromiter(
                (i for story in vocab.encode_stories(batch) for i in story), dtype=np.uint16
            )
            ids.tofile(f)
            tokens += len(ids)
            chars += sum(len(s) + 1 for s in batch)
    return tokens, chars


def build_vocab(name: str, raw_train: Path) -> Vocab:
    if name == "chars":
        return char_vocab()
    if name.startswith("bpe"):
        sample = itertools.islice(read_stories(raw_train), BPE_TRAINING_STORIES)
        return train_bpe(sample, int(name.removeprefix("bpe")))
    raise ValueError(f"unknown vocabulary {name!r}; use 'chars' or 'bpe<N>'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vocabs", nargs="+", default=["chars", "bpe512", "bpe4096"])
    args = parser.parse_args()

    raw = {
        split: Path(hf_hub_download(REPO, name, repo_type="dataset"))
        for split, name in FILES.items()
    }
    for name in args.vocabs:
        out_dir = DATA_DIR / name
        if (out_dir / "train.bin").exists():
            print(f"{name}: already built, skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        vocab = build_vocab(name, raw["train"])
        vocab.save(out_dir / "vocab.json")
        for split, path in raw.items():
            tmp = out_dir / f"{split}.bin.tmp"
            tokens, chars = encode_file(vocab, path, tmp)
            tmp.rename(out_dir / f"{split}.bin")
            print(f"{name} {split}: {tokens:,} tokens, {chars / tokens:.2f} chars per token")


if __name__ == "__main__":
    main()
