"""Measure how a model's score falls as more of its weights are damaged.

    uv run python curve.py bpe512-16m bpe512-16m-float

For each share of weights, negate that share at random (a bit flip for a
one-bit model, a sign change for a float one), measure the loss on held-out
stories, and repeat with a few different random choices. Prints one line of
JSON per model and share, so one-bit and float models can be compared.
"""

from __future__ import annotations

import argparse
import json

import torch

import checkpoint
from data import DATA_DIR
from model import flip_bits
from train import Batches, evaluate

SHARES = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
EVAL_BATCHES = 8
EVAL_BATCH_SIZE = 32


def score(name: str, share: float, seed: int, device: torch.device) -> float:
    """Bits per character on held-out stories with `share` of the weights negated."""
    model, vocab = checkpoint.load(checkpoint.RUNS_DIR / name, device)
    if model.cfg.one_bit:
        model.harden()
    flip_bits(model, share, torch.Generator().manual_seed(seed))
    source = Batches(
        DATA_DIR / vocab.name / "val.bin", model.cfg.context, EVAL_BATCH_SIZE, device, seed=1234
    )
    val = [source.next() for _ in range(EVAL_BATCHES)]
    char_lengths = torch.from_numpy(vocab.char_lengths()).to(device)
    return evaluate(model, val, char_lengths)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", help="run directories under runs/")
    parser.add_argument("--seeds", type=int, default=3, help="random choices per share")
    parser.add_argument("--device", default=None, help="cpu, mps or cuda; default picks the best")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else checkpoint.pick_device()
    for name in args.names:
        for share in SHARES:
            scores = [
                score(name, share, seed, device) for seed in range(1 if share == 0 else args.seeds)
            ]
            line = {"run": name, "share": share, "bits_per_char": [round(s, 4) for s in scores]}
            print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
