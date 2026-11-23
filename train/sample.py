"""Print stories from a trained model, healthy and with random bit damage.

    uv run python sample.py bpe512-4m --flip 0 0.001 0.01 0.05

Every damage level starts from the same prompt and the same random seed, so
differences in the text come from the damage alone.
"""

from __future__ import annotations

import argparse

import torch

import checkpoint
from model import flip_bits
from vocab import EOT, Vocab

CHUNK = 32  # tokens generated between length checks


def story(vocab: Vocab, row: list[int]) -> str:
    """Text of the first story in `row`, which starts with the story marker."""
    body = row[1:]
    end = body.index(EOT) if EOT in body else len(body)
    return vocab.decode(body[:end])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="run directory under runs/")
    parser.add_argument("--flip", type=float, nargs="+", default=[0.0], help="fractions of bits")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--chars", type=int, default=400, help="approximate length of each sample")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, mps or cuda; default picks the best")
    args = parser.parse_args()

    device = args.device or checkpoint.pick_device()
    for fraction in args.flip:
        model, vocab = checkpoint.load(checkpoint.RUNS_DIR / args.name, device)
        if model.cfg.one_bit:  # a float twin keeps its real weights; a flip negates one
            model.harden()
        flipped = flip_bits(model, fraction, torch.Generator().manual_seed(args.seed))
        total, _ = model.count_weights()
        print(f"\n=== {args.name}: {flipped:,} of {total:,} bits flipped ({fraction:.2%}) ===")

        prompt = [EOT, *vocab.encode(args.prompt)]
        ids = torch.tensor([prompt] * args.samples, device=device)
        sampler = torch.Generator().manual_seed(args.seed)
        while min(len(story(vocab, row)) for row in ids.tolist()) < args.chars:
            ids = model.generate(ids, CHUNK, sampler, args.temperature)
            if all(EOT in row[1:] for row in ids.tolist()):
                break
        for row in ids.tolist():
            print("\n" + story(vocab, row)[: args.chars])


if __name__ == "__main__":
    main()
