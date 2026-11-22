"""Damage one region of a trained model at a time and measure what breaks.

    uv run python lesion.py chars-4m --device cpu

A region is one 256-wide square of the weight picture: the query, key and
value parts of attention, the attention output, and the four quarters of each
MLP matrix. Each region in turn has its bits randomized (what the brush does),
the loss on held-out stories is measured, and the weights are restored. This
answers whether it matters where a visitor paints.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass

import torch

import checkpoint
from data import DATA_DIR
from model import GPT, BitLinear
from sample import story
from train import Batches, evaluate
from vocab import EOT

EVAL_BATCHES = 4
EVAL_BATCH_SIZE = 32


@dataclass(frozen=True)
class Region:
    """A rectangle of one weight matrix, named the way the picture labels it."""

    name: str
    layer: BitLinear
    rows: slice
    cols: slice

    def weights(self) -> torch.Tensor:
        return self.layer.weight[self.rows, self.cols]


def regions(model: GPT) -> Iterator[Region]:
    width = model.cfg.width
    everything = slice(None)

    def quarter(i: int) -> slice:
        return slice(i * width, (i + 1) * width)

    for number, block in enumerate(model.blocks, start=1):
        attention, mlp = block.attention, block.mlp
        for i, part in enumerate(("query", "key", "value")):
            yield Region(f"layer {number} {part}", attention.qkv, quarter(i), everything)
        yield Region(f"layer {number} attn out", attention.out, everything, everything)
        for i in range(4):
            yield Region(f"layer {number} mlp in {i + 1}", mlp.up, quarter(i), everything)
        for i in range(4):
            yield Region(f"layer {number} mlp out {i + 1}", mlp.down, everything, quarter(i))


@torch.no_grad()
def randomize(region: Region, fraction: float, generator: torch.Generator) -> torch.Tensor:
    """Give a random `fraction` of the region's weights a random sign. Returns the original."""
    original = region.weights().clone()
    shape = original.shape
    chosen = torch.rand(shape, generator=generator) < fraction
    flip = torch.rand(shape, generator=generator) < 0.5
    signs = torch.where(chosen & flip, -1.0, 1.0).to(original.device)
    region.layer.weight[region.rows, region.cols] = original * signs
    return original


@torch.no_grad()
def restore(region: Region, original: torch.Tensor) -> None:
    region.layer.weight[region.rows, region.cols] = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="run directory under runs/")
    parser.add_argument("--fraction", type=float, default=1.0, help="share of each region painted")
    parser.add_argument("--device", default=None, help="cpu, mps or cuda; default picks the best")
    parser.add_argument("--chars", type=int, default=90, help="length of the sample per region")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else checkpoint.pick_device()
    model, vocab = checkpoint.load(checkpoint.RUNS_DIR / args.name, device)
    model.harden()
    source = Batches(
        DATA_DIR / vocab.name / "val.bin", model.cfg.context, EVAL_BATCH_SIZE, device, seed=1234
    )
    val = [source.next() for _ in range(EVAL_BATCHES)]
    char_lengths = torch.from_numpy(vocab.char_lengths()).to(device)
    prompt = torch.tensor([[EOT, *vocab.encode("Once upon a time")]], device=device)

    def measure() -> tuple[float, str]:
        _, bits_per_char = evaluate(model, val, char_lengths)
        model.eval()
        sampler = torch.Generator().manual_seed(args.seed)
        ids = model.generate(prompt, args.chars, sampler)
        return bits_per_char, story(vocab, ids[0].tolist())[: args.chars].replace("\n", " ")

    healthy, text = measure()
    print(f"{'healthy':<22} {healthy:5.2f} bits/char          {text}")
    results = []
    for region in regions(model):
        original = randomize(region, args.fraction, torch.Generator().manual_seed(args.seed))
        loss, text = measure()
        restore(region, original)
        results.append((loss, region.name, text))
    for loss, name, text in sorted(results, reverse=True):
        print(f"{name:<22} {loss:5.2f} bits/char  +{loss - healthy:5.2f}  {text}")


if __name__ == "__main__":
    main()
