"""Train one model for a fixed amount of wall-clock time.

    uv run python train.py --vocab bpe512 --minutes 120 --name bpe512-4m

Reads data/<vocab>/, writes runs/<name>/ (model.pt, vocab.json, log.jsonl).
The learning-rate schedule follows elapsed time, not step count, so runs with
different speeds are comparable at the same budget.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import checkpoint
from data import DATA_DIR
from model import GPT, Config
from vocab import Vocab

WARMUP_STEPS = 300
FINAL_LR_FRACTION = 0.1
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_BATCHES = 20
LN2 = math.log(2)


@dataclass(frozen=True)
class Schedule:
    """Learning rate and weight decay as functions of progress through the run."""

    peak_lr: float
    one_bit: bool

    def lr(self, step: int, progress: float) -> float:
        if step < WARMUP_STEPS:
            return self.peak_lr * (step + 1) / WARMUP_STEPS
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return self.peak_lr * (FINAL_LR_FRACTION + (1 - FINAL_LR_FRACTION) * cosine)

    def weight_decay(self, progress: float) -> float:
        # One-bit training: decay for the first half only (BitNet training notes).
        return 0.0 if self.one_bit and progress >= 0.5 else WEIGHT_DECAY


class Batches:
    """Random windows from an encoded split, as (inputs, targets) on the device."""

    def __init__(self, path: Path, context: int, batch: int, device: torch.device, seed: int):
        self.ids = np.memmap(path, dtype=np.uint16, mode="r")
        self.context, self.batch, self.device = context, batch, device
        self.rng = np.random.default_rng(seed)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.ids) - self.context - 1, self.batch)
        windows = np.stack([self.ids[s : s + self.context + 1] for s in starts]).astype(np.int64)
        block = torch.from_numpy(windows).to(self.device)
        return block[:, :-1], block[:, 1:]


@torch.no_grad()
def evaluate(model: GPT, val: list[tuple[torch.Tensor, torch.Tensor]], char_lengths: torch.Tensor):
    """Return (nats per token, bits per character) on fixed validation batches."""
    model.eval()
    nats = tokens = chars = 0.0
    for x, y in val:
        logits = model(x)
        nats += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
        tokens += y.numel()
        chars += char_lengths[y].sum().item()
    model.train()
    return nats / tokens, nats / chars / LN2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vocab", required=True, help="chars, bpe512, ...")
    parser.add_argument("--name", required=True, help="run directory under runs/")
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--float", action="store_true", help="full-precision weights")
    parser.add_argument("--ternary", action="store_true", help="weights of -1, 0 or +1")
    parser.add_argument("--eval-every", type=int, default=500, help="steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = checkpoint.pick_device()
    data_dir = DATA_DIR / args.vocab
    vocab = Vocab.load(data_dir / "vocab.json")
    cfg = Config(
        vocab.size, args.context, args.layers, args.heads, args.width, not args.float, args.ternary
    )
    model = GPT(cfg).to(device)
    linear, other = model.count_weights()
    print(f"{args.name}: {linear:,} linear weights, {other:,} other, on {device}")

    train = Batches(data_dir / "train.bin", cfg.context, args.batch, device, args.seed)
    val_source = Batches(data_dir / "val.bin", cfg.context, args.batch, device, seed=1234)
    val = [val_source.next() for _ in range(EVAL_BATCHES)]
    char_lengths = torch.from_numpy(vocab.char_lengths()).to(device)

    matrices = [layer.weight for layer in model.bit_layers()]
    rest = [p for p in model.parameters() if all(p is not m for m in matrices)]
    optimizer = torch.optim.AdamW(
        [{"params": matrices, "weight_decay": WEIGHT_DECAY}, {"params": rest, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    schedule = Schedule(args.lr, cfg.one_bit)

    run_dir = checkpoint.RUNS_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "log.jsonl").open("w")
    budget = args.minutes * 60
    start = time.monotonic()
    step = 0
    while True:
        elapsed = time.monotonic() - start
        progress = elapsed / budget
        done = progress >= 1.0
        if step % args.eval_every == 0 or done:
            per_token, bits_per_char = evaluate(model, val, char_lengths)
            record = {
                "step": step,
                "minutes": round(elapsed / 60, 2),
                "val_nats_per_token": round(per_token, 4),
                "val_bits_per_char": round(bits_per_char, 4),
                "tokens_per_second": round(step * args.batch * cfg.context / max(elapsed, 1e-9)),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(record, flush=True)
            checkpoint.save(run_dir, model, vocab)
        if done:
            break

        optimizer.param_groups[0]["weight_decay"] = schedule.weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = schedule.lr(step, progress)
        x, y = train.next()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        step += 1


if __name__ == "__main__":
    main()
