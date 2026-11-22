"""Write the answers the browser engine must reproduce.

    uv run python reference.py bpe512-4m          # from an exported run
    uv run python reference.py --tiny ../web/tests/fixtures

Loads a model from its picture, feeds it a fixed run of tokens and saves the
logits at every position. The browser engine reads the same picture and must
match these within 1e-3 (milestone M3). `--tiny` first builds a small
untrained model and exports it, giving a fixture small enough to commit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import checkpoint
import export
from model import GPT, Config
from vocab import EOT, char_vocab

TOKENS = 48
TINY = Config(vocab_size=char_vocab().size, context=64, layers=2, heads=2, width=32)


@torch.no_grad()
def reference(picture: Path, seed: int = 0) -> dict:
    """Tokens of a seeded sample from the model in `picture`, and its logits at each one."""
    model, _ = export.load(picture)
    sampler = torch.Generator().manual_seed(seed)
    count = min(TOKENS, model.cfg.context)
    ids = model.generate(torch.tensor([[EOT]]), count - 1, sampler)
    logits = model(ids)[0]
    rounded = [[round(v, 6) for v in row] for row in logits.tolist()]
    return {"tokens": ids[0].tolist(), "logits": rounded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", help="run directory under runs/, already exported")
    parser.add_argument("--tiny", type=Path, help="write a tiny model and its answers here")
    args = parser.parse_args()

    if args.tiny:
        torch.manual_seed(0)
        args.tiny.mkdir(parents=True, exist_ok=True)
        picture = args.tiny / "tiny.png"
        export.export(GPT(TINY).eval(), char_vocab(), picture)
    elif args.name:
        picture = checkpoint.RUNS_DIR / args.name / "brain.png"
    else:
        parser.error("give a run name or --tiny")
    out = picture.with_suffix(".reference.json")
    out.write_text(json.dumps(reference(picture)))
    print(f"{out}: {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
