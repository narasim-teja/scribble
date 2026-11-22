"""Saving and loading a trained model together with its vocabulary."""

from __future__ import annotations

from pathlib import Path

import torch

from model import GPT, Config
from vocab import Vocab

RUNS_DIR = Path(__file__).parent / "runs"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save(run_dir: Path, model: GPT, vocab: Vocab) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(run_dir / "vocab.json")
    payload = {"config": model.cfg.to_dict(), "state": model.state_dict()}
    tmp = run_dir / "model.pt.tmp"
    torch.save(payload, tmp)
    tmp.rename(run_dir / "model.pt")


def load(run_dir: Path, device: torch.device) -> tuple[GPT, Vocab]:
    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model = GPT(Config(**payload["config"]))
    model.load_state_dict(payload["state"])
    return model.to(device).eval(), Vocab.load(run_dir / "vocab.json")
