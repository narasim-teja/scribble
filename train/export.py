"""Write a trained one-bit model as a picture, and read it back.

    uv run python export.py bpe512-4m

The picture is a square one-bit PNG. Every pixel inside a piece is one weight:
white is +scale, black is -scale. The picture is cut into squares one model
width on a side, twelve per layer, in the order the model computes them. A
manifest (text chunk) says which weights each square holds, and a side table
(private chunk) carries the few numbers that are not one bit: embeddings and
normalization gains, as 16-bit floats to keep the file small. `load` rebuilds
the model from the file alone, which is the proof that the picture is the
model: the model the page runs is the one `load` returns.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import checkpoint
from model import GPT, Config
from vocab import Vocab

FORMAT_VERSION = 2
SIDE_DTYPE = "<f2"  # little-endian 16-bit floats
SIGNATURE = b"\x89PNG\r\n\x1a\n"
MANIFEST_KEY = b"scribble"
SIDE_CHUNK = b"sbTb"  # ancillary, private, safe to copy
SQUARES_PER_LAYER = 12


@dataclass(frozen=True)
class Piece:
    """One square of the picture: a rectangle of one weight matrix.

    `rows` and `cols` are half-open ranges into the matrix; `x` and `y` are
    the square's top-left pixel. Matrix rows always run along image rows.
    """

    tensor: str
    label: str
    layer: int
    rows: tuple[int, int]
    cols: tuple[int, int]
    x: int
    y: int


def picture_side(cfg: Config) -> int:
    """Smallest power-of-two side, a multiple of the width, that holds every square."""
    squares = SQUARES_PER_LAYER * cfg.layers
    side = cfg.width
    while (side // cfg.width) ** 2 < squares:
        side *= 2
    if side & (side - 1):
        raise ValueError(f"width {cfg.width} must be a power of two to tile the picture")
    return side


def layout(cfg: Config) -> list[Piece]:
    """Where each square of weights sits, in forward-pass reading order."""
    d = cfg.width
    across = picture_side(cfg) // d
    pieces: list[Piece] = []

    def place(tensor: str, label: str, layer: int, rows: tuple[int, int], cols: tuple[int, int]):
        slot = len(pieces)
        x, y = (slot % across) * d, (slot // across) * d
        pieces.append(Piece(tensor, label, layer, rows, cols, x, y))

    for layer in range(cfg.layers):
        prefix = f"blocks.{layer}"
        for i, part in enumerate(("query", "key", "value")):
            place(f"{prefix}.attention.qkv.weight", part, layer, (i * d, (i + 1) * d), (0, d))
        place(f"{prefix}.attention.out.weight", "attn out", layer, (0, d), (0, d))
        for i in range(4):
            place(f"{prefix}.mlp.up.weight", f"mlp in {i + 1}", layer, (i * d, (i + 1) * d), (0, d))
        for i in range(4):
            cols = (i * d, (i + 1) * d)
            place(f"{prefix}.mlp.down.weight", f"mlp out {i + 1}", layer, (0, d), cols)
    return pieces


def _bit_names(model: GPT) -> list[str]:
    return [f"{name}.weight" for name, m in model.named_modules() if m in model.bit_layers()]


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def write_png(path: Path, bits: np.ndarray, manifest: dict, side_table: bytes) -> None:
    """Write `bits` (a square boolean array) as a one-bit grayscale PNG, filter 0 on every row."""
    side = bits.shape[0]
    packed = np.packbits(bits, axis=1)
    rows = np.concatenate([np.zeros((side, 1), np.uint8), packed], axis=1).tobytes()
    header = struct.pack(">IIBBBBB", side, side, 1, 0, 0, 0, 0)
    text = MANIFEST_KEY + b"\0" + json.dumps(manifest).encode("latin-1")
    path.write_bytes(
        SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"tEXt", text)
        + _chunk(SIDE_CHUNK, side_table)
        + _chunk(b"IDAT", zlib.compress(rows, 9))
        + _chunk(b"IEND", b"")
    )


def read_png(path: Path) -> tuple[np.ndarray, dict, bytes]:
    """Inverse of `write_png`. Only reads files this module wrote."""
    raw = path.read_bytes()
    if raw[:8] != SIGNATURE:
        raise ValueError(f"{path} is not a PNG")
    chunks: dict[bytes, bytes] = {}
    at = 8
    while at < len(raw):
        (length,) = struct.unpack(">I", raw[at : at + 4])
        kind = raw[at + 4 : at + 8]
        chunks[kind] = chunks.get(kind, b"") + raw[at + 8 : at + 8 + length]
        at += 12 + length
    side = struct.unpack(">I", chunks[b"IHDR"][:4])[0]
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(side, 1 + side // 8)
    if rows[:, 0].any():
        raise ValueError("expected filter type 0 on every row")
    bits = np.unpackbits(rows[:, 1:], axis=1).astype(bool)
    manifest = json.loads(chunks[b"tEXt"].split(b"\0", 1)[1].decode("latin-1"))
    return bits, manifest, chunks[SIDE_CHUNK]


@torch.no_grad()
def export(model: GPT, vocab: Vocab, path: Path) -> dict:
    """Write `model` and `vocab` to `path` as one picture. Returns the manifest."""
    cfg = model.cfg
    if cfg.ternary or not cfg.one_bit:
        raise ValueError("only a one-bit model fits one pixel per weight")
    state = {name: value.cpu() for name, value in model.state_dict().items()}
    bit_names = _bit_names(model)
    side = picture_side(cfg)
    bits = np.zeros((side, side), bool)
    pieces = layout(cfg)
    d = cfg.width
    for piece in pieces:
        weight = state[piece.tensor]
        square = weight[slice(*piece.rows), slice(*piece.cols)] >= weight.mean()
        bits[piece.y : piece.y + d, piece.x : piece.x + d] = square.numpy()

    side_table = bytearray()
    side_index = []
    for name, value in state.items():
        if name in bit_names:
            continue
        side_index.append({"name": name, "shape": list(value.shape), "offset": len(side_table)})
        side_table += value.numpy().astype(SIDE_DTYPE).tobytes()

    manifest = {
        "version": FORMAT_VERSION,
        "config": cfg.to_dict(),
        "side": side,
        "square": d,
        "pieces": [asdict(piece) for piece in pieces],
        "scales": {name: float(state[name].abs().mean()) for name in bit_names},
        "side_table": side_index,
        "bits_crc32": zlib.crc32(np.packbits(bits, axis=1).tobytes()),
        "vocab": {"name": vocab.name, "tokens": vocab.tokens, "bpe_json": vocab.bpe_json},
    }
    write_png(path, bits, manifest, bytes(side_table))
    return manifest


def load(path: Path) -> tuple[GPT, Vocab]:
    """Rebuild a hardened model and its vocabulary from a picture alone."""
    bits, manifest, side_table = read_png(path)
    if manifest["version"] != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format {manifest['version']}, expected {FORMAT_VERSION}; export it again"
        )
    if zlib.crc32(np.packbits(bits, axis=1).tobytes()) != manifest["bits_crc32"]:
        raise ValueError(f"{path}: weight bits do not match the checksum")
    model = GPT(Config(**manifest["config"])).eval()
    state = model.state_dict()
    for entry in manifest["side_table"]:
        count = int(np.prod(entry["shape"]))
        values = np.frombuffer(side_table, SIDE_DTYPE, count, entry["offset"]).astype(np.float32)
        state[entry["name"]] = torch.from_numpy(values.reshape(entry["shape"]))
    d = manifest["square"]
    for piece in manifest["pieces"]:
        square = torch.from_numpy(bits[piece["y"] : piece["y"] + d, piece["x"] : piece["x"] + d])
        scale = manifest["scales"][piece["tensor"]]
        rows, cols = slice(*piece["rows"]), slice(*piece["cols"])
        state[piece["tensor"]][rows, cols] = torch.where(square, scale, -scale)
    model.load_state_dict(state)
    for layer in model.bit_layers():
        layer.one_bit = False
    vocab = manifest["vocab"]
    return model, Vocab(vocab["name"], vocab["tokens"], vocab["bpe_json"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="run directory under runs/")
    args = parser.parse_args()

    run_dir = checkpoint.RUNS_DIR / args.name
    model, vocab = checkpoint.load(run_dir, torch.device("cpu"))
    path = run_dir / "brain.png"
    manifest = export(model, vocab, path)
    linear, other = model.count_weights()
    side = manifest["side"]
    print(f"{path}: {side} x {side}, {path.stat().st_size:,} bytes")
    print(f"{linear:,} weights as pixels ({linear / side**2:.1%} of the picture), {other:,} beside")


if __name__ == "__main__":
    main()
