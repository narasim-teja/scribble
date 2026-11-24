"""A small GPT whose linear layers can use one-bit weights.

Layout follows BitNet: there are no separate pre-attention or pre-MLP norms.
Every linear layer normalizes its own input, then multiplies by a weight
matrix that is, in the forward pass, only +scale or -scale. The browser engine
mirrors this file, so keep it simple.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class Config:
    vocab_size: int
    context: int = 256
    layers: int = 5
    heads: int = 4
    width: int = 256
    one_bit: bool = True
    # Three weight values (-scale, 0, +scale) instead of two. A training
    # experiment only: such a model cannot be exported as a one-bit picture.
    ternary: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def binarize(weight: Tensor) -> Tensor:
    """Replace each weight by +scale or -scale, where scale is the mean size."""
    scale = weight.abs().mean()
    return torch.where(weight >= weight.mean(), scale, -scale)


def ternarize(weight: Tensor) -> Tensor:
    """Replace each weight by -scale, 0 or +scale, whichever is nearest (BitNet b1.58)."""
    scale = weight.abs().mean()
    return (weight / scale).round().clamp(-1, 1) * scale


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gain


class BitLinear(nn.Module):
    """Normalize the input, then apply a bias-free linear layer.

    With `one_bit`, the forward pass sees quantized weights (two values, or
    three with `ternary`) while gradients flow to the full-precision weights
    underneath (straight-through estimator).
    """

    def __init__(self, inputs: int, outputs: int, one_bit: bool, ternary: bool = False) -> None:
        super().__init__()
        self.one_bit = one_bit
        self.quantize = ternarize if ternary else binarize
        self.norm = RMSNorm(inputs)
        self.weight = nn.Parameter(torch.randn(outputs, inputs) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        if self.one_bit:
            weight = weight + (self.quantize(weight) - weight).detach()
        return F.linear(self.norm(x), weight)


class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.qkv = BitLinear(cfg.width, 3 * cfg.width, cfg.one_bit, cfg.ternary)
        self.out = BitLinear(cfg.width, cfg.width, cfg.one_bit, cfg.ternary)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, width = x.shape
        q, k, v = (
            part.view(batch, time, self.heads, width // self.heads).transpose(1, 2)
            for part in self.qkv(x).split(width, dim=-1)
        )
        mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(mixed.transpose(1, 2).reshape(batch, time, width))


class MLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.up = BitLinear(cfg.width, 4 * cfg.width, cfg.one_bit, cfg.ternary)
        self.down = BitLinear(4 * cfg.width, cfg.width, cfg.one_bit, cfg.ternary)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.attention = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(x)
        return x + self.mlp(x)


class GPT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.width)
        self.positions = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.width)
        nn.init.normal_(self.tokens.weight, std=0.02)
        nn.init.normal_(self.positions.weight, std=0.02)

    def forward(self, ids: Tensor) -> Tensor:
        """Logits for the next token at every position. `ids` is (batch, time)."""
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.tokens(ids) + self.positions(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x) @ self.tokens.weight.T  # output layer shares the embedding

    def bit_layers(self) -> list[BitLinear]:
        return [m for m in self.modules() if isinstance(m, BitLinear)]

    def count_weights(self) -> tuple[int, int]:
        """(weights in linear layers, all other numbers)."""
        linear = sum(layer.weight.numel() for layer in self.bit_layers())
        return linear, sum(p.numel() for p in self.parameters()) - linear

    @torch.no_grad()
    def harden(self) -> None:
        """Overwrite the full-precision weights with their one-bit values.

        After this the model computes exactly what it did before, and each
        stored weight is literally +scale or -scale, ready to be damaged.
        """
        for layer in self.bit_layers():
            layer.weight.copy_(layer.quantize(layer.weight))
            layer.one_bit = False

    @torch.no_grad()
    def generate(
        self,
        ids: Tensor,
        count: int,
        generator: torch.Generator,
        temperature: float = 0.8,
    ) -> Tensor:
        """Append `count` sampled tokens to `ids`, which is (batch, time)."""
        for _ in range(count):
            logits = self(ids[:, -self.cfg.context :])[:, -1] / temperature
            probs = F.softmax(logits, dim=-1).cpu()
            choice = torch.multinomial(probs, 1, generator=generator)
            ids = torch.cat([ids, choice.to(ids.device)], dim=1)
        return ids


@torch.no_grad()
def flip_bits(model: GPT, fraction: float, generator: torch.Generator) -> int:
    """Flip a random `fraction` of the weights of a hardened model. Returns the count."""
    flipped = 0
    for layer in model.bit_layers():
        if layer.one_bit:
            raise ValueError("call model.harden() before flipping bits")
        hit = torch.rand(layer.weight.shape, generator=generator) < fraction
        layer.weight[hit.to(layer.weight.device)] *= -1
        flipped += int(hit.sum())
    return flipped
