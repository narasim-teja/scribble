"""Vocabularies: turning story text into token ids and back.

Two kinds share one interface. `chars` has one token per character. `bpe<N>`
has N word pieces learned from the data. Token 0 is always the end-of-story
marker. A vocabulary is fully described by its list of token strings, so it is
saved as JSON and needs no library to decode. Word-piece vocabularies also
keep their merge rules, which are needed only to encode.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers

EOT = 0
EOT_TEXT = "<|endoftext|>"

# Characters kept from the raw data: newline plus printable ASCII.
ALPHABET = "\n" + "".join(chr(c) for c in range(32, 127))
_KEEP = {ord(c): c for c in ALPHABET}

# Splits text into pieces that together cover every character: a newline, a
# word with its leading space, or a run of spaces. Merges never cross pieces.
_PIECES = Regex(r"\n| ?[^ \n]+| +")


def clean(story: str) -> str:
    """Drop everything outside the alphabet, and surrounding whitespace."""
    return story.translate(_Filter()).strip()


class _Filter(dict):
    def __missing__(self, key: int) -> str | None:
        return _KEEP.get(key)


@dataclass(frozen=True)
class Vocab:
    """A list of token strings. Index in the list is the token id."""

    name: str
    tokens: list[str]
    bpe_json: str | None = None  # the `tokenizers` rules; only needed to encode

    @property
    def size(self) -> int:
        return len(self.tokens)

    def decode(self, ids: Iterable[int]) -> str:
        return "".join("" if i == EOT else self.tokens[i] for i in ids)

    def char_lengths(self) -> np.ndarray:
        """Characters each token stands for. The story marker counts as one."""
        return np.array([1 if i == EOT else len(t) for i, t in enumerate(self.tokens)])

    def encode_stories(self, stories: list[str]) -> Iterator[list[int]]:
        """Yield ids for each story, each ending with the story marker."""
        if self.name == "chars":
            index = {t: i for i, t in enumerate(self.tokens)}
            for story in stories:
                yield [index[c] for c in story] + [EOT]
        else:
            for enc in self._bpe().encode_batch(stories):
                yield enc.ids + [EOT]

    def encode(self, text: str) -> list[int]:
        """Ids for a prompt, with no story marker added."""
        return next(self.encode_stories([text]))[:-1]

    def _bpe(self) -> Tokenizer:
        if self.bpe_json is None:
            raise ValueError(f"vocabulary {self.name!r} has no word-piece rules")
        if not hasattr(self, "_tok"):
            object.__setattr__(self, "_tok", Tokenizer.from_str(self.bpe_json))
        return self._tok  # type: ignore[attr-defined]

    def save(self, path: Path) -> None:
        raw = {"name": self.name, "tokens": self.tokens, "bpe_json": self.bpe_json}
        path.write_text(json.dumps(raw))

    @staticmethod
    def load(path: Path) -> Vocab:
        raw = json.loads(path.read_text())
        return Vocab(raw["name"], raw["tokens"], raw.get("bpe_json"))


def char_vocab() -> Vocab:
    return Vocab("chars", [EOT_TEXT, *ALPHABET])


def train_bpe(stories: Iterable[str], size: int) -> Vocab:
    """Learn a word-piece vocabulary of exactly `size` tokens."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Split(_PIECES, behavior="isolated")
    trainer = trainers.BpeTrainer(
        vocab_size=size,
        special_tokens=[EOT_TEXT],
        initial_alphabet=list(ALPHABET),
        show_progress=False,
    )
    tok.train_from_iterator(stories, trainer)
    by_id = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    return Vocab(f"bpe{size}", [t for t, _ in by_id], tok.to_str())
