import zlib

import numpy as np
import pytest
import torch

import export
from model import GPT, Config
from vocab import char_vocab

CFG = Config(vocab_size=char_vocab().size, context=16, layers=2, heads=2, width=32)


def trained_like() -> GPT:
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_layout_squares_do_not_overlap_and_cover_every_weight():
    side = export.picture_side(CFG)
    seen = np.zeros((side, side), int)
    for piece in export.layout(CFG):
        seen[piece.y : piece.y + CFG.width, piece.x : piece.x + CFG.width] += 1
    assert seen.max() == 1
    assert seen.sum() == GPT(CFG).count_weights()[0]


def test_picture_alone_reproduces_the_model(tmp_path):
    model = trained_like()
    ids = torch.randint(0, CFG.vocab_size, (2, 16))
    path = tmp_path / "brain.png"
    export.export(model, char_vocab(), path)
    restored, vocab = export.load(path)
    model.harden()
    assert torch.equal(restored(ids), model(ids))
    assert vocab.tokens == char_vocab().tokens


def test_flipping_one_pixel_changes_one_weight(tmp_path):
    path = tmp_path / "brain.png"
    manifest = export.export(trained_like(), char_vocab(), path)
    bits, _, side_table = export.read_png(path)
    before, _ = export.load(path)
    bits[0, 0] ^= True
    manifest["bits_crc32"] = zlib.crc32(np.packbits(bits, axis=1).tobytes())
    export.write_png(path, bits, manifest, side_table)
    after, _ = export.load(path)
    first = manifest["pieces"][0]["tensor"]
    changed = before.state_dict()[first] != after.state_dict()[first]
    assert changed.sum() == 1 and changed[0, 0]


def test_damaged_file_is_rejected(tmp_path):
    path = tmp_path / "brain.png"
    manifest = export.export(trained_like(), char_vocab(), path)
    bits, _, side_table = export.read_png(path)
    bits[3, 3] ^= True
    export.write_png(path, bits, manifest, side_table)
    with pytest.raises(ValueError, match="checksum"):
        export.load(path)
