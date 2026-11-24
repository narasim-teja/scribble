import torch

from model import GPT, Config, binarize, flip_bits, ternarize

CFG = Config(vocab_size=50, context=16, layers=2, heads=2, width=32)


def test_binarize_gives_two_values_of_mean_size():
    weight = torch.randn(64, 64)
    hard = binarize(weight)
    assert set(hard.unique().tolist()) == {-hard.max().item(), hard.max().item()}
    assert torch.isclose(hard.max(), weight.abs().mean())


def test_gradients_reach_full_precision_weights():
    model = GPT(CFG)
    model(torch.randint(0, 50, (2, 16))).sum().backward()
    assert all(layer.weight.grad.abs().sum() > 0 for layer in model.bit_layers())


def test_harden_keeps_output_and_stores_one_bit_weights():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    ids = torch.randint(0, 50, (2, 16))
    before = model(ids)
    model.harden()
    assert torch.allclose(model(ids), before, atol=1e-5)
    assert all(layer.weight.unique().numel() == 2 for layer in model.bit_layers())


def test_flip_bits_changes_about_the_requested_fraction():
    model = GPT(CFG).eval()
    model.harden()
    total, _ = model.count_weights()
    flipped = flip_bits(model, 0.1, torch.Generator().manual_seed(0))
    assert 0.08 * total < flipped < 0.12 * total


def test_weight_count_matches_formula():
    linear, _ = GPT(CFG).count_weights()
    assert linear == CFG.layers * 12 * CFG.width**2


def test_lesion_regions_tile_every_weight_once_and_restore_exactly():
    from lesion import randomize, regions, restore

    model = GPT(CFG).eval()
    model.harden()
    before = [layer.weight.clone() for layer in model.bit_layers()]
    covered = 0
    for region in regions(model):
        covered += region.weights().numel()
        original = randomize(region, 1.0, torch.Generator().manual_seed(0))
        assert (region.weights() != original).float().mean() > 0.3
        restore(region, original)
    assert covered == model.count_weights()[0]
    after = [layer.weight for layer in model.bit_layers()]
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_ternarize_gives_three_values_of_mean_size():
    weight = torch.randn(64, 64)
    hard = ternarize(weight)
    scale = weight.abs().mean().item()
    assert sorted(hard.unique().tolist()) == [-scale, 0.0, scale]


def test_harden_keeps_output_of_a_ternary_model():
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=50, context=16, layers=2, heads=2, width=32, ternary=True)).eval()
    ids = torch.randint(0, 50, (2, 16))
    before = model(ids)
    model.harden()
    assert torch.allclose(model(ids), before, atol=1e-5)
    assert all(layer.weight.unique().numel() == 3 for layer in model.bit_layers())
