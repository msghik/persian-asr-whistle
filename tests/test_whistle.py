from collections import OrderedDict

import pytest
import torch

from conftest import requires_cat
from persian_asr import whistle

TINY = dict(num_cells=2, idim=80, hdim=32, conv="vgg2l", num_heads=4, kernel_size=15)


def _fake_whistle_ckpt(path, n_phones=75):
    m = whistle.build_conformer(n_phones, TINY)
    sd = OrderedDict(("module.encoder." + k, v) for k, v in m.state_dict().items())
    torch.save({"model": sd}, path)
    return sd


@requires_cat
def test_make_s2_init_replaces_only_classifier(tmp_path):
    src, out = tmp_path / "w.pt", tmp_path / "s2.pt"
    sd = _fake_whistle_ckpt(src)
    rep = whistle.make_s2_init(src, out, vocab_size=500, seed=0)
    new = torch.load(out, weights_only=False)["model"]
    assert tuple(new["module.encoder.classifier.weight"].shape) == (500, 32)
    assert tuple(new["module.encoder.classifier.bias"].shape) == (500,)
    for k, v in sd.items():
        if "classifier" not in k:
            assert torch.equal(new[k], v), k
    assert rep["reinitialised"]["module.encoder.classifier.weight"] == [500, 32]
    assert rep["dropped"]["module.encoder.classifier.weight"] == [75, 32]
    assert (tmp_path / "s2.pt.json").is_file()
    # loads strictly into a 500-class model, like CAT's init_model path does
    m = whistle.build_conformer(500, TINY)
    m.load_state_dict(whistle.encoder_state_from_checkpoint({"model": new}), strict=True)


@requires_cat
def test_make_s2_init_is_seeded_and_does_not_touch_global_rng(tmp_path):
    src = tmp_path / "w.pt"
    _fake_whistle_ckpt(src)
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    whistle.make_s2_init(src, tmp_path / "a.pt", 500, seed=7)
    after = torch.rand(1)
    assert torch.equal(before, after)
    whistle.make_s2_init(src, tmp_path / "b.pt", 500, seed=7)
    a = torch.load(tmp_path / "a.pt", weights_only=False)["model"]["module.encoder.classifier.weight"]
    b = torch.load(tmp_path / "b.pt", weights_only=False)["model"]["module.encoder.classifier.weight"]
    assert torch.equal(a, b)


def test_encoder_state_rejects_foreign_keys():
    with pytest.raises(ValueError):
        whistle.encoder_state_from_checkpoint({"model": {"foo.bar": torch.zeros(1)}})


@requires_cat
def test_real_whistle_small_loads(whistle_small_dir):
    m = whistle.load_encoder_for_inference(whistle_small_dir / "best-3.pt")
    n = sum(p.numel() for p in m.parameters())
    assert abs(n / 1e6 - 90.01) < 0.05  # 90.011673 M verified 2026-11-24
    assert m.classifier.weight.shape == (75, 512)
