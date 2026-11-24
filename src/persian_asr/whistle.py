"""Whistle checkpoints: download (pinned), load into CAT's ConformerNet, build S2 init.

CAT checkpoint layout (verified on THU-SPMI/whistle-small best-3.pt): a dict with
key "model" -> state_dict whose keys are "module.encoder.<...>" (DDP-wrapped
AMTrainer whose .encoder is a cat.shared.encoder.ConformerNet). The output layer is
"module.encoder.classifier.{weight,bias}" with shape (75, 512) / (75,) for the
75 IPA phoneme units of CV-Lang10.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch

from . import config

CLASSIFIER_KEYS = ("module.encoder.classifier.weight", "module.encoder.classifier.bias")


def download_checkpoint(name: str, dest_root: Path) -> Path:
    """Download one pinned Whistle repo into dest_root/name (skips files already present)."""
    from huggingface_hub import snapshot_download

    spec = config.WHISTLE_CHECKPOINTS[name]
    local = dest_root / name
    snapshot_download(spec["repo_id"], revision=spec["revision"], local_dir=str(local))
    (local / "SOURCE.json").write_text(json.dumps({**spec, "name": name}, indent=2))
    return local


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_cat_checkpoint(path: str | os.PathLike, map_location: str = "cpu") -> dict:
    # trusted source (THU-SPMI or our own runs); see config.TORCH_ENV for why weights_only=False
    return torch.load(path, map_location=map_location, weights_only=False)


def encoder_state_from_checkpoint(ckpt: dict) -> "OrderedDict[str, torch.Tensor]":
    """Strip the DDP+trainer prefix: 'module.encoder.X' -> 'X'."""
    prefix = "module.encoder."
    sd = ckpt["model"]
    bad = [k for k in sd if not k.startswith(prefix)]
    if bad:
        raise ValueError(f"unexpected keys (not under {prefix}): {bad[:5]}")
    return OrderedDict((k[len(prefix):], v) for k, v in sd.items())


def build_conformer(num_classes: int, encoder_kwargs: dict | None = None) -> torch.nn.Module:
    """Instantiate CAT's ConformerNet with the Whistle-small architecture."""
    from cat.shared.encoder import ConformerNet

    kw = dict(encoder_kwargs or config.WHISTLE_SMALL_ENCODER)
    kw["num_classes"] = num_classes
    return ConformerNet(**kw)


def make_s2_init(src_ckpt: str | os.PathLike, out_ckpt: str | os.PathLike, vocab_size: int,
                 seed: int, hdim: int | None = None) -> dict:
    """Write an `init_model` checkpoint for S2: all Whistle-small encoder weights are kept,
    the 75-way phoneme classifier is replaced by a NEW randomly initialised
    Linear(hdim, vocab_size) for the Persian BPE units.

    CAT loads init_model with strict=True (cat/shared/manager.py:303-318), so the
    classifier shape must already match the new vocabulary. The new layer uses
    torch.nn.Linear's default init (same as CAT's AbsEncoder, cat/shared/encoder.py:42,
    i.e. the same init S1 gets), drawn from a dedicated seeded generator.
    Returns a report of what was loaded vs re-initialised.
    """
    ck = load_cat_checkpoint(src_ckpt)
    sd = ck["model"]
    w_key, b_key = CLASSIFIER_KEYS
    old_shape = tuple(sd[w_key].shape)
    hdim = hdim or old_shape[1]
    g_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        new = torch.nn.Linear(hdim, vocab_size)
    finally:
        torch.random.set_rng_state(g_state)
    new_sd = OrderedDict(sd)
    new_sd[w_key] = new.weight.detach().clone()
    new_sd[b_key] = new.bias.detach().clone()
    kept = [k for k in sd if k not in CLASSIFIER_KEYS]
    out = {"model": new_sd}
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_ckpt)
    report = {
        "source": str(src_ckpt),
        "output": str(out_ckpt),
        "loaded_from_whistle": {"n_tensors": len(kept),
                                "n_params": int(sum(sd[k].numel() for k in kept))},
        "reinitialised": {k: list(new_sd[k].shape) for k in CLASSIFIER_KEYS},
        "dropped": {w_key: list(old_shape)},
        "classifier_init": "torch.nn.Linear default (kaiming_uniform a=sqrt(5)), seed=%d" % seed,
        "optimizer_state": "not loaded (init_model only loads model weights)",
    }
    Path(str(out_ckpt) + ".json").write_text(json.dumps(report, indent=2))
    return report


def load_encoder_for_inference(ckpt_path: str | os.PathLike, num_classes: int | None = None,
                               device: str = "cpu") -> torch.nn.Module:
    ck = load_cat_checkpoint(ckpt_path)
    sd = encoder_state_from_checkpoint(ck)
    n = num_classes or sd["classifier.weight"].shape[0]
    model = build_conformer(n)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()
