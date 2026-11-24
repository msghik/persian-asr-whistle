"""Pinned versions and the Drive / local directory layout shared by all notebooks.

Everything persistent lives under DRIVE_ROOT (Google Drive). Everything under
LOCAL_ROOT is scratch that is rebuilt (or copied from Drive) each Colab session.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- pinned upstream versions -------------------------------------------------
# CAT commit read and tested for this project (git ls-remote, 2026-11-24).
CAT_REPO = "https://github.com/thu-spmi/CAT.git"
CAT_COMMIT = "2340970f046662f87bd1595605aa7be044716c14"
# Kaldi commit used for the minimal WFST-decoding build (scripts/build_kaldi_min.sh).
KALDI_COMMIT = "e02e35f0254bb033fab73d1df99fc34123e31d56"
# webdataset commit pinned in CAT's requirements.txt (do not use the PyPI one).
WEBDATASET_COMMIT = "d7334016f44a03c4a385971aa835c4f460d3f30a"
KENLM_URL = "https://github.com/kpu/kenlm/archive/master.zip"

# Hugging Face checkpoints (org is "THU-SPMI"; the lowercase "thu-spmi" org is empty).
# Revisions pinned to the commit current on 2026-11-24.
WHISTLE_CHECKPOINTS = {
    "whistle-small": {
        "repo_id": "THU-SPMI/whistle-small",
        "revision": "83add32935aeefac3a5fed910283e975fe946116",
        "ckpt": "best-3.pt",
        "default": True,
    },
    # Downloaded only on request: Prof. Ou said NOT to fine-tune it (its BPE is not Persian).
    "whistle-small-subword": {
        "repo_id": "THU-SPMI/whistle-small-subword",
        "revision": "ce6ccc2269708a64bf79ab32729c8ebd34a7c3ff",
        "ckpt": "Multi._subword_best-3.pt",
        "default": False,
    },
}

# CAT checkpoints contain pickled non-tensor objects; torch>=2.6 defaults to
# weights_only=True and refuses them (verified with torch 2.15). CAT's own
# torch.load calls (cat/shared/manager.py:298,305; cat/shared/coreutils.py:546)
# do not pass weights_only, so we set this env var for every CAT process.
TORCH_ENV = {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}

# Whistle-small encoder (from THU-SPMI/whistle-small config.json); S1 uses the same.
WHISTLE_SMALL_ENCODER = {
    "num_cells": 14,
    "idim": 80,
    "hdim": 512,
    "conv": "vgg2l",
    "num_heads": 4,
    "kernel_size": 15,
}


@dataclass(frozen=True)
class Paths:
    drive_root: Path
    local_root: Path

    # persistent (Drive)
    @property
    def cache(self) -> Path:          # wheels, built binaries (keyed by environment)
        return self.drive_root / "cache"

    @property
    def pretrained(self) -> Path:     # Whistle checkpoints
        return self.drive_root / "pretrained"

    @property
    def data(self) -> Path:           # raw downloads + prepared shards
        return self.drive_root / "data"

    @property
    def lm(self) -> Path:
        return self.drive_root / "lm"

    @property
    def exp(self) -> Path:            # per-run checkpoints / logs
        return self.drive_root / "exp"

    @property
    def results(self) -> Path:        # results.csv, setup summaries
        return self.drive_root / "results"

    # scratch (local disk, fast)
    @property
    def local_tools(self) -> Path:    # CAT checkout, kaldi, kenlm binaries
        return self.local_root / "tools"

    @property
    def local_work(self) -> Path:
        return self.local_root / "work"

    def ensure(self) -> "Paths":
        for p in (self.cache, self.pretrained, self.data, self.lm, self.exp,
                  self.results, self.local_tools, self.local_work):
            p.mkdir(parents=True, exist_ok=True)
        return self


def default_paths(drive_root: str | os.PathLike | None = None,
                  local_root: str | os.PathLike | None = None) -> Paths:
    drive_root = Path(drive_root or os.environ.get(
        "PERSIAN_ASR_DRIVE", "/content/drive/MyDrive/persian_asr"))
    local_root = Path(local_root or os.environ.get("PERSIAN_ASR_LOCAL", "/content/persian_asr_local"))
    return Paths(drive_root=drive_root, local_root=local_root)
