#!/usr/bin/env python3
"""Apply this project's small, documented patches to a CAT checkout (idempotent).

Usage: python scripts/patch_cat.py <cat_dir>

Patches (each is an exact-string replacement that must match once, so a CAT
version change fails loudly instead of silently mis-patching):

1. cat/ctc/train.py (CTC forward, CAT @2340970f04 lines 176-195): CAT moves labels
   and lengths to CPU and passes them to nn.CTCLoss with CUDA log-probs. torch 2.14
   raises "Expected all tensors to be on the same device" (verified 2026-11-25 on the
   RTX 6000 server). We move them to the logits' device for the non-CRF path.
"""
from __future__ import annotations

import sys
from pathlib import Path

MARK = "persian_asr patch"

PATCHES = [
    (
        "cat/ctc/train.py",
        """        labels = labels.cpu()
        lx = lx.cpu()
        ly = ly.cpu()
        if self.is_crf:""",
        """        labels = labels.cpu()
        lx = lx.cpu()
        ly = ly.cpu()
        if not self.is_crf:
            # persian_asr patch: torch>=2.x CTCLoss needs targets/lengths on the
            # same device as log_probs (CUDA logits + CPU targets raises).
            labels, lx, ly = (t.to(logits.device) for t in (labels, lx, ly))
        if self.is_crf:""",
    ),
]


def apply(cat_dir: Path) -> list[str]:
    done = []
    for rel, old, new in PATCHES:
        p = cat_dir / rel
        s = p.read_text()
        if new in s:
            done.append(f"{rel}: already patched")
            continue
        if s.count(old) != 1:
            raise SystemExit(f"{rel}: expected exactly one match for the patch context; CAT changed?")
        p.write_text(s.replace(old, new))
        done.append(f"{rel}: patched")
    return done


if __name__ == "__main__":
    for line in apply(Path(sys.argv[1])):
        print(line)
