#!/usr/bin/env python3
"""Apply this project's small, documented patches to a CAT checkout (idempotent).

Usage: python scripts/patch_cat.py <cat_dir>

Patches (each is an exact-string replacement that must match once, so a CAT
version change fails loudly instead of silently mis-patching):

1. cat/ctc/train.py (CTC forward, CAT @2340970f04 lines 176-195): CAT moves labels
   and lengths to CPU and passes them to nn.CTCLoss with CUDA log-probs. torch 2.14
   raises "Expected all tensors to be on the same device" (verified 2026-11-25 on the
   RTX 6000 server). We move them to the logits' device for the non-CRF path.
2. cat/ctc/train.py (same function, non-CRF branch): with targets on the GPU, torch uses its
   native CUDA CTC kernel. Called the way CAT calls it (transposed, non-contiguous log-probs
   inside fp16 autocast) its gradient is wrong and non-deterministic (up to 1.77 abs error,
   norm x120; garbage on padded frames). Training with it made dev loss rise from 6.8 to 11.5
   (S2 fa 1h, run moved to exp_invalid/). Fixed: fp32, padded frames zeroed with masked_fill,
   contiguous, autocast off (tests/test_cat_ctc_patch.py).
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
    (
        "cat/ctc/train.py",
        """        else:
            # [N, T, C] -> [T, N, C]
            logits = logits.transpose(0, 1)
            loss = self.criterion(
                logits, labels.to(torch.int), lx.to(torch.int), ly.to(torch.int)
            )""",
        """        else:
            # persian_asr patch: torch 2.14's CUDA CTC backward is wrong on CAT's input
            # (verified 2026-11-25 vs the CPU reference): with non-contiguous [T,N,C]
            # log-probs inside fp16 autocast, valid-frame grads are off (up to 1.8 abs,
            # norm x120) and padded frames (t >= lx) receive non-deterministic garbage.
            # Fix: fp32, padded frames zeroed (masked_fill blocks their gradient),
            # contiguous, autocast off -> grads match CPU to 3e-6, padded grads exactly 0.
            valid = torch.arange(logits.size(1), device=logits.device)[None, :] < lx[:, None]
            logits = logits.float().masked_fill(~valid[..., None], 0.0)
            # [N, T, C] -> [T, N, C]
            logits = logits.transpose(0, 1).contiguous()
            with autocast(enabled=False):
                loss = self.criterion(
                    logits, labels.to(torch.int), lx.to(torch.int), ly.to(torch.int)
                )""",
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
