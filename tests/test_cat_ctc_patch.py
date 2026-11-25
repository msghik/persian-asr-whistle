"""Regression test for scripts/patch_cat.py patch 2 (CUDA CTC gradient). GPU only."""
import inspect

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _patched_ctc(lp_ntc, y, lx, ly):
    # same formulation as the patched cat/ctc/train.py (non-CRF branch)
    valid = torch.arange(lp_ntc.size(1), device=lp_ntc.device)[None, :] < lx[:, None]
    lp = lp_ntc.float().masked_fill(~valid[..., None], 0.0).transpose(0, 1).contiguous()
    with torch.autocast("cuda", enabled=False):
        return torch.nn.CTCLoss()(lp, y, lx, ly)


@cuda
def test_patched_ctc_grad_matches_cpu():
    torch.manual_seed(0)
    N, T, C, L = 4, 120, 500, 25
    logits = torch.randn(N, T, C)
    y = torch.randint(1, C, (N * L,), dtype=torch.int32)
    lx = torch.tensor([120, 110, 100, 90], dtype=torch.int32)
    ly = torch.full((N,), L, dtype=torch.int32)

    x = logits.clone().requires_grad_(True)
    ref = torch.nn.CTCLoss()(torch.log_softmax(x, -1).transpose(0, 1), y, lx, ly)
    ref.backward()

    g = logits.clone().cuda().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = _patched_ctc(torch.log_softmax(g, -1), y.cuda(), lx.cuda(), ly.cuda())
    loss.backward()
    assert abs(loss.item() - ref.item()) < 1e-3
    pad = torch.arange(T)[None, :] >= lx[:, None]
    assert g.grad.cpu()[pad].abs().max().item() == 0.0
    assert (g.grad.cpu() - x.grad).abs().max().item() < 1e-4


def test_cat_source_is_patched():
    from cat.ctc import train
    src = inspect.getsource(train)
    assert src.count("persian_asr patch") == 2 and "masked_fill(~valid" in src
