"""Smoke tests and benchmarks used by notebooks/00_setup.ipynb (CPU-testable at tiny sizes)."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import torch

from . import features, whistle

# Small public English sample (LibriSpeech dev-clean excerpt, 73 utts, ~9 MB).
LIBRI_DUMMY_URL = ("https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/"
                   "resolve/main/clean/validation-00000-of-00001.parquet")

# Greedy whistle-small output for utterance 0 of LIBRI_DUMMY_URL, recorded on CPU
# (torch 2.15, 2026-11-24). On GPU tiny numeric differences may change a token or two,
# so a mismatch is reported but is not by itself a failure.
EXPECTED_UTT0_IPA = ("m ɹ k w ɪ l t ə ɪ z ð i ə p ɑ s l ə v ð i m ɪ d l k l æ s i z æ n d "
                     "w i ɑ ɹ ɡ l æ d t o ʊ w ɛ l k ə m h ɪ z ɡ ɑ s p l")


def _ctc_greedy(ids: list[int], blank: int = 0) -> list[int]:
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def english_phoneme_check(ckpt_dir: Path, parquet_path: Path, device: str = "cpu", n: int = 2) -> dict:
    """Run pretrained whistle-small on English speech; it should emit sensible IPA.

    Checks checkpoint loading, CAT model code, and the feature front-end
    (int16 scaling) end to end. Pass = every utterance yields >= 10 phonemes.
    """
    import pyarrow.parquet as pq
    import soundfile as sf
    from cat.shared.tokenizer import load as load_tokenizer

    tk = load_tokenizer(str(ckpt_dir / "tokenizer_mul10.tknz"))
    model = whistle.load_encoder_for_inference(ckpt_dir / "best-3.pt", device=device)
    rows = pq.read_table(parquet_path).slice(0, n).to_pylist()
    utts = []
    for r in rows:
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        f = features.fbank80(torch.from_numpy(wav), sr).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(f, torch.tensor([f.size(1)], device=device))
        frame_ids = logits[0].argmax(-1).tolist()
        toks = _ctc_greedy(frame_ids)
        utts.append({
            "ref_text": r["text"],
            "hyp_ipa": tk.decode(toks) if toks else "",
            "n_phonemes": len(toks),
            "blank_frac": round(sum(i == 0 for i in frame_ids) / max(1, len(frame_ids)), 3),
        })
    return {
        "pass": all(u["n_phonemes"] >= 10 for u in utts),
        "utt0_matches_cpu_reference": bool(utts) and utts[0]["hyp_ipa"] == EXPECTED_UTT0_IPA,
        "utts": utts,
    }


def train_step_benchmark(device: str, micro_batches: list[int], frames: int = 1000, vocab: int = 500,
                         steps: int = 3, amp: bool = True, encoder_kwargs: dict | None = None) -> dict:
    """Time fwd+bwd+Adam steps of the Whistle-small Conformer at several micro-batch sizes.

    Mirrors CAT's step: log_softmax -> CTCLoss, fp16 autocast + GradScaler, grad-norm clip 2.0
    (cat/ctc/train.py:172-197, cat/shared/manager.py:612-628). `frames` is the per-utterance
    length in 10 ms frames (1000 = 10 s; CV clips are mostly <= 10 s). Inputs are random:
    this measures speed and memory only.
    """
    torch.manual_seed(0)
    model = whistle.build_conformer(vocab, encoder_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    use_cuda = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp and use_cuda)
    ctc = torch.nn.CTCLoss()
    results: dict[int, dict] = {}
    stop = False
    for b in sorted(micro_batches):
        if stop:  # larger sizes would OOM too; don't waste GPU time
            results[b] = {"oom": True, "skipped": True}
            continue
        try:
            if use_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            x = torch.randn(b, frames, 80, device=device)
            lx = torch.full((b,), frames, dtype=torch.long, device=device)
            ly = torch.full((b,), max(1, frames // 40), dtype=torch.int)  # ~2.5 BPE tokens/s
            y = torch.randint(1, vocab, (int(ly.sum()),), dtype=torch.int)
            times = []
            for s in range(steps + 1):  # first step = warm-up, not timed
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp and use_cuda):
                    logits, lo = model(x, lx)
                    lp = torch.log_softmax(logits, -1).transpose(0, 1)
                    loss = ctc(lp.float(), y, lo.cpu().to(torch.int), ly)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if use_cuda:
                    torch.cuda.synchronize()
                if s > 0:
                    times.append(time.time() - t0)
            step_s = sum(times) / len(times)
            results[b] = {
                "step_s": round(step_s, 4),
                "audio_s_per_s": round(b * frames / 100 / step_s, 1),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if use_cuda else None,
                "finite_loss": bool(torch.isfinite(loss).item()),
            }
        except torch.OutOfMemoryError:
            results[b] = {"oom": True}
            opt.zero_grad(set_to_none=True)
            stop = True
        finally:
            x = logits = lp = loss = None  # noqa: F841  (free activations before next size)
            if use_cuda:
                torch.cuda.empty_cache()
    return {"frames": frames, "vocab": vocab, "amp": amp and use_cuda, "by_micro_batch": results}


def drive_io_benchmark(directory: Path, size_mb: int = 256) -> dict:
    """Write+read one file of `size_mb` (checkpoint-like) to measure Drive throughput."""
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"io_bench_{os.getpid()}.bin"
    data = os.urandom(1 << 20)
    t0 = time.time()
    with open(p, "wb") as f:
        for _ in range(size_mb):
            f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tw = time.time() - t0
    t0 = time.time()
    with open(p, "rb") as f:
        while f.read(1 << 24):
            pass
    tr = time.time() - t0
    p.unlink()
    return {"size_mb": size_mb, "write_MBps": round(size_mb / tw, 1), "read_MBps": round(size_mb / tr, 1)}


def checkpoint_roundtrip(directory: Path) -> dict:
    """Save a CAT-style checkpoint (model + optimizer-like state) and load it back."""
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / "roundtrip_test.pt"
    sd = {"model": {"w": torch.randn(256, 256)}, "scheduler": {"lr": 1e-4, "step": 7}, "step": 7}
    torch.save(sd, p)
    back = whistle.load_cat_checkpoint(p)
    ok = torch.equal(back["model"]["w"], sd["model"]["w"]) and back["step"] == 7
    p.unlink()
    return {"pass": bool(ok)}
