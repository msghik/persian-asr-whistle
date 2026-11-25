"""Run CAT CTC training (S1 / S2 / Polish reproduction) with one shared schedule.

Layout (all under DATA = $PERSIAN_ASR_DATA, default ~/persian_asr_data):
  DATA/prep/<lang>/<set>/{feats.scp,text}   from scripts/prepare_data.py + extract_feats.py
  DATA/cat_ws/                               CAT workspace: cat, utils symlinks + data/metainfo.json
  DATA/cat_ws/exp -> DATA/exp                experiments: DATA/exp/<run_id>/
  DATA/tok/<lang>_<size>_bpe500.tknz         one tokenizer per (lang, size), shared by S1 and S2

CAT facts this relies on (CAT @2340970f04):
* tokenizer: SentencePiece BPE trained on the run's training text (egs/cv-lang10/utils/
  pipeline/common_utils.py:618-676); we train it once per (lang, size) via CAT's
  tokenizer API and point both systems at the same file, so asr.py stage 1 skips it.
* stage 2 packs feats.scp + text with the length filter (asr.py:226-281); stage 3 calls
  cat.ctc.train via train_nn (common_utils.py:334-400).
* every evaluation saves a full checkpoint (cat/shared/manager.py:374-378) and CAT never
  deletes any; the watcher below keeps the best 3 (by dev loss) + the latest.
* --grad-accum-fold K: one optimizer update per K loader batches, so the effective
  batch is batch_size * K (cat/shared/manager.py:600-670).
"""
from __future__ import annotations

import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config

DATA = Path(os.environ.get("PERSIAN_ASR_DATA", Path.home() / "persian_asr_data"))
CAT_DIR = Path(os.environ.get("CAT_DIR", Path.home() / "persian_asr_local/tools/CAT"))
WHISTLE_SMALL = DATA / "pretrained/whistle-small/best-3.pt"

BPE_SIZE = 500
EFFECTIVE_BATCH = 128
MICRO_BATCH = 32
STEP_CAP = {"10m": 6000, "1h": 6000, "10h": 20000, "full": 40000}
SPECAUG = {"freq_mask_width_range": 0.35, "num_freq_mask": 2, "num_time_mask": 10, "time_mask_width_range": 0.05}
SCHEDULER = {"type": "SchedulerEarlyStop",
             "kwargs": {"min_step": 2000, "n_tol": 5, "gamma": 0.5, "stop_lr": 1e-6},
             "optimizer": {"type": "Adam", "kwargs": {"lr": 3e-5, "betas": [0.9, 0.98], "weight_decay": 1e-6}}}


def ensure_workspace() -> Path:
    ws = DATA / "cat_ws"
    (ws / "data").mkdir(parents=True, exist_ok=True)
    (DATA / "exp").mkdir(parents=True, exist_ok=True)
    for name, target in (("cat", CAT_DIR / "cat"), ("utils", CAT_DIR / "cat/utils"), ("exp", DATA / "exp")):
        link = ws / name
        if not link.exists():
            link.symlink_to(target)
    meta = {}
    for lang_dir in sorted((DATA / "prep").glob("*")):
        for s in sorted(lang_dir.glob("*")):
            if (s / "feats.scp").exists() and (s / "text").exists():
                meta[f"{lang_dir.name}_{s.name}"] = {"scp": str(s / "feats.scp"), "trans": str(s / "text")}
    (ws / "data/metainfo.json").write_text(json.dumps(meta, indent=1))
    return ws


def n_lines(p: Path) -> int:
    with p.open() as f:
        return sum(1 for _ in f)


def ensure_tokenizer(lang: str, train_set: str, tag: str) -> Path:
    """SentencePiece BPE-500 on the training transcripts (ids stripped), CAT's tokenizer API."""
    tok = DATA / "tok" / f"{lang}_{tag}_bpe{BPE_SIZE}.tknz"
    if tok.exists():
        return tok
    tok.parent.mkdir(parents=True, exist_ok=True)
    txt = tok.with_suffix(".txt")
    lines = [l.rstrip("\n").split("\t", 1) for l in (DATA / "prep" / lang / train_set / "text").open()]
    txt.write_text("".join((l[1] if len(l) > 1 else "") + "\n" for l in lines))
    from cat.shared import tokenizer as tknz
    t = tknz.initialize({"type": "SentencePieceTokenizer",
                         "option-train": {"model_type": "bpe", "vocab_size": BPE_SIZE,
                                          "model_prefix": str(tok.with_suffix("")), "f_text": str(txt)}})
    tknz.save(t, str(tok))
    return tok


def write_exp(run_id: str, lang: str, system: str, train_set: str, dev_set: str, tok: Path,
              seed: int, check_freq: int, init_model: Path | None, scheduler: dict, micro_batch: int) -> Path:
    exp = DATA / "exp" / run_id
    exp.mkdir(parents=True, exist_ok=True)
    fold = EFFECTIVE_BATCH // micro_batch
    from cat.shared import tokenizer as tknz
    vocab = tknz.load(str(tok)).vocab_size
    hyper = {
        "data": {"train": f"{lang}_{train_set}", "dev": f"{lang}_{dev_set}", "filter": "10:2000"},
        "tokenizer": {"type": "SentencePieceTokenizer", "|V|": vocab, "file": str(tok)},
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "train": {"bin": "cat.ctc.train",
                  "option": {"amp": True, "batch_size": micro_batch, "grad_accum_fold": fold,
                             "check_freq": check_freq, "seed": seed, "workers": 4}},
        "inference": {"avgmodel": {"mode": "best", "num": 3}},
        "commit": config.CAT_COMMIT,
    }
    if init_model is not None:
        hyper["train"]["option"]["init_model"] = str(init_model)
    enc = dict(config.WHISTLE_SMALL_ENCODER, num_classes=vocab)
    cfg = {"specaug": SPECAUG, "encoder": {"type": "ConformerNet", "kwargs": enc}, "scheduler": scheduler}
    (exp / "hyper-p.json").write_text(json.dumps(hyper, indent=4))
    (exp / "config.json").write_text(json.dumps(cfg, indent=4))
    return exp


def read_checklist(exp: Path) -> list[dict]:
    f = exp / "check/checkpoint.list"
    out = []
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        e, s, m, lr, path = line.split()[:5]
        out.append({"epoch": int(e), "step": int(s, 16), "metric": float.fromhex(m),
                    "lr": float.fromhex(lr), "path": path})
    return out


def latest_checkpoint(exp: Path) -> str | None:
    for c in reversed(read_checklist(exp)):
        p = c["path"] if os.path.isabs(c["path"]) else str(DATA / "cat_ws" / c["path"])
        if os.path.isfile(p):
            return p
    return None


class Watcher(threading.Thread):
    """Prune checkpoints (keep best-3 by dev metric + latest) and enforce the step cap."""

    def __init__(self, exp: Path, proc: subprocess.Popen, step_cap: int, keep_best: int = 3):
        super().__init__(daemon=True)
        self.exp, self.proc, self.cap, self.keep = exp, proc, step_cap, keep_best
        self.capped = False

    def prune(self) -> list[dict]:
        cl = read_checklist(self.exp)
        if not cl:
            return cl
        keep = {c["path"] for c in sorted(cl, key=lambda c: c["metric"])[: self.keep]} | {cl[-1]["path"]}
        for c in cl:
            p = Path(c["path"] if os.path.isabs(c["path"]) else DATA / "cat_ws" / c["path"])
            if c["path"] not in keep and p.exists():
                p.unlink()
        return cl

    def run(self) -> None:
        while self.proc.poll() is None:
            time.sleep(20)
            cl = self.prune()
            if cl and cl[-1]["step"] >= self.cap and not self.capped:
                self.capped = True
                (self.exp / "STEP_CAP_REACHED").write_text(f"step {cl[-1]['step']} >= cap {self.cap}\n")
                os.killpg(self.proc.pid, signal.SIGTERM)
        self.prune()


def env_info() -> dict:
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": platform.node(), "cat_commit": config.CAT_COMMIT}


def run_cat(exp: Path, stages: tuple[int, int], log: Path, step_cap: int) -> dict:
    ws = ensure_workspace()
    hyper = json.loads((exp / "hyper-p.json").read_text())
    resume = latest_checkpoint(exp)
    if resume:
        hyper["train"]["option"]["resume"] = resume
    else:
        hyper["train"]["option"].pop("resume", None)
    (exp / "hyper-p.json").write_text(json.dumps(hyper, indent=4))
    env = dict(os.environ, **config.TORCH_ENV, PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "utils/pipeline/asr.py", f"exp/{exp.name}", "--sta", str(stages[0]), "--sto", str(stages[1])]
    t0 = time.time()
    with log.open("a") as lf:
        lf.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd)} resume={resume}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=ws, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        w = Watcher(exp, proc, step_cap)
        w.start()
        rc = proc.wait()
        w.join()
    return {"returncode": rc, "wall_s": round(time.time() - t0, 1), "resumed_from": resume, "capped": w.capped}


def average_best(exp: Path, num: int = 3) -> Path:
    """CAT's model averaging (utils/avgmodel.py): mean of the `num` best checkpoints by dev metric."""
    ws = ensure_workspace()
    out = exp / "check" / f"best-{num}.pt"
    code = ("import sys; sys.path.insert(0, '.');"
            "from utils.pipeline.common_utils import model_average;"
            f"print(model_average({{'mode': 'best', 'num': {num}}}, 'exp/{exp.name}/check')[0])")
    subprocess.run([sys.executable, "-c", code], cwd=ws, check=True,
                   env=dict(os.environ, **config.TORCH_ENV))
    return out


def steps_per_epoch(n_utts: int) -> int:
    return max(1, math.ceil(n_utts / EFFECTIVE_BATCH))
