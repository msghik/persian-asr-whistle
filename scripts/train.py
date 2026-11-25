#!/usr/bin/env python3
"""Step 4: train one system at one size with CAT (re-running the same command resumes).

  python scripts/train.py --system S2 --lang fa --size 1h            # full run
  python scripts/train.py --system S1 --lang fa --size 1h --sanity   # overfit 50 utts, ~minutes
  python scripts/train.py --system PL --lang pl --size 1h            # Polish reproduction (= S2 recipe)

Systems: S1 = Whistle-small Conformer from scratch; S2 / PL = Whistle-small encoder + new
BPE-500 output layer (persian_asr.whistle.make_s2_init; JSON report next to the init file).
Same schedule for every system at a size (persian_asr.catrun.SCHEDULER, STEP_CAP).
Run inside tmux; afterwards the 3 best checkpoints are averaged to check/best-3.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persian_asr import catrun, whistle  # noqa: E402


def make_sanity_set(lang: str, n: int = 50) -> str:
    src = catrun.DATA / "prep" / lang / "train_1h"
    dst = catrun.DATA / "prep" / lang / f"sanity{n}"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("feats.scp", "text"):
        lines = (src / f).read_text().splitlines(keepends=True)[:n]
        (dst / f).write_text("".join(lines))
    return dst.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["S1", "S2", "PL"], required=True)
    ap.add_argument("--lang", choices=["fa", "pl"], required=True)
    ap.add_argument("--size", choices=["10m", "1h", "10h", "full"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", action="store_true", help="overfit 50 train utts (train = dev), 400-step cap")
    a = ap.parse_args()

    train_set = f"train_{a.size}"
    tok = catrun.ensure_tokenizer(a.lang, train_set, a.size)  # same tokenizer for S1/S2 at this size
    scheduler = json.loads(json.dumps(catrun.SCHEDULER))
    micro = catrun.MICRO_BATCH
    if a.sanity:
        train_set = dev_set = make_sanity_set(a.lang)
        run_id = f"sanity_{a.lang}_{a.system}"
        scheduler["kwargs"].update(min_step=300)
        micro, cap, check_freq = 16, 400, 20
        catrun.EFFECTIVE_BATCH = 16
    else:
        dev_set = "dev_1h"
        run_id = f"{a.lang}_{a.system}_{a.size}_bpe{catrun.BPE_SIZE}_s{a.seed}"
        cap = catrun.STEP_CAP[a.size]
        check_freq = catrun.steps_per_epoch(catrun.n_lines(catrun.DATA / "prep" / a.lang / train_set / "text"))
    catrun.ensure_workspace()

    init = None
    if a.system in ("S2", "PL"):
        init = catrun.DATA / "init" / f"whistle-small_{tok.stem}_s{a.seed}.pt"
        if not init.exists():
            from cat.shared import tokenizer as tknz
            whistle.make_s2_init(catrun.WHISTLE_SMALL, init, tknz.load(str(tok)).vocab_size, seed=a.seed)
    exp = catrun.DATA / "exp" / run_id
    if not (exp / "hyper-p.json").exists():
        catrun.write_exp(run_id, a.lang, a.system, train_set, dev_set, tok, a.seed, check_freq, init, scheduler, micro)
    info_f = exp / "run_info.json"
    info = json.loads(info_f.read_text()) if info_f.exists() else {
        "run_id": run_id, "system": a.system, "lang": a.lang, "size": a.size, "seed": a.seed,
        "train_set": train_set, "dev_set": dev_set, "tokenizer": str(tok), "init_model": str(init) if init else None,
        "effective_batch": catrun.EFFECTIVE_BATCH, "micro_batch": micro, "check_freq": check_freq,
        "step_cap": cap, "scheduler": scheduler, "env": catrun.env_info(), "segments": []}

    if (exp / "check" / "best-3.pt").exists():
        print(f"{run_id}: already finished (check/best-3.pt exists)")
        return
    t0 = time.time()
    seg = catrun.run_cat(exp, (2, 3), exp / "train.log", cap)
    info["segments"].append(seg | {"started": time.strftime("%F %T", time.localtime(t0))})
    cl = catrun.read_checklist(exp)
    finished = seg["returncode"] == 0 or seg["capped"]
    if finished and cl:
        catrun.average_best(exp)
        best = min(cl, key=lambda c: c["metric"])
        info.update(final_step=cl[-1]["step"], n_evals=len(cl), best_dev_loss=best["metric"], best_step=best["step"],
                    stop_reason="step cap" if seg["capped"] else "early stop (CAT TERMINATED)",
                    train_wall_s=round(sum(s["wall_s"] for s in info["segments"]), 1))
    info_f.write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info.get(k) for k in ("run_id", "final_step", "best_step", "best_dev_loss", "stop_reason", "train_wall_s")}))
    if not finished:
        print(f"training did not finish (rc={seg['returncode']}); see {exp / 'train.log'}; re-run the same command to resume")
        sys.exit(1)


if __name__ == "__main__":
    main()
