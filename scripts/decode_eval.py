#!/usr/bin/env python3
"""Step 5: greedy + word-LM (WFST) decoding on dev and test; append a row to results.csv.

  python scripts/decode_eval.py --run-id fa_S2_1h_bpe500_s0 --lm fa_clean
  python scripts/decode_eval.py --run-id pl_PL_1h_bpe500_s0 --lm pl_cat pl_clean

LM weight: swept on DEV only (--lmwts); the dev-best weight is then applied to test.
Hypotheses are normalized with the same normalizer as the references before scoring.
Every LM given gets its own results row. LMs are built by scripts/build_lm.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persian_asr import catrun, decoding  # noqa: E402
from persian_asr.normalize_fa import normalize, normalize_pl_cat  # noqa: E402

NORM = {"fa": normalize, "pl": normalize_pl_cat}
FIELDS = ["run_id", "system", "lang", "units", "hours", "train_set", "gpu", "lm", "lm_weight",
          "dev_wer_greedy", "dev_cer_greedy", "test_wer_greedy", "test_cer_greedy",
          "dev_wer_lm", "dev_cer_lm", "test_wer_lm", "test_cer_lm",
          "bpe", "final_step", "best_step", "epochs_approx", "stop_reason", "train_wall_h", "decode_wall_min", "date"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--lm", nargs="*", default=[], help="LM names under DATA/lm (built by build_lm.py)")
    ap.add_argument("--sets", nargs="+", default=["dev", "test"], help="first set is used to tune the LM weight")
    # Weights >= 1.1 make latgen-faster very slow (S2 fa 1h dev: 0.3-0.7 ~1.5 min, 1.1 17 min, 1.9 ~60 min)
    # and were far from the optimum (0.5); the sweep extends upward only if the best weight is the largest.
    ap.add_argument("--lmwts", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "results.csv")
    a = ap.parse_args()

    exp = catrun.DATA / "exp" / a.run_id
    info = json.loads((exp / "run_info.json").read_text())
    lang = info["lang"]
    norm = NORM[lang]
    tok = Path(info["tokenizer"])
    from cat.shared import tokenizer as tknz
    V = tknz.load(str(tok)).vocab_size
    ckpt = exp / "check" / "best-3.pt"
    t0 = time.time()

    out = {"greedy": {}}
    refs = {}
    for s in a.sets:
        refs[s] = decoding.read_text(catrun.DATA / "prep" / lang / s / "text")
        ark = exp / "decode" / s / "ark"
        decoding.compute_logprobs(ckpt, V, catrun.DATA / "prep" / lang / s / "feats.scp", ark)
        hyp = {u: norm(h) for u, h in decoding.greedy(ark, tok).items()}
        (exp / "decode" / s / "greedy.hyp").write_text("".join(f"{u}\t{hyp[u]}\n" for u in sorted(hyp)))
        out["greedy"][s] = {"wer": decoding.score(refs[s], hyp), "cer": decoding.score(refs[s], hyp, cer=True)}
        print(f"{a.run_id} {s} greedy  WER {out['greedy'][s]['wer']['er']:6.2f}  CER {out['greedy'][s]['cer']['er']:6.2f}")

    rows = []
    for lm in a.lm or [None]:
        row = {"run_id": a.run_id, "system": info["system"], "lang": lang, "units": f"BPE{catrun.BPE_SIZE}",
               "hours": info["size"], "train_set": info["train_set"], "gpu": info["env"]["gpu"], "lm": lm or "",
               "bpe": catrun.BPE_SIZE, "final_step": info.get("final_step"), "best_step": info.get("best_step"),
               "stop_reason": info.get("stop_reason"),
               "train_wall_h": round(info.get("train_wall_s", 0) / 3600, 2), "date": time.strftime("%F")}
        n_train = catrun.n_lines(catrun.DATA / "prep" / lang / info["train_set"] / "text")
        if info.get("final_step"):
            row["epochs_approx"] = round(info["final_step"] * info["effective_batch"] / n_train, 1)
        for s in a.sets:
            if f"{s}_wer_greedy" not in FIELDS:
                continue
            row[f"{s}_wer_greedy"] = out["greedy"][s]["wer"]["er"]
            row[f"{s}_cer_greedy"] = out["greedy"][s]["cer"]["er"]
        if lm:
            lm_dir = catrun.DATA / "lm" / lm
            graph = decoding.build_graph(tok, lm_dir, catrun.DATA / "graph" / f"{tok.stem}__{lm}")
            sweep = {}
            weights = list(a.lmwts)
            while weights:
                w = weights.pop(0)
                tune = a.sets[0]  # the LM weight is tuned on the FIRST set only (dev)
                hyp = {u: norm(h) for u, h in decoding.fst_decode(exp / "decode" / tune / "ark", graph,
                                                                    exp / "decode" / tune / f"fst_{lm}", w).items()}
                sweep[w] = decoding.score(refs[tune], hyp)["er"]
                print(f"  {lm} {tune} lmwt {w}: WER {sweep[w]:.2f}")
                if not weights and min(sweep, key=sweep.get) == max(sweep) and w < 2.0:
                    weights.append(round(w + 0.2, 1))  # optimum at the edge: extend the sweep
            best_w = min(sweep, key=sweep.get)
            row["lm_weight"] = best_w
            for s in a.sets:
                hyp = {u: norm(h) for u, h in decoding.fst_decode(exp / "decode" / s / "ark", graph,
                                                                    exp / "decode" / s / f"fst_{lm}", best_w).items()}
                (exp / "decode" / s / f"fst_{lm}_best.hyp").write_text("".join(f"{u}\t{hyp[u]}\n" for u in sorted(hyp)))
                w_ = decoding.score(refs[s], hyp)
                c_ = decoding.score(refs[s], hyp, cer=True)
                if f"{s}_wer_lm" in FIELDS:
                    row[f"{s}_wer_lm"], row[f"{s}_cer_lm"] = w_["er"], c_["er"]
                out.setdefault(f"lm_{lm}", {})[s] = {"lmwt": best_w, "wer": w_, "cer": c_, "dev_sweep": sweep}
                print(f"{a.run_id} {s} {lm} (lmwt {best_w}, tuned on dev)  WER {w_['er']:6.2f}  CER {c_['er']:6.2f}")
        row["decode_wall_min"] = round((time.time() - t0) / 60, 1)
        rows.append(row)

    (exp / "decode" / "scores.json").write_text(json.dumps(out, indent=1))
    a.results.parent.mkdir(parents=True, exist_ok=True)
    new = not a.results.exists()
    with a.results.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"appended {len(rows)} row(s) to {a.results}")


if __name__ == "__main__":
    main()
