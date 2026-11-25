#!/usr/bin/env python3
"""Re-score every decoded run from its saved hypotheses: strict and ZWNJ-insensitive WER/CER.

  python scripts/rescore.py            # -> results/results_rescored.csv (+ printed table)

strict          : as decode_eval (the normalized references keep ZWNJ as a word-internal character).
zwnj-insensitive: ZWNJ -> space in BOTH reference and hypothesis before scoring. Needed because the fa 1h/10h
                  BPE tokenizers (SentencePiece nmt_nfkc) turn ZWNJ into a space, so those models can never
                  output ZWNJ; tokenizers with the "_id" suffix keep it. This is the comparable metric across sizes.
For CER the two coincide except for ZWNJ characters themselves (CER removes whitespace; ZWNJ is not whitespace).
LM hypotheses are the dev-tuned ones written by decode_eval (decode/<set>/fst_<lm>_best.hyp).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persian_asr import catrun, decoding  # noqa: E402
from persian_asr.normalize_fa import ZWNJ  # noqa: E402


def zs(d: dict[str, str]) -> dict[str, str]:
    return {k: " ".join(v.replace(ZWNJ, " ").split()) for k, v in d.items()}


def main() -> None:
    rows = []
    for exp in sorted((catrun.DATA / "exp").glob("*")):
        f = exp / "run_info.json"
        if not f.exists() or not (exp / "decode/scores.json").exists():
            continue
        info = json.loads(f.read_text())
        lang = info["lang"]
        for s in ("dev", "test"):
            ref = decoding.read_text(catrun.DATA / "prep" / lang / s / "text")
            for hyp_f in sorted((exp / "decode" / s).glob("*.hyp")):
                if hyp_f.name != "greedy.hyp" and not hyp_f.name.endswith("_best.hyp"):
                    continue
                hyp = decoding.read_text(hyp_f)
                r = {"run_id": exp.name, "system": info["system"], "lang": lang, "hours": info["size"],
                     "tokenizer": Path(info["tokenizer"]).name, "set": s,
                     "decoding": "greedy" if hyp_f.name == "greedy.hyp" else hyp_f.name[len("fst_"):-len("_best.hyp")],
                     "wer": decoding.score(ref, hyp)["er"], "cer": decoding.score(ref, hyp, cer=True)["er"]}
                if lang == "fa":
                    r["wer_zwnj_insensitive"] = decoding.score(zs(ref), zs(hyp))["er"]
                    r["cer_zwnj_insensitive"] = decoding.score(zs(ref), zs(hyp), cer=True)["er"]
                rows.append(r)
    out = Path(__file__).resolve().parents[1] / "results" / "results_rescored.csv"
    fields = ["run_id", "system", "lang", "hours", "tokenizer", "set", "decoding", "wer", "cer",
              "wer_zwnj_insensitive", "cer_zwnj_insensitive"]
    with out.open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("  ".join(str(r.get(k, "")) for k in ("run_id", "set", "decoding", "wer", "cer", "wer_zwnj_insensitive", "cer_zwnj_insensitive")))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
