#!/usr/bin/env python3
"""Step 6: error analysis of decoded runs (Persian).

  python scripts/analysis.py --run-ids fa_S1_10h_bpe500_s0 fa_S2_10h_bpe500_s0 --set test --hyp fst_fa_clean_best.hyp

Per run (normalized refs vs hyps, jiwer alignment):
* S/I/D counts and WER;
* the 30 most frequent word substitutions (ref -> hyp);
* OOV-word error rate: share of reference words NOT in the LM vocabulary (prep/fa/lm_text/clean.txt)
  that are not recognised correctly (aligned as hit), vs the same for in-vocabulary words;
* ZWNJ boundaries: error rate of reference words containing ZWNJ, and "space/ZWNJ-only" errors
  (a substitution, or a substitution+deletion/insertion pair, that disappears once ZWNJ and spaces
  are removed: e.g. ref "می‌روم" vs hyp "می روم");
* ezafe: error rate of reference words whose raw CV text carried a written ezafe (ۀ, or ه + U+0654),
  located per token (approximate: raw tokens are normalized one by one);
* homograph proxy (no pronunciation lexicon is used, Prof. Ou's lexicon comes later): substitutions
  whose two words become identical after deleting the long-vowel letters ا و ی and ZWNJ, i.e. errors
  between spellings that differ only in vowel letters. This is a flag, not a homograph count.
Writes results/analysis_<set>.md and .json.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persian_asr import catrun, decoding  # noqa: E402
from persian_asr.normalize_fa import ZWNJ, normalize  # noqa: E402

VOWELS_ZW = re.compile("[اوی" + ZWNJ + "]")


def align(ref: str, hyp: str):
    """Yield (op, ref_words, hyp_words) chunks."""
    import jiwer
    if not ref:
        yield ("insert", [], hyp.split())
        return
    o = jiwer.process_words(ref, hyp)
    r, h = o.references[0], o.hypotheses[0]
    for c in o.alignments[0]:
        yield (c.type, r[c.ref_start_idx:c.ref_end_idx], h[c.hyp_start_idx:c.hyp_end_idx])


def ezafe_words(raw: str) -> set[str]:
    out = set()
    for tok in raw.split():
        if tok.rstrip("»«.,،؛؟!:\"").endswith(("ۀ", "هٔ")):
            n = normalize(tok)
            if n:
                out.add(n.split()[-1])
    return out


def analyse(refs, hyps, raws, vocab) -> dict:
    sub = dele = ins = hit = 0
    conf = collections.Counter()
    oov = {"oov": [0, 0], "iv": [0, 0]}  # [errors, total]
    zw = [0, 0]
    zw_only = 0
    ez = [0, 0]
    homo = collections.Counter()
    for u in sorted(refs):
        ref, hyp = refs[u], hyps.get(u, "")
        ezw = ezafe_words(raws.get(u, ""))
        chunks = list(align(ref, hyp))
        # space/ZWNJ-only errors: a maximal run of non-equal chunks whose ref and hyp words are
        # identical once ZWNJ and spaces are removed (jiwer splits "x‌y" -> "x y" into sub + ins)
        region_r: list[str] = []
        region_h: list[str] = []
        for op, rw, hw in chunks + [("equal", [], [])]:
            if op != "equal":
                region_r += rw
                region_h += hw
                continue
            if region_r and "".join(region_r).replace(ZWNJ, "") == "".join(region_h).replace(ZWNJ, ""):
                zw_only += len(region_r)
            region_r, region_h = [], []
        for i, (op, rw, hw) in enumerate(chunks):
            if op == "equal":
                hit += len(rw)
            elif op == "substitute":
                sub += len(rw)
                for a, b in zip(rw, hw):
                    conf[(a, b)] += 1
                    if VOWELS_ZW.sub("", a) == VOWELS_ZW.sub("", b) and a != b:
                        homo[(a, b)] += 1
            elif op == "delete":
                dele += len(rw)
            elif op == "insert":
                ins += len(hw)
            if rw and op != "insert":
                for w in rw:
                    err = op != "equal"
                    k = "iv" if w in vocab else "oov"
                    oov[k][0] += err
                    oov[k][1] += 1
                    if ZWNJ in w:
                        zw[0] += err
                        zw[1] += 1
                    if w in ezw:
                        ez[0] += err
                        ez[1] += 1
    n = hit + sub + dele
    pct = lambda e, t: round(100 * e / t, 2) if t else None
    return {
        "wer": pct(sub + dele + ins, n), "sub": sub, "del": dele, "ins": ins, "n_ref_words": n,
        "top30_substitutions": [[a, b, c] for (a, b), c in conf.most_common(30)],
        "oov_word_error_pct": pct(*oov["oov"]), "oov_ref_words": oov["oov"][1],
        "iv_word_error_pct": pct(*oov["iv"]), "iv_ref_words": oov["iv"][1],
        "zwnj_word_error_pct": pct(*zw), "zwnj_ref_words": zw[1], "space_or_zwnj_only_errors": zw_only,
        "ezafe_word_error_pct": pct(*ez), "ezafe_ref_words": ez[1],
        "vowel_letter_only_substitutions": sum(homo.values()),
        "top10_vowel_letter_only": [[a, b, c] for (a, b), c in homo.most_common(10)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-ids", nargs="+", required=True)
    ap.add_argument("--set", default="test")
    ap.add_argument("--hyp", default="greedy.hyp", help="file under exp/<run>/decode/<set>/")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    a = ap.parse_args()
    P = catrun.DATA / "prep" / "fa"
    refs = decoding.read_text(P / a.set / "text")
    import pandas as pd
    import csv
    meta = pd.read_csv(P / a.set / "meta.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    vs = pd.read_csv(next((catrun.DATA / "raw/cv27_fa").glob("cv-corpus-*/fa")) / "validated_sentences.tsv",
                     sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False).drop_duplicates("sentence_id")
    raws = dict(zip(meta.uid, meta.sentence_id.map(vs.set_index("sentence_id").sentence).fillna("")))
    vocab = {w for l in (P / "lm_text/clean.txt").open() for w in l.split()}
    res = {}
    for rid in a.run_ids:
        hyps = decoding.read_text(catrun.DATA / "exp" / rid / "decode" / a.set / a.hyp)
        res[rid] = analyse(refs, hyps, raws, vocab)
    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"analysis_{a.set}_{Path(a.hyp).stem}"
    (a.out / f"{stem}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    keys = ["wer", "sub", "del", "ins", "oov_word_error_pct", "iv_word_error_pct", "zwnj_word_error_pct",
            "zwnj_ref_words", "space_or_zwnj_only_errors", "ezafe_word_error_pct", "ezafe_ref_words",
            "vowel_letter_only_substitutions"]
    md = [f"# Error analysis ({a.set}, {a.hyp})", "", "| metric | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    md += [f"| {k} | " + " | ".join(str(res[r][k]) for r in res) + " |" for k in keys]
    for r in res:
        md += ["", f"## {r}: 30 most frequent substitutions (ref -> hyp, count)", ""]
        md += [f"{i + 1}. {x} -> {y} ({c})" for i, (x, y, c) in enumerate(res[r]["top30_substitutions"])]
    (a.out / f"{stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:len(keys) + 4]))


if __name__ == "__main__":
    main()
