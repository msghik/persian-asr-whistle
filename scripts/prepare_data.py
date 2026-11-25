#!/usr/bin/env python3
"""Step 3 (and the data part of step 2): Common Voice release -> sets, subsets, LM text, data card.

  python scripts/prepare_data.py --lang fa --cv-root <.../cv-corpus-27.0-*/fa> --out <data>/prep/fa
  python scripts/prepare_data.py --lang pl --cv-root <.../cv-corpus-27.0-*/pl> --out <data>/prep/pl

fa: text from validated_sentences.tsv (restores ZWNJ), normalize_fa, "clean" pool.
pl: clip-TSV text, CAT's Polish normalization, CAT's "cat" pool (to match the paper).
Prints a short summary to paste; full details in <out>/data_card.{json,md}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persian_asr import cvdata  # noqa: E402
from persian_asr.normalize_fa import normalize_pl_cat, normalize_with_flags  # noqa: E402

LANGS = {
    "fa": dict(norm=normalize_with_flags, convention="clean", text_from_sentences_tsv=True),
    "pl": dict(norm=lambda t: (normalize_pl_cat(t), [] if normalize_pl_cat(t) else ["empty"]),
               convention="cat", text_from_sentences_tsv=False),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=LANGS, required=True)
    ap.add_argument("--cv-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-subset-hours", type=float, default=1.0)
    a = ap.parse_args()
    L = LANGS[a.lang]

    rel = cvdata.CVRelease(a.cv_root, L["text_from_sentences_tsv"])
    dur = rel.durations()
    pool, dev, test = cvdata.build_pool(rel, L["convention"], dur)
    pool, dev, test = (cvdata.apply_norm(d, L["norm"]) for d in (pool, dev, test))

    empty = int((pool.text == "").sum())
    pool = pool[pool.text != ""]  # an empty transcript cannot be a CTC target; counted in the card
    text_dup = 0
    if L["convention"] == "clean":
        # CV has duplicate sentences under different sentence_ids (often differing only in
        # punctuation): also drop pool clips whose NORMALIZED text equals a dev/test sentence.
        banned = set(dev.text) | set(test.text)
        text_dup = int(pool.text.isin(banned).sum())
        pool = pool[~pool.text.isin(banned)]
    subsets = cvdata.nested_subsets(pool, a.seed)
    dev1 = cvdata.dev_subset(dev, a.dev_subset_hours, a.seed)

    sets = {f"train_{k}": v for k, v in subsets.items()} | {"dev": dev, "test": test, f"dev_{int(a.dev_subset_hours)}h": dev1}
    for name, df in sets.items():
        cvdata.write_set(df, a.out / name, rel.clips_dir())

    card = {"lang": a.lang, "cv_root": str(a.cv_root), "convention": L["convention"], "seed": a.seed,
            "text_source": "validated_sentences.tsv via sentence_id" if L["text_from_sentences_tsv"] else "clip TSV sentence",
            "pool_empty_after_norm_dropped": empty, "pool_dropped_norm_text_equals_dev_test": text_dup,
            "sets": {k: cvdata.card_row(v) for k, v in sets.items()},
            "flags": {k: v["flags"].str.split(",").explode().value_counts().drop("", errors="ignore").to_dict()
                      for k, v in sets.items() if k in ("train_full", "dev", "test")},
            "leakage": {f"{tr}->{ev}": cvdata.leakage(sets[tr], sets[ev])
                        for tr in ("train_1h", "train_10h", "train_full") for ev in ("dev", "test")}}

    lm_dir = a.out / "lm_text"
    lm_dir.mkdir(parents=True, exist_ok=True)
    lines, info = cvdata.lm_text(subsets["full"], dev, test)
    (lm_dir / "clean.txt").write_text("".join(l + "\n" for l in lines))
    card["lm_clean"] = info | {"dev": cvdata.oov_rate(lines, list(dev.text)), "test": cvdata.oov_rate(lines, list(test.text))}
    if L["convention"] == "cat":
        cat_lines = [t for t in subsets["full"].text if t]  # CAT trains the LM on the whole train `text`
        (lm_dir / "cat.txt").write_text("".join(l + "\n" for l in cat_lines))
        card["lm_cat"] = {"lm_lines": len(cat_lines), "test_sentences_in_lm_text_pct":
                          round(100 * test.text.isin(set(cat_lines)).mean(), 2),
                          "test": cvdata.oov_rate(cat_lines, list(test.text))}
    cvdata.save_json(card, a.out / "data_card.json")

    md = [f"# Data card: CV 27.0 {a.lang} ({L['convention']} pool, seed {a.seed})", "",
          "| set | utts | hours | speakers | sentences |", "|---|---|---|---|---|"]
    md += [f"| {k} | {v['utts']} | {v['hours']} | {v['speakers']} | {v['sentences']} |" for k, v in card["sets"].items()]
    md += ["", "| leakage (train->eval) | sentence_id % | norm. text % | speaker % |", "|---|---|---|---|"]
    md += [f"| {k} | {v['sentence_id_in_train_pct']} | {v['norm_text_in_train_pct']} | {v['speaker_in_train_pct']} |"
           for k, v in card["leakage"].items()]
    (a.out / "data_card.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print("\nLM (clean):", {k: card["lm_clean"][k] for k in ("lm_lines", "lm_unique_sentences", "lm_lines_removed_as_dev_test")},
          "| dev OOV %:", card["lm_clean"]["dev"]["oov_rate_pct"], "| test OOV %:", card["lm_clean"]["test"]["oov_rate_pct"],
          "| vocab:", card["lm_clean"]["dev"]["vocab_size"])
    if "lm_cat" in card:
        print("LM (CAT convention):", card["lm_cat"])
    print("flags:", card["flags"], "| empty-after-norm dropped from pool:", empty)


if __name__ == "__main__":
    main()
