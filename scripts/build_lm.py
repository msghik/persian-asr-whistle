#!/usr/bin/env python3
"""Word 4-gram LMs (CAT's ngram.sh: SimpleTokenizer over a word list + KenLM lmplz -o 4).

  python scripts/build_lm.py fa_clean            # Persian: training-pool text, dev/test sentences removed
  python scripts/build_lm.py pl_cat pl_clean      # Polish reproduction LMs

fa_clean : prep/fa/lm_text/clean.txt; vocabulary = words of that text; no short-sentence pruning
           (CV fa prompts are short; dropping <5-word sentences would bias the LM).
pl_cat   : CAT's convention, to compare with the paper: LM text = whole CAT training text
           (exp/Monolingual/pl/lm/hyper-p.json: train_pl = excluded_train), vocabulary = words of
           ALL text incl. dev/test (local/data_prep.sh stage 3), prune_shorter 5 (as in that config).
pl_clean : Polish training text with dev/test sentences removed; vocabulary from that text;
           prune_shorter 5 (only the text content differs from pl_cat).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from persian_asr import catrun, decoding  # noqa: E402


def text_only(src: Path, dst: Path) -> Path:
    dst.write_text("".join(l.split("\t", 1)[1] + "\n" for l in src.read_text().splitlines() if "\t" in l))
    return dst


def main() -> None:
    P = catrun.DATA / "prep"
    for name in sys.argv[1:]:
        lang, kind = name.split("_", 1)
        lm_text = P / lang / "lm_text" / f"{kind}.txt"
        if kind == "cat":
            vocab_src = [text_only(P / lang / s / "text", P / lang / "lm_text" / f"words_{s}.txt")
                         for s in ("train_full", "dev", "test")]
        else:
            vocab_src = [lm_text]
        d = decoding.build_word_lm(name, lm_text, vocab_src, prune_shorter=5 if lang == "pl" else 0)
        n_words = sum(1 for _ in (d / "word_list").open())
        print(f"{name}: {d / '4gram.arpa'} ({(d / '4gram.arpa').stat().st_size / 1e6:.1f} MB), word list {n_words}")


if __name__ == "__main__":
    main()
