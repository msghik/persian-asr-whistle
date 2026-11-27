#!/usr/bin/env python3
"""Check every number in REPORT.md against its source file.

  python scripts/check_report.py [REPORT.md]

Sources: results/results.csv, results/results_rescored.csv, results/analysis_test_fst_fa_clean_best.json,
DATA/prep/{fa,pl}/data_card.json, CAT's Polish readme (paper numbers), src/persian_asr/catrun.py (schedule).
1. Table cells (main results, data card) are parsed and compared with the source values.
2. Numbers in sentences are checked as fragments rebuilt from the sources (same 2-decimal formatting).
3. Every number left in REPORT.md that no check covered is listed as UNSOURCED (constants are allow-listed).
Exit code 1 if anything mismatches or is unsourced.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from persian_asr import catrun  # noqa: E402

DATA = catrun.DATA
f2 = lambda x: f"{float(x):.2f}"
SIZE = {"1h": "1 h", "10h": "10 h", "full": "77.7 h"}


def load():
    res = {}
    for r in csv.DictReader((REPO / "results/results.csv").open()):
        res[(r["system"], r["hours"], r["lm"])] = r
    resc = {}
    for r in csv.DictReader((REPO / "results/results_rescored.csv").open()):
        resc[(r["run_id"], r["set"], r["decoding"])] = r
    ana = json.loads((REPO / "results/analysis_test_fst_fa_clean_best.json").read_text())
    card = json.loads((DATA / "prep/fa/data_card.json").read_text())
    pl_card = json.loads((DATA / "prep/pl/data_card.json").read_text())
    readme = (catrun.CAT_DIR / "egs/cv-lang10/exp/Crosslingual/pl/Multi._phoneme_ft_subword_1h/readme.md").read_text()
    paper_nolm = re.search(r"test_pl %SER [\d.]+ \| %WER ([\d.]+)", readme).group(1)
    return res, resc, ana, card, pl_card, paper_nolm


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    report_f = Path(args[0]) if args else REPO / "REPORT.md"
    text = report_f.read_text()
    res, resc, ana, card, pl_card, paper_nolm = load()
    ok, bad = [], []
    covered: list[str] = []  # report substrings whose numbers are verified

    def check(label, got, exp, span):
        (ok if got == exp else bad).append(f"{label}: report={got} source={exp}")
        covered.append(span)

    # 1a. main results table
    for line in text.splitlines():
        m = re.match(r"\| (S1|S2) \| (1 h|10 h|77\.7 h) \| ([\d.]+) / ([\d.]+) \| ([\d.]+) / ([\d.]+) \| ([\d.]+) \|$", line)
        if not m:
            continue
        sysn, size = m.group(1), {v: k for k, v in SIZE.items()}[m.group(2)]
        r = res[(sysn, size, "fa_clean")]
        z = resc[(r["run_id"], "test", "fa_clean")]
        for i, (lab, col) in enumerate([("greedy WER", r["test_wer_greedy"]), ("greedy CER", r["test_cer_greedy"]),
                                        ("LM WER", r["test_wer_lm"]), ("LM CER", r["test_cer_lm"]),
                                        ("LM WER ZWNJ-ins", z["wer_zwnj_insensitive"])]):
            check(f"table {sysn} {size} {lab}", m.group(3 + i), f2(col), line)
    # 1b. data card table
    cols = ["train_1h", "train_10h", "train_full", "dev", "test"]
    for line in text.splitlines():
        m = re.match(r"\| (utterances|hours|speakers) \| (.+) \|$", line)
        if not m:
            continue
        key = {"utterances": "utts", "hours": "hours", "speakers": "speakers"}[m.group(1)]
        cells = [c.strip() for c in m.group(2).split("|")]
        for c, name in zip(cells, cols):
            v = card["sets"][name][key]
            exp = f"{v:,}" if key != "hours" else str(v)
            check(f"data card {name} {key}", c, exp, line)

    # 2. sentence fragments rebuilt from the sources
    lm = lambda s, h: f2(res[(s, h, "fa_clean")]["test_wer_lm"])
    s2_10 = res[("S2", "10h", "fa_clean")]
    subs = ana["fa_S2_10h_bpe500_s0"]["top30_substitutions"][:3]
    frags = {
        "summary 1h": f"from {lm('S1', '1h')} to {lm('S2', '1h')} (1 h)",
        "summary 10h": f"{lm('S1', '10h')} to {lm('S2', '10h')} (10 h)",
        "summary 77.7h": f"{lm('S1', 'full')} to {lm('S2', 'full')} (77.7 h",
        "LM vocab": f"LM vocabulary {card['lm_clean']['dev']['vocab_size']:,} words",
        "OOV dev": f"OOV rate {card['lm_clean']['dev']['oov_rate_pct']}% (dev)",
        "OOV test": f"{card['lm_clean']['test']['oov_rate_pct']}% (test)",
        "pool hours": f"({card['sets']['train_full']['hours']} h, nested 1 h and 10 h subsets)",
        "schedule 1h": f"{lm('S1mono', '1h')} at 1 h (vs {lm('S1', '1h')})",
        "schedule 10h": f"{lm('S1mono', '10h')} at 10 h (vs {lm('S1', '10h')})",
        "polish greedy": f"greedy test WER {f2(res[('PL', '1h', 'pl_cat')]['test_wer_greedy'])} vs {paper_nolm} in the paper",
        "polish coverage": f"contains {pl_card['lm_cat']['test_sentences_in_lm_text_pct']}% of the test sentences",
        "polish LMs": f"test WER is {f2(res[('PL', '1h', 'pl_cat')]['test_wer_lm'])} with that LM and "
                      f"{f2(res[('PL', '1h', 'pl_clean')]['test_wer_lm'])} with those sentences removed",
        "finding 1": f"S2 at 10 h ({lm('S2', '10h')}) beats S1 at 77.7 h ({lm('S1', 'full')})",
        "pairs": ", ".join(f"<bdi>{a}</bdi> → <bdi>{b}</bdi> ({c})" for a, b, c in subs),
        "ezafe": f"only {ana['fa_S2_10h_bpe500_s0']['ezafe_ref_words']} test words carry a written ezafe marker",
        "OOV finding": f"The {card['lm_clean']['test']['oov_rate_pct']}% test OOV rate",
        "LM WER/CER": f"lowers WER ({f2(s2_10['test_wer_greedy'])} → {f2(s2_10['test_wer_lm'])}) but raises CER "
                      f"({f2(s2_10['test_cer_greedy'])} → {f2(s2_10['test_cer_lm'])})",
        "schedule lr": f"lr {catrun.SCHEDULER['optimizer']['kwargs']['lr']:.0e}".replace("e-05", "e-5"),
        "schedule batch": f"batch {catrun.EFFECTIVE_BATCH}",
        "BPE size": f"BPE-{catrun.BPE_SIZE}",
    }
    for label, frag in frags.items():
        if frag in text:
            ok.append(f"text {label}: '{frag}'")
            covered.append(frag)
        else:
            bad.append(f"text {label}: expected fragment not found: '{frag}'")

    # 3. unsourced numbers
    rest = text
    for c in sorted(covered, key=len, reverse=True):
        rest = rest.replace(c, " ")
    allowed = {"27.0", "11.0", "2026-11-27", "1", "10", "77.7", "4", "3"}  # release ids, date, sizes, "4-gram", "3 best"
    # P1/P0, S1/S2 and "1 h dev subset" are identifiers or setup constants, not results
    nums = re.findall(r"(?<![\w-])\d[\d,]*(?:\.\d+)?(?:-\d+(?:-\d+)?)?(?![\d.])", rest)
    unsourced = [n for n in nums if n not in allowed]

    print(f"checked {len(ok)} values: {len(bad)} mismatches, {len(unsourced)} unsourced numbers")
    for b in bad:
        print("MISMATCH", b)
    for u in unsourced:
        print("UNSOURCED", u)
    if "-v" in sys.argv:
        for o in ok:
            print("ok", o)
    sys.exit(1 if bad or unsourced else 0)


if __name__ == "__main__":
    main()
