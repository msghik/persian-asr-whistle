"""Common Voice (MDC release) -> training pool, nested subsets, leakage checks, data card.

Two training-pool conventions:

* ``cat``: CAT cv-lang10 (egs/cv-lang10/local/data_prep.sh stage 1): (validated ∪ train)
  minus dev/test CLIPS. Used only for the Polish reproduction so it matches the paper.
* ``clean``: (validated ∪ train) minus every clip whose SENTENCE or SPEAKER occurs in
  dev or test. Used for Persian: on CV 27.0 fa the ``cat`` pool puts 97.1% of test
  sentences and 53.7% of test clips' speakers into training (measured 2026-11-25).

Dev and test are always CV's official dev.tsv / test.tsv.
Utterance id = clip file name without ".mp3".
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SUBSET_HOURS = {"10m": 1 / 6, "1h": 1.0, "10h": 10.0}


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


@dataclass
class CVRelease:
    root: Path                      # .../cv-corpus-27.0-YYYY-MM-DD/<lang>
    text_from_sentences_tsv: bool   # take text from validated_sentences.tsv via sentence_id

    def clips_dir(self) -> Path:
        return self.root / "clips"

    def durations(self) -> dict[str, float]:
        d = read_tsv(self.root / "clip_durations.tsv")
        return dict(zip(d.iloc[:, 0], d.iloc[:, 1].astype(float) / 1000.0))

    def load(self, split: str, dur: dict[str, float]) -> pd.DataFrame:
        df = read_tsv(self.root / f"{split}.tsv")
        if self.text_from_sentences_tsv:
            src = read_tsv(self.root / "validated_sentences.tsv").drop_duplicates("sentence_id")
            m = df.sentence_id.map(src.set_index("sentence_id").sentence)
            if m.isna().any():
                raise ValueError(f"{split}: {int(m.isna().sum())} sentence_ids missing from validated_sentences.tsv")
            df = df.assign(text_raw=m)
        else:
            df = df.assign(text_raw=df.sentence)
        df = df.assign(uid=df.path.str.replace(r"\.mp3$", "", regex=True),
                       duration=df.path.map(dur).astype(float))
        if df.duration.isna().any():
            raise ValueError(f"{split}: {int(df.duration.isna().sum())} clips without duration")
        return df[["uid", "path", "client_id", "sentence_id", "text_raw", "duration"]]


def build_pool(rel: CVRelease, convention: str, dur: dict[str, float]):
    tr, dv, te, va = (rel.load(s, dur) for s in ("train", "dev", "test", "validated"))
    pool = pd.concat([va, tr]).drop_duplicates("uid")
    ev = pd.concat([dv, te])
    pool = pool[~pool.uid.isin(set(ev.uid))]
    if convention == "clean":
        pool = pool[~pool.sentence_id.isin(set(ev.sentence_id))]
        pool = pool[~pool.client_id.isin(set(ev.client_id))]
    elif convention != "cat":
        raise ValueError(convention)
    return pool.sort_values("uid").reset_index(drop=True), dv.reset_index(drop=True), te.reset_index(drop=True)


def nested_subsets(pool: pd.DataFrame, seed: int, hours: dict[str, float] = SUBSET_HOURS) -> dict[str, pd.DataFrame]:
    """One fixed random order of the pool; each subset is the shortest prefix reaching its
    hours, so 10m ⊂ 1h ⊂ 10h ⊂ full by construction."""
    order = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cum = order.duration.cumsum() / 3600.0
    out = {}
    for name, h in sorted(hours.items(), key=lambda kv: kv[1]):
        n = int(np.searchsorted(cum.values, h) + 1)
        out[name] = order.iloc[:min(n, len(order))]
    out["full"] = order
    return out


def dev_subset(dev: pd.DataFrame, hours: float, seed: int) -> pd.DataFrame:
    order = dev.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = int(np.searchsorted((order.duration.cumsum() / 3600.0).values, hours) + 1)
    return order.iloc[:n]


def leakage(train: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """Share of eval clips whose sentence / normalized text / speaker occurs in train."""
    return {
        "eval_clips": int(len(ev)),
        "sentence_id_in_train_pct": round(100 * ev.sentence_id.isin(set(train.sentence_id)).mean(), 2),
        "norm_text_in_train_pct": round(100 * ev.text.isin(set(train.text)).mean(), 2),
        "speaker_in_train_pct": round(100 * ev.client_id.isin(set(train.client_id)).mean(), 2),
    }


def card_row(df: pd.DataFrame) -> dict:
    return {"utts": int(len(df)), "hours": round(df.duration.sum() / 3600, 2),
            "speakers": int(df.client_id.nunique()), "sentences": int(df.sentence_id.nunique())}


def apply_norm(df: pd.DataFrame, norm: Callable[[str], tuple[str, list[str]]]) -> pd.DataFrame:
    res = [norm(t) for t in df.text_raw]
    return df.assign(text=[r[0] for r in res], flags=[",".join(r[1]) for r in res])


def write_set(df: pd.DataFrame, out: Path, clips_dir: Path) -> None:
    """Kaldi-style dir (wav.scp with mp3 paths, text) plus a full metadata TSV."""
    out.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("uid")
    (out / "wav.scp").write_text("".join(f"{u}\t{clips_dir / p}\n" for u, p in zip(df.uid, df.path)))
    (out / "text").write_text("".join(f"{u}\t{t}\n" for u, t in zip(df.uid, df.text)))
    df[["uid", "path", "client_id", "sentence_id", "duration", "flags", "text"]].to_csv(
        out / "meta.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")


def lm_text(pool: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame) -> tuple[list[str], dict]:
    """LM training sentences: one line per training-pool CLIP (text frequency = reading
    frequency, as in CAT, which trains its LM on the train `text` file), minus any line
    whose normalized text equals a dev or test sentence; empty lines dropped."""
    banned = set(dev.text) | set(test.text)
    lines = [t for t in pool.text if t and t not in banned]
    info = {"lm_lines": len(lines), "lm_lines_removed_as_dev_test": int(sum(t in banned for t in pool.text)),
            "lm_unique_sentences": len(set(lines))}
    return lines, info


def oov_rate(lines: list[str], ev_text: list[str]) -> dict:
    vocab = {w for l in lines for w in l.split()}
    words = [w for t in ev_text for w in t.split()]
    oov = sum(w not in vocab for w in words)
    return {"vocab_size": len(vocab), "eval_words": len(words), "oov_words": oov,
            "oov_rate_pct": round(100 * oov / max(1, len(words)), 2)}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
