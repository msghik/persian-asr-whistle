#!/usr/bin/env python3
"""80-dim Kaldi fbank for prepared sets (mp3 -> ffmpeg 16 kHz mono s16 -> fbank80).

  python scripts/extract_feats.py --prep <data>/prep/fa --sets train_full dev test --nj 44

Features are computed once per union of utterances and written as Kaldi ark/scp
(<prep>/feats/part*.ark, feats_all.scp). Every set dir under --prep that is a subset
(e.g. train_1h, dev_1h) then gets its own feats.scp filtered from feats_all.scp.
Front-end: persian_asr.features.fbank80 (matches CAT cv-lang10; dither 0).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def read_scp(p: Path) -> dict[str, str]:
    return dict(l.rstrip("\n").split("\t", 1) for l in p.open() if l.strip())


def decode_mp3(path: str) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "s16le",
                          "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16)


def work(job):
    idx, items, out_dir = job
    import kaldiio
    import torch
    torch.set_num_threads(1)
    from persian_asr.features import fbank80
    ark = out_dir / f"part{idx:03d}.ark"
    scp = out_dir / f"part{idx:03d}.scp"
    bad = []
    with kaldiio.WriteHelper(f"ark,scp:{ark},{scp}") as w:
        for uid, path in items:
            try:
                pcm = decode_mp3(path)
                if len(pcm) < 400:
                    raise ValueError(f"too short ({len(pcm)} samples)")
                f = fbank80(torch.from_numpy(pcm.astype(np.float32) / 32768.0)).numpy()
                w(uid, f)
            except Exception as e:  # reported, never silently dropped
                bad.append((uid, repr(e)[:200]))
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", required=True, help="sets whose union is extracted")
    ap.add_argument("--nj", type=int, default=44)
    a = ap.parse_args()

    utts: dict[str, str] = {}
    for s in a.sets:
        utts.update(read_scp(a.prep / s / "wav.scp"))
    out = a.prep / "feats"
    out.mkdir(exist_ok=True)
    allscp = out / "feats_all.scp"
    if allscp.exists():
        print(f"{allscp} exists; only re-filtering set scps")
    else:
        items = sorted(utts.items())
        chunks = [items[i::a.nj] for i in range(a.nj)]
        t0 = time.time()
        with Pool(a.nj) as p:
            bad = [b for r in p.imap_unordered(work, [(i, c, out) for i, c in enumerate(chunks)]) for b in r]
        lines = sorted(l for f in sorted(out.glob("part*.scp")) for l in f.open())
        allscp.write_text("".join(lines))
        (out / "failed.tsv").write_text("".join(f"{u}\t{e}\n" for u, e in bad))
        print(f"extracted {len(lines)} / {len(items)} utts in {time.time() - t0:.0f} s; failed: {len(bad)}")
    feats = {l.split(" ", 1)[0]: l for l in allscp.open()}
    for d in sorted(p for p in a.prep.iterdir() if (p / "wav.scp").exists()):
        ids = list(read_scp(d / "wav.scp"))
        have = [feats[u] for u in ids if u in feats]
        if len(have) == len(ids):
            (d / "feats.scp").write_text("".join(have))
            print(f"{d.name}: feats.scp {len(have)} utts")
        else:
            print(f"{d.name}: {len(ids) - len(have)} of {len(ids)} utts have no features (not extracted or failed)")


if __name__ == "__main__":
    main()
