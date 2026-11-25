"""Log-probs, greedy CTC decoding, CAT word-LM + WFST decoding, and CAT-compatible scoring.

* Scoring follows CAT egs/cv-lang10/utils/wer.py:38-66,150-161: jiwer word alignment;
  consecutive spaces collapsed and stripped; CER = remove all whitespace, then align
  characters. SER = share of utterances with any error.
* Log-probs: CAT's cal_logit (cat/ctc/cal_logit.py:85-104) stores raw encoder outputs
  computed on CPU with batch size 1. We compute the same network on GPU in batches and
  store log_softmax. For WFST decoding this is equivalent: every CTC path consumes each
  frame exactly once, so the per-frame log-normalizer adds the same constant to all paths.
* WFST decoding: CAT's own scripts (utils/tool/build_decoding_graph.sh, cat/ctc/fst_decode.sh)
  with the n-gram from CAT's utils/pipeline/ngram.sh (KenLM lmplz -o 4, no pruning).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import catrun, config

_SP = re.compile(r" {2,}")


def _prep(s: str, cer: bool) -> str:
    s = _SP.sub(" ", s).strip("\n ")
    if cer:
        s = " ".join(re.sub(r"\s+", "", s))
    return s


def score(refs: dict[str, str], hyps: dict[str, str], cer: bool = False) -> dict:
    """CAT-compatible WER/CER over the reference ids (missing hyps count as empty)."""
    import jiwer
    ids = sorted(refs)
    gt = [_prep(refs[u], cer) for u in ids]
    hy = [_prep(hyps.get(u, ""), cer) for u in ids]
    sub = dele = ins = hit = 0
    for g, h in zip(gt, hy):  # per utt: jiwer rejects empty references
        if not g:
            ins += len(h.split())
            continue
        o = jiwer.process_words(g, h)
        sub, dele, ins, hit = sub + o.substitutions, dele + o.deletions, ins + o.insertions, hit + o.hits
    n = hit + sub + dele
    return {"er": round(100 * (sub + dele + ins) / max(1, n), 2), "sub": sub, "del": dele, "ins": ins,
            "n_ref": n, "ser": round(100 * sum(g != h for g, h in zip(gt, hy)) / max(1, len(ids)), 2)}


def read_text(p: Path) -> dict[str, str]:
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            k, _, v = line.partition("\t") if "\t" in line else line.partition(" ")
            out[k] = v.strip()
    return out


def compute_logprobs(ckpt: Path, num_classes: int, feats_scp: Path, out_dir: Path, nj: int = 16,
                     max_frames: int = 60000) -> int:
    """GPU forward of the trained encoder; writes out_dir/decode.{1..nj}.ark of log-probs."""
    import kaldiio
    import torch
    from .whistle import load_encoder_for_inference

    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob("decode.*.ark"))) == nj:
        return -1
    model = load_encoder_for_inference(ckpt, num_classes=num_classes, device="cuda").eval()
    feats = kaldiio.load_scp(str(feats_scp))
    lens = {u: feats[u].shape[0] for u in feats}
    order = sorted(lens, key=lens.get, reverse=True)
    shards: list[dict] = [{} for _ in range(nj)]
    i = k = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        while i < len(order):
            b = max(1, max_frames // lens[order[i]])
            ids = order[i:i + b]
            i += b
            xs = [torch.from_numpy(feats[u]) for u in ids]
            x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).cuda()
            xl = torch.tensor([len(t) for t in xs], device="cuda")
            logits, ol = model(x, xl)
            lp = torch.log_softmax(logits.float(), -1).cpu().numpy()
            for j, u in enumerate(ids):
                shards[k % nj][u] = lp[j, : int(ol[j])].astype(np.float32)
                k += 1
    for n, sh in enumerate(shards):
        kaldiio.save_ark(str(out_dir / f"decode.{n + 1}.ark"), sh)
    return len(order)


def greedy(ark_dir: Path, tokenizer_file: Path) -> dict[str, str]:
    import kaldiio
    from cat.shared import tokenizer as tknz
    tk = tknz.load(str(tokenizer_file))
    hyps = {}
    for ark in sorted(ark_dir.glob("decode.*.ark")):
        for u, lp in kaldiio.load_ark(str(ark)):
            ids = lp.argmax(-1)
            keep = [int(t) for t, prev in zip(ids, np.concatenate([[-1], ids[:-1]])) if t != prev and t != 0]
            hyps[u] = tk.decode(keep) if keep else ""
    return hyps


# ---------------------------------------------------------------- word LM + WFST (Kaldi)

def kaldi_env() -> dict:
    kaldi = Path(os.environ.get("KALDI_ROOT", Path.home() / "persian_asr_local/tools/kaldi"))
    kb = [kaldi / "src" / d for d in ("bin", "fstbin", "latbin", "lmbin")] + [kaldi / "tools/openfst/bin"]
    kenlm = Path.home() / "persian_asr_local/tools/kenlm/bin"
    venv_bin = Path(sys.executable).parent  # CAT's shell scripts call plain `python`
    path = ":".join(str(p) for p in [venv_bin, catrun.CAT_DIR / "src/bin", kenlm, *kb]) + ":" + os.environ["PATH"]
    return dict(os.environ, **config.TORCH_ENV, KALDI_ROOT=str(kaldi), PATH=path, LC_ALL="C.UTF-8")


def build_word_lm(name: str, lm_text: Path, word_list_texts: list[Path], prune_shorter: int) -> Path:
    """CAT word-level 4-gram (utils/pipeline/ngram.sh): SimpleTokenizer over a word list,
    lmplz -o 4 on word indices, ARPA output. Returns the LM dir (4gram.arpa, tokenizer)."""
    ws = catrun.ensure_workspace()
    d = catrun.DATA / "lm" / name
    d.mkdir(parents=True, exist_ok=True)
    words = sorted({w for p in word_list_texts for l in p.read_text().splitlines() for w in l.split()})
    (d / "word_list").write_text("".join(w + "\n" for w in words))
    txt = d / "lm_train.text"
    txt.write_text("".join(f"lm{i:07d}\t{l}" + ("" if l.endswith("\n") else "\n")
                           for i, l in enumerate(lm_text.read_text().splitlines(keepends=True)) if l.strip()))
    meta = json.loads((ws / "data/metainfo.json").read_text())
    meta[f"lm_{name}"] = {"scp": "", "trans": str(txt)}
    (ws / "data/metainfo.json").write_text(json.dumps(meta, indent=1))
    hyper = {"data": {"train": [f"lm_{name}"], "packing-text-lm": {"nj": 4, "prune_shorter": prune_shorter}},
             "tokenizer": {"type": "SimpleTokenizer", "option-init": {"dmap": str(d / "word_list")},
                           "file": str(d / "tokenizer_lm.tknz")},
             "commit": config.CAT_COMMIT}
    (d / "hyper-p.json").write_text(json.dumps(hyper, indent=4))
    # ngram.sh stage 3 reads decoder settings from config.json (as exp/Monolingual/pl/lm/config.json)
    (d / "config.json").write_text(json.dumps({"decoder": {"type": "NGram", "kwargs": {
        "f_binlm": str(d / "4gram.arpa"), "gram_order": 4, "num_classes": len(words) + 2}}}, indent=4))
    if not (d / "4gram.arpa").exists():
        rel = os.path.relpath(d, ws)
        subprocess.run(["bash", "utils/pipeline/ngram.sh", rel, "-o", "4", "--arpa", "--output",
                        f"{rel}/4gram.arpa", "--stop-stage", "3"], cwd=ws, env=kaldi_env(), check=True)
    return d


def build_graph(am_tokenizer: Path, lm_dir: Path, out: Path) -> Path:
    ws = catrun.ensure_workspace()
    if not (out / "TLG.fst").exists():
        subprocess.run(["bash", "utils/tool/build_decoding_graph.sh", "--word_list", str(lm_dir / "word_list"),
                        str(am_tokenizer), str(lm_dir / "tokenizer_lm.tknz"), str(lm_dir / "4gram.arpa"), str(out)],
                       cwd=ws, env=kaldi_env(), check=True)
    return out


def fst_decode(ark_dir: Path, graph: Path, out_dir: Path, lmwt: float, acwt: float = 1.0, wip: float = 0.0) -> dict[str, str]:
    ws = catrun.ensure_workspace()
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["bash", "cat/ctc/fst_decode.sh", "-f", "--acwt", str(acwt), "--lmwt", str(lmwt),
                        "--wip", str(wip), str(ark_dir), str(graph), str(out_dir)],
                       cwd=ws, env=kaldi_env(), check=True, capture_output=True, text=True)
    f_out = out_dir / f"ac{acwt}_lm{lmwt}_wip{wip}.hyp"
    if not f_out.exists():
        raise RuntimeError(f"fst_decode produced no {f_out}: {r.stdout[-500:]} {r.stderr[-1500:]}")
    return {k: v for k, v in read_text(f_out).items()}
