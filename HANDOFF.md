# HANDOFF — brief for the next Claude Code session (phoneme baselines P1/P0)

You are continuing a project run on a UChicago Chameleon RTX 6000 lease (2026-11-25 to 2026-11-29; that machine
and its disk are gone after the lease). This file is the brief. `PROGRESS.md` holds the verified technical state
with CAT file:line references, and `RESULTS_HANDOFF.md` lists every result. `REPORT.md` is the one-page draft for
Prof. Ou. Read all four before doing anything.

## Who and why
Mohammad Sadegh Gholizadeh is applying for a PhD with **Prof. Zhijian Ou** (Tsinghua EE, SPMI Lab).
Prof. Ou asked for two Persian ASR baselines, one subword-based and one phoneme-based, compared.
He wants to see progress before supporting the application.
**Deadline:** a short results report around **2026-12-10**.
**Never invent, estimate or "fill in" a result number. A failed or unfinished run is reported as such.**

## State at handoff (2026-11-27)
- **Code:** GitHub `msghik/persian-asr-whistle`, branch **`worktree-server-pipeline`** (not merged into `main`).
  `scripts/` has one entry point per step; `tests/` passes.
- **Data/models:** Google Drive `gdrive:persian_asr` (rclone remote `gdrive`), written by
  `bash scripts/sync_drive.sh --essentials`. It holds:
  - Final averaged models `exp/<run>/check/best-3.pt` with `hyper-p.json`, `config.json` and `run_info.json`.
  - Decoding hypotheses and scores (`exp/<run>/decode/`).
  - Tokenizers (`tok/`), word LMs (`lm/fa_clean`, `lm/pl_cat`, `lm/pl_clean`).
  - Prepared sets (`prep/{fa,pl}/<set>/{wav.scp,text,meta.tsv}`), data cards, LM text, logs.
  - **Not on Drive** (rebuildable): fbank features, intermediate checkpoints, raw CV audio, Kaldi/KenLM builds.
- **Subword baselines DONE** (S1/S2 at 1h, 10h, 77.7h; Polish reproduction): see `RESULTS_HANDOFF.md`.
  - **S1-mono (done 2026-11-27, synced to Drive):** S1 rerun with CAT's own from-scratch recipe schedule
    (`Mono._subword_10h`/`_1h`), so each mode uses its own recipe.
    - Test LM WER: 84.90 (10h) and 99.24 (1h), vs 83.26 / 99.25 for S1 with the shared schedule.
    - CAT's schedule does not improve S1. Run ids `fa_S1mono_*` in `results/results.csv`; details in PROGRESS.md.
- Decisions taken on the lease (the user delegated them; all documented in PROGRESS.md):
  - **Clean training pool:** no dev/test sentence, normalized text or speaker in training. Whistle's convention leaks
    97.1% of test sentences.
  - **Text source:** `validated_sentences.tsv`, because CV 27.0 fa clip TSVs drop ZWNJ.
  - **Precision:** fp16.
  - **Polish reproduction:** on CV 27.0, since CV 11.0 is unavailable.
  - **LM weight:** tuned on dev only.
  - **CAT patches:** two patches (`scripts/patch_cat.py`), needed with torch 2.14: device mismatch, and a wrong CUDA CTC
    gradient under fp16 autocast.

## Resume on a new machine
1. Survey the machine (GPU/VRAM, driver/CUDA, OS, disk, sudo, internet to HF/GitHub/MDC) and report it to the user.
2. Set up and restore:
   ```bash
   git clone -b worktree-server-pipeline https://github.com/msghik/persian-asr-whistle && cd persian-asr-whistle
   BUILD_KALDI=1 bash scripts/setup_server.sh   # Python 3.12 venv, torch, CAT + patches, KenLM, Kaldi, whistle-small, pytest
   rclone config                                 # user creates remote "gdrive" (Google Drive)
   bash scripts/sync_drive.sh --restore          # -> ~/persian_asr_data
   ```
   - Check the PyTorch build against the new GPU and driver. The lease used torch 2.14.1+cu126 (driver 545, sm_75).
     Keep `tests/test_cat_ctc_patch.py` passing on GPU.
3. Re-download raw audio with the MDC SDK. The key goes in `~/.mdc.env` as `MDC_API_KEY=...`; never print or commit it.
   - Persian `cmu5xazgy00finq074vaqo1b3`, Polish `cmu5wsxwp00e0nq07m8v0274j`; ~2 min each.
   - Extract them so the paths in `prep/*/*/wav.scp` resolve: `~/persian_asr_data/raw/cv27_fa/cv-corpus-27.0-2026-09-11/fa/clips/...`.
4. Rebuild features only if training or re-decoding:
   `python scripts/extract_feats.py --prep ~/persian_asr_data/prep/fa --sets train_full dev test`.
   Train/decode with `scripts/run_queue.sh` (resumable). Set `SYNC=1` to back up to Drive after each run.

## Next work: phoneme baselines (needs Prof. Ou's pipeline)
| ID | Model | Units | Notes |
|---|---|---|---|
| P1 | Whistle-small + phoneme CTC | IPA phonemes | **Add output units** for Persian phonemes missing from Whistle's 75-unit inventory; WFST decoding with Ou's lexicon |
| P0 | Whistle-small architecture from scratch | IPA phonemes | Same units, splits, LM, decoding |

- Use the **same** data, normalization, splits, subsets (1h, 10h, full) and word 4-gram LM as S1/S2. Compare P1/P0 with
  S2/S1 on identical test sets.
- Labels come from Prof. Ou's Persian lexicon + WFST phoneme-label pipeline. **Do not build our own G2P.** Ask the
  user for the pipeline before starting.
- `whistle.make_s2_init` swaps the whole classifier. P1 needs a variant that keeps the 75 Whistle phoneme rows and
  appends new rows for the added units; document which rows are reused.
- Do NOT fine-tune whistle-small-subword. Do NOT start JSA-SPG, lattice training or LLM-P2G.

## Research context
- **Whistle** (arXiv 2406.02166): Conformer-CTC pretrained on G2P IPA phoneme labels for 10 CV v11 languages (CV-Lang10).
  Later work: LLM-P2G (2506.04711), JSA-SPG (2507.06249), Phonemes vs Projectors (2604.09332), Latent Softmax (2608.01281).
- **Code:** https://github.com/thu-spmi/CAT/tree/master/egs/cv-lang10, pinned to `2340970f04` in `src/persian_asr/config.py`.
- **Checkpoints:** HF org **THU-SPMI**, revisions pinned in `config.py`.
- **Long-term topic:** Persian script omits short vowels and the ezafe, so G2P labels are noisy.

## Prof. Ou's instructions (follow exactly)
1. Subword baselines first (done); his Persian pipeline (lexicon + WFST phoneme labels) for the phoneme baselines.
2. Do NOT fine-tune whistle-small-subword.
3. Common Voice Scripted Speech **27.0 – Persian** with its official dev/test splits, via the Mozilla Data Collective.
4. Phoneme baseline: **add output units** for Persian phonemes missing from Whistle's inventory.

## Working rules
- **Plan first:** before writing new code, give the user a short plan with risks and GPU time per run measured on that GPU.
- **Test on CPU** where possible before using GPU time.
- **Keep PROGRESS.md current:** done, failed, numbers, next step, CAT file:line for every CAT finding. Pin seeds; log
  versions and the GPU.
- **Code goes in git; data and checkpoints never do.** Back up to Drive with `scripts/sync_drive.sh --essentials`.
- **Ask the user before:** spending money; downloads over 50 GB; deviating from the experiment matrix; deleting data
  or checkpoints; system-wide changes on a shared server. Pushing to this repo is approved (never commit secrets).
  Pushing to HF is not.
- **If CAT contradicts these instructions,** stop and explain. Read the code rather than guessing.
- **Long jobs** run in tmux and must be resumable (`scripts/run_queue.sh`, CAT `--resume`).
