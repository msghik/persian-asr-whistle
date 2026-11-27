# PROGRESS

Deadline: short results report to Prof. Zhijian Ou around **2026-12-10**.
Rule: no number is reported unless a run produced it; failed runs are reported as failed.

## 2026-11-25: moved off Colab
Colab setup failed in the CAT install cell (the KenLM wheel build; fix pushed in 4c7aedc but never confirmed on Colab).
The user reserved an RTX 6000 server at UChicago; work continues there with Claude Code running on that machine.
See `HANDOFF.md` for the full brief. The Colab notebook stays in the repo for reference; server entry points replace it.

## 2026-11-25: UChicago server (Chameleon bare-metal lease, ends 2026-11-29; cannot be extended)

**Machine (surveyed):** 1x Quadro RTX 6000 24 GB (Turing, sm_75, no bf16), driver 545.23.08 / CUDA 12.3, Ubuntu 22.04,
2x Xeon Gold 6126 (48 threads), 187 GB RAM, one 223 GB SSD (no scratch; disk is wiped when the lease ends), passwordless sudo,
no conda/Slurm/modules. HF, GitHub, PyPI, MDC reachable (HF ~100 MB/s, MDC ~140 MB/s).
**Persistence:** code -> GitHub branch `worktree-server-pipeline`; data/checkpoints -> Google Drive via `scripts/sync_drive.sh`
(rclone remote `gdrive`, user sets it up). Raw CV is not copied (MDC re-download takes ~2 min).

**Environment (verified):** Python 3.12 venv via uv (MDC SDK `datacollective` 0.8.1 needs >= 3.11), torch 2.14.1+cu126
(runs on driver 545, includes sm_75), CAT @2340970f04 + stubs + `scripts/patch_cat.py`, KenLM bins, Kaldi minimal build
(`scripts/build_kaldi_min.sh`: **built OK in 26 min with 46 jobs, 443 MB after shrink**; first confirmed run of the final script).
`pytest`: all pass (27 old + 16 new), incl. whistle-small English->IPA on CPU and on GPU (identical output).
Full recipe: `scripts/setup_server.sh`.

**GPU benchmark (random input, fp16, whistle-small Conformer, V=500):** bs128 x 5 s: 0.58 s/step, 16.4 GB;
bs64 x 10 s: 0.66 s/step, 17.0 GB; bs96 x 10 s: OOM. Real CAT training (S2 fa 1h, 32x4 accumulation): **1.25 s per update
incl. per-epoch dev eval** (measured once the CPU load dropped).

### New CAT findings (file:line at CAT 2340970f04)
10. **CTC loss crashes on torch 2.14:** `cat/ctc/train.py:176-178,195` moves labels/lengths to CPU and calls `nn.CTCLoss`
    with CUDA log-probs; torch 2.14 raises "Expected all tensors to be on the same device" (verified with a minimal test).
    Fixed by `scripts/patch_cat.py` (moves them to the logits' device; user approved).
10b. **That first fix exposed a wrong CUDA CTC gradient** (torch 2.14 native kernel, called as CAT calls it: transposed
    non-contiguous log-probs inside fp16 autocast): vs the CPU reference, max abs grad error up to 1.77, norm x120, and
    non-deterministic garbage on padded frames (t >= input length). The first S2 fa 1h run trained with it: dev loss
    fell 30.7 -> 6.78 (step 112) then ROSE to 11.5 (step 304); the reference Polish 1h curve plateaus ~5-6 then falls to
    1.44. Run moved to `exp_invalid/` (not deleted). Patch 2: fp32 + padded frames zeroed via masked_fill + contiguous +
    autocast off -> grads match CPU to 2.75e-6, padded grads exactly 0, deterministic (`tests/test_cat_ctc_patch.py`).
11. **CAT never deletes checkpoints:** every evaluation saves model+optimizer (~1 GB) (`cat/shared/manager.py:374-378`).
    `catrun.Watcher` keeps the best 3 by dev loss + the latest (verified on the sanity run).
12. **DataLoader workers are re-created every epoch:** cost ~30 s per epoch on the loaded box (measured); we use workers=0.
13. **Polish paper LM contains the test sentences:** the 4-gram is trained on `train_pl` = `data/pl/excluded_train`
    (`data/metainfo.json`, `exp/Monolingual/pl/lm/hyper-p.json`), i.e. validated ∪ train minus dev/test *clips*
    (`local/data_prep.sh` stage 1, expect.py), and the LM word list includes all dev/test words (`local/data_prep.sh` stage 3).
    On CV 27.0 pl the same construction puts **73.3% of test sentences into the LM text** (test OOV 2.09% vs 13.70% clean).
    This plausibly explains most of 69.50% -> 8.63% (to be measured: PL 1h decoded with `pl_cat` vs `pl_clean`).
14. `cat/ctc/cal_logit.py:85-104` stores raw logits (CPU, batch 1); we store GPU log_softmax (equivalent for WFST decoding).
15. `prune_shorter: 5` in the Polish LM config drops LM sentences with < 5 words (`utils/data/pack_corpus.py:94`);
    kept for pl LMs, not used for the Persian LM (short CV prompts).

### Data findings (CV 27.0, from MDC)
- **CV 11.0 Polish is not on MDC** (only 27.0) and the HF CV 11.0 mirrors return 401 -> Polish reproduction on CV 27.0 pl.
- **CV 27.0 fa clip TSVs have ZWNJ deleted** from the sentence column ("می‌شد" -> "میشد"; 15,684 recorded sentences,
  2,273 in test); `validated_sentences.tsv` keeps it. We take text from there via sentence_id.
- **Whistle's train convention leaks on fa:** (validated ∪ train) minus dev/test clips puts 97.1% of test sentences and
  53.7% of test clips' speakers into training. Decision (user delegated): **clean pool** = remove clips whose sentence_id,
  normalized text or speaker occurs in dev/test. Data card (`prep/fa/data_card.md`):

| set | utts | hours | speakers |
|---|---|---|---|
| train_10m / 1h / 10h | 169 / 1005 / 10080 | 0.17 / 1.0 / 10.0 | 65 / 135 / 214 |
| train_full | 78005 | 77.7 | 261 |
| dev / test | 10755 / 10755 | 12.64 / 14.77 | 993 / 2850 |

  Leakage of every train subset into dev/test: 0.0% (sentence_id, normalized text, speaker).
  Word LM `fa_clean`: 78,005 lines, vocab 21,052, dev OOV 7.82%, test OOV 10.38%. Latin-script utts flagged (kept):
  train 29, dev 10, test 14; no digit utterances.
- Polish (CAT pool): train_1h 799 utts / 236 spk; dev 10,096; test 10,096. Features: fa 99,515/99,515, pl 20,991/20,991, 0 failures.

### Schedule (identical for S1/S2/PL at a size; user delegated the choice)
lr 3e-5 Adam(0.9,0.98) wd 1e-6, effective batch 128 (32x4), fp16, SpecAugment as in the recipe, filter 10:2000 frames,
eval every epoch on fixed dev_1h, SchedulerEarlyStop(min_step 2000, n_tol 5, gamma 0.5, stop_lr 1e-6), step cap 6k (1h) /
20k (10h), average best 3. Tokenizer BPE 500 per (lang, size), shared by S1/S2.

### Runs
- sanity (S2, 50 utts, lr 3e-5, 400 steps): pipeline ran end to end (pack, train, prune, average, GPU log-probs, greedy,
  LM, TLG, Kaldi lattice decoding), but the model was still in the CTC blank phase (best dev loss 6.66 at step 120;
  47/50 empty hyps; WER 100%). Not a valid overfit test; the real 1h run is used as the check instead.
- fa_S2_1h (first attempt): INVALID, trained with the wrong CUDA CTC gradient (finding 10b); restarted after patch 2.
- fa_S2_1h (restarted 09:39 with patch 2): dev_1h loss 30.7 -> 4.67 (step 304) -> 2.53 (1216) -> 2.23 (1656), still
  improving; shape matches the reference Polish curve. Diagnostic only (not a result): greedy on dev_1h with the
  step-1216 checkpoint gave WER 77.99 / CER 36.22, hypotheses are real Persian text.
- **fa_S2_1h** (done): early stop at step 2248 (best 2016, dev_1h loss 2.12), 53 min train. Test WER/CER greedy
  76.42/32.03, fa_clean LM (w 0.4, tuned on dev) 61.80/35.56; dev greedy 71.80/28.76, LM 53.92/30.32.
  LM lowers WER but raises CER vs greedy (to examine in analysis: OOV words replaced by in-vocabulary words?).
- **fa_S1_1h** (done): never left the blank phase: best dev_1h loss 5.73 at step 320, early stop at 2232. Test WER/CER
  greedy 99.23/94.69, LM 99.25/94.08. Consistent with CAT's Polish mono subword 1h (98.38% WER).
- **pl_PL_1h** (Polish reproduction, CV 27.0 pl, CAT pool): early stop at 2261 (best 2240, dev_1h loss 1.40; paper curve
  min 1.44), 60 min. Test greedy WER/CER 66.44/17.81 (paper: 69.50 with beam 16, no LM). LM `pl_cat` (paper construction,
  73.3% of test sentences in LM text) 15.16/7.88 at w 0.8; `pl_clean` 42.11/19.06 at w 0.7 (paper: 8.63 at w 1.2).
  pl_cat's dev sweep tied at the top of the grid (0.8 = 0.9 = 21.10) -> edge rule fixed to extend on ties; re-decoded:
  w 1.1 gives dev 21.98 (worse), so w 0.8 and the numbers above are final.
- **fa_S2_10h** (done): early stop at 8690 (best 7505, dev_1h loss 0.856), 2.28 h. Test WER/CER greedy 50.56/15.80,
  fa_clean LM (w 0.6) 38.85/16.86; dev greedy 43.11/12.43, LM 29.21/11.91.
- **Tokenizer drops ZWNJ (finding 16):** CAT's SentencePiece tokenizer uses SentencePiece's default nmt_nfkc
  normalization, which turns ZWNJ into a space (`می‌روم` -> `می روم`; 0 of 500 pieces contain ZWNJ). The fa 1h/10h
  models therefore never output ZWNJ; greedy error on the 3,015 test reference words with ZWNJ is 100% (S2 1h and 10h).
  S1 and S2 share the tokenizer, so their comparison is unaffected. `scripts/rescore.py` adds a ZWNJ-insensitive
  WER/CER (ZWNJ -> space on both sides): S2 10h test greedy 50.56 -> 44.55, LM 38.85 -> 35.10; S2 1h test LM
  61.80 -> 58.94 (`results/results_rescored.csv`). Full-data tokenizers use `normalization_rule_name=identity`
  (suffix `_id`; verified: 17 pieces contain ZWNJ, exact round trip on 2,000 training lines).
- **Error analysis S2 (test, `results/analysis_test_*.md`):** with the word LM every OOV reference word is wrong (100%;
  test OOV rate 10.38%), greedy OOV-word error 84.64% at 10h; in-vocabulary word error 28.09% (LM) vs 36.90% (greedy).
  Most frequent substitutions at 10h are colloquial -> formal spellings: رو->را (153), هست->است (55), اون->او (53),
  یه->یک (46), اگه->اگر (45), خب->خوب (32); plus می -> می‌کند (space vs ZWNJ in the references themselves).
  Ezafe-marked reference words: only 40 in test (too few to conclude).
- **fa_S1_10h** (done): early stop at 9401 (best 9164, dev_1h loss 3.29; plateau: 3.49 at 6.2k, LR halved 4x),
  2.42 h. Test WER/CER greedy 89.55/56.75, LM (w 0.3) 83.26/63.28; ZWNJ-insensitive LM WER 81.71.
- Full-data queue (tmux `fullq`): S2 fa full started 22:17 with tokenizer `fa_full_bpe500_id` (keeps ZWNJ),
  check_freq 610; then S1 fa full. (~11 min lost: `tmux has-session -t queue` prefix-matched `queue2`, which waited
  on itself; run_queue.sh now uses `-t =name`.)
- **fa_S2_full** (done 2026-11-26 07:37): early stop at 37820 (best 25010, dev_1h loss 0.634), 9.13 h (0.87 s/update).
  Test WER/CER greedy 34.97/11.05, LM (w 0.6) 32.18/13.09; dev greedy 25.32/7.43, LM 22.68/8.48. ZWNJ-insensitive test
  WER greedy 32.71, LM 29.33. Identity tokenizer works: greedy ZWNJ-word error 100% (10h) -> 57.48%.
- **fa_S1_full** (done 2026-11-26 17:37): STEP CAP at 40260 (best 35380, dev_1h loss 1.577; 1.63 at 28.7k -> 1.59 at
  38.4k, LR 3.75e-6), 9.74 h. Test WER/CER greedy 63.77/29.14, LM (w 0.4) 52.73/30.20; dev greedy 53.02/23.02,
  LM 40.90/22.33. ZWNJ-insensitive test WER greedy 61.93, LM 50.50.
- **All planned runs (S1/S2 at 1h, 10h, 77.7h + Polish reproduction) are done.** GPU idle since 17:37 on 2026-11-26.
- **Drive backup (2026-11-27):** user configured rclone remote `gdrive`; `sync_drive.sh --essentials` uploaded 3.15 GB;
  `rclone check` = 1,364 matching files, 0 differences (only the live sync log itself failed to copy; sync logs now
  go to ~/sync_logs outside the synced tree).

### S1-mono: S1 with CAT's own from-scratch recipe schedule (2026-11-27, user request)
CAT Polish monolingual subword configs (`egs/cv-lang10/exp/Monolingual/pl/`):
- `Mono._subword_1h`:
  - Adam lr **1e-5** (`config.json:31`), betas (0.9, 0.98), wd 1e-6 (`config.json:29-36`).
  - SchedulerEarlyStop min_step 2020, n_tol 5, gamma 0.5, stop_lr 1e-6 (`config.json:21-26`).
  - Batch **80** (`hyper-p.json:25`), check_freq 10 (`hyper-p.json:26`).
- `Mono._subword_10h`: same, except lr **3e-5** (`config.json:31`), batch **160** (`hyper-p.json:25`), check_freq 51
  (`hyper-p.json:26`).
- Both:
  - No warmup; no step/epoch budget, early stopping only.
  - Reference runs: ~2.2k (1h) and ~15k (10h) updates on 1x RTX 3090, "10 hours" each (`readme.md:16`, monitor.png).
  - Reference WERs with LM: 98.38 (1h) and 59.43 (10h) (`readme.md:52`).
- **Note:** at 10h CAT's from-scratch lr equals our shared lr (3e-5); the difference is batch 160 vs 128 and min_step
  2020 vs 2000. Only 1h really differs (lr 1e-5, batch 80).

Our settings (`catrun.MONO_RECIPE`, system `S1mono`): those values; micro-batches 20x4 (1h) and 32x5 (10h).
- check_freq = 1 epoch of our data (13 / 63 steps), since CAT's 10 / 51 are ~1 epoch of its Polish set.
- Data, tokenizer, normalization, dev_1h, LM and decoding are unchanged. Safety caps 6k / 20k were not reached.
- Estimated before starting, from the old S1 10h at 0.93 s/update: ~1.16 s/update for 10h. Measured: 1.11 s/update.

| run | final / best step | best dev_1h loss | stop | train time | test greedy WER/CER | test LM WER/CER (w) | LM WER ZWNJ-ins. |
|---|---|---|---|---|---|---|---|
| fa_S1mono_10h | 8064 / 7938 | 3.390 | early stop | 2.55 h | 90.06 / 59.41 | 84.90 / 64.18 (0.2) | 83.48 |
| fa_S1mono_1h | 2327 / 988 | 5.760 | early stop | 0.61 h | 99.24 / 94.67 | 99.24 / 94.05 (0.2) | 99.27 |
| (old) fa_S1_10h | 9401 / 9164 | 3.293 | early stop | 2.42 h | 89.55 / 56.75 | 83.26 / 63.28 (0.3) | 81.71 |
| (old) fa_S1_1h | 2232 / 320 | 5.728 | early stop | 0.91 h | 99.23 / 94.69 | 99.25 / 94.08 (0.2) | 99.28 |

- Dev: S1mono 10h greedy 86.67/55.54, LM 80.03/59.63; S1mono 1h greedy 98.67/94.18, LM 98.45/93.46.
- **Conclusion:** CAT's own from-scratch schedule does not improve S1; it is slightly worse at 10h and identical at 1h.
  S1's weakness comes from training from scratch on little data, not from the shared schedule.
- Both S1mono runs chose LM weight 0.2, the bottom of the 0.2-0.9 grid; the sweep extends only upward, so lower
  weights are untested.
- Both runs were synced to Drive automatically after decoding (queue.log: SYNCED 14:26 and 15:27).
- `REPORT.md` now shows both S1 variants (S1-shared, S1-mono) and states that each mode uses its own recipe schedule.
- Queue (tmux `queue`, `scripts/run_queue.sh`): S2 fa 1h -> S1 fa 1h -> PL pl 1h -> S2 fa 10h -> S1 fa 10h, each
  followed by decode_eval (fa: LM fa_clean; pl: pl_cat and pl_clean). Log: ~/persian_asr_data/logs/queue.log.
  Estimated worst case (step caps, 1.3-1.4 s/update): ~18.5 GPU-h -> done ~Nov 26 morning; then full-data runs if time.

## Next steps (2026-11-25)
1. User: set up the rclone remote `gdrive` (`rclone config`), then `bash scripts/sync_drive.sh` before 2026-11-29.
2. After each run: check `results/results.csv` (decode_eval appends rows); after S1/S2 10h: `scripts/analysis.py`, REPORT.md.
3. If time remains after the 10h runs: S1/S2 on train_full (77.7 h; ~610 updates/epoch, cap 40k).

## Status (2026-11-24)

| Step | State |
|---|---|
| Plan | written (see below and the conversation); open questions listed at the end |
| `00_setup.ipynb` | built; executed end to end locally on CPU with `google.colab` faked (cold and warm cache). **Not yet run on Colab.** |
| `01`–`06` | not started |

Numbers so far: none (no ASR runs yet).

## What was verified locally (CPU, Ubuntu 24.04, Python 3.12, torch 2.15.0+cpu)

- CAT @ `2340970f04` installs without its CUDA extensions; `import cat, cat.ctc.train` works
  (with import-only stubs, see below).
- `THU-SPMI/whistle-small` @ `83add32935`: `best-3.pt` has 488 tensors and 90.01 M params; the classifier
  is `(75, 512)`, i.e. 75 IPA units. It loads strictly into CAT's `ConformerNet` (14 cells, hdim 512, vgg2l, 4 heads, kernel 15).
- **Front-end check:** whistle-small greedy-decodes two LibriSpeech English utterances into correct-looking IPA
  (`m ɹ k w ɪ l t ə ɪ z ð i ə p ɑ s l ...` for "MISTER QUILTER IS THE APOSTLE ...").
  This needs Kaldi int16-scale input (`wave * 32768`); with [-1, 1] input the output is 100% blanks.
- KenLM binaries and the Python module build; a tiny 3-gram trains, binarizes and scores.
- `make_s2_init` swaps only the classifier, keeps all other 486 tensors bit-identical, and loads strictly.
- Notebook 00: cold cache took 3 min 12 s (CAT install 124 s, KenLM 33 s). Warm cache took 1 min 35 s (CAT 20 s, KenLM 4 s).
- `pytest`: 25 passed (2 tests need `WHISTLE_SMALL_DIR` and `LIBRI_DUMMY_PARQUET`).

## Findings in CAT that affect the instructions (file:line at CAT `2340970f04`)

1. **AMP is fp16 only.** `cat/shared/manager.py:29,280,533` use `torch.cuda.amp.autocast` + `GradScaler(enabled=args.amp)`
   and have no bf16 switch. The instructions ask for bf16 on A100/L4. Proposal: fp16 on every GPU (what the Whistle
   recipes used, `"amp": true`), which also keeps runs comparable across GPUs. **Needs your OK** (the alternative is patching CAT).
2. **`torch.load` and PyTorch ≥ 2.6.** CAT calls `torch.load` without `weights_only` (`cat/shared/manager.py:298,305`,
   `cat/shared/coreutils.py:546`); the default is now `weights_only=True`, which rejects Whistle and CAT checkpoints
   (verified). Fix without touching CAT: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in every notebook.
3. **Compiled extensions.** `import cat` imports `cat.lm` → `gather` and `cat.rnnt` → `warp_rnnt` (`cat/__init__.py:4-8`,
   `cat/lm/train.py:17`); `cat/ctc/train.py:19` imports `ctcdecode`. ctcdecode does not compile with recent PyTorch:
   torch headers need C++20, and its bundled OpenFST 1.6.7 fails under C++20 (verified). None of the three is *called*
   on our CTC path: ctcdecode is used only with `--eval-error-rate` (`cat/ctc/train.py:70-76`), and only by `cat.ctc.decode`.
   We use import-only stubs that raise if called (`third_party/cat_ext_stubs`). Consequence: CAT's stage-4
   "WER without LM" (`cat.ctc.decode`, beam 16, no LM) is not available; we report greedy CTC.
4. **S2 initialisation.** `init_model` is loaded with strict shapes (`cat/shared/manager.py:303-318`); the
   Polish subword fine-tune recipe points at a pre-made `mul10_phone_best-3_bpe_pl.pt` whose creation script is not in
   the repo (`local/tools/unpack_mulingual_param.py` only handles lexicon/phone tokenizers). We write it with
   `persian_asr.whistle.make_s2_init`: all encoder weights from whistle-small, a new `Linear(512, V)` classifier
   (PyTorch default init, seeded), and no optimizer state.
5. **Training set convention.** cv-lang10 trains on `validated ∪ train` minus dev/test, not on CV's official
   `train.tsv` (`egs/cv-lang10/local/data_prep.sh` stage 1). This matters for "official splits". **Needs your decision.**
6. **LM weight.** The Polish readmes report FST-decoded WER at lm weights 1.2 and 1.9, but `run.sh` stage 7 only
   sweeps 0.8/0.9/1.0 on **test**. We will tune the LM weight on **dev** only and then apply it to test.
7. **Hyperparameters differ between from-scratch and fine-tune** in the Polish recipes (e.g. 1h: mono lr 1e-5,
   batch 80, min_step 2020, n_tol 5, check every 10 steps; fine-tune lr 3e-5, batch 128, min_step 140,
   n_tol 20, check every 7 steps). That conflicts with "same budget and early-stopping rule for S1 and S2", so a
   single rule is proposed (see plan).
8. **Cost of the reference runs** (`monitor.png` + readmes): pl fine-tune 1h ≈ 3.5k updates, 10h ≈ 15k updates
   (batch 128), 3 h on 2× RTX 3090 each; mono 1h ≈ 2.2k updates (dev loss stays ≈ 6, i.e. no learning;
   98.38% WER), mono 10h ≈ 15k updates (batch 160), 10 h each.
9. Features: 80-dim Kaldi fbank, 16 kHz, no CMVN (`egs/cv-lang10/local/data_prep.sh:111,123-133`). We use
   `torchaudio.compliance.kaldi.fbank` on int16-scaled audio with dither 0 (Kaldi's default dither is 1.0).

## Dev-machine-only issue

PyTorch 2.15 CPU (oneDNN) crashes with SIGFPE on a plain 3×3 `Conv2d` with a 1×1×500×80 input on AMD EPYC 9534.
Not our code, and it doesn't affect Colab (cuDNN). Tests disable oneDNN (`tests/conftest.py`).

## Next steps

1. You run `00_setup.ipynb` on Colab (QUICK) and paste the summary → I fix anything and size the batch plan.
2. Kaldi (`scripts/build_kaldi_min.sh`): local results so far:
   - Kaldi's bundled OpenBLAS 0.3.13 fails with gcc 13 on AVX-512 → switched to Ubuntu's libopenblas (static).
   - System `lapack.h` adds hidden Fortran string-length args that Kaldi's wrappers don't pass → private header copy
     with `LAPACK_FORTRAN_STRLEN_END` undefined.
   - With both fixes: OpenFST + Kaldi `bin fstbin latbin` + CAT `latgen-faster` **built OK**. OpenFST took 25 min on
     2 cores; `src` took 3,276 CPU-s (≈30 min on 2 vCPUs).
   - The unshrunk tree is 12 GB, so I added a whitelist shrink step and `lmbin` (for `arpa2fst`). The clean rebuild
     with that final script was **killed by the dev machine (low memory), so the final script is untested**.
     It is opt-in (`BUILD_KALDI=True, QUICK=False`) and not needed before notebook 01.
3. `01_reproduce_polish.ipynb` (needs CV Polish + decisions below).

## Open questions for you

See the end of the plan in the conversation (Colab plan, Drive capacity, GitHub, splits, fp16, S1/S2 schedule).
