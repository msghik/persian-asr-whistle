# Results handoff: Persian subword ASR baselines (S1/S2) + Polish Whistle reproduction

Status as of 2026-11-27. Every number below comes from a finished run (`results/results.csv`,
`results/results_rescored.csv`, `results/analysis_*`, `~/persian_asr_data/exp/*/run_info.json`). None are estimated.
Context: PhD application to Prof. Zhijian Ou (Tsinghua SPMI). He asked for subword- and phoneme-based Persian ASR
baselines; this covers the **subword** part (S1, S2). Report due about 2026-12-10. Draft one-pager: `REPORT.md`.

## Where things are
- **Code:** GitHub `msghik/persian-asr-whistle`, branch **`worktree-server-pipeline`** (latest commit `f392f20`),
  not yet merged to `main`. 46 CPU/GPU tests pass.
- **Server:** Chameleon bare-metal lease `mohammad-rtx6000-lease`, 1x Quadro RTX 6000 24 GB. **Lease ends 2026-11-29,
  can't be extended, and the disk is wiped.** Data lives in `~/persian_asr_data` (98 GB excl. raw audio).
- **Backup: NOT DONE.** No rclone remote exists yet. To save the essentials (3.2 GB: final averaged models `best-3.pt`,
  LMs, tokenizers, prepared text, hypotheses, logs), run `rclone config` (remote `gdrive`, type `drive`), then
  `bash scripts/sync_drive.sh --essentials`. Restore with `bash scripts/sync_drive.sh --restore`.
- **Docs:** `PROGRESS.md` (findings with CAT file:line), `HANDOFF.md` (original brief), `REPORT.md` (one-pager draft).

## Setup
- **Toolkit:** CAT (thu-spmi) @ `2340970f04` + 2 patches (`scripts/patch_cat.py`), torch 2.14.1+cu126, Python 3.12.
- **Model:** Whistle-small Conformer-CTC (14 blocks, d=512, vgg2l, 4 heads, kernel 15, ~90 M params), 80-dim Kaldi fbank
  (16 kHz, no CMVN, dither 0), fp16.
- **S1:** from scratch. **S2:** all 486 encoder tensors from `THU-SPMI/whistle-small` (`best-3.pt`), new random
  `Linear(512, 500)` output layer (the 75-unit IPA layer is dropped). whistle-small-subword is not used (Ou's instruction).
- **Units:** SentencePiece BPE 500, trained on each size's training text and shared by S1 and S2 at that size.
- **Schedule (identical for all systems):**
  - Adam lr 3e-5, betas (0.9, 0.98), weight decay 1e-6; effective batch 128 (32 x 4 accumulation); SpecAugment as in CAT.
  - Length filter 10–2000 frames; evaluate every epoch on a fixed 1 h dev subset.
  - SchedulerEarlyStop: min_step 2000, n_tol 5, gamma 0.5, stop_lr 1e-6.
  - Step cap 6k (1h) / 20k (10h) / 40k (full); final model = average of the 3 best checkpoints.
- **Decoding:**
  - Greedy CTC.
  - CAT's WFST decoding (Kaldi TLG + `latgen-faster`) with a word 4-gram (KenLM via CAT's `ngram.sh`).
  - LM weight swept 0.2–0.9 (extended upward if the best is at the top) on **dev only**, then applied to test.
- **Scoring:** like CAT's `utils/wer.py` (jiwer alignment; CER = whitespace removed, then characters). Hypotheses use
  the same normalizer as the references.

## Data (Common Voice 27.0 Persian via Mozilla Data Collective; official dev/test)
- **Training pool:** (validated ∪ train) minus every clip whose sentence_id, normalized text or **speaker** occurs in
  dev/test.
  - Whistle's convention (remove only dev/test *clips*) would put 97.1% of test sentences and 53.7% of test clips'
    speakers into training.
  - Removing exact text duplicates dropped another 29,631 clips.
- **Text source:** `validated_sentences.tsv` via sentence_id, because the clip TSVs in CV 27.0 fa have ZWNJ deleted
  (e.g. "می‌شد" -> "میشد"; 15,684 recorded sentences affected, 2,273 in test).
- **Normalization:** `src/persian_asr/normalize_fa.py`, unit-tested and documented.
  - Arabic ي ى ك -> ی ک; digits -> ASCII; harakat, tanvin, hamza-above and tatweel removed; punctuation -> space.
  - ZWNJ kept word-internally.
  - Latin and digit utterances are flagged, not dropped: 29 train / 10 dev / 14 test Latin; 0 digit utterances.
- **Subsets:** nested 10m ⊂ 1h ⊂ 10h ⊂ full, seed 0.

| set | utts | hours | speakers |
|---|---|---|---|
| train 1h | 1,005 | 1.0 | 135 |
| train 10h | 10,080 | 10.0 | 214 |
| train full | 78,005 | 77.7 | 261 |
| dev / test | 10,755 / 10,755 | 12.64 / 14.77 | 993 / 2,850 |

Leakage of every training subset into dev and test: **0.0%** (sentence_id, normalized text, speaker).
**LM `fa_clean`:** 78,005 training-pool sentences (no short-sentence pruning), vocabulary 21,052, OOV rate dev 7.82%,
test 10.38%. The same LM is used by every Persian system.

## Persian results (WER / CER %)
"ZWNJ-ins." = ZWNJ mapped to space in both reference and hypothesis. This is the fair metric across sizes, because the
1h/10h tokenizers cannot emit ZWNJ (see finding 4).

### Test
| system | train | greedy WER / CER | 4-gram LM WER / CER | LM weight | greedy WER ZWNJ-ins. | LM WER ZWNJ-ins. |
|---|---|---|---|---|---|---|
| S1 scratch | 1h | 99.23 / 94.69 | 99.25 / 94.08 | 0.2 | 99.01 | 99.28 |
| S2 Whistle | 1h | 76.42 / 32.03 | **61.80** / 35.56 | 0.4 | 72.36 | 58.94 |
| S1 scratch | 10h | 89.55 / 56.75 | 83.26 / 63.28 | 0.3 | 87.25 | 81.71 |
| S2 Whistle | 10h | 50.56 / 15.80 | **38.85** / 16.86 | 0.6 | 44.55 | 35.10 |
| S1 scratch | 77.7h | 63.77 / 29.14 | 52.73 / 30.20 | 0.4 | 61.93 | 50.50 |
| S2 Whistle | 77.7h | 34.97 / 11.05 | **32.18** / 13.09 | 0.6 | 32.71 | 29.33 |

### Dev
| system | train | greedy WER / CER | LM WER / CER | greedy WER ZWNJ-ins. | LM WER ZWNJ-ins. |
|---|---|---|---|---|---|
| S1 | 1h | 98.64 / 94.24 | 98.47 / 93.51 | 98.47 | 98.55 |
| S2 | 1h | 71.80 / 28.76 | 53.92 / 30.32 | 66.52 | 50.79 |
| S1 | 10h | 86.01 / 52.87 | 77.71 / 58.20 | 82.59 | 75.78 |
| S2 | 10h | 43.11 / 12.43 | 29.21 / 11.91 | 35.23 | 25.16 |
| S1 | 77.7h | 53.02 / 23.02 | 40.90 / 22.33 | 51.09 | 38.61 |
| S2 | 77.7h | 25.32 / 7.43 | 22.68 / 8.48 | 23.14 | 19.91 |

### Training details
"Best dev loss" is measured on the fixed 1 h dev subset.

| run | final step | best step | best dev loss | stop | train time | epochs | tokenizer |
|---|---|---|---|---|---|---|---|
| S1 1h | 2,232 | 320 | 5.73 | early stop | 0.91 h | 284 | fa_1h (NFKC) |
| S2 1h | 2,248 | 2,016 | 2.12 | early stop | 0.89 h | 286 | fa_1h (NFKC) |
| S1 10h | 9,401 | 9,164 | 3.29 | early stop (plateau) | 2.42 h | 119 | fa_10h (NFKC) |
| S2 10h | 8,690 | 7,505 | 0.856 | early stop | 2.28 h | 110 | fa_10h (NFKC) |
| S1 77.7h | 40,260 | 35,380 | 1.577 | **step cap** | 9.74 h | 66 | fa_full_id (keeps ZWNJ) |
| S2 77.7h | 37,820 | 25,010 | 0.634 | early stop | 9.13 h | 62 | fa_full_id (keeps ZWNJ) |

**Measured speed:** ~1.25 s/update at 1h and ~0.87 s/update at full data (eval included).
**Theoretical benchmark:** batch 128 x 5 s clips -> 0.58 s/step at 16.4 GB.

## Polish reproduction (sanity check of the S2 procedure)
- **Target:** Whistle paper "phoneme PT + subword FT", Polish 1h (CAT `egs/cv-lang10/exp/Crosslingual/pl/Multi._phoneme_ft_subword_1h`).
  Paper: 69.50% WER without LM (beam 16), 8.63% with a 4-gram (lm weight 1.2).
- **Data:** CV 11.0 Polish is no longer distributed (not on MDC; HF mirrors return 401), so CV 27.0 Polish was used with
  **CAT's own conventions**:
  - Pool = (validated ∪ train) minus dev/test clips; CAT's Polish text normalization; random 1h (799 utts, 236 speakers).
  - Dev/test 10,096 each.
- **Training:** early stop at step 2,261 (best 2,240), best dev loss 1.40 (paper curve min 1.44), 1.0 h.

| decoding | dev WER / CER | test WER / CER | paper test WER |
|---|---|---|---|
| greedy (no LM) | 68.09 / 17.41 | **66.44** / 17.81 | 69.50 (beam 16) |
| LM `pl_cat` (CAT's recipe), weight 0.8 | 21.10 / 9.99 | **15.16** / 7.88 | 8.63 (weight 1.2) |
| LM `pl_clean` (dev/test sentences removed), weight 0.7 | 38.60 / 16.62 | **42.11** / 19.06 | – |

**Why the LMs differ:**
- CAT's recipe trains the LM on all of `excluded_train` text, and its word list includes every dev/test word.
- On CV 27.0 pl, that LM text contains **73.3% of the test sentences** (test OOV 2.09%, vs 13.70% for the clean LM).

## Error analysis (test set, `results/analysis_test_*.md`)

### 10h, greedy
| metric | S1 10h | S2 10h |
|---|---|---|
| WER | 89.55 | 50.56 |
| sub / del / ins | 48,393 / 14,335 / 3,189 | 27,236 / 3,570 / 6,408 |
| in-vocab word error % | 83.54 | 36.90 |
| OOV word error % | 99.78 | 84.64 |
| ZWNJ-word error % (3,015 words) | 100.0 | 100.0 |

### 10h, LM
| metric | S1 10h | S2 10h |
|---|---|---|
| WER | 83.26 | 38.85 |
| deletions | 26,476 | 6,554 |
| in-vocab word error % | 80.30 | 28.09 |
| OOV word error % | 100.0 | 100.0 |

### 77.7h
| metric | S1 77.7h greedy | S2 77.7h greedy | S1 77.7h LM | S2 77.7h LM |
|---|---|---|---|---|
| WER | 63.77 | 34.97 | 52.73 | 32.18 |
| sub / del / ins | 35,936 / 7,269 / 3,734 | 19,004 / 3,531 / 3,208 | 25,967 / 10,916 / 1,928 | 16,282 / 4,859 / 2,546 |
| in-vocab word error % | 54.55 | 25.32 | 44.33 | 20.47 |
| ZWNJ-word error % | 77.51 | 57.48 | 69.12 | 55.02 |

### S2 across sizes
| metric | 1h | 10h | 77.7h |
|---|---|---|---|
| greedy OOV word error % | 96.33 | 84.64 | 76.31 |
| space/ZWNJ-only errors (greedy) | 372 | 1,867 | 2,285 |
| vowel-letter-only substitutions (greedy) | 2,494 | 2,572 | 2,438 |

- **Ezafe-marked reference words:** only 40 in test, too few to conclude.
- **Most frequent S2 10h substitutions (LM output):**
  - Colloquial vs formal spelling: رو->را 153, هست->است 55, اون->او 53, یه->یک 46, اگه->اگر 45, خب->خوب 32, خونه->خون 29,
    منم->من 26.
  - Space vs ZWNJ in the references: می->می‌کند 51, می->می‌شود 43, می->می‌کنم 40.
  - ZWNJ spelling variants: آن‌ها->آنها 32, درباره->درباره‌ی 25.

## Findings (for the report)
1. **Pretraining helps at every size, most at low resource.**
   - LM test WER, S1 vs S2: 1h 99.25 vs 61.80; 10h 83.26 vs 38.85; 77.7h 52.73 vs 32.18.
   - **S2 with 10h beats S1 with 77.7h.**
   - S1 1h never left the CTC blank phase, matching CAT's Polish mono subword 1h (98.38%).
   - Caveat: S1 is somewhat under-trained under the shared budget. At 10h it stopped on a plateau (CAT's Polish mono
     10h recipe ran ~15k updates). At 77.7h it hit the 40k cap while still improving slowly (dev loss 1.63 -> 1.59
     over the last 10k updates).
2. **The published Whistle Polish LM result is inflated by the test sentences being in the LM text.**
   - Same acoustic model: 15.16% with CAT's LM vs 42.11% with a clean LM. Greedy 66.44% vs paper 69.50%.
   - Note: the paper's training set (CAT's pool) also contains test sentences spoken by other speakers (CV 27 pl:
     4.0% of test sentences at 1h, 73.2% in the full pool).
3. **CV 27.0 Persian data issues:**
   - ZWNJ deleted in the clip TSVs (restored from `validated_sentences.tsv`).
   - Heavy sentence reuse: 341,785 validated clips but only 51,921 sentences, so ~6.6 recordings per sentence.
   - The clean speaker-disjoint pool keeps 77.7 h of 373 h, with only 261 speakers.
4. **ZWNJ and the tokenizer.**
   - CAT's SentencePiece tokenizer uses default nmt_nfkc normalization, which maps ZWNJ to a space. So the 1h/10h
     models can never emit ZWNJ: 100% error on the 3,015 test words containing ZWNJ.
   - Fixed for 77.7h (`normalization_rule_name=identity`): S2 greedy ZWNJ-word error drops to 57.48%.
   - Remaining space/ZWNJ errors largely reflect inconsistent spelling in the references themselves.
   - S1 and S2 share the tokenizer at each size, so the S1 vs S2 comparison is unaffected. Cross-size comparisons
     should use the ZWNJ-insensitive columns.
5. **The word LM lowers WER but raises CER** (S2 10h: WER 50.56 -> 38.85, CER 15.80 -> 16.86). With 10.38% test OOV,
   every OOV word becomes an error; the LM forces it to an in-vocabulary word.
6. **Colloquial vs formal spelling variation is the largest single error class.** CV Persian mixes registers in its
   prompts.

## Problems found and fixed (all documented in PROGRESS.md / scripts/patch_cat.py)
- **CAT CTC crash on torch 2.14:** CPU targets with CUDA log-probs (`cat/ctc/train.py:176-195`). Patch 1 moves them to
  the GPU.
- **Wrong torch 2.14 CUDA CTC gradient in fp16 autocast** with non-contiguous input: gradients up to x120 off, garbage
  on padded frames. The first S2 1h run diverged (dev loss 6.78 -> 11.5); it's kept in `exp_invalid/`.
  - Patch 2 computes CTC in fp32 with padded frames masked, contiguous, autocast off.
  - Gradients now match the CPU reference to 3e-6 (`tests/test_cat_ctc_patch.py`).
- **CAT never deletes checkpoints** (~1 GB each): a watcher keeps the best 3 + the latest.
- **DataLoader workers re-created every epoch** (~30 s each): switched to workers=0.
- **LM weights >= 1.1 made Kaldi decoding take 17–60 min each:** the sweep is now 0.2–0.9 with upward extension.
- **Kaldi minimal build:** `scripts/build_kaldi_min.sh` confirmed working (26 min, 443 MB).

## Deviations from the original brief (user delegated these decisions)
- **Training pool:** clean, speaker- and sentence-disjoint pool instead of Whistle's convention (leakage numbers above).
- **Precision:** fp16 (this GPU has no bf16).
- **Schedule:** one shared schedule for S1/S2 (above) instead of CAT's separate mono vs fine-tune recipes.
- **Polish reproduction:** on CV 27.0 instead of CV 11.0, which is unavailable.
- **LM short-sentence pruning:** Persian LM has none; Polish LMs keep CAT's `prune_shorter 5`.
- **"WER without LM":** greedy CTC instead of CAT's beam-16 search (needs the ctcdecode extension, which doesn't build
  with current torch).

## Open items / next steps
1. **Back up before 2026-11-29** (see top). Nothing is on Drive yet.
2. Optional while the GPU is idle: continue S1 77.7h past the 40k cap (resumable) to remove the under-training caveat.
3. Finalize `REPORT.md` (one page, already drafted with these numbers) and send it to Prof. Ou around 2026-12-10.
4. Merge `worktree-server-pipeline` into `main` (or open a PR).
5. **Later (needs Prof. Ou's lexicon/WFST pipeline):** phoneme baselines P1 (Whistle-small + phoneme CTC with added units
   for Persian phonemes missing from Whistle's inventory) and P0 (from scratch). Do not start JSA-SPG / LLM-P2G.
6. After S1/S2 (allowed now): inspect the user's HF collection `MohammadGholizadeh/persian-asr-datasets`, inspection
   only.

## Reproduce / resume on a new machine
```bash
git clone -b worktree-server-pipeline https://github.com/msghik/persian-asr-whistle && cd persian-asr-whistle
BUILD_KALDI=1 bash scripts/setup_server.sh                       # env, CAT + patches, KenLM, Kaldi, whistle-small, tests
# CV 27.0 data via MDC SDK (needs MDC_API_KEY in ~/.mdc.env): fa id cmu5xazgy00finq074vaqo1b3, pl id cmu5wsxwp00e0nq07m8v0274j
python scripts/prepare_data.py --lang fa --cv-root <cv-corpus-27.0-*/fa> --out ~/persian_asr_data/prep/fa
python scripts/extract_feats.py --prep ~/persian_asr_data/prep/fa --sets train_full dev test
python scripts/build_lm.py fa_clean
bash scripts/run_queue.sh "S2 fa 1h" "S1 fa 1h" ...                # train + decode/eval, resumable
python scripts/rescore.py; python scripts/analysis.py --run-ids <ids> --set test --hyp fst_fa_clean_best.hyp
```
