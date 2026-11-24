# PROGRESS

Deadline: short results report to Prof. Zhijian Ou around **2026-12-10**.
Rule: no number is reported unless a run produced it; failed runs are reported as failed.

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
