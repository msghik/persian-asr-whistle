# HANDOFF — brief for the Claude Code session on the UChicago GPU server

You are continuing a project started in another Claude Code session (2026-11-24/25). This file is
the complete brief; `PROGRESS.md` holds the verified technical state with CAT file:line references.
Read both fully before doing anything.

## Who and why
Mohammad Sadegh Gholizadeh is applying for a PhD with **Prof. Zhijian Ou** (Tsinghua EE, SPMI Lab).
Prof. Ou asked for two Persian ASR baselines, one subword-based and one phoneme-based, compared.
He wants to see progress before supporting the application.
**Deadline:** a short results report around **2026-12-10**.
**Never invent, estimate or "fill in" a result number. A failed run is reported as failed.**

## Where we run now
Colab was abandoned (setup was unreliable). Training now runs on a **reserved RTX 6000 server at the
University of Chicago**; this Claude session runs on that machine. Details are unknown at handoff and must
be checked first: exact GPU model, VRAM, driver/CUDA, OS, sudo, conda/modules, Slurm vs interactive,
disk and scratch paths, internet access to huggingface.co / github.com / datacollective.mozillafoundation.org,
and the reservation length and windows.
Repo: https://github.com/msghik/persian-asr-whistle (private).

## Research context
- **Whistle** (arXiv 2406.02166): a Conformer-CTC pretrained on G2P IPA phoneme labels for 10 Common Voice v11
  languages (CV-Lang10). It is the most data-efficient option when fine-tuning data is small.
  Later work: LLM-P2G (2506.04711), JSA-SPG (2507.06249), Phonemes vs Projectors (2604.09332), Latent Softmax (2608.01281).
- Code and recipes: https://github.com/thu-spmi/CAT/tree/master/egs/cv-lang10 (pinned commit in `src/persian_asr/config.py`).
- Checkpoints are under the HF org **THU-SPMI** (the lowercase `thu-spmi` org is empty): whistle-small,
  whistle-medium, whistle-large, whistle-small-subword, wav2vec2-cv-lang10, whistle-small-polish.
  Revisions are pinned in `config.py`.
- Long-term topic: Persian script omits short vowels and the ezafe, so G2P labels are noisy. This project only
  builds baselines. Do NOT start JSA-SPG, lattice training or LLM-P2G.

## Prof. Ou's instructions (follow exactly)
1. Start with the subword baselines. His Persian pipeline (lexicon plus WFST-based phoneme labels) comes later. Do not build our own G2P.
2. Do NOT fine-tune whistle-small-subword (its BPE units are not Persian).
3. Use the **newest Common Voice Persian release with its official splits**: Common Voice Scripted Speech
   **27.0 – Persian** (Sept 2026; 10.46 GB of MP3; 373 validated hours; 4,668 speakers). It is distributed only via the
   **Mozilla Data Collective**: https://mozilladatacollective.com/datasets/cmu5xazgy00finq074vaqo1b3. You need a login,
   accepted dataset terms, and an API key (`MDC_API_KEY`). The Python SDK is `pip install datacollective` (pre-1.0;
   pin it). The user must accept the terms and supply the key; never print it or commit it.
4. For the later phoneme baseline: **add output units** for Persian phonemes missing from Whistle's inventory.

## Experiment matrix
All systems share the same normalization, splits and word 4-gram LM.

| ID | Model | Units | Train sizes | Needs Ou's pipeline |
|---|---|---|---|---|
| S1 | Conformer-CTC, Whistle-small architecture, from scratch | Persian BPE | 1h, 10h, (full) | No |
| S2 | Whistle-small encoder + NEW Persian BPE output layer | Persian BPE | 1h, 10h, (full) | No |
| P1 | Whistle-small + phoneme CTC (added units), WFST decoding | IPA phonemes | later | Yes |
| P0 | From scratch, phoneme units | IPA phonemes | later | Yes |

- **Priority for 2026-12-10:** S1 and S2 at 1h and 10h. Run full data only if time allows.
- **Metrics:** WER and CER on dev and test, both greedy CTC and LM-decoded.
- **BPE size:** CAT's Polish subword recipes use BPE 500 at every size (verified). Use the same size for S1 and S2.
- LM decoding must be identical for S1 and S2. Primary: CAT's WFST decoding (Kaldi). Fallback if Kaldi fails:
  KenLM + CTC prefix beam search, used for both. Tune the LM weight on **dev**, never on test.

## Deliverables (convert the Colab notebooks into scripts for this server)
`src/` (normalization, data prep, training wrappers, decoding, scoring), `tests/` that run on CPU,
and one entry point per step. Each step has a short sanity mode (a few minutes, or overfitting about 50
utterances) and a flag for the full run. Each prints a short summary for the user to paste.
1. **setup:** check GPU/disk, install CAT + deps, build Kaldi (`scripts/build_kaldi_min.sh`) and KenLM,
   download Whistle checkpoints, run smoke tests (the reused `src/persian_asr/smoke.py`: whistle-small English→IPA,
   GPU batch benchmark).
2. **reproduce_polish:** reproduce ONE published Whistle crosslingual result and compare it with the paper.
   Recommended target: "Phoneme PT + subword FT", Polish 1h, **8.63% WER** with the 4-gram LM (69.50% without),
   CAT `egs/cv-lang10/exp/Crosslingual/pl/Multi._phoneme_ft_subword_1h`. This is the same procedure as S2.
   It needs CV 11.0 Polish; confirm its availability on MDC. If the result is far off, debug before touching Persian.
3. **prepare_persian:**
   - `src/normalize_fa.py`, unit-tested and applied identically to training text, LM text, references and hypotheses.
     It handles: Arabic ي/ك → ی/ک; Persian/Arabic-Indic digits; one documented ZWNJ policy; punctuation (، ؛ ؟ « »);
     harakat and tanvin; tatweel; whitespace. It **flags** (does not silently drop) Latin-script or digit utterances.
   - Split checks: speaker overlap between splits, and test sentences whose text appears in train.
   - Subsets: nested 10 min ⊂ 1h ⊂ 10h ⊂ full with a fixed seed; save the lists.
   - Word 4-gram LM on training text only, with dev and test sentences removed. Report vocabulary size and dev OOV rate.
   - Data card: hours, utterances and speakers per split.
4. **train:** one parameterized entry point for S1/S2 at a chosen size. It must resume after interruption
   (CAT `--resume`). For S2, document which layers load from whistle-small and which are re-initialized
   (`whistle.make_s2_init` writes a JSON report). Use the same training budget and early-stopping rule for S1 and S2.
5. **decode_eval:** greedy and LM decoding on dev and test. Append a row to `results.csv` with: run_id, system, units,
   hours, GPU, WER/CER greedy and LM, BPE size, epochs, wall-clock.
6. **analysis:** for the best S1 and S2:
   - substitution/insertion/deletion breakdown;
   - the 30 most frequent word confusions;
   - OOV-word error rate;
   - errors at ZWNJ/ezafe boundaries;
   - a simple homograph flag.
   Then draft `REPORT.md`: at most one page for Prof. Ou, covering setup, data card, results table,
   3–5 factual findings and next steps.
7. Later, only after S1/S2 at 1h and 10h are done, and for inspection only (no training): the user's HF collection
   https://huggingface.co/collections/MohammadGholizadeh/persian-asr-datasets. Stream it; report clips, hours and
   metadata; judge transcript type with 10 random examples; propose speaker- or show-disjoint splits.

## Working rules
- Before writing new code, give the user a short plan. It should cover risks, estimated GPU time per run on THIS GPU
  (measure with the benchmark), and whether full-data runs fit before 12-10.
- Test what you can on CPU before using GPU time.
- Keep `PROGRESS.md` current: done, failed, current numbers, next step, CAT file:line for every CAT finding.
- Pin seeds; log package versions and the GPU used in every run.
- Code and configs go in git; data and checkpoints never do (keep them in a data dir outside the repo, e.g. on scratch).
- **Ask the user before:** spending money; downloads over 50 GB; deviating from the experiment matrix;
  deleting data or checkpoints; anything that changes the shared server beyond this user's own directories
  (system-wide installs, other users' files, more GPU than reserved).
  Pushing to this repo is approved (commit with clear messages; never commit secrets). Pushing to HF is not.
- If CAT code contradicts these instructions, stop and explain. Read the code rather than guessing.
- Respect the reservation: run long jobs in tmux or Slurm, and checkpoint so a job can resume in the next window.

## Decisions still open (ask the user first)
1. **Training split:** Whistle trained on `validated ∪ train` minus dev/test, not on CV's official `train.tsv`
   (`egs/cv-lang10/local/data_prep.sh` stage 1); speakers may then overlap with dev/test. Recommendation: official
   dev/test, Whistle's training convention, and report speaker overlap.
2. **Precision:** CAT supports fp16 AMP only (`cat/shared/manager.py:280,533`). Recommendation: fp16 on every GPU.
3. **One schedule for S1 and S2:** CAT's Polish recipes differ between mono and fine-tune runs. Recommendation:
   identical settings per size (lr 3e-5, effective batch 128, SchedulerEarlyStop with the same min_step/n_tol/check_freq,
   a step cap), with early stopping on a fixed ~1h dev subset; final scoring on the full dev and test sets.

## First actions on the server
1. Read `PROGRESS.md` and `README.md`; run the machine survey (GPU model/VRAM, driver, OS, CPUs, RAM, disks, sudo,
   conda/modules/Slurm, internet to HF/GitHub/MDC) and report it to the user.
2. Ask the user the open decisions above, plus the reservation length and the data/scratch directory to use.
3. Create a venv/conda env and run `pytest` (CPU). Then port `00_setup` to a server script, run it, and report
   the summary, including the GPU benchmark → batch plan and measured hours-per-run estimates.
