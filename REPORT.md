# Persian subword ASR baselines on Common Voice 27.0: S1 (from scratch) vs S2 (Whistle-small encoder)

*Draft, updated 2026-11-27. Every number below was produced by a run in this repository (`results/`); none are estimated.*

**Setup.** CAT @2340970f04, Whistle-small Conformer-CTC (14 blocks, 512-dim, 90 M params), 80-dim fbank, fp16,
1x Quadro RTX 6000. **S1**: random init. **S2**: all encoder weights from THU-SPMI/whistle-small, new
Linear(512, 500) output layer. Both: SentencePiece BPE 500 trained on that size's training text, CAT early stopping
on a fixed 1 h dev subset, average of 3 best checkpoints. **Schedules:** S2 uses CAT's fine-tuning recipe schedule
(Adam lr 3e-5, batch 128). S1 is reported twice: with that same shared schedule ("S1-shared") and with **CAT's own
from-scratch recipe schedule** ("S1-mono", `exp/Monolingual/pl/Mono._subword_{1h,10h}`: lr 1e-5 / batch 80 at 1 h,
lr 3e-5 / batch 160 at 10 h, min_step 2020), so the comparison now uses each mode's own recipe schedule. CAT's
from-scratch recipe exists only for 1 h and 10 h, so at 77.7 h only S1-shared is available.
Decoding: greedy CTC, and CAT's WFST decoding (Kaldi) with a word 4-gram LM trained only on training-pool text;
LM weight tuned on **dev** only. WER/CER computed like CAT's `utils/wer.py`.

**Data (CV 27.0 fa, MDC; official dev/test).** Training pool = (validated ∪ train) minus every clip whose sentence,
normalized text or speaker occurs in dev or test. Whistle's convention (remove only dev/test clips) would have put
97.1% of test sentences and 53.7% of test clips' speakers into training.

| set | utts | hours | speakers |
|---|---|---|---|
| train 1h / 10h / full | 1,005 / 10,080 / 78,005 | 1.0 / 10.0 / 77.7 | 135 / 214 / 261 |
| dev / test | 10,755 / 10,755 | 12.6 / 14.8 | 993 / 2,850 |

LM: 78,005 sentences, 21,052 words; OOV rate dev 7.82%, test 10.38%. Train-test overlap: 0% (sentences, speakers).

**Results (test WER / CER %).** "ZWNJ-ins." = ZWNJ mapped to space in reference and hypothesis (see finding 3).

| system | train | greedy | + 4-gram LM | LM, ZWNJ-ins. WER |
|---|---|---|---|---|
| S1-shared scratch | 1 h | 99.23 / 94.69 | 99.25 / 94.08 | 99.28 |
| S1-mono scratch (CAT recipe) | 1 h | 99.24 / 94.67 | 99.24 / 94.05 | 99.27 |
| S2 Whistle | 1 h | 76.42 / 32.03 | **61.80** / 35.56 | 58.94 |
| S1-shared scratch | 10 h | 89.55 / 56.75 | 83.26 / 63.28 | 81.71 |
| S1-mono scratch (CAT recipe) | 10 h | 90.06 / 59.41 | 84.90 / 64.18 | 83.48 |
| S2 Whistle | 10 h | 50.56 / 15.80 | **38.85** / 16.86 | 35.10 |
| S1-shared scratch | 77.7 h | 63.77 / 29.14 | 52.73 / 30.20 | 50.50 |
| S2 Whistle | 77.7 h | 34.97 / 11.05 | **32.18** / 13.09 | 29.33 |

LM weights (tuned on dev): S1-mono chose 0.2 at both sizes, the lowest value of the 0.2-0.9 grid (lower weights untested).

**Sanity check: reproduction of a published Whistle result.** "Phoneme PT + subword FT", Polish 1 h. CV 11.0 is no
longer distributed (not on MDC, HF mirrors removed), so CV 27.0 Polish with CAT's own data and LM recipe was used.
Test WER without LM: 66.44 (paper 69.50). With CAT's LM recipe: 15.16 (paper 8.63). With the same LM built after
removing dev/test sentences: **42.11**.

**Findings.**
1. Whistle pretraining helps at every size, most at low resource: at 1 h S1 never leaves the CTC blank phase (99% WER)
   while S2 reaches 61.8%; at 10 h 83.3% vs 38.9%; at 77.7 h 52.7% vs 32.2% (LM-decoded test WER). S2 at 10 h already
   beats S1 at 77.7 h. **S1's weakness is not a schedule artefact:** rerun with CAT's own from-scratch recipe
   schedule, S1 is no better (1 h: 99.24 vs 99.25; 10 h: 84.90 vs 83.26 LM test WER). At 10 h CAT's from-scratch
   recipe differs from the shared one mainly in batch size (160 vs 128; lr is 3e-5 in both); both S1 runs stopped
   early on a dev-loss plateau (best 3.39 at 7.9k and 3.29 at 9.2k updates, vs ~15k updates in CAT's Polish run).
   At 77.7 h, S1-shared hit the 40k-update cap while still improving slowly (dev loss 1.63 -> 1.59 over the last 10k
   updates), so that one number is a slight underestimate of S1.
2. CAT's cv-lang10 LM recipe trains the 4-gram on (validated ∪ train) minus dev/test *clips*, so on CV 27.0 Polish
   the LM text contains 73.3% of the test sentences (test OOV 2.1% vs 13.7% clean). With the same acoustic model the
   LM-decoded WER rises from 15.16% to 42.11% when those sentences are removed. Most of the published 1 h LM gain
   (69.50 -> 8.63) is plausibly due to this.
3. ZWNJ: the CV 27.0 Persian clip TSVs have ZWNJ deleted ("می‌شد" -> "میشد"; we restore it from
   validated_sentences.tsv), and CAT's SentencePiece tokenizer (default NFKC normalization) turns ZWNJ into a space,
   so the 1 h/10 h models cannot emit it. Every test word with ZWNJ (3,015) is then wrong in greedy decoding; ignoring
   ZWNJ lowers S2 10 h test WER from 50.56 to 44.55 (greedy) and 38.85 to 35.10 (LM). The 77.7 h runs use a tokenizer
   that keeps ZWNJ: S2's greedy error on ZWNJ words falls from 100% (10 h) to 57.5%; the remaining 2,285
   space/ZWNJ-only errors reflect inconsistent spelling in the references themselves (e.g. "می روم" vs "می‌روم").
4. The word LM lowers WER but raises CER (S2 10 h: 50.56 -> 38.85 WER, 15.80 -> 16.86 CER): with a 10.4% test OOV
   rate, every OOV word is replaced by an in-vocabulary word (OOV-word error 100% with LM vs 84.6% greedy).
5. The most frequent S2 errors are colloquial vs formal spellings (رو->را 153x, هست->است, اون->او, یه->یک, اگه->اگر):
   CV Persian mixes registers, and both the references and the LM text contain both forms.

**Next steps.** Phoneme baselines P0/P1 with Prof. Ou's lexicon/WFST pipeline (adding
output units for Persian phonemes missing from Whistle's inventory); a sentence-disjoint LM corpus beyond CV text.
