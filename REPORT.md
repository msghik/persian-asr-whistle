# Persian subword ASR baselines on Common Voice 27.0: S1 (from scratch) vs S2 (Whistle-small encoder)

*Draft, 2026-11-25. Every number below was produced by a run in this repository (`results/`); none are estimated.*

**Setup.** CAT @2340970f04, Whistle-small Conformer-CTC (14 blocks, 512-dim, 90 M params), 80-dim fbank, fp16,
1x Quadro RTX 6000. **S1**: random init. **S2**: all encoder weights from THU-SPMI/whistle-small, new
Linear(512, 500) output layer. Both: SentencePiece BPE 500 trained on that size's training text, identical schedule
(Adam lr 3e-5, effective batch 128, CAT early stopping on a fixed 1 h dev subset, average of 3 best checkpoints).
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
| S1 scratch | 1 h | 99.23 / 94.69 | 99.25 / 94.08 | 99.28 |
| S2 Whistle | 1 h | 76.42 / 32.03 | **61.80** / 35.56 | 58.94 |
| S1 scratch | 10 h | 89.55 / 56.75 | 83.26 / 63.28 | 81.71 |
| S2 Whistle | 10 h | 50.56 / 15.80 | **38.85** / 16.86 | 35.10 |
| S1 / S2 | 77.7 h | running | running | running |

**Sanity check: reproduction of a published Whistle result.** "Phoneme PT + subword FT", Polish 1 h. CV 11.0 is no
longer distributed (not on MDC, HF mirrors removed), so CV 27.0 Polish with CAT's own data and LM recipe was used.
Test WER without LM: 66.44 (paper 69.50). With CAT's LM recipe: 15.16 (paper 8.63). With the same LM built after
removing dev/test sentences: **42.11**.

**Findings.**
1. Pretraining matters most at low resource: at 1 h, S1 never leaves the CTC blank phase (99% WER) while S2 reaches
   61.8% WER; at 10 h the gap is 83.3% vs 38.9%. S1's early stopping at 10 h was triggered by a plateau (dev loss
   3.29 at 9.4k updates); CAT's Polish monolingual 10 h recipe trained ~15k updates, so S1 may be under-trained
   under the shared schedule.
2. CAT's cv-lang10 LM recipe trains the 4-gram on (validated ∪ train) minus dev/test *clips*, so on CV 27.0 Polish
   the LM text contains 73.3% of the test sentences (test OOV 2.1% vs 13.7% clean). With the same acoustic model the
   LM-decoded WER rises from 15.16% to 42.11% when those sentences are removed. Most of the published 1 h LM gain
   (69.50 -> 8.63) is plausibly due to this.
3. ZWNJ: the CV 27.0 Persian clip TSVs have ZWNJ deleted ("می‌شد" -> "میشد"; we restore it from
   validated_sentences.tsv), and CAT's SentencePiece tokenizer (default NFKC normalization) turns ZWNJ into a space,
   so the 1 h/10 h models cannot emit it. Every test word with ZWNJ (3,015) is then wrong in greedy decoding; ignoring
   ZWNJ lowers S2 10 h test WER from 50.56 to 44.55 (greedy) and 38.85 to 35.10 (LM). Full-data runs use a tokenizer
   that keeps ZWNJ.
4. The word LM lowers WER but raises CER (S2 10 h: 50.56 -> 38.85 WER, 15.80 -> 16.86 CER): with a 10.4% test OOV
   rate, every OOV word is replaced by an in-vocabulary word (OOV-word error 100% with LM vs 84.6% greedy).
5. The most frequent S2 errors are colloquial vs formal spellings (رو->را 153x, هست->است, اون->او, یه->یک, اگه->اگر):
   CV Persian mixes registers, and both the references and the LM text contain both forms.

**Next steps.** Finish the 77.7 h runs; phoneme baselines P0/P1 with Prof. Ou's lexicon/WFST pipeline (adding
output units for Persian phonemes missing from Whistle's inventory); a sentence-disjoint LM corpus beyond CV text.
