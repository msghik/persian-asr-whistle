# Persian ASR baselines: phoneme-pretrained vs from-scratch subword models (Common Voice 27.0)

Mohammad Sadegh Gholizadeh · 2026-11-27

**Summary.** Initializing a subword CTC model from the phoneme-pretrained Whistle-small encoder (S2) instead of training it from scratch (S1) lowers LM-decoded test WER from 99.25 to 61.80 (1 h), 83.26 to 38.85 (10 h) and 52.73 to 32.18 (77.7 h). Next, I plan to build the phoneme baseline with your Persian lexicon and phoneme-label pipeline.

**Setup**

- **Data:** CV 27.0 Persian, official dev/test. Training pool: validated ∪ train minus clips whose sentence, normalized text or speaker occurs in dev/test (77.7 h, nested 1 h and 10 h subsets). Text from validated_sentences.tsv because the clip TSVs delete ZWNJ.
- **Normalization:** Arabic yeh/kaf → Persian yeh/kaf; diacritics, tatweel, punctuation removed; ZWNJ kept inside words; same for all text.
- **Systems:** S1 = Conformer-CTC with Whistle-small's architecture, from scratch. S2 = Whistle-small encoder + new Persian BPE-500 output layer. Both: same tokenizer and schedule (Adam, lr 3e-5, batch 128, early stopping on a 1 h dev subset).
- **Toolkit/decoding:** CAT, fp16. Greedy CTC; WFST with a word 4-gram LM (training-pool text), weight tuned on dev only.

**Data card**

| | train 1 h | train 10 h | train full | dev | test |
|---|---|---|---|---|---|
| utterances | 1,005 | 10,080 | 78,005 | 10,755 | 10,755 |
| hours | 1.0 | 10.0 | 77.7 | 12.64 | 14.77 |
| speakers | 135 | 214 | 261 | 993 | 2,850 |

LM vocabulary 21,052 words; OOV rate 7.82% (dev), 10.38% (test).

**Main results** (test WER / CER, %; "ZWNJ ignored" maps ZWNJ to a space in both texts, needed because the 1 h/10 h tokenizers cannot output ZWNJ)

| system | train | greedy | 4-gram LM | LM WER, ZWNJ ignored |
|---|---|---|---|---|
| S1 | 1 h | 99.23 / 94.69 | 99.25 / 94.08 | 99.28 |
| S2 | 1 h | 76.42 / 32.03 | 61.80 / 35.56 | 58.94 |
| S1 | 10 h | 89.55 / 56.75 | 83.26 / 63.28 | 81.71 |
| S2 | 10 h | 50.56 / 15.80 | 38.85 / 16.86 | 35.10 |
| S1 | 77.7 h | 63.77 / 29.14 | 52.73 / 30.20 | 50.50 |
| S2 | 77.7 h | 34.97 / 11.05 | 32.18 / 13.09 | 29.33 |

**Schedule check.** Re-running S1 with CAT's own from-scratch recipe schedule gave LM test WER 99.24 at 1 h (vs 99.25) and 84.90 at 10 h (vs 83.26), so the S1–S2 gap is not due to the schedule.

**Polish sanity check.** Whistle's "phoneme PT + subword FT" Polish 1 h procedure, reproduced on CV 27.0 (CV 11.0 is unavailable), gave greedy test WER 66.44 vs 69.50 in the paper. CAT's LM text contains 73.3% of the test sentences: test WER is 15.16 with that LM and 42.11 with those sentences removed, so LM-decoded results under this protocol depend on that coverage.

**Findings**

- Phoneme pretraining helps at every size; S2 at 10 h (38.85) beats S1 at 77.7 h (52.73).
- The largest error class is colloquial vs formal spelling (S2 10 h, LM): <bdi>رو</bdi> → <bdi>را</bdi> (153), <bdi>هست</bdi> → <bdi>است</bdi> (55), <bdi>اون</bdi> → <bdi>او</bdi> (53).
- Ezafe cannot be evaluated from text: only 40 test words carry a written ezafe marker, so phoneme-level labels are needed.
- The 10.38% test OOV rate limits the word LM: for S2 10 h it lowers WER (50.56 → 38.85) but raises CER (15.80 → 16.86).

**Next steps**

- Phoneme baseline P1 (P0 if time allows) with your lexicon and phoneme-label pipeline, adding output units for missing phonemes.
- A larger LM text with dev/test sentences excluded; phoneme-level error analysis (short vowels, ezafe).
