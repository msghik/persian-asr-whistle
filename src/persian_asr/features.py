"""Acoustic front-end, matched to how Whistle / CV-Lang10 features were made.

CAT cv-lang10 recipe (egs/cv-lang10/local/data_prep.sh:111,123-133): mp3 ->
`ffmpeg -ar 16000 -ac 1` -> Kaldi `compute-fbank-feats --num-mel-bins=80`, no CMVN.
Kaldi computes fbank on int16-valued samples, so float audio in [-1, 1] must be
scaled by 32768 first. Verified on CPU (2026-11-24): with the scale, whistle-small
greedy-decodes LibriSpeech English to correct-looking IPA; without it, 100% blanks.

Deviation (documented): Kaldi's default dither=1.0; we use dither=0.0 so features
are deterministic. Dither of 1 int16 LSB is negligible next to speech energy.
"""
from __future__ import annotations

import torch

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
KALDI_INT16_SCALE = 32768.0


def fbank80(wave: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """wave: float tensor in [-1, 1], shape (N,) or (1, N). Returns (T, 80) float32."""
    import torchaudio

    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    if wave.size(0) != 1:
        wave = wave.mean(0, keepdim=True)
    wave = wave.float()
    if sample_rate != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sample_rate, SAMPLE_RATE)
    return torchaudio.compliance.kaldi.fbank(
        wave * KALDI_INT16_SCALE,
        num_mel_bins=NUM_MEL_BINS,
        sample_frequency=SAMPLE_RATE,
        dither=0.0,
    )
