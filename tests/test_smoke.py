import torch

from conftest import requires_cat
from persian_asr import features, smoke


def test_fbank_shape_and_scale():
    t = torch.linspace(0, 1, 16000)
    wave = 0.1 * torch.sin(2 * 3.1416 * 440 * t)
    f = features.fbank80(wave)
    assert f.shape == (98, 80)
    # int16-scaled input => log energies well above the [-1,1]-scaled ones (~ +20.8 = log(32768^2))
    import torchaudio
    g = torchaudio.compliance.kaldi.fbank(wave.unsqueeze(0), num_mel_bins=80, dither=0.0)
    unclipped = g > -10  # bins where the unscaled energy is above Kaldi's log floor
    assert unclipped.sum() > 100
    assert abs((f - g)[unclipped].mean().item() - 20.79) < 0.05


def test_fbank_resamples():
    f = features.fbank80(torch.zeros(8000) + 1e-3, sample_rate=8000)
    assert f.shape[1] == 80 and abs(f.shape[0] - 98) <= 1


def test_ctc_greedy():
    assert smoke._ctc_greedy([0, 3, 3, 0, 3, 5, 5, 0]) == [3, 3, 5]


@requires_cat
def test_train_step_benchmark_cpu_tiny():
    r = smoke.train_step_benchmark("cpu", [2], frames=200, vocab=50, steps=1, amp=False,
                                   encoder_kwargs=dict(num_cells=1, idim=80, hdim=32, conv="vgg2l",
                                                       num_heads=4, kernel_size=15))
    assert r["by_micro_batch"][2]["finite_loss"]


def test_checkpoint_roundtrip(tmp_path):
    assert smoke.checkpoint_roundtrip(tmp_path)["pass"]


def test_drive_io_benchmark(tmp_path):
    r = smoke.drive_io_benchmark(tmp_path, size_mb=4)
    assert r["write_MBps"] > 0 and not list(tmp_path.iterdir())


@requires_cat
def test_english_phoneme_check_real(whistle_small_dir, libri_parquet):
    r = smoke.english_phoneme_check(whistle_small_dir, libri_parquet, n=2)
    assert r["pass"], r
    assert r["utt0_matches_cpu_reference"], r["utts"][0]["hyp_ipa"]
