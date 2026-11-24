import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

try:
    import torch

    # PyTorch 2.15 CPU oneDNN crashes (SIGFPE) on a plain 3x3 Conv2d with a 500x80 input
    # on AMD EPYC 9534 (Zen 4) — seen on the dev machine, not related to our code.
    # Colab runs convs on cuDNN. Disable oneDNN for local CPU tests only.
    torch.backends.mkldnn.enabled = False
except ImportError:
    pass


def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


requires_cat = pytest.mark.skipif(not _has("cat.shared.encoder"), reason="CAT not installed")


@pytest.fixture(scope="session")
def whistle_small_dir():
    d = os.environ.get("WHISTLE_SMALL_DIR")
    if not d or not (Path(d) / "best-3.pt").is_file():
        pytest.skip("set WHISTLE_SMALL_DIR to a downloaded THU-SPMI/whistle-small to run this test")
    return Path(d)


@pytest.fixture(scope="session")
def libri_parquet():
    p = os.environ.get("LIBRI_DUMMY_PARQUET")
    if not p or not Path(p).is_file():
        pytest.skip("set LIBRI_DUMMY_PARQUET to the librispeech_asr_dummy parquet to run this test")
    return Path(p)
