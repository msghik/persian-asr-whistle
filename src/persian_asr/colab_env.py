"""Colab environment checks: Drive, disk, GPU, precision/batch policy, versions, seeds."""
from __future__ import annotations

import importlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def mount_drive(mountpoint: str = "/content/drive") -> None:
    if os.path.isdir(os.path.join(mountpoint, "MyDrive")):
        return
    from google.colab import drive  # type: ignore

    drive.mount(mountpoint)


# --- disk ---------------------------------------------------------------------

def disk_free_gb(path: str | os.PathLike) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def require_disk(path: str | os.PathLike, min_free_gb: float, what: str) -> float:
    """Fail early with a clear message if `path` has less than `min_free_gb` free."""
    free = disk_free_gb(path)
    if free < min_free_gb:
        raise RuntimeError(
            f"Not enough free space for {what}: {free:.1f} GB free at {path}, "
            f"need at least {min_free_gb:.1f} GB. Free space (delete files in /content, "
            f"or empty Drive trash) or lower the requirement in the settings cell."
        )
    return free


# --- GPU / precision / batch --------------------------------------------------

@dataclass
class GPUInfo:
    available: bool
    name: str = "none"
    total_mem_gb: float = 0.0
    capability: str = ""
    bf16_supported: bool = False
    driver: str = ""

    @property
    def short(self) -> str:
        n = self.name.upper()
        for k in ("H100", "A100", "L4", "T4", "V100", "P100"):
            if k in n:
                return k
        return self.name if self.available else "CPU"


def gpu_info() -> GPUInfo:
    import torch

    if not torch.cuda.is_available():
        return GPUInfo(available=False)
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    driver = ""
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return GPUInfo(
        available=True,
        name=props.name,
        total_mem_gb=props.total_memory / 1e9,
        capability=f"{props.major}.{props.minor}",
        bf16_supported=bool(torch.cuda.is_bf16_supported()),
        driver=driver,
    )


def precision_policy(gpu: GPUInfo) -> dict[str, Any]:
    """Mixed-precision mode actually used for CAT training.

    CAT's trainer only supports fp16 autocast + GradScaler (`--amp`):
    cat/shared/manager.py:29 (imports), :280 (GradScaler(enabled=args.amp)),
    :533 (autocast(enabled=use_amp)) — there is no bf16 switch. So the policy is
    fp16 on every GPU (also what the Whistle recipes used: "amp": true), which
    keeps runs numerically comparable across T4/L4/A100. bf16 support is only
    recorded. Changing this would mean patching CAT; see PROGRESS.md.
    """
    if not gpu.available:
        return {"amp": False, "dtype": "fp32", "bf16_supported": False,
                "note": "no GPU: CAT training requires CUDA"}
    return {"amp": True, "dtype": "fp16", "bf16_supported": gpu.bf16_supported,
            "note": "CAT supports fp16 AMP only (cat/shared/manager.py:280,533)"}


def plan_batch(effective_batch: int, max_micro_batch: int) -> tuple[int, int]:
    """Split a fixed effective batch (utterances/update) into micro-batch x grad-accum.

    Returns (micro_batch, grad_accum_fold) with micro*accum == effective_batch exactly,
    using the largest divisor of `effective_batch` that fits in `max_micro_batch`.
    Keeping the effective batch identical on every GPU keeps runs comparable.
    """
    if effective_batch < 1 or max_micro_batch < 1:
        raise ValueError("effective_batch and max_micro_batch must be >= 1")
    for micro in range(min(effective_batch, max_micro_batch), 0, -1):
        if effective_batch % micro == 0:
            return micro, effective_batch // micro
    return 1, effective_batch  # unreachable (1 always divides)


def max_micro_from_measurements(peak_gb_by_batch: dict[int, float], total_mem_gb: float,
                                headroom: float = 0.80) -> int:
    """Largest measured micro-batch whose peak memory stays under headroom*total.

    `peak_gb_by_batch` maps micro-batch size -> measured peak GB for a worst-case
    (longest-utterance) batch. Returns 0 if none fit.
    """
    ok = [b for b, gb in peak_gb_by_batch.items() if gb is not None and gb <= headroom * total_mem_gb]
    return max(ok) if ok else 0


# --- versions / seeds -----------------------------------------------------------

_PKGS = ["torch", "torchaudio", "numpy", "sentencepiece", "kaldiio", "jiwer", "kenlm",
         "webdataset", "transformers", "huggingface_hub", "Levenshtein", "cat"]


def package_versions(pkgs: list[str] | None = None) -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version()}
    for name in pkgs or _PKGS:
        try:
            m = importlib.import_module(name)
            out[name] = str(getattr(m, "__version__", "installed"))
        except Exception as e:  # noqa: BLE001
            out[name] = f"MISSING ({type(e).__name__})"
    try:
        import torch

        out["torch.cuda"] = str(torch.version.cuda)
        out["cudnn"] = str(torch.backends.cudnn.version())
    except Exception:  # noqa: BLE001
        pass
    return out


def git_commit(repo_dir: str | os.PathLike) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path: str | os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o))
    os.replace(tmp, path)
