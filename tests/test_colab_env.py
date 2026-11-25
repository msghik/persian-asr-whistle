import pytest

from persian_asr import colab_env as ce


@pytest.mark.parametrize("eff,maxm,exp", [
    (128, 32, (32, 4)), (128, 48, (32, 4)), (128, 200, (128, 1)), (128, 1, (1, 128)),
    (80, 24, (20, 4)), (7, 4, (1, 7)),
])
def test_plan_batch_keeps_effective_batch(eff, maxm, exp):
    micro, accum = ce.plan_batch(eff, maxm)
    assert (micro, accum) == exp
    assert micro * accum == eff
    assert micro <= maxm


def test_plan_batch_rejects_bad():
    with pytest.raises(ValueError):
        ce.plan_batch(0, 4)


def test_max_micro_from_measurements():
    peaks = {8: 3.0, 16: 6.0, 32: 11.5, 64: None}
    assert ce.max_micro_from_measurements(peaks, 16.0) == 32  # 11.5 <= 12.8
    assert ce.max_micro_from_measurements(peaks, 10.0) == 16
    assert ce.max_micro_from_measurements(peaks, 2.0) == 0


def test_precision_policy_is_fp16_everywhere():
    for name, bf16 in [("Tesla T4", False), ("NVIDIA L4", True), ("NVIDIA A100-SXM4-40GB", True)]:
        g = ce.GPUInfo(available=True, name=name, total_mem_gb=16, bf16_supported=bf16)
        p = ce.precision_policy(g)
        assert p["amp"] is True and p["dtype"] == "fp16"
        assert p["bf16_supported"] == bf16
    assert ce.precision_policy(ce.GPUInfo(available=False))["amp"] is False


def test_gpu_short_names():
    assert ce.GPUInfo(True, "Tesla T4").short == "T4"
    assert ce.GPUInfo(True, "NVIDIA A100-SXM4-40GB").short == "A100"
    assert ce.GPUInfo(True, "NVIDIA L4").short == "L4"
    assert ce.GPUInfo(False).short == "CPU"


def test_require_disk(tmp_path):
    free = ce.require_disk(tmp_path, 0.001, "test")
    assert free > 0
    with pytest.raises(RuntimeError, match="Not enough free space for huge"):
        ce.require_disk(tmp_path, 1e9, "huge")


def test_require_disk_nonexistent_subdir(tmp_path):
    assert ce.disk_free_gb(tmp_path / "a" / "b") > 0


def test_write_json_atomic(tmp_path):
    p = tmp_path / "x" / "s.json"
    ce.write_json(p, {"a": 1, "gpu": ce.GPUInfo(False)})
    import json
    d = json.loads(p.read_text())
    assert d["a"] == 1 and d["gpu"]["available"] is False


def test_set_seed_reproducible():
    import torch
    ce.set_seed(3); a = torch.rand(3)
    ce.set_seed(3); b = torch.rand(3)
    assert torch.equal(a, b)


def test_missing_shared_libs_on_system_binary():
    import shutil
    from pathlib import Path
    from persian_asr import install
    ls = Path(shutil.which("ls"))
    assert install.missing_shared_libs(ls) == []
    install.ensure_runtime_libs([ls], ["should-not-be-installed"])  # nothing missing -> no apt call


def test_cached_wheel_falls_back_to_no_build_isolation(tmp_path, monkeypatch):
    from persian_asr import install
    calls = []

    def fake_run(cmd, log_file=None, cwd=None, env=None, tail=60):
        calls.append(cmd)
        if "--no-build-isolation" not in cmd:
            raise RuntimeError("isolated build failed")
        (tmp_path / "tmp" / "kenlm-0.2.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"x")

    monkeypatch.setattr(install, "run", fake_run)
    monkeypatch.setattr(install, "pip", lambda *a, **k: calls.append(["pip", *a]))
    out = install._cached_wheel(tmp_path / "cache", "kenlm-*.whl", "kenlm-src", tmp_path / "tmp",
                                fallback_build_reqs=("cmake>=3.10,<4",))
    assert out.is_file() and out.parent == tmp_path / "cache"
    assert calls[1] == ["pip", "cmake>=3.10,<4"]
    assert "--no-build-isolation" in calls[2] and "-q" not in calls[2]
    # second call hits the cache, no build
    calls.clear()
    assert install._cached_wheel(tmp_path / "cache", "kenlm-*.whl", "kenlm-src", tmp_path / "tmp") == out
    assert calls == []
