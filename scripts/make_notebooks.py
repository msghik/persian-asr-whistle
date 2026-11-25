"""Generate the Colab notebooks from cell sources kept in this file (reviewable in git).

Usage: python scripts/make_notebooks.py   -> writes notebooks/*.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str):
    return nbf.v4.new_code_cell(s.strip("\n"))


def write(name: str, cells: list) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    out = ROOT / "notebooks" / name
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    return out


# ---------------------------------------------------------------------------------------
# Shared bootstrap cell: Drive + code checkout. Used verbatim by every notebook.
BOOTSTRAP = r'''
# --- Bootstrap: mount Drive, get this repo, import it (same cell in every notebook) ---
import os, sys, subprocess, time, json, warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)  # CAT has invalid escape sequences in strings
T0 = time.time()
os.environ.update({"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})  # CAT checkpoints need it (see config.py)

from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_ROOT, exist_ok=True)

CODE_DIR = "/content/persian_asr"
if REPO_URL and "<" not in REPO_URL:
    url = REPO_URL
    try:  # private repo: add a Colab secret named GITHUB_TOKEN (key icon in the left bar)
        from google.colab import userdata
        tok = userdata.get("GITHUB_TOKEN")
        if tok:
            url = REPO_URL.replace("https://", f"https://{tok}@")
    except Exception:
        pass
    if not os.path.isdir(f"{CODE_DIR}/.git"):
        subprocess.run(["git", "clone", "-q", url, CODE_DIR], check=True)
    subprocess.run(["git", "-C", CODE_DIR, "fetch", "-q", "origin"], check=True)
    subprocess.run(["git", "-C", CODE_DIR, "checkout", "-q", REPO_REF], check=True)
    subprocess.run(["git", "-C", CODE_DIR, "pull", "-q", "--ff-only"], check=False)
else:  # no GitHub yet: use a zip of the repo uploaded to Drive
    zip_path = f"{DRIVE_ROOT}/code/persian_asr.zip"
    assert os.path.isfile(zip_path), f"Set REPO_URL, or upload the repo zip to {zip_path}"
    subprocess.run(["rm", "-rf", CODE_DIR], check=True)
    subprocess.run(["unzip", "-q", "-o", zip_path, "-d", CODE_DIR], check=True)

sys.path.insert(0, f"{CODE_DIR}/src")
from persian_asr import config, colab_env as ce
P = config.default_paths(DRIVE_ROOT, LOCAL_ROOT).ensure()
CODE_COMMIT = ce.git_commit(CODE_DIR)
print("code:", CODE_DIR, "@", CODE_COMMIT[:10], "| drive:", P.drive_root, "| local:", P.local_root)
'''


def nb00() -> Path:
    cells = [
        md("""
# 00 · Setup (run once per new Colab environment; safe to re-run)

What this does:
1. mounts Drive, checks GPU and free disk (fails early with a clear message);
2. installs CAT (pinned commit) and its Python deps, reusing wheels cached on Drive;
3. builds or restores the KenLM binaries (`lmplz`, `build_binary`) for the word 4-gram LM;
4. optionally builds or restores a minimal **Kaldi** for CAT's WFST decoding (`BUILD_KALDI`);
5. downloads the Whistle checkpoints to Drive (pinned revisions, SHA-256 logged);
6. smoke tests: whistle-small transcribes English speech to IPA on the GPU; a train-step benchmark
   (speed and memory per micro-batch → batch plan); Drive I/O and checkpoint round-trip; KenLM;
7. prints a **SUMMARY** block, which you paste back to Claude.

`QUICK = True` (default) does the short version: no Kaldi build, and the benchmark uses fewer batch sizes.
"""),
        md("## Settings"),
        code(r'''
# ===================== SETTINGS =====================
RUN_ID      = "setup"
REPO_URL    = "https://github.com/msghik/persian-asr-whistle.git"  # private: add Colab secret GITHUB_TOKEN
REPO_REF    = "main"
DRIVE_ROOT  = "/content/drive/MyDrive/persian_asr"
LOCAL_ROOT  = "/content/persian_asr_local"
SEED        = 0

QUICK       = True        # True: short sanity version. False: full benchmark (+ Kaldi if BUILD_KALDI)
BUILD_KALDI = False       # True: build minimal Kaldi once (~1-2 h, CPU only) and cache it to Drive.
                          #       Tip: do this on a CPU-only runtime to save GPU time/compute units.
CHECKPOINTS = ["whistle-small"]          # add "whistle-small-subword" only if needed (not for S2)
MIN_LOCAL_FREE_GB = 30    # local /content disk needed for this notebook (+ Kaldi build)
MIN_DRIVE_FREE_GB = 3     # Drive space needed for this notebook (checkpoints + caches)

EFFECTIVE_BATCH = 128     # utterances per update; Whistle fine-tune recipe (pl 1h/10h hyper-p.json)
BENCH_FRAMES = 1000       # 10 s utterances (worst case for Common Voice clips)
# micro-batch sizes to try; stops at the first out-of-memory, so big GPUs (A100 80 GB) reach 128
BENCH_MICRO_BATCHES = [b for b in (8, 16, 32, 64, 128) if b <= EFFECTIVE_BATCH]
# ====================================================
'''),
        md("## 1 · Drive, code, environment checks"),
        code(BOOTSTRAP),
        code(r'''
ce.set_seed(SEED)
GPU = ce.gpu_info()
PREC = ce.precision_policy(GPU)
print("GPU:", GPU)
print("precision:", PREC)
if not GPU.available:
    print("WARNING: no GPU attached. Install/caching steps work on CPU, but the GPU smoke tests are skipped. "
          "Runtime > Change runtime type > T4 GPU (or better) for the full check.")

free_local = ce.require_disk(P.local_root, MIN_LOCAL_FREE_GB, "local scratch (/content)")
free_drive = ce.require_disk(P.drive_root, MIN_DRIVE_FREE_GB, "Google Drive")
print(f"free disk: local {free_local:.1f} GB | Drive {free_drive:.1f} GB")
print("CPUs:", os.cpu_count())
'''),
        md("## 2 · Install CAT + dependencies (cached on Drive)"),
        code(r'''
from pathlib import Path
from persian_asr import install
t = time.time()
CAT_DIR = install.ensure_cat(P, Path(CODE_DIR))
T_INSTALL_CAT = time.time() - t
import importlib; importlib.invalidate_caches()
import cat, cat.ctc.train, kenlm, webdataset, ctcdecode
print(f"CAT ready at {CAT_DIR} @ {ce.git_commit(CAT_DIR)[:10]} in {T_INSTALL_CAT:.0f}s "
      f"(ctcdecode stub: {getattr(ctcdecode, 'IS_STUB', False)})")
VERSIONS = ce.package_versions()
VERSIONS["CAT"] = ce.git_commit(CAT_DIR)
VERSIONS["persian_asr"] = CODE_COMMIT
print(json.dumps(VERSIONS, indent=1))
'''),
        md("## 3 · KenLM binaries (cached on Drive)"),
        code(r'''
from persian_asr import kenlm_tools
t = time.time()
KENLM_BIN = install.ensure_kenlm_bins(P)
KENLM_SMOKE = kenlm_tools.smoke(KENLM_BIN, P.local_work / "kenlm_smoke")
T_KENLM = time.time() - t
print(KENLM_BIN, KENLM_SMOKE, f"{T_KENLM:.0f}s")
'''),
        md("""
## 4 · Kaldi for CAT WFST decoding (optional now; needed from notebook 01 on)

If a cached tarball exists on Drive this only restores it (a few minutes). Otherwise it builds only when
`BUILD_KALDI = True` **and** `QUICK = False`. Building takes about 1–2 h on Colab's 2 vCPUs and
needs no GPU.
"""),
        code(r'''
t = time.time()
KALDI_ROOT = install.ensure_kaldi(P, Path(CODE_DIR), CAT_DIR, build_if_missing=(BUILD_KALDI and not QUICK))
T_KALDI = time.time() - t
KALDI_STATUS = "absent (not built yet)"
if KALDI_ROOT:
    env = {**os.environ, **install.kaldi_env(KALDI_ROOT)}
    probes = {}
    for b in ["cat-latgen-faster", "arpa2fst", "lattice-scale", "lattice-best-path", "fsttablecompose", "fstcompile"]:
        r = subprocess.run([b, "--help"], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
        probes[b] = "ok" if "usage" in (r.stderr + r.stdout).lower() else f"FAIL rc={r.returncode}"
    KALDI_STATUS = "ok" if all(v == "ok" for v in probes.values()) else f"BROKEN {probes}"
    print(probes)
print("Kaldi:", KALDI_STATUS, f"({T_KALDI:.0f}s)", "| cache file:", install.kaldi_cache_file(P))
'''),
        md("## 5 · Whistle checkpoints → Drive"),
        code(r'''
from persian_asr import whistle
CKPT_INFO = {}
for name in CHECKPOINTS:
    d = whistle.download_checkpoint(name, P.pretrained)
    spec = config.WHISTLE_CHECKPOINTS[name]
    f = d / spec["ckpt"]
    sha_file = d / (spec["ckpt"] + ".sha256")
    if not sha_file.is_file():
        sha_file.write_text(whistle.sha256(f))
    CKPT_INFO[name] = {"path": str(f), "size_mb": round(f.stat().st_size / 1e6, 1),
                       "revision": spec["revision"][:10], "sha256": sha_file.read_text()[:16]}
print(json.dumps(CKPT_INFO, indent=1))
'''),
        md("## 6 · Smoke tests"),
        code(r'''
from persian_asr import smoke
SMOKE = {}
# 6a. whistle-small on real English speech -> IPA (checks weights + CAT model code + feature front-end)
lib = P.local_work / "librispeech_dummy.parquet"
if not lib.is_file():
    subprocess.run(["wget", "-q", "-O", str(lib), smoke.LIBRI_DUMMY_URL], check=True)
dev = "cuda" if GPU.available else "cpu"
SMOKE["english_ipa"] = smoke.english_phoneme_check(P.pretrained / "whistle-small", lib, device=dev, n=2)
for u in SMOKE["english_ipa"]["utts"]:
    print("REF:", u["ref_text"]); print("IPA:", u["hyp_ipa"])
print("pass:", SMOKE["english_ipa"]["pass"], "| utt0 identical to CPU reference:",
      SMOKE["english_ipa"]["utt0_matches_cpu_reference"])
'''),
        code(r'''
# 6b. Drive I/O + checkpoint round-trip on Drive (checkpoints are saved there during training)
SMOKE["drive_io"] = smoke.drive_io_benchmark(P.exp / "_io_test", size_mb=128 if QUICK else 512)
SMOKE["ckpt_roundtrip"] = smoke.checkpoint_roundtrip(P.exp / "_io_test")
print(SMOKE["drive_io"], SMOKE["ckpt_roundtrip"])
'''),
        code(r'''
# 6c. Training-step benchmark: Whistle-small Conformer, BPE-500 head, CTC, CAT's AMP mode.
BATCH_PLAN = None
if GPU.available:
    bench = smoke.train_step_benchmark("cuda", BENCH_MICRO_BATCHES, frames=BENCH_FRAMES, vocab=500,
                                       amp=PREC["amp"], steps=3 if QUICK else 5)
    SMOKE["train_bench"] = bench
    for b, r in bench["by_micro_batch"].items():
        print(f"micro-batch {b:3d}: {r}")
    peaks = {b: r.get("peak_mem_gb") for b, r in bench["by_micro_batch"].items() if not r.get("oom")}
    max_micro = ce.max_micro_from_measurements(peaks, GPU.total_mem_gb, headroom=0.80)
    if max_micro:
        micro, accum = ce.plan_batch(EFFECTIVE_BATCH, max_micro)
        thr = bench["by_micro_batch"][max(k for k in peaks if k <= micro)]["audio_s_per_s"]
        BATCH_PLAN = {"effective_batch": EFFECTIVE_BATCH, "micro_batch": micro, "grad_accum_fold": accum,
                      "train_audio_hours_per_gpu_hour": round(thr, 1)}  # (audio s / s) == (audio h / h)
    print("batch plan:", BATCH_PLAN)
else:
    print("skipped (no GPU)")
'''),
        md("## 7 · Summary (copy everything between the ===== lines and paste it to Claude)"),
        code(r'''
summary = {
    "run_id": RUN_ID, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "code_commit": CODE_COMMIT[:10],
    "quick": QUICK, "gpu": GPU.name, "gpu_mem_gb": round(GPU.total_mem_gb, 1), "gpu_cc": GPU.capability,
    "driver": GPU.driver, "precision": PREC, "cpus": os.cpu_count(),
    "free_local_gb": round(ce.disk_free_gb(P.local_root), 1), "free_drive_gb": round(ce.disk_free_gb(P.drive_root), 1),
    "versions": VERSIONS, "t_install_cat_s": round(T_INSTALL_CAT), "kenlm": KENLM_SMOKE,
    "kaldi": KALDI_STATUS, "t_kaldi_s": round(T_KALDI), "checkpoints": CKPT_INFO,
    "smoke": {"english_ipa_pass": SMOKE["english_ipa"]["pass"],
              "english_ipa_matches_cpu_ref": SMOKE["english_ipa"]["utt0_matches_cpu_reference"],
              "english_ipa_utt0": SMOKE["english_ipa"]["utts"][0]["hyp_ipa"],
              "drive_io": SMOKE["drive_io"], "ckpt_roundtrip": SMOKE["ckpt_roundtrip"]["pass"],
              "train_bench": SMOKE.get("train_bench")},
    "batch_plan": BATCH_PLAN, "wall_clock_s": round(time.time() - T0),
}
out = P.results / "setup" / f"setup_{time.strftime('%Y%m%d_%H%M%S')}.json"
ce.write_json(out, summary)
ok = (SMOKE["english_ipa"]["pass"] and SMOKE["ckpt_roundtrip"]["pass"] and KENLM_SMOKE["pass"]
      and not str(KALDI_STATUS).startswith("BROKEN"))
print("=" * 30, "SUMMARY 00_setup", "=" * 30)
print("STATUS:", "OK" if ok else "PROBLEM (see fields)")
print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
print("=" * 78)
print("saved:", out)
'''),
    ]
    return write("00_setup.ipynb", cells)


if __name__ == "__main__":
    print(nb00())
