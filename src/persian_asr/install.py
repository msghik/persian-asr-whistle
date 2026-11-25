"""Build-once / restore-every-session installers for CAT and its native dependencies.

Every artifact that is slow to build is cached on Drive as ONE file (a wheel or a
.tar.gz), keyed by the parts of the environment it depends on, so a new Colab
session restores it in seconds instead of rebuilding:

  <drive>/cache/<os>-py<XY>/wheels/kenlm-*.whl, webdataset-*.whl
  <drive>/cache/<os>/kenlm_bin.tar.gz             (lmplz, build_binary, query)
  <drive>/cache/<os>/kaldi_min-<commit>.tar.gz    (minimal static Kaldi + CAT latgen-faster)

If Colab's base image changes (new Ubuntu / Python), the key changes and things
are rebuilt automatically; old caches are left alone (never deleted by code).
"""
from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from . import config

PY_DEPS = [
    # CAT requirements.txt minus: the CUDA-extension git deps (stubbed, see
    # third_party/cat_ext_stubs), webdataset (pinned wheel below) and the
    # protobuf==3.20.2 pin (only a workaround for old tensorboard; pinning it
    # would downgrade Colab's protobuf and break other preinstalled packages).
    "tqdm", "matplotlib", "sentencepiece", "kaldiio", "tensorboard", "jiwer",
    "pyyaml", "transformers", "jieba", "Levenshtein", "braceexpand",
    "huggingface_hub",
]
APT_KENLM = ["libboost-program-options-dev", "libboost-system-dev", "libboost-thread-dev",
             "libboost-test-dev", "libeigen3-dev", "zlib1g-dev", "libbz2-dev", "liblzma-dev", "cmake"]
APT_KALDI = ["automake", "autoconf", "libtool", "gfortran", "sox", "subversion", "zlib1g-dev", "bc",
             "libopenblas-dev", "liblapacke-dev"]


# --- small helpers ----------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[install {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str] | str, log_file: str | os.PathLike | None = None, cwd: str | None = None,
        env: dict | None = None, tail: int = 60) -> None:
    """Run a command; on failure print the last `tail` lines and raise."""
    shell = isinstance(cmd, str)
    full_env = {**os.environ, **(env or {})}
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            r = subprocess.run(cmd, shell=shell, cwd=cwd, env=full_env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            lines = Path(log_file).read_text(errors="replace").splitlines()[-tail:]
            print("\n".join(lines))
            raise RuntimeError(f"command failed ({r.returncode}): {cmd}\nfull log: {log_file}")
    else:
        r = subprocess.run(cmd, shell=shell, cwd=cwd, env=full_env, capture_output=True, text=True)
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-6000:])
            raise RuntimeError(f"command failed ({r.returncode}): {cmd}")


def pip(*args: str, log_file: str | os.PathLike | None = None) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", *args], log_file=log_file)


def os_key() -> str:
    try:
        info = platform.freedesktop_os_release()
        return f"{info.get('ID', 'linux')}{info.get('VERSION_ID', '')}"
    except Exception:  # noqa: BLE001
        return "linux"


def py_key() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def _copy_to_drive(local: Path, drive_dst: Path) -> None:
    """Copy one big file to Drive atomically (tmp name, then rename)."""
    drive_dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = drive_dst.with_name(drive_dst.name + ".partial")
    shutil.copyfile(local, tmp)
    os.replace(tmp, drive_dst)


def _make_tar(src_dir: Path, arcname: str, out: Path, exclude_globs: tuple[str, ...] = ()) -> None:
    def _filter(ti: tarfile.TarInfo):
        for g in exclude_globs:
            if Path(ti.name).match(g):
                return None
        return ti

    out.parent.mkdir(parents=True, exist_ok=True)
    # gzip level 1: these are mostly binaries; speed matters more than size.
    with tarfile.open(out, "w:gz", compresslevel=1) as t:
        t.add(src_dir, arcname=arcname, filter=_filter)


def missing_shared_libs(binary: Path) -> list[str]:
    """Shared libraries `binary` needs that the dynamic loader cannot find (via ldd)."""
    r = subprocess.run(["ldd", str(binary)], capture_output=True, text=True)
    return [ln.split()[0] for ln in r.stdout.splitlines() if "not found" in ln]


def ensure_runtime_libs(binaries: list[Path], apt_pkgs: list[str]) -> None:
    """A binary restored from the Drive cache into a fresh VM may lack runtime .so files
    (e.g. KenLM -> libboost_*, Kaldi -> libgfortran). Install them only if missing."""
    missing = sorted({m for b in binaries for m in missing_shared_libs(b)})
    if missing:
        log(f"missing shared libs {missing}; installing {apt_pkgs}")
        apt_install(apt_pkgs)
        still = sorted({m for b in binaries for m in missing_shared_libs(b)})
        if still:
            raise RuntimeError(f"still missing shared libraries after apt install: {still}")


def apt_install(pkgs: list[str]) -> None:
    missing = [p for p in pkgs if subprocess.run(["dpkg", "-s", p], capture_output=True).returncode != 0]
    if not missing:
        return
    log(f"apt-get install {' '.join(missing)}")
    sudo = "" if os.geteuid() == 0 else "sudo "  # Colab runs as root; the server user has sudo
    run(f"{sudo}apt-get update -qq && {sudo}env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq " + " ".join(missing))


# --- python deps + CAT --------------------------------------------------------------

def _cached_wheel(cache_dir: Path, pattern: str, build_spec: str, tmp_dir: Path,
                  fallback_build_reqs: tuple[str, ...] = ()) -> Path:
    """Return a cached wheel matching `pattern`, building it with `pip wheel` if absent.

    Build output is logged in full (no -q) so a failure shows the real compiler/CMake error.
    If the normal (isolated) build fails and `fallback_build_reqs` is given, those packages
    are installed and the build is retried once with --no-build-isolation.
    """
    hits = sorted(cache_dir.glob(pattern))
    if hits:
        return hits[-1]
    log(f"building wheel {build_spec} (first time only)")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # CMake >= 4 refuses projects declaring cmake_minimum_required < 3.5 unless this is set.
    env = {"CMAKE_POLICY_VERSION_MINIMUM": "3.5"}
    cmd = [sys.executable, "-m", "pip", "wheel", "-v", "--no-deps", "-w", str(tmp_dir), build_spec]
    try:
        run(cmd, log_file=tmp_dir / "pip_wheel.log", env=env)
    except RuntimeError:
        if not fallback_build_reqs:
            raise
        log(f"isolated build failed; retrying with --no-build-isolation and {list(fallback_build_reqs)}")
        pip(*fallback_build_reqs)
        run(cmd[:4] + ["--no-build-isolation"] + cmd[4:], log_file=tmp_dir / "pip_wheel_retry.log", env=env)
    built = sorted(tmp_dir.glob(pattern))
    if not built:
        raise RuntimeError(f"wheel build produced nothing matching {pattern} in {tmp_dir}")
    dst = cache_dir / built[-1].name
    _copy_to_drive(built[-1], dst)
    return dst


def ensure_torchaudio() -> str:
    """CAT imports torchaudio (cat/shared/encoder.py:8). Install a build matching the
    preinstalled torch if Colab no longer ships it; never upgrade torch itself."""
    try:
        import torchaudio  # noqa: F401
        from torchaudio.compliance import kaldi  # noqa: F401

        return torchaudio.__version__
    except Exception:  # noqa: BLE001
        import torch

        log(f"torchaudio missing; installing one compatible with torch=={torch.__version__}")
        pip("torchaudio", f"torch=={torch.__version__}")
        import torchaudio

        return torchaudio.__version__


def ensure_cat(paths: config.Paths, repo_root: Path) -> Path:
    """Clone CAT at the pinned commit to local disk and install it (+deps) into this kernel."""
    cat_dir = paths.local_tools / "CAT"
    if not (cat_dir / ".git").is_dir():
        log(f"cloning CAT @ {config.CAT_COMMIT[:10]}")
        run(["git", "clone", "-q", config.CAT_REPO, str(cat_dir)])
    run(["git", "-C", str(cat_dir), "fetch", "-q", "--depth", "1", "origin", config.CAT_COMMIT])
    run(["git", "-C", str(cat_dir), "checkout", "-q", config.CAT_COMMIT])

    wheel_cache = paths.cache / f"{os_key()}-{py_key()}" / "wheels"
    tmp = paths.local_work / "wheel_build"
    wds = _cached_wheel(wheel_cache, "webdataset-*.whl",
                        f"git+https://github.com/webdataset/webdataset.git@{config.WEBDATASET_COMMIT}", tmp)
    if not sorted(wheel_cache.glob("kenlm-*.whl")):
        apt_install(APT_KENLM)  # compiler-side deps (cmake, zlib/bz2/lzma headers) before building
    klm = _cached_wheel(wheel_cache, "kenlm-*.whl", config.KENLM_URL, tmp,
                        fallback_build_reqs=("setuptools", "wheel", "cmake>=3.10,<4"))

    ensure_torchaudio()
    log("pip install CAT python deps + cached wheels + extension stubs")
    pip(*PY_DEPS, str(wds), str(klm), str(repo_root / "third_party" / "cat_ext_stubs"))
    pip("--no-deps", "-e", str(cat_dir))
    # make the freshly installed editable package importable in this running kernel
    if str(cat_dir) not in sys.path:
        sys.path.insert(0, str(cat_dir))
    return cat_dir


# --- KenLM binaries (lmplz / build_binary) -------------------------------------------

def ensure_kenlm_bins(paths: config.Paths) -> Path:
    bin_dir = paths.local_tools / "kenlm" / "bin"
    if (bin_dir / "lmplz").is_file() and (bin_dir / "build_binary").is_file():
        return bin_dir
    cached = paths.cache / os_key() / "kenlm_bin.tar.gz"
    if cached.is_file():
        log("restoring KenLM binaries from Drive cache")
        with tarfile.open(cached) as t:
            t.extractall(paths.local_tools / "kenlm")
    else:
        log("building KenLM binaries (first time only, ~5 min)")
        apt_install(APT_KENLM)
        src = paths.local_work / "kenlm_src"
        if not src.is_dir():
            run(f"mkdir -p {src} && cd {src} && wget -q -O kenlm.zip {config.KENLM_URL} && "
                "unzip -q kenlm.zip && mv kenlm-master/* . ")
        build = src / "build"
        build.mkdir(exist_ok=True)
        run(f"cmake .. -DCMAKE_BUILD_TYPE=Release && make -j{os.cpu_count()} lmplz build_binary query",
            cwd=str(build), log_file=paths.local_work / "kenlm_build.log")
        bin_dir.mkdir(parents=True, exist_ok=True)
        for b in ("lmplz", "build_binary", "query"):
            shutil.copy2(build / "bin" / b, bin_dir / b)
        tmp_tar = paths.local_work / "kenlm_bin.tar.gz"
        _make_tar(bin_dir, "bin", tmp_tar)
        _copy_to_drive(tmp_tar, cached)
    ensure_runtime_libs([bin_dir / "lmplz", bin_dir / "build_binary"], APT_KENLM)
    return bin_dir


# --- Kaldi (minimal, for CAT WFST decoding) --------------------------------------------

def kaldi_cache_file(paths: config.Paths) -> Path:
    return paths.cache / os_key() / f"kaldi_min-{config.KALDI_COMMIT[:10]}.tar.gz"


def ensure_kaldi(paths: config.Paths, repo_root: Path, cat_dir: Path, build_if_missing: bool) -> Path | None:
    """Restore (or build once) the minimal Kaldi tree at <local_tools>/kaldi.

    The tree must always live at the same absolute path because Kaldi's kaldi.mk and
    OpenFST's libtool binaries embed it. Returns KALDI_ROOT or None if absent and
    build_if_missing is False.
    """
    kaldi_root = paths.local_tools / "kaldi"
    marker = kaldi_root / "src" / "bin" / "cat-latgen-faster"
    cached = kaldi_cache_file(paths)
    if not marker.is_file():
        if cached.is_file():
            log(f"restoring Kaldi from Drive cache ({cached.stat().st_size / 1e9:.2f} GB)")
            with tarfile.open(cached) as t:
                t.extractall(paths.local_tools)
        elif build_if_missing:
            log("building minimal Kaldi (first time only; ~1-2 h on 2 vCPUs)")
            apt_install(APT_KALDI)
            run(["bash", str(repo_root / "scripts" / "build_kaldi_min.sh"), str(kaldi_root), str(cat_dir),
                 str(os.cpu_count() or 2)],
                log_file=paths.local_work / "kaldi_build.log",
                env={"KALDI_COMMIT": config.KALDI_COMMIT})
            tmp_tar = paths.local_work / cached.name
            log("packing Kaldi tree for the Drive cache")
            _make_tar(kaldi_root, "kaldi", tmp_tar, exclude_globs=("*.o", ".git"))
            _copy_to_drive(tmp_tar, cached)
        else:
            return None
    ensure_runtime_libs([marker, kaldi_root / "tools" / "openfst" / "bin" / "fstcompile"],
                        ["libgfortran5", "libopenblas-dev"])
    # CAT looks for latgen-faster in ../../src/bin relative to the egs dir (cat/ctc/fst_decode.sh)
    (cat_dir / "src" / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy2(marker, cat_dir / "src" / "bin" / "latgen-faster")
    return kaldi_root


def kaldi_env(kaldi_root: Path) -> dict[str, str]:
    """PATH additions for Kaldi + OpenFST binaries."""
    dirs = [kaldi_root / "src" / d for d in ("bin", "fstbin", "latbin", "lmbin")] + [kaldi_root / "tools" / "openfst" / "bin"]
    return {"KALDI_ROOT": str(kaldi_root),
            "PATH": os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])}


def list_glob(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))
