"""Execute a generated notebook locally (CPU) with google.colab faked, for sanity checks.

- DRIVE_ROOT / LOCAL_ROOT in the settings cell are redirected to a scratch dir;
- REPO_URL is blanked, so the bootstrap uses the "repo zip on Drive" fallback,
  and this script builds that zip from the working tree;
- extra settings overrides: --set NAME=python_literal (e.g. --set QUICK=False).

Usage:
  python scripts/run_notebook_locally.py notebooks/00_setup.ipynb --scratch /tmp/x [--kernel-python /path/to/python]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

FAKE_COLAB = {
    "google/colab/__init__.py": "",
    "google/colab/drive.py": "def mount(path, **kw):\n    print('[fake colab] drive.mount', path)\n",
    "google/colab/userdata.py": "def get(name):\n    raise KeyError(name)\n",
    # dev-machine-only workaround: torch CPU oneDNN SIGFPE on AMD Zen4 (see tests/conftest.py)
    "sitecustomize.py": "try:\n    import torch\n    torch.backends.mkldnn.enabled = False\nexcept Exception:\n    pass\n",
}


def zip_repo(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    skip = {".venv", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob("*"):
            if any(part in skip for part in p.relative_to(ROOT).parts) or p.is_dir():
                continue
            z.write(p, p.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--kernel-python", default=sys.executable)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    scratch = Path(a.scratch).resolve()
    drive, local, fake = scratch / "drive" / "persian_asr", scratch / "local", scratch / "fake_mods"
    for rel, src in FAKE_COLAB.items():
        (fake / rel).parent.mkdir(parents=True, exist_ok=True)
        (fake / rel).write_text(src)
    zip_repo(drive / "code" / "persian_asr.zip")

    nb = nbformat.read(a.notebook, as_version=4)
    settings = nb.cells[2]
    assert "SETTINGS" in settings.source, "cell 2 must be the settings cell"
    src = settings.source
    src = re.sub(r'^DRIVE_ROOT\s*=.*$', f'DRIVE_ROOT = "{drive}"', src, flags=re.M)
    src = re.sub(r'^LOCAL_ROOT\s*=.*$', f'LOCAL_ROOT = "{local}"', src, flags=re.M)
    src = re.sub(r'^REPO_URL\s*=.*$', 'REPO_URL = ""', src, flags=re.M)
    for kv in a.set:
        k, v = kv.split("=", 1)
        src, n = re.subn(rf'^{k}\s*=.*$', f"{k} = {v}", src, flags=re.M)
        assert n == 1, f"setting {k} not found"
    settings.source = src
    # bootstrap clones into /content/persian_asr; redirect for local runs
    for c in nb.cells:
        if c.cell_type == "code":
            c.source = c.source.replace('CODE_DIR = "/content/persian_asr"', f'CODE_DIR = "{scratch / "code"}"')

    # kernel = the given python (e.g. a fresh venv mimicking Colab)
    kdir = scratch / "kernels" / "nbtest"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "kernel.json").write_text(
        '{"argv": ["%s", "-m", "ipykernel_launcher", "-f", "{connection_file}"], '
        '"display_name": "nbtest", "language": "python"}' % a.kernel_python)
    os.environ["JUPYTER_PATH"] = str(scratch)
    os.environ["PYTHONPATH"] = str(fake)

    client = NotebookClient(nb, timeout=a.timeout, kernel_name="nbtest",
                            resources={"metadata": {"path": str(scratch)}})
    try:
        client.execute()
    finally:
        out = scratch / ("executed_" + Path(a.notebook).name)
        nbformat.write(nb, out)
        print("executed notebook:", out)
    for c in nb.cells:
        if c.cell_type == "code":
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    sys.stdout.write(o["text"])
                elif o.get("output_type") == "error":
                    print("ERROR:", o["ename"], o["evalue"])


if __name__ == "__main__":
    main()
