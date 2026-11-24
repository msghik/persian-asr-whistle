"""Thin wrappers around KenLM's lmplz / build_binary (used by the setup smoke test and 02)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def build_arpa(lmplz: Path, text_file: Path, arpa_out: Path, order: int = 4,
               extra: tuple[str, ...] = ()) -> None:
    with open(text_file, "rb") as fin, open(arpa_out, "wb") as fout:
        r = subprocess.run([str(lmplz), "-o", str(order), *extra], stdin=fin, stdout=fout,
                           stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"lmplz failed: {r.stderr.decode(errors='replace')[-2000:]}")


def build_binary(build_binary_bin: Path, arpa: Path, out: Path) -> None:
    r = subprocess.run([str(build_binary_bin), str(arpa), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"build_binary failed: {r.stderr[-2000:]}")


def smoke(bin_dir: Path, work: Path) -> dict:
    """Train a tiny 3-gram, binarize it, and score a sentence with the python module."""
    import kenlm

    work.mkdir(parents=True, exist_ok=True)
    txt = work / "tiny.txt"
    txt.write_text("سلام دنیا\nسلام بر شما\nدنیا زیباست\n" * 20, encoding="utf-8")
    arpa, binf = work / "tiny.arpa", work / "tiny.bin"
    build_arpa(bin_dir / "lmplz", txt, arpa, order=3, extra=("--discount_fallback",))
    build_binary(bin_dir / "build_binary", arpa, binf)
    m = kenlm.Model(str(binf))
    s_in, s_oov = m.score("سلام دنیا", bos=True, eos=True), m.score("خداحافظ دنیا", bos=True, eos=True)
    return {"pass": m.order == 3 and s_in > s_oov, "order": m.order,
            "logp_in_domain": round(s_in, 3), "logp_with_oov": round(s_oov, 3)}
