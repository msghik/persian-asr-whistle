import csv
from pathlib import Path

import pandas as pd
import pytest

from persian_asr import catrun, cvdata, decoding
from persian_asr.normalize_fa import ZWNJ, normalize_with_flags

COLS = ["client_id", "path", "sentence_id", "sentence", "sentence_domain", "up_votes", "down_votes",
        "age", "gender", "accents", "variant", "locale", "segment"]


def _write(p: Path, rows: list[dict], cols=COLS):
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


@pytest.fixture()
def cv(tmp_path):
    """Tiny CV release: 2 dev, 2 test, 8 validated clips with sentence/speaker overlaps."""
    root = tmp_path / "fa"
    root.mkdir()
    sents = {f"s{i}": f"جمله{i} می{ZWNJ}رود" for i in range(8)}
    clip = lambda i, spk, sid: {"client_id": spk, "path": f"c{i}.mp3", "sentence_id": sid,
                                "sentence": sents[sid].replace(ZWNJ, "")}  # CV 27 drops ZWNJ in clip TSVs
    dev = [clip(0, "dA", "s0"), clip(1, "dB", "s1")]
    test = [clip(2, "tA", "s2"), clip(3, "tB", "s3")]
    val = dev + test + [clip(4, "x", "s2"),   # test sentence, other speaker -> leaks under "cat"
                        clip(5, "tA", "s4"),  # test speaker -> leaks under "cat"
                        clip(6, "y", "s5"), clip(7, "z", "s6"), clip(8, "y", "s7"), clip(9, "z", "s5")]
    _write(root / "dev.tsv", dev)
    _write(root / "test.tsv", test)
    _write(root / "validated.tsv", val)
    _write(root / "train.tsv", [clip(6, "y", "s5")])
    _write(root / "validated_sentences.tsv", [{"sentence_id": k, "sentence": v} for k, v in sents.items()],
           cols=["sentence_id", "sentence"])
    (root / "clip_durations.tsv").write_text("clip\tduration[ms]\n" + "".join(f"c{i}.mp3\t{1800000}\n" for i in range(10)))
    return root


def test_pool_conventions_and_leakage(cv):
    rel = cvdata.CVRelease(cv, text_from_sentences_tsv=True)
    dur = rel.durations()
    pool_cat, dev, test = cvdata.build_pool(rel, "cat", dur)
    pool_clean, _, _ = cvdata.build_pool(rel, "clean", dur)
    assert set(pool_cat.uid) == {"c4", "c5", "c6", "c7", "c8", "c9"}
    assert set(pool_clean.uid) == {"c6", "c7", "c8", "c9"}
    n = lambda df: cvdata.apply_norm(df, normalize_with_flags)
    lk = cvdata.leakage(n(pool_cat), n(test))
    assert lk["sentence_id_in_train_pct"] == 50.0 and lk["speaker_in_train_pct"] == 50.0
    lk = cvdata.leakage(n(pool_clean), n(test))
    assert lk["sentence_id_in_train_pct"] == 0.0 and lk["speaker_in_train_pct"] == 0.0


def test_text_from_sentences_tsv_restores_zwnj(cv):
    dur = cvdata.CVRelease(cv, True).durations()
    a = cvdata.CVRelease(cv, True).load("dev", dur)
    b = cvdata.CVRelease(cv, False).load("dev", dur)
    assert all(ZWNJ in t for t in a.text_raw) and not any(ZWNJ in t for t in b.text_raw)


def test_nested_subsets_are_nested(cv):
    rel = cvdata.CVRelease(cv, True)
    pool, _, _ = cvdata.build_pool(rel, "cat", rel.durations())  # 6 clips x 0.5 h
    sub = cvdata.nested_subsets(pool, seed=0, hours={"a": 0.4, "b": 1.0, "c": 2.0})
    a, b, c, full = (list(sub[k].uid) for k in ("a", "b", "c", "full"))
    assert len(a) == 1 and len(b) == 2 and len(c) == 4 and len(full) == 6
    assert a == b[:1] and b == c[:2] and c == full[:4]


def test_score_matches_cat_definitions():
    refs = {"u1": "a b c d", "u2": "x y"}
    hyps = {"u1": "a  c d e", "u2": "x y"}
    s = decoding.score(refs, hyps)
    assert (s["sub"], s["del"], s["ins"], s["n_ref"]) == (0, 1, 1, 6) and s["er"] == round(100 * 2 / 6, 2)
    assert s["ser"] == 50.0
    c = decoding.score({"u": "ab cd"}, {"u": "abd"}, cer=True)  # spaces removed, then chars
    assert c["n_ref"] == 4 and c["del"] == 1


def test_read_checklist(tmp_path, monkeypatch):
    monkeypatch.setattr(catrun, "DATA", tmp_path)
    exp = tmp_path / "exp" / "r"
    (exp / "check").mkdir(parents=True)
    (exp / "check" / "checkpoint.list").write_text(
        "# header\n" + "\n".join("    ".join([f"{e:04}", f"{s:#010x}", float(m).hex(), float(3e-5).hex(), f"exp/r/check/c{s}.pt"])
                                 for e, s, m in [(1, 8, 5.0), (2, 16, 4.0)]))
    cl = catrun.read_checklist(exp)
    assert [c["step"] for c in cl] == [8, 16] and cl[1]["metric"] == 4.0
