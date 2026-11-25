import importlib.util
from pathlib import Path

from persian_asr.normalize_fa import ZWNJ

spec = importlib.util.spec_from_file_location("analysis", Path(__file__).resolve().parents[1] / "scripts/analysis.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_analyse_counts():
    mirvam = "می" + ZWNJ + "روم"
    refs = {"a": f"من {mirvam} خانه", "b": "کتاب شیر"}
    hyps = {"a": "من می روم خانه", "b": "کتب شیر اضافه"}
    raws = {"a": f"من {mirvam} خانهٔ", "b": "کتاب شیر"}
    vocab = {"من", "خانه", "کتاب", mirvam}
    r = analysis.analyse(refs, hyps, raws, vocab)
    assert (r["sub"], r["del"], r["ins"], r["n_ref_words"]) == (2, 0, 2, 5)
    assert r["zwnj_ref_words"] == 1 and r["zwnj_word_error_pct"] == 100.0
    assert r["space_or_zwnj_only_errors"] == 1          # می‌روم -> می روم
    assert r["vowel_letter_only_substitutions"] == 1    # کتاب -> کتب
    assert r["oov_ref_words"] == 1 and r["oov_word_error_pct"] == 0.0  # "شیر" OOV but correct
    assert r["ezafe_ref_words"] == 1 and r["ezafe_word_error_pct"] == 0.0  # خانهٔ -> خانه, correct
