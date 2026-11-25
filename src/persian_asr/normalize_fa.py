"""Persian text normalization, applied identically to training text, LM text,
references and hypotheses.

Policy (documented choices; see tests/test_normalize_fa.py):

* Unicode: Arabic presentation forms (U+FB50-U+FDFF, U+FE70-U+FEFF; present in CV 27.0
  fa, e.g. "ﮐ", "ﯾ") are folded with NFKC; the rest of the string is left as is
  (full-string NFKC would also rewrite ligatures and digits we handle explicitly).
* Letters: Arabic yeh ي and alef maksura ى -> Persian yeh ی; Arabic kaf ك -> Persian
  kaf ک; yeh barree ے -> ی; ae ە, heh goal ہ and heh-with-yeh ۀ -> ه; alef with hamza أ/إ and
  alef wasla ٱ -> ا. Kept: آ ؤ ئ ء ة.
* Diacritics removed: harakat and tanvin (U+064B-U+0652), hamza above/below
  (U+0654/U+0655; in CV fa mostly the written ezafe on ه), superscript alef (U+0670).
  Short vowels and the ezafe are mostly not written in Persian anyway, so removing the
  few written ones makes references consistent.
* Tatweel (U+0640) removed.
* ZWNJ (U+200C) is KEPT as a word-internal character: "می‌روم" is one word and
  "می روم" is two. This is CV's dominant spelling; WER therefore counts a ZWNJ/space
  mismatch as an error, and the analysis step reports those errors separately.
  ZWNJ is cleaned: runs collapse to one, and ZWNJ next to a space, punctuation or the
  string edge is dropped. ZWJ, zero-width space, soft hyphen, LRM/RLM/ALM and BOM are
  removed, as are the bidi controls U+202A-U+202E.
* Source text: CV 27.0 fa clip TSVs (validated/train/dev/test.tsv) have ZWNJ DELETED
  from the sentence column (e.g. "می‌شد" -> "میشد"; verified 2026-11-25), while
  validated_sentences.tsv keeps it. Data prep therefore takes text from
  validated_sentences.tsv via sentence_id (see persian_asr.cvdata).
* Digits: Persian (U+06F0-) and Arabic-Indic (U+0660-) digits -> ASCII; utterances
  with any digit are FLAGGED ("digit"), never dropped here (digits are read out as
  words, so the transcript does not match the audio literally).
* Latin letters are lower-cased and FLAGGED ("latin").
* Punctuation (Persian and ASCII, incl. ، ؛ ؟ « » ٬ … – — “ ” and brackets) -> space;
  then whitespace is collapsed and stripped.
"""
from __future__ import annotations

import re
import unicodedata

_CHAR_MAP = {
    "ي": "ی",  # ي -> ی
    "ى": "ی",  # ى -> ی
    "ے": "ی",  # ے -> ی
    "ك": "ک",  # ك -> ک
    "ە": "ه",  # ە -> ه
    "ۀ": "ه",  # ۀ -> ه
    "ہ": "ه",  # ہ (heh goal) -> ه
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "ٱ": "ا",  # ٱ -> ا
}
for _i in range(10):
    _CHAR_MAP[chr(0x06F0 + _i)] = str(_i)  # Persian digits
    _CHAR_MAP[chr(0x0660 + _i)] = str(_i)  # Arabic-Indic digits

_REMOVE = (
    [chr(c) for c in range(0x064B, 0x0653)]  # harakat, tanvin, shadda, sukun
    + ["ٔ", "ٕ", "ٰ", "ـ"]  # hamza above/below, superscript alef, tatweel
    + ["​", "‍", "­", "‎", "‏", "؜", "﻿"]
    + [chr(c) for c in range(0x202A, 0x202F)]  # bidi embedding/override controls
)
for _c in _REMOVE:
    _CHAR_MAP[_c] = None
_TRANS = str.maketrans(_CHAR_MAP)

ZWNJ = "‌"
_PUNCT = (
    "،؛؟٪٫٬۔«»…–—‐‑"
    "“”„‘’·•"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)
_PUNCT_RE = re.compile("[" + re.escape(_PUNCT) + "]")
_PRESENTATION_RE = re.compile("[ﭐ-﷿ﹰ-﻿]+")
_WS_RE = re.compile(r"\s+")
_ZWNJ_RUN = re.compile(ZWNJ + "{2,}")
_ZWNJ_EDGE = re.compile(r"(?:\s*" + ZWNJ + r"\s+|\s+" + ZWNJ + r"\s*)")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ɏ]")
_DIGIT_RE = re.compile(r"[0-9]")


def normalize(text: str) -> str:
    return normalize_with_flags(text)[0]


def normalize_with_flags(text: str) -> tuple[str, list[str]]:
    """Return (normalized text, flags). Flags: 'latin', 'digit', 'empty'."""
    s = _PRESENTATION_RE.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text)
    s = s.translate(_TRANS)
    s = _PUNCT_RE.sub(" ", s)
    s = _ZWNJ_RUN.sub(ZWNJ, s)
    s = _ZWNJ_EDGE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = s.strip(ZWNJ).strip()
    if _LATIN_RE.search(s):
        s = s.lower()
    flags = []
    if _LATIN_RE.search(s):
        flags.append("latin")
    if _DIGIT_RE.search(s):
        flags.append("digit")
    if not s:
        flags.append("empty")
    return s, flags


def normalize_pl_cat(text: str) -> str:
    """Exact port of CAT's Polish normalization (egs/cv-lang10/lang-process/pl/text_norm.sh):
    , " “ ; — . : ! ” ? « » -> space; collapse/strip spaces; lower-case."""
    s = re.sub(r"[,\"“;—.:!”?«»]", " ", text)
    s = re.sub(r" +", " ", s).strip()
    return s.lower()
