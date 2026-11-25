from persian_asr.normalize_fa import ZWNJ, normalize, normalize_pl_cat, normalize_with_flags


def test_arabic_letters_to_persian():
    assert normalize("كتاب علي") == "کتاب علی"
    assert normalize("مصطفى") == "مصطفی"


def test_presentation_forms_folded():
    assert normalize("ﮎﯾ") == "کی"  # ﮎ ﯾ presentation forms -> ک ی


def test_digits_converted_and_flagged():
    s, f = normalize_with_flags("سال ۱۴۰۲ و ٣")
    assert s == "سال 1402 و 3" and "digit" in f


def test_latin_flagged_and_lowercased():
    s, f = normalize_with_flags("برنامه Python")
    assert s == "برنامه python" and f == ["latin"]


def test_punctuation_removed():
    assert normalize("«سلام»، خوبی؟ آره؛ باشه!") == "سلام خوبی آره باشه"


def test_harakat_tanvin_hamza_tatweel_removed():
    assert normalize("اصلاً") == "اصلا"
    assert normalize("کِتابُ") == "کتاب"
    assert normalize("خانهٔ من") == "خانه من"
    assert normalize("خانۀ من") == "خانه من"
    assert normalize("بـــزرگ") == "بزرگ"


def test_zwnj_policy():
    w = "می" + ZWNJ + "روم"
    assert normalize(w) == w  # kept word-internally
    assert normalize("می" + ZWNJ * 3 + "روم") == w  # runs collapse
    assert normalize(ZWNJ + "می " + ZWNJ + " روم" + ZWNJ) == "می روم"  # edges dropped
    assert normalize("می‍روم") == "میروم"  # ZWJ removed
    assert normalize("می" + ZWNJ + "، روم") == "می روم"


def test_whitespace_and_empty():
    assert normalize("  سلام \t  دنیا \n") == "سلام دنیا"
    assert normalize_with_flags(" ؟ ")[1] == ["empty"]


def test_idempotent():
    for t in ["«کتابِ» علي، ۱۲ Python!", "می" + ZWNJ + "روم  خانهٔ"]:
        assert normalize(normalize(t)) == normalize(t)


def test_polish_cat_norm():
    assert normalize_pl_cat("Ala, ma „kota”? Tak!  Ma.") == "ala ma „kota tak ma"


def test_heh_goal_and_bidi_controls():
    assert normalize("\u202bکتابہ\u202c") == "کتابه"
