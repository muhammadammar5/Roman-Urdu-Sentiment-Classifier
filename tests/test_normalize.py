from src.normalize import normalize_text


def test_lowercase_and_whitespace():
    assert normalize_text("  Yeh   PHONE  ") == "yeh phone"


def test_removes_urls_mentions_html_hashtag_symbol():
    out = normalize_text("@ali check https://x.com/a <b>bohat</b> #acha")
    assert out == "check bohat acha"


def test_collapses_stretched_letters_and_punctuation():
    assert normalize_text("soooooo goood!!!!") == "soo good!"


def test_maps_spelling_variants():
    assert normalize_text("Achha accha bohot bht nhi") == "acha acha bohat bohat nahi"


def test_variant_lookup_after_squeezing_repeats():
    assert normalize_text("bohooot") == "bohat"   # "ooo" -> "oo" -> squeezed "bohot" -> "bohat"
    assert normalize_text("achhhha") == "acha"
    assert normalize_text("totally unknownnn") == "totally unknownn"  # unknown words only de-stretched


def test_ambiguous_words_not_merged():
    # kia (did) and kya (what) must stay different
    assert normalize_text("kia kya") == "kia kya"


def test_handles_none_and_empty():
    assert normalize_text(None) == ""
    assert normalize_text("   ") == ""


def test_is_idempotent():
    text = "Bohooot  achhaaa!!! @user http://a.b"
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_keeps_urdu_script():
    assert normalize_text("یہ بہت اچھا ہے") == "یہ بہت اچھا ہے"
