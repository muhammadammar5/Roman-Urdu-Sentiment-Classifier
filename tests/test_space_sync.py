from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_space_normalizer_matches_source():
    """space/normalize.py is a copy (a Space only ships its own folder). Keep them identical."""
    assert (ROOT / "src" / "normalize.py").read_text(encoding="utf-8") == \
           (ROOT / "space" / "normalize.py").read_text(encoding="utf-8")
