"""Roman Urdu text normalisation.

Roman Urdu has no standard spelling, so the same word appears in many forms
("acha", "achha", "accha"). Normalising these variants shrinks the effective
vocabulary and helps the model generalise.

Steps (in order):
  1. Unicode NFKC normalisation and lower-casing
  2. Remove URLs, @mentions, HTML tags and the '#' of hashtags
  3. Collapse stretched letters ("soooo" -> "soo") and repeated "!!!"/"???"
  4. Drop symbols/emoji, keeping letters, digits and ! ?
  5. Map known spelling variants to one canonical form (VARIANTS below)

The variant table is intentionally conservative: ambiguous tokens (for example
"kia"/"kiya" vs "kya", or "main"/"mein") are NOT merged, because merging them
would change meaning. Extend VARIANTS by inspecting the most frequent tokens
in your data.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List

# canonical form -> spelling variants seen in the wild
VARIANTS: Dict[str, List[str]] = {
    "acha": ["achha", "accha", "achaa", "achcha", "acchaa"],
    "achi": ["achhi", "acchi", "achee", "achiii"],
    "bohat": ["bohot", "bohut", "boht", "bhot", "bahut", "buhat", "bohhat",
              "bhut", "bht", "bahot"],
    "nahi": ["nahin", "nahee", "nhi", "nehi", "nahii"],
    "zabardast": ["zabardust", "zabrdast", "zabardasth", "zbardast"],
    "shukriya": ["shukria", "shukriah", "shukriyaa", "shukeriya"],
    "theek": ["thik", "thek", "theak", "theeck"],
    "kharab": ["kharaab", "kharb"],
    "bekar": ["bekaar"],
    "bilkul": ["bilkol", "bilkool", "bilqul", "bilkull"],
    "lekin": ["lakin", "laken", "lekan", "likan", "lkn"],
    "pasand": ["pasnd", "pasaand"],
    "mehnga": ["mehenga", "mahenga", "mahnga"],
    "sasta": ["sastaa"],
    "kuch": ["kuchh", "kch"],
    "kyun": ["kyon", "kiun", "kiyun"],
    "mujhe": ["mujhy", "mujhay", "mjhe"],
    "behtareen": ["bahtareen", "behtreen", "bahtreen", "behtarin"],
    "shandar": ["shaandaar", "shandaar"],
    "bakwas": ["bakwaas", "bkwas"],
    "hai": ["hy", "hay", "hae"],
    "hain": ["hein"],
    "wah": ["waah", "wahh"],
    "mashallah": ["mashaallah", "mashaalah"],
}

_VARIANT_MAP: Dict[str, str] = {
    variant: canonical
    for canonical, variants in VARIANTS.items()
    for variant in variants
}

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_MENTION_RE = re.compile(r"@\w+")
_HTML_RE = re.compile(r"<[^>]+>")
_REPEAT_RE = re.compile(r"(.)\1{2,}")          # 3+ of the same char -> 2
_PUNCT_RUN_RE = re.compile(r"([!?])[!?]+")     # "!!!" or "?!?" -> first char
_KEEP_RE = re.compile(r"[^\w\s!?]", re.UNICODE)
_SQUEEZE_RE = re.compile(r"(.)\1+")            # any repeat -> single (lookup only)
_SPACE_RE = re.compile(r"\s+")


def _canonical(token: str) -> str:
    if token in _VARIANT_MAP:
        return _VARIANT_MAP[token]
    squeezed = _SQUEEZE_RE.sub(r"\1", token)
    return _VARIANT_MAP.get(squeezed, token)


def normalize_text(text: object) -> str:
    """Normalise one review. Always returns a string (possibly empty)."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = text.replace("#", " ").replace("_", " ")
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _PUNCT_RUN_RE.sub(r"\1", text)
    text = _KEEP_RE.sub(" ", text)
    tokens = [_canonical(tok) for tok in _SPACE_RE.split(text.strip()) if tok]
    return " ".join(tokens)


def normalize_many(texts: Iterable[object]) -> List[str]:
    return [normalize_text(t) for t in texts]
