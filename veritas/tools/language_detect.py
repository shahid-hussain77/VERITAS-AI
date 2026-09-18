"""
Language detection — lightweight, offline.

Uses Unicode script ranges + common word patterns.
No API calls. No downloads.
"""
from __future__ import annotations

import re
from collections import Counter


# Unicode script ranges
SCRIPT_RANGES = {
    "latin":      (0x0041, 0x007A),
    "cyrillic":   (0x0400, 0x04FF),
    "arabic":     (0x0600, 0x06FF),
    "hebrew":     (0x0590, 0x05FF),
    "devanagari": (0x0900, 0x097F),
    "cjk":        (0x4E00, 0x9FFF),
    "hiragana":   (0x3040, 0x309F),
    "katakana":   (0x30A0, 0x30FF),
    "hangul":     (0xAC00, 0xD7AF),
    "thai":       (0x0E00, 0x0E7F),
    "bengali":    (0x0980, 0x09FF),
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
    "gujarati":   (0x0A80, 0x0AFF),
    "gurmukhi":   (0x0A00, 0x0A7F),
}


def _script_of(char: str) -> str:
    """Return script name for a character."""
    cp = ord(char)
    for script, (lo, hi) in SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return script
    return "other"


def detect_script(text: str) -> str:
    """Return dominant script in text."""
    if not text:
        return "unknown"

    counts: Counter = Counter()
    for ch in text:
        if ch.isalpha():
            counts[_script_of(ch)] += 1

    if not counts:
        return "unknown"

    return counts.most_common(1)[0][0]


def detect_language(text: str) -> str:
    """
    Detect language code from text.

    Returns ISO-like code:
        en, ur, hi, ar, zh, ja, ko, ru, he, th, bn, ta, te, gu, pa
    """
    script = detect_script(text)

    script_to_lang = {
        "latin": "en",           # default for Latin
        "arabic": "ur",          # Urdu default (could be ar, fa)
        "devanagari": "hi",
        "cjk": "zh",
        "hiragana": "ja",
        "katakana": "ja",
        "hangul": "ko",
        "cyrillic": "ru",
        "hebrew": "he",
        "thai": "th",
        "bengali": "bn",
        "tamil": "ta",
        "telugu": "te",
        "gujarati": "gu",
        "gurmukhi": "pa",
    }

    lang = script_to_lang.get(script, "unknown")

    # Refine Arabic → Urdu vs Arabic vs Persian
    if script == "arabic":
        lang = _refine_arabic(text)
    # Refine Latin → English vs others (basic)
    elif script == "latin":
        lang = _refine_latin(text)

    return lang


def _refine_arabic(text: str) -> str:
    """Distinguish Urdu, Arabic, Persian by common words."""
    urdu_markers = {
        "ہے", "کی", "کا", "کو", "میں", "اور", "سے", "نے", "پر", "یہ",
        "کیا", "نہیں", "ہیں", "تھا", "تھی", "کے", "کی", "ایک", "بھی",
    }
    arabic_markers = {
        "في", "من", "على", "إلى", "هذا", "هذه", "التي", "الذي", "كان",
        "هو", "هي", "أن", "لا", "ما", "مع", "عن", "كل",
    }
    persian_markers = {
        "است", "که", "را", "این", "آن", "برای", "با", "از", "می", "شود",
    }

    tokens = set(re.findall(r"\S+", text))

    urdu = len(tokens & urdu_markers)
    arabic = len(tokens & arabic_markers)
    persian = len(tokens & persian_markers)

    if urdu >= arabic and urdu >= persian and urdu > 0:
        return "ur"
    if arabic >= persian and arabic > 0:
        return "ar"
    if persian > 0:
        return "fa"
    # Default to Urdu for our use case
    return "ur"


def _refine_latin(text: str) -> str:
    """Basic Latin-script language detection."""
    lower = " " + text.lower() + " "

    # Common stopwords per language
    markers = {
        "en": [" the ", " is ", " and ", " of ", " to ", " in ", " a "],
        "fr": [" le ", " la ", " les ", " et ", " est ", " un ", " une "],
        "es": [" el ", " la ", " los ", " y ", " es ", " un ", " una "],
        "de": [" der ", " die ", " das ", " und ", " ist ", " ein "],
        "it": [" il ", " la ", " e ", " è ", " un ", " una ", " di "],
        "pt": [" o ", " a ", " os ", " e ", " é ", " um ", " uma "],
    }

    scores = {}
    for lang, words in markers.items():
        scores[lang] = sum(lower.count(w) for w in words)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "en"


def is_same_language(text_a: str, text_b: str) -> bool:
    """Check if two texts are in the same language."""
    return detect_language(text_a) == detect_language(text_b)


def language_name(code: str) -> str:
    """Human-readable language name."""
    names = {
        "en": "English",
        "ur": "Urdu",
        "hi": "Hindi",
        "ar": "Arabic",
        "fa": "Persian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "he": "Hebrew",
        "th": "Thai",
        "bn": "Bengali",
        "ta": "Tamil",
        "te": "Telugu",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
    }
    return names.get(code, code)


__all__ = [
    "detect_script",
    "detect_language",
    "is_same_language",
    "language_name",
]