"""
Text utilities — paragraph splitting, heading detection,
reference detection, text cleaning.
"""
from __future__ import annotations

import re
from typing import Optional


# ---------- Cleaning ----------
def clean_text(text: str) -> str:
    """Normalize whitespace and remove control chars."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove non-printable control chars (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(re.findall(r"\b\w+\b", text or ""))


# ---------- Paragraph splitting ----------
def split_paragraphs(text: str, min_chars: int = 40) -> list[str]:
    """
    Split text into paragraphs.

    Heuristic:
    - Split on 2+ newlines
    - Also split on single newlines if lines look like paragraphs
    - Filter out very short fragments
    """
    if not text:
        return []

    text = clean_text(text)

    # First try double-newline split
    raw = re.split(r"\n\s*\n", text)

    # If we got very few paragraphs, try single-newline split
    if len(raw) < 3:
        raw = re.split(r"\n", text)

    paragraphs = []
    buffer = ""

    for chunk in raw:
        chunk = chunk.strip()
        if not chunk:
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            continue
        # If chunk is short, merge with next
        if len(chunk) < min_chars and buffer:
            buffer += " " + chunk
        elif len(chunk) < min_chars:
            buffer = chunk
        else:
            if buffer:
                paragraphs.append(buffer)
                buffer = ""
            paragraphs.append(chunk)

    if buffer:
        paragraphs.append(buffer)

    # Filter
    return [p for p in paragraphs if len(p) >= min_chars]


# ---------- Heading detection ----------
HEADING_PATTERNS = [
    re.compile(r"^(?:chapter|section|part)\s+\d+", re.IGNORECASE),
    re.compile(r"^\d+\.\s+[A-Z]"),                    # "1. Introduction"
    re.compile(r"^\d+\.\d+\s+[A-Z]"),                 # "1.1 Background"
    re.compile(r"^[IVX]+\.\s+[A-Z]"),                 # "I. Introduction"
    re.compile(r"^(?:abstract|introduction|methodology|"
               r"results|discussion|conclusion|references|"
               r"bibliography|acknowledgements)\s*$",
               re.IGNORECASE),
]


def is_heading(text: str, max_words: int = 12) -> bool:
    """Heuristic: is this paragraph a heading?"""
    if not text:
        return False
    text = text.strip()
    words = text.split()
    if len(words) > max_words:
        return False
    if len(text) > 100:
        return False
    # Ends with period? Likely not a heading
    if text.endswith(".") and not text.endswith("..."):
        # Except numbered headings like "1. Introduction."
        if not re.match(r"^\d+\.\s", text):
            return False

    for pat in HEADING_PATTERNS:
        if pat.match(text):
            return True

    # All caps short line
    if text.isupper() and 2 <= len(words) <= 10:
        return True

    # Title Case short line (most words capitalized)
    if 2 <= len(words) <= 8:
        cap_words = sum(1 for w in words if w and w[0].isupper())
        if cap_words / len(words) >= 0.8 and not text.endswith("."):
            return True

    return False


def detect_headings(paragraphs: list[str]) -> list[int]:
    """Return indices of paragraphs that look like headings."""
    return [i for i, p in enumerate(paragraphs) if is_heading(p)]


# ---------- Reference detection ----------
REFERENCE_HEADER = re.compile(
    r"^\s*(references|bibliography|works\s+cited|"
    r"literature\s+cited|sources)\s*$",
    re.IGNORECASE,
)

# Common citation patterns
CITATION_PATTERNS = [
    # [1], [2], [12]
    re.compile(r"\[\d+(?:[,\-\s]+\d+)*\]"),
    # (Author, 2020), (Author et al., 2020)
    re.compile(r"\([A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?\)"),
    # Author (2020)
    re.compile(r"[A-Z][a-zA-Z\-]+\s+\(\d{4}[a-z]?\)"),
    # DOI
    re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b"),
]


def detect_references(text: str) -> tuple[list[str], str]:
    """
    Detect the references section and extract reference lines.

    Returns:
        (reference_lines, text_without_references)
    """
    if not text:
        return [], ""

    lines = text.split("\n")
    ref_start = None

    for i, line in enumerate(lines):
        if REFERENCE_HEADER.match(line.strip()):
            ref_start = i
            break

    if ref_start is None:
        return [], text

    # Everything after the header is references (until end)
    ref_lines = lines[ref_start + 1:]
    body_lines = lines[:ref_start]

    # Merge multi-line references: a new reference starts when a line
    # begins with [n], a number, or a capital letter + period/comma
    merged = []
    current = ""
    ref_start_pattern = re.compile(
        r"^\s*(?:\[\d+\]|\d+\.|[A-Z][a-zA-Z\-]+,\s+[A-Z]\.)"
    )

    for line in ref_lines:
        line = line.strip()
        if not line:
            if current:
                merged.append(current)
                current = ""
            continue

        if ref_start_pattern.match(line) and current:
            merged.append(current)
            current = line
        else:
            current = (current + " " + line).strip() if current else line

    if current:
        merged.append(current)

    # Filter out very short lines
    references = [r for r in merged if len(r) > 20]

    return references, "\n".join(body_lines).strip()


def extract_inline_citations(text: str) -> list[dict]:
    """
    Extract inline citations from text.

    Returns list of dicts with:
        - raw: the matched string
        - type: "numeric" | "author_year" | "doi"
        - position: character position
    """
    if not text:
        return []

    results = []
    for i, pat in enumerate(CITATION_PATTERNS):
        for m in pat.finditer(text):
            if i == 0:
                ctype = "numeric"
            elif i == 3:
                ctype = "doi"
            else:
                ctype = "author_year"
            results.append({
                "raw": m.group(0),
                "type": ctype,
                "position": m.start(),
            })

    # Sort by position
    results.sort(key=lambda x: x["position"])
    return results


def split_sentences(text: str) -> list[str]:
    """Basic sentence splitter."""
    if not text:
        return []
    # Protect common abbreviations
    protected = text
    for abbr in ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "etc.",
                 "i.e.", "e.g.", "vs.", "Fig.", "Eq.", "No.",
                 "U.S.", "U.K.", "al."]:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


__all__ = [
    "clean_text",
    "count_words",
    "split_paragraphs",
    "is_heading",
    "detect_headings",
    "detect_references",
    "extract_inline_citations",
    "split_sentences",
]