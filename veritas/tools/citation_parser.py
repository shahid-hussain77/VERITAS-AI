"""
Citation parser — extract and analyze inline citations and references.

Supports:
- Numeric: [1], [2], [1,2], [1-5]
- Author-year: (Smith, 2020), (Smith et al., 2020)
- Mixed

Detects issues:
- Citations without references
- References without citations
- Duplicate citation numbers
- Excessive citation density
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------- Data classes ----------
@dataclass
class InlineCitation:
    """An inline citation found in text."""
    raw: str
    type: str                # "numeric" | "author_year"
    numbers: list[int]       # [1, 2] for numeric
    authors: list[str]       # ["Smith"] for author-year
    year: Optional[str]      # "2020"
    position: int            # character position
    context: str = ""        # surrounding text


@dataclass
class Reference:
    """A reference entry from the references section."""
    raw: str
    index: Optional[int]     # parsed [n] if numeric
    authors: list[str] = field(default_factory=list)
    year: Optional[str] = None
    title: str = ""
    journal: str = ""
    doi: Optional[str] = None
    position: int = 0


# ---------- Regex patterns ----------
# Numeric citation: [1], [1,2], [1-5], [1, 2, 3]
NUMERIC_CITATION = re.compile(
    r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]"
)

# Author-year: (Smith, 2020), (Smith et al., 2020), (Smith & Doe, 2020)
AUTHOR_YEAR_CITATION = re.compile(
    r"\(([A-Z][a-zA-Z\-]+(?:\s+(?:and|&|et\s+al\.?))?"
    r"(?:\s+[A-Z][a-zA-Z\-]+)?(?:\s*,\s*[A-Z][a-zA-Z\-]+)*)"
    r",?\s+(\d{4}[a-z]?)\)"
)

# DOI pattern
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")

# Reference entry: starts with [n], n., or Author (Year)
REF_ENTRY_START = re.compile(
    r"^\s*(?:\[(\d+)\]|(\d+)\.|([A-Z][a-zA-Z\-]+,\s+[A-Z]))"
)


# ---------- Inline citation extraction ----------
def parse_numeric_citation(raw: str) -> list[int]:
    """Parse '[1,2,5-7]' → [1,2,5,6,7]."""
    inner = raw.strip("[]").strip()
    numbers = []
    for part in re.split(r"\s*,\s*", inner):
        if not part:
            continue
        if "-" in part or "–" in part:
            bounds = re.split(r"[-–]", part)
            if len(bounds) == 2:
                try:
                    start = int(bounds[0].strip())
                    end = int(bounds[1].strip())
                    numbers.extend(range(start, end + 1))
                except ValueError:
                    continue
        else:
            try:
                numbers.append(int(part.strip()))
            except ValueError:
                continue
    return numbers


def extract_citations(text: str, window: int = 100) -> list[InlineCitation]:
    """
    Extract all inline citations from text.

    Args:
        text: source text
        window: characters on each side to capture context
    """
    citations: list[InlineCitation] = []
    seen_positions: set[int] = set()

    # Numeric
    for m in NUMERIC_CITATION.finditer(text):
        pos = m.start()
        if pos in seen_positions:
            continue
        seen_positions.add(pos)

        raw = m.group(0)
        numbers = parse_numeric_citation(raw)

        context_start = max(0, pos - window)
        context_end = min(len(text), pos + len(raw) + window)
        context = text[context_start:context_end].replace("\n", " ")

        citations.append(InlineCitation(
            raw=raw,
            type="numeric",
            numbers=numbers,
            authors=[],
            year=None,
            position=pos,
            context=context,
        ))

    # Author-year
    for m in AUTHOR_YEAR_CITATION.finditer(text):
        pos = m.start()
        if pos in seen_positions:
            continue
        seen_positions.add(pos)

        raw = m.group(0)
        authors_str = m.group(1)
        year = m.group(2)

        authors = re.split(r"\s+(?:and|&|et\s+al\.?)\s+", authors_str)
        authors = [a.strip() for a in authors if a.strip()]

        context_start = max(0, pos - window)
        context_end = min(len(text), pos + len(raw) + window)
        context = text[context_start:context_end].replace("\n", " ")

        citations.append(InlineCitation(
            raw=raw,
            type="author_year",
            numbers=[],
            authors=authors,
            year=year,
            position=pos,
            context=context,
        ))

    citations.sort(key=lambda c: c.position)
    return citations


# ---------- Reference parsing ----------
def parse_reference_entry(raw: str) -> Reference:
    """Parse a single reference line."""
    # Try to extract [n]
    m = re.match(r"^\s*\[(\d+)\]", raw)
    idx = int(m.group(1)) if m else None

    # Authors (before year)
    year_match = re.search(r"\((\d{4}[a-z]?)\)", raw)
    if not year_match:
        year_match = re.search(r"\b(\d{4})\b", raw)
    year = year_match.group(1) if year_match else None

    # Authors: text before year
    authors = []
    if year_match:
        authors_part = raw[:year_match.start()]
        # Remove [n]
        authors_part = re.sub(r"^\s*\[\d+\]\s*", "", authors_part)
        authors_part = re.sub(r"^\s*\d+\.\s*", "", authors_part)
        authors = [
            a.strip().rstrip(".,")
            for a in re.split(r",|&|\band\b", authors_part)
            if a.strip() and len(a.strip()) < 60
        ]

    # DOI
    doi_match = DOI_PATTERN.search(raw)
    doi = doi_match.group(0) if doi_match else None

    return Reference(
        raw=raw,
        index=idx,
        authors=authors[:5],
        year=year,
        doi=doi,
    )


def extract_references(ref_lines: list[str]) -> list[Reference]:
    """Parse reference lines into Reference objects."""
    return [parse_reference_entry(line) for line in ref_lines if line.strip()]


# ---------- Analysis ----------
def analyze_citations(
    text: str,
    ref_lines: list[str],
) -> dict:
    """
    Analyze citation integrity.

    Returns dict with:
        citations: list[InlineCitation]
        references: list[Reference]
        missing_refs: list[int]         citations with no reference
        unused_refs: list[int]          references never cited
        duplicate_citations: dict       citation → count
        numeric_count: int
        author_year_count: int
        coverage: float                 0-1
        issues: list[dict]
    """
    citations = extract_citations(text)
    references = extract_references(ref_lines)

    # Numeric citation analysis
    numeric_citations = [c for c in citations if c.type == "numeric"]
    all_cited_numbers: set[int] = set()
    citation_counter: dict[int, int] = {}

    for c in numeric_citations:
        for n in c.numbers:
            all_cited_numbers.add(n)
            citation_counter[n] = citation_counter.get(n, 0) + 1

    # Reference numbers
    ref_numbers = {r.index for r in references if r.index is not None}

    # Missing references
    missing_refs = sorted(all_cited_numbers - ref_numbers)

    # Unused references
    unused_refs = sorted(ref_numbers - all_cited_numbers)

    # Duplicate citations (cited 3+ times)
    duplicate_citations = {
        n: count for n, count in citation_counter.items() if count >= 3
    }

    # Coverage
    if ref_numbers:
        coverage = len(all_cited_numbers & ref_numbers) / len(ref_numbers)
    else:
        coverage = 0.0

    # Build issues
    issues = []

    for n in missing_refs:
        issues.append({
            "type": "missing_reference",
            "severity": "high",
            "message": f"Citation [{n}] used in text but no Reference [{n}] exists",
            "citation_number": n,
        })

    for n in unused_refs:
        issues.append({
            "type": "unused_reference",
            "severity": "low",
            "message": f"Reference [{n}] listed but never cited in text",
            "citation_number": n,
        })

    for n, count in duplicate_citations.items():
        issues.append({
            "type": "excessive_citation",
            "severity": "info",
            "message": f"Citation [{n}] used {count} times",
            "citation_number": n,
            "count": count,
        })

    return {
        "citations": citations,
        "references": references,
        "missing_refs": missing_refs,
        "unused_refs": unused_refs,
        "duplicate_citations": duplicate_citations,
        "numeric_count": len(numeric_citations),
        "author_year_count": len(citations) - len(numeric_citations),
        "total_citations": len(citations),
        "total_references": len(references),
        "unique_cited_numbers": sorted(all_cited_numbers),
        "reference_numbers": sorted(ref_numbers),
        "coverage": round(coverage, 4),
        "issues": issues,
    }


# ---------- Finding unsupported claims ----------
CLAIM_INDICATORS = re.compile(
    r"\b(?:"
    r"studies\s+show|research\s+shows|"
    r"it\s+has\s+been\s+(?:shown|proven|demonstrated|found)|"
    r"according\s+to|researchers?\s+(?:found|reported|discovered)|"
    r"evidence\s+suggests|data\s+(?:shows|indicates)|"
    r"reported\s+that|found\s+that|"
    r"\d+(?:\.\d+)?%"
    r")\b",
    re.IGNORECASE,
)


def find_uncited_claims(text: str, window: int = 200) -> list[dict]:
    """
    Find sentences that look like factual claims but have no citation nearby.

    Returns list of dicts with: sentence, position, matched_pattern
    """
    from veritas.tools.text_utils import split_sentences

    sentences = split_sentences(text)
    uncited = []

    for sent in sentences:
        if len(sent.split()) < 6:
            continue

        match = CLAIM_INDICATORS.search(sent)
        if not match:
            continue

        # Check if citation exists in sentence or nearby
        has_numeric = bool(NUMERIC_CITATION.search(sent))
        has_author_year = bool(AUTHOR_YEAR_CITATION.search(sent))

        if not (has_numeric or has_author_year):
            uncited.append({
                "sentence": sent[:200],
                "matched": match.group(0),
                "reason": "claim-like statement without citation",
            })

    return uncited


__all__ = [
    "InlineCitation", "Reference",
    "extract_citations", "extract_references",
    "analyze_citations", "find_uncited_claims",
]