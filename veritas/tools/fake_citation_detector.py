"""
Fake citation detector.

Detects potentially fabricated references using:
- DOI format validation
- Journal name plausibility
- Author name heuristics
- Publication year sanity
- Structural red flags

Offline by default. Optional online verification via Crossref.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------- Known publishers (top 100-ish) ----------
KNOWN_PUBLISHERS = {
    "elsevier", "springer", "wiley", "taylor", "francis",
    "sage", "oxford", "cambridge", "ieee", "acm",
    "nature", "science", "plos", "frontiers", "mdpi",
    "emerald", "lippincott", "bmj", "jama", "lancet",
    "cell", "american", "royal", "iospress",
}


# ---------- Known journals (partial list) ----------
KNOWN_JOURNALS = {
    "nature", "science", "cell", "lancet", "jama", "bmj",
    "plos one", "scientific reports",
    "journal of machine learning research",
    "ieee transactions on pattern analysis and machine intelligence",
    "acm transactions on graphics",
    "artificial intelligence",
    "neural networks",
    "pattern recognition",
    "computer vision and image understanding",
    "journal of artificial intelligence research",
    "machine learning",
    "data mining and knowledge discovery",
}


# ---------- Suspicious journal keywords ----------
SUSPICIOUS_JOURNAL_WORDS = {
    "advanced", "innovative", "cutting-edge", "emerging",
    "breakthrough", "revolutionary", "next-generation",
    "modern", "contemporary", "international journal of",
    "global journal of", "world journal of",
    "american journal of", "european journal of",
}


# ---------- DOI format ----------
DOI_REGEX = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")


@dataclass
class FakeCitationSignal:
    """A single red flag or green flag."""
    name: str
    status: str          # "ok" | "warn" | "fail" | "info"
    score: float         # contribution to fake probability (0-1)
    detail: str = ""


@dataclass
class FakeCitationAnalysis:
    """Result of analyzing one reference."""
    reference_raw: str
    reference_index: Optional[int] = None

    signals: list[FakeCitationSignal] = field(default_factory=list)
    fake_probability: float = 0.0
    risk_level: str = "unknown"    # low | medium | high | critical

    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[str] = None
    journal: str = ""
    doi: Optional[str] = None

    def add_signal(self, name: str, status: str, score: float, detail: str = ""):
        self.signals.append(FakeCitationSignal(name, status, score, detail))

    def to_dict(self) -> dict:
        return {
            "reference_index": self.reference_index,
            "reference_preview": self.reference_raw[:200],
            "fake_probability": round(self.fake_probability, 4),
            "risk_level": self.risk_level,
            "signals": [
                {
                    "name": s.name,
                    "status": s.status,
                    "score": round(s.score, 3),
                    "detail": s.detail,
                }
                for s in self.signals
            ],
            "parsed": {
                "title": self.title,
                "authors": self.authors,
                "year": self.year,
                "journal": self.journal,
                "doi": self.doi,
            },
        }


# ---------- Parsing helpers ----------
def _parse_reference_fields(raw: str) -> dict:
    """Extract structured fields from a reference string."""
    fields = {
        "index": None,
        "title": "",
        "authors": [],
        "year": None,
        "journal": "",
        "doi": None,
    }

    # Index
    m = re.match(r"^\s*\[(\d+)\]", raw)
    if m:
        fields["index"] = int(m.group(1))

    # DOI
    doi_match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", raw)
    if doi_match:
        fields["doi"] = doi_match.group(1)

    # Year
    year_match = re.search(r"\((\d{4}[a-z]?)\)", raw)
    if not year_match:
        year_match = re.search(r"\b((?:19|20)\d{2})\b", raw)
    if year_match:
        fields["year"] = year_match.group(1)

    # Authors: text before year
    if year_match:
        authors_part = raw[:year_match.start()]
        authors_part = re.sub(r"^\s*\[\d+\]\s*", "", authors_part)
        authors_part = re.sub(r"^\s*\d+\.\s*", "", authors_part)
        authors = [
            a.strip().rstrip(".,&")
            for a in re.split(r",|&|\band\b", authors_part)
            if a.strip() and 2 < len(a.strip()) < 50
        ]
        fields["authors"] = authors[:6]

    # Title: after year, before journal
    if year_match:
        after_year = raw[year_match.end():]
        # Remove leading ). or .
        after_year = re.sub(r"^[\)\.\s]+", "", after_year)
        # Title typically up to first period or comma
        title_match = re.match(r"([^.\n]{10,200})", after_year)
        if title_match:
            fields["title"] = title_match.group(1).strip()[:200]

    # Journal: look for capitalized multi-word phrases after title
    journal_match = re.search(
        r"(Journal of [A-Z][a-zA-Z\s]{2,60}|"
        r"[A-Z][a-zA-Z]+ (?:Journal|Review|Reviews|Letters|Transactions|"
        r"Proceedings|Reports|Studies|Research))",
        raw,
    )
    if journal_match:
        fields["journal"] = journal_match.group(1).strip()

    return fields


# ---------- Individual checks ----------
def _check_doi_format(doi: Optional[str]) -> FakeCitationSignal:
    """Check DOI format."""
    if not doi:
        return FakeCitationSignal(
            "doi_presence", "warn", 0.25,
            "No DOI found (modern references usually have DOI)",
        )
    if DOI_REGEX.match(doi):
        return FakeCitationSignal(
            "doi_format", "ok", 0.0,
            f"Valid DOI format: {doi}",
        )
    return FakeCitationSignal(
        "doi_format", "fail", 0.5,
        f"Invalid DOI format: {doi}",
    )


def _check_journal_known(journal: str) -> FakeCitationSignal:
    """Check if journal is in known list."""
    if not journal:
        return FakeCitationSignal(
            "journal_presence", "warn", 0.15,
            "No recognizable journal name",
        )

    journal_lower = journal.lower().strip()

    if any(known in journal_lower for known in KNOWN_JOURNALS):
        return FakeCitationSignal(
            "journal_known", "ok", 0.0,
            f"Journal recognized: {journal}",
        )

    # Check publisher
    for pub in KNOWN_PUBLISHERS:
        if pub in journal_lower:
            return FakeCitationSignal(
                "journal_publisher", "ok", 0.05,
                f"Journal associated with known publisher: {pub}",
            )

    # Suspicious keywords
    suspicious_hits = [
        w for w in SUSPICIOUS_JOURNAL_WORDS
        if w in journal_lower
    ]
    if suspicious_hits:
        return FakeCitationSignal(
            "journal_suspicious", "warn", 0.30,
            f"Journal name contains template-like keywords: {suspicious_hits}",
        )

    return FakeCitationSignal(
        "journal_unknown", "warn", 0.15,
        f"Journal not in known list: {journal}",
    )


def _check_year_plausibility(year: Optional[str]) -> FakeCitationSignal:
    """Year should be plausible."""
    if not year:
        return FakeCitationSignal(
            "year_presence", "warn", 0.15,
            "No publication year found",
        )

    try:
        y = int(year[:4])
    except ValueError:
        return FakeCitationSignal(
            "year_format", "fail", 0.3,
            f"Invalid year: {year}",
        )

    from datetime import datetime
    current = datetime.now().year

    if y < 1900:
        return FakeCitationSignal(
            "year_too_old", "fail", 0.4,
            f"Year {y} is implausibly old",
        )
    if y > current + 1:
        return FakeCitationSignal(
            "year_future", "fail", 0.5,
            f"Year {y} is in the future",
        )
    if y >= current - 1:
        return FakeCitationSignal(
            "year_recent", "info", 0.05,
            f"Very recent (year {y})",
        )
    return FakeCitationSignal(
        "year_ok", "ok", 0.0,
        f"Year plausible: {y}",
    )


def _check_authors(authors: list[str]) -> FakeCitationSignal:
    """Check author list plausibility."""
    if not authors:
        return FakeCitationSignal(
            "authors_presence", "warn", 0.20,
            "No authors detected",
        )

    # Check for very generic names
    generic_first = {"j", "john", "james", "smith", "doe", "johnson",
                     "williams", "brown", "jones", "garcia", "miller",
                     "davis", "rodriguez", "martinez", "hernandez",
                     "lopez", "gonzalez", "wilson", "anderson", "thomas"}

    generic_count = 0
    for author in authors:
        parts = author.lower().split()
        if any(p in generic_first for p in parts):
            generic_count += 1

    if len(authors) >= 2 and generic_count == len(authors):
        return FakeCitationSignal(
            "authors_generic", "warn", 0.25,
            f"All authors have generic/common names ({authors})",
        )

    if len(authors) > 8:
        return FakeCitationSignal(
            "authors_too_many", "warn", 0.10,
            f"Unusually many authors: {len(authors)}",
        )

    return FakeCitationSignal(
        "authors_ok", "ok", 0.0,
        f"{len(authors)} author(s) look plausible",
    )


def _check_title(title: str) -> FakeCitationSignal:
    """Check title plausibility."""
    if not title:
        return FakeCitationSignal(
            "title_presence", "warn", 0.10,
            "No title extracted",
        )

    if len(title) < 15:
        return FakeCitationSignal(
            "title_short", "warn", 0.15,
            f"Title very short: {title}",
        )

    if len(title) > 300:
        return FakeCitationSignal(
            "title_long", "warn", 0.05,
            "Title unusually long",
        )

    return FakeCitationSignal(
        "title_ok", "ok", 0.0,
        "Title length plausible",
    )


def _check_template_pattern(raw: str) -> FakeCitationSignal:
    """Detect AI-template-like patterns."""
    # AI often uses these patterns
    template_markers = [
        r"\bJournal of (?:Advanced|Modern|Emerging|Innovative|"
        r"Cutting-Edge|Next-Generation)\s+\w+\s+(?:Research|Studies|Science|Technology)\b",
        r"\bInternational Journal of (?:Advanced|Modern|Emerging)\b",
    ]

    for pat in template_markers:
        if re.search(pat, raw, re.IGNORECASE):
            return FakeCitationSignal(
                "template_pattern", "fail", 0.4,
                "Reference matches common AI-generated template pattern",
            )

    return FakeCitationSignal(
        "template_ok", "ok", 0.0,
        "No obvious template pattern",
    )


# ---------- Main analysis ----------
def analyze_reference(raw: str) -> FakeCitationAnalysis:
    """Analyze a single reference for fake indicators."""
    analysis = FakeCitationAnalysis(reference_raw=raw)

    fields = _parse_reference_fields(raw)
    analysis.reference_index = fields["index"]
    analysis.title = fields["title"]
    analysis.authors = fields["authors"]
    analysis.year = fields["year"]
    analysis.journal = fields["journal"]
    analysis.doi = fields["doi"]

    # Collect signals
    analysis.signals = [
        _check_doi_format(fields["doi"]),
        _check_journal_known(fields["journal"]),
        _check_year_plausibility(fields["year"]),
        _check_authors(fields["authors"]),
        _check_title(fields["title"]),
        _check_template_pattern(raw),
    ]

    # Aggregate probability (weighted average, capped)
    total_score = sum(s.score for s in analysis.signals)
    # Normalize: max possible ~2.0, want 0-1
    analysis.fake_probability = min(1.0, total_score / 1.5)

    # Risk level
    p = analysis.fake_probability
    if p >= 0.65:
        analysis.risk_level = "critical"
    elif p >= 0.45:
        analysis.risk_level = "high"
    elif p >= 0.25:
        analysis.risk_level = "medium"
    else:
        analysis.risk_level = "low"

    return analysis


def analyze_references(ref_lines: list[str]) -> list[FakeCitationAnalysis]:
    """Analyze multiple references."""
    return [analyze_reference(line) for line in ref_lines if line.strip()]


__all__ = [
    "FakeCitationAnalysis", "FakeCitationSignal",
    "analyze_reference", "analyze_references",
]