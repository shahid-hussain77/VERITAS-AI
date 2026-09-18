"""
Paraphrase detection utilities.

Techniques:
- Synonym overlap (using WordNet-style approach with builtin dictionaries)
- Voice detection (active/passive)
- Sentence structure comparison
- Word order analysis
- Nominalization detection
- Synonym density scoring
"""
from __future__ import annotations

import re
from typing import Optional

from veritas.tools.ngram import tokenize


# ---------- Built-in synonym dictionary ----------
# Small, curated set. Extendable.
SYNONYM_GROUPS = [
    {"big", "large", "huge", "enormous", "massive", "gigantic"},
    {"small", "tiny", "little", "minute", "miniature"},
    {"fast", "quick", "rapid", "swift", "speedy"},
    {"slow", "sluggish", "gradual", "leisurely"},
    {"good", "excellent", "great", "fine", "superb", "wonderful"},
    {"bad", "poor", "awful", "terrible", "dreadful"},
    {"important", "significant", "crucial", "vital", "essential"},
    {"improve", "enhance", "better", "upgrade", "boost"},
    {"change", "alter", "modify", "transform", "adjust"},
    {"use", "utilize", "employ", "apply", "leverage"},
    {"make", "create", "produce", "generate", "build"},
    {"show", "demonstrate", "display", "reveal", "indicate"},
    {"help", "assist", "aid", "support", "facilitate"},
    {"begin", "start", "commence", "initiate", "launch"},
    {"end", "finish", "conclude", "complete", "terminate"},
    {"problem", "issue", "difficulty", "challenge", "obstacle"},
    {"answer", "response", "reply", "solution", "resolution"},
    {"method", "approach", "technique", "procedure", "way"},
    {"result", "outcome", "consequence", "effect", "conclusion"},
    {"study", "research", "investigation", "analysis", "examination"},
    {"data", "information", "facts", "figures", "statistics"},
    {"model", "framework", "structure", "system", "scheme"},
    {"cancer", "carcinoma", "malignancy", "tumor"},
    {"doctor", "physician", "medical practitioner", "clinician"},
    {"disease", "illness", "ailment", "condition", "disorder"},
    {"treatment", "therapy", "intervention", "care", "remedy"},
    {"diagnosis", "identification", "detection", "recognition"},
    {"patient", "sufferer", "client", "case"},
    {"medicine", "medication", "drug", "pharmaceutical"},
    {"health", "wellness", "wellbeing", "fitness"},
    {"climate", "weather", "atmosphere", "environment"},
    {"increase", "rise", "grow", "expand", "escalate"},
    {"decrease", "reduce", "decline", "diminish", "lessen"},
    {"predict", "forecast", "project", "estimate", "anticipate"},
    {"affect", "influence", "impact", "shape", "determine"},
    {"cause", "lead to", "result in", "trigger", "produce"},
    {"danger", "threat", "risk", "hazard", "peril"},
]

def _build_synonym_lookup() -> dict[str, set[str]]:
    """Build word → synonyms lookup."""
    lookup: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        for word in group:
            lookup.setdefault(word, set()).update(group - {word})
    return lookup


SYNONYM_LOOKUP = _build_synonym_lookup()


# ---------- Synonym density ----------
def synonym_density(text_a: str, text_b: str) -> float:
    """
    Estimate how many words in B are synonyms of words in A.

    Returns 0.0 to 1.0
    """
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))

    if not tokens_a or not tokens_b:
        return 0.0

    synonym_matches = 0
    for word in tokens_b:
        syns = SYNONYM_LOOKUP.get(word, set())
        if syns & tokens_a:
            synonym_matches += 1

    return synonym_matches / len(tokens_b)


# ---------- Voice detection ----------
PASSIVE_PATTERNS = [
    re.compile(r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b", re.I),
    re.compile(r"\b(?:is|are|was|were|be|been|being)\s+\w+en\b", re.I),
    re.compile(r"\b(?:is|are|was|were|be|been|being)\s+(?:written|done|"
               r"made|seen|taken|given|shown|known|found|used|"
               r"called|considered|regarded)\b", re.I),
]

ACTIVE_PATTERNS = [
    re.compile(r"\b(?:I|we|they|he|she|it|you)\s+\w+", re.I),
    re.compile(r"\b[A-Z][a-z]+\s+(?:conducted|found|showed|identified|"
               r"demonstrated|reported|investigated|analyzed|examined)\b",
               re.I),
]


def is_passive(sentence: str) -> bool:
    """Detect if sentence uses passive voice."""
    return any(p.search(sentence) for p in PASSIVE_PATTERNS)


def is_active(sentence: str) -> bool:
    """Detect if sentence uses active voice."""
    return any(p.search(sentence) for p in ACTIVE_PATTERNS)


def voice_changed(text_a: str, text_b: str) -> bool:
    """Check if voice changed between two texts."""
    a_passive = is_passive(text_a)
    b_passive = is_passive(text_b)
    a_active = is_active(text_a)
    b_active = is_active(text_b)

    # Passive ↔ Active
    if (a_passive and b_active) or (a_active and b_passive):
        return True
    return False


# ---------- Nominalization ----------
# verb → noun forms
NOMINALIZATION_PATTERNS = [
    (r"\bdecide", r"\bdecision"),
    (r"\bconclude", r"\bconclusion"),
    (r"\banalyze", r"\banalysis"),
    (r"\bimplement", r"\bimplementation"),
    (r"\bevaluate", r"\bevaluation"),
    (r"\bassess", r"\bassessment"),
    (r"\bapply", r"\bapplication"),
    (r"\bperform", r"\bperformance"),
    (r"\bexplain", r"\bexplanation"),
    (r"\bdevelop", r"\bdevelopment"),
    (r"\bcreate", r"\bcreation"),
    (r"\bdefine", r"\bdefinition"),
    (r"\bdescribe", r"\bdescription"),
    (r"\bmeasure", r"\bmeasurement"),
    (r"\bimprove", r"\bimprovement"),
    (r"\bmaintain", r"\bmaintenance"),
]


def nominalization_score(text_a: str, text_b: str) -> float:
    """
    Detect if B uses nominalized forms of A's verbs.

    Returns 0.0 to 1.0
    """
    a_lower = text_a.lower()
    b_lower = text_b.lower()

    matches = 0
    total = 0
    for verb_pat, noun_pat in NOMINALIZATION_PATTERNS:
        has_verb_a = bool(re.search(verb_pat, a_lower))
        has_noun_b = bool(re.search(noun_pat, b_lower))
        if has_verb_a or has_noun_b:
            total += 1
            if has_verb_a and has_noun_b:
                matches += 1

    if total == 0:
        return 0.0
    return matches / total


# ---------- Word order ----------
def word_order_change(text_a: str, text_b: str) -> float:
    """
    Measure how much word order changed between similar texts.

    Returns 0.0 (same order) to 1.0 (completely different order)
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # Common vocabulary
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    common = set_a & set_b

    if not common:
        return 0.0

    # Position sequence in A and B
    positions_a = [i for i, t in enumerate(tokens_a) if t in common]
    positions_b = [i for i, t in enumerate(tokens_b) if t in common]

    # Rank correlation (Spearman-like)
    if len(positions_a) != len(positions_b):
        # Padded comparison
        min_len = min(len(positions_a), len(positions_b))
        positions_a = positions_a[:min_len]
        positions_b = positions_b[:min_len]

    if len(positions_a) < 2:
        return 0.0

    # Normalize positions
    max_a = max(positions_a) or 1
    max_b = max(positions_b) or 1
    norm_a = [p / max_a for p in positions_a]
    norm_b = [p / max_b for p in positions_b]

    # Average absolute difference
    diffs = [abs(a - b) for a, b in zip(norm_a, norm_b)]
    return sum(diffs) / len(diffs)


# ---------- Sentence structure ----------
def structure_similarity(text_a: str, text_b: str) -> float:
    """
    Compare POS-like structure (approximated with word categories).

    Returns 0.0 to 1.0
    """
    def signature(text: str) -> list[str]:
        """Simple word category signature."""
        tokens = tokenize(text)
        sig = []
        for t in tokens[:30]:  # first 30 words
            if t in {"the", "a", "an", "this", "that", "these", "those"}:
                sig.append("DET")
            elif t in {"and", "or", "but", "yet", "so", "for"}:
                sig.append("CONJ")
            elif t in {"in", "on", "at", "by", "with", "from", "to", "for"}:
                sig.append("PREP")
            elif t in {"is", "are", "was", "were", "be", "been"}:
                sig.append("BE")
            elif t.endswith("ly"):
                sig.append("ADV")
            elif t.endswith("ing") or t.endswith("ed"):
                sig.append("VERB")
            elif t.endswith("tion") or t.endswith("ment") or t.endswith("ness"):
                sig.append("NOUN")
            elif len(t) > 3:
                sig.append("WORD")
            else:
                sig.append("SHORT")
        return sig

    sig_a = signature(text_a)
    sig_b = signature(text_b)

    if not sig_a or not sig_b:
        return 0.0

    # Levenshtein-like on sequences
    n, m = len(sig_a), len(sig_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if sig_a[i - 1] == sig_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    distance = dp[n][m]
    max_len = max(n, m)
    return 1.0 - distance / max_len


# ---------- Combined paraphrase score ----------
def paraphrase_indicators(text_a: str, text_b: str) -> dict:
    """
    Compute all paraphrase indicators.

    Returns dict with individual signals + overall score.
    """
    syn = synonym_density(text_a, text_b)
    voice = voice_changed(text_a, text_b)
    nom = nominalization_score(text_a, text_b)
    order = word_order_change(text_a, text_b)
    struct = structure_similarity(text_a, text_b)

    # Weighted aggregate
    score = (
        0.35 * syn
        + 0.20 * (1.0 if voice else 0.0)
        + 0.15 * nom
        + 0.15 * struct
        + 0.15 * order
    )

    return {
        "synonym_density": round(syn, 4),
        "voice_changed": voice,
        "nominalization": round(nom, 4),
        "word_order_change": round(order, 4),
        "structure_similarity": round(struct, 4),
        "overall": round(score, 4),
    }


__all__ = [
    "synonym_density",
    "is_passive",
    "is_active",
    "voice_changed",
    "nominalization_score",
    "word_order_change",
    "structure_similarity",
    "paraphrase_indicators",
]