"""VERITAS-AI tools."""
from veritas.tools.text_utils import (
    split_paragraphs,
    detect_headings,
    detect_references,
    clean_text,
    count_words,
)
from veritas.tools.ngram import (
    tokenize,
    ngrams,
    ngram_jaccard,
)
from veritas.tools.language_detect import (
    detect_language,
    language_name,
    is_same_language,
)

__all__ = [
    "split_paragraphs",
    "detect_headings",
    "detect_references",
    "clean_text",
    "count_words",
    "tokenize",
    "ngrams",
    "ngram_jaccard",
    "detect_language",
    "language_name",
    "is_same_language",
]