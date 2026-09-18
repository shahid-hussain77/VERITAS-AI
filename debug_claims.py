import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from veritas.tools.text_utils import split_sentences
from veritas.tools.citation_parser import (
    CLAIM_INDICATORS, NUMERIC_CITATION, AUTHOR_YEAR_CITATION,
)

text = (
    "Studies show that 75% of patients recover quickly. "
    "According to recent research, this is very common. "
    "Smith (2020) reported 94% success rates."
)

print("Original text:")
print(text)
print()

sentences = split_sentences(text)
print(f"Sentences ({len(sentences)}):")
for i, s in enumerate(sentences):
    print(f"  [{i}] {s}")
print()

for i, sent in enumerate(sentences):
    words = len(sent.split())
    match = CLAIM_INDICATORS.search(sent)
    print(f"Sentence {i}: words={words}")
    print(f"  CLAIM_INDICATORS match: {match.group(0) if match else 'None'}")
    print(f"  Has numeric citation: {bool(NUMERIC_CITATION.search(sent))}")
    print(f"  Has author-year: {bool(AUTHOR_YEAR_CITATION.search(sent))}")