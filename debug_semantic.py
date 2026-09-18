"""Debug why semantic agent found nothing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

print("[1] Imports starting...")
from veritas.tools.embeddings import Embedder
from veritas.tools.text_utils import split_sentences
import numpy as np
print("[1] Imports OK")

submitted = (
    "Artificial intelligence has the ability to make the diagnosis "
    "of diseases more efficient and accurate."
)
source = (
    "AI can significantly improve medical diagnosis through "
    "advanced pattern recognition."
)

print("\n[2] Splitting sentences...")
sub_sents = [s for s in split_sentences(submitted) if len(s.split()) >= 6]
src_sents = [s for s in split_sentences(source) if len(s.split()) >= 6]
print(f"[2] submitted sentences: {len(sub_sents)}")
print(f"[2] source sentences:    {len(src_sents)}")

if not sub_sents or not src_sents:
    print("\n⚠️  One list is empty!")
    sys.exit(1)

print("\n[3] Loading embedder (may take 30-60s first time)...")
emb = Embedder.get(verbose=True)
print("[3] Embedder loaded")

print("\n[4] Embedding submitted...")
sub_vecs = emb.embed_en(sub_sents)
print(f"[4] submitted shape: {sub_vecs.shape}")

print("\n[5] Embedding source...")
src_vecs = emb.embed_en(src_sents)
print(f"[5] source shape: {src_vecs.shape}")

print("\n[6] Computing similarity...")
sim_matrix = np.dot(sub_vecs, src_vecs.T)
print(f"[6] matrix:\n{sim_matrix}")

best = float(sim_matrix.max())
print(f"\n[7] Best similarity: {best:.4f}")
print(f"[7] Current threshold: 0.75")
print(f"[7] Match: {'YES' if best >= 0.75 else 'NO'}")

# Also try a few alternative pairs
print("\n[8] Testing alternative sentence pairs...")
pairs = [
    (
        "Artificial intelligence improves medical diagnosis.",
        "AI enhances medical diagnosis significantly.",
    ),
    (
        "Machine learning can detect cancer from images.",
        "ML algorithms identify cancer in medical images.",
    ),
    (
        "The cat sat on the mat.",
        "A feline rested on the rug.",
    ),
    (
        "The weather is nice today.",
        "It is sunny outside right now.",
    ),
    (
        "Climate change threatens agriculture.",
        "Global warming endangers farming.",
    ),
]

for i, (a, b) in enumerate(pairs):
    va = emb.embed_en(a)
    vb = emb.embed_en(b)
    sim = float(np.dot(va, vb))
    print(f"  Pair {i+1}: {sim:.4f}  |  {a[:40]}...")

print("\n[DONE]")