"""
PlagioVeritas Configuration

Central configuration for the entire system.
All paths, model names, and thresholds live here.
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field


# ---------- Paths ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ROOT_DIR / "workspace"
DOCUMENTS_DIR = WORKSPACE_DIR / "documents"
VECTORS_DIR = WORKSPACE_DIR / "vectors"
REPORTS_DIR = WORKSPACE_DIR / "reports"
DATA_DIR = ROOT_DIR / "veritas" / "data"
DB_PATH = WORKSPACE_DIR / "veritas.db"

# Create directories if they don't exist
for _dir in (WORKSPACE_DIR, DOCUMENTS_DIR, VECTORS_DIR, REPORTS_DIR, DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------- Models ----------
# LLM (via Ollama)
LLM_MODEL = os.getenv("VERITAS_LLM", "llama3.2:3b")
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT = 120  # seconds

# Embeddings (sentence-transformers)
EMBEDDING_MODEL_EN = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_MULTI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM_EN = 384
EMBEDDING_DIM_MULTI = 384

# OCR
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
OCR_LANGUAGES = "eng+urd+hin+ara"  # English + Urdu + Hindi + Arabic


# ---------- Thresholds ----------
@dataclass
class Thresholds:
    """Similarity thresholds for different agents."""
    # Exact copy
    exact_match_min: float = 0.85
    ngram_size: int = 5
    min_match_words: int = 8

    # Semantic
    semantic_min: float = 0.60           # ← CHANGED
    semantic_high: float = 0.80          # ← NEW
    semantic_critical: float = 0.90      # ← NEW

    # Paraphrase
    paraphrase_min: float = 0.65         # ← CHANGED
    paraphrase_high: float = 0.78        # ← NEW

    # Cross-language
    cross_language_min: float = 0.65     # ← CHANGED

    # Self-plagiarism
    self_plagiarism_min: float = 0.50    # ← CHANGED

    # AI-writing
    ai_writing_low: float = 0.40
    ai_writing_medium: float = 0.60
    ai_writing_high: float = 0.80

    # Citation
    citation_confidence_min: float = 0.50


THRESHOLDS = Thresholds()


# ---------- Processing ----------
@dataclass
class ProcessingConfig:
    """Document processing settings."""
    max_file_size_mb: int = 50
    max_pages: int = 500
    chunk_size_words: int = 200
    chunk_overlap_words: int = 40
    min_paragraph_chars: int = 40
    batch_parallel_workers: int = 2  # Low for 8GB RAM
    llm_max_retries: int = 2


PROCESSING = ProcessingConfig()


# ---------- Report ----------
@dataclass
class ReportConfig:
    """Report generation settings."""
    output_formats: list = field(default_factory=lambda: ["json", "html", "pdf"])
    include_evidence: bool = True
    include_explanation: bool = True
    max_evidence_per_finding: int = 5
    human_review_threshold: float = 0.5


REPORT = ReportConfig()


# ---------- Logging ----------
LOG_LEVEL = os.getenv("VERITAS_LOG_LEVEL", "INFO")


__all__ = [
    "ROOT_DIR", "WORKSPACE_DIR", "DOCUMENTS_DIR", "VECTORS_DIR",
    "REPORTS_DIR", "DATA_DIR", "DB_PATH",
    "LLM_MODEL", "LLM_TEMPERATURE", "LLM_TIMEOUT",
    "EMBEDDING_MODEL_EN", "EMBEDDING_MODEL_MULTI",
    "EMBEDDING_DIM_EN", "EMBEDDING_DIM_MULTI",
    "TESSERACT_CMD", "OCR_LANGUAGES",
    "THRESHOLDS", "PROCESSING", "REPORT", "LOG_LEVEL",
]