"""
OCR wrapper for Tesseract.

Handles:
- Images (JPG, PNG, etc.)
- Scanned PDFs (render page → image → OCR)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from veritas.config import TESSERACT_CMD, OCR_LANGUAGES


def _get_pytesseract():
    """Lazy import + configure pytesseract."""
    try:
        import pytesseract
    except ImportError:
        return None

    # Set Tesseract binary path if provided
    if TESSERACT_CMD and Path(TESSERACT_CMD).exists():
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    return pytesseract


def is_ocr_available() -> bool:
    """Check if OCR is usable."""
    pt = _get_pytesseract()
    if not pt:
        return False
    try:
        pt.get_tesseract_version()
        return True
    except Exception:
        return False


def get_available_languages() -> list[str]:
    """Return list of installed Tesseract languages."""
    pt = _get_pytesseract()
    if not pt:
        return []
    try:
        return pt.get_languages(config="")
    except Exception:
        return []


def pick_best_languages() -> str:
    """Pick available languages from our preferred set."""
    available = set(get_available_languages())
    if not available:
        return "eng"

    wanted = ["eng", "urd", "hin", "ara"]
    selected = [l for l in wanted if l in available]
    return "+".join(selected) if selected else "eng"


def ocr_image(image_path: str | Path, lang: Optional[str] = None) -> str:
    """Run OCR on a single image file."""
    pt = _get_pytesseract()
    if not pt:
        raise RuntimeError("pytesseract is not installed")

    from PIL import Image
    img = Image.open(str(image_path))

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    lang = lang or pick_best_languages()
    text = pt.image_to_string(img, lang=lang)
    return text.strip()


def ocr_pdf_page(page, lang: Optional[str] = None, dpi: int = 300) -> str:
    """
    OCR a single PyMuPDF page by rendering to image.

    Args:
        page: fitz.Page object
        lang: Tesseract language string
        dpi: rendering resolution (higher = more accurate, slower)
    """
    pt = _get_pytesseract()
    if not pt:
        raise RuntimeError("pytesseract is not installed")

    import io
    from PIL import Image

    # Render page to PNG bytes
    zoom = dpi / 72.0
    matrix = __import__("pymupdf").Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))

    lang = lang or pick_best_languages()
    text = pt.image_to_string(img, lang=lang)
    return text.strip()


__all__ = [
    "is_ocr_available",
    "get_available_languages",
    "pick_best_languages",
    "ocr_image",
    "ocr_pdf_page",
]