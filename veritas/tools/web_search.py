"""
Web search + scrape for online plagiarism detection.

Uses ddgs (new duckduckgo-search) with multiple fallbacks.
"""
from __future__ import annotations
import time
import re
from typing import Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass
class WebResult:
    url: str
    title: str
    snippet: str
    full_text: str = ""
    domain: str = ""
    fetched: bool = False
    error: str = ""


def is_available() -> bool:
    """Check if web search tools available."""
    try:
        import ddgs  # noqa
        return True
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS  # noqa
        return True
    except ImportError:
        pass
    return False


def _get_ddgs_class():
    """Get DDGS class from either package."""
    try:
        from ddgs import DDGS
        return DDGS, "ddgs"
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS
        return DDGS, "duckduckgo_search"
    except ImportError:
        return None, None


def web_search(query: str, max_results: int = 5, timeout: int = 10) -> list[WebResult]:
    """Search web. Tries multiple backends."""
    DDGS, pkg = _get_ddgs_class()
    if DDGS is None:
        return []

    results = []
    try:
        with DDGS(timeout=timeout) as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href", "") or r.get("url", "")
                results.append(WebResult(
                    url=url,
                    title=r.get("title", ""),
                    snippet=r.get("body", "") or r.get("snippet", ""),
                    domain=urlparse(url).netloc,
                ))
    except Exception as e:
        # Try fallback backend if available
        try:
            from duckduckgo_search import DDGS as AltDDGS
            with AltDDGS(timeout=timeout) as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    url = r.get("href", "") or r.get("url", "")
                    results.append(WebResult(
                        url=url,
                        title=r.get("title", ""),
                        snippet=r.get("body", "") or r.get("snippet", ""),
                        domain=urlparse(url).netloc,
                    ))
        except Exception:
            pass

    return results


def fetch_page_text(url: str, timeout: int = 15) -> str:
    """Fetch + extract clean text from URL."""
    # Try trafilatura first
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_recall=True,
            )
            if text and len(text) > 200:
                return text
    except Exception:
        pass

    # Fallback: requests + BeautifulSoup
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            # Remove scripts/styles
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            if len(text) > 200:
                return text
    except Exception:
        pass

    return ""


def fetch_pages(results: list[WebResult], max_pages: int = 3,
                delay: float = 0.5) -> list[WebResult]:
    """Fetch full text for top results."""
    fetched = 0
    for r in results:
        if fetched >= max_pages:
            break
        if not r.url or not r.url.startswith("http"):
            continue
        text = fetch_page_text(r.url)
        if text and len(text) > 200:
            r.full_text = text
            r.fetched = True
            fetched += 1
        else:
            r.error = "fetch_failed"
        time.sleep(delay)
    return results


def build_queries(text: str, num_queries: int = 3) -> list[str]:
    """Build search queries from longest sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    candidates = [
        s.strip().strip('"').strip("'")
        for s in sentences
        if 40 <= len(s) <= 200
    ]
    candidates.sort(key=len, reverse=True)

    seen = set()
    queries = []
    for c in candidates:
        key = c[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(c[:150])
        if len(queries) >= num_queries:
            break

    return queries


__all__ = [
    "WebResult", "is_available", "web_search",
    "fetch_page_text", "fetch_pages", "build_queries",
]