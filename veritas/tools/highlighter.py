"""Text highlighter for evidence display."""
from __future__ import annotations
from difflib import SequenceMatcher
import html
import re


def highlight_similar(text_a: str, text_b: str) -> tuple[str, str]:
    """
    Highlight matching segments between two texts.

    Returns (html_a, html_b) with <mark> tags on matches.
    """
    if not text_a or not text_b:
        return html.escape(text_a), html.escape(text_b)

    # Tokenize preserving whitespace
    tokens_a = re.findall(r"\S+\s*|\s+", text_a)
    tokens_b = re.findall(r"\S+\s*|\s+", text_b)

    matcher = SequenceMatcher(None, tokens_a, tokens_b)
    blocks = matcher.get_matching_blocks()

    # Build highlighted versions
    out_a = []
    out_b = []
    a_pos = 0
    b_pos = 0

    for block in blocks:
        # Non-matching part before this block
        for i in range(a_pos, block.a):
            out_a.append(html.escape(tokens_a[i]))
        for j in range(b_pos, block.b):
            out_b.append(html.escape(tokens_b[j]))

        # Matching part
        match_text = "".join(tokens_a[block.a:block.a + block.size])
        escaped = html.escape(match_text)
        out_a.append(f'<mark class="hl">{escaped}</mark>')
        out_b.append(f'<mark class="hl">{escaped}</mark>')

        a_pos = block.a + block.size
        b_pos = block.b + block.size

    # Remaining
    for i in range(a_pos, len(tokens_a)):
        out_a.append(html.escape(tokens_a[i]))
    for j in range(b_pos, len(tokens_b)):
        out_b.append(html.escape(tokens_b[j]))

    return "".join(out_a), "".join(out_b)


def inline_highlight(text: str, fragment: str) -> str:
    """Highlight fragment inside text."""
    if not text or not fragment:
        return html.escape(text)

    escaped = html.escape(text)
    frag_escaped = html.escape(fragment)
    # Escape regex specials
    frag_pattern = re.escape(frag_escaped)
    result = re.sub(
        frag_pattern,
        f'<mark class="hl">{frag_escaped}</mark>',
        escaped,
        count=1,
        flags=re.IGNORECASE,
    )
    return result


__all__ = ["highlight_similar", "inline_highlight"]