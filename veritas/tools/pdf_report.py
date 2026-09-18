"""PDF report generation using reportlab."""
from __future__ import annotations
from datetime import datetime
from io import BytesIO
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)


SEVERITY_COLORS = {
    "critical": colors.HexColor("#ef4444"),
    "high": colors.HexColor("#f97316"),
    "medium": colors.HexColor("#eab308"),
    "low": colors.HexColor("#22c55e"),
    "info": colors.HexColor("#3b82f6"),
}


def generate_pdf_report(report: dict) -> bytes:
    """Generate PDF report, return bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "H1Custom", parent=styles["Heading1"],
        textColor=colors.HexColor("#10b981"), fontSize=24,
    )
    h2 = ParagraphStyle(
        "H2Custom", parent=styles["Heading2"],
        textColor=colors.HexColor("#3b82f6"), fontSize=16,
    )
    body = styles["BodyText"]
    small = ParagraphStyle(
        "Small", parent=body, fontSize=9, textColor=colors.gray,
    )

    story = []
    score = report.get("score", {})
    findings = report.get("findings", [])

    # ---------- Cover ----------
    story.append(Paragraph("🛡️ VERITAS-AI", h1))
    story.append(Paragraph(
        "Academic Integrity Report", h2,
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        f"<b>Document:</b> {report.get('document_name', 'N/A')}",
        body,
    ))
    story.append(Paragraph(
        f"<b>Words:</b> {report.get('document_words', 0):,}",
        body,
    ))
    story.append(Paragraph(
        f"<b>Pages:</b> {report.get('document_pages', 0)}",
        body,
    ))
    story.append(Paragraph(
        f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        body,
    ))
    story.append(Spacer(1, 0.5 * cm))

    # ---------- Signals table ----------
    story.append(Paragraph("Signal Summary", h2))

    sig_data = [
        ["Signal", "Value"],
        ["Exact overlap", f"{score.get('exact_overlap', 0)*100:.1f}%"],
        ["Semantic overlap", f"{score.get('semantic_overlap', 0)*100:.1f}%"],
        ["Paraphrase", f"{score.get('paraphrase_overlap', 0)*100:.1f}%"],
        ["Cross-language", f"{score.get('cross_language_overlap', 0)*100:.1f}%"],
        ["Self-overlap", f"{score.get('self_overlap', 0)*100:.1f}%"],
        ["Source matches", str(score.get("source_matches", 0))],
        ["Citation issues", str(score.get("citation_issues", 0))],
        ["Reference issues", str(score.get("reference_issues", 0))],
        ["AI-writing signal", score.get("ai_writing_signal", "?")],
        ["Overall confidence", score.get("overall_confidence", "?")],
        ["Human review recommended",
         "YES" if report.get("human_review_recommended") else "NO"],
    ]
    sig_table = Table(sig_data, colWidths=[8 * cm, 8 * cm])
    sig_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#10b981")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 0.5 * cm))

    # ---------- Findings ----------
    story.append(Paragraph(f"Findings ({len(findings)})", h2))

    for i, f in enumerate(findings, 1):
        sev = f.get("severity", "info")
        color = SEVERITY_COLORS.get(sev, colors.grey)

        title = f"{i}. [{sev.upper()}] {f.get('title', '')}"
        story.append(Paragraph(
            f'<font color="{color.hexval()}"><b>{title}</b></font>',
            body,
        ))
        story.append(Paragraph(
            f"Confidence: {f.get('confidence', 0):.0%} | "
            f"Agent: {f.get('agent', '')}",
            small,
        ))
        story.append(Paragraph(
            f.get("description", ""), body,
        ))

        # Evidence
        ev = f.get("evidence", [])
        if ev:
            story.append(Paragraph("<b>Evidence:</b>", small))
            for j, e in enumerate(ev[:3], 1):
                ev_data = [
                    ["Submitted:", e.get("submitted_text", "")[:200]],
                    ["Source:", e.get("source_text", "")[:200]],
                    ["Similarity:", f"{e.get('similarity', 0):.2%}"],
                ]
                ev_table = Table(ev_data, colWidths=[3 * cm, 13 * cm])
                ev_table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (0, -1),
                     colors.HexColor("#f1f5f9")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(ev_table)
                story.append(Spacer(1, 0.2 * cm))

        story.append(Spacer(1, 0.3 * cm))

    # ---------- Footer ----------
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "Generated by VERITAS-AI — Local Multi-Agent Academic Integrity Engine. "
        "⚠️ This report provides evidence, not verdicts. Human review required.",
        small,
    ))

    doc.build(story)
    return buffer.getvalue()


__all__ = ["generate_pdf_report"]