"""AI Tool Detection Agent — always reports status."""
from __future__ import annotations
from typing import Optional

from veritas.agents.base import BaseAgent, AgentContext
from veritas.models.finding import (
    AgentResult, Evidence, Finding, FindingType, Severity,
)
from veritas.tools.ai_tool_detector import analyze_ai_tool
from veritas.tools.text_utils import clean_text


class AIToolAgent(BaseAgent):
    name = "ai_tool_agent"
    description = "Detects specific AI tool signatures."

    def run(self, task: str, context: Optional[AgentContext] = None) -> AgentResult:
        start = self._time_start()
        if context is None or not context.document_text:
            return AgentResult.failed(self.name, "No document text")

        text = clean_text(context.document_text)
        if len(text.split()) < 50:
            return AgentResult(
                agent=self.name, status="ok",
                findings=[self._info_finding(
                    "AI-tool detection skipped",
                    f"Document too short ({len(text.split())} words, need 50+).",
                )],
                notes="Text too short",
                duration_seconds=self._time_end(start),
            )

        self.log_info("Detecting AI tool signatures...")
        analysis = analyze_ai_tool(text)

        metrics = {
            "top_tool": analysis.top_tool,
            "confidence": round(analysis.confidence, 4),
            "signals_count": len(analysis.signals),
            "word_count": len(text.split()),
        }

        findings = []

        if analysis.signals and analysis.confidence >= 0.3:
            # ---------- Strong/medium match ----------
            evidence_list = []
            for sig in analysis.signals:
                for pat in sig.matched_patterns[:3]:
                    evidence_list.append(Evidence(
                        submitted_text=pat.get("example", ""),
                        source_text=f"pattern: {pat['pattern']}",
                        similarity=sig.score,
                        method=f"ai_tool_{sig.tool}",
                        metadata={
                            "tool": sig.tool,
                            "strength": sig.strength,
                            "count": pat["count"],
                        },
                    ))

            tool_name = {
                "gemini": "Google Gemini",
                "chatgpt": "OpenAI ChatGPT",
                "claude": "Anthropic Claude",
                "ai_generic_style": "Generic AI style",
            }.get(analysis.top_tool, analysis.top_tool)

            severity = (
                Severity.HIGH if analysis.confidence >= 0.7 else
                Severity.MEDIUM if analysis.confidence >= 0.4 else
                Severity.LOW
            )

            findings.append(self.make_finding(
                finding_type=FindingType.AI_WRITING,
                title=f"AI tool signature detected: {tool_name}",
                description=(
                    f"Stylistic patterns consistent with {tool_name} found. "
                    f"Confidence: {analysis.confidence:.0%}"
                ),
                confidence=analysis.confidence,
                evidence=evidence_list,
                explanation=[
                    f"Top match: {tool_name}",
                    f"Confidence: {analysis.confidence:.2%}",
                    "",
                    "Detected signals:",
                ] + [
                    f"  • {s.tool} ({s.strength.upper()}) — "
                    f"{sum(p['count'] for p in s.matched_patterns)} pattern hits"
                    for s in analysis.signals
                ] + [
                    "",
                    "⚠️ These are STYLISTIC HINTS, not proof.",
                    "Multiple AI tools share similar patterns.",
                    "Verify manually before concluding.",
                ],
                severity=severity,
                metadata={
                    "top_tool": analysis.top_tool,
                    "confidence": round(analysis.confidence, 4),
                    "signals": [
                        {
                            "tool": s.tool,
                            "strength": s.strength,
                            "score": round(s.score, 4),
                            "matches": s.matched_patterns,
                        }
                        for s in analysis.signals
                    ],
                },
            ))
        else:
            # ---------- No match — INFO finding (important for transparency) ----------
            findings.append(self._info_finding(
                title="AI-tool detection: No strong signatures",
                description=(
                    "Scanned for Gemini/ChatGPT/Claude stylistic patterns. "
                    "No significant AI-tool signatures detected."
                ),
                extra_explanation=[
                    f"Words analyzed: {metrics['word_count']}",
                    f"Signals checked: Gemini, ChatGPT, Claude, generic AI style",
                    f"Confidence: {analysis.confidence:.2%}",
                    "",
                    "Interpretation:",
                    "  • Low confidence means text does NOT match common AI patterns.",
                    "  • This does NOT guarantee human authorship.",
                    "  • AI tools can be prompted to write in human style.",
                    "  • Absence of signature = inconclusive, not proof.",
                ],
                metadata={
                    "top_tool": "none",
                    "confidence": round(analysis.confidence, 4),
                    "signals": [],
                },
            ))

        duration = self._time_end(start)
        self.log_success(f"AI tool: {analysis.top_tool} ({duration:.2f}s)")

        return AgentResult(
            agent=self.name, status="ok", findings=findings,
            metrics=metrics,
            notes=f"Top: {analysis.top_tool}",
            duration_seconds=duration,
        )

    def _info_finding(self, title: str, description: str,
                      extra_explanation: list = None,
                      metadata: dict = None) -> Finding:
        """Create an INFO-level finding."""
        return self.make_finding(
            finding_type=FindingType.AI_WRITING,
            title=title,
            description=description,
            confidence=0.5,
            evidence=[],
            explanation=extra_explanation or [],
            severity=Severity.INFO,
            metadata=metadata or {},
        )


__all__ = ["AIToolAgent"]