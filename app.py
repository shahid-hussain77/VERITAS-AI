"""
VERITAS-AI — Premium Gemini-style Streamlit App.

Run: streamlit run app.py --server.fileWatcherType=none
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from veritas import __version__
from veritas.config import WORKSPACE_DIR, DOCUMENTS_DIR
from veritas.orchestrator import Orchestrator
from veritas.memory.store import memory
from veritas.memory.history import save_run, list_runs, load_run, delete_run


# ---------- Page config ----------
st.set_page_config(
    page_title="VERITAS-AI · Academic Integrity Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- PREMIUM GEMINI-STYLE CSS ----------
st.markdown("""
<style>
/* ============ GLOBAL ============ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0a0e1a;
    background-image:
        radial-gradient(circle at 10% 5%, rgba(59, 130, 246, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(139, 92, 246, 0.10) 0%, transparent 40%),
        radial-gradient(circle at 50% 90%, rgba(16, 185, 129, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 20% 60%, rgba(236, 72, 153, 0.06) 0%, transparent 40%);
    background-attachment: fixed;
    color: #e2e8f0;
}

/* Main container */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Hide streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* ============ ANIMATIONS ============ */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.15); opacity: 0.7; }
}
@keyframes glow {
    0%, 100% {
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.3),
                    0 0 40px rgba(59, 130, 246, 0.15);
    }
    50% {
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.5),
                    0 0 60px rgba(139, 92, 246, 0.25);
    }
}
@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(59, 130, 246, 0.3); }
    50% { border-color: rgba(139, 92, 246, 0.6); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
}
@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}
@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* ============ HERO HEADER ============ */
.hero-container {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    animation: fadeInUp 0.8s ease-out;
}

.hero-logo {
    font-size: 4.5rem;
    display: inline-block;
    animation: float 3s ease-in-out infinite;
    filter: drop-shadow(0 0 30px rgba(16, 185, 129, 0.4));
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg,
        #60a5fa, #a78bfa, #f472b6, #34d399, #60a5fa);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 8s ease infinite;
    margin: 0.5rem 0;
    letter-spacing: -2px;
    line-height: 1.1;
}

.hero-tagline {
    color: #94a3b8;
    font-size: 1.15rem;
    font-style: italic;
    margin: 0.5rem 0 1.5rem 0;
    animation: fadeIn 1.2s ease-out 0.3s both;
}

.hero-badges {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1rem;
    animation: fadeIn 1.4s ease-out 0.5s both;
}

.hero-badge {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(59, 130, 246, 0.2);
    padding: 0.4rem 1rem;
    border-radius: 2rem;
    font-size: 0.85rem;
    color: #cbd5e1;
    transition: all 0.3s;
}
.hero-badge:hover {
    border-color: rgba(139, 92, 246, 0.6);
    transform: translateY(-2px);
}

.hero-badge .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.4rem;
    animation: pulse 2s infinite;
}
.dot.green { background: #10b981; box-shadow: 0 0 8px #10b981; }
.dot.blue { background: #3b82f6; box-shadow: 0 0 8px #3b82f6; }
.dot.purple { background: #8b5cf6; box-shadow: 0 0 8px #8b5cf6; }

/* ============ MODE CARDS ============ */
.mode-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1rem;
    margin: 1rem 0 2rem 0;
}

.mode-card {
    background: linear-gradient(135deg,
        rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    backdrop-filter: blur(20px);
    border: 1px solid rgba(59, 130, 246, 0.15);
    border-radius: 1rem;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.6s ease-out;
}

.mode-card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg,
        transparent, rgba(139, 92, 246, 0.15), transparent);
    transition: left 0.6s;
}
.mode-card:hover::before { left: 100%; }

.mode-card:hover {
    border-color: rgba(139, 92, 246, 0.5);
    transform: translateY(-4px);
    box-shadow: 0 20px 50px rgba(139, 92, 246, 0.2);
}

.mode-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    display: block;
}

.mode-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.3rem;
}

.mode-desc {
    font-size: 0.8rem;
    color: #94a3b8;
}

/* ============ METRIC CARDS ============ */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background: linear-gradient(135deg,
        rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    backdrop-filter: blur(20px);
    padding: 1.25rem;
    border-radius: 1rem;
    border: 1px solid rgba(59, 130, 246, 0.15);
    animation: fadeInUp 0.5s ease-out;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
}

.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 4px 0 0 4px;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(139, 92, 246, 0.4);
    box-shadow: 0 10px 30px rgba(139, 92, 246, 0.15);
}

.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
}

.metric-value.small {
    font-size: 1.1rem;
    font-weight: 500;
}

/* ============ FINDING CARDS ============ */
.finding-header {
    background: linear-gradient(135deg,
        rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
    backdrop-filter: blur(20px);
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    border-left: 4px solid;
    animation: slideInLeft 0.4s ease-out;
    transition: all 0.3s;
}
.finding-header:hover {
    transform: translateX(4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}
.sev-critical { border-left-color: #ef4444; }
.sev-high { border-left-color: #f97316; }
.sev-medium { border-left-color: #eab308; }
.sev-low { border-left-color: #22c55e; }
.sev-info { border-left-color: #3b82f6; }

/* ============ EVIDENCE BOXES ============ */
.evidence-box {
    background: linear-gradient(135deg,
        rgba(16, 185, 129, 0.08), rgba(15, 23, 42, 0.6));
    border: 1px solid rgba(16, 185, 129, 0.2);
    padding: 0.85rem;
    border-radius: 0.6rem;
    font-size: 0.88rem;
    color: #d1fae5;
    line-height: 1.6;
    animation: slideInLeft 0.4s ease-out;
    margin: 0.4rem 0;
}

.source-box {
    background: linear-gradient(135deg,
        rgba(245, 158, 11, 0.08), rgba(15, 23, 42, 0.6));
    border: 1px solid rgba(245, 158, 11, 0.2);
    padding: 0.85rem;
    border-radius: 0.6rem;
    font-size: 0.88rem;
    color: #fef3c7;
    line-height: 1.6;
    animation: slideInRight 0.4s ease-out;
    margin: 0.4rem 0;
}

mark.hl {
    background: #fde047;
    color: #0f172a;
    padding: 0.08rem 0.2rem;
    border-radius: 0.2rem;
    font-weight: 600;
    box-shadow: 0 0 8px rgba(253, 224, 71, 0.4);
}

/* ============ SIMILARITY BAR ============ */
.sim-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.5rem 0;
}
.sim-bar-wrapper {
    flex: 1;
    background: rgba(30, 41, 59, 0.7);
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
}
.sim-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
    border-radius: 4px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    animation: shimmer 2s infinite;
    background-size: 200% 100%;
}
.sim-value {
    font-family: 'JetBrains Mono', monospace;
    color: #10b981;
    font-weight: 600;
    min-width: 60px;
}

/* ============ PROGRESS PULSE ============ */
.pulse-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 1.2s infinite;
    margin-right: 0.5rem;
    box-shadow: 0 0 12px #10b981;
}

/* ============ GLOW BUTTONS ============ */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.6rem !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.5) !important;
}

/* ============ SIDEBAR ============ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        rgba(15, 23, 42, 0.98), rgba(10, 14, 26, 0.98));
    border-right: 1px solid rgba(59, 130, 246, 0.1);
}

section[data-testid="stSidebar"] .stRadio label {
    background: rgba(30, 41, 59, 0.5);
    padding: 0.6rem 0.9rem;
    border-radius: 0.6rem;
    margin: 0.2rem 0;
    border: 1px solid transparent;
    transition: all 0.25s;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(59, 130, 246, 0.12);
    border-color: rgba(59, 130, 246, 0.3);
}

/* ============ EXPANDER ============ */
.streamlit-expanderHeader {
    background: linear-gradient(135deg,
        rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9)) !important;
    border-radius: 0.6rem !important;
    border: 1px solid rgba(59, 130, 246, 0.15) !important;
    transition: all 0.3s !important;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.15) !important;
}

/* ============ FILE UPLOADER ============ */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg,
        rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.7));
    border: 2px dashed rgba(59, 130, 246, 0.3);
    border-radius: 1rem;
    padding: 1rem;
    transition: all 0.3s;
    animation: borderGlow 4s infinite;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(139, 92, 246, 0.6);
    background: linear-gradient(135deg,
        rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.05));
}

/* ============ HISTORY CARD ============ */
.history-card {
    background: linear-gradient(135deg,
        rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
    padding: 0.7rem;
    border-radius: 0.6rem;
    margin: 0.4rem 0;
    border-left: 3px solid #3b82f6;
    animation: slideInLeft 0.3s ease-out;
    font-size: 0.85rem;
}

/* ============ INFO BANNER ============ */
.info-banner {
    background: linear-gradient(90deg,
        rgba(16, 185, 129, 0.15), rgba(59, 130, 246, 0.15),
        rgba(139, 92, 246, 0.15));
    background-size: 200% 100%;
    animation: gradientShift 4s ease infinite;
    color: #f1f5f9;
    padding: 0.85rem 1.25rem;
    border-radius: 0.75rem;
    margin: 0.75rem 0;
    font-weight: 500;
    border: 1px solid rgba(139, 92, 246, 0.25);
    backdrop-filter: blur(10px);
}

/* ============ SECTION HEADERS ============ */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid;
    border-image: linear-gradient(90deg, #3b82f6, #8b5cf6, transparent) 1;
    animation: fadeInUp 0.5s ease-out;
}

/* ============ SCROLLBAR ============ */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #3b82f6, #8b5cf6);
    border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #60a5fa, #a78bfa);
}

/* ============ STEP INDICATOR ============ */
.step-indicator {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin: 1rem 0;
    animation: fadeIn 0.5s;
}
.step-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: rgba(59, 130, 246, 0.3);
    transition: all 0.3s;
}
.step-dot.active {
    background: #10b981;
    box-shadow: 0 0 12px #10b981;
    animation: pulse 1s infinite;
}
.step-dot.done {
    background: #3b82f6;
}

/* Metric card in columns */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg,
        rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    padding: 1rem;
    border-radius: 0.75rem;
    border-left: 3px solid #3b82f6;
    animation: fadeInUp 0.5s ease-out;
}
</style>
""", unsafe_allow_html=True)


# ---------- HERO HEADER ----------
st.markdown("""
<div class="hero-container">
    <div class="hero-logo">🛡️</div>
    <h1 class="hero-title">VERITAS-AI</h1>
    <p class="hero-tagline">Don't just detect similarity. Show the evidence.</p>
    <div class="hero-badges">
        <span class="hero-badge">
            <span class="dot green"></span>100% Local
        </span>
        <span class="hero-badge">
            <span class="dot blue"></span>Free Forever
        </span>
        <span class="hero-badge">
            <span class="dot purple"></span>Multi-Agent
        </span>
        <span class="hero-badge">
            <span class="dot green"></span>Evidence-Based
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size:2.5rem;">⚙️</div>
        <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9;">
            Control Panel
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "**Analysis Mode**",
        ["📄 Single Document", "🔀 Two Documents", "🎓 Classroom Batch"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### 🎯 Options")
    use_sources = st.checkbox("Compare vs sources", value=True)
    use_history = st.checkbox("Self-plagiarism check", value=False)
    use_web = st.checkbox(
        "🌐 Online web check",
        value=True,
        help="DuckDuckGo (free). May be slow or blocked in some regions.",
    )

    st.divider()
    st.markdown("#### 📊 Database")
    try:
        stats = memory.stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Docs", stats["documents"])
        c2.metric("Findings", stats["findings"])
        c3.metric("Runs", stats["agent_runs"])
    except Exception as e:
        st.warning(f"DB error: {e}")

    if st.button("🗑️ Reset DB", use_container_width=True):
        memory.reset()
        st.success("Reset done")
        st.rerun()

    st.divider()
    st.markdown("#### 🕐 Recent Runs")
    try:
        runs = list_runs(limit=5)
        if runs:
            for r in runs:
                st.markdown(
                    f'<div class="history-card">'
                    f'<b>{r["document"][:30]}</b><br>'
                    f'<small>{r["saved_at"]} · {r["findings"]} findings · '
                    f'{r["confidence"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"↻ Load", key=f"load_{r['run_id']}"):
                    loaded = load_run(r["run_id"])
                    if loaded:
                        st.session_state["last_result"] = loaded["result"]
                        st.success("Loaded")
                        st.rerun()
        else:
            st.caption("No runs yet")
    except Exception:
        st.caption("History unavailable")

    st.divider()
    st.caption(f"📁 `{WORKSPACE_DIR.name}/`")
    st.caption(f"**Version** v{__version__}")


# ---------- Helpers ----------
def save_uploaded(uploaded_file, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def animated_steps(steps: list, duration_per_step: float = 0.4):
    """Animate through steps with pulsing dot."""
    progress_bar = st.progress(0)
    status_area = st.empty()

    for i, step in enumerate(steps):
        progress = (i + 1) / len(steps)
        progress_bar.progress(progress)
        status_area.markdown(
            f'<div style="text-align:center; padding:0.5rem;">'
            f'<span class="pulse-dot"></span>'
            f'<span style="color:#cbd5e1; font-weight:500;">{step}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        time.sleep(duration_per_step)

    status_area.markdown(
        '<div style="text-align:center; padding:0.5rem; color:#10b981; '
        'font-weight:600;">✓ Complete</div>',
        unsafe_allow_html=True,
    )
    time.sleep(0.3)
    progress_bar.empty()
    status_area.empty()


def sim_bar(sim: float) -> str:
    """Generate animated similarity bar HTML."""
    pct = int(sim * 100)
    return (
        f'<div class="sim-container">'
        f'<div class="sim-bar-wrapper">'
        f'<div class="sim-bar-fill" style="width:{pct}%"></div>'
        f'</div>'
        f'<div class="sim-value">{sim:.1%}</div>'
        f'</div>'
    )


# ---------- REPORT RENDERER ----------
def render_report(result: dict):
    if result.get("status") != "ok":
        st.error(f"❌ Analysis failed: {result.get('error', 'unknown')}")
        return

    report = result["report"]
    score = report.get("score", {})
    findings = report.get("findings", [])

    # Banner
    st.markdown(
        f'<div class="info-banner">'
        f'✨ <b>Analysis complete</b> — '
        f'{len(findings)} finding(s) detected · '
        f'Confidence: <b>{score.get("overall_confidence", "LOW")}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Top metrics grid
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)

    conf = score.get("overall_confidence", "LOW")
    conf_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(conf, "⚪")

    metrics = [
        ("📄 Document", report.get("document_name", "?")[:28]),
        ("📝 Words", f"{report.get('document_words', 0):,}"),
        ("🔍 Findings", str(len(findings))),
        ("🎯 Confidence", f"{conf_color} {conf}"),
    ]

    for label, value in metrics:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if report.get("human_review_recommended"):
        st.warning("⚠️ **Human review strongly recommended** — significant findings detected.")

    # Signal breakdown chart
    st.markdown('<div class="section-header">🔍 Signal Breakdown</div>',
                unsafe_allow_html=True)

    signal_data = {
        "Signal": [
            "Exact overlap", "Semantic", "Paraphrase",
            "Cross-language", "Self-overlap", "Style deviation",
        ],
        "Score": [
            score.get("exact_overlap", 0),
            score.get("semantic_overlap", 0),
            score.get("paraphrase_overlap", 0),
            score.get("cross_language_overlap", 0),
            score.get("self_overlap", 0),
            score.get("style_deviation", 0),
        ],
    }
    df = pd.DataFrame(signal_data)
    fig = px.bar(
        df, x="Score", y="Signal", orientation="h",
        color="Score", color_continuous_scale="RdYlGn_r",
        range_color=(0, 1),
    )
    fig.update_layout(
        height=300, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter"),
        xaxis=dict(gridcolor="rgba(59,130,246,0.1)", range=[0, 1]),
        yaxis=dict(gridcolor="rgba(59,130,246,0.05)"),
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    # AI signal callout
    ai_sig = score.get("ai_writing_signal", "INCONCLUSIVE")
    st.markdown('<div class="section-header">🤖 AI-Writing Signal</div>',
                unsafe_allow_html=True)
    if ai_sig == "HIGH":
        st.error(f"**{ai_sig}** — Strong AI-writing indicators detected")
    elif ai_sig == "MEDIUM":
        st.warning(f"**{ai_sig}** — Moderate AI-writing indicators")
    elif ai_sig == "LOW":
        st.info(f"**{ai_sig}** — Weak AI-writing indicators")
    else:
        st.info(f"**{ai_sig}** — Not enough signal for classification")

    # Findings
    st.markdown(
        f'<div class="section-header">📑 Findings ({len(findings)})</div>',
        unsafe_allow_html=True,
    )

    if not findings:
        st.success("✅ No findings. Document appears clean.")
    else:
        sev_filter = st.multiselect(
            "Filter severity",
            ["critical", "high", "medium", "low", "info"],
            default=["critical", "high", "medium"],
            key="sev_filter",
        )
        filtered = [f for f in findings if f.get("severity") in sev_filter]
        st.caption(f"Showing **{len(filtered)}** of **{len(findings)}** findings")

        for i, f in enumerate(filtered):
            sev = f.get("severity", "info")
            icon = {
                "critical": "🔴", "high": "🟠",
                "medium": "🟡", "low": "🟢", "info": "🔵",
            }.get(sev, "⚪")

            with st.expander(
                f"{icon} **[{sev.upper()}]** {f.get('title', '')} "
                f"— {f.get('confidence', 0):.0%}",
                expanded=(i < 2 and sev in ("critical", "high")),
            ):
                st.markdown(f"**{f.get('description', '')}**")
                cc1, cc2 = st.columns(2)
                cc1.caption(f"🤖 Agent: `{f.get('agent', '')}`")
                cc2.caption(f"📊 Confidence: {f.get('confidence', 0):.2%}")

                if f.get("explanation"):
                    with st.container(border=True):
                        st.markdown("**📋 Explanation**")
                        for line in f["explanation"][:12]:
                            st.markdown(f"› {line}")

                ev = f.get("evidence", [])
                if ev:
                    st.markdown(f"**🔍 Evidence ({len(ev)} items)**")
                    for e in ev[:6]:
                        with st.container(border=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(
                                    '<div style="color:#10b981; '
                                    'font-weight:600; font-size:0.75rem; '
                                    'letter-spacing:0.1em; '
                                    'margin-bottom:0.3rem;">'
                                    '📄 SUBMITTED</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f'<div class="evidence-box">'
                                    f'{e.get("submitted_text", "")[:500]}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            with col2:
                                st.markdown(
                                    '<div style="color:#f59e0b; '
                                    'font-weight:600; font-size:0.75rem; '
                                    'letter-spacing:0.1em; '
                                    'margin-bottom:0.3rem;">'
                                    '📚 SOURCE</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f'<div class="source-box">'
                                    f'{e.get("source_text", "")[:500]}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                            sim = e.get("similarity", 0)
                            st.markdown(sim_bar(sim), unsafe_allow_html=True)

                            method = e.get("method", "")
                            st.caption(f"⚙️ Method: `{method}`")

                            src = e.get("source_document", "")
                            if src.startswith("http"):
                                st.markdown(f"🔗 [Open source page]({src})")
                            elif src:
                                st.caption(f"📎 Source: `{src}`")

    # Export
    st.markdown('<div class="section-header">📥 Export Report</div>',
                unsafe_allow_html=True)

    base = report.get("document_name", "report").replace(".", "_")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.download_button(
            "📄 Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False, default=str),
            file_name=f"{base}_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with cc2:
        st.download_button(
            "🌐 Download HTML",
            data=_build_html_report(report),
            file_name=f"{base}_report.html",
            mime="text/html",
            use_container_width=True,
        )


def _build_html_report(report: dict) -> str:
    score = report.get("score", {})
    findings = report.get("findings", [])
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VERITAS-AI Report</title>
<style>
body{{font-family:Inter,sans-serif;max-width:900px;margin:2rem auto;padding:1rem;background:#0a0e1a;color:#e2e8f0}}
h1{{background:linear-gradient(90deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.metric{{background:#1e293b;padding:.8rem;margin:.5rem 0;border-radius:.4rem;border-left:4px solid #3b82f6}}
.finding{{background:#1e293b;padding:1rem;margin:1rem 0;border-radius:.4rem;border-left:4px solid #f59e0b}}
.critical{{border-left-color:#ef4444}} .high{{border-left-color:#f97316}}
.medium{{border-left-color:#eab308}} .low{{border-left-color:#22c55e}}
</style></head><body>
<h1>🛡️ VERITAS-AI Integrity Report</h1>
<p><b>Document:</b> {report.get('document_name')}</p>
<p><b>Words:</b> {report.get('document_words', 0):,}</p>
<p><b>Pages:</b> {report.get('document_pages', 0)}</p>
<hr><h2>Signals</h2>
<div class="metric">Exact overlap: {score.get('exact_overlap', 0)*100:.1f}%</div>
<div class="metric">Semantic overlap: {score.get('semantic_overlap', 0)*100:.1f}%</div>
<div class="metric">Paraphrase: {score.get('paraphrase_overlap', 0)*100:.1f}%</div>
<div class="metric">Cross-language: {score.get('cross_language_overlap', 0)*100:.1f}%</div>
<div class="metric">Self-overlap: {score.get('self_overlap', 0)*100:.1f}%</div>
<div class="metric">AI-writing: {score.get('ai_writing_signal', '?')}</div>
<hr><h2>Findings ({len(findings)})</h2>
{''.join(f'<div class="finding {f.get("severity","info")}"><b>[{f.get("severity","").upper()}]</b> {f.get("title","")}<br><i>Confidence: {f.get("confidence",0):.0%}</i><br>{f.get("description","")}</div>' for f in findings)}
<hr><p><i>Generated by VERITAS-AI v{__version__}</i></p>
</body></html>"""


# ---------- MODE 1: Single Document ----------
if mode == "📄 Single Document":
    st.markdown(
        '<div class="section-header">📄 Single Document Analysis</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📤 Submitted Document")
        submitted_file = st.file_uploader(
            "Upload document",
            type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"],
            key="single_submitted",
            label_visibility="collapsed",
        )
    with col2:
        if use_sources:
            st.markdown("##### 📚 Source Documents (optional)")
            source_files = st.file_uploader(
                "Upload sources",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="single_sources",
                label_visibility="collapsed",
            )
        else:
            source_files = []

    if use_history:
        history_files = st.file_uploader(
            "Author's history (self-plagiarism)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="single_history",
        )
    else:
        history_files = []

    st.markdown("<br>", unsafe_allow_html=True)

    if submitted_file and st.button("🚀 Run Analysis", type="primary",
                                     use_container_width=True,
                                     key="run_single"):
        with st.spinner("💾 Preparing files..."):
            sub_path = save_uploaded(submitted_file, DOCUMENTS_DIR)
            src_paths = [str(save_uploaded(f, DOCUMENTS_DIR))
                         for f in (source_files or [])]
            hist_paths = [str(save_uploaded(f, DOCUMENTS_DIR))
                          for f in (history_files or [])]

        steps = [
            "📖 Parsing document",
            "🎯 Exact copy detection",
            "🧠 Semantic analysis",
            "🔄 Paraphrase detection",
            "🌐 Cross-language scan",
            "📖 Citation integrity",
            "🤖 AI-writing analysis",
            "🎭 AI-tool detection",
            "📊 Style analysis",
        ]
        if use_web:
            steps.append("🌐 Web plagiarism check")

        steps.append("⚖️ Fusing evidence")
        animated_steps(steps, duration_per_step=0.25)

        with st.spinner("⚙️ Running complete analysis..."):
            orch = Orchestrator(verbose=False)
            t0 = time.time()
            try:
                result = orch.run(
                    submitted_path=str(sub_path),
                    source_paths=src_paths or None,
                    author_history_paths=hist_paths or None,
                    enable_web=use_web,
                )
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.stop()

        st.success(f"✨ Complete in {time.time() - t0:.1f}s")
        st.session_state["last_result"] = result
        try:
            save_run(result, name=submitted_file.name)
        except Exception:
            pass

    if "last_result" in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        render_report(st.session_state["last_result"])


# ---------- MODE 2: Two Documents ----------
elif mode == "🔀 Two Documents":
    st.markdown(
        '<div class="section-header">🔀 Two-Document Comparison</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        a = st.file_uploader("Document A",
                             type=["pdf", "docx", "txt"], key="two_a")
    with col2:
        b = st.file_uploader("Document B",
                             type=["pdf", "docx", "txt"], key="two_b")

    if a and b and st.button("🚀 Compare", type="primary",
                              use_container_width=True, key="run_compare"):
        with st.spinner("Parsing..."):
            pa = save_uploaded(a, DOCUMENTS_DIR)
            pb = save_uploaded(b, DOCUMENTS_DIR)

        animated_steps([
            "📖 Parsing documents",
            "🔍 Cross-comparing",
            "⚖️ Generating report",
        ], duration_per_step=0.4)

        with st.spinner("Comparing..."):
            orch = Orchestrator(verbose=False)
            result = orch.run(
                submitted_path=str(pa),
                source_paths=[str(pb)],
                enable_web=use_web,
            )
        st.session_state["last_result"] = result
        try:
            save_run(result, name=f"compare_{a.name}")
        except Exception:
            pass

    if "last_result" in st.session_state:
        render_report(st.session_state["last_result"])


# ---------- MODE 3: Classroom Batch ----------
elif mode == "🎓 Classroom Batch":
    st.markdown(
        '<div class="section-header">🎓 Classroom Batch Analysis</div>',
        unsafe_allow_html=True,
    )
    st.info("📌 Upload **3 or more** student submissions to detect collusion clusters.")

    files = st.file_uploader(
        "Upload assignments",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="batch_files",
    )

    if files and len(files) >= 3 and st.button("🚀 Batch Analysis",
                                                type="primary",
                                                use_container_width=True,
                                                key="run_batch"):
        with st.spinner(f"Parsing {len(files)} documents..."):
            paths = [str(save_uploaded(f, DOCUMENTS_DIR)) for f in files]

        animated_steps([
            f"📚 Parsing {len(files)} documents",
            "🧠 Embedding all submissions",
            "🔍 Pairwise comparison",
            "🕸️ Building collusion graph",
            "⚖️ Generating classroom report",
        ], duration_per_step=0.5)

        with st.spinner("Batch comparison..."):
            orch = Orchestrator(verbose=False)
            try:
                result = orch.run(
                    submitted_path=paths[0],
                    classroom_paths=paths[1:],
                    enable_web=False,
                )
            except Exception as e:
                st.error(f"❌ Batch failed: {e}")
                st.stop()

        st.success("✨ Batch complete")
        st.session_state["last_result"] = result
        try:
            save_run(result, name=f"batch_{len(files)}_docs")
        except Exception:
            pass

    if "last_result" in st.session_state:
        render_report(st.session_state["last_result"])


# ---------- FOOTER ----------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#64748b; '
    'font-size:0.85rem; padding:1.5rem; '
    'border-top:1px solid rgba(59,130,246,0.1);">'
    '🛡️ <b>VERITAS-AI</b> v' + __version__ + ' — '
    'Local Multi-Agent Academic Integrity Engine<br>'
    '<span style="font-size:0.75rem;">'
    '⚠️ Provides evidence, not verdicts. Human review required.'
    '</span>'
    '</div>',
    unsafe_allow_html=True,
)