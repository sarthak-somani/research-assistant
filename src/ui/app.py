"""
Streamlit Command Center — Futuristic Edition
==============================================

A dark-themed, glassmorphic UI to visualise the LangGraph multi-agent
pipeline in real time and generate / download the final PDF report.

Usage::

    streamlit run src/ui/app.py
"""

from __future__ import annotations

import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so relative imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import OUTPUT_DIR, MAX_RETRIES
from src.utils.logger import setup_logging, get_logger

# ---------------------------------------------------------------------------
# Initialise structured logging so a pipeline_run_<ts>.txt is created
# ---------------------------------------------------------------------------
setup_logging()
_file_logger = get_logger("streamlit_app")

# ═══════════════════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="FMCG GenAI — Command Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Futuristic CSS — Glassmorphism + Neon Accents
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Root variables ──────────────────────────────────────── */
:root {
    --bg-deep:     #04060e;
    --bg-surface:  rgba(12, 16, 30, 0.85);
    --bg-glass:    rgba(18, 24, 48, 0.55);
    --bg-card:     rgba(22, 30, 58, 0.60);
    --border-glow: rgba(99, 102, 241, 0.25);
    --neon-indigo: #818cf8;
    --neon-purple: #a78bfa;
    --neon-cyan:   #22d3ee;
    --neon-green:  #34d399;
    --neon-red:    #f87171;
    --neon-amber:  #fbbf24;
    --text-primary:   #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted:     #64748b;
    --font-sans:  'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono:  'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
}

/* ── Global ──────────────────────────────────────────────── */
html, body, .stApp {
    background: var(--bg-deep) !important;
    color: var(--text-primary);
    font-family: var(--font-sans);
}

/* Background grid pattern */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }

/* ── Sidebar — Glassmorphism ─────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-glass) !important;
    backdrop-filter: blur(20px) saturate(1.4);
    -webkit-backdrop-filter: blur(20px) saturate(1.4);
    border-right: 1px solid var(--border-glow) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-size: 0.88rem;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--neon-indigo) !important;
}

/* ── Text areas & inputs ─────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    border-radius: 8px;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--neon-indigo) !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.25);
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 0.65rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 0 20px rgba(99,102,241,0.15);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 30px rgba(99,102,241,0.35);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
}

/* ── Download button ─────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    border: none !important;
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 0 20px rgba(16,185,129,0.15);
}
.stDownloadButton > button:hover {
    box-shadow: 0 0 30px rgba(16,185,129,0.35);
}

/* ── Expander ────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-weight: 500;
}
details[data-testid="stExpander"] {
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: 10px;
}

/* ── Metrics ─────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(12px);
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--neon-indigo) !important;
    font-weight: 800;
    font-size: 1.8rem !important;
}

/* ── Custom class: hero banner ───────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg,
        rgba(30, 27, 75, 0.7) 0%,
        rgba(15, 23, 42, 0.8) 50%,
        rgba(30, 27, 75, 0.5) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glow);
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(124,58,237,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-banner h1 {
    font-family: var(--font-sans);
    font-weight: 800;
    font-size: 2.1rem;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
    position: relative;
}
.hero-banner .subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 0.5px;
    position: relative;
}
.hero-banner .badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    color: var(--neon-indigo);
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.6rem;
    position: relative;
}

/* ── Console area ────────────────────────────────────────── */
.console-wrapper {
    background: rgba(8, 10, 20, 0.9);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    line-height: 1.7;
    max-height: 420px;
    overflow-y: auto;
    color: var(--text-secondary);
    backdrop-filter: blur(8px);
}
.console-wrapper .log-new {
    color: var(--neon-cyan);
    font-weight: 500;
}
.console-wrapper .log-ok  { color: var(--neon-green); }
.console-wrapper .log-err { color: var(--neon-red); }
.console-wrapper .log-dim { color: var(--text-muted); font-size: 0.78rem; }
.console-wrapper .log-ts  { color: var(--text-muted); font-size: 0.75rem; margin-right: 6px; }

/* ── Pipeline node cards ─────────────────────────────────── */
.node-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    transition: border-color 0.3s;
}
.node-card.active { border-color: var(--neon-cyan); box-shadow: 0 0 12px rgba(34,211,238,0.15); }
.node-card.done   { border-color: var(--neon-green); }
.node-card .icon   { font-size: 1.3rem; }
.node-card .label  { color: var(--text-primary); font-weight: 600; font-size: 0.9rem; }
.node-card .status { color: var(--text-muted); font-size: 0.78rem; margin-left: auto; }

/* ── Section headers ─────────────────────────────────────── */
.section-header {
    font-family: var(--font-sans);
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--text-primary);
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border-glow);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }

/* ── Status indicators ───────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.idle    { background: var(--text-muted); }
.status-dot.running { background: var(--neon-cyan); box-shadow: 0 0 6px var(--neon-cyan); animation: pulse 1.5s infinite; }
.status-dot.done    { background: var(--neon-green); box-shadow: 0 0 6px var(--neon-green); }
.status-dot.error   { background: var(--neon-red); box-shadow: 0 0 6px var(--neon-red); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Hero Banner
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <h1>🧠 FMCG GenAI Command Center</h1>
    <div class="subtitle">Multi-Agent LangGraph Pipeline &nbsp;·&nbsp; Real-Time Monitoring &nbsp;·&nbsp; PDF Report Generation</div>
    <div class="badge">⚡ Powered by LangGraph + Gemini</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════

with st.expander("📐  Pipeline Architecture", expanded=False):
    st.markdown("""
    ```
    ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
    │ Orchestrator│──▶│  Scraper   │──▶│  Analyst   │──▶│  Assessor  │──▶│   Critic   │
    └────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────┘
                                            ▲                                  │
                                            └────── retry (if failures) ───────┘
    ```
    """)
    arch_img_path = _PROJECT_ROOT / "assets" / "architecture.png"
    if arch_img_path.exists():
        st.image(str(arch_img_path), caption="Pipeline Architecture Diagram")
    else:
        st.caption("💡 Place an image at `assets/architecture.png` to display here.")


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — Configuration
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown(
        f'<p style="font-size:0.8rem;color:#94a3b8;">'
        f'Max Retries: <code style="color:#818cf8;">{MAX_RETRIES}</code> &nbsp;|&nbsp; '
        f'Output: <code style="color:#818cf8;">{OUTPUT_DIR}</code></p>',
        unsafe_allow_html=True,
    )
    st.divider()

    custom_query = st.text_area(
        "Research Query",
        value=(
            "Identify the top 5 emerging GenAI use cases in FMCG supply chains. "
            "For each use case, describe implementation approach, expected impact, "
            "risks, and maturity level."
        ),
        height=120,
    )

    pdf_filename = st.text_input("PDF Filename", value="output_report.pdf")

    st.divider()
    st.markdown(
        '<p style="color:#475569;font-size:0.7rem;text-align:center;letter-spacing:1px;">'
        'BUILT WITH LANGGRAPH + STREAMLIT + FPDF2</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — verdict extraction & use-case card rendering
# ═══════════════════════════════════════════════════════════════════════════

def _get_verdict(uc) -> str:
    """Safely extract critic_verdict as a plain string (handles enums, dicts, strings)."""
    raw = getattr(uc, "critic_verdict", None) if hasattr(uc, "critic_verdict") else None
    if raw is None and isinstance(uc, dict):
        raw = uc.get("critic_verdict")
    if raw is None:
        return ""
    return raw.value if hasattr(raw, "value") else str(raw)


def _render_use_case_cards(use_cases: list, header: str = "📋 Results") -> None:
    """Render expandable cards for each use case (DRY helper)."""
    st.markdown(f'<div class="section-header">{header}</div>', unsafe_allow_html=True)

    for i, uc in enumerate(use_cases, 1):
        topic = getattr(uc, "topic", "Untitled") if hasattr(uc, "topic") else uc.get("topic", "Untitled")
        maturity_raw = getattr(uc, "maturity_level", "N/A") if hasattr(uc, "maturity_level") else "N/A"
        maturity = maturity_raw.value if hasattr(maturity_raw, "value") else str(maturity_raw)
        verdict = _get_verdict(uc)
        verdict_icon = "✅" if verdict == "pass" else ("❌" if verdict == "fail" else "⏳")

        with st.expander(f"**{i}. {topic}** — `{maturity}` {verdict_icon}", expanded=False):
            desc = getattr(uc, "description", "") if hasattr(uc, "description") else ""
            if desc:
                st.markdown(f"**Description:** {desc}")

            impl = getattr(uc, "implementation_approach", "") if hasattr(uc, "implementation_approach") else ""
            if impl:
                st.markdown(f"**Implementation:** {impl}")

            ei = getattr(uc, "economic_impact", None) if hasattr(uc, "economic_impact") else None
            if ei:
                roi = getattr(ei, "estimated_roi_percentage", "N/A")
                cost_raw = getattr(ei, "implementation_cost_complexity", "N/A")
                cost = cost_raw.value if hasattr(cost_raw, "value") else str(cost_raw)
                st.markdown(f"**ROI:** {roi}% &nbsp;|&nbsp; **Cost Complexity:** {cost}")

            ra = getattr(uc, "risk_assessment", None) if hasattr(uc, "risk_assessment") else None
            if ra:
                st.markdown(f"**Bottleneck:** {getattr(ra, 'primary_bottleneck', 'N/A')}")
                st.markdown(f"**Privacy:** {getattr(ra, 'data_privacy_concerns', 'N/A')}")

            feedback = getattr(uc, "critic_feedback", []) if hasattr(uc, "critic_feedback") else []
            if feedback:
                st.markdown("**Critic Feedback:**")
                for j, fb in enumerate(feedback, 1):
                    st.caption(f"Round {j}: {fb}")


# ═══════════════════════════════════════════════════════════════════════════
# Node metadata
# ═══════════════════════════════════════════════════════════════════════════

_NODE_META: dict[str, tuple[str, str, str]] = {
    "orchestrator": ("🎯", "Orchestrator", "Decomposing query into supply-chain sub-domains"),
    "scraper":      ("🔍", "Market Scraper", "Collecting web evidence for GenAI use cases"),
    "analyst":      ("📊", "Economic Analyst", "Synthesising evidence into candidate use cases"),
    "assessor":     ("⚠️",  "Risk Assessor", "Evaluating risks and integration complexity"),
    "critic":       ("🛡️", "Red Team Critic", "Adversarial validation of each use case"),
}


# ═══════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ═══════════════════════════════════════════════════════════════════════════

if "logs" not in st.session_state:
    st.session_state.logs = []
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "use_cases" not in st.session_state:
    st.session_state.use_cases = []


# ═══════════════════════════════════════════════════════════════════════════
# Layout: metrics + run button row
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🚀 Pipeline Control</div>', unsafe_allow_html=True)

ctrl_cols = st.columns([2, 1, 1, 1, 1])

with ctrl_cols[0]:
    run_clicked = st.button("▶  Run Research Pipeline", use_container_width=True, type="primary")

# Metric placeholders
metric_total = ctrl_cols[1].empty()
metric_passed = ctrl_cols[2].empty()
metric_retries = ctrl_cols[3].empty()
metric_errors = ctrl_cols[4].empty()

# Show previous-run metrics if available
if st.session_state.pipeline_done and st.session_state.use_cases:
    _uc = st.session_state.use_cases
    _fs = st.session_state.final_state or {}
    metric_total.metric("Use Cases", len(_uc))
    _pc = sum(1 for c in _uc if _get_verdict(c) == "pass")
    metric_passed.metric("Passed", f"{_pc}/{len(_uc)}")
    metric_retries.metric("Retries", _fs.get("error_count", 0))
    metric_errors.metric("Errors", len(_fs.get("errors", [])))


# ═══════════════════════════════════════════════════════════════════════════
# Console + status
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📟 Live Console</div>', unsafe_allow_html=True)

status_placeholder = st.empty()
console_placeholder = st.empty()


def _render_console(logs: list[str]) -> None:
    """Overwrite the console container with the full log list (newest first)."""
    if not logs:
        html = '<div class="console-wrapper"><span class="log-dim">Waiting for pipeline...</span></div>'
    else:
        lines = "\n".join(logs)
        html = f'<div class="console-wrapper">{lines}</div>'
    console_placeholder.markdown(html, unsafe_allow_html=True)


def _log(msg: str, css_class: str = "") -> None:
    """Prepend a timestamped log entry to session_state.logs and re-render."""
    ts = datetime.now().strftime("%H:%M:%S")
    cls = f' class="{css_class}"' if css_class else ""
    entry = f'<div{cls}><span class="log-ts">[{ts}]</span> {msg}</div>'
    st.session_state.logs.insert(0, entry)
    _render_console(st.session_state.logs)


# Render existing logs on rerun
_render_console(st.session_state.logs)


# ═══════════════════════════════════════════════════════════════════════════
# Results area (for post-run display)
# ═══════════════════════════════════════════════════════════════════════════

results_area = st.container()


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Execution
# ═══════════════════════════════════════════════════════════════════════════

if run_clicked:
    # Reset state
    st.session_state.logs = []
    st.session_state.pipeline_done = False
    st.session_state.final_state = None
    st.session_state.use_cases = []

    # Late import to avoid circular imports at module load
    from src.graph.builder import build_graph
    from src.utils.pdf_generator import generate_fmcg_report

    initial_state = {
        "original_query": custom_query,
        "target_supply_chain_nodes": [],
        "raw_evidence": [],
        "candidate_use_cases": [],
        "final_top_5": [],
        "error_count": 0,
        "errors": [],
    }

    # ── Build graph with spinner ──────────────────────────────────────
    status_placeholder.markdown(
        '<div><span class="status-dot running"></span> Building graph...</div>',
        unsafe_allow_html=True,
    )
    _log("⚙️ Building LangGraph state machine...", "log-new")

    try:
        with st.spinner("Compiling pipeline graph..."):
            graph = build_graph()
        _log("✅ Graph compiled successfully", "log-ok")
        _file_logger.info("Graph compiled — starting pipeline stream")
        _log("🚀 <b>Starting pipeline stream...</b>", "log-new")

        status_placeholder.markdown(
            '<div><span class="status-dot running"></span> Pipeline running...</div>',
            unsafe_allow_html=True,
        )

        # ── Stream the graph ──────────────────────────────────────────
        final_state: dict = dict(initial_state)

        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue

                emoji, label, desc = _NODE_META.get(
                    node_name, ("🔧", node_name.title(), "Processing...")
                )

                _log(f"{emoji} <b>{label}</b> — {desc}", "log-new")

                # Node-specific detail logging
                if node_name == "orchestrator":
                    nodes = node_output.get("target_supply_chain_nodes", [])
                    if nodes:
                        _log(f"&nbsp;&nbsp;&nbsp;&nbsp;📋 Identified <b>{len(nodes)}</b> supply-chain sub-domains")

                elif node_name == "scraper":
                    evidence = node_output.get("raw_evidence", [])
                    _log(f"&nbsp;&nbsp;&nbsp;&nbsp;📄 Collected <b>{len(evidence)}</b> evidence blocks")

                elif node_name == "analyst":
                    cases = node_output.get("candidate_use_cases", [])
                    _log(f"&nbsp;&nbsp;&nbsp;&nbsp;📊 Generated <b>{len(cases)}</b> candidate use cases")

                elif node_name == "assessor":
                    cases = node_output.get("candidate_use_cases", [])
                    _log(f"&nbsp;&nbsp;&nbsp;&nbsp;⚠️ Risk-assessed <b>{len(cases)}</b> use cases")

                elif node_name == "critic":
                    cases = node_output.get("candidate_use_cases", [])
                    final = node_output.get("final_top_5", [])
                    err_count = node_output.get("error_count", 0)

                    passed = sum(1 for c in cases if _get_verdict(c) == "pass")
                    failed = len(cases) - passed

                    _log(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;✅ Passed: <b>{passed}</b> &nbsp;|&nbsp; "
                        f"❌ Failed: <b>{failed}</b> &nbsp;|&nbsp; "
                        f"🔄 Error count: <b>{err_count}</b>"
                    )

                    if final:
                        _log("🏁 <b>Critic promoted candidates to final_top_5 — pipeline complete!</b>", "log-ok")
                    elif err_count < MAX_RETRIES:
                        _log(
                            f"🔄 Retrying... routing back to Analyst "
                            f"(attempt {err_count + 1}/{MAX_RETRIES})",
                            "log-new",
                        )

                # Merge into running state
                final_state.update(node_output)

        # ── Pipeline complete ─────────────────────────────────────────
        _log("━" * 50, "log-dim")
        _log("✅ <b>Pipeline execution complete!</b>", "log-ok")

        use_cases = final_state.get("final_top_5", [])
        if not use_cases:
            use_cases = final_state.get("candidate_use_cases", [])

        _log(f"📦 Final use cases: <b>{len(use_cases)}</b>")

        # Save to session state
        st.session_state.final_state = final_state
        st.session_state.use_cases = use_cases
        st.session_state.pipeline_done = True

        status_placeholder.markdown(
            f'<div><span class="status-dot done"></span> '
            f'Pipeline complete — <b>{len(use_cases)}</b> use cases</div>',
            unsafe_allow_html=True,
        )

        # Update metrics
        metric_total.metric("Use Cases", len(use_cases))
        passed_count = sum(1 for c in use_cases if _get_verdict(c) == "pass")
        metric_passed.metric("Passed", f"{passed_count}/{len(use_cases)}")
        metric_retries.metric("Retries", final_state.get("error_count", 0))
        metric_errors.metric("Errors", len(final_state.get("errors", [])))

        # ── Save JSON report (same schema as main.py) ─────────────────
        _ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        _json_path = OUTPUT_DIR / f"report_{_ts}.json"

        _serialised = []
        for _uc_ser in use_cases:
            if hasattr(_uc_ser, "model_dump"):
                _serialised.append(_uc_ser.model_dump(mode="json"))
            elif isinstance(_uc_ser, dict):
                _serialised.append(_uc_ser)
            else:
                _serialised.append(str(_uc_ser))

        _all_passed = all(
            _get_verdict(c) == "pass" for c in use_cases
        ) if use_cases else False

        _report = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "topic": "Emerging GenAI Use Cases in FMCG Supply Chains",
                "total_use_cases": len(_serialised),
                "validation_status": "all_passed" if _all_passed else "partial",
                "error_count": final_state.get("error_count", 0),
                "errors": final_state.get("errors", []),
            },
            "use_cases": _serialised,
        }

        _json_path.write_text(
            json.dumps(_report, indent=2, default=str), encoding="utf-8"
        )
        _log(f"📋 JSON report saved to <code>{_json_path.name}</code>", "log-ok")
        _file_logger.info("JSON report saved: %s", _json_path)

        # ── Display Results ───────────────────────────────────────────
        if use_cases:
            with results_area:
                _render_use_case_cards(use_cases)

                # ── Generate PDF ──────────────────────────────────────
                st.markdown('<div class="section-header">📄 PDF Report</div>', unsafe_allow_html=True)

                _log("📄 Generating PDF report...", "log-new")

                try:
                    pdf_path = generate_fmcg_report(use_cases, filename=pdf_filename)
                    _log(f"✅ PDF saved to <code>{pdf_path}</code>", "log-ok")

                    pdf_bytes = Path(pdf_path).read_bytes()

                    col_dl, col_view = st.columns([1, 2])
                    with col_dl:
                        st.download_button(
                            label="⬇️  Download PDF Report",
                            data=pdf_bytes,
                            file_name=pdf_filename,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.caption(f"Size: {len(pdf_bytes) / 1024:.1f} KB")

                    with col_view:
                        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                        pdf_display = (
                            f'<iframe src="data:application/pdf;base64,{b64_pdf}" '
                            f'width="100%" height="600" style="border:1px solid rgba(99,102,241,0.2);'
                            f'border-radius:10px;" type="application/pdf"></iframe>'
                        )
                        st.markdown(pdf_display, unsafe_allow_html=True)

                except Exception as pdf_err:
                    _log(f"❌ PDF generation failed: {pdf_err}", "log-err")
                    st.error(f"PDF generation failed: {pdf_err}")

        else:
            status_placeholder.markdown(
                '<div><span class="status-dot error"></span> Pipeline completed but no use cases were generated.</div>',
                unsafe_allow_html=True,
            )

    except NotImplementedError as e:
        _log(f"⚙️ Agent not yet implemented: {e}", "log-err")
        _file_logger.error("Agent not yet implemented: %s", e)
        status_placeholder.markdown(
            '<div><span class="status-dot error"></span> Agent not yet implemented — '
            'project foundation is set up and ready for implementation.</div>',
            unsafe_allow_html=True,
        )
        st.info(
            "This is expected if agent logic has not been written yet. "
            "The project foundation is set up and ready for implementation."
        )

    except Exception as e:
        _log(f"❌ <b>Pipeline error:</b> {e}", "log-err")
        _file_logger.error("Pipeline failed: %s", e, exc_info=True)
        status_placeholder.markdown(
            f'<div><span class="status-dot error"></span> Pipeline failed: {e}</div>',
            unsafe_allow_html=True,
        )
        st.exception(e)

# ── Show previous results on rerun (if pipeline already done) ─────────
elif st.session_state.pipeline_done and st.session_state.use_cases:
    use_cases = st.session_state.use_cases
    final_state = st.session_state.final_state or {}

    with results_area:
        _render_use_case_cards(use_cases, header="📋 Results (Previous Run)")

        # PDF download from previous run
        prev_pdf = OUTPUT_DIR / pdf_filename
        if prev_pdf.exists():
            st.markdown('<div class="section-header">📄 PDF Report</div>', unsafe_allow_html=True)
            pdf_bytes = prev_pdf.read_bytes()
            col_dl, col_view = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    label="⬇️  Download PDF Report",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    use_container_width=True,
                )
            with col_view:
                b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64_pdf}" '
                    f'width="100%" height="600" style="border:1px solid rgba(99,102,241,0.2);'
                    f'border-radius:10px;" type="application/pdf"></iframe>',
                    unsafe_allow_html=True,
                )
