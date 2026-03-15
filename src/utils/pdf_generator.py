"""
PDF Report Generator
====================

Converts the final list of ``UseCase`` Pydantic models into a professional
PDF report using fpdf2.

Usage::

    from src.utils.pdf_generator import generate_fmcg_report
    filepath = generate_fmcg_report(use_cases, filename="output_report.pdf")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpdf import FPDF

from config.settings import OUTPUT_DIR

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Color Palette  (RGB tuples)
# ═══════════════════════════════════════════════════════════════════════════

_CLR_HEADER_BG = (15, 23, 42)        # Slate-900
_CLR_HEADER_FG = (255, 255, 255)     # White
_CLR_SECTION_BG = (30, 41, 59)       # Slate-800
_CLR_SECTION_FG = (226, 232, 240)    # Slate-200
_CLR_ACCENT = (99, 102, 241)         # Indigo-500
_CLR_PASS = (34, 197, 94)            # Green-500
_CLR_FAIL = (239, 68, 68)            # Red-500
_CLR_BODY = (51, 65, 85)             # Slate-700
_CLR_LIGHT = (148, 163, 184)         # Slate-400
_CLR_WHITE = (255, 255, 255)
_CLR_BG_CARD = (248, 250, 252)       # Slate-50


class _ReportPDF(FPDF):
    """Custom FPDF subclass with header/footer branding."""

    def header(self):
        if self.page_no() == 1:
            return  # Title page has its own header
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_CLR_LIGHT)
        self.cell(0, 8, "FMCG GenAI Supply Chain Report", align="L")
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_CLR_LIGHT)
        self.cell(0, 10, f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", align="C")


# ═══════════════════════════════════════════════════════════════════════════
# Helper: safe attribute access (works with UseCase objects AND dicts)
# ═══════════════════════════════════════════════════════════════════════════

def _get(obj: Any, key: str, default: Any = "") -> Any:
    """Get attribute or dict key, falling back to default."""
    if hasattr(obj, key):
        val = getattr(obj, key)
        return val if val is not None else default
    if isinstance(obj, dict):
        val = obj.get(key)
        return val if val is not None else default
    return default


def _get_nested(obj: Any, outer: str, inner: str, default: Any = "N/A") -> Any:
    """Get a nested field like economic_impact.estimated_roi_percentage."""
    parent = _get(obj, outer)
    if parent is None or parent == "":
        return default
    return _get(parent, inner, default)


def _safe(text: str) -> str:
    """Sanitise text for fpdf2's built-in Helvetica (latin-1 charset).

    Replaces common Unicode characters with ASCII equivalents, then
    encodes to latin-1 with replacement for anything left over.
    """
    replacements = {
        "\u2014": "--",   # em-dash
        "\u2013": "-",    # en-dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "*",    # bullet
        "\u2026": "...",  # ellipsis
        "\u2192": "->",   # right arrow
        "\u2190": "<-",   # left arrow
        "\u2500": "-",    # box drawing horizontal
        "\u2550": "=",    # box drawing double horizontal
        "\u26a1": "[!]",  # lightning
        "\u2705": "[v]",  # check mark
        "\u274c": "[x]",  # cross mark
    }
    for uni, ascii_rep in replacements.items():
        text = text.replace(uni, ascii_rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def generate_fmcg_report(
    use_cases: list,
    filename: str = "output_report.pdf",
) -> str:
    """
    Generate a professional PDF report from a list of UseCase objects.

    Args:
        use_cases: List of UseCase Pydantic models (or dicts).
        filename: Output filename (saved into ``OUTPUT_DIR``).

    Returns:
        Absolute filepath of the generated PDF as a string.
    """
    filepath = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title Page ────────────────────────────────────────────────────
    _build_title_page(pdf, len(use_cases))

    # ── Use Case Pages ────────────────────────────────────────────────
    for i, uc in enumerate(use_cases, 1):
        _build_use_case_page(pdf, uc, i, len(use_cases))

    # ── Save ──────────────────────────────────────────────────────────
    pdf.output(str(filepath))
    logger.info("PDF report generated: %s", filepath)
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# Page Builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_title_page(pdf: _ReportPDF, count: int) -> None:
    """Render the title / cover page."""
    pdf.add_page()

    # Dark header band
    pdf.set_fill_color(*_CLR_HEADER_BG)
    pdf.rect(0, 0, 210, 100, "F")

    # Title
    pdf.set_y(25)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*_CLR_WHITE)
    pdf.cell(0, 14, "FMCG GenAI", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 14, "Supply Chain Report", align="C", new_x="LMARGIN", new_y="NEXT")

    # Subtitle
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*_CLR_SECTION_FG)
    pdf.cell(
        0, 8,
        _safe("Multi-Agent Research Pipeline -- Validated Use Cases"),
        align="C", new_x="LMARGIN", new_y="NEXT",
    )

    # Metadata bar
    pdf.set_y(105)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_CLR_BODY)
    timestamp = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    pdf.cell(0, 8, f"Generated: {timestamp}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Total Use Cases: {count}", align="C", new_x="LMARGIN", new_y="NEXT")

    # Accent line
    pdf.set_draw_color(*_CLR_ACCENT)
    pdf.set_line_width(0.8)
    pdf.line(60, pdf.get_y() + 4, 150, pdf.get_y() + 4)


def _build_use_case_page(pdf: _ReportPDF, uc: Any, index: int, total: int) -> None:
    """Render a single use-case section."""
    pdf.add_page()

    topic = _get(uc, "topic", "Untitled Use Case")

    # ── Header band ───────────────────────────────────────────────────
    pdf.set_fill_color(*_CLR_HEADER_BG)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_y(8)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*_CLR_WHITE)
    pdf.cell(0, 10, f"Use Case {index}/{total}", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Topic title ───────────────────────────────────────────────────
    pdf.set_y(35)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*_CLR_HEADER_BG)
    pdf.multi_cell(0, 8, _safe(topic), align="L")
    pdf.ln(2)

    # ── Segment & Maturity ────────────────────────────────────────────
    segment = _safe(str(_get(uc, "supply_chain_segment", "N/A")))
    maturity_raw = _get(uc, "maturity_level", "Theoretical")
    maturity = _safe(maturity_raw.value if hasattr(maturity_raw, "value") else str(maturity_raw))

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_CLR_LIGHT)
    pdf.cell(95, 6, f"Segment: {segment}", new_x="RIGHT")
    pdf.cell(95, 6, f"Maturity: {maturity}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Description ───────────────────────────────────────────────────
    _section_heading(pdf, "Description")
    desc = _get(uc, "description", "No description provided.")
    _body_text(pdf, desc)

    # ── Implementation Approach ───────────────────────────────────────
    impl = _get(uc, "implementation_approach", "")
    if impl:
        _section_heading(pdf, "Implementation Approach")
        _body_text(pdf, impl)

    # ── Economic Impact ───────────────────────────────────────────────
    roi = _get_nested(uc, "economic_impact", "estimated_roi_percentage", "N/A")
    efficiency = _get_nested(uc, "economic_impact", "marginal_efficiency_gain_description", "N/A")
    cost_raw = _get_nested(uc, "economic_impact", "implementation_cost_complexity", "N/A")
    cost = cost_raw.value if hasattr(cost_raw, "value") else str(cost_raw)

    _section_heading(pdf, "Economic Impact")
    _kv_row(pdf, "Estimated ROI", f"{roi}%" if roi != "N/A" else "N/A")
    _kv_row(pdf, "Cost Complexity", cost)
    pdf.ln(1)
    _body_text(pdf, f"Efficiency Gains: {efficiency}")

    # ── Risk Assessment ───────────────────────────────────────────────
    bottleneck = _get_nested(uc, "risk_assessment", "primary_bottleneck", "N/A")
    privacy = _get_nested(uc, "risk_assessment", "data_privacy_concerns", "N/A")
    integration = _get_nested(uc, "risk_assessment", "integration_complexity", "N/A")

    _section_heading(pdf, "Risk Assessment")
    _kv_row(pdf, "Primary Bottleneck", str(bottleneck))
    _kv_row(pdf, "Data Privacy", str(privacy))
    _kv_row(pdf, "Integration Complexity", str(integration))

    # ── Critic Feedback ───────────────────────────────────────────────
    feedback_list = _get(uc, "critic_feedback", [])
    verdict_raw = _get(uc, "critic_verdict", "needs_revision")
    verdict = verdict_raw.value if hasattr(verdict_raw, "value") else str(verdict_raw)

    _section_heading(pdf, "Red Team Critic Feedback")

    # Verdict badge
    pdf.set_font("Helvetica", "B", 10)
    if verdict == "pass":
        pdf.set_text_color(*_CLR_PASS)
        pdf.cell(0, 6, f"Verdict: PASS", new_x="LMARGIN", new_y="NEXT")
    elif verdict == "fail":
        pdf.set_text_color(*_CLR_FAIL)
        pdf.cell(0, 6, f"Verdict: FAIL", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_text_color(*_CLR_LIGHT)
        pdf.cell(0, 6, f"Verdict: {verdict.upper()}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(1)
    if feedback_list:
        for j, fb in enumerate(feedback_list, 1):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(*_CLR_BODY)
            pdf.multi_cell(0, 5, _safe(f"Round {j}: {fb}"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
    else:
        _body_text(pdf, "No feedback recorded.")

    # ── Evidence Sources ──────────────────────────────────────────────
    sources = _get(uc, "evidence_sources", [])
    if sources:
        _section_heading(pdf, "Evidence Sources")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*_CLR_ACCENT)
        for src in sources:
            pdf.multi_cell(0, 4, _safe(f"* {src}"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Formatting Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _section_heading(pdf: _ReportPDF, title: str) -> None:
    """Render a section heading with accent underline."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*_CLR_HEADER_BG)
    pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*_CLR_ACCENT)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 40, pdf.get_y())
    pdf.ln(3)


def _body_text(pdf: _ReportPDF, text: str) -> None:
    """Render body text in the standard style."""
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_CLR_BODY)
    pdf.multi_cell(0, 5, _safe(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _kv_row(pdf: _ReportPDF, key: str, value: str) -> None:
    """Render a bold key : normal value row."""
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*_CLR_BODY)
    safe_key = _safe(f"{key}: ")
    key_width = pdf.get_string_width(safe_key) + 2
    pdf.cell(key_width, 6, safe_key)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _safe(value), new_x="LMARGIN", new_y="NEXT")
