"""
Risk Assessor Agent
===================

Takes the partially-filled ``candidate_use_cases`` from the Economic Analyst
and enriches each one with:
    - ``risk_assessment``: A ``RiskAssessment`` covering primary bottlenecks,
      data privacy concerns, and legacy-system integration complexity.
    - ``implementation_approach``: A realistic 3–5 sentence narrative covering
      tech stack, data requirements, integration points, and rollout phases.

Design Rationale:
    - Processes use cases **one at a time** so the LLM can focus its full
      context on each use case, producing deeper, more specific risk analysis.
    - Uses ``with_structured_output()`` for each use case, with a manual-
      parse fallback if structured output fails.
    - The prompt forces a cynical Enterprise Solutions Architect persona —
      no optimistic hand-waving, every risk must be specific and actionable.
    - Per-use-case error isolation: if enrichment fails for one use case,
      it keeps the Analyst's original data and continues to the next.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from config.settings import get_llm
from src.state.graph_state import (
    AgentState,
    GraphState,
    RiskAssessment,
    UseCase,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Environment Setup
# ═══════════════════════════════════════════════════════════════════════════

# Environment setup handled centrally by config.settings.get_llm()


# ═══════════════════════════════════════════════════════════════════════════
# Structured Output Schema  (per-use-case enrichment)
# ═══════════════════════════════════════════════════════════════════════════

class RiskEnrichment(BaseModel):
    """
    Structured output schema for the Risk Assessor's per-use-case analysis.

    Keeps ``RiskAssessment`` fields flat for simpler LLM output, then maps
    back to the nested Pydantic model.
    """

    implementation_approach: str = Field(
        ...,
        description=(
            "3–5 sentence narrative: specific tech stack (e.g., LangGraph, "
            "SAP BTP, Azure OpenAI), data requirements, integration points "
            "with legacy systems, and phased rollout strategy (POC → Pilot → "
            "Production). Must be realistic for FMCG enterprises."
        ),
    )
    primary_bottleneck: str = Field(
        ...,
        description=(
            "The single biggest technical bottleneck blocking adoption. "
            "Must be specific to this use case and FMCG context — not generic "
            "statements like 'data quality issues'."
        ),
    )
    data_privacy_concerns: str = Field(
        ...,
        description=(
            "Data-privacy and regulatory risks specific to this use case. "
            "Reference applicable regulations (GDPR, India's DPDP Act 2023, "
            "CCPA) and specific data categories at risk (e.g., supplier "
            "pricing data, employee PII, proprietary formulations)."
        ),
    )
    integration_complexity: str = Field(
        ...,
        description=(
            "Legacy-system integration burden: which enterprise systems "
            "(SAP S/4HANA, Oracle EBS, custom MES/SCADA, WMS) must be "
            "connected, what middleware/API layers are required, and "
            "estimated integration effort."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Constants
# ═══════════════════════════════════════════════════════════════════════════

ASSESSOR_SYSTEM_PROMPT = """\
You are a cynical, battle-hardened Enterprise Solutions Architect with 20+ \
years of experience deploying technology solutions in large FMCG / CPG \
enterprises (Unilever, Nestlé, P&G, Godrej, ITC).

You have seen countless "transformational" IT projects fail due to \
underestimated integration complexity, ignored data governance, and \
optimistic vendor promises. Your job is to be the VOICE OF REALITY.

═══════════════════════════════════════════════════════════════════
YOUR EVALUATION FRAMEWORK:
═══════════════════════════════════════════════════════════════════

For the given GenAI use case, you must provide:

1. **implementation_approach** (3–5 sentences):
   - Specify the EXACT tech stack (e.g., "LangGraph for agent orchestration, \
     Azure OpenAI GPT-4o for LLM, SAP Ariba APIs for procurement data")
   - Define data requirements (what data, from which systems, in what format)
   - Identify integration touchpoints with legacy systems
   - Outline a realistic phased rollout: POC (3 months) → Pilot (6 months) \
     → Production (12+ months)

2. **primary_bottleneck**:
   - The SINGLE biggest technical obstacle. Be BRUTALLY specific.
   - Bad example: "Data quality issues"
   - Good example: "Lack of structured, machine-readable vendor performance \
     data — most supplier evaluations exist as unstructured PDF reports \
     across 4 different regional procurement teams using different ERP modules"

3. **data_privacy_concerns**:
   - Reference SPECIFIC regulations (GDPR Article 22 for automated \
     decision-making, India's DPDP Act 2023 Section 8 for data principals, \
     CCPA if US-facing)
   - Identify SPECIFIC data categories at risk (supplier pricing = \
     commercially sensitive, employee data, proprietary formulations)

4. **integration_complexity**:
   - Name SPECIFIC systems (SAP S/4HANA, Oracle EBS, Manhattan WMS, \
     Blue Yonder, custom MES/SCADA)
   - Identify middleware requirements (MuleSoft, Dell Boomi, SAP BTP)
   - Estimate integration effort realistically

═══════════════════════════════════════════════════════════════════
CONSTRAINTS:
═══════════════════════════════════════════════════════════════════

- Do NOT be optimistic. Assume the worst-case legacy environment.
- Every statement must be actionable — a CTO should be able to create a \
  project plan from your assessment.
- Do NOT use filler phrases like "careful planning required" or \
  "thorough testing needed" — those are meaningless.
"""

ASSESSOR_USER_PROMPT = """\
═══════════════════════════════════════════════════════════════════
USE CASE TO ASSESS:
═══════════════════════════════════════════════════════════════════

Topic: {topic}
Supply Chain Segment: {segment}
Description: {description}
Maturity Level: {maturity}
Estimated ROI: {roi}%
Efficiency Gains: {efficiency}

═══════════════════════════════════════════════════════════════════
EVIDENCE CONTEXT:
═══════════════════════════════════════════════════════════════════

{evidence_text}

═══════════════════════════════════════════════════════════════════
Provide your implementation_approach, primary_bottleneck, \
data_privacy_concerns, and integration_complexity assessment.
═══════════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════════════════
# LLM Factory
# ═══════════════════════════════════════════════════════════════════════════

def _get_llm():
    """Get the centralised LLM instance for the Risk Assessor."""
    return get_llm(max_output_tokens=4096)


# ═══════════════════════════════════════════════════════════════════════════
# Node Implementation
# ═══════════════════════════════════════════════════════════════════════════

def run_risk_assessor(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node: enrich each candidate UseCase with risk assessment and
    implementation approach.

    Iterates over ``candidate_use_cases``, invoking the LLM once per use case
    with structured output. On failure, preserves the Analyst's original data.

    Args:
        state: Current LangGraph shared state.

    Returns:
        Dict updating ``candidate_use_cases`` and ``current_agent``.
    """
    candidates: list[UseCase] = state.get("candidate_use_cases", [])
    raw_evidence: list[str] = state.get("raw_evidence", [])
    existing_errors: list[str] = state.get("errors", [])

    if not candidates:
        logger.warning("Risk Assessor: no candidate use cases to assess")
        return {
            "candidate_use_cases": [],
            "current_agent": AgentState.RISK_ASSESSOR,
            "errors": existing_errors + ["Risk Assessor: no candidates provided"],
        }

    logger.info(
        "Risk Assessor: enriching %d candidate use cases with risk data",
        len(candidates),
    )

    llm = _get_llm()
    evidence_text = "\n\n".join(raw_evidence) if raw_evidence else "[No evidence available]"

    # Truncate for context safety
    if len(evidence_text) > 40_000:
        evidence_text = evidence_text[:40_000] + "\n\n[... truncated]"

    enriched: list[UseCase] = []
    new_errors: list[str] = []

    for i, use_case in enumerate(candidates, 1):
        logger.info(
            "Risk Assessor [%d/%d]: assessing '%s'",
            i, len(candidates), use_case.topic,
        )

        try:
            enriched_uc = _enrich_single_use_case(
                use_case=use_case,
                evidence_text=evidence_text,
                llm=llm,
            )
            enriched.append(enriched_uc)
            logger.info(
                "Risk Assessor [%d/%d]: successfully enriched '%s'",
                i, len(candidates), use_case.topic,
            )

        except Exception as exc:
            msg = (
                f"Risk Assessor: failed to enrich '{use_case.topic}': {exc}. "
                f"Keeping original Analyst data."
            )
            logger.error(msg, exc_info=True)
            new_errors.append(msg)
            # Preserve the Analyst's original data
            enriched.append(use_case)

        # Free tier rate limit protection (15 RPM)
        if i < len(candidates):
            logger.debug("Risk Assessor: sleeping 5s to respect API rate limits...")
            time.sleep(5)

    logger.info(
        "Risk Assessor: completed — %d/%d use cases enriched",
        len(enriched) - len(new_errors), len(candidates),
    )

    return {
        "candidate_use_cases": enriched,
        "current_agent": AgentState.RISK_ASSESSOR,
        "errors": existing_errors + new_errors if new_errors else existing_errors,
    }


def _enrich_single_use_case(
    use_case: UseCase,
    evidence_text: str,
    llm,
) -> UseCase:
    """
    Enrich a single UseCase with risk assessment and implementation approach.

    Strategy:
      1. Try ``with_structured_output(RiskEnrichment)``.
      2. If that fails, fall back to raw JSON parsing.
      3. Map the enrichment into an updated UseCase.

    Args:
        use_case: The candidate UseCase from the Analyst.
        evidence_text: Aggregated raw evidence for context.
        llm: Pre-configured Gemini LLM instance.

    Returns:
        Enriched UseCase with risk_assessment and implementation_approach.
    """
    # ── Build use-case-specific context ───────────────────────────────
    roi_str = (
        str(use_case.economic_impact.estimated_roi_percentage)
        if use_case.economic_impact
        else "N/A"
    )
    efficiency_str = (
        use_case.economic_impact.marginal_efficiency_gain_description
        if use_case.economic_impact
        else "N/A"
    )

    messages = [
        SystemMessage(content=ASSESSOR_SYSTEM_PROMPT),
        HumanMessage(
            content=ASSESSOR_USER_PROMPT.format(
                topic=use_case.topic,
                segment=use_case.supply_chain_segment,
                description=use_case.description,
                maturity=use_case.maturity_level.value if use_case.maturity_level else "N/A",
                roi=roi_str,
                efficiency=efficiency_str,
                evidence_text=evidence_text,
            )
        ),
    ]

    # ── Attempt 1: Structured output ──────────────────────────────────
    enrichment: RiskEnrichment | None = None

    try:
        structured_llm = llm.with_structured_output(RiskEnrichment)
        enrichment = structured_llm.invoke(messages)
        logger.debug("Risk Assessor: structured output succeeded for '%s'", use_case.topic)
    except (ValidationError, Exception) as exc:
        logger.warning(
            "Risk Assessor: structured output failed for '%s' (%s) — "
            "trying raw JSON fallback",
            use_case.topic, exc,
        )

    # ── Attempt 2: Raw JSON fallback ──────────────────────────────────
    if enrichment is None:
        enrichment = _fallback_parse_enrichment(llm, messages, use_case.topic)

    if enrichment is None:
        raise RuntimeError(f"Both structured and fallback parsing failed for '{use_case.topic}'")

    # ── Map enrichment back to UseCase ────────────────────────────────
    # Ensure min_length constraints are met
    bottleneck = enrichment.primary_bottleneck
    if len(bottleneck) < 10:
        bottleneck = bottleneck.ljust(10)

    privacy = enrichment.data_privacy_concerns
    if len(privacy) < 10:
        privacy = privacy.ljust(10)

    integration = enrichment.integration_complexity
    if len(integration) < 10:
        integration = integration.ljust(10)

    updated = use_case.model_copy(
        update={
            "implementation_approach": enrichment.implementation_approach,
            "risk_assessment": RiskAssessment(
                primary_bottleneck=bottleneck,
                data_privacy_concerns=privacy,
                integration_complexity=integration,
            ),
        }
    )

    return updated


def _fallback_parse_enrichment(
    llm,
    messages: list,
    topic: str,
) -> RiskEnrichment | None:
    """
    Fallback: invoke the LLM without structured output and manually parse
    the JSON response into a RiskEnrichment.
    """
    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]).strip()

        data = json.loads(raw)
        return RiskEnrichment.model_validate(data)

    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error(
            "Risk Assessor: fallback JSON parsing failed for '%s': %s",
            topic, exc,
        )
        return None
    except Exception as exc:
        logger.error(
            "Risk Assessor: fallback LLM call failed for '%s': %s",
            topic, exc,
        )
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible alias
# ═══════════════════════════════════════════════════════════════════════════
risk_assessor_node = run_risk_assessor
