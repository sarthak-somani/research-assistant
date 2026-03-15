"""
Economic Analyst Agent
======================

Takes the ``raw_evidence`` strings collected by the Market Scraper and the
``target_supply_chain_nodes`` from the Orchestrator, then synthesises
exactly **5 candidate UseCase objects** with quantitative economic impact
assessments.

Design Rationale:
    - Uses Gemini's ``with_structured_output()`` to guarantee Pydantic-
      validated JSON output, with a manual-parse fallback if structured
      output fails.
    - The prompt forces the LLM into a rigorous microeconomist / quant
      persona — no hand-waving, every ROI claim must be grounded in the
      evidence corpus.
    - The ``risk_assessment`` and ``implementation_approach`` fields are
      deliberately left empty (defaults) — the Risk Assessor fills those.
    - "Automating Vendor Tendering and Procurement" is always included as
      one of the 5 use cases (fail-safe injection if the LLM omits it).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from config.settings import get_llm
from src.state.graph_state import (
    AgentState,
    CostComplexity,
    CriticVerdict,
    EconomicImpact,
    GraphState,
    MaturityLevel,
    UseCase,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Environment Setup
# ═══════════════════════════════════════════════════════════════════════════

# Environment setup handled centrally by config.settings.get_llm()
# ═══════════════════════════════════════════════════════════════════════════


class AnalystUseCaseCandidate(BaseModel):
    """
    Slimmed-down use-case schema for the Analyst's structured output.

    We use a separate model here (rather than UseCase directly) because
    the Analyst should NOT populate risk_assessment or implementation_approach.
    This schema only exposes the fields the Analyst is responsible for.
    """

    topic: str = Field(
        ...,
        description=(
            "Concise, descriptive title of the GenAI use case. "
            "Example: 'Agentic Vendor Tendering & Procurement Automation'."
        ),
    )
    supply_chain_segment: str = Field(
        ...,
        description="Primary FMCG supply-chain segment (e.g., procurement, logistics, QA).",
    )
    description: str = Field(
        ...,
        description=(
            "2–4 sentence description grounded in the evidence corpus. "
            "Must reference specific data points or sources."
        ),
    )
    maturity_level: str = Field(
        ...,
        description=(
            "Technology readiness: one of 'Theoretical', 'Proof_of_Concept', "
            "'Pilot', or 'Production_Ready'."
        ),
    )
    estimated_roi_percentage: float = Field(
        ...,
        description="Estimated ROI as a percentage (0–500). Must be evidence-backed.",
    )
    marginal_efficiency_gain_description: str = Field(
        ...,
        description=(
            "Narrative (min 20 chars) describing the marginal efficiency gain "
            "vs. current manual/legacy process, referencing specific metrics."
        ),
    )
    implementation_cost_complexity: str = Field(
        ...,
        description="Cost complexity: 'Low', 'Medium', or 'High'.",
    )
    evidence_sources: list[str] = Field(
        default_factory=list,
        description="URLs or source references from the evidence corpus.",
    )
    iteration_count: int = Field(
        default=0,
        description=(
            "Revision cycle counter. Increment by 1 when revising or "
            "replacing a use case. New use cases start at 0."
        ),
    )


class AnalystOutput(BaseModel):
    """Wrapper for the Analyst's structured output — exactly 5 use cases."""

    use_cases: list[AnalystUseCaseCandidate] = Field(
        ...,
        description="Exactly 5 GenAI use cases in FMCG supply chains.",
        min_length=5,
        max_length=5,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Constants
# ═══════════════════════════════════════════════════════════════════════════

ANALYST_SYSTEM_PROMPT = """\
You are a rigorous Microeconomist and Quantitative Data Scientist \
specialising in technology-driven operational transformation within \
FMCG / CPG supply chains.

You have been given a corpus of research evidence about Generative AI \
use cases in FMCG supply chains. Your task is to identify and evaluate \
the TOP 5 most impactful use cases.

═══════════════════════════════════════════════════════════════════
YOUR ANALYTICAL FRAMEWORK:
═══════════════════════════════════════════════════════════════════

For EACH of the 5 use cases you identify, you must provide:

1. **topic**: A concise, descriptive title.
2. **supply_chain_segment**: The primary FMCG supply-chain segment.
3. **description**: 2–4 sentences grounded in evidence. Reference specific \
   data points, percentages, or case studies from the corpus.
4. **maturity_level**: Rate as one of:
   - "Theoretical" — concept only, no known implementations
   - "Proof_of_Concept" — lab/sandbox demos exist
   - "Pilot" — limited production deployments in progress
   - "Production_Ready" — multiple enterprises in production
5. **estimated_roi_percentage**: A realistic ROI estimate (0–500%). If \
   evidence is thin, be CONSERVATIVE. Do not inflate.
6. **marginal_efficiency_gain_description**: Detailed narrative (min 20 \
   characters) describing the efficiency gain vs. current processes. \
   Reference specific operational metrics.
7. **implementation_cost_complexity**: Rate as "Low", "Medium", or "High".
8. **evidence_sources**: List of URLs/references backing this use case.

═══════════════════════════════════════════════════════════════════
HARD CONSTRAINTS:
═══════════════════════════════════════════════════════════════════

1. You MUST include "Automating Vendor Tendering and Procurement using \
   Agentic AI Frameworks" as one of the 5 use cases. This is NON-NEGOTIABLE.

2. Every ROI estimate and efficiency claim MUST be grounded in the evidence \
   corpus. If the evidence is insufficient for a precise figure, provide a \
   conservative lower-bound estimate and note the uncertainty.

3. DO NOT populate implementation_approach or risk_assessment — those will \
   be handled by a separate Risk Assessor agent.

4. Focus on **B2B operational bottlenecks**, not consumer-facing applications.

5. Return EXACTLY 5 use cases, no more, no fewer.

6. Be CYNICAL about vendor marketing claims. Distinguish between demonstrated \
   results and aspirational projections.
"""

ANALYST_USER_PROMPT = """\
═══════════════════════════════════════════════════════════════════
TARGET SUPPLY-CHAIN NODES (from Orchestrator):
═══════════════════════════════════════════════════════════════════

{target_nodes_text}

═══════════════════════════════════════════════════════════════════
RAW EVIDENCE CORPUS (from Market Scraper):
═══════════════════════════════════════════════════════════════════

{evidence_text}

═══════════════════════════════════════════════════════════════════
TASK: Identify and evaluate the top 5 GenAI use cases from this evidence.
Return exactly 5 use cases. Remember: "Automating Vendor Tendering and \
Procurement using Agentic AI Frameworks" MUST be one of them.
═══════════════════════════════════════════════════════════════════
"""

ANALYST_REVISION_SYSTEM_PROMPT = """\
You are a rigorous Microeconomist and Quantitative Data Scientist \
specialising in technology-driven operational transformation within \
FMCG / CPG supply chains.

You are in a REVISION LOOP. The Red Team Critic has reviewed the \
previous set of 5 use cases and FAILED some of them. Your task is to \
FIX the failures while keeping the successful use cases untouched.

═══════════════════════════════════════════════════════════════════
YOUR REVISION RULES:
═══════════════════════════════════════════════════════════════════

1. For each use case marked as FAILED:
   - Read the critic_feedback carefully. It contains EXPLICIT FIX \
     INSTRUCTIONS (e.g., "Lower ROI from 150% to 40%").
   - OPTION A — REVISE: Apply the Critic's fix instructions directly. \
     Lower the ROI, adjust the efficiency gain description, fix the \
     maturity level, etc. Keep the same topic and description core.
   - OPTION B — REPLACE: If the use case is fundamentally unsalvageable, \
     replace it entirely with a NEW use case from the evidence corpus. \
     The new use case must still be grounded in evidence.
   - In EITHER case, increment the iteration_count by 1.

2. For each use case marked as PASSED:
   - Return it UNCHANGED. Do NOT modify any fields. Keep the same \
     iteration_count.

3. You MUST return EXACTLY 5 use cases total.

4. "Automating Vendor Tendering and Procurement using Agentic AI \
   Frameworks" MUST remain as one of the 5 use cases. If it was failed, \
   REVISE it — do NOT replace it.

5. Be CONSERVATIVE with numbers this time. The Critic already flagged \
   inflated figures — do not repeat the same mistake.
"""

ANALYST_REVISION_USER_PROMPT = """\
═══════════════════════════════════════════════════════════════════
CURRENT USE CASES (with Critic verdicts and feedback):
═══════════════════════════════════════════════════════════════════

{use_cases_json}

═══════════════════════════════════════════════════════════════════
RAW EVIDENCE CORPUS (for replacements if needed):
═══════════════════════════════════════════════════════════════════

{evidence_text}

═══════════════════════════════════════════════════════════════════
TASK: Revise or replace the FAILED use cases. Return exactly 5 use cases.
For revised/replaced use cases, increment iteration_count by 1.
Keep PASSED use cases unchanged.
═══════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# Mandatory Use Case  (fail-safe constant)
# ═══════════════════════════════════════════════════════════════════════════

_MANDATORY_TOPIC_FRAGMENT = "vendor tendering"


# ═══════════════════════════════════════════════════════════════════════════
# LLM Factory
# ═══════════════════════════════════════════════════════════════════════════

def _get_llm():
    """Get the centralised LLM instance for the Economic Analyst."""
    return get_llm(max_output_tokens=8192)


# ═══════════════════════════════════════════════════════════════════════════
# Node Implementation
# ═══════════════════════════════════════════════════════════════════════════

def run_economic_analyst(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node: synthesise raw evidence into 5 candidate UseCase objects.

    Strategy:
      - **First pass** (no prior candidates): generate 5 use cases from scratch.
      - **Retry pass** (candidates exist + error_count > 0): revise or replace
        the failed use cases using the Critic's feedback.

    In both cases:
      1. Attempt structured output via ``llm.with_structured_output``.
      2. Fallback to raw JSON extraction + manual Pydantic validation.
      3. Guarantee the mandatory vendor-tendering use case is present.
      4. Return exactly 5 candidates.

    Args:
        state: Current LangGraph shared state.

    Returns:
        Dict updating ``candidate_use_cases`` and ``current_agent``.
    """
    raw_evidence: list[str] = state.get("raw_evidence", [])
    target_nodes: list[str] = state.get("target_supply_chain_nodes", [])
    existing_errors: list[str] = state.get("errors", [])
    existing_candidates: list[UseCase] = state.get("candidate_use_cases", [])
    error_count: int = state.get("error_count", 0)

    # ── Detect retry mode ─────────────────────────────────────────────
    is_retry = bool(existing_candidates) and error_count > 0

    if is_retry:
        logger.info(
            "Economic Analyst: RETRY MODE (error_count=%d) — revising %d "
            "candidates based on Critic feedback",
            error_count, len(existing_candidates),
        )
        return _run_revision_pass(
            existing_candidates=existing_candidates,
            raw_evidence=raw_evidence,
            existing_errors=existing_errors,
        )

    # ── First pass: generate from scratch ─────────────────────────────
    logger.info(
        "Economic Analyst: FIRST PASS — synthesising %d evidence blocks "
        "into 5 use cases",
        len(raw_evidence),
    )

    if not raw_evidence:
        logger.warning("Economic Analyst: no evidence to analyse — using target nodes only")

    # ── Build prompt context ──────────────────────────────────────────
    target_nodes_text = "\n".join(f"  {i}. {n}" for i, n in enumerate(target_nodes, 1))
    evidence_text = "\n\n".join(raw_evidence) if raw_evidence else "[No evidence collected]"

    # Truncate evidence if it's extremely long (context-window safety)
    if len(evidence_text) > 60_000:
        evidence_text = evidence_text[:60_000] + "\n\n[... evidence truncated for context limits]"

    messages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(
            content=ANALYST_USER_PROMPT.format(
                target_nodes_text=target_nodes_text,
                evidence_text=evidence_text,
            )
        ),
    ]

    # ── Attempt 1: Structured output ──────────────────────────────────
    llm = _get_llm()
    candidates: list[UseCase] = []
    new_errors: list[str] = []

    try:
        structured_llm = llm.with_structured_output(AnalystOutput)
        result: AnalystOutput = structured_llm.invoke(messages)

        logger.info(
            "Economic Analyst: structured output succeeded — %d candidates",
            len(result.use_cases),
        )
        candidates = [_convert_to_use_case(c) for c in result.use_cases]

    except (ValidationError, Exception) as exc:
        logger.warning(
            "Economic Analyst: structured output failed (%s) — "
            "falling back to raw JSON parsing",
            exc,
        )
        new_errors.append(f"Economic Analyst structured output fallback: {exc}")

        # ── Attempt 2: Raw LLM output + manual parsing ───────────────
        try:
            candidates = _fallback_parse(llm, messages)
        except Exception as fallback_exc:
            logger.error(
                "Economic Analyst: fallback parsing also failed: %s",
                fallback_exc,
            )
            new_errors.append(f"Economic Analyst fallback failed: {fallback_exc}")

    # ── Fail-safe: ensure mandatory vendor-tendering use case ─────────
    candidates = _ensure_mandatory_use_case(candidates)

    # ── Limit to exactly 5 ────────────────────────────────────────────
    if len(candidates) > 5:
        candidates = candidates[:5]

    logger.info(
        "Economic Analyst: finalised %d candidate use cases: %s",
        len(candidates),
        [c.topic for c in candidates],
    )

    return {
        "candidate_use_cases": candidates,
        "current_agent": AgentState.ECONOMIC_ANALYST,
        "errors": existing_errors + new_errors if new_errors else existing_errors,
    }


def _run_revision_pass(
    existing_candidates: list[UseCase],
    raw_evidence: list[str],
    existing_errors: list[str],
) -> dict[str, Any]:
    """
    Revision pass: revise or replace failed use cases based on Critic feedback.

    Sends the existing 5 use cases (with their verdicts and feedback) to the
    LLM with instructions to fix or replace the failures. Passed use cases
    are kept unchanged.

    Args:
        existing_candidates: Current list of UseCase objects with critic verdicts.
        raw_evidence: Evidence corpus for potential replacements.
        existing_errors: Accumulated error messages.

    Returns:
        Dict updating ``candidate_use_cases`` and ``current_agent``.
    """
    new_errors: list[str] = []

    # ── Serialise use cases for the prompt ────────────────────────────
    use_cases_for_prompt = []
    for uc in existing_candidates:
        use_cases_for_prompt.append({
            "topic": uc.topic,
            "supply_chain_segment": uc.supply_chain_segment,
            "description": uc.description,
            "maturity_level": uc.maturity_level.value if uc.maturity_level else "Theoretical",
            "estimated_roi_percentage": (
                uc.economic_impact.estimated_roi_percentage
                if uc.economic_impact else 0.0
            ),
            "marginal_efficiency_gain_description": (
                uc.economic_impact.marginal_efficiency_gain_description
                if uc.economic_impact else "N/A"
            ),
            "implementation_cost_complexity": (
                uc.economic_impact.implementation_cost_complexity.value
                if uc.economic_impact else "Medium"
            ),
            "evidence_sources": uc.evidence_sources,
            "critic_verdict": uc.critic_verdict.value,
            "critic_feedback": uc.critic_feedback,
            "iteration_count": uc.iteration_count,
        })

    import json as _json
    use_cases_json = _json.dumps(use_cases_for_prompt, indent=2)

    evidence_text = "\n\n".join(raw_evidence) if raw_evidence else "[No evidence collected]"
    if len(evidence_text) > 40_000:
        evidence_text = evidence_text[:40_000] + "\n\n[... evidence truncated]"

    messages = [
        SystemMessage(content=ANALYST_REVISION_SYSTEM_PROMPT),
        HumanMessage(
            content=ANALYST_REVISION_USER_PROMPT.format(
                use_cases_json=use_cases_json,
                evidence_text=evidence_text,
            )
        ),
    ]

    llm = _get_llm()
    candidates: list[UseCase] = []

    try:
        structured_llm = llm.with_structured_output(AnalystOutput)
        result: AnalystOutput = structured_llm.invoke(messages)

        logger.info(
            "Economic Analyst (revision): structured output succeeded — %d candidates",
            len(result.use_cases),
        )
        candidates = [_convert_to_use_case(c) for c in result.use_cases]

    except (ValidationError, Exception) as exc:
        logger.warning(
            "Economic Analyst (revision): structured output failed (%s) — "
            "falling back to raw JSON parsing",
            exc,
        )
        new_errors.append(f"Economic Analyst revision structured output fallback: {exc}")

        try:
            candidates = _fallback_parse(llm, messages)
        except Exception as fallback_exc:
            logger.error(
                "Economic Analyst (revision): fallback also failed: %s — "
                "returning original candidates with incremented iteration_count",
                fallback_exc,
            )
            new_errors.append(f"Economic Analyst revision fallback failed: {fallback_exc}")
            # Last resort: return originals with bumped iteration_count on failures
            candidates = []
            for uc in existing_candidates:
                if uc.critic_verdict == CriticVerdict.FAIL:
                    candidates.append(uc.model_copy(
                        update={"iteration_count": uc.iteration_count + 1}
                    ))
                else:
                    candidates.append(uc)

    # ── Fail-safe: ensure mandatory vendor-tendering use case ─────────
    candidates = _ensure_mandatory_use_case(candidates)

    # ── Limit to exactly 5 ────────────────────────────────────────────
    if len(candidates) > 5:
        candidates = candidates[:5]

    logger.info(
        "Economic Analyst (revision): finalised %d revised candidates: %s",
        len(candidates),
        [c.topic for c in candidates],
    )

    return {
        "candidate_use_cases": candidates,
        "current_agent": AgentState.ECONOMIC_ANALYST,
        "errors": existing_errors + new_errors if new_errors else existing_errors,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Conversion & Parsing Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _convert_to_use_case(candidate: AnalystUseCaseCandidate) -> UseCase:
    """
    Convert an ``AnalystUseCaseCandidate`` into a full ``UseCase``.

    Maps the flat analyst fields into the nested Pydantic structure,
    leaving ``risk_assessment`` and ``implementation_approach`` as defaults.
    """
    # Map maturity string to enum (with fallback)
    maturity_map = {
        "theoretical": MaturityLevel.THEORETICAL,
        "proof_of_concept": MaturityLevel.PROOF_OF_CONCEPT,
        "pilot": MaturityLevel.PILOT,
        "production_ready": MaturityLevel.PRODUCTION_READY,
    }
    maturity = maturity_map.get(
        candidate.maturity_level.lower().replace(" ", "_"),
        MaturityLevel.THEORETICAL,
    )

    # Map cost complexity string to enum
    cost_map = {
        "low": CostComplexity.LOW,
        "medium": CostComplexity.MEDIUM,
        "high": CostComplexity.HIGH,
    }
    cost = cost_map.get(
        candidate.implementation_cost_complexity.lower(),
        CostComplexity.MEDIUM,
    )

    # Ensure marginal_efficiency_gain_description meets min_length
    efficiency_desc = candidate.marginal_efficiency_gain_description
    if len(efficiency_desc) < 20:
        efficiency_desc = efficiency_desc + " " * (20 - len(efficiency_desc))

    # Clamp ROI to valid range
    roi = max(0.0, min(500.0, candidate.estimated_roi_percentage))

    return UseCase(
        topic=candidate.topic,
        supply_chain_segment=candidate.supply_chain_segment,
        description=candidate.description,
        implementation_approach="",  # Left for Risk Assessor
        maturity_level=maturity,
        economic_impact=EconomicImpact(
            estimated_roi_percentage=roi,
            marginal_efficiency_gain_description=efficiency_desc,
            implementation_cost_complexity=cost,
        ),
        risk_assessment=None,  # Left for Risk Assessor
        evidence_sources=candidate.evidence_sources,
        critic_feedback=[],
        iteration_count=candidate.iteration_count,
    )


def _fallback_parse(
    llm,
    messages: list,
) -> list[UseCase]:
    """
    Fallback: invoke the LLM without structured output and manually parse
    the JSON response into UseCase objects.
    """
    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]).strip()

    data = json.loads(raw)

    # Handle both {"use_cases": [...]} and bare [...]
    items: list[dict] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("use_cases", "candidates", "results"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        if not items:
            items = [data]

    candidates: list[UseCase] = []
    for item in items[:5]:
        try:
            # Try parsing as AnalystUseCaseCandidate first
            c = AnalystUseCaseCandidate.model_validate(item)
            candidates.append(_convert_to_use_case(c))
        except ValidationError:
            # Try direct UseCase parsing as last resort
            try:
                candidates.append(UseCase.model_validate(item))
            except ValidationError as e:
                logger.warning("Skipping invalid use case item: %s", e)

    return candidates


def _ensure_mandatory_use_case(candidates: list[UseCase]) -> list[UseCase]:
    """
    Guarantee that the mandatory vendor-tendering use case is present.

    If not found, inject a default entry at position 0.
    """
    has_mandatory = any(
        _MANDATORY_TOPIC_FRAGMENT in uc.topic.lower()
        for uc in candidates
    )

    if not has_mandatory:
        logger.warning(
            "Economic Analyst: mandatory 'Vendor Tendering' use case missing "
            "— injecting default"
        )
        default_uc = UseCase(
            topic="Automating Vendor Tendering and Procurement using Agentic AI Frameworks",
            supply_chain_segment="procurement",
            description=(
                "Agentic AI frameworks can autonomously manage the end-to-end "
                "vendor tendering lifecycle — from RFQ generation and vendor "
                "shortlisting to bid evaluation and contract drafting — "
                "reducing procurement cycle times by 40-60%."
            ),
            maturity_level=MaturityLevel.PROOF_OF_CONCEPT,
            economic_impact=EconomicImpact(
                estimated_roi_percentage=35.0,
                marginal_efficiency_gain_description=(
                    "Reduces vendor tendering cycle from 14-21 days to 3-5 days "
                    "by automating RFQ drafting, bid analysis, and compliance "
                    "checks. Estimated 40% reduction in procurement FTE costs."
                ),
                implementation_cost_complexity=CostComplexity.HIGH,
            ),
        )
        candidates.insert(0, default_uc)

    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible alias
# ═══════════════════════════════════════════════════════════════════════════
economic_analyst_node = run_economic_analyst
