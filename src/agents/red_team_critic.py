"""
Red Team Critic Agent
=====================

The mandatory adversarial reflection loop — the final gatekeeper in the
multi-agent pipeline. Evaluates each ``candidate_use_case`` produced by the
Economic Analyst + Risk Assessor chain and issues pass/fail verdicts.

Design Rationale:
    - Processes use cases **one at a time** so the LLM can focus its full
      adversarial attention on each claim, risk, and implementation narrative.
    - Uses ``with_structured_output(CriticEvaluation)`` for deterministic
      boolean pass/fail + rationale, with a manual-parse fallback.
    - Three-axis evaluation: **Hallucination**, **Integration Reality**,
      **Tendering Rigor** (the vendor-tendering use case gets extra scrutiny).
    - On failure: appends ``critique_rationale`` to the ``UseCase.critic_feedback``
      list and increments ``error_count`` in the graph state.
    - On all-pass OR max-retries: promotes ``candidate_use_cases`` to
      ``final_top_5``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from config.settings import get_llm, MAX_RETRIES
from src.state.graph_state import (
    AgentState,
    CriticVerdict,
    GraphState,
    UseCase,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Environment Setup
# ═══════════════════════════════════════════════════════════════════════════

# Environment setup handled centrally by config.settings.get_llm()


# ═══════════════════════════════════════════════════════════════════════════
# Structured Output Schema
# ═══════════════════════════════════════════════════════════════════════════

class CriticEvaluation(BaseModel):
    """
    Structured verdict from the Red Team Critic for a single use case.

    The boolean ``passed`` drives the retry loop — a single ``False``
    triggers re-routing back to the Analyst for refinement.
    """

    passed: bool = Field(
        ...,
        description=(
            "True if the use case passes ALL three evaluation axes "
            "(hallucination, integration reality, tendering rigor). "
            "False if ANY axis fails."
        ),
    )
    critique_rationale: str = Field(
        ...,
        description=(
            "Detailed 3–5 sentence explanation of the verdict. "
            "If failed: specifically identify which axis (Hallucination, "
            "Integration Reality, or Tendering Rigor) triggered failure "
            "and what must be fixed. "
            "If passed: briefly confirm which evidence supported each claim."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Constants
# ═══════════════════════════════════════════════════════════════════════════

CRITIC_SYSTEM_PROMPT = """\
You are a pragmatic Enterprise Strategy Director and Chief Risk Officer \
for a $10B FMCG conglomerate. Your job is to catch REAL problems — \
fabricated evidence, physically impossible claims, and outright \
hallucinations — NOT to reject every use case that has slightly \
optimistic numbers.

═══════════════════════════════════════════════════════════════════
YOUR THREE EVALUATION AXES:
═══════════════════════════════════════════════════════════════════

AXIS 1 — HALLUCINATION CHECK (HARD FAIL only for fabrications):
  - Are the evidence_sources clearly fabricated or non-existent? → FAIL
  - Are the ROI claims PHYSICALLY IMPOSSIBLE (e.g., 400% ROI for a \
    simple dashboarding tool)? → FAIL
  - Is the marginal_efficiency_gain_description entirely vague with ZERO \
    measurable metrics? → FAIL
  - If the ROI is merely inflated but plausible (e.g., 120% where 40% is \
    more realistic), this is NOT a hallucination. → FAIL, but provide \
    EXPLICIT FIX INSTRUCTIONS in critique_rationale: "Lower ROI from X% \
    to Y%, adjust efficiency claim to Z."

AXIS 2 — INTEGRATION REALITY CHECK (pragmatic, not perfectionist):
  - Does the implementation_approach make basic sense for a legacy FMCG \
    enterprise? If it's reasonable but slightly optimistic on timeline, \
    that is acceptable — just note the concern.
  - Does the risk_assessment exist and identify at least one real bottleneck? \
    Generic but not completely wrong answers should PASS with a note.
  - Is the integration_complexity at least acknowledged? If yes → PASS.
  - Only FAIL if the integration story is completely disconnected from \
    reality (e.g., claiming zero SAP integration effort for an ERP-heavy \
    use case).

AXIS 3 — TENDERING RIGOR (for Vendor Tendering use case ONLY):
  - Does the Vendor Tendering use case propose a genuinely AGENTIC framework \
    (autonomous agents handling RFQ generation, bid evaluation, compliance \
    checking, negotiation) — or is it just a glorified chatbot / basic RAG \
    system? If chatbot-only → FAIL.
  - Does it address multi-step procurement workflows? At least 3 steps → PASS.
  - Is the maturity level realistic? (Most agentic procurement systems are \
    Proof_of_Concept or Pilot at best in 2026.)

═══════════════════════════════════════════════════════════════════
VERDICT RULES — BE PRAGMATIC, NOT RUTHLESS:
═══════════════════════════════════════════════════════════════════

- Set passed=TRUE if the use case is fundamentally sound, even if some \
  numbers are slightly optimistic.
- Set passed=FALSE ONLY if the use case is a fundamental hallucination \
  (fabricated evidence, impossible claims) or completely disconnected \
  from integration reality.
- If you FAIL a use case for inflated metrics (NOT hallucination), you \
  MUST provide EXPLICIT FIX INSTRUCTIONS in critique_rationale. Example: \
  "FIXABLE: Lower estimated_roi_percentage from 150 to 40. Change \
  implementation timeline from 3 months to 12 months. Adjust efficiency \
  gain to reference specific cycle-time reduction rather than vague claims."
- A product manager should be able to fix a failed use case in ONE revision \
  based solely on your critique_rationale.
- Do NOT reject for minor stylistic or phrasing issues.
"""

CRITIC_USER_PROMPT = """\
═══════════════════════════════════════════════════════════════════
USE CASE UNDER REVIEW ({index}/{total}):
═══════════════════════════════════════════════════════════════════

Topic: {topic}
Supply Chain Segment: {segment}
Description: {description}
Maturity Level: {maturity}

--- Economic Impact ---
Estimated ROI: {roi}%
Efficiency Gains: {efficiency}
Cost Complexity: {cost_complexity}

--- Risk Assessment ---
Primary Bottleneck: {bottleneck}
Data Privacy Concerns: {privacy}
Integration Complexity: {integration}

--- Implementation Approach ---
{implementation}

--- Evidence Sources ---
{evidence_sources}

--- Previous Critic Feedback (if any) ---
{previous_feedback}

═══════════════════════════════════════════════════════════════════
Is this use case "{topic}" a VENDOR TENDERING use case? {is_tendering}
If yes, apply the TENDERING RIGOR axis with extra scrutiny.

Evaluate this use case against all applicable axes and return your verdict.
═══════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_TENDERING_KEYWORDS = ("vendor tendering", "procurement", "tendering")


# ═══════════════════════════════════════════════════════════════════════════
# LLM Factory
# ═══════════════════════════════════════════════════════════════════════════

def _get_llm():
    """Get the centralised LLM instance for the Red Team Critic."""
    return get_llm(max_output_tokens=4096)


# ═══════════════════════════════════════════════════════════════════════════
# Node Implementation
# ═══════════════════════════════════════════════════════════════════════════

def run_red_team_critic(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node: adversarial validation of every candidate use case.

    Logic:
      1. Iterate through ``candidate_use_cases``.
      2. For each, invoke Gemini with the 3-axis critic prompt.
      3. Record pass/fail verdict and append ``critique_rationale`` to
         the use case's ``critic_feedback`` list.
      4. If ANY use case fails and ``error_count < MAX_RETRIES``:
         increment ``error_count`` → the graph router will send us back
         to the Analyst for refinement.
      5. If ALL pass or ``error_count >= MAX_RETRIES``:
         promote candidates to ``final_top_5``.

    Args:
        state: Current LangGraph shared state.

    Returns:
        Dict updating ``candidate_use_cases``, ``final_top_5``,
        ``error_count``, and ``current_agent``.
    """
    candidates: list[UseCase] = state.get("candidate_use_cases", [])
    error_count: int = state.get("error_count", 0)
    existing_errors: list[str] = state.get("errors", [])

    if not candidates:
        logger.warning("Red Team Critic: no candidates to evaluate")
        return {
            "candidate_use_cases": [],
            "final_top_5": [],
            "error_count": error_count,
            "current_agent": AgentState.RED_TEAM_CRITIC,
            "errors": existing_errors + ["Red Team Critic: no candidates provided"],
        }

    logger.info(
        "Red Team Critic: evaluating %d candidates (error_count=%d, max=%d)",
        len(candidates), error_count, MAX_RETRIES,
    )

    llm = _get_llm()
    reviewed: list[UseCase] = []
    failures: int = 0
    new_errors: list[str] = []

    for i, use_case in enumerate(candidates, 1):
        logger.info(
            "Red Team Critic [%d/%d]: reviewing '%s'",
            i, len(candidates), use_case.topic,
        )

        try:
            evaluation = _evaluate_single_use_case(
                use_case=use_case,
                index=i,
                total=len(candidates),
                llm=llm,
            )

            # ── Update the use case with the verdict ──────────────────
            new_feedback = list(use_case.critic_feedback) + [evaluation.critique_rationale]
            new_verdict = CriticVerdict.PASS if evaluation.passed else CriticVerdict.FAIL

            updated_uc = use_case.model_copy(
                update={
                    "critic_feedback": new_feedback,
                    "critic_verdict": new_verdict,
                }
            )
            reviewed.append(updated_uc)

            if evaluation.passed:
                logger.info(
                    "Red Team Critic [%d/%d]: ✅ PASSED — '%s'",
                    i, len(candidates), use_case.topic,
                )
            else:
                failures += 1
                logger.warning(
                    "Red Team Critic [%d/%d]: ❌ FAILED — '%s': %s",
                    i, len(candidates), use_case.topic,
                    evaluation.critique_rationale[:200],
                )

        except Exception as exc:
            msg = f"Red Team Critic: error evaluating '{use_case.topic}': {exc}"
            logger.error(msg, exc_info=True)
            new_errors.append(msg)
            # On error, conservatively treat as a pass to avoid blocking
            reviewed.append(use_case)

        # Free tier rate limit protection (15 RPM)
        if i < len(candidates):
            logger.debug("Red Team Critic: sleeping 5s to respect API rate limits...")
            time.sleep(5)

    # ── Decide: retry or finalize ─────────────────────────────────────
    new_error_count = error_count + (1 if failures > 0 else 0)
    all_passed = failures == 0

    if all_passed:
        logger.info(
            "Red Team Critic: ✅ ALL %d candidates passed — promoting to final_top_5",
            len(reviewed),
        )
        return {
            "candidate_use_cases": reviewed,
            "final_top_5": reviewed,
            "error_count": new_error_count,
            "current_agent": AgentState.RED_TEAM_CRITIC,
            "errors": existing_errors + new_errors if new_errors else existing_errors,
        }

    if new_error_count >= MAX_RETRIES:
        logger.warning(
            "Red Team Critic: ⚠️ Max retries (%d) reached with %d failures — "
            "forcing promotion to final_top_5",
            MAX_RETRIES, failures,
        )
        return {
            "candidate_use_cases": reviewed,
            "final_top_5": reviewed,
            "error_count": new_error_count,
            "current_agent": AgentState.RED_TEAM_CRITIC,
            "errors": existing_errors + new_errors if new_errors else existing_errors,
        }

    # Failures exist but retries remain → route back to Analyst
    logger.info(
        "Red Team Critic: 🔄 %d failures detected, error_count=%d (< max=%d) — "
        "flagging for retry",
        failures, new_error_count, MAX_RETRIES,
    )
    return {
        "candidate_use_cases": reviewed,
        "final_top_5": [],  # Empty → signals the router that we haven't finalized
        "error_count": new_error_count,
        "current_agent": AgentState.RED_TEAM_CRITIC,
        "errors": existing_errors + new_errors if new_errors else existing_errors,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-Use-Case Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_single_use_case(
    use_case: UseCase,
    index: int,
    total: int,
    llm,
) -> CriticEvaluation:
    """
    Evaluate a single use case against the three adversarial axes.

    Strategy:
      0. **Circuit breaker**: if ``iteration_count >= 2``, force-pass
         immediately without calling the LLM.
      1. Try ``with_structured_output(CriticEvaluation)``.
      2. If that fails, fall back to raw JSON parsing.

    Args:
        use_case: The candidate to evaluate.
        index: Position in the review batch (1-indexed).
        total: Total number of candidates.
        llm: Pre-configured Gemini instance.

    Returns:
        CriticEvaluation with pass/fail verdict and rationale.
    """
    # ── Circuit breaker: force-pass after 2 iterations ────────────────
    if use_case.iteration_count >= 2:
        logger.warning(
            "Red Team Critic [%d/%d]: ⚡ CIRCUIT BREAKER — '%s' has "
            "iteration_count=%d (>= 2), force-passing without LLM call",
            index, total, use_case.topic, use_case.iteration_count,
        )
        return CriticEvaluation(
            passed=True,
            critique_rationale=(
                "SYSTEM OVERRIDE: Forced pass due to iteration limits. "
                "This use case has been through multiple revision cycles "
                "without satisfying the Critic. It should be treated as "
                "highly theoretical and risky. Manual review recommended "
                "before any investment decisions."
            ),
        )
    # ── Detect if this is the vendor tendering use case ───────────────
    is_tendering = any(
        kw in use_case.topic.lower() for kw in _TENDERING_KEYWORDS
    )

    # ── Build context strings ─────────────────────────────────────────
    roi_str = (
        str(use_case.economic_impact.estimated_roi_percentage)
        if use_case.economic_impact else "N/A"
    )
    efficiency_str = (
        use_case.economic_impact.marginal_efficiency_gain_description
        if use_case.economic_impact else "N/A"
    )
    cost_str = (
        use_case.economic_impact.implementation_cost_complexity.value
        if use_case.economic_impact else "N/A"
    )
    bottleneck_str = (
        use_case.risk_assessment.primary_bottleneck
        if use_case.risk_assessment else "NOT ASSESSED"
    )
    privacy_str = (
        use_case.risk_assessment.data_privacy_concerns
        if use_case.risk_assessment else "NOT ASSESSED"
    )
    integration_str = (
        use_case.risk_assessment.integration_complexity
        if use_case.risk_assessment else "NOT ASSESSED"
    )
    evidence_str = (
        "\n".join(f"  - {src}" for src in use_case.evidence_sources)
        if use_case.evidence_sources else "  [No evidence sources cited]"
    )
    prev_feedback_str = (
        "\n".join(f"  Round {j+1}: {fb}" for j, fb in enumerate(use_case.critic_feedback))
        if use_case.critic_feedback else "  [First review — no prior feedback]"
    )

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(
            content=CRITIC_USER_PROMPT.format(
                index=index,
                total=total,
                topic=use_case.topic,
                segment=use_case.supply_chain_segment,
                description=use_case.description,
                maturity=use_case.maturity_level.value if use_case.maturity_level else "N/A",
                roi=roi_str,
                efficiency=efficiency_str,
                cost_complexity=cost_str,
                bottleneck=bottleneck_str,
                privacy=privacy_str,
                integration=integration_str,
                implementation=use_case.implementation_approach or "[NOT PROVIDED]",
                evidence_sources=evidence_str,
                previous_feedback=prev_feedback_str,
                is_tendering="YES — apply TENDERING RIGOR axis" if is_tendering else "NO",
            )
        ),
    ]

    # ── Attempt 1: Structured output ──────────────────────────────────
    try:
        structured_llm = llm.with_structured_output(CriticEvaluation)
        result = structured_llm.invoke(messages)
        logger.debug(
            "Red Team Critic: structured output succeeded for '%s'",
            use_case.topic,
        )
        return result
    except (ValidationError, Exception) as exc:
        logger.warning(
            "Red Team Critic: structured output failed for '%s' (%s) — "
            "trying raw JSON fallback",
            use_case.topic, exc,
        )

    # ── Attempt 2: Raw JSON fallback ──────────────────────────────────
    return _fallback_parse_evaluation(llm, messages, use_case.topic)


def _fallback_parse_evaluation(
    llm,
    messages: list,
    topic: str,
) -> CriticEvaluation:
    """
    Fallback: invoke LLM without structured output and manually parse.
    """
    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]).strip()

        data = json.loads(raw)
        return CriticEvaluation.model_validate(data)

    except (json.JSONDecodeError, ValidationError) as exc:
        logger.error(
            "Red Team Critic: fallback parsing failed for '%s': %s — "
            "defaulting to FAIL with generic rationale",
            topic, exc,
        )
        # Conservative default: FAIL if we can't parse the verdict
        return CriticEvaluation(
            passed=False,
            critique_rationale=(
                f"SYSTEM: Critic evaluation could not be parsed for '{topic}'. "
                f"Parse error: {exc}. Defaulting to FAIL for safety."
            ),
        )
    except Exception as exc:
        logger.error(
            "Red Team Critic: fallback LLM call failed for '%s': %s",
            topic, exc,
        )
        return CriticEvaluation(
            passed=False,
            critique_rationale=(
                f"SYSTEM: Critic LLM call failed for '{topic}'. "
                f"Error: {exc}. Defaulting to FAIL for safety."
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible alias
# ═══════════════════════════════════════════════════════════════════════════
red_team_critic_node = run_red_team_critic
