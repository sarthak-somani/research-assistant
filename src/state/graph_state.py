"""
Graph State Schema
==================

Defines the strict Pydantic models and TypedDict-based graph state that flow
through the LangGraph state machine. This module is the **single source of
truth** for inter-agent data contracts.

Design Principles:
    - Pydantic models enforce runtime validation and JSON serialisation.
    - TypedDict-based GraphState gives LangGraph native dict-merge semantics
      while keeping full type-safety for IDE tooling.
    - Every field uses ``pydantic.Field`` with explicit descriptions so the
      schema is self-documenting and auto-generates clean JSON-Schema for
      structured LLM output parsing.
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class MaturityLevel(str, Enum):
    """
    Technology-readiness classification for a GenAI use case,
    modelled after standard TRL (Technology Readiness Level) bands.
    """
    THEORETICAL = "Theoretical"
    PROOF_OF_CONCEPT = "Proof_of_Concept"
    PILOT = "Pilot"
    PRODUCTION_READY = "Production_Ready"


class CostComplexity(str, Enum):
    """Qualitative implementation-cost / integration-complexity rating."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class AgentState(str, Enum):
    """
    Tracks which agent is currently active inside the LangGraph state machine.
    Used for observability, logging, and conditional routing.
    """
    ORCHESTRATOR = "orchestrator"
    MARKET_SCRAPER = "market_scraper"
    ECONOMIC_ANALYST = "economic_analyst"
    RISK_ASSESSOR = "risk_assessor"
    RED_TEAM_CRITIC = "red_team_critic"
    COMPLETED = "completed"


class CriticVerdict(str, Enum):
    """Red Team Critic validation verdicts."""
    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVISION = "needs_revision"


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Sub-Models  (inter-agent data contracts)
# ═══════════════════════════════════════════════════════════════════════════

class EconomicImpact(BaseModel):
    """
    Quantitative economic-impact assessment produced by the Economic Analyst.

    Forces the LLM to move beyond vague qualitative statements into
    concrete ROI estimates and efficiency-gain narratives.
    """

    estimated_roi_percentage: float = Field(
        ...,
        ge=0,
        le=500,
        description=(
            "Estimated Return on Investment as a percentage. "
            "Must be grounded in evidence; speculative figures should be "
            "flagged with a low confidence note in the rationale."
        ),
    )
    marginal_efficiency_gain_description: str = Field(
        ...,
        min_length=20,
        description=(
            "Narrative describing the marginal efficiency gain relative to "
            "the current manual / legacy process. Should reference specific "
            "operational metrics (e.g., cycle-time reduction, error-rate "
            "decrease, throughput increase)."
        ),
    )
    implementation_cost_complexity: CostComplexity = Field(
        ...,
        description=(
            "Qualitative rating of total implementation cost and integration "
            "complexity: High (>$1M, 12+ months, deep legacy coupling), "
            "Medium ($200K–$1M, 6–12 months), Low (<$200K, <6 months)."
        ),
    )


class RiskAssessment(BaseModel):
    """
    Structured risk evaluation produced by the Risk Assessor agent.

    Covers the three critical dimensions: technical bottlenecks, data
    privacy / regulatory exposure, and legacy-system integration burden.
    """

    primary_bottleneck: str = Field(
        ...,
        min_length=10,
        description=(
            "The single biggest technical bottleneck blocking adoption "
            "(e.g., 'Lack of structured supplier-performance data across "
            "disparate ERP instances')."
        ),
    )
    data_privacy_concerns: str = Field(
        ...,
        min_length=10,
        description=(
            "Data-privacy and regulatory risks specific to this use case. "
            "Reference applicable regulations (GDPR, DPDP Act, CCPA) where "
            "relevant."
        ),
    )
    integration_complexity: str = Field(
        ...,
        min_length=10,
        description=(
            "Description of the legacy-system integration burden — which "
            "enterprise systems (SAP, Oracle, custom MES) must be connected, "
            "and what middleware / API layers are required."
        ),
    )


class UseCase(BaseModel):
    """
    The core data model representing a single GenAI use case in FMCG
    supply chains. Aggregates outputs from all downstream agents.

    This schema is intentionally strict — every field that the final JSON
    report requires is surfaced here so that validation happens at the
    Pydantic layer, not post-hoc.
    """

    topic: str = Field(
        ...,
        min_length=5,
        description=(
            "Concise, descriptive title of the GenAI use case "
            "(e.g., 'Agentic Vendor Tendering & Procurement Automation')."
        ),
    )
    supply_chain_segment: str = Field(
        default="",
        description="Primary supply-chain segment this use case targets.",
    )
    description: str = Field(
        default="",
        description="2–4 sentence description grounded in collected evidence.",
    )
    implementation_approach: str = Field(
        default="",
        description=(
            "3–5 sentence narrative: tech stack, data requirements, "
            "integration points, and phased rollout strategy."
        ),
    )
    maturity_level: MaturityLevel = Field(
        default=MaturityLevel.THEORETICAL,
        description="Current technology-readiness classification.",
    )
    economic_impact: EconomicImpact | None = Field(
        default=None,
        description="Quantitative economic-impact assessment (filled by Economic Analyst).",
    )
    risk_assessment: RiskAssessment | None = Field(
        default=None,
        description="Structured risk evaluation (filled by Risk Assessor).",
    )
    evidence_sources: list[str] = Field(
        default_factory=list,
        description="URLs / source references backing this use case.",
    )
    critic_feedback: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered history of Red Team Critic feedback strings. "
            "Each retry appends a new entry."
        ),
    )
    critic_verdict: CriticVerdict = Field(
        default=CriticVerdict.NEEDS_REVISION,
        description="Latest Red Team Critic verdict for this use case.",
    )
    iteration_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of revision cycles this use case has been through. "
            "Used by the Red Team Critic circuit breaker — if iteration_count >= 2, "
            "the use case is force-passed regardless of quality."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph Shared State  (TypedDict for native dict-merge semantics)
# ═══════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict, total=False):
    """
    The shared state dictionary that flows through the LangGraph state
    machine.

    Using ``TypedDict`` (instead of a Pydantic BaseModel) gives LangGraph
    its native partial-update / dict-merge semantics — each agent node
    returns only the keys it wants to update, and LangGraph handles the
    merge automatically.

    Fields:
        original_query:
            The verbatim user research query.
        current_agent:
            Tracks which agent is actively executing (for observability).
        target_supply_chain_nodes:
            Decomposed supply-chain sub-domains produced by the Orchestrator.
        raw_evidence:
            Unstructured evidence strings collected by the Market Scraper.
        candidate_use_cases:
            In-progress UseCase objects being refined by Analyst + Assessor.
        final_top_5:
            Validated top-5 use cases that have passed the Red Team Critic.
        error_count:
            Cumulative retry / error counter for the Red Team loop.
            When ``error_count >= MAX_RETRIES``, the graph forces completion.
        errors:
            Detailed error / warning messages for debugging and audit trail.
    """

    original_query: str
    current_agent: AgentState
    target_supply_chain_nodes: list[str]
    raw_evidence: list[str]
    candidate_use_cases: list[UseCase]
    final_top_5: list[UseCase]
    error_count: int
    errors: list[str]
