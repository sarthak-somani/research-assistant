"""
Graph Builder
=============

Constructs and compiles the LangGraph state machine that orchestrates the
full multi-agent pipeline:

    START → orchestrator → scraper → analyst → assessor → critic
                                        ↑                     │
                                        └─── (retry loop) ←──┘

The conditional edge after the Critic implements the reflection loop:
  - If **any** use case failed AND ``error_count < MAX_RETRIES``:
    route back to the Analyst for refinement (the Critic's feedback is
    already appended to each ``UseCase.critic_feedback``).
  - Otherwise (all pass OR retries exhausted): route to END.

The compiled graph is exposed as the module-level ``app`` variable so
``main.py`` can import and invoke it directly.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from src.agents.orchestrator import run_orchestrator
from src.agents.market_scraper import run_market_scraper
from src.agents.economic_analyst import run_economic_analyst
from src.agents.risk_assessor import run_risk_assessor
from src.agents.red_team_critic import run_red_team_critic
from src.state.graph_state import GraphState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Conditional Router
# ═══════════════════════════════════════════════════════════════════════════

def route_after_critic(state: GraphState) -> str:
    """
    Conditional edge router invoked after the Red Team Critic node.

    Decision logic:
      1. If ``final_top_5`` is populated → the Critic promoted the
         candidates (either all-pass or max-retries). Route to END.
      2. If ``error_count < MAX_RETRIES`` and ``final_top_5`` is empty →
         failures exist and retries remain. Route back to **analyst** so
         the Analyst can use the Critic's feedback to refine.
      3. Safety fallback: if somehow ``error_count >= MAX_RETRIES`` but
         ``final_top_5`` is still empty, route to END anyway.

    Args:
        state: Current graph state after the Critic node.

    Returns:
        ``"analyst"`` to retry, or ``"end"`` to terminate.
    """
    from config.settings import MAX_RETRIES

    final_top_5 = state.get("final_top_5", [])
    error_count = state.get("error_count", 0)

    # Case 1: Critic already promoted candidates to final_top_5
    if final_top_5:
        logger.info(
            "Router: final_top_5 populated (%d use cases) — routing to END",
            len(final_top_5),
        )
        return "end"

    # Case 2: Retries remain → loop back to analyst
    if error_count < MAX_RETRIES:
        logger.info(
            "Router: error_count=%d < max=%d, final_top_5 empty — "
            "routing back to ANALYST for refinement",
            error_count, MAX_RETRIES,
        )
        return "analyst"

    # Case 3: Safety fallback — retries exhausted
    logger.warning(
        "Router: error_count=%d >= max=%d but final_top_5 empty — "
        "forcing END (Critic may not have promoted candidates)",
        error_count, MAX_RETRIES,
    )
    return "end"


# ═══════════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph state machine.

    Node names use the user-specified labels:
      ``orchestrator``, ``scraper``, ``analyst``, ``assessor``, ``critic``

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready for ``.invoke()``.
    """
    graph = StateGraph(GraphState)

    # ── Register Nodes ────────────────────────────────────────────────
    graph.add_node("orchestrator", run_orchestrator)
    graph.add_node("scraper", run_market_scraper)
    graph.add_node("analyst", run_economic_analyst)
    graph.add_node("assessor", run_risk_assessor)
    graph.add_node("critic", run_red_team_critic)

    # ── Linear Edge Chain ─────────────────────────────────────────────
    #    START → orchestrator → scraper → analyst → assessor → critic
    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "scraper")
    graph.add_edge("scraper", "analyst")
    graph.add_edge("analyst", "assessor")
    graph.add_edge("assessor", "critic")

    # ── Conditional Edge: Reflection / Retry Loop ─────────────────────
    #    critic → analyst (retry)  OR  critic → END
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "analyst": "analyst",
            "end": END,
        },
    )

    logger.info("Graph built: START → orchestrator → scraper → analyst → assessor → critic → (analyst | END)")

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════
# Module-Level Compiled Graph (importable by main.py)
# ═══════════════════════════════════════════════════════════════════════════

app = build_graph()
