"""
Orchestrator Agent
==================

The first node in the LangGraph pipeline. Decomposes the FMCG supply-chain
research query into 3–4 **highly specific, B2B-focused sub-domains** for the
Market Scraper to investigate.

Design Rationale:
    - The system prompt hard-codes a mandatory research node to ensure
      "Automating the Vendor Tendering and Procurement Process using Agentic
      Frameworks" is always included — this is a core requirement.
    - Surface-level / consumer-facing use cases (chatbots, marketing copy) are
      explicitly excluded to keep the research focused on operational
      bottlenecks with measurable ROI.
    - We use LangChain's ``ChatOpenAI`` with ``response_format={"type": "json_object"}``
      to get deterministic JSON output, then validate with Pydantic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_llm
from src.state.graph_state import AgentState, GraphState

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Prompt Constants
# ═══════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a Staff-level Supply-Chain Strategy Consultant with 15+ years of \
experience in FMCG / CPG enterprises (Unilever, P&G, Nestlé, Godrej).

Your mandate is to decompose an FMCG supply-chain research query into \
3–4 highly specific sub-domains where Generative AI and Agentic AI \
frameworks can create measurable operational value.

═══════════════════════════════════════════════════════════════════
HARD CONSTRAINTS — you MUST follow these:
═══════════════════════════════════════════════════════════════════

1. MANDATORY NODE: You MUST always include the following research node \
   verbatim as your FIRST entry:
   "Automating the Vendor Tendering and Procurement Process using Agentic Frameworks"

2. PRIORITISE complex, B2B operational bottlenecks — areas where GenAI \
   can reduce cycle time, cut costs, or eliminate manual toil in back-office \
   and mid-office operations. Think: procurement, demand sensing, quality \
   assurance, logistics optimisation, regulatory compliance automation.

3. IGNORE and DO NOT include surface-level, customer-facing use cases such \
   as: marketing copy generation, social-media content, customer-support \
   chatbots, personalised product recommendations, or any direct-to-consumer \
   engagement. These are well-trodden and not the focus of this research.

4. Each sub-domain must be specific enough to be a standalone research \
   project — avoid vague labels like "supply chain optimisation" or \
   "AI in operations".

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT (strict JSON):
═══════════════════════════════════════════════════════════════════

Return a JSON object with exactly one key "target_nodes", whose value is \
an array of strings. Each string is a research sub-domain title.

Example:
{
  "target_nodes": [
    "Automating the Vendor Tendering and Procurement Process using Agentic Frameworks",
    "Predictive Demand Sensing for Raw-Material Inventory Optimisation",
    "..."
  ]
}

Do NOT include any text outside the JSON object.
"""

ORCHESTRATOR_USER_PROMPT = """\
Research Query:
\"\"\"{query}\"\"\"

Decompose the above query into 3–4 highly specific FMCG supply-chain \
sub-domains for deep research. Remember: the first node MUST be \
"Automating the Vendor Tendering and Procurement Process using Agentic Frameworks".
"""

# ═══════════════════════════════════════════════════════════════════════════
# Mandatory Node (fail-safe guarantee)
# ═══════════════════════════════════════════════════════════════════════════

_MANDATORY_NODE = (
    "Automating the Vendor Tendering and Procurement Process "
    "using Agentic Frameworks"
)


# ═══════════════════════════════════════════════════════════════════════════
# LLM Factory (centralised in config.settings)
# ═══════════════════════════════════════════════════════════════════════════

def _get_llm():
    """Get the centralised LLM instance for the Orchestrator."""
    return get_llm(max_output_tokens=4096)


# ═══════════════════════════════════════════════════════════════════════════
# Node Implementation
# ═══════════════════════════════════════════════════════════════════════════

def run_orchestrator(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node: decompose the research query into supply-chain sub-domains.

    This function:
      1. Constructs a system + user prompt pair that forces the LLM to
         produce B2B-focused, operationally specific research nodes.
      2. Invokes the LLM with JSON-mode output.
      3. Parses and validates the response.
      4. Guarantees that the mandatory procurement/tendering node is present
         (fail-safe injection if the LLM omits it).
      5. Returns a partial state update for LangGraph's dict-merge.

    Args:
        state: Current LangGraph shared state (TypedDict).

    Returns:
        Dict updating ``target_supply_chain_nodes`` and ``current_agent``.
    """
    query = state.get(
        "original_query",
        "Identify the top 5 emerging GenAI use cases in FMCG supply chains.",
    )

    logger.info("Orchestrator: decomposing query into research sub-domains")
    logger.debug("Orchestrator: query = %s", query)

    # ── Build messages ────────────────────────────────────────────────
    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=ORCHESTRATOR_USER_PROMPT.format(query=query),
        ),
    ]

    # ── Invoke LLM ────────────────────────────────────────────────────
    llm = _get_llm()

    try:
        response = llm.invoke(messages)
        raw_content = response.content
        logger.debug("Orchestrator: raw LLM response = %s", raw_content)
    except Exception as exc:
        logger.error("Orchestrator: LLM invocation failed — %s", exc)
        # Graceful degradation: return the mandatory node + sensible defaults
        return {
            "target_supply_chain_nodes": [
                _MANDATORY_NODE,
                "Predictive Demand Sensing for Raw-Material Inventory Optimisation",
                "AI-Driven Quality Assurance and Defect Detection in Manufacturing Lines",
                "Intelligent Logistics Route Optimisation and Freight Consolidation",
            ],
            "current_agent": AgentState.ORCHESTRATOR,
            "errors": state.get("errors", []) + [f"Orchestrator LLM error: {exc}"],
        }

    # ── Parse JSON response ───────────────────────────────────────────
    target_nodes: list[str] = _parse_target_nodes(raw_content)

    # ── Fail-safe: guarantee mandatory node is present ────────────────
    if not any(_MANDATORY_NODE.lower() in node.lower() for node in target_nodes):
        logger.warning(
            "Orchestrator: mandatory node missing from LLM output — injecting"
        )
        target_nodes.insert(0, _MANDATORY_NODE)

    # ── Deduplicate while preserving order ────────────────────────────
    seen: set[str] = set()
    unique_nodes: list[str] = []
    for node in target_nodes:
        normalised = node.strip()
        if normalised.lower() not in seen:
            seen.add(normalised.lower())
            unique_nodes.append(normalised)

    logger.info(
        "Orchestrator: produced %d research nodes: %s",
        len(unique_nodes),
        unique_nodes,
    )

    return {
        "target_supply_chain_nodes": unique_nodes,
        "current_agent": AgentState.ORCHESTRATOR,
    }


def _parse_target_nodes(raw_content: str) -> list[str]:
    """
    Extract the ``target_nodes`` list from the LLM's JSON response.

    Handles common LLM output quirks:
      - Markdown-wrapped JSON (```json ... ```)
      - Nested vs flat structures
      - Fallback to sensible defaults on parse failure

    Args:
        raw_content: Raw string content from the LLM response.

    Returns:
        List of research node strings.
    """
    # Strip markdown code fences if present
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            "Orchestrator: failed to parse LLM JSON — %s. "
            "Falling back to defaults.",
            exc,
        )
        return [
            _MANDATORY_NODE,
            "Predictive Demand Sensing for Raw-Material Inventory Optimisation",
            "AI-Driven Quality Assurance and Defect Detection in Manufacturing Lines",
        ]

    # Accept both {"target_nodes": [...]} and bare [...]
    if isinstance(data, list):
        return [str(item) for item in data]

    if isinstance(data, dict):
        # Try common key names
        for key in ("target_nodes", "target_supply_chain_nodes", "nodes", "sub_domains"):
            if key in data and isinstance(data[key], list):
                return [str(item) for item in data[key]]

        # If dict has string values, collect them
        values = [v for v in data.values() if isinstance(v, str)]
        if values:
            return values

    logger.warning("Orchestrator: unexpected JSON structure — using defaults")
    return [
        _MANDATORY_NODE,
        "Predictive Demand Sensing for Raw-Material Inventory Optimisation",
        "AI-Driven Quality Assurance and Defect Detection in Manufacturing Lines",
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible alias for agents/__init__.py
# ═══════════════════════════════════════════════════════════════════════════
orchestrator_node = run_orchestrator
