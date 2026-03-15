"""
Market Scraper Agent
====================

Takes the ``target_supply_chain_nodes`` produced by the Orchestrator and,
for each node, executes targeted web searches via Tavily, then uses
Gemini (``ChatGoogleGenerativeAI``) to synthesise the raw results into
structured, quantitative evidence strings.

Design Rationale:
    - Two-stage pipeline per node: *Search → LLM Synthesis*.
    - The synthesis prompt is aggressively tuned to extract **quantitative
      statistical evidence** (ROI %, CapEx vs OpEx, marginal efficiency
      gains) and reject vendor marketing fluff.
    - Each evidence string is self-contained: it includes the data point,
      source URL, and contextual explanation so downstream agents can
      operate without re-searching.
    - Errors on individual nodes are isolated — a failure on one node does
      not block the others (graceful degradation).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_llm, LLM_TEMPERATURE
from src.state.graph_state import AgentState, GraphState
from src.tools.search import execute_research_query

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Environment Setup
# ═══════════════════════════════════════════════════════════════════════════
# Environment setup handled centrally by config.settings.get_llm()

# ═══════════════════════════════════════════════════════════════════════════
# Prompt Constants
# ═══════════════════════════════════════════════════════════════════════════

SCRAPER_SYSTEM_PROMPT = """\
You are a Senior Research Analyst at a top-tier management consulting firm \
(McKinsey, BCG, Bain) specialising in FMCG / CPG supply-chain transformation.

You have been given raw web search results about a specific GenAI use case \
in FMCG supply chains. Your job is to extract and synthesise ONLY the \
high-value, quantitative evidence from these results.

═══════════════════════════════════════════════════════════════════
EXTRACTION RULES — follow these STRICTLY:
═══════════════════════════════════════════════════════════════════

1. HUNT for QUANTITATIVE STATISTICAL EVIDENCE:
   - ROI percentages (e.g., "achieved 23% ROI within 18 months")
   - CapEx vs OpEx tradeoffs (e.g., "$500K upfront, $50K/year run-rate")
   - Marginal efficiency gains (e.g., "reduced procurement cycle from \
     14 days to 3 days", "cut error rates by 67%")
   - Adoption statistics (e.g., "37% of FMCG firms surveyed are piloting")
   - Cost savings (e.g., "saved $2.3M annually in logistics overhead")

2. For EVERY data point you extract, you MUST cite the source URL.

3. IGNORE and DO NOT include:
   - Vendor marketing claims without supporting data
   - Vague qualitative statements ("AI is transforming supply chains")
   - Product announcements without implementation evidence
   - Analyst predictions without methodology ("AI market will reach $X by 2030")

4. If the search results contain NO quantitative evidence for this topic, \
   honestly state: "INSUFFICIENT QUANTITATIVE DATA — only qualitative \
   references found" and summarise the best qualitative findings available.

5. Structure your output as numbered evidence items, each 2–4 sentences.

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════════

Return ONLY the extracted evidence as a numbered list. Example format:

1. [SOURCE: url] Finding: Unilever reported a 28% reduction in procurement \
   cycle time after deploying an agentic AI system for vendor tendering. \
   Implementation cost was approximately $1.2M with a 14-month payback period.

2. [SOURCE: url] Finding: A BCG study across 15 FMCG companies found that \
   GenAI-powered demand sensing reduced safety stock by 18-25%, translating \
   to $3-8M annual working capital release per $1B revenue.

Do NOT add any preamble, summary, or conclusion. Only the evidence items.
"""

SCRAPER_USER_PROMPT = """\
RESEARCH NODE: "{node_title}"

Below are the raw web search results. Extract all quantitative evidence \
relevant to this GenAI use case in FMCG supply chains.

═══════════════════════════════════════════════════════════════════
RAW SEARCH RESULTS:
═══════════════════════════════════════════════════════════════════

{search_results_text}

═══════════════════════════════════════════════════════════════════

Extract the quantitative statistical evidence. Cite every source URL.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Search Query Templates (per-node specialised queries)
# ═══════════════════════════════════════════════════════════════════════════

_SEARCH_QUERY_TEMPLATES = [
    '"{node}" GenAI FMCG supply chain case study ROI statistics',
    '"{node}" artificial intelligence implementation cost savings FMCG CPG',
    '"{node}" agentic AI enterprise procurement logistics efficiency gains data',
]


# ═══════════════════════════════════════════════════════════════════════════
# LLM Factory
# ═══════════════════════════════════════════════════════════════════════════

def _get_llm():
    """
    Get the centralised LLM instance for evidence synthesis.

    Returns:
        Configured LLM chat model instance.
    """
    return get_llm(max_output_tokens=4096)


# ═══════════════════════════════════════════════════════════════════════════
# Node Implementation
# ═══════════════════════════════════════════════════════════════════════════

def run_market_scraper(state: GraphState) -> dict[str, Any]:
    """
    LangGraph node: research each supply-chain sub-domain via web search
    and synthesise quantitative evidence using Gemini.

    Pipeline per node:
      1. Generate 3 specialised search queries from the node title.
      2. Execute each query via Tavily (``execute_research_query``).
      3. Aggregate and deduplicate raw search results.
      4. Feed results into Gemini with a strict extraction prompt.
      5. Append the synthesised evidence string to ``raw_evidence``.

    Errors on individual nodes are isolated — the function continues to
    the next node and logs the failure.

    Args:
        state: Current LangGraph shared state (TypedDict).

    Returns:
        Dict updating ``raw_evidence``, ``current_agent``, and
        optionally ``errors``.
    """
    target_nodes: list[str] = state.get("target_supply_chain_nodes", [])
    existing_evidence: list[str] = state.get("raw_evidence", [])
    existing_errors: list[str] = state.get("errors", [])

    if not target_nodes:
        logger.warning("Market Scraper: no target nodes to research")
        return {
            "raw_evidence": existing_evidence,
            "current_agent": AgentState.MARKET_SCRAPER,
            "errors": existing_errors + ["Market Scraper: no target nodes provided"],
        }

    logger.info(
        "Market Scraper: researching %d supply-chain nodes", len(target_nodes)
    )

    llm = _get_llm()
    new_evidence: list[str] = []
    new_errors: list[str] = []

    for i, node_title in enumerate(target_nodes, 1):
        logger.info(
            "Market Scraper [%d/%d]: researching '%s'",
            i, len(target_nodes), node_title,
        )

        try:
            evidence_str = _research_single_node(
                node_title=node_title,
                llm=llm,
            )
            if evidence_str:
                # Tag with a header so downstream agents know the source node
                tagged_evidence = (
                    f"══ EVIDENCE FOR: {node_title} ══\n"
                    f"{evidence_str}\n"
                    f"══ END EVIDENCE ══"
                )
                new_evidence.append(tagged_evidence)
                logger.info(
                    "Market Scraper [%d/%d]: extracted %d chars of evidence for '%s'",
                    i, len(target_nodes), len(evidence_str), node_title,
                )
            else:
                msg = f"Market Scraper: no evidence extracted for '{node_title}'"
                logger.warning(msg)
                new_errors.append(msg)

        except Exception as exc:
            msg = f"Market Scraper: error researching '{node_title}': {exc}"
            logger.error(msg, exc_info=True)
            new_errors.append(msg)
            # Continue to next node — don't let one failure block all research

    logger.info(
        "Market Scraper: completed — %d evidence blocks collected, %d errors",
        len(new_evidence), len(new_errors),
    )

    return {
        "raw_evidence": existing_evidence + new_evidence,
        "current_agent": AgentState.MARKET_SCRAPER,
        "errors": existing_errors + new_errors if new_errors else existing_errors,
    }


def _research_single_node(
    node_title: str,
    llm,
) -> str:
    """
    Execute the full search → synthesis pipeline for a single research node.

    Args:
        node_title: The supply-chain sub-domain to research.
        llm: Pre-configured Gemini LLM instance.

    Returns:
        Synthesised evidence string, or empty string if nothing found.
    """
    # ── Step 1: Generate & execute search queries ─────────────────────
    all_results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for template in _SEARCH_QUERY_TEMPLATES:
        query = template.format(node=node_title)
        results = execute_research_query(query=query, max_results=3)

        for result in results:
            url = result.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            all_results.append(result)

    if not all_results:
        logger.warning(
            "Market Scraper: zero search results for '%s'", node_title
        )
        return ""

    logger.debug(
        "Market Scraper: %d unique results for '%s'",
        len(all_results), node_title,
    )

    # ── Step 2: Format search results for the LLM ────────────────────
    search_results_text = _format_search_results(all_results)

    # ── Step 3: Synthesise via Gemini ─────────────────────────────────
    messages = [
        SystemMessage(content=SCRAPER_SYSTEM_PROMPT),
        HumanMessage(
            content=SCRAPER_USER_PROMPT.format(
                node_title=node_title,
                search_results_text=search_results_text,
            ),
        ),
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        logger.error(
            "Market Scraper: Gemini synthesis failed for '%s': %s",
            node_title, exc,
        )
        # Fallback: return raw results as plain text evidence
        return _fallback_raw_evidence(node_title, all_results)


def _format_search_results(results: list[dict[str, Any]]) -> str:
    """
    Format raw search results into a clean text block for the LLM prompt.

    Args:
        results: List of normalised search result dicts.

    Returns:
        Formatted multi-line string.
    """
    lines: list[str] = []

    for i, result in enumerate(results, 1):
        url = result.get("url", "N/A")
        title = result.get("title", "Untitled")
        content = result.get("content", "")

        # Truncate very long content to stay within context limits
        if len(content) > 2000:
            content = content[:2000] + " [... truncated]"

        lines.append(
            f"--- Result {i} ---\n"
            f"URL: {url}\n"
            f"Title: {title}\n"
            f"Content:\n{content}\n"
        )

    return "\n".join(lines)


def _fallback_raw_evidence(
    node_title: str,
    results: list[dict[str, Any]],
) -> str:
    """
    Construct a fallback evidence string from raw search results when the
    LLM synthesis fails. Ensures we don't lose data on API errors.

    Args:
        node_title: The research node title.
        results: Raw search results.

    Returns:
        Plain-text evidence string.
    """
    lines = [f"[FALLBACK — LLM synthesis unavailable for '{node_title}']"]

    for i, result in enumerate(results[:5], 1):
        url = result.get("url", "N/A")
        content = result.get("content", "")[:500]
        lines.append(f"{i}. [SOURCE: {url}] {content}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible alias for agents/__init__.py
# ═══════════════════════════════════════════════════════════════════════════
market_scraper_node = run_market_scraper
