"""
GenAI FMCG Supply Chain Research Assistant — Entry Point
========================================================

Builds and invokes the LangGraph multi-agent pipeline to research
emerging GenAI use cases in FMCG supply chains and produce a validated
JSON report + professional PDF.

Usage:
    python main.py
    python main.py --query "Your custom research query"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import OUTPUT_DIR
from src.graph.builder import build_graph
from src.utils.logger import setup_logging, get_logger
from src.utils.pdf_generator import generate_fmcg_report


_DEFAULT_QUERY = (
    "Identify the top 5 emerging GenAI use cases in FMCG supply chains. "
    "For each use case, describe implementation approach, expected impact, "
    "risks, and maturity level."
)


def _get_verdict(uc) -> str:
    """Safely extract critic_verdict as a plain string (handles enums, dicts, strings)."""
    raw = getattr(uc, "critic_verdict", None) if hasattr(uc, "critic_verdict") else None
    if raw is None and isinstance(uc, dict):
        raw = uc.get("critic_verdict")
    if raw is None:
        return ""
    return raw.value if hasattr(raw, "value") else str(raw)


def main() -> None:
    """Run the multi-agent research pipeline."""
    # --- Parse CLI args ---
    parser = argparse.ArgumentParser(
        description="GenAI FMCG Supply Chain Research Assistant",
    )
    parser.add_argument(
        "--query", "-q",
        default=_DEFAULT_QUERY,
        help="Custom research query (default: top 5 GenAI FMCG use cases)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger("main")

    logger.info("=" * 60)
    logger.info("GenAI FMCG Supply Chain Research Assistant")
    logger.info("=" * 60)

    # --- Build the graph ---
    logger.info("Building LangGraph state machine...")
    graph = build_graph()
    logger.info("Graph compiled successfully")

    # --- Initialise state (TypedDict — just a plain dict) ---
    initial_state = {
        "original_query": args.query,
        "target_supply_chain_nodes": [],
        "raw_evidence": [],
        "candidate_use_cases": [],
        "final_top_5": [],
        "error_count": 0,
        "errors": [],
    }

    # --- Execute the pipeline (streaming for per-node visibility) ---
    logger.info("Invoking pipeline...")
    try:
        final_state: dict = dict(initial_state)

        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue
                logger.info("Node [%s] completed", node_name)

                # Per-node detail logging
                if node_name == "orchestrator":
                    nodes = node_output.get("target_supply_chain_nodes", [])
                    logger.info("  Identified %d supply-chain sub-domains", len(nodes))
                elif node_name == "scraper":
                    evidence = node_output.get("raw_evidence", [])
                    logger.info("  Collected %d evidence blocks", len(evidence))
                elif node_name == "analyst":
                    cases = node_output.get("candidate_use_cases", [])
                    logger.info("  Generated %d candidate use cases", len(cases))
                elif node_name == "assessor":
                    cases = node_output.get("candidate_use_cases", [])
                    logger.info("  Risk-assessed %d use cases", len(cases))
                elif node_name == "critic":
                    cases = node_output.get("candidate_use_cases", [])
                    final = node_output.get("final_top_5", [])
                    passed = sum(1 for c in cases if _get_verdict(c) == "pass")
                    logger.info(
                        "  Passed: %d | Failed: %d | error_count: %d",
                        passed, len(cases) - passed,
                        node_output.get("error_count", 0),
                    )
                    if final:
                        logger.info("  Critic promoted candidates — pipeline complete")

                final_state.update(node_output)

    except NotImplementedError as e:
        logger.error("Agent not yet implemented: %s", e)
        logger.info(
            "This is expected — agent logic has not been written yet. "
            "The project foundation is set up and ready for implementation."
        )
        sys.exit(0)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)

    # --- Extract final use cases ---
    final_use_cases = final_state.get("final_top_5", [])
    if not final_use_cases:
        final_use_cases = final_state.get("candidate_use_cases", [])

    # --- Serialise UseCase objects ---
    use_cases_serialised = []
    for uc in final_use_cases:
        if hasattr(uc, "model_dump"):
            use_cases_serialised.append(uc.model_dump(mode="json"))
        elif isinstance(uc, dict):
            use_cases_serialised.append(uc)
        else:
            use_cases_serialised.append(str(uc))

    # --- Build JSON report ---
    all_passed = all(
        _get_verdict(uc) == "pass" for uc in final_use_cases
    ) if final_use_cases else False

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"report_{timestamp}.json"

    report = {
        "report_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "topic": "Emerging GenAI Use Cases in FMCG Supply Chains",
            "total_use_cases": len(use_cases_serialised),
            "validation_status": "all_passed" if all_passed else "partial",
            "error_count": final_state.get("error_count", 0),
            "errors": final_state.get("errors", []),
        },
        "use_cases": use_cases_serialised,
    }

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("JSON report saved to %s", json_path)

    # --- Generate PDF report ---
    if final_use_cases:
        try:
            pdf_path = generate_fmcg_report(final_use_cases, filename=f"report_{timestamp}.pdf")
            logger.info("PDF report saved to %s", pdf_path)
        except Exception as pdf_err:
            logger.error("PDF generation failed: %s", pdf_err)

    logger.info("Total use cases: %d", len(use_cases_serialised))
    logger.info("Done!")


if __name__ == "__main__":
    main()
