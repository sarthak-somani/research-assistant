#!/usr/bin/env python3
"""
Integration Verification Script
================================

Runs a sequence of live health checks to verify that core integrations
(Gemini, Tavily, LangGraph) are functioning correctly.

Checks:
  1. Environment variables loaded
  2. Tavily API ping (live search)
  3. Gemini API + structured output ping
  4. Mini graph execution (constrained single-use-case query)
  5. Final output assertion (Pydantic serialisation)

Usage:
    python scripts/verify_integration.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

# ── Ensure project root is on PYTHONPATH ──────────────────────────────────
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Load env before anything else ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Console Colours (ANSI — works on Windows 10+, macOS, Linux)
# ═══════════════════════════════════════════════════════════════════════════

class C:
    """ANSI colour codes for readable terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BG_GREEN = "\033[42m"
    BG_RED   = "\033[41m"

    @staticmethod
    def ok(msg: str) -> str:
        return f"{C.GREEN}✅ {msg}{C.RESET}"

    @staticmethod
    def fail(msg: str) -> str:
        return f"{C.RED}❌ {msg}{C.RESET}"

    @staticmethod
    def warn(msg: str) -> str:
        return f"{C.YELLOW}⚠️  {msg}{C.RESET}"

    @staticmethod
    def info(msg: str) -> str:
        return f"{C.CYAN}ℹ  {msg}{C.RESET}"

    @staticmethod
    def header(msg: str) -> str:
        line = "═" * 60
        return f"\n{C.BOLD}{C.MAGENTA}{line}\n  {msg}\n{line}{C.RESET}"

    @staticmethod
    def step(num: int, total: int, msg: str) -> str:
        return f"{C.BOLD}{C.WHITE}[{num}/{total}]{C.RESET} {C.CYAN}{msg}{C.RESET}"


# Enable ANSI on Windows
if sys.platform == "win32":
    os.system("")  # Enables ANSI escape sequences in Windows terminal


# ═══════════════════════════════════════════════════════════════════════════
# Health Check Results Tracker
# ═══════════════════════════════════════════════════════════════════════════

class HealthCheckResults:
    """Tracks pass/fail for each check."""

    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, detail: str = ""):
        self.results.append((name, passed, detail))
        if passed:
            print(C.ok(f"{name}: {detail}"))
        else:
            print(C.fail(f"{name}: {detail}"))

    def summary(self):
        print(C.header("HEALTH CHECK SUMMARY"))
        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        for name, p, detail in self.results:
            status = f"{C.GREEN}PASS{C.RESET}" if p else f"{C.RED}FAIL{C.RESET}"
            print(f"  {status}  {name}")

        print()
        if passed == total:
            print(f"  {C.BG_GREEN}{C.BOLD}{C.WHITE} ALL {total} CHECKS PASSED {C.RESET}")
        else:
            print(f"  {C.BG_RED}{C.BOLD}{C.WHITE} {total - passed}/{total} CHECKS FAILED {C.RESET}")
        print()
        return passed == total


tracker = HealthCheckResults()
TOTAL_CHECKS = 5


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1: Environment Variables
# ═══════════════════════════════════════════════════════════════════════════

def check_environment():
    """Verify that required API keys are loaded."""
    print(C.step(1, TOTAL_CHECKS, "Environment Variable Check"))
    print(C.DIM + "  Checking GOOGLE_API_KEY and TAVILY_API_KEY..." + C.RESET)

    from config.settings import GOOGLE_API_KEY, TAVILY_API_KEY

    google_ok = bool(GOOGLE_API_KEY)
    tavily_ok = bool(TAVILY_API_KEY)

    if google_ok:
        masked = GOOGLE_API_KEY[:8] + "..." + GOOGLE_API_KEY[-4:]
        print(f"  GOOGLE_API_KEY: {C.GREEN}{masked}{C.RESET}")
    else:
        print(f"  GOOGLE_API_KEY: {C.RED}NOT SET{C.RESET}")

    if tavily_ok:
        masked = TAVILY_API_KEY[:8] + "..." + TAVILY_API_KEY[-4:]
        print(f"  TAVILY_API_KEY: {C.GREEN}{masked}{C.RESET}")
    else:
        print(f"  TAVILY_API_KEY: {C.RED}NOT SET{C.RESET}")

    all_ok = google_ok and tavily_ok
    tracker.record(
        "Environment Variables",
        all_ok,
        "All keys loaded" if all_ok else "Missing keys — check .env file",
    )
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2: Tavily API Ping
# ═══════════════════════════════════════════════════════════════════════════

def check_tavily():
    """Execute a live search query via Tavily."""
    print(C.step(2, TOTAL_CHECKS, "Tavily API Ping"))
    print(C.DIM + "  Querying: 'Latest FMCG supply chain news'..." + C.RESET)

    try:
        from src.tools.search import execute_research_query

        start = time.time()
        results = execute_research_query(
            query="Latest FMCG supply chain news",
            max_results=2,
        )
        elapsed = time.time() - start

        if results:
            first = results[0]
            content_preview = first.get("content", "")[:100]
            url = first.get("url", "N/A")
            print(f"  {C.GREEN}Got {len(results)} result(s) in {elapsed:.1f}s{C.RESET}")
            print(f"  URL:     {C.DIM}{url}{C.RESET}")
            print(f"  Preview: {C.DIM}{content_preview}...{C.RESET}")
            tracker.record("Tavily API", True, f"{len(results)} results in {elapsed:.1f}s")
        else:
            print(f"  {C.YELLOW}Query returned 0 results (API reachable but no data){C.RESET}")
            tracker.record("Tavily API", True, "Reachable but 0 results")

    except Exception as exc:
        print(f"  {C.RED}Error: {exc}{C.RESET}")
        tracker.record("Tavily API", False, str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 3: Gemini API + Structured Output
# ═══════════════════════════════════════════════════════════════════════════

class TestModel(BaseModel):
    """Dummy model for structured output verification."""
    status: str = Field(..., description="Should be 'OK'")
    message: str = Field(default="", description="Optional message")


def check_gemini():
    """Verify Gemini authentication and structured output enforcement."""
    print(C.step(3, TOTAL_CHECKS, "Gemini API + Structured Output Ping"))
    print(C.DIM + '  Prompting Gemini to return TestModel(status="OK")...' + C.RESET)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        from config.settings import GEMINI_MODEL, GOOGLE_API_KEY

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.0,
            google_api_key=GOOGLE_API_KEY,
        )

        structured_llm = llm.with_structured_output(TestModel)

        start = time.time()
        result: TestModel = structured_llm.invoke(
            [HumanMessage(content='Return a JSON with status="OK" and message="Gemini integration verified".')]
        )
        elapsed = time.time() - start

        print(f"  Model:   {C.GREEN}{GEMINI_MODEL}{C.RESET}")
        print(f"  Status:  {C.GREEN}{result.status}{C.RESET}")
        print(f"  Message: {C.DIM}{result.message}{C.RESET}")
        print(f"  Latency: {C.DIM}{elapsed:.1f}s{C.RESET}")
        print(f"  Type:    {C.DIM}{type(result).__name__} (Pydantic validated){C.RESET}")

        is_ok = result.status.lower() in ("ok", "200", "success")
        tracker.record(
            "Gemini Structured Output",
            is_ok,
            f"status='{result.status}' in {elapsed:.1f}s",
        )

    except Exception as exc:
        print(f"  {C.RED}Error: {exc}{C.RESET}")
        traceback.print_exc()
        tracker.record("Gemini Structured Output", False, str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 4: Mini Graph Execution
# ═══════════════════════════════════════════════════════════════════════════

def check_mini_graph():
    """Run a constrained single-use-case query through the full pipeline."""
    print(C.step(4, TOTAL_CHECKS, "Mini Graph Execution"))
    print(C.DIM + "  Running constrained query through full pipeline..." + C.RESET)
    print(C.DIM + '  Query: "Identify ONE emerging GenAI use case in inventory management."' + C.RESET)
    print()

    try:
        from src.graph.builder import build_graph

        graph = build_graph()

        initial_state = {
            "original_query": (
                "Identify ONE emerging GenAI use case in inventory management."
            ),
            "target_supply_chain_nodes": [],
            "raw_evidence": [],
            "candidate_use_cases": [],
            "final_top_5": [],
            "error_count": 0,
            "errors": [],
        }

        print(f"  {C.MAGENTA}▶ Pipeline started...{C.RESET}")
        start = time.time()

        # Stream events to show agent progression
        agent_seen = set()
        final_state = None

        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name not in agent_seen:
                    agent_seen.add(node_name)
                    agent_label = node_name.upper()
                    elapsed_so_far = time.time() - start
                    print(f"  {C.CYAN}  → {agent_label}{C.RESET} "
                          f"{C.DIM}(+{elapsed_so_far:.1f}s){C.RESET}")

                    # Show intermediate state details
                    if "target_supply_chain_nodes" in node_output:
                        nodes = node_output["target_supply_chain_nodes"]
                        print(f"    {C.DIM}Supply-chain nodes: {len(nodes)}{C.RESET}")
                        for n in nodes[:3]:
                            print(f"    {C.DIM}  • {n[:60]}{C.RESET}")

                    if "raw_evidence" in node_output:
                        evidence = node_output["raw_evidence"]
                        print(f"    {C.DIM}Evidence blocks: {len(evidence)}{C.RESET}")

                    if "candidate_use_cases" in node_output:
                        candidates = node_output["candidate_use_cases"]
                        print(f"    {C.DIM}Candidate use cases: {len(candidates)}{C.RESET}")
                        for uc in candidates[:3]:
                            topic = getattr(uc, "topic", str(uc)[:50])
                            print(f"    {C.DIM}  • {topic}{C.RESET}")

                    if "final_top_5" in node_output:
                        final = node_output["final_top_5"]
                        if final:
                            print(f"    {C.GREEN}Final top-5: {len(final)} use cases{C.RESET}")

                    if "errors" in node_output and node_output["errors"]:
                        for err in node_output["errors"][-2:]:
                            print(f"    {C.YELLOW}⚠ {err[:80]}{C.RESET}")

                # Keep track of latest state
                if final_state is None:
                    final_state = {}
                final_state.update(node_output)

        total_time = time.time() - start
        print(f"\n  {C.MAGENTA}■ Pipeline complete in {total_time:.1f}s{C.RESET}")

        # Store for check 5
        check_mini_graph._final_state = final_state
        check_mini_graph._elapsed = total_time

        tracker.record(
            "Mini Graph Execution",
            final_state is not None,
            f"Completed pipeline in {total_time:.1f}s through {len(agent_seen)} agents",
        )

    except Exception as exc:
        print(f"  {C.RED}Error: {exc}{C.RESET}")
        traceback.print_exc()
        tracker.record("Mini Graph Execution", False, str(exc))
        check_mini_graph._final_state = None


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 5: Output Assertion (Pydantic Serialisation)
# ═══════════════════════════════════════════════════════════════════════════

def check_output_assertion():
    """Verify that the pipeline output contains valid Pydantic UseCase objects."""
    print(C.step(5, TOTAL_CHECKS, "Output Assertion — Pydantic Serialisation"))

    final_state = getattr(check_mini_graph, "_final_state", None)

    if final_state is None:
        tracker.record("Output Assertion", False, "No final state — graph didn't run")
        return

    # Get the best available use cases
    use_cases = final_state.get("final_top_5", [])
    if not use_cases:
        use_cases = final_state.get("candidate_use_cases", [])

    if not use_cases:
        print(f"  {C.YELLOW}No use cases in output — pipeline may have errored{C.RESET}")
        errors = final_state.get("errors", [])
        for e in errors[:3]:
            print(f"  {C.RED}  Error: {e[:100]}{C.RESET}")
        tracker.record("Output Assertion", False, "No use cases produced")
        return

    print(f"  {C.GREEN}Found {len(use_cases)} use case(s){C.RESET}\n")

    from src.state.graph_state import UseCase

    all_valid = True
    for i, uc in enumerate(use_cases, 1):
        if isinstance(uc, UseCase):
            # Verify serialisation roundtrip
            try:
                json_str = uc.model_dump_json(indent=2)
                restored = UseCase.model_validate_json(json_str)
                data = json.loads(json_str)

                print(f"  {C.BOLD}Use Case {i}: {uc.topic}{C.RESET}")
                print(f"  {C.DIM}├── Segment:  {uc.supply_chain_segment}{C.RESET}")
                print(f"  {C.DIM}├── Maturity:  {uc.maturity_level.value}{C.RESET}")

                if uc.economic_impact:
                    print(f"  {C.DIM}├── ROI:       {uc.economic_impact.estimated_roi_percentage}%{C.RESET}")
                    print(f"  {C.DIM}├── Cost:      {uc.economic_impact.implementation_cost_complexity.value}{C.RESET}")

                if uc.risk_assessment:
                    print(f"  {C.DIM}├── Bottleneck: {uc.risk_assessment.primary_bottleneck[:60]}...{C.RESET}")

                verdict = uc.critic_verdict.value if uc.critic_verdict else "N/A"
                feedback_count = len(uc.critic_feedback)
                print(f"  {C.DIM}├── Verdict:   {verdict} ({feedback_count} feedback rounds){C.RESET}")
                print(f"  {C.DIM}└── JSON size: {len(json_str)} bytes{C.RESET}")
                print(f"  {C.GREEN}    ✓ Pydantic serialisation roundtrip OK{C.RESET}")
                print()

            except Exception as exc:
                print(f"  {C.RED}Use Case {i}: Serialisation FAILED — {exc}{C.RESET}")
                all_valid = False
        else:
            print(f"  {C.RED}Use Case {i}: Not a UseCase instance (type={type(uc).__name__}){C.RESET}")
            all_valid = False

    tracker.record(
        "Output Assertion",
        all_valid,
        f"{len(use_cases)} UseCase(s) validated, Pydantic roundtrip OK",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(C.header("INTEGRATION VERIFICATION — FMCG GenAI Research Assistant"))
    print(f"  {C.DIM}Project: {PROJECT_ROOT}{C.RESET}")
    print(f"  {C.DIM}Python:  {sys.version.split()[0]}{C.RESET}")
    print()

    overall_start = time.time()

    # ── Run checks sequentially ───────────────────────────────────────
    print(C.header("CHECK 1/5 — ENVIRONMENT"))
    env_ok = check_environment()
    print()

    if not env_ok:
        print(C.fail("Cannot proceed without API keys. Skipping live checks."))
        print(C.info("Set GOOGLE_API_KEY and TAVILY_API_KEY in your .env file."))
        tracker.summary()
        sys.exit(1)

    print(C.header("CHECK 2/5 — TAVILY API"))
    check_tavily()
    print()

    print(C.header("CHECK 3/5 — GEMINI API + STRUCTURED OUTPUT"))
    check_gemini()
    print()

    print(C.header("CHECK 4/5 — MINI GRAPH EXECUTION"))
    check_mini_graph()
    print()

    print(C.header("CHECK 5/5 — OUTPUT ASSERTION"))
    check_output_assertion()
    print()

    # ── Summary ───────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    all_passed = tracker.summary()
    print(f"  {C.DIM}Total time: {total_time:.1f}s{C.RESET}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
