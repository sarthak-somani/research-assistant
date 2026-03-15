"""Quick sequential integration checks — plain output, no ANSI."""
import sys, time, json, traceback
sys.path.insert(0, ".")

from config.settings import GOOGLE_API_KEY, TAVILY_API_KEY, GEMINI_MODEL

# ── CHECK 1: Env ──────────────────────────────────────────────────────
print("=" * 60)
print("CHECK 1/5: Environment Variables")
print("=" * 60)
g_ok = bool(GOOGLE_API_KEY)
t_ok = bool(TAVILY_API_KEY)
print(f"  GOOGLE_API_KEY: {'SET (' + GOOGLE_API_KEY[:8] + '...)' if g_ok else 'NOT SET'}")
print(f"  TAVILY_API_KEY: {'SET (' + TAVILY_API_KEY[:8] + '...)' if t_ok else 'NOT SET'}")
print(f"  >> {'PASS' if (g_ok and t_ok) else 'FAIL'}")
if not (g_ok and t_ok):
    print("Cannot proceed without keys. Exiting.")
    sys.exit(1)

# ── CHECK 2: Tavily ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 2/5: Tavily API Ping")
print("=" * 60)
try:
    from src.tools.search import execute_research_query
    start = time.time()
    results = execute_research_query("Latest FMCG supply chain news 2025", max_results=2)
    elapsed = time.time() - start
    print(f"  Results: {len(results)} in {elapsed:.1f}s")
    for i, r in enumerate(results[:2], 1):
        print(f"  [{i}] URL: {r.get('url', 'N/A')[:80]}")
        print(f"      Content: {r.get('content', '')[:100]}...")
    print(f"  >> PASS")
except Exception as e:
    print(f"  Error: {e}")
    traceback.print_exc()
    print(f"  >> FAIL")

# ── CHECK 3: Gemini Structured Output ─────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 3/5: Gemini API + Structured Output")
print("=" * 60)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel, Field

    class TestModel(BaseModel):
        status: str = Field(..., description="Should be OK")
        message: str = Field(default="", description="Optional message")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
    )
    structured_llm = llm.with_structured_output(TestModel)
    start = time.time()
    result = structured_llm.invoke(
        [HumanMessage(content='Return JSON with status="OK" and message="integration verified".')]
    )
    elapsed = time.time() - start
    print(f"  Model:   {GEMINI_MODEL}")
    print(f"  Status:  {result.status}")
    print(f"  Message: {result.message}")
    print(f"  Latency: {elapsed:.1f}s")
    print(f"  Type:    {type(result).__name__}")
    print(f"  >> PASS")
except Exception as e:
    print(f"  Error: {e}")
    traceback.print_exc()
    print(f"  >> FAIL")

# ── CHECK 4: Mini Graph Execution ─────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 4/5: Mini Graph Execution")
print("  This will take several minutes (multiple API calls)...")
print("=" * 60)
final_state = None
try:
    from src.graph.builder import build_graph
    graph = build_graph()

    initial_state = {
        "original_query": "Identify ONE emerging GenAI use case in inventory management.",
        "target_supply_chain_nodes": [],
        "raw_evidence": [],
        "candidate_use_cases": [],
        "final_top_5": [],
        "error_count": 0,
        "errors": [],
    }

    start = time.time()
    agent_seen = set()

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name not in agent_seen:
                agent_seen.add(node_name)
                elapsed_so_far = time.time() - start
                print(f"  >> {node_name.upper()} (+{elapsed_so_far:.1f}s)")

                if "target_supply_chain_nodes" in node_output:
                    nodes = node_output["target_supply_chain_nodes"]
                    print(f"     Supply-chain nodes: {len(nodes)}")
                    for n in nodes[:3]:
                        print(f"       - {n[:70]}")

                if "raw_evidence" in node_output:
                    evidence = node_output["raw_evidence"]
                    print(f"     Evidence blocks: {len(evidence)}")

                if "candidate_use_cases" in node_output:
                    candidates = node_output["candidate_use_cases"]
                    print(f"     Candidate use cases: {len(candidates)}")
                    for uc in candidates[:3]:
                        topic = getattr(uc, "topic", str(uc)[:50])
                        print(f"       - {topic}")

                if "final_top_5" in node_output:
                    final = node_output["final_top_5"]
                    if final:
                        print(f"     FINAL top-5: {len(final)} use cases")

                if "errors" in node_output and node_output["errors"]:
                    for err in node_output["errors"][-2:]:
                        print(f"     WARNING: {err[:100]}")

            final_state = {}
            final_state.update(node_output)

    total_time = time.time() - start
    print(f"\n  Pipeline complete in {total_time:.1f}s ({len(agent_seen)} agents)")
    print(f"  >> PASS")

except Exception as e:
    print(f"  Error: {e}")
    traceback.print_exc()
    print(f"  >> FAIL")

# ── CHECK 5: Output Assertion ─────────────────────────────────────────
print("\n" + "=" * 60)
print("CHECK 5/5: Output Assertion (Pydantic Serialisation)")
print("=" * 60)
try:
    if final_state is None:
        print("  No final state — graph did not complete")
        print(f"  >> FAIL")
    else:
        from src.state.graph_state import UseCase
        use_cases = final_state.get("final_top_5", [])
        if not use_cases:
            use_cases = final_state.get("candidate_use_cases", [])

        if not use_cases:
            print("  No use cases produced")
            errors = final_state.get("errors", [])
            for e in errors[:3]:
                print(f"    Error: {e[:120]}")
            print(f"  >> FAIL")
        else:
            print(f"  Found {len(use_cases)} use case(s)")
            all_valid = True
            for i, uc in enumerate(use_cases, 1):
                if isinstance(uc, UseCase):
                    json_str = uc.model_dump_json(indent=2)
                    restored = UseCase.model_validate_json(json_str)
                    print(f"\n  [{i}] {uc.topic}")
                    print(f"      Segment:   {uc.supply_chain_segment}")
                    print(f"      Maturity:  {uc.maturity_level.value}")
                    if uc.economic_impact:
                        print(f"      ROI:       {uc.economic_impact.estimated_roi_percentage}%")
                        print(f"      Cost:      {uc.economic_impact.implementation_cost_complexity.value}")
                    if uc.risk_assessment:
                        print(f"      Bottleneck: {uc.risk_assessment.primary_bottleneck[:70]}...")
                    print(f"      Verdict:   {uc.critic_verdict.value} ({len(uc.critic_feedback)} feedback rounds)")
                    print(f"      JSON size: {len(json_str)} bytes")
                    print(f"      Roundtrip: OK")
                else:
                    print(f"  [{i}] NOT a UseCase instance (type={type(uc).__name__})")
                    all_valid = False

            print(f"\n  >> {'PASS' if all_valid else 'FAIL'}")

except Exception as e:
    print(f"  Error: {e}")
    traceback.print_exc()
    print(f"  >> FAIL")

print("\n" + "=" * 60)
print("ALL CHECKS COMPLETE")
print("=" * 60)
