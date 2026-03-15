# Architecture and Data Flow

## Agent Architecture
The core of the system is built on a cyclic Directed Acyclic Graph (DAG), utilizing five specifically tailored agents. Each agent has dedicated component responsibilities:

1. **[Orchestrator](../src/agents/orchestrator.py)**: Acts as the central node managing the overall execution flow.
2. **[Market Scraper](../src/agents/market_scraper.py)**: Responsible for external data acquisition via search APIs.
3. **[Economic Analyst](../src/agents/economic_analyst.py)**: Processes raw data to extract specific microeconomic implications.
4. **[Risk Assessor](../src/agents/risk_assessor.py)**: Identifies integration hurdles, security implications, and enterprise constraints.
5. **[Red Team Critic](../src/agents/red_team_critic.py)**: Serves as the adversarial check within the cyclic loop, actively hunting for unverified claims or hallucinated metrics.

---

## 1. High-Level System Architecture

The system is a multi-agent research pipeline built on **LangGraph**, orchestrated through a compiled state machine. All agents share a strict **Pydantic TypedDict** state contract and communicate via Gemini LLM calls with structured output enforcement.

```mermaid
graph TB
    subgraph "Entry Points"
        CLI["🖥️ CLI<br/><code>main.py</code>"]
        UI["🌐 Streamlit UI<br/><code>src/ui/app.py</code>"]
    end

    subgraph "Configuration Layer"
        ENV[".env File"]
        CFG["⚙️ config/settings.py<br/>─────────────────<br/>• LLM Provider Selection<br/>• API Key Management<br/>• Search Config<br/>• Agent Parameters"]
    end

    subgraph "LangGraph State Machine"
        direction TB
        GRAPH["🔄 Graph Builder<br/><code>src/graph/builder.py</code><br/>─────────────────<br/>Compiles StateGraph → app"]
    end

    subgraph "Agent Layer"
        direction TB
        ORCH["🎯 Orchestrator"]
        SCRP["🔍 Market Scraper"]
        ANAL["📊 Economic Analyst"]
        RISK["⚠️ Risk Assessor"]
        CRIT["🛡️ Red Team Critic"]
    end

    subgraph "External Services"
        GEMINI["🤖 Gemini 2.5 Pro<br/>(Vertex AI)"]
        TAVILY["🌍 Tavily Search API<br/>(Advanced Depth)"]
    end

    subgraph "Data Layer"
        STATE["📋 GraphState<br/><code>src/state/graph_state.py</code>"]
        MODELS["📐 Pydantic Models<br/>UseCase • EconomicImpact<br/>RiskAssessment • Enums"]
    end

    subgraph "Output Layer"
        JSON_OUT["📄 JSON Report"]
        PDF_OUT["📑 PDF Report"]
        LOG_OUT["📝 Text Log"]
    end

    CLI --> GRAPH
    UI --> GRAPH
    ENV --> CFG
    CFG --> GRAPH
    CFG --> GEMINI
    CFG --> TAVILY
    GRAPH --> ORCH & SCRP & ANAL & RISK & CRIT
    ORCH & ANAL & RISK & CRIT --> GEMINI
    SCRP --> TAVILY
    SCRP --> GEMINI
    STATE --> MODELS
    GRAPH --> STATE
    GRAPH --> JSON_OUT & PDF_OUT & LOG_OUT

    style CLI fill:#1e3a5f,stroke:#4a90d9,color:#fff
    style UI fill:#1e3a5f,stroke:#4a90d9,color:#fff
    style GEMINI fill:#0d47a1,stroke:#42a5f5,color:#fff
    style TAVILY fill:#0d47a1,stroke:#42a5f5,color:#fff
    style GRAPH fill:#1b5e20,stroke:#66bb6a,color:#fff
    style STATE fill:#4a148c,stroke:#ab47bc,color:#fff
    style MODELS fill:#4a148c,stroke:#ab47bc,color:#fff
```

---

## 2. LangGraph Pipeline Flow

This diagram shows the exact execution order of the compiled state machine, including the conditional reflection/retry loop after the Critic node.

```mermaid
flowchart TD
    START(("▶ START"))
    ORCH["🎯 <b>Orchestrator</b><br/>──────────────<br/>Decomposes query into<br/>3–4 supply-chain<br/>research sub-domains"]
    SCRP["🔍 <b>Market Scraper</b><br/>──────────────<br/>3 Tavily queries per node<br/>→ LLM synthesis<br/>→ evidence strings"]
    ANAL["📊 <b>Economic Analyst</b><br/>──────────────<br/>Synthesises evidence into<br/>5 UseCase objects with<br/>quantitative ROI data"]
    RISK["⚠️ <b>Risk Assessor</b><br/>──────────────<br/>Enriches each UseCase<br/>with risk_assessment +<br/>implementation_approach"]
    CRIT["🛡️ <b>Red Team Critic</b><br/>──────────────<br/>3-axis adversarial<br/>evaluation per UseCase"]
    
    ROUTER{"🔀 <b>route_after_critic</b><br/>─────────────────<br/>final_top_5 populated?<br/>OR error_count ≥ MAX?"}
    
    FINISH(("⏹ END"))

    START --> ORCH
    ORCH -->|"target_supply_chain_nodes"| SCRP
    SCRP -->|"raw_evidence"| ANAL
    ANAL -->|"candidate_use_cases"| RISK
    RISK -->|"enriched candidates"| CRIT
    CRIT --> ROUTER

    ROUTER -->|"✅ All Pass OR<br/>Max Retries Reached<br/>→ Promote to final_top_5"| FINISH
    ROUTER -->|"❌ Failures Exist<br/>AND error_count < MAX_RETRIES<br/>→ Retry with feedback"| ANAL

    style START fill:#2e7d32,stroke:#66bb6a,color:#fff
    style FINISH fill:#c62828,stroke:#ef5350,color:#fff
    style ORCH fill:#1565c0,stroke:#42a5f5,color:#fff
    style SCRP fill:#6a1b9a,stroke:#ab47bc,color:#fff
    style ANAL fill:#ef6c00,stroke:#ffa726,color:#fff
    style RISK fill:#d84315,stroke:#ff7043,color:#fff
    style CRIT fill:#283593,stroke:#5c6bc0,color:#fff
    style ROUTER fill:#37474f,stroke:#78909c,color:#fff
```

---

## 3. State Data Flow Between Agents

Shows the exact state keys produced and consumed by each agent, illustrating the immutable, append-only state management pattern.

```mermaid
flowchart LR
    subgraph "🎯 Orchestrator"
        O_IN["<b>Reads:</b><br/>original_query"]
        O_OUT["<b>Writes:</b><br/>target_supply_chain_nodes<br/>current_agent"]
    end

    subgraph "🔍 Market Scraper"
        S_IN["<b>Reads:</b><br/>target_supply_chain_nodes"]
        S_OUT["<b>Writes:</b><br/>raw_evidence<br/>current_agent"]
    end

    subgraph "📊 Economic Analyst"
        A_IN["<b>Reads:</b><br/>raw_evidence<br/>target_supply_chain_nodes<br/>candidate_use_cases ⁱ<br/>error_count ⁱ"]
        A_OUT["<b>Writes:</b><br/>candidate_use_cases<br/>current_agent"]
    end

    subgraph "⚠️ Risk Assessor"
        R_IN["<b>Reads:</b><br/>candidate_use_cases<br/>raw_evidence"]
        R_OUT["<b>Writes:</b><br/>candidate_use_cases ↻<br/>current_agent"]
    end

    subgraph "🛡️ Red Team Critic"
        C_IN["<b>Reads:</b><br/>candidate_use_cases<br/>error_count"]
        C_OUT["<b>Writes:</b><br/>candidate_use_cases ↻<br/>final_top_5<br/>error_count<br/>current_agent"]
    end

    O_OUT --> S_IN
    S_OUT --> A_IN
    A_OUT --> R_IN
    R_OUT --> C_IN
    C_OUT -.->|"Retry Loop"| A_IN

    style O_IN fill:#1565c0,stroke:#42a5f5,color:#fff
    style O_OUT fill:#1565c0,stroke:#42a5f5,color:#fff
    style S_IN fill:#6a1b9a,stroke:#ab47bc,color:#fff
    style S_OUT fill:#6a1b9a,stroke:#ab47bc,color:#fff
    style A_IN fill:#ef6c00,stroke:#ffa726,color:#fff
    style A_OUT fill:#ef6c00,stroke:#ffa726,color:#fff
    style R_IN fill:#d84315,stroke:#ff7043,color:#fff
    style R_OUT fill:#d84315,stroke:#ff7043,color:#fff
    style C_IN fill:#283593,stroke:#5c6bc0,color:#fff
    style C_OUT fill:#283593,stroke:#5c6bc0,color:#fff
```

> *ⁱ = Used only on retry passes. ↻ = In-place enrichment of existing objects.*

---

## 4. Pydantic Data Model Hierarchy

The strict schema that enforces data contracts between agents at runtime.

```mermaid
classDiagram
    class GraphState {
        <<TypedDict>>
        +str original_query
        +AgentState current_agent
        +list~str~ target_supply_chain_nodes
        +list~str~ raw_evidence
        +list~UseCase~ candidate_use_cases
        +list~UseCase~ final_top_5
        +int error_count
        +list~str~ errors
    }

    class UseCase {
        <<BaseModel>>
        +str topic
        +str supply_chain_segment
        +str description
        +str implementation_approach
        +MaturityLevel maturity_level
        +EconomicImpact economic_impact
        +RiskAssessment risk_assessment
        +list~str~ evidence_sources
        +list~str~ critic_feedback
        +CriticVerdict critic_verdict
        +int iteration_count
    }

    class EconomicImpact {
        <<BaseModel>>
        +float estimated_roi_percentage
        +str marginal_efficiency_gain_description
        +CostComplexity implementation_cost_complexity
    }

    class RiskAssessment {
        <<BaseModel>>
        +str primary_bottleneck
        +str data_privacy_concerns
        +str integration_complexity
    }

    class MaturityLevel {
        <<Enum>>
        THEORETICAL
        PROOF_OF_CONCEPT
        PILOT
        PRODUCTION_READY
    }

    class CostComplexity {
        <<Enum>>
        LOW
        MEDIUM
        HIGH
    }

    class CriticVerdict {
        <<Enum>>
        PASS
        FAIL
        NEEDS_REVISION
    }

    class AgentState {
        <<Enum>>
        ORCHESTRATOR
        MARKET_SCRAPER
        ECONOMIC_ANALYST
        RISK_ASSESSOR
        RED_TEAM_CRITIC
        COMPLETED
    }

    GraphState *-- UseCase : candidate_use_cases / final_top_5
    GraphState *-- AgentState : current_agent
    UseCase *-- EconomicImpact : economic_impact
    UseCase *-- RiskAssessment : risk_assessment
    UseCase *-- MaturityLevel : maturity_level
    UseCase *-- CriticVerdict : critic_verdict
    EconomicImpact *-- CostComplexity : implementation_cost_complexity
```

---

## 5. Market Scraper — Search & Synthesis Pipeline

Details the two-stage pipeline executed per research node inside the Market Scraper.

```mermaid
flowchart TD
    INPUT["📥 <b>Input:</b> target_supply_chain_nodes<br/>(3–4 research sub-domains)"]

    subgraph "Per-Node Pipeline (×3–4)"
        direction TB

        Q1["🔎 Query 1:<br/>'node GenAI FMCG case study ROI'"]
        Q2["🔎 Query 2:<br/>'node AI implementation cost savings'"]
        Q3["🔎 Query 3:<br/>'node agentic AI procurement efficiency'"]

        TAVILY["🌍 <b>Tavily Advanced Search</b><br/>max_results=3 per query<br/>include_raw_content=True"]

        FILTER["🧹 <b>Filter & Normalise</b><br/>• Remove duplicates by URL<br/>• Quality gate: ≥50 chars<br/>• Prefer raw_content over snippet"]

        FORMAT["📝 <b>Format for LLM</b><br/>Structured text block<br/>with URL, title, content"]

        LLM["🤖 <b>Gemini Synthesis</b><br/>──────────────<br/>Extract quantitative evidence<br/>• ROI percentages<br/>• CapEx vs OpEx<br/>• Efficiency gains<br/>• Cost savings<br/>Cite every source URL"]

        FALLBACK["🔄 <b>Fallback</b><br/>If LLM fails → return<br/>raw results as plain text"]
    end

    OUTPUT["📤 <b>Output:</b> raw_evidence<br/>(tagged evidence strings)"]

    INPUT --> Q1 & Q2 & Q3
    Q1 & Q2 & Q3 --> TAVILY
    TAVILY --> FILTER
    FILTER --> FORMAT
    FORMAT --> LLM
    LLM --> OUTPUT
    LLM -.->|"on failure"| FALLBACK
    FALLBACK --> OUTPUT

    style INPUT fill:#4a148c,stroke:#ab47bc,color:#fff
    style OUTPUT fill:#4a148c,stroke:#ab47bc,color:#fff
    style TAVILY fill:#0d47a1,stroke:#42a5f5,color:#fff
    style LLM fill:#1b5e20,stroke:#66bb6a,color:#fff
    style FILTER fill:#37474f,stroke:#78909c,color:#fff
    style FALLBACK fill:#bf360c,stroke:#ff7043,color:#fff
```

---

## 6. Red Team Critic — Adversarial Evaluation & Retry Logic

The Critic's three-axis evaluation framework and the circuit-breaker mechanism that prevents infinite loops.

```mermaid
flowchart TD
    INPUT["📥 <b>Input:</b> candidate_use_cases<br/>+ error_count"]

    subgraph "Per-UseCase Evaluation Loop"
        direction TB

        CB{"⚡ Circuit Breaker<br/>iteration_count ≥ 2?"}

        FORCE_PASS["✅ <b>FORCED PASS</b><br/>Skip LLM call<br/>Flag as 'Theoretical'<br/>Manual review recommended"]

        subgraph "3-Axis Evaluation"
            AXIS1["<b>Axis 1: Hallucination Check</b><br/>• Fabricated evidence sources?<br/>• Impossible ROI claims?<br/>• Zero measurable metrics?"]
            AXIS2["<b>Axis 2: Integration Reality</b><br/>• Reasonable tech stack?<br/>• Real bottleneck identified?<br/>• Integration acknowledged?"]
            AXIS3["<b>Axis 3: Tendering Rigor</b><br/>(Vendor Tendering ONLY)<br/>• Genuinely agentic framework?<br/>• Multi-step procurement?<br/>• Realistic maturity level?"]
        end

        STRUCTURED["🔗 <b>Structured Output</b><br/>with_structured_output<br/>(CriticEvaluation)"]

        FALLBACK_PARSE["🔄 <b>Fallback Parse</b><br/>Raw JSON → manual validation"]

        VERDICT{"Verdict?"}

        PASS_UC["✅ PASS<br/>Append rationale to<br/>critic_feedback"]
        FAIL_UC["❌ FAIL<br/>Append fix instructions<br/>to critic_feedback<br/>Increment failures"]
    end

    DECIDE{"🔀 <b>Decision</b><br/>─────────────<br/>all_passed OR<br/>error_count ≥ MAX_RETRIES?"}

    PROMOTE["📤 <b>Promote</b><br/>candidates → final_top_5"]
    RETRY["🔄 <b>Signal Retry</b><br/>final_top_5 = [ ]<br/>error_count += 1"]

    INPUT --> CB
    CB -->|"Yes"| FORCE_PASS
    CB -->|"No"| AXIS1 & AXIS2 & AXIS3
    AXIS1 & AXIS2 & AXIS3 --> STRUCTURED
    STRUCTURED -.->|"on failure"| FALLBACK_PARSE
    STRUCTURED --> VERDICT
    FALLBACK_PARSE --> VERDICT
    VERDICT -->|"passed=True"| PASS_UC
    VERDICT -->|"passed=False"| FAIL_UC
    FORCE_PASS --> DECIDE
    PASS_UC --> DECIDE
    FAIL_UC --> DECIDE

    DECIDE -->|"Yes"| PROMOTE
    DECIDE -->|"No"| RETRY

    style INPUT fill:#283593,stroke:#5c6bc0,color:#fff
    style CB fill:#e65100,stroke:#ff9800,color:#fff
    style FORCE_PASS fill:#2e7d32,stroke:#66bb6a,color:#fff
    style PROMOTE fill:#2e7d32,stroke:#66bb6a,color:#fff
    style RETRY fill:#c62828,stroke:#ef5350,color:#fff
    style STRUCTURED fill:#1565c0,stroke:#42a5f5,color:#fff
```

---

## 7. LLM Configuration & Provider Architecture

The centralised LLM factory supporting multiple providers with fallback chains.

```mermaid
flowchart TD
    subgraph "config/settings.py"
        FACTORY["⚙️ <b>get_llm()</b><br/>Centralised LLM Factory"]
        PROVIDER{"LLM_PROVIDER?"}
    end

    subgraph "Vertex AI Path (Default)"
        VERTEX["<b>google_vertex</b><br/>ChatGoogleGenerativeAI<br/>──────────────<br/>Model: gemini-2.5-pro<br/>Auth: VERTEX_API_KEY<br/>Temperature: 0.2"]
    end

    subgraph "Google AI Studio Path"
        GOOGLE["<b>google</b><br/>ChatGoogleGenerativeAI<br/>──────────────<br/>Model: gemini-2.5-pro<br/>Auth: GOOGLE_API_KEY"]
    end

    subgraph "OpenAI Path"
        OPENAI["<b>openai</b><br/>ChatOpenAI<br/>──────────────<br/>Model: gpt-4o<br/>Auth: OPENAI_API_KEY"]
    end

    subgraph "Consumers (All Agents)"
        ORCH_LLM["Orchestrator<br/>(4096 tokens)"]
        SCRP_LLM["Market Scraper<br/>(4096 tokens)"]
        ANAL_LLM["Economic Analyst<br/>(8192 tokens)"]
        RISK_LLM["Risk Assessor<br/>(4096 tokens)"]
        CRIT_LLM["Red Team Critic<br/>(4096 tokens)"]
    end

    FACTORY --> PROVIDER
    PROVIDER -->|"google_vertex"| VERTEX
    PROVIDER -->|"google"| GOOGLE
    PROVIDER -->|"openai"| OPENAI

    VERTEX --> ORCH_LLM & SCRP_LLM & ANAL_LLM & RISK_LLM & CRIT_LLM
    GOOGLE --> ORCH_LLM & SCRP_LLM & ANAL_LLM & RISK_LLM & CRIT_LLM
    OPENAI --> ORCH_LLM & SCRP_LLM & ANAL_LLM & RISK_LLM & CRIT_LLM

    style FACTORY fill:#1b5e20,stroke:#66bb6a,color:#fff
    style PROVIDER fill:#37474f,stroke:#78909c,color:#fff
    style VERTEX fill:#0d47a1,stroke:#42a5f5,color:#fff
    style GOOGLE fill:#e65100,stroke:#ff9800,color:#fff
    style OPENAI fill:#4a148c,stroke:#ab47bc,color:#fff
```

---

## 8. Error Handling & Resilience Architecture

Shows the multi-layered error handling strategy across the pipeline.

```mermaid
flowchart TD
    subgraph "Layer 1: LLM Output Parsing"
        L1A["<b>Primary:</b> with_structured_output()<br/>Pydantic-validated JSON"]
        L1B["<b>Fallback:</b> Raw JSON parsing<br/>+ manual model_validate()"]
        L1C["<b>Last Resort:</b> Default values<br/>or conservative FAIL verdict"]
        L1A -.->|"ValidationError"| L1B
        L1B -.->|"JSONDecodeError"| L1C
    end

    subgraph "Layer 2: Search Resilience"
        L2A["<b>Tavily Search</b><br/>Exponential backoff<br/>max_retries=3"]
        L2B["<b>Quality Filter</b><br/>≥50 char gate<br/>URL deduplication"]
        L2C["<b>Fallback Evidence</b><br/>Raw results as plain text"]
        L2A --> L2B
        L2A -.->|"timeout / 429 / 503"| L2A
        L2B -.->|"LLM synthesis fails"| L2C
    end

    subgraph "Layer 3: Pipeline Circuit Breakers"
        L3A["<b>error_count Tracker</b><br/>Increments on any Critic failure batch"]
        L3B{"error_count ≥<br/>MAX_RETRIES (2)?"}
        L3C["<b>Force Promotion</b><br/>candidates → final_top_5<br/>Pipeline terminates"]
        L3D["<b>iteration_count</b><br/>Per UseCase (≥2 → force-pass)"]
        L3A --> L3B
        L3B -->|"Yes"| L3C
    end

    subgraph "Layer 4: Graceful Degradation"
        L4A["<b>Per-Node Isolation</b><br/>Scraper/Assessor/Critic errors<br/>on one UseCase don't block others"]
        L4B["<b>Mandatory Fail-safe</b><br/>Vendor Tendering use case<br/>injected if LLM omits it"]
        L4C["<b>Error Audit Trail</b><br/>All errors appended to<br/>GraphState.errors list"]
    end

    style L1A fill:#1565c0,stroke:#42a5f5,color:#fff
    style L1B fill:#e65100,stroke:#ff9800,color:#fff
    style L1C fill:#c62828,stroke:#ef5350,color:#fff
    style L3C fill:#c62828,stroke:#ef5350,color:#fff
    style L4B fill:#2e7d32,stroke:#66bb6a,color:#fff
```

---

## Memory and State Management
To prevent context collapse and state corruption across the 5-agent cyclic loop, the system manages state using a strict Pydantic `TypedDict` defined in **[GraphState](../src/state/graph_state.py)**.

We utilize the `.with_structured_output()` method to enforce a rigid data contract across all agent LLM calls. The state management system allows agents to append data immutably layer by layer. Furthermore, the system implements a strict `iteration_count` circuit breaker logic within the Critic. This prevents infinite API loop traps by force-passing stubborn use cases after a maximum number of retries, explicitly flagging them in the final output as "Theoretical". This guarantees that the pipeline will always resolve while still maintaining an auditable trail of the Critic's rejections.
