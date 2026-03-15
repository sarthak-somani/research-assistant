# GCPL AI Intern Hackathon Submission: Option A

## Overview
This repository contains the official submission for Option A: Multi-Agent Research Assistant. The primary focus of this system is to maintain high microeconomic rigor and perform thorough enterprise risk assessment. It features an adversarial reflection loop specifically designed to filter out Large Language Model (LLM) hallucinations, ensuring reliable, enterprise-grade outputs.

## System Architecture

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

## Repository Structure
```text
.
├── assets/
│   └── architecture.png
├── config/
│   └── settings.py
├── docs/
│   ├── 01_Problem_Approach.md
│   ├── 02_Architecture_Data_Flow.md
│   ├── 03_Technology_Choices.md
│   └── 04_Evaluation_and_Limitations.md
├── src/
│   ├── agents/
│   ├── core/
│   ├── ui/
│   │   └── app.py
│   └── utils/
├── .env.example
├── README.md
└── requirements.txt
```

## Documentation Library
* [Problem Approach](docs/01_Problem_Approach.md): Details the specific supply chain bottlenecks addressed and the analytical framework applied.
* [Architecture and Data Flow](docs/02_Architecture_Data_Flow.md): Explains the cyclic LangGraph setup, the roles of the five distinct agents, and the specific state management implementation.
* [Technology Choices](docs/03_Technology_Choices.md): Outlines the structural trade-offs, frameworks, APIs, and specific configurations chosen for the system.
* [Evaluation and Limitations](docs/04_Evaluation_and_Limitations.md): Covers testing outcomes, known constraints regarding context window limits, and future improvement vectors.

## Quick Start
```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run src/ui/app.py
```

## About the Author
Sarthak Somani is a second-year undergraduate student at IIT Bombay, pursuing a major in Economics and a minor in Data Science. With a strong interest in the intersection of microeconomics, artificial intelligence, and public policy, Sarthak focuses on building analytical, data-driven frameworks to solve complex enterprise bottlenecks. He serves as a Convenor at the Web and Coding Club (WnCC).
