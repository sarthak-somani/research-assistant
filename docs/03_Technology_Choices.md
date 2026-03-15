# Technology Choices

| What we chose | Why we chose it | What we considered | Trade-offs |
| --- | --- | --- | --- |
| **LangGraph** | Enables cyclic graphs essential for the Red Team Critic loop. | CrewAI | LangGraph has a steeper learning curve but provides absolute control over state routing. |
| **Vertex AI (Gemini 2.5)** | Provides a massive context window capable of ingesting unparsed HTML directly. | OpenAI | Vertex AI enforces strict rate limits requiring custom backoff decorators for concurrent scraping. |
| **Tavily API** | Designed for agentic retrieval of raw content rather than just URLs. | Standard SerpAPI | Results in higher latency per query for deep research compared to standard endpoints. |
| **Pydantic** | Enforces strict JSON schema validation across the entire graph. | Standard JSON parsing | Requires complex retry logic and custom exception handling if the LLM breaks the output schema. |
