# Problem and Approach

## Problem Interpretation
Modern Fast-Moving Consumer Goods (FMCG) supply chains suffer immensely from legacy technical debt. Fragmented visibility, entrenched vendor lock-in, and opaque procurement cycles create significant operational drag. Standard LLM wrappers and automation tools often fail because they lack the capacity to contextualize multi-stage workflows or evaluate the genuine cost of switching out embedded enterprise systems. They frequently suffer from hallucinated capabilities, suggesting theoretically perfect solutions that crumble upon contact with actual enterprise IT environments.

## Our Approach
Instead of designing a generic search-and-summarize chatbot, this system acts as a rigorous enterprise consulting pipeline. The primary objective is to force the underlying LLMs to calculate explicit microeconomic variables using strict Pydantic schemas. 

The system explicitly requires outputs to include structured fields such as `estimated_roi_percentage`, `marginal_efficiency_gain_description`, and `implementation_cost_complexity`. By enforcing this data contract, the pipeline rejects qualitative fluff and forces the agents to substantiate their recommendations against strict enterprise integration constraints and actual capital expenditure realities.
