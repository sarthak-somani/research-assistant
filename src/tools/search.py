"""
Search Tool — Tavily Integration
=================================

Provides a clean, production-grade abstraction over the Tavily Search API
using LangChain's ``TavilySearchResults`` tool. The Market Scraper agent
calls ``execute_research_query()`` without being coupled to API internals.

Features:
    - Automatic API key loading from environment
    - Quality-focused search (``search_depth="advanced"``)
    - Aggressive content filtering for high-signal results
    - Robust error handling for timeouts, rate limits, and empty results
    - Clean, normalised output format for downstream agents
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults

from config.settings import MAX_SEARCH_RESULTS, TAVILY_API_KEY

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Environment Setup
# ═══════════════════════════════════════════════════════════════════════════

# TavilySearchResults reads the key from os.environ at invocation time,
# so we inject it here to keep config centralised in settings.py.
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# ═══════════════════════════════════════════════════════════════════════════
# Tool Initialisation
# ═══════════════════════════════════════════════════════════════════════════

def _build_tavily_tool(max_results: int = MAX_SEARCH_RESULTS) -> TavilySearchResults:
    """
    Create a configured ``TavilySearchResults`` instance.

    Uses ``search_depth="advanced"`` for higher-quality, longer content
    snippets. The ``include_raw_content`` flag pulls full page text when
    available, giving the downstream LLM richer evidence.

    Args:
        max_results: Maximum number of results per query.

    Returns:
        Configured TavilySearchResults tool.
    """
    if not TAVILY_API_KEY and not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is not configured. "
            "Set it in your .env file or as an environment variable."
        )

    return TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=True,
        include_answer=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def execute_research_query(
    query: str,
    max_results: int = 3,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Execute a web search via Tavily and return cleaned, high-quality results.

    This function:
      1. Invokes the Tavily advanced search API.
      2. Filters out low-quality / empty results.
      3. Normalises output into a consistent ``{url, title, content}`` format.
      4. Retries with exponential backoff on transient failures.

    Args:
        query: The search query string. Should be specific and research-focused
               (e.g., "GenAI vendor tendering FMCG ROI case study").
        max_results: Maximum number of results to return (default 3).
        max_retries: Number of retry attempts on transient errors.
        base_delay: Initial backoff delay in seconds (doubles each retry).

    Returns:
        List of result dicts, each containing:
            - ``url``: Source URL
            - ``title``: Page title (may be empty)
            - ``content``: Cleaned text snippet / raw content

        Returns an empty list if the search fails after all retries.

    Example::

        results = execute_research_query(
            "agentic AI procurement FMCG case study ROI",
            max_results=3,
        )
        for r in results:
            print(r["url"], r["content"][:200])
    """
    tool = _build_tavily_tool(max_results=max_results)
    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Tavily search (attempt %d/%d): %s",
                attempt, max_retries, query[:80],
            )

            # TavilySearchResults.invoke() returns a list of dicts
            raw_results = tool.invoke({"query": query})

            if not raw_results:
                logger.warning("Tavily returned no results for: %s", query[:80])
                return []

            # ── Normalise & filter ────────────────────────────────────
            cleaned = _filter_and_normalise(raw_results)

            logger.info(
                "Tavily returned %d results (%d after filtering) for: %s",
                len(raw_results), len(cleaned), query[:80],
            )
            return cleaned

        except Exception as exc:
            last_exception = exc
            exc_str = str(exc).lower()

            # Identify retryable errors
            is_retryable = any(
                keyword in exc_str
                for keyword in ("timeout", "rate limit", "429", "503", "retry", "connection")
            )

            if is_retryable and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Tavily transient error (attempt %d/%d): %s. "
                    "Retrying in %.1fs...",
                    attempt, max_retries, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Tavily search failed (attempt %d/%d): %s",
                    attempt, max_retries, exc,
                )
                if not is_retryable:
                    break  # Non-retryable error — don't waste time

    logger.error(
        "Tavily search exhausted all %d retries for query: %s — last error: %s",
        max_retries, query[:80], last_exception,
    )
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _filter_and_normalise(
    raw_results: list[dict[str, Any] | str],
) -> list[dict[str, Any]]:
    """
    Filter out low-quality results and normalise into a consistent format.

    Filtering criteria:
      - Content must be non-empty and at least 50 characters
      - Duplicate URLs are removed
      - Results with error messages are skipped

    Args:
        raw_results: Raw output from ``TavilySearchResults.invoke()``.

    Returns:
        Filtered and normalised list of result dicts.
    """
    cleaned: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for item in raw_results:
        # Handle string results (Tavily sometimes returns plain strings)
        if isinstance(item, str):
            if len(item.strip()) >= 50:
                cleaned.append({
                    "url": "",
                    "title": "",
                    "content": item.strip(),
                })
            continue

        if not isinstance(item, dict):
            continue

        url = item.get("url", "").strip()
        content = item.get("content", "").strip()
        raw_content = item.get("raw_content", "").strip()
        title = item.get("title", "").strip()

        # Prefer raw_content (fuller text) over snippet if available
        best_content = raw_content if len(raw_content) > len(content) else content

        # ── Quality gate ──────────────────────────────────────────
        if len(best_content) < 50:
            logger.debug("Skipping thin result (%d chars): %s", len(best_content), url)
            continue

        # ── Dedup by URL ──────────────────────────────────────────
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        cleaned.append({
            "url": url,
            "title": title,
            "content": best_content,
        })

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible alias (used by old code paths)
# ═══════════════════════════════════════════════════════════════════════════

def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict[str, Any]]:
    """Legacy alias — delegates to ``execute_research_query``."""
    return execute_research_query(query=query, max_results=max_results)
