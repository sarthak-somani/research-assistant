"""
Centralised configuration for the GenAI FMCG Research Assistant.

Loads settings from environment variables (.env file) with sensible defaults.
Provides a shared ``get_llm()`` factory so every agent uses the same LLM
instance configuration (currently Vertex AI / Gemini).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google_vertex")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
VERTEX_API_KEY: str = os.getenv("VERTEX_API_KEY", "")

# ---------------------------------------------------------------------------
# GCP / Vertex AI
# ---------------------------------------------------------------------------
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")

# ---------------------------------------------------------------------------
# Search Configuration
# ---------------------------------------------------------------------------
SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "tavily")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
MAX_USE_CASES: int = int(os.getenv("MAX_USE_CASES", "5"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = _PROJECT_ROOT
OUTPUT_DIR: Path = _PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Centralised LLM Factory
# ---------------------------------------------------------------------------

def get_llm(max_output_tokens: int = 4096):
    """
    Create and return a configured LLM instance.

    Uses Vertex AI (``ChatVertexAI``) by default, falling back to
    Google AI Studio (``ChatGoogleGenerativeAI``) or OpenAI as configured.

    Args:
        max_output_tokens: Maximum tokens in the LLM response.

    Returns:
        A LangChain chat model instance (BaseChatModel).

    Raises:
        RuntimeError: If the required API key is missing.
    """
    if LLM_PROVIDER == "google_vertex":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not VERTEX_API_KEY:
            raise RuntimeError(
                "VERTEX_API_KEY is not set. Add it to your .env file."
            )

        # API key auth works via ChatGoogleGenerativeAI regardless of
        # whether the key was provisioned through Vertex AI or AI Studio.
        os.environ["GOOGLE_API_KEY"] = VERTEX_API_KEY
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=LLM_TEMPERATURE,
            google_api_key=VERTEX_API_KEY,
            max_output_tokens=max_output_tokens,
        )

    elif LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )

        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=LLM_TEMPERATURE,
            google_api_key=GOOGLE_API_KEY,
            max_output_tokens=max_output_tokens,
        )

    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            max_tokens=max_output_tokens,
        )

    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Expected: google_vertex | google | openai"
        )
