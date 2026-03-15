"""
Structured Logging Setup
========================

Configures structured logging using structlog for production observability.
Every agent step is logged with rich context (agent name, state snapshot, timing).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

import structlog

from config.settings import LOG_FORMAT, LOG_LEVEL, OUTPUT_DIR


def setup_logging() -> None:
    """
    Initialise structured logging for the application.

    Call this once at startup (in main.py).
    """
    # Configure standard logging to output to both stdout and a file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"pipeline_run_{timestamp}.txt"
    
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )

    # Choose renderer based on config
    if LOG_FORMAT.lower() == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named structured logger."""
    return structlog.get_logger(name)
