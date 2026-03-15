"""
Retry Utilities
===============

Provides configurable retry logic with exponential backoff for LLM API
calls and search API calls. Handles transient failures gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Multiplier for exponential backoff.
        retryable_exceptions: Tuple of exception types to retry on.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay,
                        )
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                            attempt + 1,
                            max_retries + 1,
                            func.__name__,
                            str(e),
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s",
                            max_retries + 1,
                            func.__name__,
                        )
            raise last_exception  # type: ignore[misc]

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay,
                        )
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                            attempt + 1,
                            max_retries + 1,
                            func.__name__,
                            str(e),
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s",
                            max_retries + 1,
                            func.__name__,
                        )
            raise last_exception  # type: ignore[misc]

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
