"""
Output Validators
=================

Pydantic-based validators for ensuring LLM outputs conform to expected
schemas before they enter the graph state. Catches malformed JSON, missing
fields, and type mismatches at the boundary.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def validate_llm_json(raw_output: str, model_class: type[T]) -> T | None:
    """
    Parse and validate LLM JSON output against a Pydantic model.

    Args:
        raw_output: Raw string output from the LLM.
        model_class: The Pydantic model class to validate against.

    Returns:
        A validated model instance, or None if validation fails.
    """
    try:
        # Handle markdown-wrapped JSON (```json ... ```)
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (``` markers)
            cleaned = "\n".join(lines[1:-1])

        data = json.loads(cleaned)
        return model_class.model_validate(data)

    except json.JSONDecodeError as e:
        logger.error("LLM output is not valid JSON: %s", e)
        return None
    except ValidationError as e:
        logger.error("LLM output failed Pydantic validation: %s", e)
        return None


def validate_llm_json_list(
    raw_output: str, model_class: type[T]
) -> list[T]:
    """
    Parse and validate LLM JSON array output against a Pydantic model.

    Args:
        raw_output: Raw string output from the LLM (expected JSON array).
        model_class: The Pydantic model class for each array element.

    Returns:
        A list of validated model instances (empty list on failure).
    """
    try:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])

        data = json.loads(cleaned)
        if not isinstance(data, list):
            logger.error("Expected JSON array, got %s", type(data).__name__)
            return []

        results = []
        for i, item in enumerate(data):
            try:
                results.append(model_class.model_validate(item))
            except ValidationError as e:
                logger.warning("Item %d failed validation: %s", i, e)

        return results

    except json.JSONDecodeError as e:
        logger.error("LLM output is not valid JSON: %s", e)
        return []
