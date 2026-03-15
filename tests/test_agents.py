"""
Tests for the Orchestrator agent.

Tests cover:
- run_orchestrator signature and return structure
- Mandatory node injection guarantee
- JSON parsing robustness
"""

import json

import pytest

from src.agents.orchestrator import (
    _MANDATORY_NODE,
    _parse_target_nodes,
    run_orchestrator,
)
from src.state.graph_state import AgentState


class TestParseTargetNodes:
    """Test the JSON parsing helper."""

    def test_parses_valid_json_object(self):
        raw = json.dumps({"target_nodes": ["node1", "node2", "node3"]})
        result = _parse_target_nodes(raw)
        assert result == ["node1", "node2", "node3"]

    def test_parses_bare_json_array(self):
        raw = json.dumps(["node1", "node2"])
        result = _parse_target_nodes(raw)
        assert result == ["node1", "node2"]

    def test_strips_markdown_fences(self):
        raw = '```json\n{"target_nodes": ["a", "b"]}\n```'
        result = _parse_target_nodes(raw)
        assert result == ["a", "b"]

    def test_fallback_on_invalid_json(self):
        raw = "this is not json at all"
        result = _parse_target_nodes(raw)
        assert len(result) >= 2
        assert _MANDATORY_NODE in result

    def test_alternative_key_names(self):
        raw = json.dumps({"nodes": ["x", "y"]})
        result = _parse_target_nodes(raw)
        assert result == ["x", "y"]


class TestMandatoryNodeConstant:
    """Verify the mandatory node string."""

    def test_mandatory_node_contains_vendor_tendering(self):
        assert "Vendor Tendering" in _MANDATORY_NODE
        assert "Procurement" in _MANDATORY_NODE
        assert "Agentic" in _MANDATORY_NODE
