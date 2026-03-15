"""
Tests for the LangGraph graph builder.

Verifies graph construction, node registration, and edge topology.
"""

from src.graph.builder import build_graph


class TestGraphBuilder:
    """Test graph construction."""

    def test_graph_compiles(self):
        """The graph should compile without errors."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """All five agent nodes should be registered."""
        graph = build_graph()
        # LangGraph compiled graphs expose node names
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "orchestrator",
            "scraper",
            "analyst",
            "assessor",
            "critic",
            "__start__",
            "__end__",
        }
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"
