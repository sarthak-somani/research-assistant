"""
Tests for the GraphState schema and Pydantic models.

Verifies that:
- Pydantic models enforce validation rules (min_length, ge/le, enums)
- TypedDict-based GraphState works with LangGraph dict-merge semantics
- Serialisation / deserialisation round-trips correctly
"""

import pytest

from src.state.graph_state import (
    AgentState,
    CostComplexity,
    CriticVerdict,
    EconomicImpact,
    GraphState,
    MaturityLevel,
    RiskAssessment,
    UseCase,
)


class TestEnums:
    """Test enum definitions and values."""

    def test_maturity_levels(self):
        assert MaturityLevel.THEORETICAL == "Theoretical"
        assert MaturityLevel.PROOF_OF_CONCEPT == "Proof_of_Concept"
        assert MaturityLevel.PILOT == "Pilot"
        assert MaturityLevel.PRODUCTION_READY == "Production_Ready"

    def test_agent_states(self):
        assert AgentState.ORCHESTRATOR == "orchestrator"
        assert AgentState.COMPLETED == "completed"

    def test_cost_complexity(self):
        assert CostComplexity.HIGH == "High"
        assert CostComplexity.LOW == "Low"

    def test_critic_verdicts(self):
        assert CriticVerdict.PASS == "pass"
        assert CriticVerdict.FAIL == "fail"
        assert CriticVerdict.NEEDS_REVISION == "needs_revision"


class TestEconomicImpact:
    """Test EconomicImpact Pydantic model."""

    def test_valid_impact(self):
        impact = EconomicImpact(
            estimated_roi_percentage=35.0,
            marginal_efficiency_gain_description=(
                "Reduces procurement cycle time by 40% through automated "
                "vendor scoring and RFQ generation."
            ),
            implementation_cost_complexity=CostComplexity.MEDIUM,
        )
        assert impact.estimated_roi_percentage == 35.0
        assert impact.implementation_cost_complexity == CostComplexity.MEDIUM

    def test_roi_out_of_bounds(self):
        with pytest.raises(Exception):
            EconomicImpact(
                estimated_roi_percentage=600.0,
                marginal_efficiency_gain_description="x" * 25,
                implementation_cost_complexity=CostComplexity.LOW,
            )

    def test_description_too_short(self):
        with pytest.raises(Exception):
            EconomicImpact(
                estimated_roi_percentage=10.0,
                marginal_efficiency_gain_description="short",
                implementation_cost_complexity=CostComplexity.LOW,
            )


class TestRiskAssessment:
    """Test RiskAssessment Pydantic model."""

    def test_valid_risk(self):
        risk = RiskAssessment(
            primary_bottleneck="Fragmented supplier data across multiple ERP systems",
            data_privacy_concerns="GDPR compliance for EU supplier PII in shared models",
            integration_complexity="Requires SAP S/4HANA API layer and custom middleware",
        )
        assert "ERP" in risk.primary_bottleneck

    def test_bottleneck_too_short(self):
        with pytest.raises(Exception):
            RiskAssessment(
                primary_bottleneck="short",
                data_privacy_concerns="GDPR compliance for EU supplier PII",
                integration_complexity="Requires SAP integration layer",
            )


class TestUseCase:
    """Test UseCase Pydantic model."""

    def test_minimal_use_case(self):
        uc = UseCase(topic="Agentic Vendor Tendering Automation")
        assert uc.topic == "Agentic Vendor Tendering Automation"
        assert uc.maturity_level == MaturityLevel.THEORETICAL
        assert uc.critic_feedback == []
        assert uc.critic_verdict == CriticVerdict.NEEDS_REVISION
        assert uc.economic_impact is None
        assert uc.risk_assessment is None
        assert uc.iteration_count == 0

    def test_iteration_count_default_and_custom(self):
        uc_default = UseCase(topic="Test Use Case Default")
        assert uc_default.iteration_count == 0

        uc_custom = UseCase(topic="Test Use Case Custom", iteration_count=3)
        assert uc_custom.iteration_count == 3

    def test_iteration_count_serialisation_roundtrip(self):
        uc = UseCase(topic="Iteration Roundtrip Test", iteration_count=2)
        json_str = uc.model_dump_json()
        restored = UseCase.model_validate_json(json_str)
        assert restored.iteration_count == 2

    def test_full_use_case(self):
        uc = UseCase(
            topic="AI-Driven Demand Sensing for Raw Material Procurement",
            supply_chain_segment="demand_forecasting",
            description="Uses transformer models to predict demand signals.",
            implementation_approach="Deploy on Azure ML with SAP BW integration.",
            maturity_level=MaturityLevel.PROOF_OF_CONCEPT,
            economic_impact=EconomicImpact(
                estimated_roi_percentage=42.0,
                marginal_efficiency_gain_description=(
                    "Reduces safety-stock holding costs by 25% and cuts "
                    "stockout incidents by 60%."
                ),
                implementation_cost_complexity=CostComplexity.HIGH,
            ),
            risk_assessment=RiskAssessment(
                primary_bottleneck="Requires 3+ years of historical POS data",
                data_privacy_concerns="Retailer data-sharing agreements under DPDP Act",
                integration_complexity="SAP IBP connector + custom ETL pipeline",
            ),
            evidence_sources=["https://example.com/case-study"],
            critic_feedback=["Impact score seems inflated", "Revised and grounded"],
        )
        assert uc.economic_impact.estimated_roi_percentage == 42.0
        assert len(uc.critic_feedback) == 2

    def test_topic_too_short(self):
        with pytest.raises(Exception):
            UseCase(topic="AI")


class TestGraphState:
    """Test TypedDict-based GraphState."""

    def test_graph_state_as_dict(self):
        state: GraphState = {
            "original_query": "Test query",
            "current_agent": AgentState.ORCHESTRATOR,
            "target_supply_chain_nodes": ["node1", "node2"],
            "raw_evidence": [],
            "candidate_use_cases": [],
            "final_top_5": [],
            "error_count": 0,
            "errors": [],
        }
        assert state["original_query"] == "Test query"
        assert state["error_count"] == 0

    def test_partial_state_update(self):
        """LangGraph uses partial dict updates — only changed keys are returned."""
        update = {
            "target_supply_chain_nodes": ["procurement", "logistics"],
            "current_agent": AgentState.ORCHESTRATOR,
        }
        assert len(update["target_supply_chain_nodes"]) == 2

    def test_use_case_serialisation_roundtrip(self):
        uc = UseCase(
            topic="Freight Route Optimisation via GenAI",
            maturity_level=MaturityLevel.PILOT,
        )
        json_str = uc.model_dump_json()
        restored = UseCase.model_validate_json(json_str)
        assert restored.topic == uc.topic
        assert restored.maturity_level == MaturityLevel.PILOT
