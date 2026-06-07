"""Tests for CritiqueDecision.REJECT plumbing.

Covers:
    - AnalysisState.is_rejected helper
    - Standard orchestrator marking the state FAILED on REJECT
    - ReAct orchestrator short-circuiting on REJECT in _request_critique
    - Heuristic critique fallback emitting REJECT for empty analyses
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.analysis.agents import (
    AnalysisState,
    CritiqueDecision,
    CritiqueFeedback,
    DatasetInfo,
    JobStatus,
)
from src.analysis.agents.critique.agent import CritiqueAgent
from src.analysis.orchestrator.react import ReActOrchestrator
from src.analysis.orchestrator.standard import StandardOrchestrator


def _make_state() -> AnalysisState:
    return AnalysisState(
        job_id="test-job-reject",
        dataset_info=DatasetInfo(url="https://kaggle.com/test", name="test"),
        treatment_variable="treatment",
        outcome_variable="outcome",
    )


def _reject_feedback() -> CritiqueFeedback:
    return CritiqueFeedback(
        decision=CritiqueDecision.REJECT,
        iteration=1,
        scores={
            "statistical_validity": 1,
            "assumption_checking": 1,
            "method_selection": 1,
            "completeness": 1,
            "reproducibility": 1,
            "interpretation": 1,
        },
        issues=["No methods produced any treatment effect estimate"],
        improvements=[],
        reasoning="Analysis is empty; cannot recover via iteration.",
    )


class TestIsRejected:
    """AnalysisState.is_rejected reflects the latest critique decision."""

    def test_no_critique_returns_false(self):
        state = _make_state()
        assert state.is_rejected() is False
        assert state.is_approved() is False
        assert state.should_iterate() is False

    def test_reject_decision_detected(self):
        state = _make_state()
        state.critique_history.append(_reject_feedback())
        assert state.is_rejected() is True
        assert state.is_approved() is False

    def test_approve_decision_not_rejected(self):
        state = _make_state()
        state.critique_history.append(
            CritiqueFeedback(
                decision=CritiqueDecision.APPROVE,
                iteration=1,
                scores={"statistical_validity": 5, "assumption_checking": 5,
                        "method_selection": 5, "completeness": 5,
                        "reproducibility": 5, "interpretation": 5},
                issues=[],
                improvements=[],
                reasoning="ok",
            )
        )
        assert state.is_rejected() is False
        assert state.is_approved() is True


class TestStandardOrchestratorReject:
    """StandardOrchestrator marks the state FAILED on REJECT."""

    @pytest.mark.asyncio
    async def test_request_critique_marks_failed_on_reject(self):
        orchestrator = StandardOrchestrator()

        # Mock critique that records a REJECT into state
        critique = AsyncMock()

        async def fake_execute(state):
            state.critique_history.append(_reject_feedback())
            return state

        critique.execute_with_tracing.side_effect = fake_execute
        orchestrator.register_specialist("critique", critique)

        state = _make_state()
        result = await orchestrator._run_critique(state)

        assert result.status == JobStatus.FAILED
        assert "rejected_by_critique" in (result.error_message or "")
        assert result.error_agent == "critique"


class TestReActOrchestratorReject:
    """ReActOrchestrator marks the state FAILED on REJECT in _request_critique."""

    @pytest.mark.asyncio
    async def test_request_critique_marks_failed_on_reject(self):
        orchestrator = ReActOrchestrator()

        critique = AsyncMock()

        async def fake_execute(state):
            state.critique_history.append(_reject_feedback())
            return state

        critique.execute_with_tracing.side_effect = fake_execute
        orchestrator.register_specialist("critique", critique)

        state = _make_state()
        tool_result = await orchestrator._request_critique(state, summary="x")

        # ToolResult signals error so the LLM stops the loop
        assert tool_result.status.name == "ERROR"
        assert tool_result.output is not None
        assert tool_result.output.get("decision") == "REJECT"
        # State itself is marked FAILED
        assert state.status == JobStatus.FAILED
        assert "rejected_by_critique" in (state.error_message or "")


class TestHeuristicCritiqueReject:
    """Heuristic fallback returns REJECT for genuinely empty analyses."""

    def test_empty_analysis_rejected(self):
        agent = CritiqueAgent()
        agent._state = _make_state()
        agent._investigation_evidence = []

        result = agent._auto_finalize()

        # No treatment_effects, no sensitivity_results, no DAG -> REJECT
        assert result["decision"] == "REJECT"

    def test_normal_analysis_not_rejected(self):
        from src.analysis.agents import TreatmentEffectResult

        agent = CritiqueAgent()
        state = _make_state()
        state.treatment_effects = [
            TreatmentEffectResult(
                method="OLS", estimand="ATE", estimate=1.5, std_error=0.2,
                ci_lower=1.1, ci_upper=1.9, p_value=0.001,
            ),
            TreatmentEffectResult(
                method="IPW", estimand="ATE", estimate=1.4, std_error=0.25,
                ci_lower=0.9, ci_upper=1.9, p_value=0.005,
            ),
        ]
        agent._state = state
        agent._investigation_evidence = []

        result = agent._auto_finalize()

        # Two methods with results -> never REJECT in heuristic
        assert result["decision"] != "REJECT"
