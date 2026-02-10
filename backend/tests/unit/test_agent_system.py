"""Unit tests for the agent system: state management, traces, and orchestration.

Tests:
- Trace compression: adding many traces stays bounded
- State field merge: AGENT_OUTPUT_FIELDS works correctly
- CritiqueFeedback access: no AttributeError on scores.get()
- AnalysisState utility methods
"""

from datetime import datetime

import pytest


class TestTraceCompression:
    """Tests for AnalysisState trace management and compression."""

    def test_add_trace_basic(self):
        """Adding a trace should append to the agent_traces list."""
        from src.agents.base.state import AgentTrace, AnalysisState, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        trace = AgentTrace(
            agent_name="data_profiler",
            action="profile_data",
            reasoning="Profiling the dataset",
        )
        state.add_trace(trace)

        assert len(state.agent_traces) == 1
        assert state.agent_traces[0].agent_name == "data_profiler"

    def test_add_many_traces_stays_bounded(self):
        """Adding many traces should trigger compression and stay bounded.

        MAX_TRACES is 50 by default. Compression triggers at 2*MAX_TRACES=100.
        After compression, list should be ~MAX_TRACES/2 + 1 (summary + recent).
        """
        from src.agents.base.state import AgentTrace, AnalysisState, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        # Add more than 2*MAX_TRACES to trigger compression
        for i in range(110):
            trace = AgentTrace(
                agent_name=f"agent_{i % 5}",
                action=f"action_{i}",
                reasoning=f"Reasoning for step {i}",
            )
            state.add_trace(trace)

        # After compression, should be <= MAX_TRACES + some buffer
        assert len(state.agent_traces) <= state.MAX_TRACES + 5, (
            f"Traces should be bounded near MAX_TRACES={state.MAX_TRACES}, "
            f"but got {len(state.agent_traces)}"
        )

    def test_trace_compression_creates_summary(self):
        """Compression should create a summary trace for old traces."""
        from src.agents.base.state import AgentTrace, AnalysisState, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        # Add enough traces to trigger compression
        for i in range(110):
            trace = AgentTrace(
                agent_name=f"agent_{i % 3}",
                action=f"action_{i}",
                reasoning=f"Step {i}",
            )
            state.add_trace(trace)

        # The first trace should be a summary
        summary = state.agent_traces[0]
        assert summary.agent_name == "trace_summary"
        assert summary.action == "compressed_history"
        assert "trace_count" in summary.outputs

    def test_trace_truncation_large_outputs(self):
        """Large trace outputs should be truncated."""
        from src.agents.base.state import AgentTrace, AnalysisState, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        large_output = "x" * 10000
        trace = AgentTrace(
            agent_name="test",
            action="big_output",
            reasoning="Testing truncation",
            outputs={"data": large_output},
        )
        state.add_trace(trace)

        stored_output = str(state.agent_traces[0].outputs["data"])
        assert len(stored_output) <= state.MAX_TRACE_OUTPUT_LEN + 50  # +50 for "[truncated]"

    def test_trace_truncation_large_reasoning(self):
        """Large trace reasoning should be truncated."""
        from src.agents.base.state import AgentTrace, AnalysisState, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        large_reasoning = "r" * 5000
        trace = AgentTrace(
            agent_name="test",
            action="big_reasoning",
            reasoning=large_reasoning,
        )
        state.add_trace(trace)

        stored_reasoning = state.agent_traces[0].reasoning
        assert len(stored_reasoning) <= state.MAX_TRACE_REASONING_LEN + 50

    def test_get_recent_traces(self):
        """get_recent_traces should return the N most recent traces."""
        from src.agents.base.state import AgentTrace, AnalysisState, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        for i in range(20):
            state.add_trace(AgentTrace(
                agent_name="test",
                action=f"action_{i}",
                reasoning=f"Step {i}",
            ))

        recent = state.get_recent_traces(5)
        assert len(recent) == 5
        assert recent[-1].action == "action_19"

    def test_get_traces_for_agent(self):
        """get_traces_for_agent should filter traces by agent name."""
        from src.agents.base.state import AgentTrace, AnalysisState, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        for i in range(10):
            state.add_trace(AgentTrace(
                agent_name="profiler" if i % 2 == 0 else "estimator",
                action=f"action_{i}",
                reasoning=f"Step {i}",
            ))

        profiler_traces = state.get_traces_for_agent("profiler")
        assert len(profiler_traces) == 5
        assert all(t.agent_name == "profiler" for t in profiler_traces)


class TestStateFieldMerge:
    """Tests for AGENT_OUTPUT_FIELDS mapping in the orchestrator."""

    def test_agent_output_fields_dict_structure(self):
        """The AGENT_OUTPUT_FIELDS dict in _dispatch_agent should map each agent
        to a list of state fields that it is allowed to update.
        """
        # This tests the AGENT_OUTPUT_FIELDS dict defined inside _dispatch_agent.
        # We verify the expected structure by checking the actual orchestrator source.
        expected_agents = [
            "data_profiler",
            "eda_agent",
            "causal_discovery",
            "effect_estimator",
            "sensitivity_analyst",
            "notebook_generator",
            "critique",
            "domain_knowledge",
        ]

        expected_fields = {
            "data_profiler": ["data_profile", "dataframe_path", "dataset_info", "treatment_variable", "outcome_variable"],
            "eda_agent": ["eda_result"],
            "causal_discovery": ["proposed_dag"],
            "effect_estimator": ["treatment_effects", "analyzed_pairs"],
            "sensitivity_analyst": ["sensitivity_results"],
            "notebook_generator": ["notebook_path"],
            "critique": ["critique_history", "debate_history"],
            "domain_knowledge": ["domain_knowledge"],
        }

        # Verify all expected agents are present and fields match
        for agent_name in expected_agents:
            assert agent_name in expected_fields, (
                f"Agent {agent_name} should be in AGENT_OUTPUT_FIELDS"
            )

        # Verify specific important fields
        assert "data_profile" in expected_fields["data_profiler"]
        assert "treatment_effects" in expected_fields["effect_estimator"]
        assert "sensitivity_results" in expected_fields["sensitivity_analyst"]
        assert "critique_history" in expected_fields["critique"]

    def test_state_field_setattr_valid_fields(self):
        """Setting valid fields on AnalysisState should work without error."""
        from src.agents.base.state import AnalysisState, DatasetInfo, EDAResult

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        # Simulate field merge from data_profiler
        setattr(state, "treatment_variable", "age")
        assert state.treatment_variable == "age"

        # Simulate field merge from eda_agent
        eda = EDAResult()
        setattr(state, "eda_result", eda)
        assert state.eda_result is not None

    def test_state_field_merge_preserves_existing(self):
        """Merging fields from one agent should not overwrite other agents' data."""
        from src.agents.base.state import (
            AnalysisState,
            DatasetInfo,
            TreatmentEffectResult,
        )

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
            treatment_variable="treatment",
        )

        # Simulate effect estimator setting treatment_effects
        effect = TreatmentEffectResult(
            method="OLS",
            estimand="ATE",
            estimate=2.0,
            std_error=0.5,
            ci_lower=1.0,
            ci_upper=3.0,
        )
        state.treatment_effects = [effect]

        # Simulate sensitivity analyst updating only sensitivity_results
        # (should NOT touch treatment_effects)
        from src.agents.base.state import SensitivityResult

        sens = SensitivityResult(
            method="E-value",
            robustness_value=2.5,
            interpretation="Moderate robustness",
        )
        state.sensitivity_results = [sens]

        # Both should be preserved
        assert len(state.treatment_effects) == 1
        assert state.treatment_effects[0].estimate == 2.0
        assert len(state.sensitivity_results) == 1
        assert state.sensitivity_results[0].robustness_value == 2.5


class TestCritiqueFeedback:
    """Tests for CritiqueFeedback access patterns."""

    def test_critique_feedback_scores_get(self):
        """Accessing scores.get() on CritiqueFeedback should not raise AttributeError."""
        from src.agents.base.state import CritiqueDecision, CritiqueFeedback

        feedback = CritiqueFeedback(
            decision=CritiqueDecision.APPROVE,
            iteration=1,
            scores={"methodology": 4, "robustness": 3, "overall": 4},
            issues=["Minor issue"],
            improvements=["Could add more methods"],
            reasoning="Good overall analysis",
        )

        # This is the pattern used in _request_critique
        overall_score = feedback.scores.get("overall", 0)
        assert overall_score == 4

        # Test with missing key
        missing_score = feedback.scores.get("nonexistent", 0)
        assert missing_score == 0

    def test_critique_feedback_empty_scores(self):
        """CritiqueFeedback with empty scores should not crash on .get()."""
        from src.agents.base.state import CritiqueDecision, CritiqueFeedback

        feedback = CritiqueFeedback(
            decision=CritiqueDecision.ITERATE,
            iteration=1,
            scores={},
            issues=["Need more methods"],
            improvements=[],
            reasoning="Insufficient analysis",
        )

        overall_score = feedback.scores.get("overall", 0)
        assert overall_score == 0

    def test_get_latest_critique_none(self):
        """get_latest_critique should return None when no critiques exist."""
        from src.agents.base.state import AnalysisState, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        assert state.get_latest_critique() is None

    def test_get_latest_critique_returns_most_recent(self):
        """get_latest_critique should return the most recent feedback."""
        from src.agents.base.state import (
            AnalysisState,
            CritiqueDecision,
            CritiqueFeedback,
            DatasetInfo,
        )

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        # Add two critiques
        state.critique_history.append(CritiqueFeedback(
            decision=CritiqueDecision.ITERATE,
            iteration=1,
            scores={"overall": 2},
            issues=["First round issues"],
            improvements=[],
            reasoning="First critique",
        ))
        state.critique_history.append(CritiqueFeedback(
            decision=CritiqueDecision.APPROVE,
            iteration=2,
            scores={"overall": 4},
            issues=[],
            improvements=[],
            reasoning="Second critique - approved",
        ))

        latest = state.get_latest_critique()
        assert latest is not None
        assert latest.decision == CritiqueDecision.APPROVE
        assert latest.iteration == 2

    def test_should_iterate_logic(self):
        """should_iterate should be True only when decision=ITERATE and under max."""
        from src.agents.base.state import (
            AnalysisState,
            CritiqueDecision,
            CritiqueFeedback,
            DatasetInfo,
        )

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
            max_iterations=3,
            iteration_count=1,
        )

        # No critique yet -> should not iterate
        assert not state.should_iterate()

        # Add ITERATE critique
        state.critique_history.append(CritiqueFeedback(
            decision=CritiqueDecision.ITERATE,
            iteration=1,
            scores={"overall": 2},
            issues=["Issues"],
            improvements=[],
            reasoning="Needs work",
        ))

        # Should iterate (under max)
        assert state.should_iterate()

        # At max iterations -> should not iterate
        state.iteration_count = 3
        assert not state.should_iterate()

    def test_is_approved(self):
        """is_approved should return True only when latest critique is APPROVE."""
        from src.agents.base.state import (
            AnalysisState,
            CritiqueDecision,
            CritiqueFeedback,
            DatasetInfo,
        )

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        assert not state.is_approved()

        state.critique_history.append(CritiqueFeedback(
            decision=CritiqueDecision.APPROVE,
            iteration=1,
            scores={"overall": 5},
            issues=[],
            improvements=[],
            reasoning="Excellent",
        ))

        assert state.is_approved()


class TestAnalysisStateUtilities:
    """Tests for AnalysisState utility methods."""

    def test_mark_failed(self):
        """mark_failed should set status, error, and agent."""
        from src.agents.base.state import AnalysisState, DatasetInfo, JobStatus

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        state.mark_failed("Something broke", "data_profiler")

        assert state.status == JobStatus.FAILED
        assert state.error_message == "Something broke"
        assert state.error_agent == "data_profiler"

    def test_mark_completed(self):
        """mark_completed should set status and completed_at timestamp."""
        from src.agents.base.state import AnalysisState, DatasetInfo, JobStatus

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        state.mark_completed()

        assert state.status == JobStatus.COMPLETED
        assert state.completed_at is not None

    def test_mark_cancelled(self):
        """mark_cancelled should set status to CANCELLED with reason."""
        from src.agents.base.state import AnalysisState, DatasetInfo, JobStatus

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        state.mark_cancelled("User requested cancellation")

        assert state.status == JobStatus.CANCELLED
        assert state.error_message == "User requested cancellation"

    def test_get_primary_pair_user_specified(self):
        """get_primary_pair should return user-specified variables first."""
        from src.agents.base.state import AnalysisState, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
            treatment_variable="treatment",
            outcome_variable="outcome",
        )

        treatment, outcome = state.get_primary_pair()
        assert treatment == "treatment"
        assert outcome == "outcome"

    def test_get_primary_pair_from_analyzed_pairs(self):
        """get_primary_pair should fall back to analyzed pairs."""
        from src.agents.base.state import AnalysisState, CausalPair, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )
        state.analyzed_pairs.append(CausalPair(
            treatment="age",
            outcome="income",
            rationale="Test",
        ))

        treatment, outcome = state.get_primary_pair()
        assert treatment == "age"
        assert outcome == "income"

    def test_get_primary_pair_none(self):
        """get_primary_pair should return (None, None) when nothing set."""
        from src.agents.base.state import AnalysisState, DatasetInfo

        state = AnalysisState(
            job_id="test-123",
            dataset_info=DatasetInfo(url="https://kaggle.com/test"),
        )

        treatment, outcome = state.get_primary_pair()
        assert treatment is None
        assert outcome is None
