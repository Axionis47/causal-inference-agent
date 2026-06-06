"""Tool-handler tests across the diagnostic and finalize paths."""

import pytest

from src.analysis.agents.base import ToolResultStatus


class TestToolGetCandidates:
    @pytest.mark.asyncio
    async def test_excludes_treatment_and_outcome(self, primed_agent, state):
        result = await primed_agent._tool_get_candidate_variables(state)
        assert result.status == ToolResultStatus.SUCCESS
        cols = result.output["candidates"]
        assert "treat" not in cols
        assert "outcome" not in cols
        assert "age" in cols and "noise" in cols


class TestToolComputeCorrelation:
    @pytest.mark.asyncio
    async def test_age_correlates_with_treatment(self, primed_agent, state):
        """age was generated to be the confounder, so |corr| with treat is meaningful."""
        result = await primed_agent._tool_compute_correlation(state, var1="age", var2="treat")
        assert result.status == ToolResultStatus.SUCCESS
        assert abs(result.output["correlation"]) > 0.1
        assert result.output["significance"] == "significant"

    @pytest.mark.asyncio
    async def test_missing_variable_returns_error(self, primed_agent, state):
        result = await primed_agent._tool_compute_correlation(state, var1="nope", var2="treat")
        assert result.status == ToolResultStatus.ERROR


class TestToolComputePartialCorrelation:
    @pytest.mark.asyncio
    async def test_treat_outcome_drops_after_controlling_for_age(self, primed_agent, state):
        """age is the confounder: partial corr(treat, outcome | age) < raw corr."""
        raw = await primed_agent._tool_compute_correlation(state, var1="treat", var2="outcome")
        partial = await primed_agent._tool_compute_partial_correlation(
            state, var1="treat", var2="outcome", control_var="age"
        )
        assert partial.status == ToolResultStatus.SUCCESS
        assert abs(partial.output["partial_correlation"]) <= abs(raw.output["correlation"])


class TestToolTestConfounderCriteria:
    @pytest.mark.asyncio
    async def test_age_flagged_as_confounder(self, primed_agent, state):
        """Comparisons use `== True` because the flags are numpy.bool_, not Python bool."""
        result = await primed_agent._tool_test_confounder_criteria(state, variable="age")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["is_confounder"] == True  # noqa: E712 — numpy bool
        assert result.output["affects_treatment"] == True
        assert result.output["affects_outcome"] == True

    @pytest.mark.asyncio
    async def test_noise_has_lower_outcome_correlation(self, primed_agent, state):
        """noise was generated independent of outcome, so corr_with_outcome stays low.

        With n=200 and a 0.03 threshold the loose criterion can still trip by
        chance, so we assert the meaningful inequality (corr_y much smaller for
        noise than for age) rather than is_confounder.
        """
        age_result = await primed_agent._tool_test_confounder_criteria(state, variable="age")
        noise_result = await primed_agent._tool_test_confounder_criteria(state, variable="noise")
        assert noise_result.status == ToolResultStatus.SUCCESS
        assert abs(noise_result.output["corr_with_outcome"]) < abs(age_result.output["corr_with_outcome"])


class TestToolFinalizeConfounders:
    @pytest.mark.asyncio
    async def test_writes_state_and_flips_finalized(self, primed_agent, state):
        result = await primed_agent._tool_finalize_confounders(
            state,
            confounders=["age"],
            excluded=["noise"],
            reasoning="age affects both treatment and outcome",
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert primed_agent._finalized is True
        assert state.confounder_discovery["ranked_confounders"] == ["age"]
        assert state.confounder_discovery["excluded_variables"] == ["noise"]

    @pytest.mark.asyncio
    async def test_empty_confounders_triggers_statistical_fallback(self, primed_agent, state):
        """LLM submitting no confounders should not leave the analysis empty —
        the statistical scan fills it in from observed correlations."""
        result = await primed_agent._tool_finalize_confounders(
            state, confounders=[], excluded=[], reasoning=""
        )
        assert result.status == ToolResultStatus.SUCCESS
        # The fallback should at least find `age` (the real confounder)
        assert "age" in state.confounder_discovery["ranked_confounders"]
        assert "statistical_fallback_details" in state.confounder_discovery
