"""Tool-handler tests: one class per registered profiling tool."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.agents.base import ToolResultStatus


class TestToolGetOverview:
    @pytest.mark.asyncio
    async def test_returns_overview(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_get_overview(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_samples"] == 100
        assert result.output["n_features"] == 7
        assert "binary_columns" in result.output
        assert "numeric_columns" in result.output
        assert "suggestions" in result.output

    @pytest.mark.asyncio
    async def test_identifies_binary_columns(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_get_overview(state_with_dataframe)
        assert "treat" in result.output["binary_columns"]

    @pytest.mark.asyncio
    async def test_error_when_no_dataframe(self, agent, state_with_dataframe):
        agent._df = None
        agent._profile = None
        result = await agent._tool_get_overview(state_with_dataframe)
        assert result.status == ToolResultStatus.ERROR
        assert "not loaded" in result.error


class TestToolAnalyzeColumn:
    @pytest.mark.asyncio
    async def test_analyzes_binary_column(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_analyze_column(state_with_dataframe, column="treat")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["column"] == "treat"
        assert result.output["type"] == "binary"
        assert "value_distribution" in result.output
        assert "treatment_suitability" in result.output

    @pytest.mark.asyncio
    async def test_analyzes_numeric_column(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_analyze_column(state_with_dataframe, column="income")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["type"] == "numeric"
        assert "statistics" in result.output
        assert result.output["has_variance"] == True  # numpy bool, identity check fails

    @pytest.mark.asyncio
    async def test_column_not_found(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_analyze_column(state_with_dataframe, column="nonexistent")
        assert result.status == ToolResultStatus.ERROR
        assert "not found" in result.error


class TestToolCheckTreatmentBalance:
    @pytest.mark.asyncio
    async def test_binary_treatment_balance(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_check_treatment_balance(state_with_dataframe, column="treat")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["treatment_type"] == "binary"
        assert "minority_pct" in result.output
        assert "assessment" in result.output
        assert "suitable_methods" in result.output

    @pytest.mark.asyncio
    async def test_categorical_treatment(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_check_treatment_balance(state_with_dataframe, column="region")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_unique"] == 4

    @pytest.mark.asyncio
    async def test_continuous_variable(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_check_treatment_balance(state_with_dataframe, column="income")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["treatment_type"] == "continuous"
        assert result.output["assessment"] == "DOSE_RESPONSE"


class TestToolCheckRelationship:
    @pytest.mark.asyncio
    async def test_numeric_correlation(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_check_relationship(
            state_with_dataframe, column1="age", column2="income"
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert "pearson_correlation" in result.output
        assert "spearman_correlation" in result.output
        assert "strength" in result.output

    @pytest.mark.asyncio
    async def test_binary_numeric_relationship(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_check_relationship(
            state_with_dataframe, column1="treat", column2="outcome"
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert "group_0_mean" in result.output
        assert "group_1_mean" in result.output
        assert "mean_difference" in result.output

    @pytest.mark.asyncio
    async def test_column_not_found(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_check_relationship(
            state_with_dataframe, column1="nonexistent", column2="age"
        )
        assert result.status == ToolResultStatus.ERROR


class TestToolCheckTimeDimension:
    @pytest.mark.asyncio
    async def test_no_time_dimension(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_check_time_dimension(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_time_dimension"] is False

    @pytest.mark.asyncio
    async def test_with_time_column(self, agent, state_with_dataframe):
        df = pd.DataFrame({"year": [2019, 2020, 2021, 2022], "value": [1, 2, 3, 4]})
        agent._df = df
        agent._profile = agent._compute_basic_profile(df)
        result = await agent._tool_check_time_dimension(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_time_dimension"] is True
        assert len(result.output["candidates"]) > 0


class TestToolCheckDiscontinuity:
    @pytest.mark.asyncio
    async def test_no_discontinuity_candidates(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = await agent._tool_check_discontinuity(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_rdd_candidates"] is False

    @pytest.mark.asyncio
    async def test_with_score_column(self, agent, state_with_dataframe):
        df = pd.DataFrame({
            "test_score": np.random.normal(50, 10, 100),
            "passed": np.random.binomial(1, 0.5, 100),
        })
        agent._df = df
        agent._profile = agent._compute_basic_profile(df)
        result = await agent._tool_check_discontinuity(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_rdd_candidates"] is True
        assert len(result.output["candidates"]) > 0


class TestToolFinalizeProfile:
    @pytest.mark.asyncio
    async def test_finalizes_profile(self, agent, sample_dataframe, state_with_dataframe):
        """Finalize records the LLM's structure choice and flips agent._finalized."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_finalize_profile(
            state_with_dataframe,
            treatment_candidates=["treat"],
            outcome_candidates=["outcome", "income"],
            potential_confounders=["age", "education", "gender"],
            recommended_methods=["IPW", "Matching"],
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["profile_finalized"] is True
        assert agent._finalized is True
        assert agent._final_result["treatment_candidates"] == ["treat"]
        assert agent._final_result["outcome_candidates"] == ["outcome", "income"]
